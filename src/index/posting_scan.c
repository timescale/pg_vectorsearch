/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * posting_scan.c - Posting list scan with fused cluster function
 *
 * Scans posting data (paged or flat), computes batch RaBitQ
 * distances, prunes via error bounds, and reranks survivors
 * in a single function call per cluster.
 *
 * Optimizations:
 * - Inline page access via stored base pointer (no vtable dispatch)
 * - Score + prune + rerank fused into one function (no iterator
 *   overhead, compiler can optimize across all phases)
 * - IP count padded to multiple of 4 (avoids tail kernel)
 */

#include "vs_config.h"

#include <string.h>

#include "core/log.h"
#include "core/memory.h"
#include "index/posting_scan.h"
#include "quant/fastscan.h"

#ifdef VS_SIMD_FULL
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>

#include "algo/simd_utils.h"
#endif
#endif

/* ----------------------------------------------------------------
 * Init / cleanup
 * ---------------------------------------------------------------- */

void
prism_posting_scan_init(
		PrismPostingScan   *scan,
		VsStorage		   *storage,
		char			   *page_base,
		const RaBitQParams *params,
		Dimension			dim,
		uint32_t			max_entries_per_page)
{
	memset(scan, 0, sizeof(*scan));
	scan->storage	   = storage;
	scan->page_base	   = page_base;
	scan->params	   = params;
	scan->dim		   = dim;
	scan->packed_bytes = VS_RABITQ_BYTES(dim);
	scan->cur_blkno	   = InvalidBlockNumber;

	/* Pre-allocate batch buffers — pad to multiple of 4 for IP kernel.
	 * max_entries_cap is the trusted per-page capacity these buffers are
	 * sized for; a page whose on-disk entry_count exceeds it (corruption)
	 * would overrun page_distances/page_scratch and is rejected on load. */
	scan->max_entries_cap = max_entries_per_page;
	uint32_t padded		  = (max_entries_per_page + 3) & ~3u;
	scan->page_distances  = vs_alloc(padded * sizeof(Distance));
	scan->page_scratch	  = vs_alloc(padded * sizeof(float));
}

void
prism_posting_scan_cleanup(PrismPostingScan *scan)
{
	/* Release pinned page if still held */
	if (scan->cur_page != NULL && scan->storage != NULL &&
		scan->page_base == NULL)
	{
		vs_storage_release_page(scan->storage, scan->cur_blkno);
		scan->cur_page = NULL;
	}

	vs_free(scan->page_distances);
	vs_free(scan->page_scratch);
	scan->page_distances = NULL;
	scan->page_scratch	 = NULL;

	if (scan->fs_lut != NULL)
	{
		vs_free(scan->fs_lut);
		scan->fs_lut = NULL;
	}
	if (scan->fs_accum != NULL)
	{
		vs_free(scan->fs_accum);
		scan->fs_accum = NULL;
	}
}

void
prism_posting_scan_enable_fastscan(PrismPostingScan *scan, int lut_bits)
{
	scan->fs_lut_bits = lut_bits;
	if (lut_bits == 8)
		scan->fs_lut = vs_alloc0(VS_FASTSCAN_LUT_BYTES(scan->dim));
	else
		scan->fs_lut = vs_alloc0(VS_FASTSCAN_LUT_HACC_BYTES(scan->dim));
	scan->fs_accum = vs_alloc(
			PRISM_FASTSCAN_MAX_INFLIGHT * VS_FASTSCAN_GROUP * sizeof(int32_t));

	/* Resolve dispatch once per scan (see posting_scan.h). */
	scan->fs_accum_hacc = vs_fastscan_get_accumulate_hacc();
#if defined(VS_SIMD_FULL) && (defined(__x86_64__) || defined(_M_X64))
	scan->fs_has_avx512 = vs_has_all_simd(VS_SIMD_AVX512_BW);
#else
	scan->fs_has_avx512 = false;
#endif
	scan->fs_cap_first	 = prism_fastscan_max_entries_first(scan->dim);
	scan->fs_cap_over	 = prism_fastscan_max_entries(scan->dim);
	scan->fs_space_first = prism_fastscan_page_space(scan->dim, true);
	scan->fs_space_over	 = prism_fastscan_page_space(scan->dim, false);
}

static bool advance_page(PrismPostingScan *scan);

/*
 * The checks prism_posting_check_count makes on a fastscan page, against
 * bounds cached at enable_fastscan: this runs on every page a scan reads.
 * The page's place in the stream is checked as it is scanned.
 */
static void
fastscan_check_page(
		const PrismPostingScan *scan, const PrismPostingPageOpaque *opaque)
{
	if (scan->fs_lut == NULL)
	{
		prism_posting_check_count(scan->cur_blkno, scan->cur_page, scan->dim);
		return;
	}

	bool	 first	= (opaque->flags & PRISM_POSTING_PAGE_FIRST) != 0;
	uint32_t cap	= first ? scan->fs_cap_first : scan->fs_cap_over;
	uint32_t space	= first ? scan->fs_space_first : scan->fs_space_over;
	uint32_t groups = prism_fastscan_page_groups(opaque);

	if (opaque->entry_count > cap ||
		groups * PRISM_FASTSCAN_META_BYTES + opaque->code_bytes > space ||
		opaque->code_bytes % VS_FASTSCAN_GROUP_BYTES(8) != 0)
		vs_error(
				VS_EXTENSION_NAME
				": fastscan posting page %u does not fit its page (%u "
				"entries, %u code bytes); the index may be corrupted -- "
				"REINDEX it",
				scan->cur_blkno,
				(unsigned)opaque->entry_count,
				(unsigned)opaque->code_bytes);
}

/* ----------------------------------------------------------------
 * Per-cluster begin / end
 * ---------------------------------------------------------------- */

void
prism_posting_scan_begin_cluster(
		PrismPostingScan *scan,
		RaBitQQueryState *qstate,
		BlockNumber		  posting_head)
{
	scan->qstate		  = qstate;
	scan->cur_blkno		  = posting_head;
	scan->cur_page		  = NULL;
	scan->cur_content	  = NULL;
	scan->cur_max_entries = 0;
	scan->cur_count		  = 0;
	scan->fs_lut_valid	  = false;
	scan->fs_stream_pos	  = 0;
	scan->fs_next_group	  = 0;
	scan->pages_read	  = 0;
	scan->pages_skipped	  = 0;
	scan->entries_scanned = 0;
	scan->entries_pruned  = 0;

	/* Eagerly read the first page so pt_centroid is accessible
	 * via prism_posting_pt_centroid(scan->cur_page) before _cluster
	 * is called. */
	if (posting_head != InvalidBlockNumber)
		advance_page(scan);
}

/*
 * Return pt_centroid from the first page (must be called after
 * begin_cluster). Returns NULL if no page was loaded.
 */
const float *
prism_posting_scan_pt_centroid(const PrismPostingScan *scan)
{
	if (scan->cur_page == NULL)
		return NULL;
	return prism_posting_pt_centroid(scan->cur_page);
}

void
prism_posting_scan_begin_flat(
		PrismPostingScan *scan, RaBitQQueryState *qstate, char *flat_buf)
{
	PrismFlatPostingHeader *hdr = prism_flat_posting_header(flat_buf);

	scan->qstate		  = qstate;
	scan->cur_blkno		  = InvalidBlockNumber;
	scan->cur_page		  = flat_buf;
	scan->cur_content	  = prism_flat_posting_content(flat_buf);
	scan->cur_max_entries = hdr->max_entries;
	if (hdr->entry_count > scan->max_entries_cap)
		vs_error(
				VS_EXTENSION_NAME
				": flat posting page has an invalid entry count (%u > %u); "
				"the index may be corrupted -- REINDEX it",
				(unsigned)hdr->entry_count,
				scan->max_entries_cap);
	scan->cur_count		  = hdr->entry_count;
	scan->pages_read	  = 1;
	scan->pages_skipped	  = 0;
	scan->entries_scanned = 0;
	scan->entries_pruned  = 0;
}

void
prism_posting_scan_end_cluster(PrismPostingScan *scan)
{
	/* Release current page if held via storage vtable */
	if (scan->cur_page != NULL && scan->storage != NULL &&
		scan->page_base == NULL)
	{
		vs_storage_release_page(scan->storage, scan->cur_blkno);
	}

	scan->cur_page	  = NULL;
	scan->cur_content = NULL;
	scan->qstate	  = NULL;
}

/* ----------------------------------------------------------------
 * Advance to next page in chain
 * ---------------------------------------------------------------- */

static bool
advance_page(PrismPostingScan *scan)
{
	/* Follow chain from current page */
	if (scan->cur_page != NULL)
	{
		if (scan->storage != NULL)
		{
			scan->cur_blkno = prism_posting_opaque(scan->cur_page)->next_blkno;
		}
		else
		{
			/* Flat mode: single page, no chain */
			scan->cur_blkno = InvalidBlockNumber;
		}
		scan->cur_page	  = NULL;
		scan->cur_content = NULL;
	}

	if (scan->cur_blkno == InvalidBlockNumber)
		return false;

	/* Read next page — inline for ArrayPageStorage. A scan over a
	 * non-paged posting format carries neither a page base nor a storage
	 * (see the flat branch of the standalone query context); it reaches
	 * its single page through begin_flat and has no chain to follow. */
	if (scan->page_base != NULL)
		scan->cur_page = scan->page_base + (size_t)scan->cur_blkno * BLCKSZ;
	else if (scan->storage != NULL)
		scan->cur_page = vs_storage_read_page(scan->storage, scan->cur_blkno);
	else
		return false;

	PrismPostingPageOpaque *opaque = prism_posting_opaque(scan->cur_page);

	/* Validate page identity — catch corrupted chain pointers early */
	if (opaque->page_id != PRISM_POSTING_PAGE_ID)
	{
		vs_warn(VS_EXTENSION_NAME ": posting scan hit non-posting page "
								  "(blkno=%u, page_id=0x%04X)",
				scan->cur_blkno,
				opaque->page_id);
		/* Release the pin taken above before bailing (storage-backed only;
		 * in page_base mode the page is a borrowed pointer, not a pin). */
		if (scan->storage != NULL && scan->page_base == NULL)
			vs_storage_release_page(scan->storage, scan->cur_blkno);
		scan->cur_page	  = NULL;
		scan->cur_content = NULL;
		return false;
	}

	scan->cur_content =
			(opaque->flags & PRISM_POSTING_PAGE_FIRST)
					? prism_posting_content_first(scan->cur_page, scan->dim)
					: prism_posting_content(scan->cur_page);
	scan->cur_max_entries = opaque->max_entries;
	/* Reject an entry_count past what a valid page of this format can
	 * hold. AoS pages drive the page_distances/page_scratch batch (sized
	 * to max_entries_cap), so an over-count there is an out-of-bounds
	 * write. Fastscan pages are scored per 32-vector group, and their
	 * metadata blocks and code run must fit the page. */
	if (opaque->flags & PRISM_POSTING_PAGE_FASTSCAN)
		fastscan_check_page(scan, opaque);
	else if (opaque->entry_count > scan->max_entries_cap)
		vs_error(
				VS_EXTENSION_NAME
				": posting page %u has an invalid entry count (%u > %u); "
				"the index may be corrupted -- REINDEX it",
				scan->cur_blkno,
				(unsigned)opaque->entry_count,
				scan->max_entries_cap);
	scan->cur_count = opaque->entry_count;
	scan->pages_read++;
	return true;
}

/* ----------------------------------------------------------------
 * Score + prune a full cluster into topk
 *
 * Inserts survivors with approximate (estimated) distances and
 * error bounds. No exact reranking — the caller does that after
 * extracting candidates from topk.
 * ---------------------------------------------------------------- */

void
prism_posting_scan_cluster(PrismPostingScan *scan, VsTopK *topk)
{
	Dimension dim		 = scan->dim;
	uint32_t  entry_size = PRISM_POSTING_ENTRY_SIZE(dim);

	float g_add		 = scan->qstate->g_add;
	float sum_t		 = scan->qstate->sum_transformed;
	float inv_sqrt_d = scan->qstate->inv_sqrt_d;
	float g_error	 = scan->qstate->g_error;

	Distance *distances = scan->page_distances;
	float	 *scratch	= scan->page_scratch;

	/* Process pages until chain is exhausted.
	 * For flat mode: single iteration (one page, no chain). */
	for (;;)
	{
		/* Advance to next page */
		if (scan->cur_page == NULL)
		{
			if (!advance_page(scan))
				break;
		}

		/* A page tombstoned by VACUUM (all entries dead) is skipped — no
		 * scoring kernel; the chain-follow below still advances past it. A
		 * retired chain also carries TOMBSTONED once reclaimed, but its
		 * entries moved rather than died, so it is still scored. */
		if (!prism_posting_page_all_dead(prism_posting_opaque(scan->cur_page)))
		{
			char	*content = scan->cur_content;
			uint32_t count	 = scan->cur_count;

			/* --- Score: batch IP over all entries on this page ---
			 *
			 * AoS layout: entry i's bits live at content + i * entry_size
			 * + PRISM_POSTING_ENTRY_BITS_OFFSET. The SIMD kernel just needs
			 * the first entry's bits pointer and a stride of entry_size. */
			const uint8_t *bits_base = prism_posting_first_bits(content);

			uint32_t padded = (count + 3) & ~3u;
			vs_rabitq_inner_product_multi(
					scan->qstate->transformed,
					bits_base,
					entry_size,
					dim,
					padded,
					scratch);

			/* Convert raw IPs to distances, reading f_add/f_rescale per
			 * entry via the strided AoS accessor. */
			for (uint32_t i = 0; i < count; i++)
			{
				PrismPostingEntryHeader *e =
						prism_posting_entry_at(content, i, dim);
				float final_dot = (2.0f * scratch[i] - sum_t) * inv_sqrt_d;
				distances[i]	= e->f_add + g_add -
							   2.0f * e->f_rescale * final_dot;
			}

			/* --- Prune + insert approximate distances --- */
			Distance threshold = vs_topk_threshold(topk);

			for (uint32_t i = 0; i < count; i++)
			{
				PrismPostingEntryHeader *e =
						prism_posting_entry_at(content, i, dim);
				scan->entries_scanned++;

				if (e->meta.flags & PRISM_POSTING_FLAG_DELETED)
				{
					scan->entries_pruned++;
					continue;
				}

				Distance est = distances[i];
				Distance err = e->f_error * g_error;
				Distance lb	 = est - err;

				if (lb >= threshold)
				{
					scan->entries_pruned++;
					continue;
				}

				uint64_t id = prism_posting_encode_tid(&e->meta.tid);
				vs_topk_insert(topk, est, err, id);
				threshold = vs_topk_threshold(topk);
			}
		} /* end: page not tombstoned */
		else
			scan->pages_skipped++;

		/* Follow chain: read next_blkno, then release current */
		BlockNumber prev_blkno = scan->cur_blkno;

		if (scan->storage != NULL)
			scan->cur_blkno = prism_posting_opaque(scan->cur_page)->next_blkno;
		else
			scan->cur_blkno = InvalidBlockNumber;

		if (scan->storage != NULL && scan->page_base == NULL)
			vs_storage_release_page(scan->storage, prev_blkno);

		scan->cur_page	  = NULL;
		scan->cur_content = NULL;
	}
}

/* ----------------------------------------------------------------
 * Fastscan: vectorized distance + prune for 16 entries
 * ---------------------------------------------------------------- */

#ifdef VS_SIMD_FULL
#if defined(__x86_64__) || defined(_M_X64)

/*
 * Compute distances and lower bounds for 16 entries using AVX-512.
 * Returns a bitmask of survivors (lb < threshold).
 */
VS_TARGET_AVX512 static inline __mmask16
fastscan_prune_16(
		const int32_t *accum,
		const float	  *f_add,
		const float	  *f_rescale,
		const float	  *f_error,
		float		  *est_out,
		float		  *err_out,
		float		   lut_scale,
		float		   lut_bias,
		float		   sum_t,
		float		   inv_sqrt_d,
		float		   g_add,
		float		   g_error,
		float		   threshold,
		__mmask16	   live)
{
	/* Convert 16 int32 accumulators to float */
	__m512i acc32 = _mm512_loadu_si512((const __m512i *)accum);
	__m512	acc_f = _mm512_cvtepi32_ps(acc32);

	/* binary_ip = acc * scale + bias */
	__m512 scale_v = _mm512_set1_ps(lut_scale);
	__m512 bias_v  = _mm512_set1_ps(lut_bias);
	__m512 ip	   = _mm512_fmadd_ps(acc_f, scale_v, bias_v);

	/* final_dot = (2*ip - sum_t) * inv_sqrt_d */
	__m512 two_v	  = _mm512_set1_ps(2.0f);
	__m512 sum_t_v	  = _mm512_set1_ps(sum_t);
	__m512 inv_sqrt_v = _mm512_set1_ps(inv_sqrt_d);
	__m512 final_dot =
			_mm512_mul_ps(_mm512_fmsub_ps(two_v, ip, sum_t_v), inv_sqrt_v);

	/* est = f_add + g_add - 2 * f_rescale * final_dot */
	__m512 fa	 = _mm512_loadu_ps(f_add);
	__m512 fr	 = _mm512_loadu_ps(f_rescale);
	__m512 gadd	 = _mm512_set1_ps(g_add);
	__m512 est_v = _mm512_add_ps(fa, gadd);
	est_v = _mm512_fnmadd_ps(_mm512_mul_ps(two_v, fr), final_dot, est_v);

	/* err = f_error * g_error */
	__m512 fe	  = _mm512_loadu_ps(f_error);
	__m512 gerr_v = _mm512_set1_ps(g_error);
	__m512 err_v  = _mm512_mul_ps(fe, gerr_v);

	/* lb = est - err; survivors = lb < threshold, restricted to live lanes --
	 * the masked compare clears a dead lane's result bit regardless of score.
	 */
	__m512	  lb_v	= _mm512_sub_ps(est_v, err_v);
	__m512	  thr_v = _mm512_set1_ps(threshold);
	__mmask16 surv	= _mm512_mask_cmp_ps_mask(live, lb_v, thr_v, _CMP_LT_OS);

	_mm512_storeu_ps(est_out, est_v);
	_mm512_storeu_ps(err_out, err_v);

	return surv;
}

/*
 * AVX-512 vectorized distance + prune for one 32-vector group.
 * Must be in a target-attributed function so the compiler only
 * emits AVX-512 instructions here, not in the caller.
 */
VS_TARGET_AVX512 static void
fastscan_prune_group_avx512(
		PrismPostingScan *scan,
		VsTopK			 *topk,
		const int32_t	 *accum,
		ItemPointerData	 *tids,
		const float		 *f_add,
		const float		 *f_rescale,
		const float		 *f_error,
		float			  lut_scale,
		float			  lut_bias,
		float			  sum_t,
		float			  inv_sqrt_d,
		float			  g_add,
		float			  g_error,
		uint32_t		  tombstone_mask,
		Distance		 *threshold_p)
{
	Distance threshold = *threshold_p;
	float	 est_buf[VS_FASTSCAN_GROUP];
	float	 err_buf[VS_FASTSCAN_GROUP];
	uint32_t live_mask = ~tombstone_mask;

	for (uint32_t h = 0; h < 2; h++)
	{
		uint32_t  off  = h * 16;
		__mmask16 live = (__mmask16)((live_mask >> off) & 0xFFFFu);
		__mmask16 surv = fastscan_prune_16(
				accum + off,
				f_add + off,
				f_rescale + off,
				f_error + off,
				est_buf + off,
				err_buf + off,
				lut_scale,
				lut_bias,
				sum_t,
				inv_sqrt_d,
				g_add,
				g_error,
				threshold,
				live);

		if (surv == 0)
		{
			scan->entries_pruned += 16;
			continue;
		}

		scan->entries_pruned += 16 - _mm_popcnt_u32(surv);

		while (surv != 0)
		{
			uint32_t v = off + __builtin_ctz(surv);
			surv &= surv - 1;

			uint64_t id = prism_posting_encode_tid(&tids[v]);
			vs_topk_insert(topk, est_buf[v], err_buf[v], id);
			threshold = vs_topk_threshold(topk);
		}
	}

	*threshold_p = threshold;
}

#endif
#endif

/* ----------------------------------------------------------------
 * Fastscan variant: VPSHUFB kernel for fastscan-format pages
 *
 * Walks the list's code stream page by page (see the fastscan page format
 * in posting_page.h). A group's running sums carry across pages until its
 * last code byte and its metadata block have both been read, then the
 * group is scored. Falls back to the AoS kernel for non-fastscan pages
 * (mixed chains).
 * ---------------------------------------------------------------- */

static void
fastscan_corrupt(const PrismPostingScan *scan, const char *what)
{
	vs_error(
			VS_EXTENSION_NAME
			": fastscan posting page %u breaks its list's code stream (%s); "
			"the index may be corrupted -- REINDEX it",
			scan->cur_blkno,
			what);
}

static inline int32_t *
fastscan_group_accum(const PrismPostingScan *scan, uint32_t group)
{
	return scan->fs_accum +
		   (size_t)(group % PRISM_FASTSCAN_MAX_INFLIGHT) * VS_FASTSCAN_GROUP;
}

/* Build the query LUT once per cluster, on its first fastscan page. */
static void
fastscan_ensure_lut(PrismPostingScan *scan)
{
	if (scan->fs_lut_valid)
		return;
	if (scan->fs_lut_bits == 8)
		vs_fastscan_build_lut(
				scan->qstate->transformed,
				scan->dim,
				scan->fs_lut,
				&scan->fs_lut_scale,
				&scan->fs_lut_bias);
	else
		vs_fastscan_build_lut_hacc(
				scan->qstate->transformed,
				scan->dim,
				scan->fs_lut,
				&scan->fs_lut_scale,
				&scan->fs_lut_bias);
	scan->fs_lut_valid = true;
}

/*
 * Add one run of a group's codes into its running sums. A group's first
 * run starts them: when it fits one kernel call, the kernel writes the sums
 * directly, which is the whole job for a group that sits on one page.
 */
__attribute__((always_inline)) static inline void
fastscan_accumulate_run(
		PrismPostingScan *scan,
		const uint8_t	 *codes,
		uint32_t		  offset,
		uint32_t		  len,
		int32_t			 *accum)
{
	uint32_t col_bytes = VS_FASTSCAN_GROUP_BYTES(8);
	uint32_t col	   = offset / col_bytes;
	uint32_t ncols	   = len / col_bytes;

	if (offset == 0 && ncols <= VS_FASTSCAN_MAX_KERNEL_COLS)
	{
		Dimension run_dim = (Dimension)(ncols * 8);

		if (scan->fs_lut_bits == 8)
		{
			uint16_t acc16[VS_FASTSCAN_GROUP];
			vs_fastscan_accumulate(codes, scan->fs_lut, acc16, run_dim);
			for (uint32_t v = 0; v < VS_FASTSCAN_GROUP; v++)
				accum[v] = acc16[v];
		}
		else
			scan->fs_accum_hacc(codes, scan->fs_lut, accum, run_dim);
		return;
	}

	if (offset == 0)
		memset(accum, 0, VS_FASTSCAN_GROUP * sizeof(int32_t));
	if (scan->fs_lut_bits == 8)
		vs_fastscan_accumulate_range(codes, scan->fs_lut, col, ncols, accum);
	else
		vs_fastscan_accumulate_hacc_range(
				scan->fs_accum_hacc, codes, scan->fs_lut, col, ncols, accum);
}

/* What scoring the groups of one fastscan page needs, gathered once. */
typedef struct FastscanPageCtx
{
	VsTopK	*topk;
	char	*content;
	uint32_t first_group; /* list-wide index of the page's first block */
	uint32_t meta_end;	  /* one past its last */
	uint32_t count;		  /* vectors in the page's blocks */
	uint32_t next_group;  /* next group to score */
	float	 lut_scale;
	float	 lut_bias;
	float	 g_add;
	float	 sum_t;
	float	 inv_sqrt_d;
	float	 g_error;
	Distance threshold;
} FastscanPageCtx;

/* Prune the next group, whose running sums are complete, and insert
 * survivors. */
__attribute__((always_inline)) static inline void
fastscan_score_group(PrismPostingScan *scan, FastscanPageCtx *ctx)
{
	Dimension dim	  = scan->dim;
	uint32_t  g		  = ctx->next_group - ctx->first_group;
	uint32_t  g_count = ctx->count - g * VS_FASTSCAN_GROUP;
	if (g_count > VS_FASTSCAN_GROUP)
		g_count = VS_FASTSCAN_GROUP;

	const int32_t	*accum = fastscan_group_accum(scan, ctx->next_group);
	ItemPointerData *tids  = prism_fastscan_group_tids(ctx->content, g, dim);
	float			*f_add = prism_fastscan_group_f_add(ctx->content, g, dim);
	float	*f_rescale = prism_fastscan_group_f_rescale(ctx->content, g, dim);
	float	*f_error   = prism_fastscan_group_f_error(ctx->content, g, dim);
	uint32_t tombstone_mask =
			*prism_fastscan_group_tombstone_mask(ctx->content, g, dim);

	/* Vectorized distance + prune: compute 16 distances at a
	 * time, mask survivors, iterate only the few that pass. */
	scan->entries_scanned += g_count;

#ifdef VS_SIMD_FULL
#if defined(__x86_64__) || defined(_M_X64)
	if (vs_likely(g_count == VS_FASTSCAN_GROUP) && scan->fs_has_avx512)
	{
		fastscan_prune_group_avx512(
				scan,
				ctx->topk,
				accum,
				tids,
				f_add,
				f_rescale,
				f_error,
				ctx->lut_scale,
				ctx->lut_bias,
				ctx->sum_t,
				ctx->inv_sqrt_d,
				ctx->g_add,
				ctx->g_error,
				tombstone_mask,
				&ctx->threshold);
		return;
	}
#endif
#endif

	/* Scalar fallback for partial groups (and non-AVX512 CPUs,
	 * NEON included): no masked-compare primitive to fold the
	 * exclusion into, so a dead lane is skipped explicitly, the
	 * same as a DELETED AoS entry. */
	Distance threshold = ctx->threshold;
	for (uint32_t v = 0; v < g_count; v++)
	{
		if (prism_fastscan_lane_is_tombstoned(tombstone_mask, v))
		{
			scan->entries_pruned++;
			continue;
		}

		float	 binary_ip = (float)accum[v] * ctx->lut_scale + ctx->lut_bias;
		float	 final_dot = (2.0f * binary_ip - ctx->sum_t) * ctx->inv_sqrt_d;
		Distance est = f_add[v] + ctx->g_add - 2.0f * f_rescale[v] * final_dot;
		Distance err = f_error[v] * ctx->g_error;
		Distance lb	 = est - err;

		if (lb >= threshold)
		{
			scan->entries_pruned++;
			continue;
		}

		uint64_t id = prism_posting_encode_tid(&tids[v]);
		vs_topk_insert(ctx->topk, est, err, id);
		threshold = vs_topk_threshold(ctx->topk);
	}
	ctx->threshold = threshold;
}

/*
 * Score the groups whose metadata is on this page and whose codes are
 * complete, up to (not including) list-wide group `done`.
 */
__attribute__((always_inline)) static inline void
fastscan_score_ready(
		PrismPostingScan *scan, FastscanPageCtx *ctx, uint32_t done)
{
	while (ctx->next_group < done && ctx->next_group < ctx->meta_end)
	{
		fastscan_score_group(scan, ctx);
		ctx->next_group++;
	}
}

static void
scan_fastscan_page(PrismPostingScan *scan, VsTopK *topk)
{
	char				   *content	 = scan->cur_content;
	PrismPostingPageOpaque *opaque	 = prism_posting_opaque(scan->cur_page);
	uint32_t				code_len = VS_FASTSCAN_GROUP_BYTES(scan->dim);
	uint32_t				ngroups	 = prism_fastscan_page_groups(opaque);

	/* The page continues the stream where the previous one stopped. */
	uint32_t	   start	   = scan->fs_stream_pos;
	uint32_t	   end		   = start + opaque->code_bytes;
	uint32_t	   first_group = scan->fs_next_group;
	uint32_t	   meta_end	   = first_group + ngroups;
	uint32_t	   pos		   = start;
	const uint8_t *codes	   = prism_fastscan_page_codes(content, ngroups);

	/*
	 * A page with only dead entries is skipped when everything on it belongs
	 * to its own groups: its codes end on a group boundary and every group
	 * they touch has its metadata here. Otherwise a live group's codes or
	 * running sums pass through the page and it must be read.
	 */
	if (prism_posting_page_all_dead(opaque) && end % code_len == 0 &&
		meta_end == end / code_len)
	{
		scan->fs_stream_pos = end;
		scan->fs_next_group = meta_end;
		scan->pages_skipped++;
		return;
	}

	fastscan_ensure_lut(scan);

	FastscanPageCtx ctx = {
			.topk		 = topk,
			.content	 = content,
			.first_group = first_group,
			.meta_end	 = meta_end,
			.count		 = scan->cur_count,
			.next_group	 = scan->fs_next_group,
			.lut_scale	 = scan->fs_lut_scale,
			.lut_bias	 = scan->fs_lut_bias,
			.g_add		 = scan->qstate->g_add,
			.sum_t		 = scan->qstate->sum_transformed,
			.inv_sqrt_d	 = scan->qstate->inv_sqrt_d,
			.g_error	 = scan->qstate->g_error,
			.threshold	 = vs_topk_threshold(topk),
	};

	/* The group the page's codes start in, and how far into it. */
	uint32_t g	 = pos / code_len;
	uint32_t off = pos % code_len;

	/* Groups whose codes ended on an earlier page. */
	fastscan_score_ready(scan, &ctx, g);

	while (pos < end)
	{
		const uint8_t *run_codes = codes + (pos - start);

		/*
		 * The common case: a whole group on this page with its metadata
		 * here and nothing waiting before it, scored straight away.
		 */
		if (off == 0 && end - pos >= code_len && ctx.next_group == g &&
			g < meta_end)
		{
			fastscan_accumulate_run(
					scan,
					run_codes,
					0,
					code_len,
					fastscan_group_accum(scan, g));
			pos += code_len;

			/* Fetch the next group's codes while this one is pruned. */
			uint32_t ahead = end - pos < code_len ? end - pos : code_len;
			for (uint32_t p = 0; p < ahead; p += PRISM_FASTSCAN_CHUNK_BYTES)
				__builtin_prefetch(run_codes + code_len + p, 0, 1);
			fastscan_score_group(scan, &ctx);
			ctx.next_group++;
			g++;
			continue;
		}

		uint32_t run = code_len - off;
		if (run > end - pos)
			run = end - pos;

		if (g - ctx.next_group >= PRISM_FASTSCAN_MAX_INFLIGHT)
			fastscan_corrupt(scan, "too many groups in flight");

		fastscan_accumulate_run(
				scan, run_codes, off, run, fastscan_group_accum(scan, g));
		pos += run;

		if (off + run < code_len)
			off += run;
		else
		{
			fastscan_score_ready(scan, &ctx, g + 1);
			g++;
			off = 0;
		}
	}

	if (ctx.next_group != meta_end)
		fastscan_corrupt(scan, "metadata ahead of its codes");
	scan->fs_next_group = ctx.next_group;
	scan->fs_stream_pos = end;
}

/* The fastscan pages of a list must leave no group half-read. */
static void
fastscan_check_stream_end(const PrismPostingScan *scan)
{
	uint32_t code_len = VS_FASTSCAN_GROUP_BYTES(scan->dim);

	if (scan->fs_stream_pos != scan->fs_next_group * code_len)
		vs_error(
				VS_EXTENSION_NAME
				": a fastscan posting list ends inside a group (%u code "
				"bytes for %u groups); the index may be corrupted -- "
				"REINDEX it",
				scan->fs_stream_pos,
				scan->fs_next_group);
}

void
prism_posting_scan_cluster_fastscan(PrismPostingScan *scan, VsTopK *topk)
{
	/* Process pages until chain is exhausted */
	for (;;)
	{
		if (scan->cur_page == NULL)
		{
			if (!advance_page(scan))
				break;
		}

		PrismPostingPageOpaque *opaque = prism_posting_opaque(scan->cur_page);

		if (opaque->flags & PRISM_POSTING_PAGE_FASTSCAN)
			scan_fastscan_page(scan, topk);
		else
		{
			/* Inserted AoS pages follow the fastscan ones; the AoS scan
			 * consumes the rest of the chain. */
			fastscan_check_stream_end(scan);
			if (prism_posting_page_all_dead(opaque))
			{
				/* All entries dead -- skip scoring; the chain-follow
				 * advances. A retired chain is not that: it carries
				 * TOMBSTONED too once reclaimed, but its entries were
				 * rewritten elsewhere, and a scan that followed a stale
				 * leaf pointer into it must still see them. */
				scan->pages_skipped++;
			}
			else
				prism_posting_scan_cluster(scan, topk);
		}

		/* If AoS fallback consumed the entire chain, we're done */
		if (scan->cur_page == NULL && scan->cur_blkno == InvalidBlockNumber)
			return;

		/* Follow chain: read next_blkno, then release current */
		if (scan->cur_page != NULL)
		{
			BlockNumber prev_blkno = scan->cur_blkno;
			scan->cur_blkno		   = opaque->next_blkno;

			if (scan->storage != NULL && scan->page_base == NULL)
				vs_storage_release_page(scan->storage, prev_blkno);

			scan->cur_page	  = NULL;
			scan->cur_content = NULL;
		}
	}
	fastscan_check_stream_end(scan);
}
