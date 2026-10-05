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
	scan->fs_accum = vs_alloc(VS_FASTSCAN_GROUP * sizeof(int32_t));

	/* Resolve dispatch once per scan (see posting_scan.h). */
	scan->fs_accum_hacc = vs_fastscan_get_accumulate_hacc();
#if defined(VS_SIMD_FULL) && (defined(__x86_64__) || defined(_M_X64))
	scan->fs_has_avx512 = vs_has_all_simd(VS_SIMD_AVX512_BW);
#else
	scan->fs_has_avx512 = false;
#endif
	scan->fs_max_groups_first = prism_fastscan_max_groups(scan->dim, true);
	scan->fs_max_groups_over  = prism_fastscan_max_groups(scan->dim, false);
}

static bool advance_page(PrismPostingScan *scan);

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

	/* Read next page — inline for ArrayPageStorage */
	if (scan->page_base != NULL)
		scan->cur_page = scan->page_base + (size_t)scan->cur_blkno * BLCKSZ;
	else
		scan->cur_page = vs_storage_read_page(scan->storage, scan->cur_blkno);

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
	 * write; fastscan pages are scored per 32-vector group and hold more
	 * entries per page, so bound them by the fastscan capacity instead of
	 * the smaller AoS one. */
	uint32_t page_cap;
	if (opaque->flags & PRISM_POSTING_PAGE_FASTSCAN)
		page_cap = (opaque->flags & PRISM_POSTING_PAGE_FIRST)
						 ? prism_fastscan_max_entries_first(scan->dim)
						 : prism_fastscan_max_entries(scan->dim);
	else
		page_cap = scan->max_entries_cap;
	if (opaque->entry_count > page_cap)
		vs_error(
				VS_EXTENSION_NAME
				": posting page %u has an invalid entry count (%u > %u); "
				"the index may be corrupted -- REINDEX it",
				scan->cur_blkno,
				(unsigned)opaque->entry_count,
				page_cap);
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
				scan->fs_accum + off,
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
 * Processes 32-vector groups via table-lookup accumulation.
 * Falls back to the AoS kernel for non-fastscan pages (mixed
 * chains).
 * ---------------------------------------------------------------- */

static void
scan_fastscan_page(PrismPostingScan *scan, VsTopK *topk)
{
	Dimension dim	  = scan->dim;
	uint32_t  count	  = scan->cur_count;
	char	 *content = scan->cur_content;

	PrismPostingPageOpaque *opaque = prism_posting_opaque(scan->cur_page);

	/* Fall back to AoS kernel for non-fastscan pages */
	if (!(opaque->flags & PRISM_POSTING_PAGE_FASTSCAN))
		return; /* caller handles AoS via the standard path */

	/* Build LUT from transformed query (once per cluster) */
	if (!scan->fs_lut_valid)
	{
		if (scan->fs_lut_bits == 8)
			vs_fastscan_build_lut(
					scan->qstate->transformed,
					dim,
					scan->fs_lut,
					&scan->fs_lut_scale,
					&scan->fs_lut_bias);
		else
			vs_fastscan_build_lut_hacc(
					scan->qstate->transformed,
					dim,
					scan->fs_lut,
					&scan->fs_lut_scale,
					&scan->fs_lut_bias);
		scan->fs_lut_valid = true;
	}
	float lut_scale = scan->fs_lut_scale;
	float lut_bias	= scan->fs_lut_bias;

	float	 g_add		= scan->qstate->g_add;
	float	 sum_t		= scan->qstate->sum_transformed;
	float	 inv_sqrt_d = scan->qstate->inv_sqrt_d;
	float	 g_error	= scan->qstate->g_error;
	uint32_t max_groups = (opaque->flags & PRISM_POSTING_PAGE_FIRST)
								? scan->fs_max_groups_first
								: scan->fs_max_groups_over;
	uint32_t ngroups	= (count + VS_FASTSCAN_GROUP - 1) / VS_FASTSCAN_GROUP;
	if (ngroups > max_groups)
		ngroups = max_groups;

	Distance threshold = vs_topk_threshold(topk);

	for (uint32_t g = 0; g < ngroups; g++)
	{
		uint32_t g_start = g * VS_FASTSCAN_GROUP;
		uint32_t g_count = count - g_start;
		if (g_count > VS_FASTSCAN_GROUP)
			g_count = VS_FASTSCAN_GROUP;

		/* Access group data — compute base once */
		char *gbase = prism_fastscan_group_base(content, g, dim);

		ItemPointerData *tids	   = (ItemPointerData *)gbase;
		float			*f_add	   = (float *)(gbase +
								   VS_FASTSCAN_GROUP * sizeof(ItemPointerData));
		float			*f_rescale = f_add + VS_FASTSCAN_GROUP;
		float			*f_error   = f_rescale + VS_FASTSCAN_GROUP;
		uint8_t			*codes	   = (uint8_t *)(f_error + VS_FASTSCAN_GROUP);
		uint32_t		 tombstone_mask = *(uint32_t *)(codes +
												VS_FASTSCAN_GROUP_BYTES(dim));

		/* Run VPSHUFB accumulate kernel */
		if (scan->fs_lut_bits == 8)
		{
			uint16_t acc16[VS_FASTSCAN_GROUP];
			vs_fastscan_accumulate(codes, scan->fs_lut, acc16, dim);
			for (uint32_t v2 = 0; v2 < VS_FASTSCAN_GROUP; v2++)
				scan->fs_accum[v2] = acc16[v2];
		}
		else
		{
			scan->fs_accum_hacc(codes, scan->fs_lut, scan->fs_accum, dim);
		}

		/* Prefetch next group's codes during prune phase */
		if (g + 1 < ngroups)
		{
			uint8_t *next_codes =
					prism_fastscan_group_codes(content, g + 1, dim);
			uint32_t code_bytes = VS_FASTSCAN_GROUP_BYTES(dim);
			for (uint32_t p = 0; p < code_bytes; p += 64)
				__builtin_prefetch(next_codes + p, 0, 1);
		}

		/* Vectorized distance + prune: compute 16 distances at a
		 * time, mask survivors, iterate only the few that pass. */
		scan->entries_scanned += g_count;

#ifdef VS_SIMD_FULL
#if defined(__x86_64__) || defined(_M_X64)
		if (vs_likely(g_count == VS_FASTSCAN_GROUP) && scan->fs_has_avx512)
		{
			fastscan_prune_group_avx512(
					scan,
					topk,
					tids,
					f_add,
					f_rescale,
					f_error,
					lut_scale,
					lut_bias,
					sum_t,
					inv_sqrt_d,
					g_add,
					g_error,
					tombstone_mask,
					&threshold);
		}
		else
#endif
#endif
		{
			/* Scalar fallback for partial groups (and non-AVX512 CPUs,
			 * NEON included): no masked-compare primitive to fold the
			 * exclusion into, so a dead lane is skipped explicitly, the
			 * same as a DELETED AoS entry below. */
			for (uint32_t v = 0; v < g_count; v++)
			{
				if (prism_fastscan_lane_is_tombstoned(tombstone_mask, v))
				{
					scan->entries_pruned++;
					continue;
				}

				float binary_ip = (float)scan->fs_accum[v] * lut_scale +
								  lut_bias;
				float	 final_dot = (2.0f * binary_ip - sum_t) * inv_sqrt_d;
				Distance est	   = f_add[v] + g_add -
							   2.0f * f_rescale[v] * final_dot;
				Distance err = f_error[v] * g_error;
				Distance lb	 = est - err;

				if (lb >= threshold)
				{
					scan->entries_pruned++;
					continue;
				}

				uint64_t id = prism_posting_encode_tid(&tids[v]);
				vs_topk_insert(topk, est, err, id);
				threshold = vs_topk_threshold(topk);
			}
		}
	}
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

		if (prism_posting_page_all_dead(opaque))
		{
			/* All entries dead — skip scoring; the chain-follow advances.
			 * A retired chain is not that: it carries TOMBSTONED too once
			 * reclaimed, but its entries were rewritten elsewhere, and a
			 * scan that followed a stale leaf pointer into it must still
			 * see them. */
			scan->pages_skipped++;
		}
		else if (opaque->flags & PRISM_POSTING_PAGE_FASTSCAN)
			scan_fastscan_page(scan, topk);
		else
			prism_posting_scan_cluster(scan, topk);

		/* If AoS fallback consumed the entire chain, we're done */
		if (scan->cur_page == NULL && scan->cur_blkno == InvalidBlockNumber)
			break;

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
}
