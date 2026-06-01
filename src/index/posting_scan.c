/*
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

#include "mkt_config.h"

#include <string.h>

#include "core/log.h"
#include "core/memory.h"
#include "core/platform.h"
#include "index/posting_scan.h"
#include "quant/fastscan.h"

#ifdef MKT_SIMD_FULL
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>

#include "algo/simd_utils.h"
#elif defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif
#endif

/* ----------------------------------------------------------------
 * Per-query seen-TID hash (open addressing + generation counter)
 *
 * Used to dedup vectors replicated across clusters (SOAR or
 * boundary-epsilon replication). The hash is sized at scan setup
 * so we have power-of-two capacity; a single-step linear probe
 * keeps the inner loop branch-light and the slots stay near the
 * initial bucket because the load factor is < 0.5 by construction.
 *
 * Returns true if the tid is already in the set (skip this
 * survivor); false otherwise and inserts it.
 * ---------------------------------------------------------------- */

static inline bool
seen_tid_check_and_insert(MktPostingScan *scan, uint64_t tid)
{
	if (scan->seen_tids == NULL)
		return false;

	/* splitmix64-style mixer for good distribution from sparse TIDs. */
	uint64_t h = tid * 0x9E3779B97F4A7C15ULL;
	h ^= h >> 33;
	h *= 0xff51afd7ed558ccdULL;
	h ^= h >> 33;

	uint32_t  mask = scan->seen_tids_cap - 1;
	uint32_t  slot = (uint32_t)h & mask;
	uint32_t  gen  = scan->seen_gen;
	uint64_t *tids = scan->seen_tids;
	uint32_t *gens = scan->seen_gens;

	for (;;)
	{
		if (gens[slot] != gen)
		{
			tids[slot] = tid;
			gens[slot] = gen;
			return false;
		}
		if (tids[slot] == tid)
			return true;
		slot = (slot + 1) & mask;
	}
}

/* ----------------------------------------------------------------
 * Init / cleanup
 * ---------------------------------------------------------------- */

void
mkt_posting_scan_init(
		MktPostingScan	   *scan,
		MktStorage		   *storage,
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
	scan->packed_bytes = MKT_RABITQ_BYTES(dim);
	scan->cur_blkno	   = InvalidBlockNumber;

	/* Pre-allocate batch buffers — pad to multiple of 4 for IP kernel */
	uint32_t padded		 = (max_entries_per_page + 3) & ~3u;
	scan->page_distances = mkt_alloc(padded * sizeof(Distance));
	scan->page_scratch	 = mkt_alloc(padded * sizeof(float));
}

void
mkt_posting_scan_cleanup(MktPostingScan *scan)
{
	/* Release pinned page if still held */
	if (scan->cur_page != NULL && scan->storage != NULL &&
		scan->page_base == NULL)
	{
		mkt_storage_release_page(scan->storage, scan->cur_blkno);
		scan->cur_page = NULL;
	}

	mkt_free(scan->page_distances);
	mkt_free(scan->page_scratch);
	scan->page_distances = NULL;
	scan->page_scratch	 = NULL;

	if (scan->fs_lut != NULL)
	{
		mkt_free(scan->fs_lut);
		scan->fs_lut = NULL;
	}
	if (scan->fs_accum != NULL)
	{
		mkt_free(scan->fs_accum);
		scan->fs_accum = NULL;
	}
}

void
mkt_posting_scan_enable_fastscan(MktPostingScan *scan, int lut_bits)
{
	scan->fs_lut_bits = lut_bits;
	if (lut_bits == 8)
		scan->fs_lut = mkt_alloc0(MKT_FASTSCAN_LUT_BYTES(scan->dim));
	else
		scan->fs_lut = mkt_alloc0(MKT_FASTSCAN_LUT_HACC_BYTES(scan->dim));
	scan->fs_accum = mkt_alloc(MKT_FASTSCAN_GROUP * sizeof(int32_t));
}

static bool advance_page(MktPostingScan *scan);

/* ----------------------------------------------------------------
 * Per-cluster begin / end
 * ---------------------------------------------------------------- */

void
mkt_posting_scan_begin_cluster(
		MktPostingScan	 *scan,
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
	scan->entries_scanned = 0;
	scan->entries_pruned  = 0;

	/* Eagerly read the first page so pt_centroid is accessible
	 * via mkt_posting_pt_centroid(scan->cur_page) before _cluster
	 * is called. */
	if (posting_head != InvalidBlockNumber)
		advance_page(scan);
}

/*
 * Return pt_centroid from the first page (must be called after
 * begin_cluster). Returns NULL if no page was loaded.
 */
const float *
mkt_posting_scan_pt_centroid(const MktPostingScan *scan)
{
	if (scan->cur_page == NULL)
		return NULL;
	return mkt_posting_pt_centroid(scan->cur_page);
}

void
mkt_posting_scan_begin_flat(
		MktPostingScan *scan, RaBitQQueryState *qstate, char *flat_buf)
{
	MktFlatPostingHeader *hdr = mkt_flat_posting_header(flat_buf);

	scan->qstate		  = qstate;
	scan->cur_blkno		  = InvalidBlockNumber;
	scan->cur_page		  = flat_buf;
	scan->cur_content	  = mkt_flat_posting_content(flat_buf);
	scan->cur_max_entries = hdr->max_entries;
	scan->cur_count		  = hdr->entry_count;
	scan->pages_read	  = 1;
	scan->entries_scanned = 0;
	scan->entries_pruned  = 0;
}

void
mkt_posting_scan_end_cluster(MktPostingScan *scan)
{
	/* Release current page if held via storage vtable */
	if (scan->cur_page != NULL && scan->storage != NULL &&
		scan->page_base == NULL)
	{
		mkt_storage_release_page(scan->storage, scan->cur_blkno);
	}

	scan->cur_page	  = NULL;
	scan->cur_content = NULL;
	scan->qstate	  = NULL;
}

/* ----------------------------------------------------------------
 * Advance to next page in chain
 * ---------------------------------------------------------------- */

static bool
advance_page(MktPostingScan *scan)
{
	/* Follow chain from current page */
	if (scan->cur_page != NULL)
	{
		if (scan->storage != NULL)
		{
			scan->cur_blkno = mkt_posting_opaque(scan->cur_page)->next_blkno;
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
		scan->cur_page = mkt_storage_read_page(scan->storage, scan->cur_blkno);

	MktPostingPageOpaque *opaque = mkt_posting_opaque(scan->cur_page);

	/* Validate page identity — catch corrupted chain pointers early */
	if (opaque->page_id != MKT_POSTING_PAGE_ID)
	{
		mkt_warn(
				"meerkat: posting scan hit non-posting page "
				"(blkno=%u, page_id=0x%04X)",
				scan->cur_blkno,
				opaque->page_id);
		/* Release the pin taken above before bailing (storage-backed only;
		 * in page_base mode the page is a borrowed pointer, not a pin). */
		if (scan->storage != NULL && scan->page_base == NULL)
			mkt_storage_release_page(scan->storage, scan->cur_blkno);
		scan->cur_page	  = NULL;
		scan->cur_content = NULL;
		return false;
	}

	scan->cur_content =
			(opaque->flags & MKT_POSTING_PAGE_FIRST)
					? mkt_posting_content_first(scan->cur_page, scan->dim)
					: mkt_posting_content(scan->cur_page);
	scan->cur_max_entries = opaque->max_entries;
	scan->cur_count		  = opaque->entry_count;
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
mkt_posting_scan_cluster(MktPostingScan *scan, MktTopK *topk)
{
	Dimension dim		 = scan->dim;
	uint32_t  entry_size = MKT_POSTING_ENTRY_SIZE(dim);

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

		char	*content = scan->cur_content;
		uint32_t count	 = scan->cur_count;

		/* --- Score: batch IP over all entries on this page ---
		 *
		 * AoS layout: entry i's bits live at content + i * entry_size
		 * + MKT_POSTING_ENTRY_BITS_OFFSET. The SIMD kernel just needs
		 * the first entry's bits pointer and a stride of entry_size. */
		const uint8_t *bits_base = mkt_posting_first_bits(content);

		uint32_t padded = (count + 3) & ~3u;
		mkt_rabitq_inner_product_multi(
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
			MktPostingEntryHeader *e = mkt_posting_entry_at(content, i, dim);
			float final_dot = (2.0f * scratch[i] - sum_t) * inv_sqrt_d;
			distances[i] = e->f_add + g_add - 2.0f * e->f_rescale * final_dot;
		}

		/* --- Prune + insert approximate distances --- */
		Distance threshold = mkt_topk_threshold(topk);

		for (uint32_t i = 0; i < count; i++)
		{
			MktPostingEntryHeader *e = mkt_posting_entry_at(content, i, dim);
			scan->entries_scanned++;

			if (e->meta.flags & MKT_POSTING_FLAG_DELETED)
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

			uint64_t id = mkt_posting_encode_tid(&e->meta.tid);
			if (seen_tid_check_and_insert(scan, id))
				continue;
			mkt_topk_insert(topk, est, err, id);
			threshold = mkt_topk_threshold(topk);
		}

		/* Follow chain: read next_blkno, then release current */
		BlockNumber prev_blkno = scan->cur_blkno;

		if (scan->storage != NULL)
			scan->cur_blkno = mkt_posting_opaque(scan->cur_page)->next_blkno;
		else
			scan->cur_blkno = InvalidBlockNumber;

		if (scan->storage != NULL && scan->page_base == NULL)
			mkt_storage_release_page(scan->storage, prev_blkno);

		scan->cur_page	  = NULL;
		scan->cur_content = NULL;
	}
}

/* ----------------------------------------------------------------
 * Fastscan: vectorized distance + prune for 16 entries
 * ---------------------------------------------------------------- */

#ifdef MKT_SIMD_FULL
#if defined(__x86_64__) || defined(_M_X64)

/*
 * Compute distances and lower bounds for 16 entries using AVX-512.
 * Returns a bitmask of survivors (lb < threshold).
 */
MKT_TARGET_AVX512 static inline __mmask16
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
		float		   threshold)
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

	/* lb = est - err; survivors = lb < threshold */
	__m512	  lb_v	= _mm512_sub_ps(est_v, err_v);
	__m512	  thr_v = _mm512_set1_ps(threshold);
	__mmask16 surv	= _mm512_cmp_ps_mask(lb_v, thr_v, _CMP_LT_OS);

	_mm512_storeu_ps(est_out, est_v);
	_mm512_storeu_ps(err_out, err_v);

	return surv;
}

/*
 * AVX-512 vectorized distance + prune for one 32-vector group.
 * Must be in a target-attributed function so the compiler only
 * emits AVX-512 instructions here, not in the caller.
 */
MKT_TARGET_AVX512 static void
fastscan_prune_group_avx512(
		MktPostingScan	*scan,
		MktTopK			*topk,
		ItemPointerData *tids,
		const float		*f_add,
		const float		*f_rescale,
		const float		*f_error,
		float			 lut_scale,
		float			 lut_bias,
		float			 sum_t,
		float			 inv_sqrt_d,
		float			 g_add,
		float			 g_error,
		Distance		*threshold_p)
{
	Distance threshold = *threshold_p;
	float	 est_buf[MKT_FASTSCAN_GROUP];
	float	 err_buf[MKT_FASTSCAN_GROUP];

	for (uint32_t h = 0; h < 2; h++)
	{
		uint32_t  off  = h * 16;
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
				threshold);

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

			uint64_t id = mkt_posting_encode_tid(&tids[v]);
			if (seen_tid_check_and_insert(scan, id))
				continue;
			mkt_topk_insert(topk, est_buf[v], err_buf[v], id);
			threshold = mkt_topk_threshold(topk);
		}
	}

	*threshold_p = threshold;
}

#endif
#endif

#ifdef MKT_SIMD_FULL
#if defined(__aarch64__) || defined(_M_ARM64)

/*
 * NEON vectorized distance + prune for one 32-vector group.
 *
 * Per 4-vector chunk:
 *   1. Convert 4 int32 accumulators to float
 *   2. binary_ip = acc * scale + bias
 *   3. final_dot = (2*ip - sum_t) * inv_sqrt_d
 *   4. est = f_add + g_add - 2 * f_rescale * final_dot
 *   5. err = f_error * g_error
 *   6. lb = est - err
 *   7. survivor mask = lb < threshold (extract to 4-bit per chunk)
 *
 * NEON has no native packed compare-mask, so we narrow the
 * 32-bit compare result into a single 4-bit nibble per chunk via
 * shrn / addv.
 */
static void
fastscan_prune_group_neon(
		MktPostingScan	*scan,
		MktTopK			*topk,
		ItemPointerData *tids,
		const float		*f_add,
		const float		*f_rescale,
		const float		*f_error,
		float			 lut_scale,
		float			 lut_bias,
		float			 sum_t,
		float			 inv_sqrt_d,
		float			 g_add,
		float			 g_error,
		Distance		*threshold_p)
{
	Distance threshold = *threshold_p;
	float	 est_buf[MKT_FASTSCAN_GROUP];
	float	 err_buf[MKT_FASTSCAN_GROUP];

	float32x4_t scale_v	   = vdupq_n_f32(lut_scale);
	float32x4_t bias_v	   = vdupq_n_f32(lut_bias);
	float32x4_t two_v	   = vdupq_n_f32(2.0f);
	float32x4_t sum_t_v	   = vdupq_n_f32(sum_t);
	float32x4_t inv_sqrt_v = vdupq_n_f32(inv_sqrt_d);
	float32x4_t gadd_v	   = vdupq_n_f32(g_add);
	float32x4_t gerr_v	   = vdupq_n_f32(g_error);

	uint32_t surv_bits = 0; /* 32-bit mask of survivors */

	for (uint32_t off = 0; off < MKT_FASTSCAN_GROUP; off += 4)
	{
		int32x4_t	acc32 = vld1q_s32(scan->fs_accum + off);
		float32x4_t acc_f = vcvtq_f32_s32(acc32);

		float32x4_t ip	   = vmlaq_f32(bias_v, acc_f, scale_v);
		float32x4_t two_ip = vmulq_f32(two_v, ip);
		float32x4_t final_dot =
				vmulq_f32(vsubq_f32(two_ip, sum_t_v), inv_sqrt_v);

		float32x4_t fa	= vld1q_f32(f_add + off);
		float32x4_t fr	= vld1q_f32(f_rescale + off);
		float32x4_t est = vaddq_f32(fa, gadd_v);
		est				= vmlsq_f32(est, vmulq_f32(two_v, fr), final_dot);

		float32x4_t fe	= vld1q_f32(f_error + off);
		float32x4_t err = vmulq_f32(fe, gerr_v);
		float32x4_t lb	= vsubq_f32(est, err);

		vst1q_f32(est_buf + off, est);
		vst1q_f32(err_buf + off, err);

		float32x4_t thr_v = vdupq_n_f32(threshold);
		uint32x4_t	cmp	  = vcltq_f32(lb, thr_v);

		/* Extract 4-bit survivor mask: each lane is 0xFFFFFFFF or 0.
		 * AND with weighted lane mask then horizontal reduce. */
		const uint32x4_t weights	= {1, 2, 4, 8};
		uint32x4_t		 bits		= vandq_u32(cmp, weights);
		uint32_t		 chunk_bits = vaddvq_u32(bits);
		surv_bits |= chunk_bits << off;
	}

	uint32_t surv_count = (uint32_t)__builtin_popcount(surv_bits);
	scan->entries_pruned += MKT_FASTSCAN_GROUP - surv_count;

	while (surv_bits != 0)
	{
		uint32_t v = (uint32_t)__builtin_ctz(surv_bits);
		surv_bits &= surv_bits - 1;

		uint64_t id = mkt_posting_encode_tid(&tids[v]);
		if (seen_tid_check_and_insert(scan, id))
			continue;
		mkt_topk_insert(topk, est_buf[v], err_buf[v], id);
		threshold = mkt_topk_threshold(topk);
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
scan_fastscan_page(MktPostingScan *scan, MktTopK *topk)
{
	Dimension dim	  = scan->dim;
	uint32_t  count	  = scan->cur_count;
	char	 *content = scan->cur_content;

	MktPostingPageOpaque *opaque = mkt_posting_opaque(scan->cur_page);

	/* Fall back to AoS kernel for non-fastscan pages */
	if (!(opaque->flags & MKT_POSTING_PAGE_FASTSCAN))
		return; /* caller handles AoS via the standard path */

	/* Build LUT from transformed query (once per cluster) */
	if (!scan->fs_lut_valid)
	{
		if (scan->fs_lut_bits == 8)
			mkt_fastscan_build_lut(
					scan->qstate->transformed,
					dim,
					scan->fs_lut,
					&scan->fs_lut_scale,
					&scan->fs_lut_bias);
		else
			mkt_fastscan_build_lut_hacc(
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
	uint32_t max_groups = mkt_fastscan_max_groups(
			dim, opaque->flags & MKT_POSTING_PAGE_FIRST);
	uint32_t ngroups = (count + MKT_FASTSCAN_GROUP - 1) / MKT_FASTSCAN_GROUP;
	if (ngroups > max_groups)
		ngroups = max_groups;

	Distance threshold = mkt_topk_threshold(topk);

	for (uint32_t g = 0; g < ngroups; g++)
	{
		uint32_t g_start = g * MKT_FASTSCAN_GROUP;
		uint32_t g_count = count - g_start;
		if (g_count > MKT_FASTSCAN_GROUP)
			g_count = MKT_FASTSCAN_GROUP;

		/* Access group data — compute base once */
		char *gbase = mkt_fastscan_group_base(content, g, dim);

		ItemPointerData *tids	   = (ItemPointerData *)gbase;
		float			*f_add	   = (float *)(gbase +
								   MKT_FASTSCAN_GROUP * sizeof(ItemPointerData));
		float			*f_rescale = f_add + MKT_FASTSCAN_GROUP;
		float			*f_error   = f_rescale + MKT_FASTSCAN_GROUP;
		uint8_t			*codes	   = (uint8_t *)(f_error + MKT_FASTSCAN_GROUP);

		/* Run VPSHUFB accumulate kernel */
		if (scan->fs_lut_bits == 8)
		{
			uint16_t acc16[MKT_FASTSCAN_GROUP];
			mkt_fastscan_accumulate(codes, scan->fs_lut, acc16, dim);
			for (uint32_t v2 = 0; v2 < MKT_FASTSCAN_GROUP; v2++)
				scan->fs_accum[v2] = acc16[v2];
		}
		else
		{
			mkt_fastscan_accumulate_hacc(
					codes, scan->fs_lut, scan->fs_accum, dim);
		}

		/* Prefetch next group's codes during prune phase */
		if (g + 1 < ngroups)
		{
			uint8_t *next_codes =
					mkt_fastscan_group_codes(content, g + 1, dim);
			uint32_t code_bytes = MKT_FASTSCAN_GROUP_BYTES(dim);
			for (uint32_t p = 0; p < code_bytes; p += 64)
				__builtin_prefetch(next_codes + p, 0, 1);
		}

		/* Vectorized distance + prune: compute 16 distances at a
		 * time, mask survivors, iterate only the few that pass. */
		scan->entries_scanned += g_count;

#ifdef MKT_SIMD_FULL
#if defined(__x86_64__) || defined(_M_X64)
		if (mkt_likely(g_count == MKT_FASTSCAN_GROUP) &&
			mkt_has_simd(SIMD_AVX512F))
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
					&threshold);
		}
		else
#elif defined(__aarch64__) || defined(_M_ARM64)
		if (mkt_likely(g_count == MKT_FASTSCAN_GROUP))
		{
			fastscan_prune_group_neon(
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
					&threshold);
		}
		else
#endif
#endif
		{
			/* Scalar fallback for partial groups */
			for (uint32_t v = 0; v < g_count; v++)
			{
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

				uint64_t id = mkt_posting_encode_tid(&tids[v]);
				if (seen_tid_check_and_insert(scan, id))
					continue;
				mkt_topk_insert(topk, est, err, id);
				threshold = mkt_topk_threshold(topk);
			}
		}
	}
}

void
mkt_posting_scan_cluster_fastscan(MktPostingScan *scan, MktTopK *topk)
{
	/* Process pages until chain is exhausted */
	for (;;)
	{
		if (scan->cur_page == NULL)
		{
			if (!advance_page(scan))
				break;
		}

		MktPostingPageOpaque *opaque = mkt_posting_opaque(scan->cur_page);

		if (opaque->flags & MKT_POSTING_PAGE_FASTSCAN)
			scan_fastscan_page(scan, topk);
		else
			mkt_posting_scan_cluster(scan, topk);

		/* If AoS fallback consumed the entire chain, we're done */
		if (scan->cur_page == NULL && scan->cur_blkno == InvalidBlockNumber)
			break;

		/* Follow chain: read next_blkno, then release current */
		if (scan->cur_page != NULL)
		{
			BlockNumber prev_blkno = scan->cur_blkno;
			scan->cur_blkno		   = opaque->next_blkno;

			if (scan->storage != NULL && scan->page_base == NULL)
				mkt_storage_release_page(scan->storage, prev_blkno);

			scan->cur_page	  = NULL;
			scan->cur_content = NULL;
		}
	}
}
