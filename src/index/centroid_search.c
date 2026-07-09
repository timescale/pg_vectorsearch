/*
 * centroid_search.c - Beam search over centroid tree
 *
 * Implements level-by-level descent through centroid pages using
 * format-aware distance computation. RaBitQ pages use approximate
 * distance with error bounds; float and half pages use exact L2
 * distance (error = 0). Candidate selection uses MktTopK for
 * error-bound-aware pruning.
 */

#include <math.h>
#include <string.h>

#include "algo/topk.h"
#include "algo/vecops.h"
#include "core/memory.h"
#include "index/centroid_search.h"
#include "mkt_halfvec.h"

/* ----------------------------------------------------------------
 * Internal candidate for beam search
 * ---------------------------------------------------------------- */
typedef struct Candidate
{
	BlockNumber		child_blkno; /* next level's page (or posting head) */
	ItemPointerData origin;		 /* (page, entry) this came from */
	Distance		distance;
	Distance		error; /* symmetric error (0 for exact) */
} Candidate;

/* ----------------------------------------------------------------
 * Per-scan scratch (public allocation; see centroid_search.h)
 *
 * Holds the candidate-buffer pair and score-page scratch that beam
 * search previously palloc'd per call. Allocated once at scan setup
 * and reused across all queries on the scan.
 * ---------------------------------------------------------------- */
struct MktCentroidScratch
{
	uint32_t   cand_cap;	 /* size of buf_a / buf_b */
	uint32_t   max_per_page; /* size of the f_add..symmetric_scratch arrays */
	Candidate *buf_a;
	Candidate *buf_b;
	float	  *f_add;
	float	  *f_rescale;
	Distance  *distances;
	Distance  *lower_bounds;
	float	  *multi_scratch;
	uint32_t  *symmetric_scratch;
	/* Reusable top-K + extraction buffer for select_topk_bounded.
	 * Avoids creating a fresh memctx + ub_heap + ub_ids + candidates
	 * + entries-buf on every beam-search level (was 2 sets of 4 allocs
	 * + 2 memctx creates per query). The MktTopK is initialised once
	 * at scratch_create with the worst-case k; select_topk_bounded
	 * calls mkt_topk_reset_to_k() to adjust between levels. */
	MktTopK		  level_topk;
	MktTopKEntry *entries_buf;
	uint32_t	  entries_cap;
	/* Fastscan LUT used by the FASTSCAN centroid format. The LUT only
	 * depends on the query (qstate->transformed), which is constant
	 * for the entire centroid descent — so we build it once per query
	 * and reuse for every FASTSCAN page. fs_lut_valid is cleared at
	 * the start of every beam-search call. */
	uint8_t *fs_lut;
	uint32_t fs_lut_bytes;
	float	 fs_lut_delta;
	float	 fs_lut_bias;
	bool	 fs_lut_valid;
};

MktCentroidScratch *
mkt_centroid_scratch_create(Dimension dim, uint32_t max_beam_width)
{
	uint32_t max_per_page = mkt_centroid_max_entries(dim);
	uint32_t cand_cap	  = max_beam_width * max_per_page;
	if (cand_cap < max_per_page * 4)
		cand_cap = max_per_page * 4;

	MktCentroidScratch *s = mkt_alloc(sizeof(MktCentroidScratch));
	if (s == NULL)
		return NULL;

	s->cand_cap			 = cand_cap;
	s->max_per_page		 = max_per_page;
	s->buf_a			 = mkt_alloc(cand_cap * sizeof(Candidate));
	s->buf_b			 = mkt_alloc(cand_cap * sizeof(Candidate));
	s->f_add			 = mkt_alloc(max_per_page * sizeof(float));
	s->f_rescale		 = mkt_alloc(max_per_page * sizeof(float));
	s->distances		 = mkt_alloc(max_per_page * sizeof(Distance));
	s->lower_bounds		 = mkt_alloc(max_per_page * sizeof(Distance));
	s->multi_scratch	 = mkt_alloc(max_per_page * sizeof(float));
	s->symmetric_scratch = mkt_alloc(max_per_page * sizeof(uint32_t));

	/* Reusable top-K and extract buffer (resized to actual k per
	 * select_topk_bounded call). Initial k=max_beam_width is just
	 * a starting size — the reset path repalloc's within the
	 * topk's memctx for different k. */
	mkt_topk_init(&s->level_topk, max_beam_width);
	s->entries_cap = max_beam_width * 4;
	if (s->entries_cap < 64)
		s->entries_cap = 64;
	s->entries_buf = mkt_alloc(s->entries_cap * sizeof(MktTopKEntry));

	/* Fastscan LUT (worst-case hacc size for this dim). Allocated
	 * once and reused for every centroid page scored in the
	 * FASTSCAN format. The LUT depends on the query so it's rebuilt
	 * per page; the buffer is reusable. */
	s->fs_lut_bytes = MKT_FASTSCAN_LUT_HACC_BYTES(dim);
	s->fs_lut		= mkt_alloc(s->fs_lut_bytes);
	return s;
}

void
mkt_centroid_scratch_free(MktCentroidScratch *s)
{
	if (s == NULL)
		return;
	mkt_topk_cleanup(&s->level_topk);
	mkt_free(s->entries_buf);
	mkt_free(s->buf_a);
	mkt_free(s->buf_b);
	mkt_free(s->f_add);
	mkt_free(s->f_rescale);
	mkt_free(s->distances);
	mkt_free(s->lower_bounds);
	mkt_free(s->multi_scratch);
	mkt_free(s->symmetric_scratch);
	mkt_free(s->fs_lut);
	mkt_free(s);
}

/*
 * Score all centroids on a single page, appending to candidates.
 * Returns the new candidate count.
 *
 * Dispatches based on page data format:
 *   RABITQ → batch multi-candidate scoring via cs
 *   FLOAT  → mkt_l2_distance_squared (exact, error=0)
 *   HALF   → mkt_f16_l2_squared (exact, error=0)
 */
static uint32_t
score_page(
		const MktCentroidSearchState *state,
		Page						  page,
		BlockNumber					  page_blkno,
		Dimension					  dim,
		Candidate					 *cands,
		uint32_t					  cand_count,
		uint32_t					  cand_cap,
		MktCentroidScratch			 *cs)
{
	MktCentroidPageOpaque *opaque = MKT_CENTROID_OPAQUE(page);
	uint16_t			   count  = opaque->entry_count;
	MktCentroidFormat	   fmt	  = mkt_centroid_page_format(page);

	if (count == 0)
		return cand_count;

	switch (fmt)
	{
	case MKT_CENTROID_FMT_RABITQ:
	{
		uint32_t data_size = MKT_RABITQ_DATA_SIZE(dim);

		/* Gather f_add/f_rescale in reverse order
		 * (page data grows backward: entry 0 at highest address) */
		for (uint16_t i = 0; i < count; i++)
		{
			const RaBitQData *d = mkt_centroid_data(page, count - 1 - i, dim);
			cs->f_add[i]		= d->f_add;
			cs->f_rescale[i]	= d->f_rescale;
		}

		/* bits_base = last entry's bits (lowest address) */
		const uint8_t *bits_base =
				mkt_centroid_data(page, count - 1, dim)->bits;

		/* Batch distance + error bound computation. With error_scale = 0
		 * (the default) the pruning bounds are multiplied by zero anyway,
		 * so skip computing them entirely: the batch functions take a
		 * NULL lower_bounds and omit the per-entry error derivation (a
		 * divide + sqrt per centroid that also blocks vectorization of
		 * the distance-apply loop). */
		bool	  want_bounds = (state->error_scale != 0.0f);
		Distance *lb		  = want_bounds ? cs->lower_bounds : NULL;

		if (state->qstate->mode == MKT_DISTANCE_MODE_SYMMETRIC)
			mkt_rabitq_distance_batch_symmetric_with_bound(
					state->qstate,
					cs->f_add,
					cs->f_rescale,
					bits_base,
					data_size,
					count,
					dim,
					cs->distances,
					lb,
					cs->symmetric_scratch);
		else
			mkt_rabitq_distance_batch_multi_with_bound(
					state->qstate,
					cs->f_add,
					cs->f_rescale,
					bits_base,
					data_size,
					count,
					dim,
					cs->distances,
					lb,
					cs->multi_scratch);

		/* Build candidates (result j → page entry count-1-j) */
		for (uint16_t j = 0; j < count && cand_count < cand_cap; j++)
		{
			uint16_t					page_idx = count - 1 - j;
			const MktCentroidEntryMeta *meta =
					mkt_centroid_meta(page, page_idx);

			cands[cand_count].child_blkno = meta->child_blkno;
			ItemPointerSet(&cands[cand_count].origin, page_blkno, page_idx);
			cands[cand_count].distance = cs->distances[j];
			cands[cand_count].error	   = want_bounds
											   ? state->error_scale *
														 (cs->distances[j] -
														  cs->lower_bounds[j])
											   : 0.0f;
			cand_count++;
		}
		break;
	}
	case MKT_CENTROID_FMT_FLOAT:
	{
		/* Hoist query norm out of the inner loop: it depends only on
		 * the query, not the centroid, but was previously recomputed
		 * for every entry (one full norm² per centroid scored). */
		float norm_q = (state->metric == DISTANCE_COSINE)
							 ? mkt_l2_norm_squared(state->query, dim)
							 : 0.0f;

		for (uint16_t i = 0; i < count && cand_count < cand_cap; i++)
		{
			const MktCentroidEntryMeta *meta = mkt_centroid_meta(page, i);
			const float *fvec = mkt_centroid_float_data(page, i, dim);
			Distance	 dist;

			switch (state->metric)
			{
			case DISTANCE_INNER_PRODUCT:
				dist = -mkt_dot_product(state->query, fvec, dim);
				break;
			case DISTANCE_COSINE:
			{
				float dot	 = mkt_dot_product(state->query, fvec, dim);
				float norm_v = mkt_l2_norm_squared(fvec, dim);
				float denom	 = sqrtf(norm_q * norm_v);
				dist		 = (denom > 0.0f) ? 1.0f - dot / denom : 1.0f;
				break;
			}
			default: /* L2 */
				dist = mkt_l2_distance_squared(state->query, fvec, dim);
				break;
			}

			cands[cand_count].child_blkno = meta->child_blkno;
			ItemPointerSet(&cands[cand_count].origin, page_blkno, i);
			cands[cand_count].distance = dist;
			cands[cand_count].error	   = 0.0f;
			cand_count++;
		}
		break;
	}
	case MKT_CENTROID_FMT_FASTSCAN:
	{
		/* Fastscan centroid pages: same RaBitQ codes as the RABITQ
		 * format but rearranged into 32-vector groups so we can
		 * score them with mkt_fastscan_accumulate (~150 M vec/s on
		 * Graviton 4) instead of the per-vector kernel used by the
		 * RABITQ branch above.
		 *
		 * Layout per group section:
		 *   BlockNumber child_blkno[32]
		 *   float       f_add[32]
		 *   float       f_rescale[32]
		 *   float       f_error[32]
		 *   uint8_t     codes[nsq_pairs*32]
		 *
		 * The LUT depends on the query (and via qstate->transformed,
		 * implicitly on the cluster's centroid for posting scans;
		 * for centroid descent the relevant query state is the
		 * pre-cluster qstate->transformed which is just P^T*query
		 * minus the global_mean rotation already absorbed by
		 * pt_global_mean). It is rebuilt once per centroid page
		 * scored. Amortising the LUT build across all 32 entries in
		 * a group is the whole reason this is faster than the
		 * per-vector kernel. */
		/* Build the LUT once per query — qstate->transformed is
		 * constant across the whole centroid descent, so we cache
		 * the LUT bytes + lut_delta + lut_bias in MktCentroidScratch
		 * and reuse them on every subsequent FASTSCAN page. */
		if (!cs->fs_lut_valid)
		{
			mkt_fastscan_build_lut_hacc(
					state->qstate->transformed,
					dim,
					cs->fs_lut,
					&cs->fs_lut_delta,
					&cs->fs_lut_bias);
			cs->fs_lut_valid = true;
		}
		float lut_delta = cs->fs_lut_delta;
		float lut_bias	= cs->fs_lut_bias;

		float g_add		 = state->qstate->g_add;
		float sum_t		 = state->qstate->sum_transformed;
		float inv_sqrt_d = state->qstate->inv_sqrt_d;
		float g_error	 = state->qstate->g_error;
		float err_mult	 = state->qstate->error_multiplier;

		char	*content	 = (char *)PageGetContents(page);
		uint32_t entry_count = count;
		uint32_t ngroups	 = (entry_count + MKT_FASTSCAN_GROUP - 1) /
						   MKT_FASTSCAN_GROUP;

		int32_t accum[MKT_FASTSCAN_GROUP];

		for (uint32_t g = 0; g < ngroups; g++)
		{
			uint32_t g_start = g * MKT_FASTSCAN_GROUP;
			uint32_t g_count = entry_count - g_start;
			if (g_count > MKT_FASTSCAN_GROUP)
				g_count = MKT_FASTSCAN_GROUP;

			const BlockNumber *child = (const BlockNumber *)
					mkt_centroid_fastscan_group_child(content, g, dim);
			const float *f_add_arr =
					mkt_centroid_fastscan_group_f_add(content, g, dim);
			const float *f_rescale_arr =
					mkt_centroid_fastscan_group_f_rescale(content, g, dim);
			const float *f_error_arr =
					mkt_centroid_fastscan_group_f_error(content, g, dim);
			const uint8_t *codes =
					mkt_centroid_fastscan_group_codes(content, g, dim);

			mkt_fastscan_accumulate_hacc(codes, cs->fs_lut, accum, dim);

			for (uint32_t v = 0; v < g_count && cand_count < cand_cap; v++)
			{
				/* De-quantise the LUT accumulator the same way the
				 * posting fastscan path does (see prune_group_neon
				 * in posting_scan.c). */
				float binary_ip = (float)accum[v] * lut_delta + lut_bias;
				float final_dot = (2.0f * binary_ip - sum_t) * inv_sqrt_d;

				Distance est = f_add_arr[v] + g_add -
							   2.0f * f_rescale_arr[v] * final_dot;
				/* Matches rabitq_lower_bound(): err_margin =
				 * multiplier * f_error * g_error, plus a small
				 * floating-point margin proportional to |est|.
				 * error_scale = 0 (the default) zeroes the margin, so
				 * skip the arithmetic in that case. */
				Distance err = 0.0f;
				if (state->error_scale != 0.0f)
					err = state->error_scale *
						  (err_mult * f_error_arr[v] * g_error +
						   1e-5f * fabsf(est));

				uint32_t page_idx			  = g_start + v;
				cands[cand_count].child_blkno = child[v];
				ItemPointerSet(
						&cands[cand_count].origin, page_blkno, page_idx);
				cands[cand_count].distance = est;
				cands[cand_count].error	   = err;
				cand_count++;
			}
		}
		break;
	}
	case MKT_CENTROID_FMT_HALF:
	{
		float norm_q = (state->metric == DISTANCE_COSINE)
							 ? mkt_l2_norm_squared(state->query, dim)
							 : 0.0f;

		for (uint16_t i = 0; i < count && cand_count < cand_cap; i++)
		{
			const MktCentroidEntryMeta *meta = mkt_centroid_meta(page, i);
			const half *hvec = mkt_centroid_half_data(page, i, dim);
			Distance	dist;

			switch (state->metric)
			{
			case DISTANCE_INNER_PRODUCT:
				dist = -mkt_f16_dot_product(hvec, state->query, dim);
				break;
			case DISTANCE_COSINE:
			{
				float dot	 = mkt_f16_dot_product(hvec, state->query, dim);
				float norm_v = mkt_f16_norm_sq(hvec, dim);
				float denom	 = sqrtf(norm_q * norm_v);
				dist		 = (denom > 0.0f) ? 1.0f - dot / denom : 1.0f;
				break;
			}
			default: /* L2 */
				dist = mkt_f16_l2_squared(hvec, state->query, dim);
				break;
			}

			cands[cand_count].child_blkno = meta->child_blkno;
			ItemPointerSet(&cands[cand_count].origin, page_blkno, i);
			cands[cand_count].distance = dist;
			cands[cand_count].error	   = 0.0f;
			cand_count++;
		}
		break;
	}
	}

	return cand_count;
}

/* ----------------------------------------------------------------
 * Top-K selection via MktTopK (error-bound-aware)
 *
 * Selects the best candidates using MktTopK pruning. Candidates
 * with lower_bound >= threshold are pruned. With overlapping
 * error intervals, may return more than k entries.
 *
 * Results are placed sorted in out[0..return_count).
 * out must not alias cands.
 * ---------------------------------------------------------------- */
static uint32_t
select_topk_bounded(
		MktCentroidScratch *scratch,
		Candidate		   *cands,
		uint32_t			count,
		uint32_t			k,
		Candidate		   *out,
		uint32_t			out_cap)
{
	if (count == 0)
		return 0;

	/* Reuse the per-scan topk and extract buffer instead of allocating
	 * new ones every level. mkt_topk_reset_to_k re-allocates the
	 * heap/candidates within the topk's existing memctx (cheap). */
	MktTopK *topk = &scratch->level_topk;
	mkt_topk_reset_to_k(topk, k);

	/* Centroid candidates have unique ids (the buf index), so we can
	 * skip the O(k) per-insert dedup scan. */
	for (uint32_t i = 0; i < count; i++)
		mkt_topk_insert_unique(topk, cands[i].distance, cands[i].error, i);

	/* entries_buf must hold topk->cand_count survivors; grow if needed. */
	if (topk->cand_count > scratch->entries_cap)
	{
		uint32_t new_cap = scratch->entries_cap * 2;
		while (new_cap < topk->cand_count)
			new_cap *= 2;
		mkt_free(scratch->entries_buf);
		scratch->entries_buf = mkt_alloc(new_cap * sizeof(MktTopKEntry));
		scratch->entries_cap = new_cap;
	}

	uint32_t nresults;
	mkt_topk_extract_sorted(topk, scratch->entries_buf, &nresults);

	if (nresults > out_cap)
		nresults = out_cap;

	for (uint32_t i = 0; i < nresults; i++)
	{
		uint32_t idx	= (uint32_t)scratch->entries_buf[i].id;
		out[i]			= cands[idx];
		out[i].distance = scratch->entries_buf[i].distance;
		out[i].error	= scratch->entries_buf[i].error;
	}

	return nresults;
}

uint32_t
mkt_centroid_beam_search(
		const MktCentroidSearchState *state,
		BlockNumber					  first_centroid_blkno,
		uint8_t						  nlevels,
		MktCentroidResult			 *results,
		float						 *centroid_vecs,
		MktCentroidSearchStats		 *stats)
{
	if (state == NULL || results == NULL || nlevels == 0 ||
		first_centroid_blkno == InvalidBlockNumber)
		return 0;

	Dimension dim		 = state->dim;
	uint32_t  beam_width = state->beam_width;
	uint32_t  nprobe	 = state->nprobe;

	/* Caller-owned scratch: pre-allocated buffers for candidate
	 * arrays and per-page batch scoring. Avoids 8 palloc + 2 memctx
	 * creations per query — a meaningful chunk of the allocator
	 * traffic on the hot path.
	 *
	 * If state->scratch is NULL we fall back to a one-shot
	 * allocation. The fallback exists for tests and ad-hoc callers;
	 * production query paths (MktQueryState / MktQueryCtx)
	 * pre-allocate and pass it in. */
	MktCentroidScratch *scratch		  = state->scratch;
	MktCentroidScratch *owned_scratch = NULL;
	if (scratch == NULL)
	{
		owned_scratch = mkt_centroid_scratch_create(dim, beam_width);
		if (owned_scratch == NULL)
			return 0;
		scratch = owned_scratch;
	}
	uint32_t   cand_cap = scratch->cand_cap;
	Candidate *buf_a	= scratch->buf_a;
	Candidate *buf_b	= scratch->buf_b;

	/* Invalidate the per-query fastscan LUT cache. Built lazily on
	 * first FASTSCAN page encountered, then reused for all subsequent
	 * pages in this query. */
	scratch->fs_lut_valid = false;

	/* centroid_vecs: will be used later for copying centroid vectors */
	(void)centroid_vecs;

	/*
	 * buf_a accumulates raw candidates from score_page.
	 * buf_b receives the topk-selected survivors.
	 * After selection, buf_b becomes the live set for expansion.
	 */

	uint32_t centroid_pages_read = 0;

	/* Level 0: read root centroid page(s), score ALL centroids */
	uint32_t	raw_count = 0;
	BlockNumber blkno	  = first_centroid_blkno;

	while (blkno != InvalidBlockNumber)
	{
		Page page = mkt_storage_read_page(state->storage, blkno);
		centroid_pages_read++;
		raw_count = score_page(
				state, page, blkno, dim, buf_a, raw_count, cand_cap, scratch);
		MktCentroidPageOpaque *opaque	  = MKT_CENTROID_OPAQUE(page);
		BlockNumber			   next_blkno = opaque->next_blkno;
		mkt_storage_release_page(state->storage, blkno);
		blkno = next_blkno;
	}
	if (stats)
		stats->dist_calcs += raw_count;

	/* beam_width is the intermediate-level keep; the leaf level always
	 * returns nprobe (see keep below), and beam_width*fan_out >= nprobe
	 * covers the top-nprobe leaves, so beam_width may be < nprobe. */
	if (beam_width < 1)
		beam_width = 1;

	/* Select top-K from level 0 into buf_b.
	 *
	 * Error-bound-aware selection via MktTopK: keeps the beam_width
	 * candidates with smallest upper bounds, plus any additional
	 * candidates whose lower bound overlaps the threshold. For
	 * exact formats (error=0) this returns exactly beam_width. */
	uint32_t keep		= (nlevels == 1) ? nprobe : beam_width;
	uint32_t cand_count = select_topk_bounded(
			scratch, buf_a, raw_count, keep, buf_b, cand_cap);

	/* buf_b is now the live set */
	Candidate *live		  = buf_b;
	Candidate *expand_buf = buf_a;

	/* Intermediate levels: expand winners via child_blkno */
	for (uint8_t level = 1; level < nlevels; level++)
	{
		uint32_t next_count = 0;

		for (uint32_t i = 0; i < cand_count; i++)
		{
			BlockNumber child_blkno = live[i].child_blkno;
			if (child_blkno == InvalidBlockNumber)
				continue;

			BlockNumber cb = child_blkno;
			while (cb != InvalidBlockNumber)
			{
				Page page = mkt_storage_read_page(state->storage, cb);
				centroid_pages_read++;
				next_count = score_page(
						state,
						page,
						cb,
						dim,
						expand_buf,
						next_count,
						cand_cap,
						scratch);
				MktCentroidPageOpaque *opaque = MKT_CENTROID_OPAQUE(page);
				BlockNumber			   nb	  = opaque->next_blkno;
				mkt_storage_release_page(state->storage, cb);
				cb = nb;
			}
		}
		if (stats)
			stats->dist_calcs += next_count;

		/* Select winners into live; expand_buf is the raw input. */
		keep	   = (level == nlevels - 1) ? nprobe : beam_width;
		cand_count = select_topk_bounded(
				scratch, expand_buf, next_count, keep, live, cand_cap);
	}

	/* Build results in caller-owned memory (cap at nprobe) */
	uint32_t result_count = cand_count < nprobe ? cand_count : nprobe;
	for (uint32_t i = 0; i < result_count; i++)
	{
		results[i].posting_head = live[i].child_blkno;
		results[i].distance		= live[i].distance;
		results[i].error		= live[i].error;
	}

	/* Extract centroid vectors for winning clusters.
	 * Only meaningful for FLOAT/HALF centroid pages — RaBitQ is a
	 * lossy binary encoding, so the full-precision centroid cannot
	 * be recovered. RaBitQ callers must obtain pt_centroid from its
	 * dedicated location (e.g., the first posting page). Skip the
	 * loop entirely rather than re-reading pages for nothing. */
	if (centroid_vecs != NULL && result_count > 0 && state->qstate == NULL)
	{
		BlockNumber prev_blk  = InvalidBlockNumber;
		Page		prev_page = NULL;

		for (uint32_t i = 0; i < result_count; i++)
		{
			BlockNumber	 blk = ItemPointerGetBlockNumber(&live[i].origin);
			OffsetNumber idx = ItemPointerGetOffsetNumber(&live[i].origin);

			Page page;
			if (blk == prev_blk)
			{
				page = prev_page;
			}
			else
			{
				if (prev_page != NULL)
					mkt_storage_release_page(state->storage, prev_blk);
				page = mkt_storage_read_page(state->storage, blk);
				centroid_pages_read++;
				prev_blk  = blk;
				prev_page = page;
			}

			float			 *dst = centroid_vecs + (size_t)i * dim;
			MktCentroidFormat fmt = mkt_centroid_page_format(page);

			switch (fmt)
			{
			case MKT_CENTROID_FMT_FLOAT:
			{
				const float *src = mkt_centroid_float_data(page, idx, dim);
				memcpy(dst, src, dim * sizeof(float));
				break;
			}
			case MKT_CENTROID_FMT_HALF:
			{
				const half *src = mkt_centroid_half_data(page, idx, dim);
				for (Dimension d = 0; d < dim; d++)
					dst[d] = mkt_half_to_float(src[d]);
				break;
			}
			case MKT_CENTROID_FMT_RABITQ:
			case MKT_CENTROID_FMT_FASTSCAN:
				/* Unreachable — both are lossy binary encodings and
				 * the outer guard skips this branch when the page
				 * format isn't FLOAT/HALF. */
				break;
			}
		}

		if (prev_page != NULL)
			mkt_storage_release_page(state->storage, prev_blk);
	}

	if (stats)
		stats->pages_read = centroid_pages_read;

	if (owned_scratch != NULL)
		mkt_centroid_scratch_free(owned_scratch);

	return result_count;
}
