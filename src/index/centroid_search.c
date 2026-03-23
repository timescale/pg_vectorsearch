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
	BlockNumber child_blkno; /* next level's page */
	Distance	distance;
	Distance	error;		  /* symmetric error (0 for exact) */
	BlockNumber source_blkno; /* page this candidate came from */
	uint16_t	source_entry; /* entry index on source page */
} Candidate;

/* ----------------------------------------------------------------
 * Score-page scratch — preallocated buffers for batch RaBitQ scoring
 * ---------------------------------------------------------------- */
typedef struct ScorePageScratch
{
	float	 *f_add;
	float	 *f_rescale;
	Distance *distances;
	Distance *lower_bounds;
	float	 *multi_scratch;	 /* scratch for batch_multi_with_bound */
	uint32_t *symmetric_scratch; /* scratch for batch_symmetric_with_bound */
} ScorePageScratch;

/*
 * Score all centroids on a single page, appending to candidates.
 * Returns the new candidate count.
 *
 * Dispatches based on page data format:
 *   RABITQ → batch multi-candidate scoring via sp_scratch
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
		ScorePageScratch			 *sp_scratch)
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
			const RaBitQData *d	 = mkt_centroid_data(page, count - 1 - i, dim);
			sp_scratch->f_add[i] = d->f_add;
			sp_scratch->f_rescale[i] = d->f_rescale;
		}

		/* bits_base = last entry's bits (lowest address) */
		const uint8_t *bits_base =
				mkt_centroid_data(page, count - 1, dim)->bits;

		/* Batch distance + error bound computation */
		if (state->qstate->mode == MKT_DISTANCE_MODE_SYMMETRIC)
			mkt_rabitq_distance_batch_symmetric_with_bound(
					state->qstate,
					sp_scratch->f_add,
					sp_scratch->f_rescale,
					bits_base,
					data_size,
					count,
					dim,
					sp_scratch->distances,
					sp_scratch->lower_bounds,
					sp_scratch->symmetric_scratch);
		else
			mkt_rabitq_distance_batch_multi_with_bound(
					state->qstate,
					sp_scratch->f_add,
					sp_scratch->f_rescale,
					bits_base,
					data_size,
					count,
					dim,
					sp_scratch->distances,
					sp_scratch->lower_bounds,
					sp_scratch->multi_scratch);

		/* Build candidates (result j → page entry count-1-j) */
		for (uint16_t j = 0; j < count && cand_count < cand_cap; j++)
		{
			uint16_t					page_idx = count - 1 - j;
			const MktCentroidEntryMeta *meta =
					mkt_centroid_meta(page, page_idx);

			cands[cand_count].child_blkno = meta->child_blkno;
			cands[cand_count].distance	  = sp_scratch->distances[j];
			cands[cand_count].error		  = sp_scratch->distances[j] -
									  sp_scratch->lower_bounds[j];
			cands[cand_count].source_blkno = page_blkno;
			cands[cand_count].source_entry = page_idx;
			cand_count++;
		}
		break;
	}
	case MKT_CENTROID_FMT_FLOAT:
	{
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
				float norm_q = mkt_l2_norm_squared(state->query, dim);
				float norm_v = mkt_l2_norm_squared(fvec, dim);
				float denom	 = sqrtf(norm_q * norm_v);
				dist		 = (denom > 0.0f) ? 1.0f - dot / denom : 1.0f;
				break;
			}
			default: /* L2 */
				dist = mkt_l2_distance_squared(state->query, fvec, dim);
				break;
			}

			cands[cand_count].child_blkno  = meta->child_blkno;
			cands[cand_count].distance	   = dist;
			cands[cand_count].error		   = 0.0f;
			cands[cand_count].source_blkno = page_blkno;
			cands[cand_count].source_entry = i;
			cand_count++;
		}
		break;
	}
	case MKT_CENTROID_FMT_HALF:
	{
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
				float norm_q = mkt_l2_norm_squared(state->query, dim);
				float norm_v = mkt_f16_norm_sq(hvec, dim);
				float denom	 = sqrtf(norm_q * norm_v);
				dist		 = (denom > 0.0f) ? 1.0f - dot / denom : 1.0f;
				break;
			}
			default: /* L2 */
				dist = mkt_f16_l2_squared(hvec, state->query, dim);
				break;
			}

			cands[cand_count].child_blkno  = meta->child_blkno;
			cands[cand_count].distance	   = dist;
			cands[cand_count].error		   = 0.0f;
			cands[cand_count].source_blkno = page_blkno;
			cands[cand_count].source_entry = i;
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
		Candidate *cands, uint32_t count, uint32_t k, Candidate *out)
{
	if (count == 0)
		return 0;

	MktTopK topk;
	mkt_topk_init(&topk, k);

	for (uint32_t i = 0; i < count; i++)
		mkt_topk_insert(&topk, cands[i].distance, cands[i].error, i);

	MktTopKEntry *entries = mkt_alloc(topk.cand_count * sizeof(MktTopKEntry));
	uint32_t	  nresults;
	mkt_topk_extract_sorted(&topk, entries, &nresults);

	for (uint32_t i = 0; i < nresults; i++)
	{
		uint32_t idx	= (uint32_t)entries[i].id;
		out[i]			= cands[idx];
		out[i].distance = entries[i].distance;
		out[i].error	= entries[i].error;
	}

	mkt_free(entries);
	mkt_topk_cleanup(&topk);
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

	/*
	 * Two-level memory context hierarchy for bounded memory usage.
	 */
	MktMemCtx beam_ctx	= mkt_memctx_create(NULL, "beam_search");
	MktMemCtx level_ctx = mkt_memctx_create(beam_ctx, "beam_level");
	MktMemCtx old_ctx	= mkt_memctx_switch(beam_ctx);

	/*
	 * Fixed-size candidate buffers. Use RaBitQ max (largest possible
	 * entry count) for safe upper bound regardless of page format.
	 */
	uint32_t max_per_page = mkt_centroid_max_entries(dim);
	uint32_t cand_cap	  = beam_width * max_per_page;
	if (cand_cap < max_per_page * 4)
		cand_cap = max_per_page * 4;

	Candidate *buf_a = mkt_alloc(cand_cap * sizeof(Candidate));
	Candidate *buf_b = mkt_alloc(cand_cap * sizeof(Candidate));

	/* Preallocate score-page scratch for batch RaBitQ scoring */
	ScorePageScratch sp_scratch;
	sp_scratch.f_add			 = mkt_alloc(max_per_page * sizeof(float));
	sp_scratch.f_rescale		 = mkt_alloc(max_per_page * sizeof(float));
	sp_scratch.distances		 = mkt_alloc(max_per_page * sizeof(Distance));
	sp_scratch.lower_bounds		 = mkt_alloc(max_per_page * sizeof(Distance));
	sp_scratch.multi_scratch	 = mkt_alloc(max_per_page * sizeof(float));
	sp_scratch.symmetric_scratch = mkt_alloc(max_per_page * sizeof(uint32_t));

	/* centroid_vecs extraction happens after building results */

	/*
	 * buf_a accumulates raw candidates from score_page.
	 * buf_b receives the topk-selected survivors.
	 * After selection, buf_b becomes the live set for expansion.
	 */

	/* Level 0: read root centroid page(s), score ALL centroids */
	uint32_t	raw_count = 0;
	BlockNumber blkno	  = first_centroid_blkno;

	mkt_memctx_switch(level_ctx);

	while (blkno != InvalidBlockNumber)
	{
		Page page = mkt_storage_read_page(state->storage, blkno);
		raw_count = score_page(
				state,
				page,
				blkno,
				dim,
				buf_a,
				raw_count,
				cand_cap,
				&sp_scratch);
		MktCentroidPageOpaque *opaque	  = MKT_CENTROID_OPAQUE(page);
		BlockNumber			   next_blkno = opaque->next_blkno;
		mkt_storage_release_page(state->storage, blkno);
		blkno = next_blkno;
	}
	if (stats)
		stats->dist_calcs += raw_count;

	/* Select top-K from level 0 into buf_b */
	uint32_t keep		= (nlevels == 1) ? nprobe : beam_width;
	uint32_t cand_count = select_topk_bounded(buf_a, raw_count, keep, buf_b);

	/* buf_b is now the live set */
	Candidate *live	   = buf_b;
	Candidate *scratch = buf_a;

	/* Intermediate levels: expand winners via child_blkno */
	for (uint8_t level = 1; level < nlevels; level++)
	{
		mkt_memctx_reset(level_ctx);

		uint32_t next_count = 0;

		for (uint32_t i = 0; i < cand_count; i++)
		{
			BlockNumber child_blkno = live[i].child_blkno;
			if (child_blkno == InvalidBlockNumber)
				continue;

			BlockNumber cb = child_blkno;
			while (cb != InvalidBlockNumber)
			{
				Page page  = mkt_storage_read_page(state->storage, cb);
				next_count = score_page(
						state,
						page,
						cb,
						dim,
						scratch,
						next_count,
						cand_cap,
						&sp_scratch);
				MktCentroidPageOpaque *opaque = MKT_CENTROID_OPAQUE(page);
				BlockNumber			   nb	  = opaque->next_blkno;
				mkt_storage_release_page(state->storage, cb);
				cb = nb;
			}
		}
		if (stats)
			stats->dist_calcs += next_count;

		/* Select into live (scratch → live via topk) */
		keep	   = (level == nlevels - 1) ? nprobe : beam_width;
		cand_count = select_topk_bounded(scratch, next_count, keep, live);
	}

	/* Build results in caller-owned memory (cap at nprobe) */
	uint32_t result_count = cand_count < nprobe ? cand_count : nprobe;
	for (uint32_t i = 0; i < result_count; i++)
	{
		results[i].posting_head = live[i].child_blkno;
		results[i].distance		= live[i].distance;
		results[i].error		= live[i].error;
	}

	/* Extract centroid vectors for float/half formats.
	 * Re-read the source pages (likely still cached) and copy
	 * the centroid vector data into caller-owned centroid_vecs. */
	if (centroid_vecs != NULL)
	{
		BlockNumber prev_blk  = InvalidBlockNumber;
		Page		prev_page = NULL;

		for (uint32_t i = 0; i < result_count; i++)
		{
			BlockNumber src_blk = live[i].source_blkno;
			uint16_t	src_ent = live[i].source_entry;

			/* Pin page (reuse if same as previous) */
			if (src_blk != prev_blk)
			{
				if (prev_page != NULL)
					mkt_storage_release_page(state->storage, prev_blk);
				prev_page = mkt_storage_read_page(state->storage, src_blk);
				prev_blk  = src_blk;
			}

			MktCentroidFormat fmt = mkt_centroid_page_format(prev_page);
			float			 *dst = centroid_vecs + (size_t)i * dim;

			switch (fmt)
			{
			case MKT_CENTROID_FMT_FLOAT:
			{
				const float *fvec =
						mkt_centroid_float_data(prev_page, src_ent, dim);
				memcpy(dst, fvec, dim * sizeof(float));
				break;
			}
			case MKT_CENTROID_FMT_HALF:
			{
				const half *hvec =
						mkt_centroid_half_data(prev_page, src_ent, dim);
				for (Dimension d = 0; d < dim; d++)
					dst[d] = mkt_half_to_float(hvec[d]);
				break;
			}
			default:
				/* RaBitQ centroids: can't extract float vector.
				 * Zero-fill; caller should not use centroid_vecs
				 * with RaBitQ centroid format. */
				memset(dst, 0, dim * sizeof(float));
				break;
			}
		}

		if (prev_page != NULL)
			mkt_storage_release_page(state->storage, prev_blk);
	}

	mkt_memctx_switch(old_ctx);
	mkt_memctx_delete(beam_ctx);

	return result_count;
}
