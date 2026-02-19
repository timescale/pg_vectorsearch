/*
 * centroid_search.c - Beam search over centroid tree
 *
 * Implements level-by-level descent through centroid pages using
 * format-aware distance computation. RaBitQ pages use approximate
 * distance with error bounds; float and half pages use exact L2
 * distance (error = 0). Candidate selection uses MktTopK for
 * error-bound-aware pruning.
 */

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
	BlockNumber		child_blkno; /* next level's page */
	ItemPointerData medoid_tid;
	Distance		distance;
	Distance		error; /* symmetric error (0 for exact) */
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
			uint16_t						  page_idx = count - 1 - j;
			const MktCentroidEntryMetaRaBitQ *rmeta =
					mkt_centroid_meta_rabitq(page, page_idx);

			cands[cand_count].child_blkno = rmeta->base.child_blkno;
			cands[cand_count].medoid_tid  = rmeta->medoid_tid;
			cands[cand_count].distance	  = sp_scratch->distances[j];
			cands[cand_count].error		  = sp_scratch->distances[j] -
									  sp_scratch->lower_bounds[j];
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
			Distance dist = mkt_l2_distance_squared(state->query, fvec, dim);

			cands[cand_count].child_blkno = meta->child_blkno;
			memset(&cands[cand_count].medoid_tid, 0, sizeof(ItemPointerData));
			cands[cand_count].distance = dist;
			cands[cand_count].error	   = 0.0f;
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
			Distance	dist = mkt_f16_l2_squared(hvec, state->query, dim);

			cands[cand_count].child_blkno = meta->child_blkno;
			memset(&cands[cand_count].medoid_tid, 0, sizeof(ItemPointerData));
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

/* ----------------------------------------------------------------
 * Rerank scratch — preallocated buffers for rerank_candidates
 * ---------------------------------------------------------------- */
typedef struct RerankScratch
{
	ItemPointerData *tids;
	Distance		*distances;
	Distance		*errors;
	uint32_t		*out_indices;
	Distance		*out_distances;
	Candidate		*tmp;
} RerankScratch;

static void
rerank_scratch_init(RerankScratch *s, uint32_t cap)
{
	s->tids			 = mkt_alloc(cap * sizeof(ItemPointerData));
	s->distances	 = mkt_alloc(cap * sizeof(Distance));
	s->errors		 = mkt_alloc(cap * sizeof(Distance));
	s->out_indices	 = mkt_alloc(cap * sizeof(uint32_t));
	s->out_distances = mkt_alloc(cap * sizeof(Distance));
	s->tmp			 = mkt_alloc(cap * sizeof(Candidate));
}

/*
 * Rerank candidates with exact distances via storage layer.
 * Replaces approximate distances with exact ones for candidates
 * that survive error-bound pruning. Skips if no rerank method
 * or all candidates are already exact (error == 0).
 *
 * Uses preallocated scratch buffers — no per-call allocations.
 */
static uint32_t
rerank_candidates(
		const MktCentroidSearchState *state,
		Candidate					 *cands,
		uint32_t					  count,
		uint32_t					  keep,
		RerankScratch				 *scratch,
		MktCentroidSearchStats		 *stats)
{
	if (count == 0 || state->storage->ops->rerank == NULL)
		return count;

	/* Check if any candidate has approximate distance */
	bool has_approx = false;
	for (uint32_t i = 0; i < count && !has_approx; i++)
		has_approx = (cands[i].error > 0.0f);

	if (!has_approx)
		return count < keep ? count : keep;

	if (stats)
		stats->reranked += count;

	/* Build parallel arrays from Candidate structs */
	for (uint32_t i = 0; i < count; i++)
	{
		scratch->tids[i]	  = cands[i].medoid_tid;
		scratch->distances[i] = cands[i].distance;
		scratch->errors[i]	  = cands[i].error;
	}

	uint32_t nresults = mkt_storage_rerank(
			state->storage,
			state->query_datum,
			state->dim,
			scratch->tids,
			scratch->distances,
			scratch->errors,
			count,
			keep,
			scratch->out_indices,
			scratch->out_distances);

	/* Rebuild candidate array from reranked results */
	for (uint32_t i = 0; i < nresults; i++)
	{
		uint32_t idx				= scratch->out_indices[i];
		scratch->tmp[i].child_blkno = cands[idx].child_blkno;
		scratch->tmp[i].medoid_tid	= cands[idx].medoid_tid;
		scratch->tmp[i].distance	= scratch->out_distances[i];
		scratch->tmp[i].error		= 0.0f; /* now exact */
	}
	memcpy(cands, scratch->tmp, nresults * sizeof(Candidate));

	return nresults;
}

uint32_t
mkt_centroid_beam_search(
		const MktCentroidSearchState *state,
		BlockNumber					  first_centroid_blkno,
		uint8_t						  nlevels,
		MktCentroidResult			 *results,
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

	/* Preallocate rerank scratch buffers (zero allocations per level) */
	RerankScratch rerank_scratch;
	bool		  has_rerank = (state->storage->ops->rerank != NULL);
	if (has_rerank)
		rerank_scratch_init(&rerank_scratch, cand_cap);

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
				state, page, dim, buf_a, raw_count, cand_cap, &sp_scratch);
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

	/* Rerank level 0 survivors with exact distances */
	if (has_rerank)
		cand_count = rerank_candidates(
				state, buf_b, cand_count, keep, &rerank_scratch, stats);

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

		/* Rerank this level's survivors with exact distances */
		if (has_rerank)
			cand_count = rerank_candidates(
					state, live, cand_count, keep, &rerank_scratch, stats);
	}

	/* Build results in caller-owned memory (cap at nprobe) */
	uint32_t result_count = cand_count < nprobe ? cand_count : nprobe;
	for (uint32_t i = 0; i < result_count; i++)
	{
		results[i].posting_head = live[i].child_blkno;
		results[i].medoid_tid	= live[i].medoid_tid;
		results[i].distance		= live[i].distance;
		results[i].error		= live[i].error;
	}

	mkt_memctx_switch(old_ctx);
	mkt_memctx_delete(beam_ctx);

	return result_count;
}
