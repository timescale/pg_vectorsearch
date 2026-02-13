/*
 * centroid_search.c - Beam search over centroid tree
 *
 * Implements level-by-level descent through centroid pages using
 * batch RaBitQ distance on the in-page SoA arrays. Each level
 * narrows candidates down to beam_width (or nprobe at leaf level).
 */

#include <string.h>

#include "core/memory.h"
#include "index/centroid_search.h"

/* ----------------------------------------------------------------
 * Internal candidate for beam search
 * ---------------------------------------------------------------- */
typedef struct Candidate
{
	BlockNumber		child_blkno; /* next level's page */
	uint16_t		child_count; /* for diagnostics */
	uint16_t		flags;
	ItemPointerData medoid_tid;
	Distance		distance;
} Candidate;

/* ----------------------------------------------------------------
 * Min-heap (top-K selection by smallest distance)
 *
 * We use a simple partial sort: collect all candidates into an
 * array, then partial-sort to find the top-K smallest distances.
 * For typical centroid counts (32-256), this is fast enough and
 * simpler than maintaining a max-heap.
 * ---------------------------------------------------------------- */

/*
 * Partition candidates so that the k smallest are in [0, k).
 * Uses nth_element-style partial quickselect.
 */
static void
partial_sort_candidates(Candidate *cands, uint32_t count, uint32_t k)
{
	if (k >= count)
		return;

	/* Simple insertion sort for small arrays (common case: k <= 64) */
	for (uint32_t i = 1; i < count; i++)
	{
		Candidate tmp = cands[i];
		uint32_t  j	  = i;
		while (j > 0 && cands[j - 1].distance > tmp.distance)
		{
			cands[j] = cands[j - 1];
			j--;
		}
		cands[j] = tmp;
	}
}

/*
 * Score all centroids on a single page, appending to candidates.
 * Returns the new candidate count.
 */
static uint32_t
score_page(
		const MktCentroidSearchState *state,
		Page						  page,
		Dimension					  dim,
		Candidate					 *cands,
		uint32_t					  cand_count,
		uint32_t					  cand_cap)
{
	MktCentroidPageOpaque *opaque = MKT_CENTROID_OPAQUE(page);
	uint16_t			   count  = opaque->entry_count;

	if (count == 0)
		return cand_count;

	uint32_t max = mkt_centroid_max_entries(dim);

	/* Batch compute distances on SoA arrays directly from page */
	Distance *dists = mkt_alloc(count * sizeof(Distance));
	mkt_rabitq_distance_batch_soa(
			state->qstate,
			MKT_CENTROID_F_ADD(page, max),
			MKT_CENTROID_F_RESCALE(page, max),
			MKT_CENTROID_BITS(page, max),
			count,
			dim,
			dists);

	/* Convert to candidates */
	MktCentroidEntryMeta *meta = MKT_CENTROID_META(page);
	for (uint16_t i = 0; i < count && cand_count < cand_cap; i++)
	{
		cands[cand_count].child_blkno = meta[i].child_blkno;
		cands[cand_count].child_count = meta[i].child_count;
		cands[cand_count].flags		  = meta[i].flags;
		cands[cand_count].medoid_tid  = meta[i].medoid_tid;
		cands[cand_count].distance	  = dists[i];
		cand_count++;
	}

	mkt_free(dists);
	return cand_count;
}

uint32_t
mkt_centroid_beam_search(
		const MktCentroidSearchState *state,
		BlockNumber					  first_centroid_blkno,
		uint8_t						  nlevels,
		MktCentroidResult			 *results)
{
	if (state == NULL || results == NULL || nlevels == 0 ||
		first_centroid_blkno == InvalidBlockNumber)
		return 0;

	Dimension dim		 = state->dim;
	uint32_t  beam_width = state->beam_width;
	uint32_t  nprobe	 = state->nprobe;

	/*
	 * Maximum candidates at any level. With branch factor K and
	 * beam_width B, each level can produce at most B * K candidates.
	 * We cap at a reasonable maximum to bound memory usage.
	 */
	uint32_t max_per_page = mkt_centroid_max_entries(dim);
	uint32_t cand_cap	  = beam_width * max_per_page;
	if (cand_cap < max_per_page * 4)
		cand_cap = max_per_page * 4;

	Candidate *cands = mkt_alloc(cand_cap * sizeof(Candidate));
	Candidate *next	 = mkt_alloc(cand_cap * sizeof(Candidate));

	/* Level 0: read root centroid page(s), score ALL centroids */
	uint32_t	cand_count = 0;
	BlockNumber blkno	   = first_centroid_blkno;

	while (blkno != InvalidBlockNumber)
	{
		Page page  = state->accessor->read(state->accessor->ctx, blkno);
		cand_count = score_page(state, page, dim, cands, cand_count, cand_cap);
		MktCentroidPageOpaque *opaque	  = MKT_CENTROID_OPAQUE(page);
		BlockNumber			   next_blkno = opaque->next_blkno;
		state->accessor->release(state->accessor->ctx, blkno);
		blkno = next_blkno;
	}

	/* Select top beam_width from level 0 */
	uint32_t keep = (nlevels == 1) ? nprobe : beam_width;
	if (keep > cand_count)
		keep = cand_count;
	partial_sort_candidates(cands, cand_count, keep);
	cand_count = keep;

	/* Intermediate levels: expand winners via child_blkno */
	for (uint8_t level = 1; level < nlevels; level++)
	{
		uint32_t next_count = 0;

		for (uint32_t i = 0; i < cand_count; i++)
		{
			BlockNumber child_blkno = cands[i].child_blkno;
			if (child_blkno == InvalidBlockNumber)
				continue;

			/* Read child page(s) — follow next_blkno chain */
			BlockNumber cb = child_blkno;
			while (cb != InvalidBlockNumber)
			{
				Page page  = state->accessor->read(state->accessor->ctx, cb);
				next_count = score_page(
						state, page, dim, next, next_count, cand_cap);
				MktCentroidPageOpaque *opaque = MKT_CENTROID_OPAQUE(page);
				BlockNumber			   nb	  = opaque->next_blkno;
				state->accessor->release(state->accessor->ctx, cb);
				cb = nb;
			}
		}

		/* Select top-K for next level */
		keep = (level == nlevels - 1) ? nprobe : beam_width;
		if (keep > next_count)
			keep = next_count;
		partial_sort_candidates(next, next_count, keep);

		/* Swap buffers */
		Candidate *tmp = cands;
		cands		   = next;
		next		   = tmp;
		cand_count	   = keep;
	}

	/* Build results from final candidates */
	uint32_t result_count = cand_count;
	for (uint32_t i = 0; i < result_count; i++)
	{
		results[i].posting_head = cands[i].child_blkno;
		results[i].medoid_tid	= cands[i].medoid_tid;
		results[i].distance		= cands[i].distance;
	}

	mkt_free(next);
	mkt_free(cands);

	return result_count;
}
