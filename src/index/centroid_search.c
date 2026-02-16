/*
 * centroid_search.c - Beam search over centroid tree
 *
 * Implements level-by-level descent through centroid pages using
 * RaBitQ distance on the in-page AoS entries. Each level narrows
 * candidates down to beam_width (or nprobe at leaf level).
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
	ItemPointerData medoid_tid;
	Distance		distance;
} Candidate;

/* ----------------------------------------------------------------
 * Top-K selection
 *
 * Selects the k smallest candidates by distance, placing them
 * sorted in cands[0..k). Returns actual count (min of k, count).
 * ---------------------------------------------------------------- */
typedef uint32_t (*SelectTopK)(Candidate *cands, uint32_t count, uint32_t k);

/*
 * Default: insertion sort. For typical centroid counts (32-256),
 * this is fast enough and simpler than a heap or quickselect.
 */
static uint32_t
select_topk_sort(Candidate *cands, uint32_t count, uint32_t k)
{
	if (k > count)
		k = count;

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

	return k;
}

/*
 * Score all centroids on a single page, appending to candidates.
 * Returns the new candidate count.
 *
 * scratch[] must have room for at least max_per_page distances.
 */
static uint32_t
score_page(
		const MktCentroidSearchState *state,
		Page						  page,
		Dimension					  dim,
		uint32_t					  max_per_page,
		Distance					 *scratch,
		Candidate					 *cands,
		uint32_t					  cand_count,
		uint32_t					  cand_cap)
{
	MktCentroidPageOpaque *opaque = MKT_CENTROID_OPAQUE(page);
	uint16_t			   count  = opaque->entry_count;

	(void)max_per_page;

	if (count == 0)
		return cand_count;

	/* Compute distances from AoS RaBitQData entries */
	for (uint16_t i = 0; i < count; i++)
	{
		const RaBitQData *data = mkt_centroid_data(page, i, dim);
		scratch[i]			   = mkt_rabitq_distance(state->qstate, data, dim);
	}

	/* Convert to candidates */
	for (uint16_t i = 0; i < count && cand_count < cand_cap; i++)
	{
		const MktCentroidEntryMeta *meta = mkt_centroid_meta(page, i);
		cands[cand_count].child_blkno	 = meta->child_blkno;
		cands[cand_count].medoid_tid	 = meta->medoid_tid;
		cands[cand_count].distance		 = scratch[i];
		cand_count++;
	}

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

	Dimension  dim		  = state->dim;
	uint32_t   beam_width = state->beam_width;
	uint32_t   nprobe	  = state->nprobe;
	SelectTopK select	  = select_topk_sort;

	/*
	 * Maximum candidates at any level. With branch factor K and
	 * beam_width B, each level can produce at most B * K candidates.
	 * We cap at a reasonable maximum to bound memory usage.
	 */
	uint32_t max_per_page = mkt_centroid_max_entries(dim);
	uint32_t cand_cap	  = beam_width * max_per_page;
	if (cand_cap < max_per_page * 4)
		cand_cap = max_per_page * 4;

	Candidate *cands   = mkt_alloc(cand_cap * sizeof(Candidate));
	Candidate *next	   = mkt_alloc(cand_cap * sizeof(Candidate));
	Distance  *scratch = mkt_alloc(max_per_page * sizeof(Distance));

	/* Level 0: read root centroid page(s), score ALL centroids */
	uint32_t	cand_count = 0;
	BlockNumber blkno	   = first_centroid_blkno;

	while (blkno != InvalidBlockNumber)
	{
		Page page  = mkt_storage_read_page(state->storage, blkno);
		cand_count = score_page(
				state,
				page,
				dim,
				max_per_page,
				scratch,
				cands,
				cand_count,
				cand_cap);
		MktCentroidPageOpaque *opaque	  = MKT_CENTROID_OPAQUE(page);
		BlockNumber			   next_blkno = opaque->next_blkno;
		mkt_storage_release_page(state->storage, blkno);
		blkno = next_blkno;
	}

	/* Select top beam_width from level 0 */
	uint32_t keep = (nlevels == 1) ? nprobe : beam_width;
	cand_count	  = select(cands, cand_count, keep);

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
				Page page  = mkt_storage_read_page(state->storage, cb);
				next_count = score_page(
						state,
						page,
						dim,
						max_per_page,
						scratch,
						next,
						next_count,
						cand_cap);
				MktCentroidPageOpaque *opaque = MKT_CENTROID_OPAQUE(page);
				BlockNumber			   nb	  = opaque->next_blkno;
				mkt_storage_release_page(state->storage, cb);
				cb = nb;
			}
		}

		/* Select top-K for next level */
		keep	   = (level == nlevels - 1) ? nprobe : beam_width;
		next_count = select(next, next_count, keep);

		/* Swap buffers */
		Candidate *tmp = cands;
		cands		   = next;
		next		   = tmp;
		cand_count	   = next_count;
	}

	/* Build results from final candidates */
	uint32_t result_count = cand_count;
	for (uint32_t i = 0; i < result_count; i++)
	{
		results[i].posting_head = cands[i].child_blkno;
		results[i].medoid_tid	= cands[i].medoid_tid;
		results[i].distance		= cands[i].distance;
	}

	mkt_free(scratch);
	mkt_free(next);
	mkt_free(cands);

	return result_count;
}
