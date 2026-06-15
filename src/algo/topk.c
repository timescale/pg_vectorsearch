/*
 * topk.c - Top-K collection
 *
 * Threshold heap: max-heap of K Distance values (upper bounds).
 * Candidate buffer: growable array of all entries that passed
 * the threshold check.
 *
 * Used by both standalone and PG builds.
 */

#include <stdlib.h>
#include <string.h>

#include "algo/topk.h"
#include "core/memory.h"

#define MKT_TOPK_INITIAL_CAP_MIN 32

/* ----------------------------------------------------------------
 * Threshold heap (max-heap of Distance values)
 * ---------------------------------------------------------------- */

static void
ub_sift_up(Distance *heap, uint64_t *ids, uint32_t i)
{
	while (i > 0)
	{
		uint32_t parent = (i - 1) / 2;
		if (heap[i] <= heap[parent])
			break;
		Distance tmp = heap[i];
		heap[i]		 = heap[parent];
		heap[parent] = tmp;
		uint64_t tid = ids[i];
		ids[i]		 = ids[parent];
		ids[parent]	 = tid;
		i			 = parent;
	}
}

static void
ub_sift_down(Distance *heap, uint64_t *ids, uint32_t count)
{
	uint32_t i = 0;
	for (;;)
	{
		uint32_t left	 = 2 * i + 1;
		uint32_t right	 = 2 * i + 2;
		uint32_t largest = i;

		if (left < count && heap[left] > heap[largest])
			largest = left;
		if (right < count && heap[right] > heap[largest])
			largest = right;

		if (largest == i)
			break;

		Distance tmp  = heap[i];
		heap[i]		  = heap[largest];
		heap[largest] = tmp;
		uint64_t tid  = ids[i];
		ids[i]		  = ids[largest];
		ids[largest]  = tid;
		i			  = largest;
	}
}

static void
ub_sift_down_from(Distance *heap, uint64_t *ids, uint32_t count, uint32_t i)
{
	for (;;)
	{
		uint32_t left	 = 2 * i + 1;
		uint32_t right	 = 2 * i + 2;
		uint32_t largest = i;

		if (left < count && heap[left] > heap[largest])
			largest = left;
		if (right < count && heap[right] > heap[largest])
			largest = right;

		if (largest == i)
			break;

		Distance tmp  = heap[i];
		heap[i]		  = heap[largest];
		heap[largest] = tmp;
		uint64_t tid  = ids[i];
		ids[i]		  = ids[largest];
		ids[largest]  = tid;
		i			  = largest;
	}
}

/* ----------------------------------------------------------------
 * Comparator for qsort (ascending by distance)
 * ---------------------------------------------------------------- */

static int
cmp_by_distance(const void *a, const void *b)
{
	const MktTopKEntry *ea = (const MktTopKEntry *)a;
	const MktTopKEntry *eb = (const MktTopKEntry *)b;
	if (ea->distance < eb->distance)
		return -1;
	if (ea->distance > eb->distance)
		return 1;
	return 0;
}

/* ----------------------------------------------------------------
 * Init / cleanup / create / destroy
 * ---------------------------------------------------------------- */

void
mkt_topk_init(MktTopK *topk, uint32_t k)
{
	topk->memctx   = mkt_memctx_create(NULL, "topk");
	topk->k		   = k;
	topk->ub_heap  = mkt_memctx_alloc(topk->memctx, k * sizeof(Distance));
	topk->ub_ids   = mkt_memctx_alloc(topk->memctx, k * sizeof(uint64_t));
	topk->ub_count = 0;

	uint32_t cap = k * 2;
	if (cap < MKT_TOPK_INITIAL_CAP_MIN)
		cap = MKT_TOPK_INITIAL_CAP_MIN;
	topk->candidates =
			mkt_memctx_alloc(topk->memctx, cap * sizeof(MktTopKEntry));
	topk->cand_count	= 0;
	topk->cand_capacity = cap;
}

void
mkt_topk_cleanup(MktTopK *topk)
{
	if (topk == NULL)
		return;
	if (topk->memctx != NULL)
	{
		mkt_memctx_delete(topk->memctx);
		topk->memctx = NULL;
	}
	topk->ub_heap	 = NULL;
	topk->ub_ids	 = NULL;
	topk->candidates = NULL;
}

MktTopK *
mkt_topk_create(uint32_t k)
{
	MktMemCtx ctx  = mkt_memctx_create(NULL, "topk");
	MktTopK	 *topk = mkt_memctx_alloc(ctx, sizeof(MktTopK));
	topk->memctx   = ctx;
	topk->k		   = k;
	topk->ub_heap  = mkt_memctx_alloc(ctx, k * sizeof(Distance));
	topk->ub_ids   = mkt_memctx_alloc(ctx, k * sizeof(uint64_t));
	topk->ub_count = 0;

	uint32_t cap = k * 2;
	if (cap < MKT_TOPK_INITIAL_CAP_MIN)
		cap = MKT_TOPK_INITIAL_CAP_MIN;
	topk->candidates	= mkt_memctx_alloc(ctx, cap * sizeof(MktTopKEntry));
	topk->cand_count	= 0;
	topk->cand_capacity = cap;
	return topk;
}

void
mkt_topk_destroy(MktTopK *topk)
{
	if (topk == NULL)
		return;
	mkt_memctx_delete(topk->memctx);
}

void
mkt_topk_reset(MktTopK *topk)
{
	/* Reset the arena to reclaim grown buffers, then re-allocate
	 * the initial ub_heap and candidates from the fresh arena. */
	mkt_memctx_reset(topk->memctx);

	topk->ub_heap = mkt_memctx_alloc(topk->memctx, topk->k * sizeof(Distance));
	topk->ub_ids  = mkt_memctx_alloc(topk->memctx, topk->k * sizeof(uint64_t));
	topk->ub_count = 0;

	uint32_t cap = topk->k * 2;
	if (cap < MKT_TOPK_INITIAL_CAP_MIN)
		cap = MKT_TOPK_INITIAL_CAP_MIN;
	topk->candidates =
			mkt_memctx_alloc(topk->memctx, cap * sizeof(MktTopKEntry));
	topk->cand_count	= 0;
	topk->cand_capacity = cap;
}

void
mkt_topk_reset_to_k(MktTopK *topk, uint32_t k)
{
	topk->k = k;
	mkt_topk_reset(topk);
}

void
mkt_topk_insert_unique(
		MktTopK *topk, Distance distance, Distance error, uint64_t id)
{
	Distance lb = distance - error;
	Distance ub = distance + error;

	/* Prune: lower bound exceeds threshold */
	if (lb >= mkt_topk_threshold(topk))
		return;

	/* No dedup scan: caller guarantees ids are unique. */
	if (topk->ub_count < topk->k)
	{
		topk->ub_heap[topk->ub_count] = ub;
		topk->ub_ids[topk->ub_count]  = id;
		topk->ub_count++;
		ub_sift_up(topk->ub_heap, topk->ub_ids, topk->ub_count - 1);
	}
	else if (ub < topk->ub_heap[0])
	{
		topk->ub_heap[0] = ub;
		topk->ub_ids[0]	 = id;
		ub_sift_down(topk->ub_heap, topk->ub_ids, topk->ub_count);
	}

	/* Grow candidate buffer if needed (old buffer freed with memctx) */
	if (topk->cand_count == topk->cand_capacity)
	{
		uint32_t	  new_cap = topk->cand_capacity * 2;
		MktTopKEntry *new_buf =
				mkt_memctx_alloc(topk->memctx, new_cap * sizeof(MktTopKEntry));
		memcpy(new_buf,
			   topk->candidates,
			   topk->cand_count * sizeof(MktTopKEntry));
		topk->candidates	= new_buf;
		topk->cand_capacity = new_cap;
	}

	topk->candidates[topk->cand_count++] = (MktTopKEntry){
			.distance = distance,
			.error	  = error,
			.id		  = id,
	};
}

/* ----------------------------------------------------------------
 * Insert
 * ---------------------------------------------------------------- */

void
mkt_topk_insert(MktTopK *topk, Distance distance, Distance error, uint64_t id)
{
	Distance lb = distance - error;
	Distance ub = distance + error;

	/* Prune: lower bound exceeds threshold */
	if (lb >= mkt_topk_threshold(topk))
		return;

	/* Dedup: check if this ID is already in the heap. If so,
	 * only keep the entry with the better (smaller) upper bound.
	 * This prevents duplicates (e.g. SOAR replicas) from occupying
	 * multiple heap slots and inflating the threshold. */
	for (uint32_t i = 0; i < topk->ub_count; i++)
	{
		if (topk->ub_ids[i] == id)
		{
			if (ub < topk->ub_heap[i])
			{
				topk->ub_heap[i] = ub;
				ub_sift_down_from(
						topk->ub_heap, topk->ub_ids, topk->ub_count, i);
			}
			goto append;
		}
	}

	/* Update threshold heap */
	if (topk->ub_count < topk->k)
	{
		topk->ub_heap[topk->ub_count] = ub;
		topk->ub_ids[topk->ub_count]  = id;
		topk->ub_count++;
		ub_sift_up(topk->ub_heap, topk->ub_ids, topk->ub_count - 1);
	}
	else if (ub < topk->ub_heap[0])
	{
		topk->ub_heap[0] = ub;
		topk->ub_ids[0]	 = id;
		ub_sift_down(topk->ub_heap, topk->ub_ids, topk->ub_count);
	}

append:
	/* Grow candidate buffer if needed (old buffer freed with memctx) */
	if (topk->cand_count == topk->cand_capacity)
	{
		uint32_t	  new_cap = topk->cand_capacity * 2;
		MktTopKEntry *new_buf =
				mkt_memctx_alloc(topk->memctx, new_cap * sizeof(MktTopKEntry));
		memcpy(new_buf,
			   topk->candidates,
			   topk->cand_count * sizeof(MktTopKEntry));
		topk->candidates	= new_buf;
		topk->cand_capacity = new_cap;
	}

	/* Append to candidate buffer */
	topk->candidates[topk->cand_count++] = (MktTopKEntry){
			.distance = distance,
			.error	  = error,
			.id		  = id,
	};
}

/* ----------------------------------------------------------------
 * Extract sorted
 * ---------------------------------------------------------------- */

void
mkt_topk_extract_sorted(
		MktTopK *topk, MktTopKEntry *results, uint32_t *count_out)
{
	Distance threshold = mkt_topk_threshold(topk);

	/* Filter stale candidates and copy survivors to results */
	uint32_t out = 0;
	for (uint32_t i = 0; i < topk->cand_count; i++)
	{
		Distance lb = topk->candidates[i].distance - topk->candidates[i].error;
		if (lb <= threshold)
			results[out++] = topk->candidates[i];
	}

	/* Sort by distance ascending */
	if (out > 1)
		qsort(results, out, sizeof(MktTopKEntry), cmp_by_distance);

	/* Deduplicate: keep the first (best distance) for each ID. */
	if (out > 1)
	{
		uint32_t w = 1;
		for (uint32_t r = 1; r < out; r++)
		{
			bool dup = false;
			for (uint32_t j = 0; j < w; j++)
			{
				if (results[r].id == results[j].id)
				{
					dup = true;
					break;
				}
			}
			if (!dup)
				results[w++] = results[r];
		}
		out = w;
	}

	*count_out = out;
}
