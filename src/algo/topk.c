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
#include "core/idset.h"
#include "core/memory.h"
#include "core/platform.h"

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
	topk->memctx	 = mkt_memctx_create(NULL, "topk");
	topk->k			 = k;
	topk->k_capacity = k;
	topk->ub_heap	 = mkt_memctx_alloc(topk->memctx, k * sizeof(Distance));
	topk->ub_ids	 = mkt_memctx_alloc(topk->memctx, k * sizeof(uint64_t));
	topk->ub_count	 = 0;

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
	mkt_topk_reset_to_k(topk, topk->k);
}

void
mkt_topk_reset_to_k(MktTopK *topk, uint32_t k)
{
	/* O(1) reset: buffers are retained across resets (grow-only), so a
	 * reused top-K reaches a steady state with zero allocator traffic.
	 * Only a k above the allocated capacity re-allocates — reset the
	 * arena first so the outgrown blocks are reclaimed rather than
	 * leaked into the context. The candidate buffer keeps its
	 * high-water capacity, avoiding the doubling-regrowth copies that
	 * a fresh buffer paid on every use. */
	if (mkt_unlikely(k > topk->k_capacity))
	{
		uint32_t cand_cap = topk->cand_capacity;
		uint32_t min_cap  = k * 2;

		if (cand_cap < min_cap)
			cand_cap = min_cap;
		if (cand_cap < MKT_TOPK_INITIAL_CAP_MIN)
			cand_cap = MKT_TOPK_INITIAL_CAP_MIN;

		mkt_memctx_reset(topk->memctx);
		topk->ub_heap = mkt_memctx_alloc(topk->memctx, k * sizeof(Distance));
		topk->ub_ids  = mkt_memctx_alloc(topk->memctx, k * sizeof(uint64_t));
		topk->candidates = mkt_memctx_alloc(
				topk->memctx, cand_cap * sizeof(MktTopKEntry));
		topk->cand_capacity = cand_cap;
		topk->k_capacity	= k;
	}

	topk->k			 = k;
	topk->ub_count	 = 0;
	topk->cand_count = 0;
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
			.src	  = topk->cur_src,
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
			.src	  = topk->cur_src,
	};
}

/*
 * Deduplicate a distance-sorted candidate prefix in place, keeping the
 * first (best-distance) entry per ID, tracked in an id set. A rescan
 * of prior survivors per entry would go quadratic in the survivor
 * count; the set keeps this pass linear. Returns the deduplicated
 * count, additionally truncated to `cap` unique entries when cap > 0.
 */
static uint32_t
dedup_sorted_prefix(MktTopKEntry *results, uint32_t n, uint32_t cap)
{
	if (n <= 1)
		return n;

	MktIdSet seen;
	mkt_idset_init(&seen, n);

	uint32_t w = 0;
	for (uint32_t r = 0; r < n; r++)
	{
		if (!mkt_idset_test_add(&seen, results[r].id))
			continue;
		results[w++] = results[r];
		if (cap > 0 && w == cap)
			break;
	}
	mkt_idset_cleanup(&seen);
	return w;
}

/*
 * Partially order `results` so the `want` smallest-distance entries
 * occupy the front (in arbitrary order within each side). Iterative
 * Hoare quickselect with median-of-three pivots: expected O(n), no
 * recursion, no external randomness.
 */
static void
quickselect_by_distance(MktTopKEntry *results, uint32_t n, uint32_t want)
{
	uint32_t lo = 0, hi = n;

	while (hi - lo > 1 && want > lo && want < hi)
	{
		uint32_t mid = lo + (hi - lo) / 2;
		Distance a	 = results[lo].distance;
		Distance b	 = results[mid].distance;
		Distance c	 = results[hi - 1].distance;
		Distance pivot;
		if ((a <= b && b <= c) || (c <= b && b <= a))
			pivot = b;
		else if ((b <= a && a <= c) || (c <= a && a <= b))
			pivot = a;
		else
			pivot = c;

		/* The median-of-three pivot exists in the range, so each scan
		 * has a stopper; the explicit bounds checks make that local
		 * rather than an invariant to trust (j is unsigned, so an
		 * unchecked descent past lo would wrap). */
		uint32_t i = lo, j = hi - 1;
		while (i <= j)
		{
			while (i < hi - 1 && results[i].distance < pivot)
				i++;
			while (j > lo && results[j].distance > pivot)
				j--;
			if (i <= j)
			{
				MktTopKEntry tmp = results[i];
				results[i]		 = results[j];
				results[j]		 = tmp;
				i++;
				if (j == 0)
					break;
				j--;
			}
		}
		/* Entries [lo, j] <= pivot <= entries [i, hi). Recurse into the
		 * side containing the selection boundary. */
		if (want <= j + 1)
			hi = j + 1;
		else if (want >= i)
			lo = i;
		else
			return; /* boundary falls in the pivot-equal middle band */
	}
}

/* ----------------------------------------------------------------
 * Extract sorted
 * ---------------------------------------------------------------- */

/* The uncapped form of mkt_topk_extract_sorted_capped. */
void
mkt_topk_extract_sorted(
		MktTopK *topk, MktTopKEntry *results, uint32_t *count_out)
{
	mkt_topk_extract_sorted_capped(topk, results, count_out, 0);
}

/*
 * Extract the threshold survivors sorted by distance ascending and
 * deduplicated (first, i.e. best-distance, entry per id), keeping at
 * most the best `cap` unique candidates; cap == 0 keeps them all. The
 * survivor count is unbounded when distance estimates are noisy -- a
 * loose threshold admits most scanned entries, every SOAR replica
 * included -- so sorting all survivors is wasted work when only a
 * small prefix is kept. A quickselect partition narrows to the
 * smallest 3*cap entries first: every id occurs at most twice (primary
 * posting plus at most one SOAR replica), so the smallest 2*cap
 * entries already contain the best cap unique ids and 3*cap leaves
 * margin.
 */
void
mkt_topk_extract_sorted_capped(
		MktTopK		 *topk,
		MktTopKEntry *results,
		uint32_t	 *count_out,
		uint32_t	  cap)
{
	Distance threshold = mkt_topk_threshold(topk);

	uint32_t out = 0;
	for (uint32_t i = 0; i < topk->cand_count; i++)
	{
		Distance lb = topk->candidates[i].distance - topk->candidates[i].error;
		if (lb <= threshold)
			results[out++] = topk->candidates[i];
	}

	uint32_t sel = out;
	if (cap > 0 && out > cap * 3)
	{
		sel = cap * 3;
		quickselect_by_distance(results, out, sel);
	}

	if (sel > 1)
		qsort(results, sel, sizeof(MktTopKEntry), cmp_by_distance);

	*count_out = dedup_sorted_prefix(results, sel, cap);
}

/*
 * Same as mkt_topk_extract_sorted but skips the duplicate-id pass.
 * For callers whose ids are unique by construction (array indices,
 * beam-search buffer positions), the dedup scan can never remove
 * anything and is quadratic in the survivor count.
 */
void
mkt_topk_extract_sorted_unique(
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

	*count_out = out;
}
