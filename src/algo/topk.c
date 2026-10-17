/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * topk.c - Top-K collection
 *
 * Threshold heap: max-heap of K Distance values (upper bounds).
 * Candidate buffer: growable array of all entries that passed
 * the threshold check.
 * Id map: heap position of each id, for vs_topk_insert's dedup lookup.
 *
 * Used by both standalone and PG builds.
 */

#include <stdlib.h>
#include <string.h>

#include "algo/topk.h"
#include "core/idset.h"
#include "core/memory.h"
#include "core/platform.h"

#define VS_TOPK_INITIAL_CAP_MIN 32

/* ----------------------------------------------------------------
 * Id -> heap position map
 *
 * Linear probing over a power-of-two slot array sized for at most half
 * load at k_capacity. Deletion shifts later entries back into the hole
 * rather than leaving tombstones, so probe chains stay short however
 * many evictions a scan makes.
 * ---------------------------------------------------------------- */

#define VS_TOPK_MAP_EMPTY UINT32_MAX

static inline uint32_t
map_home(const VsTopK *topk, uint64_t id)
{
	return (uint32_t)((id * VS_HASH_GOLDEN_GAMMA) >> topk->map_shift);
}

/* Slot holding `id`, or VS_TOPK_MAP_EMPTY if it is not in the heap. */
static inline uint32_t
map_find(const VsTopK *topk, uint64_t id)
{
	uint32_t slot = map_home(topk, id);
	while (topk->map_pos[slot] != VS_TOPK_MAP_EMPTY)
	{
		if (topk->map_ids[slot] == id)
			return slot;
		slot = (slot + 1) & topk->map_mask;
	}
	return VS_TOPK_MAP_EMPTY;
}

/* Record that heap position `pos` holds `id` (absent from the map). */
static inline void
map_add(VsTopK *topk, uint64_t id, uint32_t pos)
{
	uint32_t slot = map_home(topk, id);
	while (topk->map_pos[slot] != VS_TOPK_MAP_EMPTY)
		slot = (slot + 1) & topk->map_mask;
	topk->map_ids[slot] = id;
	topk->map_pos[slot] = pos;
	topk->ub_slot[pos]	= slot;
}

static void
map_remove(VsTopK *topk, uint32_t slot)
{
	uint32_t mask = topk->map_mask;
	uint32_t hole = slot;

	for (uint32_t j = (hole + 1) & mask; topk->map_pos[j] != VS_TOPK_MAP_EMPTY;
		 j			= (j + 1) & mask)
	{
		/* Move j into the hole unless that would put it before its
		 * home slot, where lookups would no longer reach it. */
		uint32_t home = map_home(topk, topk->map_ids[j]);
		if (((j - home) & mask) >= ((j - hole) & mask))
		{
			topk->map_ids[hole]				   = topk->map_ids[j];
			topk->map_pos[hole]				   = topk->map_pos[j];
			topk->ub_slot[topk->map_pos[hole]] = hole;
			hole							   = j;
		}
	}
	topk->map_pos[hole] = VS_TOPK_MAP_EMPTY;
}

/* Allocate the map on the first deduping insert, indexing whatever
 * unique inserts already put in the heap. */
static void
map_build(VsTopK *topk)
{
	uint32_t bits = 1;
	while ((1ull << bits) < 2ull * topk->k_capacity)
		bits++;

	uint32_t nslots = 1u << bits;
	topk->map_ids	= vs_memctx_alloc(topk->memctx, nslots * sizeof(uint64_t));
	topk->map_pos	= vs_memctx_alloc(topk->memctx, nslots * sizeof(uint32_t));
	topk->ub_slot =
			vs_memctx_alloc(topk->memctx, topk->k_capacity * sizeof(uint32_t));
	topk->map_mask	= nslots - 1;
	topk->map_shift = 64 - bits;

	memset(topk->map_pos, 0xFF, nslots * sizeof(uint32_t));
	for (uint32_t i = 0; i < topk->ub_count; i++)
		map_add(topk, topk->ub_ids[i], i);
}

/* ----------------------------------------------------------------
 * Threshold heap (max-heap of Distance values)
 *
 * The sifts take the heap's arrays as locals, so stores through them
 * don't force reloads of the pointers from VsTopK. With `track`, every
 * move also updates the id map.
 * ---------------------------------------------------------------- */

typedef struct HeapArrays
{
	Distance *ub;
	uint64_t *ids;
	uint32_t *slot; /* ub_slot, used only when tracking */
	uint32_t *pos;	/* map_pos, used only when tracking */
} HeapArrays;

static inline HeapArrays
heap_arrays(const VsTopK *topk)
{
	return (HeapArrays){
			.ub	  = topk->ub_heap,
			.ids  = topk->ub_ids,
			.slot = topk->ub_slot,
			.pos  = topk->map_pos,
	};
}

static inline void
ub_swap(HeapArrays h, uint32_t a, uint32_t b, bool track)
{
	Distance tmp = h.ub[a];
	h.ub[a]		 = h.ub[b];
	h.ub[b]		 = tmp;
	uint64_t tid = h.ids[a];
	h.ids[a]	 = h.ids[b];
	h.ids[b]	 = tid;

	if (track)
	{
		uint32_t sa = h.slot[a];
		uint32_t sb = h.slot[b];
		h.slot[a]	= sb;
		h.slot[b]	= sa;
		h.pos[sb]	= a;
		h.pos[sa]	= b;
	}
}

static inline void
ub_sift_up(HeapArrays h, uint32_t i, bool track)
{
	while (i > 0)
	{
		uint32_t parent = (i - 1) / 2;
		if (h.ub[i] <= h.ub[parent])
			break;
		ub_swap(h, i, parent, track);
		i = parent;
	}
}

static inline void
ub_sift_down(HeapArrays h, uint32_t count, uint32_t i, bool track)
{
	for (;;)
	{
		uint32_t left	 = 2 * i + 1;
		uint32_t right	 = 2 * i + 2;
		uint32_t largest = i;

		if (left < count && h.ub[left] > h.ub[largest])
			largest = left;
		if (right < count && h.ub[right] > h.ub[largest])
			largest = right;

		if (largest == i)
			break;

		ub_swap(h, i, largest, track);
		i = largest;
	}
}

/* Add an id that isn't in the heap. It is appended while the heap holds
 * fewer than k entries and otherwise replaces the root if ub beats it. */
__attribute__((always_inline)) static inline void
ub_insert(VsTopK *topk, Distance ub, uint64_t id, bool track)
{
	if (topk->ub_count < topk->k)
	{
		uint32_t i		 = topk->ub_count++;
		topk->ub_heap[i] = ub;
		topk->ub_ids[i]	 = id;
		if (track)
			map_add(topk, id, i);
		ub_sift_up(heap_arrays(topk), i, track);
	}
	else if (ub < topk->ub_heap[0])
	{
		if (track)
			map_remove(topk, topk->ub_slot[0]);
		topk->ub_heap[0] = ub;
		topk->ub_ids[0]	 = id;
		if (track)
			map_add(topk, id, 0);
		ub_sift_down(heap_arrays(topk), topk->ub_count, 0, track);
	}
}

/* ----------------------------------------------------------------
 * Comparator for qsort (ascending by distance)
 * ---------------------------------------------------------------- */

static int
cmp_by_distance(const void *a, const void *b)
{
	const VsTopKEntry *ea = (const VsTopKEntry *)a;
	const VsTopKEntry *eb = (const VsTopKEntry *)b;
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
vs_topk_init(VsTopK *topk, uint32_t k)
{
	topk->memctx	 = vs_memctx_create(NULL, "topk");
	topk->k			 = k;
	topk->k_capacity = k;
	topk->ub_heap	 = vs_memctx_alloc(topk->memctx, k * sizeof(Distance));
	topk->ub_ids	 = vs_memctx_alloc(topk->memctx, k * sizeof(uint64_t));
	topk->ub_count	 = 0;

	uint32_t cap = k * 2;
	if (cap < VS_TOPK_INITIAL_CAP_MIN)
		cap = VS_TOPK_INITIAL_CAP_MIN;
	topk->candidates =
			vs_memctx_alloc(topk->memctx, cap * sizeof(VsTopKEntry));
	topk->cand_count	= 0;
	topk->cand_capacity = cap;
	/* Stamped onto every insert; only the query scan ever sets it. */
	topk->cur_src = 0;

	topk->map_ids	= NULL;
	topk->map_pos	= NULL;
	topk->ub_slot	= NULL;
	topk->map_mask	= 0;
	topk->map_shift = 0;
}

void
vs_topk_cleanup(VsTopK *topk)
{
	if (topk == NULL)
		return;
	if (topk->memctx != NULL)
	{
		vs_memctx_delete(topk->memctx);
		topk->memctx = NULL;
	}
	topk->ub_heap	 = NULL;
	topk->ub_ids	 = NULL;
	topk->candidates = NULL;
	topk->map_ids	 = NULL;
	topk->map_pos	 = NULL;
	topk->ub_slot	 = NULL;
}

VsTopK *
vs_topk_create(uint32_t k)
{
	/* The handle comes from the caller's context rather than from the arena
	 * it owns. vs_topk_reset_to_k resets that arena when k outgrows it,
	 * which would reclaim the object being reset through. */
	VsTopK *topk = vs_alloc(sizeof(VsTopK));
	vs_topk_init(topk, k);
	return topk;
}

void
vs_topk_destroy(VsTopK *topk)
{
	if (topk == NULL)
		return;
	vs_topk_cleanup(topk);
	vs_free(topk);
}

void
vs_topk_reset(VsTopK *topk)
{
	vs_topk_reset_to_k(topk, topk->k);
}

void
vs_topk_reset_to_k(VsTopK *topk, uint32_t k)
{
	/* O(1) reset: buffers are retained across resets (grow-only), so a
	 * reused top-K reaches a steady state with zero allocator traffic.
	 * Only a k above the allocated capacity re-allocates — reset the
	 * arena first so the outgrown blocks are reclaimed rather than
	 * leaked into the context. The candidate buffer keeps its
	 * high-water capacity, avoiding the doubling-regrowth copies that
	 * a fresh buffer paid on every use. */
	if (vs_unlikely(k > topk->k_capacity))
	{
		uint32_t cand_cap = topk->cand_capacity;
		uint32_t min_cap  = k * 2;

		if (cand_cap < min_cap)
			cand_cap = min_cap;
		if (cand_cap < VS_TOPK_INITIAL_CAP_MIN)
			cand_cap = VS_TOPK_INITIAL_CAP_MIN;

		vs_memctx_reset(topk->memctx);
		topk->ub_heap = vs_memctx_alloc(topk->memctx, k * sizeof(Distance));
		topk->ub_ids  = vs_memctx_alloc(topk->memctx, k * sizeof(uint64_t));
		topk->candidates =
				vs_memctx_alloc(topk->memctx, cand_cap * sizeof(VsTopKEntry));
		topk->cand_capacity = cand_cap;
		topk->k_capacity	= k;
		topk->map_ids		= NULL;
		topk->map_pos		= NULL;
		topk->ub_slot		= NULL;
	}
	else if (topk->map_pos != NULL)
	{
		/* Clear only the occupied slots. */
		for (uint32_t i = 0; i < topk->ub_count; i++)
			topk->map_pos[topk->ub_slot[i]] = VS_TOPK_MAP_EMPTY;
	}

	topk->k			 = k;
	topk->ub_count	 = 0;
	topk->cand_count = 0;
}

void
vs_topk_insert_unique(
		VsTopK *topk, Distance distance, Distance error, uint64_t id)
{
	Distance lb = distance - error;
	Distance ub = distance + error;

	/* Prune: lower bound exceeds threshold. Written NaN-safe: a NaN
	 * estimate (e.g. a zero vector normalized for cosine) fails every
	 * comparison, so the naive lb >= threshold guard would admit it --
	 * and one NaN upper bound in the heap corrupts the threshold for
	 * the rest of the scan. */
	if (!(lb < vs_topk_threshold(topk)))
		return;

	/* No dedup lookup: caller guarantees ids are unique. A map that
	 * vs_topk_insert built is kept current. */
	if (topk->map_pos == NULL)
		ub_insert(topk, ub, id, false);
	else
		ub_insert(topk, ub, id, true);

	/* Grow candidate buffer if needed (old buffer freed with memctx) */
	if (topk->cand_count == topk->cand_capacity)
	{
		uint32_t	 new_cap = topk->cand_capacity * 2;
		VsTopKEntry *new_buf =
				vs_memctx_alloc(topk->memctx, new_cap * sizeof(VsTopKEntry));
		memcpy(new_buf,
			   topk->candidates,
			   topk->cand_count * sizeof(VsTopKEntry));
		topk->candidates	= new_buf;
		topk->cand_capacity = new_cap;
	}

	topk->candidates[topk->cand_count++] = (VsTopKEntry){
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
vs_topk_insert(VsTopK *topk, Distance distance, Distance error, uint64_t id)
{
	Distance lb = distance - error;
	Distance ub = distance + error;

	/* Prune: lower bound exceeds threshold. Written NaN-safe: a NaN
	 * estimate (e.g. a zero vector normalized for cosine) fails every
	 * comparison, so the naive lb >= threshold guard would admit it --
	 * and one NaN upper bound in the heap corrupts the threshold for
	 * the rest of the scan. */
	if (!(lb < vs_topk_threshold(topk)))
		return;

	/* Dedup: if this ID is already in the heap, keep only the better
	 * (smaller) upper bound. This prevents duplicates (e.g. SOAR
	 * replicas) from occupying multiple heap slots and inflating the
	 * threshold. */
	if (vs_unlikely(topk->map_pos == NULL))
		map_build(topk);

	uint32_t slot = map_find(topk, id);
	if (slot != VS_TOPK_MAP_EMPTY)
	{
		uint32_t i = topk->map_pos[slot];
		if (ub < topk->ub_heap[i])
		{
			topk->ub_heap[i] = ub;
			ub_sift_down(heap_arrays(topk), topk->ub_count, i, true);
		}
		goto append;
	}

	ub_insert(topk, ub, id, true);

append:
	/* Grow candidate buffer if needed (old buffer freed with memctx) */
	if (topk->cand_count == topk->cand_capacity)
	{
		uint32_t	 new_cap = topk->cand_capacity * 2;
		VsTopKEntry *new_buf =
				vs_memctx_alloc(topk->memctx, new_cap * sizeof(VsTopKEntry));
		memcpy(new_buf,
			   topk->candidates,
			   topk->cand_count * sizeof(VsTopKEntry));
		topk->candidates	= new_buf;
		topk->cand_capacity = new_cap;
	}

	/* Append to candidate buffer */
	topk->candidates[topk->cand_count++] = (VsTopKEntry){
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
dedup_sorted_prefix(VsTopKEntry *results, uint32_t n, uint32_t cap)
{
	if (n <= 1)
		return n;

	VsIdSet seen;
	vs_idset_init(&seen, n);

	uint32_t w = 0;
	for (uint32_t r = 0; r < n; r++)
	{
		if (!vs_idset_test_add(&seen, results[r].id))
			continue;
		results[w++] = results[r];
		if (cap > 0 && w == cap)
			break;
	}
	vs_idset_cleanup(&seen);
	return w;
}

/*
 * Partially order `results` so the `want` smallest-distance entries
 * occupy the front (in arbitrary order within each side). Iterative
 * Hoare quickselect with median-of-three pivots: expected O(n), no
 * recursion, no external randomness.
 */
static void
quickselect_by_distance(VsTopKEntry *results, uint32_t n, uint32_t want)
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
				VsTopKEntry tmp = results[i];
				results[i]		= results[j];
				results[j]		= tmp;
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

/* The uncapped form of vs_topk_extract_sorted_capped. */
void
vs_topk_extract_sorted(VsTopK *topk, VsTopKEntry *results, uint32_t *count_out)
{
	vs_topk_extract_sorted_capped(topk, results, count_out, 0);
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
vs_topk_extract_sorted_capped(
		VsTopK *topk, VsTopKEntry *results, uint32_t *count_out, uint32_t cap)
{
	Distance threshold = vs_topk_threshold(topk);

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
		qsort(results, sel, sizeof(VsTopKEntry), cmp_by_distance);

	*count_out = dedup_sorted_prefix(results, sel, cap);
}

/*
 * Same as vs_topk_extract_sorted but skips the duplicate-id pass.
 * For callers whose ids are unique by construction (array indices,
 * beam-search buffer positions), the dedup scan can never remove
 * anything and is quadratic in the survivor count.
 */
void
vs_topk_extract_sorted_unique(
		VsTopK *topk, VsTopKEntry *results, uint32_t *count_out)
{
	Distance threshold = vs_topk_threshold(topk);

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
		qsort(results, out, sizeof(VsTopKEntry), cmp_by_distance);

	*count_out = out;
}
