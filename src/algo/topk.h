/*
 * topk.h - Bounded top-K collection for nearest neighbor results
 *
 * Collects candidates that could be in the true top-K based on
 * RaBitQ error bounds. Two internal structures work together:
 *
 *   1. Threshold heap: max-heap of K upper bounds (distance + error).
 *      Tracks the Kth-smallest upper bound seen so far. A candidate
 *      is pruned when its lower bound >= this threshold (there are
 *      already K candidates that are definitely closer).
 *
 *   2. Candidate buffer: growable array of all entries that passed
 *      the threshold check. May contain more than K entries because
 *      error intervals can overlap near the Kth position.
 *
 * After the scan, the buffer contains the full rerank set. The
 * caller fetches full-precision vectors for overlapping candidates,
 * reranks, and takes the true top-K.
 *
 * Usage:
 *   MktTopK topk;
 *   mkt_topk_init(&topk, k);
 *
 *   mkt_topk_insert(&topk, distance, error, id);
 *   // ... more inserts ...
 *
 *   MktTopKEntry *results = mkt_alloc(topk.cand_count * sizeof(...));
 *   uint32_t count;
 *   mkt_topk_extract_sorted(&topk, results, &count);
 *   mkt_topk_cleanup(&topk);
 */

#ifndef MKT_TOPK_H
#define MKT_TOPK_H

#include <math.h>
#include <stdint.h>

#include "core/memory.h"
#include "mkt_types.h"

/* ----------------------------------------------------------------
 * Top-K entry
 * ---------------------------------------------------------------- */
typedef struct MktTopKEntry
{
	Distance distance; /* estimated distance */
	Distance error;	   /* symmetric error margin (>= 0) */
	uint64_t id;	   /* encoded TID or generic identifier */
	uint32_t src;	   /* diagnostic: source rank stamped at insert */
} MktTopKEntry;

/* ----------------------------------------------------------------
 * Top-K collection
 *
 * Standalone: binary max-heap array for threshold tracking.
 * PG: pairing heap via opaque pointer (defined in src/pg/topk.c).
 * ---------------------------------------------------------------- */
typedef struct MktTopK
{
	Distance	 *ub_heap;		 /* binary max-heap of K upper bounds */
	uint64_t	 *ub_ids;		 /* parallel array: ID for each heap entry */
	uint32_t	  ub_count;		 /* entries in threshold heap (<= k) */
	uint32_t	  k;			 /* target K */
	uint32_t	  k_capacity;	 /* allocated ub_heap/ub_ids capacity */
	MktTopKEntry *candidates;	 /* growable candidate buffer */
	uint32_t	  cand_count;	 /* buffered candidates */
	uint32_t	  cand_capacity; /* allocated capacity */
	uint32_t	  cur_src;		 /* diagnostic: rank stamped onto inserts */
	MktMemCtx	  memctx;		 /* owning context for all allocations */
} MktTopK;

/* ----------------------------------------------------------------
 * API
 * ---------------------------------------------------------------- */

/*
 * Initialize a top-K collection. Creates a child memory context
 * under the current context for internal buffers. Caller must be
 * in a context with the desired lifetime.
 * Use mkt_topk_cleanup() to free.
 */
void mkt_topk_init(MktTopK *topk, uint32_t k);

/* Free internal buffers (does not free the MktTopK struct itself). */
void mkt_topk_cleanup(MktTopK *topk);

/*
 * Create a heap-allocated top-K collection.
 * Returns NULL on allocation failure.
 */
MktTopK *mkt_topk_create(uint32_t k);

/* Free a heap-allocated top-K collection. */
void mkt_topk_destroy(MktTopK *topk);

/* Reset to empty state (reuses existing storage). */
void mkt_topk_reset(MktTopK *topk);

/*
 * Reset to empty state AND change k. Reuses the existing memory
 * context (preserves the per-topk arena) but re-allocates ub_heap /
 * ub_ids / candidates within it. Cheap compared to a full init +
 * cleanup pair, because the memctx itself isn't created or destroyed.
 *
 * Use this when the same MktTopK is reused across calls that may
 * want different k values (e.g. beam-search keeps beam_width
 * candidates at intermediate levels, nprobe at the last).
 */
void mkt_topk_reset_to_k(MktTopK *topk, uint32_t k);

/*
 * Same as mkt_topk_insert but skips the O(k) per-insert dedup scan.
 * Use ONLY when the caller guarantees all ids are unique. The cluster
 * scan (where SOAR / boundary replicas can collide) must keep using
 * mkt_topk_insert; centroid beam search and similar code that
 * inserts each candidate exactly once should use this fast path.
 */
/* Replay a collected entry through full insert semantics, preserving
 * its scan-rank stamp (parallel merge; see topk.c). */
void mkt_topk_insert_entry(MktTopK *topk, const MktTopKEntry *entry);

void mkt_topk_insert_unique(
		MktTopK *topk, Distance distance, Distance error, uint64_t id);

/*
 * Insert a candidate. Pruned if lower_bound >= threshold.
 * Otherwise updates the threshold heap and appends to the
 * candidate buffer.
 */
void
mkt_topk_insert(MktTopK *topk, Distance distance, Distance error, uint64_t id);

/*
 * Current pruning threshold: the Kth-smallest upper bound
 * (distance + error) seen so far. Returns INFINITY if fewer
 * than K upper bounds have been recorded.
 */
static inline Distance
mkt_topk_threshold(const MktTopK *topk)
{
	if (topk->ub_count < topk->k)
		return INFINITY;
	return topk->ub_heap[0];
}

/*
 * Extract candidates sorted by distance ascending. Filters out
 * stale entries whose lower bound now exceeds the final threshold.
 *
 * results[] must have space for topk->cand_count entries (upper
 * bound; actual count returned via *count_out may be smaller).
 *
 * Does not reset the collection — call mkt_topk_reset() or
 * mkt_topk_cleanup() when done with the results.
 */
void mkt_topk_extract_sorted(
		MktTopK *topk, MktTopKEntry *results, uint32_t *count_out);

/* As above, but skips the duplicate-id pass; ids must be unique. */
void mkt_topk_extract_sorted_unique(
		MktTopK *topk, MktTopKEntry *results, uint32_t *count_out);

#endif /* MKT_TOPK_H */
