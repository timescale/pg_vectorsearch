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
	MktTopKEntry *candidates;	 /* growable candidate buffer */
	uint32_t	  cand_count;	 /* buffered candidates */
	uint32_t	  cand_capacity; /* allocated capacity */
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

#endif /* MKT_TOPK_H */
