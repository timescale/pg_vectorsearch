/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
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
 *   VsTopK topk;
 *   vs_topk_init(&topk, k);
 *
 *   vs_topk_insert(&topk, distance, error, id);
 *   // ... more inserts ...
 *
 *   VsTopKEntry *results = vs_alloc(topk.cand_count * sizeof(...));
 *   uint32_t count;
 *   vs_topk_extract_sorted(&topk, results, &count);
 *   vs_topk_cleanup(&topk);
 */

#ifndef VS_TOPK_H
#define VS_TOPK_H

#include <math.h>
#include <stdint.h>

#include "core/memory.h"
#include "core/types.h"

/* ----------------------------------------------------------------
 * Top-K entry
 * ---------------------------------------------------------------- */
typedef struct VsTopKEntry
{
	Distance distance; /* estimated distance */
	Distance error;	   /* symmetric error margin (>= 0) */
	uint64_t id;	   /* encoded TID or generic identifier */
	uint32_t src;	   /* diagnostic: source rank stamped at insert */
} VsTopKEntry;

/* ----------------------------------------------------------------
 * Top-K collection
 *
 * Standalone: binary max-heap array for threshold tracking.
 * PG: pairing heap via opaque pointer (defined in src/pg/topk.c).
 * ---------------------------------------------------------------- */
typedef struct VsTopK
{
	Distance	*ub_heap;		/* binary max-heap of K upper bounds */
	uint64_t	*ub_ids;		/* parallel array: ID for each heap entry */
	uint32_t	 ub_count;		/* entries in threshold heap (<= k) */
	uint32_t	 k;				/* target K */
	uint32_t	 k_capacity;	/* allocated ub_heap/ub_ids capacity */
	VsTopKEntry *candidates;	/* growable candidate buffer */
	uint32_t	 cand_count;	/* buffered candidates */
	uint32_t	 cand_capacity; /* allocated capacity */
	/* When > 0, the candidate buffer is bounded to this many entries and
	 * kept as a max-heap by distance (the cand_limit smallest-distance
	 * survivors), instead of growing without bound. Set for an explicit
	 * rerank_pool, where collecting survivors past the cap is wasted work.
	 * 0 = unbounded (auto pool needs the full survivor count). */
	uint32_t cand_limit;
	uint32_t cur_src; /* diagnostic: rank stamped onto inserts */
	VsMemCtx memctx;  /* owning context for all allocations */
} VsTopK;

/* ----------------------------------------------------------------
 * API
 * ---------------------------------------------------------------- */

/*
 * Initialize a top-K collection. Creates a child memory context
 * under the current context for internal buffers. Caller must be
 * in a context with the desired lifetime.
 * Use vs_topk_cleanup() to free.
 */
void vs_topk_init(VsTopK *topk, uint32_t k);

/*
 * Bound the candidate buffer to `limit` entries (0 = unbounded). When set,
 * the buffer keeps only the `limit` smallest-distance survivors as a
 * max-heap, avoiding unbounded growth for an explicit rerank_pool. Call
 * after reset (expects cand_count == 0); ensures capacity >= limit.
 */
void vs_topk_set_cand_limit(VsTopK *topk, uint32_t limit);

/* Free internal buffers (does not free the VsTopK struct itself). */
void vs_topk_cleanup(VsTopK *topk);

/*
 * Create a heap-allocated top-K collection.
 * Returns NULL on allocation failure.
 */
VsTopK *vs_topk_create(uint32_t k);

/* Free a heap-allocated top-K collection. */
void vs_topk_destroy(VsTopK *topk);

/* Reset to empty state (reuses existing storage). */
void vs_topk_reset(VsTopK *topk);

/*
 * Reset to empty state AND change k. Reuses the existing memory
 * context (preserves the per-topk arena) but re-allocates ub_heap /
 * ub_ids / candidates within it. Cheap compared to a full init +
 * cleanup pair, because the memctx itself isn't created or destroyed.
 *
 * Use this when the same VsTopK is reused across calls that may
 * want different k values (e.g. beam-search keeps beam_width
 * candidates at intermediate levels, nprobe at the last).
 */
void vs_topk_reset_to_k(VsTopK *topk, uint32_t k);

/*
 * Same as vs_topk_insert but skips the O(k) per-insert dedup scan.
 * Use ONLY when the caller guarantees all ids are unique. The cluster
 * scan (where SOAR / boundary replicas can collide) must keep using
 * vs_topk_insert; centroid beam search and similar code that
 * inserts each candidate exactly once should use this fast path.
 */
void vs_topk_insert_unique(
		VsTopK *topk, Distance distance, Distance error, uint64_t id);

/*
 * Insert a candidate. Pruned if lower_bound >= threshold.
 * Otherwise updates the threshold heap and appends to the
 * candidate buffer.
 */
void
vs_topk_insert(VsTopK *topk, Distance distance, Distance error, uint64_t id);

/*
 * Current pruning threshold: the Kth-smallest upper bound
 * (distance + error) seen so far. Returns INFINITY if fewer
 * than K upper bounds have been recorded.
 */
static inline Distance
vs_topk_threshold(const VsTopK *topk)
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
 * Does not reset the collection — call vs_topk_reset() or
 * vs_topk_cleanup() when done with the results.
 */
void vs_topk_extract_sorted_capped(
		VsTopK *topk, VsTopKEntry *results, uint32_t *count_out, uint32_t cap);
void vs_topk_extract_sorted(
		VsTopK *topk, VsTopKEntry *results, uint32_t *count_out);

/* As above, but skips the duplicate-id pass; ids must be unique. */
void vs_topk_extract_sorted_unique(
		VsTopK *topk, VsTopKEntry *results, uint32_t *count_out);

#endif /* VS_TOPK_H */
