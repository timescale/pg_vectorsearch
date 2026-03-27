/*
 * query.h - Zero-allocation query execution
 *
 * Pre-allocated query context for executing queries against an
 * MktIndex. All buffers are allocated once at creation time;
 * the query hot path does zero allocations.
 */

#ifndef MKT_STANDALONE_QUERY_H
#define MKT_STANDALONE_QUERY_H

#include "index/centroid_search.h"
#include "standalone/index.h"

/* Opaque query context */
typedef struct MktQueryCtx MktQueryCtx;

/*
 * Create a query context for the given index.
 *
 * Pre-allocates all buffers needed for beam search, cluster scan,
 * and top-K collection. max_k and max_nprobe define the upper
 * bounds for query parameters.
 *
 * Returns NULL on failure.
 */
MktQueryCtx *
mkt_query_ctx_create(MktIndex *idx, uint32_t max_k, uint32_t max_nprobe);

/*
 * Free query context and all owned buffers.
 */
void mkt_query_ctx_destroy(MktQueryCtx *ctx);

/*
 * Execute a query. Zero allocations on this path.
 *
 * Finds the k nearest neighbors by:
 *   1. Beam search over centroid pages → nprobe clusters
 *   2. Scan posting lists in selected clusters
 *   3. Return top-k result IDs
 *
 * result_ids must have space for k entries.
 * Returns actual number of results (<= k).
 */
uint32_t mkt_query_exec(
		MktQueryCtx	   *ctx,
		const float	   *query,
		uint32_t		k,
		uint32_t		nprobe,
		MktDistanceMode mode,
		uint32_t	   *result_ids);

#endif /* MKT_STANDALONE_QUERY_H */
