/*
 * query_scan.h - Shared query execution for ANN search
 *
 * MktQueryState owns all pre-allocated buffers for query execution.
 * Shared between standalone and PG. The per-query execute path
 * does zero allocations (except rare topk candidate buffer growth).
 *
 * Usage:
 *   MktQueryState qs;
 *   mkt_query_state_init(&qs, &index_base, max_k, max_nprobe);
 *
 *   uint32_t n = mkt_query_execute(&qs, query, k, nprobe, mode, &stats);
 *   // Results in qs.candidates[0..n)
 *
 *   mkt_query_state_cleanup(&qs);
 */

#ifndef MKT_QUERY_SCAN_H
#define MKT_QUERY_SCAN_H

#include "algo/topk.h"
#include "index/centroid_search.h"
#include "index/index_base.h"
#include "index/posting_scan.h"

/* ----------------------------------------------------------------
 * Query statistics
 * ---------------------------------------------------------------- */

typedef struct MktQueryStats
{
	uint32_t centroid_pages_read;
	uint32_t posting_pages_read;
	uint32_t posting_entries_scanned;
	uint32_t clusters_scanned;
} MktQueryStats;

/* ----------------------------------------------------------------
 * Query state — pre-allocated, reused across queries
 * ---------------------------------------------------------------- */

typedef struct MktQueryState
{
	MktIndexBase *index;
	uint32_t	  max_k;
	uint32_t	  max_nprobe;

	/* Pre-allocated query buffers */
	float	*query_buf;
	float	*pt_query;
	float	*beam_transformed;
	float	*cluster_transformed;
	uint8_t *beam_query_bits;
	uint8_t *cluster_query_bits;

	RaBitQQueryState beam_qs;
	RaBitQQueryState cluster_qs;

	/* Pre-allocated search state (reset per query) */
	MktCentroidResult *beam_results;
	MktTopK			   topk;
	MktPostingScan	   pscan;
	MktTopKEntry	  *candidates;
	uint32_t		   cand_cap;
	uint32_t		   ncandidates;
} MktQueryState;

/* ----------------------------------------------------------------
 * API
 * ---------------------------------------------------------------- */

void mkt_query_state_init(
		MktQueryState *qs,
		MktIndexBase  *index,
		uint32_t	   max_k,
		uint32_t	   max_nprobe);

void mkt_query_state_cleanup(MktQueryState *qs);

/*
 * Execute one ANN search query.
 *
 * Normalizes the query (cosine), runs beam search over centroids,
 * scans posting lists, and collects approximate candidates.
 *
 * Returns: number of candidates.
 * Results are in qs->candidates[0..return_count), sorted by
 * distance ascending. The caller owns reranking (if any).
 */
uint32_t mkt_query_execute(
		MktQueryState  *qs,
		const float	   *query,
		uint32_t		k,
		uint32_t		nprobe,
		MktDistanceMode mode,
		MktQueryStats  *stats);

#endif /* MKT_QUERY_SCAN_H */
