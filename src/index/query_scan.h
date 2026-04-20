/*
 * query_scan.h - Shared posting list scan for query execution
 *
 * Scans posting lists for selected clusters, inserting approximate
 * distances into a MktTopK. Used by both standalone and PG query
 * paths — the only backend differences are how pt_centroids and
 * posting heads are obtained, and how reranking is done (caller's
 * responsibility).
 */

#ifndef MKT_QUERY_SCAN_H
#define MKT_QUERY_SCAN_H

#include "algo/topk.h"
#include "index/centroid_search.h"
#include "index/posting_scan.h"
#include "quant/rabitq.h"

/* ----------------------------------------------------------------
 * Scan parameters — populated by the caller, passed to the
 * shared scan function. All pointers must remain valid for the
 * duration of the scan.
 * ---------------------------------------------------------------- */

typedef struct MktQueryScanParams
{
	/* Pre-initialized, reused across clusters */
	MktPostingScan	 *posting_scan;
	RaBitQQueryState *cluster_qs; /* reused per cluster */

	/* Query vectors */
	const float *pt_query; /* P^T * query, precomputed once */

	Dimension		dim;
	MktDistanceMode mode;

	/* Output stats (accumulated across clusters) */
	uint32_t total_posting_pages;
	uint32_t total_posting_entries;
} MktQueryScanParams;

/*
 * Scan posting lists for selected clusters.
 *
 * For each beam result with a valid posting_head, initializes the
 * per-cluster RaBitQ query state from pt_centroids, then runs
 * mkt_posting_scan_cluster to score + prune into topk.
 *
 * The posting_head in each beam result is used directly as the
 * BlockNumber for mkt_posting_scan_begin_cluster.
 */
void mkt_query_scan_clusters(
		MktQueryScanParams		*params,
		const MktCentroidResult *beam_results,
		uint32_t				 n_results,
		MktTopK					*topk);

#endif /* MKT_QUERY_SCAN_H */
