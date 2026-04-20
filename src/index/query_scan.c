/*
 * query_scan.c - Shared posting list scan for query execution
 *
 * The cluster iteration loop, shared between standalone and PG.
 * Per-cluster: begin scan → read pt_centroid → init query state →
 * score+prune → end.
 */

#include "index/query_scan.h"

void
mkt_query_scan_clusters(
		MktQueryScanParams		*params,
		const MktCentroidResult *beam_results,
		uint32_t				 n_results,
		MktTopK					*topk)
{
	Dimension dim = params->dim;

	for (uint32_t j = 0; j < n_results; j++)
	{
		BlockNumber ph = beam_results[j].posting_head;
		if (ph == InvalidBlockNumber)
			continue;

		/* Begin scan — eagerly reads the first page */
		mkt_posting_scan_begin_cluster(
				params->posting_scan, params->cluster_qs, ph);

		/* Read P^T * centroid from the first posting page */
		const float *pt_cent = mkt_posting_scan_pt_centroid(
				params->posting_scan);
		if (pt_cent == NULL)
		{
			mkt_posting_scan_end_cluster(params->posting_scan);
			continue;
		}

		/* Per-cluster query state from pt_centroid on page.
		 * O(dim) subtraction — same path for standalone and PG. */
		mkt_rabitq_init_query_state(
				params->cluster_qs,
				params->pt_query,
				pt_cent,
				dim,
				params->mode);

		mkt_posting_scan_cluster(params->posting_scan, topk);
		params->total_posting_pages += params->posting_scan->pages_read;
		params->total_posting_entries += params->posting_scan->entries_scanned;
		mkt_posting_scan_end_cluster(params->posting_scan);
	}
}
