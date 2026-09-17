/*
 * scan.h - Index scan for mktann
 *
 * Implements ambeginscan, amrescan, amgettuple, amendscan.
 * Reads the centroid tree via beam search and returns medoid TIDs.
 */

#ifndef MKT_SCAN_H
#define MKT_SCAN_H

#include <postgres.h>

#include <access/amapi.h>
#include <access/relscan.h>

/* ----------------------------------------------------------------
 * Scan statistics (populated by execute_search)
 * ---------------------------------------------------------------- */

typedef struct MktannScanStats
{
	/* Phase timing, wall-clock nanoseconds (from the last query) */
	uint64_t centroid_ns;
	uint64_t posting_ns;
	uint64_t rerank_ns;

	/* Centroid beam search */
	uint32_t clusters_scanned;
	uint32_t centroid_pages_read;

	/* Posting scan */
	uint32_t posting_pages_read;
	uint32_t posting_pages_skipped; /* tombstoned all-dead pages skipped */
	uint32_t posting_entries_scanned;

	/* Rerank */
	uint32_t rerank_candidates;
	uint32_t rerank_heap_fetches;
	uint32_t rerank_results;

	/* Total storage reads */
	uint32_t storage_reads;

	/* Top-k the search was sized for (from the query's LIMIT, seeded by
	 * filter selectivity, or mkt.query_limit, or the built-in default) */
	uint32_t top_k;
} MktannScanStats;

/*
 * Floor on every scan's top-k. Not a default: it keeps a little slack under
 * a small LIMIT for rows the heap fetch discards as dead, and leaves a query
 * something to answer with when work_mem affords less.
 */
#define MKT_DEFAULT_K 10

const MktannScanStats *mktann_scan_get_stats(IndexScanDesc scan);

/*
 * Resolve the top-k a scan will build, from the rows its LIMIT asks for (0
 * when there is none) and the relation's estimated row count (negative when
 * unknown). Applies mkt.query_limit, the built-in floor, and the work_mem
 * ceiling.
 *
 * Called by mktann_beginscan and by the cost model.
 */
uint32_t mkt_scan_resolve_top_k(uint32_t scan_bound, double heap_rows);

/*
 * Begin an index scan. Matches ambeginscan_function signature.
 * Allocates scan state and reads the metadata page.
 */
IndexScanDesc mktann_beginscan(Relation index, int nkeys, int norderbys);

/*
 * Restart a scan with new keys/orderbys. Matches amrescan_function.
 */
void mktann_rescan(
		IndexScanDesc scan,
		ScanKey		  keys,
		int			  nkeys,
		ScanKey		  orderbys,
		int			  norderbys);

/*
 * Return next tuple from scan. Matches amgettuple_function.
 * On first call, runs beam search and caches all results.
 * Subsequent calls iterate through cached results.
 */
bool mktann_gettuple(IndexScanDesc scan, ScanDirection direction);

/*
 * End scan and free resources. Matches amendscan_function.
 */
void mktann_endscan(IndexScanDesc scan);

#endif /* MKT_SCAN_H */
