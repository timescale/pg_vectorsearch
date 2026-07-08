/*
 * mktann_scan.h - Index scan for mktann
 *
 * Implements ambeginscan, amrescan, amgettuple, amendscan.
 * Reads the centroid tree via beam search and returns medoid TIDs.
 */

#ifndef MKTANN_SCAN_H
#define MKTANN_SCAN_H

#include <postgres.h>

#include <access/amapi.h>
#include <access/relscan.h>
#include <portability/instr_time.h>

/* ----------------------------------------------------------------
 * Scan statistics (populated by execute_search)
 * ---------------------------------------------------------------- */

typedef struct MktannScanStats
{
	/* Phase timing, wall-clock nanoseconds (accumulated across rescans) */
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
} MktannScanStats;

const MktannScanStats *mktann_scan_get_stats(IndexScanDesc scan);

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

#endif /* MKTANN_SCAN_H */
