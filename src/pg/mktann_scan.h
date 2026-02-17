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
