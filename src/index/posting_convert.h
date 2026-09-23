/*
 * posting_convert.h - Convert AoS posting pages to fastscan format
 *
 * Reads an AoS posting chain and rewrites it as fastscan pages
 * with SoA group sections and VPSHUFB-packed nibble codes.
 */

#ifndef PRISM_POSTING_CONVERT_H
#define PRISM_POSTING_CONVERT_H

#include "index/posting_page.h"
#include "index/storage.h"

/*
 * Convert a cluster's AoS posting chain to fastscan format.
 *
 * Reads all entries from the AoS chain starting at aos_head,
 * repacks the 1-bit codes into VPSHUFB nibble layout, and writes
 * new fastscan pages. The pt_centroid from the first AoS page is
 * preserved on the first fastscan page.
 *
 * Returns the head block of the new fastscan chain.
 * Returns InvalidBlockNumber if the AoS chain is empty.
 *
 * The original AoS pages are NOT freed — the caller is
 * responsible for any cleanup.
 */
BlockNumber prism_posting_convert_to_fastscan(
		VsStorage *storage, BlockNumber aos_head, Dimension dim);

#endif /* PRISM_POSTING_CONVERT_H */
