/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
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
 * new fastscan pages. The chain's encode reference is carried over
 * to the new chain.
 *
 * Returns the head block of the new fastscan chain, or
 * InvalidBlockNumber when aos_head is invalid or the list already
 * starts with fastscan pages. In that case *has_fastscan_base is set
 * and nothing is written: the AoS pages after a fastscan base hold
 * vectors inserted since, which a split repacks.
 *
 * The original AoS pages are NOT freed — the caller is
 * responsible for any cleanup.
 */
BlockNumber prism_posting_convert_to_fastscan(
		VsStorage  *storage,
		BlockNumber aos_head,
		Dimension	dim,
		bool	   *has_fastscan_base);

#endif /* PRISM_POSTING_CONVERT_H */
