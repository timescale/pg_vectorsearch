/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * scan_bound.h - row target for a prism scan
 *
 * A scan asks, at rescan, how many rows the query above it will pull, so
 * its top-k is sized for the query instead of a fixed default. Registers
 * an ExecutorRun_hook that only records which query is running; the
 * resolution itself happens inside the scan (see scan_bound.c).
 */

#ifndef SCAN_BOUND_H
#define SCAN_BOUND_H

#include <access/genam.h>

/* Install the ExecutorRun hook. Called once, from _PG_init. */
void prism_scan_bound_init(void);

/*
 * Rows this scan should size its top-k for, already seeded for any filter
 * the executor applies above it. 0 when the scan does not run under a
 * usable LIMIT, in which case the caller keeps its default sizing.
 */
uint32_t prism_scan_bound(IndexScanDesc scan);

/*
 * Inflate a row target for a filter applied above the scan, from the
 * estimated surviving fraction. Shared by the executor's sizing and the
 * cost model so both price the same pool.
 */
uint32_t prism_scan_inflate_for_filter(uint32_t k, double selectivity);

#endif /* SCAN_BOUND_H */
