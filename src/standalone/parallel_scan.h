/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * parallel_scan.h - Standalone work-stealing vector scan for the index build
 *
 * The standalone analog of PostgreSQL's table_index_build_scan: a shared
 * cursor over a contiguous in-memory vector array that every participating
 * thread claims chunks from until the input is exhausted, invoking the
 * back-end- neutral PrismBuildScanCb per vector. (In a PostgreSQL build the
 * scan is table_index_build_scan instead, so this header is standalone-only.)
 */

#ifndef VS_PARALLEL_SCAN_H
#define VS_PARALLEL_SCAN_H

#include <stdatomic.h>
#include <stdint.h>

#include "core/types.h"
#include "index/parallel_build.h" /* PrismBuildScanCb */

/* Vectors scanned per claimed chunk (balances work-stealing vs. contention).
 */
#define PRISM_BUILD_SCAN_CHUNK 512

/*
 * Shared scan cursor over a contiguous vector array. Initialized once; every
 * participating thread calls prism_parallel_scan_run on the same instance.
 */
typedef struct PrismParallelScan
{
	const float		 *vectors;
	uint32_t		  nvecs;
	Dimension		  dim;
	_Atomic(uint32_t) cursor;
} PrismParallelScan;

extern void prism_parallel_scan_init(
		PrismParallelScan *ps,
		const float		  *vectors,
		uint32_t		   nvecs,
		Dimension		   dim);

/*
 * Claim and process chunks until the input is exhausted, invoking cb for each
 * vector. Safe to call concurrently from any number of threads on the same
 * PrismParallelScan; together they cover every vector exactly once. Returns
 * the number of vectors this caller processed (the callers' sum is nvecs).
 */
extern double prism_parallel_scan_run(
		PrismParallelScan *ps, PrismBuildScanCb cb, void *state);

#endif /* VS_PARALLEL_SCAN_H */
