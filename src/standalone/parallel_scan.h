/*
 * parallel_scan.h - Standalone work-stealing vector scan for the index build
 *
 * The standalone analog of PostgreSQL's table_index_build_scan: a shared
 * cursor over a contiguous in-memory vector array that every participating
 * thread claims chunks from until the input is exhausted, invoking the
 * back-end- neutral MktBuildScanCb per vector. (In a PostgreSQL build the scan
 * is table_index_build_scan instead, so this header is standalone-only.)
 */

#ifndef MKT_PARALLEL_SCAN_H
#define MKT_PARALLEL_SCAN_H

#include <stdatomic.h>
#include <stdint.h>

#include "index/parallel_build.h" /* MktBuildScanCb */
#include "mkt_types.h"

/* Vectors scanned per claimed chunk (balances work-stealing vs. contention).
 */
#define MKT_BUILD_SCAN_CHUNK 512

/*
 * Shared scan cursor over a contiguous vector array. Initialized once; every
 * participating thread calls mkt_parallel_scan_run on the same instance.
 */
typedef struct MktParallelScan
{
	const float		 *vectors;
	uint32_t		  nvecs;
	Dimension		  dim;
	_Atomic(uint32_t) cursor;
} MktParallelScan;

extern void mkt_parallel_scan_init(
		MktParallelScan *ps,
		const float		*vectors,
		uint32_t		 nvecs,
		Dimension		 dim);

/*
 * Claim and process chunks until the input is exhausted, invoking cb for each
 * vector. Safe to call concurrently from any number of threads on the same
 * MktParallelScan; together they cover every vector exactly once. Returns the
 * number of vectors this caller processed (the callers' sum is nvecs).
 */
extern double
mkt_parallel_scan_run(MktParallelScan *ps, MktBuildScanCb cb, void *state);

#endif /* MKT_PARALLEL_SCAN_H */
