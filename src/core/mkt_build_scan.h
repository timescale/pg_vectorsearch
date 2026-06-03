/*
 * mkt_build_scan.h - Parallel vector scan for the index build
 *
 * Both build phases (sampling and posting) scan every vector and hand it to a
 * callback. The scan is parallel and cooperative: each participant repeatedly
 * claims a chunk of the input until it is exhausted, so coverage is complete
 * regardless of how many participants actually run (matching PostgreSQL's
 * work-stealing table_index_build_scan).
 *
 * The callback receives a raw float pointer rather than a Datum-wrapped vector
 * so neither back-end copies on the hot path: a PG scan adapter unwraps the
 * Datum to the varlena's data pointer, and the standalone scan points straight
 * into its contiguous vector array.
 *
 * The tid identifies the vector. In PostgreSQL it is the real heap TID; in a
 * standalone build it is synthesized reversibly from the vector's index
 * (block = index, offset = 1), so the posting lists store a uniform TID in
 * both builds and the standalone query path recovers the index with
 * ItemPointerGetBlockNumber.
 */

#ifndef MKT_BUILD_SCAN_H
#define MKT_BUILD_SCAN_H

#ifdef MKT_STANDALONE
#include "core/pg_compat.h"
#else
#include <storage/itemptr.h>
#endif

#include "mkt_types.h"

/* Per-vector callback: tid identifies the vector, vec points at dim floats. */
typedef void (*MktBuildScanCb)(
		void *state, ItemPointerData tid, const float *vec);

#ifdef MKT_STANDALONE

#include <stdatomic.h>
#include <stdint.h>

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
 * MktParallelScan; together they cover every vector exactly once.
 */
extern void
mkt_parallel_scan_run(MktParallelScan *ps, MktBuildScanCb cb, void *state);

#endif /* MKT_STANDALONE */

#endif /* MKT_BUILD_SCAN_H */
