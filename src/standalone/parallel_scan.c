/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * parallel_scan_standalone.c - Work-stealing vector scan over an array
 *
 * Standalone implementation of the parallel build scan (see
 * prism_build_scan.h). PG builds scan the heap via table_index_build_scan
 * instead, so this file is compiled only for standalone.
 */

#ifdef VS_STANDALONE

#include "standalone/parallel_scan.h"

void
prism_parallel_scan_init(
		PrismParallelScan *ps,
		const float		  *vectors,
		uint32_t		   nvecs,
		Dimension		   dim)
{
	ps->vectors = vectors;
	ps->nvecs	= nvecs;
	ps->dim		= dim;
	atomic_store(&ps->cursor, 0);
}

double
prism_parallel_scan_run(
		PrismParallelScan *ps, PrismBuildScanCb cb, void *state)
{
	Dimension dim	  = ps->dim;
	double	  scanned = 0;

	for (;;)
	{
		uint32_t start = atomic_fetch_add(&ps->cursor, PRISM_BUILD_SCAN_CHUNK);
		uint32_t end;

		if (start >= ps->nvecs)
			break;

		end = start + PRISM_BUILD_SCAN_CHUNK;
		if (end > ps->nvecs)
			end = ps->nvecs;

		for (uint32_t i = start; i < end; i++)
		{
			ItemPointerData tid;

			/* Reversible: ItemPointerGetBlockNumber(tid) == i. */
			ItemPointerSet(&tid, i, 1);
			cb(state, tid, ps->vectors + (size_t)i * dim);
		}
		scanned += end - start;
	}
	return scanned;
}

#endif /* VS_STANDALONE */
