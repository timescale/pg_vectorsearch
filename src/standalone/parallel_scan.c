/*
 * parallel_scan_standalone.c - Work-stealing vector scan over an array
 *
 * Standalone implementation of the parallel build scan (see mkt_build_scan.h).
 * PG builds scan the heap via table_index_build_scan instead, so this file is
 * compiled only for standalone.
 */

#ifdef MKT_STANDALONE

#include "core/build_scan.h"

void
mkt_parallel_scan_init(
		MktParallelScan *ps,
		const float		*vectors,
		uint32_t		 nvecs,
		Dimension		 dim)
{
	ps->vectors = vectors;
	ps->nvecs	= nvecs;
	ps->dim		= dim;
	atomic_store(&ps->cursor, 0);
}

void
mkt_parallel_scan_run(MktParallelScan *ps, MktBuildScanCb cb, void *state)
{
	Dimension dim = ps->dim;

	for (;;)
	{
		uint32_t start = atomic_fetch_add(&ps->cursor, MKT_BUILD_SCAN_CHUNK);
		uint32_t end;

		if (start >= ps->nvecs)
			break;

		end = start + MKT_BUILD_SCAN_CHUNK;
		if (end > ps->nvecs)
			end = ps->nvecs;

		for (uint32_t i = start; i < end; i++)
		{
			ItemPointerData tid;

			/* Reversible: ItemPointerGetBlockNumber(tid) == i. */
			ItemPointerSet(&tid, i, 1);
			cb(state, tid, ps->vectors + (size_t)i * dim);
		}
	}
}

#endif /* MKT_STANDALONE */
