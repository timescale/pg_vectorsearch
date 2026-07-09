/*
 * build_posting.c - Parallel posting list build pipeline
 *
 * Shared between standalone and PostgreSQL builds. All functions
 * operate through the MktStorage abstraction — no direct dependency
 * on ArrayPageStorage or PG shared buffers.
 */

#include <stdlib.h>
#include <string.h>

#include "core/memory.h"
#include "index/posting_build_parallel.h"

/* ----------------------------------------------------------------
 * Page reservation
 *
 * Pre-allocate a contiguous block range per cluster so that
 * parallel workers can claim pages without locking. Each worker
 * atomically increments the cluster's next-slot counter to get
 * its page. Contiguous layout also gives sequential I/O during
 * query scans.
 *
 * ---------------------------------------------------------------- */

void
mkt_posting_reserve_init(
		MktPostingReserve *res,
		const uint32_t	  *cluster_counts,
		uint32_t		   nlist,
		uint32_t		   nworkers,
		Dimension		   dim,
		bool			   fastscan,
		bool			   replicate)
{
	(void)nworkers;

	res->starts = mkt_alloc(nlist * sizeof(BlockNumber));
	res->counts = mkt_alloc(nlist * sizeof(uint32_t));
	res->nexts	= mkt_alloc0(nlist * sizeof(mkt_atomic_uint32));
	res->nlist	= nlist;

	/* cluster_counts are raw primary-vector estimates; the shared estimator
	 * applies the format-specific capacity and replication headroom so every
	 * build path reserves identically. */
	BlockNumber total = 0;
	for (uint32_t c = 0; c < nlist; c++)
	{
		uint32_t npages = mkt_posting_estimate_pages(
				cluster_counts[c], dim, fastscan, replicate);
		res->starts[c] = total;
		res->counts[c] = npages;
		total += npages;
	}
	res->total = total;

	for (uint32_t c = 0; c < nlist; c++)
		mkt_atomic_init_u32(&res->nexts[c], 1);
}

void
mkt_posting_reserve_free(MktPostingReserve *res)
{
	mkt_free(res->starts);
	mkt_free(res->counts);
	mkt_free(res->nexts);
	res->starts = NULL;
	res->counts = NULL;
	res->nexts	= NULL;
}
