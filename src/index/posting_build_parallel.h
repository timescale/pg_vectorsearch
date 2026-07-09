/*
 * posting_build_parallel.h - Multi-worker posting list build
 *
 * Coordinates multiple workers building posting lists in parallel.
 * Handles page reservation (atomic block claiming), per-worker
 * builder lifecycle, partial page merging, and chain linking.
 *
 * Used by both standalone (pthreads) and PostgreSQL (parallel
 * workers). No PostgreSQL dependencies.
 */

#ifndef MKT_POSTING_BUILD_PARALLEL_H
#define MKT_POSTING_BUILD_PARALLEL_H

#include <stdbool.h>
#include <stdint.h>

#include "core/atomics.h"
#include "index/posting_build.h"
#include "index/posting_page.h"
#include "quant/rabitq.h"

/* ----------------------------------------------------------------
 * Page reservation — contiguous block ranges per cluster
 * ---------------------------------------------------------------- */

typedef struct MktPostingReserve
{
	BlockNumber		  *starts; /* [nlist] start block per cluster */
	uint32_t		  *counts; /* [nlist] reserved pages per cluster */
	mkt_atomic_uint32 *nexts;  /* [nlist] next slot (atomic) */
	uint32_t		   nlist;
	BlockNumber		   total; /* total reserved blocks */
} MktPostingReserve;

/*
 * Compute page reservations from cluster sizes.
 *
 * cluster_counts: [nlist] vectors per cluster
 * nworkers: currently unused — the deferred-batch path reserves exact
 *           block counts at materialize, so no per-worker partial-page
 *           headroom is added here.
 */
void mkt_posting_reserve_init(
		MktPostingReserve *res,
		const uint32_t	  *cluster_counts,
		uint32_t		   nlist,
		uint32_t		   nworkers,
		Dimension		   dim,
		bool			   fastscan,
		bool			   replicate);

void mkt_posting_reserve_free(MktPostingReserve *res);

#endif /* MKT_POSTING_BUILD_PARALLEL_H */
