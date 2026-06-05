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

/* ----------------------------------------------------------------
 * Per-worker posting build
 * ---------------------------------------------------------------- */

typedef struct MktPostingWorkerState
{
	uint32_t  worker_id; /* 0 = leader; threads (standalone) or procs (PG) */
	uint32_t  nlist;
	Dimension dim;
	bool	  fastscan;

	MktStorage		   *storage;
	const RaBitQParams *params;
	const float		   *leaf_centroids; /* [nlist * dim] */
	const float		   *pt_centroids;	/* [nlist * dim] */
	MktPostingReserve  *reserve;

	MktPostingBuilder *builders; /* [nlist] lazily initialized */
	bool			  *active;	 /* [nlist] */

	/* Shared partial page buffer: after finish, each worker's
	 * partial page for cluster c is at partials[worker_id * nlist + c].
	 * The buffer is provided by the caller (thread-local memory in
	 * standalone, DSM in PG). NULL entries = no partial for that
	 * cluster. The page data is BLCKSZ bytes per slot. */
	char *partials; /* [nlist * BLCKSZ], caller-owned */

	/* Full-page sink for deferred mode (storage == NULL): each builder
	 * streams its completed pages to this callback, which writes/forwards
	 * them — the parallel build uses it to stream pages to the leader over
	 * shm_mq. Applied to every builder in ensure_builder. */
	void (*page_sink)(void *ctx, uint32_t cluster_id, const char *page);
	void *sink_ctx;
} MktPostingWorkerState;

void mkt_posting_worker_init(
		MktPostingWorkerState *ws,
		uint32_t			   worker_id,
		uint32_t			   nlist,
		Dimension			   dim,
		bool				   fastscan,
		MktStorage			  *storage,
		const RaBitQParams	  *params,
		const float			  *leaf_centroids,
		const float			  *pt_centroids,
		MktPostingReserve	  *reserve,
		char				  *partials);

/*
 * Set a full-page sink applied to every builder this worker creates
 * (deferred mode). Call after worker_init, before adding vectors.
 */
void mkt_posting_worker_set_page_sink(
		MktPostingWorkerState *ws,
		void (*sink)(void *ctx, uint32_t cluster_id, const char *page),
		void *sink_ctx);

/*
 * Add a heap tuple to its cluster's posting list builder.
 * PG path: TID comes directly from the heap scan.
 */
void mkt_posting_worker_add_heap(
		MktPostingWorkerState *ws,
		ItemPointerData		   tid,
		const float			  *vec,
		uint32_t			   primary,
		uint32_t			   secondary);

/*
 * Finalize all active builders. Full pages have already been streamed
 * to the leader; the trailing partial page per cluster is copied to the
 * partials buffer for the leader to fold into the list head.
 */
void mkt_posting_worker_finish(MktPostingWorkerState *ws);

/*
 * Free internal arrays (builders, active). Call after the leader has
 * consumed this worker's partials.
 */
void mkt_posting_worker_cleanup(MktPostingWorkerState *ws);

#endif /* MKT_POSTING_BUILD_PARALLEL_H */
