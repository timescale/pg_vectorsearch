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
 * nworkers: headroom for partial pages (adds nworkers-1 per cluster)
 */
void mkt_posting_reserve_init(
		MktPostingReserve *res,
		const uint32_t	  *cluster_counts,
		uint32_t		   nlist,
		uint32_t		   nworkers,
		Dimension		   dim);

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
	BlockNumber		  *heads;	 /* [nlist] flushed chain heads */
	BlockNumber		  *tails;	 /* [nlist] flushed chain tails */

	/* Deferred batch output: per-cluster batch of complete pages.
	 * Populated by worker_finish when storage == NULL. */
	MktPostingBatch *batches; /* [nlist] */

	/* Shared partial page buffer: after finish, each worker's
	 * partial page for cluster c is at partials[worker_id * nlist + c].
	 * The buffer is provided by the caller (thread-local memory in
	 * standalone, DSM in PG). NULL entries = no partial for that
	 * cluster. The page data is BLCKSZ bytes per slot. */
	char *partials; /* [nlist * BLCKSZ], caller-owned */
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
 * Add a vector to its cluster's posting list builder.
 * Standalone path: vec_id is converted to an ItemPointerData.
 */
void mkt_posting_worker_add(
		MktPostingWorkerState *ws,
		uint32_t			   vec_id,
		const float			  *vec,
		uint32_t			   primary,
		uint32_t			   secondary);

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
 * Finalize all active builders. Full pages have already been flushed
 * to storage. Partial pages are copied to the partials buffer.
 * heads[] and tails[] contain the flushed chain endpoints.
 */
void mkt_posting_worker_finish(MktPostingWorkerState *ws);

/*
 * Free internal arrays (builders, active). Must be called after
 * mkt_posting_finalize() since the merge reads from partials.
 */
void mkt_posting_worker_cleanup(MktPostingWorkerState *ws);

/* ----------------------------------------------------------------
 * Post-build: merge partial pages + chain linking
 * ---------------------------------------------------------------- */

typedef struct MktPostingBuildResult
{
	BlockNumber *heads;		   /* [nlist] posting list head blocks */
	uint32_t	 total_pages;  /* final page count */
	uint32_t	 merge_input;  /* partial pages that went into merge */
	uint32_t	 merge_output; /* pages produced by merge */
} MktPostingBuildResult;

/*
 * Merge partial pages and link all page chains.
 *
 * partials: [nworkers * nlist * BLCKSZ] shared buffer with
 *           partial page data from all workers
 * worker_heads/tails: [nworkers][nlist] flushed chain endpoints
 * worker_active: [nworkers][nlist] which clusters each worker touched
 *
 * For each cluster:
 *   1. Merge partial pages into optimally packed pages
 *   2. Link all flushed chains + merge output
 *   3. Sort chain by block number for sequential I/O
 *
 * result: output — caller must free result->heads
 */
void mkt_posting_finalize(
		char				  *partials,
		BlockNumber			 **worker_heads,
		BlockNumber			 **worker_tails,
		bool				 **worker_active,
		uint32_t			   nworkers,
		MktStorage			  *storage,
		MktPostingReserve	  *reserve,
		const float			  *leaf_centroids,
		const float			  *pt_centroids,
		Dimension			   dim,
		bool				   fastscan,
		MktPostingBuildResult *result);

/*
 * Materialize deferred batch output into storage.
 *
 * Workers produce pages in deferred mode (storage == NULL),
 * accumulating complete pages in MktPostingBatch arrays. This
 * function takes all worker output, reserves contiguous blocks,
 * writes pages to storage, and links chains.
 *
 * For AoS: partial pages (in the partials buffer) are merged
 * across workers into optimally packed pages before writing.
 *
 * worker_batches: [nworkers] arrays of MktPostingBatch[nlist]
 * partials: [nworkers * nlist * BLCKSZ] or NULL (fastscan)
 * worker_active: [nworkers][nlist]
 *
 * result: output — caller must free result->heads
 */
void mkt_posting_materialize(
		MktPostingBatch		 **worker_batches,
		char				  *partials,
		bool				 **worker_active,
		uint32_t			   nworkers,
		uint32_t			   nlist,
		MktStorage			  *storage,
		const float			  *leaf_centroids,
		const float			  *pt_centroids,
		Dimension			   dim,
		bool				   fastscan,
		MktPostingBuildResult *result);

#endif /* MKT_POSTING_BUILD_PARALLEL_H */
