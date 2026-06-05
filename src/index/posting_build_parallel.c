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

/* ----------------------------------------------------------------
 * Per-worker state
 * ---------------------------------------------------------------- */

void
mkt_posting_worker_init(
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
		char				  *partials)
{
	ws->worker_id	   = worker_id;
	ws->nlist		   = nlist;
	ws->dim			   = dim;
	ws->fastscan	   = fastscan;
	ws->storage		   = storage;
	ws->params		   = params;
	ws->leaf_centroids = leaf_centroids;
	ws->pt_centroids   = pt_centroids;
	ws->reserve		   = reserve;
	ws->partials	   = partials;
	ws->page_sink	   = NULL;
	ws->sink_ctx	   = NULL;

	ws->builders = mkt_alloc((size_t)nlist * sizeof(MktPostingBuilder));
	ws->active	 = mkt_alloc0(nlist * sizeof(bool));
}

static void
ensure_builder(MktPostingWorkerState *ws, uint32_t c)
{
	if (ws->active[c])
		return;

	uint32_t	 dim  = ws->dim;
	const float *cent = ws->leaf_centroids + (size_t)c * dim;

	/*
	 * Worker 0 (the leader) owns the HEAD page of each cluster's posting
	 * chain: only the head carries the cluster's centroid metadata (and
	 * the rotated pt_centroid used for reranking), so exactly one worker
	 * must produce it. Every other worker produces CONTINUATION pages
	 * (no head metadata); their pages are linked in after the leader's
	 * head when the leader finalizes each list.
	 */
	if (ws->worker_id == 0)
	{
		const float *pt_cent = ws->pt_centroids + (size_t)c * dim;
		if (ws->fastscan)
			mkt_posting_builder_init_fastscan(
					&ws->builders[c],
					ws->storage,
					ws->params,
					dim,
					c,
					cent,
					pt_cent);
		else
			mkt_posting_builder_init(
					&ws->builders[c],
					ws->storage,
					ws->params,
					dim,
					c,
					cent,
					pt_cent);
	}
	else
	{
		if (ws->fastscan)
			mkt_posting_builder_init_continuation_fastscan(
					&ws->builders[c], ws->storage, ws->params, dim, c, cent);
		else
			mkt_posting_builder_init_continuation(
					&ws->builders[c], ws->storage, ws->params, dim, c, cent);
	}

	if (ws->reserve != NULL)
	{
		mkt_posting_builder_set_shared_reserve(
				&ws->builders[c],
				ws->reserve->starts[c],
				ws->reserve->counts[c],
				&ws->reserve->nexts[c]);
		/* Only the leader's head page anchors the chain at the cluster's
		 * reserved start block; continuation pages claim later blocks
		 * from the shared reservation. */
		if (ws->worker_id == 0)
			mkt_posting_builder_set_first_blkno(
					&ws->builders[c], ws->reserve->starts[c]);
	}

	/* Deferred mode: stream completed pages to the sink (the leader writes
	 * them) for bounded memory. */
	if (ws->storage == NULL)
		mkt_posting_builder_set_page_sink(
				&ws->builders[c], ws->page_sink, ws->sink_ctx);

	ws->active[c] = true;
}

void
mkt_posting_worker_set_page_sink(
		MktPostingWorkerState *ws,
		void (*sink)(void *ctx, uint32_t cluster_id, const char *page),
		void *sink_ctx)
{
	ws->page_sink = sink;
	ws->sink_ctx  = sink_ctx;
}

static void
worker_add_to_clusters(
		MktPostingWorkerState *ws,
		ItemPointerData		   tid,
		const float			  *vec,
		uint32_t			   primary,
		uint32_t			   secondary)
{
	uint32_t clusters[2] = {primary, secondary};
	uint32_t nclusters	 = (secondary != MKT_INVALID_CLUSTER) ? 2 : 1;

	for (uint32_t ci = 0; ci < nclusters; ci++)
	{
		uint32_t c = clusters[ci];
		ensure_builder(ws, c);
		mkt_posting_builder_add(&ws->builders[c], tid, vec);
	}
}

void
mkt_posting_worker_add_heap(
		MktPostingWorkerState *ws,
		ItemPointerData		   tid,
		const float			  *vec,
		uint32_t			   primary,
		uint32_t			   secondary)
{
	worker_add_to_clusters(ws, tid, vec, primary, secondary);
}

void
mkt_posting_worker_finish(MktPostingWorkerState *ws)
{
	/* Hold the trailing partial page (both formats) when a partials buffer is
	 * provided so the driver can fold it into the head, packing it optimally
	 * instead of leaving an under-full page per worker. */
	bool hold = ws->partials != NULL;

	for (uint32_t c = 0; c < ws->nlist; c++)
	{
		if (!ws->active[c])
			continue;

		if (hold)
			mkt_posting_builder_finish_partial(&ws->builders[c]);
		else
			mkt_posting_builder_finish(&ws->builders[c]);

		if (hold)
			memcpy(ws->partials + (size_t)c * BLCKSZ,
				   ws->builders[c].mem_page,
				   BLCKSZ);

		mkt_posting_builder_cleanup(&ws->builders[c]);
	}
}

void
mkt_posting_worker_cleanup(MktPostingWorkerState *ws)
{
	mkt_free(ws->builders);
	mkt_free(ws->active);
	ws->builders = NULL;
	ws->active	 = NULL;
}
