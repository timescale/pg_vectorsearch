/*
 * build_posting.c - Parallel posting list build pipeline
 *
 * Shared between standalone and PostgreSQL builds. All functions
 * operate through the MktStorage abstraction — no direct dependency
 * on ArrayPageStorage or PG shared buffers.
 */

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
		Dimension		   dim)
{
	(void)nworkers;

	uint32_t ent_first	  = mkt_posting_max_entries_first(dim);
	uint32_t ent_overflow = mkt_posting_max_entries(dim);

	res->starts = mkt_alloc(nlist * sizeof(BlockNumber));
	res->counts = mkt_alloc(nlist * sizeof(uint32_t));
	res->nexts	= mkt_alloc0(nlist * sizeof(mkt_atomic_uint32));
	res->nlist	= nlist;

	BlockNumber total = 0;
	for (uint32_t c = 0; c < nlist; c++)
	{
		uint32_t cnt	= cluster_counts[c];
		uint32_t npages = 1;
		if (cnt > ent_first)
			npages += (cnt - ent_first + ent_overflow - 1) / ent_overflow;
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
		char				  *partials,
		MktMemCtx			   batch_ctx)
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
	ws->batch_ctx	   = batch_ctx;

	ws->builders = mkt_alloc((size_t)nlist * sizeof(MktPostingBuilder));
	ws->active	 = mkt_alloc0(nlist * sizeof(bool));
	ws->heads	 = mkt_alloc(nlist * sizeof(BlockNumber));
	ws->tails	 = mkt_alloc(nlist * sizeof(BlockNumber));
	ws->batches	 = (storage == NULL)
						 ? mkt_alloc0(nlist * sizeof(MktPostingBatch))
						 : NULL;

	for (uint32_t c = 0; c < nlist; c++)
	{
		ws->heads[c] = InvalidBlockNumber;
		ws->tails[c] = InvalidBlockNumber;
	}
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
	 * (no head metadata); their chains are linked in after the leader's
	 * head during finalization (mkt_posting_materialize).
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

	/* Deferred mode: complete pages accumulate in this worker's batch
	 * context so they outlive the (transient) context the scan runs in. */
	if (ws->storage == NULL)
		mkt_posting_builder_set_batch_ctx(&ws->builders[c], ws->batch_ctx);

	ws->active[c] = true;
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
mkt_posting_worker_add(
		MktPostingWorkerState *ws,
		uint32_t			   vec_id,
		const float			  *vec,
		uint32_t			   primary,
		uint32_t			   secondary)
{
	ItemPointerData tid;
	mkt_posting_set_vector_id(&tid, vec_id);
	worker_add_to_clusters(ws, tid, vec, primary, secondary);
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
	bool deferred = (ws->storage == NULL);
	bool hold	  = !ws->fastscan && ws->partials != NULL;

	for (uint32_t c = 0; c < ws->nlist; c++)
	{
		if (!ws->active[c])
			continue;

		if (hold)
			mkt_posting_builder_finish_partial(&ws->builders[c]);
		else
			mkt_posting_builder_finish(&ws->builders[c]);

		if (deferred)
		{
			mkt_posting_builder_take_batch(&ws->builders[c], &ws->batches[c]);
		}
		else
		{
			ws->heads[c] = mkt_posting_builder_head(&ws->builders[c]);
			ws->tails[c] = mkt_posting_builder_tail(&ws->builders[c]);
		}

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
	if (ws->batches != NULL)
	{
		/* Batch pages live in ws->batch_ctx and are released when the
		 * caller deletes that context — only the array is freed here. */
		mkt_free(ws->batches);
		ws->batches = NULL;
	}
	mkt_free(ws->builders);
	mkt_free(ws->active);
	mkt_free(ws->heads);
	mkt_free(ws->tails);
	ws->builders = NULL;
	ws->active	 = NULL;
	ws->heads	 = NULL;
	ws->tails	 = NULL;
}

/* ----------------------------------------------------------------
 * Materialize deferred batch output into storage
 *
 * All workers ran in deferred mode (storage == NULL), so their
 * output is complete BLCKSZ page buffers without block numbers
 * or chain links. This function:
 *
 *   1. Merges partial pages (AoS) into additional batch pages
 *   2. Extends storage for the exact page count needed
 *   3. Writes all pages sequentially, assigning contiguous
 *      block numbers and chain links as they go
 *
 * No sorting or relinking needed — pages are written in order.
 * ---------------------------------------------------------------- */

/*
 * Write a page from a batch to storage at the given block,
 * setting the chain link to next_blkno.
 */
static void
write_batch_page(
		MktStorage *storage,
		BlockNumber blkno,
		const char *src_page,
		BlockNumber next_blkno)
{
	Page spage = mkt_storage_write_page(storage, blkno);
	memcpy(spage, src_page, BLCKSZ);
	mkt_posting_opaque(spage)->next_blkno = next_blkno;
	mkt_storage_commit_page(storage, blkno);
}

void
mkt_posting_materialize(
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
		MktPostingBuildResult *result)
{
	/* --- Phase 1: Merge partial pages into batch pages --- */
	MktPostingBatch *merged = mkt_alloc0(nlist * sizeof(MktPostingBatch));

	/* Merged pages are transient: written to storage in phase 3 and then
	 * discarded. Accumulate them in a scratch context that is freed in one
	 * shot at the end, rather than freeing each cluster's pages by hand. */
	MktMemCtx merge_ctx = mkt_memctx_create(NULL, "posting_merge");

	result->merge_input	 = 0;
	result->merge_output = 0;

	if (!fastscan && partials != NULL)
	{
		for (uint32_t c = 0; c < nlist; c++)
		{
			uint32_t npartials = 0;
			for (uint32_t w = 0; w < nworkers; w++)
			{
				if (!worker_active[w][c])
					continue;
				Page pg = partials + ((size_t)w * nlist + c) * BLCKSZ;
				if (mkt_posting_page_count(pg) > 0)
					npartials++;
			}
			if (npartials == 0)
				continue;

			const float *cent	 = leaf_centroids + (size_t)c * dim;
			const float *pt_cent = pt_centroids + (size_t)c * dim;

			/* First-page format when worker 0 has no batch output */
			bool			  need_first = (worker_batches[0][c].count == 0);
			MktPostingBuilder mb;
			if (need_first)
				mkt_posting_builder_init(
						&mb, NULL, NULL, dim, c, cent, pt_cent);
			else
				mkt_posting_builder_init_continuation(
						&mb, NULL, NULL, dim, c, cent);
			mkt_posting_builder_set_batch_ctx(&mb, merge_ctx);

			for (uint32_t w = 0; w < nworkers; w++)
			{
				if (!worker_active[w][c])
					continue;
				Page pg = partials + ((size_t)w * nlist + c) * BLCKSZ;
				MktPostingPageOpaque *op = mkt_posting_opaque(pg);
				if (op->entry_count == 0)
					continue;

				bool  is_fp = (op->flags & MKT_POSTING_PAGE_FIRST) != 0;
				char *ct	= is_fp ? mkt_posting_content_first(pg, dim)
									: mkt_posting_content(pg);

				for (uint32_t e = 0; e < op->entry_count; e++)
				{
					MktPostingEntryHeader *hdr =
							mkt_posting_entry_at(ct, e, dim);
					mkt_posting_builder_add_encoded(
							&mb,
							hdr->meta.tid,
							hdr->f_add,
							hdr->f_rescale,
							hdr->f_error,
							hdr->bits);
				}
			}

			mkt_posting_builder_finish(&mb);
			mkt_posting_builder_take_batch(&mb, &merged[c]);
			mkt_posting_builder_cleanup(&mb);

			result->merge_input += npartials;
			result->merge_output += merged[c].count;
		}
	}

	/* --- Phase 2: Compute exact page counts --- */
	uint32_t *cluster_pages = mkt_alloc(nlist * sizeof(uint32_t));
	/* Whether merged pages supply the first page for each cluster */
	bool	*merged_first = mkt_alloc0(nlist * sizeof(bool));
	uint32_t total_pages  = 0;

	for (uint32_t c = 0; c < nlist; c++)
	{
		uint32_t count = 0;
		for (uint32_t w = 0; w < nworkers; w++)
		{
			if (worker_active[w][c])
				count += worker_batches[w][c].count;
		}

		bool w0_has_pages = (worker_batches[0][c].count > 0);
		merged_first[c]	  = !w0_has_pages && merged[c].count > 0;

		count += merged[c].count;
		if (count == 0)
			count = 1; /* empty first page */
		cluster_pages[c] = count;
		total_pages += count;
	}

	/* --- Phase 3: Extend storage --- */
	BlockNumber base_block = mkt_storage_extend(storage, total_pages);

	/* --- Phase 4: Write pages --- */
	result->heads		= mkt_alloc(nlist * sizeof(BlockNumber));
	result->total_pages = total_pages;

	BlockNumber cur_block = base_block;

	for (uint32_t c = 0; c < nlist; c++)
	{
		uint32_t	ct		= cluster_pages[c];
		BlockNumber cl_base = cur_block;
		uint32_t	written = 0;

		bool w0_has		   = (worker_batches[0][c].count > 0);
		bool has_any_batch = w0_has || merged[c].count > 0;
		for (uint32_t w = 1; !has_any_batch && w < nworkers; w++)
		{
			if (worker_active[w][c] && worker_batches[w][c].count > 0)
				has_any_batch = true;
		}

		if (!has_any_batch)
		{
			/* Empty cluster: create first page with pt_centroid */
			BlockNumber blkno = cur_block++;
			Page		page  = mkt_storage_write_page(storage, blkno);
			uint16_t	flags = MKT_POSTING_PAGE_FIRST;
			if (fastscan)
				flags |= MKT_POSTING_PAGE_FASTSCAN;
			mkt_posting_page_init(page, c, dim, flags);
			memcpy(mkt_posting_pt_centroid_mut(page),
				   pt_centroids + (size_t)c * dim,
				   dim * sizeof(float));
			mkt_posting_opaque(page)->next_blkno = InvalidBlockNumber;
			mkt_storage_commit_page(storage, blkno);
			result->heads[c] = blkno;
			continue;
		}

		/*
		 * Write order:
		 *   If worker 0 has batch pages: w0 pages, w1..wN, merged
		 *   If merged supplies first page: merged, w0..wN
		 *   Otherwise: w1..wN (rare: w0 has no output, no partials)
		 */
		if (merged_first[c])
		{
			/* Merged pages first (first one is first-page format) */
			for (uint32_t p = 0; p < merged[c].count; p++)
			{
				BlockNumber next = (written + 1 < ct) ? cur_block + 1
													  : InvalidBlockNumber;
				write_batch_page(
						storage,
						cur_block,
						merged[c].pages + (size_t)p * BLCKSZ,
						next);
				cur_block++;
				written++;
			}
		}

		/* Worker batch pages */
		for (uint32_t w = 0; w < nworkers; w++)
		{
			if (!worker_active[w][c])
				continue;
			MktPostingBatch *batch = &worker_batches[w][c];
			for (uint32_t p = 0; p < batch->count; p++)
			{
				BlockNumber next = (written + 1 < ct) ? cur_block + 1
													  : InvalidBlockNumber;
				write_batch_page(
						storage,
						cur_block,
						batch->pages + (size_t)p * BLCKSZ,
						next);
				cur_block++;
				written++;
			}
		}

		/* Merged pages last (when not first) */
		if (!merged_first[c])
		{
			for (uint32_t p = 0; p < merged[c].count; p++)
			{
				BlockNumber next = (written + 1 < ct) ? cur_block + 1
													  : InvalidBlockNumber;
				write_batch_page(
						storage,
						cur_block,
						merged[c].pages + (size_t)p * BLCKSZ,
						next);
				cur_block++;
				written++;
			}
		}

		result->heads[c] = cl_base;
	}

	/* --- Cleanup --- */
	mkt_memctx_delete(merge_ctx); /* frees all merged[c].pages at once */
	mkt_free(merged);
	mkt_free(cluster_pages);
	mkt_free(merged_first);
}
