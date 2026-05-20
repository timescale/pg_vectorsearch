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
 * Extra headroom (nworkers - 1 pages per cluster) accounts for
 * each worker potentially holding one partial page at the end.
 * ---------------------------------------------------------------- */

void
mkt_posting_reserve_init(
		MktPostingReserve *res,
		const uint32_t	  *cluster_counts,
		uint32_t		   nlist,
		uint32_t		   nworkers,
		Dimension		   dim)
{
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
		npages += nworkers - 1;
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
		uint32_t			   thread_id,
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
	ws->thread_id	   = thread_id;
	ws->nlist		   = nlist;
	ws->dim			   = dim;
	ws->fastscan	   = fastscan;
	ws->storage		   = storage;
	ws->params		   = params;
	ws->leaf_centroids = leaf_centroids;
	ws->pt_centroids   = pt_centroids;
	ws->reserve		   = reserve;
	ws->partials	   = partials;

	ws->builders = mkt_alloc((size_t)nlist * sizeof(MktPostingBuilder));
	ws->active	 = mkt_alloc0(nlist * sizeof(bool));
	ws->heads	 = mkt_alloc(nlist * sizeof(BlockNumber));
	ws->tails	 = mkt_alloc(nlist * sizeof(BlockNumber));

	for (uint32_t c = 0; c < nlist; c++)
	{
		ws->heads[c] = InvalidBlockNumber;
		ws->tails[c] = InvalidBlockNumber;
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
	uint32_t dim		 = ws->dim;
	uint32_t clusters[2] = {primary, secondary};
	uint32_t nclusters	 = (secondary != MKT_INVALID_CLUSTER) ? 2 : 1;

	for (uint32_t ci = 0; ci < nclusters; ci++)
	{
		uint32_t c = clusters[ci];
		if (!ws->active[c])
		{
			const float *cent = ws->leaf_centroids + (size_t)c * dim;

			if (ws->thread_id == 0)
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
							&ws->builders[c],
							ws->storage,
							ws->params,
							dim,
							c,
							cent);
				else
					mkt_posting_builder_init_continuation(
							&ws->builders[c],
							ws->storage,
							ws->params,
							dim,
							c,
							cent);
			}

			if (ws->reserve != NULL)
			{
				mkt_posting_builder_set_shared_reserve(
						&ws->builders[c],
						ws->reserve->starts[c],
						ws->reserve->counts[c],
						&ws->reserve->nexts[c]);
				if (ws->thread_id == 0)
					mkt_posting_builder_set_first_blkno(
							&ws->builders[c], ws->reserve->starts[c]);
			}

			ws->active[c] = true;
		}

		ItemPointerData tid;
		mkt_posting_set_vector_id(&tid, vec_id);
		mkt_posting_builder_add(&ws->builders[c], tid, vec);
	}
}

void
mkt_posting_worker_finish(MktPostingWorkerState *ws)
{
	bool hold = !ws->fastscan && ws->partials != NULL;

	for (uint32_t c = 0; c < ws->nlist; c++)
	{
		if (!ws->active[c])
			continue;

		if (hold)
			mkt_posting_builder_finish_partial(&ws->builders[c]);
		else
			mkt_posting_builder_finish(&ws->builders[c]);

		ws->heads[c] = mkt_posting_builder_head(&ws->builders[c]);
		ws->tails[c] = mkt_posting_builder_tail(&ws->builders[c]);

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
	mkt_free(ws->heads);
	mkt_free(ws->tails);
	ws->builders = NULL;
	ws->active	 = NULL;
	ws->heads	 = NULL;
	ws->tails	 = NULL;
}

/* ----------------------------------------------------------------
 * Post-build finalization
 *
 * After all workers finish, each cluster may have:
 *   - Flushed page chains from multiple workers (disjoint block
 *     ranges, each linked internally)
 *   - One partial (non-full) page per worker, held in the shared
 *     partials buffer
 *
 * This function, for each cluster:
 *   1. Merges partial pages from all workers into optimally packed
 *      full pages via a continuation builder, eliminating per-worker
 *      waste (e.g., 8 workers × 1 half-full page → 4 full pages)
 *   2. Collects all flushed page block numbers (from worker chains
 *      and the merge output) into a single array
 *   3. Sorts by block number for sequential disk I/O during scans
 *   4. Relinks the next_blkno chain in sorted order
 *
 * The result is one contiguous, sorted page chain per cluster with
 * no wasted partial pages.
 * ---------------------------------------------------------------- */

static int
blkno_cmp(const void *a, const void *b)
{
	BlockNumber ba = *(const BlockNumber *)a;
	BlockNumber bb = *(const BlockNumber *)b;
	return (ba > bb) - (ba < bb);
}

void
mkt_posting_finalize(
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
		MktPostingBuildResult *result)
{
	uint32_t nlist = reserve->nlist;

	result->heads		 = mkt_alloc(nlist * sizeof(BlockNumber));
	result->total_pages	 = 0;
	result->merge_input	 = 0;
	result->merge_output = 0;

	uint32_t max_chain = 0;
	for (uint32_t c = 0; c < nlist; c++)
		if (reserve->counts[c] > max_chain)
			max_chain = reserve->counts[c];
	max_chain += nworkers + 64;
	BlockNumber *chain = mkt_alloc(max_chain * sizeof(BlockNumber));

	for (uint32_t c = 0; c < nlist; c++)
	{
		/* --- Merge partial pages (AoS only) --- */
		BlockNumber merge_head = InvalidBlockNumber;
		BlockNumber merge_tail = InvalidBlockNumber;

		if (!fastscan && partials != NULL)
		{
			uint32_t npartials = 0;
			for (uint32_t t = 0; t < nworkers; t++)
			{
				if (!worker_active[t][c])
					continue;
				Page pg = partials + ((size_t)t * nlist + c) * BLCKSZ;
				if (mkt_posting_page_count(pg) > 0)
					npartials++;
			}

			if (npartials > 0)
			{
				const float *cent = leaf_centroids + (size_t)c * dim;

				MktPostingBuilder mb;
				mkt_posting_builder_init_continuation(
						&mb, storage, NULL, dim, c, cent);
				mkt_posting_builder_set_shared_reserve(
						&mb,
						reserve->starts[c],
						reserve->counts[c],
						&reserve->nexts[c]);

				for (uint32_t t = 0; t < nworkers; t++)
				{
					if (!worker_active[t][c])
						continue;
					Page pg = partials + ((size_t)t * nlist + c) * BLCKSZ;
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
				merge_head = mkt_posting_builder_head(&mb);
				merge_tail = mkt_posting_builder_tail(&mb);
				mkt_posting_builder_cleanup(&mb);

				uint32_t	merge_pages = 0;
				BlockNumber blk			= merge_head;
				while (blk != InvalidBlockNumber)
				{
					merge_pages++;
					Page mpg = mkt_storage_read_page(storage, blk);
					blk		 = mkt_posting_opaque(mpg)->next_blkno;
					mkt_storage_release_page(storage, blk);
				}
				result->merge_input += npartials;
				result->merge_output += merge_pages;

				/* Link merge output after last flushed tail */
				BlockNumber prev_tail = InvalidBlockNumber;
				for (uint32_t t = nworkers; t > 0; t--)
				{
					if (worker_tails[t - 1][c] != InvalidBlockNumber)
					{
						prev_tail = worker_tails[t - 1][c];
						break;
					}
				}

				if (prev_tail != InvalidBlockNumber)
				{
					Page prev = mkt_storage_write_page(storage, prev_tail);
					mkt_posting_opaque(prev)->next_blkno = merge_head;
					mkt_storage_commit_page(storage, prev_tail);
				}
				else if (worker_heads[0][c] == InvalidBlockNumber)
				{
					/* No flushed pages at all — create synthetic
					 * first page at slot 0 */
					BlockNumber blkno = reserve->starts[c];
					Page		page  = mkt_storage_write_page(storage, blkno);
					uint16_t	flags = MKT_POSTING_PAGE_FIRST;
					if (fastscan)
						flags |= MKT_POSTING_PAGE_FASTSCAN;
					mkt_posting_page_init(page, c, dim, flags);
					memcpy(mkt_posting_pt_centroid_mut(page),
						   pt_centroids + (size_t)c * dim,
						   dim * sizeof(float));
					mkt_posting_opaque(page)->next_blkno = merge_head;
					mkt_storage_commit_page(storage, blkno);
					worker_heads[0][c] = blkno;
				}
				else
				{
					Page fp = mkt_storage_write_page(
							storage, worker_heads[0][c]);
					mkt_posting_opaque(fp)->next_blkno = merge_head;
					mkt_storage_commit_page(storage, worker_heads[0][c]);
				}

				(void)merge_tail;
			}
		}

		/* --- Ensure a first page exists --- */
		if (worker_heads[0][c] == InvalidBlockNumber)
		{
			BlockNumber blkno = reserve->starts[c];
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
			worker_heads[0][c] = blkno;
		}

		/* --- Collect all pages into chain array --- */
		uint32_t nchain = 0;
		for (uint32_t t = 0; t < nworkers; t++)
		{
			BlockNumber blk = worker_heads[t][c];
			while (blk != InvalidBlockNumber)
			{
				chain[nchain++] = blk;
				Page pg			= mkt_storage_read_page(storage, blk);
				blk				= mkt_posting_opaque(pg)->next_blkno;
				mkt_storage_release_page(storage, blk);
			}
		}

		/* --- Sort by block number --- */
		if (nchain > 1)
			qsort(chain, nchain, sizeof(BlockNumber), blkno_cmp);

		/* --- Relink in sorted order --- */
		for (uint32_t i = 0; i + 1 < nchain; i++)
		{
			Page pg = mkt_storage_write_page(storage, chain[i]);
			mkt_posting_opaque(pg)->next_blkno = chain[i + 1];
			mkt_storage_commit_page(storage, chain[i]);
		}
		if (nchain > 0)
		{
			Page last = mkt_storage_write_page(storage, chain[nchain - 1]);
			mkt_posting_opaque(last)->next_blkno = InvalidBlockNumber;
			mkt_storage_commit_page(storage, chain[nchain - 1]);
		}

		result->heads[c] = nchain > 0 ? chain[0] : InvalidBlockNumber;
		result->total_pages += nchain;
	}

	mkt_free(chain);
}
