/*
 * index.c - Standalone in-memory index build
 *
 * Builds centroid tree and per-cluster posting lists from a flat
 * vector array. Used by bindings and CLI benchmark.
 *
 * When RaBitQ encoding is enabled, posting data is written to pages
 * via MktPostingBuilder. The full-precision vectors are stored in a
 * flat array for reranking.
 */

#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "algo/distance.h"
#include "algo/hkmeans.h"
#include "algo/kmeans.h"
#include "algo/topk.h"
#include "algo/vecops.h"
#include "core/memory.h"
#include "index/build_parallel.h"
#include "index/centroid_build.h"
#include "index/centroid_page.h"
#include "index/index_build.h"
#include "index/posting_build.h"
#include "index/posting_convert.h"
#include "index/posting_page.h"
#include "standalone/index.h"

static uint64_t
now_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* Arena-safe grow: alloc new, copy old, old freed on context delete */
static void *
arena_grow(void *old, size_t old_size, size_t new_size)
{
	void *new_buf = mkt_alloc(new_size);
	if (old != NULL && old_size > 0)
		memcpy(new_buf, old, old_size);
	return new_buf;
}

/* ----------------------------------------------------------------
 * ArrayPageStorage — page array backed by memory context
 * ---------------------------------------------------------------- */

static Page
aps_read_page(MktStorage *self, BlockNumber blkno)
{
	ArrayPageStorage *s = (ArrayPageStorage *)self;
	return s->pages + (size_t)blkno * BLCKSZ;
}

static void
aps_release_page(MktStorage *self, BlockNumber blkno)
{
	(void)self;
	(void)blkno;
}

static Page
aps_write_page(MktStorage *self, BlockNumber blkno)
{
	ArrayPageStorage *s = (ArrayPageStorage *)self;
	return s->pages + (size_t)blkno * BLCKSZ;
}

static Page
aps_new_page(MktStorage *self, BlockNumber *blkno_out)
{
	ArrayPageStorage *s = (ArrayPageStorage *)self;
	pthread_mutex_lock(&s->alloc_mutex);
	if (s->next_blkno >= s->page_cap)
	{
		uint32_t new_cap  = s->page_cap * 2;
		size_t	 new_size = (size_t)new_cap * BLCKSZ;
		char	*new_buf  = realloc(s->pages, new_size);
		memset(new_buf + (size_t)s->page_cap * BLCKSZ,
			   0,
			   ((size_t)new_cap - s->page_cap) * BLCKSZ);
		s->pages	= new_buf;
		s->page_cap = new_cap;
	}
	*blkno_out = s->next_blkno++;
	pthread_mutex_unlock(&s->alloc_mutex);
	return s->pages + (size_t)*blkno_out * BLCKSZ;
}

static void
aps_commit_page(MktStorage *self, BlockNumber blkno)
{
	(void)self;
	(void)blkno;
}

static uint32_t
aps_rerank(
		MktStorage		   *self,
		const float		   *query,
		Dimension			dim,
		const MktTopKEntry *candidates,
		uint32_t			count,
		uint32_t			keep,
		uint32_t		   *out_indices,
		Distance		   *out_distances)
{
	ArrayPageStorage *s = (ArrayPageStorage *)self;

	if (s->all_vectors == NULL || count == 0)
		return 0;

	mkt_topk_reset(&s->rerank_topk);
	s->rerank_topk.k = keep;

	for (uint32_t i = 0; i < count; i++)
	{
		Distance d;
		if (candidates[i].error == 0.0f)
		{
			d = candidates[i].distance;
		}
		else
		{
			uint32_t vid = mkt_posting_decode_vector_id(candidates[i].id);
			if (vid < s->nvecs)
			{
				const float *vec = s->all_vectors + (size_t)vid * dim;
				d				 = mkt_l2_distance_squared(query, vec, dim);
			}
			else
			{
				d = candidates[i].distance;
			}
		}

		mkt_topk_insert(&s->rerank_topk, d, 0.0f, (uint64_t)i);
	}

	uint32_t nresults;
	mkt_topk_extract_sorted(&s->rerank_topk, s->rerank_entries, &nresults);
	if (nresults > s->rerank_cap)
		nresults = s->rerank_cap;

	for (uint32_t i = 0; i < nresults; i++)
	{
		out_indices[i]	 = (uint32_t)s->rerank_entries[i].id;
		out_distances[i] = s->rerank_entries[i].distance;
	}

	return nresults;
}

static const MktStorageOps array_page_storage_ops = {
		.read_page	  = aps_read_page,
		.release_page = aps_release_page,
		.write_page	  = aps_write_page,
		.new_page	  = aps_new_page,
		.commit_page  = aps_commit_page,
		.rerank		  = aps_rerank,
};

/* ----------------------------------------------------------------
 * Cluster list helpers (per-cluster vector ID lists)
 * ---------------------------------------------------------------- */

static void
cluster_list_init(MktClusterList *cl, uint32_t initial_cap)
{
	cl->count	 = 0;
	cl->capacity = initial_cap;
	cl->ids		 = mkt_alloc(initial_cap * sizeof(uint32_t));
}

static void
cluster_list_append(MktClusterList *cl, uint32_t id)
{
	if (cl->count == cl->capacity)
	{
		uint32_t old_cap = cl->capacity;
		uint32_t new_cap = old_cap * 2;
		cl->ids			 = arena_grow(
				 cl->ids,
				 old_cap * sizeof(uint32_t),
				 new_cap * sizeof(uint32_t));
		cl->capacity = new_cap;
	}
	cl->ids[cl->count++] = id;
}

/* ----------------------------------------------------------------
 * ArrayPageStorage initialization helper
 * ---------------------------------------------------------------- */

static ArrayPageStorage
make_array_page_storage(uint32_t est_pages)
{
	ArrayPageStorage s = {
			.base		= {.ops = &array_page_storage_ops},
			.pages		= calloc(est_pages, BLCKSZ),
			.next_blkno = 0,
			.page_cap	= est_pages,
	};
	pthread_mutex_init(&s.alloc_mutex, NULL);
	return s;
}

/* ----------------------------------------------------------------
 * Normalize helpers
 * ---------------------------------------------------------------- */

static void
normalize_vector(float *v, Dimension dim)
{
	float norm = mkt_l2_norm(v, dim);
	if (norm > 0.0f)
		mkt_vector_scale(v, 1.0f / norm, v, dim);
}

static void
normalize_all(float *data, uint32_t nvecs, Dimension dim)
{
	for (uint32_t i = 0; i < nvecs; i++)
		normalize_vector(data + (size_t)i * dim, dim);
}

/* ----------------------------------------------------------------
 * Parallel posting build — each thread processes a vector slice
 *
 * Mirrors the PG model: each worker independently scans its portion
 * of the heap, assigns vectors to clusters, encodes, and writes
 * posting pages. Workers share pre-allocated page ranges per cluster
 * via atomic counters. After all workers finish, page chains are
 * linked together.
 * ---------------------------------------------------------------- */

typedef struct ParPostingWorkerArg
{
	uint32_t			  thread_id;
	uint32_t			  vec_start;
	uint32_t			  vec_end;
	MktIndex			 *idx;
	const HKMeansResult	 *tree;
	const MktIndexConfig *config;
	const MktBuildParams *bp;
	Dimension			  dim;
	uint32_t			  nlist;

	/* Pre-computed assignments (NULL to re-assign) */
	const MktBatchAssignment *batch;

	/* Per-cluster shared page reservations */
	BlockNumber		  *reserve_starts;
	uint32_t		  *reserve_counts;
	_Atomic(uint32_t) *reserve_nexts;

	/* Output: per-cluster head/tail from this thread */
	BlockNumber *out_heads;
	BlockNumber *out_tails;
} ParPostingWorkerArg;

static void *
par_posting_worker_fn(void *arg)
{
	ParPostingWorkerArg *w = (ParPostingWorkerArg *)arg;

	MktMemCtx thread_ctx = mkt_memctx_create(NULL, "par_posting");
	mkt_memctx_switch(thread_ctx);

	uint32_t  nlist = w->nlist;
	Dimension dim	= w->dim;

	MktPostingBuilder *builders = malloc(
			(size_t)nlist * sizeof(MktPostingBuilder));
	bool *active = calloc(nlist, sizeof(bool));

	MktBuildWorkerBufs bufs = {0};
	if (w->batch == NULL)
		bufs = mkt_build_worker_bufs_create(dim);

	for (BlockNumber c = 0; c < nlist; c++)
	{
		w->out_heads[c] = InvalidBlockNumber;
		w->out_tails[c] = InvalidBlockNumber;
	}

	for (uint32_t i = w->vec_start; i < w->vec_end; i++)
	{
		const float *vec = w->idx->all_vectors + (size_t)i * dim;
		uint32_t	 primary, secondary;

		if (w->batch != NULL)
		{
			primary	  = w->batch->primary[i];
			secondary = w->batch->secondary[i];
		}
		else
		{
			MktBuildAssignment asgn =
					mkt_build_assign_vector(w->tree, vec, w->bp, &bufs);
			primary	  = asgn.primary;
			secondary = asgn.secondary;
		}

		/* Add to primary cluster */
		uint32_t clusters[2] = {primary, secondary};
		uint32_t nclusters	 = (secondary != MKT_INVALID_CLUSTER) ? 2 : 1;

		for (uint32_t ci = 0; ci < nclusters; ci++)
		{
			uint32_t c = clusters[ci];
			if (!active[c])
			{
				const float *cent = w->tree->leaf_centroids + (size_t)c * dim;

				if (w->thread_id == 0)
				{
					const float *pt_cent = w->idx->pt_centroids +
										   (size_t)c * dim;
					if (w->config->fastscan)
						mkt_posting_builder_init_fastscan(
								&builders[c],
								&w->idx->posting_storage.base,
								w->idx->base.params,
								dim,
								c,
								cent,
								pt_cent);
					else
						mkt_posting_builder_init(
								&builders[c],
								&w->idx->posting_storage.base,
								w->idx->base.params,
								dim,
								c,
								cent,
								pt_cent);
				}
				else
				{
					if (w->config->fastscan)
						mkt_posting_builder_init_continuation_fastscan(
								&builders[c],
								&w->idx->posting_storage.base,
								w->idx->base.params,
								dim,
								c,
								cent);
					else
						mkt_posting_builder_init_continuation(
								&builders[c],
								&w->idx->posting_storage.base,
								w->idx->base.params,
								dim,
								c,
								cent);
				}

				if (w->reserve_starts != NULL)
				{
					mkt_posting_builder_set_shared_reserve(
							&builders[c],
							w->reserve_starts[c],
							w->reserve_counts[c],
							&w->reserve_nexts[c]);
					if (w->thread_id == 0)
						mkt_posting_builder_set_first_blkno(
								&builders[c], w->reserve_starts[c]);
				}

				active[c] = true;
			}

			ItemPointerData tid;
			mkt_posting_set_vector_id(&tid, i);
			mkt_posting_builder_add(&builders[c], tid, vec);
		}
	}

	/* Finish all active builders */
	for (uint32_t c = 0; c < nlist; c++)
	{
		if (!active[c])
			continue;
		mkt_posting_builder_finish(&builders[c]);
		w->out_heads[c] = mkt_posting_builder_head(&builders[c]);
		w->out_tails[c] = mkt_posting_builder_tail(&builders[c]);
		mkt_posting_builder_cleanup(&builders[c]);
	}

	if (w->batch == NULL)
		mkt_build_worker_bufs_free(&bufs);

	free(builders);
	free(active);
	mkt_memctx_delete(thread_ctx);
	return NULL;
}

static uint32_t
sa_detect_nthreads(void)
{
	long n = sysconf(_SC_NPROCESSORS_ONLN);
	if (n < 1)
		n = 1;
	if (n > 64)
		n = 64;
	return (uint32_t)n;
}

/* ----------------------------------------------------------------
 * Build
 * ---------------------------------------------------------------- */

MktIndex *
mkt_index_build(MktVectorSource *src, const MktIndexConfig *config)
{
	if (src == NULL || src->nvecs == 0 || src->dim == 0 || config == NULL)
		return NULL;

	uint32_t  nvecs = src->nvecs;
	Dimension dim	= src->dim;

	mkt_distance_init();

	/* Long-lived context for index data. Build-phase temporaries
	 * go into a child context that gets deleted after build. */
	MktMemCtx idx_ctx	= mkt_memctx_create(NULL, "index");
	MktMemCtx build_ctx = mkt_memctx_create(idx_ctx, "index_build");
	MktMemCtx old_ctx	= mkt_memctx_switch(idx_ctx);

	MktIndex *idx			  = mkt_alloc0(sizeof(MktIndex));
	idx->memctx				  = idx_ctx;
	idx->base.dim			  = dim;
	idx->base.metric		  = config->metric;
	idx->base.centroid_format = config->centroid_fmt;

	/* Resolve nlist */
	uint32_t nlist = config->nlist;
	if (nlist == 0)
	{
		nlist = (uint32_t)sqrt((double)nvecs);
		if (nlist < 1)
			nlist = 1;
		if (nlist > 10000)
			nlist = 10000;
	}

	/* Resolve fan_out */
	uint32_t fan_out = config->fan_out;
	if (fan_out == 0)
		fan_out = mkt_auto_fan_out(0, nlist, 0);
	idx->fan_out = fan_out;

	/* Pass 1: sample vectors for clustering (build context).
	 * Use strided reads for uniform coverage across the dataset. */
	mkt_memctx_switch(build_ctx);
	uint32_t max_samples = nvecs < 256000 ? nvecs : 256000;
	uint32_t stride		 = nvecs / max_samples;
	if (stride < 1)
		stride = 1;
	float *samples = mkt_alloc((size_t)max_samples * dim * sizeof(float));

	{
		const float *vec;
		uint32_t	 id, n = 0;
		while (n < max_samples && src->next(src, stride, &vec, &id))
		{
			memcpy(samples + (size_t)n * dim, vec, dim * sizeof(float));
			n++;
		}
		max_samples = n;
	}

	/* Normalize samples for cosine */
	if (idx->base.metric == DISTANCE_COSINE)
		normalize_all(samples, max_samples, dim);

	/* Run hierarchical k-means */
	KMeansOptions km_opts = MKT_KMEANS_OPTIONS_DEFAULT;
	if (config->km_nredo > 0)
		km_opts.nredo = config->km_nredo;
	if (config->km_max_iter > 0)
		km_opts.max_iterations = config->km_max_iter;

	HKMeansResult *tree = mkt_hkmeans_f32(
			samples,
			max_samples,
			dim,
			nlist,
			fan_out,
			idx->base.metric,
			&km_opts);

	/* Switch back to index context for long-lived allocations */
	mkt_memctx_switch(idx_ctx);

	if (tree == NULL)
	{
		mkt_memctx_switch(old_ctx);
		mkt_memctx_delete(idx_ctx); /* frees idx, build_ctx, everything */
		return NULL;
	}

	nlist			  = tree->nleaves;
	idx->nlist		  = nlist;
	idx->base.nlevels = (uint8_t)tree->nlevels;

	/* Normalize leaf centroids for cosine */
	if (idx->base.metric == DISTANCE_COSINE)
	{
		for (uint32_t c = 0; c < nlist; c++)
			normalize_vector(tree->leaf_centroids + (size_t)c * dim, dim);
	}

	/* Global mean */
	idx->global_mean = mkt_alloc(dim * sizeof(float));
	mkt_vector_mean(tree->leaf_centroids, nlist, dim, idx->global_mean);
	if (idx->base.metric == DISTANCE_COSINE)
		normalize_vector(idx->global_mean, dim);

	/* Save leaf centroids for per-cluster query preparation */
	idx->leaf_centroids = mkt_alloc((size_t)nlist * dim * sizeof(float));
	memcpy(idx->leaf_centroids,
		   tree->leaf_centroids,
		   (size_t)nlist * dim * sizeof(float));

	/* RaBitQ params */
	idx->base.params = mkt_rabitq_create(dim, 42);

	/* Precompute P^T * centroids for zero-alloc query path.
	 * This turns per-cluster O(dim²) matrix multiply into O(dim)
	 * vector subtraction at query time. */
	idx->pt_centroids = mkt_alloc((size_t)nlist * dim * sizeof(float));
	for (uint32_t c = 0; c < nlist; c++)
		mkt_rabitq_rotate(
				idx->base.params,
				idx->leaf_centroids + (size_t)c * dim,
				idx->pt_centroids + (size_t)c * dim);

	idx->base.pt_global_mean = mkt_alloc(dim * sizeof(float));
	mkt_rabitq_rotate(
			idx->base.params, idx->global_mean, idx->base.pt_global_mean);

	/* Build centroid pages */
	uint32_t est_centroid_pages = nlist + 100;
	idx->centroid_storage		= make_array_page_storage(est_centroid_pages);

	uint32_t max_ent =
			mkt_centroid_max_entries_fmt(dim, idx->base.centroid_format);

	/* Temporary arrays for centroid page layout (build context) */
	mkt_memctx_switch(build_ctx);
	BlockNumber *node_first_blkno = mkt_alloc(
			tree->nnodes * sizeof(BlockNumber));
	idx->base.first_centroid = 0;
	mkt_compute_centroid_layout(
			tree, max_ent, idx->base.first_centroid, node_first_blkno);

	/* Switch back to index context for posting data.
	 * Posting lists are built before centroid pages so that
	 * centroid leaf entries store actual posting block numbers. */
	mkt_memctx_switch(idx_ctx);

	/* Allocate flat vectors array for reranking */
	idx->all_vectors = mkt_alloc((size_t)nvecs * dim * sizeof(float));

	/* Initialize per-cluster ID lists */
	uint32_t est_per_cluster = nvecs / nlist + 1;
	idx->clusters			 = mkt_alloc0(nlist * sizeof(MktClusterList));
	for (uint32_t c = 0; c < nlist; c++)
		cluster_list_init(&idx->clusters[c], est_per_cluster);

	/* Pass 2: assign all vectors to clusters in parallel.
	 * Read all vectors into the rerank array, normalize if cosine,
	 * then run threaded batch assignment. */
	const MktBuildParams bp = {
			.dim			  = dim,
			.metric			  = config->metric,
			.soar_lambda	  = config->soar_lambda,
			.boundary_epsilon = config->boundary_epsilon,
	};

	src->reset(src);
	idx->nvecs = 0;

	{
		const float *vec;
		uint32_t	 id;
		while (src->next(src, 1, &vec, &id))
		{
			memcpy(idx->all_vectors + (size_t)id * dim,
				   vec,
				   dim * sizeof(float));
			idx->nvecs++;
		}
	}

	/* Normalize all vectors for cosine before parallel assignment */
	if (config->metric == DISTANCE_COSINE)
		normalize_all(idx->all_vectors, idx->nvecs, dim);

	uint32_t nworkers = config->nworkers;
	if (nworkers == 0)
		nworkers = sa_detect_nthreads();

	MktBatchAssignment batch = mkt_batch_assignment_create(idx->nvecs);

	mkt_build_assign_batch_parallel(
			idx->all_vectors, idx->nvecs, dim, tree, &bp, nworkers, &batch);

	/* Distribute assignments to cluster lists (sequential) */
	for (uint32_t i = 0; i < idx->nvecs; i++)
	{
		cluster_list_append(&idx->clusters[batch.primary[i]], i);

		if (batch.secondary[i] != MKT_INVALID_CLUSTER)
			cluster_list_append(&idx->clusters[batch.secondary[i]], i);
	}

	/* Compute max cluster size (needed for flat mode scan buffers) */
	idx->max_cluster_size = 0;
	for (uint32_t c = 0; c < nlist; c++)
		if (idx->clusters[c].count > idx->max_cluster_size)
			idx->max_cluster_size = idx->clusters[c].count;

	/* Build posting data if RaBitQ encoding is requested */
	if (config->encode_rabitq)
	{
		idx->posting_fmt = config->posting_fmt;

		if (config->posting_fmt == MKT_POSTING_FMT_PAGES)
		{
			/* Estimate per-cluster page counts from known sizes.
			 * Reserve contiguous block ranges so all pages for a
			 * cluster are adjacent in the page array. */
			uint32_t ent_first	  = mkt_posting_max_entries_first(dim);
			uint32_t ent_overflow = mkt_posting_max_entries(dim);

			BlockNumber *reserve_starts = malloc(nlist * sizeof(BlockNumber));
			uint32_t	*reserve_counts = malloc(nlist * sizeof(uint32_t));
			_Atomic(uint32_t) *reserve_nexts =
					calloc(nlist, sizeof(_Atomic(uint32_t)));

			BlockNumber total_reserved = 0;
			for (uint32_t c = 0; c < nlist; c++)
			{
				uint32_t cnt	= idx->clusters[c].count;
				uint32_t npages = 1; /* first page (with pt_centroid) */
				if (cnt > ent_first)
					npages += (cnt - ent_first + ent_overflow - 1) /
							  ent_overflow;
				/* Each thread may produce a partial page per cluster,
				 * so add (nworkers - 1) extra pages. */
				npages += nworkers - 1;
				reserve_starts[c] = total_reserved;
				reserve_counts[c] = npages;
				total_reserved += npages;
			}

			uint32_t est_extra	 = total_reserved / 10 + 100;
			idx->posting_storage = make_array_page_storage(
					total_reserved + est_extra);
			idx->posting_storage.next_blkno = total_reserved;
			idx->posting_heads = mkt_alloc(nlist * sizeof(BlockNumber));

			/* Slot 0 of each cluster is reserved for thread 0's
			 * first page (which carries pt_centroid). Shared
			 * atomic counters start at 1 so other threads skip
			 * slot 0. */
			for (uint32_t c = 0; c < nlist; c++)
				atomic_store(&reserve_nexts[c], 1);

			/* Parallel posting build: each thread processes its
			 * 1/N slice of vectors, like an independent PG worker. */
			uint64_t t_posting = now_ns();
			{
				uint32_t nt = nworkers;
				if (nt > nlist)
					nt = nlist;

				pthread_t			*threads = malloc(nt * sizeof(pthread_t));
				ParPostingWorkerArg *pargs	 = malloc(
						  nt * sizeof(ParPostingWorkerArg));

				/* Per-thread output arrays */
				BlockNumber **all_heads = malloc(nt * sizeof(BlockNumber *));
				BlockNumber **all_tails = malloc(nt * sizeof(BlockNumber *));
				for (uint32_t t = 0; t < nt; t++)
				{
					all_heads[t] = malloc(nlist * sizeof(BlockNumber));
					all_tails[t] = malloc(nlist * sizeof(BlockNumber));
				}

				uint32_t per_thread = idx->nvecs / nt;
				uint32_t remainder	= idx->nvecs % nt;
				uint32_t voff		= 0;

				for (uint32_t t = 0; t < nt; t++)
				{
					uint32_t chunk = per_thread + (t < remainder ? 1 : 0);
					pargs[t]	   = (ParPostingWorkerArg){
								  .thread_id	  = t,
								  .vec_start	  = voff,
								  .vec_end		  = voff + chunk,
								  .idx			  = idx,
								  .tree			  = tree,
								  .config		  = config,
								  .bp			  = &bp,
								  .dim			  = dim,
								  .nlist		  = nlist,
								  .batch		  = &batch,
								  .reserve_starts = reserve_starts,
								  .reserve_counts = reserve_counts,
								  .reserve_nexts  = reserve_nexts,
								  .out_heads	  = all_heads[t],
								  .out_tails	  = all_tails[t],
					  };
					pthread_create(
							&threads[t],
							NULL,
							par_posting_worker_fn,
							&pargs[t]);
					voff += chunk;
				}

				for (uint32_t t = 0; t < nt; t++)
					pthread_join(threads[t], NULL);

				double ms_posting = (double)(now_ns() - t_posting) / 1e6;

				/* Collect all pages per cluster and sort by block
				 * number so the chain follows sequential I/O order. */
				uint32_t total_pages = 0;
				uint32_t empty_pages = 0;
				{
					uint32_t max_pages = 0;
					for (uint32_t c = 0; c < nlist; c++)
						if (reserve_counts[c] > max_pages)
							max_pages = reserve_counts[c];
					max_pages += 64;
					BlockNumber *chain = malloc(
							max_pages * sizeof(BlockNumber));

					for (uint32_t c = 0; c < nlist; c++)
					{
						uint32_t nchain = 0;

						/* If thread 0 had no vectors for this cluster,
						 * create a synthetic first page at slot 0. */
						if (all_heads[0][c] == InvalidBlockNumber)
						{
							BlockNumber blkno = reserve_starts[c];
							Page		page  = idx->posting_storage.pages +
										(size_t)blkno * BLCKSZ;
							uint16_t flags = MKT_POSTING_PAGE_FIRST;
							if (config->fastscan)
								flags |= MKT_POSTING_PAGE_FASTSCAN;
							mkt_posting_page_init(page, c, dim, flags);
							memcpy(mkt_posting_pt_centroid_mut(page),
								   idx->pt_centroids + (size_t)c * dim,
								   dim * sizeof(float));
							mkt_posting_opaque(page)->next_blkno =
									InvalidBlockNumber;
							chain[nchain++] = blkno;
						}

						/* Collect pages from all threads */
						for (uint32_t t = 0; t < nt; t++)
						{
							BlockNumber blk = all_heads[t][c];
							while (blk != InvalidBlockNumber)
							{
								chain[nchain++] = blk;
								Page pg			= idx->posting_storage.pages +
										  (size_t)blk * BLCKSZ;
								blk = mkt_posting_opaque(pg)->next_blkno;
							}
						}

						/* Sort by block number for sequential access */
						for (uint32_t i = 1; i < nchain; i++)
							for (uint32_t j = i;
								 j > 0 && chain[j] < chain[j - 1];
								 j--)
							{
								BlockNumber tmp = chain[j];
								chain[j]		= chain[j - 1];
								chain[j - 1]	= tmp;
							}

						/* Rewrite chain links in sorted order */
						for (uint32_t i = 0; i + 1 < nchain; i++)
						{
							Page pg = idx->posting_storage.pages +
									  (size_t)chain[i] * BLCKSZ;
							mkt_posting_opaque(pg)->next_blkno = chain[i + 1];
						}
						if (nchain > 0)
						{
							Page last = idx->posting_storage.pages +
										(size_t)chain[nchain - 1] * BLCKSZ;
							mkt_posting_opaque(last)->next_blkno =
									InvalidBlockNumber;
						}

						/* Merge trailing partial pages: walk the chain
						 * backwards, collect entries from non-full
						 * pages, and repack them into fewer pages. */
						if (nchain > 1 && !config->fastscan)
						{
							uint32_t ent_max = mkt_posting_max_entries(dim);
							uint32_t tail_entries = 0;
							uint32_t merge_from	  = nchain;

							for (uint32_t i = nchain; i > 1; i--)
							{
								Page pg = idx->posting_storage.pages +
										  (size_t)chain[i - 1] * BLCKSZ;
								MktPostingPageOpaque *op = mkt_posting_opaque(
										pg);
								if (op->entry_count == ent_max)
									break;
								tail_entries += op->entry_count;
								merge_from = i - 1;
							}

							uint32_t merged_pages = (tail_entries + ent_max -
													 1) /
													ent_max;
							uint32_t orig_pages = nchain - merge_from;

							if (merged_pages < orig_pages && tail_entries > 0)
							{
								MktPostingBuilder mb;
								const float		 *cent = tree->leaf_centroids +
													(size_t)c * dim;
								mkt_posting_builder_init_continuation(
										&mb,
										&idx->posting_storage.base,
										NULL,
										dim,
										c,
										cent);
								mkt_posting_builder_set_shared_reserve(
										&mb,
										reserve_starts[c],
										reserve_counts[c],
										&reserve_nexts[c]);

								for (uint32_t i = merge_from; i < nchain; i++)
								{
									Page pg = idx->posting_storage.pages +
											  (size_t)chain[i] * BLCKSZ;
									MktPostingPageOpaque *op =
											mkt_posting_opaque(pg);
									bool  is_fp = (op->flags &
												   MKT_POSTING_PAGE_FIRST) != 0;
									char *ct =
											is_fp ? mkt_posting_content_first(
															pg, dim)
												  : mkt_posting_content(pg);
									for (uint32_t e = 0; e < op->entry_count;
										 e++)
									{
										MktPostingEntryHeader *hdr =
												mkt_posting_entry_at(
														ct, e, dim);
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
								BlockNumber mh = mkt_posting_builder_head(&mb);
								BlockNumber mt = mkt_posting_builder_tail(&mb);
								mkt_posting_builder_cleanup(&mb);

								/* Truncate chain: replace merged tail
								 * pages with the merge output. */
								nchain = merge_from;

								/* Walk merge chain, append blocks */
								BlockNumber blk = mh;
								while (blk != InvalidBlockNumber)
								{
									chain[nchain++] = blk;
									Page pg = idx->posting_storage.pages +
											  (size_t)blk * BLCKSZ;
									blk = mkt_posting_opaque(pg)->next_blkno;
								}

								/* Relink the junction */
								if (merge_from > 0)
								{
									Page prev = idx->posting_storage.pages +
												(size_t)chain[merge_from - 1] *
														BLCKSZ;
									mkt_posting_opaque(prev)->next_blkno =
											chain[merge_from];
								}
								Page last = idx->posting_storage.pages +
											(size_t)chain[nchain - 1] * BLCKSZ;
								mkt_posting_opaque(last)->next_blkno =
										InvalidBlockNumber;

								(void)mt;
							}
						}

						idx->posting_heads[c] = nchain > 0
													  ? chain[0]
													  : InvalidBlockNumber;
						total_pages += nchain;
					}

					/* Count empty reserved pages (not part of
					 * any chain — will be used by future inserts) */
					uint32_t used_slots = atomic_load(&reserve_nexts[0]);
					for (uint32_t c = 1; c < nlist; c++)
					{
						uint32_t slot = atomic_load(&reserve_nexts[c]);
						used_slots += slot;
					}
					BlockNumber total_slots = 0;
					for (uint32_t c = 0; c < nlist; c++)
						total_slots += reserve_counts[c];
					empty_pages = total_slots > used_slots
										? total_slots - used_slots
										: 0;

					free(chain);
				}

				uint32_t overflow = idx->posting_storage.next_blkno >
													total_reserved
										  ? idx->posting_storage.next_blkno -
													total_reserved
										  : 0;

				fprintf(stderr,
						"  Posting build: %.1fms "
						"(%u threads)\n",
						ms_posting,
						nt);
				fprintf(stderr,
						"  Pages: %u total, %u empty, "
						"%u overflow\n",
						total_pages,
						empty_pages,
						overflow);

				for (uint32_t t = 0; t < nt; t++)
				{
					free(all_heads[t]);
					free(all_tails[t]);
				}
				free(all_heads);
				free(all_tails);
				free(threads);
				free(pargs);
			}

			free(reserve_starts);
			free(reserve_counts);
			free(reserve_nexts);

			if (config->fastscan)
				idx->base.fastscan = config->fastscan;
		}
		else
		{
			/* Flat mode: one buffer per cluster */
			idx->flat_pages = mkt_alloc(nlist * sizeof(char *));

			for (uint32_t c = 0; c < nlist; c++)
			{
				MktClusterList *cl	 = &idx->clusters[c];
				const float	   *cent = tree->leaf_centroids + (size_t)c * dim;

				MktFlatPostingBuilder builder;
				mkt_flat_posting_builder_init(
						&builder, idx->base.params, dim, c, cent, cl->count);

				for (uint32_t i = 0; i < cl->count; i++)
				{
					uint32_t		vid = cl->ids[i];
					const float	   *vec = idx->all_vectors + (size_t)vid * dim;
					ItemPointerData tid;
					mkt_posting_set_vector_id(&tid, vid);
					mkt_flat_posting_builder_add(&builder, tid, vec);
				}

				idx->flat_pages[c] = mkt_flat_posting_builder_finish(&builder);
				mkt_flat_posting_builder_cleanup(&builder);
			}
		}

		idx->has_posting_data = true;
	}

	idx->has_replication = config->soar_lambda > 0.0 ||
						   config->boundary_epsilon > 0.0;

	/* Write centroid pages — after posting lists so leaf entries
	 * store actual posting block numbers (not just cluster indices). */
	{
		/* For paged mode, store actual posting block numbers so the
		 * shared scan can use posting_head directly. For flat mode,
		 * store cluster indices (the inline loop maps them). */
		BlockNumber *leaf_heads;
		if (idx->has_posting_data &&
			config->posting_fmt == MKT_POSTING_FMT_PAGES)
		{
			leaf_heads = idx->posting_heads;
		}
		else
		{
			leaf_heads = mkt_alloc(nlist * sizeof(BlockNumber));
			for (uint32_t i = 0; i < nlist; i++)
				leaf_heads[i] = (BlockNumber)i;
		}

		mkt_memctx_switch(build_ctx);
		mkt_write_centroid_tree(
				&idx->centroid_storage.base,
				tree,
				dim,
				fan_out,
				idx->base.centroid_format,
				idx->base.centroid_format == MKT_CENTROID_FMT_RABITQ
						? idx->base.params
						: NULL,
				idx->global_mean,
				leaf_heads,
				node_first_blkno,
				NULL); /* pt_centroids on posting pages, not here */
		mkt_memctx_switch(idx_ctx);
	}

	mkt_hkmeans_result_destroy(tree);

	mkt_memctx_switch(old_ctx);
	mkt_memctx_delete(build_ctx);

	/* Wire storage pointers in base (concrete storage owned above) */
	idx->base.centroid_storage = &idx->centroid_storage.base;
	idx->base.posting_storage  = &idx->posting_storage.base;
	idx->base.page_base		   = idx->posting_storage.pages;

	/* Wire rerank data on posting storage */
	idx->posting_storage.all_vectors = idx->all_vectors;
	idx->posting_storage.nvecs		 = idx->nvecs;
	idx->posting_storage.metric		 = idx->base.metric;

	/* Pre-allocate rerank buffers */
	uint32_t rerank_cap = 256;
	mkt_topk_init(&idx->posting_storage.rerank_topk, rerank_cap);
	idx->posting_storage.rerank_entries = mkt_alloc(
			rerank_cap * sizeof(MktTopKEntry));
	idx->posting_storage.rerank_cap = rerank_cap;

	return idx;
}

void
mkt_index_destroy(MktIndex *idx)
{
	if (idx == NULL)
		return;

	mkt_topk_cleanup(&idx->posting_storage.rerank_topk);

	free(idx->posting_storage.pages);
	free(idx->centroid_storage.pages);

	/* Deleting the memory context frees idx and all owned data. */
	mkt_memctx_delete(idx->memctx);
}
