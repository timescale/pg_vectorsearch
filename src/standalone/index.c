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
#include <stdlib.h>
#include <string.h>
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
		uint32_t new_cap = s->page_cap * 2;
		s->pages		 = arena_grow(
				s->pages,
				(size_t)s->page_cap * BLCKSZ,
				(size_t)new_cap * BLCKSZ);
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
			.pages		= mkt_alloc0((size_t)est_pages * BLCKSZ),
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
 * Parallel posting build helpers
 * ---------------------------------------------------------------- */

typedef struct PostingWorkerArg
{
	uint32_t			  c_start;
	uint32_t			  c_end;
	MktIndex			 *idx;
	const HKMeansResult	 *tree;
	const MktIndexConfig *config;
	Dimension			  dim;
} PostingWorkerArg;

static void *
posting_worker_fn(void *arg)
{
	PostingWorkerArg *w = (PostingWorkerArg *)arg;

	/* Each thread needs its own memory context for mkt_alloc */
	MktMemCtx thread_ctx = mkt_memctx_create(NULL, "posting_worker");
	mkt_memctx_switch(thread_ctx);

	for (uint32_t c = w->c_start; c < w->c_end; c++)
	{
		MktClusterList *cl		= &w->idx->clusters[c];
		const float	   *cent	= w->tree->leaf_centroids + (size_t)c * w->dim;
		const float	   *pt_cent = w->idx->pt_centroids + (size_t)c * w->dim;

		MktPostingBuilder b;
		if (w->config->fastscan)
			mkt_posting_builder_init_fastscan(
					&b,
					&w->idx->posting_storage.base,
					w->idx->base.params,
					w->dim,
					c,
					cent,
					pt_cent);
		else
			mkt_posting_builder_init(
					&b,
					&w->idx->posting_storage.base,
					w->idx->base.params,
					w->dim,
					c,
					cent,
					pt_cent);

		for (uint32_t i = 0; i < cl->count; i++)
		{
			uint32_t		vid = cl->ids[i];
			const float	   *vec = w->idx->all_vectors + (size_t)vid * w->dim;
			ItemPointerData tid;
			mkt_posting_set_vector_id(&tid, vid);
			mkt_posting_builder_add(&b, tid, vec);
		}

		w->idx->posting_heads[c] = mkt_posting_builder_finish(&b);
		mkt_posting_builder_cleanup(&b);
	}

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
			/* Paged mode: BLCKSZ pages in ArrayPageStorage */
			uint32_t ent_per_page = mkt_posting_max_entries(dim);
			uint32_t est_pages	  = (nvecs / ent_per_page) + nlist + 100;
			idx->posting_storage  = make_array_page_storage(est_pages);
			idx->posting_heads	  = mkt_alloc(nlist * sizeof(BlockNumber));

			/* Parallel posting build: each thread processes a range
			 * of clusters. Page allocation is mutex-protected. */
			{
				uint32_t nt = nworkers;
				if (nt > nlist)
					nt = nlist;

				pthread_t		 *threads = malloc(nt * sizeof(pthread_t));
				PostingWorkerArg *pargs	  = malloc(
						  nt * sizeof(PostingWorkerArg));

				uint32_t per = nlist / nt;
				uint32_t rem = nlist % nt;
				uint32_t off = 0;

				for (uint32_t t = 0; t < nt; t++)
				{
					uint32_t chunk = per + (t < rem ? 1 : 0);
					pargs[t]	   = (PostingWorkerArg){
								  .c_start = off,
								  .c_end   = off + chunk,
								  .idx	   = idx,
								  .tree	   = tree,
								  .config  = config,
								  .dim	   = dim,
					  };
					pthread_create(
							&threads[t], NULL, posting_worker_fn, &pargs[t]);
					off += chunk;
				}

				for (uint32_t t = 0; t < nt; t++)
					pthread_join(threads[t], NULL);

				free(threads);
				free(pargs);
			}

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

	/* Deleting the memory context frees idx and all owned data. */
	mkt_memctx_delete(idx->memctx);
}
