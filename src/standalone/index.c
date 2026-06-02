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
#include "algo/kmeans_internal.h"
#include "algo/topk.h"
#include "algo/vecops.h"
#include "core/memory.h"
#include "index/centroid_build.h"
#include "index/centroid_page.h"
#include "index/index_build.h"
#include "index/posting_build.h"
#include "index/posting_build_parallel.h"
#include "index/posting_convert.h"
#include "index/posting_page.h"
#include "standalone/index.h"
#include "standalone/thread_pool.h"

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
		MktMemCtx prev	  = mkt_memctx_switch(s->memctx);
		uint32_t  new_cap = s->page_cap * 2;
		s->pages		  = arena_grow(
				 s->pages,
				 (size_t)s->page_cap * BLCKSZ,
				 (size_t)new_cap * BLCKSZ);
		s->page_cap = new_cap;
		mkt_memctx_switch(prev);
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

static BlockNumber
aps_extend(MktStorage *self, uint32_t npages)
{
	ArrayPageStorage *s = (ArrayPageStorage *)self;
	pthread_mutex_lock(&s->alloc_mutex);

	BlockNumber start  = s->next_blkno;
	uint32_t	needed = start + npages;

	while (needed > s->page_cap)
	{
		MktMemCtx prev	  = mkt_memctx_switch(s->memctx);
		uint32_t  new_cap = s->page_cap * 2;
		if (new_cap < needed)
			new_cap = needed;
		s->pages = arena_grow(
				s->pages,
				(size_t)s->page_cap * BLCKSZ,
				(size_t)new_cap * BLCKSZ);
		s->page_cap = new_cap;
		mkt_memctx_switch(prev);
	}

	s->next_blkno = needed;
	pthread_mutex_unlock(&s->alloc_mutex);
	return start;
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
		.extend		  = aps_extend,
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
make_array_page_storage(uint32_t est_pages, MktMemCtx memctx)
{
	ArrayPageStorage s = {
			.base		= {.ops = &array_page_storage_ops},
			.pages		= mkt_alloc0((size_t)est_pages * BLCKSZ),
			.next_blkno = 0,
			.page_cap	= est_pages,
			.memctx		= memctx,
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
 * K-means parallel dispatch bridge
 *
 * Adapts the thread pool's parallel_for to the k-means callback
 * signature. K-means doesn't know about MktThreadPool.
 * ---------------------------------------------------------------- */

static void
km_parallel_for(void *ctx, uint32_t total, KMeansWorkFn work_fn, void *arg)
{
	mkt_thread_pool_parallel_for((MktThreadPool *)ctx, total, work_fn, arg);
}

static void
km_iterate(
		void		  *ctx,
		uint32_t	   total,
		KMeansWorkFn   work_fn,
		KMeansReduceFn reduce_fn,
		void		  *arg,
		uint32_t	   max_iterations)
{
	mkt_thread_pool_iterate(
			(MktThreadPool *)ctx,
			total,
			work_fn,
			(MktReduceFn)reduce_fn,
			arg,
			max_iterations);
}

/* ----------------------------------------------------------------
 * K-means iterate context for root/child k-means
 *
 * Uses kmeans_assign_accumulate (shared kernel) for the work step
 * and kmeans_merge_centroids for the reduce step, matching PG's
 * parallel k-means path.
 * ---------------------------------------------------------------- */

typedef struct KmIterCtx
{
	const float	  *samples;	  /* all samples (flat array) */
	float		  *centroids; /* current centroids [k * dim] */
	float		  *norms_c;	  /* centroid norms [k] */
	uint32_t	   k;
	Dimension	   dim;
	DistanceMetric metric;
	uint32_t	   nthreads;
	float		   tolerance;

	/* Per-thread accumulators */
	float	 *sums;		 /* [nthreads * k * dim] */
	uint32_t *cnts;		 /* [nthreads * k] */
	float	 *costs;	 /* [nthreads] */
	float	 *old_cents; /* [k * dim] for convergence */

	/* Optional filter for child k-means */
	const uint32_t *filter;		/* root assignments, NULL for root */
	uint32_t		filter_val; /* child index */
} KmIterCtx;

static void
km_iter_work(uint32_t thread_id, uint32_t start, uint32_t end, void *arg)
{
	KmIterCtx *ctx	   = arg;
	float	  *my_sums = ctx->sums + (size_t)thread_id * ctx->k * ctx->dim;
	uint32_t  *my_cnts = ctx->cnts + (size_t)thread_id * ctx->k;
	float	  *my_cost = &ctx->costs[thread_id];

	memset(my_sums, 0, (size_t)ctx->k * ctx->dim * sizeof(float));
	memset(my_cnts, 0, ctx->k * sizeof(uint32_t));
	*my_cost = 0.0f;

	kmeans_assign_accumulate(
			ctx->samples,
			NULL,
			start,
			end,
			ctx->centroids,
			ctx->norms_c,
			ctx->k,
			ctx->dim,
			ctx->metric,
			ctx->filter,
			ctx->filter_val,
			my_sums,
			my_cnts,
			my_cost);
}

static bool
km_iter_reduce(void *arg, uint32_t iteration)
{
	(void)iteration;
	KmIterCtx *ctx = arg;
	uint32_t   nt  = ctx->nthreads;
	uint32_t   k   = ctx->k;
	Dimension  dim = ctx->dim;

	memcpy(ctx->old_cents, ctx->centroids, (size_t)k * dim * sizeof(float));

	/* Build pointer arrays for merge (stack-allocated) */
	const float	   **all_sums = (const float **)alloca(nt * sizeof(float *));
	const uint32_t **all_cnts = (const uint32_t **)alloca(
			nt * sizeof(uint32_t *));

	for (uint32_t t = 0; t < nt; t++)
	{
		all_sums[t] = ctx->sums + (size_t)t * k * dim;
		all_cnts[t] = ctx->cnts + (size_t)t * k;
	}

	float total_cost;
	float shift_sq = kmeans_merge_centroids(
			ctx->centroids,
			ctx->norms_c,
			ctx->old_cents,
			all_sums,
			all_cnts,
			ctx->costs,
			nt,
			k,
			dim,
			ctx->metric,
			&total_cost);

	float tol_sq = ctx->tolerance * ctx->tolerance;
	return shift_sq >= tol_sq; /* true = continue */
}

/* ----------------------------------------------------------------
 * Parallel root assignment callback
 * ---------------------------------------------------------------- */

typedef struct RootAssignCtx
{
	const float	  *samples;
	const float	  *centroids;
	const float	  *norms_c;
	uint32_t	   k;
	Dimension	   dim;
	DistanceMetric metric;
	uint32_t	  *assignments;
} RootAssignCtx;

static void
par_root_assign_fn(uint32_t thread_id, uint32_t start, uint32_t end, void *arg)
{
	(void)thread_id;
	RootAssignCtx *ctx = arg;
	kmeans_assign(
			ctx->samples,
			start,
			end,
			ctx->centroids,
			ctx->norms_c,
			ctx->k,
			ctx->dim,
			ctx->metric,
			ctx->assignments);
}

/* ----------------------------------------------------------------
 * Parallel sample copy callback
 * ---------------------------------------------------------------- */

typedef struct SampleCopyCtx
{
	const float *src;
	float		*dst;
	Dimension	 dim;
	uint32_t	 stride;
} SampleCopyCtx;

static void
par_sample_copy_fn(uint32_t thread_id, uint32_t start, uint32_t end, void *arg)
{
	(void)thread_id;
	SampleCopyCtx *ctx = (SampleCopyCtx *)arg;
	Dimension	   dim = ctx->dim;
	size_t		   esz = dim * sizeof(float);

	for (uint32_t i = start; i < end; i++)
		memcpy(ctx->dst + (size_t)i * dim,
			   ctx->src + (size_t)(i * ctx->stride) * dim,
			   esz);
}

/* ----------------------------------------------------------------
 * Parallel centroid rotation callback
 * ---------------------------------------------------------------- */

typedef struct RotateCtx
{
	const RaBitQParams *params;
	const float		   *leaf_centroids;
	float			   *pt_centroids;
	Dimension			dim;
} RotateCtx;

static void
par_rotate_fn(uint32_t thread_id, uint32_t start, uint32_t end, void *arg)
{
	(void)thread_id;
	RotateCtx *ctx = (RotateCtx *)arg;
	Dimension  dim = ctx->dim;

	for (uint32_t c = start; c < end; c++)
		mkt_rabitq_rotate(
				ctx->params,
				ctx->leaf_centroids + (size_t)c * dim,
				ctx->pt_centroids + (size_t)c * dim);
}

/* ----------------------------------------------------------------
 * Parallel posting build callback
 *
 * Each thread creates its own memory context for transient
 * allocations. The MktPostingWorkerState is pre-initialized
 * by the caller and lives in the main context.
 * ---------------------------------------------------------------- */

typedef struct ParPostingCtx
{
	MktPostingWorkerState *workers;
	const float			  *all_vectors;
	const HKMeansResult	  *tree;
	MktBuildParams		   bp;
} ParPostingCtx;

static void
par_posting_fn(uint32_t thread_id, uint32_t start, uint32_t end, void *arg)
{
	ParPostingCtx		  *ctx = (ParPostingCtx *)arg;
	MktPostingWorkerState *ws  = &ctx->workers[thread_id];
	Dimension			   dim = ws->dim;

	MktMemCtx thread_ctx = mkt_memctx_create(NULL, "par_posting");
	MktMemCtx old_ctx	 = mkt_memctx_switch(thread_ctx);

	MktBuildWorkerBufs bufs	   = mkt_build_worker_bufs_create(dim);
	MktMemCtx		   tmp_ctx = mkt_memctx_create(thread_ctx, "par_tmp");

	for (uint32_t i = start; i < end; i++)
	{
		mkt_memctx_switch(tmp_ctx);
		const float		  *vec = ctx->all_vectors + (size_t)i * dim;
		MktBuildAssignment asgn =
				mkt_build_assign_vector(ctx->tree, vec, &ctx->bp, &bufs);
		mkt_memctx_switch(thread_ctx);
		mkt_posting_worker_add(
				ws, i, asgn.enc_vector, asgn.primary, asgn.secondary);
		mkt_memctx_reset(tmp_ctx);
	}

	mkt_posting_worker_finish(ws);
	mkt_build_worker_bufs_free(&bufs);

	mkt_memctx_switch(old_ctx);
	mkt_memctx_delete(thread_ctx);
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
 * Assign vectors to cluster lists (serial, inline)
 *
 * Used only by flat mode and no-rabitq brute-force path which need
 * per-cluster vector ID lists. The pages mode parallel path does
 * assignment inline in each posting worker instead.
 * ---------------------------------------------------------------- */

static void
assign_to_cluster_lists(
		MktIndex			 *idx,
		const HKMeansResult	 *tree,
		const MktBuildParams *bp,
		Dimension			  dim,
		uint32_t			  nlist)
{
	MktBuildWorkerBufs bufs = mkt_build_worker_bufs_create(dim);

	for (uint32_t i = 0; i < idx->nvecs; i++)
	{
		const float		  *vec = idx->all_vectors + (size_t)i * dim;
		MktBuildAssignment asgn =
				mkt_build_assign_vector(tree, vec, bp, &bufs);

		cluster_list_append(&idx->clusters[asgn.primary], i);
		if (asgn.secondary != MKT_INVALID_CLUSTER)
			cluster_list_append(&idx->clusters[asgn.secondary], i);
	}

	mkt_build_worker_bufs_free(&bufs);

	idx->max_cluster_size = 0;
	for (uint32_t c = 0; c < nlist; c++)
		if (idx->clusters[c].count > idx->max_cluster_size)
			idx->max_cluster_size = idx->clusters[c].count;
}

/* ----------------------------------------------------------------
 * Build
 * ---------------------------------------------------------------- */

MktIndex *
mkt_index_build(
		MktVectorSource		 *src,
		const MktIndexConfig *config,
		MktBuildStats		 *stats)
{
	if (src == NULL || src->nvecs == 0 || src->dim == 0 || config == NULL)
		return NULL;

	uint32_t  nvecs = src->nvecs;
	Dimension dim	= src->dim;

	mkt_distance_init();

	/* Create thread pool early — used by k-means and posting build */
	uint32_t nworkers;
	if (config->nworkers < 0)
	{
		uint32_t ncpu = sa_detect_nthreads();
		nworkers	  = ncpu > 1 ? ncpu - 1 : 0;
	}
	else
	{
		nworkers = (uint32_t)config->nworkers;
	}
	MktThreadPool *pool = mkt_thread_pool_create(nworkers);

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

	/* --- Phase: load vectors --- */
	uint64_t t_phase = now_ns();

	idx->all_vectors = mkt_alloc((size_t)nvecs * dim * sizeof(float));

	if (src->read_all != NULL && src->read_all(src, idx->all_vectors))
	{
		idx->nvecs = nvecs;
	}
	else
	{
		idx->nvecs = 0;
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

	if (idx->base.metric == DISTANCE_COSINE)
		normalize_all(idx->all_vectors, idx->nvecs, dim);

	double ms_sample = (double)(now_ns() - t_phase) / 1e6;

	/* --- Phase: kmeans --- */
	t_phase = now_ns();
	mkt_memctx_switch(build_ctx);

	/* Subsample by stride into a contiguous buffer for
	 * cache-friendly k-means iteration. */
	uint32_t max_samples = idx->nvecs < 256000 ? idx->nvecs : 256000;
	uint32_t stride		 = idx->nvecs / max_samples;
	if (stride < 1)
		stride = 1;
	uint32_t km_nvecs = (stride > 1) ? max_samples : idx->nvecs;

	float *km_vectors;
	if (stride > 1)
	{
		km_vectors = mkt_alloc((size_t)km_nvecs * dim * sizeof(float));
		SampleCopyCtx sc_ctx = {
				.src	= idx->all_vectors,
				.dst	= km_vectors,
				.dim	= dim,
				.stride = stride,
		};
		mkt_thread_pool_parallel_for(
				pool, km_nvecs, par_sample_copy_fn, &sc_ctx);
	}
	else
	{
		km_vectors = idx->all_vectors;
	}

	KMeansOptions km_opts = MKT_KMEANS_OPTIONS_DEFAULT;
	if (config->km_nredo > 0)
		km_opts.nredo = config->km_nredo;
	if (config->km_max_iter > 0)
		km_opts.max_iterations = config->km_max_iter;
	km_opts.parallel_for = km_parallel_for;
	km_opts.iterate		 = km_iterate;
	km_opts.parallel_ctx = pool;
	km_opts.nthreads	 = nworkers + 1;

	/* Compute tree depth to decide explicit vs fallback path */
	uint32_t nlevels = 1;
	{
		uint32_t n = nlist;
		while (n > fan_out)
		{
			n = (n + fan_out - 1) / fan_out;
			nlevels++;
		}
	}

	uint32_t	   km_k		= fan_out < nlist ? fan_out : nlist;
	HKMeansResult *tree		= NULL;
	uint32_t	   nthreads = nworkers + 1;

	if (nlevels == 2)
	{
		/*
		 * Explicit root + child k-means using the shared
		 * kernels (kmeans_assign_accumulate + merge). This
		 * matches PG's parallel build path.
		 */

		/* --- Root k-means --- */
		float *centroids = mkt_alloc((size_t)km_k * dim * sizeof(float));
		float *norms_c	 = mkt_alloc(km_k * sizeof(float));

		/* Initial centroids: first km_k samples */
		uint32_t init_k = km_k < km_nvecs ? km_k : km_nvecs;
		memcpy(centroids, km_vectors, (size_t)init_k * dim * sizeof(float));

		if (idx->base.metric == DISTANCE_L2)
			for (uint32_t c = 0; c < init_k; c++)
				norms_c[c] =
						mkt_l2_norm_squared(centroids + (size_t)c * dim, dim);

		float *old_cents = mkt_alloc((size_t)km_k * dim * sizeof(float));
		float *sums		 = mkt_alloc0(
				 (size_t)nthreads * km_k * dim * sizeof(float));
		uint32_t *cnts = mkt_alloc0(
				(size_t)nthreads * km_k * sizeof(uint32_t));
		float *costs = mkt_alloc0(nthreads * sizeof(float));

		KmIterCtx root_ctx = {
				.samples	= km_vectors,
				.centroids	= centroids,
				.norms_c	= norms_c,
				.k			= km_k,
				.dim		= dim,
				.metric		= idx->base.metric,
				.nthreads	= nthreads,
				.tolerance	= km_opts.tolerance,
				.sums		= sums,
				.cnts		= cnts,
				.costs		= costs,
				.old_cents	= old_cents,
				.filter		= NULL,
				.filter_val = 0,
		};

		mkt_thread_pool_iterate(
				pool,
				km_nvecs,
				km_iter_work,
				(MktReduceFn)km_iter_reduce,
				&root_ctx,
				km_opts.max_iterations);

		/* --- Root assignment --- */
		uint32_t *root_asgn = mkt_alloc(km_nvecs * sizeof(uint32_t));

		RootAssignCtx ra_ctx = {
				.samples	 = km_vectors,
				.centroids	 = centroids,
				.norms_c	 = norms_c,
				.k			 = km_k,
				.dim		 = dim,
				.metric		 = idx->base.metric,
				.assignments = root_asgn,
		};
		mkt_thread_pool_parallel_for(
				pool, km_nvecs, par_root_assign_fn, &ra_ctx);

		/* Save root centroids before reusing buffers */
		float *root_cents = mkt_alloc((size_t)km_k * dim * sizeof(float));
		memcpy(root_cents, centroids, (size_t)km_k * dim * sizeof(float));

		/* --- Child k-means --- */
		float	**child_centroids = mkt_alloc(km_k * sizeof(float *));
		uint32_t *child_ks		  = mkt_alloc(km_k * sizeof(uint32_t));

		for (uint32_t child = 0; child < km_k; child++)
		{
			/* Count samples for this child */
			uint32_t child_count = 0;
			for (uint32_t i = 0; i < km_nvecs; i++)
				if (root_asgn[i] == child)
					child_count++;

			uint32_t child_k = fan_out < child_count ? fan_out : child_count;
			if (child_k < 1)
				child_k = 1;
			child_ks[child] = child_k;

			if (child_k <= 1)
			{
				child_centroids[child] = mkt_alloc(dim * sizeof(float));
				memcpy(child_centroids[child],
					   root_cents + (size_t)child * dim,
					   dim * sizeof(float));
				continue;
			}

			/* Pick initial child centroids: first child_k
			 * samples assigned to this child */
			float *child_cents = mkt_alloc(
					(size_t)child_k * dim * sizeof(float));
			uint32_t picked = 0;
			for (uint32_t i = 0; i < km_nvecs && picked < child_k; i++)
			{
				if (root_asgn[i] != child)
					continue;
				memcpy(child_cents + (size_t)picked * dim,
					   km_vectors + (size_t)i * dim,
					   dim * sizeof(float));
				picked++;
			}

			float *child_norms = mkt_alloc(child_k * sizeof(float));
			if (idx->base.metric == DISTANCE_L2)
				for (uint32_t c = 0; c < child_k; c++)
					child_norms[c] = mkt_l2_norm_squared(
							child_cents + (size_t)c * dim, dim);

			/* Reuse accumulator buffers (child_k <= km_k) */
			float *ch_old  = mkt_alloc((size_t)child_k * dim * sizeof(float));
			float *ch_sums = mkt_alloc0(
					(size_t)nthreads * child_k * dim * sizeof(float));
			uint32_t *ch_cnts = mkt_alloc0(
					(size_t)nthreads * child_k * sizeof(uint32_t));
			float *ch_costs = mkt_alloc0(nthreads * sizeof(float));

			KmIterCtx child_ctx = {
					.samples	= km_vectors,
					.centroids	= child_cents,
					.norms_c	= child_norms,
					.k			= child_k,
					.dim		= dim,
					.metric		= idx->base.metric,
					.nthreads	= nthreads,
					.tolerance	= km_opts.tolerance,
					.sums		= ch_sums,
					.cnts		= ch_cnts,
					.costs		= ch_costs,
					.old_cents	= ch_old,
					.filter		= root_asgn,
					.filter_val = child,
			};

			mkt_thread_pool_iterate(
					pool,
					km_nvecs,
					km_iter_work,
					(MktReduceFn)km_iter_reduce,
					&child_ctx,
					km_opts.max_iterations);

			child_centroids[child] = mkt_alloc(
					(size_t)child_k * dim * sizeof(float));
			memcpy(child_centroids[child],
				   child_cents,
				   (size_t)child_k * dim * sizeof(float));
		}

		tree = mkt_hkmeans_build_two_level(
				root_cents,
				km_k,
				(const float **)child_centroids,
				child_ks,
				dim);
	}
	else
	{
		/* nlevels != 2: fall back to mkt_hkmeans_f32 */
		tree = mkt_hkmeans_f32(
				km_vectors,
				km_nvecs,
				NULL,
				dim,
				nlist,
				fan_out,
				idx->base.metric,
				&km_opts);
	}

	/* --- Phase: setup --- */
	double ms_kmeans = (double)(now_ns() - t_phase) / 1e6;
	t_phase			 = now_ns();

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
			normalize_vector(hk_leaf_centroids(tree) + (size_t)c * dim, dim);
	}

	/* Global mean */
	idx->global_mean = mkt_alloc(dim * sizeof(float));
	mkt_vector_mean(hk_leaf_centroids(tree), nlist, dim, idx->global_mean);
	if (idx->base.metric == DISTANCE_COSINE)
		normalize_vector(idx->global_mean, dim);

	/* Save leaf centroids for per-cluster query preparation */
	idx->leaf_centroids = mkt_alloc((size_t)nlist * dim * sizeof(float));
	memcpy(idx->leaf_centroids,
		   hk_leaf_centroids(tree),
		   (size_t)nlist * dim * sizeof(float));

	/* RaBitQ params */
	idx->base.params = mkt_rabitq_create(dim, 42);

	/* Precompute P^T * centroids for zero-alloc query path. */
	idx->pt_centroids	 = mkt_alloc((size_t)nlist * dim * sizeof(float));
	RotateCtx rotate_ctx = {
			.params			= idx->base.params,
			.leaf_centroids = idx->leaf_centroids,
			.pt_centroids	= idx->pt_centroids,
			.dim			= dim,
	};
	mkt_thread_pool_parallel_for(pool, nlist, par_rotate_fn, &rotate_ctx);

	idx->base.pt_global_mean = mkt_alloc(dim * sizeof(float));
	mkt_rabitq_rotate(
			idx->base.params, idx->global_mean, idx->base.pt_global_mean);

	/* Build centroid pages */
	uint32_t est_centroid_pages = nlist + 100;
	idx->centroid_storage =
			make_array_page_storage(est_centroid_pages, idx_ctx);

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

	/* Initialize per-cluster ID lists */
	uint32_t est_per_cluster = nvecs / nlist + 1;
	idx->clusters			 = mkt_alloc0(nlist * sizeof(MktClusterList));
	for (uint32_t c = 0; c < nlist; c++)
		cluster_list_init(&idx->clusters[c], est_per_cluster);

	double				 ms_setup = (double)(now_ns() - t_phase) / 1e6;
	const MktBuildParams bp		  = {
				  .dim				= dim,
				  .metric			= config->metric,
				  .soar_lambda		= config->soar_lambda,
				  .boundary_epsilon = config->boundary_epsilon,
	  };

	/* --- Phase: posting --- */
	t_phase = now_ns();

	uint32_t nt = nworkers + 1; /* workers + leader */

	double	 ms_parallel = 0, ms_merge = 0;
	uint32_t stat_nworkers = nworkers, stat_pages = 0;
	uint32_t stat_merge_in = 0, stat_merge_out = 0;
	if (config->encode_rabitq)
	{
		idx->posting_fmt = config->posting_fmt;

		if (config->posting_fmt == MKT_POSTING_FMT_PAGES)
		{
			/* Shared partial page buffer — must be malloc'd
			 * since workers write from their own contexts. */
			char *partials = config->fastscan
								   ? NULL
								   : calloc((size_t)nt * nlist, BLCKSZ);

			mkt_memctx_switch(build_ctx);
			MktPostingWorkerState *workers = mkt_alloc(
					nt * sizeof(MktPostingWorkerState));
			MktPostingBatch **all_batches = mkt_alloc(
					nt * sizeof(MktPostingBatch *));
			bool **all_active = mkt_alloc(nt * sizeof(bool *));
			/* Each worker needs its own context for its deferred batch
			 * pages: arena allocation is not thread-safe within a single
			 * context, so concurrent workers must not share one. They are
			 * parented under a single context so one delete frees them all
			 * after the merge, and they outlive the worker threads. */
			MktMemCtx batch_parent =
					mkt_memctx_create(build_ctx, "par_posting_batches");
			MktMemCtx *batch_ctxs = mkt_alloc(nt * sizeof(MktMemCtx));
			for (uint32_t t = 0; t < nt; t++)
				batch_ctxs[t] =
						mkt_memctx_create(batch_parent, "par_posting_batch");
			mkt_memctx_switch(idx_ctx);

			/* Deferred mode: workers produce batch pages
			 * in memory (storage == NULL), leader
			 * materializes them to storage afterward. */
			for (uint32_t t = 0; t < nt; t++)
			{
				char *t_partials = partials ? partials + (size_t)t * nlist *
																 BLCKSZ
											: NULL;
				mkt_posting_worker_init(
						&workers[t],
						t,
						nlist,
						dim,
						config->fastscan != 0,
						NULL,
						idx->base.params,
						idx->leaf_centroids,
						idx->pt_centroids,
						NULL,
						t_partials,
						batch_ctxs[t]);
			}

			ParPostingCtx posting_ctx = {
					.workers	 = workers,
					.all_vectors = idx->all_vectors,
					.tree		 = tree,
					.bp			 = bp,
			};

			uint64_t t_posting = now_ns();
			mkt_thread_pool_parallel_for(
					pool, idx->nvecs, par_posting_fn, &posting_ctx);
			uint64_t t_parallel = now_ns() - t_posting;

			/* Gather per-thread outputs */
			for (uint32_t t = 0; t < nt; t++)
			{
				all_batches[t] = workers[t].batches;
				all_active[t]  = workers[t].active;
			}

			/* Create posting storage and materialize */
			idx->posting_storage = make_array_page_storage(1024, idx_ctx);
			idx->posting_heads	 = mkt_alloc(nlist * sizeof(BlockNumber));

			uint64_t			  t_merge_start = now_ns();
			MktPostingBuildResult build_result;
			mkt_posting_materialize(
					all_batches,
					partials,
					all_active,
					nt,
					nlist,
					&idx->posting_storage.base,
					idx->leaf_centroids,
					idx->pt_centroids,
					dim,
					config->fastscan != 0,
					&build_result);

			uint64_t t_merge = now_ns() - t_merge_start;

			memcpy(idx->posting_heads,
				   build_result.heads,
				   nlist * sizeof(BlockNumber));
			mkt_free(build_result.heads);

			ms_parallel	   = (double)t_parallel / 1e6;
			ms_merge	   = (double)t_merge / 1e6;
			stat_pages	   = build_result.total_pages;
			stat_merge_in  = build_result.merge_input;
			stat_merge_out = build_result.merge_output;

			for (uint32_t t = 0; t < nt; t++)
				mkt_posting_worker_cleanup(&workers[t]);
			mkt_memctx_delete(batch_parent); /* frees all worker batches */
			free(partials);

			if (config->fastscan)
				idx->base.fastscan = config->fastscan;
		}
		else
		{
			/* Flat mode needs per-cluster vector lists */
			assign_to_cluster_lists(idx, tree, &bp, dim, nlist);

			idx->flat_pages = mkt_alloc(nlist * sizeof(char *));

			for (uint32_t c = 0; c < nlist; c++)
			{
				MktClusterList *cl = &idx->clusters[c];
				const float *cent  = hk_leaf_centroids(tree) + (size_t)c * dim;

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
	else
	{
		/* No RaBitQ — still need cluster lists for brute-force */
		assign_to_cluster_lists(idx, tree, &bp, dim, nlist);
	}

	idx->has_replication = config->soar_lambda > 0.0 ||
						   config->boundary_epsilon > 0.0;

	/* --- Phase: centroid pages --- */
	double ms_posting = (double)(now_ns() - t_phase) / 1e6;
	t_phase			  = now_ns();
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

	double ms_centroid = (double)(now_ns() - t_phase) / 1e6;

	mkt_thread_pool_destroy(pool);
	mkt_free(tree);

	mkt_memctx_switch(old_ctx);
	mkt_memctx_delete(build_ctx);

	/* Fill build stats */
	if (stats != NULL)
	{
		stats->ms_sample   = ms_sample;
		stats->ms_kmeans   = ms_kmeans;
		stats->ms_setup	   = ms_setup;
		stats->ms_posting  = ms_posting;
		stats->ms_parallel = ms_parallel;
		stats->ms_merge	   = ms_merge;
		stats->ms_centroid = ms_centroid;
		stats->ms_total	   = ms_sample + ms_kmeans + ms_setup + ms_posting +
						  ms_centroid;
		stats->nworkers		= stat_nworkers;
		stats->total_pages	= stat_pages;
		stats->merge_input	= stat_merge_in;
		stats->merge_output = stat_merge_out;
	}

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
