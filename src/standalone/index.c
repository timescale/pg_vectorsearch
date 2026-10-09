/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * index.c - Standalone in-memory index build
 *
 * Builds centroid tree and per-cluster posting lists from a flat
 * vector array. Used by bindings and CLI benchmark.
 *
 * When RaBitQ encoding is enabled, posting data is written to pages
 * via PrismPostingBuilder. The full-precision vectors are stored in a
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
#include "core/log.h"
#include "core/memory.h"
#include "index/centroid_build.h"
#include "index/centroid_page.h"
#include "index/index_build.h"
#include "index/parallel_build.h"
#include "index/posting_build.h"
#include "index/posting_convert.h"
#include "index/posting_page.h"
#include "quant/rabitq.h"
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
	void *new_buf = vs_alloc(new_size);
	if (old != NULL && old_size > 0)
		memcpy(new_buf, old, old_size);
	return new_buf;
}

/* ----------------------------------------------------------------
 * ArrayPageStorage — page array backed by memory context
 * ---------------------------------------------------------------- */

static Page
aps_read_page(VsStorage *self, BlockNumber blkno)
{
	ArrayPageStorage *s = (ArrayPageStorage *)self;
	return s->pages + (size_t)blkno * BLCKSZ;
}

static void
aps_release_page(VsStorage *self, BlockNumber blkno)
{
	(void)self;
	(void)blkno;
}

static Page
aps_write_page(VsStorage *self, BlockNumber blkno)
{
	ArrayPageStorage *s = (ArrayPageStorage *)self;
	return s->pages + (size_t)blkno * BLCKSZ;
}

static Page
aps_new_page(VsStorage *self, BlockNumber *blkno_out)
{
	ArrayPageStorage *s = (ArrayPageStorage *)self;
	pthread_mutex_lock(&s->alloc_mutex);
	if (s->next_blkno >= s->page_cap)
	{
		VsMemCtx prev	 = vs_memctx_switch(s->memctx);
		uint32_t new_cap = s->page_cap * 2;
		s->pages		 = arena_grow(
				s->pages,
				(size_t)s->page_cap * BLCKSZ,
				(size_t)new_cap * BLCKSZ);
		s->page_cap = new_cap;
		vs_memctx_switch(prev);
	}
	*blkno_out = s->next_blkno++;
	pthread_mutex_unlock(&s->alloc_mutex);
	return s->pages + (size_t)*blkno_out * BLCKSZ;
}

static void
aps_commit_page(VsStorage *self, BlockNumber blkno)
{
	(void)self;
	(void)blkno;
}

static BlockNumber
aps_extend(VsStorage *self, uint32_t npages)
{
	ArrayPageStorage *s = (ArrayPageStorage *)self;
	pthread_mutex_lock(&s->alloc_mutex);

	BlockNumber start  = s->next_blkno;
	uint32_t	needed = start + npages;

	while (needed > s->page_cap)
	{
		VsMemCtx prev	 = vs_memctx_switch(s->memctx);
		uint32_t new_cap = s->page_cap * 2;
		if (new_cap < needed)
			new_cap = needed;
		s->pages = arena_grow(
				s->pages,
				(size_t)s->page_cap * BLCKSZ,
				(size_t)new_cap * BLCKSZ);
		s->page_cap = new_cap;
		vs_memctx_switch(prev);
	}

	s->next_blkno = needed;
	pthread_mutex_unlock(&s->alloc_mutex);
	return start;
}

static uint32_t
aps_rerank(
		VsStorage		  *self,
		const float		  *query,
		Dimension		   dim,
		const VsTopKEntry *candidates,
		uint32_t		   count,
		uint32_t		   keep,
		uint32_t		  *out_indices,
		Distance		  *out_distances)
{
	ArrayPageStorage *s = (ArrayPageStorage *)self;

	if (s->all_vectors == NULL || count == 0)
		return 0;

	vs_topk_reset_to_k(&s->rerank_topk, keep);

	for (uint32_t i = 0; i < count; i++)
	{
		Distance d;
		if (candidates[i].error == 0.0f)
		{
			d = candidates[i].distance;
		}
		else
		{
			uint32_t vid = prism_posting_decode_vector_id(candidates[i].id);
			if (vid < s->nvecs)
			{
				const float *vec = s->all_vectors + (size_t)vid * dim;
				d				 = vs_l2_distance_squared(query, vec, dim);
			}
			else
			{
				d = candidates[i].distance;
			}
		}

		vs_topk_insert(&s->rerank_topk, d, 0.0f, (uint64_t)i);
	}

	uint32_t nresults;
	vs_topk_extract_sorted(&s->rerank_topk, s->rerank_entries, &nresults);
	if (nresults > s->rerank_cap)
		nresults = s->rerank_cap;

	for (uint32_t i = 0; i < nresults; i++)
	{
		out_indices[i]	 = (uint32_t)s->rerank_entries[i].id;
		out_distances[i] = s->rerank_entries[i].distance;
	}

	return nresults;
}

static const VsStorageOps array_page_storage_ops = {
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
cluster_list_init(PrismClusterList *cl, uint32_t initial_cap)
{
	cl->count	 = 0;
	cl->capacity = initial_cap;
	cl->ids		 = vs_alloc(initial_cap * sizeof(uint32_t));
}

static void
cluster_list_append(PrismClusterList *cl, uint32_t id)
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
make_array_page_storage(uint32_t est_pages, VsMemCtx memctx)
{
	/*
	 * No zero-fill: every page is fully overwritten on its first write
	 * (flush copies the whole BLCKSZ buffer), and reserved/spill pages
	 * that no worker writes are never linked into a chain, so they are
	 * never read. Zeroing the whole pre-reserved array (hundreds of MB
	 * for large builds) would be pure overhead.
	 */
	ArrayPageStorage s = {
			.base		= {.ops = &array_page_storage_ops},
			.pages		= vs_alloc((size_t)est_pages * BLCKSZ),
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
	float norm = vs_l2_norm(v, dim);
	if (norm > 0.0f)
		vec32_scale(v, 1.0f / norm, v, dim);
}

static void
normalize_all(float *data, uint32_t nvecs, Dimension dim)
{
	for (uint32_t i = 0; i < nvecs; i++)
		normalize_vector(data + (size_t)i * dim, dim);
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
		PrismIndex				*idx,
		const HKMeansResult		*tree,
		const PrismAssignParams *bp,
		Dimension				 dim,
		uint32_t				 nlist)
{
	PrismBuildWorkerBufs bufs = prism_build_worker_bufs_create(dim);

	for (uint32_t i = 0; i < idx->nvecs; i++)
	{
		const float			*vec = idx->all_vectors + (size_t)i * dim;
		PrismBuildAssignment asgn =
				prism_build_assign_vector(tree, vec, bp, &bufs);

		cluster_list_append(&idx->clusters[asgn.primary], i);
		if (asgn.secondary != PRISM_INVALID_CLUSTER)
			cluster_list_append(&idx->clusters[asgn.secondary], i);
	}

	prism_build_worker_bufs_free(&bufs);

	idx->max_cluster_size = 0;
	for (uint32_t c = 0; c < nlist; c++)
		if (idx->clusters[c].count > idx->max_cluster_size)
			idx->max_cluster_size = idx->clusters[c].count;
}

/* ----------------------------------------------------------------
 * Build
 * ---------------------------------------------------------------- */

PrismIndex *
prism_index_build(
		Vec32Source			   *src,
		const PrismIndexConfig *config,
		PrismBuildStats		   *stats)
{
	/* Each path below writes only the phases it ran, and ms_refine has no
	 * standalone producer at all, while callers read the struct whole. */
	if (stats != NULL)
		memset(stats, 0, sizeof(*stats));

	if (src == NULL || src->nvecs == 0 || src->dim == 0 || config == NULL)
		return NULL;
	if (src->dim > PRISM_INDEX_MAX_DIM)
	{
		vs_warn("index build: %u dimensions exceeds the layout ceiling %u",
				(unsigned)src->dim,
				(unsigned)PRISM_INDEX_MAX_DIM);
		return NULL;
	}

	uint32_t  nvecs = src->nvecs;
	Dimension dim	= src->dim;

	vs_distance_init();

	/* Resolve the worker count for the parallel driver (paged builds). */
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

	/* Long-lived context for index data. Build-phase temporaries
	 * go into a child context that gets deleted after build. */
	VsMemCtx idx_ctx   = vs_memctx_create(NULL, "index");
	VsMemCtx build_ctx = vs_memctx_create(idx_ctx, "index_build");
	VsMemCtx old_ctx   = vs_memctx_switch(idx_ctx);

	PrismIndex *idx			  = vs_alloc0(sizeof(PrismIndex));
	idx->memctx				  = idx_ctx;
	idx->base.dim			  = dim;
	idx->base.metric		  = config->metric;
	idx->base.centroid_format = config->centroid_fmt;

	/* Resolve nlist */
	uint32_t nlist = config->nlist;
	if (nlist == 0)
		nlist = prism_auto_nlist((double)nvecs, dim, config->target_pages);

	/* Resolve fan_out */
	uint32_t fan_out = config->fan_out;
	if (fan_out == 0)
		fan_out = prism_auto_fan_out(0, nlist, 0);
	idx->fan_out = fan_out;

	/* --- Phase: load vectors --- */
	uint64_t t_phase = now_ns();

	idx->all_vectors = vs_alloc((size_t)nvecs * dim * sizeof(float));

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

	/*
	 * Paged RaBitQ builds with at least one worker run the shared parallel
	 * build driver (sampling + k-means + bounded streaming posting) — the same
	 * code path the PostgreSQL extension uses. Flat and serial builds keep the
	 * in-memory path below.
	 */
	bool use_driver = config->encode_rabitq &&
					  config->posting_fmt == PRISM_POSTING_FMT_PAGES;

	HKMeansResult *tree			  = NULL;
	uint32_t	   drv_nlist	  = 0; /* driver path: streamed tree shape */
	uint8_t		   drv_nlevels	  = 0;
	float		  *km_vectors	  = NULL;
	uint32_t	   km_nvecs		  = 0;
	double		   ms_kmeans	  = 0;
	float		  *sa_global_mean = NULL; /* driver's leaf-centroid mean */

	if (use_driver)
	{
		t_phase = now_ns();

		/* The driver produces fastscan-packed pages when requested; the query
		 * path must know to read them as fastscan (not AoS). */
		idx->base.fastscan = config->fastscan != 0;

		/* Upper bound on leaves (fan_out^nlevels, matching the tree the driver
		 * builds). */
		uint32_t max_nlist = prism_max_nlist(nlist, fan_out);

		/*
		 * Long-lived posting storage for the driver's streamed pages. The
		 * driver reserves a centroid region at the front (left unwritten here
		 * — the centroid pages go to centroid_storage below) and writes
		 * posting pages after it. Head blocks are formula-derived
		 * (first_posting + leaf), so no head array is needed.
		 */
		uint32_t est_pages	 = idx->nvecs / 4 + max_nlist + 256;
		idx->posting_storage = make_array_page_storage(est_pages, idx_ctx);

		RelationData heap_rel = {
				.vectors = idx->all_vectors,
				.nvecs	 = idx->nvecs,
				.dim	 = dim,
		};
		RelationData index_rel = {
				.page_count = &idx->posting_storage.next_blkno,
		};
		/* The driver streams worker→leader, so it needs at least one worker;
		 * a serial (nworkers==0) PAGES build runs through the driver with one.
		 */
		IndexInfo index_info = {
				.ii_ParallelWorkers = (int)(nworkers > 0 ? nworkers : 1)};
		PrismBuildConfig cfg = {
				.dim			 = dim,
				.metric			 = config->metric,
				.centroid_format = idx->base.centroid_format,
				/* The tree expands to up to fan_out^nlevels leaves; size the
				 * shared regions for that bound (matches the PG caller). */
				.nlist			  = max_nlist,
				.fan_out		  = fan_out,
				.soar_lambda	  = config->soar_lambda,
				.boundary_epsilon = config->boundary_epsilon,
				.fastscan		  = config->fastscan != 0,
		};

		double heap_tuples = 0, indtuples = 0, soar_dupes = 0;

		/* Build temporaries (samples, accumulators, the tree) live in
		 * build_ctx and are freed after the build; page growth uses the
		 * storage's own (idx_ctx) context, so the index pages outlive it. */
		vs_memctx_switch(build_ctx);
		bool ok = do_parallel_build(
				&heap_rel,
				&index_rel,
				&index_info,
				&cfg,
				&idx->posting_storage.base,
				NULL, /* no build-progress seam in standalone (no-op stub) */
				&drv_nlist,
				&drv_nlevels,
				&heap_tuples,
				&indtuples,
				&soar_dupes,
				/* The driver streams the centroid tree into posting_storage
				 * (no in-RAM tree). Capture the leaf-centroid mean it encoded
				 * against so the page-backed query centering matches. */
				&sa_global_mean,
				&idx->first_posting);
		vs_memctx_switch(idx_ctx);

		/* The driver always launches at least one worker, so it does not fail
		 * here; on the off chance it does, the empty guard below returns. */
		if (!ok)
			drv_nlist = 0;

		ms_kmeans = (double)(now_ns() - t_phase) / 1e6;
	}

	if (!use_driver)
	{
		/* --- Phase: kmeans (serial; FLAT/no-RaBitQ do not need parallel)
		 * --- */
		t_phase = now_ns();
		vs_memctx_switch(build_ctx);

		/* Subsample by stride into a contiguous buffer for cache-friendly
		 * k-means iteration. */
		uint32_t max_samples = idx->nvecs < 256000 ? idx->nvecs : 256000;
		uint32_t stride		 = max_samples > 0 ? idx->nvecs / max_samples : 1;
		if (stride < 1)
			stride = 1;
		km_nvecs = (stride > 1) ? max_samples : idx->nvecs;

		if (stride > 1)
		{
			km_vectors = vs_alloc((size_t)km_nvecs * dim * sizeof(float));
			for (uint32_t i = 0; i < km_nvecs; i++)
				memcpy(km_vectors + (size_t)i * dim,
					   idx->all_vectors + (size_t)i * stride * dim,
					   dim * sizeof(float));
		}
		else
		{
			km_vectors = idx->all_vectors;
		}

		KMeansOptions km_opts = VS_KMEANS_OPTIONS_DEFAULT;
		if (config->km_nredo > 0)
			km_opts.nredo = config->km_nredo;
		if (config->km_max_iter > 0)
			km_opts.max_iterations = config->km_max_iter;

		tree = vs_hkmeans_f32(
				km_vectors,
				km_nvecs,
				NULL,
				dim,
				nlist,
				fan_out,
				idx->base.metric,
				&km_opts);

		ms_kmeans = (double)(now_ns() - t_phase) / 1e6;
	} /* end serial in-memory k-means */

	/* --- Phase: setup --- */
	t_phase = now_ns();

	vs_memctx_switch(idx_ctx);

	if (use_driver ? drv_nlist == 0 : tree == NULL)
	{
		vs_memctx_switch(old_ctx);
		vs_memctx_delete(idx_ctx); /* frees idx, build_ctx, everything */
		return NULL;
	}

	if (use_driver)
	{
		/* Paged parallel build: do_parallel_build streamed the whole centroid
		 * tree + posting heads into posting_storage (no in-RAM tree). The
		 * query routes page-backed over those pages via idx->base, so skip the
		 * tree-centroid finalize entirely (the returned tree is only a carrier
		 * of leaf/level counts). */
		nlist			  = drv_nlist;
		idx->nlist		  = nlist;
		idx->base.nlevels = drv_nlevels;
		idx->base.nlist	  = nlist;
		idx->base.fan_out = (uint8_t)(fan_out <= UINT8_MAX ? fan_out
														   : UINT8_MAX);
		/* Query knobs: mirror the PG GUC defaults so the paged beam search
		 * behaves identically in both engines (a zero beam scale would
		 * collapse the beam to width 1 and skip the coverage floors). */
		idx->base.centroid_error_scale = 0.0f;
		idx->base.centroid_beam_scale  = 0.5f;
		/* The streaming build reserves block 0 for the meta page and writes
		 * the root centroid page in place at block 1, after its subtrees. */
		idx->base.first_centroid = 1;
		idx->base.params		 = vs_rabitq_create(dim, VS_RABITQ_BUILD_SEED);
		/* The paged query needs no per-cluster lists, but the bindings API
		 * reads idx->clusters[c].count for build stats; give it zeroed entries
		 * (count 0 -> the API falls back to size estimates). */
		idx->clusters = vs_alloc0((size_t)nlist * sizeof(PrismClusterList));
		idx->base.pt_global_mean = vs_alloc(dim * sizeof(float));
		if (sa_global_mean != NULL)
			vs_rabitq_rotate(
					idx->base.params,
					sa_global_mean,
					idx->base.pt_global_mean);
		else
			memset(idx->base.pt_global_mean, 0, (size_t)dim * sizeof(float));
		idx->has_posting_data = true;
		idx->posting_fmt	  = config->posting_fmt;

		vs_memctx_switch(old_ctx);
		vs_memctx_delete(build_ctx);

		if (stats != NULL)
		{
			stats->ms_sample = ms_sample;
			stats->ms_kmeans = ms_kmeans;
			stats->ms_total	 = ms_sample + ms_kmeans;
			stats->nworkers	 = nworkers;
		}

		idx->base.centroid_storage = &idx->posting_storage.base;
		idx->base.posting_storage  = &idx->posting_storage.base;
		idx->base.page_base		   = idx->posting_storage.pages;

		idx->posting_storage.all_vectors = idx->all_vectors;
		idx->posting_storage.nvecs		 = idx->nvecs;
		idx->posting_storage.metric		 = idx->base.metric;
		uint32_t rerank_cap				 = 256;
		vs_topk_init(&idx->posting_storage.rerank_topk, rerank_cap);
		idx->posting_storage.rerank_entries = vs_alloc(
				rerank_cap * sizeof(VsTopKEntry));
		idx->posting_storage.rerank_cap = rerank_cap;

		return idx;
	}

	nlist			  = tree->nleaves;
	idx->nlist		  = nlist;
	idx->base.nlevels = (uint8_t)tree->nlevels;
	idx->base.fan_out = (uint8_t)(fan_out <= UINT8_MAX ? fan_out : UINT8_MAX);
	idx->base.nlist	  = nlist;

	/* Normalize leaf centroids for cosine */
	if (idx->base.metric == DISTANCE_COSINE)
	{
		for (uint32_t c = 0; c < nlist; c++)
			normalize_vector(hk_leaf_centroids(tree) + (size_t)c * dim, dim);
	}

	/* Global mean */
	idx->global_mean = vs_alloc(dim * sizeof(float));
	vs_global_mean(
			hk_leaf_centroids(tree),
			nlist,
			dim,
			idx->base.metric,
			idx->global_mean);

	/* Save leaf centroids for per-cluster query preparation */
	idx->leaf_centroids = vs_alloc((size_t)nlist * dim * sizeof(float));
	memcpy(idx->leaf_centroids,
		   hk_leaf_centroids(tree),
		   (size_t)nlist * dim * sizeof(float));

	/* RaBitQ params */
	idx->base.params = vs_rabitq_create(dim, VS_RABITQ_BUILD_SEED);

	/* Precompute P^T * centroids for zero-alloc query path. */
	idx->pt_centroids = vs_alloc((size_t)nlist * dim * sizeof(float));
	for (uint32_t c = 0; c < nlist; c++)
		vs_rabitq_rotate(
				idx->base.params,
				idx->leaf_centroids + (size_t)c * dim,
				idx->pt_centroids + (size_t)c * dim);

	idx->base.pt_global_mean = vs_alloc(dim * sizeof(float));
	vs_rabitq_rotate(
			idx->base.params, idx->global_mean, idx->base.pt_global_mean);

	/* Build centroid pages */
	uint32_t est_centroid_pages = nlist + 100;
	idx->centroid_storage =
			make_array_page_storage(est_centroid_pages, idx_ctx);

	uint32_t max_ent =
			prism_centroid_max_entries_fmt(dim, idx->base.centroid_format);

	/* Temporary arrays for centroid page layout (build context) */
	vs_memctx_switch(build_ctx);
	BlockNumber *node_first_blkno = vs_alloc(
			tree->nnodes * sizeof(BlockNumber));
	idx->base.first_centroid = 0;
	prism_compute_centroid_layout(
			tree, max_ent, idx->base.first_centroid, node_first_blkno);

	/* Switch back to index context for posting data.
	 * Posting lists are built before centroid pages so that
	 * centroid leaf entries store actual posting block numbers. */
	vs_memctx_switch(idx_ctx);

	/* Initialize per-cluster ID lists */
	uint32_t est_per_cluster = nlist > 0 ? nvecs / nlist + 1 : 1;
	idx->clusters			 = vs_alloc0(nlist * sizeof(PrismClusterList));
	for (uint32_t c = 0; c < nlist; c++)
		cluster_list_init(&idx->clusters[c], est_per_cluster);

	double					ms_setup = (double)(now_ns() - t_phase) / 1e6;
	const PrismAssignParams bp		 = {
				  .dim				= dim,
				  .metric			= config->metric,
				  .soar_lambda		= config->soar_lambda,
				  .boundary_epsilon = config->boundary_epsilon,
	  };

	/* --- Phase: posting --- */
	t_phase = now_ns();

	double	 ms_parallel = 0, ms_merge = 0;
	uint32_t stat_nworkers = nworkers, stat_pages = 0;
	uint32_t stat_merge_in = 0, stat_merge_out = 0;
	if (config->encode_rabitq)
	{
		idx->posting_fmt = config->posting_fmt;

		/* The driver path returned above, so this is always the flat
		 * mode, which needs per-cluster vector lists. */
		assign_to_cluster_lists(idx, tree, &bp, dim, nlist);

		idx->flat_pages = vs_alloc(nlist * sizeof(char *));

		for (uint32_t c = 0; c < nlist; c++)
		{
			PrismClusterList *cl   = &idx->clusters[c];
			const float		 *cent = hk_leaf_centroids(tree) + (size_t)c * dim;

			PrismFlatPostingBuilder builder;
			prism_flat_posting_builder_init(
					&builder, idx->base.params, dim, c, cent, cl->count);

			for (uint32_t i = 0; i < cl->count; i++)
			{
				uint32_t		vid = cl->ids[i];
				const float	   *vec = idx->all_vectors + (size_t)vid * dim;
				ItemPointerData tid;
				prism_posting_set_vector_id(&tid, vid);
				prism_flat_posting_builder_add(&builder, tid, vec);
			}

			idx->flat_pages[c] = prism_flat_posting_builder_finish(&builder);
			prism_flat_posting_builder_cleanup(&builder);
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
		/* This (non-driver) path stores cluster indices as leaf child blocks;
		 * the inline flat query maps them to per-cluster lists. Leaf j's child
		 * is posting_base + global_leaf_index, so posting_base = 0 yields the
		 * leaf (cluster) index directly. The paged build takes the use_driver
		 * path, which writes the centroid tree itself and returns above. */
		vs_memctx_switch(build_ctx);
		bool needs_rq_params =
				(idx->base.centroid_format == PRISM_CENTROID_FMT_RABITQ ||
				 idx->base.centroid_format == PRISM_CENTROID_FMT_FASTSCAN);
		prism_write_centroid_tree(
				&idx->centroid_storage.base,
				tree,
				dim,
				fan_out,
				0, /* in-RAM tree levels are absolute */
				idx->base.centroid_format,
				needs_rq_params ? idx->base.params : NULL,
				idx->global_mean,
				0, /* posting_base: leaf child = global leaf (cluster) index */
				node_first_blkno,
				NULL,  /* pt_centroids on posting pages, not here */
				NULL); /* in-RAM assignment descends the exact float tree */
		vs_memctx_switch(idx_ctx);
	}

	double ms_centroid = (double)(now_ns() - t_phase) / 1e6;

	vs_free(tree);

	vs_memctx_switch(old_ctx);
	vs_memctx_delete(build_ctx);

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
	vs_topk_init(&idx->posting_storage.rerank_topk, rerank_cap);
	idx->posting_storage.rerank_entries = vs_alloc(
			rerank_cap * sizeof(VsTopKEntry));
	idx->posting_storage.rerank_cap = rerank_cap;

	return idx;
}

void
prism_index_destroy(PrismIndex *idx)
{
	if (idx == NULL)
		return;

	vs_topk_cleanup(&idx->posting_storage.rerank_topk);

	/* Deleting the memory context frees idx and all owned data. */
	vs_memctx_delete(idx->memctx);
}
