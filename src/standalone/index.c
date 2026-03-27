/*
 * index.c - Standalone in-memory index build
 *
 * Builds centroid tree and per-cluster posting lists from a flat
 * vector array. Used by bindings and CLI benchmark.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "algo/distance.h"
#include "algo/hkmeans.h"
#include "algo/kmeans.h"
#include "algo/vecops.h"
#include "core/memory.h"
#include "index/centroid_build.h"
#include "index/centroid_page.h"
#include "index/index_build.h"
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
	return s->pages + (size_t)*blkno_out * BLCKSZ;
}

static void
aps_commit_page(MktStorage *self, BlockNumber blkno)
{
	(void)self;
	(void)blkno;
}

static const MktStorageOps array_page_storage_ops = {
		.read_page	  = aps_read_page,
		.release_page = aps_release_page,
		.write_page	  = aps_write_page,
		.new_page	  = aps_new_page,
		.commit_page  = aps_commit_page,
};

/* ----------------------------------------------------------------
 * Posting list helpers
 * ---------------------------------------------------------------- */

static void
posting_list_init(
		MktPostingList *pl, uint32_t initial_cap, Dimension dim, bool rabitq)
{
	pl->count	 = 0;
	pl->capacity = initial_cap;
	pl->ids		 = mkt_alloc(initial_cap * sizeof(uint32_t));
	pl->vectors	 = mkt_alloc((size_t)initial_cap * dim * sizeof(float));

	if (rabitq)
	{
		uint32_t packed = MKT_RABITQ_BYTES(dim);
		pl->f_add		= mkt_alloc(initial_cap * sizeof(float));
		pl->f_rescale	= mkt_alloc(initial_cap * sizeof(float));
		pl->f_error		= mkt_alloc(initial_cap * sizeof(float));
		pl->bits		= mkt_alloc((size_t)initial_cap * packed);
	}
	else
	{
		pl->f_add	  = NULL;
		pl->f_rescale = NULL;
		pl->f_error	  = NULL;
		pl->bits	  = NULL;
	}
}

static void
posting_list_append(
		MktPostingList *pl, uint32_t id, const float *vec, Dimension dim)
{
	if (pl->count == pl->capacity)
	{
		uint32_t old_cap = pl->capacity;
		uint32_t new_cap = old_cap * 2;

		pl->ids = arena_grow(
				pl->ids,
				old_cap * sizeof(uint32_t),
				new_cap * sizeof(uint32_t));
		pl->vectors = arena_grow(
				pl->vectors,
				(size_t)old_cap * dim * sizeof(float),
				(size_t)new_cap * dim * sizeof(float));

		if (pl->f_add != NULL)
		{
			uint32_t packed = MKT_RABITQ_BYTES(dim);
			pl->f_add		= arena_grow(
					  pl->f_add,
					  old_cap * sizeof(float),
					  new_cap * sizeof(float));
			pl->f_rescale = arena_grow(
					pl->f_rescale,
					old_cap * sizeof(float),
					new_cap * sizeof(float));
			pl->f_error = arena_grow(
					pl->f_error,
					old_cap * sizeof(float),
					new_cap * sizeof(float));
			pl->bits = arena_grow(
					pl->bits,
					(size_t)old_cap * packed,
					(size_t)new_cap * packed);
		}
		pl->capacity = new_cap;
	}

	pl->ids[pl->count] = id;
	memcpy(pl->vectors + (size_t)pl->count * dim, vec, dim * sizeof(float));
	pl->count++;
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

	MktIndex *idx	  = mkt_alloc0(sizeof(MktIndex));
	idx->memctx		  = idx_ctx;
	idx->dim		  = dim;
	idx->metric		  = config->metric;
	idx->centroid_fmt = config->centroid_fmt;

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
	if (idx->metric == DISTANCE_COSINE)
		normalize_all(samples, max_samples, dim);

	/* Run hierarchical k-means */
	KMeansOptions km_opts = MKT_KMEANS_OPTIONS_DEFAULT;
	if (config->km_nredo > 0)
		km_opts.nredo = config->km_nredo;
	if (config->km_max_iter > 0)
		km_opts.max_iterations = config->km_max_iter;

	HKMeansResult *tree = mkt_hkmeans_f32(
			samples, max_samples, dim, nlist, fan_out, idx->metric, &km_opts);

	/* Switch back to index context for long-lived allocations */
	mkt_memctx_switch(idx_ctx);

	if (tree == NULL)
	{
		mkt_memctx_switch(old_ctx);
		mkt_memctx_delete(idx_ctx); /* frees idx, build_ctx, everything */
		return NULL;
	}

	nlist		 = tree->nleaves;
	idx->nlist	 = nlist;
	idx->nlevels = (uint8_t)tree->nlevels;

	/* Normalize leaf centroids for cosine */
	if (idx->metric == DISTANCE_COSINE)
	{
		for (uint32_t c = 0; c < nlist; c++)
			normalize_vector(tree->leaf_centroids + (size_t)c * dim, dim);
	}

	/* Global mean */
	idx->global_mean = mkt_alloc(dim * sizeof(float));
	mkt_vector_mean(tree->leaf_centroids, nlist, dim, idx->global_mean);
	if (idx->metric == DISTANCE_COSINE)
		normalize_vector(idx->global_mean, dim);

	/* Save leaf centroids for per-cluster query preparation */
	idx->leaf_centroids = mkt_alloc((size_t)nlist * dim * sizeof(float));
	memcpy(idx->leaf_centroids,
		   tree->leaf_centroids,
		   (size_t)nlist * dim * sizeof(float));

	/* RaBitQ params */
	idx->rq_params = mkt_rabitq_create(dim, 42);

	/* Precompute P^T * centroids for zero-alloc query path.
	 * This turns per-cluster O(dim²) matrix multiply into O(dim)
	 * vector subtraction at query time. */
	idx->pt_centroids = mkt_alloc((size_t)nlist * dim * sizeof(float));
	for (uint32_t c = 0; c < nlist; c++)
		mkt_rabitq_rotate(
				idx->rq_params,
				idx->leaf_centroids + (size_t)c * dim,
				idx->pt_centroids + (size_t)c * dim);

	idx->pt_global_mean = mkt_alloc(dim * sizeof(float));
	mkt_rabitq_rotate(idx->rq_params, idx->global_mean, idx->pt_global_mean);

	/* Build centroid pages */
	uint32_t est_pages	  = nlist + 100;
	idx->centroid_storage = (ArrayPageStorage){
			.base		= {.ops = &array_page_storage_ops},
			.pages		= mkt_alloc0((size_t)est_pages * BLCKSZ),
			.next_blkno = 0,
			.page_cap	= est_pages,
	};

	uint32_t max_ent = mkt_centroid_max_entries_fmt(dim, idx->centroid_fmt);

	/* Temporary arrays for centroid page layout (build context) */
	mkt_memctx_switch(build_ctx);
	BlockNumber *node_first_blkno = mkt_alloc(
			tree->nnodes * sizeof(BlockNumber));
	idx->first_centroid = 0;
	mkt_compute_centroid_layout(
			tree, max_ent, idx->first_centroid, node_first_blkno);

	BlockNumber *leaf_heads = mkt_alloc(nlist * sizeof(BlockNumber));
	for (uint32_t i = 0; i < nlist; i++)
		leaf_heads[i] = (BlockNumber)i;

	mkt_write_centroid_tree(
			&idx->centroid_storage.base,
			tree,
			dim,
			fan_out,
			idx->centroid_fmt,
			idx->centroid_fmt == MKT_CENTROID_FMT_RABITQ ? idx->rq_params
														 : NULL,
			idx->global_mean,
			leaf_heads,
			node_first_blkno);

	/* Switch back to index context for posting lists */
	mkt_memctx_switch(idx_ctx);

	/* Initialize posting lists */
	uint32_t est_per_cluster = nvecs / nlist + 1;
	idx->lists				 = mkt_alloc0(nlist * sizeof(MktPostingList));
	for (uint32_t c = 0; c < nlist; c++)
		posting_list_init(
				&idx->lists[c], est_per_cluster, dim, config->encode_rabitq);

	/* Assign vectors to clusters.
	 * Stay in idx_ctx so posting list growth allocations are
	 * long-lived. Only norm_buf is temporary (freed with build_ctx). */
	float *norm_buf = NULL;
	if (idx->metric == DISTANCE_COSINE)
	{
		mkt_memctx_switch(build_ctx);
		norm_buf = mkt_alloc(dim * sizeof(float));
		mkt_memctx_switch(idx_ctx);
	}

	/* Pass 2: stream all vectors, route to clusters */
	src->reset(src);
	idx->nvecs = 0;

	{
		const float *vec;
		uint32_t	 id;
		while (src->next(src, 1, &vec, &id))
		{
			/* Normalize for cosine */
			if (idx->metric == DISTANCE_COSINE)
			{
				memcpy(norm_buf, vec, dim * sizeof(float));
				normalize_vector(norm_buf, dim);
				vec = norm_buf;
			}

			uint32_t c = mkt_hkmeans_assign(tree, vec, idx->metric, NULL);
			posting_list_append(&idx->lists[c], id, vec, dim);
			idx->nvecs++;
		}
	}

	/* Encode RaBitQ for posting lists if requested.
	 * Use build_ctx for temporary RaBitQData allocations. */
	mkt_memctx_switch(build_ctx);

	/* Encode RaBitQ for posting lists if requested */
	if (config->encode_rabitq)
	{
		for (uint32_t c = 0; c < nlist; c++)
		{
			MktPostingList *pl		 = &idx->lists[c];
			VectorRef		cent_ref = {
						  .data = tree->leaf_centroids + (size_t)c * dim,
						  .dim	= dim,
			  };

			for (uint32_t i = 0; i < pl->count; i++)
			{
				VectorRef vref = {
						.data = pl->vectors + (size_t)i * dim,
						.dim  = dim,
				};
				/* Encode into temp RaBitQData, extract SoA fields */
				uint32_t	data_sz = MKT_RABITQ_DATA_SIZE(dim);
				RaBitQData *tmp		= mkt_alloc(data_sz);
				mkt_rabitq_encode_into(idx->rq_params, vref, cent_ref, tmp);
				pl->f_add[i]	 = tmp->f_add;
				pl->f_rescale[i] = tmp->f_rescale;

				/* Derive f_error: C_error * sqrt(f_rescale² - f_add) */
				float f_rsq = tmp->f_rescale * tmp->f_rescale;
				if (f_rsq > tmp->f_add && dim > 1)
				{
					float c_err = 2.0f * MKT_RABITQ_EPSILON /
								  sqrtf((float)(dim - 1));
					pl->f_error[i] = c_err * sqrtf(f_rsq - tmp->f_add);
				}
				else
				{
					pl->f_error[i] = 2e-4f * sqrtf(tmp->f_add);
				}

				uint32_t packed = MKT_RABITQ_BYTES(dim);
				memcpy(pl->bits + (size_t)i * packed, tmp->bits, packed);
			}
		}
	}

	mkt_hkmeans_result_destroy(tree);

	mkt_memctx_switch(old_ctx);
	mkt_memctx_delete(build_ctx);

	return idx;
}

void
mkt_index_destroy(MktIndex *idx)
{
	if (idx == NULL)
		return;

	/* Deleting the memory context frees idx and all owned data. */
	mkt_memctx_delete(idx->memctx);
}
