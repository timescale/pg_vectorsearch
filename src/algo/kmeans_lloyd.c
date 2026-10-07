/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * kmeans_lloyd.c - Lloyd's k-means assignment step
 *
 * Brute-force assignment: compute the full N×K distance matrix each
 * iteration. Two backends share the same block-processing structure:
 *
 * CBLAS path:
 *   Uses sgemm for the [block × dim] × [dim × nlist] dot product
 *   matrix. BLAS cache-tiling makes this 10-50x faster than naive
 *   loops for large K.
 *
 * Builtin path:
 *   Batch dot-product kernel with VS_TARGET_CLONES for AVX-512/AVX2
 *   auto-vectorization. FMA generation requires -ffp-contract=fast
 *   (set in meson.build).
 *
 * Both use the decomposition:
 *   ||x - c||² = ||x||² + ||c||² - 2⟨x,c⟩
 * The ⟨x,c⟩ term is computed in batch, norms are precomputed.
 */

#include "vs_config.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef VS_HAVE_CBLAS
/* See matrix.c for the rationale on the Apple branch. */
#ifdef __APPLE__
#include <vecLib/cblas_new.h>
#else
#include <cblas.h>
#endif
#endif

#include "algo/kmeans_lloyd.h"
#include "algo/vecops.h"
#include "core/log.h"
#include "core/memory.h"
#include "types/vec16.h"

/*
 * Precompute ||c||² for all centroids (L2 only).
 */
static void
precompute_norms_c(KMeansState *st)
{
	for (uint32_t j = 0; j < st->nlist; j++)
		st->norms_c[j] = vs_l2_norm_squared(
				st->centroids + (size_t)j * st->dim, st->dim);
}

/*
 * Assignment step: CBLAS path for one block.
 *
 * Computes the N_block x K distance matrix using sgemm, then finds
 * the nearest centroid per vector (argmin per row).
 *
 * For L2:  dist[i][j] = ||x_i||² + ||c_j||² - 2⟨x_i, c_j⟩
 * For IP:  dist[i][j] = -⟨x_i, c_j⟩
 * For cos: dist[i][j] = 1 - ⟨x_i, c_j⟩
 */
#ifdef VS_HAVE_CBLAS
__attribute__((always_inline)) static inline void
lloyd_assign_block_cblas_impl(
		KMeansState		   *st,
		uint32_t			block_start,
		uint32_t			block_count,
		const Vec32TypeOps *ops)
{
	uint32_t nlist = st->nlist;
	uint32_t dim   = st->dim;
	float	*dist  = st->dist_block;

	/* Get float32 view of this block */
	const float *block_vecs;
	if (st->indices != NULL)
	{
		/* Gather indexed vectors into vec_block */
		size_t esz = ops->element_size;
		for (uint32_t i = 0; i < block_count; i++)
		{
			uint32_t	idx = st->indices[block_start + i];
			const void *src = (const char *)st->vectors +
							  (size_t)idx * dim * esz;
			/* Unreachable
			 * unless the standalone arena is exhausted, which no
			 * caller handles. */
			/* NOLINTNEXTLINE(clang-analyzer-core.NullPointerArithm) */
			ops->to_float_one(src, st->vec_block + (size_t)i * dim, dim);
		}
		block_vecs = st->vec_block;
	}
	else
	{
		/* Zero-copy for f32, converts for f16 */
		const void *raw = (const char *)st->vectors +
						  (size_t)block_start * dim * ops->element_size;
		block_vecs = ops->to_float_block(raw, st->vec_block, block_count, dim);
	}

	/* Compute dot products via sgemm */
	float alpha = -2.0f;
	if (st->metric == DISTANCE_INNER_PRODUCT || st->metric == DISTANCE_COSINE)
		alpha = -1.0f;

	cblas_sgemm(
			CblasRowMajor,
			CblasNoTrans,
			CblasTrans,
			(int)block_count, /* M: rows */
			(int)nlist,		  /* N: cols */
			(int)dim,		  /* K: inner dim */
			alpha,
			block_vecs,
			(int)dim,
			st->centroids,
			(int)dim,
			0.0f,
			dist,
			(int)nlist);

	/* Add norms / offset */
	switch (st->metric)
	{
	case DISTANCE_L2:
		for (uint32_t i = 0; i < block_count; i++)
		{
			float nx = st->norms_x[block_start + i];
			for (uint32_t j = 0; j < nlist; j++)
				dist[(size_t)i * nlist + j] += nx + st->norms_c[j];
		}
		break;
	case DISTANCE_COSINE:
		for (uint32_t i = 0; i < block_count; i++)
			for (uint32_t j = 0; j < nlist; j++)
				dist[(size_t)i * nlist + j] += 1.0f;
		break;
	case DISTANCE_INNER_PRODUCT:
		/* dist = -⟨x,c⟩, already correct */
		break;
	}

	/* Find argmin per row */
	for (uint32_t i = 0; i < block_count; i++)
	{
		uint32_t vec_idx  = block_start + i;
		float	 min_dist = dist[(size_t)i * nlist];
		uint32_t min_j	  = 0;

		for (uint32_t j = 1; j < nlist; j++)
		{
			float d = dist[(size_t)i * nlist + j];
			if (d < min_dist)
			{
				min_dist = d;
				min_j	 = j;
			}
		}

		st->assignments[vec_idx] = min_j;
		st->total_cost += min_dist;
	}
}

static void
lloyd_assign_block_cblas_f32(
		KMeansState *st, uint32_t block_start, uint32_t block_count)
{
	lloyd_assign_block_cblas_impl(
			st, block_start, block_count, &vs_f32_type_ops);
}

static void
lloyd_assign_block_cblas_f16(
		KMeansState *st, uint32_t block_start, uint32_t block_count)
{
	lloyd_assign_block_cblas_impl(
			st, block_start, block_count, &vs_f16_type_ops);
}

#if defined(VS_F16C_SUPPORT) && !defined(VS_SIMD_NONE)
static void
lloyd_assign_block_cblas_f16c(
		KMeansState *st, uint32_t block_start, uint32_t block_count)
{
	lloyd_assign_block_cblas_impl(
			st, block_start, block_count, &vs_f16c_type_ops);
}
#endif
#endif /* VS_HAVE_CBLAS */

/*
 * Batch dot-product matrix: dots[i*nlist + j] = dot(vecs[i], cents[j])
 *
 * VS_TARGET_CLONES generates AVX-512, AVX2, and default versions.
 * The innermost loop over dim is auto-vectorized by the compiler,
 * eliminating per-vector function pointer dispatch overhead.
 *
 * FMA generation requires -ffp-contract=fast (set in meson.build).
 * Without it, GCC in an ISO C mode generates separate vmulps +
 * horizontal scalar adds instead of vfmadd231ps accumulate.
 */
VS_TARGET_CLONES static void
lloyd_compute_dot_products(
		const float *vecs,
		const float *centroids,
		float		*dots,
		uint32_t	 block_count,
		uint32_t	 nlist,
		uint32_t	 dim)
{
	for (uint32_t i = 0; i < block_count; i++)
	{
		const float *v = vecs + (size_t)i * dim;
		for (uint32_t j = 0; j < nlist; j++)
		{
			const float *c	 = centroids + (size_t)j * dim;
			float		 dot = 0.0f;
			for (uint32_t d = 0; d < dim; d++)
				/* Unreachable
				 * unless the standalone arena is exhausted, which no
				 * caller handles. */
				/* NOLINTNEXTLINE(clang-analyzer-core.NullDereference) */
				dot += v[d] * c[d];
			dots[(size_t)i * nlist + j] = dot;
		}
	}
}

/*
 * Convert dot products to distances and find argmin per row.
 */
static void
lloyd_dots_to_assignments(
		KMeansState *st,
		float		*dist,
		uint32_t	 block_start,
		uint32_t	 block_count)
{
	uint32_t nlist = st->nlist;

	/* Convert dot products to distances based on metric */
	switch (st->metric)
	{
	case DISTANCE_L2:
		/* dist = ||x||² + ||c||² - 2*dot(x,c) */
		for (uint32_t i = 0; i < block_count; i++)
		{
			float nx = st->norms_x[block_start + i];
			for (uint32_t j = 0; j < nlist; j++)
			{
				size_t idx = (size_t)i * nlist + j;
				dist[idx]  = nx + st->norms_c[j] - 2.0f * dist[idx];
			}
		}
		break;
	case DISTANCE_INNER_PRODUCT:
		/* dist = -dot(x,c) */
		for (uint32_t i = 0; i < block_count; i++)
			for (uint32_t j = 0; j < nlist; j++)
			{
				size_t idx = (size_t)i * nlist + j;
				dist[idx]  = -dist[idx];
			}
		break;
	case DISTANCE_COSINE:
		/* dist = 1 - dot(x,c)  (vectors pre-normalized) */
		for (uint32_t i = 0; i < block_count; i++)
			for (uint32_t j = 0; j < nlist; j++)
			{
				size_t idx = (size_t)i * nlist + j;
				dist[idx]  = 1.0f - dist[idx];
			}
		break;
	}

	/* Find argmin per row */
	for (uint32_t i = 0; i < block_count; i++)
	{
		uint32_t vec_idx  = block_start + i;
		float	 min_dist = dist[(size_t)i * nlist];
		uint32_t min_j	  = 0;

		for (uint32_t j = 1; j < nlist; j++)
		{
			float d = dist[(size_t)i * nlist + j];
			if (d < min_dist)
			{
				min_dist = d;
				min_j	 = j;
			}
		}

		st->assignments[vec_idx] = min_j;
		st->total_cost += min_dist;
	}
}

/*
 * Assignment step: builtin fallback for one block.
 *
 * For f32 input: uses batch dot-product + norm decomposition.
 * For f16 input: preconverts block to f32, then uses the same f32 kernel.
 */
static void
lloyd_assign_block_builtin_f32(
		KMeansState *st, uint32_t block_start, uint32_t block_count)
{
	uint32_t nlist = st->nlist;
	uint32_t dim   = st->dim;
	float	*dist  = st->dist_block;

	const float *block_vecs;
	if (st->indices != NULL)
	{
		/* Gather indexed vectors into contiguous block */
		for (uint32_t i = 0; i < block_count; i++)
		{
			uint32_t idx = st->indices[block_start + i];
			/* Unreachable unless the standalone arena is exhausted, which
			 * no caller handles. */
			/* NOLINTNEXTLINE(clang-analyzer-core.NonNullParamChecker) */
			memcpy(st->vec_block + (size_t)i * dim,
				   (const float *)st->vectors + (size_t)idx * dim,
				   dim * sizeof(float));
		}
		block_vecs = st->vec_block;
	}
	else
	{
		block_vecs = (const float *)st->vectors + (size_t)block_start * dim;
	}

	lloyd_compute_dot_products(
			block_vecs, st->centroids, dist, block_count, nlist, dim);
	lloyd_dots_to_assignments(st, dist, block_start, block_count);
}

static void
lloyd_assign_block_builtin_f16(
		KMeansState *st, uint32_t block_start, uint32_t block_count)
{
	uint32_t nlist = st->nlist;
	uint32_t dim   = st->dim;

	/* Preconvert f16 block to f32 — O(block*dim), saves O(block*K*dim) */
	if (st->indices != NULL)
	{
		for (uint32_t i = 0; i < block_count; i++)
		{
			uint32_t	idx = st->indices[block_start + i];
			const half *src = (const half *)st->vectors + (size_t)idx * dim;
			/* Unreachable
			 * unless the standalone arena is exhausted, which no
			 * caller handles. */
			/* NOLINTNEXTLINE(clang-analyzer-core.NullPointerArithm) */
			vs_half_to_float_array(src, st->vec_block + (size_t)i * dim, dim);
		}
	}
	else
	{
		const half *src = (const half *)st->vectors +
						  (size_t)block_start * dim;
		vs_half_to_float_array(src, st->vec_block, block_count * dim);
	}

	lloyd_compute_dot_products(
			st->vec_block,
			st->centroids,
			st->dist_block,
			block_count,
			nlist,
			dim);
	lloyd_dots_to_assignments(st, st->dist_block, block_start, block_count);
}

/* Block assignment function pointer — selected once per lloyd_assign call */
typedef void (*lloyd_block_fn)(KMeansState *, uint32_t, uint32_t);

static lloyd_block_fn
lloyd_select_block_fn(KMeansState *st, bool use_cblas)
{
#ifdef VS_HAVE_CBLAS
	if (use_cblas)
	{
		switch (st->vec_type)
		{
		case VS_VEC_F32:
			return lloyd_assign_block_cblas_f32;
#if defined(VS_F16C_SUPPORT) && !defined(VS_SIMD_NONE)
		case VS_VEC_F16C:
			return lloyd_assign_block_cblas_f16c;
#endif
		default:
			return lloyd_assign_block_cblas_f16;
		}
	}
#endif
	(void)use_cblas;
	switch (st->vec_type)
	{
	case VS_VEC_F32:
		return lloyd_assign_block_builtin_f32;
	default:
		return lloyd_assign_block_builtin_f16;
	}
}

/*
 * Adaptive block size for the parallel builtin path.
 *
 * The distance buffer is [block × nlist] floats per thread. With
 * fixed block=4096, large nlist blows past L3 (e.g., 4096×8000×4
 * = 128MB per thread). Cap the buffer at ~2MB so the distance
 * matrix, vector block, and centroid tile all fit in L3.
 */
static uint32_t
lloyd_parallel_block_size(uint32_t nlist, uint32_t nvecs)
{
	uint32_t block	= KMEANS_BLOCK_SIZE;
	uint32_t buf_sz = block * nlist * (uint32_t)sizeof(float);
	if (buf_sz > 8 * 1024 * 1024)
	{
		block = 8 * 1024 * 1024 / (nlist * (uint32_t)sizeof(float));
		if (block < 64)
			block = 64;
	}
	if (block > nvecs)
		block = nvecs;
	return block;
}

static void
lloyd_assign_range(
		KMeansState	  *st,
		lloyd_block_fn block_fn,
		float		  *dist_buf,
		uint32_t	   block_size,
		uint32_t	   range_start,
		uint32_t	   range_end,
		float		  *cost_out)
{
	float *saved_dist = st->dist_block;
	st->dist_block	  = dist_buf;

	float cost = 0.0f;
	for (uint32_t start = range_start; start < range_end; start += block_size)
	{
		uint32_t count = range_end - start;
		if (count > block_size)
			count = block_size;
		st->total_cost = 0.0f;
		block_fn(st, start, count);
		cost += st->total_cost;
	}
	*cost_out	   = cost;
	st->dist_block = saved_dist;
}

typedef struct
{
	KMeansState	  *st;
	lloyd_block_fn block_fn;
	float		  *dist_bufs; /* [nthreads * block * nlist] */
	float		  *costs;	  /* [nthreads] */
	float		  *vec_bufs;  /* [nthreads * block * dim] or NULL */
	uint32_t	   block;
} LloydParCtx;

static void
lloyd_par_worker(uint32_t thread_id, uint32_t start, uint32_t end, void *arg)
{
	LloydParCtx *ctx   = (LloydParCtx *)arg;
	uint32_t	 nlist = ctx->st->nlist;
	uint32_t	 dim   = ctx->st->dim;
	uint32_t	 block = ctx->block;

	KMeansState local = *ctx->st;
	local.dist_block  = ctx->dist_bufs + (size_t)thread_id * block * nlist;
	if (ctx->vec_bufs)
		local.vec_block = ctx->vec_bufs + (size_t)thread_id * block * dim;

	lloyd_assign_range(
			&local,
			ctx->block_fn,
			local.dist_block,
			ctx->block,
			start,
			end,
			&ctx->costs[thread_id]);
}

void
lloyd_assign(KMeansState *st, bool use_cblas)
{
	st->total_cost = 0.0f;

	if (st->metric == DISTANCE_L2)
		precompute_norms_c(st);

	lloyd_block_fn block_fn = lloyd_select_block_fn(st, use_cblas);

	lloyd_assign_range(
			st,
			block_fn,
			st->dist_block,
			KMEANS_BLOCK_SIZE,
			0,
			st->nvecs,
			&st->total_cost);
}

/* ----------------------------------------------------------------
 * Fused iterative assign + update via iterate callback
 * ---------------------------------------------------------------- */

typedef struct
{
	KMeansState	  *st;
	lloyd_block_fn block_fn;
	uint32_t	   nthreads;
	uint32_t	   block;

	/* Per-thread buffers */
	float	 *dist_bufs;	 /* [nthreads * block * nlist] */
	float	 *vec_bufs;		 /* [nthreads * block * dim] or NULL */
	float	 *centroid_sums; /* [nthreads * nlist * dim] */
	uint32_t *centroid_cnts; /* [nthreads * nlist] */
	float	 *costs;		 /* [nthreads] */

	/* Convergence */
	float	*old_centroids; /* [nlist * dim] saved before iteration */
	float	 tolerance;
	bool	 verbose;
	uint32_t completed_iters;
} LloydIterCtx;

/*
 * Work function: assign vectors [start, end) to nearest centroids,
 * then accumulate per-thread centroid sums for the update step.
 *
 * Each thread gets its own dist/vec buffers and centroid accumulators
 * so there is no shared mutable state during the parallel phase.
 */
static void
lloyd_iter_work(uint32_t thread_id, uint32_t start, uint32_t end, void *arg)
{
	LloydIterCtx *ctx	= (LloydIterCtx *)arg;
	KMeansState	 *st	= ctx->st;
	uint32_t	  nlist = st->nlist;
	uint32_t	  dim	= st->dim;
	uint32_t	  block = ctx->block;

	/* 1. Set up thread-local KMeansState with private buffers */
	KMeansState local = *st;
	local.dist_block  = ctx->dist_bufs + (size_t)thread_id * block * nlist;
	if (ctx->vec_bufs)
		local.vec_block = ctx->vec_bufs + (size_t)thread_id * block * dim;

	/* 2. Assign: compute distances and find nearest centroid */
	lloyd_assign_range(
			&local,
			ctx->block_fn,
			local.dist_block,
			block,
			start,
			end,
			&ctx->costs[thread_id]);

	/* 3. Accumulate: add each vector to its assigned centroid's sum */
	float	 *my_sums = ctx->centroid_sums + (size_t)thread_id * nlist * dim;
	uint32_t *my_cnts = ctx->centroid_cnts + (size_t)thread_id * nlist;

	for (uint32_t i = start; i < end; i++)
	{
		ClusterId	 c	 = st->assignments[i];
		uint32_t	 idx = st->indices ? st->indices[i] : i;
		const float *vec = (const float *)st->vectors + (size_t)idx * dim;
		float		*sum = my_sums + (size_t)c * dim;
		for (uint32_t d = 0; d < dim; d++)
			sum[d] += vec[d];
		my_cnts[c]++;
	}
}

/*
 * Reduce function: merge per-thread results, update centroids,
 * and check convergence. Runs on the leader thread between
 * barrier-synchronized iterations.
 *
 * Returns true to continue iterating, false to stop.
 */
static bool
lloyd_iter_reduce(void *arg, uint32_t iteration)
{
	LloydIterCtx *ctx	= (LloydIterCtx *)arg;
	KMeansState	 *st	= ctx->st;
	uint32_t	  nlist = st->nlist;
	uint32_t	  dim	= st->dim;
	uint32_t	  nt	= ctx->nthreads;

	/* 1. Sum per-thread costs into total */
	st->total_cost = 0.0f;
	for (uint32_t t = 0; t < nt; t++)
		st->total_cost += ctx->costs[t];

	/* 2. Merge per-thread centroid accumulators into new_centroids */
	memset(st->new_centroids, 0, (size_t)nlist * dim * sizeof(float));
	memset(st->cluster_sizes, 0, nlist * sizeof(uint32_t));

	for (uint32_t t = 0; t < nt; t++)
	{
		float	 *sums = ctx->centroid_sums + (size_t)t * nlist * dim;
		uint32_t *cnts = ctx->centroid_cnts + (size_t)t * nlist;

		for (uint32_t c = 0; c < nlist; c++)
		{
			st->cluster_sizes[c] += cnts[c];
			float *dst = st->new_centroids + (size_t)c * dim;
			float *src = sums + (size_t)c * dim;
			for (uint32_t d = 0; d < dim; d++)
				dst[d] += src[d];
		}
	}

	/* 3. Compute new centroids as mean of assigned vectors */
	for (uint32_t c = 0; c < nlist; c++)
	{
		if (st->cluster_sizes[c] == 0)
			continue;
		float  inv	= 1.0f / (float)st->cluster_sizes[c];
		float *cent = st->new_centroids + (size_t)c * dim;
		for (uint32_t d = 0; d < dim; d++)
			cent[d] *= inv;
	}

	/* 4. Re-normalize centroids for cosine metric */
	if (st->metric == DISTANCE_COSINE)
	{
		for (uint32_t c = 0; c < nlist; c++)
		{
			if (st->cluster_sizes[c] == 0)
				continue;
			float *cent = st->new_centroids + (size_t)c * dim;
			float  norm = vs_l2_norm(cent, dim);
			if (norm > 1e-10f)
				vec32_scale(cent, 1.0f / norm, cent, dim);
		}
	}

	/* 5. Swap old and new centroids */
	float *tmp		  = st->centroids;
	st->centroids	  = st->new_centroids;
	st->new_centroids = tmp;

	/* 6. Check convergence: max centroid movement (squared) */
	float shift_sq = kmeans_max_centroid_shift_between(
			st->centroids, ctx->old_centroids, nlist, dim);

	if (ctx->verbose)
		vs_log("  iter %u: cost=%.4f, max_shift=%.6f\n",
			   iteration,
			   st->total_cost,
			   sqrtf(shift_sq));

	ctx->completed_iters = iteration + 1;

	float tol_sq = ctx->tolerance * ctx->tolerance;
	if (shift_sq < tol_sq)
		return false;

	/* 7. Prepare for next iteration: save centroids, reset accumulators */
	memcpy(ctx->old_centroids,
		   st->centroids,
		   (size_t)nlist * dim * sizeof(float));
	memset(ctx->centroid_sums, 0, (size_t)nt * nlist * dim * sizeof(float));
	memset(ctx->centroid_cnts, 0, (size_t)nt * nlist * sizeof(uint32_t));
	memset(ctx->costs, 0, nt * sizeof(float));

	/* 8. Precompute centroid norms for next iteration's distance calc */
	if (st->metric == DISTANCE_L2)
		precompute_norms_c(st);

	return true;
}

uint32_t
lloyd_iterate(KMeansState *st, bool use_cblas, const KMeansOptions *opts)
{
	(void)use_cblas;

	uint32_t nt	   = 1; /* serial: the driver owns parallelism, not k-means */
	uint32_t nlist = st->nlist;
	uint32_t dim   = st->dim;
	uint32_t block = lloyd_parallel_block_size(nlist, st->nvecs);

	lloyd_block_fn block_fn = lloyd_select_block_fn(st, use_cblas);

	/* Precompute norms before first iteration */
	if (st->metric == DISTANCE_L2)
		precompute_norms_c(st);

	VS_MEMCTX_SCOPE(iter_ctx);
	VsMemCtx old_ctx = vs_memctx_switch(iter_ctx);

	float *dist_bufs = vs_alloc((size_t)nt * block * nlist * sizeof(float));
	float *vec_bufs	 = NULL;
	if (st->vec_block != NULL)
		vec_bufs = vs_alloc((size_t)nt * block * dim * sizeof(float));

	float *centroid_sums = vs_alloc0((size_t)nt * nlist * dim * sizeof(float));
	uint32_t *centroid_cnts = vs_alloc0((size_t)nt * nlist * sizeof(uint32_t));
	float	 *costs			= vs_alloc0(nt * sizeof(float));
	float	 *old_centroids = vs_alloc((size_t)nlist * dim * sizeof(float));

	vs_memctx_switch(old_ctx);

	memcpy(old_centroids, st->centroids, (size_t)nlist * dim * sizeof(float));

	LloydIterCtx ctx = {
			.st				 = st,
			.block_fn		 = block_fn,
			.nthreads		 = nt,
			.block			 = block,
			.dist_bufs		 = dist_bufs,
			.vec_bufs		 = vec_bufs,
			.centroid_sums	 = centroid_sums,
			.centroid_cnts	 = centroid_cnts,
			.costs			 = costs,
			.old_centroids	 = old_centroids,
			.tolerance		 = opts->tolerance,
			.verbose		 = opts->verbose,
			.completed_iters = 0,
	};

	for (uint32_t iter = 0; iter < opts->max_iterations; iter++)
	{
		lloyd_iter_work(0, 0, st->nvecs, &ctx);
		if (!lloyd_iter_reduce(&ctx, iter))
			break;
	}

	/* Final assignment (reduce already did the last centroid update) */
	memset(costs, 0, nt * sizeof(float));

	LloydParCtx par_ctx = {
			.st		   = st,
			.block_fn  = block_fn,
			.dist_bufs = dist_bufs,
			.costs	   = costs,
			.vec_bufs  = vec_bufs,
			.block	   = block,
	};

	lloyd_par_worker(0, 0, st->nvecs, &par_ctx);
	st->total_cost = 0.0f;
	for (uint32_t t = 0; t < nt; t++)
		st->total_cost += costs[t];

	return ctx.completed_iters;
}
