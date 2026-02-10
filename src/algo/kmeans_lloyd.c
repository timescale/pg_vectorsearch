/*
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
 *   Batch dot-product kernel with MKT_TARGET_CLONES for AVX-512/AVX2
 *   auto-vectorization. FMA generation requires -ffp-contract=fast
 *   (set in meson.build).
 *
 * Both use the decomposition:
 *   ||x - c||² = ||x||² + ||c||² - 2⟨x,c⟩
 * The ⟨x,c⟩ term is computed in batch, norms are precomputed.
 */

#include "mkt_config.h"

#include <stddef.h>
#include <stdint.h>

#ifdef MKT_HAVE_CBLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

#include "algo/kmeans_lloyd.h"
#include "algo/vecops.h"

/*
 * Precompute ||c||² for all centroids (L2 only).
 */
static void
precompute_norms_c(KMeansState *st)
{
	for (uint32_t j = 0; j < st->nlist; j++)
		st->norms_c[j] = mkt_l2_norm_squared(
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
#ifdef MKT_HAVE_CBLAS
static void
lloyd_assign_block_cblas(
		KMeansState *st, uint32_t block_start, uint32_t block_count)
{
	uint32_t nlist = st->nlist;
	uint32_t dim   = st->dim;
	float	*dist  = st->dist_block;

	const float *block_vecs = st->vectors + (size_t)block_start * dim;

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
#endif /* MKT_HAVE_CBLAS */

/*
 * Batch dot-product matrix: dots[i*nlist + j] = dot(vecs[i], cents[j])
 *
 * MKT_TARGET_CLONES generates AVX-512, AVX2, and default versions.
 * The innermost loop over dim is auto-vectorized by the compiler,
 * eliminating per-vector function pointer dispatch overhead.
 *
 * FMA generation requires -ffp-contract=fast (set in meson.build for
 * release builds). Without it, GCC with -std=c2x generates separate
 * vmulps + horizontal scalar adds instead of vfmadd231ps accumulate.
 */
MKT_TARGET_CLONES static void
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
				dot += v[d] * c[d];
			dots[(size_t)i * nlist + j] = dot;
		}
	}
}

/*
 * Assignment step: builtin fallback for one block.
 *
 * Uses batch dot-product + norm decomposition (same approach as CBLAS
 * path) to eliminate per-vector function pointer dispatch overhead.
 */
static void
lloyd_assign_block_builtin(
		KMeansState *st, uint32_t block_start, uint32_t block_count)
{
	uint32_t nlist = st->nlist;
	uint32_t dim   = st->dim;
	float	*dist  = st->dist_block;

	const float *block_vecs = st->vectors + (size_t)block_start * dim;

	/* Compute all dot products in one batch */
	lloyd_compute_dot_products(
			block_vecs, st->centroids, dist, block_count, nlist, dim);

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

void
lloyd_assign(KMeansState *st, bool use_cblas)
{
	st->total_cost = 0.0f;

	/* Precompute centroid norms for L2 */
	if (st->metric == DISTANCE_L2)
		precompute_norms_c(st);

	for (uint32_t start = 0; start < st->nvecs; start += KMEANS_BLOCK_SIZE)
	{
		uint32_t count = st->nvecs - start;
		if (count > KMEANS_BLOCK_SIZE)
			count = KMEANS_BLOCK_SIZE;

#ifdef MKT_HAVE_CBLAS
		if (use_cblas)
		{
			lloyd_assign_block_cblas(st, start, count);
			continue;
		}
#endif
		(void)use_cblas; /* suppress unused warning when no CBLAS */
		lloyd_assign_block_builtin(st, start, count);
	}
}
