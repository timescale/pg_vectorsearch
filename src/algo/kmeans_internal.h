/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * kmeans_internal.h - Shared internal types for k-means implementations
 *
 * This header is shared between kmeans.c (Lloyd's / CBLAS), and variant
 * algorithm files (kmeans_hamerly.c, kmeans_elkan.c, etc.).
 *
 * Not part of the public API — do not include from outside src/algo/.
 */

#ifndef VS_KMEANS_INTERNAL_H
#define VS_KMEANS_INTERNAL_H

#include <float.h>
#include <stdint.h>

#include "algo/kmeans.h"
#include "types/vec16.h"
#include "types/vec32.h"

/* Block size for assignment step (matches FAISS) */
#define KMEANS_BLOCK_SIZE 4096

/*
 * Redefine VS_TARGET_CLONES locally to avoid pulling in <immintrin.h>
 * from simd_utils.h, which can affect codegen for the CBLAS path.
 */
#ifndef VS_TARGET_CLONES
#if !defined(VS_SIMD_NONE) && !defined(VS_COVERAGE) && \
		__has_attribute(target_clones) &&              \
		(defined(__x86_64__) || defined(__i386__))
/* Same list as simd_utils.h, kept local to avoid immintrin.h. */
#if defined(__clang__) || __GNUC__ >= 12
#define VS_TARGET_CLONES \
	__attribute__((      \
			target_clones("default", "arch=x86-64-v3", "arch=x86-64-v4")))
#else
#define VS_TARGET_CLONES \
	__attribute__((      \
			target_clones("default", "arch=haswell", "arch=skylake-avx512")))
#endif
#else
#define VS_TARGET_CLONES
#endif
#endif

/*
 * Internal state for one k-means run.
 *
 * Shared across all algorithm variants. The core fields (vectors through
 * total_cost) are used by every variant. Algorithm-specific state is
 * stored externally (e.g., HamerlyState) and passed alongside.
 */
typedef struct KMeansState
{
	void *memctx; /* Arena context (VsMemCtx) — used by kmeans.c */

	/* Input (not owned) */
	const void	   *vectors;
	const uint32_t *indices;  /* NULL = identity mapping [0..nvecs) */
	VecType			vec_type; /* element type (f32, f16, f16c) */
	uint32_t		nvecs;
	uint32_t		nlist;
	Dimension		dim;
	DistanceMetric	metric;

	/* Working state (owned) — always float32 */
	float	  *centroids;	  /* [nlist * dim] */
	float	  *vec_block;	  /* [BLOCK_SIZE * dim] BLAS convert buf */
	ClusterId *assignments;	  /* [nvecs] */
	uint32_t  *cluster_sizes; /* [nlist] */
	float	  *norms_x;		  /* [nvecs] precomputed ||x||^2 (L2) */
	float	  *norms_c;		  /* [nlist] precomputed ||c||^2 (L2) */
	float	  *dist_block;	  /* [KMEANS_BLOCK_SIZE * nlist] work buf */
	float	  *new_centroids; /* [nlist * dim] accumulator */
	float	   total_cost;
} KMeansState;

/* Get pointer to vector i in the input array */
__attribute__((always_inline)) static inline const void *
km_get_vector(const KMeansState *st, uint32_t i, size_t elem_size)
{
	uint32_t idx = st->indices ? st->indices[i] : i;
	return (const char *)st->vectors + (size_t)idx * st->dim * elem_size;
}

/*
 * Max centroid movement between two centroid arrays.
 * Returns the maximum squared L2 shift.
 */
float kmeans_max_centroid_shift_between(
		const float *a, const float *b, uint32_t nlist, Dimension dim);

/*
 * The per-worker accumulators the reduce combines, plus the scratch it
 * needs to do so.
 *
 * Each worker assigned its own slice of the vectors to the nearest
 * centroid and accumulated, per cluster, the component-wise sum of the
 * vectors that landed there and how many there were. Summing those is
 * what produces the new centroids: sums and counts add across workers,
 * whereas per-worker means could not be combined without re-weighting
 * them by their counts.
 */
typedef struct KMeansReduce
{
	/* [nworkers][nlist * dim] component-wise sum of the vectors each
	 * worker assigned to each cluster. */
	const float *const *worker_vector_sums;

	/* [nworkers][nlist] how many vectors each worker assigned to each
	 * cluster. Dividing a cluster's summed vector by the total is what
	 * yields its mean. */
	const uint32_t *const *worker_vector_counts;

	/* [nworkers] each worker's summed distance from its vectors to the
	 * centroids they were assigned to -- the k-means objective. The
	 * reduce sums these into out_total_cost; the parallel build
	 * discards the result, so nothing reads it on this path yet. */
	const float *worker_costs;

	uint32_t nworkers;

	/* [nlist * dim] centroids as they stood before this iteration.
	 * Caller-owned: the reduce reads it to report how far the furthest
	 * centroid moved, which is the convergence test. */
	const float *prev_centroids;

	/* [nlist] vectors per cluster once the workers are summed. Zeroed
	 * on entry. Caller-owned so one allocation serves every iteration,
	 * and so the bound on nlist sits with the caller that knows it --
	 * this layer accepts any nlist. */
	uint32_t *cluster_vector_counts;
} KMeansReduce;

/*
 * Merge the per-worker accumulators and update the centroids in place.
 *
 * This is the reduce step of parallel k-means (BSP pattern).
 * Called by the leader between barrier-synchronized iterations.
 *
 * Only the root level of the tree reaches here. The root is one
 * clustering over the whole sample, so it can only be split by data,
 * which is what makes a reduce necessary. Below the root the samples
 * are already partitioned, so each participant builds a whole child
 * subtree alone through the serial path and nothing needs merging.
 * In practice that means nlist is the tree's fan-out -- a few hundred,
 * not the leaf count -- though nothing here depends on that.
 *
 * Outputs:
 *   centroids: [nlist * dim] updated positions (in-place)
 *   norms_c:   [nlist] updated centroid norms (L2, may be NULL)
 *
 * Returns the maximum squared centroid shift (for convergence check).
 */
float kmeans_merge_centroids(
		float			   *centroids,
		float			   *norms_c,
		uint32_t			nlist,
		Dimension			dim,
		DistanceMetric		metric,
		const KMeansReduce *reduce,
		float			   *out_total_cost);

/*
 * Scalar assign+accumulate kernel for a range of vectors.
 *
 * For each vector in [start, end): find nearest centroid, add
 * to per-worker sums/counts. Supports optional root-assignment
 * filtering for hierarchical child k-means — when filter is
 * non-NULL, only vectors where filter[i] == filter_val are
 * processed.
 *
 * Used by both standalone (via Lloyd iterate) and PG parallel
 * workers (via Barrier iterate). The CBLAS path in Lloyd uses
 * its own batched sgemm kernel instead.
 */
void kmeans_assign_accumulate(
		const float	   *vectors,
		const uint32_t *indices,
		uint32_t		start,
		uint32_t		end,
		const float	   *centroids,
		const float	   *norms_c,
		uint32_t		k,
		Dimension		dim,
		DistanceMetric	metric,
		const uint32_t *filter,
		uint32_t		filter_val,
		float		   *out_sums,
		uint32_t	   *out_cnts,
		float		   *out_cost);

/*
 * Assignment-only kernel for a range of vectors.
 *
 * For each vector in [start, end): find nearest centroid and
 * write the centroid index to out_assignments[i]. Unlike
 * kmeans_assign_accumulate, this does not accumulate sums
 * or counts — it only outputs assignments.
 *
 * Used for root assignment in hierarchical k-means (both
 * standalone and PG paths).
 */
void kmeans_assign(
		const float	  *vectors,
		uint32_t	   start,
		uint32_t	   end,
		const float	  *centroids,
		const float	  *norms_c,
		uint32_t	   k,
		Dimension	   dim,
		DistanceMetric metric,
		uint32_t	  *out_assignments);

#endif /* VS_KMEANS_INTERNAL_H */
