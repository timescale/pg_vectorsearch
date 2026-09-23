/*
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
#define VS_TARGET_CLONES \
	__attribute__((      \
			target_clones("default", "arch=x86-64-v3", "arch=x86-64-v4")))
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
 * Merge per-worker centroid accumulators and update centroids.
 *
 * This is the reduce step of parallel k-means (BSP pattern).
 * Called by the leader between barrier-synchronized iterations.
 *
 * Inputs:
 *   worker_sums:  [nworkers][nlist * dim] per-worker centroid sums
 *   worker_cnts:  [nworkers][nlist] per-worker cluster counts
 *   worker_costs: [nworkers] per-worker total costs
 *   nworkers:     number of workers
 *   old_cents:    [nlist * dim] centroids from before this iteration
 *
 * Outputs:
 *   centroids:    [nlist * dim] updated centroid positions (in-place)
 *   norms_c:      [nlist] updated centroid norms (for L2, may be NULL)
 *
 * Returns the maximum squared centroid shift (for convergence check).
 */
float kmeans_merge_centroids(
		float				  *centroids,
		float				  *norms_c,
		const float			  *old_cents,
		const float *const	  *worker_sums,
		const uint32_t *const *worker_cnts,
		const float			  *worker_costs,
		uint32_t			   nworkers,
		uint32_t			   nlist,
		Dimension			   dim,
		DistanceMetric		   metric,
		float				  *out_total_cost);

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
