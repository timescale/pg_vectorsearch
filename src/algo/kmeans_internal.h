/*
 * kmeans_internal.h - Shared internal types for k-means implementations
 *
 * This header is shared between kmeans.c (Lloyd's / CBLAS), and variant
 * algorithm files (kmeans_hamerly.c, kmeans_elkan.c, etc.).
 *
 * Not part of the public API — do not include from outside src/algo/.
 */

#ifndef MKT_KMEANS_INTERNAL_H
#define MKT_KMEANS_INTERNAL_H

#include <float.h>
#include <stdint.h>

#include "algo/kmeans.h"
#include "mkt_halfvec.h"
#include "mkt_vector.h"

/* Block size for assignment step (matches FAISS) */
#define KMEANS_BLOCK_SIZE 4096

/*
 * Redefine MKT_TARGET_CLONES locally to avoid pulling in <immintrin.h>
 * from simd_utils.h, which can affect codegen for the CBLAS path.
 */
#ifndef MKT_TARGET_CLONES
#if !defined(MKT_SIMD_NONE) && !defined(MKT_COVERAGE) && \
		__has_attribute(target_clones) &&                \
		(defined(__x86_64__) || defined(__i386__))
#define MKT_TARGET_CLONES \
	__attribute__((       \
			target_clones("default", "arch=x86-64-v3", "arch=x86-64-v4")))
#else
#define MKT_TARGET_CLONES
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
	void *memctx; /* Arena context (MktMemCtx) — used by kmeans.c */

	/* Input (not owned) */
	const void	   *vectors;
	const uint32_t *indices;  /* NULL = identity mapping [0..nvecs) */
	MktVecType		vec_type; /* element type (f32, f16, f16c) */
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

	/* Parallel dispatch (NULL = serial) */
	KMeansParallelFn parallel_for;
	KMeansIterateFn	 iterate;
	void			*parallel_ctx;
	uint32_t		 nthreads;
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

#endif /* MKT_KMEANS_INTERNAL_H */
