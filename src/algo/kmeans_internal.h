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
#include "mkt_types.h"

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
	/* Input (not owned) */
	const float	  *vectors;
	uint32_t	   nvecs;
	uint32_t	   nlist;
	Dimension	   dim;
	DistanceMetric metric;

	/* Working state (owned) */
	float	  *centroids;	  /* [nlist * dim] */
	ClusterId *assignments;	  /* [nvecs] */
	uint32_t  *cluster_sizes; /* [nlist] */
	float	  *norms_x;		  /* [nvecs] precomputed ||x||^2 (L2) */
	float	  *norms_c;		  /* [nlist] precomputed ||c||^2 (L2) */
	float	  *dist_block;	  /* [KMEANS_BLOCK_SIZE * nlist] work buf */
	float	  *new_centroids; /* [nlist * dim] accumulator */
	float	   total_cost;
} KMeansState;

#endif /* MKT_KMEANS_INTERNAL_H */
