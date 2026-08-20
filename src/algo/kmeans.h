/*
 * kmeans.h - K-means clustering with BLAS-accelerated assignment
 *
 * Standard Lloyd's algorithm with k-means++ initialization and optional
 * CBLAS sgemm for the assignment step. The assignment converts O(N*K)
 * individual distance computations into a single matrix multiply.
 *
 * Supports L2, inner product, and cosine distance metrics.
 * For cosine: input vectors must be pre-normalized (unit length).
 */

#ifndef MKT_KMEANS_H
#define MKT_KMEANS_H

#include <stdbool.h>
#include <stdint.h>

#include "core/types.h"

/*
 * KMeansResult - Output of k-means clustering
 *
 * All arrays are heap-allocated and owned by the result. Call
 * mkt_kmeans_result_destroy() to free.
 */
typedef struct KMeansResult
{
	float	  *centroids;	  /* [nlist * dim] row-major */
	ClusterId *assignments;	  /* [nvecs] cluster id per vector */
	uint32_t  *cluster_sizes; /* [nlist] vectors per cluster */
	uint32_t   nlist;
	Dimension  dim;
	uint32_t   iterations; /* actual iterations in best run */
	float	   total_cost; /* sum of distances to assigned centroids */
} KMeansResult;

/*
 * KMeansAlgorithm - Assignment step algorithm
 */
typedef enum KMeansAlgorithm
{
	KMEANS_ALGO_AUTO = 0, /* CBLAS if available, else Hamerly/Lloyd */
	KMEANS_ALGO_LLOYD,	  /* Standard Lloyd's (block dot product) */
	KMEANS_ALGO_HAMERLY,  /* Hamerly's accelerated (L2 only) */
	KMEANS_ALGO_ELKAN,	  /* Elkan's accelerated (L2 only, O(nk) mem) */
	KMEANS_ALGO_CBLAS,	  /* CBLAS sgemm (requires BLAS library) */
} KMeansAlgorithm;

/*
 * KMeansOptions - Configuration for k-means
 */
typedef struct KMeansOptions
{
	uint32_t		max_iterations; /* default: 20 */
	float			tolerance;		/* convergence threshold, default: 1e-4 */
	uint64_t		seed;			/* random seed, default: 42 */
	uint32_t		nredo;			/* number of restarts, default: 1 */
	bool			verbose;		/* print per-iteration stats */
	KMeansAlgorithm algorithm;		/* assignment algorithm, default: auto */
	const float	   *initial_centroids; /* skip init, use these */
} KMeansOptions;

#define MKT_KMEANS_OPTIONS_DEFAULT \
	{.max_iterations = 20,         \
	 .tolerance		 = 1e-4f,      \
	 .seed			 = 42,         \
	 .nredo			 = 1,          \
	 .verbose		 = false,      \
	 .algorithm		 = KMEANS_ALGO_AUTO}

/*
 * Run k-means clustering on typed vectors.
 *
 * vec_type selects how input vectors are accessed (float32, float16,
 * or float16 with hand-written F16C SIMD). Centroids are always
 * float32. For f16 input, distance computation uses mixed-type
 * operations (half × float32) to avoid bulk conversion.
 *
 * For DISTANCE_COSINE: input vectors MUST be pre-normalized (unit
 * length). The function normalizes centroids after each update step
 * internally. The caller should normalize vectors during sampling.
 *
 * Runs nredo independent attempts and returns the best (lowest cost).
 *
 * Parameters:
 *   vectors:  [nvecs * dim] row-major input vectors (or full array
 *             when indices is non-NULL)
 *   indices:  optional index array [nvecs] mapping logical positions
 *             to rows in vectors. NULL = contiguous [0..nvecs).
 *   vec_type: element type (MKT_VEC_F32, MKT_VEC_F16, MKT_VEC_F16C)
 *   nvecs:    number of vectors (or indices) to cluster
 *   dim:      vector dimension
 *   nlist:    number of clusters (K)
 *   metric:   distance metric (L2, IP, or cosine)
 *   options:  configuration (NULL for defaults)
 *
 * Returns allocated result on success, NULL on failure.
 */
KMeansResult *mkt_kmeans(
		const void			*vectors,
		const uint32_t		*indices,
		MktVecType			 vec_type,
		uint32_t			 nvecs,
		Dimension			 dim,
		uint32_t			 nlist,
		DistanceMetric		 metric,
		const KMeansOptions *options);

/*
 * Convenience wrapper for float32 vectors.
 */
static inline KMeansResult *
mkt_kmeans_f32(
		const float			*vectors,
		uint32_t			 nvecs,
		Dimension			 dim,
		uint32_t			 nlist,
		DistanceMetric		 metric,
		const KMeansOptions *options)
{
	return mkt_kmeans(
			vectors, NULL, MKT_VEC_F32, nvecs, dim, nlist, metric, options);
}

/*
 * Free a k-means result and all owned arrays.
 */
void mkt_kmeans_result_destroy(KMeansResult *result);

/*
 * Get human-readable algorithm name.
 */
const char *mkt_kmeans_algo_name(KMeansAlgorithm algo);

/*
 * CBLAS runtime control (legacy, used by PostgreSQL extension).
 * For benchmarks, prefer KMeansOptions.algorithm instead.
 */
void		mkt_kmeans_set_use_cblas(bool use_cblas);
bool		mkt_kmeans_get_use_cblas(void);
const char *mkt_kmeans_impl_name(void);

/*
 * Check if BLAS is configured for single-threaded operation.
 *
 * Returns true if OMP_NUM_THREADS=1 is set in the environment.
 * BLAS libraries (BLIS, OpenBLAS, MKL) read this at load time to
 * size their thread pools. Must be set before process start.
 *
 * PostgreSQL: set in systemd unit or pg_ctl environment.
 * CLI: prefix command with OMP_NUM_THREADS=1.
 */
bool mkt_cblas_is_single_threaded(void);

/*
 * Pin BLAS to single-threaded via runtime APIs (BLIS, OpenBLAS).
 * Call once at startup. Uses weak symbols — safe when the linked
 * BLAS doesn't provide the function.
 */
void mkt_cblas_pin_single_thread(void);

#endif /* MKT_KMEANS_H */
