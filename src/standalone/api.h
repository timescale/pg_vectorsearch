/*
 * api.h - Public C API for standalone pg_vectorsearch library
 *
 * Single entry point for building and querying in-memory ANN indexes.
 * Used by the CLI benchmark, Python ctypes bindings, and ann-benchmarks.
 */

#ifndef VS_STANDALONE_API_H
#define VS_STANDALONE_API_H

#include <stdint.h>

#include "index/index_build.h"
#include "standalone/vec32_source.h"

/* Opaque index handle (wraps PrismIndex + PrismQueryCtx) */
typedef struct VsHandle VsHandle;

/* Index build statistics */
typedef struct PrismBuildInfo
{
	uint32_t		nlist;		 /* actual number of clusters */
	uint32_t		nlevels;	 /* tree depth */
	uint32_t		nvecs;		 /* total vectors in index */
	uint32_t		max_cluster; /* largest cluster size */
	uint32_t		min_cluster; /* smallest cluster size */
	PrismBuildStats stats;		 /* phase timings + posting stats */
} PrismBuildInfo;

/*
 * Build an index by streaming vectors from a source.
 *
 * The source is iterated twice (sample + assign), then no longer
 * needed. Caller owns the source lifecycle.
 *
 * metric:       "angular" (cosine) or "euclidean" (L2).
 * centroid_fmt: "rabitq", "float32", or "float16".
 * nlist:        number of clusters (0 = auto: sqrt(nvecs)).
 * fan_out:      tree branching factor (0 = auto from nlist).
 * km_nredo:     k-means restarts (0 = default).
 * km_max_iter:  k-means max iterations (0 = default).
 * info:         if non-NULL, filled with build statistics.
 *
 * Returns NULL on failure.
 */
VsHandle *vs_handle_create(
		Vec32Source	   *src,
		uint32_t		nlist,
		uint32_t		fan_out,
		const char	   *metric,
		const char	   *centroid_fmt,
		const char	   *posting_fmt,
		uint32_t		km_nredo,
		uint32_t		km_max_iter,
		double			soar_lambda,
		double			boundary_epsilon,
		int				fastscan,
		int32_t			nworkers,
		PrismBuildInfo *info);

/*
 * Convenience: build from a flat float32 array.
 *
 * Wraps the array in an VsArraySource and calls vs_handle_create.
 */
VsHandle *vs_handle_create_from_array(
		const float	   *vectors,
		uint32_t		nvecs,
		uint32_t		dim,
		uint32_t		nlist,
		uint32_t		fan_out,
		const char	   *metric,
		const char	   *centroid_fmt,
		const char	   *posting_fmt,
		uint32_t		km_nredo,
		uint32_t		km_max_iter,
		double			soar_lambda,
		double			boundary_epsilon,
		int				fastscan,
		int32_t			nworkers,
		PrismBuildInfo *info);

/*
 * Query the index for the k nearest neighbors.
 *
 * distance_mode: "asymmetric" or "symmetric".
 * result_ids must have space for k uint32_t entries.
 * Returns actual number of results (may be < k).
 */
uint32_t vs_handle_query(
		VsHandle	*handle,
		const float *query,
		uint32_t	 k,
		uint32_t	 nprobe,
		const char	*distance_mode,
		bool		 rerank,
		uint32_t	*result_ids);

/*
 * Free the index and all associated memory.
 */
void vs_handle_destroy(VsHandle *handle);

#endif /* VS_STANDALONE_API_H */
