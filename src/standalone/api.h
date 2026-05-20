/*
 * api.h - Public C API for standalone meerkat library
 *
 * Single entry point for building and querying in-memory ANN indexes.
 * Used by the CLI benchmark, Python ctypes bindings, and ann-benchmarks.
 */

#ifndef MKT_STANDALONE_API_H
#define MKT_STANDALONE_API_H

#include <stdint.h>

#include "index/index_build.h"
#include "standalone/vector_source.h"

/* Opaque index handle (wraps MktIndex + MktQueryCtx) */
typedef struct MktHandle MktHandle;

/* Index build statistics */
typedef struct MktBuildInfo
{
	uint32_t	  nlist;	   /* actual number of clusters */
	uint32_t	  nlevels;	   /* tree depth */
	uint32_t	  nvecs;	   /* total vectors in index */
	uint32_t	  max_cluster; /* largest cluster size */
	uint32_t	  min_cluster; /* smallest cluster size */
	MktBuildStats stats;	   /* phase timings + posting stats */
} MktBuildInfo;

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
MktHandle *mkt_handle_create(
		MktVectorSource *src,
		uint32_t		 nlist,
		uint32_t		 fan_out,
		const char		*metric,
		const char		*centroid_fmt,
		const char		*posting_fmt,
		uint32_t		 km_nredo,
		uint32_t		 km_max_iter,
		double			 soar_lambda,
		double			 boundary_epsilon,
		int				 fastscan,
		uint32_t		 nworkers,
		MktBuildInfo	*info);

/*
 * Convenience: build from a flat float32 array.
 *
 * Wraps the array in an MktArraySource and calls mkt_handle_create.
 */
MktHandle *mkt_handle_create_from_array(
		const float	 *vectors,
		uint32_t	  nvecs,
		uint32_t	  dim,
		uint32_t	  nlist,
		uint32_t	  fan_out,
		const char	 *metric,
		const char	 *centroid_fmt,
		const char	 *posting_fmt,
		uint32_t	  km_nredo,
		uint32_t	  km_max_iter,
		double		  soar_lambda,
		double		  boundary_epsilon,
		int			  fastscan,
		uint32_t	  nworkers,
		MktBuildInfo *info);

/*
 * Query the index for the k nearest neighbors.
 *
 * distance_mode: "asymmetric" or "symmetric".
 * result_ids must have space for k uint32_t entries.
 * Returns actual number of results (may be < k).
 */
uint32_t mkt_handle_query(
		MktHandle	*handle,
		const float *query,
		uint32_t	 k,
		uint32_t	 nprobe,
		const char	*distance_mode,
		bool		 rerank,
		uint32_t	*result_ids);

/*
 * Free the index and all associated memory.
 */
void mkt_handle_destroy(MktHandle *handle);

#endif /* MKT_STANDALONE_API_H */
