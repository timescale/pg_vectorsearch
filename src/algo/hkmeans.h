/*
 * hkmeans.h - Hierarchical k-means tree builder
 *
 * Builds a BFS-ordered tree of centroids using hierarchical k-means.
 * At each node, runs mkt_kmeans_f32() to split vectors into fan_out
 * children, then recurses on each partition.
 *
 * Used by both the PG IAM build (mktann_build.c) and the CLI
 * benchmark (bench_search.c).
 */

#ifndef MKT_HKMEANS_H
#define MKT_HKMEANS_H

#include <stdint.h>

#include "algo/kmeans.h"
#include "mkt_types.h"

/*
 * HKMeansNode - One node in the BFS-ordered tree
 *
 * Each node represents a k-means clustering of the vectors
 * assigned to it by its parent. The root node clusters all
 * input vectors.
 */
typedef struct HKMeansNode
{
	float	*centroids;	  /* [nchildren * dim] row-major (owned) */
	uint32_t nchildren;	  /* actual cluster count (<= fan_out) */
	uint32_t level;		  /* tree level (0 = root) */
	uint32_t first_child; /* index in nodes[] of first child */
	uint32_t first_leaf;  /* offset into leaf_centroids (leaf-parent only) */
} HKMeansNode;

/* Sentinel for leaf nodes with no children */
#define HKMEANS_NO_CHILD UINT32_MAX

/*
 * HKMeansResult - Complete hierarchical k-means tree
 */
typedef struct HKMeansResult
{
	HKMeansNode *nodes;			 /* BFS order (owned) */
	float		*leaf_centroids; /* [nleaves * dim] flat array (owned) */
	uint32_t	 nnodes;		 /* total internal nodes */
	uint32_t	 nlevels;		 /* tree depth */
	uint32_t	 nleaves;		 /* total leaf centroids */
	uint32_t	 fan_out;		 /* children per node (max) */
	Dimension	 dim;			 /* vector dimension */
} HKMeansResult;

/*
 * Build a hierarchical k-means tree.
 *
 * Tree depth: max(1, ceil(log(nlist) / log(fan_out)))
 * When nlist <= fan_out, the tree has a single level (flat).
 *
 * Parameters:
 *   vectors:  [nvecs * dim] row-major input vectors
 *   nvecs:    number of input vectors
 *   dim:      vector dimension
 *   nlist:    target number of leaf centroids
 *   fan_out:  max children per node
 *   metric:   distance metric for k-means
 *   options:  k-means configuration (NULL for defaults)
 *
 * Returns allocated result on success, NULL on failure.
 * Caller must free with mkt_hkmeans_result_destroy().
 */
HKMeansResult *mkt_hkmeans_f32(
		const float			*vectors,
		uint32_t			 nvecs,
		Dimension			 dim,
		uint32_t			 nlist,
		uint32_t			 fan_out,
		DistanceMetric		 metric,
		const KMeansOptions *options);

/*
 * Assign a vector to a leaf centroid via greedy tree descent.
 *
 * Returns the leaf index (0..nleaves-1). Optionally writes the
 * distance to the nearest leaf centroid into *out_dist.
 *
 * Cost: O(fan_out * nlevels) vs O(nleaves) for brute-force.
 * For cosine metric, vec must be pre-normalized.
 */
uint32_t mkt_hkmeans_assign(
		const HKMeansResult *tree,
		const float			*vec,
		DistanceMetric		 metric,
		Distance			*out_dist);

/*
 * Free a hierarchical k-means result and all owned data.
 */
void mkt_hkmeans_result_destroy(HKMeansResult *result);

#endif /* MKT_HKMEANS_H */
