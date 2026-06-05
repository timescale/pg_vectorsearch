/*
 * hkmeans.h - Hierarchical k-means tree builder
 *
 * Builds a BFS-ordered tree of centroids using hierarchical k-means.
 * At each node, runs mkt_kmeans_f32() to split vectors into fan_out
 * children, then recurses on each partition.
 *
 * The result is self-contained: all data is stored in a single
 * contiguous allocation using byte offsets instead of pointers.
 * This allows memcpy into shared memory (DSM) for parallel builds.
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
 * centroid_offset is a byte offset from the HKMeansResult base,
 * making the tree self-contained and memcpy-able.
 */
typedef struct HKMeansNode
{
	uint32_t centroid_offset; /* byte offset from HKMeansResult* */
	uint32_t nchildren;		  /* actual cluster count (<= fan_out) */
	uint32_t level;			  /* tree level (0 = root) */
	uint32_t first_child;	  /* index in nodes[] of first child */
	uint32_t first_leaf; /* offset into leaf_centroids (leaf-parent only) */
} HKMeansNode;

/* Sentinel for leaf nodes with no children */
#define HKMEANS_NO_CHILD UINT32_MAX

/*
 * HKMeansResult - Complete hierarchical k-means tree
 *
 * Self-contained: all data (nodes, leaf centroids, internal
 * centroids) is stored in a single contiguous allocation.
 * The struct can be memcpy'd into shared memory (DSM) and
 * used directly by other processes without deserialization.
 *
 * Layout:
 *   [HKMeansResult header]
 *   [HKMeansNode nodes[nnodes]]         — at nodes_offset
 *   [float leaf_centroids[nleaves*dim]] — at leaf_offset
 *   [float internal_centroids[...]]     — packed after leaves
 */
typedef struct HKMeansResult
{
	uint32_t  nodes_offset; /* byte offset to nodes[] */
	uint32_t  leaf_offset;	/* byte offset to leaf centroids */
	uint32_t  total_size;	/* total allocation size in bytes */
	uint32_t  nnodes;		/* total internal nodes */
	uint32_t  nlevels;		/* tree depth */
	uint32_t  nleaves;		/* total leaf centroids */
	uint32_t  fan_out;		/* children per node (max) */
	Dimension dim;			/* vector dimension */
} HKMeansResult;

/* ----------------------------------------------------------------
 * Accessors — resolve byte offsets to typed pointers
 * ---------------------------------------------------------------- */

static inline HKMeansNode *
hk_nodes(const HKMeansResult *r)
{
	return (HKMeansNode *)((char *)r + r->nodes_offset);
}

static inline float *
hk_leaf_centroids(const HKMeansResult *r)
{
	return (float *)((char *)r + r->leaf_offset);
}

static inline float *
hk_node_centroids(const HKMeansResult *r, const HKMeansNode *node)
{
	return (float *)((char *)r + node->centroid_offset);
}

/* ----------------------------------------------------------------
 * Public API
 * ---------------------------------------------------------------- */

/*
 * Build a hierarchical k-means tree.
 *
 * Returns a single contiguous allocation on success, NULL on failure.
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
 * Route a vector to its nearest leaf centroid by descending the tree.
 * Returns the leaf index (0..nleaves-1).
 *
 * Optionally writes the distance to the nearest leaf centroid into
 * *out_distance (may be NULL).
 */
uint32_t mkt_hkmeans_assign(
		const HKMeansResult *tree,
		const float			*vec,
		DistanceMetric		 metric,
		Distance			*out_distance);

/*
 * Free a tree built by mkt_hkmeans_f32(). The tree is a single contiguous
 * allocation, so this is just a free of the base pointer.
 */
void mkt_hkmeans_result_destroy(HKMeansResult *result);

#endif /* MKT_HKMEANS_H */
