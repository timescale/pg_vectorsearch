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
 * Used by both the PG IAM build (build.c) and the CLI
 * benchmark (bench_search.c).
 */

#ifndef MKT_HKMEANS_H
#define MKT_HKMEANS_H

#include <stdint.h>

#include "algo/kmeans.h"
#include "core/types.h"

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
 * indices: optional index array for indirect access (NULL = identity).
 *          When non-NULL, vector i is at vectors[indices[i] * dim].
 *          This allows subsampling without copying.
 *
 * Returns a single contiguous allocation on success, NULL on failure.
 * Caller must free with mkt_free().
 */
HKMeansResult *mkt_hkmeans_f32(
		const float			*vectors,
		uint32_t			 nvecs,
		const uint32_t		*indices,
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

/* Upper bound on k / beam_width for mkt_hkmeans_assign_topk (keeps the
 * beam scratch on the stack). */
#define MKT_HK_MAX_TOPK 64

/*
 * Beam-search the tree for the k nearest leaf centroids.
 *
 * Maintains a beam of the best `beam_width` nodes per level, then keeps
 * the k nearest leaves at the leaf level. Approximate for k/beam_width
 * smaller than the tree fan-out, but far cheaper than scanning all
 * leaves — used for secondary (boundary) cluster assignment during
 * build. k and beam_width are clamped to MKT_HK_MAX_TOPK.
 *
 * out_leaves[k] receives leaf indices sorted by ascending distance;
 * out_dists[k] (optional) the matching distances. Returns the number of
 * leaves written (<= k).
 */
uint32_t mkt_hkmeans_assign_topk(
		const HKMeansResult *tree,
		const float			*vec,
		DistanceMetric		 metric,
		uint32_t			 k,
		uint32_t			 beam_width,
		uint32_t			*out_leaves,
		Distance			*out_dists);

/*
 * Upper bound (bytes) on the contiguous size of a tree built for `nlist`
 * leaves with `fan_out`. Used to size the fixed per-subtree DSM slots the
 * parallel build's participants write their subtrees into.
 */
size_t
mkt_hkmeans_max_blob_size(uint32_t nlist, uint32_t fan_out, Dimension dim);

/*
 * Same bound with every level's width capped at max_leaves: a node exists
 * only where a training vector landed, so a subtree clustered from
 * max_leaves vectors can never exceed it, whatever the nlist target.
 */
size_t mkt_hkmeans_max_blob_size_capped(
		uint32_t nlist, uint32_t fan_out, Dimension dim, uint64_t max_leaves);

/*
 * Tree depth for `nlist` leaves at `fan_out` — the same value the tree build
 * uses internally. Exposed so the streaming (page-backed) centroid-tree build
 * can compute the level structure without materializing a tree.
 */
uint32_t mkt_hkmeans_nlevels(uint32_t nlist, uint32_t fan_out);

/*
 * Build a one-level (flat) tree directly from pre-computed leaf centroids.
 *
 * For a flat clustering (nleaves <= fan_out) the root k-means already produced
 * every leaf centroid, so the parallel build can assemble the tree straight
 * from them rather than re-gathering the samples and re-clustering. Produces
 * the same shape mkt_hkmeans_f32 does for a single-level build: one
 * leaf-parent root with nleaves children. Returns a contiguous allocation;
 * caller frees with mkt_free().
 */
HKMeansResult *mkt_hkmeans_build_flat(
		const float *centroids,
		uint32_t	 nleaves,
		uint32_t	 fan_out,
		Dimension	 dim);

#endif /* MKT_HKMEANS_H */
