/*
 * index_build.h - Shared index build utilities
 *
 * Generic helpers for building meerkat indexes, usable from both
 * the PostgreSQL IAM build and the standalone CLI. All functions
 * operate on the MktStorage abstraction and HKMeansResult tree.
 */

#ifndef MKT_INDEX_BUILD_H
#define MKT_INDEX_BUILD_H

#include <stdint.h>

#include "algo/hkmeans.h"
#include "index/centroid_build.h"
#include "index/centroid_page.h"
#include "index/storage.h"
#include "mkt_types.h"
#include "quant/rabitq.h"

/* ----------------------------------------------------------------
 * Build statistics — shared between standalone and PG builds
 * ---------------------------------------------------------------- */

typedef struct MktBuildStats
{
	/* Phase timings (milliseconds) */
	double ms_total;	/* total build time */
	double ms_sample;	/* sampling vectors for clustering */
	double ms_kmeans;	/* hierarchical k-means clustering */
	double ms_setup;	/* RaBitQ params, centroid rotation, etc */
	double ms_posting;	/* posting build total (parallel + merge) */
	double ms_parallel; /* parallel encode+write phase */
	double ms_merge;	/* serial partial page merge */
	double ms_centroid; /* writing centroid pages */

	/* Posting merge stats */
	uint32_t nworkers;
	uint32_t total_pages;
	uint32_t merge_input;
	uint32_t merge_output;
} MktBuildStats;

void mkt_build_stats_print(const MktBuildStats *s);

/* ----------------------------------------------------------------
 * Centroid layout + write
 * ---------------------------------------------------------------- */

/*
 * Compute the block layout for centroid pages in a BFS tree.
 *
 * Assigns sequential block numbers to each BFS node's centroid
 * pages, starting at first_blkno. Each node gets
 * ceil(nchildren / max_entries) pages.
 *
 * node_first_blkno must have space for tree->nnodes entries.
 * Returns the next available block number after all centroid pages.
 */
BlockNumber mkt_compute_centroid_layout(
		const HKMeansResult *tree,
		uint32_t			 max_entries,
		BlockNumber			 first_blkno,
		BlockNumber			*node_first_blkno);

/*
 * Write centroid pages for all BFS nodes in the tree.
 *
 * Iterates the tree in BFS order, encoding each node's centroids
 * and writing them to pages via the storage abstraction. Leaf
 * nodes get MKT_CENTROID_FLAG_LEAF; internal nodes get child_blkno
 * pointers from node_first_blkno. Leaf nodes optionally get
 * posting_heads as child block numbers.
 *
 * posting_heads may be NULL (centroid-only build without posting
 * lists).
 */
/*
 * pt_centroids: optional P^T * centroid array [nlist * dim] for leaf
 * entries, stored alongside routing data. NULL to skip.
 */
void mkt_write_centroid_tree(
		MktStorage			*storage,
		const HKMeansResult *tree,
		Dimension			 dim,
		uint32_t			 fan_out,
		MktCentroidFormat	 centroid_format,
		const RaBitQParams	*rq_params,
		const float			*global_mean,
		const BlockNumber	*posting_heads,
		const BlockNumber	*node_first_blkno,
		const float			*pt_centroids);

/*
 * Auto-tune fan_out from nlist.
 *
 * When fan_out equals default_fan_out, derive a value that gives a
 * balanced tree with ~nlist leaves. Uses sqrt for moderate nlist,
 * cbrt for large nlist (> 256^2 = 65536). When fan_out has been
 * explicitly set (differs from default), it is returned unchanged
 * (capped to nlist if larger).
 */
uint32_t
mkt_auto_fan_out(uint32_t fan_out, uint32_t nlist, uint32_t default_fan_out);

/*
 * Auto-tune nlist from a (possibly estimated) vector count: sqrt(count),
 * clamped to [1, 10000]. Used by every build path when nlist is not set
 * explicitly. The count source differs by back-end (reltuples / heap-block
 * estimate in PostgreSQL, the in-memory vector count standalone), but the
 * resolution is the same.
 */
uint32_t mkt_auto_nlist(double count);

/*
 * Find secondary cluster by plain distance (2nd-nearest centroid),
 * given a distance-sorted candidate list (e.g. from a tree beam
 * descent). The nearest candidate that is not primary_cluster is the
 * 2nd-nearest centroid.
 *
 * Returns the secondary cluster index, or primary_cluster if the gap
 * ratio exceeds epsilon (no replication needed).
 * gap_ratio = (dist_2nd - dist_primary) / |dist_primary|
 */
uint32_t mkt_find_secondary_cluster(
		const uint32_t *cand_leaves,
		const Distance *cand_dists,
		uint32_t		ncand,
		uint32_t		primary_cluster,
		Distance		primary_dist,
		double			epsilon);

/*
 * Find secondary cluster via SOAR (Spilling with Orthogonality-
 * Amplified Residuals).
 *
 * Computes the orthogonality-amplified distance for each candidate
 * centroid:
 *   OA(vec, c) = ||vec - c||^2 + lambda * dot(vec - c, r)^2
 * where r is the normalized residual from the primary centroid.
 *
 * Returns the centroid minimizing OA distance (excluding primary).
 * When lambda=0, this degenerates to standard 2nd-nearest.
 */
uint32_t mkt_find_soar_secondary(
		const float *vec,
		const float *leaf_centroids,
		uint32_t	 nleaves,
		Dimension	 dim,
		uint32_t	 primary_cluster,
		const float *normalized_residual,
		double		 lambda);

#endif /* MKT_INDEX_BUILD_H */
