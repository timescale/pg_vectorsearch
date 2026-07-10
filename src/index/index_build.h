/*
 * index_build.h - Shared index build utilities
 *
 * Generic helpers for building meerkat indexes, usable from both the
 * PostgreSQL IAM build and the standalone CLI. All functions operate on the
 * MktStorage abstraction and (where a tree is materialized at all) the
 * HKMeansResult tree.
 *
 * The routing tree is the centroid tree: the hierarchy of centroid pages
 * that routes queries and inserts down to the posting lists. These are the
 * builders that write it straight to pages.
 *
 * Who runs what: the build has a serial shape and a parallel shape, and
 * this header serves both.
 *
 *   Serial (one process; do_serial_build in pg/mktann_build.c):
 *     mkt_routing_tree_plan   clusters the sample once (recording each
 *                             node into a blob store) and sizes the page
 *                             layout;
 *     mkt_routing_tree_write  replays the recorded nodes and streams
 *                             every centroid + head page.
 *
 *   Parallel (leader + N workers; see parallel_build.h for the barrier
 *   choreography). Clustering is divided by ROOT CHILD: after the
 *   cooperative sampling scan and the barrier-synchronized root k-means,
 *   the root's children are scheduled largest-first in batches of
 *   nparticipants, and each participant clusters one child's subtree per
 *   batch into its slot of a bounded DSM ring. Page writing is NOT
 *   divided: between the batch barriers the leader alone consumes each
 *   batch (recording layout counts, spilling the subtree blobs), and after
 *   the last batch the leader alone streams every page --
 *     mkt_routing_subtree_write  writes one worker-built subtree's pages
 *                                at its reserved block range and maps its
 *                                local leaves into the global leaf index
 *                                space (leaf_offset = prefix sum of the
 *                                preceding children's leaf counts);
 *     mkt_centroid_write_node    writes the root page above the subtree
 *                                roots.
 *   The posting scan that follows is cooperative again (workers route and
 *   encode into a shared sort; the leader merges), but that machinery
 *   lives in posting_build.h / parallel_build.h, not here.
 *
 *   The standalone in-RAM build (no paging) uses mkt_write_centroid_tree
 *   directly on a fully materialized tree.
 */

#ifndef MKT_INDEX_BUILD_H
#define MKT_INDEX_BUILD_H

struct MktBlobStore;
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
	double ms_refine;	/* full-table leaf-centroid refinement */
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
 * pointers from node_first_blkno. Leaf entry j of a node gets the
 * formula-derived posting head posting_base + node->first_leaf + j
 * (no O(nlist) posting-head array).
 *
 * posting_base may be InvalidBlockNumber (centroid-only build without
 * posting lists).
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
		uint8_t				 level_offset,
		MktCentroidFormat	 centroid_format,
		const RaBitQParams	*rq_params,
		const float			*global_mean,
		BlockNumber			 posting_base,
		const BlockNumber	*node_first_blkno,
		const float			*pt_centroids);

/* ----------------------------------------------------------------
 * Streaming (page-backed) centroid-tree build
 *
 * Builds the hierarchical k-means tree top-down (DFS) and streams centroid
 * pages straight to storage, never materializing the whole tree in RAM. Peak
 * memory is the caller's sample buffer + an O(fan_out*depth*dim) recursion
 * stack; the whole tree would be O(nlist*dim). Runs as two passes over the
 * same recursion: a PLAN pass that clusters the sample once -- recording
 * each node's clustering into the caller's blob store -- and reports the
 * tree shape without writing; then a WRITE pass that replays the recorded
 * nodes and emits the pages. Posting-list heads are formula-derived from
 * the global leaf index, so the plan only sizes the page layout.
 *
 * These two entry points are the SERIAL build's whole clustering story
 * (single process, no barriers). The parallel build divides the same work
 * differently -- workers cluster per-root-child subtrees and only the
 * leader writes pages, via mkt_routing_subtree_write below -- but both
 * shapes produce the identical page layout: reserved centroid blocks
 * first, then the head region at first_posting, heads formula-derived.
 *
 * global_mean (the encoder centering) must be known before any page is
 * written; the PLAN pass reports the mean of the leaf centroids
 * (plan.leaf_mean) for that. The anchor is the leaf-centroid mean, not the
 * per-vector sample mean: it centers the quantization on what the tree
 * actually stores, and the quality of every centroid and posting code
 * depends on it.
 * ---------------------------------------------------------------- */

/*
 * Write one tree node's centroid page(s) in the node's format (fastscan or
 * encoder-based). child_count is the non-leaf entries' child capacity;
 * pass 0 for leaf-parent nodes.
 */
void mkt_centroid_write_node(
		MktStorage				  *storage,
		Dimension				   dim,
		const float				  *cents,
		uint32_t				   n,
		MktCentroidFormat		   fmt,
		uint8_t					   level,
		uint16_t				   flags,
		uint16_t				   child_count,
		const struct RaBitQParams *rq_params,
		const float				  *global_mean,
		const BlockNumber		  *child_blks,
		BlockNumber				   blkno);

typedef struct MktStreamTreePlan
{
	uint32_t nleaves;
	uint32_t nlevels;
	uint32_t centroid_pages; /* pages the write pass will emit */
	float	*leaf_mean;		 /* [dim] unweighted mean of the leaf centroids
							  * (mkt_alloc; caller frees with mkt_free) */
} MktStreamTreePlan;

/*
 * PLAN pass: cluster the sample and report the tree shape (leaf count,
 * depth, centroid page count, leaf-centroid mean) without writing anything.
 * Returns false on k-means failure.
 */
/*
 * Reserve the fixed page layout: extend the relation so blocks
 * [0, end_blkno) exist before any is written. mkt_storage_extend extends BY
 * npages; the count doubles as the absolute layout end only because the
 * relation holds nothing but the meta-page slot yet -- the reserved layout
 * (heads at first_posting + leaf) silently shifts if a page ever sneaks in
 * before this point, so the invariant is pinned here.
 */
static inline void
mkt_build_reserve_layout(MktStorage *storage, BlockNumber end_blkno)
{
	BlockNumber ext_base = mkt_storage_extend(storage, end_blkno);
	Assert(ext_base == 0 || ext_base == InvalidBlockNumber);
	(void)ext_base;
}

bool mkt_routing_tree_plan(
		const float			*vectors,
		uint32_t			 nvecs,
		Dimension			 dim,
		uint32_t			 nlist,
		uint32_t			 fan_out,
		DistanceMetric		 metric,
		MktCentroidFormat	 format,
		const KMeansOptions *opts,
		struct MktBlobStore *store,
		MktStreamTreePlan	*out);

/*
 * Per-leaf callback fired during the write pass, once per leaf, with the
 * leaf's global index and its float centroid (still resident at that moment).
 * The build uses it to write each posting-list head page carrying pt_centroid
 * = P^T*centroid — the exact encode reference — which cannot be recovered from
 * a compressed centroid page afterward. May be NULL.
 */
typedef void (*MktStreamLeafCb)(
		void *arg, uint32_t leaf, const float *centroid);

/*
 * WRITE pass: cluster the sample again (identical tree) and stream the
 * centroid pages to `storage` via on-demand block allocation (post-order, root
 * last). Leaf c's head is first_posting + c, and on_leaf (if set) fires per
 * leaf so the caller can write that leaf's head page from the resident float
 * centroid. Centroid pages occupy reserved blocks [first_centroid,
 * first_centroid + plan.centroid_pages) post-order (root last), and posting
 * heads live in the far posting area — so the caller must pre-extend the
 * relation to cover both before calling. Returns the root block (the value the
 * metadata page's first_centroid must carry), or InvalidBlockNumber on
 * failure.
 */
BlockNumber mkt_routing_tree_write(
		MktStorage			*storage,
		uint32_t			 nvecs,
		Dimension			 dim,
		uint32_t			 nlist,
		uint32_t			 fan_out,
		MktCentroidFormat	 format,
		const RaBitQParams	*rq_params,
		const float			*global_mean,
		struct MktBlobStore *store,
		BlockNumber			 first_posting,
		BlockNumber			 first_centroid,
		MktStreamLeafCb		 on_leaf,
		void				*on_leaf_arg);

/*
 * Stream one already-built subtree (an HKMeansResult a parallel worker
 * clustered into its ring slot, then spilled through the blob store) to
 * centroid pages at its reserved block range,
 * BFS layout (subtree root at first_block). Leader-only: workers cluster,
 * the leader writes -- it calls this once per root child, replaying the
 * blobs in the batch-schedule order while placing each at the child's own
 * reserved range, so the whole tree is never materialized as one blob and
 * page writing needs no cross-process coordination.
 *
 * The caller supplies the subtree's coordinates in the global layout, both
 * prefix sums over the preceding children (from the PLAN pass's per-child
 * counts): first_block for the block range, and leaf_offset for the leaf
 * index space -- leaf local_leaf's head is first_posting + leaf_offset +
 * local_leaf (formula-derived). level_offset places the subtree's
 * relative node levels at their absolute tree depth (1 under the root).
 * on_leaf fires per leaf with its float centroid so the caller can write
 * the head page. Returns the subtree root block (== first_block).
 */
BlockNumber mkt_routing_subtree_write(
		MktStorage			*storage,
		const HKMeansResult *subtree,
		Dimension			 dim,
		uint32_t			 fan_out,
		uint8_t				 level_offset,
		MktCentroidFormat	 format,
		const RaBitQParams	*rq_params,
		const float			*global_mean,
		BlockNumber			 first_posting,
		uint32_t			 leaf_offset,
		BlockNumber			 first_block,
		MktStreamLeafCb		 on_leaf,
		void				*on_leaf_arg);

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
 * Auto-tune nlist from a (possibly estimated) vector count: ~one list per
 * 256 vectors (count / 256), floored at sqrt(count) so small tables still get
 * enough lists to build. Used by every build path when nlist is not set
 * explicitly. The count source differs by back-end (reltuples / heap-block
 * estimate in PostgreSQL, the in-memory vector count standalone). Back-ends
 * clamp the result to their own nlist ceiling.
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
 * Computes the orthogonality-amplified distance for each searched centroid:
 *   OA(vec, c) = ||vec - c||^2 + lambda * dot(vec - c, r)^2
 * where r is the normalized residual from the primary centroid, and returns
 * the centroid minimizing OA distance (excluding primary). When lambda=0 this
 * degenerates to the standard 2nd-nearest.
 *
 * The search set is `count` leaves: the ids cand_leaves[0..count) when
 * cand_leaves is non-NULL (the beam-descent candidates — O(count)), otherwise
 * a full scan of leaves 0..count (pass count = nleaves). The candidate form is
 * used in production because the SOAR optimum is always among the nearest
 * leaves, so the result is unchanged while scaling to large nlist; NULL is for
 * callers/tests that want the exhaustive scan.
 */
uint32_t mkt_find_soar_secondary(
		const float	   *vec,
		const float	   *leaf_centroids,
		const uint32_t *cand_leaves,
		uint32_t		count,
		Dimension		dim,
		uint32_t		primary_cluster,
		const float	   *normalized_residual,
		double			lambda);

#endif /* MKT_INDEX_BUILD_H */
