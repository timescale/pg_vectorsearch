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
void mkt_write_centroid_tree(
		MktStorage			*storage,
		const HKMeansResult *tree,
		Dimension			 dim,
		uint32_t			 fan_out,
		MktCentroidFormat	 centroid_format,
		const RaBitQParams	*rq_params,
		const float			*global_mean,
		const BlockNumber	*posting_heads,
		const BlockNumber	*node_first_blkno);

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

#endif /* MKT_INDEX_BUILD_H */
