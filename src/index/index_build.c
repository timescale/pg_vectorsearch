/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * index_build.c - Shared index build utilities
 *
 * Generic helpers for building prism indexes, usable from both
 * the PostgreSQL IAM build and the standalone CLI.
 */

#include <inttypes.h>
#include <math.h>
#include <string.h>

#include "algo/kmeans.h"
#include "algo/vecops.h"
#include "core/log.h"
#include "core/memory.h"
#include "index/centroid_build.h"
#include "index/index_build.h"
#include "index/parallel_build.h" /* PrismBlobStore seam (plan/write replay) */
#include "quant/fastscan.h"

/* ----------------------------------------------------------------
 * Exact internal-centroid collection — see index_build.h
 * ---------------------------------------------------------------- */

/*
 * A collection is the collector's sealed output: the exact centroids
 * of the internal tree nodes, keyed by where each centroid's
 * compressed twin sits on disk (centroid page, entry index on that
 * page). Serialized as one contiguous allocation with no internal
 * pointers, so it can be handed across the parallel-build seam (a DSM
 * segment under PostgreSQL, a plain heap allocation standalone) and
 * consumed in place by any backend that maps it.
 *
 *   +--------------------------------+
 *   | ExactCentroidCollectionHeader  |  base, npages, nslots, dim
 *   +--------------------------------+
 *   | uint32_t page_off[npages]      |  slot base per page, or NONE
 *   +--------------------------------+
 *   | float cents[nslots * dim]      |  centroids, densely packed
 *   +--------------------------------+
 *
 * The collection covers the centroid-page region [base, base +
 * npages): a
 * page's page_off element is indexed by its offset from the starting
 * page (blkno - base). Pages that carry internal entries hold the
 * index of their first centroid in cents[]; their entries then map
 * 1:1 and in page-entry order to the following slots, so entry i of
 * page b scores against cents + (page_off[b - base] + i) * dim. Slot
 * ranges of different pages are disjoint, but not ordered: cents[] is
 * appended in tree-write order (post-order serial, per-subtree
 * parallel), not block order. Leaf-level pages — collection skips
 * them — hold PRISM_EXACT_INTERNAL_NONE instead.
 *
 * An empty collection (collecting off or over budget) is just the header
 * with npages = 0 and base = InvalidBlockNumber, which no block can
 * match — the consumer's scoring hook stays inert. nslots and dim are
 * self-description (sizing and debugging); the view reads only base
 * and npages, and slot extents come from page_off plus the pages' own
 * entry counts.
 *
 * Lifecycle: the LEADER produces it — it alone streams the tree to
 * pages, collecting into PrismExactCentroidCollector as it writes —
 * then sizes and fills the collection from the collector and publishes it
 * before the tree-ready barrier. The WORKERS attach after that
 * barrier and wrap it in a PrismExactInternalCentroids view to score
 * the internal tree levels while routing rows to their posting lists
 * (and again in the refine pass); they only ever read it. The collection's
 * memory is deliberately not context-managed and not the collector's:
 * under PostgreSQL it is a DSM segment (process-shared, refcounted by
 * the resource-owner machinery — a memory context is process-private
 * and could not cross the seam), standalone a heap allocation the
 * leader frees after the threads join. That split also lets the
 * leader clean up the collector right after serializing, halving the
 * peak, while the collection lives on until the last worker detaches.
 * The serial build skips serialization entirely: same process, so its view
 * points straight into the collector's arrays.
 */
typedef struct ExactCentroidCollectionHeader
{
	BlockNumber base;
	uint32_t	npages;
	uint32_t	nslots;
	uint32_t	dim;
} ExactCentroidCollectionHeader;

/* Fallback initial cents[] capacity (slots) when the caller passes no
 * expected_slots pre-size; add_node doubles from here. Covers a
 * typical two-level tree (fan_out + 1 nodes) without regrowing while
 * staying well under any realistic budget. */
#define CENTROID_COLLECTOR_INIT_CAPACITY 256

void
prism_exact_centroid_collector_init(
		PrismExactCentroidCollector *c,
		Dimension					 dim,
		PrismCentroidFormat			 fmt,
		BlockNumber					 base,
		uint32_t					 npages,
		uint64_t					 max_bytes,
		uint64_t					 expected_slots)
{
	memset(c, 0, sizeof(*c));
	c->dim	  = dim;
	c->base	  = base;
	c->npages = npages;
	c->stride = prism_centroid_max_entries_fmt(dim, fmt);
	/* Every allocation goes to the collector's dedicated child context
	 * (see the ownership contract on PrismExactCentroidCollector). */
	c->ctx		= vs_memctx_create(vs_memctx_current(), "vs exact centroids");
	c->page_off = vs_memctx_alloc(c->ctx, (size_t)npages * sizeof(uint32_t));
	memset(c->page_off, 0xFF, (size_t)npages * sizeof(uint32_t));
	c->max_bytes = max_bytes;
	/* Pre-size to the caller's estimate so growth is a rounding case,
	 * never past what the budget admits; an over-budget estimate is
	 * pointless to allocate in full -- collection will latch overflowed
	 * at the boundary anyway. */
	uint64_t cap	 = expected_slots > 0 ? expected_slots
										  : CENTROID_COLLECTOR_INIT_CAPACITY;
	uint64_t max_cap = max_bytes / ((uint64_t)dim * sizeof(float));
	if (cap > max_cap)
		cap = max_cap;
	if (cap == 0)
		cap = 1;
	c->cap	 = (uint32_t)cap;
	c->cents = vs_memctx_alloc(c->ctx, (size_t)c->cap * dim * sizeof(float));
}

/*
 * Ensure cents[] holds at least total slots. Reached only when the
 * init pre-size undershot (per-level rounding in the estimate): grow
 * geometrically — a fixed-step policy would copy O(n^2) bytes over
 * the collection's lifetime, doubling copies less than 2x the final
 * size in total — and clamp the capacity to the budget so the
 * allocation never overshoots what the caller's budget check admits
 * (that check guarantees the clamped capacity still fits).
 */
static void
exact_centroid_collector_reserve(
		PrismExactCentroidCollector *c, uint32_t total)
{
	if (total <= c->cap)
		return;

	uint64_t max_cap = c->max_bytes / ((uint64_t)c->dim * sizeof(float));
	Assert(total <= max_cap);
	uint64_t new_cap = (uint64_t)c->cap * 2;
	while (new_cap < total)
		new_cap *= 2;
	if (new_cap > max_cap)
		new_cap = max_cap;

	/* Grow inside the collector's own context, not the streaming
	 * scratch context this is called under (that scratch dies at
	 * the end of the write pass, while the collector must survive
	 * until after the encode scan and the refine pass). The old
	 * array is freed eagerly (a no-op on the standalone arena) so
	 * both generations are never held at once. */
	float *grown =
			vs_memctx_alloc(c->ctx, (size_t)new_cap * c->dim * sizeof(float));
	memcpy(grown, c->cents, (size_t)c->nslots * c->dim * sizeof(float));
	vs_free(c->cents);
	c->cents = grown;
	c->cap	 = (uint32_t)new_cap;
}

void
prism_exact_centroid_collector_add_node(
		PrismExactCentroidCollector *c,
		BlockNumber					 first_blk,
		const float					*cents,
		uint32_t					 n)
{
	if (n == 0 || c->overflowed)
		return;

	/* Enforce the collection budget: degrade to an empty (inert)
	 * collection rather than grow without bound. The arrays are dead
	 * once the flag latches (every consumer checks it first), so
	 * release them right away — the encode scan that follows is the
	 * build's memory peak. */
	uint64_t need = ((uint64_t)c->nslots + n) * c->dim * sizeof(float);
	if (need > c->max_bytes)
	{
		vs_warn("exact centroid collection over budget (needs more "
				"than %" PRIu64 " of %" PRIu64 " bytes); build descent "
				"falls back to estimated internal scoring — consider "
				"raising the build memory budget",
				need,
				c->max_bytes);
		c->overflowed = true;
		prism_exact_centroid_collector_cleanup(c);
		return;
	}

	exact_centroid_collector_reserve(c, c->nslots + n);

	/* The node's entries pack stride per page from first_blk — the same
	 * packing every centroid writer uses (each node starts on a fresh
	 * reserved page). Record each page's slot base and append its
	 * centroids in page-entry order. */
	uint32_t npages_node = (n + c->stride - 1) / c->stride;
	for (uint32_t p = 0; p < npages_node; p++)
	{
		BlockNumber blk = first_blk + p;
		if (blk < c->base || blk - c->base >= c->npages)
		{
			/* Plan/write divergence: a node landed outside the planned
			 * centroid region. Degrade to inert rather than write
			 * page_off out of bounds in release builds. */
			Assert(false);
			c->overflowed = true;
			prism_exact_centroid_collector_cleanup(c);
			return;
		}

		uint32_t count = n - p * c->stride;
		if (count > c->stride)
			count = c->stride;

		c->page_off[blk - c->base] = c->nslots;
		memcpy(c->cents + (size_t)c->nslots * c->dim,
			   cents + (size_t)p * c->stride * c->dim,
			   (size_t)count * c->dim * sizeof(float));
		c->nslots += count;
	}
}

void
prism_exact_centroid_collector_cleanup(PrismExactCentroidCollector *c)
{
	/* Everything the collector allocated lives in its dedicated
	 * context: one delete releases it all, exactly once (idempotent —
	 * a second call is a no-op). Callers must not run this until the
	 * last view consumer is done. */
	if (c->ctx != NULL)
		vs_memctx_delete(c->ctx);
	c->ctx		= NULL;
	c->page_off = NULL;
	c->cents	= NULL;
}

void
prism_exact_centroid_view(
		const PrismExactCentroidCollector *c,
		PrismExactInternalCentroids		  *view)
{
	if (c->overflowed)
	{
		/* Inert view: a zero-page region matches no block. */
		view->cents	   = NULL;
		view->page_off = NULL;
		view->base	   = InvalidBlockNumber;
		view->npages   = 0;
		return;
	}
	view->cents	   = c->cents;
	view->page_off = c->page_off;
	view->base	   = c->base;
	view->npages   = c->npages;
}

uint64_t
prism_exact_centroid_collection_size(const PrismExactCentroidCollector *c)
{
	if (c == NULL || c->overflowed)
		return sizeof(ExactCentroidCollectionHeader);
	return sizeof(ExactCentroidCollectionHeader) +
		   (uint64_t)c->npages * sizeof(uint32_t) +
		   (uint64_t)c->nslots * c->dim * sizeof(float);
}

void
prism_exact_centroid_collection_write(
		const PrismExactCentroidCollector *c, void *collection)
{
	ExactCentroidCollectionHeader *hdr = (ExactCentroidCollectionHeader *)
			collection;

	if (c == NULL || c->overflowed)
	{
		/* Empty collection: a zero-page region matches no block, so the view
		 * built from it leaves the scoring hook inert. */
		hdr->base	= InvalidBlockNumber;
		hdr->npages = 0;
		hdr->nslots = 0;
		hdr->dim	= 0;
		return;
	}

	hdr->base	= c->base;
	hdr->npages = c->npages;
	hdr->nslots = c->nslots;
	hdr->dim	= c->dim;

	char *p = (char *)collection + sizeof(*hdr);
	memcpy(p, c->page_off, (size_t)c->npages * sizeof(uint32_t));
	p += (size_t)c->npages * sizeof(uint32_t);
	memcpy(p, c->cents, (size_t)c->nslots * c->dim * sizeof(float));
}

void
prism_exact_centroid_collection_view(
		const void *collection, PrismExactInternalCentroids *view)
{
	const ExactCentroidCollectionHeader *hdr =
			(const ExactCentroidCollectionHeader *)collection;
	const char *p = (const char *)collection + sizeof(*hdr);

	view->page_off = (const uint32_t *)p;
	view->cents	 = (const float *)(p + (size_t)hdr->npages * sizeof(uint32_t));
	view->base	 = hdr->base;
	view->npages = hdr->npages;
}

/* Assign each node of a materialized tree its first block, packing nodes
 * in index order from first_blkno; returns the block after the last. */
BlockNumber
prism_compute_centroid_layout(
		const HKMeansResult *tree,
		uint32_t			 max_entries,
		BlockNumber			 first_blkno,
		BlockNumber			*node_first_blkno)
{
	BlockNumber next = first_blkno;

	for (uint32_t i = 0; i < tree->nnodes; i++)
	{
		uint32_t n		= hk_nodes(tree)[i].nchildren;
		uint32_t npages = (n + max_entries - 1) / max_entries;
		if (npages == 0)
			npages = 1;
		node_first_blkno[i] = next;
		next += npages;
	}

	return next;
}

/* Write a fully materialized (in-RAM) HKMeansResult tree to centroid
 * pages. The streamed builders below replace this for the paged builds;
 * the standalone in-RAM build still writes through it. */
void
prism_write_centroid_tree(
		VsStorage					*storage,
		const HKMeansResult			*tree,
		Dimension					 dim,
		uint32_t					 fan_out,
		uint8_t						 level_offset,
		PrismCentroidFormat			 centroid_format,
		const RaBitQParams			*rq_params,
		const float					*global_mean,
		BlockNumber					 posting_base,
		const BlockNumber			*node_first_blkno,
		const float					*pt_centroids,
		PrismExactCentroidCollector *collector)
{
	/* Leaf child blocks are formula-derived (posting_base + global leaf
	 * index), so no O(nlist) posting-head array is needed. Each node has at
	 * most fan_out leaf entries, so this scratch is O(fan_out). */
	BlockNumber *leaf_blks =
			(posting_base != InvalidBlockNumber)
					? vs_alloc((size_t)fan_out * sizeof(BlockNumber))
					: NULL;

	for (uint32_t i = 0; i < tree->nnodes; i++)
	{
		const HKMeansNode *node	   = &hk_nodes(tree)[i];
		bool			   is_leaf = (node->level == tree->nlevels - 1);

		uint16_t flags		 = is_leaf ? PRISM_CENTROID_FLAG_LEAF : 0;
		uint16_t child_count = is_leaf ? 0 : (uint16_t)fan_out;

		const BlockNumber *child_blks;
		if (is_leaf && leaf_blks != NULL)
		{
			for (uint32_t j = 0; j < node->nchildren; j++)
				leaf_blks[j] = posting_base + node->first_leaf + j;
			child_blks = leaf_blks;
		}
		else if (!is_leaf)
			child_blks = &node_first_blkno[node->first_child];
		else
			child_blks = NULL;

		/* Pass pt_centroids for leaf nodes only */
		const float *leaf_pt = (is_leaf && pt_centroids != NULL)
									 ? pt_centroids +
											   (size_t)node->first_leaf * dim
									 : NULL;

		prism_centroid_write_node(
				storage,
				dim,
				hk_node_centroids(tree, node),
				node->nchildren,
				centroid_format,
				(uint8_t)(level_offset + node->level),
				flags,
				child_count,
				rq_params,
				global_mean,
				child_blks,
				leaf_pt,
				node_first_blkno[i],
				collector);
	}

	if (leaf_blks != NULL)
		vs_free(leaf_blks);
}

/* ----------------------------------------------------------------
 * Streaming (page-backed) centroid-tree build
 * ---------------------------------------------------------------- */

typedef struct RoutingTreeCtx
{
	/* inputs */
	const float	  *vectors;
	Dimension	   dim;
	uint32_t	   nlist;
	uint32_t	   fan_out;
	uint32_t	   nlevels;
	DistanceMetric metric;
	KMeansOptions  opts;
	/* write-phase */
	VsStorage		   *storage;
	PrismCentroidFormat format;
	const RaBitQParams *rq_params;
	const float		   *global_mean;
	BlockNumber			first_posting; /* leaf c's head = first_posting + c */
	PrismStreamLeafCb	on_leaf;
	void			   *on_leaf_arg;
	/* write-phase: optional exact internal-centroid collection */
	PrismExactCentroidCollector *collector;
	/* plan-phase page-count helpers */
	uint32_t max_ent; /* non-fastscan entries/page */
	uint32_t fs_gpp;  /* fastscan groups/page */
	/* Node replay: the plan pass records each node's clustering
	 * ({k, centroids, assignments}, recursion order) here, and the write
	 * pass replays it instead of re-running k-means. Both passes require
	 * the store. */
	PrismBlobStore *store;
	float		   *replay_cents; /* write: [fan_out * dim] */
	/* One node's uint16 assignments, shared by both passes (the plan pass
	 * narrows into it before recording, the replay pass reads records back
	 * into it). Sized once for the root -- the first and largest node --
	 * so it is never grown. */
	uint16_t	*assign_scratch;   /* [nvecs] */
	BlockNumber *leaf_blk_scratch; /* write: [fan_out] leaf head blocks */
	/* accumulators */
	uint32_t nleaves;			/* running (== next first_leaf) */
	double	*leaf_sum;			/* plan only, [dim]: leaf-centroid sum for the
								 * leaf_mean the caller uses as global_mean */
	uint32_t	centroid_pages; /* plan only */
	BlockNumber next_blk;		/* write only: next reserved centroid block */
	bool		ok;
} RoutingTreeCtx;

/*
 * Write one tree node's centroid page(s) in the node's format: the fastscan
 * writer carries its own encoder; every other format goes through the
 * generic encoder + page writer. Shared by the streaming DFS and the
 * parallel leader's root-page write.
 */
void
prism_centroid_write_node(
		VsStorage					*storage,
		Dimension					 dim,
		const float					*cents,
		uint32_t					 n,
		PrismCentroidFormat			 fmt,
		uint8_t						 level,
		uint16_t					 flags,
		uint16_t					 child_count,
		const RaBitQParams			*rq_params,
		const float					*global_mean,
		const BlockNumber			*child_blks,
		const float					*leaf_pt,
		BlockNumber					 blkno,
		PrismExactCentroidCollector *collector)
{
	/* Every internal node of every build shape passes through here; the
	 * leaf level (LEAF flag) is exact-re-ranked per row instead. */
	if (collector != NULL && (flags & PRISM_CENTROID_FLAG_LEAF) == 0)
		prism_exact_centroid_collector_add_node(collector, blkno, cents, n);

	if (fmt == PRISM_CENTROID_FMT_FASTSCAN)
		/* fastscan carries its own encoder; pt_centroids live on posting
		 * pages, never inline. */
		prism_centroid_write_fastscan_pages(
				storage,
				dim,
				n,
				level,
				flags,
				rq_params,
				cents,
				global_mean,
				child_blks,
				blkno);
	else
	{
		CentroidEncoderState est;
		CentroidEncoder		*enc = centroid_encoder_init(
				&est, fmt, cents, dim, rq_params, global_mean);
		prism_centroid_write_pages(
				storage,
				dim,
				n,
				fmt,
				level,
				flags,
				child_count,
				enc,
				child_blks,
				leaf_pt,
				blkno);
	}
}

/* Pages one node of `n` entries occupies — must match the writers' packing. */
static uint32_t
node_npages(const RoutingTreeCtx *c, uint32_t n)
{
	if (c->format == PRISM_CENTROID_FMT_FASTSCAN)
	{
		uint32_t ngroups = (n + VS_FASTSCAN_GROUP - 1) / VS_FASTSCAN_GROUP;
		uint32_t gpp	 = c->fs_gpp ? c->fs_gpp : 1;
		return (ngroups + gpp - 1) / gpp;
	}
	uint32_t me = c->max_ent ? c->max_ent : 1;
	return (n + me - 1) / me;
}

/*
 * Write one node's centroid page(s) at the next reserved block (post-order, so
 * blocks are assigned in write order); returns the node's first block. The
 * relation is pre-extended by the caller to cover the centroid + posting area,
 * so writes use reserved blocks rather than appending (posting heads, which
 * live in the far posting area, are written during this pass too).
 */
static BlockNumber
write_node_pages(
		RoutingTreeCtx	  *c,
		const float		  *cents,
		uint32_t		   n,
		uint32_t		   level,
		bool			   is_leaf,
		const BlockNumber *child_blks)
{
	uint16_t	flags = is_leaf ? PRISM_CENTROID_FLAG_LEAF : 0;
	BlockNumber start = c->next_blk;

	prism_centroid_write_node(
			c->storage,
			c->dim,
			cents,
			n,
			c->format,
			(uint8_t)level,
			flags,
			is_leaf ? 0 : (uint16_t)c->fan_out,
			c->rq_params,
			c->global_mean,
			child_blks,
			NULL,
			start,
			c->collector);

	c->next_blk += node_npages(c, n);
	return start;
}

/*
 * DFS one node. slice[count] are indices into c->vectors (NULL == identity for
 * the root). Returns the node's first block (write phase) or
 * InvalidBlockNumber (plan phase). Post-order: children are written before the
 * parent so the parent's entries can carry their child block numbers.
 */
/* Read exactly nbytes of a node record from the replay store. */
static bool
replay_read(struct PrismBlobStore *store, void *dst, uint64_t nbytes)
{
	return prism_pbuild_blobstore_get(store, dst, nbytes) == nbytes;
}

/*
 * Replay one node's clustering record from the blob store into the ctx's
 * replay scratch. The plan pass wrote, per node in recursion order: the
 * cluster count k (uint32), the centroids (k * dim floats) and the
 * per-vector assignments (count uint16s); this reads the same three fields
 * in the same order. Returns false on a short read or an out-of-range k,
 * which can only mean the plan and write passes diverged.
 */
static bool
replay_node_record(RoutingTreeCtx *c, uint32_t count, uint32_t *out_k)
{
	uint32_t k = 0;

	if (!replay_read(c->store, &k, sizeof(k)))
		return false;
	if (k == 0 || k > c->fan_out)
		return false;
	if (!replay_read(
				c->store,
				c->replay_cents,
				(uint64_t)k * c->dim * sizeof(float)))
		return false;
	if (!replay_read(
				c->store,
				c->assign_scratch,
				(uint64_t)count * sizeof(uint16_t)))
		return false;

	*out_k = k;
	return true;
}

/*
 * Record one node's clustering into the blob store, in the exact field
 * order replay_node_record reads back: k, centroids[k * dim],
 * assignments[count]. Assignments are narrowed to uint16 -- a node
 * clusters into at most fan_out (<= the meta page's 8-bit fan_out) groups.
 */
static void
record_node_clustering(
		RoutingTreeCtx *c, const KMeansResult *km, uint32_t count)
{
	const size_t assign_nbytes = (size_t)count * sizeof(uint16_t);
	uint16_t	*a16		   = c->assign_scratch;

	for (uint32_t v = 0; v < count; v++)
		a16[v] = (uint16_t)km->assignments[v];

	prism_pbuild_blobstore_put(c->store, &km->nlist, sizeof(km->nlist));
	prism_pbuild_blobstore_put(
			c->store,
			km->centroids,
			(uint64_t)km->nlist * c->dim * sizeof(float));
	prism_pbuild_blobstore_put(c->store, a16, assign_nbytes);
}

/* Compact the non-empty clusters' centroids to the front, preserving
 * ascending cluster order (descent indexes the kept centroids densely). */
static uint32_t
compact_nonempty_centroids(
		float		   *cents,
		const uint32_t *counts,
		uint32_t		nclusters,
		Dimension		dim)
{
	uint32_t kept = 0;
	for (uint32_t cl = 0; cl < nclusters; cl++)
	{
		if (counts[cl] == 0)
			continue;
		if (kept != cl)
			memcpy(cents + (size_t)kept * dim,
				   cents + (size_t)cl * dim,
				   (size_t)dim * sizeof(float));
		kept++;
	}
	return kept;
}

static void plan_node_recurse(
		RoutingTreeCtx *c,
		const uint32_t *slice,
		uint32_t		count,
		uint32_t		level);
static BlockNumber
replay_node_recurse(RoutingTreeCtx *c, uint32_t count, uint32_t level);

/* Leaf parent, plan side: the non-empty clusters become leaves. Nothing is
 * written; count the leaf pages, tally the leaves, and fold the kept
 * centroids into the leaf sum (the global mean's numerator). */
static void
plan_leaf_parent(
		RoutingTreeCtx *c, const KMeansResult *km, const uint32_t *counts)
{
	uint32_t kept = 0;
	for (uint32_t cl = 0; cl < km->nlist; cl++)
	{
		if (counts[cl] == 0)
			continue;
		if (c->leaf_sum != NULL)
		{
			const float *lc = km->centroids + (size_t)cl * c->dim;
			for (Dimension d = 0; d < c->dim; d++)
				c->leaf_sum[d] += lc[d];
		}
		kept++;
	}
	c->centroid_pages += node_npages(c, kept);
	c->nleaves += kept;
}

/* Internal node, plan side: gather each non-empty child's slice of the
 * sample and recurse, in ascending cluster order -- the recursion order
 * that frames the tape -- then count this node's page(s). */
static void
plan_internal_node(
		RoutingTreeCtx	   *c,
		const KMeansResult *km,
		const uint32_t	   *counts,
		const uint32_t	   *slice,
		uint32_t			count,
		uint32_t			level)
{
	uint32_t kept = 0;
	for (uint32_t cl = 0; cl < km->nlist && c->ok; cl++)
	{
		if (counts[cl] == 0)
			continue;

		uint32_t  sub_n = counts[cl];
		uint32_t *sub	= vs_alloc((size_t)sub_n * sizeof(uint32_t));
		uint32_t  idx	= 0;
		for (uint32_t v = 0; v < count; v++)
			if (km->assignments[v] == cl)
				sub[idx++] = slice ? slice[v] : v;
		plan_node_recurse(c, sub, sub_n, level + 1);
		vs_free(sub);
		kept++;
	}
	c->centroid_pages += node_npages(c, kept);
}

/*
 * Plan pass -- writes the tape. Recursive DFS from the given node down
 * (the mutual recursion runs through plan_internal_node): cluster the
 * node's slice of the sample (slice[count] indexes c->vectors; NULL =
 * identity at the root), record the clustering for the replay pass,
 * descend into the non-empty children, and accumulate the layout totals
 * (centroid pages, leaves, the leaf-centroid sum) that must be known
 * before any page can be written.
 */
static void
plan_node_recurse(
		RoutingTreeCtx *c,
		const uint32_t *slice,
		uint32_t		count,
		uint32_t		level)
{
	if (!c->ok)
		return;

	uint32_t k = (c->nlevels == 1) ? c->nlist : c->fan_out;
	if (k > count)
		k = count;

	KMeansResult *km = vs_kmeans(
			c->vectors,
			slice,
			VS_VEC_F32,
			count,
			c->dim,
			k,
			c->metric,
			&c->opts);
	if (km == NULL)
	{
		c->ok = false;
		return;
	}
	c->opts.initial_centroids = NULL; /* root only (matches vs_hkmeans_f32) */

	if (c->store != NULL)
		record_node_clustering(c, km, count);

	/* Count assignments per cluster (km->cluster_sizes may be stale). */
	uint32_t *counts = vs_alloc0((size_t)km->nlist * sizeof(uint32_t));
	for (uint32_t v = 0; v < count; v++)
		counts[km->assignments[v]]++;

	if (level == c->nlevels - 1)
		plan_leaf_parent(c, km, counts);
	else
		plan_internal_node(c, km, counts, slice, count, level);

	vs_free(counts);
	vs_kmeans_result_destroy(km);
}

/* Leaf parent, replay side: compact the non-empty clusters into leaves,
 * write the node's page(s) with the formula-derived leaf head blocks, and
 * emit each leaf's head page from its resident float centroid. */
static BlockNumber
replay_leaf_parent(
		RoutingTreeCtx *c,
		uint32_t		nclusters,
		const uint32_t *counts,
		uint32_t		level)
{
	float	*cents = c->replay_cents;
	uint32_t kept =
			compact_nonempty_centroids(cents, counts, nclusters, c->dim);

	/* Cosine routes and encodes against unit-norm references; k-means
	 * means drift below unit norm, so normalize the leaf centroids in
	 * place before the leaf codes and head references are written from
	 * them (the flat-tree path already does). */
	if (c->metric == DISTANCE_COSINE)
		for (uint32_t kk = 0; kk < kept; kk++)
			vs_l2_normalize(cents + (size_t)kk * c->dim, c->dim);

	/* Leaf heads are formula-derived (first_posting + global leaf index);
	 * kept <= fan_out, so the pass-lifetime scratch covers every node. */
	BlockNumber *leaf_blks = NULL;
	if (c->first_posting != InvalidBlockNumber)
	{
		leaf_blks = c->leaf_blk_scratch;
		for (uint32_t kk = 0; kk < kept; kk++)
			leaf_blks[kk] = c->first_posting + c->nleaves + kk;
	}
	BlockNumber blk = write_node_pages(c, cents, kept, level, true, leaf_blks);

	if (c->on_leaf != NULL)
		for (uint32_t kk = 0; kk < kept; kk++)
			c->on_leaf(
					c->on_leaf_arg,
					c->nleaves + kk,
					cents + (size_t)kk * c->dim);
	c->nleaves += kept;
	return blk;
}

/* Internal node, replay side: recurse into each non-empty child in
 * ascending cluster order -- the same order the plan pass recursed, which
 * is what keeps the tape in frame -- then write this node's page(s) above
 * the children. The children's records overwrite the shared replay
 * scratch, so this node's centroids are snapshotted first. */
static BlockNumber
replay_internal_node(
		RoutingTreeCtx *c,
		uint32_t		nclusters,
		const uint32_t *counts,
		uint32_t		level)
{
	const size_t vec_nbytes	  = (size_t)c->dim * sizeof(float);
	const size_t cents_nbytes = (size_t)nclusters * vec_nbytes;

	float *my_cents = vs_alloc(cents_nbytes);
	memcpy(my_cents, c->replay_cents, cents_nbytes);

	BlockNumber *child_blocks = vs_alloc(
			(size_t)nclusters * sizeof(BlockNumber));
	uint32_t kept = 0;
	for (uint32_t cl = 0; cl < nclusters && c->ok; cl++)
	{
		if (counts[cl] == 0)
			continue;
		if (kept != cl)
			memcpy(my_cents + (size_t)kept * c->dim,
				   my_cents + (size_t)cl * c->dim,
				   vec_nbytes);
		child_blocks[kept] = replay_node_recurse(c, counts[cl], level + 1);
		kept++;
	}

	BlockNumber blk = InvalidBlockNumber;
	if (c->ok)
		blk = write_node_pages(c, my_cents, kept, level, false, child_blocks);
	vs_free(child_blocks);
	vs_free(my_cents);
	return blk;
}

/*
 * Replay pass -- reads the tape. Recursive DFS from the given node down
 * (the mutual recursion runs through replay_internal_node): consume the
 * record the plan pass wrote for this node and write its page(s). The
 * record carries the centroids and the assignments, and each child's size
 * falls out of the assignment counts, so unlike the plan walker this one
 * needs no sample vectors and descends on counts alone. Post-order:
 * children are written before the parent so the parent's entries can carry
 * their block numbers. Returns the node's first block.
 */
static BlockNumber
replay_node_recurse(RoutingTreeCtx *c, uint32_t count, uint32_t level)
{
	if (!c->ok)
		return InvalidBlockNumber;

	uint32_t nclusters = 0;
	if (!replay_node_record(c, count, &nclusters))
	{
		c->ok = false;
		return InvalidBlockNumber;
	}

	/* Count assignments per cluster. */
	uint32_t *counts = vs_alloc0((size_t)nclusters * sizeof(uint32_t));
	for (uint32_t v = 0; v < count; v++)
		counts[c->assign_scratch[v]]++;

	BlockNumber blk;
	if (level == c->nlevels - 1)
		blk = replay_leaf_parent(c, nclusters, counts, level);
	else
		blk = replay_internal_node(c, nclusters, counts, level);

	vs_free(counts);
	return blk;
}

/* Shared walker state: tree geometry and page capacities. The plan entry
 * adds the sample/metric/k-means knobs; the replay entry adds the storage,
 * tape and replay scratch. */
static void
routing_tree_ctx_init(
		RoutingTreeCtx	   *c,
		Dimension			dim,
		uint32_t			nlist,
		uint32_t			fan_out,
		PrismCentroidFormat format)
{
	memset(c, 0, sizeof(*c));
	c->dim	   = dim;
	c->nlist   = nlist;
	c->fan_out = fan_out < 2 ? 2 : fan_out;
	c->nlevels = vs_hkmeans_nlevels(nlist, c->fan_out);
	c->format  = format;
	c->max_ent = prism_centroid_max_entries_fmt(dim, format);
	c->fs_gpp  = prism_centroid_fastscan_max_groups(dim);
	c->ok	   = true;
}

/* Plan-pass entry ("write the tape"): cluster the sample once, record
 * every node, and return the layout totals -- see the header comment. */
bool
prism_routing_tree_plan(
		const float			*vectors,
		uint32_t			 nvecs,
		Dimension			 dim,
		uint32_t			 nlist,
		uint32_t			 fan_out,
		DistanceMetric		 metric,
		PrismCentroidFormat	 format,
		const KMeansOptions *opts,
		PrismBlobStore		*store,
		PrismStreamTreePlan *out)
{
	RoutingTreeCtx c;
	routing_tree_ctx_init(&c, dim, nlist, fan_out, format);
	/* Only the plan pass clusters, so only it carries the sample, the
	 * metric and the k-means options. */
	c.vectors  = vectors;
	c.metric   = metric;
	c.opts	   = opts ? *opts : (KMeansOptions)VS_KMEANS_OPTIONS_DEFAULT;
	c.store	   = store;
	c.leaf_sum = vs_alloc0((size_t)dim * sizeof(double));
	if (store != NULL)
	{
		/* Recording narrows each node's assignments into this scratch; the
		 * root -- first and largest -- sizes it for the whole pass. */
		c.assign_scratch = vs_alloc((size_t)nvecs * sizeof(uint16_t));
	}

	/*
	 * The recursion's per-node scratch (k-means temporaries, index slices,
	 * child-block arrays) lives in its own context so anything a node fails
	 * to free is reclaimed here, at the end of the pass, rather than
	 * accumulating for the rest of the build. This pass-scoped context is
	 * the deliberate leak-containment boundary: node lifetimes nest (a
	 * parent's k-means result and child arrays stay live across its
	 * children), so finer-grained reclamation such as resets at node or
	 * sibling boundaries would free live ancestor state — and would turn a
	 * missed free (a bounded, observable leak) into a use-after-free.
	 * Within a node, explicit frees remain the mechanism; the context caps
	 * their blast radius at one pass. The output (leaf_mean) is allocated
	 * in the caller's context outside the switch.
	 */
	VsMemCtx scratch = vs_memctx_create(NULL, "vs stream plan");
	VsMemCtx old_ctx = vs_memctx_switch(scratch);
	plan_node_recurse(&c, NULL, nvecs, 0);
	vs_memctx_switch(old_ctx);
	vs_memctx_delete(scratch);

	if (c.assign_scratch != NULL)
		vs_free(c.assign_scratch);
	if (!c.ok)
	{
		vs_free(c.leaf_sum);
		return false;
	}

	out->nleaves		= c.nleaves;
	out->nlevels		= c.nlevels;
	out->centroid_pages = c.centroid_pages;
	out->leaf_mean		= vs_alloc((size_t)dim * sizeof(float));
	for (Dimension d = 0; d < dim; d++)
		out->leaf_mean[d] = c.nleaves > 0
								  ? (float)(c.leaf_sum[d] / (double)c.nleaves)
								  : 0.0f;
	vs_free(c.leaf_sum);
	return true;
}

/* Replay-pass entry ("read the tape"): stream every centroid + head page
 * from the recorded nodes; returns the root block. */
BlockNumber
prism_routing_tree_write(
		VsStorage					*storage,
		uint32_t					 nvecs,
		Dimension					 dim,
		DistanceMetric				 metric,
		uint32_t					 nlist,
		uint32_t					 fan_out,
		PrismCentroidFormat			 format,
		const RaBitQParams			*rq_params,
		const float					*global_mean,
		PrismBlobStore				*store,
		BlockNumber					 first_posting,
		BlockNumber					 first_centroid,
		PrismStreamLeafCb			 on_leaf,
		void						*on_leaf_arg,
		PrismExactCentroidCollector *collector)
{
	/* Replay never clusters: everything it needs is on the tape, so it
	 * takes no sample vectors, metric or k-means options. */
	if (store == NULL)
		return InvalidBlockNumber;

	RoutingTreeCtx c;
	routing_tree_ctx_init(&c, dim, nlist, fan_out, format);
	c.metric  = metric;
	c.storage = storage;
	c.store	  = store;
	{
		/* A node clusters into at most fan_out groups (the flat root's k =
		 * nlist <= fan_out). The assignment scratch covers the root's full
		 * sample; deeper slices are strictly smaller. */
		c.replay_cents	   = vs_alloc((size_t)c.fan_out * dim * sizeof(float));
		c.assign_scratch   = vs_alloc((size_t)nvecs * sizeof(uint16_t));
		c.leaf_blk_scratch = vs_alloc((size_t)c.fan_out * sizeof(BlockNumber));
	}
	c.rq_params		= rq_params;
	c.global_mean	= global_mean;
	c.first_posting = first_posting;
	c.next_blk		= first_centroid;
	c.on_leaf		= on_leaf;
	c.on_leaf_arg	= on_leaf_arg;
	c.collector		= collector;

	/* Same pass-scoped scratch as the PLAN pass -- the leak-containment
	 * boundary (see the note there). on_leaf runs under it too; its
	 * allocations must not outlive the call. */
	VsMemCtx	scratch = vs_memctx_create(NULL, "vs stream write");
	VsMemCtx	old_ctx = vs_memctx_switch(scratch);
	BlockNumber root	= replay_node_recurse(&c, nvecs, 0);
	vs_memctx_switch(old_ctx);
	vs_memctx_delete(scratch);
	if (c.replay_cents != NULL)
		vs_free(c.replay_cents);
	if (c.assign_scratch != NULL)
		vs_free(c.assign_scratch);
	if (c.leaf_blk_scratch != NULL)
		vs_free(c.leaf_blk_scratch);
	return c.ok ? root : InvalidBlockNumber;
}

/* Parallel build: write one worker-built (materialized) subtree's pages at
 * its reserved block range, mapping its leaves into the global leaf index
 * space -- see the header comment. */
BlockNumber
prism_routing_subtree_write(
		VsStorage					*storage,
		const HKMeansResult			*subtree,
		Dimension					 dim,
		DistanceMetric				 metric,
		uint32_t					 fan_out,
		uint8_t						 level_offset,
		PrismCentroidFormat			 format,
		const RaBitQParams			*rq_params,
		const float					*global_mean,
		BlockNumber					 first_posting,
		uint32_t					 leaf_offset,
		BlockNumber					 first_block,
		PrismStreamLeafCb			 on_leaf,
		void						*on_leaf_arg,
		PrismExactCentroidCollector *collector)
{
	uint32_t max_ent = prism_centroid_max_entries_fmt(dim, format);

	/* Lay the subtree's nodes out at reserved blocks starting at first_block
	 * (BFS: node 0 = subtree root at first_block). */
	BlockNumber *nfb = vs_alloc((size_t)subtree->nnodes * sizeof(BlockNumber));
	(void)prism_compute_centroid_layout(subtree, max_ent, first_block, nfb);

	/* Leaf entries link to formula-derived posting heads (first_posting +
	 * global leaf index); leaf_offset maps the subtree's local leaf indices to
	 * the global index space. */
	/* Subtree node levels are subtree-relative; level_offset places them at
	 * their absolute depth (the tree root above them is level 0). */
	/* Cosine: normalize the subtree's leaf centroids in place before the
	 * leaf codes and head references are written from them (matches the
	 * flat-tree path; k-means means drift below unit norm). */
	if (metric == DISTANCE_COSINE)
	{
		float *lv = hk_leaf_centroids(subtree);
		for (uint32_t li = 0; li < subtree->nleaves; li++)
			vs_l2_normalize(lv + (size_t)li * dim, dim);
	}

	prism_write_centroid_tree(
			storage,
			subtree,
			dim,
			fan_out,
			level_offset,
			format,
			rq_params,
			global_mean,
			first_posting + leaf_offset,
			nfb,
			NULL,
			collector);
	vs_free(nfb);

	/* Head pages carry pt_centroid from the resident float leaf centroids. */
	if (on_leaf != NULL)
	{
		const float *leaves = hk_leaf_centroids(subtree);
		for (uint32_t i = 0; i < subtree->nleaves; i++)
			on_leaf(on_leaf_arg, leaf_offset + i, leaves + (size_t)i * dim);
	}

	return first_block;
}

/* Resolve the tree fan-out when the user left it at the default: sqrt of
 * the partition count, falling to cbrt when that exceeds a page's worth of
 * children. */
uint32_t
prism_auto_fan_out(uint32_t fan_out, uint32_t nlist, uint32_t default_fan_out)
{
	if (fan_out != default_fan_out || nlist <= fan_out)
		return (nlist <= fan_out) ? nlist : fan_out;

	uint32_t f = (uint32_t)ceil(sqrt((double)nlist));
	if (f > 256)
		f = (uint32_t)ceil(cbrt((double)nlist));
	return f;
}

/* Automatic partition count from the (estimated) row count. */
uint32_t
prism_auto_nlist(double count)
{
	/*
	 * One list per PRISM_TARGET_ENTRIES_PER_LIST vectors -- see the constant
	 * for why that is the target. The hierarchical centroid tree keeps routing
	 * cheap even at a high list count, so nlist scales linearly with the row
	 * count.
	 *
	 * Floor it at sqrt(count): for small tables count/target collapses toward
	 * a single list, which under-partitions and can starve the k-means build.
	 * The linear target overtakes the sqrt floor at target^2 rows (65536 at
	 * the current target). Back-ends clamp the result to their own nlist
	 * ceiling.
	 */
	double	 c		   = count > 1.0 ? count : 1.0;
	uint32_t linear	   = (uint32_t)(c / (double)PRISM_TARGET_ENTRIES_PER_LIST +
									0.5);
	uint32_t min_lists = (uint32_t)sqrt(c);
	uint32_t nlist	   = linear > min_lists ? linear : min_lists;
	return nlist < 1 ? 1 : nlist;
}

/*
 * Vectors per posting list at a given row count -- the resting size
 * maintenance aims each list at. Inverts prism_auto_nlist so the two cannot
 * drift apart; see the header for the sqrt-floor regime and the nlist == 0
 * convention.
 */
uint32_t
prism_target_entries_per_list(double count, uint32_t nlist)
{
	double c = count > 1.0 ? count : 1.0;

	if (nlist == 0)
		nlist = prism_auto_nlist(c);
	if (nlist < 1)
		nlist = 1;

	uint32_t per_list = (uint32_t)(c / (double)nlist + 0.5);
	return per_list < 1 ? 1 : per_list;
}

/* 2nd-nearest cluster from a distance-sorted candidate list, or
 * primary_cluster when the runner-up is not within epsilon (no replication
 * warranted) -- see the header comment. */
uint32_t
prism_find_secondary_cluster(
		const uint32_t *cand_leaves,
		const Distance *cand_dists,
		uint32_t		ncand,
		uint32_t		primary_cluster,
		Distance		primary_dist,
		double			epsilon)
{
	/* Candidates are sorted by ascending distance, so the first one
	 * that is not the primary is the 2nd-nearest centroid. */
	for (uint32_t i = 0; i < ncand; i++)
	{
		if (cand_leaves[i] == primary_cluster)
			continue;

		Distance best2	   = cand_dists[i];
		double	 gap	   = (double)best2 - (double)primary_dist;
		double	 gap_ratio = (primary_dist != 0.0)
								   ? gap / fabs((double)primary_dist)
								   : INFINITY;

		if (gap_ratio <= epsilon)
			return cand_leaves[i];
		break;
	}

	return primary_cluster;
}

/*
 * SOAR secondary cluster: the leaf (other than the primary) minimizing the
 * orthogonality-amplified distance, decomposed so both terms use the SIMD
 * vecops kernels:
 *
 *   oa(c) = ||v - c||^2 + lambda * (r_hat . (v - c))^2
 *         = ||v - c||^2 + lambda * (r_hat.v - r_hat.c)^2
 *
 * r_hat.v is constant across centroids, so only ||v - c||^2 and r_hat.c are
 * per-centroid. Since lambda * (...)^2 >= 0, ||v - c||^2 is a lower bound on
 * oa: when it already exceeds the running best we skip the dot product (exact
 * pruning — no recall impact).
 *
 * The search set is `count` leaves: the ids cand_leaves[0..count) when
 * cand_leaves is non-NULL, otherwise leaves 0..count (a full scan, called with
 * count == nleaves). The candidate form is O(count) rather than O(nleaves);
 * because the minimizer always has a small ||v - c||^2 (a far leaf cannot win)
 * it lies among the nearest leaves the beam already found, so the result is
 * unchanged while scaling to large nlist — where a full scan would read the
 * entire (multi-hundred-MB) leaf-centroid array per replicated vector.
 */
uint32_t
prism_find_soar_secondary(
		const float	   *vec,
		const float	   *leaf_centroids,
		const uint32_t *cand_leaves,
		uint32_t		count,
		Dimension		dim,
		uint32_t		primary_cluster,
		const float	   *normalized_residual,
		double			lambda)
{
	float	 qrv	 = vs_dot_product(normalized_residual, vec, dim);
	float	 lam	 = (float)lambda;
	float	 best_oa = INFINITY;
	uint32_t best_c	 = primary_cluster;

	for (uint32_t k = 0; k < count; k++)
	{
		uint32_t i = cand_leaves ? cand_leaves[k] : k;
		if (i == primary_cluster)
			continue;

		const float *cent = leaf_centroids + (size_t)i * dim;

		float l2 = vs_l2_distance_squared(vec, cent, dim);
		if (l2 >= best_oa)
			continue; /* oa >= l2 >= best_oa: cannot improve */

		float rc  = vs_dot_product(normalized_residual, cent, dim);
		float gap = qrv - rc;
		float oa  = l2 + lam * gap * gap;
		if (oa < best_oa)
		{
			best_oa = oa;
			best_c	= i;
		}
	}

	return best_c;
}

/* Log the per-phase build time summary. */
void
prism_build_stats_print(const PrismBuildStats *s)
{
	/* Per-phase breakdown — the single shared build summary, filled by both
	 * the PostgreSQL build (via the build-progress seam) and the standalone
	 * build. */
	vs_log("build: sample %.1fms, kmeans %.1fms, refine %.1fms, setup "
		   "%.1fms, "
		   "posting %.1fms, centroid %.1fms, total %.1fms\n",
		   s->ms_sample,
		   s->ms_kmeans,
		   s->ms_refine,
		   s->ms_setup,
		   s->ms_posting,
		   s->ms_centroid,
		   s->ms_total);

	/* Posting-merge sub-detail: only the standalone parallel path populates
	 * these. Skip the line (and its zeros) when unset — e.g. the PostgreSQL
	 * build, which tracks posting as one phase. */
	if (s->ms_parallel > 0.0 || s->ms_merge > 0.0 || s->total_pages > 0)
		vs_log("build: posting detail — parallel %.1fms + merge %.1fms, "
			   "%u workers + leader, %u pages, "
			   "%u partial pages merged into %u\n",
			   s->ms_parallel,
			   s->ms_merge,
			   s->nworkers,
			   s->total_pages,
			   s->merge_input,
			   s->merge_output);
}
