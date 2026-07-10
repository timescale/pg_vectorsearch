/*
 * index_build.c - Shared index build utilities
 *
 * Generic helpers for building meerkat indexes, usable from both
 * the PostgreSQL IAM build and the standalone CLI.
 */

#include <math.h>
#include <string.h>

#include "algo/kmeans.h"
#include "algo/vecops.h"
#include "core/log.h"
#include "core/memory.h"
#include "index/centroid_build.h"
#include "index/index_build.h"
#include "index/parallel_build.h" /* MktBlobStore seam (plan/write replay) */
#include "quant/fastscan.h"

/* Assign each node of a materialized tree its first block, packing nodes
 * in index order from first_blkno; returns the block after the last. */
BlockNumber
mkt_compute_centroid_layout(
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
mkt_write_centroid_tree(
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
		const float			*pt_centroids)
{
	/* Leaf child blocks are formula-derived (posting_base + global leaf
	 * index), so no O(nlist) posting-head array is needed. Each node has at
	 * most fan_out leaf entries, so this scratch is O(fan_out). */
	BlockNumber *leaf_blks =
			(posting_base != InvalidBlockNumber)
					? mkt_alloc((size_t)fan_out * sizeof(BlockNumber))
					: NULL;

	for (uint32_t i = 0; i < tree->nnodes; i++)
	{
		const HKMeansNode *node	   = &hk_nodes(tree)[i];
		bool			   is_leaf = (node->level == tree->nlevels - 1);

		uint16_t flags		 = is_leaf ? MKT_CENTROID_FLAG_LEAF : 0;
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

		if (centroid_format == MKT_CENTROID_FMT_FASTSCAN)
		{
			(void)child_count;
			(void)pt_centroids; /* pt_centroids live on posting pages */
			mkt_centroid_write_fastscan_pages(
					storage,
					dim,
					node->nchildren,
					(uint8_t)(level_offset + node->level),
					flags,
					rq_params,
					hk_node_centroids(tree, node),
					global_mean,
					child_blks,
					node_first_blkno[i]);
			continue;
		}

		CentroidEncoderState enc_state;
		CentroidEncoder		*encoder = centroid_encoder_init(
				&enc_state,
				centroid_format,
				hk_node_centroids(tree, node),
				dim,
				rq_params,
				global_mean);

		/* Pass pt_centroids for leaf nodes only */
		const float *leaf_pt = (is_leaf && pt_centroids != NULL)
									 ? pt_centroids +
											   (size_t)node->first_leaf * dim
									 : NULL;

		mkt_centroid_write_pages(
				storage,
				dim,
				node->nchildren,
				centroid_format,
				(uint8_t)(level_offset + node->level),
				flags,
				child_count,
				encoder,
				child_blks,
				leaf_pt,
				node_first_blkno[i]);
	}

	if (leaf_blks != NULL)
		mkt_free(leaf_blks);
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
	MktStorage		   *storage;
	MktCentroidFormat	format;
	const RaBitQParams *rq_params;
	const float		   *global_mean;
	BlockNumber			first_posting; /* leaf c's head = first_posting + c */
	MktStreamLeafCb		on_leaf;
	void			   *on_leaf_arg;
	/* plan-phase page-count helpers */
	uint32_t max_ent; /* non-fastscan entries/page */
	uint32_t fs_gpp;  /* fastscan groups/page */
	/* Node replay: the plan pass records each node's clustering
	 * ({k, centroids, assignments}, recursion order) here, and the write
	 * pass replays it instead of re-running k-means. Both passes require
	 * the store. */
	MktBlobStore *store;
	float		 *replay_cents; /* write: [fan_out * dim] */
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
mkt_centroid_write_node(
		MktStorage		   *storage,
		Dimension			dim,
		const float		   *cents,
		uint32_t			n,
		MktCentroidFormat	fmt,
		uint8_t				level,
		uint16_t			flags,
		uint16_t			child_count,
		const RaBitQParams *rq_params,
		const float		   *global_mean,
		const BlockNumber  *child_blks,
		BlockNumber			blkno)
{
	if (fmt == MKT_CENTROID_FMT_FASTSCAN)
		mkt_centroid_write_fastscan_pages(
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
		mkt_centroid_write_pages(
				storage,
				dim,
				n,
				fmt,
				level,
				flags,
				child_count,
				enc,
				child_blks,
				NULL,
				blkno);
	}
}

/* Pages one node of `n` entries occupies — must match the writers' packing. */
static uint32_t
node_npages(const RoutingTreeCtx *c, uint32_t n)
{
	if (c->format == MKT_CENTROID_FMT_FASTSCAN)
	{
		uint32_t ngroups = (n + MKT_FASTSCAN_GROUP - 1) / MKT_FASTSCAN_GROUP;
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
	uint16_t	flags = is_leaf ? MKT_CENTROID_FLAG_LEAF : 0;
	BlockNumber start = c->next_blk;

	mkt_centroid_write_node(
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
			start);

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
replay_read(struct MktBlobStore *store, void *dst, uint64_t nbytes)
{
	return mkt_pbuild_blobstore_get(store, dst, nbytes) == nbytes;
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

	mkt_pbuild_blobstore_put(c->store, &km->nlist, sizeof(km->nlist));
	mkt_pbuild_blobstore_put(
			c->store,
			km->centroids,
			(uint64_t)km->nlist * c->dim * sizeof(float));
	mkt_pbuild_blobstore_put(c->store, a16, assign_nbytes);
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
		uint32_t *sub	= mkt_alloc((size_t)sub_n * sizeof(uint32_t));
		uint32_t  idx	= 0;
		for (uint32_t v = 0; v < count; v++)
			if (km->assignments[v] == cl)
				sub[idx++] = slice ? slice[v] : v;
		plan_node_recurse(c, sub, sub_n, level + 1);
		mkt_free(sub);
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

	KMeansResult *km = mkt_kmeans(
			c->vectors,
			slice,
			MKT_VEC_F32,
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
	c->opts.initial_centroids = NULL; /* root only (matches mkt_hkmeans_f32) */

	if (c->store != NULL)
		record_node_clustering(c, km, count);

	/* Count assignments per cluster (km->cluster_sizes may be stale). */
	uint32_t *counts = mkt_alloc0((size_t)km->nlist * sizeof(uint32_t));
	for (uint32_t v = 0; v < count; v++)
		counts[km->assignments[v]]++;

	if (level == c->nlevels - 1)
		plan_leaf_parent(c, km, counts);
	else
		plan_internal_node(c, km, counts, slice, count, level);

	mkt_free(counts);
	mkt_kmeans_result_destroy(km);
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

	float *my_cents = mkt_alloc(cents_nbytes);
	memcpy(my_cents, c->replay_cents, cents_nbytes);

	BlockNumber *child_blocks = mkt_alloc(
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
	mkt_free(child_blocks);
	mkt_free(my_cents);
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
	uint32_t *counts = mkt_alloc0((size_t)nclusters * sizeof(uint32_t));
	for (uint32_t v = 0; v < count; v++)
		counts[c->assign_scratch[v]]++;

	BlockNumber blk;
	if (level == c->nlevels - 1)
		blk = replay_leaf_parent(c, nclusters, counts, level);
	else
		blk = replay_internal_node(c, nclusters, counts, level);

	mkt_free(counts);
	return blk;
}

/* Shared walker state: tree geometry and page capacities. The plan entry
 * adds the sample/metric/k-means knobs; the replay entry adds the storage,
 * tape and replay scratch. */
static void
routing_tree_ctx_init(
		RoutingTreeCtx	 *c,
		Dimension		  dim,
		uint32_t		  nlist,
		uint32_t		  fan_out,
		MktCentroidFormat format)
{
	memset(c, 0, sizeof(*c));
	c->dim	   = dim;
	c->nlist   = nlist;
	c->fan_out = fan_out < 2 ? 2 : fan_out;
	c->nlevels = mkt_hkmeans_nlevels(nlist, c->fan_out);
	c->format  = format;
	c->max_ent = mkt_centroid_max_entries_fmt(dim, format);
	c->fs_gpp  = mkt_centroid_fastscan_max_groups(dim);
	c->ok	   = true;
}

/* Plan-pass entry ("write the tape"): cluster the sample once, record
 * every node, and return the layout totals -- see the header comment. */
bool
mkt_routing_tree_plan(
		const float			*vectors,
		uint32_t			 nvecs,
		Dimension			 dim,
		uint32_t			 nlist,
		uint32_t			 fan_out,
		DistanceMetric		 metric,
		MktCentroidFormat	 format,
		const KMeansOptions *opts,
		MktBlobStore		*store,
		MktStreamTreePlan	*out)
{
	RoutingTreeCtx c;
	routing_tree_ctx_init(&c, dim, nlist, fan_out, format);
	/* Only the plan pass clusters, so only it carries the sample, the
	 * metric and the k-means options. */
	c.vectors  = vectors;
	c.metric   = metric;
	c.opts	   = opts ? *opts : (KMeansOptions)MKT_KMEANS_OPTIONS_DEFAULT;
	c.store	   = store;
	c.leaf_sum = mkt_alloc0((size_t)dim * sizeof(double));
	if (store != NULL)
	{
		/* Recording narrows each node's assignments into this scratch; the
		 * root -- first and largest -- sizes it for the whole pass. */
		c.assign_scratch = mkt_alloc((size_t)nvecs * sizeof(uint16_t));
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
	MktMemCtx scratch = mkt_memctx_create(NULL, "mkt stream plan");
	MktMemCtx old_ctx = mkt_memctx_switch(scratch);
	plan_node_recurse(&c, NULL, nvecs, 0);
	mkt_memctx_switch(old_ctx);
	mkt_memctx_delete(scratch);

	if (c.assign_scratch != NULL)
		mkt_free(c.assign_scratch);
	if (!c.ok)
	{
		mkt_free(c.leaf_sum);
		return false;
	}

	out->nleaves		= c.nleaves;
	out->nlevels		= c.nlevels;
	out->centroid_pages = c.centroid_pages;
	out->leaf_mean		= mkt_alloc((size_t)dim * sizeof(float));
	for (Dimension d = 0; d < dim; d++)
		out->leaf_mean[d] = c.nleaves > 0
								  ? (float)(c.leaf_sum[d] / (double)c.nleaves)
								  : 0.0f;
	mkt_free(c.leaf_sum);
	return true;
}

/* Replay-pass entry ("read the tape"): stream every centroid + head page
 * from the recorded nodes; returns the root block. */
BlockNumber
mkt_routing_tree_write(
		MktStorage		   *storage,
		uint32_t			nvecs,
		Dimension			dim,
		uint32_t			nlist,
		uint32_t			fan_out,
		MktCentroidFormat	format,
		const RaBitQParams *rq_params,
		const float		   *global_mean,
		MktBlobStore	   *store,
		BlockNumber			first_posting,
		BlockNumber			first_centroid,
		MktStreamLeafCb		on_leaf,
		void			   *on_leaf_arg)
{
	/* Replay never clusters: everything it needs is on the tape, so it
	 * takes no sample vectors, metric or k-means options. */
	if (store == NULL)
		return InvalidBlockNumber;

	RoutingTreeCtx c;
	routing_tree_ctx_init(&c, dim, nlist, fan_out, format);
	c.storage = storage;
	c.store	  = store;
	{
		/* A node clusters into at most fan_out groups (the flat root's k =
		 * nlist <= fan_out). The assignment scratch covers the root's full
		 * sample; deeper slices are strictly smaller. */
		c.replay_cents	 = mkt_alloc((size_t)c.fan_out * dim * sizeof(float));
		c.assign_scratch = mkt_alloc((size_t)nvecs * sizeof(uint16_t));
		c.leaf_blk_scratch = mkt_alloc(
				(size_t)c.fan_out * sizeof(BlockNumber));
	}
	c.rq_params		= rq_params;
	c.global_mean	= global_mean;
	c.first_posting = first_posting;
	c.next_blk		= first_centroid;
	c.on_leaf		= on_leaf;
	c.on_leaf_arg	= on_leaf_arg;

	/* Same pass-scoped scratch as the PLAN pass -- the leak-containment
	 * boundary (see the note there). on_leaf runs under it too; its
	 * allocations must not outlive the call. */
	MktMemCtx	scratch = mkt_memctx_create(NULL, "mkt stream write");
	MktMemCtx	old_ctx = mkt_memctx_switch(scratch);
	BlockNumber root	= replay_node_recurse(&c, nvecs, 0);
	mkt_memctx_switch(old_ctx);
	mkt_memctx_delete(scratch);
	if (c.replay_cents != NULL)
		mkt_free(c.replay_cents);
	if (c.assign_scratch != NULL)
		mkt_free(c.assign_scratch);
	if (c.leaf_blk_scratch != NULL)
		mkt_free(c.leaf_blk_scratch);
	return c.ok ? root : InvalidBlockNumber;
}

/* Parallel build: write one worker-built (materialized) subtree's pages at
 * its reserved block range, mapping its leaves into the global leaf index
 * space -- see the header comment. */
BlockNumber
mkt_routing_subtree_write(
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
		void				*on_leaf_arg)
{
	uint32_t max_ent = mkt_centroid_max_entries_fmt(dim, format);

	/* Lay the subtree's nodes out at reserved blocks starting at first_block
	 * (BFS: node 0 = subtree root at first_block). */
	BlockNumber *nfb = mkt_alloc(
			(size_t)subtree->nnodes * sizeof(BlockNumber));
	(void)mkt_compute_centroid_layout(subtree, max_ent, first_block, nfb);

	/* Leaf entries link to formula-derived posting heads (first_posting +
	 * global leaf index); leaf_offset maps the subtree's local leaf indices to
	 * the global index space. */
	/* Subtree node levels are subtree-relative; level_offset places them at
	 * their absolute depth (the tree root above them is level 0). */
	mkt_write_centroid_tree(
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
			NULL);
	mkt_free(nfb);

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
mkt_auto_fan_out(uint32_t fan_out, uint32_t nlist, uint32_t default_fan_out)
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
mkt_auto_nlist(double count)
{
	/*
	 * Target ~256 vectors per IVF list -- the k-means convergence floor (the
	 * densest partitioning that still gives each centroid enough training
	 * data) and the measured recall/QPS sweet spot. The hierarchical centroid
	 * tree keeps routing cheap even at a high list count, so nlist scales
	 * linearly with the row count.
	 *
	 * Floor it at sqrt(count): for small tables count/256 collapses toward a
	 * single list, which under-partitions and can starve the k-means build.
	 * The linear target overtakes the sqrt floor at 256^2 = 65536 rows.
	 * Back-ends clamp the result to their own nlist ceiling.
	 */
	double	 c		   = count > 1.0 ? count : 1.0;
	uint32_t linear	   = (uint32_t)(c / 256.0 + 0.5);
	uint32_t min_lists = (uint32_t)sqrt(c);
	uint32_t nlist	   = linear > min_lists ? linear : min_lists;
	return nlist < 1 ? 1 : nlist;
}

/* 2nd-nearest cluster from a distance-sorted candidate list, or
 * primary_cluster when the runner-up is not within epsilon (no replication
 * warranted) -- see the header comment. */
uint32_t
mkt_find_secondary_cluster(
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
mkt_find_soar_secondary(
		const float	   *vec,
		const float	   *leaf_centroids,
		const uint32_t *cand_leaves,
		uint32_t		count,
		Dimension		dim,
		uint32_t		primary_cluster,
		const float	   *normalized_residual,
		double			lambda)
{
	float	 qrv	 = mkt_dot_product(normalized_residual, vec, dim);
	float	 lam	 = (float)lambda;
	float	 best_oa = INFINITY;
	uint32_t best_c	 = primary_cluster;

	for (uint32_t k = 0; k < count; k++)
	{
		uint32_t i = cand_leaves ? cand_leaves[k] : k;
		if (i == primary_cluster)
			continue;

		const float *cent = leaf_centroids + (size_t)i * dim;

		float l2 = mkt_l2_distance_squared(vec, cent, dim);
		if (l2 >= best_oa)
			continue; /* oa >= l2 >= best_oa: cannot improve */

		float rc  = mkt_dot_product(normalized_residual, cent, dim);
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
mkt_build_stats_print(const MktBuildStats *s)
{
	/* Per-phase breakdown — the single shared build summary, filled by both
	 * the PostgreSQL build (via the build-progress seam) and the standalone
	 * build. */
	mkt_log("build: sample %.1fms, kmeans %.1fms, refine %.1fms, setup "
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
		mkt_log("build: posting detail — parallel %.1fms + merge %.1fms, "
				"%u workers + leader, %u pages, "
				"%u partial pages merged into %u\n",
				s->ms_parallel,
				s->ms_merge,
				s->nworkers,
				s->total_pages,
				s->merge_input,
				s->merge_output);
}
