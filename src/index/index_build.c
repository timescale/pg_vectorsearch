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

void
mkt_write_centroid_tree(
		MktStorage			*storage,
		const HKMeansResult *tree,
		Dimension			 dim,
		uint32_t			 fan_out,
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
					(uint8_t)node->level,
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
				(uint8_t)node->level,
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

typedef struct StreamCtx
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
	bool				emit;
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
	/* Node replay through the caller's blob store: the plan pass records
	 * each node's clustering ({k, centroids, assignments}, recursion order)
	 * and the write pass replays it instead of re-running k-means. NULL
	 * keeps the self-contained re-cluster behavior (unit/standalone
	 * callers). */
	MktBlobStore *store;
	float		 *replay_cents;	 /* write: [fan_out * dim] */
	uint16_t	 *replay_assign; /* write: [nvecs] */
	/* accumulators */
	uint32_t  nleaves; /* running (== next first_leaf) */
	double	 *leaf_sum; /* plan only, [dim]: leaf-centroid sum for the
						 * leaf_mean the caller uses as global_mean */
	uint32_t	centroid_pages; /* plan only */
	BlockNumber next_blk;		/* write only: next reserved centroid block */
	bool		ok;
} StreamCtx;

/* Pages one node of `n` entries occupies — must match the writers' packing. */
static uint32_t
stream_pages_for(const StreamCtx *c, uint32_t n)
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
stream_write_node(
		StreamCtx		  *c,
		const float		  *cents,
		uint32_t		   n,
		uint32_t		   level,
		bool			   is_leaf,
		const BlockNumber *child_blks)
{
	uint16_t	flags = is_leaf ? MKT_CENTROID_FLAG_LEAF : 0;
	BlockNumber start = c->next_blk;

	if (c->format == MKT_CENTROID_FMT_FASTSCAN)
		mkt_centroid_write_fastscan_pages(
				c->storage,
				c->dim,
				n,
				(uint8_t)level,
				flags,
				c->rq_params,
				cents,
				c->global_mean,
				child_blks,
				start);
	else
	{
		CentroidEncoderState est;
		CentroidEncoder		*enc = centroid_encoder_init(
				&est, c->format, cents, c->dim, c->rq_params, c->global_mean);
		uint16_t child_count = is_leaf ? 0 : (uint16_t)c->fan_out;
		mkt_centroid_write_pages(
				c->storage,
				c->dim,
				n,
				c->format,
				(uint8_t)level,
				flags,
				child_count,
				enc,
				child_blks,
				NULL,
				start);
	}

	c->next_blk += stream_pages_for(c, n);
	return start;
}

/*
 * DFS one node. slice[count] are indices into c->vectors (NULL == identity for
 * the root). Returns the node's first block (write phase) or
 * InvalidBlockNumber (plan phase). Post-order: children are written before the
 * parent so the parent's entries can carry their child block numbers.
 */
static BlockNumber
stream_node(
		StreamCtx *c, const uint32_t *slice, uint32_t count, uint32_t level)
{
	if (!c->ok)
		return InvalidBlockNumber;

	bool	 is_leaf_parent = (level == c->nlevels - 1);
	uint32_t k				= (c->nlevels == 1) ? c->nlist : c->fan_out;
	if (k > count)
		k = count;

	/* This node's clustering: replayed from the blob store on the write
	 * pass when the plan pass recorded it, otherwise computed (and, on the
	 * plan pass, recorded). The record travels in recursion order, so the
	 * write pass consumes it in exactly the order it was produced. */
	KMeansResult   *km			= NULL;
	float		   *cents		= NULL;
	const uint32_t *assign32	= NULL;
	const uint16_t *assign16	= NULL;
	uint32_t		nclusters	= 0;

	if (c->store != NULL && c->emit)
	{
		uint32_t rec_k = 0;
		if (mkt_pbuild_blobstore_get(c->store, &rec_k, sizeof(rec_k)) !=
					sizeof(rec_k) ||
			rec_k == 0 || rec_k > c->fan_out ||
			mkt_pbuild_blobstore_get(
					c->store,
					c->replay_cents,
					(uint64_t)rec_k * c->dim * sizeof(float)) !=
					(uint64_t)rec_k * c->dim * sizeof(float) ||
			mkt_pbuild_blobstore_get(
					c->store,
					c->replay_assign,
					(uint64_t)count * sizeof(uint16_t)) !=
					(uint64_t)count * sizeof(uint16_t))
		{
			c->ok = false;
			return InvalidBlockNumber;
		}
		nclusters = rec_k;
		cents	  = c->replay_cents;
		assign16  = c->replay_assign;
	}
	else
	{
		km = mkt_kmeans(
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
			return InvalidBlockNumber;
		}
		nclusters = km->nlist;
		cents	  = km->centroids;
		assign32  = km->assignments;

		if (c->store != NULL)
		{
			/* Record for the write pass. Assignments fit uint16: a node
			 * clusters into at most fan_out (<= the meta page's 8-bit
			 * fan_out) groups. */
			uint16_t *a16 = mkt_alloc((size_t)count * sizeof(uint16_t));
			for (uint32_t v = 0; v < count; v++)
				a16[v] = (uint16_t)km->assignments[v];
			mkt_pbuild_blobstore_put(c->store, &nclusters, sizeof(nclusters));
			mkt_pbuild_blobstore_put(
					c->store, cents, (uint64_t)nclusters * c->dim * sizeof(float));
			mkt_pbuild_blobstore_put(
					c->store, a16, (uint64_t)count * sizeof(uint16_t));
			mkt_free(a16);
		}
	}
	c->opts.initial_centroids = NULL; /* root only (matches mkt_hkmeans_f32) */

#define STREAM_ASSIGN(v) (assign32 ? assign32[(v)] : (uint32_t)assign16[(v)])

	/* Count assignments per cluster (km->cluster_sizes may be stale). */
	uint32_t *counts = mkt_alloc0((size_t)nclusters * sizeof(uint32_t));
	for (uint32_t v = 0; v < count; v++)
		counts[STREAM_ASSIGN(v)]++;

	BlockNumber blk = InvalidBlockNumber;

	if (is_leaf_parent)
	{
		/* Keep non-empty clusters as leaves, compacting their centroids so the
		 * kept centroids stay contiguous (descent indexes them densely). */
		uint32_t kept = 0;
		for (uint32_t cl = 0; cl < nclusters; cl++)
		{
			if (counts[cl] == 0)
				continue;
			if (kept != cl)
				memcpy(cents + (size_t)kept * c->dim,
					   cents + (size_t)cl * c->dim,
					   (size_t)c->dim * sizeof(float));
			kept++;
		}
		if (c->emit)
		{
			/* Leaf heads are formula-derived (first_posting + global leaf
			 * index); kept <= fan_out, so this scratch is O(fan_out). */
			BlockNumber *leaf_blks = NULL;
			if (c->first_posting != InvalidBlockNumber)
			{
				leaf_blks = mkt_alloc((size_t)kept * sizeof(BlockNumber));
				for (uint32_t kk = 0; kk < kept; kk++)
					leaf_blks[kk] = c->first_posting + c->nleaves + kk;
			}
			blk = stream_write_node(c, cents, kept, level, true, leaf_blks);
			if (leaf_blks != NULL)
				mkt_free(leaf_blks);
			/* Emit each leaf's head page from its resident float centroid. */
			if (c->on_leaf != NULL)
				for (uint32_t kk = 0; kk < kept; kk++)
					c->on_leaf(
							c->on_leaf_arg,
							c->nleaves + kk,
							cents + (size_t)kk * c->dim);
		}
		else
		{
			c->centroid_pages += stream_pages_for(c, kept);
			if (c->leaf_sum != NULL)
				for (uint32_t kk = 0; kk < kept; kk++)
				{
					const float *lc = cents + (size_t)kk * c->dim;
					for (Dimension d = 0; d < c->dim; d++)
						c->leaf_sum[d] += lc[d];
				}
		}
		c->nleaves += kept;
	}
	else
	{
		/* Recurse each non-empty cluster (post-order), then write this node.
		 */
		BlockNumber *child_blocks = mkt_alloc(
				(size_t)nclusters * sizeof(BlockNumber));
		/* The children's replay records interleave with this node's slice
		 * gathers, so the parent's replayed assignments must survive the
		 * recursion: snapshot them (the shared replay scratch is reused by
		 * every level). O(count) like the slice arrays themselves. */
		uint16_t *my_assign = NULL;
		if (assign16 != NULL)
		{
			my_assign = mkt_alloc((size_t)count * sizeof(uint16_t));
			memcpy(my_assign, assign16, (size_t)count * sizeof(uint16_t));
			assign16 = my_assign;
		}
		float *my_cents = NULL;
		if (km == NULL)
		{
			my_cents = mkt_alloc((size_t)nclusters * c->dim * sizeof(float));
			memcpy(my_cents,
				   cents,
				   (size_t)nclusters * c->dim * sizeof(float));
			cents = my_cents;
		}
		uint32_t kept = 0;
		for (uint32_t cl = 0; cl < nclusters && c->ok; cl++)
		{
			if (counts[cl] == 0)
				continue;
			if (kept != cl)
				memcpy(cents + (size_t)kept * c->dim,
					   cents + (size_t)cl * c->dim,
					   (size_t)c->dim * sizeof(float));

			uint32_t  sub_n = counts[cl];
			uint32_t *sub	= mkt_alloc((size_t)sub_n * sizeof(uint32_t));
			uint32_t  idx	= 0;
			for (uint32_t v = 0; v < count; v++)
				if (STREAM_ASSIGN(v) == cl)
					sub[idx++] = slice ? slice[v] : v;
			child_blocks[kept] = stream_node(c, sub, sub_n, level + 1);
			mkt_free(sub);
			kept++;
		}
		if (c->emit)
			blk = stream_write_node(c, cents, kept, level, false, child_blocks);
		if (my_assign != NULL)
			mkt_free(my_assign);
		if (my_cents != NULL)
			mkt_free(my_cents);
		else
			c->centroid_pages += stream_pages_for(c, kept);
		mkt_free(child_blocks);
	}

	mkt_free(counts);
	if (km != NULL)
		mkt_kmeans_result_destroy(km);
	return blk;
#undef STREAM_ASSIGN
}

static void
stream_ctx_init(
		StreamCtx			*c,
		Dimension			 dim,
		uint32_t			 nlist,
		uint32_t			 fan_out,
		DistanceMetric		 metric,
		MktCentroidFormat	 format,
		const KMeansOptions *opts)
{
	memset(c, 0, sizeof(*c));
	c->dim	   = dim;
	c->nlist   = nlist;
	c->fan_out = fan_out < 2 ? 2 : fan_out;
	c->nlevels = mkt_hkmeans_nlevels(nlist, c->fan_out);
	c->metric  = metric;
	c->format  = format;
	c->opts	   = opts ? *opts : (KMeansOptions)MKT_KMEANS_OPTIONS_DEFAULT;
	c->max_ent = mkt_centroid_max_entries_fmt(dim, format);
	c->fs_gpp  = mkt_centroid_fastscan_max_groups(dim);
	c->ok	   = true;
}

bool
mkt_stream_centroid_plan(
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
	StreamCtx c;
	stream_ctx_init(&c, dim, nlist, fan_out, metric, format, opts);
	c.vectors = vectors;
	c.emit	  = false;
	c.store	  = store;
	c.leaf_sum = mkt_alloc0((size_t)dim * sizeof(double));

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
	 * their blast radius at one pass. Outputs (leaf_counts, leaf_mean) are
	 * allocated in the caller's context outside the switch.
	 */
	MktMemCtx scratch = mkt_memctx_create(NULL, "mkt stream plan");
	MktMemCtx old_ctx = mkt_memctx_switch(scratch);
	stream_node(&c, NULL, nvecs, 0);
	mkt_memctx_switch(old_ctx);
	mkt_memctx_delete(scratch);

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

BlockNumber
mkt_stream_centroid_write(
		MktStorage			*storage,
		const float			*vectors,
		uint32_t			 nvecs,
		Dimension			 dim,
		uint32_t			 nlist,
		uint32_t			 fan_out,
		DistanceMetric		 metric,
		MktCentroidFormat	 format,
		const RaBitQParams	*rq_params,
		const float			*global_mean,
		const KMeansOptions *opts,
		MktBlobStore		*store,
		BlockNumber			 first_posting,
		BlockNumber			 first_centroid,
		MktStreamLeafCb		 on_leaf,
		void				*on_leaf_arg)
{
	StreamCtx c;
	stream_ctx_init(&c, dim, nlist, fan_out, metric, format, opts);
	c.vectors		= vectors;
	c.emit			= true;
	c.storage		= storage;
	c.store			= store;
	if (store != NULL)
	{
		/* A node clusters into at most fan_out groups (the flat root's k =
		 * nlist <= fan_out). The assignment scratch covers the root's full
		 * sample; deeper slices are strictly smaller. */
		c.replay_cents =
				mkt_alloc((size_t)c.fan_out * dim * sizeof(float));
		c.replay_assign = mkt_alloc((size_t)nvecs * sizeof(uint16_t));
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
	BlockNumber root	= stream_node(&c, NULL, nvecs, 0);
	mkt_memctx_switch(old_ctx);
	mkt_memctx_delete(scratch);
	if (c.replay_cents != NULL)
		mkt_free(c.replay_cents);
	if (c.replay_assign != NULL)
		mkt_free(c.replay_assign);
	return c.ok ? root : InvalidBlockNumber;
}

BlockNumber
mkt_write_subtree_streaming(
		MktStorage			*storage,
		const HKMeansResult *subtree,
		Dimension			 dim,
		uint32_t			 fan_out,
		MktCentroidFormat	 format,
		const RaBitQParams	*rq_params,
		const float			*global_mean,
		BlockNumber			 first_posting,
		uint32_t			 leaf_offset,
		BlockNumber			 first_block,
		MktStreamLeafCb		 on_leaf,
		void				*on_leaf_arg,
		uint32_t			*out_pages)
{
	uint32_t max_ent = mkt_centroid_max_entries_fmt(dim, format);

	/* Lay the subtree's nodes out at reserved blocks starting at first_block
	 * (BFS: node 0 = subtree root at first_block). */
	BlockNumber *nfb = mkt_alloc(
			(size_t)subtree->nnodes * sizeof(BlockNumber));
	BlockNumber next =
			mkt_compute_centroid_layout(subtree, max_ent, first_block, nfb);

	/* Leaf entries link to formula-derived posting heads (first_posting +
	 * global leaf index); leaf_offset maps the subtree's local leaf indices to
	 * the global index space. */
	mkt_write_centroid_tree(
			storage,
			subtree,
			dim,
			fan_out,
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

	if (out_pages != NULL)
		*out_pages = (uint32_t)(next - first_block);
	return first_block;
}

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
