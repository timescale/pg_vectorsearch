/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * hkmeans.c - Hierarchical k-means tree builder
 *
 * BFS tree construction using vs_kmeans() with indexed access
 * at each node — avoids copying sub-vector arrays.
 *
 * The result is packed into a single contiguous allocation so it
 * can be memcpy'd into shared memory for parallel builds.
 */

#include <math.h>
#include <string.h>

#include "algo/distance.h"
#include "algo/hkmeans.h"
#include "core/memory.h"

/* BFS work queue entry */
typedef struct HKWorkItem
{
	uint32_t *vec_indices; /* original vector indices; NULL = identity */
	uint32_t  count;
	uint32_t  level;
	uint32_t  idx_in_level;
} HKWorkItem;

/*
 * Compute the number of tree levels needed.
 *
 * nlevels = max(1, ceil(log(nlist) / log(fan_out)))
 * When nlist <= fan_out, nlevels = 1 (flat).
 */
static uint32_t
compute_nlevels(uint32_t nlist, uint32_t fan_out)
{
	if (nlist <= fan_out)
		return 1;

	double	 levels = ceil(log((double)nlist) / log((double)fan_out));
	uint32_t n		= (uint32_t)levels;
	return n < 1 ? 1 : n;
}

uint32_t
vs_hkmeans_nlevels(uint32_t nlist, uint32_t fan_out)
{
	if (fan_out < 2)
		fan_out = 2;
	return compute_nlevels(nlist, fan_out);
}

/*
 * power_u32 - Compute base^exp for small unsigned integers.
 */
static uint32_t
power_u32(uint32_t base, uint32_t exp)
{
	uint32_t result = 1;
	for (uint32_t i = 0; i < exp; i++)
		result *= base;
	return result;
}

/*
 * Upper bound on total nodes: sum of fan_out^l for l = 0..nlevels-1.
 */
static uint32_t
max_total_nodes(uint32_t fan_out, uint32_t nlevels)
{
	uint32_t total = 0;
	for (uint32_t l = 0; l < nlevels; l++)
		total += power_u32(fan_out, l);
	return total;
}

/*
 * Temporary node used during BFS construction. Holds pointers
 * to centroid data before everything is packed into the final
 * contiguous allocation.
 */
typedef struct TmpNode
{
	float	*centroids;
	uint32_t nchildren;
	uint32_t level;
	uint32_t first_child;
	uint32_t first_leaf;
	size_t	 cent_bytes;
	bool	 is_leaf_parent;
} TmpNode;

HKMeansResult *
vs_hkmeans_f32(
		const float			*vectors,
		uint32_t			 nvecs,
		const uint32_t		*indices,
		Dimension			 dim,
		uint32_t			 nlist,
		uint32_t			 fan_out,
		DistanceMetric		 metric,
		const KMeansOptions *options)
{
	if (vectors == NULL || nvecs == 0 || dim == 0 || nlist == 0 || fan_out < 2)
		return NULL;

	uint32_t nlevels   = compute_nlevels(nlist, fan_out);
	uint32_t max_nodes = max_total_nodes(fan_out, nlevels);

	/* Work context for BFS temporaries */
	VsMemCtx work_ctx	= vs_memctx_create(NULL, "hkmeans_work");
	VsMemCtx caller_ctx = vs_memctx_switch(work_ctx);

	/* Temporary nodes (pointers, packed later) */
	TmpNode *tmp_nodes = vs_alloc(max_nodes * sizeof(TmpNode));
	memset(tmp_nodes, 0, max_nodes * sizeof(TmpNode));
	uint32_t nnodes	 = 0;
	uint32_t nleaves = 0;

	/* If caller provided indices, copy them as the root's
	 * vec_indices so vs_kmeans uses indirect access. */
	uint32_t *root_indices = NULL;
	if (indices != NULL)
	{
		root_indices = vs_alloc(nvecs * sizeof(uint32_t));
		memcpy(root_indices, indices, nvecs * sizeof(uint32_t));
	}

	/* BFS work queue */
	HKWorkItem *queue  = vs_alloc(max_nodes * sizeof(HKWorkItem));
	uint32_t	q_tail = 0;

	queue[q_tail++] = (HKWorkItem){
			.vec_indices  = root_indices,
			.count		  = nvecs,
			.level		  = 0,
			.idx_in_level = 0,
	};

	bool	  ok	 = true;
	uint32_t *counts = vs_alloc(fan_out * sizeof(uint32_t));

	/* Mutable copy: initial_centroids applies only to root */
	KMeansOptions local_opts = VS_KMEANS_OPTIONS_DEFAULT;
	if (options != NULL)
		local_opts = *options;

	for (uint32_t qi = 0; qi < q_tail && ok; qi++)
	{
		HKWorkItem item			  = queue[qi];
		bool	   is_leaf_parent = (item.level == nlevels - 1);

		/*
		 * Determine K for this node.
		 *
		 * Single-level (flat): K = nlist (clamped to vector count)
		 * Multi-level root/internal: K = fan_out (clamped)
		 */
		uint32_t k;
		if (nlevels == 1)
			k = nlist < item.count ? nlist : item.count;
		else
			k = fan_out < item.count ? fan_out : item.count;

		/* Use indexed k-means — no vector copy needed */
		KMeansResult *km = vs_kmeans(
				vectors,
				item.vec_indices,
				VS_VEC_F32,
				item.count,
				dim,
				k,
				metric,
				&local_opts);

		/* Initial centroids only for root node */
		local_opts.initial_centroids = NULL;

		if (km == NULL)
		{
			ok = false;
			break;
		}

		uint32_t node_idx = nnodes++;
		TmpNode *tn		  = &tmp_nodes[node_idx];

		size_t cent_sz = (size_t)km->nlist * dim * sizeof(float);
		tn->centroids  = vs_alloc(cent_sz);
		tn->cent_bytes = cent_sz;
		memcpy(tn->centroids, km->centroids, cent_sz);
		tn->nchildren	   = km->nlist;
		tn->level		   = item.level;
		tn->first_child	   = HKMEANS_NO_CHILD;
		tn->first_leaf	   = 0;
		tn->is_leaf_parent = is_leaf_parent;

		if (is_leaf_parent)
			nleaves += km->nlist;
		else
		{
			/*
			 * Count vectors per cluster from assignments.
			 *
			 * We cannot use km->cluster_sizes because k-means
			 * does a final reassignment after the last update
			 * step, so cluster_sizes may be stale.
			 */
			memset(counts, 0, km->nlist * sizeof(uint32_t));
			for (uint32_t v = 0; v < item.count; v++)
				counts[km->assignments[v]]++;

			/*
			 * Keep only non-empty clusters as children, compacting their
			 * centroids to match. k-means can leave a cluster empty on
			 * degenerate/collapsing data; tree descent indexes a node's
			 * children as first_child + c over its centroids, so the kept
			 * centroids and the enqueued child nodes must stay 1:1 and
			 * contiguous. (Empty internal clusters previously left
			 * nchildren > children-created, so descent could read past the
			 * nodes array.) item.count > 0 here, so at least one cluster is
			 * non-empty and kept >= 1.
			 */
			uint32_t kept	= 0;
			tn->first_child = q_tail;
			for (uint32_t c = 0; c < km->nlist; c++)
			{
				uint32_t sub_n = counts[c];
				if (sub_n == 0)
					continue;

				if (kept != c)
					memcpy(tn->centroids + (size_t)kept * dim,
						   km->centroids + (size_t)c * dim,
						   (size_t)dim * sizeof(float));

				uint32_t *sub_indices = vs_alloc(sub_n * sizeof(uint32_t));
				uint32_t  idx		  = 0;
				for (uint32_t v = 0; v < item.count; v++)
				{
					if (km->assignments[v] == c)
					{
						uint32_t orig = item.vec_indices ? item.vec_indices[v]
														 : v;
						sub_indices[idx++] = orig;
					}
				}

				queue[q_tail++] = (HKWorkItem){
						.vec_indices  = sub_indices,
						.count		  = sub_n,
						.level		  = item.level + 1,
						.idx_in_level = item.idx_in_level * fan_out + kept,
				};
				kept++;
			}
			tn->nchildren  = kept;
			tn->cent_bytes = (size_t)kept * dim * sizeof(float);
		}

		vs_kmeans_result_destroy(km);
	}

	if (!ok)
	{
		vs_memctx_switch(caller_ctx);
		vs_memctx_delete(work_ctx);
		return NULL;
	}

	/* Compute first_leaf for leaf-parent nodes */
	uint32_t leaf_off = 0;
	for (uint32_t i = 0; i < nnodes; i++)
	{
		if (!tmp_nodes[i].is_leaf_parent)
			continue;
		tmp_nodes[i].first_leaf = leaf_off;
		leaf_off += tmp_nodes[i].nchildren;
	}

	/*
	 * Pack the whole tree into one contiguous allocation, laid out as
	 * [header][nodes][leaf centroids][internal centroids] and addressed
	 * by byte offsets rather than pointers. This is what lets a built
	 * tree be memcpy'd into shared memory and used by another process
	 * (the PG parallel build) with no pointer fix-up or deserialization.
	 */
	size_t hdr_sz	 = sizeof(HKMeansResult);
	size_t nodes_sz	 = (size_t)nnodes * sizeof(HKMeansNode);
	size_t leaf_sz	 = (size_t)nleaves * dim * sizeof(float);
	size_t intern_sz = 0;
	for (uint32_t i = 0; i < nnodes; i++)
	{
		if (!tmp_nodes[i].is_leaf_parent)
			intern_sz += tmp_nodes[i].cent_bytes;
	}
	size_t total = hdr_sz + nodes_sz + leaf_sz + intern_sz;

	vs_memctx_switch(caller_ctx);
	HKMeansResult *result = vs_alloc(total);
	memset(result, 0, total);

	result->nodes_offset = (uint32_t)hdr_sz;
	result->leaf_offset	 = (uint32_t)(hdr_sz + nodes_sz);
	result->total_size	 = (uint32_t)total;
	result->nnodes		 = nnodes;
	result->nlevels		 = nlevels;
	result->nleaves		 = nleaves;
	result->fan_out		 = fan_out;
	result->dim			 = dim;

	HKMeansNode *nodes		= hk_nodes(result);
	float		*leaf_cents = hk_leaf_centroids(result);
	char		*intern_dst = (char *)result + hdr_sz + nodes_sz + leaf_sz;

	/* Copy leaf centroids into the leaf area */
	for (uint32_t i = 0; i < nnodes; i++)
	{
		TmpNode *tn = &tmp_nodes[i];
		if (!tn->is_leaf_parent)
			continue;
		memcpy(leaf_cents + (size_t)tn->first_leaf * dim,
			   tn->centroids,
			   tn->cent_bytes);
	}

	/* Copy internal centroids and build final nodes */
	for (uint32_t i = 0; i < nnodes; i++)
	{
		TmpNode *tn = &tmp_nodes[i];

		nodes[i].nchildren	 = tn->nchildren;
		nodes[i].level		 = tn->level;
		nodes[i].first_child = tn->first_child;
		nodes[i].first_leaf	 = tn->first_leaf;

		if (tn->is_leaf_parent)
		{
			/* Point into the leaf centroids area */
			nodes[i].centroid_offset = result->leaf_offset +
									   (uint32_t)((size_t)tn->first_leaf *
												  dim * sizeof(float));
		}
		else
		{
			/* Copy into the internal area */
			memcpy(intern_dst, tn->centroids, tn->cent_bytes);
			nodes[i].centroid_offset = (uint32_t)((size_t)(intern_dst -
														   (char *)result));
			intern_dst += tn->cent_bytes;
		}
	}

	/* Bulk-free all BFS temporaries */
	vs_memctx_delete(work_ctx);

	return result;
}

size_t
vs_hkmeans_max_blob_size_capped(
		uint32_t nlist, uint32_t fan_out, Dimension dim, uint64_t max_leaves)
{
	if (fan_out < 2)
		fan_out = 2;
	if (max_leaves < 1)
		max_leaves = 1;

	uint32_t nlevels = compute_nlevels(nlist, fan_out);

	/* All worst-case counts are fan_out powers; at large nlist with a small
	 * fan_out they exceed 32 bits, so the whole bound is computed in 64-bit
	 * (the caller compares it against the blob format's 32-bit offset limit
	 * and fails the build rather than wrapping into an undersized slot).
	 *
	 * max_leaves caps every level's node count: a node exists only where at
	 * least one training vector landed, so no level can hold more nodes
	 * than the tree has vectors -- the depth stays the full nlevels (few
	 * vectors under a deep target degenerate into chains), but each level's
	 * width is min(fan_out^l, max_leaves). UINT64_MAX = the analytic
	 * worst case. */
	uint64_t width		  = 1; /* fan_out^l, capped at max_leaves */
	uint64_t nleaves	  = 0;
	uint64_t nnodes		  = 0; /* sum of capped widths, l = 0..nlevels-1 */
	uint64_t intern_nodes = 0; /* same sum, one level shorter */
	for (uint32_t l = 0; l < nlevels; l++)
	{
		nnodes += width;
		if (nlevels >= 2 && l < nlevels - 1)
			intern_nodes += width;
		if (width >= max_leaves / fan_out)
			width = max_leaves;
		else
			width *= fan_out;
	}
	nleaves = width;

	return sizeof(HKMeansResult) + nnodes * sizeof(HKMeansNode) +
		   nleaves * dim * sizeof(float) +
		   intern_nodes * fan_out * dim * sizeof(float);
}

size_t
vs_hkmeans_max_blob_size(uint32_t nlist, uint32_t fan_out, Dimension dim)
{
	return vs_hkmeans_max_blob_size_capped(nlist, fan_out, dim, UINT64_MAX);
}

HKMeansResult *
vs_hkmeans_build_flat(
		const float *centroids,
		uint32_t	 nleaves,
		uint32_t	 fan_out,
		Dimension	 dim)
{
	if (centroids == NULL || nleaves == 0 || dim == 0)
		return NULL;

	size_t hdr_sz	= sizeof(HKMeansResult);
	size_t nodes_sz = sizeof(HKMeansNode); /* single leaf-parent root */
	size_t leaf_sz	= (size_t)nleaves * dim * sizeof(float);
	size_t total	= hdr_sz + nodes_sz + leaf_sz;

	HKMeansResult *result = vs_alloc(total);
	memset(result, 0, total);

	result->nodes_offset = (uint32_t)hdr_sz;
	result->leaf_offset	 = (uint32_t)(hdr_sz + nodes_sz);
	result->total_size	 = (uint32_t)total;
	result->nnodes		 = 1;
	result->nlevels		 = 1;
	result->nleaves		 = nleaves;
	result->fan_out		 = fan_out;
	result->dim			 = dim;

	HKMeansNode *root	  = hk_nodes(result);
	root->level			  = 0;
	root->nchildren		  = nleaves;
	root->first_child	  = HKMEANS_NO_CHILD; /* leaf-parent */
	root->first_leaf	  = 0;
	root->centroid_offset = result->leaf_offset;
	memcpy(hk_leaf_centroids(result), centroids, leaf_sz);

	return result;
}

uint32_t
vs_hkmeans_assign(
		const HKMeansResult *tree,
		const float			*vec,
		DistanceMetric		 metric,
		Distance			*out_distance)
{
	Dimension		   dim		= tree->dim;
	const HKMeansNode *nodes	= hk_nodes(tree);
	uint32_t		   node_idx = 0;

	for (uint32_t level = 0; level < tree->nlevels; level++)
	{
		const HKMeansNode *node		 = &nodes[node_idx];
		const float		  *cents	 = hk_node_centroids(tree, node);
		Distance		   best_dist = INFINITY;
		uint32_t		   best_c	 = 0;

		for (uint32_t c = 0; c < node->nchildren; c++)
		{
			const float *centroid = cents + (size_t)c * dim;
			Vec32Ref	 qref	  = {.data = vec, .dim = dim};
			Vec32Ref	 cref	  = {.data = centroid, .dim = dim};
			Distance	 d		  = vs_distance(qref, cref, metric);
			if (d < best_dist)
			{
				best_dist = d;
				best_c	  = c;
			}
		}

		/*
		 * A node with no internal children is a leaf-parent: its children are
		 * leaf centroids, reached via first_leaf. This is the authoritative
		 * test -- using level == nlevels - 1 instead breaks on non-uniform
		 * depth trees, where a branch that bottoms out early leaves a
		 * leaf-parent above the max level; first_child (NO_CHILD = UINT32_MAX)
		 * would then be added to best_c and index past the nodes array.
		 */
		if (node->first_child == HKMEANS_NO_CHILD)
		{
			if (out_distance != NULL)
				*out_distance = best_dist;
			return node->first_leaf + best_c;
		}

		node_idx = node->first_child + best_c;
	}

	if (out_distance != NULL)
		*out_distance = INFINITY;
	return 0;
}

/*
 * Insert (id, d) into an unsorted bounded set of the `cap` smallest
 * entries. Keeps at most `cap` items; evicts the current max when full.
 */
static inline void
topk_insert(
		uint32_t *ids,
		Distance *dists,
		uint32_t *n,
		uint32_t  cap,
		uint32_t  id,
		Distance  d)
{
	if (*n < cap)
	{
		ids[*n]	  = id;
		dists[*n] = d;
		(*n)++;
		return;
	}

	/* Full: replace the worst entry if this one is better. */
	uint32_t worst_i = 0;
	Distance worst_d = dists[0];
	for (uint32_t i = 1; i < cap; i++)
		if (dists[i] > worst_d)
		{
			worst_d = dists[i];
			worst_i = i;
		}
	if (d < worst_d)
	{
		ids[worst_i]   = id;
		dists[worst_i] = d;
	}
}

/* Insertion sort by ascending distance (n is small, <= VS_HK_MAX_TOPK). */
static inline void
topk_sort(uint32_t *ids, Distance *dists, uint32_t n)
{
	for (uint32_t i = 1; i < n; i++)
	{
		Distance d	= dists[i];
		uint32_t id = ids[i];
		uint32_t j	= i;
		while (j > 0 && dists[j - 1] > d)
		{
			dists[j] = dists[j - 1];
			ids[j]	 = ids[j - 1];
			j--;
		}
		dists[j] = d;
		ids[j]	 = id;
	}
}

uint32_t
vs_hkmeans_assign_topk(
		const HKMeansResult *tree,
		const float			*vec,
		DistanceMetric		 metric,
		uint32_t			 k,
		uint32_t			 beam_width,
		uint32_t			*out_leaves,
		Distance			*out_dists)
{
	if (k == 0)
		return 0;
	if (k > VS_HK_MAX_TOPK)
		k = VS_HK_MAX_TOPK;
	if (beam_width < 1)
		beam_width = 1;
	if (beam_width > VS_HK_MAX_TOPK)
		beam_width = VS_HK_MAX_TOPK;

	const Dimension	   dim	 = tree->dim;
	const HKMeansNode *nodes = hk_nodes(tree);

	/* Current beam: indices into nodes[] for the next level to expand. */
	uint32_t beam[VS_HK_MAX_TOPK];
	uint32_t beam_n = 1;
	beam[0]			= 0; /* root */

	/*
	 * Terminal leaves found so far. A leaf-parent (first_child == NO_CHILD)
	 * can appear at any level on a non-uniform-depth tree, so its leaf
	 * children are collected here directly rather than assumed to all sit at
	 * nlevels - 1.
	 */
	uint32_t res_id[VS_HK_MAX_TOPK] = {0};
	Distance res_d[VS_HK_MAX_TOPK]	= {0};
	uint32_t res_n					= 0;

	/* Internal child nodes to expand at the next level. next_n is what
	 * resets them per level, so one initialization covers the descent. */
	uint32_t next[VS_HK_MAX_TOPK]	= {0};
	Distance next_d[VS_HK_MAX_TOPK] = {0};

	for (uint32_t level = 0; level < tree->nlevels && beam_n > 0; level++)
	{
		uint32_t next_n = 0;

		for (uint32_t b = 0; b < beam_n; b++)
		{
			const HKMeansNode *node	 = &nodes[beam[b]];
			const float		  *cents = hk_node_centroids(tree, node);
			bool node_leaf			 = (node->first_child == HKMEANS_NO_CHILD);
			for (uint32_t c = 0; c < node->nchildren; c++)
			{
				Vec32Ref qref = {.data = vec, .dim = dim};
				Vec32Ref cref = {.data = cents + (size_t)c * dim, .dim = dim};
				Distance d	  = vs_distance(qref, cref, metric);
				if (node_leaf)
					topk_insert(
							res_id, res_d, &res_n, k, node->first_leaf + c, d);
				else
					topk_insert(
							next,
							next_d,
							&next_n,
							beam_width,
							node->first_child + c,
							d);
			}
		}

		memcpy(beam, next, next_n * sizeof(uint32_t));
		beam_n = next_n;
	}

	/* A tree whose every node is internal collects nothing, so the sort
	 * and the copy below have nothing to work on. */
	if (res_n == 0)
		return 0;

	topk_sort(res_id, res_d, res_n);
	for (uint32_t i = 0; i < res_n; i++)
	{
		out_leaves[i] = res_id[i];
		if (out_dists != NULL)
			out_dists[i] = res_d[i];
	}
	return res_n;
}
