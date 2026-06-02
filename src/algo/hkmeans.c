/*
 * hkmeans.c - Hierarchical k-means tree builder
 *
 * BFS tree construction using mkt_kmeans() with indexed access
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
	uint32_t *vec_indices;
	uint32_t  count;
	uint32_t  level;
	uint32_t  idx_in_level;
} HKWorkItem;

static uint32_t
compute_nlevels(uint32_t nlist, uint32_t fan_out)
{
	if (nlist <= fan_out)
		return 1;

	double	 levels = ceil(log((double)nlist) / log((double)fan_out));
	uint32_t n		= (uint32_t)levels;
	return n < 1 ? 1 : n;
}

static uint32_t
power_u32(uint32_t base, uint32_t exp)
{
	uint32_t result = 1;
	for (uint32_t i = 0; i < exp; i++)
		result *= base;
	return result;
}

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
mkt_hkmeans_f32(
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
	MktMemCtx work_ctx	 = mkt_memctx_create(NULL, "hkmeans_work");
	MktMemCtx caller_ctx = mkt_memctx_switch(work_ctx);

	/* Temporary nodes (pointers, packed later) */
	TmpNode *tmp_nodes = mkt_alloc(max_nodes * sizeof(TmpNode));
	memset(tmp_nodes, 0, max_nodes * sizeof(TmpNode));
	uint32_t nnodes	 = 0;
	uint32_t nleaves = 0;

	/* If caller provided indices, copy them as the root's
	 * vec_indices so mkt_kmeans uses indirect access. */
	uint32_t *root_indices = NULL;
	if (indices != NULL)
	{
		root_indices = mkt_alloc(nvecs * sizeof(uint32_t));
		memcpy(root_indices, indices, nvecs * sizeof(uint32_t));
	}

	/* BFS work queue */
	HKWorkItem *queue  = mkt_alloc(max_nodes * sizeof(HKWorkItem));
	uint32_t	q_tail = 0;

	queue[q_tail++] = (HKWorkItem){
			.vec_indices  = root_indices,
			.count		  = nvecs,
			.level		  = 0,
			.idx_in_level = 0,
	};

	bool	  ok	 = true;
	uint32_t *counts = mkt_alloc(fan_out * sizeof(uint32_t));

	/* Mutable copy: initial_centroids applies only to root */
	KMeansOptions local_opts = MKT_KMEANS_OPTIONS_DEFAULT;
	if (options != NULL)
		local_opts = *options;

	for (uint32_t qi = 0; qi < q_tail && ok; qi++)
	{
		HKWorkItem item			  = queue[qi];
		bool	   is_leaf_parent = (item.level == nlevels - 1);

		uint32_t k;
		if (nlevels == 1)
			k = nlist < item.count ? nlist : item.count;
		else
			k = fan_out < item.count ? fan_out : item.count;

		KMeansResult *km = mkt_kmeans(
				vectors,
				item.vec_indices,
				MKT_VEC_F32,
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
		tn->centroids  = mkt_alloc(cent_sz);
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
			memset(counts, 0, km->nlist * sizeof(uint32_t));
			for (uint32_t v = 0; v < item.count; v++)
				counts[km->assignments[v]]++;

			bool first = true;
			for (uint32_t c = 0; c < km->nlist; c++)
			{
				uint32_t sub_n = counts[c];
				if (sub_n == 0)
					continue;

				uint32_t *sub_indices = mkt_alloc(sub_n * sizeof(uint32_t));
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

				if (first)
				{
					tn->first_child = q_tail;
					first			= false;
				}

				queue[q_tail++] = (HKWorkItem){
						.vec_indices  = sub_indices,
						.count		  = sub_n,
						.level		  = item.level + 1,
						.idx_in_level = item.idx_in_level * fan_out + c,
				};
			}
		}

		mkt_kmeans_result_destroy(km);
	}

	if (!ok)
	{
		mkt_memctx_switch(caller_ctx);
		mkt_memctx_delete(work_ctx);
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

	/* Pack everything into a single contiguous allocation */
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

	mkt_memctx_switch(caller_ctx);
	HKMeansResult *result = mkt_alloc(total);
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
	mkt_memctx_delete(work_ctx);

	return result;
}

HKMeansResult *
mkt_hkmeans_build_two_level(
		const float	   *root_centroids,
		uint32_t		fan_out,
		const float	  **child_centroids,
		const uint32_t *child_k,
		Dimension		dim)
{
	if (root_centroids == NULL || fan_out == 0 || dim == 0)
		return NULL;

	uint32_t nleaves = 0;
	for (uint32_t c = 0; c < fan_out; c++)
		nleaves += child_k[c];

	/* Tree has 1 root + fan_out child nodes = 1 + fan_out total */
	uint32_t nnodes	 = 1 + fan_out;
	uint32_t nlevels = 2;

	size_t hdr_sz	 = sizeof(HKMeansResult);
	size_t nodes_sz	 = (size_t)nnodes * sizeof(HKMeansNode);
	size_t leaf_sz	 = (size_t)nleaves * dim * sizeof(float);
	size_t intern_sz = (size_t)fan_out * dim * sizeof(float);
	size_t total	 = hdr_sz + nodes_sz + leaf_sz + intern_sz;

	HKMeansResult *result = mkt_alloc(total);
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

	/* Root node: internal centroids, fan_out children */
	nodes[0].nchildren		 = fan_out;
	nodes[0].level			 = 0;
	nodes[0].first_child	 = 1;
	nodes[0].first_leaf		 = 0;
	nodes[0].centroid_offset = (uint32_t)((size_t)(intern_dst -
												   (char *)result));
	memcpy(intern_dst, root_centroids, (size_t)fan_out * dim * sizeof(float));

	/* Child nodes: each is a leaf-parent */
	uint32_t leaf_off = 0;
	for (uint32_t c = 0; c < fan_out; c++)
	{
		uint32_t nidx = 1 + c;

		nodes[nidx].nchildren		= child_k[c];
		nodes[nidx].level			= 1;
		nodes[nidx].first_child		= HKMEANS_NO_CHILD;
		nodes[nidx].first_leaf		= leaf_off;
		nodes[nidx].centroid_offset = result->leaf_offset +
									  (uint32_t)((size_t)leaf_off * dim *
												 sizeof(float));

		if (child_centroids[c] != NULL && child_k[c] > 0)
		{
			memcpy(leaf_cents + (size_t)leaf_off * dim,
				   child_centroids[c],
				   (size_t)child_k[c] * dim * sizeof(float));
		}

		leaf_off += child_k[c];
	}

	return result;
}

uint32_t
mkt_hkmeans_assign(
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
			VectorRef	 qref	  = {.data = vec, .dim = dim};
			VectorRef	 cref	  = {.data = centroid, .dim = dim};
			Distance	 d		  = mkt_distance(qref, cref, metric);
			if (d < best_dist)
			{
				best_dist = d;
				best_c	  = c;
			}
		}

		bool is_leaf = (level == tree->nlevels - 1);
		if (is_leaf)
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
