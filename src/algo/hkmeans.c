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
mkt_hkmeans_f32(
		const float			*vectors,
		uint32_t			 nvecs,
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

	/* BFS work queue. The root clusters all vectors via identity
	 * mapping (vec_indices == NULL); children get explicit subsets. */
	HKWorkItem *queue  = mkt_alloc(max_nodes * sizeof(HKWorkItem));
	uint32_t	q_tail = 0;

	queue[q_tail++] = (HKWorkItem){
			.vec_indices  = NULL,
			.count		  = nvecs,
			.level		  = 0,
			.idx_in_level = 0,
	};

	bool	  ok	 = true;
	uint32_t *counts = mkt_alloc(fan_out * sizeof(uint32_t));

	/* Local copy so a NULL caller still gets sane defaults. */
	KMeansOptions local_opts = MKT_KMEANS_OPTIONS_DEFAULT;
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
		KMeansResult *km = mkt_kmeans(
				vectors,
				item.vec_indices,
				MKT_VEC_F32,
				item.count,
				dim,
				k,
				metric,
				&local_opts);

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

void
mkt_hkmeans_result_destroy(HKMeansResult *result)
{
	/* The whole tree is one contiguous allocation. */
	if (result != NULL)
		mkt_free(result);
}
