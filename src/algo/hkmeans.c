/*
 * hkmeans.c - Hierarchical k-means tree builder
 *
 * BFS tree construction using mkt_kmeans() with indexed access
 * at each node — avoids copying sub-vector arrays. Extracted
 * from bench_search.c so both the PG IAM build and CLI benchmark
 * share the same clustering logic.
 */

#include <math.h>
#include <string.h>

#include "algo/hkmeans.h"
#include "core/memory.h"

/* BFS work queue entry */
typedef struct HKWorkItem
{
	uint32_t *vec_indices; /* original vector indices (owned if level>0) */
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

	/* Result in caller's context */
	HKMeansResult *result = mkt_alloc(sizeof(HKMeansResult));
	result->nodes		  = mkt_alloc(max_nodes * sizeof(HKMeansNode));
	memset(result->nodes, 0, max_nodes * sizeof(HKMeansNode));
	result->nnodes	= 0;
	result->nlevels = nlevels;
	result->fan_out = fan_out;
	result->dim		= dim;
	result->nleaves = 0;

	/* Work context for BFS temporaries (queue, counts, sub_indices) */
	MktMemCtx work_ctx	 = mkt_memctx_create(NULL, "hkmeans_work");
	MktMemCtx caller_ctx = mkt_memctx_switch(work_ctx);

	/* BFS work queue */
	HKWorkItem *queue  = mkt_alloc(max_nodes * sizeof(HKWorkItem));
	uint32_t	q_tail = 0;

	/* Root: all vectors (NULL indices = identity mapping) */
	queue[q_tail++] = (HKWorkItem){
			.vec_indices  = NULL,
			.count		  = nvecs,
			.level		  = 0,
			.idx_in_level = 0,
	};

	bool	 ok			 = true;
	uint32_t level_start = 0;
	uint32_t level_end	 = q_tail; /* 1 after root enqueue */

	for (uint32_t level = 0; level < nlevels && ok; level++)
	{
		bool is_leaf_parent = (level == nlevels - 1);

		for (uint32_t qi = level_start; qi < level_end && ok; qi++)
		{
			HKWorkItem item = queue[qi];

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
					options);

			if (km == NULL)
			{
				ok = false;
				break;
			}

			/* Store node in result (allocate in caller ctx) */
			uint32_t	 node_idx = result->nnodes++;
			HKMeansNode *node	  = &result->nodes[node_idx];

			size_t cent_sz	= (size_t)km->nlist * dim * sizeof(float);
			node->centroids = mkt_memctx_alloc(caller_ctx, cent_sz);
			memcpy(node->centroids, km->centroids, cent_sz);
			node->nchildren	   = km->nlist;
			node->level		   = level;
			node->idx_in_level = item.idx_in_level;
			node->nvecs_in	   = item.count;
			node->first_child  = HKMEANS_NO_CHILD;

			/* Store vec_indices (generate identity for root) */
			size_t vi_sz	  = item.count * sizeof(uint32_t);
			node->vec_indices = mkt_memctx_alloc(caller_ctx, vi_sz);
			if (item.vec_indices != NULL)
				memcpy(node->vec_indices, item.vec_indices, vi_sz);
			else
			{
				for (uint32_t v = 0; v < item.count; v++)
					node->vec_indices[v] = v;
			}

			/* Copy assignments */
			size_t asgn_sz	  = item.count * sizeof(ClusterId);
			node->assignments = mkt_memctx_alloc(caller_ctx, asgn_sz);
			memcpy(node->assignments, km->assignments, asgn_sz);

			/* Count leaf centroids */
			if (is_leaf_parent)
				result->nleaves += km->nlist;

			/* Enqueue children for non-leaf nodes */
			if (!is_leaf_parent)
			{
				/*
				 * Count vectors per cluster from assignments.
				 *
				 * We cannot use km->cluster_sizes because k-means
				 * does a final reassignment after the last update
				 * step, so cluster_sizes may be stale.
				 */
				uint32_t *counts = mkt_alloc(km->nlist * sizeof(uint32_t));
				memset(counts, 0, km->nlist * sizeof(uint32_t));
				for (uint32_t v = 0; v < item.count; v++)
					counts[km->assignments[v]]++;

				bool first = true;
				for (uint32_t c = 0; c < km->nlist; c++)
				{
					uint32_t sub_n = counts[c];
					if (sub_n == 0)
						continue;

					uint32_t *sub_indices = mkt_alloc(
							sub_n * sizeof(uint32_t));
					uint32_t idx = 0;
					for (uint32_t v = 0; v < item.count; v++)
					{
						if (km->assignments[v] == c)
						{
							uint32_t orig	   = item.vec_indices
													   ? item.vec_indices[v]
													   : v;
							sub_indices[idx++] = orig;
						}
					}

					if (first)
					{
						node->first_child = q_tail;
						first			  = false;
					}

					queue[q_tail++] = (HKWorkItem){
							.vec_indices  = sub_indices,
							.count		  = sub_n,
							.level		  = level + 1,
							.idx_in_level = item.idx_in_level * fan_out + c,
					};
				}

				mkt_free(counts);
			}

			mkt_kmeans_result_destroy(km);
		}

		level_start = level_end;
		level_end	= q_tail;
	}

	/* Bulk-free all BFS temporaries */
	mkt_memctx_switch(caller_ctx);
	mkt_memctx_delete(work_ctx);

	if (!ok)
	{
		mkt_hkmeans_result_destroy(result);
		return NULL;
	}

	return result;
}

void
mkt_hkmeans_result_destroy(HKMeansResult *result)
{
	if (result == NULL)
		return;

	for (uint32_t i = 0; i < result->nnodes; i++)
	{
		HKMeansNode *n = &result->nodes[i];
		mkt_free(n->centroids);
		mkt_free(n->vec_indices);
		mkt_free(n->assignments);
	}
	mkt_free(result->nodes);
	mkt_free(result);
}

float *
mkt_hkmeans_leaf_centroids(const HKMeansResult *result)
{
	if (result == NULL || result->nleaves == 0)
		return NULL;

	uint32_t  leaf_level = result->nlevels - 1;
	Dimension dim		 = result->dim;
	float	 *out = mkt_alloc((size_t)result->nleaves * dim * sizeof(float));

	uint32_t offset = 0;
	for (uint32_t i = 0; i < result->nnodes; i++)
	{
		const HKMeansNode *node = &result->nodes[i];
		if (node->level != leaf_level)
			continue;

		memcpy(out + (size_t)offset * dim,
			   node->centroids,
			   (size_t)node->nchildren * dim * sizeof(float));
		offset += node->nchildren;
	}

	return out;
}
