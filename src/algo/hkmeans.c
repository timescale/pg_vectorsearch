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
#include "algo/vecops.h"
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

	bool	  ok	 = true;
	uint32_t *counts = mkt_alloc(fan_out * sizeof(uint32_t));

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
				options);

		if (km == NULL)
		{
			ok = false;
			break;
		}

		/* Store node in result.
		 * Leaf-parent centroids go to work_ctx (consolidated later);
		 * internal node centroids go to caller_ctx (permanent). */
		uint32_t	 node_idx = result->nnodes++;
		HKMeansNode *node	  = &result->nodes[node_idx];

		node->level		  = item.level;
		node->first_child = HKMEANS_NO_CHILD;
		node->first_leaf  = 0;

		/* Count leaf centroids */
		if (is_leaf_parent)
		{
			size_t cent_sz	= (size_t)km->nlist * dim * sizeof(float);
			node->centroids = mkt_alloc(cent_sz);
			memcpy(node->centroids, km->centroids, cent_sz);
			node->nchildren = km->nlist;
			result->nleaves += km->nlist;
		}
		else
		/* Enqueue children for non-leaf nodes */
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
			 * Compact centroids to exclude empty clusters.
			 * The centroid array must be 1:1 with child nodes,
			 * since the build uses centroids[i] with
			 * child_blkno[i] from node_first_blkno.
			 */
			uint32_t nactive = 0;
			for (uint32_t c = 0; c < km->nlist; c++)
			{
				if (counts[c] > 0)
					nactive++;
			}

			size_t cent_sz	= (size_t)nactive * dim * sizeof(float);
			node->centroids = mkt_memctx_alloc(caller_ctx, cent_sz);
			node->nchildren = nactive;

			uint32_t compact_idx = 0;
			bool	 first		 = true;
			for (uint32_t c = 0; c < km->nlist; c++)
			{
				uint32_t sub_n = counts[c];
				if (sub_n == 0)
					continue;

				/* Copy centroid for this non-empty cluster */
				memcpy(node->centroids + (size_t)compact_idx * dim,
					   km->centroids + (size_t)c * dim,
					   dim * sizeof(float));
				compact_idx++;

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
					node->first_child = q_tail;
					first			  = false;
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

	/*
	 * Consolidate leaf centroids into a single flat array in
	 * caller context. Leaf-parent node->centroids then point
	 * into this array.
	 */
	result->leaf_centroids = mkt_memctx_alloc(
			caller_ctx, (size_t)result->nleaves * dim * sizeof(float));

	uint32_t leaf_off = 0;
	for (uint32_t i = 0; i < result->nnodes; i++)
	{
		HKMeansNode *node = &result->nodes[i];
		if (node->level != nlevels - 1)
			continue;
		node->first_leaf = leaf_off;
		memcpy(result->leaf_centroids + (size_t)leaf_off * dim,
			   node->centroids,
			   (size_t)node->nchildren * dim * sizeof(float));
		node->centroids = result->leaf_centroids + (size_t)leaf_off * dim;
		leaf_off += node->nchildren;
	}

	/* Bulk-free all BFS temporaries (including temp leaf centroids) */
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

	/* Internal node centroids are individually allocated;
	 * leaf-parent centroids point into result->leaf_centroids. */
	uint32_t leaf_level = result->nlevels - 1;
	for (uint32_t i = 0; i < result->nnodes; i++)
	{
		if (result->nodes[i].level != leaf_level)
			mkt_free(result->nodes[i].centroids);
	}
	mkt_free(result->leaf_centroids);
	mkt_free(result->nodes);
	mkt_free(result);
}

/* ----------------------------------------------------------------
 * Tree traversal for centroid assignment
 * ---------------------------------------------------------------- */

static Distance
hkmeans_dist(
		const float *a, const float *b, Dimension dim, DistanceMetric metric)
{
	switch (metric)
	{
	case DISTANCE_INNER_PRODUCT:
	case DISTANCE_COSINE:
		return -mkt_dot_product(a, b, dim);
	case DISTANCE_L2:
	default:
		return mkt_l2_distance_squared(a, b, dim);
	}
}

/*
 * Assign a vector to a leaf centroid via greedy tree descent.
 *
 * At each level, picks the nearest child centroid. Cost is
 * O(fan_out * nlevels) instead of O(nleaves) for brute force.
 *
 * For cosine metric, the input vector must be pre-normalized.
 */
uint32_t
mkt_hkmeans_assign(
		const HKMeansResult *tree,
		const float			*vec,
		DistanceMetric		 metric,
		Distance			*out_dist)
{
	Dimension dim  = tree->dim;
	uint32_t  node = 0; /* start at root */

	for (;;)
	{
		const HKMeansNode *n = &tree->nodes[node];

		/* Find nearest child centroid */
		uint32_t best	= 0;
		Distance best_d = hkmeans_dist(vec, n->centroids, dim, metric);

		for (uint32_t c = 1; c < n->nchildren; c++)
		{
			Distance d = hkmeans_dist(
					vec, n->centroids + (size_t)c * dim, dim, metric);
			if (d < best_d)
			{
				best_d = d;
				best   = c;
			}
		}

		/* Leaf-parent: return leaf index */
		if (n->first_child == HKMEANS_NO_CHILD)
		{
			if (out_dist)
				*out_dist = best_d;
			return n->first_leaf + best;
		}

		/* Descend to child */
		node = n->first_child + best;
	}
}
