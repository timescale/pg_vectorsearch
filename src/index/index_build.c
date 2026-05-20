/*
 * index_build.c - Shared index build utilities
 *
 * Generic helpers for building meerkat indexes, usable from both
 * the PostgreSQL IAM build and the standalone CLI.
 */

#include <math.h>

#include "algo/distance.h"
#include "algo/vecops.h"
#include "core/log.h"
#include "core/memory.h"
#include "index/index_build.h"

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
		uint32_t n		= tree->nodes[i].nchildren;
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
		const BlockNumber	*posting_heads,
		const BlockNumber	*node_first_blkno,
		const float			*pt_centroids)
{
	for (uint32_t i = 0; i < tree->nnodes; i++)
	{
		HKMeansNode *node	 = &tree->nodes[i];
		bool		 is_leaf = (node->level == tree->nlevels - 1);

		uint16_t flags		 = is_leaf ? MKT_CENTROID_FLAG_LEAF : 0;
		uint16_t child_count = is_leaf ? 0 : (uint16_t)fan_out;

		CentroidEncoderState enc_state;
		CentroidEncoder		*encoder = centroid_encoder_init(
				&enc_state,
				centroid_format,
				node->centroids,
				dim,
				rq_params,
				global_mean);

		const BlockNumber *child_blks;
		if (is_leaf && posting_heads != NULL)
			child_blks = &posting_heads[node->first_leaf];
		else if (!is_leaf)
			child_blks = &node_first_blkno[node->first_child];
		else
			child_blks = NULL;

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
mkt_find_secondary_cluster(
		const float	  *vec,
		const float	  *leaf_centroids,
		uint32_t	   nleaves,
		Dimension	   dim,
		DistanceMetric metric,
		uint32_t	   primary_cluster,
		Distance	   primary_dist,
		double		   epsilon)
{
	VectorRef qref	= {.data = vec, .dim = dim};
	Distance  best2 = INFINITY;
	uint32_t  c2	= primary_cluster;

	for (uint32_t i = 0; i < nleaves; i++)
	{
		if (i == primary_cluster)
			continue;
		VectorRef cref =
				{.data = leaf_centroids + (size_t)i * dim, .dim = dim};
		Distance d = mkt_distance(qref, cref, metric);
		if (d < best2)
		{
			best2 = d;
			c2	  = i;
		}
	}

	double gap		 = (double)best2 - (double)primary_dist;
	double gap_ratio = (primary_dist != 0.0) ? gap / fabs((double)primary_dist)
											 : INFINITY;

	if (c2 != primary_cluster && gap_ratio <= epsilon)
		return c2;

	return primary_cluster;
}

uint32_t
mkt_find_soar_secondary(
		const float *vec,
		const float *leaf_centroids,
		uint32_t	 nleaves,
		Dimension	 dim,
		uint32_t	 primary_cluster,
		const float *normalized_residual,
		double		 lambda)
{
	Distance best_oa = INFINITY;
	uint32_t best_c	 = primary_cluster;

	for (uint32_t i = 0; i < nleaves; i++)
	{
		if (i == primary_cluster)
			continue;

		const float *cent	 = leaf_centroids + (size_t)i * dim;
		double		 sq_dist = 0.0;
		double		 dot	 = 0.0;

		for (Dimension d = 0; d < dim; d++)
		{
			double diff = (double)vec[d] - (double)cent[d];
			sq_dist += diff * diff;
			dot += diff * (double)normalized_residual[d];
		}

		Distance oa = (Distance)(sq_dist + lambda * dot * dot);
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
	mkt_log("build: sample %.1fms, kmeans %.1fms, setup %.1fms, "
			"posting %.1fms "
			"(parallel %.1fms + merge %.1fms, "
			"%u workers + leader, %u pages, "
			"%u partial pages merged into %u), "
			"centroid %.1fms, total %.1fms\n",
			s->ms_sample,
			s->ms_kmeans,
			s->ms_setup,
			s->ms_posting,
			s->ms_parallel,
			s->ms_merge,
			s->nworkers,
			s->total_pages,
			s->merge_input,
			s->merge_output,
			s->ms_centroid,
			s->ms_total);
}
