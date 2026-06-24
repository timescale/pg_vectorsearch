/*
 * index_build.c - Shared index build utilities
 *
 * Generic helpers for building meerkat indexes, usable from both
 * the PostgreSQL IAM build and the standalone CLI.
 */

#include <math.h>

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
		const BlockNumber	*posting_heads,
		const BlockNumber	*node_first_blkno,
		const float			*pt_centroids)
{
	for (uint32_t i = 0; i < tree->nnodes; i++)
	{
		const HKMeansNode *node	   = &hk_nodes(tree)[i];
		bool			   is_leaf = (node->level == tree->nlevels - 1);

		uint16_t flags		 = is_leaf ? MKT_CENTROID_FLAG_LEAF : 0;
		uint16_t child_count = is_leaf ? 0 : (uint16_t)fan_out;

		const BlockNumber *child_blks;
		if (is_leaf && posting_heads != NULL)
			child_blks = &posting_heads[node->first_leaf];
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
	/*
	 * Orthogonality-amplified distance, decomposed so both terms use the
	 * SIMD vecops kernels:
	 *
	 *   oa(c) = ||v - c||^2 + lambda * (r_hat . (v - c))^2
	 *         = ||v - c||^2 + lambda * (r_hat.v - r_hat.c)^2
	 *
	 * r_hat.v is constant across centroids, so only ||v - c||^2 and
	 * r_hat.c are per-centroid. Since lambda * (...)^2 >= 0, ||v - c||^2
	 * is a lower bound on oa: when it already exceeds the running best we
	 * skip the dot product (exact pruning — no recall impact).
	 */
	float	 qrv	 = mkt_dot_product(normalized_residual, vec, dim);
	float	 lam	 = (float)lambda;
	float	 best_oa = INFINITY;
	uint32_t best_c	 = primary_cluster;

	for (uint32_t i = 0; i < nleaves; i++)
	{
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
