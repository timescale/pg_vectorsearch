/*
 * build_parallel.c - Parallel index build support
 *
 * Per-vector processing functions shared between standalone and PG
 * build paths. No PostgreSQL dependencies.
 */

#include <math.h>
#include <string.h>

#include "algo/hkmeans.h"
#include "core/memory.h"
#include "index/build_parallel.h"
#include "index/index_build.h"
#include "mkt_types.h"

static void
normalize_vec(float *out, const float *in, Dimension dim)
{
	float norm = 0.0f;
	for (Dimension d = 0; d < dim; d++)
		norm += in[d] * in[d];

	if (norm > 0.0f)
	{
		float inv = 1.0f / sqrtf(norm);
		for (Dimension d = 0; d < dim; d++)
			out[d] = in[d] * inv;
	}
	else
	{
		memcpy(out, in, dim * sizeof(float));
	}
}

MktBuildWorkerBufs
mkt_build_worker_bufs_create(Dimension dim)
{
	return (MktBuildWorkerBufs){
			.norm_buf	  = mkt_alloc(dim * sizeof(float)),
			.residual_buf = mkt_alloc(dim * sizeof(float)),
	};
}

void
mkt_build_worker_bufs_free(MktBuildWorkerBufs *bufs)
{
	mkt_free(bufs->norm_buf);
	mkt_free(bufs->residual_buf);
	bufs->norm_buf	   = NULL;
	bufs->residual_buf = NULL;
}

MktBuildAssignment
mkt_build_assign_vector(
		const HKMeansResult	 *tree,
		const float			 *vec,
		const MktBuildParams *params,
		MktBuildWorkerBufs	 *bufs)
{
	const Dimension dim = params->dim;

	/* Find nearest centroid via tree descent */
	Distance min_dist;
	uint32_t best_c = mkt_hkmeans_assign(tree, vec, params->metric, &min_dist);

	/* Normalize for cosine so RaBitQ encodes in L2-equivalent space */
	const float *enc_vec = vec;
	if (params->metric == DISTANCE_COSINE)
	{
		normalize_vec(bufs->norm_buf, vec, dim);
		enc_vec = bufs->norm_buf;
	}

	/* Determine secondary cluster for SOAR/boundary replication */
	uint32_t secondary = MKT_INVALID_CLUSTER;

	bool has_soar	  = params->soar_lambda > 0.0;
	bool has_boundary = params->boundary_epsilon > 0.0;

	if (has_soar || has_boundary)
	{
		const float *leaves	 = tree->leaf_centroids;
		uint32_t	 nleaves = tree->nleaves;

		/* Boundary gate */
		uint32_t boundary_c2 = best_c;
		if (has_boundary)
			boundary_c2 = mkt_find_secondary_cluster(
					enc_vec,
					leaves,
					nleaves,
					dim,
					params->metric,
					best_c,
					min_dist,
					params->boundary_epsilon);

		bool should_replicate = has_boundary ? (boundary_c2 != best_c) : true;

		if (should_replicate)
		{
			if (has_soar)
			{
				/* SOAR placement: pick secondary by OA distance */
				const float *cent = leaves + (size_t)best_c * dim;
				float		*r	  = bufs->residual_buf;
				float		 norm = 0.0f;
				for (Dimension d = 0; d < dim; d++)
				{
					r[d] = enc_vec[d] - cent[d];
					norm += r[d] * r[d];
				}
				if (norm > 1e-7f)
				{
					float inv = 1.0f / sqrtf(norm);
					for (Dimension d = 0; d < dim; d++)
						r[d] *= inv;
				}
				secondary = mkt_find_soar_secondary(
						enc_vec,
						leaves,
						nleaves,
						dim,
						best_c,
						r,
						params->soar_lambda);
			}
			else
			{
				secondary = boundary_c2;
			}

			/* Don't replicate to the same cluster */
			if (secondary == best_c)
				secondary = MKT_INVALID_CLUSTER;
		}
	}

	return (MktBuildAssignment){
			.primary	= best_c,
			.secondary	= secondary,
			.enc_vector = enc_vec,
	};
}
