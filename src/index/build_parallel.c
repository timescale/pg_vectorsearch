/*
 * build_parallel.c - Parallel index build support
 *
 * Per-vector processing functions shared between standalone and PG
 * build paths. No PostgreSQL dependencies.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "algo/hkmeans.h"
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
	/* Use malloc (not mkt_alloc) because worker threads may not
	 * have a memory context in standalone mode. */
	return (MktBuildWorkerBufs){
			.norm_buf	  = malloc(dim * sizeof(float)),
			.residual_buf = malloc(dim * sizeof(float)),
	};
}

void
mkt_build_worker_bufs_free(MktBuildWorkerBufs *bufs)
{
	free(bufs->norm_buf);
	free(bufs->residual_buf);
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

/* ----------------------------------------------------------------
 * Batch parallel assignment via pthreads
 * ---------------------------------------------------------------- */

#include <pthread.h>

typedef struct BatchWorkerArg
{
	const float			 *vectors;
	uint32_t			  start;
	uint32_t			  end;
	Dimension			  dim;
	const HKMeansResult	 *tree;
	const MktBuildParams *params;
	uint32_t			 *out_primary;
	uint32_t			 *out_secondary;
} BatchWorkerArg;

static void *
batch_worker_fn(void *arg)
{
	BatchWorkerArg	  *w	= (BatchWorkerArg *)arg;
	MktBuildWorkerBufs bufs = mkt_build_worker_bufs_create(w->dim);

	for (uint32_t i = w->start; i < w->end; i++)
	{
		const float		  *vec = w->vectors + (size_t)i * w->dim;
		MktBuildAssignment asgn =
				mkt_build_assign_vector(w->tree, vec, w->params, &bufs);

		w->out_primary[i]	= asgn.primary;
		w->out_secondary[i] = asgn.secondary;
	}

	mkt_build_worker_bufs_free(&bufs);
	return NULL;
}

static uint32_t
detect_nthreads(void)
{
	long n = sysconf(_SC_NPROCESSORS_ONLN);
	if (n < 1)
		n = 1;
	if (n > 64)
		n = 64;
	return (uint32_t)n;
}

MktBatchAssignment
mkt_batch_assignment_create(uint32_t count)
{
	return (MktBatchAssignment){
			.primary   = malloc(count * sizeof(uint32_t)),
			.secondary = malloc(count * sizeof(uint32_t)),
			.count	   = count,
	};
}

void
mkt_batch_assignment_free(MktBatchAssignment *ba)
{
	free(ba->primary);
	free(ba->secondary);
	ba->primary	  = NULL;
	ba->secondary = NULL;
	ba->count	  = 0;
}

void
mkt_build_assign_batch_parallel(
		const float			 *vectors,
		uint32_t			  count,
		Dimension			  dim,
		const HKMeansResult	 *tree,
		const MktBuildParams *params,
		uint32_t			  nthreads,
		MktBatchAssignment	 *out)
{
	if (nthreads == 0)
		nthreads = detect_nthreads();
	if (nthreads > count)
		nthreads = count;
	if (nthreads <= 1)
	{
		/* Single-threaded fallback */
		MktBuildWorkerBufs bufs = mkt_build_worker_bufs_create(dim);
		for (uint32_t i = 0; i < count; i++)
		{
			const float		  *vec = vectors + (size_t)i * dim;
			MktBuildAssignment asgn =
					mkt_build_assign_vector(tree, vec, params, &bufs);
			out->primary[i]	  = asgn.primary;
			out->secondary[i] = asgn.secondary;
		}
		mkt_build_worker_bufs_free(&bufs);
		return;
	}

	pthread_t	   *threads = malloc(nthreads * sizeof(pthread_t));
	BatchWorkerArg *args	= malloc(nthreads * sizeof(BatchWorkerArg));

	uint32_t per_thread = count / nthreads;
	uint32_t remainder	= count % nthreads;

	uint32_t offset = 0;
	for (uint32_t t = 0; t < nthreads; t++)
	{
		uint32_t chunk = per_thread + (t < remainder ? 1 : 0);
		args[t]		   = (BatchWorkerArg){
					   .vectors		  = vectors,
					   .start		  = offset,
					   .end			  = offset + chunk,
					   .dim			  = dim,
					   .tree		  = tree,
					   .params		  = params,
					   .out_primary	  = out->primary,
					   .out_secondary = out->secondary,
		   };
		pthread_create(&threads[t], NULL, batch_worker_fn, &args[t]);
		offset += chunk;
	}

	for (uint32_t t = 0; t < nthreads; t++)
		pthread_join(threads[t], NULL);

	free(threads);
	free(args);
}
