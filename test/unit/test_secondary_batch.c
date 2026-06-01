/*
 * test_secondary_batch.c - Batched secondary (boundary + SOAR) assignment
 *
 * Validates the CBLAS sgemm batched kernel against an independent
 * brute-force reference that scans all leaf centroids.
 */

#include <math.h>
#include <string.h>

#include "algo/hkmeans.h"
#include "algo/vecops.h"
#include "core/memory.h"
#include "index/posting_build.h"
#include "mkt_test.h"

TEST_GROUP(SecondaryBatch);
TEST_MEMCTX_FIXTURE();

static float *
make_clustered_data(
		uint32_t nclusters, uint32_t per_cluster, Dimension dim, uint64_t seed)
{
	uint32_t nvecs = nclusters * per_cluster;
	float	*data  = mkt_alloc((size_t)nvecs * dim * sizeof(float));

	uint32_t rng = (uint32_t)seed;
	for (uint32_t c = 0; c < nclusters; c++)
		for (uint32_t i = 0; i < per_cluster; i++)
		{
			uint32_t idx = c * per_cluster + i;
			for (Dimension d = 0; d < dim; d++)
			{
				rng			 = rng * 1103515245 + 12345;
				float noise	 = ((float)(rng % 1000) / 500.0f - 1.0f) * 0.3f;
				float center = (d == c % dim) ? 10.0f : 0.0f;
				data[(size_t)idx * dim + d] = center + noise;
			}
		}

	return data;
}

/*
 * Brute-force reference, mirroring the kernel's formulas (dot-based L2)
 * so the two agree up to sgemm accumulation order.
 */
static uint32_t
ref_secondary(
		const float			 *leaf_cents,
		const float			 *cent_norms,
		uint32_t			  nleaves,
		Dimension			  dim,
		const float			 *v,
		uint32_t			  p,
		float				  primary_dist,
		const MktBuildParams *bp)
{
	bool  has_b = bp->boundary_epsilon > 0.0;
	bool  has_s = bp->soar_lambda > 0.0;
	float nx	= mkt_l2_norm_squared(v, dim);

	float	 best2 = INFINITY;
	uint32_t c2	   = p;
	for (uint32_t j = 0; j < nleaves; j++)
	{
		if (j == p)
			continue;
		const float *c = leaf_cents + (size_t)j * dim;
		float d = nx + cent_norms[j] - 2.0f * mkt_dot_product(v, c, dim);
		if (d < best2)
		{
			best2 = d;
			c2	  = j;
		}
	}

	bool repl;
	if (has_b)
	{
		double pd  = (double)primary_dist;
		double gap = ((double)best2 - pd) / (pd != 0.0 ? fabs(pd) : 1.0);
		repl	   = (c2 != p) && (gap <= bp->boundary_epsilon);
	}
	else
		repl = true;

	if (!repl)
		return MKT_INVALID_CLUSTER;
	if (!has_s)
		return (c2 != p) ? c2 : MKT_INVALID_CLUSTER;

	/* SOAR */
	float *r	= mkt_alloc(dim * sizeof(float));
	float  norm = 0.0f;
	for (Dimension d = 0; d < dim; d++)
	{
		r[d] = v[d] - leaf_cents[(size_t)p * dim + d];
		norm += r[d] * r[d];
	}
	if (norm > 1e-7f)
	{
		float inv = 1.0f / sqrtf(norm);
		for (Dimension d = 0; d < dim; d++)
			r[d] *= inv;
	}
	float	 qrv	 = mkt_dot_product(r, v, dim);
	float	 best_oa = INFINITY;
	uint32_t bc		 = p;
	for (uint32_t j = 0; j < nleaves; j++)
	{
		if (j == p)
			continue;
		const float *c = leaf_cents + (size_t)j * dim;
		float l2  = nx + cent_norms[j] - 2.0f * mkt_dot_product(v, c, dim);
		float gap = qrv - mkt_dot_product(r, c, dim);
		float oa  = l2 + (float)bp->soar_lambda * gap * gap;
		if (oa < best_oa)
		{
			best_oa = oa;
			bc		= j;
		}
	}
	mkt_free(r);
	return (bc != p) ? bc : MKT_INVALID_CLUSTER;
}

static void
run_compare(MktTestResult *result, const MktBuildParams *bp, uint64_t seed)
{
	uint32_t  fan_out = 4;
	uint32_t  nlist	  = 16;
	Dimension dim	  = bp->dim;
	uint32_t  nvecs	  = 16 * 40;
	float	 *data	  = make_clustered_data(16, 40, dim, seed);

	KMeansOptions  opts = MKT_KMEANS_OPTIONS_DEFAULT;
	HKMeansResult *tree = mkt_hkmeans_f32(
			data, nvecs, NULL, dim, nlist, fan_out, DISTANCE_L2, &opts);
	ASSERT_NOT_NULL(tree, "tree built");

	const float *leaves	 = hk_leaf_centroids(tree);
	uint32_t	 nleaves = tree->nleaves;

	/* Reference centroid norms. */
	float *cent_norms = mkt_alloc(nleaves * sizeof(float));
	for (uint32_t j = 0; j < nleaves; j++)
		cent_norms[j] = mkt_l2_norm_squared(leaves + (size_t)j * dim, dim);

	/* Primary assignment per vector. */
	uint32_t *primary = mkt_alloc(nvecs * sizeof(uint32_t));
	float	 *pdist	  = mkt_alloc(nvecs * sizeof(float));
	uint32_t *ref	  = mkt_alloc(nvecs * sizeof(uint32_t));
	for (uint32_t i = 0; i < nvecs; i++)
	{
		Distance d;
		primary[i] = mkt_hkmeans_assign(
				tree, data + (size_t)i * dim, DISTANCE_L2, &d);
		pdist[i] = (float)d;
		ref[i]	 = ref_secondary(
				  leaves,
				  cent_norms,
				  nleaves,
				  dim,
				  data + (size_t)i * dim,
				  primary[i],
				  pdist[i],
				  bp);
	}

	/* Batched assignment. */
	MktSecondaryBatch sb;
	mkt_secondary_batch_init(&sb, leaves, nleaves, dim, nvecs);
	uint32_t *got = mkt_alloc(nvecs * sizeof(uint32_t));
	mkt_secondary_batch_assign(&sb, data, nvecs, primary, pdist, bp, got);
	mkt_secondary_batch_free(&sb);

	uint32_t mismatches = 0;
	for (uint32_t i = 0; i < nvecs; i++)
		if (got[i] != ref[i])
			mismatches++;
	ASSERT_EQ(
			0, mismatches, "batched secondary matches brute-force reference");

	mkt_free(got);
	mkt_free(ref);
	mkt_free(pdist);
	mkt_free(primary);
	mkt_free(cent_norms);
	mkt_free(tree);
	mkt_free(data);
}

TEST(available)
{
	ASSERT_TRUE(
			mkt_secondary_batch_available(),
			"CBLAS batched path available in tests");
}

TEST(soar_only_matches_reference)
{
	MktBuildParams bp = {
			.dim			  = 8,
			.metric			  = DISTANCE_L2,
			.soar_lambda	  = 1.0,
			.boundary_epsilon = 0.0,
	};
	run_compare(result, &bp, 17);
}

TEST(boundary_only_matches_reference)
{
	MktBuildParams bp = {
			.dim		 = 8,
			.metric		 = DISTANCE_L2,
			.soar_lambda = 0.0,
			.boundary_epsilon =
					10.0, /* generous: replicate whenever c2 != p */
	};
	run_compare(result, &bp, 23);
}

TEST(soar_and_boundary_matches_reference)
{
	MktBuildParams bp = {
			.dim			  = 8,
			.metric			  = DISTANCE_L2,
			.soar_lambda	  = 1.0,
			.boundary_epsilon = 0.5,
	};
	run_compare(result, &bp, 29);
}
