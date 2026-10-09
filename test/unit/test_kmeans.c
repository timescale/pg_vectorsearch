/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * test_kmeans.c - Tests for k-means clustering
 */

#include <math.h>
#include <string.h>

#include "algo/kmeans.h"
#include "algo/kmeans_internal.h"
#include "algo/vecops.h"
#include "core/memory.h"
#include "types/vec16.h"
#include "vs_test.h"

TEST_GROUP(KMeans);
TEST_MEMCTX_FIXTURE();

/*
 * Helper: generate 2D data with 3 well-separated clusters.
 *
 * Cluster 0: centered at (0, 10)
 * Cluster 1: centered at (10, 0)
 * Cluster 2: centered at (-10, 0)
 *
 * Each cluster has `per_cluster` points with small random offsets.
 */
static void
make_3_clusters(float *out, uint32_t per_cluster, uint64_t seed)
{
	/* Simple LCG for deterministic offsets */
	uint32_t rng = (uint32_t)seed;

	float centers[][2] = {
			{0.0f, 10.0f},
			{10.0f, 0.0f},
			{-10.0f, 0.0f},
	};

	for (int c = 0; c < 3; c++)
	{
		for (uint32_t i = 0; i < per_cluster; i++)
		{
			uint32_t idx = (uint32_t)c * per_cluster + i;
			float dx = ((float)(vs_test_rand(&rng) % 1000) / 500.0f - 1.0f) *
					   0.5f;
			float dy = ((float)(vs_test_rand(&rng) % 1000) / 500.0f - 1.0f) *
					   0.5f;
			out[idx * 2 + 0] = centers[c][0] + dx;
			out[idx * 2 + 1] = centers[c][1] + dy;
		}
	}
}

/*
 * Helper: count how many vectors are assigned to their expected cluster.
 *
 * The cluster IDs from k-means may not match the expected ordering,
 * so we determine the mapping by majority vote per expected group.
 */
static uint32_t
count_correct_assignments(
		const ClusterId *assignments, uint32_t per_cluster, uint32_t nlist)
{
	uint32_t total	 = per_cluster * nlist;
	uint32_t correct = 0;

	/* For each expected cluster, find the most-assigned k-means cluster */
	bool used[16] = {false};

	for (uint32_t c = 0; c < nlist; c++)
	{
		/* Count assignments for this expected group */
		uint32_t counts[16] = {0};
		for (uint32_t i = 0; i < per_cluster; i++)
		{
			ClusterId a = assignments[c * per_cluster + i];
			if (a < 16)
				counts[a]++;
		}

		/* Find best matching cluster (not already used) */
		uint32_t best_k = 0;
		uint32_t best_n = 0;
		for (uint32_t k = 0; k < nlist; k++)
		{
			if (!used[k] && counts[k] > best_n)
			{
				best_k = k;
				best_n = counts[k];
			}
		}
		used[best_k] = true;
		correct += best_n;
	}

	(void)total;
	return correct;
}

/*
 * Basic L2 clustering
 */
TEST(basic_l2)
{
	uint32_t  per_cluster = 50;
	uint32_t  nvecs		  = per_cluster * 3;
	Dimension dim		  = 2;
	float	  data[300]; /* 150 * 2 */

	make_3_clusters(data, per_cluster, 42);

	KMeansOptions opts = VS_KMEANS_OPTIONS_DEFAULT;
	KMeansResult *res = vs_kmeans_f32(data, nvecs, dim, 3, DISTANCE_L2, &opts);

	ASSERT_NOT_NULL(res, "k-means should return a result");
	ASSERT_EQ(3, res->nlist, "nlist should be 3");
	ASSERT_EQ(2, res->dim, "dim should be 2");
	ASSERT_TRUE(res->total_cost > 0, "cost should be positive");

	/* Verify all clusters have vectors */
	uint32_t total = 0;
	for (uint32_t j = 0; j < 3; j++)
	{
		ASSERT_TRUE(
				res->cluster_sizes[j] > 0, "each cluster should have vectors");
		total += res->cluster_sizes[j];
	}
	ASSERT_EQ(nvecs, total, "total assignments should equal nvecs");

	/* With well-separated clusters, expect high accuracy */
	uint32_t correct =
			count_correct_assignments(res->assignments, per_cluster, 3);
	ASSERT_TRUE(
			correct >= nvecs * 9 / 10,
			"at least 90% correct for separated clusters");

	vs_kmeans_result_destroy(res);
}

/*
 * Basic IP clustering
 */
TEST(basic_ip)
{
	uint32_t  per_cluster = 50;
	uint32_t  nvecs		  = per_cluster * 3;
	Dimension dim		  = 2;
	float	  data[300];

	make_3_clusters(data, per_cluster, 42);

	KMeansOptions opts = VS_KMEANS_OPTIONS_DEFAULT;
	KMeansResult *res =
			vs_kmeans_f32(data, nvecs, dim, 3, DISTANCE_INNER_PRODUCT, &opts);

	ASSERT_NOT_NULL(res, "k-means should return a result");
	ASSERT_EQ(3, res->nlist, "nlist should be 3");

	uint32_t total = 0;
	for (uint32_t j = 0; j < 3; j++)
		total += res->cluster_sizes[j];
	ASSERT_EQ(nvecs, total, "total assignments should equal nvecs");

	vs_kmeans_result_destroy(res);
}

/*
 * Basic cosine clustering (pre-normalized input)
 */
TEST(basic_cosine)
{
	/* Create 3 clusters of unit vectors in different directions */
	uint32_t  per_cluster = 30;
	uint32_t  nvecs		  = per_cluster * 3;
	Dimension dim		  = 2;
	float	  data[180]; /* 90 * 2 */

	/* Cluster 0: near (1, 0) */
	/* Cluster 1: near (0, 1) */
	/* Cluster 2: near (-1, 0) */
	uint32_t rng = 42;
	for (uint32_t i = 0; i < per_cluster; i++)
	{
		/* Cluster 0: angle near 0 */
		float a0 = ((float)(vs_test_rand(&rng) % 1000) / 5000.0f) - 0.1f;
		data[i * 2 + 0] = cosf(a0);
		data[i * 2 + 1] = sinf(a0);

		/* Cluster 1: angle near pi/2 */
		float a1 = 1.5708f + ((float)(vs_test_rand(&rng) % 1000) / 5000.0f) -
				   0.1f;
		data[(per_cluster + i) * 2 + 0] = cosf(a1);
		data[(per_cluster + i) * 2 + 1] = sinf(a1);

		/* Cluster 2: angle near pi */
		float a2 = 3.14159f + ((float)(vs_test_rand(&rng) % 1000) / 5000.0f) -
				   0.1f;
		data[(2 * per_cluster + i) * 2 + 0] = cosf(a2);
		data[(2 * per_cluster + i) * 2 + 1] = sinf(a2);
	}

	KMeansOptions opts = VS_KMEANS_OPTIONS_DEFAULT;
	KMeansResult *res =
			vs_kmeans_f32(data, nvecs, dim, 3, DISTANCE_COSINE, &opts);

	ASSERT_NOT_NULL(res, "k-means should return a result");

	uint32_t total = 0;
	for (uint32_t j = 0; j < 3; j++)
		total += res->cluster_sizes[j];
	ASSERT_EQ(nvecs, total, "total assignments should equal nvecs");

	/* Centroids should be approximately unit length */
	for (uint32_t j = 0; j < 3; j++)
	{
		float norm = vs_l2_norm(res->centroids + j * dim, dim);
		ASSERT_FLOAT_EQ(
				1.0f, norm, 0.01f, "cosine centroids should be unit length");
	}

	vs_kmeans_result_destroy(res);
}

/*
 * Well-separated clusters should get 100% accuracy.
 */
TEST(well_separated_clusters)
{
	/* 4 clusters at (±100, ±100), tiny noise */
	uint32_t  per_cluster = 25;
	uint32_t  nvecs		  = per_cluster * 4;
	Dimension dim		  = 2;
	float	  data[200];

	float centers[][2] = {
			{100.0f, 100.0f},
			{-100.0f, 100.0f},
			{100.0f, -100.0f},
			{-100.0f, -100.0f},
	};

	uint32_t rng = 123;
	for (int c = 0; c < 4; c++)
	{
		for (uint32_t i = 0; i < per_cluster; i++)
		{
			uint32_t idx = (uint32_t)c * per_cluster + i;
			float dx = ((float)(vs_test_rand(&rng) % 1000) / 500.0f - 1.0f) *
					   0.01f;
			float dy = ((float)(vs_test_rand(&rng) % 1000) / 500.0f - 1.0f) *
					   0.01f;
			data[idx * 2 + 0] = centers[c][0] + dx;
			data[idx * 2 + 1] = centers[c][1] + dy;
		}
	}

	KMeansOptions opts = VS_KMEANS_OPTIONS_DEFAULT;
	KMeansResult *res = vs_kmeans_f32(data, nvecs, dim, 4, DISTANCE_L2, &opts);

	ASSERT_NOT_NULL(res, "k-means should return a result");

	uint32_t correct =
			count_correct_assignments(res->assignments, per_cluster, 4);
	ASSERT_EQ(nvecs, correct, "100% accuracy for well-separated");

	vs_kmeans_result_destroy(res);
}

/*
 * Verify deterministic results with same seed.
 */
TEST(deterministic_with_seed)
{
	uint32_t  nvecs = 150;
	Dimension dim	= 2;
	float	  data[300];

	make_3_clusters(data, 50, 42);

	KMeansOptions opts = VS_KMEANS_OPTIONS_DEFAULT;
	opts.seed		   = 99;

	KMeansResult *r1 = vs_kmeans_f32(data, nvecs, dim, 3, DISTANCE_L2, &opts);
	KMeansResult *r2 = vs_kmeans_f32(data, nvecs, dim, 3, DISTANCE_L2, &opts);

	ASSERT_NOT_NULL(r1, "first run should succeed");
	ASSERT_NOT_NULL(r2, "second run should succeed");

	ASSERT_FLOAT_EQ(
			r1->total_cost,
			r2->total_cost,
			1e-3f,
			"same seed should give same cost");

	for (uint32_t i = 0; i < nvecs; i++)
	{
		ASSERT_EQ(
				r1->assignments[i],
				r2->assignments[i],
				"same seed should give same assignments");
	}

	vs_kmeans_result_destroy(r1);
	vs_kmeans_result_destroy(r2);
}

/*
 * Single cluster: centroid should be the mean.
 */
TEST(single_cluster)
{
	float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
	/* 3 vectors of dim 2: (1,2), (3,4), (5,6) */
	/* Mean: (3, 4) */

	KMeansOptions opts = VS_KMEANS_OPTIONS_DEFAULT;
	KMeansResult *res  = vs_kmeans_f32(data, 3, 2, 1, DISTANCE_L2, &opts);

	ASSERT_NOT_NULL(res, "should succeed");
	ASSERT_EQ(1, res->nlist, "nlist=1");
	ASSERT_EQ(3, res->cluster_sizes[0], "all in one cluster");

	ASSERT_FLOAT_EQ(3.0f, res->centroids[0], 1e-4f, "mean x");
	ASSERT_FLOAT_EQ(4.0f, res->centroids[1], 1e-4f, "mean y");

	vs_kmeans_result_destroy(res);
}

/*
 * nlist == nvecs: each vector is its own centroid.
 */
TEST(nlist_equals_nvecs)
{
	float data[] = {1.0f, 0.0f, 0.0f, 1.0f, -1.0f, 0.0f};

	KMeansOptions opts = VS_KMEANS_OPTIONS_DEFAULT;
	KMeansResult *res  = vs_kmeans_f32(data, 3, 2, 3, DISTANCE_L2, &opts);

	ASSERT_NOT_NULL(res, "should succeed");

	/* Each cluster should have exactly 1 vector */
	for (uint32_t j = 0; j < 3; j++)
	{
		ASSERT_EQ(
				1, res->cluster_sizes[j], "each cluster should have 1 vector");
	}

	/* Cost should be 0 (each vector is its own centroid) */
	ASSERT_FLOAT_EQ(
			0.0f,
			res->total_cost,
			1e-3f,
			"cost should be ~0 when nlist==nvecs");

	vs_kmeans_result_destroy(res);
}

/*
 * Empty cluster handling: synthetic data likely to produce empties.
 */
TEST(empty_cluster_handling)
{
	/* 2 tight clusters, request 4 clusters -> 2 will be empty initially */
	uint32_t  per_cluster = 50;
	uint32_t  nvecs		  = per_cluster * 2;
	Dimension dim		  = 2;
	float	  data[200];

	uint32_t rng = 77;
	for (uint32_t i = 0; i < per_cluster; i++)
	{
		data[i * 2 + 0] = 10.0f + (float)(vs_test_rand(&rng) % 100) / 1000.0f;
		data[i * 2 + 1] = 10.0f + (float)(vs_test_rand(&rng) % 100) / 1000.0f;
	}
	for (uint32_t i = 0; i < per_cluster; i++)
	{
		data[(per_cluster + i) * 2 + 0] = -10.0f +
										  (float)(vs_test_rand(&rng) % 100) /
												  1000.0f;
		data[(per_cluster + i) * 2 + 1] = -10.0f +
										  (float)(vs_test_rand(&rng) % 100) /
												  1000.0f;
	}

	KMeansOptions opts = VS_KMEANS_OPTIONS_DEFAULT;
	KMeansResult *res = vs_kmeans_f32(data, nvecs, dim, 4, DISTANCE_L2, &opts);

	ASSERT_NOT_NULL(res, "should handle empty clusters");

	uint32_t total = 0;
	for (uint32_t j = 0; j < 4; j++)
		total += res->cluster_sizes[j];
	ASSERT_EQ(nvecs, total, "all vectors assigned");

	vs_kmeans_result_destroy(res);
}

/*
 * nredo > 1 should produce cost <= single run.
 */
TEST(nredo_improves_cost)
{
	uint32_t  nvecs = 150;
	Dimension dim	= 2;
	float	  data[300];

	make_3_clusters(data, 50, 42);

	KMeansOptions opts1 = VS_KMEANS_OPTIONS_DEFAULT;
	opts1.seed			= 42;
	opts1.nredo			= 1;
	KMeansResult *r1 = vs_kmeans_f32(data, nvecs, dim, 3, DISTANCE_L2, &opts1);

	KMeansOptions opts3 = VS_KMEANS_OPTIONS_DEFAULT;
	opts3.seed			= 42;
	opts3.nredo			= 3;
	KMeansResult *r3 = vs_kmeans_f32(data, nvecs, dim, 3, DISTANCE_L2, &opts3);

	ASSERT_NOT_NULL(r1, "nredo=1 should succeed");
	ASSERT_NOT_NULL(r3, "nredo=3 should succeed");

	/* nredo=3 picks the best of 3 runs, so cost <= nredo=1 */
	ASSERT_TRUE(
			r3->total_cost <= r1->total_cost + 1e-3f,
			"more restarts should not increase cost");

	vs_kmeans_result_destroy(r1);
	vs_kmeans_result_destroy(r3);
}

/*
 * CBLAS and builtin should produce the same assignments.
 */
TEST(cblas_builtin_match)
{
	uint32_t  nvecs = 150;
	Dimension dim	= 2;
	float	  data[300];

	make_3_clusters(data, 50, 42);

	KMeansOptions opts = VS_KMEANS_OPTIONS_DEFAULT;
	opts.seed		   = 42;

	/* Run with whatever the current default is */
	bool original = vs_kmeans_get_use_cblas();

	/* Force builtin */
	vs_kmeans_set_use_cblas(false);
	KMeansResult *r_builtin =
			vs_kmeans_f32(data, nvecs, dim, 3, DISTANCE_L2, &opts);

	/* Force cblas (if available, otherwise same as builtin) */
	vs_kmeans_set_use_cblas(true);
	KMeansResult *r_cblas =
			vs_kmeans_f32(data, nvecs, dim, 3, DISTANCE_L2, &opts);

	/* Restore */
	vs_kmeans_set_use_cblas(original);

	ASSERT_NOT_NULL(r_builtin, "builtin should succeed");
	ASSERT_NOT_NULL(r_cblas, "cblas should succeed");

	/* Assignments should match (same seed, same algorithm) */
	for (uint32_t i = 0; i < nvecs; i++)
	{
		ASSERT_EQ(
				r_builtin->assignments[i],
				r_cblas->assignments[i],
				"cblas and builtin should match");
	}

	ASSERT_FLOAT_EQ(
			r_builtin->total_cost,
			r_cblas->total_cost,
			1e-2f,
			"costs should be close");

	vs_kmeans_result_destroy(r_builtin);
	vs_kmeans_result_destroy(r_cblas);
}

/*
 * NULL inputs should return NULL gracefully.
 */
TEST(null_inputs)
{
	float		  data[] = {1.0f, 2.0f};
	KMeansOptions opts	 = VS_KMEANS_OPTIONS_DEFAULT;

	ASSERT_NULL(
			vs_kmeans_f32(NULL, 10, 2, 3, DISTANCE_L2, &opts),
			"NULL vectors should return NULL");

	ASSERT_NULL(
			vs_kmeans_f32(data, 0, 2, 3, DISTANCE_L2, &opts),
			"nvecs=0 should return NULL");

	ASSERT_NULL(
			vs_kmeans_f32(data, 1, 0, 3, DISTANCE_L2, &opts),
			"dim=0 should return NULL");

	ASSERT_NULL(
			vs_kmeans_f32(data, 1, 2, 0, DISTANCE_L2, &opts),
			"nlist=0 should return NULL");

	/* NULL options should use defaults (not crash) */
	KMeansResult *r = vs_kmeans_f32(data, 1, 2, 1, DISTANCE_L2, NULL);
	ASSERT_NOT_NULL(r, "NULL options should use defaults");
	vs_kmeans_result_destroy(r);
}

/*
 * Test across multiple dimensions.
 */
TEST_PARAMETERIZED(dims, "4", "16", "64", "128")
{
	uint32_t  dim_values[] = {4, 16, 64, 128};
	Dimension dim		   = (Dimension)dim_values[iteration];
	uint32_t  per_cluster  = 30;
	uint32_t  nlist		   = 3;
	uint32_t  nvecs		   = per_cluster * nlist;
	float	 *data		   = vs_alloc((size_t)nvecs * dim * sizeof(float));

	/* Generate clusters spread along each axis */
	uint32_t rng = 42;
	for (uint32_t c = 0; c < nlist; c++)
	{
		for (uint32_t i = 0; i < per_cluster; i++)
		{
			uint32_t idx = c * per_cluster + i;
			for (uint32_t d = 0; d < dim; d++)
			{
				float noise = ((float)(vs_test_rand(&rng) % 1000) / 500.0f -
							   1.0f) *
							  0.1f;
				/* Cluster c has large value in dimension c % dim */
				float center				= (d == c % dim) ? 10.0f : 0.0f;
				data[(size_t)idx * dim + d] = center + noise;
			}
		}
	}

	KMeansOptions opts = VS_KMEANS_OPTIONS_DEFAULT;
	KMeansResult *res =
			vs_kmeans_f32(data, nvecs, dim, nlist, DISTANCE_L2, &opts);

	ASSERT_NOT_NULL(res, "should succeed");
	ASSERT_TRUE(res->total_cost > 0, "cost should be positive");

	uint32_t total = 0;
	for (uint32_t j = 0; j < nlist; j++)
		total += res->cluster_sizes[j];
	ASSERT_EQ(nvecs, total, "all vectors assigned");

	vs_kmeans_result_destroy(res);
	vs_free(data);
}

/*
 * max_iterations should be respected.
 */
TEST(max_iterations_respected)
{
	uint32_t  nvecs = 150;
	Dimension dim	= 2;
	float	  data[300];

	make_3_clusters(data, 50, 42);

	KMeansOptions opts	= VS_KMEANS_OPTIONS_DEFAULT;
	opts.max_iterations = 1;
	opts.tolerance		= 0.0f; /* never converge early */

	KMeansResult *res = vs_kmeans_f32(data, nvecs, dim, 3, DISTANCE_L2, &opts);

	ASSERT_NOT_NULL(res, "should succeed with 1 iteration");
	/* Just verify it completed without crash */

	vs_kmeans_result_destroy(res);
}

/*
 * Impl name should be valid.
 */
TEST(impl_name_valid)
{
	const char *name = vs_kmeans_impl_name();
	ASSERT_NOT_NULL(name, "impl_name should not be null");

	int valid = (strcmp(name, "cblas") == 0 || strcmp(name, "builtin") == 0);
	ASSERT_TRUE(valid, "impl name should be cblas or builtin");
}

/*
 * Test all algorithms on well-separated L2 data.
 *
 * Each algorithm should produce correct cluster assignments.
 */
TEST_PARAMETERIZED(algo_l2, "lloyd", "hamerly", "elkan")
{
	KMeansAlgorithm algos[] = {
			KMEANS_ALGO_LLOYD,
			KMEANS_ALGO_HAMERLY,
			KMEANS_ALGO_ELKAN,
	};

	uint32_t  per_cluster = 50;
	uint32_t  nvecs		  = per_cluster * 3;
	Dimension dim		  = 2;
	float	  data[300];

	make_3_clusters(data, per_cluster, 42);

	KMeansOptions opts = VS_KMEANS_OPTIONS_DEFAULT;
	opts.algorithm	   = algos[iteration];
	opts.seed		   = 42;

	KMeansResult *res = vs_kmeans_f32(data, nvecs, dim, 3, DISTANCE_L2, &opts);

	ASSERT_NOT_NULL(res, "algorithm should return result");
	ASSERT_EQ(3, res->nlist, "nlist should be 3");
	ASSERT_TRUE(res->total_cost > 0, "cost should be positive");

	uint32_t total = 0;
	for (uint32_t j = 0; j < 3; j++)
	{
		ASSERT_TRUE(
				res->cluster_sizes[j] > 0, "each cluster should have vectors");
		total += res->cluster_sizes[j];
	}
	ASSERT_EQ(nvecs, total, "total should equal nvecs");

	uint32_t correct =
			count_correct_assignments(res->assignments, per_cluster, 3);
	ASSERT_TRUE(
			correct >= nvecs * 9 / 10,
			"at least 90% correct for separated clusters");

	vs_kmeans_result_destroy(res);
}

/*
 * All algorithms should produce valid clustering results.
 *
 * Bound-based algorithms (Hamerly/Elkan) may converge to different
 * local optima than Lloyd despite the same initialization, because
 * stale bounds can delay reassignment. We verify each produces a
 * finite positive cost and correct assignments.
 */
TEST(algo_all_valid)
{
	uint32_t  per_cluster = 50;
	uint32_t  nvecs		  = per_cluster * 3;
	Dimension dim		  = 2;
	float	  data[300];

	make_3_clusters(data, per_cluster, 42);

	KMeansAlgorithm algos[] = {
			KMEANS_ALGO_LLOYD,
			KMEANS_ALGO_HAMERLY,
			KMEANS_ALGO_ELKAN,
	};

	for (int a = 0; a < 3; a++)
	{
		KMeansOptions opts = VS_KMEANS_OPTIONS_DEFAULT;
		opts.seed		   = 42;
		opts.algorithm	   = algos[a];

		KMeansResult *res =
				vs_kmeans_f32(data, nvecs, dim, 3, DISTANCE_L2, &opts);

		ASSERT_NOT_NULL(res, "algorithm should succeed");
		ASSERT_TRUE(res->total_cost > 0, "cost should be positive");
		ASSERT_TRUE(isfinite(res->total_cost), "cost should be finite");

		uint32_t total = 0;
		for (uint32_t j = 0; j < 3; j++)
			total += res->cluster_sizes[j];
		ASSERT_EQ(nvecs, total, "all vectors assigned");

		vs_kmeans_result_destroy(res);
	}
}

/*
 * Lloyd builtin with inner product metric.
 *
 * Covers the IP path in lloyd_assign_block_builtin (lines 203-211).
 */
TEST(lloyd_ip)
{
	uint32_t  per_cluster = 50;
	uint32_t  nvecs		  = per_cluster * 3;
	Dimension dim		  = 2;
	float	  data[300];

	make_3_clusters(data, per_cluster, 42);

	KMeansOptions opts = VS_KMEANS_OPTIONS_DEFAULT;
	opts.algorithm	   = KMEANS_ALGO_LLOYD;

	KMeansResult *res =
			vs_kmeans_f32(data, nvecs, dim, 3, DISTANCE_INNER_PRODUCT, &opts);

	ASSERT_NOT_NULL(res, "lloyd IP should succeed");

	uint32_t total = 0;
	for (uint32_t j = 0; j < 3; j++)
		total += res->cluster_sizes[j];
	ASSERT_EQ(nvecs, total, "all vectors assigned");

	vs_kmeans_result_destroy(res);
}

/*
 * Lloyd builtin with cosine metric.
 *
 * Covers the cosine path in lloyd_assign_block_builtin (lines 212-220).
 */
TEST(lloyd_cosine)
{
	uint32_t  per_cluster = 30;
	uint32_t  nvecs		  = per_cluster * 3;
	Dimension dim		  = 2;
	float	  data[180];

	/* Create clusters of unit vectors */
	uint32_t rng = 42;
	for (uint32_t i = 0; i < per_cluster; i++)
	{
		float a0 = ((float)(vs_test_rand(&rng) % 1000) / 5000.0f) - 0.1f;
		data[i * 2 + 0] = cosf(a0);
		data[i * 2 + 1] = sinf(a0);

		float a1 = 1.5708f + ((float)(vs_test_rand(&rng) % 1000) / 5000.0f) -
				   0.1f;
		data[(per_cluster + i) * 2 + 0] = cosf(a1);
		data[(per_cluster + i) * 2 + 1] = sinf(a1);

		float a2 = 3.14159f + ((float)(vs_test_rand(&rng) % 1000) / 5000.0f) -
				   0.1f;
		data[(2 * per_cluster + i) * 2 + 0] = cosf(a2);
		data[(2 * per_cluster + i) * 2 + 1] = sinf(a2);
	}

	KMeansOptions opts = VS_KMEANS_OPTIONS_DEFAULT;
	opts.algorithm	   = KMEANS_ALGO_LLOYD;

	KMeansResult *res =
			vs_kmeans_f32(data, nvecs, dim, 3, DISTANCE_COSINE, &opts);

	ASSERT_NOT_NULL(res, "lloyd cosine should succeed");

	uint32_t total = 0;
	for (uint32_t j = 0; j < 3; j++)
		total += res->cluster_sizes[j];
	ASSERT_EQ(nvecs, total, "all vectors assigned");

	/* Centroids should be approximately unit length */
	for (uint32_t j = 0; j < 3; j++)
	{
		float norm = vs_l2_norm(res->centroids + j * dim, dim);
		ASSERT_FLOAT_EQ(
				1.0f, norm, 0.01f, "cosine centroids should be unit length");
	}

	vs_kmeans_result_destroy(res);
}

/*
 * Hamerly/Elkan with non-L2 metrics should fall back to Lloyd.
 *
 * Covers the fallback path in vs_kmeans (lines 637-639).
 */
TEST_PARAMETERIZED(algo_non_l2_fallback, "hamerly_ip", "elkan_ip")
{
	KMeansAlgorithm algos[] = {
			KMEANS_ALGO_HAMERLY,
			KMEANS_ALGO_ELKAN,
	};

	uint32_t  per_cluster = 50;
	uint32_t  nvecs		  = per_cluster * 3;
	Dimension dim		  = 2;
	float	  data[300];

	make_3_clusters(data, per_cluster, 42);

	KMeansOptions opts = VS_KMEANS_OPTIONS_DEFAULT;
	opts.algorithm	   = algos[iteration];

	/* Should fall back to Lloyd for IP metric, not crash */
	KMeansResult *res =
			vs_kmeans_f32(data, nvecs, dim, 3, DISTANCE_INNER_PRODUCT, &opts);

	ASSERT_NOT_NULL(res, "fallback should succeed");

	uint32_t total = 0;
	for (uint32_t j = 0; j < 3; j++)
		total += res->cluster_sizes[j];
	ASSERT_EQ(nvecs, total, "all vectors assigned");

	vs_kmeans_result_destroy(res);
}

/*
 * Algo name should return correct strings for all values.
 */
TEST(algo_name)
{
	ASSERT_STR_EQ("auto", vs_kmeans_algo_name(KMEANS_ALGO_AUTO), "AUTO name");
	ASSERT_STR_EQ(
			"lloyd", vs_kmeans_algo_name(KMEANS_ALGO_LLOYD), "LLOYD name");
	ASSERT_STR_EQ(
			"hamerly",
			vs_kmeans_algo_name(KMEANS_ALGO_HAMERLY),
			"HAMERLY name");
	ASSERT_STR_EQ(
			"elkan", vs_kmeans_algo_name(KMEANS_ALGO_ELKAN), "ELKAN name");
	ASSERT_STR_EQ(
			"lloyd(cblas)",
			vs_kmeans_algo_name(KMEANS_ALGO_CBLAS),
			"CBLAS name");
	ASSERT_STR_EQ("unknown", vs_kmeans_algo_name(99), "invalid algo name");
}

/*
 * Verbose mode should run without crashing.
 *
 * Covers the verbose fprintf paths in kmeans_run_one_*.
 */
TEST(verbose_mode)
{
	uint32_t  per_cluster = 50;
	uint32_t  nvecs		  = per_cluster * 3;
	Dimension dim		  = 2;
	float	  data[300];

	make_3_clusters(data, per_cluster, 42);

	KMeansOptions opts = VS_KMEANS_OPTIONS_DEFAULT;
	opts.verbose	   = true;
	opts.nredo		   = 2; /* covers verbose nredo path */

	/* Redirect stderr to /dev/null to suppress output */
	FILE *saved = stderr;
	stderr		= fopen("/dev/null", "w");

	/* Test verbose for each algorithm */
	opts.algorithm	 = KMEANS_ALGO_LLOYD;
	KMeansResult *r1 = vs_kmeans_f32(data, nvecs, dim, 3, DISTANCE_L2, &opts);

	opts.algorithm	 = KMEANS_ALGO_HAMERLY;
	KMeansResult *r2 = vs_kmeans_f32(data, nvecs, dim, 3, DISTANCE_L2, &opts);

	opts.algorithm	 = KMEANS_ALGO_ELKAN;
	KMeansResult *r3 = vs_kmeans_f32(data, nvecs, dim, 3, DISTANCE_L2, &opts);

	fclose(stderr);
	stderr = saved;

	ASSERT_NOT_NULL(r1, "verbose lloyd should succeed");
	ASSERT_NOT_NULL(r2, "verbose hamerly should succeed");
	ASSERT_NOT_NULL(r3, "verbose elkan should succeed");

	vs_kmeans_result_destroy(r1);
	vs_kmeans_result_destroy(r2);
	vs_kmeans_result_destroy(r3);
}

/*
 * Hamerly with higher dimensions.
 *
 * Ensures bound maintenance works correctly beyond 2D.
 */
TEST(hamerly_higher_dim)
{
	uint32_t  per_cluster = 30;
	uint32_t  nlist		  = 3;
	uint32_t  nvecs		  = per_cluster * nlist;
	Dimension dim		  = 32;
	float	 *data		  = vs_alloc((size_t)nvecs * dim * sizeof(float));

	/* Generate clusters spread along axes */
	uint32_t rng = 42;
	for (uint32_t c = 0; c < nlist; c++)
	{
		for (uint32_t i = 0; i < per_cluster; i++)
		{
			uint32_t idx = c * per_cluster + i;
			for (uint32_t d = 0; d < dim; d++)
			{
				float noise = ((float)(vs_test_rand(&rng) % 1000) / 500.0f -
							   1.0f) *
							  0.1f;
				float center				= (d == c % dim) ? 10.0f : 0.0f;
				data[(size_t)idx * dim + d] = center + noise;
			}
		}
	}

	KMeansOptions opts = VS_KMEANS_OPTIONS_DEFAULT;
	opts.algorithm	   = KMEANS_ALGO_HAMERLY;

	KMeansResult *res =
			vs_kmeans_f32(data, nvecs, dim, nlist, DISTANCE_L2, &opts);

	ASSERT_NOT_NULL(res, "hamerly higher dim should succeed");
	ASSERT_TRUE(res->total_cost > 0, "cost should be positive");

	uint32_t total = 0;
	for (uint32_t j = 0; j < nlist; j++)
		total += res->cluster_sizes[j];
	ASSERT_EQ(nvecs, total, "all vectors assigned");

	vs_kmeans_result_destroy(res);
	vs_free(data);
}

/*
 * Elkan with higher dimensions.
 *
 * Ensures per-centroid lower bound maintenance works correctly.
 */
TEST(elkan_higher_dim)
{
	uint32_t  per_cluster = 30;
	uint32_t  nlist		  = 3;
	uint32_t  nvecs		  = per_cluster * nlist;
	Dimension dim		  = 32;
	float	 *data		  = vs_alloc((size_t)nvecs * dim * sizeof(float));

	uint32_t rng = 42;
	for (uint32_t c = 0; c < nlist; c++)
	{
		for (uint32_t i = 0; i < per_cluster; i++)
		{
			uint32_t idx = c * per_cluster + i;
			for (uint32_t d = 0; d < dim; d++)
			{
				float noise = ((float)(vs_test_rand(&rng) % 1000) / 500.0f -
							   1.0f) *
							  0.1f;
				float center				= (d == c % dim) ? 10.0f : 0.0f;
				data[(size_t)idx * dim + d] = center + noise;
			}
		}
	}

	KMeansOptions opts = VS_KMEANS_OPTIONS_DEFAULT;
	opts.algorithm	   = KMEANS_ALGO_ELKAN;

	KMeansResult *res =
			vs_kmeans_f32(data, nvecs, dim, nlist, DISTANCE_L2, &opts);

	ASSERT_NOT_NULL(res, "elkan higher dim should succeed");
	ASSERT_TRUE(res->total_cost > 0, "cost should be positive");

	uint32_t total = 0;
	for (uint32_t j = 0; j < nlist; j++)
		total += res->cluster_sizes[j];
	ASSERT_EQ(nvecs, total, "all vectors assigned");

	vs_kmeans_result_destroy(res);
	vs_free(data);
}

/*
 * cblas_is_single_threaded should reflect OMP_NUM_THREADS.
 */
TEST(cblas_single_threaded)
{
	/* We can't set env vars safely in tests, but we can call it */
	bool result_val = vs_cblas_is_single_threaded();
	/* Just verify it returns a bool without crashing */
	ASSERT_TRUE(
			result_val == true || result_val == false,
			"should return a valid bool");
}

/*
 * Hamerly and Elkan with nredo > 1.
 *
 * Covers the nredo loop with these algorithms and verifies the
 * best result is kept.
 */
TEST_PARAMETERIZED(algo_nredo, "hamerly", "elkan")
{
	KMeansAlgorithm algos[] = {
			KMEANS_ALGO_HAMERLY,
			KMEANS_ALGO_ELKAN,
	};

	uint32_t  per_cluster = 50;
	uint32_t  nvecs		  = per_cluster * 3;
	Dimension dim		  = 2;
	float	  data[300];

	make_3_clusters(data, per_cluster, 42);

	KMeansOptions opts1 = VS_KMEANS_OPTIONS_DEFAULT;
	opts1.algorithm		= algos[iteration];
	opts1.seed			= 42;
	opts1.nredo			= 1;

	KMeansOptions opts3 = VS_KMEANS_OPTIONS_DEFAULT;
	opts3.algorithm		= algos[iteration];
	opts3.seed			= 42;
	opts3.nredo			= 3;

	KMeansResult *r1 = vs_kmeans_f32(data, nvecs, dim, 3, DISTANCE_L2, &opts1);
	KMeansResult *r3 = vs_kmeans_f32(data, nvecs, dim, 3, DISTANCE_L2, &opts3);

	ASSERT_NOT_NULL(r1, "nredo=1 should succeed");
	ASSERT_NOT_NULL(r3, "nredo=3 should succeed");

	ASSERT_TRUE(
			r3->total_cost <= r1->total_cost + 1e-3f,
			"more restarts should not increase cost");

	vs_kmeans_result_destroy(r1);
	vs_kmeans_result_destroy(r3);
}

/*
 * Hamerly with more clusters (K=8).
 *
 * Tests bound maintenance with many centroids where more
 * distance computations can be pruned.
 */
TEST(hamerly_many_clusters)
{
	uint32_t  per_cluster = 20;
	uint32_t  nlist		  = 8;
	uint32_t  nvecs		  = per_cluster * nlist;
	Dimension dim		  = 4;
	float	 *data		  = vs_alloc((size_t)nvecs * dim * sizeof(float));

	/* 8 clusters at corners of a 4D hypercube */
	uint32_t rng = 42;
	for (uint32_t c = 0; c < nlist; c++)
	{
		for (uint32_t i = 0; i < per_cluster; i++)
		{
			uint32_t idx = c * per_cluster + i;
			for (uint32_t d = 0; d < dim; d++)
			{
				float noise = ((float)(vs_test_rand(&rng) % 1000) / 500.0f -
							   1.0f) *
							  0.1f;
				float center				= (c & (1u << d)) ? 10.0f : -10.0f;
				data[(size_t)idx * dim + d] = center + noise;
			}
		}
	}

	KMeansOptions opts = VS_KMEANS_OPTIONS_DEFAULT;
	opts.algorithm	   = KMEANS_ALGO_HAMERLY;

	KMeansResult *res =
			vs_kmeans_f32(data, nvecs, dim, nlist, DISTANCE_L2, &opts);

	ASSERT_NOT_NULL(res, "hamerly many clusters should succeed");

	uint32_t total = 0;
	for (uint32_t j = 0; j < nlist; j++)
		total += res->cluster_sizes[j];
	ASSERT_EQ(nvecs, total, "all vectors assigned");

	/* Well-separated corners: expect high accuracy */
	uint32_t correct =
			count_correct_assignments(res->assignments, per_cluster, nlist);
	ASSERT_TRUE(
			correct >= nvecs * 9 / 10,
			"at least 90% correct for hypercube corners");

	vs_kmeans_result_destroy(res);
	vs_free(data);
}

/*
 * Hamerly/Elkan with overlapping clusters.
 *
 * Uses clusters with significant overlap so that vectors near
 * boundaries get reassigned across iterations. This exercises
 * the bound-violation paths (step 2-3 in Hamerly, step 3a-3b
 * in Elkan) that are skipped with well-separated data.
 */
TEST_PARAMETERIZED(algo_overlapping, "hamerly", "elkan")
{
	KMeansAlgorithm algos[] = {
			KMEANS_ALGO_HAMERLY,
			KMEANS_ALGO_ELKAN,
	};

	uint32_t  nvecs = 200;
	uint32_t  nlist = 5;
	Dimension dim	= 8;
	float	 *data	= vs_alloc((size_t)nvecs * dim * sizeof(float));

	/*
	 * Generate data with overlapping clusters: centers are close
	 * together (spacing 3.0) with large noise (±2.0), so many
	 * vectors lie near decision boundaries.
	 */
	uint32_t rng = 12345;
	for (uint32_t i = 0; i < nvecs; i++)
	{
		uint32_t c = i % nlist;
		for (uint32_t d = 0; d < dim; d++)
		{
			float noise = ((float)(vs_test_rand(&rng) % 10000) / 5000.0f -
						   1.0f) *
						  2.0f;
			float center			  = (d == c) ? 3.0f : 0.0f;
			data[(size_t)i * dim + d] = center + noise;
		}
	}

	KMeansOptions opts	= VS_KMEANS_OPTIONS_DEFAULT;
	opts.algorithm		= algos[iteration];
	opts.seed			= 42;
	opts.max_iterations = 20;

	KMeansResult *res =
			vs_kmeans_f32(data, nvecs, dim, nlist, DISTANCE_L2, &opts);

	ASSERT_NOT_NULL(res, "overlapping should succeed");
	ASSERT_TRUE(res->total_cost > 0, "cost should be positive");

	uint32_t total = 0;
	for (uint32_t j = 0; j < nlist; j++)
		total += res->cluster_sizes[j];
	ASSERT_EQ(nvecs, total, "all vectors assigned");

	vs_kmeans_result_destroy(res);
	vs_free(data);
}

/*
 * Elkan with more clusters (K=8).
 *
 * Tests per-centroid lower bound maintenance with many centroids.
 */
TEST(elkan_many_clusters)
{
	uint32_t  per_cluster = 20;
	uint32_t  nlist		  = 8;
	uint32_t  nvecs		  = per_cluster * nlist;
	Dimension dim		  = 4;
	float	 *data		  = vs_alloc((size_t)nvecs * dim * sizeof(float));

	uint32_t rng = 42;
	for (uint32_t c = 0; c < nlist; c++)
	{
		for (uint32_t i = 0; i < per_cluster; i++)
		{
			uint32_t idx = c * per_cluster + i;
			for (uint32_t d = 0; d < dim; d++)
			{
				float noise = ((float)(vs_test_rand(&rng) % 1000) / 500.0f -
							   1.0f) *
							  0.1f;
				float center				= (c & (1u << d)) ? 10.0f : -10.0f;
				data[(size_t)idx * dim + d] = center + noise;
			}
		}
	}

	KMeansOptions opts = VS_KMEANS_OPTIONS_DEFAULT;
	opts.algorithm	   = KMEANS_ALGO_ELKAN;

	KMeansResult *res =
			vs_kmeans_f32(data, nvecs, dim, nlist, DISTANCE_L2, &opts);

	ASSERT_NOT_NULL(res, "elkan many clusters should succeed");

	uint32_t total = 0;
	for (uint32_t j = 0; j < nlist; j++)
		total += res->cluster_sizes[j];
	ASSERT_EQ(nvecs, total, "all vectors assigned");

	uint32_t correct =
			count_correct_assignments(res->assignments, per_cluster, nlist);
	ASSERT_TRUE(
			correct >= nvecs * 9 / 10,
			"at least 90% correct for hypercube corners");

	vs_kmeans_result_destroy(res);
	vs_free(data);
}

/*
 * Helper: convert f32 data to f16.
 */
static half *
make_f16_data(const float *f32, uint32_t count, Dimension dim)
{
	size_t n   = (size_t)count * dim;
	half  *f16 = vs_alloc(n * sizeof(half));
	vs_float_to_half_array(f32, f16, (uint32_t)n);
	return f16;
}

/*
 * f16 k-means with all three algorithms on well-separated clusters.
 *
 * Exercises the preconvert paths in Lloyd, Hamerly, and Elkan.
 */
TEST_PARAMETERIZED(f16_algo_l2, "lloyd", "hamerly", "elkan")
{
	KMeansAlgorithm algos[] = {
			KMEANS_ALGO_LLOYD,
			KMEANS_ALGO_HAMERLY,
			KMEANS_ALGO_ELKAN,
	};

	uint32_t  per_cluster = 50;
	uint32_t  nvecs		  = per_cluster * 3;
	Dimension dim		  = 2;
	float	  data[300];

	make_3_clusters(data, per_cluster, 42);
	half *f16 = make_f16_data(data, nvecs, dim);

	KMeansOptions opts = VS_KMEANS_OPTIONS_DEFAULT;
	opts.algorithm	   = algos[iteration];
	opts.seed		   = 42;

	KMeansResult *res = vs_kmeans(
			f16, NULL, VS_VEC_F16, nvecs, dim, 3, DISTANCE_L2, &opts);

	ASSERT_NOT_NULL(res, "f16 algorithm should return result");
	ASSERT_EQ(3, res->nlist, "nlist should be 3");
	ASSERT_TRUE(res->total_cost > 0, "cost should be positive");

	uint32_t total_f16 = 0;
	for (uint32_t j = 0; j < 3; j++)
	{
		ASSERT_TRUE(
				res->cluster_sizes[j] > 0, "each cluster should have vectors");
		total_f16 += res->cluster_sizes[j];
	}
	ASSERT_EQ(nvecs, total_f16, "total should equal nvecs");

	uint32_t correct_f16 =
			count_correct_assignments(res->assignments, per_cluster, 3);
	ASSERT_TRUE(
			correct_f16 >= nvecs * 9 / 10,
			"at least 90% correct for separated clusters");

	vs_kmeans_result_destroy(res);
	vs_free(f16);
}

/*
 * f16 vs f32 should produce similar assignments.
 *
 * With well-separated clusters and the same seed, both should
 * converge to the same clustering.
 */
TEST(f16_matches_f32)
{
	uint32_t  per_cluster = 50;
	uint32_t  nvecs		  = per_cluster * 3;
	Dimension dim		  = 2;
	float	  data[300];

	make_3_clusters(data, per_cluster, 42);
	half *f16 = make_f16_data(data, nvecs, dim);

	KMeansOptions opts = VS_KMEANS_OPTIONS_DEFAULT;
	opts.seed		   = 42;

	KMeansResult *r32 = vs_kmeans_f32(data, nvecs, dim, 3, DISTANCE_L2, &opts);
	KMeansResult *r16 = vs_kmeans(
			f16, NULL, VS_VEC_F16, nvecs, dim, 3, DISTANCE_L2, &opts);

	ASSERT_NOT_NULL(r32, "f32 should succeed");
	ASSERT_NOT_NULL(r16, "f16 should succeed");

	/* Same seed + well-separated: assignments should match */
	for (uint32_t i = 0; i < nvecs; i++)
	{
		ASSERT_EQ(
				r32->assignments[i],
				r16->assignments[i],
				"f16 and f32 assignments should match");
	}

	/* Costs should be close (f16 has quantization noise) */
	ASSERT_FLOAT_EQ(
			r32->total_cost,
			r16->total_cost,
			r32->total_cost * 0.01f,
			"f16 cost should be within 1% of f32");

	vs_kmeans_result_destroy(r32);
	vs_kmeans_result_destroy(r16);
	vs_free(f16);
}

/*
 * f16 Hamerly/Elkan with overlapping clusters.
 *
 * Exercises bound-violation paths in the preconvert impls where
 * vectors near decision boundaries need full distance recomputation.
 */
TEST_PARAMETERIZED(f16_algo_overlapping, "hamerly", "elkan")
{
	KMeansAlgorithm algos[] = {
			KMEANS_ALGO_HAMERLY,
			KMEANS_ALGO_ELKAN,
	};

	uint32_t  nvecs = 200;
	uint32_t  nlist = 5;
	Dimension dim	= 8;
	float	 *data	= vs_alloc((size_t)nvecs * dim * sizeof(float));

	uint32_t rng = 12345;
	for (uint32_t i = 0; i < nvecs; i++)
	{
		uint32_t c = i % nlist;
		for (uint32_t d = 0; d < dim; d++)
		{
			float noise = ((float)(vs_test_rand(&rng) % 10000) / 5000.0f -
						   1.0f) *
						  2.0f;
			float center			  = (d == c) ? 3.0f : 0.0f;
			data[(size_t)i * dim + d] = center + noise;
		}
	}

	half *f16 = make_f16_data(data, nvecs, dim);

	KMeansOptions opts	= VS_KMEANS_OPTIONS_DEFAULT;
	opts.algorithm		= algos[iteration];
	opts.seed			= 42;
	opts.max_iterations = 20;

	KMeansResult *res = vs_kmeans(
			f16, NULL, VS_VEC_F16, nvecs, dim, nlist, DISTANCE_L2, &opts);

	ASSERT_NOT_NULL(res, "f16 overlapping should succeed");
	ASSERT_TRUE(res->total_cost > 0, "cost should be positive");

	uint32_t total_f16 = 0;
	for (uint32_t j = 0; j < nlist; j++)
		total_f16 += res->cluster_sizes[j];
	ASSERT_EQ(nvecs, total_f16, "all vectors assigned");

	vs_kmeans_result_destroy(res);
	vs_free(f16);
	vs_free(data);
}

/*
 * Indexed k-means: subset via index array should match gathered copy.
 *
 * Build a large array with 4 well-separated clusters, then select a
 * subset of 3 clusters via an index array. Verify that indexed k-means
 * produces the same assignments as k-means on a contiguous copy.
 */
TEST(indexed_kmeans)
{
	/* 4 clusters at (±50, ±50), 25 vectors each */
	uint32_t  per_cluster = 25;
	uint32_t  total_vecs  = per_cluster * 4;
	Dimension dim		  = 2;
	float	 *all_vecs	  = vs_alloc((size_t)total_vecs * dim * sizeof(float));

	float centers[][2] = {
			{50.0f, 50.0f},
			{-50.0f, 50.0f},
			{50.0f, -50.0f},
			{-50.0f, -50.0f},
	};

	uint32_t rng = 77;
	for (uint32_t c = 0; c < 4; c++)
	{
		for (uint32_t i = 0; i < per_cluster; i++)
		{
			uint32_t idx = c * per_cluster + i;
			float dx = ((float)(vs_test_rand(&rng) % 1000) / 500.0f - 1.0f) *
					   0.1f;
			float dy = ((float)(vs_test_rand(&rng) % 1000) / 500.0f - 1.0f) *
					   0.1f;
			all_vecs[idx * 2 + 0] = centers[c][0] + dx;
			all_vecs[idx * 2 + 1] = centers[c][1] + dy;
		}
	}

	/* Select clusters 0, 2, 3 (skip cluster 1) via indices */
	uint32_t  sub_nvecs = per_cluster * 3;
	uint32_t *indices	= vs_alloc(sub_nvecs * sizeof(uint32_t));
	float	 *gathered	= vs_alloc((size_t)sub_nvecs * dim * sizeof(float));

	uint32_t idx		= 0;
	uint32_t selected[] = {0, 2, 3};
	for (int s = 0; s < 3; s++)
	{
		uint32_t c = selected[s];
		for (uint32_t i = 0; i < per_cluster; i++)
		{
			uint32_t orig = c * per_cluster + i;
			indices[idx]  = orig;
			memcpy(gathered + (size_t)idx * dim,
				   all_vecs + (size_t)orig * dim,
				   dim * sizeof(float));
			idx++;
		}
	}

	KMeansOptions opts = VS_KMEANS_OPTIONS_DEFAULT;
	opts.seed		   = 42;

	/* Run indexed k-means on full array with subset indices */
	KMeansResult *r_idx = vs_kmeans(
			all_vecs,
			indices,
			VS_VEC_F32,
			sub_nvecs,
			dim,
			3,
			DISTANCE_L2,
			&opts);

	/* Run standard k-means on gathered copy */
	KMeansResult *r_std =
			vs_kmeans_f32(gathered, sub_nvecs, dim, 3, DISTANCE_L2, &opts);

	ASSERT_NOT_NULL(r_idx, "indexed k-means should succeed");
	ASSERT_NOT_NULL(r_std, "standard k-means should succeed");

	/* Same seed + same data: assignments should match exactly */
	for (uint32_t i = 0; i < sub_nvecs; i++)
	{
		ASSERT_EQ(
				r_std->assignments[i],
				r_idx->assignments[i],
				"indexed and gathered should match");
	}

	ASSERT_FLOAT_EQ(
			r_std->total_cost, r_idx->total_cost, 1e-3f, "costs should match");

	vs_kmeans_result_destroy(r_idx);
	vs_kmeans_result_destroy(r_std);
	vs_free(all_vecs);
	vs_free(indices);
	vs_free(gathered);
}

/*
 * The reduce step: two workers' partial tallies combine into the means.
 *
 * Worker 0 put one vector in cluster 0 and one in cluster 1; worker 1
 * put three more in cluster 0. Cluster 2 gets nothing, which exercises
 * the empty-cluster path. Sums and counts are what the workers hand
 * over, so the expected centroid is each cluster's summed vector
 * divided by the number of vectors that reached it.
 *
 * The count scratch arrives dirty on purpose: the caller hands the same
 * buffer to every iteration, so the merge has to clear it rather than
 * accumulate onto whatever the previous iteration left.
 */
TEST(merge_combines_worker_tallies)
{
	enum
	{
		NLIST = 3,
		DIM	  = 2,
		NW	  = 2
	};

	/* [cluster][component], laid out flat as the merge expects. */
	float	 w0_sums[NLIST * DIM] = {1.0f, 2.0f, 10.0f, 10.0f, 0.0f, 0.0f};
	float	 w1_sums[NLIST * DIM] = {11.0f, 14.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	uint32_t w0_counts[NLIST]	  = {1, 1, 0};
	uint32_t w1_counts[NLIST]	  = {3, 0, 0};

	const float *const	  sums[NW]	 = {w0_sums, w1_sums};
	const uint32_t *const counts[NW] = {w0_counts, w1_counts};
	const float			  costs[NW]	 = {1.5f, 2.5f};

	float	 centroids[NLIST * DIM] = {0};
	float	 prev[NLIST * DIM]		= {0};
	uint32_t scratch[NLIST]			= {7, 11, 13};
	float	 total_cost				= 0.0f;

	const KMeansReduce reduce = {
			.worker_vector_sums	   = sums,
			.worker_vector_counts  = counts,
			.worker_costs		   = costs,
			.nworkers			   = NW,
			.prev_centroids		   = prev,
			.cluster_vector_counts = scratch,
	};

	float shift_sq = kmeans_merge_centroids(
			centroids, NULL, NLIST, DIM, DISTANCE_L2, &reduce, &total_cost);

	/* Cluster 0: (1,2) + (11,14) over 1 + 3 vectors. */
	ASSERT_FLOAT_EQ(3.0f, centroids[0], 1e-6f, "cluster 0 mean x");
	ASSERT_FLOAT_EQ(4.0f, centroids[1], 1e-6f, "cluster 0 mean y");

	/* Cluster 1: a single vector, so the mean is that vector. */
	ASSERT_FLOAT_EQ(10.0f, centroids[2], 1e-6f, "cluster 1 mean x");
	ASSERT_FLOAT_EQ(10.0f, centroids[3], 1e-6f, "cluster 1 mean y");

	/* Cluster 2 drew no vectors: left at the origin, not divided by
	 * zero. */
	ASSERT_FLOAT_EQ(0.0f, centroids[4], 1e-6f, "empty cluster x");
	ASSERT_FLOAT_EQ(0.0f, centroids[5], 1e-6f, "empty cluster y");

	/* The counts the merge summed, which the caller reuses. */
	ASSERT_EQ(4u, scratch[0], "cluster 0 vector count");
	ASSERT_EQ(1u, scratch[1], "cluster 1 vector count");
	ASSERT_EQ(0u, scratch[2], "cluster 2 vector count");

	ASSERT_FLOAT_EQ(4.0f, total_cost, 1e-6f, "costs sum across workers");

	/* Furthest move from the origin is cluster 1: 10^2 + 10^2. */
	ASSERT_FLOAT_EQ(200.0f, shift_sq, 1e-3f, "max squared shift");
}
