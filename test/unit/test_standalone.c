/*
 * test_standalone.c - Unit tests for standalone index, query, and API
 *
 * Tests the full pipeline: build index → create query context → query
 * → verify results. Exercises api.c, index.c, and query.c.
 */

#include <math.h>
#include <string.h>

#include "core/memory.h"
#include "mkt_test.h"
#include "standalone/api.h"
#include "standalone/index.h"
#include "standalone/query.h"

TEST_GROUP(Standalone);
TEST_MEMCTX_FIXTURE();

/* ----------------------------------------------------------------
 * Helpers
 * ---------------------------------------------------------------- */

/* Generate random float32 vectors (deterministic via seed) */
static float *
make_vectors(uint32_t nvecs, uint32_t dim, uint32_t seed)
{
	srand(seed);
	float *data = mkt_alloc((size_t)nvecs * dim * sizeof(float));
	for (uint32_t i = 0; i < nvecs * dim; i++)
		data[i] = (float)(rand() % 10000 - 5000) / 5000.0f;
	return data;
}

/* Convenience: build index from array */
static MktIndex *
build_from_array(
		const float			 *vecs,
		uint32_t			  nvecs,
		uint32_t			  dim,
		const MktIndexConfig *config)
{
	MktArraySource src;
	mkt_array_source_init(&src, vecs, nvecs, dim);
	return mkt_index_build(&src.base, config);
}

/* Brute-force nearest neighbor for ground truth */
static void
brute_force_knn(
		const float *vecs,
		uint32_t	 nvecs,
		uint32_t	 dim,
		const float *query,
		uint32_t	 k,
		uint32_t	*ids)
{
	/* Simple O(n*k) selection */
	float *dists = mkt_alloc(nvecs * sizeof(float));
	for (uint32_t i = 0; i < nvecs; i++)
	{
		float d = 0;
		for (uint32_t j = 0; j < dim; j++)
		{
			float diff = vecs[(size_t)i * dim + j] - query[j];
			d += diff * diff;
		}
		dists[i] = d;
	}

	for (uint32_t i = 0; i < k; i++)
	{
		uint32_t best = 0;
		for (uint32_t j = 1; j < nvecs; j++)
			if (dists[j] < dists[best])
				best = j;
		ids[i]		= best;
		dists[best] = INFINITY;
	}
	mkt_free(dists);
}

/* ----------------------------------------------------------------
 * MktIndex tests
 * ---------------------------------------------------------------- */

TEST(index_build_basic)
{
	float *vecs = make_vectors(1000, 32, 42);

	MktIndexConfig config = {
			.nlist		   = 10,
			.metric		   = DISTANCE_L2,
			.centroid_fmt  = MKT_CENTROID_FMT_RABITQ,
			.encode_rabitq = true,
	};

	MktIndex *idx = build_from_array(vecs, 1000, 32, &config);
	ASSERT_NOT_NULL(idx, "build should succeed");
	ASSERT_EQ(idx->base.dim, 32, "dim should match");
	ASSERT_EQ(idx->nvecs, 1000, "nvecs should match");
	ASSERT_TRUE(idx->nlist > 0, "should have clusters");
	ASSERT_TRUE(idx->base.nlevels > 0, "should have levels");

	mkt_index_destroy(idx);
}

TEST(index_build_cosine)
{
	float *vecs = make_vectors(500, 16, 99);

	MktIndexConfig config = {
			.nlist		   = 5,
			.metric		   = DISTANCE_COSINE,
			.centroid_fmt  = MKT_CENTROID_FMT_RABITQ,
			.encode_rabitq = true,
	};

	MktIndex *idx = build_from_array(vecs, 500, 16, &config);
	ASSERT_NOT_NULL(idx, "cosine build should succeed");
	ASSERT_EQ(idx->base.metric, DISTANCE_COSINE, "metric should be cosine");

	mkt_index_destroy(idx);
}

TEST(index_build_null_fails)
{
	MktIndexConfig config = {.nlist = 10, .metric = DISTANCE_L2};

	ASSERT_NULL(mkt_index_build(NULL, &config), "null src should fail");

	float		   dummy = 1.0f;
	MktArraySource src;

	mkt_array_source_init(&src, &dummy, 0, 32);
	ASSERT_NULL(mkt_index_build(&src.base, &config), "zero nvecs should fail");

	mkt_array_source_init(&src, &dummy, 100, 0);
	ASSERT_NULL(mkt_index_build(&src.base, &config), "zero dim should fail");

	mkt_array_source_init(&src, &dummy, 100, 32);
	ASSERT_NULL(mkt_index_build(&src.base, NULL), "null config should fail");
}

TEST(index_build_float32_centroids)
{
	float *vecs = make_vectors(500, 16, 77);

	MktIndexConfig config = {
			.nlist		  = 5,
			.metric		  = DISTANCE_L2,
			.centroid_fmt = MKT_CENTROID_FMT_FLOAT,
	};

	MktIndex *idx = build_from_array(vecs, 500, 16, &config);
	ASSERT_NOT_NULL(idx, "float32 centroid build should succeed");

	mkt_index_destroy(idx);
}

/* ----------------------------------------------------------------
 * MktQueryCtx tests
 * ---------------------------------------------------------------- */

TEST(query_ctx_create_destroy)
{
	float *vecs = make_vectors(500, 16, 42);

	MktIndexConfig config = {
			.nlist		   = 5,
			.metric		   = DISTANCE_L2,
			.centroid_fmt  = MKT_CENTROID_FMT_RABITQ,
			.encode_rabitq = true,
	};

	MktIndex	*idx  = build_from_array(vecs, 500, 16, &config);
	MktQueryCtx *qctx = mkt_query_ctx_create(idx, 10, 5);
	ASSERT_NOT_NULL(qctx, "query ctx should succeed");

	mkt_query_ctx_destroy(qctx);
	mkt_index_destroy(idx);
}

TEST(query_ctx_null_fails)
{
	ASSERT_NULL(mkt_query_ctx_create(NULL, 10, 5), "null idx should fail");
}

TEST(query_exec_returns_results)
{
	uint32_t dim = 32, nvecs = 1000, k = 5, nprobe = 10;
	float	*vecs = make_vectors(nvecs, dim, 42);

	MktIndexConfig config = {
			.nlist		   = 10,
			.metric		   = DISTANCE_L2,
			.centroid_fmt  = MKT_CENTROID_FMT_RABITQ,
			.encode_rabitq = true,
	};

	MktIndex	*idx  = build_from_array(vecs, nvecs, dim, &config);
	MktQueryCtx *qctx = mkt_query_ctx_create(idx, k, nprobe);

	/* Query with the first vector — scan all clusters */
	uint32_t result_ids[5];
	uint32_t count = mkt_query_exec(
			qctx,
			vecs,
			k,
			nprobe,
			MKT_DISTANCE_MODE_ASYMMETRIC,
			true,
			result_ids);

	ASSERT_TRUE(count > 0, "should return results");
	ASSERT_TRUE(count <= k, "should not exceed k");

	/* All result IDs should be valid vector indices */
	for (uint32_t i = 0; i < count; i++)
		ASSERT_TRUE(result_ids[i] < nvecs, "result ID in range");

	mkt_query_ctx_destroy(qctx);
	mkt_index_destroy(idx);
}

TEST(query_exec_recall)
{
	uint32_t dim = 32, nvecs = 2000, k = 10;
	float	*vecs = make_vectors(nvecs, dim, 42);

	MktIndexConfig config = {
			.nlist		   = 20,
			.metric		   = DISTANCE_L2,
			.centroid_fmt  = MKT_CENTROID_FMT_RABITQ,
			.encode_rabitq = true,
	};

	MktIndex	*idx  = build_from_array(vecs, nvecs, dim, &config);
	MktQueryCtx *qctx = mkt_query_ctx_create(idx, k, 20);

	/* Run 10 queries and check average recall */
	uint32_t total_hits = 0;
	uint32_t nqueries	= 10;

	for (uint32_t q = 0; q < nqueries; q++)
	{
		const float *query = vecs + (size_t)(q * 100) * dim;

		uint32_t result_ids[10];
		uint32_t count = mkt_query_exec(
				qctx,
				query,
				k,
				10,
				MKT_DISTANCE_MODE_ASYMMETRIC,
				true,
				result_ids);

		/* Compute ground truth */
		uint32_t gt_ids[10];
		brute_force_knn(vecs, nvecs, dim, query, k, gt_ids);

		/* Count hits */
		for (uint32_t i = 0; i < count; i++)
			for (uint32_t g = 0; g < k; g++)
				if (result_ids[i] == gt_ids[g])
					total_hits++;
	}

	double recall = (double)total_hits / (nqueries * k);
	/* With nprobe=10 on 20 clusters, recall should be reasonable */
	ASSERT_TRUE(recall > 0.2, "recall should be > 0.2");

	mkt_query_ctx_destroy(qctx);
	mkt_index_destroy(idx);
}

TEST(query_exec_null_fails)
{
	uint32_t ids[10];
	float	 query[32] = {0};
	ASSERT_EQ(
			mkt_query_exec(
					NULL,
					query,
					10,
					5,
					MKT_DISTANCE_MODE_ASYMMETRIC,
					true,
					ids),
			0,
			"null ctx should return 0");
}

TEST(query_exec_brute_force_path)
{
	uint32_t dim = 16, nvecs = 200, k = 5;
	float	*vecs = make_vectors(nvecs, dim, 55);

	/* No RaBitQ encoding — exercises brute-force L2 path */
	MktIndexConfig config = {
			.nlist		  = 5,
			.metric		  = DISTANCE_L2,
			.centroid_fmt = MKT_CENTROID_FMT_FLOAT,
	};

	MktIndex	*idx  = build_from_array(vecs, nvecs, dim, &config);
	MktQueryCtx *qctx = mkt_query_ctx_create(idx, k, 5);

	uint32_t result_ids[5];
	uint32_t count = mkt_query_exec(
			qctx, vecs, k, 5, MKT_DISTANCE_MODE_ASYMMETRIC, true, result_ids);

	ASSERT_EQ(count, k, "should return k results");
	for (uint32_t i = 0; i < count; i++)
		ASSERT_TRUE(result_ids[i] < nvecs, "result ID in range");

	mkt_query_ctx_destroy(qctx);
	mkt_index_destroy(idx);
}

/* ----------------------------------------------------------------
 * Bindings API tests
 * ---------------------------------------------------------------- */

TEST(bindings_create_destroy)
{
	float *vecs = make_vectors(500, 16, 42);

	MktBuildInfo info;
	MktHandle	*handle = mkt_handle_create_from_array(
			  vecs,
			  500,
			  16,
			  5,
			  0,
			  "euclidean",
			  "rabitq",
			  NULL,
			  0,
			  0,
			  0.0,
			  0.0,
			  &info);

	ASSERT_NOT_NULL(handle, "create should succeed");
	ASSERT_TRUE(info.nlist > 0, "should have clusters");
	ASSERT_EQ(info.nvecs, 500, "nvecs should match");
	ASSERT_TRUE(info.min_cluster > 0, "min cluster > 0");
	ASSERT_TRUE(info.max_cluster >= info.min_cluster, "max >= min");

	mkt_handle_destroy(handle);
}

TEST(bindings_query)
{
	uint32_t dim = 32, nvecs = 1000;
	float	*vecs = make_vectors(nvecs, dim, 42);

	MktHandle *handle = mkt_handle_create_from_array(
			vecs,
			nvecs,
			dim,
			10,
			0,
			"euclidean",
			"rabitq",
			NULL,
			0,
			0,
			0.0,
			0.0,
			NULL);
	ASSERT_NOT_NULL(handle, "create should succeed");

	uint32_t result_ids[10];
	uint32_t count = mkt_handle_query(
			handle, vecs, 10, 5, "asymmetric", true, result_ids);

	ASSERT_EQ(count, 10, "should return 10 results");
	ASSERT_TRUE(result_ids[0] < nvecs, "result ID in range");

	mkt_handle_destroy(handle);
}

TEST(bindings_angular_metric)
{
	uint32_t dim = 16, nvecs = 500;
	float	*vecs = make_vectors(nvecs, dim, 77);

	MktHandle *handle = mkt_handle_create_from_array(
			vecs,
			nvecs,
			dim,
			5,
			0,
			"angular",
			"rabitq",
			NULL,
			0,
			0,
			0.0,
			0.0,
			NULL);
	ASSERT_NOT_NULL(handle, "angular create should succeed");

	uint32_t result_ids[5];
	uint32_t count = mkt_handle_query(
			handle, vecs, 5, 5, "asymmetric", true, result_ids);

	ASSERT_TRUE(count > 0, "should return results");

	mkt_handle_destroy(handle);
}

TEST(bindings_null_destroy)
{
	/* Should not crash */
	mkt_handle_destroy(NULL);
	ASSERT_TRUE(true, "null destroy should not crash");
}

TEST(bindings_null_query)
{
	uint32_t ids[10];
	float	 query[32] = {0};
	ASSERT_EQ(
			mkt_handle_query(NULL, query, 10, 5, "asymmetric", true, ids),
			0,
			"null handle should return 0");
}

TEST(bindings_symmetric_mode)
{
	uint32_t dim = 32, nvecs = 500;
	float	*vecs = make_vectors(nvecs, dim, 42);

	MktHandle *handle = mkt_handle_create_from_array(
			vecs,
			nvecs,
			dim,
			5,
			0,
			"euclidean",
			"rabitq",
			NULL,
			0,
			0,
			0.0,
			0.0,
			NULL);

	uint32_t result_ids[5];
	uint32_t count = mkt_handle_query(
			handle, vecs, 5, 5, "symmetric", true, result_ids);

	ASSERT_TRUE(count > 0, "symmetric query should return results");

	mkt_handle_destroy(handle);
}

TEST(bindings_kmeans_params)
{
	float *vecs = make_vectors(500, 16, 42);

	MktBuildInfo info;
	MktHandle	*handle = mkt_handle_create_from_array(
			  vecs,
			  500,
			  16,
			  5,
			  0,
			  "euclidean",
			  "rabitq",
			  NULL,
			  2,
			  20,
			  0.0,
			  0.0,
			  &info);

	ASSERT_NOT_NULL(handle, "create with kmeans params should succeed");
	ASSERT_TRUE(info.nlist > 0, "should have clusters");

	mkt_handle_destroy(handle);
}
