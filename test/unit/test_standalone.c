/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * test_standalone.c - Unit tests for standalone index, query, and API
 *
 * Tests the full pipeline: build index → create query context → query
 * → verify results. Exercises api.c, index.c, and query.c.
 */

#include <math.h>
#include <string.h>

#include "algo/distance.h"
#include "algo/vecops.h"
#include "core/memory.h"
#include "index/posting_build.h"
#include "index/posting_convert.h"
#include "index/posting_page.h"
#include "index/posting_scan.h"
#include "index/query_scan.h"
#include "index/storage.h"
#include "quant/fastscan.h"
#include "quant/rabitq.h"
#include "standalone/api.h"
#include "standalone/index.h"
#include "standalone/query.h"
#include "vs_test.h"

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
	float *data = vs_alloc((size_t)nvecs * dim * sizeof(float));
	for (uint32_t i = 0; i < nvecs * dim; i++)
		data[i] = (float)(rand() % 10000 - 5000) / 5000.0f;
	return data;
}

/* Convenience: build index from array */
static PrismIndex *
build_from_array(
		const float			   *vecs,
		uint32_t				nvecs,
		uint32_t				dim,
		const PrismIndexConfig *config)
{
	VsArraySource src;
	vs_array_source_init(&src, vecs, nvecs, dim);
	return prism_index_build(&src.base, config, NULL);
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
	float *dists = vs_alloc(nvecs * sizeof(float));
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
	vs_free(dists);
}

/* ----------------------------------------------------------------
 * PrismIndex tests
 * ---------------------------------------------------------------- */

TEST(index_build_basic)
{
	float *vecs = make_vectors(1000, 32, 42);

	PrismIndexConfig config = {
			.nlist		   = 10,
			.metric		   = DISTANCE_L2,
			.centroid_fmt  = PRISM_CENTROID_FMT_RABITQ,
			.encode_rabitq = true,
	};

	PrismIndex *idx = build_from_array(vecs, 1000, 32, &config);
	ASSERT_NOT_NULL(idx, "build should succeed");
	ASSERT_EQ(idx->base.dim, 32, "dim should match");
	ASSERT_EQ(idx->nvecs, 1000, "nvecs should match");
	ASSERT_TRUE(idx->nlist > 0, "should have clusters");
	ASSERT_TRUE(idx->base.nlevels > 0, "should have levels");

	prism_index_destroy(idx);
}

TEST(index_build_cosine)
{
	float *vecs = make_vectors(500, 16, 99);

	PrismIndexConfig config = {
			.nlist		   = 5,
			.metric		   = DISTANCE_COSINE,
			.centroid_fmt  = PRISM_CENTROID_FMT_RABITQ,
			.encode_rabitq = true,
	};

	PrismIndex *idx = build_from_array(vecs, 500, 16, &config);
	ASSERT_NOT_NULL(idx, "cosine build should succeed");
	ASSERT_EQ(idx->base.metric, DISTANCE_COSINE, "metric should be cosine");

	prism_index_destroy(idx);
}

TEST(index_build_null_fails)
{
	PrismIndexConfig config = {.nlist = 10, .metric = DISTANCE_L2};

	ASSERT_NULL(
			prism_index_build(NULL, &config, NULL), "null src should fail");

	float		  dummy = 1.0f;
	VsArraySource src;

	vs_array_source_init(&src, &dummy, 0, 32);
	ASSERT_NULL(
			prism_index_build(&src.base, &config, NULL),
			"zero nvecs should fail");

	vs_array_source_init(&src, &dummy, 100, 0);
	ASSERT_NULL(
			prism_index_build(&src.base, &config, NULL),
			"zero dim should fail");

	vs_array_source_init(&src, &dummy, 100, 32);
	ASSERT_NULL(
			prism_index_build(&src.base, NULL, NULL),
			"null config should fail");
}

TEST(index_build_float32_centroids)
{
	float *vecs = make_vectors(500, 16, 77);

	PrismIndexConfig config = {
			.nlist		  = 5,
			.metric		  = DISTANCE_L2,
			.centroid_fmt = PRISM_CENTROID_FMT_FLOAT,
	};

	PrismIndex *idx = build_from_array(vecs, 500, 16, &config);
	ASSERT_NOT_NULL(idx, "float32 centroid build should succeed");

	prism_index_destroy(idx);
}

/* ----------------------------------------------------------------
 * Auto nprobe
 * ---------------------------------------------------------------- */

TEST(auto_nprobe)
{
	/* Floor: small indexes probe at least 10 lists, but never more
	 * than the lists that exist. */
	ASSERT_EQ(prism_auto_nprobe(4), 4, "clamped to nlist");
	ASSERT_EQ(prism_auto_nprobe(10), 10, "floor at 10");
	ASSERT_EQ(prism_auto_nprobe(100), 10, "floor still binding at 100");

	/* Curve: ~0.5 * sqrt(nlist). */
	ASSERT_EQ(prism_auto_nprobe(40000), 100, "10M-scale auto nlist");
	ASSERT_EQ(prism_auto_nprobe(240000), 245, "50M-scale auto nlist");
	ASSERT_EQ(prism_auto_nprobe(400000), 317, "100M-scale auto nlist");

	/* Cap: past the measured range explicit settings take over. */
	ASSERT_EQ(prism_auto_nprobe(2000000000), 2048, "capped at 2048");
}

/* ----------------------------------------------------------------
 * PrismQueryCtx tests
 * ---------------------------------------------------------------- */

TEST(query_ctx_create_destroy)
{
	float *vecs = make_vectors(500, 16, 42);

	PrismIndexConfig config = {
			.nlist		   = 5,
			.metric		   = DISTANCE_L2,
			.centroid_fmt  = PRISM_CENTROID_FMT_RABITQ,
			.encode_rabitq = true,
	};

	PrismIndex	  *idx	= build_from_array(vecs, 500, 16, &config);
	PrismQueryCtx *qctx = prism_query_ctx_create(idx, 10, 5);
	ASSERT_NOT_NULL(qctx, "query ctx should succeed");

	prism_query_ctx_destroy(qctx);
	prism_index_destroy(idx);
}

TEST(query_ctx_null_fails)
{
	ASSERT_NULL(prism_query_ctx_create(NULL, 10, 5), "null idx should fail");
}

TEST(query_exec_returns_results)
{
	uint32_t dim = 32, nvecs = 1000, k = 5, nprobe = 10;
	float	*vecs = make_vectors(nvecs, dim, 42);

	PrismIndexConfig config = {
			.nlist		   = 10,
			.metric		   = DISTANCE_L2,
			.centroid_fmt  = PRISM_CENTROID_FMT_RABITQ,
			.encode_rabitq = true,
	};

	PrismIndex	  *idx	= build_from_array(vecs, nvecs, dim, &config);
	PrismQueryCtx *qctx = prism_query_ctx_create(idx, k, nprobe);

	/* Query with the first vector — scan all clusters */
	uint32_t result_ids[5];
	uint32_t count = prism_query_exec(
			qctx,
			vecs,
			k,
			nprobe,
			VS_DISTANCE_MODE_ASYMMETRIC,
			true,
			result_ids);

	ASSERT_TRUE(count > 0, "should return results");
	ASSERT_TRUE(count <= k, "should not exceed k");

	/* All result IDs should be valid vector indices */
	for (uint32_t i = 0; i < count; i++)
		ASSERT_TRUE(result_ids[i] < nvecs, "result ID in range");

	prism_query_ctx_destroy(qctx);
	prism_index_destroy(idx);
}

TEST(query_exec_recall)
{
	uint32_t dim = 32, nvecs = 2000, k = 10;
	float	*vecs = make_vectors(nvecs, dim, 42);

	PrismIndexConfig config = {
			.nlist		   = 20,
			.metric		   = DISTANCE_L2,
			.centroid_fmt  = PRISM_CENTROID_FMT_RABITQ,
			.encode_rabitq = true,
	};

	PrismIndex	  *idx	= build_from_array(vecs, nvecs, dim, &config);
	PrismQueryCtx *qctx = prism_query_ctx_create(idx, k, 20);

	/* Run 10 queries and check average recall */
	uint32_t total_hits = 0;
	uint32_t nqueries	= 10;

	for (uint32_t q = 0; q < nqueries; q++)
	{
		const float *query = vecs + (size_t)(q * 100) * dim;

		uint32_t result_ids[10];
		uint32_t count = prism_query_exec(
				qctx,
				query,
				k,
				10,
				VS_DISTANCE_MODE_ASYMMETRIC,
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

	prism_query_ctx_destroy(qctx);
	prism_index_destroy(idx);
}

/*
 * Verify each cluster head's stamped metadata against a ground-truth walk.
 * The build populates head->tail_blkno / head->live_count directly (no runtime
 * chain walk): the parallel leader builds each list from the cluster-sorted
 * entry stream and stamps the head as it writes — so the stamped tail must be
 * the real last block and the stamped live count the real entry count.
 */
/* Sums prism_posting_page_count over a chain; `state` is a uint32_t *. */
static bool
sum_page_count(PrismPostingChainPos *pos, void *state)
{
	*(uint32_t *)state += prism_posting_page_count(pos->page);
	return true;
}

typedef struct ChainTally
{
	uint32_t	count;
	BlockNumber tail; /* last page the walk saw */
} ChainTally;

static bool
tally_page(PrismPostingChainPos *pos, void *state)
{
	ChainTally *t = state;

	t->count += prism_posting_page_count(pos->page);
	t->tail = pos->blkno;
	return true;
}

static void
verify_head_meta(VsTestResult *result, PrismIndex *idx)
{
	VsStorage *st = idx->base.posting_storage;
	for (uint32_t c = 0; c < idx->nlist; c++)
	{
		BlockNumber head = idx->first_posting + c;
		if (head == InvalidBlockNumber)
			continue;

		ChainTally tally = {.tail = head};

		prism_posting_chain_walk(st, head, tally_page, &tally);

		BlockNumber tail  = tally.tail;
		uint32_t	count = tally.count;

		Page hp = vs_storage_read_page(st, head);
		ASSERT_EQ(
				count,
				prism_posting_head_live_count(hp),
				"stamped live_count must match the walked entry count");
		ASSERT_EQ(
				tail,
				prism_posting_head_tail(hp),
				"stamped tail_blkno must match the actual chain tail");
		vs_storage_release_page(st, head);
	}
}

/*
 * Paged + parallel build runs the shared do_parallel_build driver (the same
 * code path as the PG extension). Build, query, and check recall to confirm
 * the driver produces a correct, queryable index in the standalone back-end.
 */
TEST(query_exec_recall_pages_parallel)
{
	uint32_t dim = 32, nvecs = 2000, k = 10;
	float	*vecs = make_vectors(nvecs, dim, 42);

	PrismIndexConfig config = {
			.nlist		   = 20,
			.metric		   = DISTANCE_L2,
			.centroid_fmt  = PRISM_CENTROID_FMT_RABITQ,
			.encode_rabitq = true,
			.posting_fmt   = PRISM_POSTING_FMT_PAGES,
			.nworkers	   = 4, /* force the parallel driver */
	};

	PrismIndex *idx = build_from_array(vecs, nvecs, dim, &config);
	ASSERT_NOT_NULL(idx, "paged parallel build should succeed");

	verify_head_meta(result, idx);

	PrismQueryCtx *qctx = prism_query_ctx_create(idx, k, 20);

	uint32_t total_hits = 0;
	uint32_t nqueries	= 10;
	for (uint32_t q = 0; q < nqueries; q++)
	{
		const float *query = vecs + (size_t)(q * 100) * dim;

		uint32_t result_ids[10];
		uint32_t count = prism_query_exec(
				qctx,
				query,
				k,
				20,
				VS_DISTANCE_MODE_ASYMMETRIC,
				true,
				result_ids);

		uint32_t gt_ids[10];
		brute_force_knn(vecs, nvecs, dim, query, k, gt_ids);

		for (uint32_t i = 0; i < count; i++)
			for (uint32_t g = 0; g < k; g++)
				if (result_ids[i] == gt_ids[g])
					total_hits++;
	}

	double recall = (double)total_hits / (nqueries * k);
	ASSERT_TRUE(recall > 0.2, "paged parallel recall should be > 0.2");

	prism_query_ctx_destroy(qctx);
	prism_index_destroy(idx);
}

TEST(query_exec_recall_pages_parallel_depth3)
{
	/* Depth-3 streamed tree through the driver (explicit fan_out; the auto
	 * sqrt fan_out never exceeds two levels at test scale): batched subtree
	 * ring across several batches, blob replay, formula heads. */
	uint32_t dim = 32, nvecs = 2000, k = 10;
	float	*vecs = make_vectors(nvecs, dim, 42);

	PrismIndexConfig config = {
			.nlist		   = 40,
			.fan_out	   = 4,
			.metric		   = DISTANCE_L2,
			.centroid_fmt  = PRISM_CENTROID_FMT_RABITQ,
			.encode_rabitq = true,
			.posting_fmt   = PRISM_POSTING_FMT_PAGES,
			.nworkers	   = 2,
	};

	PrismIndex *idx = build_from_array(vecs, nvecs, dim, &config);
	ASSERT_NOT_NULL(idx, "paged parallel depth-3 build should succeed");

	verify_head_meta(result, idx);

	PrismQueryCtx *qctx = prism_query_ctx_create(idx, k, 40);

	uint32_t total_hits = 0;
	uint32_t nqueries	= 10;
	for (uint32_t q = 0; q < nqueries; q++)
	{
		const float *query = vecs + (size_t)(q * 100) * dim;

		uint32_t result_ids[10];
		uint32_t count = prism_query_exec(
				qctx,
				query,
				k,
				40,
				VS_DISTANCE_MODE_ASYMMETRIC,
				true,
				result_ids);

		uint32_t truth[10];
		brute_force_knn(vecs, nvecs, dim, query, k, truth);
		for (uint32_t i = 0; i < count; i++)
			for (uint32_t j = 0; j < k; j++)
				if (result_ids[i] == truth[j])
				{
					total_hits++;
					break;
				}
	}
	/* This asserts pipeline correctness, not search quality: a framing or
	 * replay bug streams a garbage subtree and collapses recall to near
	 * zero, while a healthy build lands in 0.84-0.91 (the parallel build is
	 * not bit-reproducible -- work-stealing scan order varies the k-means
	 * seeding -- so the exact value moves run to run). */
	double recall = (double)total_hits / (nqueries * k);
	ASSERT_TRUE(recall > 0.75, "depth-3 paged recall at full probe");

	prism_query_ctx_destroy(qctx);
	prism_index_destroy(idx);
}

TEST(query_exec_null_fails)
{
	uint32_t ids[10];
	float	 query[32] = {0};
	ASSERT_EQ(
			prism_query_exec(
					NULL,
					query,
					10,
					5,
					VS_DISTANCE_MODE_ASYMMETRIC,
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
	PrismIndexConfig config = {
			.nlist		  = 5,
			.metric		  = DISTANCE_L2,
			.centroid_fmt = PRISM_CENTROID_FMT_FLOAT,
	};

	PrismIndex	  *idx	= build_from_array(vecs, nvecs, dim, &config);
	PrismQueryCtx *qctx = prism_query_ctx_create(idx, k, 5);

	uint32_t result_ids[5];
	uint32_t count = prism_query_exec(
			qctx, vecs, k, 5, VS_DISTANCE_MODE_ASYMMETRIC, true, result_ids);

	ASSERT_EQ(count, k, "should return k results");
	for (uint32_t i = 0; i < count; i++)
		ASSERT_TRUE(result_ids[i] < nvecs, "result ID in range");

	prism_query_ctx_destroy(qctx);
	prism_index_destroy(idx);
}

/* ----------------------------------------------------------------
 * Bindings API tests
 * ---------------------------------------------------------------- */

TEST(bindings_create_destroy)
{
	float *vecs = make_vectors(500, 16, 42);

	PrismBuildInfo info;
	VsHandle	  *handle = vs_handle_create_from_array(
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
			 false,
			 0,
			 &info);

	ASSERT_NOT_NULL(handle, "create should succeed");
	ASSERT_TRUE(info.nlist > 0, "should have clusters");
	ASSERT_EQ(info.nvecs, 500, "nvecs should match");
	ASSERT_TRUE(info.min_cluster > 0, "min cluster > 0");
	ASSERT_TRUE(info.max_cluster >= info.min_cluster, "max >= min");

	vs_handle_destroy(handle);
}

TEST(bindings_query)
{
	uint32_t dim = 32, nvecs = 1000;
	float	*vecs = make_vectors(nvecs, dim, 42);

	VsHandle *handle = vs_handle_create_from_array(
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
			false,
			0,
			NULL);
	ASSERT_NOT_NULL(handle, "create should succeed");

	uint32_t result_ids[10];
	uint32_t count = vs_handle_query(
			handle, vecs, 10, 5, "asymmetric", true, result_ids);

	ASSERT_EQ(count, 10, "should return 10 results");
	ASSERT_TRUE(result_ids[0] < nvecs, "result ID in range");

	vs_handle_destroy(handle);
}

TEST(posting_convert_aos_to_fastscan)
{
	uint32_t dim   = 32;
	uint32_t nvecs = 200;

	vs_distance_init();
	vs_rabitq_init_simd();
	vs_fastscan_init_simd();

	/* Build an AoS index (fastscan=false) */
	float *cvecs = make_vectors(nvecs, dim, 42);

	VsArraySource array_src;
	vs_array_source_init(&array_src, cvecs, nvecs, dim);

	PrismIndexConfig cfg = {
			.nlist		   = 5,
			.metric		   = DISTANCE_L2,
			.encode_rabitq = true,
			.posting_fmt   = PRISM_POSTING_FMT_PAGES,
			.fastscan	   = 0,
	};

	PrismIndex *idx = prism_index_build(&array_src.base, &cfg, NULL);
	ASSERT_NOT_NULL(idx, "AoS index built");
	ASSERT_TRUE(
			idx->first_posting != InvalidBlockNumber, "cluster 0 has pages");

	/* Count AoS entries for cluster 0 */
	uint32_t	aos_count = 0;
	BlockNumber blkno	  = idx->first_posting;
	while (blkno != InvalidBlockNumber)
	{
		Page page = idx->posting_storage.pages + (size_t)blkno * BLCKSZ;
		PrismPostingPageOpaque *op = prism_posting_opaque(page);
		aos_count += op->entry_count;
		blkno = op->next_blkno;
	}
	ASSERT_TRUE(aos_count > 0, "cluster 0 has entries");

	/* Convert cluster 0 to fastscan */
	BlockNumber fs_head = prism_posting_convert_to_fastscan(
			&idx->posting_storage.base, idx->first_posting, dim);
	ASSERT_TRUE(fs_head != InvalidBlockNumber, "fastscan chain created");

	/* Count fastscan entries */
	uint32_t fs_count = 0;
	blkno			  = fs_head;
	while (blkno != InvalidBlockNumber)
	{
		Page page = idx->posting_storage.pages + (size_t)blkno * BLCKSZ;
		PrismPostingPageOpaque *op = prism_posting_opaque(page);
		ASSERT_TRUE(
				op->flags & PRISM_POSTING_PAGE_FASTSCAN,
				"converted page has fastscan flag");
		fs_count += op->entry_count;
		blkno = op->next_blkno;
	}
	ASSERT_EQ(fs_count, aos_count, "fastscan has same entry count as AoS");

	prism_index_destroy(idx);
}

TEST(bindings_query_fastscan)
{
	uint32_t dim = 32, nvecs = 1000;
	float	*vecs = make_vectors(nvecs, dim, 42);

	VsHandle *handle = vs_handle_create_from_array(
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
			16,
			0,
			NULL);
	ASSERT_NOT_NULL(handle, "fastscan create should succeed");

	uint32_t result_ids[10];
	uint32_t count = vs_handle_query(
			handle, vecs, 10, 5, "asymmetric", true, result_ids);

	ASSERT_EQ(count, 10, "should return 10 results");
	ASSERT_TRUE(result_ids[0] < nvecs, "result ID in range");

	vs_handle_destroy(handle);
}

TEST(bindings_angular_metric)
{
	uint32_t dim = 16, nvecs = 500;
	float	*vecs = make_vectors(nvecs, dim, 77);

	VsHandle *handle = vs_handle_create_from_array(
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
			false,
			0,
			NULL);
	ASSERT_NOT_NULL(handle, "angular create should succeed");

	uint32_t result_ids[5];
	uint32_t count = vs_handle_query(
			handle, vecs, 5, 5, "asymmetric", true, result_ids);

	ASSERT_TRUE(count > 0, "should return results");

	vs_handle_destroy(handle);
}

TEST(bindings_null_destroy)
{
	/* Should not crash */
	vs_handle_destroy(NULL);
	ASSERT_TRUE(true, "null destroy should not crash");
}

TEST(bindings_null_query)
{
	uint32_t ids[10];
	float	 query[32] = {0};
	ASSERT_EQ(
			vs_handle_query(NULL, query, 10, 5, "asymmetric", true, ids),
			0,
			"null handle should return 0");
}

TEST(bindings_symmetric_mode)
{
	uint32_t dim = 32, nvecs = 500;
	float	*vecs = make_vectors(nvecs, dim, 42);

	VsHandle *handle = vs_handle_create_from_array(
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
			false,
			0,
			NULL);

	uint32_t result_ids[5];
	uint32_t count =
			vs_handle_query(handle, vecs, 5, 5, "symmetric", true, result_ids);

	ASSERT_TRUE(count > 0, "symmetric query should return results");

	vs_handle_destroy(handle);
}

TEST(bindings_kmeans_params)
{
	float *vecs = make_vectors(500, 16, 42);

	PrismBuildInfo info;
	VsHandle	  *handle = vs_handle_create_from_array(
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
			 false,
			 0,
			 &info);

	ASSERT_NOT_NULL(handle, "create with kmeans params should succeed");
	ASSERT_TRUE(info.nlist > 0, "should have clusters");

	vs_handle_destroy(handle);
}

/* ----------------------------------------------------------------
 * Regression: parallel fastscan build must not drop trailing partials
 * ---------------------------------------------------------------- */

/* Sum the entries actually written across all posting-list pages. */
static uint32_t
count_posting_entries(PrismIndex *idx)
{
	VsStorage *st	 = idx->base.posting_storage;
	uint32_t   total = 0;

	for (uint32_t c = 0; c < idx->nlist; c++)
	{
		prism_posting_chain_walk(
				st, idx->first_posting + c, sum_page_count, &total);
	}
	return total;
}

/*
 * Regression guard for the parallel fastscan build. Workers RaBitQ-encode
 * every vector and feed it to the cluster-keyed sorter; the leader builds each
 * list's fastscan pages from the merged stream, including the trailing
 * (under-full) group. With no SOAR/boundary replication every input vector is
 * written to the posting pages exactly once — walk the pages and require all
 * of them to survive (a dropped trailing entry would show up as written <
 * nvecs).
 */
TEST(parallel_fastscan_no_lost_partials)
{
	uint32_t dim = 32, nvecs = 10000;
	float	*vecs = make_vectors(nvecs, dim, 7);

	PrismIndexConfig config = {
			.nlist		   = 8,
			.metric		   = DISTANCE_L2,
			.centroid_fmt  = PRISM_CENTROID_FMT_RABITQ,
			.encode_rabitq = true,
			.posting_fmt   = PRISM_POSTING_FMT_PAGES,
			.fastscan	   = 8,
			.nworkers	   = 4, /* force the shared parallel driver */
	};

	PrismIndex *idx = build_from_array(vecs, nvecs, dim, &config);
	ASSERT_NOT_NULL(idx, "parallel fastscan build should succeed");

	uint32_t written = count_posting_entries(idx);
	ASSERT_EQ(
			nvecs,
			written,
			"every vector must be written to the posting pages (no dropped "
			"trailing partials)");

	verify_head_meta(result, idx);

	prism_index_destroy(idx);
}

/*
 * Build/query reachability tripwire: every indexed row should be
 * findable by the query router at a modest probe count. The build
 * places rows by exact centroid distance while queries route through
 * the compressed tree with exact candidate re-ranking, so a small
 * fraction of rows can sit one rank past a narrow probe set — the
 * floors below are calibrated with slack. A regression in build
 * placement (rows filed outside the query's reach) trips them long
 * before it costs measurable benchmark recall.
 */
TEST(paged_build_self_reachability)
{
	uint32_t dim = 24, nvecs = 20000;
	float	*vecs = make_vectors(nvecs, dim, 7);

	PrismIndexConfig config = {
			.nlist		   = 256,
			.metric		   = DISTANCE_L2,
			.centroid_fmt  = PRISM_CENTROID_FMT_RABITQ,
			.encode_rabitq = true,
			.posting_fmt   = PRISM_POSTING_FMT_PAGES,
			.nworkers	   = 2,
	};

	PrismIndex *idx = build_from_array(vecs, nvecs, dim, &config);
	ASSERT_NOT_NULL(idx, "paged parallel build should succeed");

	PrismQueryCtx *qctx = prism_query_ctx_create(idx, 1, 16);

	uint32_t nq		 = 500;
	uint32_t hits_4	 = 0;
	uint32_t hits_16 = 0;
	for (uint32_t q = 0; q < nq; q++)
	{
		const float *self = vecs + (size_t)(q * (nvecs / nq)) * dim;
		uint32_t	 id;
		uint32_t	 got;

		got = prism_query_exec(
				qctx, self, 1, 4, VS_DISTANCE_MODE_ASYMMETRIC, true, &id);
		if (got == 1 && id == q * (nvecs / nq))
			hits_4++;

		got = prism_query_exec(
				qctx, self, 1, 16, VS_DISTANCE_MODE_ASYMMETRIC, true, &id);
		if (got == 1 && id == q * (nvecs / nq))
			hits_16++;
	}

	TEST_PRINT(
			"  self-reachability: np4 %u/%u, np16 %u/%u\n",
			hits_4,
			nq,
			hits_16,
			nq);
	/* Floors calibrated on this synthetic regime (uniform data, 1-bit
	 * routing at low dim is noisy): healthy builds measure ~65%/~85%.
	 * The sharper signal is the SLOPE — misplaced rows are unreachable
	 * at any probe count, so a placement regression flattens the curve
	 * (measured during the assignment-regression post-mortem: broken
	 * builds were rank-flat while healthy ones climbed steeply). */
	ASSERT_TRUE(
			hits_4 >= (nq * 50) / 100,
			"row placement must be reachable at nprobe 4 (>=50%)");
	ASSERT_TRUE(
			hits_16 >= (nq * 75) / 100,
			"row placement must be reachable at nprobe 16 (>=75%)");
	ASSERT_TRUE(
			hits_16 >= hits_4 + (nq * 5) / 100,
			"reachability must climb with nprobe (flat = misplaced rows)");

	prism_query_ctx_destroy(qctx);
	prism_index_destroy(idx);
	vs_free(vecs);
}

/*
 * Build-time assignment parity: the page-backed build descent must file
 * each row in its exact nearest cluster when the candidate pool covers
 * every leaf. The build scores the INTERNAL tree levels against the exact
 * float centroids collected during the streaming tree write
 * (PrismExactInternalCentroids) and exact-re-ranks the TOPK leaf finalists
 * against the head pages' full-precision pt_centroids; with nlist <=
 * PRISM_SECONDARY_TOPK the finalists cover ALL leaves, so the primary
 * assignment must equal a brute-force nearest-centroid scan exactly —
 * both sides compute the same metric kernel on the same rotated floats
 * (P^T preserves L2 and dot products alike). Parameterized over the
 * centroid format (each descent scores — and collects exact internal
 * centroids for — its own format branch) and the metric (inner product
 * must rank the finalists by -dot to match query-time routing, not L2).
 */
static void
check_paged_assignment_parity(
		VsTestResult *result, PrismCentroidFormat fmt, DistanceMetric metric)
{
	uint32_t dim = 24, nvecs = 3000;
	float	*vecs = make_vectors(nvecs, dim, 11);

	PrismIndexConfig config = {
			.nlist		   = 27,
			.fan_out	   = 3, /* 3-level tree: scored internal levels */
			.metric		   = metric,
			.centroid_fmt  = fmt,
			.encode_rabitq = true,
			.posting_fmt   = PRISM_POSTING_FMT_PAGES,
			.nworkers	   = 2,
			/* soar_lambda / boundary_epsilon left 0: no replication, so
			 * list membership identifies the primary assignment. */
	};

	PrismIndex *idx = build_from_array(vecs, nvecs, dim, &config);
	ASSERT_NOT_NULL(idx, "paged parallel build should succeed");
	ASSERT_TRUE(
			idx->base.nlevels >= 3,
			"tree must have scored internal levels (the exact ones)");
	ASSERT_TRUE(
			idx->nlist <= PRISM_SECONDARY_TOPK,
			"candidate pool must cover every leaf for exact parity");

	VsStorage *st = idx->base.posting_storage;

	/* Gather each cluster's encode reference (P^T * centroid) and each
	 * vector's assigned cluster from the posting pages. */
	float	 *pt_cents = vs_alloc((size_t)idx->nlist * dim * sizeof(float));
	uint32_t *assigned = vs_alloc(nvecs * sizeof(uint32_t));
	for (uint32_t i = 0; i < nvecs; i++)
		assigned[i] = UINT32_MAX;

	for (uint32_t c = 0; c < idx->nlist; c++)
	{
		BlockNumber blk = idx->first_posting + c;

		Page head = vs_storage_read_page(st, blk);
		memcpy(pt_cents + (size_t)c * dim,
			   prism_posting_pt_centroid(head),
			   (size_t)dim * sizeof(float));
		vs_storage_release_page(st, blk);

		while (blk != InvalidBlockNumber)
		{
			Page	 pg		 = vs_storage_read_page(st, blk);
			bool	 first	 = (prism_posting_opaque(pg)->flags &
							PRISM_POSTING_PAGE_FIRST) != 0;
			char	*content = first ? prism_posting_content_first(pg, dim)
									 : prism_posting_content(pg);
			uint32_t count	 = prism_posting_page_count(pg);

			for (uint32_t i = 0; i < count; i++)
			{
				const PrismPostingEntryHeader *e =
						prism_posting_entry_at(content, i, dim);
				uint32_t vid = prism_posting_get_vector_id(&e->meta.tid);
				ASSERT_TRUE(vid < nvecs, "tid decodes to a vector id");
				ASSERT_EQ(
						UINT32_MAX,
						assigned[vid],
						"no replication: one list per vector");
				assigned[vid] = c;
			}
			BlockNumber next = prism_posting_opaque(pg)->next_blkno;
			vs_storage_release_page(st, blk);
			blk = next;
		}
	}

	/* Brute-force nearest centroid in the rotated space, computed with
	 * the same kernel on the same floats the assign path used — so
	 * parity must be exact, not approximate. Inner product ranks by
	 * -dot, matching the query descent; the others by squared L2. */
	bool   rank_by_dot = (metric == DISTANCE_INNER_PRODUCT);
	float *pt_vec	   = vs_alloc((size_t)dim * sizeof(float));
	for (uint32_t i = 0; i < nvecs; i++)
	{
		vs_rabitq_rotate(idx->base.params, vecs + (size_t)i * dim, pt_vec);

		uint32_t best	   = 0;
		float	 best_dist = 0.0f;
		for (uint32_t c = 0; c < idx->nlist; c++)
		{
			const float *cand = pt_cents + (size_t)c * dim;
			float		 d = rank_by_dot ? -vs_dot_product(pt_vec, cand, dim)
										 : vs_l2_distance_squared(pt_vec, cand, dim);
			if (c == 0 || d < best_dist)
			{
				best_dist = d;
				best	  = c;
			}
		}
		ASSERT_EQ(
				best,
				assigned[i],
				"row must be filed in its exact nearest cluster");
	}

	vs_free(pt_vec);
	vs_free(pt_cents);
	vs_free(assigned);
	prism_index_destroy(idx);
	vs_free(vecs);
}

TEST(paged_build_assignment_parity)
{
	check_paged_assignment_parity(
			result, PRISM_CENTROID_FMT_RABITQ, DISTANCE_L2);
}

TEST(paged_build_assignment_parity_float)
{
	check_paged_assignment_parity(
			result, PRISM_CENTROID_FMT_FLOAT, DISTANCE_L2);
}

TEST(paged_build_assignment_parity_fastscan)
{
	check_paged_assignment_parity(
			result, PRISM_CENTROID_FMT_FASTSCAN, DISTANCE_L2);
}

TEST(paged_build_assignment_parity_ip)
{
	/* Rows must be filed in their -dot-nearest list, not L2-nearest:
	 * inner-product queries route by -dot, and a build/query metric
	 * mismatch misfiles every row whose two orders disagree. */
	check_paged_assignment_parity(
			result, PRISM_CENTROID_FMT_FLOAT, DISTANCE_INNER_PRODUCT);
}
