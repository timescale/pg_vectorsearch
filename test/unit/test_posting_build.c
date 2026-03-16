/*
 * test_posting_build.c - Unit tests for streaming posting list builder
 *
 * Tests cover:
 * - Basic build and scan-back roundtrip
 * - Empty cluster (no adds)
 * - Multi-page spanning
 * - Distance sanity (finite, non-negative)
 */

#include <math.h>
#include <string.h>

#include "core/memory.h"
#include "core/pg_compat.h"
#include "index/posting_build.h"
#include "index/posting_scan.h"
#include "mkt_test.h"
#include "quant/rabitq.h"

TEST_GROUP(PostingBuild);
TEST_MEMCTX_FIXTURE();

/* ----------------------------------------------------------------
 * Mock storage (same pattern as test_posting_scan.c)
 * ---------------------------------------------------------------- */

typedef struct TestStorage
{
	char	*pages;
	uint32_t num_pages;
	uint32_t next_page;
} TestStorage;

typedef struct TestMktStorage
{
	MktStorage	 base;
	TestStorage *ts;
} TestMktStorage;

static Page
test_read_page(MktStorage *self, BlockNumber blkno)
{
	TestMktStorage *tms = (TestMktStorage *)self;
	return tms->ts->pages + (size_t)blkno * BLCKSZ;
}

static void
test_release_page(MktStorage *self, BlockNumber blkno)
{
	(void)self;
	(void)blkno;
}

static Page
test_write_page(MktStorage *self, BlockNumber blkno)
{
	TestMktStorage *tms = (TestMktStorage *)self;
	return tms->ts->pages + (size_t)blkno * BLCKSZ;
}

static Page
test_new_page(MktStorage *self, BlockNumber *blkno_out)
{
	TestMktStorage *tms = (TestMktStorage *)self;
	*blkno_out			= tms->ts->next_page++;
	return tms->ts->pages + (size_t)(*blkno_out) * BLCKSZ;
}

static void
test_commit_page(MktStorage *self, BlockNumber blkno)
{
	(void)self;
	(void)blkno;
}

static const MktStorageOps test_storage_ops = {
		.read_page	  = test_read_page,
		.release_page = test_release_page,
		.write_page	  = test_write_page,
		.new_page	  = test_new_page,
		.commit_page  = test_commit_page,
		.rerank		  = NULL,
};

static TestStorage *
create_test_storage(uint32_t num_pages)
{
	TestStorage *ts = mkt_alloc(sizeof(TestStorage));
	ts->pages		= mkt_alloc((size_t)num_pages * BLCKSZ);
	ts->num_pages	= num_pages;
	ts->next_page	= 0;
	memset(ts->pages, 0, (size_t)num_pages * BLCKSZ);
	return ts;
}

static void
destroy_test_storage(TestStorage *ts)
{
	mkt_free(ts->pages);
	mkt_free(ts);
}

static TestMktStorage
make_storage(TestStorage *ts)
{
	return (TestMktStorage){
			.base.ops = &test_storage_ops,
			.ts		  = ts,
	};
}

/* ----------------------------------------------------------------
 * Helpers
 * ---------------------------------------------------------------- */

static float *
make_vector(Dimension dim, int seed)
{
	float *data = mkt_alloc(dim * sizeof(float));
	for (Dimension i = 0; i < dim; i++)
		data[i] = (float)((i * 17 + seed) % 100 - 50) / 10.0f;
	return data;
}

/* Collect all results from a posting scan on one cluster */
static uint32_t
collect_scan_results(
		MktPostingScan		 *scan,
		VectorRef			  query_ref,
		VectorRef			  cent_ref,
		BlockNumber			  head,
		MktPostingScanResult *results,
		uint32_t			  max_results)
{
	mkt_posting_scan_begin_cluster(scan, query_ref, cent_ref, head);

	uint32_t count = 0;
	while (count < max_results && mkt_posting_scan_next(scan, &results[count]))
	{
		count++;
	}

	mkt_posting_scan_end_cluster(scan);
	return count;
}

/* ----------------------------------------------------------------
 * build_cluster_basic — build a few vectors, scan back, verify TIDs
 * ---------------------------------------------------------------- */

TEST(build_cluster_basic)
{
	Dimension	   dim = 64;
	const uint32_t n   = 5;

	RaBitQParams *params = mkt_rabitq_create(dim, 42);
	ASSERT_NOT_NULL(params, "params created");

	float *centroid = mkt_alloc(dim * sizeof(float));
	for (Dimension i = 0; i < dim; i++)
		centroid[i] = 0.0f;
	VectorRef cent_ref = {.data = centroid, .dim = dim};

	TestStorage	  *ts	 = create_test_storage(4);
	TestMktStorage tstor = make_storage(ts);

	/* Build posting list via builder */
	MktPostingBuilder pb;
	mkt_posting_builder_init(&pb, &tstor.base, params, dim, 0, centroid);

	for (uint32_t i = 0; i < n; i++)
	{
		float	 *v	   = make_vector(dim, (int)i * 100);
		VectorRef vref = {.data = v, .dim = dim};
		mkt_posting_builder_add(&pb, vref, (BlockNumber)(200 + i), 1);
	}

	BlockNumber head = mkt_posting_builder_finish(&pb);
	mkt_posting_builder_cleanup(&pb);

	ASSERT_TRUE(head != InvalidBlockNumber, "should return valid head");

	/* Verify first page metadata */
	Page				  page	 = ts->pages + (size_t)head * BLCKSZ;
	MktPostingPageOpaque *opaque = MKT_POSTING_OPAQUE(page);
	ASSERT_EQ(0, opaque->cluster_id, "cluster_id should be 0");
	ASSERT_EQ(
			MKT_POSTING_PAGE_FIRST,
			opaque->flags & MKT_POSTING_PAGE_FIRST,
			"first page should have FIRST flag");

	/* Scan back and verify all TIDs returned */
	float	 *query		= make_vector(dim, 9999);
	VectorRef query_ref = {.data = query, .dim = dim};

	MktPostingScan scan;
	mkt_posting_scan_init(&scan, &tstor.base, params, dim);

	MktPostingScanResult results[20];
	uint32_t			 nresults = collect_scan_results(
			&scan, query_ref, cent_ref, head, results, 20);

	ASSERT_EQ(n, nresults, "should return all entries");

	/* Verify TID block numbers */
	bool found[5] = {false};
	for (uint32_t i = 0; i < nresults; i++)
	{
		BlockNumber blk = ItemPointerGetBlockNumber(&results[i].tid);
		ASSERT_TRUE(blk >= 200 && blk < 205, "TID block in range");
		found[blk - 200] = true;
	}
	for (uint32_t i = 0; i < n; i++)
		ASSERT_TRUE(found[i], "all TIDs present");

	mkt_posting_scan_cleanup(&scan);
	destroy_test_storage(ts);
	mkt_rabitq_destroy(params);
}

/* ----------------------------------------------------------------
 * build_cluster_empty — no adds returns InvalidBlockNumber
 * ---------------------------------------------------------------- */

TEST(build_cluster_empty)
{
	Dimension dim = 64;

	RaBitQParams *params = mkt_rabitq_create(dim, 42);
	ASSERT_NOT_NULL(params, "params created");

	float *centroid = mkt_alloc(dim * sizeof(float));
	for (Dimension i = 0; i < dim; i++)
		centroid[i] = 0.0f;

	TestStorage	  *ts	 = create_test_storage(1);
	TestMktStorage tstor = make_storage(ts);

	MktPostingBuilder pb;
	mkt_posting_builder_init(&pb, &tstor.base, params, dim, 0, centroid);

	BlockNumber head = mkt_posting_builder_finish(&pb);
	mkt_posting_builder_cleanup(&pb);

	ASSERT_EQ(
			InvalidBlockNumber,
			head,
			"empty build should return InvalidBlockNumber");

	destroy_test_storage(ts);
	mkt_rabitq_destroy(params);
}

/* ----------------------------------------------------------------
 * build_cluster_multi_page — enough vectors to span multiple pages
 * ---------------------------------------------------------------- */

TEST(build_cluster_multi_page)
{
	Dimension dim = 64;

	RaBitQParams *params = mkt_rabitq_create(dim, 42);
	ASSERT_NOT_NULL(params, "params created");

	float *centroid = mkt_alloc(dim * sizeof(float));
	for (Dimension i = 0; i < dim; i++)
		centroid[i] = 0.0f;
	VectorRef cent_ref = {.data = centroid, .dim = dim};

	/* Calculate how many entries fit per page, then exceed it */
	uint32_t per_page = mkt_posting_max_entries(dim);
	uint32_t n		  = per_page * 2 + 3; /* spans 3 pages */

	/* Allocate enough pages */
	uint32_t	   npages = (n / per_page) + 2;
	TestStorage	  *ts	  = create_test_storage(npages);
	TestMktStorage tstor  = make_storage(ts);

	MktPostingBuilder pb;
	mkt_posting_builder_init(&pb, &tstor.base, params, dim, 7, centroid);

	for (uint32_t i = 0; i < n; i++)
	{
		float	 *v	   = make_vector(dim, (int)i);
		VectorRef vref = {.data = v, .dim = dim};
		mkt_posting_builder_add(&pb, vref, (BlockNumber)(1000 + i), 1);
	}

	BlockNumber head = mkt_posting_builder_finish(&pb);
	mkt_posting_builder_cleanup(&pb);

	ASSERT_TRUE(head != InvalidBlockNumber, "should return valid head");

	/* Verify multiple pages were used */
	ASSERT_TRUE(ts->next_page >= 3, "should use at least 3 pages");

	/* Verify page chain */
	Page				  page	 = ts->pages + (size_t)head * BLCKSZ;
	MktPostingPageOpaque *opaque = MKT_POSTING_OPAQUE(page);
	ASSERT_EQ(7, opaque->cluster_id, "cluster_id should be 7");
	ASSERT_TRUE(
			opaque->next_blkno != InvalidBlockNumber,
			"first page should chain to next");

	/* Scan back and verify all entries returned */
	float	 *query		= make_vector(dim, 9999);
	VectorRef query_ref = {.data = query, .dim = dim};

	MktPostingScan scan;
	mkt_posting_scan_init(&scan, &tstor.base, params, dim);

	MktPostingScanResult *results = mkt_alloc(
			n * sizeof(MktPostingScanResult));
	uint32_t nresults =
			collect_scan_results(&scan, query_ref, cent_ref, head, results, n);

	ASSERT_EQ(n, nresults, "should return all entries");

	mkt_posting_scan_cleanup(&scan);
	mkt_free(results);
	destroy_test_storage(ts);
	mkt_rabitq_destroy(params);
}

/* ----------------------------------------------------------------
 * build_multi_cluster_interleaved — multiple builders sharing storage
 *
 * Simulates the streaming build where nlist builders are active
 * simultaneously and vectors are added in arbitrary order.
 * ---------------------------------------------------------------- */

TEST(build_multi_cluster_interleaved)
{
	Dimension	   dim		   = 64;
	const uint32_t nclusters   = 3;
	const uint32_t per_cluster = 4;

	RaBitQParams *params = mkt_rabitq_create(dim, 42);
	ASSERT_NOT_NULL(params, "params created");

	/* Each cluster has a different centroid */
	float *centroids = mkt_alloc(nclusters * dim * sizeof(float));
	for (uint32_t c = 0; c < nclusters; c++)
		for (Dimension i = 0; i < dim; i++)
			centroids[c * dim + i] = (float)c;

	/* Enough pages for all clusters */
	TestStorage	  *ts	 = create_test_storage(20);
	TestMktStorage tstor = make_storage(ts);

	/* Initialize all builders */
	MktPostingBuilder builders[3];
	for (uint32_t c = 0; c < nclusters; c++)
	{
		mkt_posting_builder_init(
				&builders[c],
				&tstor.base,
				params,
				dim,
				c,
				centroids + (size_t)c * dim);
	}

	/* Add vectors interleaved across clusters:
	 * cluster 0, cluster 1, cluster 2, cluster 0, ... */
	for (uint32_t i = 0; i < per_cluster; i++)
	{
		for (uint32_t c = 0; c < nclusters; c++)
		{
			float	   *v	 = make_vector(dim, (int)(c * 100 + i));
			VectorRef	vref = {.data = v, .dim = dim};
			BlockNumber blk	 = (BlockNumber)(c * 1000 + i);
			mkt_posting_builder_add(&builders[c], vref, blk, 1);
		}
	}

	/* Finish all builders and collect heads */
	BlockNumber heads[3];
	for (uint32_t c = 0; c < nclusters; c++)
	{
		heads[c] = mkt_posting_builder_finish(&builders[c]);
		mkt_posting_builder_cleanup(&builders[c]);
		ASSERT_TRUE(
				heads[c] != InvalidBlockNumber,
				"each cluster should have a valid head");
	}

	/* Verify each cluster independently scans back correctly */
	float	 *query		= make_vector(dim, 9999);
	VectorRef query_ref = {.data = query, .dim = dim};

	MktPostingScan scan;
	mkt_posting_scan_init(&scan, &tstor.base, params, dim);

	for (uint32_t c = 0; c < nclusters; c++)
	{
		VectorRef cent_ref = {
				.data = centroids + (size_t)c * dim,
				.dim  = dim,
		};

		MktPostingScanResult results[20];
		uint32_t			 nresults = collect_scan_results(
				&scan, query_ref, cent_ref, heads[c], results, 20);

		ASSERT_EQ(
				per_cluster, nresults, "each cluster should have all entries");

		/* Verify TIDs belong to this cluster */
		for (uint32_t i = 0; i < nresults; i++)
		{
			BlockNumber blk = ItemPointerGetBlockNumber(&results[i].tid);
			ASSERT_TRUE(
					blk >= c * 1000 && blk < c * 1000 + per_cluster,
					"TID block should match cluster");
		}

		/* Verify cluster_id on first page */
		Page				  page	 = ts->pages + (size_t)heads[c] * BLCKSZ;
		MktPostingPageOpaque *opaque = MKT_POSTING_OPAQUE(page);
		ASSERT_EQ(c, opaque->cluster_id, "cluster_id should match");
	}

	mkt_posting_scan_cleanup(&scan);
	destroy_test_storage(ts);
	mkt_rabitq_destroy(params);
}

/* ----------------------------------------------------------------
 * build_cluster_large_dim768 — realistic embedding dimension
 *
 * dim=768, 200 vectors → should span ~3 pages (max ~72/page).
 * Verifies all TIDs round-trip and page count matches.
 * ---------------------------------------------------------------- */

TEST(build_cluster_large_dim768)
{
	Dimension	   dim = 768;
	const uint32_t n   = 200;

	RaBitQParams *params = mkt_rabitq_create(dim, 42);
	ASSERT_NOT_NULL(params, "params created");

	float *centroid = mkt_alloc(dim * sizeof(float));
	for (Dimension i = 0; i < dim; i++)
		centroid[i] = 0.0f;
	VectorRef cent_ref = {.data = centroid, .dim = dim};

	uint32_t	   per_page	 = mkt_posting_max_entries(dim);
	uint32_t	   exp_pages = (n + per_page - 1) / per_page;
	uint32_t	   npages	 = exp_pages + 2;
	TestStorage	  *ts		 = create_test_storage(npages);
	TestMktStorage tstor	 = make_storage(ts);

	MktPostingBuilder pb;
	mkt_posting_builder_init(&pb, &tstor.base, params, dim, 0, centroid);

	for (uint32_t i = 0; i < n; i++)
	{
		float	 *v	   = make_vector(dim, (int)i);
		VectorRef vref = {.data = v, .dim = dim};
		mkt_posting_builder_add(&pb, vref, (BlockNumber)(500 + i), 1);
	}

	BlockNumber head = mkt_posting_builder_finish(&pb);
	mkt_posting_builder_cleanup(&pb);

	ASSERT_TRUE(head != InvalidBlockNumber, "should return valid head");
	ASSERT_TRUE(
			ts->next_page >= exp_pages, "should use expected number of pages");

	/* Scan back and verify all TIDs */
	float	 *query		= make_vector(dim, 9999);
	VectorRef query_ref = {.data = query, .dim = dim};

	MktPostingScan scan;
	mkt_posting_scan_init(&scan, &tstor.base, params, dim);

	MktPostingScanResult *results = mkt_alloc(
			n * sizeof(MktPostingScanResult));
	uint32_t nresults =
			collect_scan_results(&scan, query_ref, cent_ref, head, results, n);

	ASSERT_EQ(n, nresults, "should return all 200 entries");

	/* Verify all TID block numbers present */
	bool *found = mkt_alloc(n * sizeof(bool));
	memset(found, 0, n * sizeof(bool));
	for (uint32_t i = 0; i < nresults; i++)
	{
		BlockNumber blk = ItemPointerGetBlockNumber(&results[i].tid);
		ASSERT_TRUE(blk >= 500 && blk < 500 + n, "TID block in range");
		found[blk - 500] = true;
	}
	for (uint32_t i = 0; i < n; i++)
		ASSERT_TRUE(found[i], "all TIDs present");

	mkt_posting_scan_cleanup(&scan);
	mkt_free(found);
	mkt_free(results);
	destroy_test_storage(ts);
	mkt_rabitq_destroy(params);
}

/* ----------------------------------------------------------------
 * build_cluster_many_pages — stress multi-page chains (10+ pages)
 *
 * dim=64, enough vectors to create 10+ pages.
 * Verifies scan returns all entries and page chain length.
 * ---------------------------------------------------------------- */

TEST(build_cluster_many_pages)
{
	Dimension dim = 64;

	RaBitQParams *params = mkt_rabitq_create(dim, 42);
	ASSERT_NOT_NULL(params, "params created");

	float *centroid = mkt_alloc(dim * sizeof(float));
	for (Dimension i = 0; i < dim; i++)
		centroid[i] = 0.0f;
	VectorRef cent_ref = {.data = centroid, .dim = dim};

	uint32_t per_page	  = mkt_posting_max_entries(dim);
	uint32_t target_pages = 11;
	uint32_t n			  = per_page * target_pages + 3;

	uint32_t	   npages = (n / per_page) + 4;
	TestStorage	  *ts	  = create_test_storage(npages);
	TestMktStorage tstor  = make_storage(ts);

	MktPostingBuilder pb;
	mkt_posting_builder_init(&pb, &tstor.base, params, dim, 0, centroid);

	for (uint32_t i = 0; i < n; i++)
	{
		float	 *v	   = make_vector(dim, (int)i);
		VectorRef vref = {.data = v, .dim = dim};
		mkt_posting_builder_add(&pb, vref, (BlockNumber)(2000 + i), 1);
	}

	BlockNumber head = mkt_posting_builder_finish(&pb);
	mkt_posting_builder_cleanup(&pb);

	ASSERT_TRUE(head != InvalidBlockNumber, "should return valid head");
	ASSERT_TRUE(ts->next_page >= target_pages, "should use 10+ pages");

	/* Walk the page chain and count pages */
	uint32_t	chain_len = 0;
	BlockNumber blk		  = head;
	while (blk != InvalidBlockNumber)
	{
		chain_len++;
		Page				  page = ts->pages + (size_t)blk * BLCKSZ;
		MktPostingPageOpaque *op   = MKT_POSTING_OPAQUE(page);
		blk						   = op->next_blkno;
	}
	ASSERT_TRUE(chain_len >= target_pages, "chain should span 10+ pages");

	/* Scan back and verify all entries */
	float	 *query		= make_vector(dim, 9999);
	VectorRef query_ref = {.data = query, .dim = dim};

	MktPostingScan scan;
	mkt_posting_scan_init(&scan, &tstor.base, params, dim);

	MktPostingScanResult *results = mkt_alloc(
			n * sizeof(MktPostingScanResult));
	uint32_t nresults =
			collect_scan_results(&scan, query_ref, cent_ref, head, results, n);

	ASSERT_EQ(n, nresults, "should return all entries");

	mkt_posting_scan_cleanup(&scan);
	mkt_free(results);
	destroy_test_storage(ts);
	mkt_rabitq_destroy(params);
}

/* ----------------------------------------------------------------
 * build_multi_cluster_skewed — uneven cluster sizes
 *
 * 3 clusters: 200 vectors, 5 vectors, 0 vectors (empty).
 * Verifies builder handles skewed distributions and empty clusters.
 * ---------------------------------------------------------------- */

TEST(build_multi_cluster_skewed)
{
	Dimension dim = 64;

	RaBitQParams *params = mkt_rabitq_create(dim, 42);
	ASSERT_NOT_NULL(params, "params created");

	const uint32_t cluster_sizes[3] = {200, 5, 0};
	const uint32_t nclusters		= 3;

	float *centroids = mkt_alloc(nclusters * dim * sizeof(float));
	for (uint32_t c = 0; c < nclusters; c++)
		for (Dimension i = 0; i < dim; i++)
			centroids[c * dim + i] = (float)c * 10.0f;

	/* Allocate enough pages for the large cluster */
	uint32_t	   per_page = mkt_posting_max_entries(dim);
	uint32_t	   npages	= (cluster_sizes[0] / per_page) + 4;
	TestStorage	  *ts		= create_test_storage(npages);
	TestMktStorage tstor	= make_storage(ts);

	/* Initialize all builders */
	MktPostingBuilder builders[3];
	for (uint32_t c = 0; c < nclusters; c++)
	{
		mkt_posting_builder_init(
				&builders[c],
				&tstor.base,
				params,
				dim,
				c,
				centroids + (size_t)c * dim);
	}

	/* Add vectors to clusters according to sizes */
	for (uint32_t c = 0; c < nclusters; c++)
	{
		for (uint32_t i = 0; i < cluster_sizes[c]; i++)
		{
			float	 *v	   = make_vector(dim, (int)(c * 1000 + i));
			VectorRef vref = {.data = v, .dim = dim};
			mkt_posting_builder_add(
					&builders[c], vref, (BlockNumber)(c * 10000 + i), 1);
		}
	}

	/* Finish all builders */
	BlockNumber heads[3];
	for (uint32_t c = 0; c < nclusters; c++)
	{
		heads[c] = mkt_posting_builder_finish(&builders[c]);
		mkt_posting_builder_cleanup(&builders[c]);
	}

	/* Cluster 0 (200 vectors): should have valid head */
	ASSERT_TRUE(
			heads[0] != InvalidBlockNumber,
			"large cluster should have valid head");

	/* Cluster 1 (5 vectors): should have valid head */
	ASSERT_TRUE(
			heads[1] != InvalidBlockNumber,
			"small cluster should have valid head");

	/* Cluster 2 (0 vectors): should be InvalidBlockNumber */
	ASSERT_EQ(
			InvalidBlockNumber,
			heads[2],
			"empty cluster should return InvalidBlockNumber");

	/* Scan back the large cluster */
	float	 *query		= make_vector(dim, 9999);
	VectorRef query_ref = {.data = query, .dim = dim};

	MktPostingScan scan;
	mkt_posting_scan_init(&scan, &tstor.base, params, dim);

	MktPostingScanResult *results = mkt_alloc(
			cluster_sizes[0] * sizeof(MktPostingScanResult));
	VectorRef cent0_ref = {.data = centroids, .dim = dim};
	uint32_t  nresults	= collect_scan_results(
			  &scan, query_ref, cent0_ref, heads[0], results, cluster_sizes[0]);

	ASSERT_EQ(
			cluster_sizes[0],
			nresults,
			"large cluster should return all 200 entries");

	/* Scan back the small cluster */
	MktPostingScanResult small_results[10];
	VectorRef			 cent1_ref = {
					   .data = centroids + dim,
					   .dim	 = dim,
	   };
	uint32_t nsmall = collect_scan_results(
			&scan, query_ref, cent1_ref, heads[1], small_results, 10);

	ASSERT_EQ(
			cluster_sizes[1],
			nsmall,
			"small cluster should return all 5 entries");

	mkt_posting_scan_cleanup(&scan);
	mkt_free(results);
	destroy_test_storage(ts);
	mkt_rabitq_destroy(params);
}

/* ----------------------------------------------------------------
 * build_cluster_dim3 — match regression test dimension
 *
 * dim=3, 50 vectors. Exercises the same encoding path as the PG
 * regression test and catches dim=3 edge cases in RaBitQ encoding.
 * ---------------------------------------------------------------- */

TEST(build_cluster_dim3)
{
	Dimension	   dim = 3;
	const uint32_t n   = 50;

	RaBitQParams *params = mkt_rabitq_create(dim, 42);
	ASSERT_NOT_NULL(params, "params created");

	float *centroid = mkt_alloc(dim * sizeof(float));
	for (Dimension i = 0; i < dim; i++)
		centroid[i] = 0.25f;
	VectorRef cent_ref = {.data = centroid, .dim = dim};

	TestStorage	  *ts	 = create_test_storage(4);
	TestMktStorage tstor = make_storage(ts);

	MktPostingBuilder pb;
	mkt_posting_builder_init(&pb, &tstor.base, params, dim, 0, centroid);

	for (uint32_t i = 0; i < n; i++)
	{
		float	 *v	   = make_vector(dim, (int)i * 7);
		VectorRef vref = {.data = v, .dim = dim};
		mkt_posting_builder_add(&pb, vref, (BlockNumber)(100 + i), 1);
	}

	BlockNumber head = mkt_posting_builder_finish(&pb);
	mkt_posting_builder_cleanup(&pb);

	ASSERT_TRUE(head != InvalidBlockNumber, "should return valid head");

	/* Scan back */
	float	 *query		= make_vector(dim, 4321);
	VectorRef query_ref = {.data = query, .dim = dim};

	MktPostingScan scan;
	mkt_posting_scan_init(&scan, &tstor.base, params, dim);

	MktPostingScanResult results[60];
	uint32_t			 nresults = collect_scan_results(
			&scan, query_ref, cent_ref, head, results, 60);

	ASSERT_EQ(n, nresults, "should return all 50 entries");

	/* Verify distances are finite (note: at dim=3, RaBitQ quantization
	 * error can produce slightly negative estimated distances) */
	for (uint32_t i = 0; i < nresults; i++)
	{
		ASSERT_TRUE(
				isfinite(results[i].distance), "distance should be finite");
	}

	/* Verify all TID block numbers present */
	bool found[50] = {false};
	for (uint32_t i = 0; i < nresults; i++)
	{
		BlockNumber blk = ItemPointerGetBlockNumber(&results[i].tid);
		ASSERT_TRUE(blk >= 100 && blk < 150, "TID block in range");
		found[blk - 100] = true;
	}
	for (uint32_t i = 0; i < n; i++)
		ASSERT_TRUE(found[i], "all TIDs present");

	mkt_posting_scan_cleanup(&scan);
	destroy_test_storage(ts);
	mkt_rabitq_destroy(params);
}

/* ----------------------------------------------------------------
 * build_cluster_roundtrip — verify distances are finite/non-negative
 * ---------------------------------------------------------------- */

TEST(build_cluster_roundtrip)
{
	Dimension	   dim = 64;
	const uint32_t n   = 8;

	RaBitQParams *params = mkt_rabitq_create(dim, 42);
	ASSERT_NOT_NULL(params, "params created");

	float *centroid = mkt_alloc(dim * sizeof(float));
	for (Dimension i = 0; i < dim; i++)
		centroid[i] = 0.0f;
	VectorRef cent_ref = {.data = centroid, .dim = dim};

	TestStorage	  *ts	 = create_test_storage(4);
	TestMktStorage tstor = make_storage(ts);

	MktPostingBuilder pb;
	mkt_posting_builder_init(&pb, &tstor.base, params, dim, 0, centroid);

	for (uint32_t i = 0; i < n; i++)
	{
		float	 *v	   = make_vector(dim, (int)i * 50);
		VectorRef vref = {.data = v, .dim = dim};
		mkt_posting_builder_add(&pb, vref, (BlockNumber)i, 1);
	}

	BlockNumber head = mkt_posting_builder_finish(&pb);
	mkt_posting_builder_cleanup(&pb);

	ASSERT_TRUE(head != InvalidBlockNumber, "should return valid head");

	/* Scan back and check distance properties */
	float	 *query		= make_vector(dim, 7777);
	VectorRef query_ref = {.data = query, .dim = dim};

	MktPostingScan scan;
	mkt_posting_scan_init(&scan, &tstor.base, params, dim);

	MktPostingScanResult results[20];
	uint32_t			 nresults = collect_scan_results(
			&scan, query_ref, cent_ref, head, results, 20);

	ASSERT_EQ(n, nresults, "should return all entries");

	for (uint32_t i = 0; i < nresults; i++)
	{
		ASSERT_TRUE(
				isfinite(results[i].distance), "distance should be finite");
		ASSERT_TRUE(
				results[i].distance >= 0.0f,
				"distance should be non-negative");
		ASSERT_TRUE(isfinite(results[i].error), "error should be finite");
		ASSERT_TRUE(results[i].error >= 0.0f, "error should be non-negative");
	}

	mkt_posting_scan_cleanup(&scan);
	destroy_test_storage(ts);
	mkt_rabitq_destroy(params);
}
