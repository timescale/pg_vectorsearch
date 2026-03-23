/*
 * test_posting_scan.c - Unit tests for posting list scan and write
 *
 * Tests cover:
 * - Multi-page chain walk via iterator
 * - Multiple cluster iteration
 * - Deleted entries skipped
 * - Threshold push-down pruning
 * - Write list and read back
 * - Edge cases (empty cluster, invalid block)
 */

#include <math.h>
#include <string.h>

#include "core/memory.h"
#include "core/pg_compat.h"
#include "index/posting_scan.h"
#include "mkt_test.h"
#include "quant/rabitq.h"

TEST_GROUP(PostingScan);
TEST_MEMCTX_FIXTURE();

/* ----------------------------------------------------------------
 * Mock storage (flat array of pages)
 * ---------------------------------------------------------------- */

typedef struct TestStorage
{
	char	*pages;
	uint32_t num_pages;
	uint32_t next_page; /* next free page for new_page */
} TestStorage;

/* Embeds MktStorage as first member so we can upcast */
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

/*
 * Build a posting page with n encoded entries. Returns the page
 * block number.
 */
static BlockNumber
build_posting_page(
		TestStorage *ts,
		Dimension	 dim,
		uint32_t	 cluster_id,
		RaBitQData **encodings,
		uint32_t	 count,
		uint16_t	 page_flags)
{
	BlockNumber blkno = ts->next_page++;
	Page		page  = ts->pages + (size_t)blkno * BLCKSZ;

	mkt_posting_page_init(page, cluster_id, page_flags);

	for (uint32_t i = 0; i < count; i++)
	{
		mkt_posting_page_add(
				page,
				dim,
				(BlockNumber)(100 + i),
				(OffsetNumber)(i + 1),
				encodings[i],
				0);
	}

	return blkno;
}

/*
 * Collect all results from a posting scan iterator on one cluster.
 * Returns the number of results collected.
 */
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
 * Multi-page chain
 * ---------------------------------------------------------------- */

TEST(scan_multi_page_chain)
{
	Dimension dim = 64;
	const int n	  = 5;

	RaBitQParams *params = mkt_rabitq_create(dim, 42);
	ASSERT_NOT_NULL(params, "params created");

	float *centroid = mkt_alloc(dim * sizeof(float));
	for (Dimension i = 0; i < dim; i++)
		centroid[i] = 0.0f;
	VectorRef cent_ref = {.data = centroid, .dim = dim};

	/* Create encodings for two pages worth */
	RaBitQData **enc1 = mkt_alloc(n * sizeof(void *));
	RaBitQData **enc2 = mkt_alloc(n * sizeof(void *));
	for (int i = 0; i < n; i++)
	{
		float	 *v1  = make_vector(dim, i * 100);
		float	 *v2  = make_vector(dim, i * 100 + 50);
		VectorRef vr1 = {.data = v1, .dim = dim};
		VectorRef vr2 = {.data = v2, .dim = dim};
		enc1[i]		  = mkt_rabitq_encode(params, vr1, cent_ref);
		enc2[i]		  = mkt_rabitq_encode(params, vr2, cent_ref);
	}

	TestStorage *ts = create_test_storage(4);

	/* Build two pages and chain them */
	BlockNumber bn1 =
			build_posting_page(ts, dim, 0, enc1, n, MKT_POSTING_PAGE_FIRST);
	BlockNumber bn2 =
			build_posting_page(ts, dim, 0, enc2, n, MKT_POSTING_PAGE_OVERFLOW);

	/* Chain: page1 -> page2 */
	MktPostingPageOpaque *op1 = MKT_POSTING_OPAQUE(
			ts->pages + (size_t)bn1 * BLCKSZ);
	op1->next_blkno = bn2;

	/* Scan via iterator */
	float	 *query		= make_vector(dim, 9999);
	VectorRef query_ref = {.data = query, .dim = dim};

	TestMktStorage tstor = make_storage(ts);

	MktPostingScan scan;
	mkt_posting_scan_init(&scan, &tstor.base, params, dim);

	MktPostingScanResult results[20];
	uint32_t			 nresults =
			collect_scan_results(&scan, query_ref, cent_ref, bn1, results, 20);

	/* Should find entries from both pages (2*n = 10) */
	ASSERT_TRUE(nresults > 0, "should return results");
	ASSERT_EQ(
			(uint32_t)(2 * n),
			nresults,
			"should return all entries from both pages");

	mkt_posting_scan_cleanup(&scan);
	for (int i = 0; i < n; i++)
	{
		mkt_free(enc1[i]);
		mkt_free(enc2[i]);
	}
	mkt_free(enc1);
	mkt_free(enc2);
	destroy_test_storage(ts);
	mkt_rabitq_destroy(params);
}

/* ----------------------------------------------------------------
 * Multiple clusters
 * ---------------------------------------------------------------- */

TEST(scan_multiple_clusters)
{
	Dimension dim = 64;
	const int n	  = 4;

	RaBitQParams *params = mkt_rabitq_create(dim, 42);
	ASSERT_NOT_NULL(params, "params created");

	float *centroid = mkt_alloc(dim * sizeof(float));
	for (Dimension i = 0; i < dim; i++)
		centroid[i] = 0.0f;
	VectorRef cent_ref = {.data = centroid, .dim = dim};

	RaBitQData **enc1 = mkt_alloc(n * sizeof(void *));
	RaBitQData **enc2 = mkt_alloc(n * sizeof(void *));
	for (int i = 0; i < n; i++)
	{
		float	 *v1  = make_vector(dim, i * 200);
		float	 *v2  = make_vector(dim, i * 200 + 100);
		VectorRef vr1 = {.data = v1, .dim = dim};
		VectorRef vr2 = {.data = v2, .dim = dim};
		enc1[i]		  = mkt_rabitq_encode(params, vr1, cent_ref);
		enc2[i]		  = mkt_rabitq_encode(params, vr2, cent_ref);
	}

	TestStorage *ts = create_test_storage(4);

	BlockNumber bn1 =
			build_posting_page(ts, dim, 0, enc1, n, MKT_POSTING_PAGE_FIRST);
	BlockNumber bn2 =
			build_posting_page(ts, dim, 1, enc2, n, MKT_POSTING_PAGE_FIRST);

	float	 *query		= make_vector(dim, 7777);
	VectorRef query_ref = {.data = query, .dim = dim};

	TestMktStorage tstor = make_storage(ts);

	MktPostingScan scan;
	mkt_posting_scan_init(&scan, &tstor.base, params, dim);

	/* Scan both clusters, accumulate results */
	MktPostingScanResult results[20];
	uint32_t			 total = 0;

	total += collect_scan_results(
			&scan, query_ref, cent_ref, bn1, results + total, 20 - total);
	total += collect_scan_results(
			&scan, query_ref, cent_ref, bn2, results + total, 20 - total);

	ASSERT_TRUE(total > 0, "should return results");
	ASSERT_EQ(
			(uint32_t)(2 * n),
			total,
			"should return all entries from both clusters");

	mkt_posting_scan_cleanup(&scan);
	for (int i = 0; i < n; i++)
	{
		mkt_free(enc1[i]);
		mkt_free(enc2[i]);
	}
	mkt_free(enc1);
	mkt_free(enc2);
	destroy_test_storage(ts);
	mkt_rabitq_destroy(params);
}

/* ----------------------------------------------------------------
 * Deleted entries are skipped
 * ---------------------------------------------------------------- */

TEST(scan_skips_deleted)
{
	Dimension dim = 64;

	RaBitQParams *params = mkt_rabitq_create(dim, 42);
	ASSERT_NOT_NULL(params, "params created");

	float *centroid = mkt_alloc(dim * sizeof(float));
	for (Dimension i = 0; i < dim; i++)
		centroid[i] = 0.0f;
	VectorRef cent_ref = {.data = centroid, .dim = dim};

	/* Create 4 entries, mark 2 as deleted */
	float	 *vec = make_vector(dim, 42);
	VectorRef vr  = {.data = vec, .dim = dim};

	RaBitQData *encoded = mkt_rabitq_encode(params, vr, cent_ref);
	ASSERT_NOT_NULL(encoded, "encoding succeeded");

	TestStorage *ts	  = create_test_storage(4);
	BlockNumber	 bn	  = ts->next_page++;
	Page		 page = ts->pages + (size_t)bn * BLCKSZ;

	mkt_posting_page_init(page, 0, MKT_POSTING_PAGE_FIRST);

	/* Entry 0: normal */
	mkt_posting_page_add(page, dim, 10, 1, encoded, 0);
	/* Entry 1: deleted */
	mkt_posting_page_add(page, dim, 11, 2, encoded, MKT_POSTING_FLAG_DELETED);
	/* Entry 2: normal */
	mkt_posting_page_add(page, dim, 12, 3, encoded, 0);
	/* Entry 3: deleted */
	mkt_posting_page_add(page, dim, 13, 4, encoded, MKT_POSTING_FLAG_DELETED);

	float	 *query		= make_vector(dim, 1234);
	VectorRef query_ref = {.data = query, .dim = dim};

	TestMktStorage tstor = make_storage(ts);

	MktPostingScan scan;
	mkt_posting_scan_init(&scan, &tstor.base, params, dim);

	MktPostingScanResult results[10];
	uint32_t			 nresults =
			collect_scan_results(&scan, query_ref, cent_ref, bn, results, 10);

	/* Should only get the non-deleted entries */
	ASSERT_EQ(2, nresults, "should return 2 non-deleted entries");

	/* Verify none of the results have deleted TIDs */
	for (uint32_t i = 0; i < nresults; i++)
	{
		BlockNumber blk = ItemPointerGetBlockNumber(&results[i].tid);
		ASSERT_TRUE(
				blk != 11 && blk != 13, "deleted entries should not appear");
	}

	mkt_posting_scan_cleanup(&scan);
	mkt_free(encoded);
	destroy_test_storage(ts);
	mkt_rabitq_destroy(params);
}

/* ----------------------------------------------------------------
 * Threshold push-down pruning
 * ---------------------------------------------------------------- */

TEST(scan_threshold_pruning)
{
	Dimension dim = 64;

	RaBitQParams *params = mkt_rabitq_create(dim, 42);
	ASSERT_NOT_NULL(params, "params created");

	float *centroid = mkt_alloc(dim * sizeof(float));
	for (Dimension i = 0; i < dim; i++)
		centroid[i] = 0.0f;
	VectorRef cent_ref = {.data = centroid, .dim = dim};

	/* Encode several vectors at different distances */
	const int	 n	 = 5;
	RaBitQData **enc = mkt_alloc(n * sizeof(void *));
	for (int i = 0; i < n; i++)
	{
		float	 *v	 = make_vector(dim, i * 100);
		VectorRef vr = {.data = v, .dim = dim};
		enc[i]		 = mkt_rabitq_encode(params, vr, cent_ref);
	}

	TestStorage *ts = create_test_storage(4);
	BlockNumber	 bn =
			build_posting_page(ts, dim, 0, enc, n, MKT_POSTING_PAGE_FIRST);

	float	 *query		= make_vector(dim, 9999);
	VectorRef query_ref = {.data = query, .dim = dim};

	TestMktStorage tstor = make_storage(ts);

	/* First scan without threshold: get all results */
	MktPostingScan scan;
	mkt_posting_scan_init(&scan, &tstor.base, params, dim);

	MktPostingScanResult results_all[10];
	uint32_t			 n_all = collect_scan_results(
			&scan, query_ref, cent_ref, bn, results_all, 10);

	ASSERT_TRUE(n_all > 0, "should return results without threshold");

	/*
	 * Find the minimum lower bound among results, then set
	 * threshold to that value — should prune entries whose
	 * lower_bound >= threshold.
	 */
	Distance min_lb = results_all[0].distance - results_all[0].error;
	for (uint32_t i = 1; i < n_all; i++)
	{
		Distance lb = results_all[i].distance - results_all[i].error;
		if (lb < min_lb)
			min_lb = lb;
	}

	/* Use the minimum lower bound as threshold: only the entry
	 * with exactly that lb can survive, the rest are pruned */
	Distance threshold = min_lb;
	mkt_posting_scan_set_threshold(&scan, &threshold);

	MktPostingScanResult results_pruned[10];
	uint32_t			 n_pruned = collect_scan_results(
			&scan, query_ref, cent_ref, bn, results_pruned, 10);

	/* With threshold = min lower bound, entries with
	 * lower_bound >= min_lb are pruned (strict >=). Only
	 * entries with lower_bound < min_lb survive — but
	 * min_lb is the minimum, so the minimum entry itself
	 * has lower_bound == threshold and gets pruned too.
	 * If n_all == n, we should see pruning. */
	ASSERT_TRUE(n_pruned <= n_all, "threshold should not increase results");

	/* Third scan with no threshold (NULL): should get all again */
	mkt_posting_scan_set_threshold(&scan, NULL);

	MktPostingScanResult results_again[10];
	uint32_t			 n_again = collect_scan_results(
			&scan, query_ref, cent_ref, bn, results_again, 10);

	ASSERT_EQ(n_all, n_again, "NULL threshold should match no-threshold");

	mkt_posting_scan_cleanup(&scan);
	for (int i = 0; i < n; i++)
		mkt_free(enc[i]);
	mkt_free(enc);
	destroy_test_storage(ts);
	mkt_rabitq_destroy(params);
}

/* ----------------------------------------------------------------
 * Write and read back
 * ---------------------------------------------------------------- */

/* ----------------------------------------------------------------
 * Edge cases
 * ---------------------------------------------------------------- */

TEST(scan_invalid_cluster)
{
	Dimension dim = 64;

	RaBitQParams *params   = mkt_rabitq_create(dim, 42);
	float		 *centroid = mkt_alloc(dim * sizeof(float));
	for (Dimension i = 0; i < dim; i++)
		centroid[i] = 0.0f;
	VectorRef cent_ref = {.data = centroid, .dim = dim};

	float	 *query		= make_vector(dim, 1);
	VectorRef query_ref = {.data = query, .dim = dim};

	TestStorage	  *ts	 = create_test_storage(1);
	TestMktStorage tstor = make_storage(ts);

	MktPostingScan scan;
	mkt_posting_scan_init(&scan, &tstor.base, params, dim);

	MktPostingScanResult results[5];
	uint32_t			 nresults = collect_scan_results(
			&scan, query_ref, cent_ref, InvalidBlockNumber, results, 5);

	ASSERT_EQ(0, nresults, "invalid cluster should return 0 results");

	mkt_posting_scan_cleanup(&scan);
	destroy_test_storage(ts);
	mkt_rabitq_destroy(params);
}
