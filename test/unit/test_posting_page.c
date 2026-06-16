/*
 * test_posting_page.c - Unit tests for posting page format and operations
 *
 * Tests cover:
 * - Struct size verification (binary compatibility with PG)
 * - Page capacity calculation
 * - SoA region non-overlap
 * - Page init/add/access round-trip
 * - vector_id packing in ItemPointerData
 * - Page builder: single-page and multi-page chains
 * - Fused cluster scan: score + prune + rerank
 */

#include <math.h>
#include <string.h>

#include "core/memory.h"
#include "index/posting_build.h"
#include "index/posting_page.h"
#include "index/posting_scan.h"
#include "index/storage.h"
#include "mkt_test.h"
#include "quant/rabitq.h"
#include "standalone/pg_compat.h"

TEST_GROUP(PostingPage);
TEST_MEMCTX_FIXTURE();

/* Helper: create TID from vector_id (standalone encoding) */
static inline ItemPointerData
vid_to_tid(uint32_t vid)
{
	ItemPointerData tid;
	mkt_posting_set_vector_id(&tid, vid);
	return tid;
}

/* ----------------------------------------------------------------
 * Minimal ArrayPageStorage for tests
 * ---------------------------------------------------------------- */

typedef struct TestPageStorage
{
	MktStorage base;
	char	  *pages;
	uint32_t   next_blkno;
	uint32_t   page_cap;
} TestPageStorage;

static Page
test_read_page(MktStorage *self, BlockNumber blkno)
{
	TestPageStorage *s = (TestPageStorage *)self;
	return s->pages + (size_t)blkno * BLCKSZ;
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
	TestPageStorage *s = (TestPageStorage *)self;
	return s->pages + (size_t)blkno * BLCKSZ;
}

static Page
test_new_page(MktStorage *self, BlockNumber *blkno_out)
{
	TestPageStorage *s = (TestPageStorage *)self;
	*blkno_out		   = s->next_blkno++;
	return s->pages + (size_t)*blkno_out * BLCKSZ;
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
};

static TestPageStorage
make_test_storage(uint32_t num_pages)
{
	return (TestPageStorage){
			.base		= {.ops = &test_storage_ops},
			.pages		= mkt_alloc0((size_t)num_pages * BLCKSZ),
			.next_blkno = 0,
			.page_cap	= num_pages,
	};
}

/* ----------------------------------------------------------------
 * Struct size tests
 * ---------------------------------------------------------------- */

TEST(posting_entry_meta_size)
{
	ASSERT_EQ(
			8,
			sizeof(MktPostingEntryMeta),
			"MktPostingEntryMeta must be 8 bytes");
}

TEST(posting_page_opaque_size)
{
	/* 24B since adding per-cluster head metadata (live_count + tail_blkno)
	 * for runtime inserts / LIRE size tracking. */
	ASSERT_EQ(
			24,
			sizeof(MktPostingPageOpaque),
			"MktPostingPageOpaque must be 24 bytes");
}

/* ----------------------------------------------------------------
 * Page capacity tests
 * ---------------------------------------------------------------- */

TEST(page_capacity_768d)
{
	uint32_t max = mkt_posting_max_entries(768);
	/* Per entry: 8 (meta) + 12 (3 floats) + 96 (bits) = 116B
	 * Usable: 8192 - 24 (header) - 16 (opaque) = 8152B
	 * 8152 / 116 = 70 */
	ASSERT_EQ(70, max, "768d should fit 70 entries per page");
}

TEST(page_capacity_128d)
{
	uint32_t max = mkt_posting_max_entries(128);
	/* Per entry: 8 + 12 + 16 = 36B
	 * 8152 / 36 = 226 */
	ASSERT_EQ(226, max, "128d should fit 226 entries per page");
}

TEST(page_capacity_1536d)
{
	uint32_t max = mkt_posting_max_entries(1536);
	/* Per entry: 8 + 12 + 192 = 212B
	 * 8152 / 212 = 38 */
	ASSERT_EQ(38, max, "1536d should fit 38 entries per page");
}

/* ----------------------------------------------------------------
 * SoA region non-overlap test
 * ---------------------------------------------------------------- */

TEST(soa_regions_no_overlap)
{
	Dimension dim = 768;
	uint32_t  max = mkt_posting_max_entries(dim);

	/* All regions must fit within usable space */
	size_t meta_size	  = (size_t)max * sizeof(MktPostingEntryMeta);
	size_t f_add_size	  = (size_t)max * sizeof(float);
	size_t f_rescale_size = (size_t)max * sizeof(float);
	size_t f_error_size	  = (size_t)max * sizeof(float);
	size_t bits_size	  = (size_t)max * MKT_RABITQ_BYTES(dim);

	size_t total = meta_size + f_add_size + f_rescale_size + f_error_size +
				   bits_size;

	ASSERT_TRUE(
			total <= mkt_posting_page_usable(),
			"SoA arrays must fit in usable space");
}

/* ----------------------------------------------------------------
 * vector_id packing round-trip
 * ---------------------------------------------------------------- */

TEST(vector_id_round_trip_small)
{
	ItemPointerData tid;
	mkt_posting_set_vector_id(&tid, 42);
	ASSERT_EQ(42, mkt_posting_get_vector_id(&tid), "small vector_id");
}

TEST(vector_id_round_trip_large)
{
	ItemPointerData tid;
	uint32_t		vid = 1000000;
	mkt_posting_set_vector_id(&tid, vid);
	ASSERT_EQ(vid, mkt_posting_get_vector_id(&tid), "large vector_id");
}

TEST(vector_id_round_trip_max)
{
	ItemPointerData tid;
	/* Max safe value with 32-bit block + 16-bit offset packing */
	uint32_t vid = 0x00FFFFFF;
	mkt_posting_set_vector_id(&tid, vid);
	ASSERT_EQ(vid, mkt_posting_get_vector_id(&tid), "max vector_id");
}

/* ----------------------------------------------------------------
 * Page init and add round-trip
 * ---------------------------------------------------------------- */

TEST(page_init_basic)
{
	Page page = mkt_alloc0(BLCKSZ);

	mkt_posting_page_init(page, 0, 128, MKT_POSTING_PAGE_FIRST);

	MktPostingPageOpaque *opaque = mkt_posting_opaque(page);
	ASSERT_EQ(
			InvalidBlockNumber,
			opaque->next_blkno,
			"next_blkno should be invalid");
	ASSERT_EQ(0, opaque->entry_count, "entry_count should be 0");
	ASSERT_EQ(0, opaque->cluster_id, "cluster_id should be 0");
	ASSERT_EQ(MKT_POSTING_PAGE_FIRST, opaque->flags, "flags should be FIRST");
	ASSERT_EQ(
			MKT_POSTING_PAGE_ID,
			opaque->page_id,
			"page_id should be MKT_POSTING_PAGE_ID");
}

TEST(page_add_single_entry)
{
	Dimension dim	 = 128;
	uint32_t  packed = MKT_RABITQ_BYTES(dim);

	Page page = mkt_alloc0(BLCKSZ);
	mkt_posting_page_init(page, 0, dim, 0);

	/* Create test data */
	ItemPointerData tid;
	mkt_posting_set_vector_id(&tid, 42);
	uint8_t *bits = mkt_alloc(packed);
	memset(bits, 0xAB, packed);

	bool ok = mkt_posting_page_add(page, dim, tid, 1.0f, 2.0f, 0.5f, bits, 0);
	ASSERT_TRUE(ok, "add should succeed");
	ASSERT_EQ(1, mkt_posting_page_count(page), "count should be 1");

	/* Verify round-trip via the AoS entry accessor */
	MktPostingEntryHeader *e0 = mkt_posting_entry(page, 0, dim);
	ASSERT_EQ(42, mkt_posting_get_vector_id(&e0->meta.tid), "tid round-trip");
	ASSERT_FLOAT_EQ(1.0f, e0->f_add, 1e-6f, "f_add round-trip");
	ASSERT_FLOAT_EQ(2.0f, e0->f_rescale, 1e-6f, "f_rescale round-trip");
	ASSERT_FLOAT_EQ(0.5f, e0->f_error, 1e-6f, "f_error round-trip");
	ASSERT_EQ(
			0xAB,
			mkt_posting_entry_bits(
					page, mkt_posting_max_entries(dim), dim, 0)[0],
			"bits round-trip");
}

TEST(page_fill_to_capacity)
{
	Dimension dim	 = 128;
	uint32_t  max	 = mkt_posting_max_entries(dim);
	uint32_t  packed = MKT_RABITQ_BYTES(dim);

	Page page = mkt_alloc0(BLCKSZ);
	mkt_posting_page_init(page, 0, dim, 0);

	uint8_t *bits = mkt_alloc(packed);
	memset(bits, 0, packed);

	/* Fill page */
	for (uint32_t i = 0; i < max; i++)
	{
		ItemPointerData tid;
		mkt_posting_set_vector_id(&tid, i);
		bool ok = mkt_posting_page_add(
				page, dim, tid, (float)i, 1.0f, 0.1f, bits, 0);
		ASSERT_TRUE(ok, "add should succeed");
	}
	ASSERT_EQ(max, mkt_posting_page_count(page), "page should be full");
	ASSERT_FALSE(mkt_posting_page_has_room(page), "page should have no room");

	/* One more should fail */
	ItemPointerData tid;
	mkt_posting_set_vector_id(&tid, max);
	bool ok = mkt_posting_page_add(page, dim, tid, 0, 0, 0, bits, 0);
	ASSERT_FALSE(ok, "add to full page should fail");
}

/* ----------------------------------------------------------------
 * Builder tests
 * ---------------------------------------------------------------- */

TEST(builder_single_page)
{
	Dimension		dim		= 128;
	TestPageStorage storage = make_test_storage(16);

	RaBitQParams *params   = mkt_rabitq_create(dim, 42);
	float		 *centroid = mkt_alloc0(dim * sizeof(float));

	MktPostingBuilder builder;
	mkt_posting_builder_init(
			&builder, &storage.base, params, dim, 0, centroid, centroid);

	/* Add a few vectors */
	float *vec = mkt_alloc(dim * sizeof(float));
	for (uint32_t i = 0; i < 5; i++)
	{
		for (Dimension d = 0; d < dim; d++)
			vec[d] = (float)((i * 17 + d * 3) % 100 - 50) / 10.0f;
		mkt_posting_builder_add(&builder, vid_to_tid(i), vec);
	}

	BlockNumber head = mkt_posting_builder_finish(&builder);
	ASSERT_NEQ(InvalidBlockNumber, head, "head should be valid");

	/* Verify page contents */
	Page page = mkt_storage_read_page(&storage.base, head);
	ASSERT_EQ(5, mkt_posting_page_count(page), "should have 5 entries");

	/* Verify vector IDs — first page content starts after pt_centroid */
	char *content = mkt_posting_content_first(page, dim);
	for (uint32_t i = 0; i < 5; i++)
	{
		MktPostingEntryHeader *e   = mkt_posting_entry_at(content, i, dim);
		uint32_t			   vid = mkt_posting_get_vector_id(&e->meta.tid);
		ASSERT_EQ(i, vid, "vector_id round-trip through builder");
	}

	mkt_storage_release_page(&storage.base, head);
	mkt_posting_builder_cleanup(&builder);
	mkt_rabitq_destroy(params);
}

TEST(builder_multi_page_chain)
{
	Dimension dim		   = 768;
	uint32_t  max_per_page = mkt_posting_max_entries(dim);
	/* Add enough vectors to require at least 2 pages */
	uint32_t nvecs = max_per_page + 10;

	TestPageStorage storage	 = make_test_storage(64);
	RaBitQParams   *params	 = mkt_rabitq_create(dim, 42);
	float		   *centroid = mkt_alloc0(dim * sizeof(float));

	MktPostingBuilder builder;
	mkt_posting_builder_init(
			&builder, &storage.base, params, dim, 0, centroid, centroid);

	float *vec = mkt_alloc(dim * sizeof(float));
	for (uint32_t i = 0; i < nvecs; i++)
	{
		for (Dimension d = 0; d < dim; d++)
			vec[d] = (float)((i * 7 + d * 11) % 200 - 100) / 50.0f;
		mkt_posting_builder_add(&builder, vid_to_tid(i), vec);
	}

	BlockNumber head = mkt_posting_builder_finish(&builder);
	ASSERT_NEQ(InvalidBlockNumber, head, "head should be valid");

	/* Walk the chain and count entries */
	uint32_t	total_entries = 0;
	uint32_t	pages_seen	  = 0;
	BlockNumber blkno		  = head;

	while (blkno != InvalidBlockNumber)
	{
		Page	 page  = mkt_storage_read_page(&storage.base, blkno);
		uint32_t count = mkt_posting_page_count(page);
		total_entries += count;
		pages_seen++;
		blkno = mkt_posting_opaque(page)->next_blkno;
		mkt_storage_release_page(&storage.base, blkno);
	}

	ASSERT_EQ(nvecs, total_entries, "total entries across chain");
	ASSERT_TRUE(pages_seen >= 2, "chain should span multiple pages");

	mkt_posting_builder_cleanup(&builder);
	mkt_rabitq_destroy(params);
}

/* ----------------------------------------------------------------
 * Scan tests (fused cluster function)
 * ---------------------------------------------------------------- */

/*
 * Helper: build vectors array for reranking (mirrors what
 * standalone/index.c does — flat array indexed by vector_id).
 */
static float *
make_test_vectors(uint32_t nvecs, Dimension dim)
{
	float *vecs = mkt_alloc(nvecs * dim * sizeof(float));
	for (uint32_t i = 0; i < nvecs; i++)
		for (Dimension d = 0; d < dim; d++)
			vecs[(size_t)i * dim + d] = (float)((i * 13 + d * 7) % 100 - 50) /
										10.0f;
	return vecs;
}

/*
 * Helper: set up query state for scan tests.
 */
static void
setup_query_state(
		RaBitQQueryState *qstate,
		RaBitQParams	 *params,
		const float		 *centroid,
		float			 *query,
		float			 *pt_query,
		Dimension		  dim)
{
	for (Dimension d = 0; d < dim; d++)
		query[d] = (float)(d % 10) / 5.0f;

	uint32_t packed		= MKT_RABITQ_BYTES(dim);
	qstate->transformed = mkt_alloc_aligned(dim * sizeof(float), 64);
	qstate->query_bits	= mkt_alloc_aligned(packed, 64);
	mkt_rabitq_init_query_constants(qstate, dim);
	mkt_rabitq_rotate(params, query, pt_query);
	mkt_rabitq_init_query_state(
			qstate, pt_query, centroid, dim, MKT_DISTANCE_MODE_ASYMMETRIC);
}

TEST(scan_processes_all_entries)
{
	Dimension		dim		 = 128;
	TestPageStorage storage	 = make_test_storage(16);
	RaBitQParams   *params	 = mkt_rabitq_create(dim, 42);
	float		   *centroid = mkt_alloc0(dim * sizeof(float));

	/* Build vectors and posting list */
	uint32_t nvecs = 20;
	float	*vecs  = make_test_vectors(nvecs, dim);

	MktPostingBuilder builder;
	mkt_posting_builder_init(
			&builder, &storage.base, params, dim, 0, centroid, centroid);
	for (uint32_t i = 0; i < nvecs; i++)
		mkt_posting_builder_add(
				&builder, vid_to_tid(i), vecs + (size_t)i * dim);
	BlockNumber head = mkt_posting_builder_finish(&builder);

	/* Prepare query state */
	float			*query	  = mkt_alloc(dim * sizeof(float));
	float			*pt_query = mkt_alloc(dim * sizeof(float));
	RaBitQQueryState qstate;
	setup_query_state(&qstate, params, centroid, query, pt_query, dim);

	/* Scan with large k — all entries should be reranked */
	MktPostingScan scan;
	mkt_posting_scan_init(
			&scan,
			&storage.base,
			NULL,
			params,
			dim,
			mkt_posting_max_entries(dim));

	MktTopK topk;
	mkt_topk_init(&topk, nvecs);

	mkt_posting_scan_begin_cluster(&scan, &qstate, head);
	mkt_posting_scan_cluster(&scan, &topk);
	mkt_posting_scan_end_cluster(&scan);

	ASSERT_EQ(nvecs, scan.entries_scanned, "all entries scanned");
	ASSERT_EQ(nvecs, topk.cand_count, "all entries in topk");

	mkt_topk_cleanup(&topk);
	mkt_posting_scan_cleanup(&scan);
	mkt_posting_builder_cleanup(&builder);
	mkt_rabitq_destroy(params);
}

TEST(scan_prunes_with_tight_topk)
{
	Dimension		dim		 = 128;
	TestPageStorage storage	 = make_test_storage(16);
	RaBitQParams   *params	 = mkt_rabitq_create(dim, 42);
	float		   *centroid = mkt_alloc0(dim * sizeof(float));

	/* Build vectors and posting list */
	uint32_t nvecs = 50;
	float	*vecs  = make_test_vectors(nvecs, dim);

	MktPostingBuilder builder;
	mkt_posting_builder_init(
			&builder, &storage.base, params, dim, 0, centroid, centroid);
	for (uint32_t i = 0; i < nvecs; i++)
		mkt_posting_builder_add(
				&builder, vid_to_tid(i), vecs + (size_t)i * dim);
	BlockNumber head = mkt_posting_builder_finish(&builder);

	/* Prepare query state */
	float			*query	  = mkt_alloc(dim * sizeof(float));
	float			*pt_query = mkt_alloc(dim * sizeof(float));
	RaBitQQueryState qstate;
	setup_query_state(&qstate, params, centroid, query, pt_query, dim);

	/* Scan with small k — threshold should prune some entries */
	MktPostingScan scan;
	mkt_posting_scan_init(
			&scan,
			&storage.base,
			NULL,
			params,
			dim,
			mkt_posting_max_entries(dim));

	MktTopK topk;
	mkt_topk_init(&topk, 5);

	mkt_posting_scan_begin_cluster(&scan, &qstate, head);
	mkt_posting_scan_cluster(&scan, &topk);
	mkt_posting_scan_end_cluster(&scan);

	ASSERT_EQ(nvecs, scan.entries_scanned, "all entries scanned");
	ASSERT_TRUE(scan.entries_pruned > 0, "some entries pruned");

	mkt_topk_cleanup(&topk);
	mkt_posting_scan_cleanup(&scan);
	mkt_posting_builder_cleanup(&builder);
	mkt_rabitq_destroy(params);
}

TEST(scan_stats_tracking)
{
	Dimension		dim		 = 128;
	TestPageStorage storage	 = make_test_storage(16);
	RaBitQParams   *params	 = mkt_rabitq_create(dim, 42);
	float		   *centroid = mkt_alloc0(dim * sizeof(float));

	/* Build vectors and posting list */
	uint32_t nvecs = 30;
	float	*vecs  = make_test_vectors(nvecs, dim);

	MktPostingBuilder builder;
	mkt_posting_builder_init(
			&builder, &storage.base, params, dim, 0, centroid, centroid);
	for (uint32_t i = 0; i < nvecs; i++)
		mkt_posting_builder_add(
				&builder, vid_to_tid(i), vecs + (size_t)i * dim);
	BlockNumber head = mkt_posting_builder_finish(&builder);

	/* Prepare query state */
	float			*query	  = mkt_alloc(dim * sizeof(float));
	float			*pt_query = mkt_alloc(dim * sizeof(float));
	RaBitQQueryState qstate;
	setup_query_state(&qstate, params, centroid, query, pt_query, dim);

	MktPostingScan scan;
	mkt_posting_scan_init(
			&scan,
			&storage.base,
			NULL,
			params,
			dim,
			mkt_posting_max_entries(dim));

	MktTopK topk;
	mkt_topk_init(&topk, nvecs);

	mkt_posting_scan_begin_cluster(&scan, &qstate, head);
	mkt_posting_scan_cluster(&scan, &topk);
	mkt_posting_scan_end_cluster(&scan);

	ASSERT_EQ(nvecs, scan.entries_scanned, "entries_scanned");
	ASSERT_TRUE(scan.pages_read >= 1, "pages_read >= 1");

	mkt_topk_cleanup(&topk);
	mkt_posting_scan_cleanup(&scan);
	mkt_posting_builder_cleanup(&builder);
	mkt_rabitq_destroy(params);
}
