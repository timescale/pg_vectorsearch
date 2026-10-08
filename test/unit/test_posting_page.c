/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
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
#include <sys/wait.h>
#include <unistd.h>

#include "core/memory.h"
#include "index/posting_build.h"
#include "index/posting_page.h"
#include "index/posting_scan.h"
#include "index/posting_split.h"
#include "index/storage.h"
#include "posting_fixtures.h"
#include "quant/rabitq.h"
#include "standalone/pg_compat.h"
#include "vs_test.h"

TEST_GROUP(PostingPage);
TEST_MEMCTX_FIXTURE();

/* ----------------------------------------------------------------
 * Struct size tests
 * ---------------------------------------------------------------- */

static void
mark_page_deleted(PrismPostingPageOpaque *op, void *state)
{
	(void)state;
	op->flags |= PRISM_POSTING_PAGE_DELETED;
}

TEST(posting_entry_meta_size)
{
	ASSERT_EQ(
			8,
			sizeof(PrismPostingEntryMeta),
			"PrismPostingEntryMeta must be 8 bytes");
}

TEST(posting_page_opaque_size)
{
	/* 24B since adding per-cluster head metadata (live_count + tail_blkno)
	 * for runtime inserts / LIRE size tracking. */
	ASSERT_EQ(
			24,
			sizeof(PrismPostingPageOpaque),
			"PrismPostingPageOpaque must be 24 bytes");
}

/* ----------------------------------------------------------------
 * Page capacity tests
 * ---------------------------------------------------------------- */

TEST(page_capacity_768d)
{
	uint32_t max = prism_posting_max_entries(768);
	/* Per entry: 8 (meta) + 12 (3 floats) + 96 (bits) = 116B
	 * Usable: 8192 - 24 (header) - 16 (opaque) = 8152B
	 * 8152 / 116 = 70 */
	ASSERT_EQ(70, max, "768d should fit 70 entries per page");
}

TEST(page_capacity_128d)
{
	uint32_t max = prism_posting_max_entries(128);
	/* Per entry: 8 + 12 + 16 = 36B
	 * 8152 / 36 = 226 */
	ASSERT_EQ(226, max, "128d should fit 226 entries per page");
}

TEST(page_capacity_1536d)
{
	uint32_t max = prism_posting_max_entries(1536);
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
	uint32_t  max = prism_posting_max_entries(dim);

	/* All regions must fit within usable space */
	size_t meta_size	  = (size_t)max * sizeof(PrismPostingEntryMeta);
	size_t f_add_size	  = (size_t)max * sizeof(float);
	size_t f_rescale_size = (size_t)max * sizeof(float);
	size_t f_error_size	  = (size_t)max * sizeof(float);
	size_t bits_size	  = (size_t)max * VS_RABITQ_BYTES(dim);

	size_t total = meta_size + f_add_size + f_rescale_size + f_error_size +
				   bits_size;

	ASSERT_TRUE(
			total <= prism_posting_page_usable(),
			"SoA arrays must fit in usable space");
}

/* ----------------------------------------------------------------
 * vector_id packing round-trip
 * ---------------------------------------------------------------- */

TEST(vector_id_round_trip_small)
{
	ItemPointerData tid;
	prism_posting_set_vector_id(&tid, 42);
	ASSERT_EQ(42, prism_posting_get_vector_id(&tid), "small vector_id");
}

TEST(vector_id_round_trip_large)
{
	ItemPointerData tid;
	uint32_t		vid = 1000000;
	prism_posting_set_vector_id(&tid, vid);
	ASSERT_EQ(vid, prism_posting_get_vector_id(&tid), "large vector_id");
}

TEST(vector_id_round_trip_max)
{
	ItemPointerData tid;
	/* Max safe value with 32-bit block + 16-bit offset packing */
	uint32_t vid = 0x00FFFFFF;
	prism_posting_set_vector_id(&tid, vid);
	ASSERT_EQ(vid, prism_posting_get_vector_id(&tid), "max vector_id");
}

/* ----------------------------------------------------------------
 * Page init and add round-trip
 * ---------------------------------------------------------------- */

TEST(page_init_basic)
{
	Page page = vs_alloc0(BLCKSZ);

	prism_posting_page_init(page, 0, 128, PRISM_POSTING_PAGE_FIRST);

	PrismPostingPageOpaque *opaque = prism_posting_opaque(page);
	ASSERT_EQ(
			InvalidBlockNumber,
			opaque->next_blkno,
			"next_blkno should be invalid");
	ASSERT_EQ(0, opaque->entry_count, "entry_count should be 0");
	ASSERT_EQ(0, opaque->cluster_id, "cluster_id should be 0");
	ASSERT_EQ(
			PRISM_POSTING_PAGE_FIRST, opaque->flags, "flags should be FIRST");
	ASSERT_EQ(
			PRISM_POSTING_PAGE_ID,
			opaque->page_id,
			"page_id should be PRISM_POSTING_PAGE_ID");
}

TEST(page_add_single_entry)
{
	Dimension dim	 = 128;
	uint32_t  packed = VS_RABITQ_BYTES(dim);

	Page page = vs_alloc0(BLCKSZ);
	prism_posting_page_init(page, 0, dim, 0);

	/* Create test data */
	ItemPointerData tid;
	prism_posting_set_vector_id(&tid, 42);
	uint8_t *bits = vs_alloc(packed);
	memset(bits, 0xAB, packed);

	bool ok =
			prism_posting_page_add(page, dim, tid, 1.0f, 2.0f, 0.5f, bits, 0);
	ASSERT_TRUE(ok, "add should succeed");
	ASSERT_EQ(1, prism_posting_page_count(page), "count should be 1");

	/* Verify round-trip via the AoS entry accessor */
	PrismPostingEntryHeader *e0 = prism_posting_entry(page, 0, dim);
	ASSERT_EQ(
			42, prism_posting_get_vector_id(&e0->meta.tid), "tid round-trip");
	ASSERT_FLOAT_EQ(1.0f, e0->f_add, 1e-6f, "f_add round-trip");
	ASSERT_FLOAT_EQ(2.0f, e0->f_rescale, 1e-6f, "f_rescale round-trip");
	ASSERT_FLOAT_EQ(0.5f, e0->f_error, 1e-6f, "f_error round-trip");
	ASSERT_EQ(
			0xAB,
			prism_posting_entry_bits(
					page, prism_posting_max_entries(dim), dim, 0)[0],
			"bits round-trip");
}

TEST(page_fill_to_capacity)
{
	Dimension dim	 = 128;
	uint32_t  max	 = prism_posting_max_entries(dim);
	uint32_t  packed = VS_RABITQ_BYTES(dim);

	Page page = vs_alloc0(BLCKSZ);
	prism_posting_page_init(page, 0, dim, 0);

	uint8_t *bits = vs_alloc(packed);
	memset(bits, 0, packed);

	/* Fill page */
	for (uint32_t i = 0; i < max; i++)
	{
		ItemPointerData tid;
		prism_posting_set_vector_id(&tid, i);
		bool ok = prism_posting_page_add(
				page, dim, tid, (float)i, 1.0f, 0.1f, bits, 0);
		ASSERT_TRUE(ok, "add should succeed");
	}
	ASSERT_EQ(max, prism_posting_page_count(page), "page should be full");
	ASSERT_FALSE(
			prism_posting_page_has_room(page), "page should have no room");

	/* One more should fail */
	ItemPointerData tid;
	prism_posting_set_vector_id(&tid, max);
	bool ok = prism_posting_page_add(page, dim, tid, 0, 0, 0, bits, 0);
	ASSERT_FALSE(ok, "add to full page should fail");
}

/* ----------------------------------------------------------------
 * Builder tests
 * ---------------------------------------------------------------- */

TEST(builder_single_page)
{
	Dimension		dim		= 128;
	TestPageStorage storage = make_test_storage(16);

	RaBitQParams *params   = vs_rabitq_create(dim, 42);
	float		 *centroid = vs_alloc0(dim * sizeof(float));

	PrismPostingBuilder builder;
	prism_posting_builder_init(
			&builder, &storage.base, params, dim, 0, centroid, centroid);

	/* Add a few vectors */
	float *vec = vs_alloc(dim * sizeof(float));
	for (uint32_t i = 0; i < 5; i++)
	{
		for (Dimension d = 0; d < dim; d++)
			vec[d] = (float)((i * 17 + d * 3) % 100 - 50) / 10.0f;
		prism_posting_builder_add(&builder, vid_to_tid(i), vec);
	}

	BlockNumber head = prism_posting_builder_finish(&builder);
	ASSERT_NEQ(InvalidBlockNumber, head, "head should be valid");

	/* Verify page contents */
	Page page = vs_storage_read_page(&storage.base, head);
	ASSERT_EQ(5, prism_posting_page_count(page), "should have 5 entries");

	/* Verify vector IDs — first page content starts after pt_centroid */
	char *content = prism_posting_content_first(page, dim);
	for (uint32_t i = 0; i < 5; i++)
	{
		PrismPostingEntryHeader *e = prism_posting_entry_at(content, i, dim);
		uint32_t vid			   = prism_posting_get_vector_id(&e->meta.tid);
		ASSERT_EQ(i, vid, "vector_id round-trip through builder");
	}

	vs_storage_release_page(&storage.base, head);
	prism_posting_builder_cleanup(&builder);
	vs_rabitq_destroy(params);
}

TEST(builder_multi_page_chain)
{
	Dimension dim		   = 768;
	uint32_t  max_per_page = prism_posting_max_entries(dim);
	/* Add enough vectors to require at least 2 pages */
	uint32_t nvecs = max_per_page + 10;

	TestPageStorage storage	 = make_test_storage(64);
	RaBitQParams   *params	 = vs_rabitq_create(dim, 42);
	float		   *centroid = vs_alloc0(dim * sizeof(float));

	PrismPostingBuilder builder;
	prism_posting_builder_init(
			&builder, &storage.base, params, dim, 0, centroid, centroid);

	float *vec = vs_alloc(dim * sizeof(float));
	for (uint32_t i = 0; i < nvecs; i++)
	{
		for (Dimension d = 0; d < dim; d++)
			vec[d] = (float)((i * 7 + d * 11) % 200 - 100) / 50.0f;
		prism_posting_builder_add(&builder, vid_to_tid(i), vec);
	}

	BlockNumber head = prism_posting_builder_finish(&builder);
	ASSERT_NEQ(InvalidBlockNumber, head, "head should be valid");

	/* Walk the chain and count entries */
	uint32_t	total_entries = 0;
	uint32_t	pages_seen	  = 0;
	BlockNumber blkno		  = head;

	while (blkno != InvalidBlockNumber)
	{
		Page	 page  = vs_storage_read_page(&storage.base, blkno);
		uint32_t count = prism_posting_page_count(page);
		total_entries += count;
		pages_seen++;
		blkno = prism_posting_opaque(page)->next_blkno;
		vs_storage_release_page(&storage.base, blkno);
	}

	ASSERT_EQ(nvecs, total_entries, "total entries across chain");
	ASSERT_TRUE(pages_seen >= 2, "chain should span multiple pages");

	prism_posting_builder_cleanup(&builder);
	vs_rabitq_destroy(params);
}

/* ----------------------------------------------------------------
 * Scan tests (fused cluster function)
 * ---------------------------------------------------------------- */

/*
 * Helper: build vectors array for reranking (mirrors what
 * standalone/index.c does — flat array indexed by vector_id).
 */
TEST(scan_processes_all_entries)
{
	Dimension		dim		 = 128;
	TestPageStorage storage	 = make_test_storage(16);
	RaBitQParams   *params	 = vs_rabitq_create(dim, 42);
	float		   *centroid = vs_alloc0(dim * sizeof(float));

	/* Build vectors and posting list */
	uint32_t nvecs = 20;
	float	*vecs  = make_test_vectors(nvecs, dim);

	BlockNumber head =
			build_cluster(&storage, params, dim, centroid, vecs, nvecs, false);

	/* Prepare query state */
	RaBitQQueryState qstate;
	setup_query_state(&qstate, params, centroid, dim);

	/* Scan with large k — all entries should be reranked */
	PrismPostingScan scan;
	prism_posting_scan_init(
			&scan,
			&storage.base,
			NULL,
			params,
			dim,
			prism_posting_max_entries(dim));

	VsTopK topk;
	vs_topk_init(&topk, nvecs);

	prism_posting_scan_begin_cluster(&scan, &qstate, head);
	prism_posting_scan_cluster(&scan, &topk);
	prism_posting_scan_end_cluster(&scan);

	ASSERT_EQ(nvecs, scan.entries_scanned, "all entries scanned");
	ASSERT_EQ(nvecs, topk.cand_count, "all entries in topk");

	vs_topk_cleanup(&topk);
	prism_posting_scan_cleanup(&scan);
	vs_rabitq_destroy(params);
}

/*
 * A chain the tree no longer points at is still scanned.
 *
 * A split rewrites a list into new chains and retires the old one, marking
 * every page DELETED but leaving it linked and intact, so a scan that captured
 * the old head before the flip still sees a complete list. The later reclaim
 * pass then ORs TOMBSTONED onto those same pages -- the flag VACUUM uses for
 * "every entry here is dead" -- and the scan used to test only that bit and
 * skip the page. The entries are not dead, though; they were rewritten
 * elsewhere, so skipping drops results such a scan is entitled to. DELETED is
 * what separates the two cases.
 */
TEST(scan_reads_a_retired_chain)
{
	Dimension		dim		 = 128;
	TestPageStorage storage	 = make_test_storage(16);
	RaBitQParams   *params	 = vs_rabitq_create(dim, 42);
	float		   *centroid = vs_alloc0(dim * sizeof(float));

	uint32_t nvecs = 20;
	float	*vecs  = make_test_vectors(nvecs, dim);

	BlockNumber head =
			build_cluster(&storage, params, dim, centroid, vecs, nvecs, false);

	/*
	 * Retire the chain the way the split's PG-side retire pass does: DELETED
	 * on every page, contents untouched. Then reclaim it, which is where
	 * TOMBSTONED joins DELETED.
	 */
	prism_posting_chain_mutate(&storage.base, head, mark_page_deleted, NULL);

	RaBitQQueryState qstate;
	setup_query_state(&qstate, params, centroid, dim);

	PrismPostingScan scan;
	prism_posting_scan_init(
			&scan,
			&storage.base,
			NULL,
			params,
			dim,
			prism_posting_max_entries(dim));

	/*
	 * Baseline first: the chain scores every entry while it is still a live
	 * one. Without it the count after reclaim could be a state the chain was
	 * in all along rather than one the flags left untouched.
	 */
	VsTopK live;
	vs_topk_init(&live, nvecs);
	prism_posting_scan_begin_cluster(&scan, &qstate, head);
	prism_posting_scan_cluster(&scan, &live);
	prism_posting_scan_end_cluster(&scan);
	ASSERT_EQ(nvecs, scan.entries_scanned, "live chain scores every entry");
	vs_topk_cleanup(&live);

	/* Now reclaim it: TOMBSTONED joins DELETED, which is the state a scan
	 * holding a stale head meets. */
	prism_posting_chain_tombstone(&storage.base, head);

	VsTopK topk;
	vs_topk_init(&topk, nvecs);

	prism_posting_scan_begin_cluster(&scan, &qstate, head);
	prism_posting_scan_cluster(&scan, &topk);
	prism_posting_scan_end_cluster(&scan);

	/* begin_cluster zeroes the per-cluster stats, so this is the retired
	 * chain's own count, not a running total. */
	ASSERT_EQ(nvecs, scan.entries_scanned, "retired chain is still scanned");
	ASSERT_EQ(nvecs, topk.cand_count, "its entries still reach the top-k");

	vs_topk_cleanup(&topk);
	prism_posting_scan_cleanup(&scan);
	vs_rabitq_destroy(params);
}

/*
 * The complement: TOMBSTONED without DELETED is VACUUM's all-dead page, and
 * skipping it is the point -- there is nothing left on it to score.
 */
TEST(scan_skips_an_all_dead_page)
{
	Dimension		dim		 = 128;
	TestPageStorage storage	 = make_test_storage(16);
	RaBitQParams   *params	 = vs_rabitq_create(dim, 42);
	float		   *centroid = vs_alloc0(dim * sizeof(float));

	uint32_t nvecs = 20;
	float	*vecs  = make_test_vectors(nvecs, dim);

	BlockNumber head =
			build_cluster(&storage, params, dim, centroid, vecs, nvecs, false);

	RaBitQQueryState qstate;
	setup_query_state(&qstate, params, centroid, dim);

	PrismPostingScan scan;
	prism_posting_scan_init(
			&scan,
			&storage.base,
			NULL,
			params,
			dim,
			prism_posting_max_entries(dim));

	/*
	 * Baseline first, or a zero at the end would be indistinguishable from a
	 * chain that had nothing scannable on it to begin with.
	 */
	VsTopK live;
	vs_topk_init(&live, nvecs);
	prism_posting_scan_begin_cluster(&scan, &qstate, head);
	prism_posting_scan_cluster(&scan, &live);
	prism_posting_scan_end_cluster(&scan);
	ASSERT_EQ(nvecs, scan.entries_scanned, "live chain scores every entry");
	vs_topk_cleanup(&live);

	/* TOMBSTONED with no DELETED: VACUUM's all-dead page. */
	prism_posting_chain_tombstone(&storage.base, head);

	VsTopK topk;
	vs_topk_init(&topk, nvecs);

	prism_posting_scan_begin_cluster(&scan, &qstate, head);
	prism_posting_scan_cluster(&scan, &topk);
	prism_posting_scan_end_cluster(&scan);

	ASSERT_EQ(0u, scan.entries_scanned, "all-dead page is skipped");
	ASSERT_EQ(0u, topk.cand_count, "and nothing reaches the top-k");

	vs_topk_cleanup(&topk);
	prism_posting_scan_cleanup(&scan);
	vs_rabitq_destroy(params);
}

TEST(scan_prunes_with_tight_topk)
{
	Dimension		dim		 = 128;
	TestPageStorage storage	 = make_test_storage(16);
	RaBitQParams   *params	 = vs_rabitq_create(dim, 42);
	float		   *centroid = vs_alloc0(dim * sizeof(float));

	/* Build vectors and posting list */
	uint32_t nvecs = 50;
	float	*vecs  = make_test_vectors(nvecs, dim);

	BlockNumber head =
			build_cluster(&storage, params, dim, centroid, vecs, nvecs, false);

	/* Prepare query state */
	RaBitQQueryState qstate;
	setup_query_state(&qstate, params, centroid, dim);

	/* Scan with small k — threshold should prune some entries */
	PrismPostingScan scan;
	prism_posting_scan_init(
			&scan,
			&storage.base,
			NULL,
			params,
			dim,
			prism_posting_max_entries(dim));

	VsTopK topk;
	vs_topk_init(&topk, 5);

	prism_posting_scan_begin_cluster(&scan, &qstate, head);
	prism_posting_scan_cluster(&scan, &topk);
	prism_posting_scan_end_cluster(&scan);

	ASSERT_EQ(nvecs, scan.entries_scanned, "all entries scanned");
	ASSERT_TRUE(scan.entries_pruned > 0, "some entries pruned");

	vs_topk_cleanup(&topk);
	prism_posting_scan_cleanup(&scan);
	vs_rabitq_destroy(params);
}

TEST(scan_stats_tracking)
{
	Dimension		dim		 = 128;
	TestPageStorage storage	 = make_test_storage(16);
	RaBitQParams   *params	 = vs_rabitq_create(dim, 42);
	float		   *centroid = vs_alloc0(dim * sizeof(float));

	/* Build vectors and posting list */
	uint32_t nvecs = 30;
	float	*vecs  = make_test_vectors(nvecs, dim);

	BlockNumber head =
			build_cluster(&storage, params, dim, centroid, vecs, nvecs, false);

	/* Prepare query state */
	RaBitQQueryState qstate;
	setup_query_state(&qstate, params, centroid, dim);

	PrismPostingScan scan;
	prism_posting_scan_init(
			&scan,
			&storage.base,
			NULL,
			params,
			dim,
			prism_posting_max_entries(dim));

	VsTopK topk;
	vs_topk_init(&topk, nvecs);

	prism_posting_scan_begin_cluster(&scan, &qstate, head);
	prism_posting_scan_cluster(&scan, &topk);
	prism_posting_scan_end_cluster(&scan);

	ASSERT_EQ(nvecs, scan.entries_scanned, "entries_scanned");
	ASSERT_TRUE(scan.pages_read >= 1, "pages_read >= 1");

	vs_topk_cleanup(&topk);
	prism_posting_scan_cleanup(&scan);
	vs_rabitq_destroy(params);
}

/* Counts the pages a chain walk visits, for the corrupt-chain test below. */
static bool
count_pages_cb(PrismPostingChainPos *pos, void *state)
{
	(void)pos;
	(*(uint32_t *)state)++;
	return true;
}

/*
 * A posting page's entry_count is read straight off disk and drives the entry
 * loops in the chain-walk callbacks, which stride by the AoS entry size. A
 * count past what this page's format can hold must be rejected before it
 * reads past the page. Note the ceiling is per format: a fastscan page packs
 * more entries than an AoS one, so bounding an AoS page by the larger figure
 * would not help. Runs in a forked child because vs_error aborts.
 */
TEST(chain_walk_rejects_corrupt_entry_count)
{
	Dimension	  dim	 = 64;
	RaBitQParams *params = vs_rabitq_create(dim, 7);
	ASSERT_NOT_NULL(params, "params created");

	TestPageStorage st = make_test_storage(32);

	float *centroid = make_test_vectors(1, dim);
	float *vecs		= make_test_vectors(8, dim);

	BlockNumber head =
			build_cluster(&st, params, dim, centroid, vecs, 8, false);
	Page p = test_read_page(&st.base, head);

	PrismPostingPageOpaque *op	= prism_posting_opaque(p);
	uint32_t				cap = prism_posting_page_cap(op, dim);

	/* Sanity: the fixture built a well-formed page. */
	ASSERT_TRUE(op->entry_count <= cap, "built page is within its ceiling");

	op->entry_count = (uint16_t)(cap + 1);

	fflush(NULL);
	pid_t pid = fork();
	ASSERT_TRUE(pid >= 0, "fork succeeded");
	if (pid == 0)
	{
		if (!vs_test_expect_abort())
			_exit(2);
		prism_posting_check_count(head, op, dim);
		_exit(0); /* reached only if the check failed to fire */
	}

	int status = 0;
	waitpid(pid, &status, 0);
	ASSERT_TRUE(
			WIFEXITED(status) && WEXITSTATUS(status) == VS_TEST_ABORTED,
			"a count past the page ceiling aborts");

	vs_free(st.pages);
	vs_free(vecs);
	vs_free(centroid);
	vs_rabitq_destroy(params);
}

/*
 * Neither format's ceiling bounds the other: fastscan packs more at dim 256
 * and fewer at dim 768. So a single AoS-derived bound is too strict at one
 * dimension and too lax at the other, which is why the count check has to
 * pick the ceiling per format.
 */
TEST(neither_format_cap_bounds_the_other)
{
	ASSERT_TRUE(
			prism_fastscan_max_entries(256) > prism_posting_max_entries(256),
			"fastscan holds more than AoS at dim 256");
	ASSERT_TRUE(
			prism_posting_max_entries(768) > prism_fastscan_max_entries(768),
			"AoS holds more than fastscan at dim 768");
}

/*
 * A first page gives up room to the encode reference, and past
 * PRISM_INDEX_MAX_DIM the reference alone fills the page. The subtraction is
 * unsigned, so without a guard it reports a capacity of millions -- and the
 * count checks take that as their bound.
 */
TEST(first_page_cap_does_not_underflow)
{
	uint32_t cap_at_limit = prism_posting_max_entries_first(
			PRISM_INDEX_MAX_DIM);

	ASSERT_TRUE(
			cap_at_limit >= 1,
			"a first page still holds an entry at the dim limit");

	for (Dimension dim = PRISM_INDEX_MAX_DIM + 1; dim <= 8192; dim += 37)
	{
		ASSERT_TRUE(
				prism_posting_max_entries_first(dim) <=
						prism_posting_max_entries(dim),
				"no first page holds more than an overflow page");
		ASSERT_TRUE(
				prism_fastscan_max_entries_first(dim) <=
						prism_fastscan_max_entries(dim),
				"same for fastscan");
	}
}

/*
 * A chain whose next_blkno leaves the posting pages is corrupt. The walk must
 * stop rather than cast a foreign page's bytes as a posting opaque and follow
 * whatever they happen to contain.
 */
TEST(chain_walk_stops_leaving_posting_pages)
{
	Dimension	  dim	 = 64;
	RaBitQParams *params = vs_rabitq_create(dim, 11);
	ASSERT_NOT_NULL(params, "params created");

	TestPageStorage st = make_test_storage(32);

	float *centroid = make_test_vectors(1, dim);
	float *vecs		= make_test_vectors(4, dim);

	BlockNumber head =
			build_cluster(&st, params, dim, centroid, vecs, 4, false);

	/* Point the head at a page that was never initialized as a posting
	 * page, as a truncated or half-written extend would leave it. */
	BlockNumber stray = 20;
	memset(test_read_page(&st.base, stray), 0, BLCKSZ);
	prism_posting_opaque(test_read_page(&st.base, head))->next_blkno = stray;

	/*
	 * In a child under an alarm: without the guard the walk reads the stray
	 * page's zeroed bytes as next_blkno and cycles forever, so a regression
	 * has to fail rather than hang the suite.
	 */
	fflush(NULL);
	pid_t pid = fork();
	ASSERT_TRUE(pid >= 0, "fork succeeded");
	if (pid == 0)
	{
		alarm(5);
		uint32_t visited = 0;
		prism_posting_chain_walk(&st.base, head, count_pages_cb, &visited);
		_exit((int)visited);
	}

	int status = 0;
	waitpid(pid, &status, 0);
	ASSERT_TRUE(
			WIFEXITED(status),
			"the walk terminated instead of following the stray page");
	ASSERT_EQ(
			1,
			WEXITSTATUS(status),
			"the walk stopped at the non-posting page");

	vs_free(st.pages);
	vs_free(vecs);
	vs_free(centroid);
	vs_rabitq_destroy(params);
}
