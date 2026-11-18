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
#include "index/posting_convert.h"
#include "index/posting_insert.h"
#include "index/posting_page.h"
#include "index/posting_scan.h"
#include "index/posting_split.h"
#include "index/storage.h"
#include "posting_fixtures.h"
#include "quant/rabitq.h"
#include "standalone/pg_compat.h"
#include "standalone/query.h"
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

	prism_posting_page_init(page, 0, 128, PRISM_POSTING_HEAD_START);

	PrismPostingPageOpaque *opaque = prism_posting_opaque(page);
	ASSERT_EQ(
			InvalidBlockNumber,
			opaque->next_blkno,
			"next_blkno should be invalid");
	ASSERT_EQ(0, opaque->entry_count, "entry_count should be 0");
	ASSERT_EQ(0, opaque->cluster_id, "cluster_id should be 0");
	ASSERT_EQ(
			PRISM_POSTING_HEAD_START,
			opaque->flags,
			"flags should be the head's START");
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
	char *content = prism_posting_page_content(page, dim);
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

/*
 * A spilled list (dim past the inline threshold) is head + reference pages +
 * entry pages, in that chain order. With a fixed head block the reference
 * pages sit at the blocks right after it (the bulk-built head region strides
 * by their count); the entries then go wherever the storage appends.
 */
static void
check_spilled_chain(
		VsTestResult	*result,
		TestPageStorage *st,
		BlockNumber		 head,
		Dimension		 dim,
		const float		*ref,
		uint32_t		 nentries,
		bool			 fixed)
{
	uint32_t npages = prism_posting_reference_pages(
			dim, PRISM_POSTING_REFERENCE_FLOAT32);
	BlockNumber blk	 = head;
	BlockNumber prev = InvalidBlockNumber;
	BlockNumber end	 = InvalidBlockNumber;

	/* A full page of reference on the START and MIDDLE pages, the rest on
	 * the END page where the entries start. */
	for (uint32_t c = 0; c + 1 < npages; c++)
	{
		Page						  p = vs_storage_read_page(&st->base, blk);
		const PrismPostingPageOpaque *op = prism_posting_opaque(p);
		ASSERT_EQ(
				c == 0 ? PRISM_POSTING_HEAD_START : PRISM_POSTING_HEAD_MIDDLE,
				prism_posting_head_part(op->flags),
				"the head's START page, then MIDDLE pages");
		ASSERT_TRUE(
				prism_posting_is_reference_only_page(op->flags, dim),
				"which hold only reference");
		ASSERT_EQ(0u, op->entry_count, "no entries on a reference page");
		ASSERT_EQ(0u, op->max_entries, "nor capacity");
		ASSERT_EQ(7u, op->cluster_id, "the list's cluster id");
		if (fixed && c > 0)
			ASSERT_EQ(prev + 1, blk, "reference pages follow the head");
		prev = blk;
		blk	 = op->next_blkno;
	}

	uint32_t total = 0;
	end			   = blk;
	while (blk != InvalidBlockNumber)
	{
		Page						  p = vs_storage_read_page(&st->base, blk);
		const PrismPostingPageOpaque *op = prism_posting_opaque(p);
		ASSERT_FALSE(
				prism_posting_is_reference_only_page(op->flags, dim),
				"entry pages hold entries");
		ASSERT_EQ(
				blk == end ? PRISM_POSTING_HEAD_END : PRISM_POSTING_HEAD_NONE,
				prism_posting_head_part(op->flags),
				"the first entry page is the head's END page");
		ASSERT_EQ(7u, op->cluster_id, "the list's cluster id");
		if (blk == end && fixed)
			ASSERT_EQ(prev + 1, blk, "it follows the reference pages");
		if (blk != end)
			ASSERT_TRUE(op->entry_count > 0, "a later page holds entries");
		total += op->entry_count;
		prev = blk;
		blk	 = op->next_blkno;
	}
	ASSERT_EQ(nentries, total, "every entry is on the chain");

	Page hp = vs_storage_read_page(&st->base, head);
	ASSERT_EQ(nentries, prism_posting_head_live_count(hp), "live_count");
	ASSERT_EQ(prev, prism_posting_head_tail(hp), "tail is the last page");

	float	   *got		= vs_alloc((size_t)dim * sizeof(float));
	BlockNumber entries = prism_posting_reference_read(
			&st->base, head, dim, PRISM_POSTING_REFERENCE_FLOAT32, got);
	ASSERT_MEM_EQ(ref, got, (size_t)dim * sizeof(float), "reference gathers");
	ASSERT_EQ(end, entries, "entries start on the head END page");
	if (fixed)
		ASSERT_EQ(head + npages - 1, entries, "the last reference block");
	vs_free(got);
}

static void
fill_ref(float *ref, Dimension dim)
{
	for (Dimension d = 0; d < dim; d++)
		ref[d] = (float)d * 0.25f - 3.0f;
}

TEST(builder_spilled_list_layout)
{
	Dimension	  dim	 = 4096;
	RaBitQParams *params = vs_rabitq_create(dim, 42);
	float		 *ref	 = vs_alloc((size_t)dim * sizeof(float));
	float		 *vec	 = vs_alloc((size_t)dim * sizeof(float));
	fill_ref(ref, dim);

	uint32_t sizes[] = {0, 1, 40};
	for (uint32_t t = 0; t < 3; t++)
	{
		/* Head region [5, 8) reserved for one leaf; appends start at 8. */
		TestPageStorage st;
		test_storage_init(&st, 64, 8);

		PrismPostingBuilder b;
		prism_posting_builder_init(&b, &st.base, params, dim, 7, ref, ref);
		prism_posting_builder_set_first_blkno(&b, 5);
		for (uint32_t i = 0; i < sizes[t]; i++)
		{
			for (Dimension d = 0; d < dim; d++)
				vec[d] = ref[d] + (float)((i * 7 + d) % 13) - 6.0f;
			prism_posting_builder_add(&b, vid_to_tid(i), vec);
		}
		BlockNumber head = prism_posting_builder_finish(&b);
		prism_posting_builder_cleanup(&b);
		ASSERT_EQ(5u, head, "the head is the fixed block");
		check_spilled_chain(result, &st, head, dim, ref, sizes[t], true);
		vs_free(st.pages);
	}

	/* Without a fixed head: everything is appended, chain order still. */
	{
		TestPageStorage st = make_test_storage(64);

		PrismPostingBuilder b;
		prism_posting_builder_init(&b, &st.base, params, dim, 7, ref, ref);
		for (uint32_t i = 0; i < 20; i++)
		{
			for (Dimension d = 0; d < dim; d++)
				vec[d] = ref[d] + (float)((i * 5 + d) % 11) - 5.0f;
			prism_posting_builder_add(&b, vid_to_tid(i), vec);
		}
		BlockNumber head = prism_posting_builder_finish(&b);
		prism_posting_builder_cleanup(&b);
		check_spilled_chain(result, &st, head, dim, ref, 20, false);
		vs_free(st.pages);
	}

	vs_free(ref);
	vs_free(vec);
	vs_rabitq_destroy(params);
}

/*
 * The parallel bulk build pre-writes every head (and, spilled, its reference
 * pages) and later adopts it to append the sorted entries. A spilled adoption
 * leaves the head and middle reference pages untouched and appends to the
 * head END page; an adoption that appends nothing leaves the head's
 * stamp alone.
 */
TEST(builder_adopts_a_spilled_head)
{
	Dimension	  dim	 = 4096;
	RaBitQParams *params = vs_rabitq_create(dim, 42);
	float		 *ref	 = vs_alloc((size_t)dim * sizeof(float));
	float		 *vec	 = vs_alloc((size_t)dim * sizeof(float));
	RaBitQData	 *rd	 = vs_alloc(VS_RABITQ_DATA_SIZE(dim));
	fill_ref(ref, dim);

	TestPageStorage st;
	test_storage_init(&st, 64, 8);

	/* The pre-scan writer: head + reference pages, no entries. */
	PrismPostingBuilder b;
	prism_posting_builder_init(&b, &st.base, params, dim, 7, ref, ref);
	prism_posting_builder_set_first_blkno(&b, 5);
	ASSERT_EQ(5u, prism_posting_builder_finish(&b), "pre-written at 5");
	prism_posting_builder_cleanup(&b);
	check_spilled_chain(result, &st, 5, dim, ref, 0, true);

	/* Adopt and append nothing: byte-identical head. */
	char before[BLCKSZ];
	memcpy(before, vs_storage_read_page(&st.base, 5), BLCKSZ);
	prism_posting_builder_adopt_head(&b, &st.base, params, dim, 7, 5, false);
	ASSERT_EQ(5u, prism_posting_builder_finish(&b), "same head");
	prism_posting_builder_cleanup(&b);
	ASSERT_MEM_EQ(
			before,
			vs_storage_read_page(&st.base, 5),
			BLCKSZ,
			"an empty adoption rewrites nothing");
	ASSERT_EQ(8u, st.next_blkno, "and appends nothing");

	/* Adopt and append 31 encoded entries: 15 on the head END page,
	 * which is rewritten in place, then 15 and 1 on two appended pages. */
	prism_posting_builder_adopt_head(&b, &st.base, params, dim, 7, 5, false);
	Vec32Ref cref = {.data = ref, .dim = dim};
	for (uint32_t i = 0; i < 31; i++)
	{
		for (Dimension d = 0; d < dim; d++)
			vec[d] = ref[d] + (float)((i * 3 + d) % 17) - 8.0f;
		Vec32Ref vref = {.data = vec, .dim = dim};
		vs_rabitq_encode_into(params, vref, cref, rd);
		prism_posting_builder_add_encoded(
				&b,
				vid_to_tid(i),
				rd->f_add,
				rd->f_rescale,
				vs_rabitq_derive_f_error(rd->f_add, rd->f_rescale, dim),
				rd->bits);
	}
	ASSERT_EQ(5u, prism_posting_builder_finish(&b), "same head");
	prism_posting_builder_cleanup(&b);
	check_spilled_chain(result, &st, 5, dim, ref, 31, true);
	ASSERT_EQ(10u, st.next_blkno, "two pages appended after the chain");

	vs_free(st.pages);
	vs_free(rd);
	vs_free(ref);
	vs_free(vec);
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
			PRISM_POSTING_REFERENCE_FLOAT32,
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
 * Scanning a spilled cluster: the reference is gathered from the chain, the
 * entries are scored from the pages after it, and a reference the caller
 * gathered beforehand (the probe re-rank's) is taken as-is with no reference
 * page read again.
 */
TEST(scan_spilled_cluster)
{
	Dimension		dim		 = 4096;
	TestPageStorage storage	 = make_test_storage(16);
	RaBitQParams   *params	 = vs_rabitq_create(dim, 42);
	float		   *centroid = vs_alloc(dim * sizeof(float));
	uint32_t		nvecs	 = 40; /* 15 per page, from the head END page */
	float		   *vecs	 = make_test_vectors(nvecs, dim);

	for (Dimension d = 0; d < dim; d++)
		centroid[d] = (float)(d % 7) * 0.1f;

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
			PRISM_POSTING_REFERENCE_FLOAT32,
			prism_posting_max_entries(dim));
	ASSERT_NOT_NULL(scan.pt_ref, "the scan has a spilled-reference buffer");

	VsTopK topk;
	vs_topk_init(&topk, nvecs);

	prism_posting_scan_begin_cluster(&scan, &qstate, head);
	const float *ref = prism_posting_scan_pt_centroid(&scan);
	ASSERT_NOT_NULL(ref, "reference known after begin_cluster");
	ASSERT_MEM_EQ(
			centroid,
			ref,
			dim * sizeof(float),
			"the gathered reference is the one the builder wrote");
	prism_posting_scan_cluster(&scan, &topk);
	prism_posting_scan_end_cluster(&scan);

	ASSERT_EQ(nvecs, scan.entries_scanned, "all entries scanned");
	ASSERT_EQ(nvecs, topk.cand_count, "all entries in topk");
	ASSERT_EQ(
			prism_posting_reference_pages(
					dim, PRISM_POSTING_REFERENCE_FLOAT32) +
					2,
			scan.pages_read,
			"reference pages, the last with entries, then two more");

	/* Pre-gathered: same entries, the reference pages still counted. */
	float	   *mine	= vs_alloc(dim * sizeof(float));
	BlockNumber entries = prism_posting_reference_read(
			&storage.base, head, dim, PRISM_POSTING_REFERENCE_FLOAT32, mine);
	ASSERT_MEM_EQ(centroid, mine, dim * sizeof(float), "gather matches");
	ASSERT_EQ(
			head +
					prism_posting_reference_pages(
							dim, PRISM_POSTING_REFERENCE_FLOAT32) -
					1,
			entries,
			"entries start on the head END page");

	vs_topk_cleanup(&topk);
	vs_topk_init(&topk, nvecs);
	prism_posting_scan_begin_cluster_at(&scan, &qstate, entries, mine);
	ASSERT_PTR_EQ(
			mine,
			prism_posting_scan_pt_centroid(&scan),
			"the caller's buffer is used as-is");
	prism_posting_scan_cluster(&scan, &topk);
	prism_posting_scan_end_cluster(&scan);
	ASSERT_EQ(nvecs, topk.cand_count, "all entries in topk again");
	ASSERT_EQ(
			prism_posting_reference_pages(
					dim, PRISM_POSTING_REFERENCE_FLOAT32) +
					2,
			scan.pages_read,
			"the same page count as a self-gathering begin");

	/* An empty spilled cluster still reports its reference. */
	BlockNumber empty =
			build_cluster(&storage, params, dim, centroid, vecs, 0, false);
	prism_posting_scan_begin_cluster(&scan, &qstate, empty);
	ASSERT_NOT_NULL(
			prism_posting_scan_pt_centroid(&scan),
			"an empty list has a reference");
	vs_topk_cleanup(&topk);
	vs_topk_init(&topk, nvecs);
	prism_posting_scan_cluster(&scan, &topk);
	prism_posting_scan_end_cluster(&scan);
	ASSERT_EQ(0u, topk.cand_count, "and no entries");

	vs_topk_cleanup(&topk);
	prism_posting_scan_cleanup(&scan);
	vs_rabitq_destroy(params);
}

/*
 * A builder with a float16 reference rounds it once, marks every head page
 * REF16 and no later page, and encodes raw vectors against the rounded
 * reference the pages store, so the scan scores them against what it reads.
 * 2048 and 3072 dimensions are inline at float16, 4096 spills.
 */
static void
check_ref16_builder(VsTestResult *result, Dimension dim, bool fastscan)
{
	RaBitQParams   *params = vs_rabitq_create(dim, 42);
	TestPageStorage st	   = make_test_storage(64);
	uint32_t		nvecs  = 70;
	float		   *vecs   = make_test_vectors(nvecs, dim);
	float		   *c	   = vs_alloc((size_t)dim * sizeof(float));
	float		   *pt	   = vs_alloc((size_t)dim * sizeof(float));
	float		   *want   = vs_alloc((size_t)dim * sizeof(float));
	float		   *got	   = vs_alloc((size_t)dim * sizeof(float));

	for (Dimension d = 0; d < dim; d++)
		c[d] = (float)(d % 7) * 0.1f + 0.013f;
	vs_rabitq_rotate(params, c, pt);
	memcpy(want, pt, (size_t)dim * sizeof(float));
	prism_posting_reference_round16(want, dim);

	PrismPostingBuilder b;
	prism_posting_builder_init_ref(
			&b,
			&st.base,
			params,
			dim,
			0,
			c,
			pt,
			fastscan,
			PRISM_POSTING_REFERENCE_FLOAT16);
	for (uint32_t i = 0; i < nvecs; i++)
		prism_posting_builder_add(&b, vid_to_tid(i), vecs + (size_t)i * dim);
	BlockNumber head = prism_posting_builder_finish(&b);
	prism_posting_builder_cleanup(&b);

	BlockNumber entries = prism_posting_reference_read(
			&st.base, head, dim, PRISM_POSTING_REFERENCE_FLOAT16, got);
	ASSERT_MEM_EQ(
			want, got, (size_t)dim * sizeof(float), "the rounded reference");
	ASSERT_EQ(
			prism_posting_spilled(dim, PRISM_POSTING_REFERENCE_FLOAT16)
					? head + 1
					: head,
			entries,
			"entries start on the last head page");

	/* REF16 on the head pages, not on the pages after them. */
	uint32_t npages = 0;
	uint32_t nhead	= prism_posting_reference_pages(
			 dim, PRISM_POSTING_REFERENCE_FLOAT16);
	BlockNumber blk = head;
	while (blk != InvalidBlockNumber)
	{
		Page		page = vs_storage_read_page(&st.base, blk);
		uint16_t	f	 = prism_posting_opaque(page)->flags;
		BlockNumber next = prism_posting_opaque(page)->next_blkno;

		ASSERT_EQ(
				npages < nhead,
				prism_posting_flags_ref_format(f) ==
						PRISM_POSTING_REFERENCE_FLOAT16,
				"REF16 exactly on the head pages");
		vs_storage_release_page(&st.base, blk);
		blk = next;
		npages++;
	}
	ASSERT_TRUE(npages > nhead, "the list continues past its head");

	/* The first entry was encoded against the rounded reference. */
	if (!fastscan)
	{
		RaBitQData	 *enc = vs_alloc(VS_RABITQ_DATA_SIZE(dim));
		RaBitQScratch scratch;
		float		 *res = vs_alloc((size_t)dim * sizeof(float));

		vs_rabitq_scratch_init(&scratch, dim);
		vs_rabitq_rotate(params, vecs, res);
		for (Dimension d = 0; d < dim; d++)
			res[d] -= want[d];
		vs_rabitq_encode_from_pt(params, res, enc, &scratch);

		Page page = vs_storage_read_page(&st.base, entries);
		const PrismPostingEntryHeader *e = prism_posting_entry_at(
				prism_posting_page_content(page, dim), 0, dim);
		ASSERT_MEM_EQ(
				enc->bits,
				e->bits,
				VS_RABITQ_BYTES(dim),
				"the code is relative to the rounded reference");
		ASSERT_TRUE(e->f_add == enc->f_add, "and so is f_add");
		vs_storage_release_page(&st.base, entries);
		vs_rabitq_scratch_cleanup(&scratch);
	}

	/* The scan widens the reference and reaches every entry. */
	RaBitQQueryState qstate;
	setup_query_state(&qstate, params, want, dim);
	PrismPostingScan scan;
	prism_posting_scan_init(
			&scan,
			&st.base,
			NULL,
			params,
			dim,
			PRISM_POSTING_REFERENCE_FLOAT16,
			prism_posting_max_entries(dim));
	if (fastscan)
		prism_posting_scan_enable_fastscan(&scan, 16);
	VsTopK topk;
	vs_topk_init(&topk, nvecs);
	prism_posting_scan_begin_cluster(&scan, &qstate, head);
	const float *ref = prism_posting_scan_pt_centroid(&scan);
	ASSERT_NOT_NULL(ref, "the scan has the reference");
	ASSERT_MEM_EQ(
			want, ref, (size_t)dim * sizeof(float), "widened from float16");
	prism_posting_scan_cluster(&scan, &topk);
	prism_posting_scan_end_cluster(&scan);
	ASSERT_EQ(nvecs, scan.entries_scanned, "every entry scanned");

	vs_topk_cleanup(&topk);
	prism_posting_scan_cleanup(&scan);
	vs_free(st.pages);
	vs_rabitq_destroy(params);
}

/* Arguments for beginning a scan in a child that may abort. */
typedef struct ScanBeginArgs
{
	VsStorage				   *storage;
	RaBitQParams			   *params;
	Dimension					dim;
	PrismPostingReferenceFormat ref_format;
	BlockNumber					head;
} ScanBeginArgs;

static void
run_scan_begin(void *arg)
{
	const ScanBeginArgs *a = arg;
	RaBitQQueryState	 qstate;
	PrismPostingScan	 scan;

	memset(&qstate, 0, sizeof(qstate));
	prism_posting_scan_init(
			&scan,
			a->storage,
			NULL,
			a->params,
			a->dim,
			a->ref_format,
			prism_posting_max_entries(a->dim));
	prism_posting_scan_begin_cluster(&scan, &qstate, a->head);
	prism_posting_scan_end_cluster(&scan);
	prism_posting_scan_cleanup(&scan);
}

/*
 * An inline head page whose reference format is not the index's is refused
 * before its reference is read. A float16 list at 3072 dimensions whose head
 * lost its REF16 bit would otherwise be read as 12 KB of float32 from an
 * 8 KB page; a float32 list with the bit set as half its reference.
 */
TEST(scan_rejects_a_head_of_the_other_width)
{
	struct
	{
		Dimension					dim;
		PrismPostingReferenceFormat ref_format;
	} cases[] = {
			{3072, PRISM_POSTING_REFERENCE_FLOAT16},
			{768, PRISM_POSTING_REFERENCE_FLOAT32},
	};

	for (uint32_t k = 0; k < 2; k++)
	{
		Dimension		dim	   = cases[k].dim;
		RaBitQParams   *params = vs_rabitq_create(dim, 42);
		TestPageStorage st	   = make_test_storage(4);
		float		   *pt	   = vs_alloc0((size_t)dim * sizeof(float));

		PrismPostingBuilder b;
		prism_posting_builder_init_ref(
				&b,
				&st.base,
				params,
				dim,
				0,
				pt,
				pt,
				false,
				cases[k].ref_format);
		BlockNumber head = prism_posting_builder_finish(&b);
		prism_posting_builder_cleanup(&b);

		ScanBeginArgs args = {
				.storage	= &st.base,
				.params		= params,
				.dim		= dim,
				.ref_format = cases[k].ref_format,
				.head		= head,
		};
		ASSERT_FALSE(
				vs_test_aborts(run_scan_begin, &args, NULL, 0),
				"the head as built is accepted");

		prism_posting_opaque(vs_storage_write_page(&st.base, head))->flags ^=
				PRISM_POSTING_PAGE_REF16;
		ASSERT_TRUE(
				vs_test_aborts(run_scan_begin, &args, NULL, 0),
				"a head of the other width is refused");

		vs_free(pt);
		vs_free(st.pages);
		vs_rabitq_destroy(params);
	}
}

TEST(ref16_builder_encodes_against_the_rounded_reference)
{
	Dimension dims[] = {2048, 3072, 4096};

	for (uint32_t i = 0; i < sizeof(dims) / sizeof(dims[0]); i++)
	{
		check_ref16_builder(result, dims[i], false);
		check_ref16_builder(result, dims[i], true);
	}
}

/*
 * Float16 halves the reference, so a list spills only past 3820
 * dimensions, and a spilled one needs half the head pages. A spilled list
 * keeps its START and END pages even where the reference fits on one.
 */
TEST(ref16_layout)
{
	ASSERT_EQ(
			2u * (uint32_t)PRISM_POSTING_REFERENCE_PAGE_FLOATS,
			prism_posting_reference_page_capacity(
					PRISM_POSTING_REFERENCE_FLOAT16),
			"a page holds twice as many halves");
	ASSERT_FALSE(
			prism_posting_spilled(3072, PRISM_POSTING_REFERENCE_FLOAT16),
			"3072 is inline");
	ASSERT_FALSE(
			prism_posting_spilled(3820, PRISM_POSTING_REFERENCE_FLOAT16),
			"3820 is inline");
	ASSERT_TRUE(
			prism_posting_spilled(3821, PRISM_POSTING_REFERENCE_FLOAT16),
			"3821 spills");
	ASSERT_TRUE(
			prism_posting_max_entries_first(
					3820, PRISM_POSTING_REFERENCE_FLOAT16) >= 1,
			"an inline head still holds an entry at the threshold");
	ASSERT_EQ(
			(prism_posting_page_usable() - 3072 * 2) /
					PRISM_POSTING_ENTRY_SIZE(3072),
			prism_posting_max_entries_first(
					3072, PRISM_POSTING_REFERENCE_FLOAT16),
			"3072: what fits after 6 KiB of reference");

	ASSERT_EQ(
			2u,
			prism_posting_reference_pages(
					3821, PRISM_POSTING_REFERENCE_FLOAT16),
			"START + END");
	ASSERT_EQ(
			3821u,
			prism_posting_reference_page_elems(
					3821, 0, PRISM_POSTING_REFERENCE_FLOAT16),
			"all of it on START");
	ASSERT_EQ(
			0u,
			prism_posting_ref_prefix_size(
					3821, PRISM_POSTING_REFERENCE_FLOAT16),
			"none on END");
	ASSERT_EQ(
			2u,
			prism_posting_reference_pages(
					4096, PRISM_POSTING_REFERENCE_FLOAT16),
			"4096: two");
	ASSERT_EQ(
			24u,
			prism_posting_reference_page_elems(
					4096, 1, PRISM_POSTING_REFERENCE_FLOAT16),
			"24 halves on END");
	ASSERT_EQ(
			3u,
			prism_posting_reference_pages(
					8192, PRISM_POSTING_REFERENCE_FLOAT16),
			"8192: three");

	for (Dimension dim = 3821; dim <= PRISM_INDEX_MAX_DIM; dim += 37)
	{
		uint32_t npages = prism_posting_reference_pages(
				dim, PRISM_POSTING_REFERENCE_FLOAT16);
		uint32_t total = 0;

		ASSERT_TRUE(npages >= 2, "a spilled list has START and END");
		for (uint32_t i = 0; i < npages; i++)
			total += prism_posting_reference_page_elems(
					dim, i, PRISM_POSTING_REFERENCE_FLOAT16);
		ASSERT_EQ(dim, total, "the head pages cover the vector exactly");
	}
	for (Dimension dim = 3821; dim <= PRISM_INDEX_MAX_DIM; dim++)
		if (prism_index_dim_supported(dim))
			ASSERT_TRUE(
					prism_posting_max_entries_first(
							dim, PRISM_POSTING_REFERENCE_FLOAT16) >= 1,
					"the END page holds an entry at every supported dim");
}

/*
 * The build gate: inline dimensions always, spilled ones only where the
 * fast rotation applies, nothing past the ceiling.
 */
TEST(dimension_gate)
{
	ASSERT_TRUE(prism_index_dim_supported(1), "1");
	ASSERT_TRUE(prism_index_dim_supported(1000), "1000 (inline, dense ok)");
	ASSERT_TRUE(prism_index_dim_supported(1968), "the inline threshold");
	ASSERT_FALSE(prism_index_dim_supported(1969), "1969 = 11 * 179");
	ASSERT_TRUE(prism_index_dim_supported(2048), "2048");
	ASSERT_FALSE(prism_index_dim_supported(3000), "3000 = 8 * 375");
	ASSERT_TRUE(prism_index_dim_supported(3072), "3072 = 1024 * 3");
	ASSERT_TRUE(prism_index_dim_supported(4096), "4096");
	ASSERT_FALSE(prism_index_dim_supported(4097), "4097 is odd");
	ASSERT_TRUE(prism_index_dim_supported(6144), "6144 = 2048 * 3");
	ASSERT_TRUE(prism_index_dim_supported(8192), "the ceiling");

	/* Every spilled dimension the gate admits has a START and an END
	 * page: none fits its reference on one page. */
	for (uint32_t dim = 1; dim <= PRISM_INDEX_MAX_DIM; dim++)
		if (prism_index_dim_supported(dim) &&
			prism_posting_spilled(
					(Dimension)dim, PRISM_POSTING_REFERENCE_FLOAT32))
			ASSERT_TRUE(
					prism_posting_reference_pages(
							(Dimension)dim, PRISM_POSTING_REFERENCE_FLOAT32) >=
							2,
					"a spilled head has START and END pages");
	ASSERT_FALSE(prism_index_dim_supported(8193), "past the ceiling");
	ASSERT_FALSE(prism_index_dim_supported(0), "zero");
	ASSERT_FALSE(prism_index_dim_supported(65568), "65568, 32 as a Dimension");
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
			PRISM_POSTING_REFERENCE_FLOAT32,
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
			PRISM_POSTING_REFERENCE_FLOAT32,
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
			PRISM_POSTING_REFERENCE_FLOAT32,
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
			PRISM_POSTING_REFERENCE_FLOAT32,
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
typedef struct CheckCountArgs
{
	BlockNumber blkno;
	Page		page;
	Dimension	dim;
} CheckCountArgs;

static void
run_check_count(void *arg)
{
	const CheckCountArgs *a = arg;

	prism_posting_check_count(a->blkno, a->page, a->dim);
}

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

	CheckCountArgs args = {.blkno = head, .page = p, .dim = dim};
	ASSERT_TRUE(
			vs_test_aborts(run_check_count, &args, NULL, 0),
			"a count past the page ceiling aborts");

	vs_free(st.pages);
	vs_free(vecs);
	vs_free(centroid);
	vs_rabitq_destroy(params);
}

/*
 * The fastscan ceiling allows for group metadata deferred from the previous
 * page, so it sits above the AoS one. An AoS-derived bound would reject a
 * valid fastscan page, which is why the count check picks the ceiling per
 * format, and why buffers sized for any page take the larger.
 */
TEST(page_caps_differ_by_format)
{
	for (Dimension dim = 4; dim <= PRISM_INDEX_MAX_DIM; dim += 4)
	{
		uint32_t aos = prism_posting_max_entries(dim);
		uint32_t fs	 = prism_fastscan_max_entries(dim);

		ASSERT_TRUE(fs > aos, "fastscan ceiling exceeds AoS");
		ASSERT_EQ(
				fs,
				prism_posting_max_entries_any_format(dim),
				"buffers for any page take the larger ceiling");
	}
}

/*
 * A first page gives up room to the encode reference; once that leaves no
 * room for an entry the layout spills and the head holds no entries at all.
 * The inline threshold is pinned to the number every pre-spill index was
 * built under, and the capacities are guarded rather than subtracted
 * blindly -- past the threshold an unsigned subtraction would report a
 * capacity of millions, which the count checks take as their bound and
 * fastscan=auto as a fit.
 */
TEST(spill_threshold_pins_the_inline_layout)
{
	ASSERT_FALSE(
			prism_posting_spilled(
					PRISM_INDEX_NARROW_MAX_DIM,
					PRISM_POSTING_REFERENCE_FLOAT32),
			"the inline threshold itself is inline");
	ASSERT_TRUE(
			prism_posting_spilled(
					PRISM_INDEX_NARROW_MAX_DIM + 1,
					PRISM_POSTING_REFERENCE_FLOAT32),
			"one more dimension spills");
	ASSERT_TRUE(
			prism_posting_max_entries_first(
					PRISM_INDEX_NARROW_MAX_DIM,
					PRISM_POSTING_REFERENCE_FLOAT32) >= 1,
			"an inline head still holds an entry at the threshold");
	ASSERT_EQ(
			1u,
			prism_posting_reference_pages(
					PRISM_INDEX_NARROW_MAX_DIM,
					PRISM_POSTING_REFERENCE_FLOAT32),
			"an inline list has its reference on the head alone");

	/* Reference pages: ceil(dim / floats-per-page), 2036 at 8 KB pages. */
	ASSERT_EQ(
			2036u,
			(uint32_t)PRISM_POSTING_REFERENCE_PAGE_FLOATS,
			"floats per page");
	ASSERT_EQ(
			2u,
			prism_posting_reference_pages(
					2037, PRISM_POSTING_REFERENCE_FLOAT32),
			"2037 needs two");
	ASSERT_EQ(
			2u,
			prism_posting_reference_pages(
					4072, PRISM_POSTING_REFERENCE_FLOAT32),
			"4072 fills two");
	ASSERT_EQ(
			3u,
			prism_posting_reference_pages(
					4073, PRISM_POSTING_REFERENCE_FLOAT32),
			"4073 needs three");
	ASSERT_EQ(
			3u,
			prism_posting_reference_pages(
					4096, PRISM_POSTING_REFERENCE_FLOAT32),
			"4096 needs three");
	ASSERT_EQ(
			5u,
			prism_posting_reference_pages(
					8192, PRISM_POSTING_REFERENCE_FLOAT32),
			"8192 needs five");

	for (Dimension dim = PRISM_POSTING_REFERENCE_PAGE_FLOATS + 1;
		 dim <= PRISM_INDEX_MAX_DIM;
		 dim += 37)
	{
		uint32_t npages = prism_posting_reference_pages(
				dim, PRISM_POSTING_REFERENCE_FLOAT32);
		uint32_t total = 0;
		for (uint32_t i = 0; i < npages; i++)
		{
			uint32_t n = prism_posting_reference_page_elems(
					dim, i, PRISM_POSTING_REFERENCE_FLOAT32);
			ASSERT_TRUE(n >= 1, "every head page carries reference");
			total += n;
		}
		ASSERT_EQ(dim, total, "the head pages cover the vector exactly");
		ASSERT_EQ(
				0u,
				prism_posting_reference_page_elems(
						dim, npages, PRISM_POSTING_REFERENCE_FLOAT32),
				"nothing past the END page");

		/* The START and MIDDLE pages hold only reference; the entries
		 * start after the rest of it, on the END page. */
		uint32_t prefix = (uint32_t)MAXALIGN(
				prism_posting_reference_page_elems(
						dim, npages - 1, PRISM_POSTING_REFERENCE_FLOAT32) *
				sizeof(float));
		PrismPostingPageOpaque head_op = {.flags = PRISM_POSTING_HEAD_START};
		PrismPostingPageOpaque end_op  = {.flags = PRISM_POSTING_HEAD_END};

		ASSERT_EQ(
				prefix,
				prism_posting_ref_prefix_size(
						dim, PRISM_POSTING_REFERENCE_FLOAT32),
				"prefix");
		ASSERT_EQ(
				0u, prism_posting_page_cap(&head_op, dim), "head holds none");
		ASSERT_EQ(
				(prism_posting_page_usable() - prefix) /
						PRISM_POSTING_ENTRY_SIZE(dim),
				prism_posting_page_cap(&end_op, dim),
				"the END page holds what fits after the reference");
		ASSERT_EQ(
				prism_posting_page_usable() - prefix,
				prism_fastscan_page_space(
						dim, true, PRISM_POSTING_REFERENCE_FLOAT32),
				"and as much fastscan stream");
		ASSERT_TRUE(
				prism_posting_max_entries(dim) >= 1,
				"a later page holds an entry up to the ceiling");
		ASSERT_TRUE(
				prism_fastscan_max_groups(
						dim, false, PRISM_POSTING_REFERENCE_FLOAT32) *
								VS_FASTSCAN_GROUP ==
						prism_fastscan_max_entries(dim),
				"fastscan capacity is whole groups");
	}

	/* At 4096 the END page holds 24 floats of reference, so nearly the
	 * whole page is left for entries. */
	ASSERT_EQ(
			96u,
			prism_posting_ref_prefix_size(
					4096, PRISM_POSTING_REFERENCE_FLOAT32),
			"96-byte prefix");
	ASSERT_EQ(
			15u,
			prism_posting_max_entries_first(
					4096, PRISM_POSTING_REFERENCE_FLOAT32),
			"15 AoS entries");
	ASSERT_EQ(15u, prism_posting_max_entries(4096), "as on a later page");
	ASSERT_EQ(
			prism_posting_page_usable() - 96,
			prism_fastscan_page_space(
					4096, true, PRISM_POSTING_REFERENCE_FLOAT32),
			"the stream starts on the END page");

	PrismPostingPageOpaque op = {.flags = PRISM_POSTING_HEAD_START};
	ASSERT_EQ(0u, prism_posting_page_cap(&op, 4096), "a spilled START: 0");
	ASSERT_TRUE(
			prism_posting_page_cap(&op, 768) > 0,
			"an inline START holds entries");
	op.flags = PRISM_POSTING_HEAD_MIDDLE | PRISM_POSTING_PAGE_FASTSCAN;
	ASSERT_EQ(0u, prism_posting_page_cap(&op, 4096), "a MIDDLE page: 0");
}

/*
 * Head addressing: leaf c's head is first_posting + c * stride, and the
 * route back divides. Stride 1 is the inline layout; a spilled layout
 * strides by its reference page count so the head pages are contiguous.
 */
TEST(head_stride_round_trips)
{
	uint32_t strides[] =
			{1,
			 prism_posting_reference_pages(
					 4096, PRISM_POSTING_REFERENCE_FLOAT32)};
	for (uint32_t s = 0; s < 2; s++)
	{
		for (uint32_t leaf = 0; leaf < 100; leaf += 7)
		{
			BlockNumber head = prism_leaf_head_blkno(17, leaf, strides[s]);
			ASSERT_EQ(
					leaf,
					prism_route_head_to_leaf(17, head, strides[s]),
					"head -> leaf inverts leaf -> head");
		}
	}
	ASSERT_EQ(
			3u,
			prism_posting_reference_pages(
					4096, PRISM_POSTING_REFERENCE_FLOAT32),
			"4096 strides by 3");
}

/*
 * Write a reference as pages and gather it back. Inline: the head carries
 * it and the entries start on the head. Spilled: the head pages are read in
 * order and the entries start on the END page.
 */
static void
check_reference_roundtrip(
		VsTestResult			   *result,
		Dimension					dim,
		PrismPostingReferenceFormat ref_format)
{
	uint32_t		npages = prism_posting_reference_pages(dim, ref_format);
	TestPageStorage st	   = make_test_storage(npages + 2);
	float		   *ref	   = vs_alloc((size_t)dim * sizeof(float));
	float		   *got	   = vs_alloc((size_t)dim * sizeof(float));

	float	*want  = vs_alloc((size_t)dim * sizeof(float));
	uint16_t flags = prism_posting_ref_format_flag(ref_format);

	for (Dimension d = 0; d < dim; d++)
		ref[d] = (float)d * 0.5f - 7.0f;
	memcpy(want, ref, (size_t)dim * sizeof(float));
	if (ref_format == PRISM_POSTING_REFERENCE_FLOAT16)
		prism_posting_reference_round16(want, dim);

	BlockNumber blk;
	Page		page = vs_storage_new_page(&st.base, &blk);
	BlockNumber head = blk;
	if (!prism_posting_spilled(dim, ref_format))
	{
		prism_posting_reference_page_fill(page, 42, dim, 0, ref, flags);
		vs_storage_commit_page(&st.base, blk);
	}
	else
	{
		BlockNumber prev = InvalidBlockNumber;
		for (uint32_t i = 0; i < npages; i++)
		{
			if (i > 0)
				page = vs_storage_new_page(&st.base, &blk);
			prism_posting_reference_page_fill(page, 42, dim, i, ref, flags);
			bool end = i + 1 == npages;
			ASSERT_EQ(
					end ? prism_posting_max_entries_first(dim, ref_format)
						: 0u,
					(uint32_t)prism_posting_opaque(page)->max_entries,
					"only the head END page has entry capacity");
			if (prev != InvalidBlockNumber)
				prism_posting_opaque(vs_storage_write_page(&st.base, prev))
						->next_blkno = blk;
			vs_storage_commit_page(&st.base, blk);
			prev = blk;
		}
		/* An entry page after the chain, so the gather can report it. */
		page = vs_storage_new_page(&st.base, &blk);
		prism_posting_page_init(page, 42, dim, 0);
		prism_posting_opaque(vs_storage_write_page(&st.base, prev))
				->next_blkno = blk;
		vs_storage_commit_page(&st.base, blk);
	}

	BlockNumber entries =
			prism_posting_reference_read(&st.base, head, dim, ref_format, got);
	ASSERT_MEM_EQ(
			want, got, (size_t)dim * sizeof(float), "the gather matches");
	if (!prism_posting_spilled(dim, ref_format))
		ASSERT_EQ(head, entries, "inline entries start on the head");
	else
		ASSERT_EQ(
				head + npages - 1,
				entries,
				"spilled entries start on the head END page");

	vs_free(ref);
	vs_free(want);
	vs_free(got);
	vs_free(st.pages);
}

TEST(reference_round_trips_inline_and_spilled)
{
	Dimension dims[] =
			{768,
			 PRISM_INDEX_NARROW_MAX_DIM,
			 2037,
			 2048,
			 4096,
			 4097,
			 PRISM_INDEX_MAX_DIM};
	for (uint32_t i = 0; i < sizeof(dims) / sizeof(dims[0]); i++)
		check_reference_roundtrip(
				result, dims[i], PRISM_POSTING_REFERENCE_FLOAT32);
}

/*
 * A float16 reference reads back as the float16 rounding of what was
 * written, inline up to 3820 dimensions and spilled past it, including
 * where it spills but fits a page and the END page holds none of it.
 */
TEST(reference16_round_trips_inline_and_spilled)
{
	Dimension dims[] = {768, 2048, 3072, 3820, 3821, 4072, 4073, 4096, 8192};

	for (uint32_t i = 0; i < sizeof(dims) / sizeof(dims[0]); i++)
		check_reference_roundtrip(
				result, dims[i], PRISM_POSTING_REFERENCE_FLOAT16);
}

/*
 * A head page that belongs to another list, or a chain that ends early, is
 * corruption: the gather must refuse rather than hand back a vector
 * stitched from foreign pages.
 */
typedef struct ReferenceReadArgs
{
	VsStorage				   *storage;
	BlockNumber					head;
	Dimension					dim;
	PrismPostingReferenceFormat ref_format;
	float					   *out;
} ReferenceReadArgs;

static void
run_reference_read(void *arg)
{
	const ReferenceReadArgs *a = arg;

	prism_posting_reference_read(
			a->storage, a->head, a->dim, a->ref_format, a->out);
}

TEST(reference_read_rejects_a_foreign_page)
{
	Dimension dim	 = 4096;
	uint32_t  npages = prism_posting_reference_pages(
			 dim, PRISM_POSTING_REFERENCE_FLOAT32);
	TestPageStorage st	 = make_test_storage(npages + 1);
	float		   *ref	 = vs_alloc0((size_t)dim * sizeof(float));
	float		   *got	 = vs_alloc((size_t)dim * sizeof(float));
	BlockNumber		prev = InvalidBlockNumber;
	BlockNumber		head = InvalidBlockNumber;

	for (uint32_t i = 0; i < npages; i++)
	{
		BlockNumber blk;
		Page		page = vs_storage_new_page(&st.base, &blk);
		/* The END page carries a different cluster id. */
		prism_posting_reference_page_fill(
				page, i + 1 == npages ? 43 : 42, dim, i, ref, 0);
		if (prev != InvalidBlockNumber)
			prism_posting_opaque(vs_storage_write_page(&st.base, prev))
					->next_blkno = blk;
		else
			head = blk;
		prev = blk;
	}

	ReferenceReadArgs args =
			{.storage = &st.base, .head = head, .dim = dim, .out = got};
	ASSERT_TRUE(
			vs_test_aborts(run_reference_read, &args, NULL, 0),
			"a foreign page aborts the gather");

	/* And a chain cut short. */
	prism_posting_opaque(vs_storage_write_page(&st.base, head))->next_blkno =
			InvalidBlockNumber;
	ASSERT_TRUE(
			vs_test_aborts(run_reference_read, &args, NULL, 0),
			"a short chain aborts the gather");

	vs_free(ref);
	vs_free(got);
	vs_free(st.pages);
}

/*
 * A head page whose reference width is not the one the index uses is
 * refused: read as the other width, its bytes would be a different vector.
 */
TEST(reference_read_rejects_the_other_width)
{
	Dimension dims[] = {768, 4096};

	for (uint32_t k = 0; k < 2; k++)
	{
		Dimension dim	 = dims[k];
		uint32_t  npages = prism_posting_reference_pages(
				 dim, PRISM_POSTING_REFERENCE_FLOAT16);
		TestPageStorage st	 = make_test_storage(npages + 1);
		float		   *ref	 = vs_alloc0((size_t)dim * sizeof(float));
		float		   *got	 = vs_alloc((size_t)dim * sizeof(float));
		BlockNumber		prev = InvalidBlockNumber;
		BlockNumber		head = InvalidBlockNumber;

		for (uint32_t i = 0; i < npages; i++)
		{
			BlockNumber blk;
			Page		page = vs_storage_new_page(&st.base, &blk);

			prism_posting_reference_page_fill(
					page, 42, dim, i, ref, PRISM_POSTING_PAGE_REF16);
			if (prev != InvalidBlockNumber)
				prism_posting_opaque(vs_storage_write_page(&st.base, prev))
						->next_blkno = blk;
			else
				head = blk;
			prev = blk;
		}

		ReferenceReadArgs args =
				{.storage	 = &st.base,
				 .head		 = head,
				 .dim		 = dim,
				 .ref_format = PRISM_POSTING_REFERENCE_FLOAT16,
				 .out		 = got};
		ASSERT_FALSE(
				vs_test_aborts(run_reference_read, &args, NULL, 0),
				"read as float16, it is accepted");
		args.ref_format = PRISM_POSTING_REFERENCE_FLOAT32;
		ASSERT_TRUE(
				vs_test_aborts(run_reference_read, &args, NULL, 0),
				"read as float32, it is refused");

		vs_free(ref);
		vs_free(got);
		vs_free(st.pages);
	}
}

/*
 * A chain whose next_blkno leaves the posting pages is corrupt. The walk must
 * stop rather than cast a foreign page's bytes as a posting opaque and follow
 * whatever they happen to contain.
 */
typedef struct ChainWalkArgs
{
	VsStorage  *storage;
	BlockNumber head;
} ChainWalkArgs;

/* Exits with the number of pages visited, or dies by alarm if it cycles. */
static void
run_chain_walk(void *arg)
{
	const ChainWalkArgs *a		 = arg;
	uint32_t			 visited = 0;

	alarm(5);
	prism_posting_chain_walk(a->storage, a->head, count_pages_cb, &visited);
	_exit((int)visited);
}

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
	ChainWalkArgs args = {.storage = &st.base, .head = head};
	int status		   = vs_test_run_in_child(run_chain_walk, &args, NULL, 0);
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

typedef struct ConvertArgs
{
	VsStorage  *storage;
	BlockNumber head;
	Dimension	dim;
} ConvertArgs;

static void
run_convert(void *arg)
{
	const ConvertArgs *a = arg;
	bool			   has_base;

	prism_posting_convert_to_fastscan(a->storage, a->head, a->dim, &has_base);
}

/*
 * Conversion reads its input as AoS. A list that starts with fastscan pages,
 * after its reference pages if it has any, is reported as already converted
 * and nothing is written. A fastscan page after AoS ones is a chain no
 * writer makes, and is refused rather than misread as AoS entries.
 */
TEST(convert_refuses_fastscan_pages)
{
	const Dimension dims[] = {128, 4096};

	for (uint32_t di = 0; di < sizeof(dims) / sizeof(dims[0]); di++)
	{
		Dimension		dim		 = dims[di];
		TestPageStorage st		 = make_test_storage(64);
		RaBitQParams   *params	 = vs_rabitq_create(dim, 42);
		float		   *centroid = vs_alloc0(dim * sizeof(float));
		float		   *vecs	 = make_test_vectors(40, dim);
		bool			has_base;

		BlockNumber fs =
				build_cluster(&st, params, dim, centroid, vecs, 40, true);
		uint32_t pages = st.next_blkno;
		ASSERT_EQ(
				InvalidBlockNumber,
				prism_posting_convert_to_fastscan(
						&st.base, fs, dim, &has_base),
				"a fastscan list is not converted again");
		ASSERT_TRUE(has_base, "it has a fastscan base");
		ASSERT_EQ(pages, st.next_blkno, "and nothing is written");

		/* An AoS list whose chain runs on into the fastscan one. */
		BlockNumber aos =
				build_cluster(&st, params, dim, centroid, vecs, 40, false);
		BlockNumber tail = prism_posting_head_tail(
				test_read_page(&st.base, aos));
		prism_posting_opaque(test_read_page(&st.base, tail))->next_blkno = fs;

		ConvertArgs args = {.storage = &st.base, .head = aos, .dim = dim};
		ASSERT_TRUE(
				vs_test_aborts(run_convert, &args, NULL, 0),
				"a fastscan page after AoS ones aborts the conversion");

		vs_free(vecs);
		vs_free(centroid);
		vs_free(st.pages);
		vs_rabitq_destroy(params);
	}
}

/* ----------------------------------------------------------------
 * Page format dispatch
 * ---------------------------------------------------------------- */

static int
cmp_candidate_id(const void *a, const void *b)
{
	uint64_t ia = ((const VsTopKEntry *)a)->id;
	uint64_t ib = ((const VsTopKEntry *)b)->id;
	return (ia > ib) - (ia < ib);
}

/*
 * Scans one cluster with k large enough that nothing is pruned and copies
 * every candidate into out (if not NULL), sorted by id. Returns the count.
 */
static uint32_t
scan_candidates(
		TestPageStorage	 *st,
		RaBitQParams	 *params,
		RaBitQQueryState *qstate,
		Dimension		  dim,
		BlockNumber		  head,
		uint32_t		  k,
		bool			  enable_fastscan,
		VsTopKEntry		 *out)
{
	PrismPostingScan scan;
	prism_posting_scan_init(
			&scan,
			&st->base,
			NULL,
			params,
			dim,
			PRISM_POSTING_REFERENCE_FLOAT32,
			prism_posting_max_entries(dim));
	if (enable_fastscan)
		prism_posting_scan_enable_fastscan(&scan, 16);

	VsTopK topk;
	vs_topk_init(&topk, k);
	prism_posting_scan_begin_cluster(&scan, qstate, head);
	prism_posting_scan_cluster(&scan, &topk);
	prism_posting_scan_end_cluster(&scan);

	uint32_t n = topk.cand_count;
	if (out != NULL)
	{
		memcpy(out, topk.candidates, n * sizeof(VsTopKEntry));
		qsort(out, n, sizeof(VsTopKEntry), cmp_candidate_id);
	}
	vs_topk_cleanup(&topk);
	prism_posting_scan_cleanup(&scan);
	return n;
}

/* Encoded id of vector vid, as the scan reports it. */
static uint64_t
vid_id(uint32_t vid)
{
	ItemPointerData tid = vid_to_tid(vid);
	return prism_posting_encode_tid(&tid);
}

/*
 * A flat buffer has no page header, so the standalone scan must not read page
 * flags off it. With small vector ids the special-space offset a page header
 * would hold is zero, which overlays the opaque on the flat header and puts
 * flags in the top half of cluster_id. Setting FASTSCAN there makes a misread
 * certain.
 */
TEST(flat_scan_ignores_page_flags)
{
	Dimension	  dim	   = 128;
	RaBitQParams *params   = vs_rabitq_create(dim, 42);
	float		 *centroid = vs_alloc0(dim * sizeof(float));
	uint32_t	  nvecs	   = 20;
	float		 *vecs	   = make_test_vectors(nvecs, dim);

	PrismFlatPostingBuilder builder;
	prism_flat_posting_builder_init(
			&builder,
			params,
			dim,
			(uint32_t)PRISM_POSTING_PAGE_FASTSCAN << 16,
			centroid,
			nvecs);
	for (uint32_t i = 0; i < nvecs; i++)
		prism_flat_posting_builder_add(
				&builder, vid_to_tid(i), vecs + (size_t)i * dim);
	char *flat = prism_flat_posting_builder_finish(&builder);
	prism_flat_posting_builder_cleanup(&builder);

	RaBitQQueryState qstate;
	setup_query_state(&qstate, params, centroid, dim);

	PrismPostingScan scan;
	prism_posting_scan_init(
			&scan,
			NULL,
			NULL,
			params,
			dim,
			PRISM_POSTING_REFERENCE_FLOAT32,
			nvecs);

	VsTopK topk;
	vs_topk_init(&topk, nvecs);
	prism_posting_scan_begin_flat(&scan, &qstate, flat);
	prism_posting_scan_cluster_standalone(&scan, &topk);
	prism_posting_scan_end_cluster(&scan);

	ASSERT_EQ(nvecs, scan.entries_scanned, "flat scan scores every entry");
	ASSERT_EQ(nvecs, topk.cand_count, "all entries in topk");

	vs_topk_cleanup(&topk);
	prism_posting_scan_cleanup(&scan);
	vs_free(flat);
	vs_rabitq_destroy(params);
}

/*
 * A fastscan page is scored with the fastscan kernel. Read as AoS it would
 * still count every entry but report garbage TIDs, so check the ids.
 */
TEST(scan_scores_fastscan_page)
{
	Dimension		dim		 = 128;
	TestPageStorage st		 = make_test_storage(16);
	RaBitQParams   *params	 = vs_rabitq_create(dim, 42);
	float		   *centroid = vs_alloc0(dim * sizeof(float));
	uint32_t		nvecs	 = 40;
	float		   *vecs	 = make_test_vectors(nvecs, dim);

	BlockNumber head =
			build_cluster(&st, params, dim, centroid, vecs, nvecs, true);

	RaBitQQueryState qstate;
	setup_query_state(&qstate, params, centroid, dim);

	VsTopKEntry *got = vs_alloc(nvecs * sizeof(VsTopKEntry));
	ASSERT_EQ(
			nvecs,
			scan_candidates(&st, params, &qstate, dim, head, nvecs, true, got),
			"every entry is a candidate");
	for (uint32_t i = 0; i < nvecs; i++)
		ASSERT_EQ(vid_id(i), got[i].id, "candidate ids are the built ones");

	vs_free(got);
	vs_rabitq_destroy(params);
}

typedef struct AosScanArgs
{
	TestPageStorage	 *st;
	RaBitQParams	 *params;
	RaBitQQueryState *qstate;
	Dimension		  dim;
	BlockNumber		  head;
	uint32_t		  k;
} AosScanArgs;

/* Scans a cluster without fastscan state. */
static void
run_aos_scan(void *arg)
{
	const AosScanArgs *a = arg;

	scan_candidates(
			a->st, a->params, a->qstate, a->dim, a->head, a->k, false, NULL);
}

/*
 * Without fastscan state the scan refuses a fastscan page rather than
 * reading it as AoS. Runs in a forked child because vs_error aborts.
 */
TEST(scan_rejects_fastscan_page_without_state)
{
	Dimension		dim		 = 128;
	TestPageStorage st		 = make_test_storage(16);
	RaBitQParams   *params	 = vs_rabitq_create(dim, 42);
	float		   *centroid = vs_alloc0(dim * sizeof(float));
	uint32_t		nvecs	 = 40;
	float		   *vecs	 = make_test_vectors(nvecs, dim);

	BlockNumber head =
			build_cluster(&st, params, dim, centroid, vecs, nvecs, true);

	RaBitQQueryState qstate;
	setup_query_state(&qstate, params, centroid, dim);

	AosScanArgs args =
			{.st	 = &st,
			 .params = params,
			 .qstate = &qstate,
			 .dim	 = dim,
			 .head	 = head,
			 .k		 = nvecs};
	ASSERT_TRUE(
			vs_test_aborts(run_aos_scan, &args, NULL, 0),
			"a fastscan page without fastscan state aborts");

	vs_rabitq_destroy(params);
}

/*
 * A list's first entry page holds fewer AoS entries than a later page, since
 * the reference comes first. The scan bounds its entry count by that page's
 * own capacity: a count that would be valid on a later page would read past
 * the end of this one. Covers an inline head (768) and an END page (3072).
 */
TEST(scan_rejects_an_overfull_first_entry_page)
{
	const Dimension dims[] = {768, 3072};

	for (uint32_t di = 0; di < sizeof(dims) / sizeof(dims[0]); di++)
	{
		Dimension		dim		 = dims[di];
		TestPageStorage st		 = make_test_storage(16);
		RaBitQParams   *params	 = vs_rabitq_create(dim, 42);
		float		   *centroid = vs_alloc0(dim * sizeof(float));
		float		   *vecs	 = make_test_vectors(4, dim);

		BlockNumber head =
				build_cluster(&st, params, dim, centroid, vecs, 4, false);
		float	   *ref	  = vs_alloc((size_t)dim * sizeof(float));
		BlockNumber first = prism_posting_reference_read(
				&st.base, head, dim, PRISM_POSTING_REFERENCE_FLOAT32, ref);
		vs_free(ref);

		PrismPostingPageOpaque *op = prism_posting_opaque(
				test_read_page(&st.base, first));
		ASSERT_TRUE(
				prism_posting_max_entries_first(
						dim, PRISM_POSTING_REFERENCE_FLOAT32) <
						prism_posting_max_entries(dim),
				"the first entry page holds fewer entries");
		op->entry_count = (uint16_t)prism_posting_max_entries(dim);

		RaBitQQueryState qstate;
		setup_query_state(&qstate, params, centroid, dim);
		AosScanArgs args =
				{.st	 = &st,
				 .params = params,
				 .qstate = &qstate,
				 .dim	 = dim,
				 .head	 = head,
				 .k		 = 4};
		ASSERT_TRUE(
				vs_test_aborts(run_aos_scan, &args, NULL, 0),
				"a later page's count on the first entry page aborts");

		vs_free(vecs);
		vs_free(centroid);
		vs_free(st.pages);
		vs_rabitq_destroy(params);
	}
}

/*
 * A chain that starts AoS and continues fastscan scores each page with its
 * own kernel: its candidates are exactly those of the two lists scanned on
 * their own. The AoS list is linked to the fastscan list by hand.
 */
TEST(scan_scores_mixed_chain)
{
	Dimension		dim		 = 128;
	TestPageStorage st		 = make_test_storage(32);
	RaBitQParams   *params	 = vs_rabitq_create(dim, 42);
	float		   *centroid = vs_alloc0(dim * sizeof(float));
	uint32_t		naos	 = 20;
	uint32_t		nfs		 = 40;
	uint32_t		n		 = naos + nfs;
	float		   *vecs	 = make_test_vectors(n, dim);

	BlockNumber aos_head =
			build_cluster(&st, params, dim, centroid, vecs, naos, false);
	BlockNumber fs_head = build_cluster_from(
			&st,
			params,
			dim,
			centroid,
			vecs + (size_t)naos * dim,
			nfs,
			true,
			naos);

	RaBitQQueryState qstate;
	setup_query_state(&qstate, params, centroid, dim);

	/* Reference: each list on its own. Ids sort AoS first, then fastscan. */
	VsTopKEntry *ref = vs_alloc(n * sizeof(VsTopKEntry));
	ASSERT_EQ(
			naos,
			scan_candidates(&st, params, &qstate, dim, aos_head, n, true, ref),
			"AoS list alone");
	ASSERT_EQ(
			nfs,
			scan_candidates(
					&st, params, &qstate, dim, fs_head, n, true, ref + naos),
			"fastscan list alone");

	PrismPostingPageOpaque *aos_op = prism_posting_opaque(
			test_read_page(&st.base, aos_head));
	ASSERT_EQ(
			InvalidBlockNumber,
			aos_op->next_blkno,
			"the AoS list is a single page");
	aos_op->next_blkno = fs_head;

	VsTopKEntry *got = vs_alloc(n * sizeof(VsTopKEntry));
	ASSERT_EQ(
			n,
			scan_candidates(&st, params, &qstate, dim, aos_head, n, true, got),
			"the mixed chain returns both lists");
	for (uint32_t i = 0; i < n; i++)
	{
		ASSERT_EQ(vid_id(i), ref[i].id, "reference ids are the built ones");
		ASSERT_EQ(ref[i].id, got[i].id, "same candidate ids");
		ASSERT_EQ(ref[i].distance, got[i].distance, "same distances");
		ASSERT_EQ(ref[i].error, got[i].error, "same error bounds");
	}

	vs_free(got);
	vs_free(ref);
	vs_rabitq_destroy(params);
}

/* ----------------------------------------------------------------
 * Fastscan code stream
 * ---------------------------------------------------------------- */

/*
 * Scan a cluster with k = n and return each TID's estimated distance.
 * lut_bits is the fastscan table width, or 0 for an AoS scan. The code stream
 * is read as if first_group groups came before it, which only moves the
 * positions the scan carries.
 */
static void
scan_estimates_from(
		TestPageStorage *st,
		RaBitQParams	*params,
		Dimension		 dim,
		const float		*centroid,
		BlockNumber		 head,
		uint32_t		 n,
		int				 lut_bits,
		uint32_t		 first_group,
		float			*est_by_vid)
{
	RaBitQQueryState qstate;
	setup_query_state(&qstate, params, centroid, dim);

	PrismPostingScan scan;
	prism_posting_scan_init(
			&scan,
			&st->base,
			NULL,
			params,
			dim,
			PRISM_POSTING_REFERENCE_FLOAT32,
			prism_posting_max_entries(dim));
	if (lut_bits != 0)
		prism_posting_scan_enable_fastscan(&scan, lut_bits);

	VsTopK topk;
	vs_topk_init(&topk, n);
	prism_posting_scan_begin_cluster(&scan, &qstate, head);
	scan.fs_next_group = first_group;
	scan.fs_stream_pos = (uint64_t)first_group * VS_FASTSCAN_GROUP_BYTES(dim);
	prism_posting_scan_cluster(&scan, &topk);
	prism_posting_scan_end_cluster(&scan);

	VsTopKEntry *res = vs_alloc((size_t)topk.cand_count * sizeof(*res));
	uint32_t	 nres;
	vs_topk_extract_sorted(&topk, res, &nres);
	for (uint32_t i = 0; i < n; i++)
		est_by_vid[i] = NAN;
	for (uint32_t i = 0; i < nres; i++)
		est_by_vid[prism_posting_decode_vector_id(res[i].id)] =
				res[i].distance;

	vs_topk_cleanup(&topk);
	prism_posting_scan_cleanup(&scan);
}

static void
scan_estimates(
		TestPageStorage *st,
		RaBitQParams	*params,
		Dimension		 dim,
		const float		*centroid,
		BlockNumber		 head,
		uint32_t		 n,
		int				 lut_bits,
		float			*est_by_vid)
{
	scan_estimates_from(
			st, params, dim, centroid, head, n, lut_bits, 0, est_by_vid);
}

/* One dimension and list size of fastscan_stream_round_trip. */
static void
check_stream_round_trip(VsTestResult *result, Dimension dim, uint32_t n)
{
	uint32_t		code_len = VS_FASTSCAN_GROUP_BYTES(dim);
	TestPageStorage st		 = make_test_storage(2048);
	RaBitQParams   *params	 = vs_rabitq_create(dim, 5);
	float		   *centroid = vs_alloc0(dim * sizeof(float));
	float		   *vecs	 = make_test_vectors(n, dim);

	BlockNumber aos =
			build_cluster(&st, params, dim, centroid, vecs, n, false);
	BlockNumber fs = build_cluster(&st, params, dim, centroid, vecs, n, true);

	/* Walk the pages, carrying the stream position from page to
	 * page as the scan does. A spilled list's entries start on its
	 * head END page. */
	uint32_t	pos = 0, next_group = 0, entries = 0;
	uint32_t	prev_free = 0;
	BlockNumber start	  = fs;
	if (prism_posting_spilled(dim, PRISM_POSTING_REFERENCE_FLOAT32))
	{
		float *ref = vs_alloc((size_t)dim * sizeof(float));
		start	   = prism_posting_reference_read(
				 &st.base, fs, dim, PRISM_POSTING_REFERENCE_FLOAT32, ref);
		vs_free(ref);
	}
	BlockNumber blk = start;
	while (blk != InvalidBlockNumber)
	{
		Page					p  = test_read_page(&st.base, blk);
		PrismPostingPageOpaque *op = prism_posting_opaque(p);
		bool	 first	 = prism_posting_has_ref_prefix(op->flags, dim);
		uint32_t ngroups = prism_fastscan_page_groups(op);

		prism_posting_check_count(blk, p, dim);
		ASSERT_TRUE(
				op->flags & PRISM_POSTING_PAGE_FASTSCAN,
				"every page is fastscan");
		ASSERT_TRUE(
				next_group + ngroups <= (pos + op->code_bytes) / code_len,
				"metadata only for groups whose codes are complete");

		/* A page ends with a chunk or more free only when its last
		 * group's metadata did not fit and opens this page. */
		bool opens_deferred = ngroups > 0 &&
							  (next_group + 1) * code_len <= pos;
		if (blk != start)
			ASSERT_TRUE(
					prev_free < PRISM_FASTSCAN_CHUNK_BYTES || opens_deferred,
					"free space left only for deferred metadata");

		uint32_t used = ngroups * PRISM_FASTSCAN_META_BYTES + op->code_bytes;
		prev_free	  = prism_fastscan_page_space(
							dim, first, PRISM_POSTING_REFERENCE_FLOAT32) -
					used;
		pos += op->code_bytes;
		next_group += ngroups;
		entries += op->entry_count;
		blk = op->next_blkno;
	}
	uint32_t ngroups = (n + VS_FASTSCAN_GROUP - 1) / VS_FASTSCAN_GROUP;
	ASSERT_EQ(ngroups * code_len, pos, "all codes in the stream");
	ASSERT_EQ(ngroups, next_group, "metadata for every group");
	ASSERT_EQ(n, entries, "every vector counted once");

	/* Same estimates as AoS, within LUT quantization. The 8-bit
	 * table is coarse, so it only has to find every vector. */
	float *want = vs_alloc(n * sizeof(float));
	float *got	= vs_alloc(n * sizeof(float));
	scan_estimates(&st, params, dim, centroid, aos, n, 0, want);
	scan_estimates(&st, params, dim, centroid, fs, n, 16, got);
	for (uint32_t i = 0; i < n; i++)
	{
		ASSERT_FALSE(isnan(got[i]), "fastscan returns every vector");
		ASSERT_TRUE(
				fabsf(got[i] - want[i]) <= 1e-3f * (fabsf(want[i]) + 1.0f),
				"fastscan estimate matches AoS");
	}
	scan_estimates(&st, params, dim, centroid, fs, n, 8, got);
	for (uint32_t i = 0; i < n; i++)
		ASSERT_FALSE(isnan(got[i]), "8-bit fastscan returns every vector");

	vs_free(want);
	vs_free(got);
	vs_free(vecs);
	vs_free(centroid);
	vs_free(st.pages);
	vs_rabitq_destroy(params);
}

/*
 * The fastscan pages of a list carry one code stream: each page picks up
 * where the previous one stopped, group metadata comes in group order, and
 * a page ends only when the next chunk no longer fits, or when metadata is
 * deferred to the next page. A scan over that stream finds every vector at
 * the distance the AoS layout estimates, at dimensions where a group no
 * longer fits a page as well as where it does, and where the stream starts
 * after a spilled reference chain.
 */
TEST(fastscan_stream_round_trip)
{
	const Dimension dims[] =
			{16, 96, 100, 768, 1024, 1536, 1968, 2048, 4096, 8192};
	const uint32_t counts[] = {1, 33, 500};

	/* Each case in its own context: the arena frees nothing until it is
	 * deleted, and every case allocates its own page storage. */
	for (uint32_t di = 0; di < sizeof(dims) / sizeof(dims[0]); di++)
		for (uint32_t ci = 0; ci < sizeof(counts) / sizeof(counts[0]); ci++)
		{
			VsMemCtx ctx = vs_memctx_create(NULL, "stream round trip");
			VsMemCtx old = vs_memctx_switch(ctx);

			check_stream_round_trip(result, dims[di], counts[ci]);
			vs_memctx_switch(old);
			vs_memctx_delete(ctx);
		}
}

static bool
vid_in_first_group(ItemPointerData tid, void *state)
{
	(void)state;
	return prism_posting_get_vector_id(&tid) < VS_FASTSCAN_GROUP;
}

/*
 * A page whose entries are all dead still carries the next group's codes
 * when that group starts on it, so the scan must read it rather than skip
 * it. Here the head page holds group 0, which dies, and the start of group
 * 1, which stays live.
 */
TEST(fastscan_reads_a_dead_page_a_live_group_crosses)
{
	Dimension		dim		 = 768;
	uint32_t		n		 = 2 * VS_FASTSCAN_GROUP;
	uint32_t		code_len = VS_FASTSCAN_GROUP_BYTES(dim);
	TestPageStorage st		 = make_test_storage(16);
	RaBitQParams   *params	 = vs_rabitq_create(dim, 5);
	float		   *centroid = vs_alloc0(dim * sizeof(float));
	float		   *vecs	 = make_test_vectors(n, dim);

	BlockNumber aos =
			build_cluster(&st, params, dim, centroid, vecs, n, false);
	BlockNumber fs = build_cluster(&st, params, dim, centroid, vecs, n, true);

	const PrismPostingPageOpaque *head = prism_posting_opaque(
			test_read_page(&st.base, fs));
	ASSERT_EQ(1u, prism_fastscan_page_groups(head), "head scores group 0");
	ASSERT_TRUE(head->code_bytes > code_len, "group 1's codes start on it");

	ASSERT_EQ(
			VS_FASTSCAN_GROUP,
			prism_posting_tombstone_chain(
					&st.base, dim, fs, vid_in_first_group, NULL),
			"group 0 dies");
	ASSERT_TRUE(
			prism_posting_page_all_dead(
					prism_posting_opaque(test_read_page(&st.base, fs))),
			"the head page is all dead");

	float *want = vs_alloc(n * sizeof(float));
	float *got	= vs_alloc(n * sizeof(float));
	scan_estimates(&st, params, dim, centroid, aos, n, 0, want);
	scan_estimates(&st, params, dim, centroid, fs, n, 16, got);
	for (uint32_t i = 0; i < n; i++)
	{
		if (i < VS_FASTSCAN_GROUP)
		{
			ASSERT_TRUE(isnan(got[i]), "a dead vector is not returned");
			continue;
		}
		ASSERT_FALSE(isnan(got[i]), "a live vector is returned");
		ASSERT_TRUE(
				fabsf(got[i] - want[i]) <= 1e-3f * (fabsf(want[i]) + 1.0f),
				"at the estimate AoS gives it");
	}

	vs_free(want);
	vs_free(got);
	vs_free(vecs);
	vs_free(centroid);
	vs_free(st.pages);
	vs_rabitq_destroy(params);
}

/*
 * Stream positions are list-wide byte counts, which pass 4 GiB in a list of
 * about 17 million vectors at 1968 dimensions. Reading the stream as if that
 * many groups came first must give the same estimates.
 */
TEST(fastscan_stream_positions_past_4gib)
{
	Dimension		dim		 = 768;
	uint32_t		n		 = 200;
	uint32_t		code_len = VS_FASTSCAN_GROUP_BYTES(dim);
	uint32_t		offset	 = (uint32_t)((UINT64_C(5) << 32) / code_len);
	TestPageStorage st		 = make_test_storage(64);
	RaBitQParams   *params	 = vs_rabitq_create(dim, 5);
	float		   *centroid = vs_alloc0(dim * sizeof(float));
	float		   *vecs	 = make_test_vectors(n, dim);

	BlockNumber fs = build_cluster(&st, params, dim, centroid, vecs, n, true);

	float *want = vs_alloc(n * sizeof(float));
	float *got	= vs_alloc(n * sizeof(float));
	scan_estimates(&st, params, dim, centroid, fs, n, 16, want);
	scan_estimates_from(&st, params, dim, centroid, fs, n, 16, offset, got);
	for (uint32_t i = 0; i < n; i++)
	{
		ASSERT_FALSE(isnan(got[i]), "every vector is returned");
		ASSERT_FLOAT_EQ(want[i], got[i], 0.0f, "at the same estimate");
	}

	vs_free(want);
	vs_free(got);
	vs_free(vecs);
	vs_free(centroid);
	vs_free(st.pages);
	vs_rabitq_destroy(params);
}

typedef struct FastscanScanArgs
{
	TestPageStorage *st;
	RaBitQParams	*params;
	Dimension		 dim;
	const float		*centroid;
	BlockNumber		 head;
	uint32_t		 n;
} FastscanScanArgs;

/* Scans a fastscan cluster with the 16-bit lookup table. */
static void
run_fastscan_scan(void *arg)
{
	const FastscanScanArgs *a	= arg;
	float				   *est = vs_alloc(a->n * sizeof(float));

	scan_estimates(
			a->st, a->params, a->dim, a->centroid, a->head, a->n, 16, est);
	vs_free(est);
}

/*
 * A code run that ends inside a chunk would start the next page's kernel
 * call at a column the lookup table does not line up with. The scan refuses
 * the page before any kernel reads it, which the reason it gives shows: the
 * stream checks would only catch the damage a page later.
 */
TEST(fastscan_scan_rejects_a_run_that_splits_a_chunk)
{
	Dimension		dim		 = 1024;
	uint32_t		n		 = 200;
	TestPageStorage st		 = make_test_storage(64);
	RaBitQParams   *params	 = vs_rabitq_create(dim, 5);
	float		   *centroid = vs_alloc0(dim * sizeof(float));
	float		   *vecs	 = make_test_vectors(n, dim);

	BlockNumber head =
			build_cluster(&st, params, dim, centroid, vecs, n, true);
	PrismPostingPageOpaque *op = prism_posting_opaque(
			test_read_page(&st.base, head));
	ASSERT_TRUE(
			op->code_bytes % VS_FASTSCAN_GROUP_BYTES(dim) != 0,
			"the head's run ends inside a group");
	op->code_bytes -= PRISM_FASTSCAN_CHUNK_BYTES / 2;

	FastscanScanArgs args =
			{.st	   = &st,
			 .params   = params,
			 .dim	   = dim,
			 .centroid = centroid,
			 .head	   = head,
			 .n		   = n};
	char msg[4096];
	ASSERT_TRUE(
			vs_test_aborts(run_fastscan_scan, &args, msg, sizeof(msg)),
			"a run that splits a chunk aborts the scan");
	ASSERT_TRUE(
			strstr(msg, "a code run ends inside a chunk") != NULL,
			"before any kernel reads the page");

	vs_free(vecs);
	vs_free(centroid);
	vs_free(st.pages);
	vs_rabitq_destroy(params);
}

/*
 * A page whose code run is shorter than recorded breaks the stream: some
 * group's metadata arrives before its codes are complete. The scan refuses
 * it rather than scoring codes against the wrong group.
 */
TEST(fastscan_scan_rejects_broken_stream)
{
	Dimension		dim		 = 768;
	uint32_t		n		 = 200;
	TestPageStorage st		 = make_test_storage(64);
	RaBitQParams   *params	 = vs_rabitq_create(dim, 5);
	float		   *centroid = vs_alloc0(dim * sizeof(float));
	float		   *vecs	 = make_test_vectors(n, dim);

	BlockNumber head =
			build_cluster(&st, params, dim, centroid, vecs, n, true);
	Page hp = test_read_page(&st.base, head);
	Page p2 = test_read_page(&st.base, prism_posting_opaque(hp)->next_blkno);
	prism_posting_opaque(p2)->code_bytes -= PRISM_FASTSCAN_CHUNK_BYTES;

	FastscanScanArgs args =
			{.st	   = &st,
			 .params   = params,
			 .dim	   = dim,
			 .centroid = centroid,
			 .head	   = head,
			 .n		   = n};
	ASSERT_TRUE(
			vs_test_aborts(run_fastscan_scan, &args, NULL, 0),
			"a broken code stream aborts the scan");

	vs_free(vecs);
	vs_free(centroid);
	vs_free(st.pages);
	vs_rabitq_destroy(params);
}
