/*
 * test_posting_page.c - Unit tests for posting list page layout
 *
 * Tests cover:
 * - Struct size verification (binary compatibility)
 * - TID round-trip through posting entry metadata
 * - Page capacity calculation
 * - Page init and add operations
 * - Fill-to-capacity and overflow rejection
 * - Multi-entry round-trip with distinct values
 */

#include <string.h>

#include "core/memory.h"
#include "core/pg_compat.h"
#include "index/posting_page.h"
#include "mkt_test.h"
#include "quant/rabitq.h"

TEST_GROUP(PostingPage);
TEST_MEMCTX_FIXTURE();

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
	ASSERT_EQ(
			16,
			sizeof(MktPostingPageOpaque),
			"MktPostingPageOpaque must be 16 bytes");
}

/* ----------------------------------------------------------------
 * TID round-trip through posting metadata
 * ---------------------------------------------------------------- */

TEST(posting_tid_round_trip)
{
	MktPostingEntryMeta meta;
	memset(&meta, 0, sizeof(meta));

	ItemPointerSet(&meta.tid, 42, 7);

	ASSERT_EQ(
			42,
			ItemPointerGetBlockNumber(&meta.tid),
			"block number round-trip");
	ASSERT_EQ(
			7,
			ItemPointerGetOffsetNumber(&meta.tid),
			"offset number round-trip");
}

TEST(posting_tid_large_values)
{
	MktPostingEntryMeta meta;
	memset(&meta, 0, sizeof(meta));

	BlockNumber	 blk = 0x00ABCDEF;
	OffsetNumber off = 0x1234;
	ItemPointerSet(&meta.tid, blk, off);

	ASSERT_EQ(
			blk,
			ItemPointerGetBlockNumber(&meta.tid),
			"large block round-trip");
	ASSERT_EQ(
			off,
			ItemPointerGetOffsetNumber(&meta.tid),
			"large offset round-trip");
}

/* ----------------------------------------------------------------
 * Page capacity tests
 * ---------------------------------------------------------------- */

TEST(posting_capacity_768d)
{
	uint32_t max = mkt_posting_max_entries(768);
	/* Per entry: 8 (meta) + 8 (f_add+f_rescale) + 96 (bits) = 112
	 * Usable: 8192 - 24 - 16 = 8152
	 * 8152 / 112 = 72 */
	uint32_t entry_sz = mkt_posting_entry_bytes(768);
	uint32_t expected = (uint32_t)(MKT_POSTING_PAGE_USABLE / entry_sz);
	ASSERT_EQ(expected, max, "768d capacity mismatch");
}

TEST(posting_capacity_128d)
{
	uint32_t max	  = mkt_posting_max_entries(128);
	uint32_t entry_sz = mkt_posting_entry_bytes(128);
	uint32_t expected = (uint32_t)(MKT_POSTING_PAGE_USABLE / entry_sz);
	ASSERT_EQ(expected, max, "128d capacity mismatch");
}

TEST(posting_capacity_1536d)
{
	uint32_t max	  = mkt_posting_max_entries(1536);
	uint32_t entry_sz = mkt_posting_entry_bytes(1536);
	uint32_t expected = (uint32_t)(MKT_POSTING_PAGE_USABLE / entry_sz);
	ASSERT_EQ(expected, max, "1536d capacity mismatch");
}

/* ----------------------------------------------------------------
 * Bidirectional region non-overlap
 * ---------------------------------------------------------------- */

TEST(posting_bidir_no_overlap)
{
	Dimension dim		= 768;
	uint32_t  max		= mkt_posting_max_entries(dim);
	uint32_t  data_size = MKT_RABITQ_DATA_SIZE(dim);

	size_t meta_start = SizeOfPageHeaderData;
	size_t meta_end	  = meta_start + max * sizeof(MktPostingEntryMeta);

	size_t opaque_start = BLCKSZ - MAXALIGN(sizeof(MktPostingPageOpaque));
	size_t data_start	= opaque_start - (size_t)max * data_size;

	ASSERT_TRUE(
			meta_end <= data_start,
			"meta region must not overlap data region");
}

/* ----------------------------------------------------------------
 * Page init
 * ---------------------------------------------------------------- */

TEST(posting_page_init_basic)
{
	char page_buf[BLCKSZ];
	Page page = page_buf;

	mkt_posting_page_init(page, 42, MKT_POSTING_PAGE_FIRST);

	MktPostingPageOpaque *opaque = MKT_POSTING_OPAQUE(page);
	ASSERT_EQ(
			InvalidBlockNumber,
			opaque->next_blkno,
			"next_blkno should be invalid");
	ASSERT_EQ(42, opaque->cluster_id, "cluster_id should be 42");
	ASSERT_EQ(0, opaque->entry_count, "entry_count should be 0");
	ASSERT_EQ(MKT_POSTING_PAGE_FIRST, opaque->flags, "flags should be FIRST");
	ASSERT_EQ(
			MKT_POSTING_PAGE_ID,
			opaque->page_id,
			"page_id should be MKT_POSTING_PAGE_ID");
}

/* ----------------------------------------------------------------
 * Page add single entry
 * ---------------------------------------------------------------- */

TEST(posting_page_add_single)
{
	Dimension dim = 128;

	RaBitQParams *params = mkt_rabitq_create(dim, 42);
	ASSERT_NOT_NULL(params, "params created");

	float *vec		= mkt_alloc(dim * sizeof(float));
	float *centroid = mkt_alloc(dim * sizeof(float));
	for (Dimension i = 0; i < dim; i++)
	{
		vec[i]		= (float)((i * 17 + 3) % 100 - 50) / 10.0f;
		centroid[i] = 0.0f;
	}

	VectorRef	vec_ref	 = {.data = vec, .dim = dim};
	VectorRef	cent_ref = {.data = centroid, .dim = dim};
	RaBitQData *encoded	 = mkt_rabitq_encode(params, vec_ref, cent_ref);
	ASSERT_NOT_NULL(encoded, "encoding succeeded");

	char page_buf[BLCKSZ];
	Page page = page_buf;
	mkt_posting_page_init(page, 0, MKT_POSTING_PAGE_FIRST);

	bool added = mkt_posting_page_add(page, dim, 5, 10, encoded, 0);
	ASSERT_TRUE(added, "entry should be added");

	MktPostingPageOpaque *opaque = MKT_POSTING_OPAQUE(page);
	ASSERT_EQ(1, opaque->entry_count, "entry_count should be 1");

	const MktPostingEntryMeta *meta = mkt_posting_meta(page, 0);
	ASSERT_EQ(0, meta->flags, "flags should be 0");
	ASSERT_EQ(5, ItemPointerGetBlockNumber(&meta->tid), "block number");
	ASSERT_EQ(10, ItemPointerGetOffsetNumber(&meta->tid), "offset number");

	/* Verify RaBitQData round-trip */
	const RaBitQData *data		   = mkt_posting_data(page, 0, dim);
	uint32_t		  packed_bytes = MKT_RABITQ_BYTES(dim);
	ASSERT_FLOAT_EQ(encoded->f_add, data->f_add, 1e-6f, "f_add");
	ASSERT_FLOAT_EQ(encoded->f_rescale, data->f_rescale, 1e-6f, "f_rescale");
	int bits_match = memcmp(encoded->bits, data->bits, packed_bytes) == 0;
	ASSERT_TRUE(bits_match, "bits round-trip");

	mkt_free(encoded);
	mkt_rabitq_destroy(params);
}

/* ----------------------------------------------------------------
 * Fill to capacity
 * ---------------------------------------------------------------- */

TEST(posting_page_fill_to_capacity)
{
	Dimension dim = 768;
	uint32_t  max = mkt_posting_max_entries(dim);

	RaBitQParams *params = mkt_rabitq_create(dim, 42);
	ASSERT_NOT_NULL(params, "params created");

	char page_buf[BLCKSZ];
	Page page = page_buf;
	mkt_posting_page_init(page, 0, MKT_POSTING_PAGE_FIRST);

	float *vec		= mkt_alloc(dim * sizeof(float));
	float *centroid = mkt_alloc(dim * sizeof(float));
	for (Dimension i = 0; i < dim; i++)
	{
		vec[i]		= (float)((i * 7 + 1) % 100 - 50) / 10.0f;
		centroid[i] = 0.0f;
	}

	VectorRef	vec_ref	 = {.data = vec, .dim = dim};
	VectorRef	cent_ref = {.data = centroid, .dim = dim};
	RaBitQData *encoded	 = mkt_rabitq_encode(params, vec_ref, cent_ref);
	ASSERT_NOT_NULL(encoded, "encoding succeeded");

	for (uint32_t i = 0; i < max; i++)
	{
		bool added = mkt_posting_page_add(
				page, dim, 0, (OffsetNumber)i, encoded, 0);

		char msg[64];
		snprintf(msg, sizeof(msg), "entry %u should be added", i);
		ASSERT_TRUE(added, msg);
	}

	MktPostingPageOpaque *opaque = MKT_POSTING_OPAQUE(page);
	ASSERT_EQ(max, opaque->entry_count, "page should be full");

	/* One more should fail */
	bool added = mkt_posting_page_add(page, dim, 0, 0, encoded, 0);
	ASSERT_FALSE(added, "page should be full");

	mkt_free(encoded);
	mkt_rabitq_destroy(params);
}

TEST(posting_page_has_room)
{
	Dimension dim = 128;
	char	  page_buf[BLCKSZ];
	Page	  page = page_buf;
	mkt_posting_page_init(page, 0, 0);

	ASSERT_TRUE(
			mkt_posting_page_has_room(page, dim),
			"empty page should have room");

	RaBitQParams *params   = mkt_rabitq_create(dim, 42);
	float		 *vec	   = mkt_alloc(dim * sizeof(float));
	float		 *centroid = mkt_alloc(dim * sizeof(float));
	for (Dimension i = 0; i < dim; i++)
	{
		vec[i]		= (float)i / 10.0f;
		centroid[i] = 0.0f;
	}

	VectorRef	vec_ref	 = {.data = vec, .dim = dim};
	VectorRef	cent_ref = {.data = centroid, .dim = dim};
	RaBitQData *encoded	 = mkt_rabitq_encode(params, vec_ref, cent_ref);

	uint32_t max = mkt_posting_max_entries(dim);
	for (uint32_t i = 0; i < max; i++)
		mkt_posting_page_add(page, dim, 0, (OffsetNumber)i, encoded, 0);

	ASSERT_FALSE(
			mkt_posting_page_has_room(page, dim),
			"full page should not have room");

	mkt_free(encoded);
	mkt_rabitq_destroy(params);
}

/* ----------------------------------------------------------------
 * Multi-entry round-trip
 * ---------------------------------------------------------------- */

TEST(posting_page_multi_entry)
{
	Dimension dim		   = 64;
	uint32_t  packed_bytes = MKT_RABITQ_BYTES(dim);
	const int count		   = 10;

	RaBitQParams *params = mkt_rabitq_create(dim, 42);
	ASSERT_NOT_NULL(params, "params created");

	char page_buf[BLCKSZ];
	Page page = page_buf;
	mkt_posting_page_init(page, 99, MKT_POSTING_PAGE_FIRST);

	float *centroid = mkt_alloc(dim * sizeof(float));
	for (Dimension i = 0; i < dim; i++)
		centroid[i] = 0.0f;
	VectorRef cent_ref = {.data = centroid, .dim = dim};

	RaBitQData **encodings = mkt_alloc(count * sizeof(void *));
	for (int e = 0; e < count; e++)
	{
		float *vec = mkt_alloc(dim * sizeof(float));
		for (Dimension i = 0; i < dim; i++)
			vec[i] = (float)((i * 17 + e * 31) % 100 - 50) / 10.0f;

		VectorRef vec_ref = {.data = vec, .dim = dim};
		encodings[e]	  = mkt_rabitq_encode(params, vec_ref, cent_ref);
		ASSERT_NOT_NULL(encodings[e], "encoding succeeded");

		bool added = mkt_posting_page_add(
				page,
				dim,
				(BlockNumber)e,
				(OffsetNumber)(e + 1),
				encodings[e],
				0);
		ASSERT_TRUE(added, "entry added");
	}

	/* Verify all entries */
	for (int e = 0; e < count; e++)
	{
		const MktPostingEntryMeta *meta = mkt_posting_meta(page, e);
		char					   msg[128];

		snprintf(msg, sizeof(msg), "entry %d block", e);
		ASSERT_EQ((BlockNumber)e, ItemPointerGetBlockNumber(&meta->tid), msg);

		snprintf(msg, sizeof(msg), "entry %d offset", e);
		ASSERT_EQ(
				(OffsetNumber)(e + 1),
				ItemPointerGetOffsetNumber(&meta->tid),
				msg);

		const RaBitQData *data = mkt_posting_data(page, e, dim);

		snprintf(msg, sizeof(msg), "entry %d f_add", e);
		ASSERT_FLOAT_EQ(encodings[e]->f_add, data->f_add, 1e-6f, msg);

		snprintf(msg, sizeof(msg), "entry %d f_rescale", e);
		ASSERT_FLOAT_EQ(encodings[e]->f_rescale, data->f_rescale, 1e-6f, msg);

		snprintf(msg, sizeof(msg), "entry %d bits", e);
		int bits_match = memcmp(encodings[e]->bits,
								data->bits,
								packed_bytes) == 0;
		ASSERT_TRUE(bits_match, msg);

		mkt_free(encodings[e]);
	}

	mkt_free(encodings);
	mkt_rabitq_destroy(params);
}

/* ----------------------------------------------------------------
 * Deleted flag
 * ---------------------------------------------------------------- */

TEST(posting_entry_deleted_flag)
{
	Dimension dim = 64;

	RaBitQParams *params = mkt_rabitq_create(dim, 42);
	ASSERT_NOT_NULL(params, "params created");

	float *vec		= mkt_alloc(dim * sizeof(float));
	float *centroid = mkt_alloc(dim * sizeof(float));
	for (Dimension i = 0; i < dim; i++)
	{
		vec[i]		= (float)i;
		centroid[i] = 0.0f;
	}

	VectorRef	vec_ref	 = {.data = vec, .dim = dim};
	VectorRef	cent_ref = {.data = centroid, .dim = dim};
	RaBitQData *encoded	 = mkt_rabitq_encode(params, vec_ref, cent_ref);

	char page_buf[BLCKSZ];
	Page page = page_buf;
	mkt_posting_page_init(page, 0, 0);

	/* Add with DELETED flag */
	bool added = mkt_posting_page_add(
			page, dim, 1, 1, encoded, MKT_POSTING_FLAG_DELETED);
	ASSERT_TRUE(added, "deleted entry added");

	const MktPostingEntryMeta *meta = mkt_posting_meta(page, 0);
	ASSERT_TRUE(
			meta->flags & MKT_POSTING_FLAG_DELETED,
			"DELETED flag should be set");

	mkt_free(encoded);
	mkt_rabitq_destroy(params);
}
