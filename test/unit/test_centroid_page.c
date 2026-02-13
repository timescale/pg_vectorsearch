/*
 * test_centroid_page.c - Unit tests for centroid page layout and operations
 *
 * Tests cover:
 * - Struct size verification (binary compatibility with PG)
 * - Page capacity calculation
 * - SoA region non-overlap
 * - Page init/add round-trip
 * - ItemPointerData accessors
 * - Medoid TID round-trip
 */

#include <string.h>

#include "core/memory.h"
#include "core/pg_compat.h"
#include "index/centroid_page.h"
#include "mkt_test.h"
#include "quant/rabitq.h"

TEST_GROUP(CentroidPage);
TEST_MEMCTX_FIXTURE();

/* ----------------------------------------------------------------
 * Struct size tests
 * ---------------------------------------------------------------- */

TEST(item_pointer_data_size)
{
	ASSERT_EQ(
			6,
			sizeof(ItemPointerData),
			"ItemPointerData must be 6 bytes (PG binary compat)");
}

TEST(centroid_entry_meta_size)
{
	ASSERT_EQ(
			16,
			sizeof(MktCentroidEntryMeta),
			"MktCentroidEntryMeta must be 16 bytes");
}

TEST(centroid_page_opaque_size)
{
	ASSERT_EQ(
			16,
			sizeof(MktCentroidPageOpaque),
			"MktCentroidPageOpaque must be 16 bytes");
}

/* ----------------------------------------------------------------
 * Page capacity tests
 * ---------------------------------------------------------------- */

TEST(page_capacity_768d)
{
	uint32_t max = mkt_centroid_max_entries(768);
	/* Per entry: 16 (meta) + 4 (f_add) + 4 (f_rescale) + 96 (bits) = 120
	 * Usable: 8192 - 24 - 16 = 8152
	 * 8152 / 120 = 67 */
	ASSERT_EQ(67, max, "768d should fit 67 entries per page");
}

TEST(page_capacity_128d)
{
	uint32_t max = mkt_centroid_max_entries(128);
	/* Per entry: 16 + 4 + 4 + 16 = 40
	 * 8152 / 40 = 203 */
	ASSERT_EQ(203, max, "128d should fit 203 entries per page");
}

TEST(page_capacity_1536d)
{
	uint32_t max = mkt_centroid_max_entries(1536);
	/* Per entry: 16 + 4 + 4 + 192 = 216
	 * 8152 / 216 = 37 */
	ASSERT_EQ(37, max, "1536d should fit 37 entries per page");
}

/* ----------------------------------------------------------------
 * ItemPointerData round-trip tests
 * ---------------------------------------------------------------- */

TEST(item_pointer_set_get)
{
	ItemPointerData tip;
	ItemPointerSet(&tip, 42, 7);

	ASSERT_EQ(42, ItemPointerGetBlockNumber(&tip), "block number round-trip");
	ASSERT_EQ(7, ItemPointerGetOffsetNumber(&tip), "offset number round-trip");
}

TEST(item_pointer_large_block)
{
	ItemPointerData tip;
	BlockNumber		blk = 0x00ABCDEF;
	ItemPointerSet(&tip, blk, 100);

	ASSERT_EQ(
			blk,
			ItemPointerGetBlockNumber(&tip),
			"large block number round-trip");
	ASSERT_EQ(
			100, ItemPointerGetOffsetNumber(&tip), "offset with large block");
}

TEST(item_pointer_max_values)
{
	ItemPointerData tip;
	BlockNumber		max_blk = 0xFFFFFFFE; /* InvalidBlockNumber - 1 */
	OffsetNumber	max_off = 0xFFFF;
	ItemPointerSet(&tip, max_blk, max_off);

	ASSERT_EQ(
			max_blk, ItemPointerGetBlockNumber(&tip), "max block round-trip");
	ASSERT_EQ(
			max_off,
			ItemPointerGetOffsetNumber(&tip),
			"max offset round-trip");
}

/* ----------------------------------------------------------------
 * SoA region non-overlap test
 * ---------------------------------------------------------------- */

TEST(soa_regions_no_overlap)
{
	Dimension dim		   = 768;
	uint32_t  max		   = mkt_centroid_max_entries(dim);
	uint32_t  packed_bytes = MKT_RABITQ_BYTES(dim);

	/* Compute region boundaries */
	size_t meta_start	= SizeOfPageHeaderData;
	size_t meta_end		= meta_start + max * sizeof(MktCentroidEntryMeta);
	size_t f_add_start	= meta_end;
	size_t f_add_end	= f_add_start + max * sizeof(float);
	size_t f_resc_start = f_add_end;
	size_t f_resc_end	= f_resc_start + max * sizeof(float);
	size_t bits_start	= f_resc_end;
	size_t bits_end		= bits_start + (size_t)max * packed_bytes;
	size_t opaque_start = BLCKSZ - sizeof(MktCentroidPageOpaque);

	char msg[256];

	/* Regions must not overlap */
	snprintf(
			msg,
			sizeof(msg),
			"meta [%zu,%zu) must not overlap f_add [%zu,%zu)",
			meta_start,
			meta_end,
			f_add_start,
			f_add_end);
	ASSERT_TRUE(meta_end <= f_add_start, msg);

	snprintf(
			msg,
			sizeof(msg),
			"f_add [%zu,%zu) must not overlap f_rescale [%zu,%zu)",
			f_add_start,
			f_add_end,
			f_resc_start,
			f_resc_end);
	ASSERT_TRUE(f_add_end <= f_resc_start, msg);

	snprintf(
			msg,
			sizeof(msg),
			"f_rescale [%zu,%zu) must not overlap bits [%zu,%zu)",
			f_resc_start,
			f_resc_end,
			bits_start,
			bits_end);
	ASSERT_TRUE(f_resc_end <= bits_start, msg);

	snprintf(
			msg,
			sizeof(msg),
			"bits [%zu,%zu) must not overlap opaque [%zu,%zu)",
			bits_start,
			bits_end,
			opaque_start,
			opaque_start + sizeof(MktCentroidPageOpaque));
	ASSERT_TRUE(bits_end <= opaque_start, msg);
}

/* ----------------------------------------------------------------
 * Page init and add round-trip
 * ---------------------------------------------------------------- */

TEST(page_init_basic)
{
	char page_buf[BLCKSZ];
	Page page = page_buf;

	mkt_centroid_page_init(page, 0, 0);

	MktCentroidPageOpaque *opaque = MKT_CENTROID_OPAQUE(page);
	ASSERT_EQ(
			InvalidBlockNumber,
			opaque->next_blkno,
			"next_blkno should be invalid");
	ASSERT_EQ(0, opaque->entry_count, "entry_count should be 0");
	ASSERT_EQ(0, opaque->level, "level should be 0");
	ASSERT_EQ(
			MKT_CENTROID_PAGE_ID,
			opaque->page_id,
			"page_id should be MKT_CENTROID_PAGE_ID");
	ASSERT_EQ(0, opaque->first_global_idx, "first_global_idx should be 0");
}

TEST(page_init_with_level)
{
	char page_buf[BLCKSZ];
	Page page = page_buf;

	mkt_centroid_page_init(page, 2, 64);

	MktCentroidPageOpaque *opaque = MKT_CENTROID_OPAQUE(page);
	ASSERT_EQ(2, opaque->level, "level should be 2");
	ASSERT_EQ(64, opaque->first_global_idx, "first_global_idx should be 64");
}

TEST(page_add_single_entry)
{
	Dimension dim = 128;

	/* Create RaBitQ params for encoding */
	RaBitQParams *params = mkt_rabitq_create(dim, 42);
	ASSERT_NOT_NULL(params, "params created");

	/* Create a test vector and encode */
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

	/* Set up page */
	char page_buf[BLCKSZ];
	Page page = page_buf;
	mkt_centroid_page_init(page, 0, 0);

	/* Add entry */
	ItemPointerData tid;
	ItemPointerSet(&tid, 5, 10);

	bool added = mkt_centroid_page_add(
			page, dim, 42, 8, MKT_CENTROID_FLAG_LEAF, &tid, encoded);
	ASSERT_TRUE(added, "entry should be added");

	/* Verify */
	MktCentroidPageOpaque *opaque = MKT_CENTROID_OPAQUE(page);
	ASSERT_EQ(1, opaque->entry_count, "entry_count should be 1");

	uint32_t			  max  = mkt_centroid_max_entries(dim);
	MktCentroidEntryMeta *meta = MKT_CENTROID_META(page);
	ASSERT_EQ(42, meta[0].child_blkno, "child_blkno");
	ASSERT_EQ(8, meta[0].child_count, "child_count");
	ASSERT_EQ(MKT_CENTROID_FLAG_LEAF, meta[0].flags, "flags");
	ASSERT_EQ(
			5, ItemPointerGetBlockNumber(&meta[0].medoid_tid), "medoid block");
	ASSERT_EQ(
			10,
			ItemPointerGetOffsetNumber(&meta[0].medoid_tid),
			"medoid offset");

	float *f_add	 = MKT_CENTROID_F_ADD(page, max);
	float *f_rescale = MKT_CENTROID_F_RESCALE(page, max);
	ASSERT_FLOAT_EQ(encoded->f_add, f_add[0], 1e-6f, "f_add round-trip");
	ASSERT_FLOAT_EQ(
			encoded->f_rescale, f_rescale[0], 1e-6f, "f_rescale round-trip");

	uint8_t *bits		  = MKT_CENTROID_BITS(page, max);
	uint32_t packed_bytes = MKT_RABITQ_BYTES(dim);
	int		 bits_match	  = memcmp(encoded->bits, bits, packed_bytes) == 0;
	ASSERT_TRUE(bits_match, "bits round-trip");

	mkt_free(encoded);
	mkt_rabitq_destroy(params);
}

TEST(page_add_fill_to_capacity)
{
	Dimension dim = 768;
	uint32_t  max = mkt_centroid_max_entries(dim);

	RaBitQParams *params = mkt_rabitq_create(dim, 42);
	ASSERT_NOT_NULL(params, "params created");

	char page_buf[BLCKSZ];
	Page page = page_buf;
	mkt_centroid_page_init(page, 0, 0);

	/* Create a test vector */
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

	/* Fill page to capacity */
	for (uint32_t i = 0; i < max; i++)
	{
		ItemPointerData tid;
		ItemPointerSet(&tid, 0, (OffsetNumber)i);

		bool added = mkt_centroid_page_add(page, dim, i, 1, 0, &tid, encoded);

		char msg[64];
		snprintf(msg, sizeof(msg), "entry %u should be added", i);
		ASSERT_TRUE(added, msg);
	}

	/* Verify full */
	MktCentroidPageOpaque *opaque = MKT_CENTROID_OPAQUE(page);
	ASSERT_EQ(max, opaque->entry_count, "page should be full");

	/* One more should fail */
	ItemPointerData tid;
	ItemPointerSet(&tid, 0, 0);
	bool added = mkt_centroid_page_add(page, dim, 999, 1, 0, &tid, encoded);
	ASSERT_FALSE(added, "page should be full");

	mkt_free(encoded);
	mkt_rabitq_destroy(params);
}

TEST(page_has_room)
{
	Dimension dim = 128;
	char	  page_buf[BLCKSZ];
	Page	  page = page_buf;
	mkt_centroid_page_init(page, 0, 0);

	ASSERT_TRUE(
			mkt_centroid_page_has_room(page, dim),
			"empty page should have room");

	/* Fill it */
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

	uint32_t max = mkt_centroid_max_entries(dim);
	for (uint32_t i = 0; i < max; i++)
	{
		ItemPointerData tid;
		ItemPointerSet(&tid, 0, (OffsetNumber)i);
		mkt_centroid_page_add(page, dim, i, 1, 0, &tid, encoded);
	}

	ASSERT_FALSE(
			mkt_centroid_page_has_room(page, dim),
			"full page should not have room");

	mkt_free(encoded);
	mkt_rabitq_destroy(params);
}

/* ----------------------------------------------------------------
 * Multiple entry round-trip with distinct values
 * ---------------------------------------------------------------- */

TEST(page_multi_entry_round_trip)
{
	Dimension dim		   = 64;
	uint32_t  max		   = mkt_centroid_max_entries(dim);
	uint32_t  packed_bytes = MKT_RABITQ_BYTES(dim);
	const int count		   = 10;

	RaBitQParams *params = mkt_rabitq_create(dim, 42);
	ASSERT_NOT_NULL(params, "params created");

	char page_buf[BLCKSZ];
	Page page = page_buf;
	mkt_centroid_page_init(page, 1, 100);

	float *centroid = mkt_alloc(dim * sizeof(float));
	for (Dimension i = 0; i < dim; i++)
		centroid[i] = 0.0f;
	VectorRef cent_ref = {.data = centroid, .dim = dim};

	/* Add entries with distinct vectors */
	RaBitQData **encodings = mkt_alloc(count * sizeof(void *));
	for (int e = 0; e < count; e++)
	{
		float *vec = mkt_alloc(dim * sizeof(float));
		for (Dimension i = 0; i < dim; i++)
			vec[i] = (float)((i * 17 + e * 31) % 100 - 50) / 10.0f;

		VectorRef vec_ref = {.data = vec, .dim = dim};
		encodings[e]	  = mkt_rabitq_encode(params, vec_ref, cent_ref);
		ASSERT_NOT_NULL(encodings[e], "encoding succeeded");

		ItemPointerData tid;
		ItemPointerSet(&tid, (BlockNumber)e, (OffsetNumber)(e + 1));

		bool added = mkt_centroid_page_add(
				page, dim, 100 + e, e + 1, 0, &tid, encodings[e]);
		ASSERT_TRUE(added, "entry added");
	}

	/* Verify all entries */
	MktCentroidEntryMeta *meta		= MKT_CENTROID_META(page);
	float				 *f_add		= MKT_CENTROID_F_ADD(page, max);
	float				 *f_rescale = MKT_CENTROID_F_RESCALE(page, max);
	uint8_t				 *bits		= MKT_CENTROID_BITS(page, max);

	for (int e = 0; e < count; e++)
	{
		char msg[128];

		snprintf(msg, sizeof(msg), "entry %d child_blkno", e);
		ASSERT_EQ((BlockNumber)(100 + e), meta[e].child_blkno, msg);

		snprintf(msg, sizeof(msg), "entry %d child_count", e);
		ASSERT_EQ((uint16_t)(e + 1), meta[e].child_count, msg);

		snprintf(msg, sizeof(msg), "entry %d medoid block", e);
		ASSERT_EQ(
				(BlockNumber)e,
				ItemPointerGetBlockNumber(&meta[e].medoid_tid),
				msg);

		snprintf(msg, sizeof(msg), "entry %d medoid offset", e);
		ASSERT_EQ(
				(OffsetNumber)(e + 1),
				ItemPointerGetOffsetNumber(&meta[e].medoid_tid),
				msg);

		snprintf(msg, sizeof(msg), "entry %d f_add", e);
		ASSERT_FLOAT_EQ(encodings[e]->f_add, f_add[e], 1e-6f, msg);

		snprintf(msg, sizeof(msg), "entry %d f_rescale", e);
		ASSERT_FLOAT_EQ(encodings[e]->f_rescale, f_rescale[e], 1e-6f, msg);

		snprintf(msg, sizeof(msg), "entry %d bits", e);
		int bits_match = memcmp(encodings[e]->bits,
								bits + (size_t)e * packed_bytes,
								packed_bytes) == 0;
		ASSERT_TRUE(bits_match, msg);

		mkt_free(encodings[e]);
	}

	mkt_free(encodings);
	mkt_rabitq_destroy(params);
}
