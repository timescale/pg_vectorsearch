/*
 * test_centroid_page.c - Unit tests for centroid page layout and operations
 *
 * Tests cover:
 * - Struct size verification (binary compatibility with PG)
 * - Page capacity calculation
 * - Bidirectional region non-overlap
 * - Page init/add round-trip
 * - ItemPointerData accessors
 */

#include <string.h>

#include "core/memory.h"
#include "index/centroid_page.h"
#include "quant/rabitq.h"
#include "standalone/pg_compat.h"
#include "vs_test.h"

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
			8,
			sizeof(PrismCentroidEntryMeta),
			"PrismCentroidEntryMeta must be 8 bytes");
}

TEST(centroid_page_opaque_size)
{
	ASSERT_EQ(
			12,
			sizeof(PrismCentroidPageOpaque),
			"PrismCentroidPageOpaque must be 12 bytes");
}

/* ----------------------------------------------------------------
 * Page capacity tests
 * ---------------------------------------------------------------- */

TEST(page_capacity_768d)
{
	uint32_t max = prism_centroid_max_entries(768);
	/* Per entry: 8 (meta) + 104 (RaBitQData: 8 header + 96 bits) = 112
	 * Usable: 8192 - 24 - 12 = 8156
	 * 8156 / 112 = 72 */
	ASSERT_EQ(72, max, "768d should fit 72 entries per page");
}

TEST(page_capacity_128d)
{
	uint32_t max = prism_centroid_max_entries(128);
	/* Per entry: 8 (meta) + 24 (RaBitQData: 8 header + 16 bits) = 32
	 * 8156 / 32 = 254 */
	ASSERT_EQ(254, max, "128d should fit 254 entries per page");
}

TEST(page_capacity_1536d)
{
	uint32_t max = prism_centroid_max_entries(1536);
	/* Per entry: 8 (meta) + 200 (RaBitQData: 8 header + 192 bits) = 208
	 * 8156 / 208 = 39 */
	ASSERT_EQ(39, max, "1536d should fit 39 entries per page");
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
 * Bidirectional region non-overlap test
 * ---------------------------------------------------------------- */

TEST(bidir_regions_no_overlap)
{
	Dimension dim		= 768;
	uint32_t  max		= prism_centroid_max_entries(dim);
	uint32_t  data_size = VS_RABITQ_DATA_SIZE(dim);

	/* Forward region: metadata */
	size_t meta_start = SizeOfPageHeaderData;
	size_t meta_end	  = meta_start + max * sizeof(PrismCentroidEntryMeta);

	/* Backward region: RaBitQData entries */
	size_t opaque_start = BLCKSZ - sizeof(PrismCentroidPageOpaque);
	size_t data_start	= opaque_start - (size_t)max * data_size;

	ASSERT_TRUE(
			meta_end <= data_start,
			"meta region must not overlap data region");
}

/* ----------------------------------------------------------------
 * Page init and add round-trip
 * ---------------------------------------------------------------- */

TEST(page_init_basic)
{
	char page_buf[BLCKSZ];
	Page page = page_buf;

	prism_centroid_page_init(page, 0);

	PrismCentroidPageOpaque *opaque = PRISM_CENTROID_OPAQUE(page);
	ASSERT_EQ(
			InvalidBlockNumber,
			opaque->next_blkno,
			"next_blkno should be invalid");
	ASSERT_EQ(0, opaque->entry_count, "entry_count should be 0");
	ASSERT_EQ(0, opaque->level, "level should be 0");
	ASSERT_EQ(
			PRISM_CENTROID_PAGE_ID,
			opaque->page_id,
			"page_id should be PRISM_CENTROID_PAGE_ID");
}

TEST(page_init_with_level)
{
	char page_buf[BLCKSZ];
	Page page = page_buf;

	prism_centroid_page_init(page, 2);

	PrismCentroidPageOpaque *opaque = PRISM_CENTROID_OPAQUE(page);
	ASSERT_EQ(2, opaque->level, "level should be 2");
}

TEST(page_add_single_entry)
{
	Dimension dim = 128;

	/* Create RaBitQ params for encoding */
	RaBitQParams *params = vs_rabitq_create(dim, 42);
	ASSERT_NOT_NULL(params, "params created");

	/* Create a test vector and encode */
	float *vec		= vs_alloc(dim * sizeof(float));
	float *centroid = vs_alloc(dim * sizeof(float));
	for (Dimension i = 0; i < dim; i++)
	{
		vec[i]		= (float)((i * 17 + 3) % 100 - 50) / 10.0f;
		centroid[i] = 0.0f;
	}

	Vec32Ref	vec_ref	 = {.data = vec, .dim = dim};
	Vec32Ref	cent_ref = {.data = centroid, .dim = dim};
	RaBitQData *encoded	 = vs_rabitq_encode(params, vec_ref, cent_ref);
	ASSERT_NOT_NULL(encoded, "encoding succeeded");

	/* Set up page */
	char page_buf[BLCKSZ];
	Page page = page_buf;
	prism_centroid_page_init(page, 0);

	/* Add entry */
	bool added = prism_centroid_page_add(
			page, dim, 42, 8, PRISM_CENTROID_FLAG_LEAF, encoded);
	ASSERT_TRUE(added, "entry should be added");

	/* Verify */
	PrismCentroidPageOpaque *opaque = PRISM_CENTROID_OPAQUE(page);
	ASSERT_EQ(1, opaque->entry_count, "entry_count should be 1");

	const PrismCentroidEntryMeta *meta = prism_centroid_meta(page, 0);
	ASSERT_EQ(42, meta->child_blkno, "child_blkno");
	ASSERT_EQ(8, meta->child_count, "child_count");
	ASSERT_EQ(PRISM_CENTROID_FLAG_LEAF, meta->flags, "flags");

	/* Verify RaBitQData round-trip via direct pointer */
	const RaBitQData *data		   = prism_centroid_data(page, 0, dim);
	uint32_t		  packed_bytes = VS_RABITQ_BYTES(dim);
	ASSERT_FLOAT_EQ(encoded->f_add, data->f_add, 1e-6f, "f_add round-trip");
	ASSERT_FLOAT_EQ(
			encoded->f_rescale,
			data->f_rescale,
			1e-6f,
			"f_rescale round-trip");
	int bits_match = memcmp(encoded->bits, data->bits, packed_bytes) == 0;
	ASSERT_TRUE(bits_match, "bits round-trip");

	vs_free(encoded);
	vs_rabitq_destroy(params);
}

TEST(page_add_fill_to_capacity)
{
	Dimension dim = 768;
	uint32_t  max = prism_centroid_max_entries(dim);

	RaBitQParams *params = vs_rabitq_create(dim, 42);
	ASSERT_NOT_NULL(params, "params created");

	char page_buf[BLCKSZ];
	Page page = page_buf;
	prism_centroid_page_init(page, 0);

	/* Create a test vector */
	float *vec		= vs_alloc(dim * sizeof(float));
	float *centroid = vs_alloc(dim * sizeof(float));
	for (Dimension i = 0; i < dim; i++)
	{
		vec[i]		= (float)((i * 7 + 1) % 100 - 50) / 10.0f;
		centroid[i] = 0.0f;
	}

	Vec32Ref	vec_ref	 = {.data = vec, .dim = dim};
	Vec32Ref	cent_ref = {.data = centroid, .dim = dim};
	RaBitQData *encoded	 = vs_rabitq_encode(params, vec_ref, cent_ref);
	ASSERT_NOT_NULL(encoded, "encoding succeeded");

	/* Fill page to capacity */
	for (uint32_t i = 0; i < max; i++)
	{
		bool added = prism_centroid_page_add(page, dim, i, 1, 0, encoded);

		char msg[64];
		snprintf(msg, sizeof(msg), "entry %u should be added", i);
		ASSERT_TRUE(added, msg);
	}

	/* Verify full */
	PrismCentroidPageOpaque *opaque = PRISM_CENTROID_OPAQUE(page);
	ASSERT_EQ(max, opaque->entry_count, "page should be full");

	/* One more should fail */
	bool added = prism_centroid_page_add(page, dim, 999, 1, 0, encoded);
	ASSERT_FALSE(added, "page should be full");

	vs_free(encoded);
	vs_rabitq_destroy(params);
}

TEST(page_has_room)
{
	Dimension dim = 128;
	char	  page_buf[BLCKSZ];
	Page	  page = page_buf;
	prism_centroid_page_init(page, 0);

	ASSERT_TRUE(
			prism_centroid_page_has_room(page, dim, false),
			"empty page should have room");

	/* Fill it */
	RaBitQParams *params   = vs_rabitq_create(dim, 42);
	float		 *vec	   = vs_alloc(dim * sizeof(float));
	float		 *centroid = vs_alloc(dim * sizeof(float));
	for (Dimension i = 0; i < dim; i++)
	{
		vec[i]		= (float)i / 10.0f;
		centroid[i] = 0.0f;
	}

	Vec32Ref	vec_ref	 = {.data = vec, .dim = dim};
	Vec32Ref	cent_ref = {.data = centroid, .dim = dim};
	RaBitQData *encoded	 = vs_rabitq_encode(params, vec_ref, cent_ref);

	uint32_t max = prism_centroid_max_entries(dim);
	for (uint32_t i = 0; i < max; i++)
		prism_centroid_page_add(page, dim, i, 1, 0, encoded);

	ASSERT_FALSE(
			prism_centroid_page_has_room(page, dim, false),
			"full page should not have room");

	vs_free(encoded);
	vs_rabitq_destroy(params);
}

/* ----------------------------------------------------------------
 * Multiple entry round-trip with distinct values
 * ---------------------------------------------------------------- */

TEST(page_multi_entry_round_trip)
{
	Dimension dim		   = 64;
	uint32_t  packed_bytes = VS_RABITQ_BYTES(dim);
	const int count		   = 10;

	RaBitQParams *params = vs_rabitq_create(dim, 42);
	ASSERT_NOT_NULL(params, "params created");

	char page_buf[BLCKSZ];
	Page page = page_buf;
	prism_centroid_page_init(page, 1);

	float *centroid = vs_alloc(dim * sizeof(float));
	for (Dimension i = 0; i < dim; i++)
		centroid[i] = 0.0f;
	Vec32Ref cent_ref = {.data = centroid, .dim = dim};

	/* Add entries with distinct vectors */
	RaBitQData **encodings = vs_alloc(count * sizeof(void *));
	for (int e = 0; e < count; e++)
	{
		float *vec = vs_alloc(dim * sizeof(float));
		for (Dimension i = 0; i < dim; i++)
			vec[i] = (float)((i * 17 + e * 31) % 100 - 50) / 10.0f;

		Vec32Ref vec_ref = {.data = vec, .dim = dim};
		encodings[e]	 = vs_rabitq_encode(params, vec_ref, cent_ref);
		ASSERT_NOT_NULL(encodings[e], "encoding succeeded");

		bool added = prism_centroid_page_add(
				page, dim, 100 + e, e + 1, 0, encodings[e]);
		ASSERT_TRUE(added, "entry added");
	}

	/* Verify all entries */
	for (int e = 0; e < count; e++)
	{
		const PrismCentroidEntryMeta *meta = prism_centroid_meta(page, e);
		char						  msg[128];

		snprintf(msg, sizeof(msg), "entry %d child_blkno", e);
		ASSERT_EQ((BlockNumber)(100 + e), meta->child_blkno, msg);

		snprintf(msg, sizeof(msg), "entry %d child_count", e);
		ASSERT_EQ((uint16_t)(e + 1), meta->child_count, msg);

		/* Verify RaBitQData via direct pointer */
		const RaBitQData *data = prism_centroid_data(page, e, dim);

		snprintf(msg, sizeof(msg), "entry %d f_add", e);
		ASSERT_FLOAT_EQ(encodings[e]->f_add, data->f_add, 1e-6f, msg);

		snprintf(msg, sizeof(msg), "entry %d f_rescale", e);
		ASSERT_FLOAT_EQ(encodings[e]->f_rescale, data->f_rescale, 1e-6f, msg);

		snprintf(msg, sizeof(msg), "entry %d bits", e);
		int bits_match = memcmp(encodings[e]->bits,
								data->bits,
								packed_bytes) == 0;
		ASSERT_TRUE(bits_match, msg);

		vs_free(encodings[e]);
	}

	vs_free(encodings);
	vs_rabitq_destroy(params);
}

/* ----------------------------------------------------------------
 * Multi-format page tests (float32 and float16)
 * ---------------------------------------------------------------- */

TEST(page_capacity_float_768d)
{
	uint32_t max =
			prism_centroid_max_entries_fmt(768, PRISM_CENTROID_FMT_FLOAT);
	/* Per entry: 8 (meta) + 3072 (768*4) = 3080
	 * Usable: 8156
	 * 8156 / 3080 = 2 */
	ASSERT_EQ(2, max, "768d float should fit 2 entries per page");
}

TEST(page_capacity_float_128d)
{
	uint32_t max =
			prism_centroid_max_entries_fmt(128, PRISM_CENTROID_FMT_FLOAT);
	/* Per entry: 8 + 512 = 520
	 * 8156 / 520 = 15 */
	ASSERT_EQ(15, max, "128d float should fit 15 entries per page");
}

TEST(page_capacity_half_768d)
{
	uint32_t max =
			prism_centroid_max_entries_fmt(768, PRISM_CENTROID_FMT_HALF);
	/* Per entry: 8 + 1536 = 1544
	 * 8156 / 1544 = 5 */
	ASSERT_EQ(5, max, "768d half should fit 5 entries per page");
}

TEST(page_capacity_half_128d)
{
	uint32_t max =
			prism_centroid_max_entries_fmt(128, PRISM_CENTROID_FMT_HALF);
	/* Per entry: 8 (meta) + 256 (data) = 264
	 * 8156 / 264 = 30 */
	ASSERT_EQ(30, max, "128d half should fit 30 entries per page");
}

TEST(page_init_fmt_stores_format)
{
	char page_buf[BLCKSZ];
	Page page = page_buf;

	prism_centroid_page_init_fmt(page, 0, PRISM_CENTROID_FMT_FLOAT);
	ASSERT_EQ(
			PRISM_CENTROID_FMT_FLOAT,
			prism_centroid_page_format(page),
			"float format stored in opaque");

	prism_centroid_page_init_fmt(page, 1, PRISM_CENTROID_FMT_HALF);
	ASSERT_EQ(
			PRISM_CENTROID_FMT_HALF,
			prism_centroid_page_format(page),
			"half format stored in opaque");

	prism_centroid_page_init_fmt(page, 2, PRISM_CENTROID_FMT_RABITQ);
	ASSERT_EQ(
			PRISM_CENTROID_FMT_RABITQ,
			prism_centroid_page_format(page),
			"rabitq format stored in opaque");
}

TEST(page_init_default_is_rabitq)
{
	char page_buf[BLCKSZ];
	Page page = page_buf;

	prism_centroid_page_init(page, 0);
	ASSERT_EQ(
			PRISM_CENTROID_FMT_RABITQ,
			prism_centroid_page_format(page),
			"default init should be rabitq format");
}

TEST(page_float_add_round_trip)
{
	Dimension dim = 128;

	char page_buf[BLCKSZ];
	Page page = page_buf;
	prism_centroid_page_init_fmt(page, 0, PRISM_CENTROID_FMT_FLOAT);

	/* Create test vector */
	float *vec = vs_alloc(dim * sizeof(float));
	for (Dimension i = 0; i < dim; i++)
		vec[i] = (float)((i * 17 + 3) % 100 - 50) / 10.0f;

	bool added = prism_centroid_page_add_entry(
			page, dim, 42, 8, PRISM_CENTROID_FLAG_LEAF, vec);
	ASSERT_TRUE(added, "float entry should be added");

	/* Verify metadata */
	PrismCentroidPageOpaque *opaque = PRISM_CENTROID_OPAQUE(page);
	ASSERT_EQ(1, opaque->entry_count, "entry_count should be 1");

	const PrismCentroidEntryMeta *meta = prism_centroid_meta(page, 0);
	ASSERT_EQ(42, meta->child_blkno, "child_blkno");
	ASSERT_EQ(8, meta->child_count, "child_count");
	ASSERT_EQ(PRISM_CENTROID_FLAG_LEAF, meta->flags, "flags");

	/* Verify float data round-trip */
	const float *data = prism_centroid_float_data(page, 0, dim);
	for (Dimension i = 0; i < dim; i++)
	{
		char msg[64];
		snprintf(msg, sizeof(msg), "float[%u] round-trip", i);
		ASSERT_FLOAT_EQ(vec[i], data[i], 1e-6f, msg);
	}

	vs_free(vec);
}

TEST(page_half_add_round_trip)
{
	Dimension dim = 128;

	char page_buf[BLCKSZ];
	Page page = page_buf;
	prism_centroid_page_init_fmt(page, 0, PRISM_CENTROID_FMT_HALF);

	/* Create test vector as halfs */
	half *hvec = vs_alloc(dim * sizeof(half));
	for (Dimension i = 0; i < dim; i++)
		hvec[i] = vs_float_to_half((float)((i * 17 + 3) % 100 - 50) / 10.0f);

	bool added = prism_centroid_page_add_entry(page, dim, 99, 4, 0, hvec);
	ASSERT_TRUE(added, "half entry should be added");

	/* Verify metadata */
	PrismCentroidPageOpaque *opaque = PRISM_CENTROID_OPAQUE(page);
	ASSERT_EQ(1, opaque->entry_count, "entry_count should be 1");

	const PrismCentroidEntryMeta *meta = prism_centroid_meta(page, 0);
	ASSERT_EQ(99, meta->child_blkno, "child_blkno");
	ASSERT_EQ(4, meta->child_count, "child_count");

	/* Verify half data round-trip (bit-exact) */
	const half *data	   = prism_centroid_half_data(page, 0, dim);
	int			bits_match = memcmp(hvec, data, dim * sizeof(half)) == 0;
	ASSERT_TRUE(bits_match, "half vector bit-exact round-trip");

	vs_free(hvec);
}

TEST(page_float_fill_to_capacity)
{
	Dimension dim = 768;
	uint32_t  max =
			prism_centroid_max_entries_fmt(dim, PRISM_CENTROID_FMT_FLOAT);

	char page_buf[BLCKSZ];
	Page page = page_buf;
	prism_centroid_page_init_fmt(page, 0, PRISM_CENTROID_FMT_FLOAT);

	float *vec = vs_alloc(dim * sizeof(float));
	for (Dimension i = 0; i < dim; i++)
		vec[i] = (float)i / 100.0f;

	for (uint32_t i = 0; i < max; i++)
	{
		bool added = prism_centroid_page_add_entry(page, dim, i, 1, 0, vec);

		char msg[64];
		snprintf(msg, sizeof(msg), "float entry %u added", i);
		ASSERT_TRUE(added, msg);
	}

	/* Verify full */
	PrismCentroidPageOpaque *opaque = PRISM_CENTROID_OPAQUE(page);
	ASSERT_EQ(max, opaque->entry_count, "page should be full");

	/* One more should fail */
	bool added = prism_centroid_page_add_entry(page, dim, 999, 1, 0, vec);
	ASSERT_FALSE(added, "full float page should reject");

	ASSERT_FALSE(
			prism_centroid_page_has_room(page, dim, false),
			"full float page has_room should be false");

	vs_free(vec);
}

TEST(page_half_multi_entry_round_trip)
{
	Dimension dim	= 64;
	const int count = 10;

	char page_buf[BLCKSZ];
	Page page = page_buf;
	prism_centroid_page_init_fmt(page, 1, PRISM_CENTROID_FMT_HALF);

	half **vectors = vs_alloc(count * sizeof(void *));

	for (int e = 0; e < count; e++)
	{
		vectors[e] = vs_alloc(dim * sizeof(half));
		for (Dimension i = 0; i < dim; i++)
			vectors[e][i] = vs_float_to_half(
					(float)((i * 17 + e * 31) % 100 - 50) / 10.0f);

		bool added = prism_centroid_page_add_entry(
				page, dim, 100 + e, e + 1, 0, vectors[e]);
		ASSERT_TRUE(added, "half entry added");
	}

	/* Verify all entries */
	for (int e = 0; e < count; e++)
	{
		const PrismCentroidEntryMeta *meta = prism_centroid_meta(page, e);
		char						  msg[128];

		snprintf(msg, sizeof(msg), "entry %d child_blkno", e);
		ASSERT_EQ((BlockNumber)(100 + e), meta->child_blkno, msg);

		const half *data = prism_centroid_half_data(page, e, dim);
		snprintf(msg, sizeof(msg), "entry %d half bits", e);
		int bits_match = memcmp(vectors[e], data, dim * sizeof(half)) == 0;
		ASSERT_TRUE(bits_match, msg);

		vs_free(vectors[e]);
	}

	vs_free(vectors);
}

TEST(page_entry_data_generic_accessor)
{
	Dimension dim = 64;

	char page_buf[BLCKSZ];
	Page page = page_buf;

	/* Float format: generic and typed return same address */
	prism_centroid_page_init_fmt(page, 0, PRISM_CENTROID_FMT_FLOAT);
	float *fvec = vs_alloc(dim * sizeof(float));
	for (Dimension i = 0; i < dim; i++)
		fvec[i] = (float)i;

	prism_centroid_page_add_entry(page, dim, 0, 1, 0, fvec);

	const void	*g1 = prism_centroid_entry_data(page, 0, dim);
	const float *t1 = prism_centroid_float_data(page, 0, dim);
	ASSERT_EQ(
			(uintptr_t)g1,
			(uintptr_t)t1,
			"generic and typed float same address");

	/* Half format: generic and typed return same address */
	prism_centroid_page_init_fmt(page, 0, PRISM_CENTROID_FMT_HALF);
	half *hvec = vs_alloc(dim * sizeof(half));
	for (Dimension i = 0; i < dim; i++)
		hvec[i] = vs_float_to_half((float)i);

	prism_centroid_page_add_entry(page, dim, 0, 1, 0, hvec);

	const void *g2 = prism_centroid_entry_data(page, 0, dim);
	const half *t2 = prism_centroid_half_data(page, 0, dim);
	ASSERT_EQ(
			(uintptr_t)g2,
			(uintptr_t)t2,
			"generic and typed half same address");

	vs_free(fvec);
	vs_free(hvec);
}
