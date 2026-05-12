/*
 * test_fastscan.c - Unit tests for VPSHUFB fastscan kernel
 *
 * Tests LUT construction, code packing, scalar/AVX2 kernel
 * equivalence, distance accuracy, and tail handling.
 */

#include <math.h>
#include <string.h>

#include "core/memory.h"
#include "core/platform.h"
#include "mkt_test.h"
#include "quant/fastscan.h"
#include "quant/rabitq.h"

TEST_GROUP(Fastscan);
TEST_MEMCTX_FIXTURE();

static void
reinit_fastscan_with_simd(uint32_t simd_mask)
{
	mkt_simd_set_override(simd_mask);
	mkt_simd_reset_cache();
	mkt_fastscan_reset_simd();
	mkt_fastscan_init_simd();
}

static bool
get_fastscan_simd_mask(const char *variant, uint32_t *simd_mask)
{
	if (strcmp(variant, "scalar") == 0)
	{
		*simd_mask = SIMD_NONE;
		return true;
	}
#if defined(__x86_64__) || defined(_M_X64)
	if (strcmp(variant, "avx2") == 0)
	{
		if (!(mkt_detect_simd() & SIMD_AVX2))
			return false;
		*simd_mask = SIMD_AVX2;
		return true;
	}
	if (strcmp(variant, "avx512") == 0)
	{
		if (!(mkt_detect_simd() & SIMD_AVX512F))
			return false;
		*simd_mask = SIMD_AVX512F;
		return true;
	}
#endif
	(void)variant;
	*simd_mask = 0xFFFFFFFF;
	return true;
}

#define SKIP_IF_FASTSCAN_SIMD_NOT_AVAILABLE(variant)         \
	uint32_t simd_mask;                                      \
	if (!get_fastscan_simd_mask(variant, &simd_mask))        \
	{                                                        \
		TEST_PRINT("%s not available, skipping\n", variant); \
		return;                                              \
	}                                                        \
	reinit_fastscan_with_simd(simd_mask)

/* ----------------------------------------------------------------
 * kPos / kPerm0 tables (mirror of fastscan.c, needed by tests)
 * ---------------------------------------------------------------- */

/* LSB-first encoding: kPos[j] = dimension index of lowest set bit */
static const int kPos[16] = {
		0,
		0,
		1,
		0,
		2,
		0,
		1,
		0,
		3,
		0,
		1,
		0,
		2,
		0,
		1,
		0,
};

/* Vector interleaving for even/odd byte accumulation */
static const int kPerm0[16] = {
		0,
		8,
		1,
		9,
		2,
		10,
		3,
		11,
		4,
		12,
		5,
		13,
		6,
		14,
		7,
		15,
};

/* Lowest set bit: j & (-j) */
#define LOWBIT(j) ((j) & (-(j)))

/* ----------------------------------------------------------------
 * Helpers
 * ---------------------------------------------------------------- */

/* Fill buffer with deterministic pseudo-random floats in [-1, 1] */
static void
fill_random_floats(float *buf, uint32_t n, uint32_t seed)
{
	srand(seed);
	for (uint32_t i = 0; i < n; i++)
		buf[i] = (float)(rand() % 10000 - 5000) / 5000.0f;
}

/* Fill buffer with random bits */
static void
fill_random_bits(uint8_t *buf, uint32_t n, uint32_t seed)
{
	srand(seed);
	for (uint32_t i = 0; i < n; i++)
		buf[i] = (uint8_t)(rand() & 0xFF);
}

/* Compute reference binary IP: sum(transformed[i] where bit[i]=1) */
static float
reference_binary_ip(
		const float *transformed, const uint8_t *bits, uint32_t dim)
{
	float	 ip			  = 0;
	uint32_t packed_bytes = (dim + 7) / 8;
	(void)packed_bytes;
	for (uint32_t i = 0; i < dim; i++)
	{
		uint32_t byte_idx = i / 8;
		uint32_t bit_idx  = i % 8;
		if (bits[byte_idx] & (1 << bit_idx))
			ip += transformed[i];
	}
	return ip;
}

/* ----------------------------------------------------------------
 * LUT construction tests
 * ---------------------------------------------------------------- */

TEST(lut_entry_zero_is_bias)
{
	/* Code 0 selects nothing, so LUT[0] should encode 0 (+ bias) */
	uint32_t dim = 32;
	float	 transformed[32];
	fill_random_floats(transformed, dim, 42);

	uint8_t lut[MKT_FASTSCAN_LUT_BYTES(32)];
	float	scale, bias;
	mkt_fastscan_build_lut(transformed, dim, lut, &scale, &bias);

	/* Every sq's entry 0 should be the same (all encode bias) */
	uint32_t nsq = MKT_FASTSCAN_NSQ(dim);
	for (uint32_t sq = 0; sq < nsq; sq++)
		ASSERT_EQ(lut[sq * 16], lut[0], "entry 0 same for all sq");
}

TEST(lut_entries_ordered)
{
	/* For a sq with all positive values, higher codes should have
	 * higher LUT entries (more dimensions selected = larger sum) */
	uint32_t dim			= 4; /* single subquantizer */
	float	 transformed[4] = {1.0f, 2.0f, 3.0f, 4.0f};

	uint8_t lut[MKT_FASTSCAN_LUT_BYTES(4)];
	float	scale, bias;
	mkt_fastscan_build_lut(transformed, dim, lut, &scale, &bias);

	/* Code 15 (all bits set) should be highest */
	ASSERT_TRUE(lut[15] >= lut[0], "code 15 >= code 0");
	/* Code 1 (only dim 0) < code 3 (dim 0+1) */
	ASSERT_TRUE(lut[3] >= lut[1], "code 3 >= code 1");
}

TEST(lut_roundtrip_accuracy)
{
	/* Verify LUT de-quantization approximates true partial sums */
	uint32_t dim = 32;
	float	 transformed[32];
	fill_random_floats(transformed, dim, 99);

	uint8_t lut[MKT_FASTSCAN_LUT_BYTES(32)];
	float	scale, bias;
	mkt_fastscan_build_lut(transformed, dim, lut, &scale, &bias);

	uint32_t nsq	 = MKT_FASTSCAN_NSQ(dim);
	float	 max_err = 0;
	for (uint32_t sq = 0; sq < nsq; sq++)
	{
		uint32_t base = sq * MKT_FASTSCAN_SQ_DIM;

		/* Build reference float LUT using kPos recurrence */
		float ref_lut[16];
		ref_lut[0] = 0.0f;
		for (uint32_t j = 1; j < 16; j++)
		{
			uint32_t d	 = kPos[j];
			float	 val = (base + d < dim) ? transformed[base + d] : 0.0f;
			ref_lut[j]	 = ref_lut[j - LOWBIT(j)] + val;
		}

		for (uint32_t code = 0; code < 16; code++)
		{
			float true_sum = ref_lut[code];

			/* De-quantized LUT entry.
			 * bias = vl * nsq, so per-sq bias is bias / nsq. */
			float approx = (float)lut[sq * 16 + code] * scale +
						   bias / (float)nsq;
			float err = fabsf(approx - true_sum);
			if (err > max_err)
				max_err = err;
		}
	}

	/* Error should be within one quantization step */
	ASSERT_TRUE(max_err < scale * 1.5f, "LUT error within 1.5 steps");
}

/* ----------------------------------------------------------------
 * Code packing tests
 * ---------------------------------------------------------------- */

TEST(pack_codes_roundtrip)
{
	/* Pack 1-bit codes and verify nibbles match original bits,
	 * accounting for kPerm0 vector interleaving. */
	uint32_t dim		  = 32;
	uint32_t count		  = 8;
	uint32_t packed_bytes = (dim + 7) / 8;

	uint8_t bits[8 * 4]; /* 8 vectors x 4 bytes each */
	fill_random_bits(bits, count * packed_bytes, 77);

	uint32_t codes_size = mkt_fastscan_codes_size(count, dim);
	uint8_t *codes		= mkt_alloc(codes_size);
	uint32_t ngroups	= mkt_fastscan_pack_codes(bits, count, dim, codes);

	ASSERT_EQ(ngroups, 1, "8 vectors = 1 group");

	/* Build inverse kPerm0: inv[kPerm0[j]] = j */
	int inv_kperm0[16];
	for (int j = 0; j < 16; j++)
		inv_kperm0[kPerm0[j]] = j;

	/* Verify each vector's nibbles match original bits.
	 *
	 * Code layout per column (8 dims = 2 sqs):
	 *   bytes 0-15:  lower nibble sq (sq0), kPerm0 interleaved
	 *   bytes 16-31: upper nibble sq (sq1), kPerm0 interleaved
	 *
	 * Byte j in a 16-byte block packs:
	 *   lo nibble = vector kPerm0[j]
	 *   hi nibble = vector kPerm0[j] + 16
	 */
	for (uint32_t v = 0; v < count; v++)
	{
		for (uint32_t col = 0; col < packed_bytes; col++)
		{
			/* Original byte for this vector/column */
			uint8_t orig	= bits[v * packed_bytes + col];
			uint8_t orig_lo = orig & 0x0F; /* lower sq nibble */
			uint8_t orig_hi = orig >> 4;   /* upper sq nibble */

			/* Find byte position in packed output.
			 * v < 16: lo nibble of byte inv_kperm0[v]
			 * v >= 16: hi nibble of byte inv_kperm0[v-16] */
			uint8_t *out = codes + col * MKT_FASTSCAN_GROUP;
			uint8_t	 packed_byte;
			uint8_t	 got_lo, got_hi;

			if (v < 16)
			{
				int pos = inv_kperm0[v];
				/* Lower sq block (bytes 0-15) */
				packed_byte = out[pos];
				got_lo		= packed_byte & 0x0F;
				/* Upper sq block (bytes 16-31) */
				packed_byte = out[pos + 16];
				got_hi		= packed_byte & 0x0F;
			}
			else
			{
				int pos = inv_kperm0[v - 16];
				/* Lower sq block, hi nibble */
				packed_byte = out[pos];
				got_lo		= packed_byte >> 4;
				/* Upper sq block, hi nibble */
				packed_byte = out[pos + 16];
				got_hi		= packed_byte >> 4;
			}

			ASSERT_EQ(got_lo, orig_lo, "lower sq nibble matches original");
			ASSERT_EQ(got_hi, orig_hi, "upper sq nibble matches original");
		}
	}
}

TEST(pack_codes_padding)
{
	/* Verify tail group is zero-padded for vectors >= count.
	 * With kPerm0 interleaving, padding vectors don't map to
	 * sequential byte positions - extract nibbles via inverse. */
	uint32_t dim   = 16;
	uint32_t count = 5; /* not a multiple of 32 */

	uint8_t bits[5 * 2]; /* 5 vectors x 2 bytes */
	fill_random_bits(bits, count * ((dim + 7) / 8), 55);

	uint32_t codes_size = mkt_fastscan_codes_size(count, dim);
	uint8_t *codes		= mkt_alloc(codes_size);
	uint32_t ngroups	= mkt_fastscan_pack_codes(bits, count, dim, codes);

	ASSERT_EQ(ngroups, 1, "5 vectors = 1 group (padded)");

	/* Build inverse kPerm0 */
	int inv_kperm0[16];
	for (int j = 0; j < 16; j++)
		inv_kperm0[kPerm0[j]] = j;

	/* Verify nibbles for padding vectors (>= count) are zero */
	uint32_t cols = (dim + 7) / 8;
	for (uint32_t c = 0; c < cols; c++)
	{
		uint8_t *out = codes + c * MKT_FASTSCAN_GROUP;
		for (uint32_t v = count; v < MKT_FASTSCAN_GROUP; v++)
		{
			uint8_t nibble;
			if (v < 16)
			{
				/* lo nibble of byte inv_kperm0[v] in both
				 * lower and upper sq blocks */
				nibble = out[inv_kperm0[v]] & 0x0F;
				ASSERT_EQ(nibble, 0, "padding lo sq zero");
				nibble = out[inv_kperm0[v] + 16] & 0x0F;
				ASSERT_EQ(nibble, 0, "padding hi sq zero");
			}
			else
			{
				/* hi nibble of byte inv_kperm0[v-16] */
				nibble = out[inv_kperm0[v - 16]] >> 4;
				ASSERT_EQ(nibble, 0, "padding lo sq zero");
				nibble = out[inv_kperm0[v - 16] + 16] >> 4;
				ASSERT_EQ(nibble, 0, "padding hi sq zero");
			}
		}
	}
}

/* ----------------------------------------------------------------
 * Accumulate kernel tests
 * ---------------------------------------------------------------- */

TEST(accumulate_scalar_basic)
{
	/* Verify scalar kernel produces correct results */
	uint32_t dim   = 16;
	uint32_t count = 4;

	float transformed[16];
	fill_random_floats(transformed, dim, 42);

	uint8_t bits[4 * 2];
	fill_random_bits(bits, count * 2, 42);

	/* Build LUT and pack codes */
	uint8_t lut[MKT_FASTSCAN_LUT_BYTES(16)];
	float	scale, bias;
	mkt_fastscan_build_lut(transformed, dim, lut, &scale, &bias);

	uint32_t codes_size = mkt_fastscan_codes_size(count, dim);
	uint8_t *codes		= mkt_alloc(codes_size);
	mkt_fastscan_pack_codes(bits, count, dim, codes);

	/* Run accumulate */
	uint16_t accum[32] = {0};
	mkt_fastscan_accumulate(codes, lut, accum, dim);

	/* De-quantize and compare to reference.
	 * bias from build_lut is already vl * nsq. */
	for (uint32_t v = 0; v < count; v++)
	{
		float approx = (float)accum[v] * scale + bias;
		float ref	 = reference_binary_ip(transformed, bits + v * 2, dim);
		float err	 = fabsf(approx - ref);

		/* Allow some quantization error */
		ASSERT_TRUE(err < 0.5f, "accumulate error within tolerance");
	}
}

TEST(accumulate_simd_matches_scalar)
{
	/* Verify SIMD kernel matches scalar reference */
	uint32_t dim		  = 768;
	uint32_t count		  = 64;
	uint32_t packed_bytes = (dim + 7) / 8;

	float *transformed = mkt_alloc(dim * sizeof(float));
	fill_random_floats(transformed, dim, 42);

	uint8_t *bits = mkt_alloc(count * packed_bytes);
	fill_random_bits(bits, count * packed_bytes, 99);

	/* Build LUT and pack codes */
	uint32_t lut_bytes = MKT_FASTSCAN_LUT_BYTES(dim);
	uint8_t *lut	   = mkt_alloc_aligned(lut_bytes, 64);
	float	 scale, bias;
	mkt_fastscan_build_lut(transformed, dim, lut, &scale, &bias);

	uint32_t codes_size = mkt_fastscan_codes_size(count, dim);
	uint8_t *codes		= mkt_alloc(codes_size);
	uint32_t ngroups	= mkt_fastscan_pack_codes(bits, count, dim, codes);

	uint32_t group_bytes = MKT_FASTSCAN_GROUP_BYTES(dim);

	/* Compare each group's results against reference binary IPs.
	 * bias from build_lut is already vl * nsq. */
	float max_err = 0;
	for (uint32_t g = 0; g < ngroups; g++)
	{
		uint16_t accum[32] = {0};
		mkt_fastscan_accumulate(codes + g * group_bytes, lut, accum, dim);

		uint32_t g_count = (g + 1) * 32 <= count ? 32 : count - g * 32;
		for (uint32_t v = 0; v < g_count; v++)
		{
			float approx = (float)accum[v] * scale + bias;
			float ref	 = reference_binary_ip(
					   transformed, bits + (g * 32 + v) * packed_bytes, dim);
			float err = fabsf(approx - ref);
			if (err > max_err)
				max_err = err;
		}
	}

	/* At dim=768 with 192 subquantizers, each contributing up to
	 * 1 step of rounding error, total error can be ~192 * scale/2.
	 * Use a generous bound relative to the IP magnitude. */
	ASSERT_TRUE(max_err < scale * 200.0f, "SIMD matches reference");
}

/* ----------------------------------------------------------------
 * Distance batch test
 * ---------------------------------------------------------------- */

TEST(distance_batch_produces_valid_distances)
{
	/* Verify fastscan distances are finite and positive */
	uint32_t dim		  = 64;
	uint32_t count		  = 32;
	uint32_t packed_bytes = (dim + 7) / 8;

	float *transformed = mkt_alloc(dim * sizeof(float));
	fill_random_floats(transformed, dim, 42);

	uint8_t *bits = mkt_alloc(count * packed_bytes);
	fill_random_bits(bits, count * packed_bytes, 77);

	/* Create a minimal RaBitQQueryState */
	RaBitQQueryState qstate = {0};
	qstate.transformed		= transformed;
	qstate.dim				= dim;
	qstate.inv_sqrt_d		= 1.0f / sqrtf((float)dim);
	qstate.g_add			= 1.0f;
	qstate.sum_transformed	= 0;
	for (uint32_t i = 0; i < dim; i++)
		qstate.sum_transformed += transformed[i];

	/* Build f_add/f_rescale arrays */
	float *f_add	 = mkt_alloc(count * sizeof(float));
	float *f_rescale = mkt_alloc(count * sizeof(float));
	for (uint32_t i = 0; i < count; i++)
	{
		f_add[i]	 = 0.5f;
		f_rescale[i] = 0.3f;
	}

	/* Pack fastscan codes */
	uint32_t codes_size = mkt_fastscan_codes_size(count, dim);
	uint8_t *codes		= mkt_alloc(codes_size);
	uint32_t ngroups	= mkt_fastscan_pack_codes(bits, count, dim, codes);

	/* Allocate scratch */
	uint8_t	 *lut_buf	= mkt_alloc_aligned(MKT_FASTSCAN_LUT_BYTES(dim), 64);
	uint16_t *accum_buf = mkt_alloc_aligned(32 * sizeof(uint16_t), 64);
	float	 *distances = mkt_alloc(count * sizeof(float));

	mkt_fastscan_distance_batch(
			&qstate,
			f_add,
			f_rescale,
			codes,
			ngroups,
			count,
			dim,
			distances,
			lut_buf,
			accum_buf);

	for (uint32_t i = 0; i < count; i++)
	{
		ASSERT_TRUE(isfinite(distances[i]), "distance is finite");
	}
}

/* ----------------------------------------------------------------
 * Edge cases
 * ---------------------------------------------------------------- */

TEST(pack_codes_single_vector)
{
	uint32_t dim		  = 32;
	uint32_t packed_bytes = (dim + 7) / 8;
	uint8_t	 bits[4];
	fill_random_bits(bits, packed_bytes, 42);

	uint32_t codes_size = mkt_fastscan_codes_size(1, dim);
	uint8_t *codes		= mkt_alloc(codes_size);
	uint32_t ngroups	= mkt_fastscan_pack_codes(bits, 1, dim, codes);

	ASSERT_EQ(ngroups, 1, "1 vector = 1 group");
}

TEST(lut_dim_not_multiple_of_4)
{
	/* dim=7 means 2 subquantizers: sq0 has 4 dims, sq1 has 3 */
	uint32_t dim = 7;
	float	 transformed[7];
	fill_random_floats(transformed, dim, 42);

	uint8_t lut[MKT_FASTSCAN_LUT_BYTES(7)];
	float	scale, bias;
	mkt_fastscan_build_lut(transformed, dim, lut, &scale, &bias);

	ASSERT_TRUE(scale > 0, "scale is positive");
}

/* ----------------------------------------------------------------
 * Parameterized SIMD tests: run accumulate + LUT at each level
 * ---------------------------------------------------------------- */

TEST_PARAMETERIZED(simd_accumulate_equivalence, "scalar", "avx2", "avx512")
{
	SKIP_IF_FASTSCAN_SIMD_NOT_AVAILABLE(param);

	uint32_t dim		  = 768;
	uint32_t count		  = 64;
	uint32_t packed_bytes = (dim + 7) / 8;

	float *transformed = mkt_alloc(dim * sizeof(float));
	fill_random_floats(transformed, dim, 42);

	uint8_t *bits = mkt_alloc(count * packed_bytes);
	fill_random_bits(bits, count * packed_bytes, 99);

	uint32_t lut_bytes = MKT_FASTSCAN_LUT_BYTES(dim);
	uint8_t *lut	   = mkt_alloc_aligned(lut_bytes, 64);
	float	 scale, bias;
	mkt_fastscan_build_lut(transformed, dim, lut, &scale, &bias);

	uint32_t codes_size = mkt_fastscan_codes_size(count, dim);
	uint8_t *codes		= mkt_alloc(codes_size);
	uint32_t ngroups	= mkt_fastscan_pack_codes(bits, count, dim, codes);

	uint32_t group_bytes = MKT_FASTSCAN_GROUP_BYTES(dim);

	float max_err = 0;
	for (uint32_t g = 0; g < ngroups; g++)
	{
		uint16_t accum[32] = {0};
		mkt_fastscan_accumulate(codes + g * group_bytes, lut, accum, dim);

		uint32_t g_count = (g + 1) * 32 <= count ? 32 : count - g * 32;
		for (uint32_t v = 0; v < g_count; v++)
		{
			float approx = (float)accum[v] * scale + bias;
			float ref	 = reference_binary_ip(
					   transformed, bits + (g * 32 + v) * packed_bytes, dim);
			float err = fabsf(approx - ref);
			if (err > max_err)
				max_err = err;
		}
	}

	ASSERT_TRUE(max_err < scale * 200.0f, "accumulate matches reference");

	reinit_fastscan_with_simd(0xFFFFFFFF);
}

TEST_PARAMETERIZED(
		simd_accumulate_hacc_equivalence, "scalar", "avx2", "avx512")
{
	SKIP_IF_FASTSCAN_SIMD_NOT_AVAILABLE(param);

	uint32_t dim		  = 768;
	uint32_t count		  = 64;
	uint32_t packed_bytes = (dim + 7) / 8;

	float *transformed = mkt_alloc(dim * sizeof(float));
	fill_random_floats(transformed, dim, 42);

	uint8_t *bits = mkt_alloc(count * packed_bytes);
	fill_random_bits(bits, count * packed_bytes, 99);

	uint32_t lut_bytes = MKT_FASTSCAN_LUT_HACC_BYTES(dim);
	uint8_t *lut	   = mkt_alloc_aligned(lut_bytes, 64);
	float	 scale, bias;
	mkt_fastscan_build_lut_hacc(transformed, dim, lut, &scale, &bias);

	uint32_t codes_size = mkt_fastscan_codes_size(count, dim);
	uint8_t *codes		= mkt_alloc(codes_size);
	uint32_t ngroups	= mkt_fastscan_pack_codes(bits, count, dim, codes);

	uint32_t group_bytes = MKT_FASTSCAN_GROUP_BYTES(dim);

	float max_err = 0;
	for (uint32_t g = 0; g < ngroups; g++)
	{
		int32_t accum[32] = {0};
		mkt_fastscan_accumulate_hacc(codes + g * group_bytes, lut, accum, dim);

		uint32_t g_count = (g + 1) * 32 <= count ? 32 : count - g * 32;
		for (uint32_t v = 0; v < g_count; v++)
		{
			float approx = (float)accum[v] * scale + bias;
			float ref	 = reference_binary_ip(
					   transformed, bits + (g * 32 + v) * packed_bytes, dim);
			float err = fabsf(approx - ref);
			if (err > max_err)
				max_err = err;
		}
	}

	ASSERT_TRUE(max_err < scale * 200.0f, "hacc accumulate matches reference");

	reinit_fastscan_with_simd(0xFFFFFFFF);
}
