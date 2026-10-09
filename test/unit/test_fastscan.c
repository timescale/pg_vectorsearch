/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * test_fastscan.c - Unit tests for VPSHUFB fastscan kernel
 *
 * Tests LUT construction, code packing, scalar/AVX2 kernel
 * equivalence, distance accuracy, and tail handling.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "core/memory.h"
#include "core/platform.h"
#include "quant/fastscan.h"
#include "quant/rabitq.h"
#include "vs_test.h"

TEST_GROUP(Fastscan);
TEST_MEMCTX_FIXTURE();

static void
reinit_fastscan_with_simd(uint32_t simd_mask)
{
	vs_simd_set_override(simd_mask);
	vs_simd_reset_cache();
	vs_fastscan_reset_simd();
	vs_fastscan_init_simd();
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
		if (!(vs_detect_simd() & SIMD_AVX2))
			return false;
		*simd_mask = SIMD_AVX2;
		return true;
	}
	if (strcmp(variant, "avx512") == 0)
	{
		if (!vs_has_all_simd(VS_SIMD_AVX512_BW))
			return false;
		*simd_mask = VS_SIMD_AVX512_BW;
		return true;
	}
#elif defined(__aarch64__) || defined(_M_ARM64)
	if (strcmp(variant, "neon") == 0)
	{
		if (!(vs_detect_simd() & SIMD_NEON))
			return false;
		*simd_mask = SIMD_NEON;
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
	uint32_t rng = seed;
	for (uint32_t i = 0; i < n; i++)
		buf[i] = (float)((int)(vs_test_rand(&rng) % 10000) - 5000) / 5000.0f;
}

/* Fill buffer with random bits */
static void
fill_random_bits(uint8_t *buf, uint32_t n, uint32_t seed)
{
	uint32_t rng = seed;
	for (uint32_t i = 0; i < n; i++)
		buf[i] = (uint8_t)(vs_test_rand(&rng) & 0xFF);
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

	uint8_t lut[VS_FASTSCAN_LUT_BYTES(32)];
	float	scale, bias;
	vs_fastscan_build_lut(transformed, dim, lut, &scale, &bias);

	/* Every sq's entry 0 should be the same (all encode bias) */
	uint32_t nsq = VS_FASTSCAN_NSQ(dim);
	for (uint32_t sq = 0; sq < nsq; sq++)
		ASSERT_EQ(lut[sq * 16], lut[0], "entry 0 same for all sq");
}

TEST(lut_entries_ordered)
{
	/* For a sq with all positive values, higher codes should have
	 * higher LUT entries (more dimensions selected = larger sum) */
	uint32_t dim			= 4; /* single subquantizer */
	float	 transformed[4] = {1.0f, 2.0f, 3.0f, 4.0f};

	uint8_t lut[VS_FASTSCAN_LUT_BYTES(4)];
	float	scale, bias;
	vs_fastscan_build_lut(transformed, dim, lut, &scale, &bias);

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

	uint8_t lut[VS_FASTSCAN_LUT_BYTES(32)];
	float	scale, bias;
	vs_fastscan_build_lut(transformed, dim, lut, &scale, &bias);

	uint32_t nsq	 = VS_FASTSCAN_NSQ(dim);
	float	 max_err = 0;
	for (uint32_t sq = 0; sq < nsq; sq++)
	{
		uint32_t base = sq * VS_FASTSCAN_SQ_DIM;

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
	uint32_t dim = 32;
	/* A full group, so the v >= 16 half of the layout below -- the high
	 * nibble of each byte -- is covered as well as the low one. */
	uint32_t count		  = VS_FASTSCAN_GROUP;
	uint32_t packed_bytes = (dim + 7) / 8;

	uint8_t bits[VS_FASTSCAN_GROUP * 4]; /* 4 bytes per vector */
	fill_random_bits(bits, count * packed_bytes, 77);

	uint32_t codes_size = vs_fastscan_codes_size(count, dim);
	uint8_t *codes		= vs_alloc(codes_size);
	uint32_t ngroups	= vs_fastscan_pack_codes(bits, count, dim, codes);

	ASSERT_EQ(ngroups, 1, "a full group packs as one group");

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
			uint8_t *out = codes + col * VS_FASTSCAN_GROUP;
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

TEST(unpack_codes_roundtrip)
{
	/* pack -> unpack must reproduce the original 1-bit codes exactly,
	 * including a partial trailing group (count not a multiple of 32). */
	uint32_t dim		  = 96; /* packed_bytes = 12 */
	uint32_t count		  = 40; /* 2 groups; 2nd holds 8 vectors */
	uint32_t packed_bytes = (dim + 7) / 8;

	uint8_t bits[40 * 12];
	fill_random_bits(bits, count * packed_bytes, 123);

	uint8_t *codes = vs_alloc(vs_fastscan_codes_size(count, dim));
	vs_fastscan_pack_codes(bits, count, dim, codes);

	uint8_t *out = vs_alloc(count * packed_bytes);
	vs_fastscan_unpack_codes(codes, count, dim, out);

	for (uint32_t i = 0; i < count * packed_bytes; i++)
		ASSERT_EQ(out[i], bits[i], "unpacked byte matches original");

	vs_free(codes);
	vs_free(out);
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

	uint32_t codes_size = vs_fastscan_codes_size(count, dim);
	uint8_t *codes		= vs_alloc(codes_size);
	uint32_t ngroups	= vs_fastscan_pack_codes(bits, count, dim, codes);

	ASSERT_EQ(ngroups, 1, "5 vectors = 1 group (padded)");

	/* Build inverse kPerm0 */
	int inv_kperm0[16];
	for (int j = 0; j < 16; j++)
		inv_kperm0[kPerm0[j]] = j;

	/* Verify nibbles for padding vectors (>= count) are zero */
	uint32_t cols = (dim + 7) / 8;
	for (uint32_t c = 0; c < cols; c++)
	{
		uint8_t *out = codes + c * VS_FASTSCAN_GROUP;
		for (uint32_t v = count; v < VS_FASTSCAN_GROUP; v++)
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
	uint8_t lut[VS_FASTSCAN_LUT_BYTES(16)];
	float	scale, bias;
	vs_fastscan_build_lut(transformed, dim, lut, &scale, &bias);

	uint32_t codes_size = vs_fastscan_codes_size(count, dim);
	uint8_t *codes		= vs_alloc(codes_size);
	vs_fastscan_pack_codes(bits, count, dim, codes);

	/* Run accumulate */
	uint16_t accum[32] = {0};
	vs_fastscan_accumulate(codes, lut, accum, dim);

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

	float *transformed = vs_alloc(dim * sizeof(float));
	fill_random_floats(transformed, dim, 42);

	uint8_t *bits = vs_alloc(count * packed_bytes);
	fill_random_bits(bits, count * packed_bytes, 99);

	/* Build LUT and pack codes */
	uint32_t lut_bytes = VS_FASTSCAN_LUT_BYTES(dim);
	uint8_t *lut	   = vs_alloc_aligned(lut_bytes, 64);
	float	 scale, bias;
	vs_fastscan_build_lut(transformed, dim, lut, &scale, &bias);

	uint32_t codes_size = vs_fastscan_codes_size(count, dim);
	uint8_t *codes		= vs_alloc(codes_size);
	uint32_t ngroups	= vs_fastscan_pack_codes(bits, count, dim, codes);

	uint32_t group_bytes = VS_FASTSCAN_GROUP_BYTES(dim);

	/* Compare each group's results against reference binary IPs.
	 * bias from build_lut is already vl * nsq. */
	float max_err = 0;
	for (uint32_t g = 0; g < ngroups; g++)
	{
		uint16_t accum[32] = {0};
		vs_fastscan_accumulate(codes + g * group_bytes, lut, accum, dim);

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

	float *transformed = vs_alloc(dim * sizeof(float));
	fill_random_floats(transformed, dim, 42);

	uint8_t *bits = vs_alloc(count * packed_bytes);
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
	float *f_add	 = vs_alloc(count * sizeof(float));
	float *f_rescale = vs_alloc(count * sizeof(float));
	for (uint32_t i = 0; i < count; i++)
	{
		f_add[i]	 = 0.5f;
		f_rescale[i] = 0.3f;
	}

	/* Pack fastscan codes */
	uint32_t codes_size = vs_fastscan_codes_size(count, dim);
	uint8_t *codes		= vs_alloc(codes_size);
	uint32_t ngroups	= vs_fastscan_pack_codes(bits, count, dim, codes);

	/* Allocate scratch */
	uint8_t	 *lut_buf	= vs_alloc_aligned(VS_FASTSCAN_LUT_BYTES(dim), 64);
	uint16_t *accum_buf = vs_alloc_aligned(32 * sizeof(uint16_t), 64);
	float	 *distances = vs_alloc(count * sizeof(float));

	vs_fastscan_distance_batch(
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

	uint32_t codes_size = vs_fastscan_codes_size(1, dim);
	uint8_t *codes		= vs_alloc(codes_size);
	uint32_t ngroups	= vs_fastscan_pack_codes(bits, 1, dim, codes);

	ASSERT_EQ(ngroups, 1, "1 vector = 1 group");
}

TEST(lut_dim_not_multiple_of_4)
{
	/* dim=7 means 2 subquantizers: sq0 has 4 dims, sq1 has 3 */
	uint32_t dim = 7;
	float	 transformed[7];
	fill_random_floats(transformed, dim, 42);

	uint8_t lut[VS_FASTSCAN_LUT_BYTES(7)];
	float	scale, bias;
	vs_fastscan_build_lut(transformed, dim, lut, &scale, &bias);

	ASSERT_TRUE(scale > 0, "scale is positive");
}

/* ----------------------------------------------------------------
 * Parameterized SIMD tests: run accumulate + LUT at each level
 * ---------------------------------------------------------------- */

TEST_PARAMETERIZED(
		simd_accumulate_equivalence, "scalar", "avx2", "avx512", "neon")
{
	SKIP_IF_FASTSCAN_SIMD_NOT_AVAILABLE(param);

	uint32_t dim		  = 768;
	uint32_t count		  = 64;
	uint32_t packed_bytes = (dim + 7) / 8;

	float *transformed = vs_alloc(dim * sizeof(float));
	fill_random_floats(transformed, dim, 42);

	uint8_t *bits = vs_alloc(count * packed_bytes);
	fill_random_bits(bits, count * packed_bytes, 99);

	uint32_t lut_bytes = VS_FASTSCAN_LUT_BYTES(dim);
	uint8_t *lut	   = vs_alloc_aligned(lut_bytes, 64);
	float	 scale, bias;
	vs_fastscan_build_lut(transformed, dim, lut, &scale, &bias);

	uint32_t codes_size = vs_fastscan_codes_size(count, dim);
	uint8_t *codes		= vs_alloc(codes_size);
	uint32_t ngroups	= vs_fastscan_pack_codes(bits, count, dim, codes);

	uint32_t group_bytes = VS_FASTSCAN_GROUP_BYTES(dim);

	float max_err = 0;
	for (uint32_t g = 0; g < ngroups; g++)
	{
		uint16_t accum[32] = {0};
		vs_fastscan_accumulate(codes + g * group_bytes, lut, accum, dim);

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
		simd_accumulate_hacc_equivalence, "scalar", "avx2", "avx512", "neon")
{
	SKIP_IF_FASTSCAN_SIMD_NOT_AVAILABLE(param);

	uint32_t dim		  = 768;
	uint32_t count		  = 64;
	uint32_t packed_bytes = (dim + 7) / 8;

	float *transformed = vs_alloc(dim * sizeof(float));
	fill_random_floats(transformed, dim, 42);

	uint8_t *bits = vs_alloc(count * packed_bytes);
	fill_random_bits(bits, count * packed_bytes, 99);

	uint32_t lut_bytes = VS_FASTSCAN_LUT_HACC_BYTES(dim);
	uint8_t *lut	   = vs_alloc_aligned(lut_bytes, 64);
	float	 scale, bias;
	vs_fastscan_build_lut_hacc(transformed, dim, lut, &scale, &bias);

	uint32_t codes_size = vs_fastscan_codes_size(count, dim);
	uint8_t *codes		= vs_alloc(codes_size);
	uint32_t ngroups	= vs_fastscan_pack_codes(bits, count, dim, codes);

	uint32_t group_bytes = VS_FASTSCAN_GROUP_BYTES(dim);

	float max_err = 0;
	for (uint32_t g = 0; g < ngroups; g++)
	{
		int32_t accum[32] = {0};
		vs_fastscan_accumulate_hacc(codes + g * group_bytes, lut, accum, dim);

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

/* ----------------------------------------------------------------
 * Regression (F2/F4): the SIMD accumulate kernels read codes/LUT in
 * 64B chunks, but a group's code region is ceil(dim/8)*32 bytes -- a
 * multiple of 32, not always 64. When ceil(dim/8) is odd (e.g. dim 8,
 * 100, 1000; but not 768) the final chunk read 32 bytes past the code
 * region (and, for the 8-bit path, past the LUT). The buffers here are
 * plain malloc'd at exactly the required size so the sanitizer CI
 * (sanitizers.yml) flags any reintroduced over-read -- the unit test's
 * usual arena allocator would hide it inside a larger block. Values are
 * also checked against the scalar reference at each dim.
 * ---------------------------------------------------------------- */
TEST_PARAMETERIZED(
		fastscan_odd_dim_no_overread, "scalar", "avx2", "avx512", "neon")
{
	SKIP_IF_FASTSCAN_SIMD_NOT_AVAILABLE(param);

	/* ceil(dim/8): 8->1, 100->13, 1000->125 are odd (the buggy case);
	 * 768->96 is the even control that never triggered the over-read. */
	const uint32_t dims[] = {8, 100, 768, 1000};
	const uint32_t count  = 40; /* 2 groups: exercises a non-final and a
								 * final group */

	for (uint32_t di = 0; di < sizeof(dims) / sizeof(dims[0]); di++)
	{
		uint32_t dim		  = dims[di];
		uint32_t packed_bytes = (dim + 7) / 8;
		uint32_t group_bytes  = VS_FASTSCAN_GROUP_BYTES(dim);
		uint32_t ngroups = (count + VS_FASTSCAN_GROUP - 1) / VS_FASTSCAN_GROUP;

		float *transformed = malloc(dim * sizeof(float));
		fill_random_floats(transformed, dim, 42 + di);
		uint8_t *bits = malloc((size_t)count * packed_bytes);
		fill_random_bits(bits, count * packed_bytes, 99 + di);

		/* Exact-size, instrumented buffers: an over-read past group_bytes
		 * (codes) or the LUT size faults under ASan. */
		uint8_t *codes = malloc((size_t)ngroups * group_bytes);
		vs_fastscan_pack_codes(bits, count, dim, codes);

		float	 scale, bias;
		uint8_t *lut8 = malloc(VS_FASTSCAN_LUT_BYTES(dim));
		vs_fastscan_build_lut(transformed, dim, lut8, &scale, &bias);
		float	 hscale, hbias;
		uint8_t *lut16 = malloc(VS_FASTSCAN_LUT_HACC_BYTES(dim));
		vs_fastscan_build_lut_hacc(transformed, dim, lut16, &hscale, &hbias);

		for (uint32_t g = 0; g < ngroups; g++)
		{
			uint16_t acc8[VS_FASTSCAN_GROUP];
			int32_t	 acc16[VS_FASTSCAN_GROUP];
			vs_fastscan_accumulate(
					codes + (size_t)g * group_bytes, lut8, acc8, dim);
			vs_fastscan_accumulate_hacc(
					codes + (size_t)g * group_bytes, lut16, acc16, dim);

			uint32_t g_count = (g + 1) * VS_FASTSCAN_GROUP <= count
									 ? VS_FASTSCAN_GROUP
									 : count - g * VS_FASTSCAN_GROUP;
			for (uint32_t v = 0; v < g_count; v++)
			{
				const uint8_t *vb = bits +
									(size_t)(g * VS_FASTSCAN_GROUP + v) *
											packed_bytes;
				float ref	   = reference_binary_ip(transformed, vb, dim);
				float approx8  = (float)acc8[v] * scale + bias;
				float approx16 = (float)acc16[v] * hscale + hbias;
				ASSERT_TRUE(
						fabsf(approx8 - ref) < scale * 200.0f,
						"8-bit accumulate matches reference at odd dim");
				ASSERT_TRUE(
						fabsf(approx16 - ref) < hscale * 200.0f,
						"hacc accumulate matches reference at odd dim");
			}
		}

		free(transformed);
		free(bits);
		free(codes);
		free(lut8);
		free(lut16);
	}

	reinit_fastscan_with_simd(0xFFFFFFFF);
}
