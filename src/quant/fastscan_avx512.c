/*
 * fastscan_avx512.c - AVX-512 VPSHUFB accumulate kernel
 *
 * Matches the RaBitQ Library's approach: loads 64B codes + 64B LUT
 * per iteration (4 subquantizers), uses even/odd byte accumulation
 * with 4 uint16 accumulator registers.
 *
 * Each iteration:
 *   1. Load 64B codes (4 columns x 16 bytes per nibble group)
 *   2. Load 64B LUT  (4 subquantizer tables x 16 entries)
 *   3. Split lo/hi nibbles (2 ops)
 *   4. 2x VPSHUFB: res_lo (vectors 0-15), res_hi (vectors 16-31)
 *   5. Accumulate with even/odd byte trick (4 add + 2 srli = 6 ops)
 *
 * After loop: subtract upper byte contamination, shuffle to output
 * order, store 32 uint16 results.
 */

#include "mkt_config.h"

#ifdef MKT_SIMD_FULL

#if defined(__x86_64__) || defined(_M_X64)

#include <immintrin.h>
#include <stdint.h>
#include <string.h>

#include "quant/fastscan.h"

__attribute__((target("avx512f,avx512bw"))) void
mkt_fastscan_accumulate_avx512(
		const uint8_t *codes,
		const uint8_t *lut,
		uint16_t	  *accum,
		Dimension	   dim)
{
	uint32_t code_length = MKT_FASTSCAN_GROUP_BYTES(dim);

	const __m512i lo_mask = _mm512_set1_epi8(0x0F);
	__m512i		  accu0	  = _mm512_setzero_si512();
	__m512i		  accu1	  = _mm512_setzero_si512();
	__m512i		  accu2	  = _mm512_setzero_si512();
	__m512i		  accu3	  = _mm512_setzero_si512();

	for (uint32_t i = 0; i < code_length; i += 64)
	{
		__m512i c	= _mm512_loadu_si512((const __m512i *)(codes + i));
		__m512i tab = _mm512_loadu_si512((const __m512i *)(lut + i));

		__m512i lo = _mm512_and_si512(c, lo_mask);
		__m512i hi = _mm512_and_si512(_mm512_srli_epi16(c, 4), lo_mask);

		__m512i res_lo = _mm512_shuffle_epi8(tab, lo);
		__m512i res_hi = _mm512_shuffle_epi8(tab, hi);

		/* Even/odd byte accumulation:
		 * accu0 accumulates even-byte values (vectors 0-7 per lane)
		 * accu1 accumulates odd-byte values (vectors 8-15 per lane)
		 * The add_epi16 on uint8 results causes upper byte
		 * contamination which accu1 tracks via srli. */
		accu0 = _mm512_add_epi16(accu0, res_lo);
		accu1 = _mm512_add_epi16(accu1, _mm512_srli_epi16(res_lo, 8));
		accu2 = _mm512_add_epi16(accu2, res_hi);
		accu3 = _mm512_add_epi16(accu3, _mm512_srli_epi16(res_hi, 8));
	}

	/* Remove upper byte contamination */
	accu0 = _mm512_sub_epi16(accu0, _mm512_slli_epi16(accu1, 8));
	accu2 = _mm512_sub_epi16(accu2, _mm512_slli_epi16(accu3, 8));

	/* Combine 4 x 128-bit lanes into final 32 uint16 results.
	 *
	 * Each accumulator has 4 x 128-bit lanes. We need to sum
	 * corresponding lanes across accumulators and reorder to
	 * get vectors 0-31 in order.
	 *
	 * accu0 lanes: [0-7 from lane0, 0-7 from lane1, ...]
	 * accu1 lanes: [8-15 from lane0, 8-15 from lane1, ...]
	 *
	 * Use blend + shuffle_i64x2 to cross-lane sum. */
	__m512i ret1 = _mm512_add_epi16(
			_mm512_mask_blend_epi64(0b11110000, accu0, accu1),
			_mm512_shuffle_i64x2(accu0, accu1, 0b01001110));
	__m512i ret2 = _mm512_add_epi16(
			_mm512_mask_blend_epi64(0b11110000, accu2, accu3),
			_mm512_shuffle_i64x2(accu2, accu3, 0b01001110));

	__m512i ret = _mm512_setzero_si512();
	ret = _mm512_add_epi16(ret, _mm512_shuffle_i64x2(ret1, ret2, 0b10001000));
	ret = _mm512_add_epi16(ret, _mm512_shuffle_i64x2(ret1, ret2, 0b11011101));

	_mm512_storeu_si512((__m512i *)accum, ret);
}

/* ----------------------------------------------------------------
 * AVX-512 high-accuracy accumulate (uint16 LUT → int32 output)
 *
 * Two VPSHUFB passes per code block: one for lo-byte table, one
 * for hi-byte table. Same codes in registers for both passes.
 * 8 accumulators: accu[lo/hi][0-3].
 * Final: result = lo_result + (hi_result << 8)
 * ---------------------------------------------------------------- */

/* Accumulate VPSHUFB results into even/odd accumulators */
#define HACC_ACCUM_512(tab, lo, hi, a)                                     \
	do                                                                     \
	{                                                                      \
		__m512i rlo_ = _mm512_shuffle_epi8(tab, lo);                       \
		__m512i rhi_ = _mm512_shuffle_epi8(tab, hi);                       \
		a[0]		 = _mm512_add_epi16(a[0], rlo_);                       \
		a[1]		 = _mm512_add_epi16(a[1], _mm512_srli_epi16(rlo_, 8)); \
		a[2]		 = _mm512_add_epi16(a[2], rhi_);                       \
		a[3]		 = _mm512_add_epi16(a[3], _mm512_srli_epi16(rhi_, 8)); \
	} while (0)

/* Reduce 4 ZMM accumulators to 16 int32 results */
static inline __attribute__((target("avx512f,avx512bw"))) __m512i
hacc_reduce_16(__m512i even, __m512i odd)
{
	__m256i e = _mm256_add_epi16(
			_mm512_castsi512_si256(even), _mm512_extracti64x4_epi64(even, 1));
	__m256i o = _mm256_add_epi16(
			_mm512_castsi512_si256(odd), _mm512_extracti64x4_epi64(odd, 1));
	e = _mm256_sub_epi16(e, _mm256_slli_epi16(o, 8));
	return _mm512_add_epi32(
			_mm512_cvtepu16_epi32(_mm256_permute2f128_si256(e, o, 0x21)),
			_mm512_cvtepu16_epi32(_mm256_blend_epi32(e, o, 0xF0)));
}

__attribute__((target("avx512f,avx512bw"))) void
mkt_fastscan_accumulate_hacc_avx512(
		const uint8_t *codes,
		const uint8_t *lut,
		int32_t		  *accum,
		Dimension	   dim)
{
	uint32_t	  nsq	  = MKT_FASTSCAN_NSQ(dim);
	const __m512i lo_mask = _mm512_set1_epi8(0x0F);

	__m512i accu[2][4];
	for (int q = 0; q < 2; q++)
		for (int r = 0; r < 4; r++)
			accu[q][r] = _mm512_setzero_si512();

	for (uint32_t m = 0; m < nsq; m += 4)
	{
		__m512i c  = _mm512_loadu_si512((const __m512i *)codes);
		__m512i lo = _mm512_and_si512(c, lo_mask);
		__m512i hi = _mm512_and_si512(_mm512_srli_epi16(c, 4), lo_mask);

		__m512i tab_lo = _mm512_loadu_si512((const __m512i *)lut);
		HACC_ACCUM_512(tab_lo, lo, hi, accu[0]);

		__m512i tab_hi = _mm512_loadu_si512((const __m512i *)(lut + 64));
		HACC_ACCUM_512(tab_hi, lo, hi, accu[1]);

		codes += 64;
		lut += 128;
	}

	/* Reduce and combine lo + (hi << 8) */
	__m512i lo0 = hacc_reduce_16(accu[0][0], accu[0][1]);
	__m512i lo1 = hacc_reduce_16(accu[0][2], accu[0][3]);
	__m512i hi0 = hacc_reduce_16(accu[1][0], accu[1][1]);
	__m512i hi1 = hacc_reduce_16(accu[1][2], accu[1][3]);

	_mm512_storeu_si512(
			(__m512i *)accum,
			_mm512_add_epi32(lo0, _mm512_slli_epi32(hi0, 8)));
	_mm512_storeu_si512(
			(__m512i *)(accum + 16),
			_mm512_add_epi32(lo1, _mm512_slli_epi32(hi1, 8)));
}

#undef HACC_ACCUM_512

/* ----------------------------------------------------------------
 * AVX-512 high-accuracy LUT construction
 * ---------------------------------------------------------------- */

__attribute__((target("avx512f,avx512bw,avx512vl"))) void
mkt_fastscan_build_lut_hacc_avx512(
		const float *transformed,
		Dimension	 dim,
		uint8_t		*lut_out,
		float		*delta_out,
		float		*bias_out)
{
	uint32_t nsq = MKT_FASTSCAN_NSQ(dim);

	/* Vectorized min/max */
	__m512 pos_sum = _mm512_setzero_ps();
	__m512 neg_sum = _mm512_setzero_ps();
	__m512 zero	   = _mm512_setzero_ps();

	uint32_t d = 0;
	for (; d + 16 <= dim; d += 16)
	{
		__m512 v = _mm512_loadu_ps(transformed + d);
		pos_sum	 = _mm512_add_ps(pos_sum, _mm512_max_ps(v, zero));
		neg_sum	 = _mm512_add_ps(neg_sum, _mm512_min_ps(v, zero));
	}
	float global_max = _mm512_reduce_add_ps(pos_sum);
	float global_min = _mm512_reduce_add_ps(neg_sum);
	for (; d < dim; d++)
	{
		if (transformed[d] > 0)
			global_max += transformed[d];
		else
			global_min += transformed[d];
	}

	float range = global_max - global_min;
	if (range < MKT_FASTSCAN_MIN_RANGE)
		range = MKT_FASTSCAN_MIN_RANGE;

	float delta		= range / (float)UINT16_MAX;
	float inv_delta = 1.0f / delta;
	*delta_out		= delta;
	*bias_out		= global_min * (float)nsq;

	float bias_scaled = -global_min * inv_delta + 0.5f;

	uint32_t lut_bytes = MKT_FASTSCAN_LUT_HACC_BYTES(dim);
	memset(lut_out, 0, lut_bytes);

	__m512	lo_v  = _mm512_setzero_ps();
	__m512	hi_v  = _mm512_set1_ps((float)UINT16_MAX);
	__m512i mask8 = _mm512_set1_epi32(0xFF);

	const float *q = transformed;
	for (uint32_t sq = 0; sq < nsq; sq++)
	{
		Dimension base = sq * 4;
		float	  s0   = (base + 0 < dim) ? q[0] * inv_delta : 0.0f;
		float	  s1   = (base + 1 < dim) ? q[1] * inv_delta : 0.0f;
		float	  s2   = (base + 2 < dim) ? q[2] * inv_delta : 0.0f;
		float	  s3   = (base + 3 < dim) ? q[3] * inv_delta : 0.0f;

		float p01 = s0 + s1;
		float p23 = s2 + s3;
		float f0  = bias_scaled;
		float f4  = f0 + s2;
		float f8  = f0 + s3;
		float f12 = f0 + p23;

		__m512 entries = _mm512_setr_ps(
				f0,
				f0 + s0,
				f0 + s1,
				f0 + p01,
				f4,
				f4 + s0,
				f4 + s1,
				f4 + p01,
				f8,
				f8 + s0,
				f8 + s1,
				f8 + p01,
				f12,
				f12 + s0,
				f12 + s1,
				f12 + p01);

		entries = _mm512_max_ps(entries, lo_v);
		entries = _mm512_min_ps(entries, hi_v);

		__m512i int_ent = _mm512_cvttps_epi32(entries);

		/* Split into lo and hi bytes */
		__m512i lo_bytes = _mm512_and_si512(int_ent, mask8);
		__m512i hi_bytes =
				_mm512_and_si512(_mm512_srli_epi32(int_ent, 8), mask8);

		/* Pack 16 int32 → 16 uint8 */
		__m256i lo16 = _mm512_cvtepi32_epi16(lo_bytes);
		__m128i lo8	 = _mm256_cvtepi16_epi8(lo16);

		__m256i hi16 = _mm512_cvtepi32_epi16(hi_bytes);
		__m128i hi8	 = _mm256_cvtepi16_epi8(hi16);

		/* Store to interleaved layout */
		uint32_t group4		  = sq / 4;
		uint32_t pos_in_group = sq % 4;
		uint8_t *lo_dst		  = lut_out + group4 * 128 + pos_in_group * 16;
		uint8_t *hi_dst		  = lo_dst + 64;

		_mm_storeu_si128((__m128i *)lo_dst, lo8);
		_mm_storeu_si128((__m128i *)hi_dst, hi8);

		q += 4;
	}
}

/* ----------------------------------------------------------------
 * AVX-512 LUT construction
 *
 * Vectorizes both the min/max scan and the per-sq LUT build.
 * Processes 4 subquantizers (16 floats) per iteration, producing
 * 64 uint8 LUT entries.
 * ---------------------------------------------------------------- */

__attribute__((target("avx512f,avx512bw,avx512vl"))) void
mkt_fastscan_build_lut_avx512(
		const float *transformed,
		Dimension	 dim,
		uint8_t		*lut_out,
		float		*delta_out,
		float		*bias_out)
{
	uint32_t nsq	   = MKT_FASTSCAN_NSQ(dim);
	uint32_t nsq_pairs = MKT_FASTSCAN_NSQ_PAIRS(dim);

	/* Vectorized min/max: split positive/negative and sum */
	__m512 pos_sum = _mm512_setzero_ps();
	__m512 neg_sum = _mm512_setzero_ps();
	__m512 zero	   = _mm512_setzero_ps();

	uint32_t d = 0;
	for (; d + 16 <= dim; d += 16)
	{
		__m512 v = _mm512_loadu_ps(transformed + d);
		pos_sum	 = _mm512_add_ps(pos_sum, _mm512_max_ps(v, zero));
		neg_sum	 = _mm512_add_ps(neg_sum, _mm512_min_ps(v, zero));
	}
	float global_max = _mm512_reduce_add_ps(pos_sum);
	float global_min = _mm512_reduce_add_ps(neg_sum);
	for (; d < dim; d++)
	{
		if (transformed[d] > 0)
			global_max += transformed[d];
		else
			global_min += transformed[d];
	}

	float range = global_max - global_min;
	if (range < MKT_FASTSCAN_MIN_RANGE)
		range = MKT_FASTSCAN_MIN_RANGE;

	float delta		= range / (float)UINT8_MAX;
	float inv_delta = 1.0f / delta;
	*delta_out		= delta;
	*bias_out		= global_min * (float)nsq;

	/* +0.5f for round-half-up via truncation (matches scalar) */
	float bias_scaled = -global_min * inv_delta + 0.5f;

	memset(lut_out, 0, nsq_pairs * 2 * 16);

	__m512 lo_v = _mm512_setzero_ps();
	__m512 hi_v = _mm512_set1_ps((float)UINT8_MAX);

	const float *q = transformed;
	for (uint32_t sq = 0; sq < nsq; sq++)
	{
		uint8_t *out = lut_out + sq * 16;

		Dimension base = sq * 4;
		float	  s0   = (base + 0 < dim) ? q[0] * inv_delta : 0.0f;
		float	  s1   = (base + 1 < dim) ? q[1] * inv_delta : 0.0f;
		float	  s2   = (base + 2 < dim) ? q[2] * inv_delta : 0.0f;
		float	  s3   = (base + 3 < dim) ? q[3] * inv_delta : 0.0f;

		float p01 = s0 + s1;
		float p23 = s2 + s3;

		float f0  = bias_scaled;
		float f4  = f0 + s2;
		float f8  = f0 + s3;
		float f12 = f0 + p23;

		/* Build 16 float entries in a ZMM register */
		__m512 entries = _mm512_setr_ps(
				f0,
				f0 + s0,
				f0 + s1,
				f0 + p01,
				f4,
				f4 + s0,
				f4 + s1,
				f4 + p01,
				f8,
				f8 + s0,
				f8 + s1,
				f8 + p01,
				f12,
				f12 + s0,
				f12 + s1,
				f12 + p01);

		/* Clamp to [0, 255] and truncate to int32 (matching scalar) */
		entries = _mm512_max_ps(entries, lo_v);
		entries = _mm512_min_ps(entries, hi_v);
		/* Truncate (not round) to match scalar (int)(val) behavior.
		 * The +0.5f in bias_scaled provides round-half-up. */
		__m512i int_ent = _mm512_cvttps_epi32(entries);

		/* Pack 16 int32 → 16 uint8 via two-stage narrowing:
		 * epi32 → epi16 → epi8. Use pmovdw + pmovwb. */
		__m256i i16 = _mm512_cvtepi32_epi16(int_ent);
		__m128i i8	= _mm256_cvtepi16_epi8(i16);

		_mm_storeu_si128((__m128i *)out, i8);

		q += 4;
	}
}

#endif /* x86_64 */

#endif /* MKT_SIMD_FULL */
