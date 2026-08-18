/*
 * fastscan_avx2.c - AVX2 VPSHUFB accumulate kernel
 *
 * AVX2 version of the even/odd byte accumulation. Processes 64B
 * per iteration in two 32B loads (since AVX2 registers are 256-bit).
 * Each iteration handles 4 subquantizers.
 */

#include "mkt_config.h"

#ifdef MKT_SIMD_FULL

#if defined(__x86_64__) || defined(_M_X64)

#include <immintrin.h>

#include "algo/simd_utils.h"
#include "quant/fastscan.h"

/* Accumulate one 32-byte block into even/odd accumulators */
#define ACCUM_32B(codes_ptr, lut_ptr, a0, a1, a2, a3, mask)              \
	do                                                                   \
	{                                                                    \
		__m256i c_	 = _mm256_loadu_si256((const __m256i *)(codes_ptr)); \
		__m256i tab_ = _mm256_loadu_si256((const __m256i *)(lut_ptr));   \
		__m256i lo_	 = _mm256_and_si256(c_, mask);                       \
		__m256i hi_	 = _mm256_and_si256(_mm256_srli_epi16(c_, 4), mask); \
		__m256i rlo_ = _mm256_shuffle_epi8(tab_, lo_);                   \
		__m256i rhi_ = _mm256_shuffle_epi8(tab_, hi_);                   \
		a0			 = _mm256_add_epi16(a0, rlo_);                       \
		a1			 = _mm256_add_epi16(a1, _mm256_srli_epi16(rlo_, 8)); \
		a2			 = _mm256_add_epi16(a2, rhi_);                       \
		a3			 = _mm256_add_epi16(a3, _mm256_srli_epi16(rhi_, 8)); \
	} while (0)

/* Reduce even/odd accumulators to 16 uint16 results */
static inline MKT_TARGET_AVX2 __m256i
reduce_accu_pair(__m256i even, __m256i odd)
{
	even = _mm256_sub_epi16(even, _mm256_slli_epi16(odd, 8));
	return _mm256_add_epi16(
			_mm256_permute2f128_si256(even, odd, 0x21),
			_mm256_blend_epi32(even, odd, 0xF0));
}

MKT_TARGET_AVX2 void
mkt_fastscan_accumulate_avx2(
		const uint8_t *codes,
		const uint8_t *lut,
		uint16_t	  *accum,
		Dimension	   dim)
{
	uint32_t code_length = MKT_FASTSCAN_GROUP_BYTES(dim);

	__m256i low_mask = _mm256_set1_epi8(0x0F);
	__m256i accu0	 = _mm256_setzero_si256();
	__m256i accu1	 = _mm256_setzero_si256();
	__m256i accu2	 = _mm256_setzero_si256();
	__m256i accu3	 = _mm256_setzero_si256();

	/* One 32B column (8 dims) per iteration. code_length is a multiple
	 * of 32 but not necessarily 64, so stepping 64B (two columns) at a
	 * time would read a phantom column past the code region whenever
	 * ceil(dim/8) is odd. The accumulators are order-independent, so a
	 * 32B stride is equivalent and never over-reads. */
	for (uint32_t i = 0; i < code_length; i += 32)
		ACCUM_32B(codes + i, lut + i, accu0, accu1, accu2, accu3, low_mask);

	_mm256_storeu_si256((__m256i *)accum, reduce_accu_pair(accu0, accu1));
	_mm256_storeu_si256(
			(__m256i *)(accum + 16), reduce_accu_pair(accu2, accu3));
}

#undef ACCUM_32B

/* ----------------------------------------------------------------
 * AVX2 high-accuracy accumulate (uint16 LUT → int32 output)
 *
 * Two VPSHUFB passes per 32B code block (lo-byte table, hi-byte
 * table). 8 accumulators: accu[lo/hi][0-3].
 * ---------------------------------------------------------------- */

#define HACC_ACCUM_256(tab, lo, hi, a)                                     \
	do                                                                     \
	{                                                                      \
		__m256i rlo_ = _mm256_shuffle_epi8(tab, lo);                       \
		__m256i rhi_ = _mm256_shuffle_epi8(tab, hi);                       \
		a[0]		 = _mm256_add_epi16(a[0], rlo_);                       \
		a[1]		 = _mm256_add_epi16(a[1], _mm256_srli_epi16(rlo_, 8)); \
		a[2]		 = _mm256_add_epi16(a[2], rhi_);                       \
		a[3]		 = _mm256_add_epi16(a[3], _mm256_srli_epi16(rhi_, 8)); \
	} while (0)

/* Reduce even/odd accumulators to 16 uint16, then widen to int32.
 * Returns two __m256i: lo 8 int32, hi 8 int32 via output pointers. */
static inline MKT_TARGET_AVX2 void
hacc_reduce_avx2(__m256i even, __m256i odd, __m256i *out_lo8, __m256i *out_hi8)
{
	__m256i r16 = reduce_accu_pair(even, odd);
	*out_lo8	= _mm256_cvtepu16_epi32(_mm256_castsi256_si128(r16));
	*out_hi8	= _mm256_cvtepu16_epi32(_mm256_extracti128_si256(r16, 1));
}

MKT_TARGET_AVX2 void
mkt_fastscan_accumulate_hacc_avx2(
		const uint8_t *codes,
		const uint8_t *lut,
		int32_t		  *accum,
		Dimension	   dim)
{
	__m256i low_mask = _mm256_set1_epi8(0x0F);

	__m256i accu[2][4];
	for (int q = 0; q < 2; q++)
		for (int r = 0; r < 4; r++)
			accu[q][r] = _mm256_setzero_si256();

	/* One 32B column (2 subquantizers) per iteration. The high-accuracy
	 * LUT is laid out in 128B blocks of 4 subquantizers ([4x16 lo][4x16
	 * hi]); column c lives in block c/2 at within-block offset (c%2)*32,
	 * with the hi table 64B after the lo. Iterating whole columns reads
	 * exactly code_length bytes -- stepping 64B (4 sq) at a time would
	 * read a phantom column past the code region when ceil(dim/8) is
	 * odd. Trailing phantom subquantizers (nsq not a multiple of 2) have
	 * a zeroed LUT, so they contribute nothing. */
	uint32_t ncols = MKT_FASTSCAN_GROUP_BYTES(dim) / MKT_FASTSCAN_GROUP;
	for (uint32_t c = 0; c < ncols; c++)
	{
		__m256i cc = _mm256_loadu_si256((const __m256i *)(codes + c * 32));
		__m256i lo = _mm256_and_si256(cc, low_mask);
		__m256i hi = _mm256_and_si256(_mm256_srli_epi16(cc, 4), low_mask);

		const uint8_t *lut_blk = lut + (c / 2) * 128 + (c % 2) * 32;
		__m256i		   tab_lo  = _mm256_loadu_si256((const __m256i *)lut_blk);
		HACC_ACCUM_256(tab_lo, lo, hi, accu[0]);

		__m256i tab_hi = _mm256_loadu_si256((const __m256i *)(lut_blk + 64));
		HACC_ACCUM_256(tab_hi, lo, hi, accu[1]);
	}

	/* Reduce: vectors 0-15 (accu[][0,1]), 16-31 (accu[][2,3]) */
	for (int half = 0; half < 2; half++)
	{
		__m256i lo_a, lo_b, hi_a, hi_b;
		hacc_reduce_avx2(
				accu[0][half * 2], accu[0][half * 2 + 1], &lo_a, &lo_b);
		hacc_reduce_avx2(
				accu[1][half * 2], accu[1][half * 2 + 1], &hi_a, &hi_b);
		int32_t *out = accum + half * 16;
		_mm256_storeu_si256(
				(__m256i *)out,
				_mm256_add_epi32(lo_a, _mm256_slli_epi32(hi_a, 8)));
		_mm256_storeu_si256(
				(__m256i *)(out + 8),
				_mm256_add_epi32(lo_b, _mm256_slli_epi32(hi_b, 8)));
	}
}

#undef HACC_ACCUM_256

#endif /* x86_64 */

#endif /* MKT_SIMD_FULL */
