/*
 * rabitq_avx2.c - AVX2 optimized binary inner product for RaBitQ
 *
 * The hot path in RaBitQ distance computation is:
 *   sum = 0
 *   for each bit i:
 *     if bits[i] == 1:
 *       sum += transformed[i]
 *
 * This AVX2 implementation processes 8 floats at a time using masked
 * addition. The key optimization is expanding each byte of the bit array
 * into 8 float masks.
 */

#include "mkt_config.h"

#ifdef MKT_SIMD_FULL

#if defined(__x86_64__) || defined(_M_X64)

#include <immintrin.h>

#include "algo/simd_utils.h"
#include "mkt_types.h"

/*
 * Expand a byte to 8 float masks for AVX2.
 *
 * Input: byte with 8 bits (LSB-first)
 * Output: __m256 where each lane is either all 1s (0xFFFFFFFF) or all 0s
 *
 * The bit order is LSB-first (bit 0 -> float 0, bit 7 -> float 7).
 */
MKT_TARGET_AVX2 static inline __m256
expand_byte_to_mask_avx2(uint8_t byte)
{
	/* Broadcast byte to all positions */
	__m256i vbyte = _mm256_set1_epi32(byte);

	/* Bit positions for each float (LSB-first: 0,1,2,3,4,5,6,7) */
	__m256i bit_positions = _mm256_setr_epi32(
			1 << 0, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6, 1 << 7);

	/* Test each bit: (byte & bit_pos) != 0 */
	__m256i masked = _mm256_and_si256(vbyte, bit_positions);

	/* Compare to zero - creates all 1s for non-zero, all 0s for zero */
	__m256i cmp = _mm256_cmpeq_epi32(masked, _mm256_setzero_si256());

	/* Invert: we want 1s where bit is set */
	__m256i result = _mm256_xor_si256(cmp, _mm256_set1_epi32(-1));

	return _mm256_castsi256_ps(result);
}

/*
 * AVX2 binary inner product implementation.
 *
 * Processes 8 floats at a time using masked addition.
 */
MKT_TARGET_AVX2 float
mkt_rabitq_inner_product_avx2(
		const float *transformed, const uint8_t *bits, Dimension dim)
{
	__m256 sum = _mm256_setzero_ps();

	/* Main loop: process 8 floats (1 byte of bits) at a time */
	Dimension i = 0;
	for (; i + 8 <= dim; i += 8)
	{
		/* Load 8 transformed values */
		__m256 t = _mm256_loadu_ps(transformed + i);

		/* Get the byte containing bits for these 8 floats */
		uint8_t byte = bits[i / 8];

		/* Expand byte to mask */
		__m256 mask = expand_byte_to_mask_avx2(byte);

		/* Masked addition: only add where bit is set */
		__m256 masked_t = _mm256_and_ps(t, mask);
		sum				= _mm256_add_ps(sum, masked_t);
	}

	/* Horizontal sum */
	float result = mkt_horizontal_sum_avx2(sum);

	/* Handle tail elements (LSB-first) */
	for (; i < dim; i++)
	{
		int byte_idx = i / 8;
		int bit_idx	 = i % 8;
		int bit		 = (bits[byte_idx] >> bit_idx) & 1;
		if (bit)
			result += transformed[i];
	}

	return result;
}

/*
 * AVX2 sign extraction for RaBitQ encoding.
 *
 * Extracts sign bits from transformed floats into packed bytes.
 * Uses AVX2 compare and movemask to generate 8 bits at a time.
 */
MKT_TARGET_AVX2 void
mkt_rabitq_extract_signs_avx2(
		const float *transformed, uint8_t *bits, Dimension dim)
{
	__m256 zero = _mm256_setzero_ps();

	/* Main loop: process 8 floats -> 1 byte at a time */
	Dimension i = 0;
	for (; i + 8 <= dim; i += 8)
	{
		/* Load 8 floats */
		__m256 t = _mm256_loadu_ps(transformed + i);

		/* Compare > 0 */
		__m256 cmp = _mm256_cmp_ps(t, zero, _CMP_GT_OQ);

		/* Extract sign bits as 8-bit mask (LSB-first matches our bit order) */
		int mask = _mm256_movemask_ps(cmp);

		/* Store as single byte */
		bits[i / 8] = (uint8_t)mask;
	}

	/* Handle tail elements */
	if (i < dim)
	{
		int byte_idx   = i / 8;
		bits[byte_idx] = 0;
		for (; i < dim; i++)
		{
			int bit_idx = i % 8;
			if (transformed[i] > 0)
				bits[byte_idx] |= (1 << bit_idx);
		}
	}
}

#endif /* x86_64 */

#endif /* MKT_SIMD_FULL */
