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
#include "quant/rabitq.h"

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

/*
 * AVX2 popcount helper using nibble lookup table (Mula's method).
 *
 * Uses _mm256_shuffle_epi8 as a parallel 4-bit lookup table to count
 * bits in each byte.
 */
MKT_TARGET_AVX2 static inline __m256i
popcount_avx2(__m256i v)
{
	const __m256i lut = _mm256_setr_epi8(
			0,
			1,
			1,
			2,
			1,
			2,
			2,
			3,
			1,
			2,
			2,
			3,
			2,
			3,
			3,
			4,
			0,
			1,
			1,
			2,
			1,
			2,
			2,
			3,
			1,
			2,
			2,
			3,
			2,
			3,
			3,
			4);
	const __m256i mask_lo = _mm256_set1_epi8(0x0F);

	__m256i lo = _mm256_and_si256(v, mask_lo);
	__m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), mask_lo);

	return _mm256_add_epi8(
			_mm256_shuffle_epi8(lut, lo), _mm256_shuffle_epi8(lut, hi));
}

/*
 * AVX2 Hamming distance using lookup-table popcount.
 *
 * Processes 32 bytes per iteration. Uses _mm256_sad_epu8 to
 * horizontally sum byte-level popcounts into 64-bit accumulators.
 */
MKT_TARGET_AVX2 uint32_t
mkt_rabitq_hamming_avx2(
		const uint8_t *a, const uint8_t *b, uint32_t packed_bytes)
{
	__m256i total = _mm256_setzero_si256();

	uint32_t i = 0;
	for (; i + 32 <= packed_bytes; i += 32)
	{
		__m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
		__m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));
		__m256i x  = _mm256_xor_si256(va, vb);

		__m256i pc = popcount_avx2(x);

		/* Sum byte popcounts into 64-bit accumulators via SAD */
		total = _mm256_add_epi64(
				total, _mm256_sad_epu8(pc, _mm256_setzero_si256()));
	}

	uint64_t result = mkt_horizontal_sum_epi64_avx2(total);

	/* Scalar tail */
	for (; i < packed_bytes; i++)
		result += (uint64_t)__builtin_popcount(a[i] ^ b[i]);

	return (uint32_t)result;
}

/*
 * Multi-candidate AVX2 Hamming distance.
 */
MKT_TARGET_AVX2 void
mkt_rabitq_hamming_multi_avx2(
		const uint8_t *query_bits,
		const uint8_t *data_bits,
		uint32_t	   stride,
		uint32_t	   packed_bytes,
		uint32_t	   count,
		uint32_t	  *results)
{
	for (uint32_t c = 0; c < count; c++)
	{
		results[c] = mkt_rabitq_hamming_avx2(
				query_bits, data_bits + c * stride, packed_bytes);
	}
}

/*
 * AVX2 multi-candidate vertical inner product.
 *
 * Processes 4 candidates per dimension chunk. Each iteration:
 * 1. Load 8 floats from transformed[] (1 ymm register)
 * 2. For each of 4 candidates: expand 1 byte to mask, masked-and-add
 * 3. After all dimensions: horizontal sum each accumulator
 *
 * Tail candidates (count % 4) use the single-candidate kernel.
 */
MKT_TARGET_AVX2 void
mkt_rabitq_inner_product_multi_avx2(
		const float	  *transformed,
		const uint8_t *bits,
		uint32_t	   stride,
		Dimension	   dim,
		uint32_t	   count,
		float		  *results)
{
	uint32_t groups = count / 4;
	uint32_t tail	= count % 4;

	for (uint32_t g = 0; g < groups; g++)
	{
		uint32_t base = g * 4;

		const uint8_t *b0 = bits + (size_t)base * stride;
		const uint8_t *b1 = bits + (size_t)(base + 1) * stride;
		const uint8_t *b2 = bits + (size_t)(base + 2) * stride;
		const uint8_t *b3 = bits + (size_t)(base + 3) * stride;

		__m256 sum0 = _mm256_setzero_ps();
		__m256 sum1 = _mm256_setzero_ps();
		__m256 sum2 = _mm256_setzero_ps();
		__m256 sum3 = _mm256_setzero_ps();

		/* Main loop: process 8 floats (1 byte of bits) at a time */
		Dimension i = 0;
		for (; i + 8 <= dim; i += 8)
		{
			__m256 t = _mm256_loadu_ps(transformed + i);

			uint32_t bi = i / 8;

			__m256 mask0 = expand_byte_to_mask_avx2(b0[bi]);
			__m256 mask1 = expand_byte_to_mask_avx2(b1[bi]);
			__m256 mask2 = expand_byte_to_mask_avx2(b2[bi]);
			__m256 mask3 = expand_byte_to_mask_avx2(b3[bi]);

			sum0 = _mm256_add_ps(sum0, _mm256_and_ps(t, mask0));
			sum1 = _mm256_add_ps(sum1, _mm256_and_ps(t, mask1));
			sum2 = _mm256_add_ps(sum2, _mm256_and_ps(t, mask2));
			sum3 = _mm256_add_ps(sum3, _mm256_and_ps(t, mask3));
		}

		results[base + 0] = mkt_horizontal_sum_avx2(sum0);
		results[base + 1] = mkt_horizontal_sum_avx2(sum1);
		results[base + 2] = mkt_horizontal_sum_avx2(sum2);
		results[base + 3] = mkt_horizontal_sum_avx2(sum3);

		/* Scalar tail for remaining dimensions */
		for (; i < dim; i++)
		{
			int byte_idx = i / 8;
			int bit_idx	 = i % 8;

			if ((b0[byte_idx] >> bit_idx) & 1)
				results[base + 0] += transformed[i];
			if ((b1[byte_idx] >> bit_idx) & 1)
				results[base + 1] += transformed[i];
			if ((b2[byte_idx] >> bit_idx) & 1)
				results[base + 2] += transformed[i];
			if ((b3[byte_idx] >> bit_idx) & 1)
				results[base + 3] += transformed[i];
		}
	}

	/* Handle remaining candidates with single-candidate kernel */
	for (uint32_t i = groups * 4; i < groups * 4 + tail; i++)
	{
		results[i] = mkt_rabitq_inner_product_avx2(
				transformed, bits + (size_t)i * stride, dim);
	}
}

#endif /* x86_64 */

#endif /* MKT_SIMD_FULL */
