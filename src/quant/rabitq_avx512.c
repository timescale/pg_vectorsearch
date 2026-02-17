/*
 * rabitq_avx512.c - AVX-512 optimized binary inner product for RaBitQ
 *
 * AVX-512 provides native masking support which makes this operation
 * extremely efficient. We can process 16 floats at a time using a
 * 16-bit mask directly.
 */

#include "mkt_config.h"

#ifdef MKT_SIMD_FULL

#if defined(__x86_64__) || defined(_M_X64)

#include <immintrin.h>

#include "algo/simd_utils.h"
#include "mkt_types.h"
#include "quant/rabitq.h"

/*
 * AVX-512 binary inner product implementation.
 *
 * Processes 16 floats at a time using native AVX-512 masking.
 * The mask register directly controls which elements participate
 * in the addition, making this very efficient.
 */
MKT_TARGET_AVX512 float
mkt_rabitq_inner_product_avx512(
		const float *transformed, const uint8_t *bits, Dimension dim)
{
	__m512 sum = _mm512_setzero_ps();

	/* Main loop: process 16 floats (2 bytes of bits) at a time */
	Dimension i = 0;
	for (; i + 16 <= dim; i += 16)
	{
		/* Load 16 transformed values */
		__m512 t = _mm512_loadu_ps(transformed + i);

		/* Get 2 bytes containing bits for these 16 floats (LSB-first) */
		/* bits[i/8] contains bits for i..i+7, bits[i/8+1] for i+8..i+15 */
		/* With LSB-first storage, bit order matches AVX-512 mask directly */
		uint8_t byte0 = bits[i / 8];	 /* bits for i+0..i+7 */
		uint8_t byte1 = bits[i / 8 + 1]; /* bits for i+8..i+15 */

		/* Combine into 16-bit mask (byte0 = low 8 bits, byte1 = high 8 bits)
		 */
		__mmask16 k = (__mmask16)((uint16_t)byte1 << 8 | byte0);

		/* Native masked add: only adds elements where mask bit is 1 */
		sum = _mm512_mask_add_ps(sum, k, sum, t);
	}

	/* Horizontal sum */
	float result = mkt_horizontal_sum_avx512(sum);

	/* Scalar tail for remaining elements */

	/* Scalar tail (LSB-first) */
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
 * AVX-512 sign extraction for RaBitQ encoding.
 *
 * Extracts sign bits from transformed floats into packed bytes.
 * Uses AVX-512 compare to generate a 16-bit mask directly.
 */
MKT_TARGET_AVX512 void
mkt_rabitq_extract_signs_avx512(
		const float *transformed, uint8_t *bits, Dimension dim)
{
	__m512 zero = _mm512_setzero_ps();

	/* Main loop: process 16 floats -> 2 bytes at a time */
	Dimension i = 0;
	for (; i + 16 <= dim; i += 16)
	{
		/* Load 16 floats */
		__m512 t = _mm512_loadu_ps(transformed + i);

		/* Compare > 0, returns 16-bit mask */
		__mmask16 mask = _mm512_cmp_ps_mask(t, zero, _CMP_GT_OQ);

		/* Store as 2 bytes (LSB-first: low byte = bits 0-7, high = 8-15) */
		bits[i / 8]		= (uint8_t)(mask & 0xFF);
		bits[i / 8 + 1] = (uint8_t)(mask >> 8);
	}

	/* Handle remaining floats (8 at a time with AVX2-style or scalar) */
	for (; i < dim; i++)
	{
		int byte_idx = i / 8;
		int bit_idx	 = i % 8;
		if (bit_idx == 0)
			bits[byte_idx] = 0; /* Clear byte on first bit */
		if (transformed[i] > 0)
			bits[byte_idx] |= (1 << bit_idx);
	}
}

/*
 * AVX-512 VPOPCNTDQ Hamming distance.
 *
 * Processes 512 bits (64 bytes) per iteration using XOR + VPOPCNTDQ.
 * VPOPCNTDQ computes popcount of each 64-bit element in a ZMM register.
 * Requires Ice Lake (2019+) or later.
 */
MKT_TARGET_AVX512_VPOPCNTDQ uint32_t
mkt_rabitq_hamming_avx512(
		const uint8_t *a, const uint8_t *b, uint32_t packed_bytes)
{
	__m512i total = _mm512_setzero_si512();

	uint32_t i = 0;
	for (; i + 64 <= packed_bytes; i += 64)
	{
		__m512i va = _mm512_loadu_si512(a + i);
		__m512i vb = _mm512_loadu_si512(b + i);
		__m512i x  = _mm512_xor_si512(va, vb);
		total	   = _mm512_add_epi64(total, _mm512_popcnt_epi64(x));
	}

	uint64_t result = mkt_horizontal_sum_epi64_avx512(total);

	/* Scalar tail */
	for (; i < packed_bytes; i++)
		result += (uint64_t)__builtin_popcount(a[i] ^ b[i]);

	return (uint32_t)result;
}

/*
 * Multi-candidate AVX-512 VPOPCNTDQ Hamming distance.
 */
MKT_TARGET_AVX512_VPOPCNTDQ void
mkt_rabitq_hamming_multi_avx512(
		const uint8_t *query_bits,
		const uint8_t *data_bits,
		uint32_t	   stride,
		uint32_t	   packed_bytes,
		uint32_t	   count,
		uint32_t	  *results)
{
	for (uint32_t c = 0; c < count; c++)
	{
		results[c] = mkt_rabitq_hamming_avx512(
				query_bits, data_bits + c * stride, packed_bytes);
	}
}

#endif /* x86_64 */

#endif /* MKT_SIMD_FULL */
