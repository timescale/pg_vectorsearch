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
#include "core/types.h"
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

/*
 * AVX-512 multi-candidate vertical inner product.
 *
 * Processes 4 candidates per dimension chunk. Each iteration:
 * 1. Load 16 floats from transformed[] (1 zmm register)
 * 2. For each of 4 candidates: load 2 bytes of bits, form mask,
 *    masked-add into that candidate's accumulator
 * 3. After all dimensions: horizontal sum each accumulator
 *
 * Tail candidates (count % 4) use the single-candidate kernel.
 */
MKT_TARGET_AVX512 void
mkt_rabitq_inner_product_multi_avx512(
		const float	  *transformed,
		const uint8_t *bits,
		uint32_t	   stride,
		Dimension	   dim,
		uint32_t	   count,
		float		  *results)
{
	/* Process groups of 4 candidates */
	uint32_t groups = count / 4;
	uint32_t tail	= count % 4;

	for (uint32_t g = 0; g < groups; g++)
	{
		uint32_t base = g * 4;

		const uint8_t *b0 = bits + (size_t)base * stride;
		const uint8_t *b1 = bits + (size_t)(base + 1) * stride;
		const uint8_t *b2 = bits + (size_t)(base + 2) * stride;
		const uint8_t *b3 = bits + (size_t)(base + 3) * stride;

		__m512 sum0 = _mm512_setzero_ps();
		__m512 sum1 = _mm512_setzero_ps();
		__m512 sum2 = _mm512_setzero_ps();
		__m512 sum3 = _mm512_setzero_ps();

		/* Main loop: process 16 floats at a time */
		Dimension i = 0;
		for (; i + 16 <= dim; i += 16)
		{
			__m512 t = _mm512_loadu_ps(transformed + i);

			uint32_t bi = i / 8;

			__mmask16 k0 = (__mmask16)((uint16_t)b0[bi + 1] << 8 | b0[bi]);
			__mmask16 k1 = (__mmask16)((uint16_t)b1[bi + 1] << 8 | b1[bi]);
			__mmask16 k2 = (__mmask16)((uint16_t)b2[bi + 1] << 8 | b2[bi]);
			__mmask16 k3 = (__mmask16)((uint16_t)b3[bi + 1] << 8 | b3[bi]);

			sum0 = _mm512_mask_add_ps(sum0, k0, sum0, t);
			sum1 = _mm512_mask_add_ps(sum1, k1, sum1, t);
			sum2 = _mm512_mask_add_ps(sum2, k2, sum2, t);
			sum3 = _mm512_mask_add_ps(sum3, k3, sum3, t);
		}

		results[base + 0] = mkt_horizontal_sum_avx512(sum0);
		results[base + 1] = mkt_horizontal_sum_avx512(sum1);
		results[base + 2] = mkt_horizontal_sum_avx512(sum2);
		results[base + 3] = mkt_horizontal_sum_avx512(sum3);

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
		results[i] = mkt_rabitq_inner_product_avx512(
				transformed, bits + (size_t)i * stride, dim);
	}
}

#endif /* x86_64 */

#endif /* MKT_SIMD_FULL */
