/*
 * rabitq_neon.c - ARM NEON optimized binary inner product for RaBitQ
 *
 * NEON processes 4 floats at a time. We expand each nibble (4 bits)
 * into 4 float masks for selective addition.
 */

#include "mkt_config.h"

#ifdef MKT_SIMD_FULL

#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>

#include "algo/simd_utils.h"
#include "mkt_types.h"
#include "quant/rabitq.h"

/*
 * Lookup table for nibble to mask expansion.
 * Each entry contains 4 uint32_t masks (0x0 or 0xFFFFFFFF).
 * Index by nibble value (0-15), LSB-first ordering.
 */
static const uint32_t g_nibble_masks[16][4] = {
		{0x00000000, 0x00000000, 0x00000000, 0x00000000}, /* 0000 */
		{0xFFFFFFFF, 0x00000000, 0x00000000, 0x00000000}, /* 0001 */
		{0x00000000, 0xFFFFFFFF, 0x00000000, 0x00000000}, /* 0010 */
		{0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0x00000000}, /* 0011 */
		{0x00000000, 0x00000000, 0xFFFFFFFF, 0x00000000}, /* 0100 */
		{0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000}, /* 0101 */
		{0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000}, /* 0110 */
		{0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000}, /* 0111 */
		{0x00000000, 0x00000000, 0x00000000, 0xFFFFFFFF}, /* 1000 */
		{0xFFFFFFFF, 0x00000000, 0x00000000, 0xFFFFFFFF}, /* 1001 */
		{0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF}, /* 1010 */
		{0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF}, /* 1011 */
		{0x00000000, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF}, /* 1100 */
		{0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF}, /* 1101 */
		{0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF}, /* 1110 */
		{0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF}, /* 1111 */
};

/*
 * Expand 4 bits to 4 float masks for NEON.
 *
 * Input: 4 bits (in low nibble, LSB-first: bit 0 -> float 0)
 * Output: uint32x4_t where each lane is either 0xFFFFFFFF or 0x0
 *
 * Uses a lookup table + vld1q_u32 for maximum portability.
 */
static inline uint32x4_t
expand_nibble_to_mask_neon(uint8_t nibble)
{
	return vld1q_u32(g_nibble_masks[nibble & 0x0F]);
}

/*
 * NEON sign extraction for RaBitQ encoding.
 *
 * Extracts sign bits (1 = positive, 0 = negative/zero) from transformed
 * vectors. NEON processes 4 floats at a time, so we process 8 floats
 * to produce one byte.
 */
void
mkt_rabitq_extract_signs_neon(
		const float *transformed, uint8_t *bits, Dimension dim)
{
	float32x4_t zero = vdupq_n_f32(0.0f);

	/* Main loop: process 8 floats -> 1 byte at a time */
	Dimension i = 0;
	for (; i + 8 <= dim; i += 8)
	{
		/* Load 8 floats in two batches of 4 */
		float32x4_t t0 = vld1q_f32(transformed + i);
		float32x4_t t1 = vld1q_f32(transformed + i + 4);

		/* Compare > 0 (returns all 1s or all 0s per lane) */
		uint32x4_t cmp0 = vcgtq_f32(t0, zero);
		uint32x4_t cmp1 = vcgtq_f32(t1, zero);

		/* Extract sign bits: narrow to 16-bit, then 8-bit, then extract */
		/* Each comparison result is 0xFFFFFFFF or 0x00000000 */
		/* We need to extract the MSB of each 32-bit lane */
		uint16x4_t narrow0	= vshrn_n_u32(cmp0, 16);
		uint16x4_t narrow1	= vshrn_n_u32(cmp1, 16);
		uint16x8_t combined = vcombine_u16(narrow0, narrow1);
		uint8x8_t  narrow8	= vshrn_n_u16(combined, 8);

		/* Now we have 8 bytes, each 0xFF or 0x00 */
		/* Extract bits: lane 0 -> bit 0, lane 1 -> bit 1, etc. */
		uint8_t byte = 0;
		uint8_t tmp[8];
		vst1_u8(tmp, narrow8);
		for (int j = 0; j < 8; j++)
		{
			if (tmp[j])
				byte |= (1 << j);
		}
		bits[i / 8] = byte;
	}

	/* Handle tail elements */
	if (i < dim)
	{
		int		byte_idx = i / 8;
		uint8_t byte	 = 0;

		for (int bit_idx = 0; i < dim && bit_idx < 8; i++, bit_idx++)
		{
			if (transformed[i] > 0.0f)
				byte |= (1 << bit_idx);
		}
		bits[byte_idx] = byte;
	}
}

/*
 * NEON binary inner product implementation.
 *
 * Processes 4 floats at a time using masked addition.
 */
float
mkt_rabitq_inner_product_neon(
		const float *transformed, const uint8_t *bits, Dimension dim)
{
	float32x4_t sum = vdupq_n_f32(0.0f);

	/* Main loop: process 8 floats (1 byte = 2 nibbles) at a time */
	Dimension i = 0;
	for (; i + 8 <= dim; i += 8)
	{
		uint8_t byte = bits[i / 8];

		/* Low nibble (bits for i+0..i+3, LSB-first) */
		uint8_t		low_nibble = byte & 0x0F;
		float32x4_t t0		   = vld1q_f32(transformed + i);
		uint32x4_t	mask0	   = expand_nibble_to_mask_neon(low_nibble);
		float32x4_t masked0	   = vreinterpretq_f32_u32(
				   vandq_u32(vreinterpretq_u32_f32(t0), mask0));
		sum = vaddq_f32(sum, masked0);

		/* High nibble (bits for i+4..i+7, LSB-first) */
		uint8_t		high_nibble = (byte >> 4) & 0x0F;
		float32x4_t t1			= vld1q_f32(transformed + i + 4);
		uint32x4_t	mask1		= expand_nibble_to_mask_neon(high_nibble);
		float32x4_t masked1		= vreinterpretq_f32_u32(
				vandq_u32(vreinterpretq_u32_f32(t1), mask1));
		sum = vaddq_f32(sum, masked1);
	}

	/* Process remaining 4 floats if any (LSB-first) */
	if (i + 4 <= dim)
	{
		uint8_t		byte   = bits[i / 8];
		uint8_t		nibble = (i % 8 == 0) ? byte & 0x0F : (byte >> 4) & 0x0F;
		float32x4_t t	   = vld1q_f32(transformed + i);
		uint32x4_t	mask   = expand_nibble_to_mask_neon(nibble);
		float32x4_t masked = vreinterpretq_f32_u32(
				vandq_u32(vreinterpretq_u32_f32(t), mask));
		sum = vaddq_f32(sum, masked);
		i += 4;
	}

	/* Horizontal sum */
	float result = mkt_horizontal_sum_neon(sum);

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
 * Helper: masked-add (sum += t & mask) as one expression.
 *
 * Compiler-friendly form that the NEON backend lowers to a single
 * AND + FADD (the reinterprets are zero-cost).
 */
static inline float32x4_t
masked_add_neon(float32x4_t sum, float32x4_t t, uint32x4_t mask)
{
	return vaddq_f32(
			sum,
			vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(t), mask)));
}

/*
 * NEON multi-candidate vertical inner product.
 *
 * Processes 8 candidates per dimension chunk using nibble lookup.
 * Each iteration loads 8 floats from transformed[] (two q-regs), then
 * for each of 8 candidates: expand low+high nibble to masks, masked-add.
 *
 * The 8-wide group exposes more in-flight independent FADDs to keep the
 * Apple Silicon SIMD pipes saturated. The LUT-based mask expansion is
 * preserved because vld1q_u32 from a small static table is faster than
 * a vdupq+vtstq combo on Apple Silicon (load ports are wider than the
 * GPR->SIMD transfer path).
 *
 * Tail candidates (count % 8) use the single-candidate kernel.
 */
void
mkt_rabitq_inner_product_multi_neon(
		const float	  *transformed,
		const uint8_t *bits,
		uint32_t	   stride,
		Dimension	   dim,
		uint32_t	   count,
		float		  *results)
{
	uint32_t groups = count / 8;
	uint32_t tail	= count % 8;

	for (uint32_t g = 0; g < groups; g++)
	{
		uint32_t base = g * 8;

		const uint8_t *b0 = bits + (size_t)(base + 0) * stride;
		const uint8_t *b1 = bits + (size_t)(base + 1) * stride;
		const uint8_t *b2 = bits + (size_t)(base + 2) * stride;
		const uint8_t *b3 = bits + (size_t)(base + 3) * stride;
		const uint8_t *b4 = bits + (size_t)(base + 4) * stride;
		const uint8_t *b5 = bits + (size_t)(base + 5) * stride;
		const uint8_t *b6 = bits + (size_t)(base + 6) * stride;
		const uint8_t *b7 = bits + (size_t)(base + 7) * stride;

		float32x4_t sum0 = vdupq_n_f32(0.0f);
		float32x4_t sum1 = vdupq_n_f32(0.0f);
		float32x4_t sum2 = vdupq_n_f32(0.0f);
		float32x4_t sum3 = vdupq_n_f32(0.0f);
		float32x4_t sum4 = vdupq_n_f32(0.0f);
		float32x4_t sum5 = vdupq_n_f32(0.0f);
		float32x4_t sum6 = vdupq_n_f32(0.0f);
		float32x4_t sum7 = vdupq_n_f32(0.0f);

		Dimension i = 0;
		for (; i + 8 <= dim; i += 8)
		{
			uint32_t bi = i / 8;

			/* Low nibble (floats i+0..i+3) */
			float32x4_t t0 = vld1q_f32(transformed + i);

			sum0 = masked_add_neon(
					sum0, t0, expand_nibble_to_mask_neon(b0[bi] & 0x0F));
			sum1 = masked_add_neon(
					sum1, t0, expand_nibble_to_mask_neon(b1[bi] & 0x0F));
			sum2 = masked_add_neon(
					sum2, t0, expand_nibble_to_mask_neon(b2[bi] & 0x0F));
			sum3 = masked_add_neon(
					sum3, t0, expand_nibble_to_mask_neon(b3[bi] & 0x0F));
			sum4 = masked_add_neon(
					sum4, t0, expand_nibble_to_mask_neon(b4[bi] & 0x0F));
			sum5 = masked_add_neon(
					sum5, t0, expand_nibble_to_mask_neon(b5[bi] & 0x0F));
			sum6 = masked_add_neon(
					sum6, t0, expand_nibble_to_mask_neon(b6[bi] & 0x0F));
			sum7 = masked_add_neon(
					sum7, t0, expand_nibble_to_mask_neon(b7[bi] & 0x0F));

			/* High nibble (floats i+4..i+7) */
			float32x4_t t1 = vld1q_f32(transformed + i + 4);

			sum0 = masked_add_neon(
					sum0,
					t1,
					expand_nibble_to_mask_neon((b0[bi] >> 4) & 0x0F));
			sum1 = masked_add_neon(
					sum1,
					t1,
					expand_nibble_to_mask_neon((b1[bi] >> 4) & 0x0F));
			sum2 = masked_add_neon(
					sum2,
					t1,
					expand_nibble_to_mask_neon((b2[bi] >> 4) & 0x0F));
			sum3 = masked_add_neon(
					sum3,
					t1,
					expand_nibble_to_mask_neon((b3[bi] >> 4) & 0x0F));
			sum4 = masked_add_neon(
					sum4,
					t1,
					expand_nibble_to_mask_neon((b4[bi] >> 4) & 0x0F));
			sum5 = masked_add_neon(
					sum5,
					t1,
					expand_nibble_to_mask_neon((b5[bi] >> 4) & 0x0F));
			sum6 = masked_add_neon(
					sum6,
					t1,
					expand_nibble_to_mask_neon((b6[bi] >> 4) & 0x0F));
			sum7 = masked_add_neon(
					sum7,
					t1,
					expand_nibble_to_mask_neon((b7[bi] >> 4) & 0x0F));
		}

		/* Process remaining 4 floats if any (only nibble half of a byte) */
		if (i + 4 <= dim)
		{
			uint32_t bi	   = i / 8;
			uint8_t	 shift = (i % 8 == 0) ? 0 : 4;

			float32x4_t t = vld1q_f32(transformed + i);

			sum0 = masked_add_neon(
					sum0,
					t,
					expand_nibble_to_mask_neon((b0[bi] >> shift) & 0x0F));
			sum1 = masked_add_neon(
					sum1,
					t,
					expand_nibble_to_mask_neon((b1[bi] >> shift) & 0x0F));
			sum2 = masked_add_neon(
					sum2,
					t,
					expand_nibble_to_mask_neon((b2[bi] >> shift) & 0x0F));
			sum3 = masked_add_neon(
					sum3,
					t,
					expand_nibble_to_mask_neon((b3[bi] >> shift) & 0x0F));
			sum4 = masked_add_neon(
					sum4,
					t,
					expand_nibble_to_mask_neon((b4[bi] >> shift) & 0x0F));
			sum5 = masked_add_neon(
					sum5,
					t,
					expand_nibble_to_mask_neon((b5[bi] >> shift) & 0x0F));
			sum6 = masked_add_neon(
					sum6,
					t,
					expand_nibble_to_mask_neon((b6[bi] >> shift) & 0x0F));
			sum7 = masked_add_neon(
					sum7,
					t,
					expand_nibble_to_mask_neon((b7[bi] >> shift) & 0x0F));
			i += 4;
		}

		results[base + 0] = mkt_horizontal_sum_neon(sum0);
		results[base + 1] = mkt_horizontal_sum_neon(sum1);
		results[base + 2] = mkt_horizontal_sum_neon(sum2);
		results[base + 3] = mkt_horizontal_sum_neon(sum3);
		results[base + 4] = mkt_horizontal_sum_neon(sum4);
		results[base + 5] = mkt_horizontal_sum_neon(sum5);
		results[base + 6] = mkt_horizontal_sum_neon(sum6);
		results[base + 7] = mkt_horizontal_sum_neon(sum7);

		/* Scalar tail for remaining dimensions */
		for (; i < dim; i++)
		{
			int	  byte_idx = i / 8;
			int	  bit_idx  = i % 8;
			float v		   = transformed[i];

			if ((b0[byte_idx] >> bit_idx) & 1)
				results[base + 0] += v;
			if ((b1[byte_idx] >> bit_idx) & 1)
				results[base + 1] += v;
			if ((b2[byte_idx] >> bit_idx) & 1)
				results[base + 2] += v;
			if ((b3[byte_idx] >> bit_idx) & 1)
				results[base + 3] += v;
			if ((b4[byte_idx] >> bit_idx) & 1)
				results[base + 4] += v;
			if ((b5[byte_idx] >> bit_idx) & 1)
				results[base + 5] += v;
			if ((b6[byte_idx] >> bit_idx) & 1)
				results[base + 6] += v;
			if ((b7[byte_idx] >> bit_idx) & 1)
				results[base + 7] += v;
		}
	}

	/* Handle remaining candidates with single-candidate kernel */
	for (uint32_t i = groups * 8; i < groups * 8 + tail; i++)
	{
		results[i] = mkt_rabitq_inner_product_neon(
				transformed, bits + (size_t)i * stride, dim);
	}
}

#endif /* aarch64 */

#endif /* MKT_SIMD_FULL */
