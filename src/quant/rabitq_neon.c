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

#endif /* aarch64 */

#endif /* MKT_SIMD_FULL */
