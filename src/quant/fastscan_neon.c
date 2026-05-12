/*
 * fastscan_neon.c - ARM NEON table-lookup accumulate kernel
 *
 * NEON has 128-bit vectors and VQTBL1Q for 16-byte table lookups.
 * Each iteration processes one 16-byte code block (one subquantizer)
 * + one 16-byte LUT block, mapping directly to the fastscan layout
 * without cross-lane reductions.
 *
 * Layout mapping after VTBL + even/odd byte accumulation:
 *   accu0 (u16x8) = total IP for vectors 0..7
 *   accu1 (u16x8) = total IP for vectors 8..15
 *   accu2 (u16x8) = total IP for vectors 16..23
 *   accu3 (u16x8) = total IP for vectors 24..31
 *
 * This follows from kPerm0 = {0,8,1,9,2,10,3,11,4,12,5,13,6,14,7,15}:
 *   byte[2i]  of res_lo is for vector kPerm0[2i]   = i      (i in 0..7)
 *   byte[2i+1] of res_lo is for vector kPerm0[2i+1] = i + 8
 *   byte[2i]  of res_hi is for vector kPerm0[2i]+16 = i + 16
 *   byte[2i+1] of res_hi is for vector kPerm0[2i+1]+16 = i + 24
 */

#include "mkt_config.h"

#ifdef MKT_SIMD_FULL

#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>
#include <stdint.h>

#include "quant/fastscan.h"

/* Accumulate one VQTBL1 result into even/odd byte accumulators.
 * a_even += res (as u16, low byte of each lane = byte[2i])
 * a_odd  += res >> 8 (as u16, low byte = byte[2i+1]) */
#define ACCUM_RES(res, a_even, a_odd)                             \
	do                                                            \
	{                                                             \
		uint16x8_t r16_ = vreinterpretq_u16_u8(res);              \
		a_even			= vaddq_u16(a_even, r16_);                \
		a_odd			= vaddq_u16(a_odd, vshrq_n_u16(r16_, 8)); \
	} while (0)

void
mkt_fastscan_accumulate_neon(
		const uint8_t *codes,
		const uint8_t *lut,
		uint16_t	  *accum,
		Dimension	   dim)
{
	uint32_t code_length = MKT_FASTSCAN_GROUP_BYTES(dim);

	const uint8x16_t lo_mask = vdupq_n_u8(0x0F);

	/* Two banks of accumulators (a, b) to break the add-chain across
	 * unrolled iterations. Combined at the end. code_length is always
	 * a multiple of 32: each column (8 dims) contributes 2 sq blocks. */
	uint16x8_t accu0a = vdupq_n_u16(0), accu0b = vdupq_n_u16(0);
	uint16x8_t accu1a = vdupq_n_u16(0), accu1b = vdupq_n_u16(0);
	uint16x8_t accu2a = vdupq_n_u16(0), accu2b = vdupq_n_u16(0);
	uint16x8_t accu3a = vdupq_n_u16(0), accu3b = vdupq_n_u16(0);

	for (uint32_t i = 0; i < code_length; i += 32)
	{
		uint8x16_t c0 = vld1q_u8(codes + i);
		uint8x16_t c1 = vld1q_u8(codes + i + 16);
		uint8x16_t t0 = vld1q_u8(lut + i);
		uint8x16_t t1 = vld1q_u8(lut + i + 16);

		uint8x16_t lo0 = vandq_u8(c0, lo_mask);
		uint8x16_t hi0 = vshrq_n_u8(c0, 4);
		uint8x16_t lo1 = vandq_u8(c1, lo_mask);
		uint8x16_t hi1 = vshrq_n_u8(c1, 4);

		uint8x16_t rlo0 = vqtbl1q_u8(t0, lo0);
		uint8x16_t rhi0 = vqtbl1q_u8(t0, hi0);
		uint8x16_t rlo1 = vqtbl1q_u8(t1, lo1);
		uint8x16_t rhi1 = vqtbl1q_u8(t1, hi1);

		ACCUM_RES(rlo0, accu0a, accu1a);
		ACCUM_RES(rhi0, accu2a, accu3a);
		ACCUM_RES(rlo1, accu0b, accu1b);
		ACCUM_RES(rhi1, accu2b, accu3b);
	}

	uint16x8_t accu0 = vaddq_u16(accu0a, accu0b);
	uint16x8_t accu1 = vaddq_u16(accu1a, accu1b);
	uint16x8_t accu2 = vaddq_u16(accu2a, accu2b);
	uint16x8_t accu3 = vaddq_u16(accu3a, accu3b);

	/* Remove upper byte contamination from even-byte accumulators */
	accu0 = vsubq_u16(accu0, vshlq_n_u16(accu1, 8));
	accu2 = vsubq_u16(accu2, vshlq_n_u16(accu3, 8));

	vst1q_u16(accum, accu0);
	vst1q_u16(accum + 8, accu1);
	vst1q_u16(accum + 16, accu2);
	vst1q_u16(accum + 24, accu3);
}

#undef ACCUM_RES

/* ----------------------------------------------------------------
 * NEON high-accuracy accumulate (uint16 LUT → int32 output)
 *
 * Two VQTBL1 passes per code block: one for lo-byte table, one for
 * hi-byte table. Result = lo + (hi << 8) widened to int32.
 *
 * LUT layout per group of 4 sqs (128B):
 *   bytes 0..63: 4 x 16-byte lo-byte tables
 *   bytes 64..127: 4 x 16-byte hi-byte tables
 * ---------------------------------------------------------------- */

#define HACC_ACCUM(res, a_even, a_odd)                            \
	do                                                            \
	{                                                             \
		uint16x8_t r16_ = vreinterpretq_u16_u8(res);              \
		a_even			= vaddq_u16(a_even, r16_);                \
		a_odd			= vaddq_u16(a_odd, vshrq_n_u16(r16_, 8)); \
	} while (0)

void
mkt_fastscan_accumulate_hacc_neon(
		const uint8_t *codes,
		const uint8_t *lut,
		int32_t		  *accum,
		Dimension	   dim)
{
	uint32_t		 nsq	 = MKT_FASTSCAN_NSQ(dim);
	const uint8x16_t lo_mask = vdupq_n_u8(0x0F);

	/* Dual accumulator banks (A, B) for unrolled iterations.
	 * For paired sqs (m, m+1) with m even, m%4 is 0 or 2, so both
	 * tables live in adjacent 16-byte slots within the same 128B
	 * group-of-4 LUT block — adjacent loads. */
	uint16x8_t accu_loA[4], accu_loB[4];
	uint16x8_t accu_hiA[4], accu_hiB[4];
	for (int k = 0; k < 4; k++)
	{
		accu_loA[k] = vdupq_n_u16(0);
		accu_loB[k] = vdupq_n_u16(0);
		accu_hiA[k] = vdupq_n_u16(0);
		accu_hiB[k] = vdupq_n_u16(0);
	}

	uint32_t nsq_paired = nsq & ~1u;
	uint32_t m;
	for (m = 0; m < nsq_paired; m += 2)
	{
		uint8x16_t c0 = vld1q_u8(codes);
		uint8x16_t c1 = vld1q_u8(codes + 16);

		uint8x16_t lo0 = vandq_u8(c0, lo_mask);
		uint8x16_t hi0 = vshrq_n_u8(c0, 4);
		uint8x16_t lo1 = vandq_u8(c1, lo_mask);
		uint8x16_t hi1 = vshrq_n_u8(c1, 4);

		uint32_t	   g4  = m / 4;
		uint32_t	   p4  = m & 3; /* 0 or 2 since m is even */
		const uint8_t *loA = lut + g4 * 128 + p4 * 16;
		const uint8_t *hiA = loA + 64;

		uint8x16_t tab_loA = vld1q_u8(loA);
		uint8x16_t tab_loB = vld1q_u8(loA + 16);
		uint8x16_t tab_hiA = vld1q_u8(hiA);
		uint8x16_t tab_hiB = vld1q_u8(hiA + 16);

		HACC_ACCUM(vqtbl1q_u8(tab_loA, lo0), accu_loA[0], accu_loA[1]);
		HACC_ACCUM(vqtbl1q_u8(tab_loA, hi0), accu_loA[2], accu_loA[3]);
		HACC_ACCUM(vqtbl1q_u8(tab_hiA, lo0), accu_hiA[0], accu_hiA[1]);
		HACC_ACCUM(vqtbl1q_u8(tab_hiA, hi0), accu_hiA[2], accu_hiA[3]);

		HACC_ACCUM(vqtbl1q_u8(tab_loB, lo1), accu_loB[0], accu_loB[1]);
		HACC_ACCUM(vqtbl1q_u8(tab_loB, hi1), accu_loB[2], accu_loB[3]);
		HACC_ACCUM(vqtbl1q_u8(tab_hiB, lo1), accu_hiB[0], accu_hiB[1]);
		HACC_ACCUM(vqtbl1q_u8(tab_hiB, hi1), accu_hiB[2], accu_hiB[3]);

		codes += 32;
	}

	uint16x8_t accu_lo[4];
	uint16x8_t accu_hi[4];
	for (int k = 0; k < 4; k++)
	{
		accu_lo[k] = vaddq_u16(accu_loA[k], accu_loB[k]);
		accu_hi[k] = vaddq_u16(accu_hiA[k], accu_hiB[k]);
	}

	/* Odd-tail single sq if nsq is odd */
	if (m < nsq)
	{
		uint8x16_t c  = vld1q_u8(codes);
		uint8x16_t lo = vandq_u8(c, lo_mask);
		uint8x16_t hi = vshrq_n_u8(c, 4);

		uint32_t	   g4 = m / 4;
		uint32_t	   p4 = m & 3;
		const uint8_t *l  = lut + g4 * 128 + p4 * 16;
		const uint8_t *h  = l + 64;

		uint8x16_t tab_lo = vld1q_u8(l);
		uint8x16_t tab_hi = vld1q_u8(h);

		HACC_ACCUM(vqtbl1q_u8(tab_lo, lo), accu_lo[0], accu_lo[1]);
		HACC_ACCUM(vqtbl1q_u8(tab_lo, hi), accu_lo[2], accu_lo[3]);
		HACC_ACCUM(vqtbl1q_u8(tab_hi, lo), accu_hi[0], accu_hi[1]);
		HACC_ACCUM(vqtbl1q_u8(tab_hi, hi), accu_hi[2], accu_hi[3]);
	}

	/* Remove upper byte contamination from even-byte accumulators.
	 * The odd-byte accumulators (1, 3) are already clean. */
	accu_lo[0] = vsubq_u16(accu_lo[0], vshlq_n_u16(accu_lo[1], 8));
	accu_lo[2] = vsubq_u16(accu_lo[2], vshlq_n_u16(accu_lo[3], 8));
	accu_hi[0] = vsubq_u16(accu_hi[0], vshlq_n_u16(accu_hi[1], 8));
	accu_hi[2] = vsubq_u16(accu_hi[2], vshlq_n_u16(accu_hi[3], 8));

	/* Combine lo + (hi << 8) into int32 output.
	 * accu_lo[k] / accu_hi[k] each hold u16 results for 8 vectors:
	 *   k=0 -> vectors 0..7,  k=1 -> vectors 8..15
	 *   k=2 -> vectors 16..23, k=3 -> vectors 24..31 */
	for (int blk = 0; blk < 4; blk++)
	{
		uint32x4_t lo_lo = vmovl_u16(vget_low_u16(accu_lo[blk]));
		uint32x4_t lo_hi = vmovl_u16(vget_high_u16(accu_lo[blk]));
		uint32x4_t hi_lo = vmovl_u16(vget_low_u16(accu_hi[blk]));
		uint32x4_t hi_hi = vmovl_u16(vget_high_u16(accu_hi[blk]));

		int32x4_t r_lo = vreinterpretq_s32_u32(
				vaddq_u32(lo_lo, vshlq_n_u32(hi_lo, 8)));
		int32x4_t r_hi = vreinterpretq_s32_u32(
				vaddq_u32(lo_hi, vshlq_n_u32(hi_hi, 8)));

		vst1q_s32(accum + blk * 8, r_lo);
		vst1q_s32(accum + blk * 8 + 4, r_hi);
	}
}

#undef HACC_ACCUM

#endif /* aarch64 */

#endif /* MKT_SIMD_FULL */
