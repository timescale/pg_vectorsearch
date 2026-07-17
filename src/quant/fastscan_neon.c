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
#include <string.h>

#include "algo/simd_utils.h"
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

/* ----------------------------------------------------------------
 * NEON LUT construction
 *
 * The scalar builder (fastscan_build_lut_scalar /
 * fastscan_build_lut_hacc_scalar in fastscan.c) computes 16 entries
 * per subquantizer via scalar float ops plus a branchy per-entry
 * clamp -- it was never vectorized for ARM (only AVX-512 has a
 * hand-optimized builder; AVX2 and NEON both fell back to it). At
 * dim=768 (nsq=192), `mkt bench fastscan-kernel` measures ~369K
 * uint16-LUT builds/s scalar (~2.71us/build) vs ~804K/s here
 * (~1.24us/build), a ~2.2x improvement. Per probed cluster, the
 * scalar LUT build cost rivals or exceeds the whole fastscan
 * accumulate phase for that cluster (~2.2us for an average
 * ~240-vector cluster at ~9.3ns/vector), so it's a significant
 * fraction of the posting-scan phase at typical nlist/nprobe ratios.
 *
 * This builds all 4 entries for a given f_base value (f0/f4/f8/f12)
 * in one instruction via broadcast-add against a shared {0, s0, s1,
 * p01} vector, replacing the scalar per-entry adds. The clamp
 * (max/min against [0, range]) and float->uint conversion are done
 * 4-wide instead of once per entry with a branch.
 */

void
mkt_fastscan_build_lut_neon(
		const float *transformed,
		Dimension	 dim,
		uint8_t		*lut_out,
		float		*delta_out,
		float		*bias_out)
{
	uint32_t nsq	   = MKT_FASTSCAN_NSQ(dim);
	uint32_t nsq_pairs = MKT_FASTSCAN_NSQ_PAIRS(dim);

	/* Vectorized min/max: split positive/negative and sum */
	float32x4_t pos_sum = vdupq_n_f32(0.0f);
	float32x4_t neg_sum = vdupq_n_f32(0.0f);
	float32x4_t zero4	 = vdupq_n_f32(0.0f);

	Dimension d = 0;
	for (; d + 4 <= dim; d += 4)
	{
		float32x4_t v = vld1q_f32(transformed + d);
		pos_sum		  = vaddq_f32(pos_sum, vmaxq_f32(v, zero4));
		neg_sum		  = vaddq_f32(neg_sum, vminq_f32(v, zero4));
	}
	float global_max = mkt_horizontal_sum_neon(pos_sum);
	float global_min = mkt_horizontal_sum_neon(neg_sum);
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

	float bias_scaled = -global_min * inv_delta + 0.5f;

	memset(lut_out, 0, nsq_pairs * 2 * 16);

	float32x4_t lo_v = vdupq_n_f32(0.0f);
	float32x4_t hi_v = vdupq_n_f32((float)UINT8_MAX);

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

		float		addend_arr[4] = {0.0f, s0, s1, p01};
		float32x4_t addend		  = vld1q_f32(addend_arr);

		float32x4_t e0 = vaddq_f32(vdupq_n_f32(f0), addend);
		float32x4_t e1 = vaddq_f32(vdupq_n_f32(f4), addend);
		float32x4_t e2 = vaddq_f32(vdupq_n_f32(f8), addend);
		float32x4_t e3 = vaddq_f32(vdupq_n_f32(f12), addend);

		e0 = vminq_f32(vmaxq_f32(e0, lo_v), hi_v);
		e1 = vminq_f32(vmaxq_f32(e1, lo_v), hi_v);
		e2 = vminq_f32(vmaxq_f32(e2, lo_v), hi_v);
		e3 = vminq_f32(vmaxq_f32(e3, lo_v), hi_v);

		/* Truncate toward zero, matching scalar (int)(val) -- safe
		 * since values are already clamped to [0, UINT8_MAX]. */
		uint32x4_t i0 = vcvtq_u32_f32(e0);
		uint32x4_t i1 = vcvtq_u32_f32(e1);
		uint32x4_t i2 = vcvtq_u32_f32(e2);
		uint32x4_t i3 = vcvtq_u32_f32(e3);

		uint16x4_t n0 = vmovn_u32(i0);
		uint16x4_t n1 = vmovn_u32(i1);
		uint16x4_t n2 = vmovn_u32(i2);
		uint16x4_t n3 = vmovn_u32(i3);

		uint8x8_t b01 = vmovn_u16(vcombine_u16(n0, n1)); /* entries 0..7 */
		uint8x8_t b23 = vmovn_u16(vcombine_u16(n2, n3)); /* entries 8..15 */

		vst1_u8(out, b01);
		vst1_u8(out + 8, b23);

		q += 4;
	}
}

/* ----------------------------------------------------------------
 * NEON high-accuracy LUT construction (uint16 entries -> split
 * lo/hi byte tables). Same broadcast-add trick as the uint8
 * builder above; the extra work is splitting each uint16 entry
 * into its low and high byte before storing to the interleaved
 * lo/hi layout (see fastscan_build_lut_hacc_scalar).
 * ---------------------------------------------------------------- */

void
mkt_fastscan_build_lut_hacc_neon(
		const float *transformed,
		Dimension	 dim,
		uint8_t		*lut_out,
		float		*delta_out,
		float		*bias_out)
{
	uint32_t nsq = MKT_FASTSCAN_NSQ(dim);

	float32x4_t pos_sum = vdupq_n_f32(0.0f);
	float32x4_t neg_sum = vdupq_n_f32(0.0f);
	float32x4_t zero4	 = vdupq_n_f32(0.0f);

	Dimension d = 0;
	for (; d + 4 <= dim; d += 4)
	{
		float32x4_t v = vld1q_f32(transformed + d);
		pos_sum		  = vaddq_f32(pos_sum, vmaxq_f32(v, zero4));
		neg_sum		  = vaddq_f32(neg_sum, vminq_f32(v, zero4));
	}
	float global_max = mkt_horizontal_sum_neon(pos_sum);
	float global_min = mkt_horizontal_sum_neon(neg_sum);
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

	float32x4_t lo_v = vdupq_n_f32(0.0f);
	float32x4_t hi_v = vdupq_n_f32((float)UINT16_MAX);

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

		float		addend_arr[4] = {0.0f, s0, s1, p01};
		float32x4_t addend		  = vld1q_f32(addend_arr);

		float32x4_t e0 = vaddq_f32(vdupq_n_f32(f0), addend);
		float32x4_t e1 = vaddq_f32(vdupq_n_f32(f4), addend);
		float32x4_t e2 = vaddq_f32(vdupq_n_f32(f8), addend);
		float32x4_t e3 = vaddq_f32(vdupq_n_f32(f12), addend);

		e0 = vminq_f32(vmaxq_f32(e0, lo_v), hi_v);
		e1 = vminq_f32(vmaxq_f32(e1, lo_v), hi_v);
		e2 = vminq_f32(vmaxq_f32(e2, lo_v), hi_v);
		e3 = vminq_f32(vmaxq_f32(e3, lo_v), hi_v);

		uint32x4_t i0 = vcvtq_u32_f32(e0);
		uint32x4_t i1 = vcvtq_u32_f32(e1);
		uint32x4_t i2 = vcvtq_u32_f32(e2);
		uint32x4_t i3 = vcvtq_u32_f32(e3);

		uint16x4_t n0 = vmovn_u32(i0);
		uint16x4_t n1 = vmovn_u32(i1);
		uint16x4_t n2 = vmovn_u32(i2);
		uint16x4_t n3 = vmovn_u32(i3);

		/* n0/n1 -> entries 0..7, n2/n3 -> entries 8..15 */
		uint16x8_t lo8_16 = vcombine_u16(n0, n1);
		uint16x8_t hi8_16 = vcombine_u16(n2, n3);

		uint8x8_t lo_byte_a = vmovn_u16(lo8_16);	   /* low bytes 0..7 */
		uint8x8_t lo_byte_b = vmovn_u16(hi8_16);	   /* low bytes 8..15 */
		uint8x8_t hi_byte_a = vshrn_n_u16(lo8_16, 8); /* high bytes 0..7 */
		uint8x8_t hi_byte_b = vshrn_n_u16(hi8_16, 8); /* high bytes 8..15 */

		uint8x16_t lo_bytes = vcombine_u8(lo_byte_a, lo_byte_b);
		uint8x16_t hi_bytes = vcombine_u8(hi_byte_a, hi_byte_b);

		uint32_t group4		  = sq / 4;
		uint32_t pos_in_group = sq % 4;
		uint8_t *lo_dst		  = lut_out + group4 * 128 + pos_in_group * 16;
		uint8_t *hi_dst		  = lo_dst + 64;

		vst1q_u8(lo_dst, lo_bytes);
		vst1q_u8(hi_dst, hi_bytes);

		q += 4;
	}
}

#endif /* aarch64 */

#endif /* MKT_SIMD_FULL */
