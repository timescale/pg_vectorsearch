/*
 * fastscan_sve2.c - SVE2 table-lookup accumulate kernel
 *
 * SVE2 version of the fastscan accumulate kernel. At 128-bit VL (e.g.
 * Neoverse-V2 / Graviton 4) this is structurally equivalent to the NEON
 * kernel; on wider SVE2 cores (256-bit VL, 512-bit VL) the same source
 * scales automatically via the VL-agnostic loop bound.
 *
 * The file-level `#pragma GCC target("+sve2")` enables SVE2 intrinsics
 * regardless of the project-wide -march/-mcpu flags, so a portable
 * binary built without -march=native still picks this up at runtime
 * when the CPU advertises HWCAP2_SVE2.
 *
 * Algorithm: same as fastscan_neon.c. The code stream is processed in
 * fixed 16-byte LUT/code blocks (one subquantizer per block). We unroll
 * 4 such blocks per iteration (64B) to break the dependency chain
 * across 4 banks of even/odd accumulators.
 *
 * SVE2 USRA (svsra_n_u16) folds shift-right + add, matching the NEON
 * vsraq_n_u16 trick. svtbl_u8 is the SVE counterpart of vqtbl1q_u8;
 * within a 128-bit segment its semantics match (indices >= 16 yield 0,
 * which is exactly the lookup behavior we want for 4-bit nibbles).
 */

#include "mkt_config.h"

#ifdef MKT_SIMD_FULL

#if defined(__aarch64__) || defined(_M_ARM64)

/* Enable SVE2 for this entire translation unit. This must come before
 * <arm_sve.h> so __ARM_FEATURE_SVE2 is defined when the header is
 * parsed. */
#pragma GCC target("+sve2")

#include <arm_sve.h>
#include <stdint.h>
#include <string.h>

#include "quant/fastscan.h"

/*
 * Accumulate kernel: 4-bank, 64-byte unrolled.
 *
 * Mirrors the NEON layout exactly so the dispatch swap is apples-to-
 * apples. svtbl_u8 returns 0 for indices >= svcntb(); at VL=128 indices
 * >=16 are out-of-range nibbles, but our LUT lookup uses index ranges
 * [0,15] (4-bit nibbles), so this is safe.
 *
 * On Graviton 4 (VL=128), 64B/iter = one outer loop step. On wider VL
 * cores the inner svld1 / svtbl ops cover more bytes per call, but the
 * 4-bank structure is preserved by stride-16 indexing within the iter.
 */
__attribute__((target("+sve2"))) void
mkt_fastscan_accumulate_sve2(
		const uint8_t *codes,
		const uint8_t *lut,
		uint16_t	  *accum,
		Dimension	   dim)
{
	uint32_t code_length = MKT_FASTSCAN_GROUP_BYTES(dim);

	/* All operations use the full 128-bit predicate (VL=128 on V2).
	 * We process exactly 16 bytes per svld1, so a full predicate is
	 * correct regardless of VL: it just means "use all elements". */
	svbool_t pg = svptrue_b8();

	/* 4 banks of even/odd accumulators, mirroring the NEON kernel.
	 * Each accu*X holds 8 u16 values (16 bytes), tracking 8 vectors. */
	svuint16_t accu0a = svdup_n_u16(0), accu0b = svdup_n_u16(0);
	svuint16_t accu0c = svdup_n_u16(0), accu0d = svdup_n_u16(0);
	svuint16_t accu1a = svdup_n_u16(0), accu1b = svdup_n_u16(0);
	svuint16_t accu1c = svdup_n_u16(0), accu1d = svdup_n_u16(0);
	svuint16_t accu2a = svdup_n_u16(0), accu2b = svdup_n_u16(0);
	svuint16_t accu2c = svdup_n_u16(0), accu2d = svdup_n_u16(0);
	svuint16_t accu3a = svdup_n_u16(0), accu3b = svdup_n_u16(0);
	svuint16_t accu3c = svdup_n_u16(0), accu3d = svdup_n_u16(0);

	uint32_t i = 0;

	/* Main loop: 64B per iteration (4 sq blocks). */
	for (; i + 64 <= code_length; i += 64)
	{
		svuint8_t c0 = svld1_u8(pg, codes + i);
		svuint8_t c1 = svld1_u8(pg, codes + i + 16);
		svuint8_t c2 = svld1_u8(pg, codes + i + 32);
		svuint8_t c3 = svld1_u8(pg, codes + i + 48);

		svuint8_t t0 = svld1_u8(pg, lut + i);
		svuint8_t t1 = svld1_u8(pg, lut + i + 16);
		svuint8_t t2 = svld1_u8(pg, lut + i + 32);
		svuint8_t t3 = svld1_u8(pg, lut + i + 48);

		svuint8_t lo0 = svand_n_u8_x(pg, c0, 0x0F);
		svuint8_t hi0 = svlsr_n_u8_x(pg, c0, 4);
		svuint8_t lo1 = svand_n_u8_x(pg, c1, 0x0F);
		svuint8_t hi1 = svlsr_n_u8_x(pg, c1, 4);
		svuint8_t lo2 = svand_n_u8_x(pg, c2, 0x0F);
		svuint8_t hi2 = svlsr_n_u8_x(pg, c2, 4);
		svuint8_t lo3 = svand_n_u8_x(pg, c3, 0x0F);
		svuint8_t hi3 = svlsr_n_u8_x(pg, c3, 4);

		svuint16_t rlo0 = svreinterpret_u16_u8(svtbl_u8(t0, lo0));
		svuint16_t rhi0 = svreinterpret_u16_u8(svtbl_u8(t0, hi0));
		svuint16_t rlo1 = svreinterpret_u16_u8(svtbl_u8(t1, lo1));
		svuint16_t rhi1 = svreinterpret_u16_u8(svtbl_u8(t1, hi1));
		svuint16_t rlo2 = svreinterpret_u16_u8(svtbl_u8(t2, lo2));
		svuint16_t rhi2 = svreinterpret_u16_u8(svtbl_u8(t2, hi2));
		svuint16_t rlo3 = svreinterpret_u16_u8(svtbl_u8(t3, lo3));
		svuint16_t rhi3 = svreinterpret_u16_u8(svtbl_u8(t3, hi3));

		/* even += res, odd += res >> 8 (folded into USRA) */
		accu0a = svadd_u16_x(pg, accu0a, rlo0);
		accu1a = svsra_n_u16(accu1a, rlo0, 8);
		accu2a = svadd_u16_x(pg, accu2a, rhi0);
		accu3a = svsra_n_u16(accu3a, rhi0, 8);

		accu0b = svadd_u16_x(pg, accu0b, rlo1);
		accu1b = svsra_n_u16(accu1b, rlo1, 8);
		accu2b = svadd_u16_x(pg, accu2b, rhi1);
		accu3b = svsra_n_u16(accu3b, rhi1, 8);

		accu0c = svadd_u16_x(pg, accu0c, rlo2);
		accu1c = svsra_n_u16(accu1c, rlo2, 8);
		accu2c = svadd_u16_x(pg, accu2c, rhi2);
		accu3c = svsra_n_u16(accu3c, rhi2, 8);

		accu0d = svadd_u16_x(pg, accu0d, rlo3);
		accu1d = svsra_n_u16(accu1d, rlo3, 8);
		accu2d = svadd_u16_x(pg, accu2d, rhi3);
		accu3d = svsra_n_u16(accu3d, rhi3, 8);
	}

	/* Tail: remaining 32B block (when code_length not a multiple of 64). */
	if (i < code_length)
	{
		svuint8_t c0 = svld1_u8(pg, codes + i);
		svuint8_t c1 = svld1_u8(pg, codes + i + 16);
		svuint8_t t0 = svld1_u8(pg, lut + i);
		svuint8_t t1 = svld1_u8(pg, lut + i + 16);

		svuint8_t lo0 = svand_n_u8_x(pg, c0, 0x0F);
		svuint8_t hi0 = svlsr_n_u8_x(pg, c0, 4);
		svuint8_t lo1 = svand_n_u8_x(pg, c1, 0x0F);
		svuint8_t hi1 = svlsr_n_u8_x(pg, c1, 4);

		svuint16_t rlo0 = svreinterpret_u16_u8(svtbl_u8(t0, lo0));
		svuint16_t rhi0 = svreinterpret_u16_u8(svtbl_u8(t0, hi0));
		svuint16_t rlo1 = svreinterpret_u16_u8(svtbl_u8(t1, lo1));
		svuint16_t rhi1 = svreinterpret_u16_u8(svtbl_u8(t1, hi1));

		accu0a = svadd_u16_x(pg, accu0a, rlo0);
		accu1a = svsra_n_u16(accu1a, rlo0, 8);
		accu2a = svadd_u16_x(pg, accu2a, rhi0);
		accu3a = svsra_n_u16(accu3a, rhi0, 8);

		accu0b = svadd_u16_x(pg, accu0b, rlo1);
		accu1b = svsra_n_u16(accu1b, rlo1, 8);
		accu2b = svadd_u16_x(pg, accu2b, rhi1);
		accu3b = svsra_n_u16(accu3b, rhi1, 8);
	}

	/* Combine 4 banks per accumulator. */
	svuint16_t accu0 = svadd_u16_x(
			pg,
			svadd_u16_x(pg, accu0a, accu0b),
			svadd_u16_x(pg, accu0c, accu0d));
	svuint16_t accu1 = svadd_u16_x(
			pg,
			svadd_u16_x(pg, accu1a, accu1b),
			svadd_u16_x(pg, accu1c, accu1d));
	svuint16_t accu2 = svadd_u16_x(
			pg,
			svadd_u16_x(pg, accu2a, accu2b),
			svadd_u16_x(pg, accu2c, accu2d));
	svuint16_t accu3 = svadd_u16_x(
			pg,
			svadd_u16_x(pg, accu3a, accu3b),
			svadd_u16_x(pg, accu3c, accu3d));

	/* Remove upper-byte contamination from even-byte accumulators
	 * (same trick as NEON: subtract odd<<8 from even). */
	accu0 = svsub_u16_x(pg, accu0, svlsl_n_u16_x(pg, accu1, 8));
	accu2 = svsub_u16_x(pg, accu2, svlsl_n_u16_x(pg, accu3, 8));

	svst1_u16(pg, accum, accu0);
	svst1_u16(pg, accum + 8, accu1);
	svst1_u16(pg, accum + 16, accu2);
	svst1_u16(pg, accum + 24, accu3);
}

/* ----------------------------------------------------------------
 * SVE2 high-accuracy accumulate (uint16 LUT → int32 output).
 *
 * Mirrors mkt_fastscan_accumulate_hacc_neon: two TBL passes per code
 * block (lo-byte and hi-byte LUTs). LUT layout is identical so that
 * data is interchangeable between the NEON and SVE2 paths.
 * ---------------------------------------------------------------- */

__attribute__((target("+sve2"))) void
mkt_fastscan_accumulate_hacc_sve2(
		const uint8_t *codes,
		const uint8_t *lut,
		int32_t		  *accum,
		Dimension	   dim)
{
	uint32_t nsq = MKT_FASTSCAN_NSQ(dim);
	svbool_t pg	 = svptrue_b8();

	/* 4 even/odd accumulator slots × 2 banks (A/B) × 2 tables (lo/hi).
	 * SVE types are sizeless so they cannot live in arrays; the
	 * compiler keeps these in registers. */
	svuint16_t loA0 = svdup_n_u16(0), loA1 = svdup_n_u16(0);
	svuint16_t loA2 = svdup_n_u16(0), loA3 = svdup_n_u16(0);
	svuint16_t loB0 = svdup_n_u16(0), loB1 = svdup_n_u16(0);
	svuint16_t loB2 = svdup_n_u16(0), loB3 = svdup_n_u16(0);
	svuint16_t hiA0 = svdup_n_u16(0), hiA1 = svdup_n_u16(0);
	svuint16_t hiA2 = svdup_n_u16(0), hiA3 = svdup_n_u16(0);
	svuint16_t hiB0 = svdup_n_u16(0), hiB1 = svdup_n_u16(0);
	svuint16_t hiB2 = svdup_n_u16(0), hiB3 = svdup_n_u16(0);

	uint32_t nsq_paired = nsq & ~1u;
	uint32_t m;
	for (m = 0; m < nsq_paired; m += 2)
	{
		svuint8_t c0 = svld1_u8(pg, codes);
		svuint8_t c1 = svld1_u8(pg, codes + 16);

		svuint8_t lo0 = svand_n_u8_x(pg, c0, 0x0F);
		svuint8_t hi0 = svlsr_n_u8_x(pg, c0, 4);
		svuint8_t lo1 = svand_n_u8_x(pg, c1, 0x0F);
		svuint8_t hi1 = svlsr_n_u8_x(pg, c1, 4);

		uint32_t	   g4  = m / 4;
		uint32_t	   p4  = m & 3;
		const uint8_t *loA = lut + g4 * 128 + p4 * 16;
		const uint8_t *hiA = loA + 64;

		svuint8_t tab_loA = svld1_u8(pg, loA);
		svuint8_t tab_loB = svld1_u8(pg, loA + 16);
		svuint8_t tab_hiA = svld1_u8(pg, hiA);
		svuint8_t tab_hiB = svld1_u8(pg, hiA + 16);

		svuint16_t r;
		r	 = svreinterpret_u16_u8(svtbl_u8(tab_loA, lo0));
		loA0 = svadd_u16_x(pg, loA0, r);
		loA1 = svsra_n_u16(loA1, r, 8);

		r	 = svreinterpret_u16_u8(svtbl_u8(tab_loA, hi0));
		loA2 = svadd_u16_x(pg, loA2, r);
		loA3 = svsra_n_u16(loA3, r, 8);

		r	 = svreinterpret_u16_u8(svtbl_u8(tab_hiA, lo0));
		hiA0 = svadd_u16_x(pg, hiA0, r);
		hiA1 = svsra_n_u16(hiA1, r, 8);

		r	 = svreinterpret_u16_u8(svtbl_u8(tab_hiA, hi0));
		hiA2 = svadd_u16_x(pg, hiA2, r);
		hiA3 = svsra_n_u16(hiA3, r, 8);

		r	 = svreinterpret_u16_u8(svtbl_u8(tab_loB, lo1));
		loB0 = svadd_u16_x(pg, loB0, r);
		loB1 = svsra_n_u16(loB1, r, 8);

		r	 = svreinterpret_u16_u8(svtbl_u8(tab_loB, hi1));
		loB2 = svadd_u16_x(pg, loB2, r);
		loB3 = svsra_n_u16(loB3, r, 8);

		r	 = svreinterpret_u16_u8(svtbl_u8(tab_hiB, lo1));
		hiB0 = svadd_u16_x(pg, hiB0, r);
		hiB1 = svsra_n_u16(hiB1, r, 8);

		r	 = svreinterpret_u16_u8(svtbl_u8(tab_hiB, hi1));
		hiB2 = svadd_u16_x(pg, hiB2, r);
		hiB3 = svsra_n_u16(hiB3, r, 8);

		codes += 32;
	}

	svuint16_t lo0acc = svadd_u16_x(pg, loA0, loB0);
	svuint16_t lo1acc = svadd_u16_x(pg, loA1, loB1);
	svuint16_t lo2acc = svadd_u16_x(pg, loA2, loB2);
	svuint16_t lo3acc = svadd_u16_x(pg, loA3, loB3);
	svuint16_t hi0acc = svadd_u16_x(pg, hiA0, hiB0);
	svuint16_t hi1acc = svadd_u16_x(pg, hiA1, hiB1);
	svuint16_t hi2acc = svadd_u16_x(pg, hiA2, hiB2);
	svuint16_t hi3acc = svadd_u16_x(pg, hiA3, hiB3);

	/* Odd-tail single sq if nsq is odd. */
	if (m < nsq)
	{
		svuint8_t c	 = svld1_u8(pg, codes);
		svuint8_t lo = svand_n_u8_x(pg, c, 0x0F);
		svuint8_t hi = svlsr_n_u8_x(pg, c, 4);

		uint32_t	   g4 = m / 4;
		uint32_t	   p4 = m & 3;
		const uint8_t *l  = lut + g4 * 128 + p4 * 16;
		const uint8_t *h  = l + 64;

		svuint8_t tab_lo = svld1_u8(pg, l);
		svuint8_t tab_hi = svld1_u8(pg, h);

		svuint16_t r;
		r	   = svreinterpret_u16_u8(svtbl_u8(tab_lo, lo));
		lo0acc = svadd_u16_x(pg, lo0acc, r);
		lo1acc = svsra_n_u16(lo1acc, r, 8);

		r	   = svreinterpret_u16_u8(svtbl_u8(tab_lo, hi));
		lo2acc = svadd_u16_x(pg, lo2acc, r);
		lo3acc = svsra_n_u16(lo3acc, r, 8);

		r	   = svreinterpret_u16_u8(svtbl_u8(tab_hi, lo));
		hi0acc = svadd_u16_x(pg, hi0acc, r);
		hi1acc = svsra_n_u16(hi1acc, r, 8);

		r	   = svreinterpret_u16_u8(svtbl_u8(tab_hi, hi));
		hi2acc = svadd_u16_x(pg, hi2acc, r);
		hi3acc = svsra_n_u16(hi3acc, r, 8);
	}

	/* Remove upper-byte contamination from even-byte accumulators. */
	lo0acc = svsub_u16_x(pg, lo0acc, svlsl_n_u16_x(pg, lo1acc, 8));
	lo2acc = svsub_u16_x(pg, lo2acc, svlsl_n_u16_x(pg, lo3acc, 8));
	hi0acc = svsub_u16_x(pg, hi0acc, svlsl_n_u16_x(pg, hi1acc, 8));
	hi2acc = svsub_u16_x(pg, hi2acc, svlsl_n_u16_x(pg, hi3acc, 8));

	/* Widen lo + (hi << 8) into int32 output per 8-vector block.
	 * svunpklo/svunpkhi each turn 8×u16 → 4×u32. At VL=128 the four
	 * accumulators (k=0..3) cover vectors 0..7, 8..15, 16..23, 24..31. */
	svbool_t pg32 = svptrue_b32();

#define WIDEN_AND_STORE(lo_acc, hi_acc, off)                                \
	do                                                                      \
	{                                                                       \
		svuint32_t lo_lo = svunpklo_u32(lo_acc);                            \
		svuint32_t lo_hi = svunpkhi_u32(lo_acc);                            \
		svuint32_t hi_lo = svunpklo_u32(hi_acc);                            \
		svuint32_t hi_hi = svunpkhi_u32(hi_acc);                            \
		svint32_t  r_lo	 = svreinterpret_s32_u32(                           \
				  svadd_u32_x(pg32, lo_lo, svlsl_n_u32_x(pg32, hi_lo, 8))); \
		svint32_t r_hi = svreinterpret_s32_u32(                             \
				svadd_u32_x(pg32, lo_hi, svlsl_n_u32_x(pg32, hi_hi, 8)));   \
		svst1_s32(pg32, accum + (off), r_lo);                               \
		svst1_s32(pg32, accum + (off) + 4, r_hi);                           \
	} while (0)

	WIDEN_AND_STORE(lo0acc, hi0acc, 0);
	WIDEN_AND_STORE(lo1acc, hi1acc, 8);
	WIDEN_AND_STORE(lo2acc, hi2acc, 16);
	WIDEN_AND_STORE(lo3acc, hi3acc, 24);

#undef WIDEN_AND_STORE
}

#endif /* aarch64 */

#endif /* MKT_SIMD_FULL */
