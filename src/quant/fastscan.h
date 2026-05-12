/*
 * fastscan.h - VPSHUFB-based fast scan for RaBitQ
 *
 * Repacks 1-bit RaBitQ sign codes into 4-bit nibble layout for
 * VPSHUFB table-lookup distance computation. Processes 32 vectors
 * per batch using the even/odd byte accumulation trick.
 *
 * Based on the approach from the RaBitQ Library (Gao & Long) and
 * FAISS FastScan (Andre et al., "Cache locality is not enough").
 *
 * Data layout per batch of 32 vectors:
 *   - Packed codes: dim/8 columns x 32 bytes, with kPerm0
 *     vector interleaving for even/odd byte accumulation
 *   - LUT: dim/4 subquantizers x 16 uint8 entries
 *
 * Query LUT:
 *   - 16-entry uint8 table per subquantizer, precomputing
 *     partial inner products for all 16 sign combinations
 *   - Quantized from float with linear scale + bias
 *   - De-quantized after accumulation: ip = accum * delta + bias
 */

#ifndef MKT_FASTSCAN_H
#define MKT_FASTSCAN_H

#include <stdint.h>

#include "mkt_types.h"

/* Minimum LUT range to avoid division by near-zero */
#define MKT_FASTSCAN_MIN_RANGE 1e-10f

/* Dimensions per subquantizer */
#define MKT_FASTSCAN_SQ_DIM 4

/* Vectors per VPSHUFB batch */
#define MKT_FASTSCAN_GROUP 32

/* Number of subquantizers for a given dimension */
#define MKT_FASTSCAN_NSQ(dim) \
	(((dim) + MKT_FASTSCAN_SQ_DIM - 1) / MKT_FASTSCAN_SQ_DIM)

/* Number of subquantizer pairs (two nibbles per byte) */
#define MKT_FASTSCAN_NSQ_PAIRS(dim) ((MKT_FASTSCAN_NSQ(dim) + 1) / 2)

/* Bytes per 32-vector group in packed fastscan layout.
 * Each column (8 dims = 2 sq) uses 32 bytes. */
#define MKT_FASTSCAN_GROUP_BYTES(dim) \
	((uint32_t)(((dim) + 7) / 8) * MKT_FASTSCAN_GROUP)

/* Bytes for the query lookup table.
 * nsq_pairs * 2 to include phantom sq when nsq is odd. */
#define MKT_FASTSCAN_LUT_BYTES(dim) \
	((uint32_t)MKT_FASTSCAN_NSQ_PAIRS(dim) * 2 * 16)

/*
 * Build a uint8 lookup table per subquantizer from the
 * transformed query vector.
 *
 * lut_out:    output buffer, MKT_FASTSCAN_LUT_BYTES(dim) bytes
 * delta_out:  quantization step size for de-quantization
 * bias_out:   LUT min value (sum_vl = vl * nsq)
 */
void mkt_fastscan_build_lut(
		const float *transformed,
		Dimension	 dim,
		uint8_t		*lut_out,
		float		*delta_out,
		float		*bias_out);

/*
 * Repack 1-bit RaBitQ codes into fastscan layout with kPerm0
 * vector interleaving for even/odd byte accumulation.
 *
 * Input:  bits_1bit[count * packed_bytes]
 * Output: codes_out[ngroups * group_bytes]
 *
 * Returns the number of 32-vector groups.
 */
uint32_t mkt_fastscan_pack_codes(
		const uint8_t *bits_1bit,
		uint32_t	   count,
		Dimension	   dim,
		uint8_t		  *codes_out);

/*
 * Required output buffer size for packed fastscan codes.
 */
static inline uint32_t
mkt_fastscan_codes_size(uint32_t count, Dimension dim)
{
	uint32_t ngroups = (count + MKT_FASTSCAN_GROUP - 1) / MKT_FASTSCAN_GROUP;
	uint32_t group_bytes = MKT_FASTSCAN_GROUP_BYTES(dim);
	return ngroups * group_bytes;
}

/*
 * VPSHUFB accumulate kernel -- process one 32-vector group.
 *
 * Uses even/odd byte accumulation: 4 uint16 accumulators
 * track vectors {0-7, 8-15, 16-23, 24-31} via kPerm0
 * interleaving. No explicit uint8->uint16 widening in the
 * inner loop.
 *
 * codes:  packed codes for one group
 * lut:    uint8 lookup table
 * accum:  output, 32 uint16 accumulated distances
 * dim:    vector dimension (determines loop count)
 */
void mkt_fastscan_accumulate(
		const uint8_t *codes,
		const uint8_t *lut,
		uint16_t	  *accum,
		Dimension	   dim);

/* ----------------------------------------------------------------
 * High-accuracy mode (uint16 LUT → int32 accumulators)
 *
 * Splits each uint16 LUT entry into lo/hi byte tables and does
 * two VPSHUFB passes per code block. ~2× kernel cost but
 * eliminates uint8 quantization error → matches float recall.
 *
 * LUT layout: [lo_64B, hi_64B] interleaved per 4 subquantizers.
 * Total LUT size: 2 × nsq × 16 bytes.
 * ---------------------------------------------------------------- */

/* LUT bytes for high-accuracy mode (2× the uint8 version) */
#define MKT_FASTSCAN_LUT_HACC_BYTES(dim) \
	((uint32_t)MKT_FASTSCAN_NSQ_PAIRS(dim) * 2 * 16 * 2)

/*
 * Build a uint16-precision LUT, split into lo/hi byte tables.
 * Same interface as build_lut but lut_out must be
 * MKT_FASTSCAN_LUT_HACC_BYTES(dim) bytes.
 */
void mkt_fastscan_build_lut_hacc(
		const float *transformed,
		Dimension	 dim,
		uint8_t		*lut_out,
		float		*delta_out,
		float		*bias_out);

/*
 * High-accuracy accumulate: two VPSHUFB passes per code block.
 * Output is 32 int32 values (not uint16).
 */
void mkt_fastscan_accumulate_hacc(
		const uint8_t *codes,
		const uint8_t *lut,
		int32_t		  *accum,
		Dimension	   dim);

/*
 * Batch distance computation using fastscan.
 */
struct RaBitQQueryState;

void mkt_fastscan_distance_batch(
		const struct RaBitQQueryState *qstate,
		const float					  *f_add,
		const float					  *f_rescale,
		const uint8_t				  *codes,
		uint32_t					   ngroups,
		uint32_t					   count,
		Dimension					   dim,
		float						  *distances,
		uint8_t						  *lut_buf,
		uint16_t					  *accum_buf);

void		mkt_fastscan_init_simd(void);
const char *mkt_fastscan_impl_name(void);

/* SIMD implementations */
#ifdef MKT_SIMD_FULL
#if defined(__x86_64__) || defined(_M_X64)

void mkt_fastscan_accumulate_avx2(
		const uint8_t *codes,
		const uint8_t *lut,
		uint16_t	  *accum,
		Dimension	   dim);

void mkt_fastscan_accumulate_hacc_avx2(
		const uint8_t *codes,
		const uint8_t *lut,
		int32_t		  *accum,
		Dimension	   dim);

void mkt_fastscan_accumulate_avx512(
		const uint8_t *codes,
		const uint8_t *lut,
		uint16_t	  *accum,
		Dimension	   dim);

void mkt_fastscan_build_lut_avx512(
		const float *transformed,
		Dimension	 dim,
		uint8_t		*lut_out,
		float		*delta_out,
		float		*bias_out);

void mkt_fastscan_accumulate_hacc_avx512(
		const uint8_t *codes,
		const uint8_t *lut,
		int32_t		  *accum,
		Dimension	   dim);

void mkt_fastscan_build_lut_hacc_avx512(
		const float *transformed,
		Dimension	 dim,
		uint8_t		*lut_out,
		float		*delta_out,
		float		*bias_out);

#endif
#endif

#endif /* MKT_FASTSCAN_H */
