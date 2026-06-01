/*
 * fastscan.c - VPSHUFB-based fast scan for RaBitQ
 *
 * Implements the fastscan approach from the RaBitQ Library:
 * - LUT construction with kPos bit-to-dimension mapping
 * - Code packing with kPerm0 vector interleaving
 * - Even/odd byte accumulation (no explicit widening)
 * - SIMD dispatch (AVX-512 > AVX2 > scalar)
 */

#include "mkt_config.h"

#include <limits.h>
#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

#include "algo/simd_utils.h"
#include "core/platform.h"
#include "quant/fastscan.h"
#include "quant/rabitq.h"

/* ----------------------------------------------------------------
 * kPos: maps 4-bit code to dimension index for LUT construction.
 *
 * For code j, kPos[j] gives the dimension index of the lowest
 * set bit. Used in the incremental LUT build:
 *   lut[j] = lut[j - LOWBIT(j)] + query[kPos[j]]
 * ---------------------------------------------------------------- */

/* LUT entry mapping (implicit in build_lut code):
 * bit 0 → s0 (dim 0), bit 1 → s1 (dim 1),
 * bit 2 → s2 (dim 2), bit 3 → s3 (dim 3).
 * Entry j = sum of s_i for each set bit i in j. */

/* ----------------------------------------------------------------
 * kPerm0: vector interleaving for even/odd byte accumulation.
 *
 * Maps logical vector index (0-15) to byte position within a
 * 16-byte block. Even positions get vectors 0-7, odd positions
 * get vectors 8-15. This enables uint16 accumulation without
 * explicit widening: even bytes accumulate one group, odd bytes
 * another.
 * ---------------------------------------------------------------- */

static const int kPerm0[16] = {
		0,
		8,
		1,
		9,
		2,
		10,
		3,
		11,
		4,
		12,
		5,
		13,
		6,
		14,
		7,
		15,
};

/* Lowest set bit: j & (-j) */
#define LOWBIT(j) ((j) & (-(j)))

/* ----------------------------------------------------------------
 * LUT construction
 * ---------------------------------------------------------------- */

static void
fastscan_build_lut_scalar(
		const float *transformed,
		Dimension	 dim,
		uint8_t		*lut_out,
		float		*delta_out,
		float		*bias_out)
{
	uint32_t nsq	   = MKT_FASTSCAN_NSQ(dim);
	uint32_t nsq_pairs = MKT_FASTSCAN_NSQ_PAIRS(dim);

	/* Compute global min/max analytically from transformed values.
	 * Min LUT entry per sq = min(0, sum of negative values).
	 * Max LUT entry per sq = max(0, sum of positive values).
	 * Global min = sum of all sq mins. Global max = sum of all sq maxes. */
	float global_min = 0.0f;
	float global_max = 0.0f;
	for (Dimension d = 0; d < dim; d++)
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

	/* Build uint8 LUT directly (no intermediate float buffer).
	 * Pre-scale the bias and query values so each entry is just
	 * additions + truncation to uint8. */
	float bias_scaled = -global_min * inv_delta + 0.5f;

	memset(lut_out, 0, nsq_pairs * 2 * 16);

	const float *q = transformed;
	for (uint32_t sq = 0; sq < nsq; sq++)
	{
		uint8_t *out = lut_out + sq * 16;

		/* Pre-scale the 4 values for this subquantizer */
		Dimension base = sq * 4;
		float	  s0   = (base + 0 < dim) ? q[0] * inv_delta : 0.0f;
		float	  s1   = (base + 1 < dim) ? q[1] * inv_delta : 0.0f;
		float	  s2   = (base + 2 < dim) ? q[2] * inv_delta : 0.0f;
		float	  s3   = (base + 3 < dim) ? q[3] * inv_delta : 0.0f;

		/* Build 16 entries via incremental sums.
		 * Uses kPos mapping: bit 0→dim 0, bit 1→dim 1, etc. */
		float p01 = s0 + s1;
		float p23 = s2 + s3;

		float f0  = bias_scaled;
		float f4  = f0 + s2;
		float f8  = f0 + s3;
		float f12 = f0 + p23;

#define Q(v)                    \
	((uint8_t)((int)(v) < 0 ? 0 \
							: ((int)(v) > UINT8_MAX ? UINT8_MAX : (int)(v))))
		out[0]	= Q(f0);
		out[1]	= Q(f0 + s0);
		out[2]	= Q(f0 + s1);
		out[3]	= Q(f0 + p01);
		out[4]	= Q(f4);
		out[5]	= Q(f4 + s0);
		out[6]	= Q(f4 + s1);
		out[7]	= Q(f4 + p01);
		out[8]	= Q(f8);
		out[9]	= Q(f8 + s0);
		out[10] = Q(f8 + s1);
		out[11] = Q(f8 + p01);
		out[12] = Q(f12);
		out[13] = Q(f12 + s0);
		out[14] = Q(f12 + s1);
		out[15] = Q(f12 + p01);
#undef Q
		q += 4;
	}
}

/* ----------------------------------------------------------------
 * High-accuracy LUT construction (uint16 → split lo/hi tables)
 * ---------------------------------------------------------------- */

static void
fastscan_build_lut_hacc_scalar(
		const float *transformed,
		Dimension	 dim,
		uint8_t		*lut_out,
		float		*delta_out,
		float		*bias_out)
{
	uint32_t nsq = MKT_FASTSCAN_NSQ(dim);

	/* Same min/max as uint8 version */
	float global_min = 0.0f;
	float global_max = 0.0f;
	for (Dimension d = 0; d < dim; d++)
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

	/* Build uint16 LUT entries, then split into lo/hi byte tables.
	 * Layout: for each group of 4 sq (matching 64B code blocks),
	 * [lo_table_64B, hi_table_64B]. */
	uint32_t lut_bytes = MKT_FASTSCAN_LUT_HACC_BYTES(dim);
	memset(lut_out, 0, lut_bytes);

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

		float entries[16] = {
				f0,
				f0 + s0,
				f0 + s1,
				f0 + p01,
				f4,
				f4 + s0,
				f4 + s1,
				f4 + p01,
				f8,
				f8 + s0,
				f8 + s1,
				f8 + p01,
				f12,
				f12 + s0,
				f12 + s1,
				f12 + p01,
		};

		/* Quantize to uint16 and split into lo/hi bytes.
		 * Group of 4 sq → 128B block: [lo_64B, hi_64B].
		 * Within each 64B: 4 × 16-byte tables. */
		uint32_t group4		  = sq / 4;
		uint32_t pos_in_group = sq % 4;
		uint8_t *lo_base	  = lut_out + group4 * 128 + pos_in_group * 16;
		uint8_t *hi_base	  = lo_base + 64;

		for (uint32_t j = 0; j < 16; j++)
		{
			float scaled = entries[j];
			int	  val	 = (int)scaled;
			if (val < 0)
				val = 0;
			if (val > UINT16_MAX)
				val = UINT16_MAX;
			lo_base[j] = (uint8_t)(val & 0xFF);
			hi_base[j] = (uint8_t)((val >> 8) & 0xFF);
		}

		q += 4;
	}
}

/* ----------------------------------------------------------------
 * High-accuracy scalar accumulate (reference)
 * ---------------------------------------------------------------- */

static void
fastscan_accumulate_hacc_scalar(
		const uint8_t *codes,
		const uint8_t *lut,
		int32_t		  *accum,
		Dimension	   dim)
{
	uint32_t code_length = MKT_FASTSCAN_GROUP_BYTES(dim);

	int32_t lo_accum[MKT_FASTSCAN_GROUP] = {0};
	int32_t hi_accum[MKT_FASTSCAN_GROUP] = {0};

	for (uint32_t i = 0; i < code_length; i += 16)
	{
		const uint8_t *c = codes + i;
		/* lo/hi tables are interleaved per group of 4 sq (64B each).
		 * Compute which 128B block and offset within it. */
		uint32_t	   sq_idx = i / 16;
		uint32_t	   grp4	  = sq_idx / 4;
		uint32_t	   pos	  = sq_idx % 4;
		const uint8_t *lo_tab = lut + grp4 * 128 + pos * 16;
		const uint8_t *hi_tab = lo_tab + 64;

		for (uint32_t j = 0; j < 16; j++)
		{
			uint8_t byte_val = c[j];
			uint8_t lo_nib	 = byte_val & 0x0F;
			uint8_t hi_nib	 = byte_val >> 4;

			lo_accum[kPerm0[j]] += lo_tab[lo_nib];
			lo_accum[kPerm0[j] + 16] += lo_tab[hi_nib];

			hi_accum[kPerm0[j]] += hi_tab[lo_nib];
			hi_accum[kPerm0[j] + 16] += hi_tab[hi_nib];
		}
	}

	for (uint32_t v = 0; v < MKT_FASTSCAN_GROUP; v++)
		accum[v] = lo_accum[v] + (hi_accum[v] << 8);
}

/* ----------------------------------------------------------------
 * Code packing with kPerm0 interleaving
 *
 * Each column (8 dims = 1 byte of 1-bit code) produces two
 * 16-byte blocks: lower nibble sq first, upper nibble sq second.
 * This matches the sequential LUT order (sq0, sq1, sq2, ...).
 *
 * Within each 16-byte block, vectors are interleaved with kPerm0:
 * byte j packs kPerm0[j] vector (lo nibble) and kPerm0[j]+16
 * vector (hi nibble). This enables the even/odd byte accumulation
 * trick in the SIMD kernel.
 *
 * Memory layout per group per column:
 *   bytes 0-15:  lower nibble sq (sq0, sq2, ...) with kPerm0
 *   bytes 16-31: upper nibble sq (sq1, sq3, ...) with kPerm0
 * ---------------------------------------------------------------- */

uint32_t
mkt_fastscan_pack_codes(
		const uint8_t *bits_1bit,
		uint32_t	   count,
		Dimension	   dim,
		uint8_t		  *codes_out)
{
	uint32_t packed_bytes = (dim + 7) / 8;
	uint32_t ngroups = (count + MKT_FASTSCAN_GROUP - 1) / MKT_FASTSCAN_GROUP;

	memset(codes_out, 0, (size_t)ngroups * MKT_FASTSCAN_GROUP_BYTES(dim));

	for (uint32_t g = 0; g < ngroups; g++)
	{
		uint32_t g_start = g * MKT_FASTSCAN_GROUP;

		for (uint32_t col = 0; col < packed_bytes; col++)
		{
			/* Collect one byte (8 dims) from each of 32 vectors */
			uint8_t raw[MKT_FASTSCAN_GROUP];
			for (uint32_t v = 0; v < MKT_FASTSCAN_GROUP; v++)
			{
				uint32_t vi = g_start + v;
				raw[v]		= (vi < count)
									? bits_1bit[(size_t)vi * packed_bytes + col]
									: 0;
			}

			/* Split into upper and lower nibbles */
			uint8_t upper[MKT_FASTSCAN_GROUP]; /* dims 4-7 = sq1 */
			uint8_t lower[MKT_FASTSCAN_GROUP]; /* dims 0-3 = sq0 */
			for (uint32_t v = 0; v < MKT_FASTSCAN_GROUP; v++)
			{
				upper[v] = raw[v] >> 4;
				lower[v] = raw[v] & 0x0F;
			}

			/* Pack with kPerm0 interleaving.
			 * Lower nibble sq first (to match sequential LUT). */
			uint8_t *out = codes_out +
						   (size_t)g * MKT_FASTSCAN_GROUP_BYTES(dim) +
						   (size_t)col * MKT_FASTSCAN_GROUP;

			for (uint32_t j = 0; j < 16; j++)
			{
				out[j]		= lower[kPerm0[j]] | (lower[kPerm0[j] + 16] << 4);
				out[j + 16] = upper[kPerm0[j]] | (upper[kPerm0[j] + 16] << 4);
			}
		}
	}

	return ngroups;
}

/*
 * Inverse of mkt_fastscan_pack_codes: reconstruct the per-vector 1-bit
 * RaBitQ codes from the packed fastscan layout. Used when merging a
 * fastscan posting page back into a builder (the parallel build folds
 * workers' trailing partial pages into the head). bits_out must hold
 * count * packed_bytes bytes.
 */
void
mkt_fastscan_unpack_codes(
		const uint8_t *codes, uint32_t count, Dimension dim, uint8_t *bits_out)
{
	uint32_t packed_bytes = (dim + 7) / 8;
	uint32_t ngroups = (count + MKT_FASTSCAN_GROUP - 1) / MKT_FASTSCAN_GROUP;

	for (uint32_t g = 0; g < ngroups; g++)
	{
		uint32_t g_start = g * MKT_FASTSCAN_GROUP;

		for (uint32_t col = 0; col < packed_bytes; col++)
		{
			const uint8_t *in = codes +
								(size_t)g * MKT_FASTSCAN_GROUP_BYTES(dim) +
								(size_t)col * MKT_FASTSCAN_GROUP;

			/* Undo the kPerm0 nibble interleaving (inverse of the pack). */
			uint8_t lower[MKT_FASTSCAN_GROUP];
			uint8_t upper[MKT_FASTSCAN_GROUP];
			for (uint32_t j = 0; j < 16; j++)
			{
				lower[kPerm0[j]]	  = in[j] & 0x0F;
				lower[kPerm0[j] + 16] = in[j] >> 4;
				upper[kPerm0[j]]	  = in[j + 16] & 0x0F;
				upper[kPerm0[j] + 16] = in[j + 16] >> 4;
			}

			for (uint32_t v = 0; v < MKT_FASTSCAN_GROUP; v++)
			{
				uint32_t vi = g_start + v;
				if (vi < count)
					bits_out[(size_t)vi * packed_bytes + col] =
							(uint8_t)((upper[v] << 4) | lower[v]);
			}
		}
	}
}

/* ----------------------------------------------------------------
 * Scalar accumulate kernel (reference implementation)
 *
 * Even/odd byte accumulation: add uint8 results as uint16,
 * track upper byte contamination, subtract at end.
 * ---------------------------------------------------------------- */

static void
fastscan_accumulate_scalar(
		const uint8_t *codes,
		const uint8_t *lut,
		uint16_t	  *accum,
		Dimension	   dim)
{
	uint32_t code_length = MKT_FASTSCAN_GROUP_BYTES(dim);

	memset(accum, 0, MKT_FASTSCAN_GROUP * sizeof(uint16_t));

	/* Process 64 bytes per iteration: 4 sq blocks of 16 bytes each.
	 * Each 16-byte block has codes for one sq, with kPerm0
	 * interleaving: byte j packs kPerm0[j] (lo) and kPerm0[j]+16 (hi).
	 * The corresponding LUT 16-byte block is the table for that sq. */
	for (uint32_t i = 0; i < code_length; i += 16)
	{
		const uint8_t *c   = codes + i;
		const uint8_t *tab = lut + i;

		for (uint32_t j = 0; j < 16; j++)
		{
			uint8_t byte_val = c[j];
			uint8_t lo_nib	 = byte_val & 0x0F;
			uint8_t hi_nib	 = byte_val >> 4;

			/* VPSHUFB per 128-bit lane: lookup in this sq's table */
			accum[kPerm0[j]] += tab[lo_nib];
			accum[kPerm0[j] + 16] += tab[hi_nib];
		}
	}
}

/* ----------------------------------------------------------------
 * SIMD dispatch
 * ---------------------------------------------------------------- */

typedef void (*MktFastscanAccumulateFn)(
		const uint8_t *, const uint8_t *, uint16_t *, Dimension);

typedef void (*MktFastscanAccumulateHaccFn)(
		const uint8_t *, const uint8_t *, int32_t *, Dimension);

typedef void (*MktFastscanBuildLutFn)(
		const float *, Dimension, uint8_t *, float *, float *);

static MktFastscanAccumulateFn	   g_fastscan_accumulate_fn		 = NULL;
static MktFastscanAccumulateHaccFn g_fastscan_accumulate_hacc_fn = NULL;
static MktFastscanBuildLutFn	   g_fastscan_build_lut_fn		 = NULL;
static MktFastscanBuildLutFn	   g_fastscan_build_lut_hacc_fn	 = NULL;
static atomic_bool				   g_fastscan_initialized		 = false;

void
mkt_fastscan_reset_simd(void)
{
	atomic_store(&g_fastscan_initialized, false);
}

void
mkt_fastscan_init_simd(void)
{
	if (atomic_load(&g_fastscan_initialized))
		return;

#ifdef MKT_SIMD_FULL
#if defined(__x86_64__) || defined(_M_X64)
	SimdCapability caps = mkt_detect_simd();
	if (caps & SIMD_AVX512F)
	{
		g_fastscan_accumulate_fn	  = mkt_fastscan_accumulate_avx512;
		g_fastscan_accumulate_hacc_fn = mkt_fastscan_accumulate_hacc_avx512;
		g_fastscan_build_lut_fn		  = mkt_fastscan_build_lut_avx512;
		g_fastscan_build_lut_hacc_fn  = mkt_fastscan_build_lut_hacc_avx512;
		atomic_store(&g_fastscan_initialized, true);
		return;
	}
	if (caps & SIMD_AVX2)
	{
		g_fastscan_accumulate_fn	  = mkt_fastscan_accumulate_avx2;
		g_fastscan_accumulate_hacc_fn = mkt_fastscan_accumulate_hacc_avx2;
		g_fastscan_build_lut_fn		  = fastscan_build_lut_scalar;
		g_fastscan_build_lut_hacc_fn  = fastscan_build_lut_hacc_scalar;
		atomic_store(&g_fastscan_initialized, true);
		return;
	}
#elif defined(__aarch64__) || defined(_M_ARM64)
	SimdCapability caps = mkt_detect_simd();
#ifdef MKT_HAVE_SME2
	if (caps & SIMD_SME2)
	{
		/* SME2 streaming-mode kernels. Preferred over SVE2 when both
		 * are advertised, and the only ARM accelerated path on cores
		 * that expose SME2 without architectural SVE2 (Apple M4+). */
		g_fastscan_accumulate_fn	  = mkt_fastscan_accumulate_sme2;
		g_fastscan_accumulate_hacc_fn = mkt_fastscan_accumulate_hacc_sme2;
		g_fastscan_build_lut_fn		  = mkt_fastscan_build_lut_neon;
		g_fastscan_build_lut_hacc_fn  = mkt_fastscan_build_lut_hacc_neon;
		atomic_store(&g_fastscan_initialized, true);
		return;
	}
#endif
#ifdef MKT_HAVE_SVE2
	if (caps & SIMD_SVE2)
	{
		/* SVE2 accumulate kernels; reuse NEON build_lut (LUT layout is
		 * identical and the build kernel is not on the inner hot path). */
		g_fastscan_accumulate_fn	  = mkt_fastscan_accumulate_sve2;
		g_fastscan_accumulate_hacc_fn = mkt_fastscan_accumulate_hacc_sve2;
		g_fastscan_build_lut_fn		  = mkt_fastscan_build_lut_neon;
		g_fastscan_build_lut_hacc_fn  = mkt_fastscan_build_lut_hacc_neon;
		atomic_store(&g_fastscan_initialized, true);
		return;
	}
#endif
#ifdef MKT_HAVE_SVE
	if (caps & SIMD_SVE)
	{
		/* SVE (non-SVE2) accumulate kernels for cores like Graviton 3
		 * that expose SVE but not SVE2. Same algorithm as the SVE2
		 * path with the USRA fuse decomposed into add+shift. */
		g_fastscan_accumulate_fn	  = mkt_fastscan_accumulate_sve;
		g_fastscan_accumulate_hacc_fn = mkt_fastscan_accumulate_hacc_sve;
		g_fastscan_build_lut_fn		  = mkt_fastscan_build_lut_neon;
		g_fastscan_build_lut_hacc_fn  = mkt_fastscan_build_lut_hacc_neon;
		atomic_store(&g_fastscan_initialized, true);
		return;
	}
#endif
	if (caps & SIMD_NEON)
	{
		g_fastscan_accumulate_fn	  = mkt_fastscan_accumulate_neon;
		g_fastscan_accumulate_hacc_fn = mkt_fastscan_accumulate_hacc_neon;
		g_fastscan_build_lut_fn		  = mkt_fastscan_build_lut_neon;
		g_fastscan_build_lut_hacc_fn  = mkt_fastscan_build_lut_hacc_neon;
		atomic_store(&g_fastscan_initialized, true);
		return;
	}
#endif
#endif

	g_fastscan_accumulate_fn	  = fastscan_accumulate_scalar;
	g_fastscan_accumulate_hacc_fn = fastscan_accumulate_hacc_scalar;
	g_fastscan_build_lut_fn		  = fastscan_build_lut_scalar;
	g_fastscan_build_lut_hacc_fn  = fastscan_build_lut_hacc_scalar;
	atomic_store(&g_fastscan_initialized, true);
}

void
mkt_fastscan_build_lut(
		const float *transformed,
		Dimension	 dim,
		uint8_t		*lut_out,
		float		*delta_out,
		float		*bias_out)
{
	if (mkt_unlikely(!atomic_load(&g_fastscan_initialized)))
		mkt_fastscan_init_simd();
	g_fastscan_build_lut_fn(transformed, dim, lut_out, delta_out, bias_out);
}

void
mkt_fastscan_build_lut_hacc(
		const float *transformed,
		Dimension	 dim,
		uint8_t		*lut_out,
		float		*delta_out,
		float		*bias_out)
{
	if (mkt_unlikely(!atomic_load(&g_fastscan_initialized)))
		mkt_fastscan_init_simd();
	g_fastscan_build_lut_hacc_fn(
			transformed, dim, lut_out, delta_out, bias_out);
}

void
mkt_fastscan_accumulate(
		const uint8_t *codes,
		const uint8_t *lut,
		uint16_t	  *accum,
		Dimension	   dim)
{
	if (mkt_unlikely(!atomic_load(&g_fastscan_initialized)))
		mkt_fastscan_init_simd();
	g_fastscan_accumulate_fn(codes, lut, accum, dim);
}

void
mkt_fastscan_accumulate_hacc(
		const uint8_t *codes,
		const uint8_t *lut,
		int32_t		  *accum,
		Dimension	   dim)
{
	if (mkt_unlikely(!atomic_load(&g_fastscan_initialized)))
		mkt_fastscan_init_simd();
	g_fastscan_accumulate_hacc_fn(codes, lut, accum, dim);
}

const char *
mkt_fastscan_impl_name(void)
{
	if (!atomic_load(&g_fastscan_initialized))
		mkt_fastscan_init_simd();

#ifdef MKT_SIMD_FULL
#if defined(__x86_64__) || defined(_M_X64)
	if (g_fastscan_accumulate_fn == mkt_fastscan_accumulate_avx512)
		return "avx512";
	if (g_fastscan_accumulate_fn == mkt_fastscan_accumulate_avx2)
		return "avx2";
#elif defined(__aarch64__) || defined(_M_ARM64)
#ifdef MKT_HAVE_SME2
	if (g_fastscan_accumulate_fn == mkt_fastscan_accumulate_sme2)
		return "sme2";
#endif
#ifdef MKT_HAVE_SVE2
	if (g_fastscan_accumulate_fn == mkt_fastscan_accumulate_sve2)
		return "sve2";
#endif
#ifdef MKT_HAVE_SVE
	if (g_fastscan_accumulate_fn == mkt_fastscan_accumulate_sve)
		return "sve";
#endif
	if (g_fastscan_accumulate_fn == mkt_fastscan_accumulate_neon)
		return "neon";
#endif
#endif

	return "scalar";
}

/* ----------------------------------------------------------------
 * Batch distance computation
 * ---------------------------------------------------------------- */

void
mkt_fastscan_distance_batch(
		const RaBitQQueryState *qstate,
		const float			   *f_add,
		const float			   *f_rescale,
		const uint8_t		   *codes,
		uint32_t				ngroups,
		uint32_t				count,
		Dimension				dim,
		float				   *distances,
		uint8_t				   *lut_buf,
		uint16_t			   *accum_buf)
{
	if (qstate == NULL || codes == NULL || distances == NULL || count == 0)
		return;

	/* Build LUT from transformed query */
	float lut_delta, lut_bias;
	mkt_fastscan_build_lut(
			qstate->transformed, dim, lut_buf, &lut_delta, &lut_bias);

	uint32_t group_bytes = MKT_FASTSCAN_GROUP_BYTES(dim);
	float	 g_add		 = qstate->g_add;
	float	 sum_t		 = qstate->sum_transformed;
	float	 inv_sqrt_d	 = qstate->inv_sqrt_d;

	for (uint32_t g = 0; g < ngroups; g++)
	{
		const uint8_t *group_codes = codes + (size_t)g * group_bytes;
		uint32_t	   g_start	   = g * MKT_FASTSCAN_GROUP;
		uint32_t	   g_count	   = (g_start + MKT_FASTSCAN_GROUP <= count)
										   ? MKT_FASTSCAN_GROUP
										   : count - g_start;

		mkt_fastscan_accumulate(group_codes, lut_buf, accum_buf, dim);

		for (uint32_t v = 0; v < g_count; v++)
		{
			uint32_t vi = g_start + v;

			/* De-quantize: ip = accum * delta + nsq * vl */
			float binary_ip = (float)accum_buf[v] * lut_delta + lut_bias;

			float final_dot = (2.0f * binary_ip - sum_t) * inv_sqrt_d;
			distances[vi]	= f_add[vi] + g_add -
							2.0f * f_rescale[vi] * final_dot;
		}
	}
}
