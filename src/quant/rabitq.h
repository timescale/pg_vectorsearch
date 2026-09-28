/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * rabitq.h - RaBitQ (Randomized Binary Quantization) for ANN search
 *
 * RaBitQ compresses D-dimensional vectors to D bits (32x compression) while
 * maintaining 95-99% recall through theoretical error bounds. Key insight:
 * vertices of a hypercube (±1/√D per coordinate) are evenly spread on the
 * unit hypersphere.
 *
 * Algorithm:
 * 1. Transform input via random orthogonal matrix P (ensures isotropic
 *    distribution)
 * 2. Compute residual: r = P^T * (data - centroid)
 * 3. Extract sign bits: bits[i] = (r[i] > 0) ? 1 : 0
 * 4. Compute factors for distance estimation: f_add, f_rescale
 * 5. At query time: est_dist = f_add + f_rescale * inner_product(q', bits)
 * 6. Lower bound: lower_bound = est_dist - f_error * g_error
 *
 * Reference: RaBitQ-Library (https://github.com/nmslib/RaBitQ-Library)
 */

#ifndef VS_RABITQ_H
#define VS_RABITQ_H

#include <stdbool.h>
#include <stdint.h>

#include "core/types.h"
#include "quant/fast_rotate.h"

/*
 * Error constant from RaBitQ paper (empirically tuned by authors).
 * Controls the tightness of error bounds.
 */
#define VS_RABITQ_EPSILON 1.9f

/*
 * Rotation seed for every index build. The rotation matrix is derived from
 * the seed at decode time too, so every site that creates build-time RaBitQ
 * params MUST use this seed: an index encoded under one seed and read under
 * another decodes garbage with no error.
 */
#define VS_RABITQ_BUILD_SEED UINT64_C(42)

/*
 * RaBitQVector - Quantized vector (PostgreSQL varlena-compatible)
 *
 * Stores D bits packed into ceil(D/8) bytes, plus two float factors
 * for distance estimation. The data portion (f_add, f_rescale, bits[])
 * is layout-compatible with RaBitQData for zero-copy access via
 * VS_RABITQ_DATA().
 *
 * This struct is reserved for the future PostgreSQL SQL type. Encoding
 * functions produce RaBitQData (compact form) instead.
 *
 * Total size: 16 bytes header + ceil(dim/8) bytes data
 */
typedef struct RaBitQVector
{
	int32_t vl_len_;   /* varlena header (for PG compatibility) */
	int16_t dim;	   /* dimensions = number of bits */
	int16_t flags;	   /* reserved for future use */
	float	f_add;	   /* additive factor for distance estimation */
	float	f_rescale; /* scaling factor for distance estimation */
	uint8_t bits[]; /* D/8 bytes, LSB-first bit packing (FAISS-compatible) */
} RaBitQVector;

/* Calculate size of presentation vector structure (for future PG type) */
#define VS_RABITQ_VECTOR_SIZE(dim) \
	(offsetof(RaBitQVector, bits) + (((dim) + 7) / 8))

/* Number of bytes needed to store dim bits */
#define VS_RABITQ_BYTES(dim) (((dim) + 7) / 8)

/*
 * RaBitQData - Compact quantized vector (primary encoding form)
 *
 * Stores only f_add and f_rescale. f_error is derived at query time:
 *   f_error = C_error * sqrt(f_rescale² - f_add)
 * where C_error = 2 * VS_RABITQ_EPSILON / sqrt(dim - 1).
 *
 * Total size: 8 bytes header + ceil(dim/8) bytes data
 */
typedef struct RaBitQData
{
	float	f_add;	   /* ||v-c||² - additive distance factor */
	float	f_rescale; /* dp_multiplier - scaling factor */
	uint8_t bits[];	   /* D/8 bytes, LSB-first bit packing */
} RaBitQData;

/* Calculate size of compact quantized vector (4-byte aligned for
 * safe struct access when stored in posting pages) */
#define VS_RABITQ_DATA_SIZE(dim) \
	(((offsetof(RaBitQData, bits) + VS_RABITQ_BYTES(dim)) + 3) & ~3u)

/* Access compact data portion of a presentation vector (zero-copy cast) */
#define VS_RABITQ_DATA(v) ((RaBitQData *)&(v)->f_add)

/*
 * RaBitQBatch - Batch of encoded vectors in separate arrays
 *
 * Used for batch encoding where separate arrays for each field enable
 * efficient SIMD processing and bulk page insertion.
 */
typedef struct RaBitQBatch
{
	uint16_t count;		   /* number of encoded vectors */
	uint16_t packed_bytes; /* ceil(dim/8) per vector */
	float	*f_add;		   /* [count] */
	float	*f_rescale;	   /* [count] */
	uint8_t *bits;		   /* [count * packed_bytes] */
} RaBitQBatch;

/*
 * RaBitQParams - Quantizer parameters (shared per index)
 *
 * Contains the random orthogonal matrix and derived values. Generated
 * once during index creation and shared across all vectors in the index.
 * The matrix P ensures isotropic distribution of residuals, which is
 * essential for RaBitQ's error bounds.
 */
typedef struct RaBitQParams
{
	Dimension dim;			/* Vector dimension */
	uint32_t  packed_bytes; /* ceil(dim / 8) */
	uint64_t  seed;			/* Seed for reproducibility */
	/*
	 * Rotation. When use_fast_rotate is set (dim is a supported N*K
	 * shape) the orthonormal rotation P^T*x is computed by the O(d log d)
	 * Randomized Hadamard Transform in `fr`, and the dense matrix P is
	 * neither built nor stored (the trailing P[] is allocated only for
	 * unsupported dims, which fall back to the dense O(d^2) multiply).
	 */
	bool			   use_fast_rotate;
	VsFastRotateParams fr;
	float			   P[FLEXIBLE_ARRAY_MEMBER]; /* dense P^T (dense path) */
} RaBitQParams;

/*
 * Dense layout size: header + the dim*dim float matrix. Cast to a fixed
 * 64-bit type before the multiply so the product is computed in 64 bits on
 * every platform (plain int32 overflows past dim ~46340, and size_t is still
 * 32-bit on ILP32). Used when an explicit dense matrix is stored
 * (vs_rabitq_create_from_matrix) or when the fast rotation is unavailable.
 */
#define VS_RABITQ_PARAMS_DENSE_SIZE(dim) \
	(offsetof(RaBitQParams, P) + (uint64_t)(dim) * (dim) * sizeof(float))

/*
 * Allocation size for seed-derived params. When the fast (Hadamard) rotation
 * supports the dim, the dense matrix P is not built or stored, so only the
 * header (which includes the small `fr` params) is needed -- saving the
 * O(dim^2) per-backend matrix and its O(dim^3) build. Otherwise the full
 * dense layout is allocated.
 */
static inline uint64_t
vs_rabitq_params_size(Dimension dim)
{
	return vs_fast_rotate_supported(dim) ? (uint64_t)offsetof(RaBitQParams, P)
										 : VS_RABITQ_PARAMS_DENSE_SIZE(dim);
}

#define VS_RABITQ_PARAMS_SIZE(dim) vs_rabitq_params_size(dim)

/*
 * RaBitQScratch - Pre-allocated scratch buffers for encoding
 *
 * Avoids per-vector allocation in vs_rabitq_encode_into. Create once
 * per builder/thread, reuse across all encode calls.
 */
typedef struct RaBitQScratch
{
	float *residual;	/* [dim], 64-byte aligned */
	float *transformed; /* [dim], 64-byte aligned */
	float *xu_cb;		/* [dim], 64-byte aligned */
} RaBitQScratch;

void vs_rabitq_scratch_init(RaBitQScratch *scratch, Dimension dim);
void vs_rabitq_scratch_cleanup(RaBitQScratch *scratch);

/*
 * RaBitQQueryState - Query state (amortizes work across vectors)
 *
 * Precomputes query-specific values that are reused when comparing against
 * multiple quantized vectors. The transformation and factors are computed
 * once per query, then used for all distance calculations.
 *
 * Distance formula (FAISS-style for better accuracy):
 *   est_dist = g_add + f_add - 2 * f_rescale * final_dot
 * where:
 *   final_dot = (2 * binary_ip - sum_transformed) * inv_sqrt_d
 *   binary_ip = sum of transformed[i] where bit[i] = 1
 *
 * The distance_fn and distance_with_bound_fn pointers are set at
 * query preparation time based on the selected VsDistanceMode,
 * enabling zero-branch dispatch in the hot loop.
 */

/* Forward declaration for function pointer types */
struct RaBitQQueryState;

typedef Distance (*RaBitQDistanceFn)(
		const struct RaBitQQueryState *qstate,
		const RaBitQData			  *data,
		Dimension					   dim);
typedef void (*RaBitQDistanceWithBoundFn)(
		const struct RaBitQQueryState *qstate,
		const RaBitQData			  *data,
		Dimension					   dim,
		Distance					  *est_dist,
		Distance					  *lower_bound);

typedef struct RaBitQQueryState
{
	float	 *transformed;		/* P^T * (query - centroid) */
	uint8_t	 *query_bits;		/* sign(transformed), packed bits */
	float	  g_add;			/* ||query - centroid||² */
	float	  g_error;			/* sqrt(g_add) for error bound */
	float	  g_scale;			/* mean(|transformed|) for symmetric */
	float	  sum_transformed;	/* sum(transformed) for distance formula */
	float	  inv_sqrt_d;		/* 1 / sqrt(dim) */
	float	  c_error;			/* 2*ε/√(d-1), for deriving f_error */
	float	  error_multiplier; /* 1.0 asymmetric, 3.0 symmetric */
	Dimension dim;

	/* Runtime dispatch (set by prepare_query_ex) */
	VsDistanceMode			  mode;
	RaBitQDistanceFn		  distance_fn;
	RaBitQDistanceWithBoundFn distance_with_bound_fn;
} RaBitQQueryState;

/*
 * Lifecycle - create/destroy quantizer parameters
 */

/*
 * Create new RaBitQ parameters with heap allocation.
 *
 * Generates a random orthogonal matrix of size dim x dim using the given
 * seed. The matrix is created via QR decomposition of a random Gaussian
 * matrix.
 *
 * Returns NULL on allocation failure.
 */
/*
 * Derive the per-vector error factor f_error from the encoded f_add /
 * f_rescale (see the RaBitQData comment for the formula). Shared by the
 * posting and centroid build paths so the formula lives in one place.
 */
float vs_rabitq_derive_f_error(float f_add, float f_rescale, Dimension dim);

RaBitQParams *vs_rabitq_create(Dimension dim, uint64_t seed);

/*
 * Create RaBitQ parameters from an existing rotation matrix.
 * Copies the matrix into the new allocation.
 */
RaBitQParams *
vs_rabitq_create_from_matrix(Dimension dim, uint64_t seed, const float *P);

/*
 * Initialize RaBitQ parameters in pre-allocated memory.
 *
 * Same as vs_rabitq_create but uses caller-provided buffer.
 * Buffer must be at least VS_RABITQ_PARAMS_SIZE(dim) bytes.
 * Generates the rotation matrix P in-place.
 *
 * Returns 0 on success, -1 on failure.
 */
int vs_rabitq_init(RaBitQParams *params, Dimension dim, uint64_t seed);

/*
 * Free heap-allocated RaBitQ parameters.
 */
void vs_rabitq_destroy(RaBitQParams *params);

/*
 * Cleanup internal resources (no-op since P is now inline).
 * Kept for API compatibility with stack-allocated usage.
 */
void vs_rabitq_cleanup(RaBitQParams *params);

/*
 * Encoding - convert full-precision vectors to binary codes
 */

/*
 * Encode a vector to compact RaBitQ format with heap allocation.
 *
 * Computes residual from centroid, transforms through P^T, extracts
 * sign bits, and computes estimation factors.
 *
 * Returns NULL on failure.
 */
RaBitQData *vs_rabitq_encode(
		const RaBitQParams *params, Vec32Ref input, Vec32Ref centroid);

/*
 * Encode a vector into pre-allocated output buffer.
 *
 * Output buffer must be at least VS_RABITQ_DATA_SIZE(dim) bytes.
 *
 * Returns 0 on success, -1 on failure.
 */
int vs_rabitq_encode_into(
		const RaBitQParams *params,
		Vec32Ref			input,
		Vec32Ref			centroid,
		RaBitQData		   *output);

/*
 * Encode with pre-allocated scratch buffers (zero per-call allocation).
 */
int vs_rabitq_encode_into_ex(
		const RaBitQParams *params,
		Vec32Ref			input,
		Vec32Ref			centroid,
		RaBitQData		   *output,
		RaBitQScratch	   *scratch);

/*
 * Encode from an already-rotated residual: pt_residual = P^T * (input -
 * centroid). The caller supplies the rotated residual directly, so this skips
 * the residual subtraction and P^T multiply that encode_into_ex performs; it
 * runs only the sign-extract + factor math. Used by the runtime insert path,
 * which rotates the inserted vector once and subtracts the posting head's
 * stored pt_centroid (P^T is linear: P^T*(v-c) = P^T*v - P^T*c), avoiding any
 * dependency on the raw leaf centroid (unavailable for RABITQ/FASTSCAN
 * centroid formats). Only scratch->xu_cb is used.
 *
 * pt_residual must be [params->dim] floats. Returns 0 on success, -1 on
 * failure.
 */
int vs_rabitq_encode_from_pt(
		const RaBitQParams *params,
		const float		   *pt_residual,
		RaBitQData		   *output,
		RaBitQScratch	   *scratch);

/*
 * Batch encode multiple vectors into separate output arrays.
 *
 * More efficient than calling vs_rabitq_encode_into() repeatedly because:
 * 1. Matrix P is loaded into cache once and reused for all vectors
 * 2. Centroid rotation is computed once and reused
 * 3. Batched matrix-vector multiplication enables better SIMD utilization
 *
 * Memory layout:
 *   vectors:   count vectors, each dim elements, contiguous
 *   vec_type:  element type (VS_VEC_F32, VS_VEC_F16, VS_VEC_F16C)
 *   f_add:     count floats (output)
 *   f_rescale: count floats (output)
 *   bits:      count * packed_bytes bytes (output)
 *
 * For non-f32 input, vectors are converted to float32 at entry.
 * This is O(count × dim), negligible vs the O(count × dim²) rotation.
 *
 * Returns 0 on success, -1 on failure.
 */
int vs_rabitq_encode_batch(
		const RaBitQParams *params,
		const void		   *vectors,
		VecType				vec_type,
		Vec32Ref			centroid,
		float			   *f_add,
		float			   *f_rescale,
		uint8_t			   *bits,
		uint16_t			count);

/*
 * Batch encode with heap-allocated RaBitQBatch output.
 *
 * Convenience wrapper that allocates a RaBitQBatch and calls
 * vs_rabitq_encode_batch() with the batch's arrays.
 *
 * Returns NULL on failure.
 */
RaBitQBatch *vs_rabitq_encode_batch_alloc(
		const RaBitQParams *params,
		const void		   *vectors,
		VecType				vec_type,
		Vec32Ref			centroid,
		uint16_t			count);

/*
 * Create an empty RaBitQBatch with allocated arrays.
 *
 * Returns NULL on allocation failure.
 */
RaBitQBatch *vs_rabitq_batch_create(uint16_t count, Dimension dim);

/*
 * Free a RaBitQBatch and its arrays.
 */
void vs_rabitq_batch_destroy(RaBitQBatch *batch);

/*
 * Query preparation - precompute query-specific factors
 */

/*
 * Prepare query state for efficient distance computation.
 *
 * Transforms the query through P^T and precomputes factors that are
 * reused when comparing against multiple quantized vectors.
 *
 * Returns NULL on failure.
 */
RaBitQQueryState *vs_rabitq_prepare_query(
		const RaBitQParams *params, Vec32Ref query, Vec32Ref centroid);

/*
 * Free query state.
 */
void vs_rabitq_free_query(RaBitQQueryState *state);

/*
 * Prepare query state with explicit distance mode.
 *
 * Like vs_rabitq_prepare_query() but additionally sets function pointers
 * for the selected mode, enabling zero-branch dispatch via the inline
 * helpers below.
 *
 * Returns NULL on failure.
 */
RaBitQQueryState *vs_rabitq_prepare_query_ex(
		const RaBitQParams *params,
		Vec32Ref			query,
		Vec32Ref			centroid,
		VsDistanceMode		mode);

/*
 * Pre-rotation API — eliminates per-cluster matrix multiply
 *
 * Instead of calling vs_rabitq_prepare_query_ex() per cluster
 * (which does O(dim²) matrix multiply each time), precompute:
 *   - P^T * centroid at index build time (once per cluster)
 *   - P^T * query at query time (once per query)
 * Then per cluster: transformed = pt_query - pt_centroid (O(dim))
 *
 * This is valid because P^T is linear:
 *   P^T * (query - centroid) = P^T * query - P^T * centroid
 */

/*
 * Rotate a vector through P^T into pre-allocated output buffer.
 * Output must have space for dim floats, 64-byte aligned preferred.
 */
void vs_rabitq_rotate(
		const RaBitQParams *params, const float *input, float *output);

/*
 * Initialize a pre-allocated query state from already-rotated vectors.
 *
 * Computes transformed = pt_query - pt_centroid (vector subtraction),
 * then derives all scalar fields (g_add, g_error, query_bits, etc.).
 * No matrix multiply — O(dim) instead of O(dim²).
 *
 * state->transformed and state->query_bits must be pre-allocated by
 * the caller (dim floats and packed_bytes bytes respectively).
 *
 * Call vs_rabitq_init_query_constants() once at context creation to
 * set dim-dependent constants (inv_sqrt_d, c_error) that don't change
 * per cluster.
 */
void vs_rabitq_init_query_constants(RaBitQQueryState *state, Dimension dim);

void vs_rabitq_init_query_state(
		RaBitQQueryState *state,
		const float		 *pt_query,
		const float		 *pt_centroid,
		Dimension		  dim,
		VsDistanceMode	  mode);

/*
 * Dispatch helpers - call through function pointers set at prepare time
 */

static inline Distance
vs_rabitq_distance_dispatch(
		const RaBitQQueryState *qstate, const RaBitQData *data, Dimension dim)
{
	return qstate->distance_fn(qstate, data, dim);
}

static inline void
vs_rabitq_distance_dispatch_with_bound(
		const RaBitQQueryState *qstate,
		const RaBitQData	   *data,
		Dimension				dim,
		Distance			   *est_dist,
		Distance			   *lower_bound)
{
	qstate->distance_with_bound_fn(qstate, data, dim, est_dist, lower_bound);
}

/*
 * Distance computation - estimate L2 distance from quantized codes
 */

/*
 * Compute estimated L2 squared distance from compact data.
 *
 * Uses the precomputed query state and compact vector factors:
 *   est_dist = g_add + f_add - 2 * f_rescale * final_dot
 */
Distance vs_rabitq_distance(
		const RaBitQQueryState *query_state,
		const RaBitQData	   *data,
		Dimension				dim);

/*
 * Compute estimated distance with derived error bound from compact data.
 *
 * f_error is derived from f_add and f_rescale using c_error in query
 * state. Returns both estimated distance and lower bound guaranteed
 * to be <= true distance.
 */
void vs_rabitq_distance_with_bound(
		const RaBitQQueryState *query_state,
		const RaBitQData	   *data,
		Dimension				dim,
		Distance			   *est_dist,
		Distance			   *lower_bound);

/*
 * Batch distance on separate arrays
 *
 * Computes estimated L2 distances for 'count' vectors whose fields
 * are stored in separate contiguous arrays: f_add[], f_rescale[],
 * and bits[]. Used by benchmarks and the multi-candidate kernel.
 *
 * The scalar arithmetic on contiguous f_add[]/f_rescale[] arrays
 * auto-vectorizes with the compiler.
 */
void vs_rabitq_distance_batch(
		const RaBitQQueryState *qstate,
		const float			   *f_add,
		const float			   *f_rescale,
		const uint8_t		   *bits,
		uint32_t				count,
		Dimension				dim,
		Distance			   *distances);

/*
 * Internal SIMD dispatch (called automatically)
 */

/*
 * Initialize SIMD dispatch for RaBitQ operations.
 * Called automatically on first use, but can be called explicitly
 * for deterministic initialization timing.
 *
 * Returns 0 on success.
 */
int vs_rabitq_init_simd(void);

/*
 * Get name of the active SIMD implementation.
 * Returns one of: "avx512", "avx2", "neon", "compiler"
 */
const char *vs_rabitq_impl_name(void);

/*
 * Force re-initialization of SIMD dispatch (for testing).
 */
void vs_rabitq_force_reinit(void);

/*
 * Hamming distance - XOR + popcount between two bit vectors
 */

/*
 * Compute Hamming distance between two packed bit vectors.
 *
 * Returns the number of bit positions where a and b differ.
 */
uint32_t vs_rabitq_hamming_distance(
		const uint8_t *a, const uint8_t *b, uint32_t packed_bytes);

/*
 * Multi-candidate Hamming distance (vertical SIMD).
 *
 * Computes Hamming distances from query_bits to count data vectors.
 * Candidate i's bits start at data_bits + i * stride.
 */
void vs_rabitq_hamming_distance_multi(
		const uint8_t *query_bits,
		const uint8_t *data_bits,
		uint32_t	   stride,
		uint32_t	   packed_bytes,
		uint32_t	   count,
		uint32_t	  *results);

/*
 * Symmetric distance - both query and data are 1-bit quantized
 *
 * Uses Hamming distance (XOR + popcount) instead of asymmetric inner
 * product (mask + add). ~32x fewer iterations of the inner loop,
 * but with additional query quantization error.
 *
 * Formula:
 *   sym_dot   = dim - 2 * hamming(query_bits, data_bits)
 *   final_dot = sym_dot * inv_sqrt_d
 *   est_dist  = f_add + g_add - 2 * f_rescale * g_scale * final_dot
 */

Distance vs_rabitq_distance_symmetric(
		const RaBitQQueryState *qstate, const RaBitQData *data, Dimension dim);

void vs_rabitq_distance_symmetric_with_bound(
		const RaBitQQueryState *qstate,
		const RaBitQData	   *data,
		Dimension				dim,
		Distance			   *est_dist,
		Distance			   *lower_bound);

void vs_rabitq_distance_batch_symmetric(
		const RaBitQQueryState *qstate,
		const float			   *f_add,
		const float			   *f_rescale,
		const uint8_t		   *bits,
		uint32_t				count,
		Dimension				dim,
		Distance			   *distances);

/*
 * Multi-candidate inner product (vertical SIMD)
 *
 * Computes inner products for multiple candidates in a single pass over
 * transformed[]. Loads transformed[] once per dimension chunk and
 * processes N candidates simultaneously.
 *
 * Candidate i's bits start at bits + i * stride.
 */
typedef void (*InnerProductMultiFn)(
		const float	  *transformed,
		const uint8_t *bits,
		uint32_t	   stride,
		Dimension	   dim,
		uint32_t	   count,
		float		  *results);

void vs_rabitq_inner_product_multi(
		const float	  *transformed,
		const uint8_t *bits,
		uint32_t	   stride,
		Dimension	   dim,
		uint32_t	   count,
		float		  *results);

/*
 * Batch distance using multi-candidate inner product
 *
 * Same interface as vs_rabitq_distance_batch() but uses the
 * vertical SIMD inner product to process multiple candidates per
 * pass over transformed[].
 */
void vs_rabitq_distance_batch_multi(
		const RaBitQQueryState *qstate,
		const float			   *f_add,
		const float			   *f_rescale,
		const uint8_t		   *bits,
		uint32_t				count,
		Dimension				dim,
		Distance			   *distances);

/*
 * Batch distance with error bounds (multi-candidate inner product)
 *
 * Combines vs_rabitq_inner_product_multi with per-entry error bound
 * derivation. Bits are accessed via stride (not packed_bytes), allowing
 * direct use on interleaved page data where stride = data_size.
 *
 * scratch: caller-provided buffer of at least count floats, reusable
 * across calls to avoid per-call allocation.
 */
void vs_rabitq_distance_batch_multi_with_bound(
		const RaBitQQueryState *qstate,
		const float			   *f_add,
		const float			   *f_rescale,
		const uint8_t		   *bits,
		uint32_t				stride,
		uint32_t				count,
		Dimension				dim,
		Distance			   *distances,
		Distance			   *lower_bounds,
		float				   *scratch);

/*
 * Batch symmetric distance with error bounds
 *
 * Combines vs_rabitq_hamming_distance_multi with per-entry error
 * bound derivation. Like the asymmetric variant, bits are accessed
 * via stride for direct use on interleaved page data.
 *
 * scratch: caller-provided buffer of at least count uint32_t's,
 * reusable across calls to avoid per-call allocation.
 */
void vs_rabitq_distance_batch_symmetric_with_bound(
		const RaBitQQueryState *qstate,
		const float			   *f_add,
		const float			   *f_rescale,
		const uint8_t		   *bits,
		uint32_t				stride,
		uint32_t				count,
		Dimension				dim,
		Distance			   *distances,
		Distance			   *lower_bounds,
		uint32_t			   *scratch);

/*
 * Get name of the active Hamming SIMD implementation.
 * Returns one of: "avx512-vpopcntdq", "avx2", "compiler"
 */
const char *vs_rabitq_hamming_impl_name(void);

/*
 * Hand-optimized SIMD implementations (simd=full only)
 *
 * These are resolved via function pointers in vs_rabitq_init_simd().
 */
#ifdef VS_SIMD_FULL

#if defined(__x86_64__) || defined(_M_X64)
/* AVX-512 implementations */
float vs_rabitq_inner_product_avx512(
		const float *transformed, const uint8_t *bits, Dimension dim);
void vs_rabitq_extract_signs_avx512(
		const float *transformed, uint8_t *bits, Dimension dim);
void vs_rabitq_inner_product_multi_avx512(
		const float	  *transformed,
		const uint8_t *bits,
		uint32_t	   stride,
		Dimension	   dim,
		uint32_t	   count,
		float		  *results);

/* AVX-512 VPOPCNTDQ Hamming implementations */
uint32_t vs_rabitq_hamming_avx512(
		const uint8_t *a, const uint8_t *b, uint32_t packed_bytes);
void vs_rabitq_hamming_multi_avx512(
		const uint8_t *query_bits,
		const uint8_t *data_bits,
		uint32_t	   stride,
		uint32_t	   packed_bytes,
		uint32_t	   count,
		uint32_t	  *results);

/* AVX2 implementations */
float vs_rabitq_inner_product_avx2(
		const float *transformed, const uint8_t *bits, Dimension dim);
void vs_rabitq_extract_signs_avx2(
		const float *transformed, uint8_t *bits, Dimension dim);
void vs_rabitq_inner_product_multi_avx2(
		const float	  *transformed,
		const uint8_t *bits,
		uint32_t	   stride,
		Dimension	   dim,
		uint32_t	   count,
		float		  *results);

/* AVX2 lookup-table Hamming implementations */
uint32_t vs_rabitq_hamming_avx2(
		const uint8_t *a, const uint8_t *b, uint32_t packed_bytes);
void vs_rabitq_hamming_multi_avx2(
		const uint8_t *query_bits,
		const uint8_t *data_bits,
		uint32_t	   stride,
		uint32_t	   packed_bytes,
		uint32_t	   count,
		uint32_t	  *results);
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
/* NEON implementations */
float vs_rabitq_inner_product_neon(
		const float *transformed, const uint8_t *bits, Dimension dim);
void vs_rabitq_extract_signs_neon(
		const float *transformed, uint8_t *bits, Dimension dim);
void vs_rabitq_inner_product_multi_neon(
		const float	  *transformed,
		const uint8_t *bits,
		uint32_t	   stride,
		Dimension	   dim,
		uint32_t	   count,
		float		  *results);
#endif

#endif /* VS_SIMD_FULL */

#endif /* VS_RABITQ_H */
