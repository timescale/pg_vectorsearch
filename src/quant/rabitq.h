/*
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

#ifndef MKT_RABITQ_H
#define MKT_RABITQ_H

#include <stdint.h>

#include "mkt_types.h"

/*
 * Error constant from RaBitQ paper (empirically tuned by authors).
 * Controls the tightness of error bounds.
 */
#define MKT_RABITQ_EPSILON 1.9f

/*
 * RaBitQVector - Quantized vector (PostgreSQL varlena-compatible)
 *
 * Stores D bits packed into ceil(D/8) bytes, plus two float factors
 * for distance estimation. The data portion (f_add, f_rescale, bits[])
 * is layout-compatible with RaBitQData for zero-copy access via
 * MKT_RABITQ_DATA().
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
#define MKT_RABITQ_VECTOR_SIZE(dim) \
	(offsetof(RaBitQVector, bits) + (((dim) + 7) / 8))

/* Number of bytes needed to store dim bits */
#define MKT_RABITQ_BYTES(dim) (((dim) + 7) / 8)

/*
 * RaBitQData - Compact quantized vector (primary encoding form)
 *
 * Stores only f_add and f_rescale. f_error is derived at query time:
 *   f_error = C_error * sqrt(f_rescale² - f_add)
 * where C_error = 2 * MKT_RABITQ_EPSILON / sqrt(dim - 1).
 *
 * Total size: 8 bytes header + ceil(dim/8) bytes data
 */
typedef struct RaBitQData
{
	float	f_add;	   /* ||v-c||² - additive distance factor */
	float	f_rescale; /* dp_multiplier - scaling factor */
	uint8_t bits[];	   /* D/8 bytes, LSB-first bit packing */
} RaBitQData;

/* Calculate size of compact quantized vector */
#define MKT_RABITQ_DATA_SIZE(dim) \
	(offsetof(RaBitQData, bits) + MKT_RABITQ_BYTES(dim))

/* Access compact data portion of a presentation vector (zero-copy cast) */
#define MKT_RABITQ_DATA(v) ((RaBitQData *)&(v)->f_add)

/*
 * RaBitQBatch - Batch of encoded vectors in SoA layout
 *
 * Used for batch encoding where separate arrays for each field enable
 * efficient scatter into posting page SoA regions.
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
	float	 *P;   /* Random orthogonal matrix (dim x dim), row-major */
	Dimension dim; /* Vector dimension */
	uint32_t  packed_bytes; /* ceil(dim / 8) */
	uint64_t  seed;			/* Seed for reproducibility */
} RaBitQParams;

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
 */
typedef struct RaBitQQueryState
{
	float	 *transformed;	   /* P^T * (query - centroid) */
	float	  g_add;		   /* ||query - centroid||² */
	float	  g_error;		   /* sqrt(g_add) for error bound */
	float	  sum_transformed; /* sum(transformed) for distance formula */
	float	  inv_sqrt_d;	   /* 1 / sqrt(dim) */
	float	  c_error;		   /* 2*ε/√(d-1), for deriving f_error */
	Dimension dim;
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
RaBitQParams *mkt_rabitq_create(Dimension dim, uint64_t seed);

/*
 * Initialize RaBitQ parameters in pre-allocated memory.
 *
 * Same as mkt_rabitq_create but uses caller-provided struct.
 * The P matrix is still heap-allocated internally.
 *
 * Returns 0 on success, -1 on failure.
 */
int mkt_rabitq_init(RaBitQParams *params, Dimension dim, uint64_t seed);

/*
 * Free heap-allocated RaBitQ parameters.
 */
void mkt_rabitq_destroy(RaBitQParams *params);

/*
 * Free internal resources without freeing the struct itself.
 * Use when struct is stack-allocated.
 */
void mkt_rabitq_cleanup(RaBitQParams *params);

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
RaBitQData *mkt_rabitq_encode(
		const RaBitQParams *params, VectorRef input, VectorRef centroid);

/*
 * Encode a vector into pre-allocated output buffer.
 *
 * Output buffer must be at least MKT_RABITQ_DATA_SIZE(dim) bytes.
 *
 * Returns 0 on success, -1 on failure.
 */
int mkt_rabitq_encode_into(
		const RaBitQParams *params,
		VectorRef			input,
		VectorRef			centroid,
		RaBitQData		   *output);

/*
 * Batch encode multiple vectors into SoA output arrays.
 *
 * More efficient than calling mkt_rabitq_encode_into() repeatedly because:
 * 1. Matrix P is loaded into cache once and reused for all vectors
 * 2. Centroid rotation is computed once and reused
 * 3. Batched matrix-vector multiplication enables better SIMD utilization
 *
 * Memory layout:
 *   vectors:   count vectors, each dim floats, contiguous
 *   f_add:     count floats (output)
 *   f_rescale: count floats (output)
 *   bits:      count * packed_bytes bytes (output)
 *
 * Returns 0 on success, -1 on failure.
 */
int mkt_rabitq_encode_batch(
		const RaBitQParams *params,
		const float		   *vectors,
		VectorRef			centroid,
		float			   *f_add,
		float			   *f_rescale,
		uint8_t			   *bits,
		uint16_t			count);

/*
 * Batch encode with heap-allocated RaBitQBatch output.
 *
 * Convenience wrapper that allocates a RaBitQBatch and calls
 * mkt_rabitq_encode_batch() with the batch's arrays.
 *
 * Returns NULL on failure.
 */
RaBitQBatch *mkt_rabitq_encode_batch_alloc(
		const RaBitQParams *params,
		const float		   *vectors,
		VectorRef			centroid,
		uint16_t			count);

/*
 * Create an empty RaBitQBatch with allocated arrays.
 *
 * Returns NULL on allocation failure.
 */
RaBitQBatch *mkt_rabitq_batch_create(uint16_t count, Dimension dim);

/*
 * Free a RaBitQBatch and its arrays.
 */
void mkt_rabitq_batch_destroy(RaBitQBatch *batch);

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
RaBitQQueryState *mkt_rabitq_prepare_query(
		const RaBitQParams *params, VectorRef query, VectorRef centroid);

/*
 * Free query state.
 */
void mkt_rabitq_free_query(RaBitQQueryState *state);

/*
 * Distance computation - estimate L2 distance from quantized codes
 */

/*
 * Compute estimated L2 squared distance from compact data.
 *
 * Uses the precomputed query state and compact vector factors:
 *   est_dist = g_add + f_add - 2 * f_rescale * final_dot
 */
Distance mkt_rabitq_distance(
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
void mkt_rabitq_distance_with_bound(
		const RaBitQQueryState *query_state,
		const RaBitQData	   *data,
		Dimension				dim,
		Distance			   *est_dist,
		Distance			   *lower_bound);

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
int mkt_rabitq_init_simd(void);

/*
 * Get name of the active SIMD implementation.
 * Returns one of: "avx512", "avx2", "neon", "compiler"
 */
const char *mkt_rabitq_impl_name(void);

/*
 * Force re-initialization of SIMD dispatch (for testing).
 */
void mkt_rabitq_force_reinit(void);

#endif /* MKT_RABITQ_H */
