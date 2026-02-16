/*
 * rabitq.c - RaBitQ (Randomized Binary Quantization) implementation
 *
 * Core implementation with SIMD dispatch for the binary inner product
 * operation. The hot path during search is computing:
 *
 *   sum = 0
 *   for each bit i:
 *     if bits[i] == 1:
 *       sum += transformed_query[i]
 *
 * This is optimized with AVX2/AVX-512/NEON intrinsics.
 */

#include "mkt_config.h"

#include <math.h>
#include <stdatomic.h>
#include <string.h>

#include "algo/simd_utils.h"
#include "algo/vecops.h"
#include "core/memory.h"
#include "core/platform.h"
#include "mkt_halfvec.h"
#include "mkt_vector.h"
#include "quant/matrix.h"
#include "quant/rabitq.h"

/*
 * Compiler-Vectorized Implementation
 *
 * Uses target_clones to generate multiple versions for different ISAs.
 * The dynamic linker selects the best version at load time.
 *
 * Structured for auto-vectorization: processes 8 floats per byte with
 * explicit mask expansion that compilers can optimize.
 */

MKT_TARGET_CLONES static float
rabitq_inner_product_compiler(
		const float *transformed, const uint8_t *bits, Dimension dim)
{
	float sum = 0.0f;

	/* Main loop: process 8 floats (1 byte) at a time.
	 * This structure helps auto-vectorization by:
	 * 1. Fixed iteration count inner loop (8 iterations)
	 * 2. Simple bit test with mask array lookup
	 * 3. Multiply instead of conditional for branchless code
	 */
	Dimension i = 0;
	for (; i + 8 <= dim; i += 8)
	{
		uint8_t byte = bits[i / 8];

		/* Unrolled: multiply by bit value (0 or 1) */
		sum += transformed[i + 0] * ((byte >> 0) & 1);
		sum += transformed[i + 1] * ((byte >> 1) & 1);
		sum += transformed[i + 2] * ((byte >> 2) & 1);
		sum += transformed[i + 3] * ((byte >> 3) & 1);
		sum += transformed[i + 4] * ((byte >> 4) & 1);
		sum += transformed[i + 5] * ((byte >> 5) & 1);
		sum += transformed[i + 6] * ((byte >> 6) & 1);
		sum += transformed[i + 7] * ((byte >> 7) & 1);
	}

	/* Handle tail elements */
	for (; i < dim; i++)
	{
		int byte_idx = i / 8;
		int bit_idx	 = i % 8;
		int bit		 = (bits[byte_idx] >> bit_idx) & 1;
		sum += transformed[i] * bit;
	}

	return sum;
}

/*
 * Compiler-vectorized multi-candidate inner product baseline.
 *
 * Simple loop calling the single-candidate function. This is the
 * baseline that hand-optimized vertical SIMD kernels must beat.
 */
MKT_TARGET_CLONES static void
rabitq_inner_product_multi_compiler(
		const float	  *transformed,
		const uint8_t *bits,
		uint32_t	   stride,
		Dimension	   dim,
		uint32_t	   count,
		float		  *results)
{
	for (uint32_t i = 0; i < count; i++)
		results[i] = rabitq_inner_product_compiler(
				transformed, bits + (size_t)i * stride, dim);
}

/*
 * Function pointer dispatch for inner product and sign extraction
 */
typedef float (*InnerProductFn)(const float *, const uint8_t *, Dimension);
typedef void (*ExtractSignsFn)(const float *, uint8_t *, Dimension);
static InnerProductFn	   g_inner_product_fn		= NULL;
static InnerProductMultiFn g_inner_product_multi_fn = NULL;
static ExtractSignsFn	   g_extract_signs_fn		= NULL;
static const char		  *g_impl_name				= NULL;
static _Atomic(bool)	   g_rabitq_initialized		= false;

/* Forward declaration for compiler-vectorized fallback */
MKT_TARGET_CLONES static void rabitq_extract_signs_compiler(
		const float *transformed, uint8_t *bits, Dimension dim);

void
mkt_rabitq_force_reinit(void)
{
	g_rabitq_initialized	 = false;
	g_inner_product_fn		 = NULL;
	g_inner_product_multi_fn = NULL;
	g_extract_signs_fn		 = NULL;
	g_impl_name				 = NULL;
}

int
mkt_rabitq_init_simd(void)
{
	if (g_rabitq_initialized)
		return 0;

#ifdef MKT_SIMD_FULL
	SimdCapability caps = mkt_detect_simd();

#if defined(__x86_64__) || defined(_M_X64)
	if (caps & SIMD_AVX512F)
	{
		g_inner_product_fn		 = mkt_rabitq_inner_product_avx512;
		g_inner_product_multi_fn = mkt_rabitq_inner_product_multi_avx512;
		g_extract_signs_fn		 = mkt_rabitq_extract_signs_avx512;
		g_impl_name				 = "avx512";
	}
	else if (caps & SIMD_AVX2)
	{
		g_inner_product_fn		 = mkt_rabitq_inner_product_avx2;
		g_inner_product_multi_fn = mkt_rabitq_inner_product_multi_avx2;
		g_extract_signs_fn		 = mkt_rabitq_extract_signs_avx2;
		g_impl_name				 = "avx2";
	}
	else
	{
		g_inner_product_fn		 = rabitq_inner_product_compiler;
		g_inner_product_multi_fn = rabitq_inner_product_multi_compiler;
		g_extract_signs_fn		 = rabitq_extract_signs_compiler;
		g_impl_name				 = "compiler";
	}
#elif defined(__aarch64__) || defined(_M_ARM64)
	if (caps & SIMD_NEON)
	{
		g_inner_product_fn		 = mkt_rabitq_inner_product_neon;
		g_inner_product_multi_fn = mkt_rabitq_inner_product_multi_neon;
		g_extract_signs_fn		 = mkt_rabitq_extract_signs_neon;
		g_impl_name				 = "neon";
	}
	else
	{
		g_inner_product_fn		 = rabitq_inner_product_compiler;
		g_inner_product_multi_fn = rabitq_inner_product_multi_compiler;
		g_extract_signs_fn		 = rabitq_extract_signs_compiler;
		g_impl_name				 = "compiler";
	}
#else
	g_inner_product_fn		 = rabitq_inner_product_compiler;
	g_inner_product_multi_fn = rabitq_inner_product_multi_compiler;
	g_extract_signs_fn		 = rabitq_extract_signs_compiler;
	g_impl_name				 = "compiler";
#endif

#else
	/* simd=compiler or simd=none mode */
	g_inner_product_fn		 = rabitq_inner_product_compiler;
	g_inner_product_multi_fn = rabitq_inner_product_multi_compiler;
	g_extract_signs_fn		 = rabitq_extract_signs_compiler;
	g_impl_name				 = "compiler";
#endif

	g_rabitq_initialized = true;
	return 0;
}

const char *
mkt_rabitq_impl_name(void)
{
	if (mkt_unlikely(!g_rabitq_initialized))
		mkt_rabitq_init_simd();
	return g_impl_name;
}

/*
 * Internal helper: compute inner product using dispatched function
 */
static inline float
rabitq_inner_product(
		const float *transformed, const uint8_t *bits, Dimension dim)
{
	if (mkt_unlikely(!g_rabitq_initialized))
		mkt_rabitq_init_simd();
	return g_inner_product_fn(transformed, bits, dim);
}

/*
 * Compiler-Vectorized Sign Extraction
 *
 * Extracts sign bits from transformed floats. Structured for
 * auto-vectorization by processing 8 floats at a time with explicit comparison
 * and bit packing.
 */
MKT_TARGET_CLONES static void
rabitq_extract_signs_compiler(
		const float *transformed, uint8_t *bits, Dimension dim)
{
	Dimension packed_bytes = (dim + 7) / 8;
	memset(bits, 0, packed_bytes);

	/* Main loop: process 8 floats -> 1 byte at a time */
	Dimension i = 0;
	for (; i + 8 <= dim; i += 8)
	{
		uint8_t byte = 0;
		/* Explicit bit tests help vectorization */
		byte |= (transformed[i + 0] > 0) ? (1 << 0) : 0;
		byte |= (transformed[i + 1] > 0) ? (1 << 1) : 0;
		byte |= (transformed[i + 2] > 0) ? (1 << 2) : 0;
		byte |= (transformed[i + 3] > 0) ? (1 << 3) : 0;
		byte |= (transformed[i + 4] > 0) ? (1 << 4) : 0;
		byte |= (transformed[i + 5] > 0) ? (1 << 5) : 0;
		byte |= (transformed[i + 6] > 0) ? (1 << 6) : 0;
		byte |= (transformed[i + 7] > 0) ? (1 << 7) : 0;
		bits[i / 8] = byte;
	}

	/* Handle tail elements */
	for (; i < dim; i++)
	{
		int byte_idx = i / 8;
		int bit_idx	 = i % 8;
		if (transformed[i] > 0)
			bits[byte_idx] |= (1 << bit_idx);
	}
}

/*
 * Internal helper: extract sign bits using dispatched function
 */
static inline void
rabitq_extract_signs(const float *transformed, uint8_t *bits, Dimension dim)
{
	if (mkt_unlikely(!g_rabitq_initialized))
		mkt_rabitq_init_simd();
	g_extract_signs_fn(transformed, bits, dim);
}

/*
 * Lifecycle functions
 */

RaBitQParams *
mkt_rabitq_create(Dimension dim, uint64_t seed)
{
	RaBitQParams *params = mkt_alloc(sizeof(RaBitQParams));
	if (params == NULL)
		return NULL;

	if (mkt_rabitq_init(params, dim, seed) != 0)
	{
		mkt_free(params);
		return NULL;
	}

	return params;
}

int
mkt_rabitq_init(RaBitQParams *params, Dimension dim, uint64_t seed)
{
	if (params == NULL || dim == 0)
		return -1;

	params->dim			 = dim;
	params->seed		 = seed;
	params->packed_bytes = MKT_RABITQ_BYTES(dim);

	/* Allocate orthogonal matrix */
	params->P = mkt_alloc_aligned((size_t)dim * dim * sizeof(float), 64);
	if (params->P == NULL)
		return -1;

	/* Generate random orthogonal matrix */
	if (mkt_random_orthogonal_matrix(params->P, dim, seed) != 0)
	{
		mkt_free_aligned(params->P);
		params->P = NULL;
		return -1;
	}

	return 0;
}

void
mkt_rabitq_cleanup(RaBitQParams *params)
{
	if (params == NULL)
		return;

	if (params->P != NULL)
	{
		mkt_free_aligned(params->P);
		params->P = NULL;
	}
}

void
mkt_rabitq_destroy(RaBitQParams *params)
{
	if (params == NULL)
		return;

	mkt_rabitq_cleanup(params);
	mkt_free(params);
}

/*
 * Encoding functions
 */

RaBitQData *
mkt_rabitq_encode(
		const RaBitQParams *params, VectorRef input, VectorRef centroid)
{
	if (params == NULL || input.data == NULL || centroid.data == NULL)
		return NULL;

	if (input.dim != params->dim || centroid.dim != params->dim)
		return NULL;

	size_t		size   = MKT_RABITQ_DATA_SIZE(params->dim);
	RaBitQData *output = mkt_alloc(size);
	if (output == NULL)
		return NULL;

	if (mkt_rabitq_encode_into(params, input, centroid, output) != 0)
	{
		mkt_free(output);
		return NULL;
	}

	return output;
}

int
mkt_rabitq_encode_into(
		const RaBitQParams *params,
		VectorRef			input,
		VectorRef			centroid,
		RaBitQData		   *output)
{
	if (params == NULL || input.data == NULL || centroid.data == NULL ||
		output == NULL)
		return -1;

	if (input.dim != params->dim || centroid.dim != params->dim)
		return -1;

	Dimension dim = params->dim;

	/* Allocate temporary buffers */
	float *residual	   = mkt_alloc_aligned(dim * sizeof(float), 64);
	float *transformed = mkt_alloc_aligned(dim * sizeof(float), 64);
	float *xu_cb	   = mkt_alloc_aligned(dim * sizeof(float), 64);

	if (residual == NULL || transformed == NULL || xu_cb == NULL)
	{
		if (residual)
			mkt_free_aligned(residual);
		if (transformed)
			mkt_free_aligned(transformed);
		if (xu_cb)
			mkt_free_aligned(xu_cb);
		return -1;
	}

	/* Step 1: Compute residual = input - centroid */
	mkt_vector_sub(input.data, centroid.data, residual, dim);

	/* Step 2: Transform residual through P^T
	 * RaBitQ applies the same rotation to all vectors (data, centroid, query).
	 * The library stores Q^T and multiplies by it, which is effectively P^T.
	 * Since P^T * (data - centroid) = P^T * data - P^T * centroid, applying
	 * P^T to the residual is equivalent to rotating both vectors then
	 * subtracting.
	 */
	mkt_matrix_transpose_vector_mul(params->P, residual, transformed, dim);

	/* Step 3: Extract sign bits (LSB-first packing, FAISS-compatible) */
	rabitq_extract_signs(transformed, output->bits, dim);

	/* Step 4: Compute xu_cb = binary_code - 0.5 (for 1-bit quantization) */
	/* In 1-bit RaBitQ, cb = -0.5, so xu_cb[i] = bit[i] - 0.5 */
	float cb = -0.5f;
	for (Dimension i = 0; i < dim; i++)
	{
		int byte_idx = i / 8;
		int bit_idx	 = i % 8; /* LSB-first */
		int bit		 = (output->bits[byte_idx] >> bit_idx) & 1;
		xu_cb[i]	 = (float)bit + cb;
	}

	/* Step 5: Compute factors for distance estimation */
	float l2_sqr = mkt_l2_norm_squared(transformed, dim);

	float ip_resi_xucb = mkt_dot_product(transformed, xu_cb, dim);

	/* Handle corner case: avoid division by near-zero in f_rescale */
	if (fabsf(ip_resi_xucb) < 1e-10f)
		ip_resi_xucb = 1e-10f;

	/* L2 distance factors (FAISS-style formula for better accuracy)
	 *
	 * FAISS uses a correction factor based on the actual distribution of
	 * values: est_dist = ||v-c||² + ||q-c||² - 2 * dp_multiplier * final_dot
	 * where:
	 *   dp_multiplier = ||v-c||² * sqrt(d) / ||v-c||_1
	 *
	 * The L1 norm is encoded in ip_resi_xucb (which equals 0.5 * ||v-c||_1
	 * because xu_cb = bit - 0.5, so the dot product accumulates ±0.5 * |val|).
	 *
	 * f_add = ||v-c||² (the L2 squared distance to centroid)
	 * f_rescale = dp_multiplier = ||v-c||² * sqrt(d) / ||v-c||_1
	 *
	 * f_error is derived at query time from f_add and f_rescale.
	 */
	float sqrt_d	  = sqrtf((float)dim);
	float l1_norm	  = 2.0f * fabsf(ip_resi_xucb); /* ||v-c||_1 */
	output->f_add	  = l2_sqr;
	output->f_rescale = l2_sqr * sqrt_d / l1_norm; /* dp_multiplier */

	/* Cleanup */
	mkt_free_aligned(residual);
	mkt_free_aligned(transformed);
	mkt_free_aligned(xu_cb);

	return 0;
}

/*
 * Batch encode implementation — always_inline so that the specialized
 * wrappers below pass a static const MktVectorTypeOps from the header,
 * enabling the compiler to inline through every vtable function pointer.
 * MKT_TARGET_CLONES on the wrappers generates AVX2/AVX-512 variants.
 *
 * For f32 input, to_float_block returns the input pointer (zero-copy).
 * For f16, it bulk-converts all vectors in one SIMD-dispatched call.
 * Bulk conversion via to_float_block amortizes call overhead for f16
 * (one SIMD-dispatched call vs N per-vector calls).
 */
__attribute__((always_inline)) static inline int
rabitq_encode_batch_impl(
		const RaBitQParams	   *params,
		const void			   *vectors,
		VectorRef				centroid,
		float				   *f_add,
		float				   *f_rescale,
		uint8_t				   *bits,
		uint16_t				count,
		const MktVectorTypeOps *ops)
{
	Dimension dim		   = params->dim;
	uint32_t  packed_bytes = params->packed_bytes;

	/* Bulk-convert to f32 if needed. For f32, returns input pointer
	 * (zero-copy). For f16, converts into conv_buf via SIMD. */
	float *conv_buf =
			mkt_alloc_aligned((size_t)count * dim * sizeof(float), 64);
	const float *fvecs = ops->to_float_block(vectors, conv_buf, count, dim);

	/* Allocate batch buffers */
	float *residuals =
			mkt_alloc_aligned((size_t)count * dim * sizeof(float), 64);
	float *transformed =
			mkt_alloc_aligned((size_t)count * dim * sizeof(float), 64);
	float *cent_rotated = mkt_alloc_aligned(dim * sizeof(float), 64);

	if (residuals == NULL || transformed == NULL || cent_rotated == NULL)
	{
		if (residuals)
			mkt_free_aligned(residuals);
		if (transformed)
			mkt_free_aligned(transformed);
		if (cent_rotated)
			mkt_free_aligned(cent_rotated);
		mkt_free_aligned(conv_buf);
		return -1;
	}

	/* Step 1: Compute all residuals = vectors[i] - centroid */
	for (uint16_t i = 0; i < count; i++)
	{
		const float *vec = fvecs + i * dim;
		float		*res = residuals + i * dim;
		mkt_vector_sub(vec, centroid.data, res, dim);
	}

	/* Step 2: Batch transform all residuals through P^T
	 * This is the key optimization: matrix P stays in cache while
	 * processing all vectors.
	 */
	mkt_matrix_transpose_vector_mul_batch(
			params->P, residuals, transformed, count, dim);

	/* Step 3: Rotate centroid once (shared across all vectors) */
	mkt_matrix_transpose_vector_mul(
			params->P, centroid.data, cent_rotated, dim);

	/* Step 4: Process each transformed vector to extract bits and factors */
	for (uint16_t i = 0; i < count; i++)
	{
		const float *trans	  = transformed + i * dim;
		uint8_t		*vec_bits = bits + (size_t)i * packed_bytes;

		/* Extract sign bits (LSB-first packing, FAISS-compatible) */
		rabitq_extract_signs(trans, vec_bits, dim);

		/* Compute xu_cb = binary_code - 0.5 */
		float cb		   = -0.5f;
		float ip_resi_xucb = 0.0f;
		float ip_cent_xucb = 0.0f;
		float l2_sqr	   = 0.0f;

		for (Dimension j = 0; j < dim; j++)
		{
			int	  byte_idx = j / 8;
			int	  bit_idx  = j % 8;
			int	  bit	   = (vec_bits[byte_idx] >> bit_idx) & 1;
			float xu_cb_j  = (float)bit + cb;

			ip_resi_xucb += trans[j] * xu_cb_j;
			ip_cent_xucb += cent_rotated[j] * xu_cb_j;
			l2_sqr += trans[j] * trans[j];
		}

		/* ip_cent_xucb computed for future centered distance support */
		(void)ip_cent_xucb;

		/* Handle corner case: avoid division by near-zero in f_rescale */
		if (fabsf(ip_resi_xucb) < 1e-10f)
			ip_resi_xucb = 1e-10f;

		/* L2 distance factors (FAISS-style)
		 * dp_multiplier = ||v-c||² * sqrt(d) / ||v-c||_1
		 * ip_resi_xucb = 0.5 * ||v-c||_1, so ||v-c||_1 = 2 * |ip_resi_xucb|
		 */
		float sqrt_d  = sqrtf((float)dim);
		float l1_norm = 2.0f * fabsf(ip_resi_xucb);
		f_add[i]	  = l2_sqr;
		f_rescale[i]  = l2_sqr * sqrt_d / l1_norm; /* dp_multiplier */
	}

	/* Cleanup */
	mkt_free_aligned(residuals);
	mkt_free_aligned(transformed);
	mkt_free_aligned(cent_rotated);
	mkt_free_aligned(conv_buf);

	return 0;
}

/* Specialized wrappers — MKT_TARGET_CLONES generates SIMD variants */

MKT_TARGET_CLONES static int
rabitq_encode_batch_f32(
		const RaBitQParams *params,
		const void		   *vectors,
		VectorRef			centroid,
		float			   *f_add,
		float			   *f_rescale,
		uint8_t			   *bits,
		uint16_t			count)
{
	return rabitq_encode_batch_impl(
			params,
			vectors,
			centroid,
			f_add,
			f_rescale,
			bits,
			count,
			&mkt_f32_type_ops);
}

MKT_TARGET_CLONES static int
rabitq_encode_batch_f16(
		const RaBitQParams *params,
		const void		   *vectors,
		VectorRef			centroid,
		float			   *f_add,
		float			   *f_rescale,
		uint8_t			   *bits,
		uint16_t			count)
{
	return rabitq_encode_batch_impl(
			params,
			vectors,
			centroid,
			f_add,
			f_rescale,
			bits,
			count,
			&mkt_f16_type_ops);
}

#if defined(MKT_F16C_SUPPORT) && !defined(MKT_SIMD_NONE)
MKT_TARGET_F16C_AVX2 static int
rabitq_encode_batch_f16c(
		const RaBitQParams *params,
		const void		   *vectors,
		VectorRef			centroid,
		float			   *f_add,
		float			   *f_rescale,
		uint8_t			   *bits,
		uint16_t			count)
{
	return rabitq_encode_batch_impl(
			params,
			vectors,
			centroid,
			f_add,
			f_rescale,
			bits,
			count,
			&mkt_f16c_type_ops);
}
#endif

int
mkt_rabitq_encode_batch(
		const RaBitQParams *params,
		const void		   *vectors,
		MktVecType			vec_type,
		VectorRef			centroid,
		float			   *f_add,
		float			   *f_rescale,
		uint8_t			   *bits,
		uint16_t			count)
{
	if (params == NULL || vectors == NULL || centroid.data == NULL ||
		f_add == NULL || f_rescale == NULL || bits == NULL || count == 0)
		return -1;

	if (centroid.dim != params->dim)
		return -1;

	/* Single dispatch point — selects the inline vtable once */
	switch (vec_type)
	{
	case MKT_VEC_F32:
		return rabitq_encode_batch_f32(
				params, vectors, centroid, f_add, f_rescale, bits, count);
#if defined(MKT_F16C_SUPPORT) && !defined(MKT_SIMD_NONE)
	case MKT_VEC_F16C:
		return rabitq_encode_batch_f16c(
				params, vectors, centroid, f_add, f_rescale, bits, count);
#endif
	default:
		return rabitq_encode_batch_f16(
				params, vectors, centroid, f_add, f_rescale, bits, count);
	}
}

/*
 * Batch helpers
 */

RaBitQBatch *
mkt_rabitq_batch_create(uint16_t count, Dimension dim)
{
	if (count == 0 || dim == 0)
		return NULL;

	RaBitQBatch *batch = mkt_alloc(sizeof(RaBitQBatch));
	if (batch == NULL)
		return NULL;

	uint32_t packed_bytes = MKT_RABITQ_BYTES(dim);

	batch->count		= count;
	batch->packed_bytes = (uint16_t)packed_bytes;
	batch->f_add		= mkt_alloc(count * sizeof(float));
	batch->f_rescale	= mkt_alloc(count * sizeof(float));
	batch->bits			= mkt_alloc((size_t)count * packed_bytes);

	if (batch->f_add == NULL || batch->f_rescale == NULL ||
		batch->bits == NULL)
	{
		mkt_rabitq_batch_destroy(batch);
		return NULL;
	}

	return batch;
}

void
mkt_rabitq_batch_destroy(RaBitQBatch *batch)
{
	if (batch == NULL)
		return;

	if (batch->f_add)
		mkt_free(batch->f_add);
	if (batch->f_rescale)
		mkt_free(batch->f_rescale);
	if (batch->bits)
		mkt_free(batch->bits);
	mkt_free(batch);
}

RaBitQBatch *
mkt_rabitq_encode_batch_alloc(
		const RaBitQParams *params,
		const void		   *vectors,
		MktVecType			vec_type,
		VectorRef			centroid,
		uint16_t			count)
{
	if (params == NULL || vectors == NULL || centroid.data == NULL ||
		count == 0)
		return NULL;

	RaBitQBatch *batch = mkt_rabitq_batch_create(count, params->dim);
	if (batch == NULL)
		return NULL;

	if (mkt_rabitq_encode_batch(
				params,
				vectors,
				vec_type,
				centroid,
				batch->f_add,
				batch->f_rescale,
				batch->bits,
				count) != 0)
	{
		mkt_rabitq_batch_destroy(batch);
		return NULL;
	}

	return batch;
}

/*
 * Query preparation
 */

RaBitQQueryState *
mkt_rabitq_prepare_query(
		const RaBitQParams *params, VectorRef query, VectorRef centroid)
{
	if (params == NULL || query.data == NULL || centroid.data == NULL)
		return NULL;

	if (query.dim != params->dim || centroid.dim != params->dim)
		return NULL;

	Dimension dim = params->dim;

	RaBitQQueryState *state = mkt_alloc(sizeof(RaBitQQueryState));
	if (state == NULL)
		return NULL;

	state->transformed = mkt_alloc_aligned(dim * sizeof(float), 64);
	if (state->transformed == NULL)
	{
		mkt_free(state);
		return NULL;
	}

	state->dim = dim;

	/* Compute residual = query - centroid */
	float *residual = mkt_alloc_aligned(dim * sizeof(float), 64);
	if (residual == NULL)
	{
		mkt_free_aligned(state->transformed);
		mkt_free(state);
		return NULL;
	}

	mkt_vector_sub(query.data, centroid.data, residual, dim);

	/* Transform through P^T */
	mkt_matrix_transpose_vector_mul(
			params->P, residual, state->transformed, dim);

	/* Compute g_add = ||query - centroid||^2 */
	state->g_add = mkt_l2_norm_squared(state->transformed, dim);

	/* Compute g_error = sqrt(g_add) for error bound */
	state->g_error = sqrtf(state->g_add);

	/* Compute sum of transformed values for FAISS-style distance formula */
	state->sum_transformed = mkt_vector_sum(state->transformed, dim);

	/* Precompute 1/sqrt(dim) for distance formula */
	state->inv_sqrt_d = 1.0f / sqrtf((float)dim);

	/* Precompute C_error = 2*ε/√(d-1) for deriving f_error from compact
	 * data */
	if (dim > 1)
		state->c_error = 2.0f * MKT_RABITQ_EPSILON / sqrtf((float)(dim - 1));
	else
		state->c_error = 0.0f;

	mkt_free_aligned(residual);

	return state;
}

void
mkt_rabitq_free_query(RaBitQQueryState *state)
{
	if (state == NULL)
		return;

	if (state->transformed != NULL)
		mkt_free_aligned(state->transformed);

	mkt_free(state);
}

/*
 * Distance computation
 */

Distance
mkt_rabitq_distance(
		const RaBitQQueryState *query_state,
		const RaBitQData	   *data,
		Dimension				dim)
{
	if (query_state == NULL || data == NULL)
		return -1.0f;

	if (query_state->dim != dim)
		return -1.0f;

	/* Compute binary inner product: sum of transformed[i] where bit[i] = 1 */
	float binary_ip =
			rabitq_inner_product(query_state->transformed, data->bits, dim);

	/* FAISS-style distance formula (better accuracy with non-quantized query):
	 *
	 * est_dist = ||v-c||² + ||q-c||² - 2 * dp_multiplier * final_dot
	 *
	 * where:
	 * - ||v-c||² = f_add (stored per vector)
	 * - ||q-c||² = g_add (stored in query state)
	 * - dp_multiplier = ||v-c||² * sqrt(d) / ||v-c||_1 = f_rescale
	 * - final_dot = (2 * binary_ip - sum_transformed) / sqrt(d)
	 *
	 * FAISS normalizes dp_multiplier by the L1 norm to correct for the
	 * distribution of residual values. This gives better distance estimates.
	 */
	float final_dot = (2.0f * binary_ip - query_state->sum_transformed) *
					  query_state->inv_sqrt_d;

	float est_dist = data->f_add + query_state->g_add -
					 2.0f * data->f_rescale * final_dot;

	return est_dist;
}

void
mkt_rabitq_distance_batch(
		const RaBitQQueryState *qstate,
		const float			   *f_add,
		const float			   *f_rescale,
		const uint8_t		   *bits,
		uint32_t				count,
		Dimension				dim,
		Distance			   *distances)
{
	if (qstate == NULL || f_add == NULL || f_rescale == NULL || bits == NULL ||
		distances == NULL || count == 0)
		return;

	uint32_t packed_bytes = MKT_RABITQ_BYTES(dim);

	for (uint32_t i = 0; i < count; i++)
	{
		float binary_ip = rabitq_inner_product(
				qstate->transformed, bits + (size_t)i * packed_bytes, dim);

		float final_dot = (2.0f * binary_ip - qstate->sum_transformed) *
						  qstate->inv_sqrt_d;

		distances[i] = f_add[i] + qstate->g_add -
					   2.0f * f_rescale[i] * final_dot;
	}
}

void
mkt_rabitq_inner_product_multi(
		const float	  *transformed,
		const uint8_t *bits,
		uint32_t	   stride,
		Dimension	   dim,
		uint32_t	   count,
		float		  *results)
{
	if (mkt_unlikely(!g_rabitq_initialized))
		mkt_rabitq_init_simd();
	g_inner_product_multi_fn(transformed, bits, stride, dim, count, results);
}

void
mkt_rabitq_distance_batch_multi(
		const RaBitQQueryState *qstate,
		const float			   *f_add,
		const float			   *f_rescale,
		const uint8_t		   *bits,
		uint32_t				count,
		Dimension				dim,
		Distance			   *distances)
{
	if (qstate == NULL || f_add == NULL || f_rescale == NULL || bits == NULL ||
		distances == NULL || count == 0)
		return;

	uint32_t packed_bytes = MKT_RABITQ_BYTES(dim);

	/* Compute all inner products in a single multi-candidate pass */
	float *ips = mkt_alloc_aligned(count * sizeof(float), 64);
	if (ips == NULL)
		return;

	mkt_rabitq_inner_product_multi(
			qstate->transformed, bits, packed_bytes, dim, count, ips);

	/* Apply distance formula to all candidates */
	float g_add		 = qstate->g_add;
	float sum_t		 = qstate->sum_transformed;
	float inv_sqrt_d = qstate->inv_sqrt_d;

	for (uint32_t i = 0; i < count; i++)
	{
		float final_dot = (2.0f * ips[i] - sum_t) * inv_sqrt_d;
		distances[i]	= f_add[i] + g_add - 2.0f * f_rescale[i] * final_dot;
	}

	mkt_free_aligned(ips);
}

void
mkt_rabitq_distance_with_bound(
		const RaBitQQueryState *query_state,
		const RaBitQData	   *data,
		Dimension				dim,
		Distance			   *est_dist,
		Distance			   *lower_bound)
{
	if (query_state == NULL || data == NULL || est_dist == NULL ||
		lower_bound == NULL)
	{
		if (est_dist)
			*est_dist = -1.0f;
		if (lower_bound)
			*lower_bound = -1.0f;
		return;
	}

	/* Compute estimated distance */
	*est_dist = mkt_rabitq_distance(query_state, data, dim);

	/* Derive f_error from f_add and f_rescale.
	 *
	 * The ratio f_rescale²/f_add corresponds to (||v-c||² * d / ||v-c||_1²)
	 * which equals (l2_sqr * xu_cb_norm_sq) / ip_resi_xucb² in encoding.
	 * When ratio <= 1.0 (rounding), use a small safety margin.
	 * Otherwise: f_error = c_error * sqrt(f_rescale² - f_add).
	 */
	float f_rsq = data->f_rescale * data->f_rescale;
	float ratio = f_rsq / data->f_add;
	float f_error;

	if (ratio <= 1.0f || dim <= 1)
		f_error = 2e-4f * sqrtf(data->f_add);
	else
		f_error = query_state->c_error * sqrtf(f_rsq - data->f_add);

	/* Compute lower bound: est_dist - f_error * g_error
	 *
	 * Note: We add a small relative epsilon (1e-5 * est_dist) to handle
	 * floating-point imprecision.
	 */
	float error_margin = f_error * query_state->g_error;
	float fp_margin	   = 1e-5f * fabsf(*est_dist);
	*lower_bound	   = *est_dist - error_margin - fp_margin;
}
