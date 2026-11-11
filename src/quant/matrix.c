/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * matrix.c - Random orthogonal matrix generation for RaBitQ
 *
 * Generates random orthogonal matrices via QR decomposition of Gaussian
 * random matrices, using modified Gram-Schmidt.
 *
 * The orthogonal matrix P ensures that residual vectors have isotropic
 * distribution, which is essential for RaBitQ's theoretical error bounds.
 */

#include "vs_config.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#ifdef VS_HAVE_CBLAS
/*
 * macOS: cblas lives inside Accelerate's vecLib sub-framework. We
 * include it directly rather than via the `Accelerate/Accelerate.h`
 * umbrella because the umbrella pulls in MacTypes.h, which
 * `typedef long Size` clashes with PostgreSQL's `typedef size_t
 * Size` in the PG-extension build.
 *
 * The legacy `vecLib/cblas.h` deprecates every prototype past macOS
 * 13.3; under -DACCELERATE_NEW_LAPACK (set in meson.build for the
 * Apple branch) we use `vecLib/cblas_new.h` instead, which aliases
 * the same call sites to the non-deprecated LP64 symbols.
 */
#ifdef __APPLE__
#include <vecLib/cblas_new.h>
#else
#include <cblas.h>
#endif
#endif

#if defined(VS_SIMD_FULL) && (defined(__x86_64__) || defined(_M_X64))
#include <immintrin.h>
#elif defined(VS_SIMD_FULL) && (defined(__aarch64__) || defined(_M_ARM64))
#include <arm_neon.h>
#endif

#include "algo/simd_utils.h"
#include "algo/vecops.h"
#include "core/memory.h"
#include "core/platform.h"
#include "core/types.h"
#include "quant/matrix.h"

/*
 * Simple xoshiro256** PRNG for reproducible random generation.
 * Chosen for speed and quality - not cryptographic.
 */
typedef struct
{
	uint64_t s[4];
} Xoshiro256State;

static inline uint64_t
rotl(uint64_t x, int k)
{
	return (x << k) | (x >> (64 - k));
}

static uint64_t
xoshiro256_next(Xoshiro256State *state)
{
	uint64_t *s		 = state->s;
	uint64_t  result = rotl(s[1] * 5, 7) * 9;
	uint64_t  t		 = s[1] << 17;

	s[2] ^= s[0];
	s[3] ^= s[1];
	s[1] ^= s[2];
	s[0] ^= s[3];
	s[2] ^= t;
	s[3] = rotl(s[3], 45);

	return result;
}

static void
xoshiro256_seed(Xoshiro256State *state, uint64_t seed)
{
	/* SplitMix64 to expand seed into state */
	for (int i = 0; i < 4; i++)
	{
		seed += 0x9e3779b97f4a7c15ULL;
		uint64_t z	= seed;
		z			= (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
		z			= (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
		state->s[i] = z ^ (z >> 31);
	}
}

/*
 * Generate uniform random double in [0, 1)
 */
static double
random_uniform(Xoshiro256State *state)
{
	/* Use upper 53 bits for double precision */
	uint64_t x = xoshiro256_next(state) >> 11;
	return (double)x / (double)(1ULL << 53);
}

/*
 * Generate standard normal random variable using Box-Muller transform.
 * Returns two independent samples for efficiency.
 */
static void
random_normal_pair(Xoshiro256State *state, double *n1, double *n2)
{
	double u1 = random_uniform(state);
	double u2 = random_uniform(state);

	/* Avoid log(0) */
	if (u1 < 1e-15)
		u1 = 1e-15;

	double r	 = sqrt(-2.0 * log(u1));
	double theta = 2.0 * M_PI * u2;

	*n1 = r * cos(theta);
	*n2 = r * sin(theta);
}

/*
 * Fill matrix with standard normal random values.
 * Matrix is dim x dim, stored in row-major order.
 */
static void
fill_gaussian_matrix(float *matrix, Dimension dim, uint64_t seed)
{
	Xoshiro256State state;
	xoshiro256_seed(&state, seed);

	size_t n = (size_t)dim * dim;
	size_t i = 0;

	/* Generate pairs of normal values */
	while (i + 1 < n)
	{
		double n1, n2;
		random_normal_pair(&state, &n1, &n2);
		matrix[i]	  = (float)n1;
		matrix[i + 1] = (float)n2;
		i += 2;
	}

	/* Handle odd size */
	if (i < n)
	{
		double n1, n2;
		random_normal_pair(&state, &n1, &n2);
		(void)n2;
		matrix[i] = (float)n1;
	}
}

/*
 * The matrix is regenerated from its seed wherever an index is read, so its
 * bits must not depend on the compiler. The QR code below is evaluated in
 * source order, without reassociation or fused multiply-adds. The clang
 * pragma needs -ffp-contract=fast-honor-pragmas, which meson.build sets.
 */
#if defined(__clang__)
#define STRICT_FP_FN
#define STRICT_FP_BODY _Pragma("clang fp reassociate(off) contract(off)")
#elif defined(__GNUC__)
#define STRICT_FP_FN \
	__attribute__((optimize("no-associative-math", "fp-contract=off")))
#define STRICT_FP_BODY
#else
#define STRICT_FP_FN
#define STRICT_FP_BODY
#endif

/* Columns orthogonalized together, packed so a strip stays in cache. */
#define QR_STRIP 32

/*
 * Squared norm of a contiguous column. Summed in the order gcc 14 and later
 * give it at -O2 on x86-64, so the matrix matches indexes built that way.
 */
STRICT_FP_FN static float
column_norm2(const float *x, Dimension n)
{
	STRICT_FP_BODY
	float	  norm = 0.0f;
	Dimension i	   = 0;

	if (n > 20)
	{
		float s0 = 0.0f;
		float s1 = 0.0f;

		for (; i + 4 <= n; i += 4)
		{
			s0 = (x[i + 2] * x[i + 2] + x[i] * x[i]) + s0;
			s1 = (x[i + 3] * x[i + 3] + x[i + 1] * x[i + 1]) + s1;
		}
		norm = s0 + s1;
	}
	for (; i < n; i++)
		norm += x[i] * x[i];
	return norm;
}

/*
 * One pass down a strip of QR_STRIP columns. With p set, subtract
 * p_dots[c] * p from each column c. With q set, store q . column c in
 * q_dots[c], after that update. p and q are columns with the given stride,
 * outside the strip.
 *
 * Every kernel gives each column a lane of its own and rounds each multiply
 * and add separately, so they all produce the same bits. The SIMD kernels
 * use operators for arithmetic, not intrinsics: clang compiles intrinsics
 * with the file's flags, which would let them fuse.
 */
typedef void (*StripSweepFn)(
		float		*strip,
		Dimension	 n,
		const float *p,
		const float *p_dots,
		const float *q,
		float		*q_dots,
		size_t		 stride);

STRICT_FP_FN static void
strip_sweep_scalar(
		float *restrict strip,
		Dimension	 n,
		const float *p,
		const float *restrict p_dots,
		const float *q,
		float *restrict q_dots,
		size_t stride)
{
	STRICT_FP_BODY
	float acc[QR_STRIP] = {0};

	for (Dimension i = 0; i < n; i++)
	{
		float *restrict row = strip + (size_t)i * QR_STRIP;

		if (p != NULL)
		{
			float pi = p[i * stride];

			for (int c = 0; c < QR_STRIP; c++)
				row[c] = row[c] - p_dots[c] * pi;
		}
		if (q != NULL)
		{
			float qi = q[i * stride];

			for (int c = 0; c < QR_STRIP; c++)
				acc[c] += qi * row[c];
		}
	}

	if (q != NULL)
		memcpy(q_dots, acc, sizeof(acc));
}

#if defined(VS_SIMD_FULL) && (defined(__x86_64__) || defined(_M_X64))
/* Plain AVX2 on purpose: with FMA a lane would round differently. */
#define AVX2_LANES (QR_STRIP / 8)

__attribute__((target("avx2"))) STRICT_FP_FN static void
strip_sweep_avx2(
		float		*strip,
		Dimension	 n,
		const float *p,
		const float *p_dots,
		const float *q,
		float		*q_dots,
		size_t		 stride)
{
	STRICT_FP_BODY
	__m256 acc[AVX2_LANES];
	__m256 pd[AVX2_LANES];

#pragma GCC unroll 8
	for (int v = 0; v < AVX2_LANES; v++)
	{
		acc[v] = _mm256_setzero_ps();
		pd[v]  = p != NULL ? _mm256_loadu_ps(p_dots + 8 * v) : acc[v];
	}

	for (Dimension i = 0; i < n; i++)
	{
		float *row = strip + (size_t)i * QR_STRIP;
		__m256 pi  = _mm256_set1_ps(p != NULL ? p[i * stride] : 0.0f);
		__m256 qi  = _mm256_set1_ps(q != NULL ? q[i * stride] : 0.0f);

#pragma GCC unroll 8
		for (int v = 0; v < AVX2_LANES; v++)
		{
			__m256 a = _mm256_loadu_ps(row + 8 * v);

			if (p != NULL)
			{
				a = a - pd[v] * pi;
				_mm256_storeu_ps(row + 8 * v, a);
			}
			if (q != NULL)
				acc[v] = acc[v] + qi * a;
		}
	}

	if (q != NULL)
	{
#pragma GCC unroll 8
		for (int v = 0; v < AVX2_LANES; v++)
			_mm256_storeu_ps(q_dots + 8 * v, acc[v]);
	}
}

#define AVX512_LANES (QR_STRIP / 16)

__attribute__((target("avx512f"))) STRICT_FP_FN static void
strip_sweep_avx512(
		float		*strip,
		Dimension	 n,
		const float *p,
		const float *p_dots,
		const float *q,
		float		*q_dots,
		size_t		 stride)
{
	STRICT_FP_BODY
	__m512 acc[AVX512_LANES];
	__m512 pd[AVX512_LANES];

#pragma GCC unroll 8
	for (int v = 0; v < AVX512_LANES; v++)
	{
		acc[v] = _mm512_setzero_ps();
		pd[v]  = p != NULL ? _mm512_loadu_ps(p_dots + 16 * v) : acc[v];
	}

	for (Dimension i = 0; i < n; i++)
	{
		float *row = strip + (size_t)i * QR_STRIP;
		__m512 pi  = _mm512_set1_ps(p != NULL ? p[i * stride] : 0.0f);
		__m512 qi  = _mm512_set1_ps(q != NULL ? q[i * stride] : 0.0f);

#pragma GCC unroll 8
		for (int v = 0; v < AVX512_LANES; v++)
		{
			__m512 a = _mm512_loadu_ps(row + 16 * v);

			if (p != NULL)
			{
				a = a - pd[v] * pi;
				_mm512_storeu_ps(row + 16 * v, a);
			}
			if (q != NULL)
				acc[v] = acc[v] + qi * a;
		}
	}

	if (q != NULL)
	{
#pragma GCC unroll 8
		for (int v = 0; v < AVX512_LANES; v++)
			_mm512_storeu_ps(q_dots + 16 * v, acc[v]);
	}
}
#endif /* x86_64 */

#if defined(VS_SIMD_FULL) && (defined(__aarch64__) || defined(_M_ARM64))
#define NEON_LANES (QR_STRIP / 4)

STRICT_FP_FN static void
strip_sweep_neon(
		float		*strip,
		Dimension	 n,
		const float *p,
		const float *p_dots,
		const float *q,
		float		*q_dots,
		size_t		 stride)
{
	STRICT_FP_BODY
	float32x4_t acc[NEON_LANES];
	float32x4_t pd[NEON_LANES];

#pragma GCC unroll 8
	for (int v = 0; v < NEON_LANES; v++)
	{
		acc[v] = vdupq_n_f32(0.0f);
		pd[v]  = p != NULL ? vld1q_f32(p_dots + 4 * v) : acc[v];
	}

	for (Dimension i = 0; i < n; i++)
	{
		float	   *row = strip + (size_t)i * QR_STRIP;
		float32x4_t pi	= vdupq_n_f32(p != NULL ? p[i * stride] : 0.0f);
		float32x4_t qi	= vdupq_n_f32(q != NULL ? q[i * stride] : 0.0f);

#pragma GCC unroll 8
		for (int v = 0; v < NEON_LANES; v++)
		{
			float32x4_t a = vld1q_f32(row + 4 * v);

			if (p != NULL)
			{
				a = a - pd[v] * pi;
				vst1q_f32(row + 4 * v, a);
			}
			if (q != NULL)
				acc[v] = acc[v] + qi * a;
		}
	}

	if (q != NULL)
	{
#pragma GCC unroll 8
		for (int v = 0; v < NEON_LANES; v++)
			vst1q_f32(q_dots + 4 * v, acc[v]);
	}
}
#endif /* aarch64 */

/* Resolved per matrix, so a SIMD override set in a test takes effect.
 * Only simd=full builds have the hand-written kernels. */
static StripSweepFn
resolve_strip_sweep(void)
{
#if defined(VS_SIMD_FULL) && (defined(__x86_64__) || defined(_M_X64))
	if (vs_has_simd(SIMD_AVX512F))
		return strip_sweep_avx512;
	if (vs_has_simd(SIMD_AVX2))
		return strip_sweep_avx2;
#elif defined(VS_SIMD_FULL) && (defined(__aarch64__) || defined(_M_ARM64))
	if (vs_has_simd(SIMD_NEON))
		return strip_sweep_neon;
#endif
	return strip_sweep_scalar;
}

/*
 * QR decomposition using modified Gram-Schmidt, overwriting the row-major
 * A with Q.
 *
 * Columns are processed in strips. A strip first has every finished column
 * to its left projected out, one at a time and vectorized across the strip,
 * then is orthogonalized within itself. Each column still sees the
 * classic column-by-column order of operations.
 */
STRICT_FP_FN static int
gram_schmidt_qr(float *A, Dimension dim)
{
	STRICT_FP_BODY
	StripSweepFn strip_sweep = resolve_strip_sweep();
	float		 dots[2][QR_STRIP];

	/* A context of its own, since standalone vs_free() keeps the memory. */
	VS_MEMCTX_SCOPE(qr_ctx);
	if (qr_ctx == NULL)
		return -1;

	VsMemCtx old   = vs_memctx_switch(qr_ctx);
	float	*strip = vs_alloc((size_t)dim * QR_STRIP * sizeof(float));
	float	*col   = vs_alloc((size_t)dim * sizeof(float));

	vs_memctx_switch(old);
	if (strip == NULL || col == NULL)
		return -1;

	for (Dimension s = 0; s < dim; s += QR_STRIP)
	{
		Dimension width = dim - s < QR_STRIP ? dim - s : QR_STRIP;

		/* Unused strip columns stay zero and come out of every update so. */
		memset(strip, 0, (size_t)dim * QR_STRIP * sizeof(float));
		for (Dimension i = 0; i < dim; i++)
			memcpy(strip + (size_t)i * QR_STRIP,
				   A + (size_t)i * dim + s,
				   width * sizeof(float));

		/* Each update by column j - 1 is fused with the dots for column j. */
		for (Dimension j = 0; j < s; j++)
			strip_sweep(
					strip,
					dim,
					j > 0 ? A + j - 1 : NULL,
					dots[(j + 1) % 2],
					A + j,
					dots[j % 2],
					dim);
		if (s > 0)
			strip_sweep(
					strip, dim, A + s - 1, dots[(s + 1) % 2], NULL, NULL, dim);

		for (Dimension c = 0; c < width; c++)
		{
			for (Dimension i = 0; i < dim; i++)
				col[i] = strip[(size_t)i * QR_STRIP + c];

			float norm = sqrtf(column_norm2(col, dim));

			/* Only a degenerate Gaussian draw has a zero column. */
			if (norm < 1e-10f)
				norm = 1.0f;

			for (Dimension i = 0; i < dim; i++)
			{
				col[i] /= norm;
				strip[(size_t)i * QR_STRIP + c] = col[i];
			}

			if (c + 1 == width)
				break;

			/* Columns up to c are done, so a zero dot leaves them alone. */
			strip_sweep(strip, dim, NULL, NULL, col, dots[0], 1);
			memset(dots[0], 0, (c + 1) * sizeof(float));
			strip_sweep(strip, dim, col, dots[0], NULL, NULL, 1);
		}

		for (Dimension i = 0; i < dim; i++)
			memcpy(A + (size_t)i * dim + s,
				   strip + (size_t)i * QR_STRIP,
				   width * sizeof(float));
	}

	return 0;
}

/*
 * Generate random orthogonal matrix using QR decomposition.
 *
 * Creates a Gaussian random matrix and orthogonalizes it via QR.
 * The resulting Q matrix is uniformly distributed over O(n)
 * (orthogonal group) according to Haar measure.
 *
 * Parameters:
 *   matrix: Output buffer (dim * dim floats, row-major)
 *   dim:    Matrix dimension
 *   seed:   Random seed for reproducibility
 *
 * Returns 0 on success, -1 on failure.
 */
int
vs_random_orthogonal_matrix(float *matrix, Dimension dim, uint64_t seed)
{
	if (matrix == NULL || dim == 0)
		return -1;

	/* Fill with Gaussian random values */
	fill_gaussian_matrix(matrix, dim, seed);

	/* QR decomposition to get orthogonal matrix */
	return gram_schmidt_qr(matrix, dim);
}

/*
 * Verify matrix is approximately orthogonal (for testing).
 *
 * Checks that M * M^T ≈ I within tolerance.
 *
 * Returns 1 if orthogonal, 0 if not.
 */
int
vs_matrix_is_orthogonal(const float *matrix, Dimension dim, float tolerance)
{
	if (matrix == NULL || dim == 0)
		return 0;

	/* Check M * M^T = I */
	for (Dimension i = 0; i < dim; i++)
	{
		for (Dimension j = 0; j < dim; j++)
		{
			/* Compute (i,j) element of M * M^T = dot(row_i, row_j) */
			float dot =
					vs_dot_product(matrix + i * dim, matrix + j * dim, dim);

			/* Expected value: 1 on diagonal, 0 elsewhere */
			float expected = (i == j) ? 1.0f : 0.0f;
			float diff	   = fabsf(dot - expected);

			if (diff > tolerance)
				return 0;
		}
	}

	return 1;
}

/*
 * Matrix-vector multiplication: result = M^T * v
 *
 * Computes the product of the transpose of M with vector v.
 * This is the common operation in RaBitQ for transforming residuals.
 *
 * Rewritten as: result = sum_j (v[j] * row_j)
 * This accesses rows (contiguous in row-major), enabling auto-vectorization.
 *
 * Parameters:
 *   M:      Input matrix (dim x dim, row-major)
 *   v:      Input vector (dim elements)
 *   result: Output vector (dim elements)
 *   dim:    Dimension
 */
void
vs_matrix_transpose_vector_mul(
		const float *M, const float *v, float *result, Dimension dim)
{
	/* Zero result */
	memset(result, 0, dim * sizeof(float));

	/* Accumulate v[j] * row_j for each row j */
	for (Dimension j = 0; j < dim; j++)
	{
		const float *row   = M + j * dim;
		float		 scale = v[j];

		/* This inner loop auto-vectorizes well */
		for (Dimension i = 0; i < dim; i++)
			result[i] += scale * row[i];
	}
}

/*
 * Matrix-vector multiplication: result = M * v
 *
 * Each output element is the dot product of a row with v.
 * Rows are contiguous in row-major storage, enabling SIMD.
 *
 * Parameters:
 *   M:      Input matrix (dim x dim, row-major)
 *   v:      Input vector (dim elements)
 *   result: Output vector (dim elements)
 *   dim:    Dimension
 */
void
vs_matrix_vector_mul(
		const float *M, const float *v, float *result, Dimension dim)
{
	for (Dimension i = 0; i < dim; i++)
		result[i] = vs_dot_product(M + i * dim, v, dim);
}

/*
 * Batched matrix-vector multiplication: results = M^T * vectors
 *
 * Computes M^T * v for multiple vectors at once. This is more efficient
 * than calling vs_matrix_transpose_vector_mul() repeatedly because:
 * 1. Matrix M is loaded into cache once and reused for all vectors
 * 2. Inner loop processes contiguous memory (good for SIMD)
 *
 * Memory layout:
 *   vectors: count vectors of dim elements each, contiguous (vectors[i*dim +
 * j]) results: count output vectors, same layout
 *
 * Parameters:
 *   M:       Input matrix (dim x dim, row-major)
 *   vectors: Input vectors (count * dim elements)
 *   results: Output vectors (count * dim elements)
 *   count:   Number of vectors to process
 *   dim:     Vector/matrix dimension
 */
VS_TARGET_CLONES static void
matrix_transpose_mul_batch_inner(
		const float *M_row,
		const float *vectors,
		float		*results,
		uint32_t	 count,
		Dimension	 dim,
		Dimension	 j)
{
	/* Process each vector - inner loop is contiguous and vectorizes well */
	for (uint32_t i = 0; i < count; i++)
	{
		float		 scale	= vectors[i * dim + j];
		float		*result = results + i * dim;
		const float *row	= M_row;

		for (Dimension k = 0; k < dim; k++)
			result[k] += scale * row[k];
	}
}

static void
matrix_transpose_mul_batch_builtin(
		const float *M,
		const float *vectors,
		float		*results,
		uint32_t	 count,
		Dimension	 dim)
{
	/* Zero all results */
	memset(results, 0, (size_t)count * dim * sizeof(float));

	/* Process each row of M - row stays in cache while we update all vectors
	 */
	for (Dimension j = 0; j < dim; j++)
	{
		const float *M_row = M + j * dim;
		matrix_transpose_mul_batch_inner(
				M_row, vectors, results, count, dim, j);
	}
}

#ifdef VS_HAVE_CBLAS
static void
matrix_transpose_mul_batch_cblas(
		const float *M,
		const float *vectors,
		float		*results,
		uint32_t	 count,
		Dimension	 dim)
{
	/*
	 * Use CBLAS sgemm for optimized matrix multiplication.
	 *
	 * We want: results[i] = M^T * vectors[i] for each vector i
	 * In row-major form: results = vectors * M
	 *
	 * sgemm: C = alpha * A * B + beta * C
	 * Where: A = vectors (count x dim), B = M (dim x dim), C = results
	 */
	cblas_sgemm(
			CblasRowMajor,
			CblasNoTrans,
			CblasNoTrans,
			(int)count, /* M */
			(int)dim,	/* N */
			(int)dim,	/* K */
			1.0f,		/* alpha */
			vectors,	/* A */
			(int)dim,	/* lda */
			M,			/* B */
			(int)dim,	/* ldb */
			0.0f,		/* beta */
			results,	/* C */
			(int)dim	/* ldc */
	);
}
#endif /* VS_HAVE_CBLAS */

/*
 * Runtime selection of matrix multiplication implementation.
 * Default: use CBLAS if available, else built-in.
 */
static bool g_use_cblas = true;

void
vs_matrix_set_use_cblas(bool use_cblas)
{
	g_use_cblas = use_cblas;
}

bool
vs_matrix_get_use_cblas(void)
{
#ifdef VS_HAVE_CBLAS
	return g_use_cblas;
#else
	return false;
#endif
}

const char *
vs_matrix_impl_name(void)
{
#ifdef VS_HAVE_CBLAS
	return g_use_cblas ? "cblas" : "builtin";
#else
	return "builtin";
#endif
}

void
vs_matrix_transpose_vector_mul_batch(
		const float *M,
		const float *vectors,
		float		*results,
		uint32_t	 count,
		Dimension	 dim)
{
#ifdef VS_HAVE_CBLAS
	if (g_use_cblas)
	{
		matrix_transpose_mul_batch_cblas(M, vectors, results, count, dim);
		return;
	}
#endif
	matrix_transpose_mul_batch_builtin(M, vectors, results, count, dim);
}
