/*
 * matrix.c - Random orthogonal matrix generation for RaBitQ
 *
 * Generates random orthogonal matrices via QR decomposition of Gaussian
 * random matrices. Uses the Householder algorithm for numerical stability.
 *
 * The orthogonal matrix P ensures that residual vectors have isotropic
 * distribution, which is essential for RaBitQ's theoretical error bounds.
 */

#include "mkt_config.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#ifdef MKT_HAVE_CBLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

#include "algo/simd_utils.h"
#include "algo/vecops.h"
#include "core/memory.h"
#include "mkt_types.h"

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
 * QR decomposition using modified Gram-Schmidt.
 *
 * Input: A (dim x dim matrix, row-major)
 * Output: Q (orthogonal matrix, row-major)
 *
 * The input matrix A is overwritten with Q.
 */
static void
gram_schmidt_qr(float *A, Dimension dim)
{
	/* Process columns */
	for (Dimension j = 0; j < dim; j++)
	{
		/* Get pointer to column j (stored as row j in transposed view) */
		/* For row-major, column j elements are at A[0*dim+j], A[1*dim+j], ...
		 */

		/* Compute norm of column j */
		float norm = 0.0f;
		for (Dimension i = 0; i < dim; i++)
			norm += A[i * dim + j] * A[i * dim + j];
		norm = sqrtf(norm);

		/* Handle near-zero columns (shouldn't happen with Gaussian init) */
		if (norm < 1e-10f)
			norm = 1.0f;

		/* Normalize column j */
		for (Dimension i = 0; i < dim; i++)
			A[i * dim + j] /= norm;

		/* Orthogonalize remaining columns against column j */
		for (Dimension k = j + 1; k < dim; k++)
		{
			/* Compute dot product of columns j and k */
			float dot = 0.0f;
			for (Dimension i = 0; i < dim; i++)
				dot += A[i * dim + j] * A[i * dim + k];

			/* Subtract projection: col_k = col_k - dot * col_j */
			for (Dimension i = 0; i < dim; i++)
				A[i * dim + k] -= dot * A[i * dim + j];
		}
	}
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
mkt_random_orthogonal_matrix(float *matrix, Dimension dim, uint64_t seed)
{
	if (matrix == NULL || dim == 0)
		return -1;

	/* Fill with Gaussian random values */
	fill_gaussian_matrix(matrix, dim, seed);

	/* QR decomposition to get orthogonal matrix */
	gram_schmidt_qr(matrix, dim);

	return 0;
}

/*
 * Verify matrix is approximately orthogonal (for testing).
 *
 * Checks that M * M^T ≈ I within tolerance.
 *
 * Returns 1 if orthogonal, 0 if not.
 */
int
mkt_matrix_is_orthogonal(const float *matrix, Dimension dim, float tolerance)
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
					mkt_dot_product(matrix + i * dim, matrix + j * dim, dim);

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
mkt_matrix_transpose_vector_mul(
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
mkt_matrix_vector_mul(
		const float *M, const float *v, float *result, Dimension dim)
{
	for (Dimension i = 0; i < dim; i++)
		result[i] = mkt_dot_product(M + i * dim, v, dim);
}

/*
 * Batched matrix-vector multiplication: results = M^T * vectors
 *
 * Computes M^T * v for multiple vectors at once. This is more efficient
 * than calling mkt_matrix_transpose_vector_mul() repeatedly because:
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
MKT_TARGET_CLONES static void
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

#ifdef MKT_HAVE_CBLAS
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
#endif /* MKT_HAVE_CBLAS */

/*
 * Runtime selection of matrix multiplication implementation.
 * Default: use CBLAS if available, else built-in.
 */
static bool g_use_cblas = true;

void
mkt_matrix_set_use_cblas(bool use_cblas)
{
	g_use_cblas = use_cblas;
}

bool
mkt_matrix_get_use_cblas(void)
{
#ifdef MKT_HAVE_CBLAS
	return g_use_cblas;
#else
	return false;
#endif
}

const char *
mkt_matrix_impl_name(void)
{
#ifdef MKT_HAVE_CBLAS
	return g_use_cblas ? "cblas" : "builtin";
#else
	return "builtin";
#endif
}

void
mkt_matrix_transpose_vector_mul_batch(
		const float *M,
		const float *vectors,
		float		*results,
		uint32_t	 count,
		Dimension	 dim)
{
#ifdef MKT_HAVE_CBLAS
	if (g_use_cblas)
	{
		matrix_transpose_mul_batch_cblas(M, vectors, results, count, dim);
		return;
	}
#endif
	matrix_transpose_mul_batch_builtin(M, vectors, results, count, dim);
}
