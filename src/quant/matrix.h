/*
 * matrix.h - Matrix operations for vector quantization
 *
 * Provides matrix-vector multiplication and random orthogonal matrix
 * generation for RaBitQ and other quantization methods.
 */

#ifndef VS_MATRIX_H
#define VS_MATRIX_H

#include <stdint.h>

#include "core/types.h"

/*
 * Generate random orthogonal matrix using QR decomposition.
 *
 * Creates a Gaussian random matrix and orthogonalizes it via QR.
 * The resulting Q matrix is uniformly distributed over O(n).
 *
 * Parameters:
 *   matrix: Output buffer (dim * dim floats, row-major)
 *   dim:    Matrix dimension
 *   seed:   Random seed for reproducibility
 *
 * Returns 0 on success, -1 on failure.
 */
int vs_random_orthogonal_matrix(float *matrix, Dimension dim, uint64_t seed);

/*
 * Verify matrix is approximately orthogonal (for testing).
 *
 * Checks that M * M^T ≈ I within tolerance.
 *
 * Returns 1 if orthogonal, 0 if not.
 */
int
vs_matrix_is_orthogonal(const float *matrix, Dimension dim, float tolerance);

/*
 * Matrix-vector multiplication: result = M * v
 *
 * Parameters:
 *   M:      Input matrix (dim x dim, row-major)
 *   v:      Input vector (dim elements)
 *   result: Output vector (dim elements)
 *   dim:    Dimension
 */
void vs_matrix_vector_mul(
		const float *M, const float *v, float *result, Dimension dim);

/*
 * Matrix-vector multiplication: result = M^T * v
 *
 * Computes the product of the transpose of M with vector v.
 *
 * Parameters:
 *   M:      Input matrix (dim x dim, row-major)
 *   v:      Input vector (dim elements)
 *   result: Output vector (dim elements)
 *   dim:    Dimension
 */
void vs_matrix_transpose_vector_mul(
		const float *M, const float *v, float *result, Dimension dim);

/*
 * Batched matrix-vector multiplication: results = M^T * vectors
 *
 * Computes M^T * v for multiple vectors at once. More efficient than
 * calling vs_matrix_transpose_vector_mul() repeatedly because:
 * 1. Matrix M is loaded into cache once and reused for all vectors
 * 2. Inner loop processes contiguous memory (good for SIMD)
 *
 * Memory layout:
 *   vectors: count vectors, each dim elements, contiguous (vectors[i*dim + j])
 *   results: count output vectors, same layout
 *
 * Parameters:
 *   M:       Input matrix (dim x dim, row-major)
 *   vectors: Input vectors (count * dim elements)
 *   results: Output vectors (count * dim elements)
 *   count:   Number of vectors to process
 *   dim:     Vector/matrix dimension
 */
void vs_matrix_transpose_vector_mul_batch(
		const float *M,
		const float *vectors,
		float		*results,
		uint32_t	 count,
		Dimension	 dim);

/*
 * CBLAS runtime control for benchmarking.
 *
 * When CBLAS is available, these functions allow switching between
 * the CBLAS implementation and the built-in implementation at runtime.
 * This is useful for comparing performance in benchmarks.
 */

/* Set whether to use CBLAS for batch matrix operations (default: true) */
void vs_matrix_set_use_cblas(bool use_cblas);

/* Get current CBLAS usage setting (false if CBLAS not available) */
bool vs_matrix_get_use_cblas(void);

/* Get name of current implementation ("cblas" or "builtin") */
const char *vs_matrix_impl_name(void);

#endif /* VS_MATRIX_H */
