/*
 * vecops.h - General vector operations with SIMD dispatch
 *
 * Provides optimized implementations of common vector operations used
 * throughout the codebase (quantization, clustering, distance computation).
 */

#ifndef MKT_VECOPS_H
#define MKT_VECOPS_H

#include <math.h>

#include "mkt_types.h"

/*
 * Dot product: sum(a[i] * b[i])
 *
 * Returns the inner product of two vectors.
 */
float mkt_dot_product(const float *a, const float *b, Dimension dim);

/*
 * L2 norm squared: sum(v[i]^2)
 *
 * Returns ||v||^2. Use mkt_l2_norm() if you need the actual norm.
 */
float mkt_l2_norm_squared(const float *v, Dimension dim);

/*
 * L2 norm: sqrt(sum(v[i]^2))
 *
 * Returns ||v||. Equivalent to sqrt(mkt_l2_norm_squared(v, dim)).
 */
float mkt_l2_norm(const float *v, Dimension dim);

/*
 * Vector sum: sum(v[i])
 *
 * Returns the sum of all elements.
 */
float mkt_vector_sum(const float *v, Dimension dim);

/*
 * Vector subtraction: out[i] = a[i] - b[i]
 *
 * Computes element-wise difference. Output may alias input.
 */
void mkt_vector_sub(const float *a, const float *b, float *out, Dimension dim);

/*
 * Vector addition: out[i] = a[i] + b[i]
 *
 * Computes element-wise sum. Output may alias input.
 */
void mkt_vector_add(const float *a, const float *b, float *out, Dimension dim);

/*
 * Vector scale: out[i] = v[i] * scalar
 *
 * Multiplies each element by a scalar. Output may alias input.
 */
void mkt_vector_scale(const float *v, float scalar, float *out, Dimension dim);

/*
 * Vector mean: out[d] = mean of vectors[i][d] for all i
 *
 * Computes the element-wise mean of nvecs row-major vectors.
 * Output must be preallocated with at least dim floats.
 */
void mkt_vector_mean(
		const float *vectors, uint32_t nvecs, Dimension dim, float *out);

/*
 * Normalize a vector to unit length in-place: v[i] /= ||v||
 *
 * If the vector has zero norm, it is left unchanged.
 */
static inline void
mkt_normalize(float *v, Dimension dim)
{
	float norm = mkt_l2_norm(v, dim);
	if (norm > 0.0f)
		mkt_vector_scale(v, 1.0f / norm, v, dim);
}

/*
 * L2 distance squared: sum((a[i] - b[i])^2)
 *
 * Returns ||a - b||^2. Equivalent to mkt_l2_norm_squared of the difference,
 * but without needing a temporary buffer.
 */
float mkt_l2_distance_squared(const float *a, const float *b, Dimension dim);

/*
 * Initialize SIMD dispatch for vector operations.
 * Called automatically on first use, but can be called explicitly
 * for deterministic initialization timing.
 */
int mkt_vecops_init(void);

/*
 * Get name of the active SIMD implementation.
 * Returns one of: "avx512", "avx2", "neon", "compiler"
 */
const char *mkt_vecops_impl_name(void);

/*
 * Force re-initialization of SIMD dispatch (for testing).
 */
void mkt_vecops_force_reinit(void);

#endif /* MKT_VECOPS_H */
