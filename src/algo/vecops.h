/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * vecops.h - General vector operations with SIMD dispatch
 *
 * Provides optimized implementations of common vector operations used
 * throughout the codebase (quantization, clustering, distance computation).
 */

#ifndef VS_VECOPS_H
#define VS_VECOPS_H

#include "core/types.h"

/*
 * Dot product: sum(a[i] * b[i])
 *
 * Returns the inner product of two vectors.
 */
float vs_dot_product(const float *a, const float *b, Dimension dim);

/*
 * L2 norm squared: sum(v[i]^2)
 *
 * Returns ||v||^2. Use vs_l2_norm() if you need the actual norm.
 */
float vs_l2_norm_squared(const float *v, Dimension dim);

/*
 * L2 norm: sqrt(sum(v[i]^2))
 *
 * Returns ||v||. Equivalent to sqrt(vs_l2_norm_squared(v, dim)).
 */
float vs_l2_norm(const float *v, Dimension dim);

/*
 * Vector sum: sum(v[i])
 *
 * Returns the sum of all elements.
 */
float vec32_sum(const float *v, Dimension dim);

/*
 * Vector subtraction: out[i] = a[i] - b[i]
 *
 * Computes element-wise difference. Output may alias input.
 */
void vec32_sub(const float *a, const float *b, float *out, Dimension dim);

/*
 * Vector addition: out[i] = a[i] + b[i]
 *
 * Computes element-wise sum. Output may alias input.
 */
void vec32_add(const float *a, const float *b, float *out, Dimension dim);

/*
 * Vector scale: out[i] = v[i] * scalar
 *
 * Multiplies each element by a scalar. Output may alias input.
 */
void vec32_scale(const float *v, float scalar, float *out, Dimension dim);

/*
 * Normalize a raw float vector to unit L2 length, in place. No-op for a
 * zero vector. (vec32_normalize in vec32.h is the Vec32 form.)
 */
static inline void
vs_l2_normalize(float *v, Dimension dim)
{
	float norm = vs_l2_norm(v, dim);
	if (norm > 0.0f)
		vec32_scale(v, 1.0f / norm, v, dim);
}

/*
 * Vector mean: out[d] = mean of vectors[i][d] for all i
 *
 * Computes the element-wise mean of nvecs row-major vectors.
 * Output must be preallocated with at least dim floats.
 */
void
vec32_mean(const float *vectors, uint32_t nvecs, Dimension dim, float *out);

/*
 * Global mean for index metadata: the element-wise mean of the (leaf)
 * centroids, L2-normalized when the metric is cosine so it lives on the
 * unit sphere with the vectors it recenters. Every writer of the
 * metapage's global mean must use this so the cosine normalization
 * cannot be missed at one site.
 */
void vs_global_mean(
		const float	  *centroids,
		uint32_t	   ncentroids,
		Dimension	   dim,
		DistanceMetric metric,
		float		  *out);

/*
 * L2 distance squared: sum((a[i] - b[i])^2)
 *
 * Returns ||a - b||^2. Equivalent to vs_l2_norm_squared of the difference,
 * but without needing a temporary buffer.
 */
float vs_l2_distance_squared(const float *a, const float *b, Dimension dim);

/*
 * Initialize SIMD dispatch for vector operations.
 * Called automatically on first use, but can be called explicitly
 * for deterministic initialization timing.
 */
int vs_vecops_init(void);

/*
 * Get name of the active SIMD implementation.
 * Returns one of: "avx512", "avx2", "neon", "compiler"
 */
const char *vs_vecops_impl_name(void);

/*
 * Force re-initialization of SIMD dispatch (for testing).
 */
void vs_vecops_force_reinit(void);

#endif /* VS_VECOPS_H */
