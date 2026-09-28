/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * vecops.c - General vector operations with SIMD dispatch
 *
 * Provides compiler-vectorized implementations with target_clones for
 * automatic ISA selection. Hand-optimized SIMD implementations can be
 * added in vecops_avx2.c, vecops_avx512.c, vecops_neon.c if needed.
 */

#include "vs_config.h"

#include <math.h>
#include <stdatomic.h>
#include <stddef.h>
#include <string.h>

#include "algo/simd_utils.h"
#include "algo/vecops.h"
#include "core/platform.h"

/*
 * Compiler-Vectorized Implementations
 *
 * These use target_clones to generate multiple versions for different ISAs.
 * The dynamic linker selects the best version at load time.
 */

VS_TARGET_CLONES static float
vecops_dot_product_compiler(const float *a, const float *b, Dimension dim)
{
	float sum = 0.0f;
	for (Dimension i = 0; i < dim; i++)
		sum += a[i] * b[i];
	return sum;
}

VS_TARGET_CLONES static float
vecops_l2_norm_squared_compiler(const float *v, Dimension dim)
{
	float sum = 0.0f;
	for (Dimension i = 0; i < dim; i++)
		sum += v[i] * v[i];
	return sum;
}

VS_TARGET_CLONES static float
vecops_vector_sum_compiler(const float *v, Dimension dim)
{
	float sum = 0.0f;
	for (Dimension i = 0; i < dim; i++)
		sum += v[i];
	return sum;
}

VS_TARGET_CLONES static void
vecops_vector_sub_compiler(
		const float *a, const float *b, float *out, Dimension dim)
{
	for (Dimension i = 0; i < dim; i++)
		out[i] = a[i] - b[i];
}

VS_TARGET_CLONES static void
vecops_vector_add_compiler(
		const float *a, const float *b, float *out, Dimension dim)
{
	for (Dimension i = 0; i < dim; i++)
		out[i] = a[i] + b[i];
}

VS_TARGET_CLONES static void
vecops_vector_scale_compiler(
		const float *v, float scalar, float *out, Dimension dim)
{
	for (Dimension i = 0; i < dim; i++)
		out[i] = v[i] * scalar;
}

VS_TARGET_CLONES static float
vecops_l2_distance_squared_compiler(
		const float *a, const float *b, Dimension dim)
{
	float sum = 0.0f;
	for (Dimension i = 0; i < dim; i++)
	{
		float diff = a[i] - b[i];
		sum += diff * diff;
	}
	return sum;
}

/*
 * Function Pointer Dispatch
 */

typedef float (*DotProductFn)(const float *, const float *, Dimension);
typedef float (*NormFn)(const float *, Dimension);
typedef float (*SumFn)(const float *, Dimension);
typedef void (*BinaryOpFn)(const float *, const float *, float *, Dimension);
typedef void (*ScaleFn)(const float *, float, float *, Dimension);
typedef float (*DistanceFn)(const float *, const float *, Dimension);

static DotProductFn g_dot_product_fn		 = NULL;
static NormFn		g_l2_norm_squared_fn	 = NULL;
static SumFn		g_vector_sum_fn			 = NULL;
static BinaryOpFn	g_vector_sub_fn			 = NULL;
static BinaryOpFn	g_vector_add_fn			 = NULL;
static ScaleFn		g_vector_scale_fn		 = NULL;
static DistanceFn	g_l2_distance_squared_fn = NULL;

static const char	*g_impl_name   = NULL;
static _Atomic(bool) g_initialized = false;

void
vs_vecops_force_reinit(void)
{
	g_initialized			 = false;
	g_dot_product_fn		 = NULL;
	g_l2_norm_squared_fn	 = NULL;
	g_vector_sum_fn			 = NULL;
	g_vector_sub_fn			 = NULL;
	g_vector_add_fn			 = NULL;
	g_vector_scale_fn		 = NULL;
	g_l2_distance_squared_fn = NULL;
	g_impl_name				 = NULL;
}

int
vs_vecops_init(void)
{
	if (g_initialized)
		return 0;

	/*
	 * For now, use compiler-vectorized implementations for all operations.
	 * Hand-optimized SIMD can be added later if profiling shows benefit.
	 *
	 * The compiler does a good job with these simple loops, especially
	 * with target_clones generating AVX2/AVX-512 versions automatically.
	 */
	g_dot_product_fn		 = vecops_dot_product_compiler;
	g_l2_norm_squared_fn	 = vecops_l2_norm_squared_compiler;
	g_vector_sum_fn			 = vecops_vector_sum_compiler;
	g_vector_sub_fn			 = vecops_vector_sub_compiler;
	g_vector_add_fn			 = vecops_vector_add_compiler;
	g_vector_scale_fn		 = vecops_vector_scale_compiler;
	g_l2_distance_squared_fn = vecops_l2_distance_squared_compiler;
	g_impl_name				 = "compiler";

	g_initialized = true;
	return 0;
}

const char *
vs_vecops_impl_name(void)
{
	if (vs_unlikely(!g_initialized))
		vs_vecops_init();
	return g_impl_name;
}

/*
 * Public API
 */

float
vs_dot_product(const float *a, const float *b, Dimension dim)
{
	if (vs_unlikely(!g_initialized))
		vs_vecops_init();
	return g_dot_product_fn(a, b, dim);
}

float
vs_l2_norm_squared(const float *v, Dimension dim)
{
	if (vs_unlikely(!g_initialized))
		vs_vecops_init();
	return g_l2_norm_squared_fn(v, dim);
}

float
vs_l2_norm(const float *v, Dimension dim)
{
	return sqrtf(vs_l2_norm_squared(v, dim));
}

float
vec32_sum(const float *v, Dimension dim)
{
	if (vs_unlikely(!g_initialized))
		vs_vecops_init();
	return g_vector_sum_fn(v, dim);
}

void
vec32_sub(const float *a, const float *b, float *out, Dimension dim)
{
	if (vs_unlikely(!g_initialized))
		vs_vecops_init();
	g_vector_sub_fn(a, b, out, dim);
}

void
vec32_add(const float *a, const float *b, float *out, Dimension dim)
{
	if (vs_unlikely(!g_initialized))
		vs_vecops_init();
	g_vector_add_fn(a, b, out, dim);
}

void
vec32_scale(const float *v, float scalar, float *out, Dimension dim)
{
	if (vs_unlikely(!g_initialized))
		vs_vecops_init();
	g_vector_scale_fn(v, scalar, out, dim);
}

void
vec32_mean(const float *vectors, uint32_t nvecs, Dimension dim, float *out)
{
	memset(out, 0, dim * sizeof(float));
	for (uint32_t i = 0; i < nvecs; i++)
		vec32_add(out, vectors + (size_t)i * dim, out, dim);
	vec32_scale(out, 1.0f / (float)nvecs, out, dim);
}

void
vs_global_mean(
		const float	  *centroids,
		uint32_t	   ncentroids,
		Dimension	   dim,
		DistanceMetric metric,
		float		  *out)
{
	vec32_mean(centroids, ncentroids, dim, out);
	if (metric == DISTANCE_COSINE)
		vs_l2_normalize(out, dim);
}

float
vs_l2_distance_squared(const float *a, const float *b, Dimension dim)
{
	if (vs_unlikely(!g_initialized))
		vs_vecops_init();
	return g_l2_distance_squared_fn(a, b, dim);
}
