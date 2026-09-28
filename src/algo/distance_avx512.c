/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * distance_avx512.c - AVX-512 SIMD distance implementations
 *
 * Explicit SIMD implementations using AVX-512 intrinsics for x86-64 CPUs
 * with AVX-512F support. Processes 16 floats per iteration (512 bits).
 *
 * Uses per-function target attributes instead of compiler flags to enable
 * AVX-512 code generation. This allows the file to be compiled as part of
 * the main build without requiring separate library compilation.
 */

#include <immintrin.h>
#include <math.h>

#include "algo/distance.h"
#include "algo/simd_utils.h"
#include "core/platform.h"

/*
 * L2 squared distance using AVX-512.
 *
 * Processes 16 floats per iteration using FMA (fused multiply-add).
 * Remaining elements handled with scalar tail loop.
 */
VS_TARGET_AVX512 Distance
vs_distance_l2_avx512(Vec32Ref a, Vec32Ref b)
{
	if (vs_unlikely(
				a.dim != b.dim || a.dim == 0 || a.data == NULL ||
				b.data == NULL))
		return -1.0f;

	const float *pa	 = a.data;
	const float *pb	 = b.data;
	Dimension	 dim = a.dim;

	__m512 sum_vec = _mm512_setzero_ps();

	/* Main loop: 16 floats per iteration */
	Dimension i = 0;
	for (; i + 16 <= dim; i += 16)
	{
		__m512 va	= _mm512_loadu_ps(pa + i);
		__m512 vb	= _mm512_loadu_ps(pb + i);
		__m512 diff = _mm512_sub_ps(va, vb);
		sum_vec		= _mm512_fmadd_ps(diff, diff, sum_vec);
	}

	/* Horizontal reduction */
	float sum = vs_horizontal_sum_avx512(sum_vec);

	/* Scalar tail for remaining elements */
	for (; i < dim; i++)
	{
		float diff = pa[i] - pb[i];
		sum += diff * diff;
	}

	return sum;
}

/*
 * Negative inner product using AVX-512.
 */
VS_TARGET_AVX512 Distance
vs_distance_ip_avx512(Vec32Ref a, Vec32Ref b)
{
	if (vs_unlikely(
				a.dim != b.dim || a.dim == 0 || a.data == NULL ||
				b.data == NULL))
		return -1.0f;

	const float *pa	 = a.data;
	const float *pb	 = b.data;
	Dimension	 dim = a.dim;

	__m512 dot_vec = _mm512_setzero_ps();

	Dimension i = 0;
	for (; i + 16 <= dim; i += 16)
	{
		__m512 va = _mm512_loadu_ps(pa + i);
		__m512 vb = _mm512_loadu_ps(pb + i);
		dot_vec	  = _mm512_fmadd_ps(va, vb, dot_vec);
	}

	float dot = vs_horizontal_sum_avx512(dot_vec);

	for (; i < dim; i++)
		dot += pa[i] * pb[i];

	return -dot; /* Negate for distance metric */
}

/*
 * Cosine distance using AVX-512.
 *
 * Computes three parallel reductions: dot product, norm_a, norm_b.
 */
VS_TARGET_AVX512 Distance
vs_distance_cosine_avx512(Vec32Ref a, Vec32Ref b)
{
	if (vs_unlikely(
				a.dim != b.dim || a.dim == 0 || a.data == NULL ||
				b.data == NULL))
		return -1.0f;

	const float *pa	 = a.data;
	const float *pb	 = b.data;
	Dimension	 dim = a.dim;

	__m512 dot_vec	  = _mm512_setzero_ps();
	__m512 norm_a_vec = _mm512_setzero_ps();
	__m512 norm_b_vec = _mm512_setzero_ps();

	Dimension i = 0;
	for (; i + 16 <= dim; i += 16)
	{
		__m512 va  = _mm512_loadu_ps(pa + i);
		__m512 vb  = _mm512_loadu_ps(pb + i);
		dot_vec	   = _mm512_fmadd_ps(va, vb, dot_vec);
		norm_a_vec = _mm512_fmadd_ps(va, va, norm_a_vec);
		norm_b_vec = _mm512_fmadd_ps(vb, vb, norm_b_vec);
	}

	float dot	 = vs_horizontal_sum_avx512(dot_vec);
	float norm_a = vs_horizontal_sum_avx512(norm_a_vec);
	float norm_b = vs_horizontal_sum_avx512(norm_b_vec);

	/* Scalar tail */
	for (; i < dim; i++)
	{
		float va = pa[i];
		float vb = pb[i];
		dot += va * vb;
		norm_a += va * va;
		norm_b += vb * vb;
	}

	float denom = sqrtf(norm_a) * sqrtf(norm_b);
	if (denom < 1e-8f)
		return 1.0f; /* Maximum distance for zero vectors */

	return 1.0f - (dot / denom);
}

/*
 * Batch L2 distance - simple single-accumulator loop.
 */
VS_TARGET_AVX512 int
vs_distance_batch_l2_avx512(
		Vec32Ref	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances)
{
	if (vs_unlikely(
				query.dim != dim || query.data == NULL || vectors == NULL ||
				distances == NULL))
		return -1;

	const float	  *q	 = query.data;
	const uint32_t dim32 = dim; /* Use 32-bit for efficient loop codegen */

	/* Precompute loop bounds */
	const size_t unrolled_bytes = (dim32 / 32) *
								  128; /* 32 floats = 128 bytes */
	const uint32_t unrolled_elems = (dim32 / 32) * 32;

	for (uint32_t v = 0; v < count; v++)
	{
		const float *vec = vectors + v * dim32;

		__m512 sum0 = _mm512_setzero_ps();
		__m512 sum1 = _mm512_setzero_ps();

		/* 2x unrolled loop: 32 floats per iteration */
		size_t offset = 0;
		while (offset < unrolled_bytes)
		{
			__m512 vq0 = _mm512_loadu_ps(
					(const float *)((const char *)q + offset));
			__m512 vv0 = _mm512_loadu_ps(
					(const float *)((const char *)vec + offset));
			__m512 diff0 = _mm512_sub_ps(vq0, vv0);
			sum0		 = _mm512_fmadd_ps(diff0, diff0, sum0);

			__m512 vq1 = _mm512_loadu_ps(
					(const float *)((const char *)q + offset + 64));
			__m512 vv1 = _mm512_loadu_ps(
					(const float *)((const char *)vec + offset + 64));
			__m512 diff1 = _mm512_sub_ps(vq1, vv1);
			sum1		 = _mm512_fmadd_ps(diff1, diff1, sum1);

			offset += 128;
		}

		/* Combine accumulators */
		__m512 sum_vec = _mm512_add_ps(sum0, sum1);

		/* Handle remaining 16-float chunk if dim not divisible by 32 */
		if (unrolled_elems + 16 <= dim32)
		{
			__m512 vq = _mm512_loadu_ps(
					(const float *)((const char *)q + offset));
			__m512 vv = _mm512_loadu_ps(
					(const float *)((const char *)vec + offset));
			__m512 diff = _mm512_sub_ps(vq, vv);
			sum_vec		= _mm512_fmadd_ps(diff, diff, sum_vec);
		}

		float sum = vs_horizontal_sum_avx512(sum_vec);

		/* Scalar tail */
		uint32_t tail_start = (dim32 / 16) * 16;
		for (uint32_t i = tail_start; i < dim32; i++)
		{
			float diff = q[i] - vec[i];
			sum += diff * diff;
		}

		distances[v] = sum;
	}

	return 0;
}

/*
 * Batch inner product - 2x unrolled loop.
 */
VS_TARGET_AVX512 int
vs_distance_batch_ip_avx512(
		Vec32Ref	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances)
{
	if (vs_unlikely(
				query.dim != dim || query.data == NULL || vectors == NULL ||
				distances == NULL))
		return -1;

	const float	  *q	 = query.data;
	const uint32_t dim32 = dim; /* Use 32-bit for efficient loop codegen */

	/* Precompute loop bounds */
	const size_t unrolled_bytes = (dim32 / 32) *
								  128; /* 32 floats = 128 bytes */
	const uint32_t unrolled_elems = (dim32 / 32) * 32;

	for (uint32_t v = 0; v < count; v++)
	{
		const float *vec = vectors + v * dim32;

		__m512 dot0 = _mm512_setzero_ps();
		__m512 dot1 = _mm512_setzero_ps();

		/* 2x unrolled loop: 32 floats per iteration */
		size_t offset = 0;
		while (offset < unrolled_bytes)
		{
			__m512 vq0 = _mm512_loadu_ps(
					(const float *)((const char *)q + offset));
			__m512 vv0 = _mm512_loadu_ps(
					(const float *)((const char *)vec + offset));
			dot0 = _mm512_fmadd_ps(vq0, vv0, dot0);

			__m512 vq1 = _mm512_loadu_ps(
					(const float *)((const char *)q + offset + 64));
			__m512 vv1 = _mm512_loadu_ps(
					(const float *)((const char *)vec + offset + 64));
			dot1 = _mm512_fmadd_ps(vq1, vv1, dot1);

			offset += 128;
		}

		/* Combine accumulators */
		__m512 dot_vec = _mm512_add_ps(dot0, dot1);

		/* Handle remaining 16-float chunk if dim not divisible by 32 */
		if (unrolled_elems + 16 <= dim32)
		{
			__m512 vq = _mm512_loadu_ps(
					(const float *)((const char *)q + offset));
			__m512 vv = _mm512_loadu_ps(
					(const float *)((const char *)vec + offset));
			dot_vec = _mm512_fmadd_ps(vq, vv, dot_vec);
		}

		float dot = vs_horizontal_sum_avx512(dot_vec);

		/* Scalar tail */
		uint32_t tail_start = (dim32 / 16) * 16;
		for (uint32_t i = tail_start; i < dim32; i++)
			dot += q[i] * vec[i];

		distances[v] = -dot;
	}

	return 0;
}

/*
 * Batch cosine distance with 4 accumulators.
 */
VS_TARGET_AVX512 int
vs_distance_batch_cosine_avx512(
		Vec32Ref	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances)
{
	if (vs_unlikely(
				query.dim != dim || query.data == NULL || vectors == NULL ||
				distances == NULL))
		return -1;

	const float	  *q	 = query.data;
	const uint32_t dim32 = dim; /* Use 32-bit for efficient loop codegen */

	/* Pre-compute query norm (reused for all vectors) */
	__m512	 norm_q_vec = _mm512_setzero_ps();
	uint32_t i			= 0;
	for (; i + 16 <= dim32; i += 16)
	{
		__m512 vq  = _mm512_loadu_ps(q + i);
		norm_q_vec = _mm512_fmadd_ps(vq, vq, norm_q_vec);
	}
	float norm_q = vs_horizontal_sum_avx512(norm_q_vec);
	for (; i < dim32; i++)
		norm_q += q[i] * q[i];

	float sqrt_norm_q = sqrtf(norm_q);

	for (uint32_t v = 0; v < count; v++)
	{
		const float *vec = vectors + v * dim32;

		if (v + VS_PREFETCH_DISTANCE < count)
			vs_prefetch_read(vectors + (v + VS_PREFETCH_DISTANCE) * dim32);

		/* Use 4 accumulators to hide FMA latency */
		__m512 dot0	   = _mm512_setzero_ps();
		__m512 dot1	   = _mm512_setzero_ps();
		__m512 dot2	   = _mm512_setzero_ps();
		__m512 dot3	   = _mm512_setzero_ps();
		__m512 norm_v0 = _mm512_setzero_ps();
		__m512 norm_v1 = _mm512_setzero_ps();
		__m512 norm_v2 = _mm512_setzero_ps();
		__m512 norm_v3 = _mm512_setzero_ps();

		i = 0;

		/* Process 64 floats per iteration with 4 accumulators */
		for (; i + 64 <= dim32; i += 64)
		{
			__m512 vq0 = _mm512_loadu_ps(q + i);
			__m512 vv0 = _mm512_loadu_ps(vec + i);
			dot0	   = _mm512_fmadd_ps(vq0, vv0, dot0);
			norm_v0	   = _mm512_fmadd_ps(vv0, vv0, norm_v0);

			__m512 vq1 = _mm512_loadu_ps(q + i + 16);
			__m512 vv1 = _mm512_loadu_ps(vec + i + 16);
			dot1	   = _mm512_fmadd_ps(vq1, vv1, dot1);
			norm_v1	   = _mm512_fmadd_ps(vv1, vv1, norm_v1);

			__m512 vq2 = _mm512_loadu_ps(q + i + 32);
			__m512 vv2 = _mm512_loadu_ps(vec + i + 32);
			dot2	   = _mm512_fmadd_ps(vq2, vv2, dot2);
			norm_v2	   = _mm512_fmadd_ps(vv2, vv2, norm_v2);

			__m512 vq3 = _mm512_loadu_ps(q + i + 48);
			__m512 vv3 = _mm512_loadu_ps(vec + i + 48);
			dot3	   = _mm512_fmadd_ps(vq3, vv3, dot3);
			norm_v3	   = _mm512_fmadd_ps(vv3, vv3, norm_v3);
		}

		/* Combine accumulators */
		__m512 dot_vec = _mm512_add_ps(
				_mm512_add_ps(dot0, dot1), _mm512_add_ps(dot2, dot3));
		__m512 norm_v_vec = _mm512_add_ps(
				_mm512_add_ps(norm_v0, norm_v1),
				_mm512_add_ps(norm_v2, norm_v3));

		/* Handle remaining 16-float chunks */
		for (; i + 16 <= dim32; i += 16)
		{
			__m512 vq  = _mm512_loadu_ps(q + i);
			__m512 vv  = _mm512_loadu_ps(vec + i);
			dot_vec	   = _mm512_fmadd_ps(vq, vv, dot_vec);
			norm_v_vec = _mm512_fmadd_ps(vv, vv, norm_v_vec);
		}

		float dot	 = vs_horizontal_sum_avx512(dot_vec);
		float norm_v = vs_horizontal_sum_avx512(norm_v_vec);

		/* Scalar tail */
		for (; i < dim32; i++)
		{
			dot += q[i] * vec[i];
			norm_v += vec[i] * vec[i];
		}

		float denom = sqrt_norm_q * sqrtf(norm_v);
		if (denom < 1e-8f)
			distances[v] = 1.0f;
		else
			distances[v] = 1.0f - (dot / denom);
	}

	return 0;
}
