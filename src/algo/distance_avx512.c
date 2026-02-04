/*
 * distance_avx512.c - AVX-512 SIMD distance implementations
 *
 * Explicit SIMD implementations using AVX-512 intrinsics for x86-64 CPUs
 * with AVX-512F support. Processes 16 floats per iteration (512 bits).
 *
 * Compiled with -mavx512f -mavx512dq flags.
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
Distance
mkt_distance_l2_avx512(VectorRef a, VectorRef b)
{
	if (mkt_unlikely(
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
	float sum = mkt_horizontal_sum_avx512(sum_vec);

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
Distance
mkt_distance_ip_avx512(VectorRef a, VectorRef b)
{
	if (mkt_unlikely(
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

	float dot = mkt_horizontal_sum_avx512(dot_vec);

	for (; i < dim; i++)
		dot += pa[i] * pb[i];

	return -dot; /* Negate for distance metric */
}

/*
 * Cosine distance using AVX-512.
 *
 * Computes three parallel reductions: dot product, norm_a, norm_b.
 */
Distance
mkt_distance_cosine_avx512(VectorRef a, VectorRef b)
{
	if (mkt_unlikely(
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

	float dot	 = mkt_horizontal_sum_avx512(dot_vec);
	float norm_a = mkt_horizontal_sum_avx512(norm_a_vec);
	float norm_b = mkt_horizontal_sum_avx512(norm_b_vec);

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
 * Batch L2 distance with prefetching.
 */
int
mkt_distance_batch_l2_avx512(
		VectorRef	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances)
{
	if (mkt_unlikely(
				query.dim != dim || query.data == NULL || vectors == NULL ||
				distances == NULL))
		return -1;

	const float *q = query.data;

	for (uint32_t v = 0; v < count; v++)
	{
		const float *vec = vectors + v * dim;

		/* Prefetch 2 vectors ahead */
		if (v + MKT_PREFETCH_DISTANCE < count)
			mkt_prefetch_read(vectors + (v + MKT_PREFETCH_DISTANCE) * dim);

		__m512 sum_vec = _mm512_setzero_ps();

		Dimension i = 0;
		for (; i + 16 <= dim; i += 16)
		{
			__m512 vq	= _mm512_loadu_ps(q + i);
			__m512 vv	= _mm512_loadu_ps(vec + i);
			__m512 diff = _mm512_sub_ps(vq, vv);
			sum_vec		= _mm512_fmadd_ps(diff, diff, sum_vec);
		}

		float sum = mkt_horizontal_sum_avx512(sum_vec);

		for (; i < dim; i++)
		{
			float diff = q[i] - vec[i];
			sum += diff * diff;
		}

		distances[v] = sum;
	}

	return 0;
}

/*
 * Batch inner product with prefetching.
 */
int
mkt_distance_batch_ip_avx512(
		VectorRef	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances)
{
	if (mkt_unlikely(
				query.dim != dim || query.data == NULL || vectors == NULL ||
				distances == NULL))
		return -1;

	const float *q = query.data;

	for (uint32_t v = 0; v < count; v++)
	{
		const float *vec = vectors + v * dim;

		if (v + MKT_PREFETCH_DISTANCE < count)
			mkt_prefetch_read(vectors + (v + MKT_PREFETCH_DISTANCE) * dim);

		__m512 dot_vec = _mm512_setzero_ps();

		Dimension i = 0;
		for (; i + 16 <= dim; i += 16)
		{
			__m512 vq = _mm512_loadu_ps(q + i);
			__m512 vv = _mm512_loadu_ps(vec + i);
			dot_vec	  = _mm512_fmadd_ps(vq, vv, dot_vec);
		}

		float dot = mkt_horizontal_sum_avx512(dot_vec);

		for (; i < dim; i++)
			dot += q[i] * vec[i];

		distances[v] = -dot;
	}

	return 0;
}

/*
 * Batch cosine distance with prefetching.
 */
int
mkt_distance_batch_cosine_avx512(
		VectorRef	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances)
{
	if (mkt_unlikely(
				query.dim != dim || query.data == NULL || vectors == NULL ||
				distances == NULL))
		return -1;

	const float *q = query.data;

	/* Pre-compute query norm (reused for all vectors) */
	__m512	  norm_q_vec = _mm512_setzero_ps();
	Dimension i			 = 0;
	for (; i + 16 <= dim; i += 16)
	{
		__m512 vq  = _mm512_loadu_ps(q + i);
		norm_q_vec = _mm512_fmadd_ps(vq, vq, norm_q_vec);
	}
	float norm_q = mkt_horizontal_sum_avx512(norm_q_vec);
	for (; i < dim; i++)
		norm_q += q[i] * q[i];

	float sqrt_norm_q = sqrtf(norm_q);

	for (uint32_t v = 0; v < count; v++)
	{
		const float *vec = vectors + v * dim;

		if (v + MKT_PREFETCH_DISTANCE < count)
			mkt_prefetch_read(vectors + (v + MKT_PREFETCH_DISTANCE) * dim);

		__m512 dot_vec	  = _mm512_setzero_ps();
		__m512 norm_v_vec = _mm512_setzero_ps();

		i = 0;
		for (; i + 16 <= dim; i += 16)
		{
			__m512 vq  = _mm512_loadu_ps(q + i);
			__m512 vv  = _mm512_loadu_ps(vec + i);
			dot_vec	   = _mm512_fmadd_ps(vq, vv, dot_vec);
			norm_v_vec = _mm512_fmadd_ps(vv, vv, norm_v_vec);
		}

		float dot	 = mkt_horizontal_sum_avx512(dot_vec);
		float norm_v = mkt_horizontal_sum_avx512(norm_v_vec);

		for (; i < dim; i++)
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
