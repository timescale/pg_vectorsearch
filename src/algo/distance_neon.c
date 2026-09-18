/*
 * distance_neon.c - ARM NEON SIMD distance implementations
 *
 * Explicit SIMD implementations using ARM NEON intrinsics.
 * Processes 4 floats per iteration (128 bits).
 *
 * Requires ARMv8.1+ for vaddvq_f32 (horizontal reduction).
 *
 * Note: No target pragmas needed here:
 * - AArch64: NEON is mandatory, always available
 * - ARMv7: Built as separate library with -mfpu=neon flag
 */

#include <arm_neon.h>
#include <math.h>
#include <stddef.h>

#include "algo/distance.h"
#include "algo/simd_utils.h"
#include "core/platform.h"

/*
 * L2 squared distance using NEON.
 */
Distance
mkt_distance_l2_neon(Vec32Ref a, Vec32Ref b)
{
	if (mkt_unlikely(
				a.dim != b.dim || a.dim == 0 || a.data == NULL ||
				b.data == NULL))
		return -1.0f;

	const float *pa	 = a.data;
	const float *pb	 = b.data;
	Dimension	 dim = a.dim;

	float32x4_t sum_vec = vdupq_n_f32(0.0f);

	/* Main loop: 4 floats per iteration */
	Dimension i = 0;
	for (; i + 4 <= dim; i += 4)
	{
		float32x4_t va	 = vld1q_f32(pa + i);
		float32x4_t vb	 = vld1q_f32(pb + i);
		float32x4_t diff = vsubq_f32(va, vb);
		sum_vec			 = vfmaq_f32(sum_vec, diff, diff);
	}

	/* Horizontal reduction */
	float sum = mkt_horizontal_sum_neon(sum_vec);

	/* Scalar tail */
	for (; i < dim; i++)
	{
		float diff = pa[i] - pb[i];
		sum += diff * diff;
	}

	return sum;
}

/*
 * Negative inner product using NEON.
 */
Distance
mkt_distance_ip_neon(Vec32Ref a, Vec32Ref b)
{
	if (mkt_unlikely(
				a.dim != b.dim || a.dim == 0 || a.data == NULL ||
				b.data == NULL))
		return -1.0f;

	const float *pa	 = a.data;
	const float *pb	 = b.data;
	Dimension	 dim = a.dim;

	float32x4_t dot_vec = vdupq_n_f32(0.0f);

	Dimension i = 0;
	for (; i + 4 <= dim; i += 4)
	{
		float32x4_t va = vld1q_f32(pa + i);
		float32x4_t vb = vld1q_f32(pb + i);
		dot_vec		   = vfmaq_f32(dot_vec, va, vb);
	}

	float dot = mkt_horizontal_sum_neon(dot_vec);

	for (; i < dim; i++)
		dot += pa[i] * pb[i];

	return -dot;
}

/*
 * Cosine distance using NEON.
 */
Distance
mkt_distance_cosine_neon(Vec32Ref a, Vec32Ref b)
{
	if (mkt_unlikely(
				a.dim != b.dim || a.dim == 0 || a.data == NULL ||
				b.data == NULL))
		return -1.0f;

	const float *pa	 = a.data;
	const float *pb	 = b.data;
	Dimension	 dim = a.dim;

	float32x4_t dot_vec	   = vdupq_n_f32(0.0f);
	float32x4_t norm_a_vec = vdupq_n_f32(0.0f);
	float32x4_t norm_b_vec = vdupq_n_f32(0.0f);

	Dimension i = 0;
	for (; i + 4 <= dim; i += 4)
	{
		float32x4_t va = vld1q_f32(pa + i);
		float32x4_t vb = vld1q_f32(pb + i);
		dot_vec		   = vfmaq_f32(dot_vec, va, vb);
		norm_a_vec	   = vfmaq_f32(norm_a_vec, va, va);
		norm_b_vec	   = vfmaq_f32(norm_b_vec, vb, vb);
	}

	float dot	 = mkt_horizontal_sum_neon(dot_vec);
	float norm_a = mkt_horizontal_sum_neon(norm_a_vec);
	float norm_b = mkt_horizontal_sum_neon(norm_b_vec);

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
		return 1.0f;

	return 1.0f - (dot / denom);
}

/*
 * Batch L2 distance with prefetching.
 */
int
mkt_distance_batch_l2_neon(
		Vec32Ref	 query,
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

		float32x4_t sum_vec = vdupq_n_f32(0.0f);

		Dimension i = 0;
		for (; i + 4 <= dim; i += 4)
		{
			float32x4_t vq	 = vld1q_f32(q + i);
			float32x4_t vv	 = vld1q_f32(vec + i);
			float32x4_t diff = vsubq_f32(vq, vv);
			sum_vec			 = vfmaq_f32(sum_vec, diff, diff);
		}

		float sum = mkt_horizontal_sum_neon(sum_vec);

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
mkt_distance_batch_ip_neon(
		Vec32Ref	 query,
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

		float32x4_t dot_vec = vdupq_n_f32(0.0f);

		Dimension i = 0;
		for (; i + 4 <= dim; i += 4)
		{
			float32x4_t vq = vld1q_f32(q + i);
			float32x4_t vv = vld1q_f32(vec + i);
			dot_vec		   = vfmaq_f32(dot_vec, vq, vv);
		}

		float dot = mkt_horizontal_sum_neon(dot_vec);

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
mkt_distance_batch_cosine_neon(
		Vec32Ref	 query,
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

	/* Pre-compute query norm */
	float32x4_t norm_q_vec = vdupq_n_f32(0.0f);
	Dimension	i		   = 0;
	for (; i + 4 <= dim; i += 4)
	{
		float32x4_t vq = vld1q_f32(q + i);
		norm_q_vec	   = vfmaq_f32(norm_q_vec, vq, vq);
	}
	float norm_q = mkt_horizontal_sum_neon(norm_q_vec);
	for (; i < dim; i++)
		norm_q += q[i] * q[i];

	float sqrt_norm_q = sqrtf(norm_q);

	for (uint32_t v = 0; v < count; v++)
	{
		const float *vec = vectors + v * dim;

		if (v + MKT_PREFETCH_DISTANCE < count)
			mkt_prefetch_read(vectors + (v + MKT_PREFETCH_DISTANCE) * dim);

		float32x4_t dot_vec	   = vdupq_n_f32(0.0f);
		float32x4_t norm_v_vec = vdupq_n_f32(0.0f);

		i = 0;
		for (; i + 4 <= dim; i += 4)
		{
			float32x4_t vq = vld1q_f32(q + i);
			float32x4_t vv = vld1q_f32(vec + i);
			dot_vec		   = vfmaq_f32(dot_vec, vq, vv);
			norm_v_vec	   = vfmaq_f32(norm_v_vec, vv, vv);
		}

		float dot	 = mkt_horizontal_sum_neon(dot_vec);
		float norm_v = mkt_horizontal_sum_neon(norm_v_vec);

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
