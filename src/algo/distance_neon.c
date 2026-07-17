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
 *
 * Four independent accumulators break the loop-carried dependency
 * chain: a single `sum_vec = vfmaq_f32(sum_vec, ...)` chain forces
 * each FMA to wait for the previous one's ~4-cycle latency, capping
 * throughput well below Neoverse V2's 2 FMA/cycle issue rate. Four
 * accumulators let independent iterations overlap in the OOO
 * scheduler; only the final horizontal reduction combines them.
 */
Distance
mkt_distance_l2_neon(VectorRef a, VectorRef b)
{
	if (mkt_unlikely(
				a.dim != b.dim || a.dim == 0 || a.data == NULL ||
				b.data == NULL))
		return -1.0f;

	const float *pa	 = a.data;
	const float *pb	 = b.data;
	Dimension	 dim = a.dim;

	float32x4_t sum0 = vdupq_n_f32(0.0f);
	float32x4_t sum1 = vdupq_n_f32(0.0f);
	float32x4_t sum2 = vdupq_n_f32(0.0f);
	float32x4_t sum3 = vdupq_n_f32(0.0f);

	/* Main loop: 16 floats (4 independent lanes of 4) per iteration */
	Dimension i = 0;
	for (; i + 16 <= dim; i += 16)
	{
		float32x4_t d0 = vsubq_f32(vld1q_f32(pa + i), vld1q_f32(pb + i));
		float32x4_t d1 =
				vsubq_f32(vld1q_f32(pa + i + 4), vld1q_f32(pb + i + 4));
		float32x4_t d2 =
				vsubq_f32(vld1q_f32(pa + i + 8), vld1q_f32(pb + i + 8));
		float32x4_t d3 =
				vsubq_f32(vld1q_f32(pa + i + 12), vld1q_f32(pb + i + 12));
		sum0 = vfmaq_f32(sum0, d0, d0);
		sum1 = vfmaq_f32(sum1, d1, d1);
		sum2 = vfmaq_f32(sum2, d2, d2);
		sum3 = vfmaq_f32(sum3, d3, d3);
	}
	for (; i + 4 <= dim; i += 4)
	{
		float32x4_t d0 = vsubq_f32(vld1q_f32(pa + i), vld1q_f32(pb + i));
		sum0		   = vfmaq_f32(sum0, d0, d0);
	}

	/* Horizontal reduction */
	float sum = mkt_horizontal_sum_neon(vaddq_f32(
			vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3)));

	/* Scalar tail */
	for (; i < dim; i++)
	{
		float diff = pa[i] - pb[i];
		sum += diff * diff;
	}

	return sum;
}

/*
 * Negative inner product using NEON. See mkt_distance_l2_neon for
 * why four independent accumulators beat one.
 */
Distance
mkt_distance_ip_neon(VectorRef a, VectorRef b)
{
	if (mkt_unlikely(
				a.dim != b.dim || a.dim == 0 || a.data == NULL ||
				b.data == NULL))
		return -1.0f;

	const float *pa	 = a.data;
	const float *pb	 = b.data;
	Dimension	 dim = a.dim;

	float32x4_t dot0 = vdupq_n_f32(0.0f);
	float32x4_t dot1 = vdupq_n_f32(0.0f);
	float32x4_t dot2 = vdupq_n_f32(0.0f);
	float32x4_t dot3 = vdupq_n_f32(0.0f);

	Dimension i = 0;
	for (; i + 16 <= dim; i += 16)
	{
		dot0 = vfmaq_f32(dot0, vld1q_f32(pa + i), vld1q_f32(pb + i));
		dot1 = vfmaq_f32(
				dot1, vld1q_f32(pa + i + 4), vld1q_f32(pb + i + 4));
		dot2 = vfmaq_f32(
				dot2, vld1q_f32(pa + i + 8), vld1q_f32(pb + i + 8));
		dot3 = vfmaq_f32(
				dot3, vld1q_f32(pa + i + 12), vld1q_f32(pb + i + 12));
	}
	for (; i + 4 <= dim; i += 4)
		dot0 = vfmaq_f32(dot0, vld1q_f32(pa + i), vld1q_f32(pb + i));

	float dot = mkt_horizontal_sum_neon(vaddq_f32(
			vaddq_f32(dot0, dot1), vaddq_f32(dot2, dot3)));

	for (; i < dim; i++)
		dot += pa[i] * pb[i];

	return -dot;
}

/*
 * Cosine distance using NEON. Each of the three quantities (dot,
 * norm_a, norm_b) gets two independent accumulators (see
 * mkt_distance_l2_neon) so the FMA dependency chain per quantity is
 * half as long.
 */
Distance
mkt_distance_cosine_neon(VectorRef a, VectorRef b)
{
	if (mkt_unlikely(
				a.dim != b.dim || a.dim == 0 || a.data == NULL ||
				b.data == NULL))
		return -1.0f;

	const float *pa	 = a.data;
	const float *pb	 = b.data;
	Dimension	 dim = a.dim;

	float32x4_t dot0 = vdupq_n_f32(0.0f), dot1 = vdupq_n_f32(0.0f);
	float32x4_t na0 = vdupq_n_f32(0.0f), na1 = vdupq_n_f32(0.0f);
	float32x4_t nb0 = vdupq_n_f32(0.0f), nb1 = vdupq_n_f32(0.0f);

	Dimension i = 0;
	for (; i + 8 <= dim; i += 8)
	{
		float32x4_t va0 = vld1q_f32(pa + i);
		float32x4_t vb0 = vld1q_f32(pb + i);
		float32x4_t va1 = vld1q_f32(pa + i + 4);
		float32x4_t vb1 = vld1q_f32(pb + i + 4);

		dot0 = vfmaq_f32(dot0, va0, vb0);
		na0	 = vfmaq_f32(na0, va0, va0);
		nb0	 = vfmaq_f32(nb0, vb0, vb0);

		dot1 = vfmaq_f32(dot1, va1, vb1);
		na1	 = vfmaq_f32(na1, va1, va1);
		nb1	 = vfmaq_f32(nb1, vb1, vb1);
	}
	for (; i + 4 <= dim; i += 4)
	{
		float32x4_t va0 = vld1q_f32(pa + i);
		float32x4_t vb0 = vld1q_f32(pb + i);
		dot0			= vfmaq_f32(dot0, va0, vb0);
		na0				= vfmaq_f32(na0, va0, va0);
		nb0				= vfmaq_f32(nb0, vb0, vb0);
	}

	float dot	 = mkt_horizontal_sum_neon(vaddq_f32(dot0, dot1));
	float norm_a = mkt_horizontal_sum_neon(vaddq_f32(na0, na1));
	float norm_b = mkt_horizontal_sum_neon(vaddq_f32(nb0, nb1));

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
