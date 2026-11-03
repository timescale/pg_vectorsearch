/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * distance_neon.c - ARM NEON SIMD distance implementations
 *
 * Explicit SIMD implementations using ARM NEON intrinsics.
 *
 * The inner loops keep four independent accumulators and consume 16
 * floats per iteration. A single accumulator would serialize on FMA
 * latency (~4 cycles) and leave most of the FP issue width idle.
 *
 * The L2 and inner-product batch loops go further and score two
 * vectors per pass, so one query load feeds both. Those loops are
 * limited by L1 load throughput, not by FP work.
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

#define VS_NEON_UNROLL 16

/* Reduce four accumulators to a scalar. */
static inline float
neon_hsum4(float32x4_t a, float32x4_t b, float32x4_t c, float32x4_t d)
{
	return vaddvq_f32(vaddq_f32(vaddq_f32(a, b), vaddq_f32(c, d)));
}

/*
 * Squared L2 distance over dim floats.
 */
static inline float
neon_l2_sq(const float *a, const float *b, Dimension dim)
{
	float32x4_t s0 = vdupq_n_f32(0.0f);
	float32x4_t s1 = s0;
	float32x4_t s2 = s0;
	float32x4_t s3 = s0;

	Dimension i = 0;
	for (; i + VS_NEON_UNROLL <= dim; i += VS_NEON_UNROLL)
	{
		float32x4x4_t va = vld1q_f32_x4(a + i);
		float32x4x4_t vb = vld1q_f32_x4(b + i);

		float32x4_t d0 = vsubq_f32(va.val[0], vb.val[0]);
		float32x4_t d1 = vsubq_f32(va.val[1], vb.val[1]);
		float32x4_t d2 = vsubq_f32(va.val[2], vb.val[2]);
		float32x4_t d3 = vsubq_f32(va.val[3], vb.val[3]);

		s0 = vfmaq_f32(s0, d0, d0);
		s1 = vfmaq_f32(s1, d1, d1);
		s2 = vfmaq_f32(s2, d2, d2);
		s3 = vfmaq_f32(s3, d3, d3);
	}

	for (; i + 4 <= dim; i += 4)
	{
		float32x4_t d = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
		s0			  = vfmaq_f32(s0, d, d);
	}

	float sum = neon_hsum4(s0, s1, s2, s3);

	for (; i < dim; i++)
	{
		float d = a[i] - b[i];
		sum += d * d;
	}

	return sum;
}

/*
 * Squared L2 distance of a against two vectors at once.
 *
 * Sharing the query load between both vectors cuts L1 load traffic by a
 * quarter, which is what limits the single-vector loop.
 */
static inline void
neon_l2_sq_x2(
		const float *a,
		const float *b0,
		const float *b1,
		Dimension	 dim,
		float		*out0,
		float		*out1)
{
	float32x4_t p0 = vdupq_n_f32(0.0f);
	float32x4_t p1 = p0, p2 = p0, p3 = p0;
	float32x4_t r0 = p0, r1 = p0, r2 = p0, r3 = p0;

	Dimension i = 0;
	for (; i + VS_NEON_UNROLL <= dim; i += VS_NEON_UNROLL)
	{
		float32x4x4_t va  = vld1q_f32_x4(a + i);
		float32x4x4_t vb0 = vld1q_f32_x4(b0 + i);
		float32x4x4_t vb1 = vld1q_f32_x4(b1 + i);

		float32x4_t d0 = vsubq_f32(va.val[0], vb0.val[0]);
		float32x4_t d1 = vsubq_f32(va.val[1], vb0.val[1]);
		float32x4_t d2 = vsubq_f32(va.val[2], vb0.val[2]);
		float32x4_t d3 = vsubq_f32(va.val[3], vb0.val[3]);

		float32x4_t e0 = vsubq_f32(va.val[0], vb1.val[0]);
		float32x4_t e1 = vsubq_f32(va.val[1], vb1.val[1]);
		float32x4_t e2 = vsubq_f32(va.val[2], vb1.val[2]);
		float32x4_t e3 = vsubq_f32(va.val[3], vb1.val[3]);

		p0 = vfmaq_f32(p0, d0, d0);
		p1 = vfmaq_f32(p1, d1, d1);
		p2 = vfmaq_f32(p2, d2, d2);
		p3 = vfmaq_f32(p3, d3, d3);

		r0 = vfmaq_f32(r0, e0, e0);
		r1 = vfmaq_f32(r1, e1, e1);
		r2 = vfmaq_f32(r2, e2, e2);
		r3 = vfmaq_f32(r3, e3, e3);
	}

	for (; i + 4 <= dim; i += 4)
	{
		float32x4_t va = vld1q_f32(a + i);
		float32x4_t d  = vsubq_f32(va, vld1q_f32(b0 + i));
		float32x4_t e  = vsubq_f32(va, vld1q_f32(b1 + i));
		p0			   = vfmaq_f32(p0, d, d);
		r0			   = vfmaq_f32(r0, e, e);
	}

	float s0 = neon_hsum4(p0, p1, p2, p3);
	float s1 = neon_hsum4(r0, r1, r2, r3);

	for (; i < dim; i++)
	{
		float d = a[i] - b0[i];
		float e = a[i] - b1[i];
		s0 += d * d;
		s1 += e * e;
	}

	*out0 = s0;
	*out1 = s1;
}

/*
 * Dot product over dim floats.
 */
static inline float
neon_dot(const float *a, const float *b, Dimension dim)
{
	float32x4_t s0 = vdupq_n_f32(0.0f);
	float32x4_t s1 = s0;
	float32x4_t s2 = s0;
	float32x4_t s3 = s0;

	Dimension i = 0;
	for (; i + VS_NEON_UNROLL <= dim; i += VS_NEON_UNROLL)
	{
		float32x4x4_t va = vld1q_f32_x4(a + i);
		float32x4x4_t vb = vld1q_f32_x4(b + i);

		s0 = vfmaq_f32(s0, va.val[0], vb.val[0]);
		s1 = vfmaq_f32(s1, va.val[1], vb.val[1]);
		s2 = vfmaq_f32(s2, va.val[2], vb.val[2]);
		s3 = vfmaq_f32(s3, va.val[3], vb.val[3]);
	}

	for (; i + 4 <= dim; i += 4)
		s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));

	float dot = neon_hsum4(s0, s1, s2, s3);

	for (; i < dim; i++)
		dot += a[i] * b[i];

	return dot;
}

/*
 * Dot product of a against two vectors at once, sharing the a loads.
 */
static inline void
neon_dot_x2(
		const float *a,
		const float *b0,
		const float *b1,
		Dimension	 dim,
		float		*out0,
		float		*out1)
{
	float32x4_t p0 = vdupq_n_f32(0.0f);
	float32x4_t p1 = p0, p2 = p0, p3 = p0;
	float32x4_t r0 = p0, r1 = p0, r2 = p0, r3 = p0;

	Dimension i = 0;
	for (; i + VS_NEON_UNROLL <= dim; i += VS_NEON_UNROLL)
	{
		float32x4x4_t va  = vld1q_f32_x4(a + i);
		float32x4x4_t vb0 = vld1q_f32_x4(b0 + i);
		float32x4x4_t vb1 = vld1q_f32_x4(b1 + i);

		p0 = vfmaq_f32(p0, va.val[0], vb0.val[0]);
		p1 = vfmaq_f32(p1, va.val[1], vb0.val[1]);
		p2 = vfmaq_f32(p2, va.val[2], vb0.val[2]);
		p3 = vfmaq_f32(p3, va.val[3], vb0.val[3]);

		r0 = vfmaq_f32(r0, va.val[0], vb1.val[0]);
		r1 = vfmaq_f32(r1, va.val[1], vb1.val[1]);
		r2 = vfmaq_f32(r2, va.val[2], vb1.val[2]);
		r3 = vfmaq_f32(r3, va.val[3], vb1.val[3]);
	}

	for (; i + 4 <= dim; i += 4)
	{
		float32x4_t va = vld1q_f32(a + i);
		p0			   = vfmaq_f32(p0, va, vld1q_f32(b0 + i));
		r0			   = vfmaq_f32(r0, va, vld1q_f32(b1 + i));
	}

	float s0 = neon_hsum4(p0, p1, p2, p3);
	float s1 = neon_hsum4(r0, r1, r2, r3);

	for (; i < dim; i++)
	{
		s0 += a[i] * b0[i];
		s1 += a[i] * b1[i];
	}

	*out0 = s0;
	*out1 = s1;
}

/*
 * Self dot product (squared L2 norm) over dim floats.
 */
static inline float
neon_norm_sq(const float *a, Dimension dim)
{
	float32x4_t s0 = vdupq_n_f32(0.0f);
	float32x4_t s1 = s0;
	float32x4_t s2 = s0;
	float32x4_t s3 = s0;

	Dimension i = 0;
	for (; i + VS_NEON_UNROLL <= dim; i += VS_NEON_UNROLL)
	{
		float32x4x4_t va = vld1q_f32_x4(a + i);

		s0 = vfmaq_f32(s0, va.val[0], va.val[0]);
		s1 = vfmaq_f32(s1, va.val[1], va.val[1]);
		s2 = vfmaq_f32(s2, va.val[2], va.val[2]);
		s3 = vfmaq_f32(s3, va.val[3], va.val[3]);
	}

	for (; i + 4 <= dim; i += 4)
	{
		float32x4_t v = vld1q_f32(a + i);
		s0			  = vfmaq_f32(s0, v, v);
	}

	float sum = neon_hsum4(s0, s1, s2, s3);

	for (; i < dim; i++)
		sum += a[i] * a[i];

	return sum;
}

/*
 * Dot product and squared norm of b in one pass.
 */
static inline void
neon_dot_and_norm(
		const float *a,
		const float *b,
		Dimension	 dim,
		float		*out_dot,
		float		*out_norm_b)
{
	float32x4_t d0 = vdupq_n_f32(0.0f);
	float32x4_t d1 = d0;
	float32x4_t d2 = d0;
	float32x4_t d3 = d0;
	float32x4_t n0 = d0;
	float32x4_t n1 = d0;
	float32x4_t n2 = d0;
	float32x4_t n3 = d0;

	Dimension i = 0;
	for (; i + VS_NEON_UNROLL <= dim; i += VS_NEON_UNROLL)
	{
		float32x4x4_t va = vld1q_f32_x4(a + i);
		float32x4x4_t vb = vld1q_f32_x4(b + i);

		d0 = vfmaq_f32(d0, va.val[0], vb.val[0]);
		d1 = vfmaq_f32(d1, va.val[1], vb.val[1]);
		d2 = vfmaq_f32(d2, va.val[2], vb.val[2]);
		d3 = vfmaq_f32(d3, va.val[3], vb.val[3]);

		n0 = vfmaq_f32(n0, vb.val[0], vb.val[0]);
		n1 = vfmaq_f32(n1, vb.val[1], vb.val[1]);
		n2 = vfmaq_f32(n2, vb.val[2], vb.val[2]);
		n3 = vfmaq_f32(n3, vb.val[3], vb.val[3]);
	}

	for (; i + 4 <= dim; i += 4)
	{
		float32x4_t va = vld1q_f32(a + i);
		float32x4_t vb = vld1q_f32(b + i);
		d0			   = vfmaq_f32(d0, va, vb);
		n0			   = vfmaq_f32(n0, vb, vb);
	}

	float dot	 = neon_hsum4(d0, d1, d2, d3);
	float norm_b = neon_hsum4(n0, n1, n2, n3);

	for (; i < dim; i++)
	{
		dot += a[i] * b[i];
		norm_b += b[i] * b[i];
	}

	*out_dot	= dot;
	*out_norm_b = norm_b;
}

/*
 * Dot product and both squared norms in one pass.
 */
static inline void
neon_cosine_sums(
		const float *a,
		const float *b,
		Dimension	 dim,
		float		*out_dot,
		float		*out_norm_a,
		float		*out_norm_b)
{
	float32x4_t d0 = vdupq_n_f32(0.0f);
	float32x4_t d1 = d0, d2 = d0, d3 = d0;
	float32x4_t m0 = d0, m1 = d0, m2 = d0, m3 = d0;
	float32x4_t n0 = d0, n1 = d0, n2 = d0, n3 = d0;

	Dimension i = 0;
	for (; i + VS_NEON_UNROLL <= dim; i += VS_NEON_UNROLL)
	{
		float32x4x4_t va = vld1q_f32_x4(a + i);
		float32x4x4_t vb = vld1q_f32_x4(b + i);

		d0 = vfmaq_f32(d0, va.val[0], vb.val[0]);
		d1 = vfmaq_f32(d1, va.val[1], vb.val[1]);
		d2 = vfmaq_f32(d2, va.val[2], vb.val[2]);
		d3 = vfmaq_f32(d3, va.val[3], vb.val[3]);

		m0 = vfmaq_f32(m0, va.val[0], va.val[0]);
		m1 = vfmaq_f32(m1, va.val[1], va.val[1]);
		m2 = vfmaq_f32(m2, va.val[2], va.val[2]);
		m3 = vfmaq_f32(m3, va.val[3], va.val[3]);

		n0 = vfmaq_f32(n0, vb.val[0], vb.val[0]);
		n1 = vfmaq_f32(n1, vb.val[1], vb.val[1]);
		n2 = vfmaq_f32(n2, vb.val[2], vb.val[2]);
		n3 = vfmaq_f32(n3, vb.val[3], vb.val[3]);
	}

	for (; i + 4 <= dim; i += 4)
	{
		float32x4_t va = vld1q_f32(a + i);
		float32x4_t vb = vld1q_f32(b + i);
		d0			   = vfmaq_f32(d0, va, vb);
		m0			   = vfmaq_f32(m0, va, va);
		n0			   = vfmaq_f32(n0, vb, vb);
	}

	float dot	 = neon_hsum4(d0, d1, d2, d3);
	float norm_a = neon_hsum4(m0, m1, m2, m3);
	float norm_b = neon_hsum4(n0, n1, n2, n3);

	for (; i < dim; i++)
	{
		dot += a[i] * b[i];
		norm_a += a[i] * a[i];
		norm_b += b[i] * b[i];
	}

	*out_dot	= dot;
	*out_norm_a = norm_a;
	*out_norm_b = norm_b;
}

static inline Distance
neon_cosine_distance(float dot, float sqrt_norm_a, float norm_b)
{
	float denom = sqrt_norm_a * sqrtf(norm_b);
	if (denom < 1e-8f)
		return 1.0f;

	return 1.0f - (dot / denom);
}

/*
 * L2 squared distance using NEON.
 */
Distance
vs_distance_l2_neon(Vec32Ref a, Vec32Ref b)
{
	if (vs_unlikely(
				a.dim != b.dim || a.dim == 0 || a.data == NULL ||
				b.data == NULL))
		return -1.0f;

	return neon_l2_sq(a.data, b.data, a.dim);
}

/*
 * Negative inner product using NEON.
 */
Distance
vs_distance_ip_neon(Vec32Ref a, Vec32Ref b)
{
	if (vs_unlikely(
				a.dim != b.dim || a.dim == 0 || a.data == NULL ||
				b.data == NULL))
		return -1.0f;

	return -neon_dot(a.data, b.data, a.dim);
}

/*
 * Cosine distance using NEON.
 */
Distance
vs_distance_cosine_neon(Vec32Ref a, Vec32Ref b)
{
	if (vs_unlikely(
				a.dim != b.dim || a.dim == 0 || a.data == NULL ||
				b.data == NULL))
		return -1.0f;

	float dot;
	float norm_a;
	float norm_b;
	neon_cosine_sums(a.data, b.data, a.dim, &dot, &norm_a, &norm_b);

	return neon_cosine_distance(dot, sqrtf(norm_a), norm_b);
}

/*
 * Batch L2 distance with prefetching.
 */
int
vs_distance_batch_l2_neon(
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

	const float *q = query.data;

	uint32_t v = 0;
	for (; v + 2 <= count; v += 2)
	{
		if (v + VS_PREFETCH_DISTANCE < count)
			vs_prefetch_read(vectors + (v + VS_PREFETCH_DISTANCE) * dim);

		neon_l2_sq_x2(
				q,
				vectors + (size_t)v * dim,
				vectors + (size_t)(v + 1) * dim,
				dim,
				&distances[v],
				&distances[v + 1]);
	}

	for (; v < count; v++)
		distances[v] = neon_l2_sq(q, vectors + (size_t)v * dim, dim);

	return 0;
}

/*
 * Batch inner product with prefetching.
 */
int
vs_distance_batch_ip_neon(
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

	const float *q = query.data;

	uint32_t v = 0;
	for (; v + 2 <= count; v += 2)
	{
		if (v + VS_PREFETCH_DISTANCE < count)
			vs_prefetch_read(vectors + (v + VS_PREFETCH_DISTANCE) * dim);

		float d0;
		float d1;
		neon_dot_x2(
				q,
				vectors + (size_t)v * dim,
				vectors + (size_t)(v + 1) * dim,
				dim,
				&d0,
				&d1);
		distances[v]	 = -d0;
		distances[v + 1] = -d1;
	}

	for (; v < count; v++)
		distances[v] = -neon_dot(q, vectors + (size_t)v * dim, dim);

	return 0;
}

/*
 * Batch cosine distance with prefetching.
 */
int
vs_distance_batch_cosine_neon(
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

	const float *q			 = query.data;
	const float	 sqrt_norm_q = sqrtf(neon_norm_sq(q, dim));

	for (uint32_t v = 0; v < count; v++)
	{
		if (v + VS_PREFETCH_DISTANCE < count)
			vs_prefetch_read(vectors + (v + VS_PREFETCH_DISTANCE) * dim);

		float dot;
		float norm_v;
		neon_dot_and_norm(q, vectors + (size_t)v * dim, dim, &dot, &norm_v);
		distances[v] = neon_cosine_distance(dot, sqrt_norm_q, norm_v);
	}

	return 0;
}
