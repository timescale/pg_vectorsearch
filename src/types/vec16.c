/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * vec16.c - Half-precision vector operations and type dispatch
 *
 * Implements:
 * - Bulk half<->float conversion with SIMD dispatch
 * - Vec32TypeOps for float32 (zero-copy wrappers)
 * - Vec32TypeOps for float16 (convert-in-register distance)
 * - Vec16 lifecycle
 */

#include "vs_config.h"

#include <string.h>

#include "core/memory.h"
#include "types/vec16.h"
#include "types/vec32.h"

/* ----------------------------------------------------------------
 * Bulk conversion
 *
 * On x86 with F16C: _mm256_cvtph_ps / _mm256_cvtps_ph (8 at a time)
 * On ARM with FP16: vcvt_f32_f16 / vcvt_f16_f32 (4 at a time)
 * Fallback: scalar loop via vs_half_to_float / vs_float_to_half
 * ---------------------------------------------------------------- */

/* Whether a float16, as bits, is infinite or NaN: all exponent bits set. */
static inline bool
half_bits_nonfinite(uint16_t bits)
{
	return (bits & 0x7C00) == 0x7C00;
}

/* Index of the first non-finite float16 in dst[from, n), or n. */
static uint32_t
first_nonfinite_half(const half *dst, uint32_t from, uint32_t n)
{
	for (uint32_t i = from; i < n; i++)
	{
		uint16_t bits;

		memcpy(&bits, &dst[i], sizeof(bits));
		if (half_bits_nonfinite(bits))
			return i;
	}
	return n;
}

#if defined(VS_F16C_SUPPORT) && !defined(VS_SIMD_NONE)

__attribute__((target("avx,f16c"))) void
vs_half_to_float_array(const half *src, float *dst, uint32_t n)
{
	uint32_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m128i h8 = _mm_loadu_si128((const __m128i *)(src + i));
		__m256	f8 = _mm256_cvtph_ps(h8);
		_mm256_storeu_ps(dst + i, f8);
	}
	for (; i < n; i++)
		dst[i] = vs_half_to_float(src[i]);
}

__attribute__((target("avx,f16c"))) void
vs_float_to_half_array(const float *src, half *dst, uint32_t n)
{
	uint32_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256	f8 = _mm256_loadu_ps(src + i);
		__m128i h8 = _mm256_cvtps_ph(
				f8, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
		_mm_storeu_si128((__m128i *)(dst + i), h8);
	}
	for (; i < n; i++)
		dst[i] = vs_float_to_half(src[i]);
}

/* The conversion loop above, ORing each lane's "exponent all ones" into an
 * accumulator; the first bad index is found only when something flagged. */
__attribute__((target("avx,f16c"))) uint32_t
vs_float_to_half_array_checked(const float *src, half *dst, uint32_t n)
{
	const __m128i exp_mask = _mm_set1_epi16(0x7C00);
	__m128i		  bad	   = _mm_setzero_si128();
	uint32_t	  i		   = 0;

	for (; i + 8 <= n; i += 8)
	{
		__m256	f8 = _mm256_loadu_ps(src + i);
		__m128i h8 = _mm256_cvtps_ph(
				f8, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
		_mm_storeu_si128((__m128i *)(dst + i), h8);
		bad = _mm_or_si128(
				bad, _mm_cmpeq_epi16(_mm_and_si128(h8, exp_mask), exp_mask));
	}
	uint32_t body = i;
	for (; i < n; i++)
		dst[i] = vs_float_to_half(src[i]);

	if (_mm_movemask_epi8(bad) != 0)
		return first_nonfinite_half(dst, 0, body);
	return first_nonfinite_half(dst, body, n);
}

#elif defined(__aarch64__) && defined(__ARM_FP16_FORMAT_IEEE) && \
		!defined(VS_SIMD_NONE)

#include <arm_neon.h>

void
vs_half_to_float_array(const half *src, float *dst, uint32_t n)
{
	uint32_t i = 0;
	for (; i + 4 <= n; i += 4)
	{
		float16x4_t h4 = vld1_f16((const float16_t *)(src + i));
		float32x4_t f4 = vcvt_f32_f16(h4);
		vst1q_f32(dst + i, f4);
	}
	for (; i < n; i++)
		dst[i] = vs_half_to_float(src[i]);
}

void
vs_float_to_half_array(const float *src, half *dst, uint32_t n)
{
	uint32_t i = 0;
	for (; i + 4 <= n; i += 4)
	{
		float32x4_t f4 = vld1q_f32(src + i);
		float16x4_t h4 = vcvt_f16_f32(f4);
		vst1_f16((float16_t *)(dst + i), h4);
	}
	for (; i < n; i++)
		dst[i] = vs_float_to_half(src[i]);
}

/* As the F16C version: flag lanes in the loop, locate them after it. */
uint32_t
vs_float_to_half_array_checked(const float *src, half *dst, uint32_t n)
{
	const uint16x4_t exp_mask = vdup_n_u16(0x7C00);
	uint16x4_t		 bad	  = vdup_n_u16(0);
	uint32_t		 i		  = 0;

	for (; i + 4 <= n; i += 4)
	{
		float16x4_t h4 = vcvt_f16_f32(vld1q_f32(src + i));
		uint16x4_t	b4 = vreinterpret_u16_f16(h4);

		vst1_f16((float16_t *)(dst + i), h4);
		bad = vorr_u16(bad, vceq_u16(vand_u16(b4, exp_mask), exp_mask));
	}
	uint32_t body = i;
	for (; i < n; i++)
		dst[i] = vs_float_to_half(src[i]);

	if (vmaxv_u16(bad) != 0)
		return first_nonfinite_half(dst, 0, body);
	return first_nonfinite_half(dst, body, n);
}

#else

/* Scalar fallback */
void
vs_half_to_float_array(const half *src, float *dst, uint32_t n)
{
	for (uint32_t i = 0; i < n; i++)
		dst[i] = vs_half_to_float(src[i]);
}

void
vs_float_to_half_array(const float *src, half *dst, uint32_t n)
{
	for (uint32_t i = 0; i < n; i++)
		dst[i] = vs_float_to_half(src[i]);
}

uint32_t
vs_float_to_half_array_checked(const float *src, half *dst, uint32_t n)
{
	vs_float_to_half_array(src, dst, n);
	return first_nonfinite_half(dst, 0, n);
}

#endif

/* ----------------------------------------------------------------
 * Vec16 lifecycle
 * ---------------------------------------------------------------- */

Vec16 *
vec16_create(Dimension dim)
{
	if (dim == 0 || dim > VEC32_MAX_DIM)
		return NULL;

	size_t size = VEC16_SIZE(dim);
	Vec16 *v	= vs_alloc0(size);
	if (v == NULL)
		return NULL;

	VS_SET_VARSIZE(v, size);
	v->dim = (int16_t)dim;
	return v;
}

Vec16 *
vec16_from_floats(const float *values, Dimension dim)
{
	if (values == NULL)
		return NULL;

	Vec16 *v = vec16_create(dim);
	if (v == NULL)
		return NULL;

	if (vs_float_to_half_array_checked(values, v->x, dim) < dim)
	{
		vec16_free(v);
		return NULL;
	}
	return v;
}

void
vec16_free(Vec16 *v)
{
	vs_free(v);
}

bool
vec16_set(Vec16 *v, const float *values)
{
	if (v == NULL || values == NULL)
		return false;
	uint32_t dim = (uint32_t)v->dim;

	return vs_float_to_half_array_checked(values, v->x, dim) == dim;
}
