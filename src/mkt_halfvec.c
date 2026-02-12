/*
 * mkt_halfvec.c - Half-precision vector operations and type dispatch
 *
 * Implements:
 * - Bulk half<->float conversion with SIMD dispatch
 * - MktVectorTypeOps for float32 (zero-copy wrappers)
 * - MktVectorTypeOps for float16 (convert-in-register distance)
 * - MktHalfVector lifecycle
 */

#include "mkt_config.h"

#include <string.h>

#include "core/memory.h"
#include "mkt_halfvec.h"
#include "mkt_vector.h"

/* ----------------------------------------------------------------
 * Bulk conversion
 *
 * On x86 with F16C: _mm256_cvtph_ps / _mm256_cvtps_ph (8 at a time)
 * On ARM with FP16: vcvt_f32_f16 / vcvt_f16_f32 (4 at a time)
 * Fallback: scalar loop via mkt_half_to_float / mkt_float_to_half
 * ---------------------------------------------------------------- */

#if defined(MKT_F16C_SUPPORT) && !defined(MKT_SIMD_NONE)

__attribute__((target("avx,f16c"))) void
mkt_half_to_float_array(const half *src, float *dst, uint32_t n)
{
	uint32_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m128i h8 = _mm_loadu_si128((const __m128i *)(src + i));
		__m256	f8 = _mm256_cvtph_ps(h8);
		_mm256_storeu_ps(dst + i, f8);
	}
	for (; i < n; i++)
		dst[i] = mkt_half_to_float(src[i]);
}

__attribute__((target("avx,f16c"))) void
mkt_float_to_half_array(const float *src, half *dst, uint32_t n)
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
		dst[i] = mkt_float_to_half(src[i]);
}

#elif defined(__aarch64__) && defined(__ARM_FP16_FORMAT_IEEE) && \
		!defined(MKT_SIMD_NONE)

#include <arm_neon.h>

void
mkt_half_to_float_array(const half *src, float *dst, uint32_t n)
{
	uint32_t i = 0;
	for (; i + 4 <= n; i += 4)
	{
		float16x4_t h4 = vld1_f16((const float16_t *)(src + i));
		float32x4_t f4 = vcvt_f32_f16(h4);
		vst1q_f32(dst + i, f4);
	}
	for (; i < n; i++)
		dst[i] = mkt_half_to_float(src[i]);
}

void
mkt_float_to_half_array(const float *src, half *dst, uint32_t n)
{
	uint32_t i = 0;
	for (; i + 4 <= n; i += 4)
	{
		float32x4_t f4 = vld1q_f32(src + i);
		float16x4_t h4 = vcvt_f16_f32(f4);
		vst1_f16((float16_t *)(dst + i), h4);
	}
	for (; i < n; i++)
		dst[i] = mkt_float_to_half(src[i]);
}

#else

/* Scalar fallback */
void
mkt_half_to_float_array(const half *src, float *dst, uint32_t n)
{
	for (uint32_t i = 0; i < n; i++)
		dst[i] = mkt_half_to_float(src[i]);
}

void
mkt_float_to_half_array(const float *src, half *dst, uint32_t n)
{
	for (uint32_t i = 0; i < n; i++)
		dst[i] = mkt_float_to_half(src[i]);
}

#endif

/* ----------------------------------------------------------------
 * MktHalfVector lifecycle
 * ---------------------------------------------------------------- */

MktHalfVector *
mkt_halfvec_create(Dimension dim)
{
	if (dim == 0 || dim > MKT_VECTOR_MAX_DIM)
		return NULL;

	size_t		   size = MKT_HALFVEC_SIZE(dim);
	MktHalfVector *v	= mkt_alloc0(size);
	if (v == NULL)
		return NULL;

	MKT_SET_VARSIZE(v, size);
	v->dim = (int16_t)dim;
	return v;
}

MktHalfVector *
mkt_halfvec_from_floats(const float *values, Dimension dim)
{
	if (values == NULL)
		return NULL;

	MktHalfVector *v = mkt_halfvec_create(dim);
	if (v == NULL)
		return NULL;

	mkt_float_to_half_array(values, v->x, dim);
	return v;
}

void
mkt_halfvec_free(MktHalfVector *v)
{
	mkt_free(v);
}

void
mkt_halfvec_set(MktHalfVector *v, const float *values)
{
	if (v == NULL || values == NULL)
		return;
	mkt_float_to_half_array(values, v->x, v->dim);
}
