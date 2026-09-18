/*
 * vec16.h - Half-precision (float16) vector type and type dispatch
 *
 * Provides:
 * - Platform-adaptive half type (F16C, _Float16, or uint16_t fallback)
 * - Vec16 struct (binary-compatible with pgvector's HalfVector)
 * - Vec32TypeOps vtable for type-generic k-means and quantization
 * - Scalar and bulk conversion between half and float
 *
 * The vtable enables mixed-type distance computation (vec_type × float32
 * centroid) so that halfvec inputs read half the memory without bulk
 * conversion. Full float32 conversion is only done where unavoidable
 * (CBLAS sgemm, RaBitQ rotation).
 */

#ifndef VEC16_H
#define VEC16_H

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "core/types.h"

/* ----------------------------------------------------------------
 * Half type detection (independent flags, prefer _Float16)
 *
 * _Float16 is preferred when available because the compiler can
 * auto-vectorize (float)h casts into bulk vcvtph2ps (8 at a time),
 * whereas _cvtsh_ss is a scalar intrinsic (1 at a time).  With
 * -mf16c, the compiler emits F16C instructions for _Float16 casts.
 *
 * F16C intrinsics are the fallback when _Float16 is not supported
 * but -mf16c is available (e.g. older compilers).
 * ---------------------------------------------------------------- */

#if defined(__FLT16_MAX__) && !defined(__FreeBSD__) && \
		(!defined(__i386__) || defined(__SSE2__))
#define MKT_FLT16_SUPPORT
#endif

#if defined(__F16C__)
#include <immintrin.h>
#define MKT_F16C_SUPPORT
#endif

#ifdef MKT_FLT16_SUPPORT
typedef _Float16 half;
#define MKT_HALF_MAX FLT16_MAX
#else
typedef uint16_t half;
#define MKT_HALF_MAX 65504
#endif

/* ----------------------------------------------------------------
 * Scalar conversion (inline, 3-tier)
 * ---------------------------------------------------------------- */

MKT_VTABLE_INLINE float
mkt_half_to_float(half h)
{
#if defined(MKT_FLT16_SUPPORT)
	return (float)h;
#elif defined(MKT_F16C_SUPPORT)
	return _cvtsh_ss(h);
#else
	/* IEEE 754 bit manipulation fallback */
	uint16_t bits = h;
	uint32_t sign = (uint32_t)(bits >> 15) << 31;
	uint32_t exp  = (bits >> 10) & 0x1F;
	uint32_t frac = bits & 0x3FF;
	uint32_t f32;

	if (exp == 0)
	{
		if (frac == 0)
		{
			/* ±zero */
			f32 = sign;
		}
		else
		{
			/* subnormal: normalize */
			exp = 1;
			while ((frac & 0x400) == 0)
			{
				frac <<= 1;
				exp--;
			}
			frac &= 0x3FF;
			f32 = sign | ((uint32_t)(exp + 127 - 15) << 23) |
				  ((uint32_t)frac << 13);
		}
	}
	else if (exp == 31)
	{
		/* Inf or NaN */
		f32 = sign | 0x7F800000u | ((uint32_t)frac << 13);
	}
	else
	{
		/* normal */
		f32 = sign | ((uint32_t)(exp + 127 - 15) << 23) |
			  ((uint32_t)frac << 13);
	}

	float result;
	memcpy(&result, &f32, sizeof(float));
	return result;
#endif
}

static inline half
mkt_float_to_half(float f)
{
#if defined(MKT_FLT16_SUPPORT)
	return (_Float16)f;
#elif defined(MKT_F16C_SUPPORT)
	return _cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#else
	/* IEEE 754 bit manipulation fallback */
	uint32_t bits;
	memcpy(&bits, &f, sizeof(uint32_t));

	uint32_t sign = (bits >> 16) & 0x8000;
	int32_t	 exp  = ((bits >> 23) & 0xFF) - 127 + 15;
	uint32_t frac = bits & 0x7FFFFF;

	if (exp <= 0)
	{
		if (exp < -10)
		{
			/* Too small, flush to zero */
			return (half)(sign);
		}
		/* Subnormal in half precision */
		frac |= 0x800000;
		uint32_t shift = (uint32_t)(1 - exp);
		/* Round to nearest, ties to even */
		uint32_t round_bit = 1u << (shift + 12);
		frac += round_bit >> 1;
		if ((frac & round_bit) && (frac & (round_bit - 1)) == 0)
			frac &= ~(round_bit >> 1); /* tie: round to even */
		return (half)(sign | (frac >> (shift + 13)));
	}
	else if (exp >= 31)
	{
		if (exp == 31 && frac != 0)
		{
			/* NaN: preserve some significand bits */
			return (half)(sign | 0x7E00 | (frac >> 13));
		}
		/* Overflow: Inf */
		return (half)(sign | 0x7C00);
	}

	/* Round to nearest, ties to even */
	frac += 0x1000; /* round bit at position 12 */
	if (frac & 0x800000)
	{
		frac = 0;
		exp++;
		if (exp >= 31)
			return (half)(sign | 0x7C00); /* overflow to Inf */
	}

	return (half)(sign | ((uint32_t)exp << 10) | (frac >> 13));
#endif
}

/* ----------------------------------------------------------------
 * Special value checks
 * ---------------------------------------------------------------- */

static inline bool
mkt_half_is_nan(half h)
{
#ifdef MKT_FLT16_SUPPORT
	return isnan(h);
#else
	uint16_t bits = h;
	return ((bits & 0x7C00) == 0x7C00) && ((bits & 0x03FF) != 0);
#endif
}

static inline bool
mkt_half_is_inf(half h)
{
#ifdef MKT_FLT16_SUPPORT
	return isinf(h);
#else
	uint16_t bits = h;
	return ((bits & 0x7FFF) == 0x7C00);
#endif
}

static inline bool
mkt_half_is_zero(half h)
{
#ifdef MKT_FLT16_SUPPORT
	return h == (_Float16)0;
#else
	return (h & 0x7FFF) == 0;
#endif
}

/* ----------------------------------------------------------------
 * Vec16 (binary-compatible with pgvector HalfVector)
 * ---------------------------------------------------------------- */

typedef struct Vec16
{
	int32_t vl_len_; /* varlena header / size in standalone mode */
	int16_t dim;	 /* number of dimensions */
	int16_t unused;	 /* reserved for future use, always zero */
	half	x[];	 /* flexible array member */
} Vec16;

#define VEC16_SIZE(dim) (offsetof(Vec16, x) + sizeof(half) * (dim))
#define VEC16_DIM(v)	((v)->dim)
#define VEC16_DATA(v)	((v)->x)

/* ----------------------------------------------------------------
 * Bulk conversion (SIMD-dispatched in vec16.c)
 * ---------------------------------------------------------------- */

void mkt_half_to_float_array(const half *src, float *dst, uint32_t n);
void mkt_float_to_half_array(const float *src, half *dst, uint32_t n);

/* ----------------------------------------------------------------
 * Vec16 lifecycle
 * ---------------------------------------------------------------- */

Vec16 *vec16_create(Dimension dim);
Vec16 *vec16_from_floats(const float *values, Dimension dim);
void   vec16_free(Vec16 *v);
void   vec16_set(Vec16 *v, const float *values);

/* Convert to Vec32Ref (requires caller-provided float32 buffer) */
static inline Vec32Ref
Vec16ToRef(const Vec16 *hv, float *buffer)
{
	mkt_half_to_float_array(hv->x, buffer, hv->dim);
	return (Vec32Ref){.data = buffer, .dim = (Dimension)hv->dim};
}

/* ----------------------------------------------------------------
 * Inline vtable for compile-time specialization
 *
 * These always_inline functions + static const vtable enable the
 * compiler to inline through vtable function pointers when the
 * pointer target is known at compile time. Used by k-means and
 * other hot loops that dispatch once at the entry point.
 * ---------------------------------------------------------------- */

MKT_VTABLE_INLINE float
mkt_f16_dot_product(const void *vec, const float *centroid, Dimension dim)
{
	const half *v	= (const half *)vec;
	float		sum = 0.0f;
	for (Dimension d = 0; d < dim; d++)
		sum += mkt_half_to_float(v[d]) * centroid[d];
	return sum;
}

MKT_VTABLE_INLINE float
mkt_f16_l2_squared(const void *vec, const float *centroid, Dimension dim)
{
	const half *v	= (const half *)vec;
	float		sum = 0.0f;
	for (Dimension d = 0; d < dim; d++)
	{
		float diff = mkt_half_to_float(v[d]) - centroid[d];
		sum += diff * diff;
	}
	return sum;
}

MKT_VTABLE_INLINE float
mkt_f16_norm_sq(const void *vec, Dimension dim)
{
	const half *v	= (const half *)vec;
	float		sum = 0.0f;
	for (Dimension d = 0; d < dim; d++)
	{
		float val = mkt_half_to_float(v[d]);
		sum += val * val;
	}
	return sum;
}

MKT_VTABLE_INLINE void
mkt_f16_sum_to_float(const void *vec, float *accum, Dimension dim)
{
	const half *v = (const half *)vec;
	for (Dimension d = 0; d < dim; d++)
		accum[d] += mkt_half_to_float(v[d]);
}

MKT_VTABLE_INLINE void
mkt_f16_to_float_one(const void *src, float *dst, Dimension dim)
{
	mkt_half_to_float_array((const half *)src, dst, dim);
}

MKT_VTABLE_INLINE const float *
mkt_f16_to_float_block(
		const void *src, float *dst, uint32_t count, Dimension dim)
{
	mkt_half_to_float_array((const half *)src, dst, (uint32_t)count * dim);
	return dst;
}

static const Vec32TypeOps mkt_f16_type_ops = {
		.name			= "float16",
		.element_size	= sizeof(half),
		.dot_product	= mkt_f16_dot_product,
		.l2_squared		= mkt_f16_l2_squared,
		.norm_sq		= mkt_f16_norm_sq,
		.sum_to_float	= mkt_f16_sum_to_float,
		.to_float_one	= mkt_f16_to_float_one,
		.to_float_block = mkt_f16_to_float_block,
};

/* ----------------------------------------------------------------
 * F16C inline vtable for compile-time specialization
 *
 * Hand-written AVX2+FMA+F16C: loads 8 halfs via _mm256_cvtph_ps,
 * accumulates with FMA, reduces via horizontal sum. Scalar tail
 * for remainder. Used by k-means and RaBitQ wrappers that target
 * AVX2 explicitly rather than relying on TARGET_CLONES
 * auto-vectorization.
 * ---------------------------------------------------------------- */

#if defined(MKT_F16C_SUPPORT) && !defined(MKT_SIMD_NONE)

#include "algo/simd_utils.h"

MKT_TARGET_F16C_AVX2 MKT_VTABLE_INLINE float
mkt_f16c_dot_product(const void *vec, const float *centroid, Dimension dim)
{
	const half *v	 = (const half *)vec;
	__m256		sum8 = _mm256_setzero_ps();
	Dimension	d	 = 0;

	for (; d + 8 <= dim; d += 8)
	{
		__m128i h8 = _mm_loadu_si128((const __m128i *)(v + d));
		__m256	fv = _mm256_cvtph_ps(h8);
		__m256	fc = _mm256_loadu_ps(centroid + d);
		sum8	   = _mm256_fmadd_ps(fv, fc, sum8);
	}

	float sum = mkt_horizontal_sum_avx2(sum8);

	for (; d < dim; d++)
		sum += mkt_half_to_float(v[d]) * centroid[d];

	return sum;
}

MKT_TARGET_F16C_AVX2 MKT_VTABLE_INLINE float
mkt_f16c_l2_squared(const void *vec, const float *centroid, Dimension dim)
{
	const half *v	 = (const half *)vec;
	__m256		sum8 = _mm256_setzero_ps();
	Dimension	d	 = 0;

	for (; d + 8 <= dim; d += 8)
	{
		__m128i h8	 = _mm_loadu_si128((const __m128i *)(v + d));
		__m256	fv	 = _mm256_cvtph_ps(h8);
		__m256	fc	 = _mm256_loadu_ps(centroid + d);
		__m256	diff = _mm256_sub_ps(fv, fc);
		sum8		 = _mm256_fmadd_ps(diff, diff, sum8);
	}

	float sum = mkt_horizontal_sum_avx2(sum8);

	for (; d < dim; d++)
	{
		float diff = mkt_half_to_float(v[d]) - centroid[d];
		sum += diff * diff;
	}

	return sum;
}

MKT_TARGET_F16C_AVX2 MKT_VTABLE_INLINE float
mkt_f16c_norm_sq(const void *vec, Dimension dim)
{
	const half *v	 = (const half *)vec;
	__m256		sum8 = _mm256_setzero_ps();
	Dimension	d	 = 0;

	for (; d + 8 <= dim; d += 8)
	{
		__m128i h8 = _mm_loadu_si128((const __m128i *)(v + d));
		__m256	fv = _mm256_cvtph_ps(h8);
		sum8	   = _mm256_fmadd_ps(fv, fv, sum8);
	}

	float sum = mkt_horizontal_sum_avx2(sum8);

	for (; d < dim; d++)
	{
		float val = mkt_half_to_float(v[d]);
		sum += val * val;
	}

	return sum;
}

MKT_TARGET_F16C_AVX2 MKT_VTABLE_INLINE void
mkt_f16c_sum_to_float(const void *vec, float *accum, Dimension dim)
{
	const half *v = (const half *)vec;
	Dimension	d = 0;

	for (; d + 8 <= dim; d += 8)
	{
		__m128i h8 = _mm_loadu_si128((const __m128i *)(v + d));
		__m256	fv = _mm256_cvtph_ps(h8);
		__m256	ac = _mm256_loadu_ps(accum + d);
		_mm256_storeu_ps(accum + d, _mm256_add_ps(ac, fv));
	}

	for (; d < dim; d++)
		accum[d] += mkt_half_to_float(v[d]);
}

static const Vec32TypeOps mkt_f16c_type_ops = {
		.name			= "float16-f16c",
		.element_size	= sizeof(half),
		.dot_product	= mkt_f16c_dot_product,
		.l2_squared		= mkt_f16c_l2_squared,
		.norm_sq		= mkt_f16c_norm_sq,
		.sum_to_float	= mkt_f16c_sum_to_float,
		.to_float_one	= mkt_f16_to_float_one,
		.to_float_block = mkt_f16_to_float_block,
};

#endif /* MKT_F16C_SUPPORT && !MKT_SIMD_NONE */

#endif /* VEC16_H */
