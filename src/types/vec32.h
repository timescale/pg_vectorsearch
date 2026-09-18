/*
 * vec32.h - Vector type compatible with pgvector
 *
 * The struct layout is identical to pgvector's Vector type, enabling
 * zero-copy interoperability when running inside PostgreSQL.
 */

#ifndef VEC32_H
#define VEC32_H

#include <stdint.h>
#include <string.h>

#include "core/types.h"

#define VEC32_MAX_DIM 16000

/*
 * Vec32: Binary-compatible with pgvector's Vector type.
 *
 * The struct layout is identical in both standalone and PostgreSQL modes.
 * In standalone mode, vl_len_ stores the total size (not used as varlena).
 * In PostgreSQL mode, vl_len_ is managed by SET_VARSIZE/VARSIZE macros.
 */
typedef struct Vec32
{
	int32_t vl_len_; /* varlena header / size in standalone mode */
	int16_t dim;	 /* number of dimensions */
	int16_t unused;	 /* reserved for future use, always zero */
	float	x[];	 /* flexible array member */
} Vec32;

#define VEC32_SIZE(dim) (offsetof(Vec32, x) + sizeof(float) * (dim))
#define VEC32_DIM(v)	((v)->dim)
#define VEC32_DATA(v)	((v)->x)

/* Convert to Vec32Ref for internal operations */
static inline Vec32Ref
Vec32ToRef(const Vec32 *v)
{
	return (Vec32Ref){.data = v->x, .dim = (Dimension)v->dim};
}

/* Convert to VectorMut for mutable operations */
static inline VectorMut
Vec32ToMut(Vec32 *v)
{
	return (VectorMut){.data = v->x, .dim = (Dimension)v->dim};
}

/* Allocation and lifecycle */
Vec32 *vec32_create(Dimension dim);
Vec32 *vec32_copy(const Vec32 *src);
void   vec32_free(Vec32 *v);

/* Initialization */
void vec32_set(Vec32 *v, const float *values);
void vec32_zero(Vec32 *v);
void vec32_fill(Vec32 *v, float value);

/* Operations */
float vec32_dot(const Vec32 *a, const Vec32 *b);
float vec32_norm(const Vec32 *v);
void  vec32_normalize(Vec32 *v);

/* ----------------------------------------------------------------
 * Inline vtable for compile-time specialization
 *
 * These inline functions + static const vtable enable the
 * compiler to inline through vtable function pointers when the
 * pointer target is known at compile time. Used by k-means and
 * other hot loops that dispatch once at the entry point.
 * ---------------------------------------------------------------- */

MKT_VTABLE_INLINE float
mkt_f32_dot_product(const void *vec, const float *centroid, Dimension dim)
{
	const float *v	 = (const float *)vec;
	float		 sum = 0.0f;
	for (Dimension d = 0; d < dim; d++)
		sum += v[d] * centroid[d];
	return sum;
}

MKT_VTABLE_INLINE float
mkt_f32_l2_squared(const void *vec, const float *centroid, Dimension dim)
{
	const float *v	 = (const float *)vec;
	float		 sum = 0.0f;
	for (Dimension d = 0; d < dim; d++)
	{
		float diff = v[d] - centroid[d];
		sum += diff * diff;
	}
	return sum;
}

MKT_VTABLE_INLINE float
mkt_f32_norm_sq(const void *vec, Dimension dim)
{
	const float *v	 = (const float *)vec;
	float		 sum = 0.0f;
	for (Dimension d = 0; d < dim; d++)
		sum += v[d] * v[d];
	return sum;
}

MKT_VTABLE_INLINE void
mkt_f32_sum_to_float(const void *vec, float *accum, Dimension dim)
{
	const float *v = (const float *)vec;
	for (Dimension d = 0; d < dim; d++)
		accum[d] += v[d];
}

MKT_VTABLE_INLINE void
mkt_f32_to_float_one(const void *src, float *dst, Dimension dim)
{
	memcpy(dst, src, (size_t)dim * sizeof(float));
}

MKT_VTABLE_INLINE const float *
mkt_f32_to_float_block(
		const void *src, float *dst, uint32_t count, Dimension dim)
{
	(void)dst;
	(void)count;
	(void)dim;
	return (const float *)src;
}

static const Vec32TypeOps mkt_f32_type_ops = {
		.name			= "float32",
		.element_size	= sizeof(float),
		.dot_product	= mkt_f32_dot_product,
		.l2_squared		= mkt_f32_l2_squared,
		.norm_sq		= mkt_f32_norm_sq,
		.sum_to_float	= mkt_f32_sum_to_float,
		.to_float_one	= mkt_f32_to_float_one,
		.to_float_block = mkt_f32_to_float_block,
};

#endif /* VEC32_H */
