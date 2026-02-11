/*
 * mkt_vector.h - Vector type compatible with pgvector
 *
 * The struct layout is identical to pgvector's Vector type, enabling
 * zero-copy interoperability when running inside PostgreSQL.
 */

#ifndef MKT_VECTOR_H
#define MKT_VECTOR_H

#include <stdint.h>
#include <string.h>

#include "mkt_types.h"

#define MKT_VECTOR_MAX_DIM 16000

/*
 * MktVector: Binary-compatible with pgvector's Vector type.
 *
 * The struct layout is identical in both standalone and PostgreSQL modes.
 * In standalone mode, vl_len_ stores the total size (not used as varlena).
 * In PostgreSQL mode, vl_len_ is managed by SET_VARSIZE/VARSIZE macros.
 */
typedef struct MktVector
{
	int32_t vl_len_; /* varlena header / size in standalone mode */
	int16_t dim;	 /* number of dimensions */
	int16_t unused;	 /* reserved for future use, always zero */
	float	x[];	 /* flexible array member */
} MktVector;

#define MKT_VECTOR_SIZE(dim) (offsetof(MktVector, x) + sizeof(float) * (dim))
#define MKT_VECTOR_DIM(v)	 ((v)->dim)
#define MKT_VECTOR_DATA(v)	 ((v)->x)

/* Convert to VectorRef for internal operations */
static inline VectorRef
MktVectorToRef(const MktVector *v)
{
	return (VectorRef){.data = v->x, .dim = (Dimension)v->dim};
}

/* Convert to VectorMut for mutable operations */
static inline VectorMut
MktVectorToMut(MktVector *v)
{
	return (VectorMut){.data = v->x, .dim = (Dimension)v->dim};
}

/* Allocation and lifecycle */
MktVector *mkt_vector_create(Dimension dim);
MktVector *mkt_vector_copy(const MktVector *src);
void	   mkt_vector_free(MktVector *v);

/* Initialization */
void mkt_vector_set(MktVector *v, const float *values);
void mkt_vector_zero(MktVector *v);
void mkt_vector_fill(MktVector *v, float value);

/* Operations */
float mkt_vector_dot(const MktVector *a, const MktVector *b);
float mkt_vector_norm(const MktVector *v);
void  mkt_vector_normalize(MktVector *v);

/* ----------------------------------------------------------------
 * Inline vtable for compile-time specialization
 *
 * These always_inline functions + static const vtable enable the
 * compiler to inline through vtable function pointers when the
 * pointer target is known at compile time. Used by k-means and
 * other hot loops that dispatch once at the entry point.
 * ---------------------------------------------------------------- */

__attribute__((always_inline)) static inline float
mkt_f32_dot_product(const void *vec, const float *centroid, Dimension dim)
{
	const float *v	 = (const float *)vec;
	float		 sum = 0.0f;
	for (Dimension d = 0; d < dim; d++)
		sum += v[d] * centroid[d];
	return sum;
}

__attribute__((always_inline)) static inline float
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

__attribute__((always_inline)) static inline float
mkt_f32_norm_sq(const void *vec, Dimension dim)
{
	const float *v	 = (const float *)vec;
	float		 sum = 0.0f;
	for (Dimension d = 0; d < dim; d++)
		sum += v[d] * v[d];
	return sum;
}

__attribute__((always_inline)) static inline void
mkt_f32_sum_to_float(const void *vec, float *accum, Dimension dim)
{
	const float *v = (const float *)vec;
	for (Dimension d = 0; d < dim; d++)
		accum[d] += v[d];
}

__attribute__((always_inline)) static inline void
mkt_f32_to_float_one(const void *src, float *dst, Dimension dim)
{
	memcpy(dst, src, (size_t)dim * sizeof(float));
}

__attribute__((always_inline)) static inline const float *
mkt_f32_to_float_block(
		const void *src, float *dst, uint32_t count, Dimension dim)
{
	(void)dst;
	(void)count;
	(void)dim;
	return (const float *)src;
}

static const MktVectorTypeOps mkt_f32_type_ops = {
		.name			= "float32",
		.element_size	= sizeof(float),
		.dot_product	= mkt_f32_dot_product,
		.l2_squared		= mkt_f32_l2_squared,
		.norm_sq		= mkt_f32_norm_sq,
		.sum_to_float	= mkt_f32_sum_to_float,
		.to_float_one	= mkt_f32_to_float_one,
		.to_float_block = mkt_f32_to_float_block,
};

#endif /* MKT_VECTOR_H */
