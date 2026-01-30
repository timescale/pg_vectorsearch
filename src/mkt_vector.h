/*
 * mkt_vector.h - Vector type compatible with pgvector
 *
 * The struct layout is identical to pgvector's Vector type, enabling
 * zero-copy interoperability when running inside PostgreSQL.
 */

#ifndef MKT_VECTOR_H
#define MKT_VECTOR_H

#include <stdint.h>

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

#endif /* MKT_VECTOR_H */
