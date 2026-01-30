/*
 * mkt_vector.c - Vector operations
 */

#include <math.h>
#include <string.h>

#include "mkt_memory.h"
#include "mkt_vector.h"

MktVector *
mkt_vector_create(Dimension dim)
{
	if (dim == 0 || dim > MKT_VECTOR_MAX_DIM)
		return NULL;

	size_t	   size = MKT_VECTOR_SIZE(dim);
	MktVector *v	= mkt_alloc(size);
	if (v == NULL)
		return NULL;

	v->vl_len_ = (int32_t)size; /* Store size directly in standalone mode */
	v->dim	   = (int16_t)dim;
	v->unused  = 0;

	return v;
}

MktVector *
mkt_vector_copy(const MktVector *src)
{
	if (src == NULL)
		return NULL;

	MktVector *dst = mkt_vector_create((Dimension)src->dim);
	if (dst == NULL)
		return NULL;

	memcpy(dst->x, src->x, (size_t)src->dim * sizeof(float));
	return dst;
}

void
mkt_vector_free(MktVector *v)
{
	mkt_free(v);
}

void
mkt_vector_set(MktVector *v, const float *values)
{
	memcpy(v->x, values, (size_t)v->dim * sizeof(float));
}

void
mkt_vector_zero(MktVector *v)
{
	memset(v->x, 0, (size_t)v->dim * sizeof(float));
}

void
mkt_vector_fill(MktVector *v, float value)
{
	for (int16_t i = 0; i < v->dim; i++)
		v->x[i] = value;
}

float
mkt_vector_dot(const MktVector *a, const MktVector *b)
{
	float sum = 0.0f;
	for (int16_t i = 0; i < a->dim; i++)
		sum += a->x[i] * b->x[i];
	return sum;
}

float
mkt_vector_norm(const MktVector *v)
{
	return sqrtf(mkt_vector_dot(v, v));
}

void
mkt_vector_normalize(MktVector *v)
{
	float norm = mkt_vector_norm(v);
	if (norm > 0.0f)
	{
		for (int16_t i = 0; i < v->dim; i++)
			v->x[i] /= norm;
	}
}
