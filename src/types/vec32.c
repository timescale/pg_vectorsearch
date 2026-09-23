/*
 * vec32.c - Vector operations
 */

#include <math.h>
#include <string.h>

#include "core/memory.h"
#include "types/vec32.h"

/* (No extern vtable — inline vtable in vec32.h, dispatch via VecType)
 */

/* ----------------------------------------------------------------
 * Vec32 lifecycle
 * ---------------------------------------------------------------- */

Vec32 *
vec32_create(Dimension dim)
{
	if (dim == 0 || dim > VEC32_MAX_DIM)
		return NULL;

	size_t size = VEC32_SIZE(dim);
	Vec32 *v	= vs_alloc(size);
	if (v == NULL)
		return NULL;

	VS_SET_VARSIZE(v, size);
	v->dim	  = (int16_t)dim;
	v->unused = 0;

	return v;
}

Vec32 *
vec32_copy(const Vec32 *src)
{
	if (src == NULL)
		return NULL;

	Vec32 *dst = vec32_create((Dimension)src->dim);
	if (dst == NULL)
		return NULL;

	memcpy(dst->x, src->x, (size_t)src->dim * sizeof(float));
	return dst;
}

void
vec32_free(Vec32 *v)
{
	vs_free(v);
}

void
vec32_set(Vec32 *v, const float *values)
{
	memcpy(v->x, values, (size_t)v->dim * sizeof(float));
}

void
vec32_zero(Vec32 *v)
{
	memset(v->x, 0, (size_t)v->dim * sizeof(float));
}

void
vec32_fill(Vec32 *v, float value)
{
	for (int16_t i = 0; i < v->dim; i++)
		v->x[i] = value;
}

float
vec32_dot(const Vec32 *a, const Vec32 *b)
{
	float sum = 0.0f;
	for (int16_t i = 0; i < a->dim; i++)
		sum += a->x[i] * b->x[i];
	return sum;
}

float
vec32_norm(const Vec32 *v)
{
	return sqrtf(vec32_dot(v, v));
}

void
vec32_normalize(Vec32 *v)
{
	float norm = vec32_norm(v);
	if (norm > 0.0f)
	{
		for (int16_t i = 0; i < v->dim; i++)
			v->x[i] /= norm;
	}
}
