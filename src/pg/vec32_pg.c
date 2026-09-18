/*
 * vec32_pg.c - PostgreSQL functions for mkt.vec32 type
 *
 * Type I/O, distance functions, comparison operators, casts.
 * Distance functions bridge to SIMD-accelerated meerkat core.
 */

#include <postgres.h>

#include <catalog/pg_type.h>
#include <fmgr.h>
#include <lib/stringinfo.h>
#include <libpq/pqformat.h>
#include <utils/array.h>
#include <utils/float.h>

#include "algo/distance.h"
#include "support_pg.h"

/* ----------------------------------------------------------------
 * Type I/O
 * ---------------------------------------------------------------- */

PG_FUNCTION_INFO_V1(mkt_vec32_in);

Datum
mkt_vec32_in(PG_FUNCTION_ARGS)
{
	char *str	 = PG_GETARG_CSTRING(0);
	int32 typmod = PG_GETARG_INT32(2);
	float values[VEC32_MAX_DIM];
	int	  dim = 0;
	char *p	  = str;

	/* Expect leading '[' */
	while (*p && (*p == ' ' || *p == '\t'))
		p++;
	if (*p != '[')
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("vec32 must start with \"[\"")));
	p++;

	/* Parse comma-separated floats */
	while (*p)
	{
		while (*p == ' ' || *p == '\t')
			p++;
		if (*p == ']')
			break;

		if (dim > 0)
		{
			if (*p != ',')
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
						 errmsg("expected \",\" or \"]\" in vec32")));
			p++;
			while (*p == ' ' || *p == '\t')
				p++;
		}

		if (dim >= VEC32_MAX_DIM)
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("vec32 cannot have more than %d dimensions",
							VEC32_MAX_DIM)));

		char *end;
		errno	  = 0;
		float val = strtof(p, &end);
		if (end == p || errno == ERANGE)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("invalid input syntax for type vec32: \"%s\"",
							str)));

		mkt_pg_check_value_finite(val);
		values[dim++] = val;
		p			  = end;
	}

	if (*p != ']')
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("vec32 must end with \"]\"")));
	p++;

	/* Check no trailing content */
	while (*p == ' ' || *p == '\t')
		p++;
	if (*p != '\0')
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("unexpected characters after \"]\" in vec32")));

	if (dim < 1)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("vec32 must have at least 1 dimension")));

	mkt_pg_check_expected_dim(dim, typmod);

	Vec32 *result = mkt_pg_vec32_alloc(dim);
	memcpy(result->x, values, dim * sizeof(float));

	PG_RETURN_VEC32_P(result);
}

PG_FUNCTION_INFO_V1(mkt_vec32_out);

Datum
mkt_vec32_out(PG_FUNCTION_ARGS)
{
	Vec32		  *v = PG_GETARG_VEC32_P(0);
	StringInfoData buf;

	initStringInfo(&buf);
	appendStringInfoChar(&buf, '[');

	for (int i = 0; i < v->dim; i++)
	{
		if (i > 0)
			appendStringInfoChar(&buf, ',');
		appendStringInfo(&buf, "%g", v->x[i]);
	}

	appendStringInfoChar(&buf, ']');

	PG_RETURN_CSTRING(buf.data);
}

PG_FUNCTION_INFO_V1(mkt_vec32_typmod_in);

Datum
mkt_vec32_typmod_in(PG_FUNCTION_ARGS)
{
	ArrayType *ta = PG_GETARG_ARRAYTYPE_P(0);
	int		   n;
	int32	  *tl = ArrayGetIntegerTypmods(ta, &n);

	if (n != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid type modifier")));

	int dim = tl[0];

	mkt_pg_check_dim_valid(dim);

	PG_RETURN_INT32(dim);
}

/* ----------------------------------------------------------------
 * Distance functions
 * ---------------------------------------------------------------- */

PG_FUNCTION_INFO_V1(mkt_l2_distance);

Datum
mkt_l2_distance(PG_FUNCTION_ARGS)
{
	Vec32 *a = PG_GETARG_VEC32_P(0);
	Vec32 *b = PG_GETARG_VEC32_P(1);

	mkt_pg_check_dims_match(a->dim, b->dim);

	Vec32Ref ra	 = Vec32ToRef(a);
	Vec32Ref rb	 = Vec32ToRef(b);
	float	 d2	 = mkt_distance_l2(ra, rb);
	double	 res = sqrt((double)d2);

	PG_RETURN_FLOAT8(res);
}

PG_FUNCTION_INFO_V1(mkt_inner_product);

Datum
mkt_inner_product(PG_FUNCTION_ARGS)
{
	Vec32 *a = PG_GETARG_VEC32_P(0);
	Vec32 *b = PG_GETARG_VEC32_P(1);

	mkt_pg_check_dims_match(a->dim, b->dim);

	Vec32Ref ra	 = Vec32ToRef(a);
	Vec32Ref rb	 = Vec32ToRef(b);
	float	 nip = mkt_distance_ip(ra, rb);
	/* mkt_distance_ip returns -dot, so negate to get actual dot */
	double res = (double)(-nip);

	PG_RETURN_FLOAT8(res);
}

PG_FUNCTION_INFO_V1(mkt_cosine_distance);

Datum
mkt_cosine_distance(PG_FUNCTION_ARGS)
{
	Vec32 *a = PG_GETARG_VEC32_P(0);
	Vec32 *b = PG_GETARG_VEC32_P(1);

	mkt_pg_check_dims_match(a->dim, b->dim);

	Vec32Ref ra	 = Vec32ToRef(a);
	Vec32Ref rb	 = Vec32ToRef(b);
	double	 res = (double)mkt_distance_cosine(ra, rb);

	PG_RETURN_FLOAT8(res);
}

/* Private distance functions for operators (preserve metric semantics) */

PG_FUNCTION_INFO_V1(mkt_vec32_l2_squared_distance);

Datum
mkt_vec32_l2_squared_distance(PG_FUNCTION_ARGS)
{
	Vec32 *a = PG_GETARG_VEC32_P(0);
	Vec32 *b = PG_GETARG_VEC32_P(1);

	mkt_pg_check_dims_match(a->dim, b->dim);

	Vec32Ref ra	 = Vec32ToRef(a);
	Vec32Ref rb	 = Vec32ToRef(b);
	double	 res = (double)mkt_distance_l2(ra, rb);

	PG_RETURN_FLOAT8(res);
}

PG_FUNCTION_INFO_V1(mkt_vec32_negative_inner_product);

Datum
mkt_vec32_negative_inner_product(PG_FUNCTION_ARGS)
{
	Vec32 *a = PG_GETARG_VEC32_P(0);
	Vec32 *b = PG_GETARG_VEC32_P(1);

	mkt_pg_check_dims_match(a->dim, b->dim);

	Vec32Ref ra	 = Vec32ToRef(a);
	Vec32Ref rb	 = Vec32ToRef(b);
	double	 res = (double)mkt_distance_ip(ra, rb);

	PG_RETURN_FLOAT8(res);
}

/* ----------------------------------------------------------------
 * Comparison functions (for btree opclass)
 * ---------------------------------------------------------------- */

static int
vec32_cmp_internal(Vec32 *a, Vec32 *b)
{
	int min_dim = (a->dim < b->dim) ? a->dim : b->dim;

	for (int i = 0; i < min_dim; i++)
	{
		if (a->x[i] < b->x[i])
			return -1;
		if (a->x[i] > b->x[i])
			return 1;
	}

	if (a->dim < b->dim)
		return -1;
	if (a->dim > b->dim)
		return 1;

	return 0;
}

PG_FUNCTION_INFO_V1(mkt_vec32_cmp);

Datum
mkt_vec32_cmp(PG_FUNCTION_ARGS)
{
	Vec32 *a = PG_GETARG_VEC32_P(0);
	Vec32 *b = PG_GETARG_VEC32_P(1);
	PG_RETURN_INT32(vec32_cmp_internal(a, b));
}

PG_FUNCTION_INFO_V1(mkt_vec32_lt);

Datum
mkt_vec32_lt(PG_FUNCTION_ARGS)
{
	Vec32 *a = PG_GETARG_VEC32_P(0);
	Vec32 *b = PG_GETARG_VEC32_P(1);
	PG_RETURN_BOOL(vec32_cmp_internal(a, b) < 0);
}

PG_FUNCTION_INFO_V1(mkt_vec32_le);

Datum
mkt_vec32_le(PG_FUNCTION_ARGS)
{
	Vec32 *a = PG_GETARG_VEC32_P(0);
	Vec32 *b = PG_GETARG_VEC32_P(1);
	PG_RETURN_BOOL(vec32_cmp_internal(a, b) <= 0);
}

PG_FUNCTION_INFO_V1(mkt_vec32_eq);

Datum
mkt_vec32_eq(PG_FUNCTION_ARGS)
{
	Vec32 *a = PG_GETARG_VEC32_P(0);
	Vec32 *b = PG_GETARG_VEC32_P(1);
	PG_RETURN_BOOL(vec32_cmp_internal(a, b) == 0);
}

PG_FUNCTION_INFO_V1(mkt_vec32_ne);

Datum
mkt_vec32_ne(PG_FUNCTION_ARGS)
{
	Vec32 *a = PG_GETARG_VEC32_P(0);
	Vec32 *b = PG_GETARG_VEC32_P(1);
	PG_RETURN_BOOL(vec32_cmp_internal(a, b) != 0);
}

PG_FUNCTION_INFO_V1(mkt_vec32_ge);

Datum
mkt_vec32_ge(PG_FUNCTION_ARGS)
{
	Vec32 *a = PG_GETARG_VEC32_P(0);
	Vec32 *b = PG_GETARG_VEC32_P(1);
	PG_RETURN_BOOL(vec32_cmp_internal(a, b) >= 0);
}

PG_FUNCTION_INFO_V1(mkt_vec32_gt);

Datum
mkt_vec32_gt(PG_FUNCTION_ARGS)
{
	Vec32 *a = PG_GETARG_VEC32_P(0);
	Vec32 *b = PG_GETARG_VEC32_P(1);
	PG_RETURN_BOOL(vec32_cmp_internal(a, b) > 0);
}

/* ----------------------------------------------------------------
 * Utility functions
 * ---------------------------------------------------------------- */

PG_FUNCTION_INFO_V1(mkt_vec32_dims);

Datum
mkt_vec32_dims(PG_FUNCTION_ARGS)
{
	Vec32 *v = PG_GETARG_VEC32_P(0);
	PG_RETURN_INT32(v->dim);
}

PG_FUNCTION_INFO_V1(mkt_pg_vec32_norm);

Datum
mkt_pg_vec32_norm(PG_FUNCTION_ARGS)
{
	Vec32 *v   = PG_GETARG_VEC32_P(0);
	double res = 0.0;

	for (int i = 0; i < v->dim; i++)
		res += (double)v->x[i] * (double)v->x[i];

	PG_RETURN_FLOAT8(sqrt(res));
}

/* ----------------------------------------------------------------
 * Cast functions
 * ---------------------------------------------------------------- */

PG_FUNCTION_INFO_V1(mkt_vec32);

Datum
mkt_vec32(PG_FUNCTION_ARGS)
{
	Vec32 *v	  = PG_GETARG_VEC32_P(0);
	int32  typmod = PG_GETARG_INT32(1);

	mkt_pg_check_expected_dim(v->dim, typmod);

	PG_RETURN_VEC32_P(v);
}

PG_FUNCTION_INFO_V1(mkt_array_to_vec32);

Datum
mkt_array_to_vec32(PG_FUNCTION_ARGS)
{
	ArrayType *arr	  = PG_GETARG_ARRAYTYPE_P(0);
	int32	   typmod = PG_GETARG_INT32(1);
	Oid		   elemtype;
	int		   ndims;
	int		  *dims;
	int		   dim;
	Datum	  *elems;
	bool	  *nulls;
	int		   nelems;

	elemtype = ARR_ELEMTYPE(arr);
	ndims	 = ARR_NDIM(arr);

	if (ndims > 1)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("array must be 1-D")));

	if (ndims == 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("array must not be empty")));

	dims = ARR_DIMS(arr);
	dim	 = dims[0];

	mkt_pg_check_dim_valid(dim);
	mkt_pg_check_expected_dim(dim, typmod);

	deconstruct_array(
			arr,
			elemtype,
			(elemtype == FLOAT4OID) ? 4 : 8,
			(elemtype == FLOAT4OID) ? true : true,
			(elemtype == FLOAT4OID) ? TYPALIGN_INT : TYPALIGN_DOUBLE,
			&elems,
			&nulls,
			&nelems);

	Vec32 *result = mkt_pg_vec32_alloc(dim);

	for (int i = 0; i < dim; i++)
	{
		if (nulls[i])
			ereport(ERROR,
					(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
					 errmsg("array must not contain nulls")));

		float val;
		if (elemtype == FLOAT4OID)
			val = DatumGetFloat4(elems[i]);
		else
			val = (float)DatumGetFloat8(elems[i]);

		mkt_pg_check_value_finite(val);
		result->x[i] = val;
	}

	PG_RETURN_VEC32_P(result);
}

PG_FUNCTION_INFO_V1(mkt_vec32_to_float4);

Datum
mkt_vec32_to_float4(PG_FUNCTION_ARGS)
{
	Vec32 *v	 = PG_GETARG_VEC32_P(0);
	Datum *elems = (Datum *)palloc(v->dim * sizeof(Datum));

	for (int i = 0; i < v->dim; i++)
		elems[i] = Float4GetDatum(v->x[i]);

	ArrayType *result = construct_array(
			elems, v->dim, FLOAT4OID, sizeof(float4), true, TYPALIGN_INT);

	pfree(elems);

	PG_RETURN_ARRAYTYPE_P(result);
}
