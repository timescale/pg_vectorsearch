/*
 * vec16_pg.c - PostgreSQL functions for mkt.vec16 type
 *
 * Type I/O, distance functions, comparison operators, casts.
 * Distance functions convert to float32 then delegate to SIMD core.
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

/* Stack threshold for float32 conversion buffers (4 KB = 1024 floats) */
#define VEC16_STACK_DIM 1024

/* ----------------------------------------------------------------
 * Helper: convert vec16 to Vec32Ref with float32 buffer.
 * Uses stack allocation for small dims, palloc for large.
 * ---------------------------------------------------------------- */

#define HALFVEC_TO_FLOAT(hv, buf_name, ref_name)                          \
	float  buf_name##_stack[VEC16_STACK_DIM];                             \
	float *buf_name = ((hv)->dim <= VEC16_STACK_DIM)                      \
							? buf_name##_stack                            \
							: (float *)palloc((hv)->dim * sizeof(float)); \
	mkt_half_to_float_array((hv)->x, buf_name, (hv)->dim);                \
	Vec32Ref ref_name = {.data = buf_name, .dim = (Dimension)(hv)->dim}

#define HALFVEC_FREE_BUF(hv, buf_name)   \
	do                                   \
	{                                    \
		if ((hv)->dim > VEC16_STACK_DIM) \
			pfree(buf_name);             \
	} while (0)

/* ----------------------------------------------------------------
 * Type I/O
 * ---------------------------------------------------------------- */

PG_FUNCTION_INFO_V1(mkt_vec16_in);

Datum
mkt_vec16_in(PG_FUNCTION_ARGS)
{
	char *str	 = PG_GETARG_CSTRING(0);
	int32 typmod = PG_GETARG_INT32(2);
	float values[VEC32_MAX_DIM];
	int	  dim = 0;
	char *p	  = str;

	while (*p && (*p == ' ' || *p == '\t'))
		p++;
	if (*p != '[')
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("vec16 must start with \"[\"")));
	p++;

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
						 errmsg("expected \",\" or \"]\" in vec16")));
			p++;
			while (*p == ' ' || *p == '\t')
				p++;
		}

		if (dim >= VEC32_MAX_DIM)
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("vec16 cannot have more than %d dimensions",
							VEC32_MAX_DIM)));

		char *end;
		errno	  = 0;
		float val = strtof(p, &end);
		if (end == p || errno == ERANGE)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("invalid input syntax for type vec16: \"%s\"",
							str)));

		mkt_pg_check_value_finite(val);
		values[dim++] = val;
		p			  = end;
	}

	if (*p != ']')
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("vec16 must end with \"]\"")));
	p++;

	while (*p == ' ' || *p == '\t')
		p++;
	if (*p != '\0')
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("unexpected characters after \"]\" in vec16")));

	if (dim < 1)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("vec16 must have at least 1 dimension")));

	mkt_pg_check_expected_dim(dim, typmod);

	Vec16 *result = mkt_pg_vec16_alloc(dim);
	mkt_float_to_half_array(values, result->x, dim);

	PG_RETURN_VEC16_P(result);
}

PG_FUNCTION_INFO_V1(mkt_vec16_out);

Datum
mkt_vec16_out(PG_FUNCTION_ARGS)
{
	Vec16		  *v = PG_GETARG_VEC16_P(0);
	StringInfoData buf;

	initStringInfo(&buf);
	appendStringInfoChar(&buf, '[');

	for (int i = 0; i < v->dim; i++)
	{
		if (i > 0)
			appendStringInfoChar(&buf, ',');
		appendStringInfo(&buf, "%g", mkt_half_to_float(v->x[i]));
	}

	appendStringInfoChar(&buf, ']');

	PG_RETURN_CSTRING(buf.data);
}

PG_FUNCTION_INFO_V1(mkt_vec16_typmod_in);

Datum
mkt_vec16_typmod_in(PG_FUNCTION_ARGS)
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

PG_FUNCTION_INFO_V1(mkt_vec16_l2_distance);

Datum
mkt_vec16_l2_distance(PG_FUNCTION_ARGS)
{
	Vec16 *a = PG_GETARG_VEC16_P(0);
	Vec16 *b = PG_GETARG_VEC16_P(1);

	mkt_pg_check_dims_match(a->dim, b->dim);

	HALFVEC_TO_FLOAT(a, buf_a, ra);
	HALFVEC_TO_FLOAT(b, buf_b, rb);

	double res = sqrt((double)mkt_distance_l2(ra, rb));

	HALFVEC_FREE_BUF(a, buf_a);
	HALFVEC_FREE_BUF(b, buf_b);

	PG_RETURN_FLOAT8(res);
}

PG_FUNCTION_INFO_V1(mkt_vec16_inner_product);

Datum
mkt_vec16_inner_product(PG_FUNCTION_ARGS)
{
	Vec16 *a = PG_GETARG_VEC16_P(0);
	Vec16 *b = PG_GETARG_VEC16_P(1);

	mkt_pg_check_dims_match(a->dim, b->dim);

	HALFVEC_TO_FLOAT(a, buf_a, ra);
	HALFVEC_TO_FLOAT(b, buf_b, rb);

	double res = (double)(-mkt_distance_ip(ra, rb));

	HALFVEC_FREE_BUF(a, buf_a);
	HALFVEC_FREE_BUF(b, buf_b);

	PG_RETURN_FLOAT8(res);
}

PG_FUNCTION_INFO_V1(mkt_vec16_cosine_distance);

Datum
mkt_vec16_cosine_distance(PG_FUNCTION_ARGS)
{
	Vec16 *a = PG_GETARG_VEC16_P(0);
	Vec16 *b = PG_GETARG_VEC16_P(1);

	mkt_pg_check_dims_match(a->dim, b->dim);

	HALFVEC_TO_FLOAT(a, buf_a, ra);
	HALFVEC_TO_FLOAT(b, buf_b, rb);

	double res = (double)mkt_distance_cosine(ra, rb);

	HALFVEC_FREE_BUF(a, buf_a);
	HALFVEC_FREE_BUF(b, buf_b);

	PG_RETURN_FLOAT8(res);
}

/* Private distance functions for operators */

PG_FUNCTION_INFO_V1(mkt_vec16_l2_squared_distance);

Datum
mkt_vec16_l2_squared_distance(PG_FUNCTION_ARGS)
{
	Vec16 *a = PG_GETARG_VEC16_P(0);
	Vec16 *b = PG_GETARG_VEC16_P(1);

	mkt_pg_check_dims_match(a->dim, b->dim);

	HALFVEC_TO_FLOAT(a, buf_a, ra);
	HALFVEC_TO_FLOAT(b, buf_b, rb);

	double res = (double)mkt_distance_l2(ra, rb);

	HALFVEC_FREE_BUF(a, buf_a);
	HALFVEC_FREE_BUF(b, buf_b);

	PG_RETURN_FLOAT8(res);
}

PG_FUNCTION_INFO_V1(mkt_vec16_negative_inner_product);

Datum
mkt_vec16_negative_inner_product(PG_FUNCTION_ARGS)
{
	Vec16 *a = PG_GETARG_VEC16_P(0);
	Vec16 *b = PG_GETARG_VEC16_P(1);

	mkt_pg_check_dims_match(a->dim, b->dim);

	HALFVEC_TO_FLOAT(a, buf_a, ra);
	HALFVEC_TO_FLOAT(b, buf_b, rb);

	double res = (double)mkt_distance_ip(ra, rb);

	HALFVEC_FREE_BUF(a, buf_a);
	HALFVEC_FREE_BUF(b, buf_b);

	PG_RETURN_FLOAT8(res);
}

/* ----------------------------------------------------------------
 * Comparison functions (for btree opclass)
 * ---------------------------------------------------------------- */

static int
vec16_cmp_internal(Vec16 *a, Vec16 *b)
{
	int min_dim = (a->dim < b->dim) ? a->dim : b->dim;

	for (int i = 0; i < min_dim; i++)
	{
		float fa = mkt_half_to_float(a->x[i]);
		float fb = mkt_half_to_float(b->x[i]);
		if (fa < fb)
			return -1;
		if (fa > fb)
			return 1;
	}

	if (a->dim < b->dim)
		return -1;
	if (a->dim > b->dim)
		return 1;

	return 0;
}

PG_FUNCTION_INFO_V1(mkt_vec16_cmp);

Datum
mkt_vec16_cmp(PG_FUNCTION_ARGS)
{
	Vec16 *a = PG_GETARG_VEC16_P(0);
	Vec16 *b = PG_GETARG_VEC16_P(1);
	PG_RETURN_INT32(vec16_cmp_internal(a, b));
}

PG_FUNCTION_INFO_V1(mkt_vec16_lt);

Datum
mkt_vec16_lt(PG_FUNCTION_ARGS)
{
	Vec16 *a = PG_GETARG_VEC16_P(0);
	Vec16 *b = PG_GETARG_VEC16_P(1);
	PG_RETURN_BOOL(vec16_cmp_internal(a, b) < 0);
}

PG_FUNCTION_INFO_V1(mkt_vec16_le);

Datum
mkt_vec16_le(PG_FUNCTION_ARGS)
{
	Vec16 *a = PG_GETARG_VEC16_P(0);
	Vec16 *b = PG_GETARG_VEC16_P(1);
	PG_RETURN_BOOL(vec16_cmp_internal(a, b) <= 0);
}

PG_FUNCTION_INFO_V1(mkt_vec16_eq);

Datum
mkt_vec16_eq(PG_FUNCTION_ARGS)
{
	Vec16 *a = PG_GETARG_VEC16_P(0);
	Vec16 *b = PG_GETARG_VEC16_P(1);
	PG_RETURN_BOOL(vec16_cmp_internal(a, b) == 0);
}

PG_FUNCTION_INFO_V1(mkt_vec16_ne);

Datum
mkt_vec16_ne(PG_FUNCTION_ARGS)
{
	Vec16 *a = PG_GETARG_VEC16_P(0);
	Vec16 *b = PG_GETARG_VEC16_P(1);
	PG_RETURN_BOOL(vec16_cmp_internal(a, b) != 0);
}

PG_FUNCTION_INFO_V1(mkt_vec16_ge);

Datum
mkt_vec16_ge(PG_FUNCTION_ARGS)
{
	Vec16 *a = PG_GETARG_VEC16_P(0);
	Vec16 *b = PG_GETARG_VEC16_P(1);
	PG_RETURN_BOOL(vec16_cmp_internal(a, b) >= 0);
}

PG_FUNCTION_INFO_V1(mkt_vec16_gt);

Datum
mkt_vec16_gt(PG_FUNCTION_ARGS)
{
	Vec16 *a = PG_GETARG_VEC16_P(0);
	Vec16 *b = PG_GETARG_VEC16_P(1);
	PG_RETURN_BOOL(vec16_cmp_internal(a, b) > 0);
}

/* ----------------------------------------------------------------
 * Utility functions
 * ---------------------------------------------------------------- */

PG_FUNCTION_INFO_V1(mkt_vec16_dims);

Datum
mkt_vec16_dims(PG_FUNCTION_ARGS)
{
	Vec16 *v = PG_GETARG_VEC16_P(0);
	PG_RETURN_INT32(v->dim);
}

PG_FUNCTION_INFO_V1(mkt_vec16_norm);

Datum
mkt_vec16_norm(PG_FUNCTION_ARGS)
{
	Vec16 *v   = PG_GETARG_VEC16_P(0);
	double res = 0.0;

	for (int i = 0; i < v->dim; i++)
	{
		double val = (double)mkt_half_to_float(v->x[i]);
		res += val * val;
	}

	PG_RETURN_FLOAT8(sqrt(res));
}

/* ----------------------------------------------------------------
 * Cast functions
 * ---------------------------------------------------------------- */

PG_FUNCTION_INFO_V1(mkt_vec16);

Datum
mkt_vec16(PG_FUNCTION_ARGS)
{
	Vec16 *v	  = PG_GETARG_VEC16_P(0);
	int32  typmod = PG_GETARG_INT32(1);

	mkt_pg_check_expected_dim(v->dim, typmod);

	PG_RETURN_VEC16_P(v);
}

PG_FUNCTION_INFO_V1(mkt_vec16_to_vec32);

Datum
mkt_vec16_to_vec32(PG_FUNCTION_ARGS)
{
	Vec16 *hv	  = PG_GETARG_VEC16_P(0);
	int32  typmod = PG_GETARG_INT32(1);

	mkt_pg_check_expected_dim(hv->dim, typmod);

	Vec32 *result = mkt_pg_vec32_alloc(hv->dim);
	mkt_half_to_float_array(hv->x, result->x, hv->dim);

	PG_RETURN_VEC32_P(result);
}

PG_FUNCTION_INFO_V1(mkt_vec32_to_vec16);

Datum
mkt_vec32_to_vec16(PG_FUNCTION_ARGS)
{
	Vec32 *v	  = PG_GETARG_VEC32_P(0);
	int32  typmod = PG_GETARG_INT32(1);

	mkt_pg_check_expected_dim(v->dim, typmod);

	Vec16 *result = mkt_pg_vec16_alloc(v->dim);
	mkt_float_to_half_array(v->x, result->x, v->dim);

	PG_RETURN_VEC16_P(result);
}

PG_FUNCTION_INFO_V1(mkt_array_to_vec16);

Datum
mkt_array_to_vec16(PG_FUNCTION_ARGS)
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

	/* Convert via float32 intermediate */
	float *floats = (float *)palloc(dim * sizeof(float));
	for (int i = 0; i < dim; i++)
	{
		if (nulls[i])
			ereport(ERROR,
					(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
					 errmsg("array must not contain nulls")));

		if (elemtype == FLOAT4OID)
			floats[i] = DatumGetFloat4(elems[i]);
		else
			floats[i] = (float)DatumGetFloat8(elems[i]);

		mkt_pg_check_value_finite(floats[i]);
	}

	Vec16 *result = mkt_pg_vec16_alloc(dim);
	mkt_float_to_half_array(floats, result->x, dim);

	pfree(floats);

	PG_RETURN_VEC16_P(result);
}
