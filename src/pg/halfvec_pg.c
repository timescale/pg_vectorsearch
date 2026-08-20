/*
 * mkt_pg_halfvec.c - PostgreSQL functions for mkt.halfvec type
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
#define MKT_HALFVEC_STACK_DIM 1024

/* ----------------------------------------------------------------
 * Helper: convert halfvec to VectorRef with float32 buffer.
 * Uses stack allocation for small dims, palloc for large.
 * ---------------------------------------------------------------- */

#define HALFVEC_TO_FLOAT(hv, buf_name, ref_name)                          \
	float  buf_name##_stack[MKT_HALFVEC_STACK_DIM];                       \
	float *buf_name = ((hv)->dim <= MKT_HALFVEC_STACK_DIM)                \
							? buf_name##_stack                            \
							: (float *)palloc((hv)->dim * sizeof(float)); \
	mkt_half_to_float_array((hv)->x, buf_name, (hv)->dim);                \
	VectorRef ref_name = {.data = buf_name, .dim = (Dimension)(hv)->dim}

#define HALFVEC_FREE_BUF(hv, buf_name)         \
	do                                         \
	{                                          \
		if ((hv)->dim > MKT_HALFVEC_STACK_DIM) \
			pfree(buf_name);                   \
	} while (0)

/* ----------------------------------------------------------------
 * Type I/O
 * ---------------------------------------------------------------- */

PG_FUNCTION_INFO_V1(mkt_halfvec_in);

Datum
mkt_halfvec_in(PG_FUNCTION_ARGS)
{
	char *str	 = PG_GETARG_CSTRING(0);
	int32 typmod = PG_GETARG_INT32(2);
	float values[MKT_VECTOR_MAX_DIM];
	int	  dim = 0;
	char *p	  = str;

	while (*p && (*p == ' ' || *p == '\t'))
		p++;
	if (*p != '[')
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("halfvec must start with \"[\"")));
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
						 errmsg("expected \",\" or \"]\" in halfvec")));
			p++;
			while (*p == ' ' || *p == '\t')
				p++;
		}

		if (dim >= MKT_VECTOR_MAX_DIM)
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("halfvec cannot have more than %d dimensions",
							MKT_VECTOR_MAX_DIM)));

		char *end;
		errno	  = 0;
		float val = strtof(p, &end);
		if (end == p || errno == ERANGE)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("invalid input syntax for type halfvec: \"%s\"",
							str)));

		mkt_pg_check_value_finite(val);
		values[dim++] = val;
		p			  = end;
	}

	if (*p != ']')
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("halfvec must end with \"]\"")));
	p++;

	while (*p == ' ' || *p == '\t')
		p++;
	if (*p != '\0')
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("unexpected characters after \"]\" in halfvec")));

	if (dim < 1)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("halfvec must have at least 1 dimension")));

	mkt_pg_check_expected_dim(dim, typmod);

	MktHalfVector *result = mkt_pg_halfvec_alloc(dim);
	mkt_float_to_half_array(values, result->x, dim);

	PG_RETURN_MKT_HALFVEC_P(result);
}

PG_FUNCTION_INFO_V1(mkt_halfvec_out);

Datum
mkt_halfvec_out(PG_FUNCTION_ARGS)
{
	MktHalfVector *v = PG_GETARG_MKT_HALFVEC_P(0);
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

PG_FUNCTION_INFO_V1(mkt_halfvec_typmod_in);

Datum
mkt_halfvec_typmod_in(PG_FUNCTION_ARGS)
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

PG_FUNCTION_INFO_V1(mkt_halfvec_l2_distance);

Datum
mkt_halfvec_l2_distance(PG_FUNCTION_ARGS)
{
	MktHalfVector *a = PG_GETARG_MKT_HALFVEC_P(0);
	MktHalfVector *b = PG_GETARG_MKT_HALFVEC_P(1);

	mkt_pg_check_dims_match(a->dim, b->dim);

	HALFVEC_TO_FLOAT(a, buf_a, ra);
	HALFVEC_TO_FLOAT(b, buf_b, rb);

	double res = sqrt((double)mkt_distance_l2(ra, rb));

	HALFVEC_FREE_BUF(a, buf_a);
	HALFVEC_FREE_BUF(b, buf_b);

	PG_RETURN_FLOAT8(res);
}

PG_FUNCTION_INFO_V1(mkt_halfvec_inner_product);

Datum
mkt_halfvec_inner_product(PG_FUNCTION_ARGS)
{
	MktHalfVector *a = PG_GETARG_MKT_HALFVEC_P(0);
	MktHalfVector *b = PG_GETARG_MKT_HALFVEC_P(1);

	mkt_pg_check_dims_match(a->dim, b->dim);

	HALFVEC_TO_FLOAT(a, buf_a, ra);
	HALFVEC_TO_FLOAT(b, buf_b, rb);

	double res = (double)(-mkt_distance_ip(ra, rb));

	HALFVEC_FREE_BUF(a, buf_a);
	HALFVEC_FREE_BUF(b, buf_b);

	PG_RETURN_FLOAT8(res);
}

PG_FUNCTION_INFO_V1(mkt_halfvec_cosine_distance);

Datum
mkt_halfvec_cosine_distance(PG_FUNCTION_ARGS)
{
	MktHalfVector *a = PG_GETARG_MKT_HALFVEC_P(0);
	MktHalfVector *b = PG_GETARG_MKT_HALFVEC_P(1);

	mkt_pg_check_dims_match(a->dim, b->dim);

	HALFVEC_TO_FLOAT(a, buf_a, ra);
	HALFVEC_TO_FLOAT(b, buf_b, rb);

	double res = (double)mkt_distance_cosine(ra, rb);

	HALFVEC_FREE_BUF(a, buf_a);
	HALFVEC_FREE_BUF(b, buf_b);

	PG_RETURN_FLOAT8(res);
}

/* Private distance functions for operators */

PG_FUNCTION_INFO_V1(mkt_halfvec_l2_squared_distance);

Datum
mkt_halfvec_l2_squared_distance(PG_FUNCTION_ARGS)
{
	MktHalfVector *a = PG_GETARG_MKT_HALFVEC_P(0);
	MktHalfVector *b = PG_GETARG_MKT_HALFVEC_P(1);

	mkt_pg_check_dims_match(a->dim, b->dim);

	HALFVEC_TO_FLOAT(a, buf_a, ra);
	HALFVEC_TO_FLOAT(b, buf_b, rb);

	double res = (double)mkt_distance_l2(ra, rb);

	HALFVEC_FREE_BUF(a, buf_a);
	HALFVEC_FREE_BUF(b, buf_b);

	PG_RETURN_FLOAT8(res);
}

PG_FUNCTION_INFO_V1(mkt_halfvec_negative_inner_product);

Datum
mkt_halfvec_negative_inner_product(PG_FUNCTION_ARGS)
{
	MktHalfVector *a = PG_GETARG_MKT_HALFVEC_P(0);
	MktHalfVector *b = PG_GETARG_MKT_HALFVEC_P(1);

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
mkt_halfvec_cmp_internal(MktHalfVector *a, MktHalfVector *b)
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

PG_FUNCTION_INFO_V1(mkt_halfvec_cmp);

Datum
mkt_halfvec_cmp(PG_FUNCTION_ARGS)
{
	MktHalfVector *a = PG_GETARG_MKT_HALFVEC_P(0);
	MktHalfVector *b = PG_GETARG_MKT_HALFVEC_P(1);
	PG_RETURN_INT32(mkt_halfvec_cmp_internal(a, b));
}

PG_FUNCTION_INFO_V1(mkt_halfvec_lt);

Datum
mkt_halfvec_lt(PG_FUNCTION_ARGS)
{
	MktHalfVector *a = PG_GETARG_MKT_HALFVEC_P(0);
	MktHalfVector *b = PG_GETARG_MKT_HALFVEC_P(1);
	PG_RETURN_BOOL(mkt_halfvec_cmp_internal(a, b) < 0);
}

PG_FUNCTION_INFO_V1(mkt_halfvec_le);

Datum
mkt_halfvec_le(PG_FUNCTION_ARGS)
{
	MktHalfVector *a = PG_GETARG_MKT_HALFVEC_P(0);
	MktHalfVector *b = PG_GETARG_MKT_HALFVEC_P(1);
	PG_RETURN_BOOL(mkt_halfvec_cmp_internal(a, b) <= 0);
}

PG_FUNCTION_INFO_V1(mkt_halfvec_eq);

Datum
mkt_halfvec_eq(PG_FUNCTION_ARGS)
{
	MktHalfVector *a = PG_GETARG_MKT_HALFVEC_P(0);
	MktHalfVector *b = PG_GETARG_MKT_HALFVEC_P(1);
	PG_RETURN_BOOL(mkt_halfvec_cmp_internal(a, b) == 0);
}

PG_FUNCTION_INFO_V1(mkt_halfvec_ne);

Datum
mkt_halfvec_ne(PG_FUNCTION_ARGS)
{
	MktHalfVector *a = PG_GETARG_MKT_HALFVEC_P(0);
	MktHalfVector *b = PG_GETARG_MKT_HALFVEC_P(1);
	PG_RETURN_BOOL(mkt_halfvec_cmp_internal(a, b) != 0);
}

PG_FUNCTION_INFO_V1(mkt_halfvec_ge);

Datum
mkt_halfvec_ge(PG_FUNCTION_ARGS)
{
	MktHalfVector *a = PG_GETARG_MKT_HALFVEC_P(0);
	MktHalfVector *b = PG_GETARG_MKT_HALFVEC_P(1);
	PG_RETURN_BOOL(mkt_halfvec_cmp_internal(a, b) >= 0);
}

PG_FUNCTION_INFO_V1(mkt_halfvec_gt);

Datum
mkt_halfvec_gt(PG_FUNCTION_ARGS)
{
	MktHalfVector *a = PG_GETARG_MKT_HALFVEC_P(0);
	MktHalfVector *b = PG_GETARG_MKT_HALFVEC_P(1);
	PG_RETURN_BOOL(mkt_halfvec_cmp_internal(a, b) > 0);
}

/* ----------------------------------------------------------------
 * Utility functions
 * ---------------------------------------------------------------- */

PG_FUNCTION_INFO_V1(mkt_halfvec_dims);

Datum
mkt_halfvec_dims(PG_FUNCTION_ARGS)
{
	MktHalfVector *v = PG_GETARG_MKT_HALFVEC_P(0);
	PG_RETURN_INT32(v->dim);
}

PG_FUNCTION_INFO_V1(mkt_halfvec_norm);

Datum
mkt_halfvec_norm(PG_FUNCTION_ARGS)
{
	MktHalfVector *v   = PG_GETARG_MKT_HALFVEC_P(0);
	double		   res = 0.0;

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

PG_FUNCTION_INFO_V1(mkt_halfvec);

Datum
mkt_halfvec(PG_FUNCTION_ARGS)
{
	MktHalfVector *v	  = PG_GETARG_MKT_HALFVEC_P(0);
	int32		   typmod = PG_GETARG_INT32(1);

	mkt_pg_check_expected_dim(v->dim, typmod);

	PG_RETURN_MKT_HALFVEC_P(v);
}

PG_FUNCTION_INFO_V1(mkt_halfvec_to_vector);

Datum
mkt_halfvec_to_vector(PG_FUNCTION_ARGS)
{
	MktHalfVector *hv	  = PG_GETARG_MKT_HALFVEC_P(0);
	int32		   typmod = PG_GETARG_INT32(1);

	mkt_pg_check_expected_dim(hv->dim, typmod);

	MktVector *result = mkt_pg_vector_alloc(hv->dim);
	mkt_half_to_float_array(hv->x, result->x, hv->dim);

	PG_RETURN_MKT_VECTOR_P(result);
}

PG_FUNCTION_INFO_V1(mkt_vector_to_halfvec);

Datum
mkt_vector_to_halfvec(PG_FUNCTION_ARGS)
{
	MktVector *v	  = PG_GETARG_MKT_VECTOR_P(0);
	int32	   typmod = PG_GETARG_INT32(1);

	mkt_pg_check_expected_dim(v->dim, typmod);

	MktHalfVector *result = mkt_pg_halfvec_alloc(v->dim);
	mkt_float_to_half_array(v->x, result->x, v->dim);

	PG_RETURN_MKT_HALFVEC_P(result);
}

PG_FUNCTION_INFO_V1(mkt_array_to_halfvec);

Datum
mkt_array_to_halfvec(PG_FUNCTION_ARGS)
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

	MktHalfVector *result = mkt_pg_halfvec_alloc(dim);
	mkt_float_to_half_array(floats, result->x, dim);

	pfree(floats);

	PG_RETURN_MKT_HALFVEC_P(result);
}
