/*
 * mkt_pg_rabitq.c - PostgreSQL functions for mkt.rabitq type
 *
 * Type I/O, comparison operators, accessor functions, and encoding.
 * Text format: {bits:f_add:f_rescale} where bits are 0/1 characters.
 */

#include <postgres.h>

#include <fmgr.h>
#include <lib/stringinfo.h>
#include <utils/array.h>

#include "mkt_pg.h"
#include "quant/matrix.h"

/* ----------------------------------------------------------------
 * Type I/O
 * ---------------------------------------------------------------- */

PG_FUNCTION_INFO_V1(mkt_rabitq_in);

Datum
mkt_rabitq_in(PG_FUNCTION_ARGS)
{
	char   *str	   = PG_GETARG_CSTRING(0);
	int32	typmod = PG_GETARG_INT32(2);
	uint8_t bits_buf[MKT_VECTOR_MAX_DIM / 8];
	int		dim = 0;
	char   *p	= str;

	memset(bits_buf, 0, sizeof(bits_buf));

	/* Skip whitespace */
	while (*p == ' ' || *p == '\t')
		p++;

	/* Expect opening '{' */
	if (*p != '{')
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("rabitq must start with \"{\"")));
	p++;

	/* Parse bit characters */
	while (*p && *p != ':' && *p != '}')
	{
		if (dim >= MKT_VECTOR_MAX_DIM)
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("rabitq cannot have more than %d dimensions",
							MKT_VECTOR_MAX_DIM)));

		if (*p == '1')
			bits_buf[dim / 8] |= (1 << (dim % 8));
		else if (*p != '0')
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("invalid bit character '%c' in rabitq", *p)));

		dim++;
		p++;
	}

	if (dim < 1)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("rabitq must have at least 1 dimension")));

	/* Expect ':' separator */
	if (*p != ':')
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("expected \":\" after bits in rabitq")));
	p++;

	/* Parse f_add */
	char *end;
	errno		= 0;
	float f_add = strtof(p, &end);
	if (end == p || errno == ERANGE)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("invalid f_add value in rabitq: \"%s\"", str)));
	mkt_pg_check_value_finite(f_add);
	p = end;

	/* Expect ':' separator */
	if (*p != ':')
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("expected \":\" after f_add in rabitq")));
	p++;

	/* Parse f_rescale */
	errno			= 0;
	float f_rescale = strtof(p, &end);
	if (end == p || errno == ERANGE)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("invalid f_rescale value in rabitq: \"%s\"", str)));
	mkt_pg_check_value_finite(f_rescale);
	p = end;

	/* Expect closing '}' */
	if (*p != '}')
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("rabitq must end with \"}\"")));
	p++;

	/* Check no trailing content */
	while (*p == ' ' || *p == '\t')
		p++;
	if (*p != '\0')
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("unexpected characters after \"}\" in "
						"rabitq")));

	mkt_pg_check_expected_dim(dim, typmod);

	RaBitQVector *result = mkt_pg_rabitq_alloc(dim);
	result->f_add		 = f_add;
	result->f_rescale	 = f_rescale;
	memcpy(result->bits, bits_buf, MKT_RABITQ_BYTES(dim));

	PG_RETURN_RABITQ_P(result);
}

PG_FUNCTION_INFO_V1(mkt_rabitq_out);

Datum
mkt_rabitq_out(PG_FUNCTION_ARGS)
{
	RaBitQVector  *v = PG_GETARG_RABITQ_P(0);
	StringInfoData buf;

	initStringInfo(&buf);
	appendStringInfoChar(&buf, '{');

	for (int i = 0; i < v->dim; i++)
		appendStringInfoChar(
				&buf, (v->bits[i / 8] >> (i % 8)) & 1 ? '1' : '0');

	appendStringInfo(&buf, ":%.8g:%.8g}", v->f_add, v->f_rescale);

	PG_RETURN_CSTRING(buf.data);
}

PG_FUNCTION_INFO_V1(mkt_rabitq_typmod_in);

Datum
mkt_rabitq_typmod_in(PG_FUNCTION_ARGS)
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
 * Accessor functions
 * ---------------------------------------------------------------- */

PG_FUNCTION_INFO_V1(mkt_rabitq_dims);

Datum
mkt_rabitq_dims(PG_FUNCTION_ARGS)
{
	RaBitQVector *v = PG_GETARG_RABITQ_P(0);
	PG_RETURN_INT32(v->dim);
}

PG_FUNCTION_INFO_V1(mkt_rabitq_f_add);

Datum
mkt_rabitq_f_add(PG_FUNCTION_ARGS)
{
	RaBitQVector *v = PG_GETARG_RABITQ_P(0);
	PG_RETURN_FLOAT8((double)v->f_add);
}

PG_FUNCTION_INFO_V1(mkt_rabitq_f_rescale);

Datum
mkt_rabitq_f_rescale(PG_FUNCTION_ARGS)
{
	RaBitQVector *v = PG_GETARG_RABITQ_P(0);
	PG_RETURN_FLOAT8((double)v->f_rescale);
}

/* ----------------------------------------------------------------
 * Comparison functions (for btree opclass)
 *
 * Compare dim first, then memcmp on bits. f_add/f_rescale are
 * intentionally excluded -- bits are the quantized identity.
 * ---------------------------------------------------------------- */

static int
mkt_rabitq_cmp_internal(RaBitQVector *a, RaBitQVector *b)
{
	if (a->dim != b->dim)
		return (a->dim < b->dim) ? -1 : 1;

	int nbytes = MKT_RABITQ_BYTES(a->dim);
	return memcmp(a->bits, b->bits, nbytes);
}

PG_FUNCTION_INFO_V1(mkt_rabitq_cmp);

Datum
mkt_rabitq_cmp(PG_FUNCTION_ARGS)
{
	RaBitQVector *a = PG_GETARG_RABITQ_P(0);
	RaBitQVector *b = PG_GETARG_RABITQ_P(1);
	PG_RETURN_INT32(mkt_rabitq_cmp_internal(a, b));
}

PG_FUNCTION_INFO_V1(mkt_rabitq_lt);

Datum
mkt_rabitq_lt(PG_FUNCTION_ARGS)
{
	RaBitQVector *a = PG_GETARG_RABITQ_P(0);
	RaBitQVector *b = PG_GETARG_RABITQ_P(1);
	PG_RETURN_BOOL(mkt_rabitq_cmp_internal(a, b) < 0);
}

PG_FUNCTION_INFO_V1(mkt_rabitq_le);

Datum
mkt_rabitq_le(PG_FUNCTION_ARGS)
{
	RaBitQVector *a = PG_GETARG_RABITQ_P(0);
	RaBitQVector *b = PG_GETARG_RABITQ_P(1);
	PG_RETURN_BOOL(mkt_rabitq_cmp_internal(a, b) <= 0);
}

PG_FUNCTION_INFO_V1(mkt_rabitq_eq);

Datum
mkt_rabitq_eq(PG_FUNCTION_ARGS)
{
	RaBitQVector *a = PG_GETARG_RABITQ_P(0);
	RaBitQVector *b = PG_GETARG_RABITQ_P(1);
	PG_RETURN_BOOL(mkt_rabitq_cmp_internal(a, b) == 0);
}

PG_FUNCTION_INFO_V1(mkt_rabitq_ne);

Datum
mkt_rabitq_ne(PG_FUNCTION_ARGS)
{
	RaBitQVector *a = PG_GETARG_RABITQ_P(0);
	RaBitQVector *b = PG_GETARG_RABITQ_P(1);
	PG_RETURN_BOOL(mkt_rabitq_cmp_internal(a, b) != 0);
}

PG_FUNCTION_INFO_V1(mkt_rabitq_ge);

Datum
mkt_rabitq_ge(PG_FUNCTION_ARGS)
{
	RaBitQVector *a = PG_GETARG_RABITQ_P(0);
	RaBitQVector *b = PG_GETARG_RABITQ_P(1);
	PG_RETURN_BOOL(mkt_rabitq_cmp_internal(a, b) >= 0);
}

PG_FUNCTION_INFO_V1(mkt_rabitq_gt);

Datum
mkt_rabitq_gt(PG_FUNCTION_ARGS)
{
	RaBitQVector *a = PG_GETARG_RABITQ_P(0);
	RaBitQVector *b = PG_GETARG_RABITQ_P(1);
	PG_RETURN_BOOL(mkt_rabitq_cmp_internal(a, b) > 0);
}

/* ----------------------------------------------------------------
 * Cast function (typmod enforcement)
 * ---------------------------------------------------------------- */

PG_FUNCTION_INFO_V1(mkt_rabitq);

Datum
mkt_rabitq(PG_FUNCTION_ARGS)
{
	RaBitQVector *v		 = PG_GETARG_RABITQ_P(0);
	int32		  typmod = PG_GETARG_INT32(1);

	mkt_pg_check_expected_dim(v->dim, typmod);

	PG_RETURN_RABITQ_P(v);
}

/* ----------------------------------------------------------------
 * Encode function
 * ---------------------------------------------------------------- */

PG_FUNCTION_INFO_V1(mkt_rabitq_encode_pg);

Datum
mkt_rabitq_encode_pg(PG_FUNCTION_ARGS)
{
	MktVector	   *input	 = PG_GETARG_MKT_VECTOR_P(0);
	MktVector	   *centroid = PG_GETARG_MKT_VECTOR_P(1);
	RaBitQParamsPG *params	 = PG_GETARG_RABITQ_PARAMS_P(2);

	mkt_pg_check_dims_match(input->dim, centroid->dim);
	mkt_pg_check_dims_match(input->dim, params->dim);

	int dim = input->dim;

	/* Build RaBitQParams from the PG varlena's pre-computed matrix */
	RaBitQParams *rparams =
			mkt_rabitq_create_from_matrix(dim, params->seed, params->P);

	RaBitQVector *result = mkt_pg_rabitq_alloc(dim);

	VectorRef	input_ref	 = MktVectorToRef(input);
	VectorRef	centroid_ref = MktVectorToRef(centroid);
	RaBitQData *data		 = MKT_RABITQ_DATA(result);

	int ret = mkt_rabitq_encode_into(rparams, input_ref, centroid_ref, data);
	if (ret != 0)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("rabitq encoding failed")));

	PG_RETURN_RABITQ_P(result);
}
