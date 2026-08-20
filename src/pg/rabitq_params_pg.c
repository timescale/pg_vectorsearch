/*
 * mkt_pg_rabitq_params.c - PostgreSQL functions for mkt.rabitq_params type
 *
 * Stores the orthogonal transform matrix P for RaBitQ encoding.
 * Generated via rabitq_params_generate(dim, seed); text I/O outputs
 * {dim:seed} since inspecting dim^2 floats isn't useful.
 */

#include <postgres.h>

#include <fmgr.h>
#include <lib/stringinfo.h>
#include <utils/builtins.h>
#include <utils/memutils.h>

#include "index/posting_page.h"
#include "quant/matrix.h"
#include "support_pg.h"

/*
 * The generator caps dim at MKT_INDEX_MAX_DIM (see
 * mkt_pg_check_rabitq_params_dim_valid). Prove at compile time that a
 * matrix that large still fits a single allocation, so raising the cap
 * without revisiting MKT_RABITQ_PARAMS_PG_SIZE breaks the build here
 * rather than silently over-allocating (or, on 32-bit, overflowing the
 * size computation) at run time.
 */
StaticAssertDecl(
		offsetof(RaBitQParamsPG, P) + (uint64_t)MKT_INDEX_MAX_DIM *
											  MKT_INDEX_MAX_DIM *
											  sizeof(float) <=
				MaxAllocSize,
		"rabitq_params matrix at MKT_INDEX_MAX_DIM exceeds MaxAllocSize; "
		"revisit MKT_RABITQ_PARAMS_PG_SIZE and the dim cap together");

/* ----------------------------------------------------------------
 * Type I/O
 * ---------------------------------------------------------------- */

PG_FUNCTION_INFO_V1(mkt_rabitq_params_in);

Datum
mkt_rabitq_params_in(PG_FUNCTION_ARGS)
{
	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("cannot parse rabitq_params from text"),
			 errhint("Use rabitq_params_generate(dim, seed) "
					 "instead.")));
	PG_RETURN_NULL();
}

PG_FUNCTION_INFO_V1(mkt_rabitq_params_out);

Datum
mkt_rabitq_params_out(PG_FUNCTION_ARGS)
{
	RaBitQParamsPG *p = PG_GETARG_RABITQ_PARAMS_P(0);
	StringInfoData	buf;

	initStringInfo(&buf);
	appendStringInfo(
			&buf, "{%d:%llu}", (int)p->dim, (unsigned long long)p->seed);

	PG_RETURN_CSTRING(buf.data);
}

/* ----------------------------------------------------------------
 * Generate function
 * ---------------------------------------------------------------- */

PG_FUNCTION_INFO_V1(mkt_rabitq_params_generate_pg);

Datum
mkt_rabitq_params_generate_pg(PG_FUNCTION_ARGS)
{
	int32 dim  = PG_GETARG_INT32(0);
	int64 seed = PG_GETARG_INT64(1);

	mkt_pg_check_rabitq_params_dim_valid(dim);

	Size			size   = MKT_RABITQ_PARAMS_PG_SIZE(dim);
	RaBitQParamsPG *result = (RaBitQParamsPG *)palloc0(size);
	SET_VARSIZE(result, size);
	result->dim	   = (int16_t)dim;
	result->unused = 0;
	result->seed   = (uint64_t)seed;

	int ret = mkt_random_orthogonal_matrix(
			result->P, (Dimension)dim, (uint64_t)seed);
	if (ret != 0)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("failed to generate orthogonal matrix"
						" for rabitq_params")));

	PG_RETURN_POINTER(result);
}

/* ----------------------------------------------------------------
 * Accessor functions
 * ---------------------------------------------------------------- */

PG_FUNCTION_INFO_V1(mkt_rabitq_params_dim);

Datum
mkt_rabitq_params_dim(PG_FUNCTION_ARGS)
{
	RaBitQParamsPG *p = PG_GETARG_RABITQ_PARAMS_P(0);
	PG_RETURN_INT32((int32)p->dim);
}

PG_FUNCTION_INFO_V1(mkt_rabitq_params_seed);

Datum
mkt_rabitq_params_seed(PG_FUNCTION_ARGS)
{
	RaBitQParamsPG *p = PG_GETARG_RABITQ_PARAMS_P(0);
	PG_RETURN_INT64((int64)p->seed);
}
