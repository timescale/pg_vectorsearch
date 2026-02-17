/*
 * mkt_pg.c - Meerkat PostgreSQL extension entry point
 */

#include <postgres.h>

#include <fmgr.h>
#include <utils/guc.h>

#include "algo/distance.h"
#include "mkt_pg.h"

PG_MODULE_MAGIC;

/* GUC variables */
int mkt_distance_mode = MKT_DISTANCE_MODE_ASYMMETRIC;

static const struct config_enum_entry mkt_distance_mode_options[] = {
		{"asymmetric", MKT_DISTANCE_MODE_ASYMMETRIC, false},
		{"symmetric", MKT_DISTANCE_MODE_SYMMETRIC, false},
		{NULL, 0, false},
};

void _PG_init(void);

void
_PG_init(void)
{
	DefineCustomEnumVariable(
			"mkt.distance_mode",
			"RaBitQ distance computation mode.",
			"asymmetric (accurate) or symmetric (4-12x faster)",
			&mkt_distance_mode,
			MKT_DISTANCE_MODE_ASYMMETRIC,
			mkt_distance_mode_options,
			PGC_USERSET,
			0,
			NULL,
			NULL,
			NULL);

	MarkGUCPrefixReserved("mkt");

	mkt_distance_init();
	mkt_rabitq_init_simd();
}

/* ----------------------------------------------------------------
 * Validation helpers
 * ---------------------------------------------------------------- */

void
mkt_pg_check_dim_valid(int dim)
{
	if (dim < 1)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("vector must have at least 1 dimension")));
	if (dim > MKT_VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("vector cannot have more than %d dimensions",
						MKT_VECTOR_MAX_DIM)));
}

void
mkt_pg_check_dims_match(int dim_a, int dim_b)
{
	if (dim_a != dim_b)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("different vector dimensions %d and %d",
						dim_a,
						dim_b)));
}

void
mkt_pg_check_expected_dim(int actual, int expected)
{
	if (expected != -1 && actual != expected)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("expected %d dimensions, not %d", expected, actual)));
}

void
mkt_pg_check_value_finite(float val)
{
	if (isinf(val))
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("infinite value not allowed in vector")));
	if (isnan(val))
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("NaN value not allowed in vector")));
}
