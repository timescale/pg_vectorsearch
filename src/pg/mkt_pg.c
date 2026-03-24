/*
 * mkt_pg.c - Meerkat PostgreSQL extension entry point
 */

#include <postgres.h>

#include <access/reloptions.h>
#include <catalog/namespace.h>
#include <fmgr.h>
#include <optimizer/paths.h>
#include <utils/guc.h>

#include "algo/distance.h"
#include "mkt_pg.h"

PG_MODULE_MAGIC;

/* GUC variables */
int	  mkt_distance_mode = MKT_DISTANCE_MODE_DEFAULT;
int	  mkt_nprobe		= 10;
int64 mkt_query_limit	= -1;

/* Hook chain */
static set_rel_pathlist_hook_type prev_pathlist_hook = NULL;

static const struct config_enum_entry mkt_distance_mode_options[] = {
		{"default", MKT_DISTANCE_MODE_DEFAULT, false},
		{"asymmetric", MKT_DISTANCE_MODE_ASYMMETRIC, false},
		{"symmetric", MKT_DISTANCE_MODE_SYMMETRIC, false},
		{NULL, 0, false},
};

/* Index reloptions */
relopt_kind mktann_relopt_kind;

static relopt_enum_elt_def distance_mode_relopt_members[] = {
		{"asymmetric", MKT_DISTANCE_MODE_ASYMMETRIC},
		{"symmetric", MKT_DISTANCE_MODE_SYMMETRIC},
		{NULL, 0},
};

/* ----------------------------------------------------------------
 * Planner hook: capture LIMIT for scan optimization
 *
 * root->limit_tuples is set by preprocess_limit() before
 * set_rel_pathlist runs. We store it so the scan can derive
 * a dynamic top-K budget from the query LIMIT.
 * ---------------------------------------------------------------- */
static void
mkt_set_rel_pathlist(
		PlannerInfo *root, RelOptInfo *rel, Index rti, RangeTblEntry *rte)
{
	if (prev_pathlist_hook)
		prev_pathlist_hook(root, rel, rti, rte);

	if (root->limit_tuples > 0)
		mkt_query_limit = (int64)root->limit_tuples;
	else
		mkt_query_limit = -1;
}

void _PG_init(void);

void
_PG_init(void)
{
	DefineCustomEnumVariable(
			"mkt.distance_mode",
			"RaBitQ distance computation mode.",
			"default (use index setting), asymmetric, or symmetric",
			&mkt_distance_mode,
			MKT_DISTANCE_MODE_DEFAULT,
			mkt_distance_mode_options,
			PGC_USERSET,
			0,
			NULL,
			NULL,
			NULL);

	DefineCustomIntVariable(
			"mkt.nprobe",
			"Number of clusters to probe during index scan.",
			NULL,
			&mkt_nprobe,
			MKT_DEFAULT_NPROBE,
			MKT_MIN_NPROBE,
			MKT_MAX_NPROBE,
			PGC_USERSET,
			0,
			NULL,
			NULL,
			NULL);

	MarkGUCPrefixReserved("mkt");

	mktann_relopt_kind = add_reloption_kind();
	add_enum_reloption(
			mktann_relopt_kind,
			"distance_mode",
			"RaBitQ distance computation mode",
			distance_mode_relopt_members,
			MKT_DISTANCE_MODE_ASYMMETRIC,
			"symmetric is faster but has larger estimation error",
			NoLock);
	add_int_reloption(
			mktann_relopt_kind,
			"fan_out",
			"Children per tree node (2-65535)",
			MKT_DEFAULT_FAN_OUT,
			MKT_MIN_FAN_OUT,
			MKT_MAX_FAN_OUT,
			NoLock);
	add_int_reloption(
			mktann_relopt_kind,
			"nlist",
			"Number of clusters (0 = auto: sqrt(ntuples))",
			MKT_DEFAULT_NLIST,
			MKT_MIN_NLIST,
			MKT_MAX_NLIST,
			NoLock);
	add_bool_reloption(
			mktann_relopt_kind,
			"centroid_compression",
			"Use RaBitQ compression for centroid pages",
			false,
			NoLock);

	mkt_distance_init();
	mkt_rabitq_init_simd();

	prev_pathlist_hook	  = set_rel_pathlist_hook;
	set_rel_pathlist_hook = mkt_set_rel_pathlist;
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

/* ----------------------------------------------------------------
 * Type OID helpers
 * ---------------------------------------------------------------- */

Oid
mkt_halfvec_type_oid(void)
{
	return TypenameGetTypid("halfvec");
}

/* ----------------------------------------------------------------
 * Metric identifier support functions (FUNCTION 2 in opclasses)
 * ---------------------------------------------------------------- */

PG_FUNCTION_INFO_V1(mktann_metric_l2);

Datum
mktann_metric_l2(PG_FUNCTION_ARGS)
{
	PG_RETURN_INT32(DISTANCE_L2);
}

PG_FUNCTION_INFO_V1(mktann_metric_ip);

Datum
mktann_metric_ip(PG_FUNCTION_ARGS)
{
	PG_RETURN_INT32(DISTANCE_INNER_PRODUCT);
}

PG_FUNCTION_INFO_V1(mktann_metric_cosine);

Datum
mktann_metric_cosine(PG_FUNCTION_ARGS)
{
	PG_RETURN_INT32(DISTANCE_COSINE);
}
