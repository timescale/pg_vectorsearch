/*
 * mkt_pg.c - Meerkat PostgreSQL extension entry point
 */

#include <postgres.h>

#include <access/reloptions.h>
#include <catalog/namespace.h>
#include <fmgr.h>
#include <utils/builtins.h>
#include <utils/guc.h>

#include "algo/distance.h"
#include "algo/kmeans.h"
#include "git_commit.h"
#include "mkt_pg.h"
#include "mktann_explain.h"
#include "mktann_storage.h"

PG_MODULE_MAGIC;

/* GUC variables */
int			mkt_distance_mode		 = MKT_DISTANCE_MODE_DEFAULT;
int			mkt_nprobe				 = 10;
int			mkt_query_limit			 = 0;
int			mkt_fastscan_bits		 = 16;
bool		mkt_rerank				 = true;
bool		mkt_log_build_stats		 = false;
double		mkt_centroid_error_scale = 0.0;
double		mkt_centroid_beam_scale	 = 0.25;
int			mkt_leaf_refine_iters	 = 2;
static bool mkt_recent_buffers		 = true;

static void
mkt_recent_buffers_assign_hook(bool newval, void *extra)
{
	mktann_storage_set_recent_buffers(newval);
}

static const struct config_enum_entry mkt_distance_mode_options[] = {
		{"default", MKT_DISTANCE_MODE_DEFAULT, false},
		{"asymmetric", MKT_DISTANCE_MODE_ASYMMETRIC, false},
		{"symmetric", MKT_DISTANCE_MODE_SYMMETRIC, false},
		{NULL, 0, false},
};

static const struct config_enum_entry mkt_fastscan_bits_options[] = {
		{"8", 8, false},
		{"16", 16, false},
		{NULL, 0, false},
};

/* Index reloptions */
relopt_kind mktann_relopt_kind;

static relopt_enum_elt_def distance_mode_relopt_members[] = {
		{"asymmetric", MKT_DISTANCE_MODE_ASYMMETRIC},
		{"symmetric", MKT_DISTANCE_MODE_SYMMETRIC},
		{NULL, 0},
};

/* "true"/"false" are accepted aliases for "on"/"off". */
static relopt_enum_elt_def centroid_compression_relopt_members[] = {
		{"auto", MKT_CENTROID_COMPRESSION_AUTO},
		{"on", MKT_CENTROID_COMPRESSION_ON},
		{"true", MKT_CENTROID_COMPRESSION_ON},
		{"off", MKT_CENTROID_COMPRESSION_OFF},
		{"false", MKT_CENTROID_COMPRESSION_OFF},
		{NULL, 0},
};

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
			"Number of clusters to probe per query.",
			NULL,
			&mkt_nprobe,
			10,
			1,
			10000,
			PGC_USERSET,
			0,
			NULL,
			NULL,
			NULL);

	DefineCustomIntVariable(
			"mkt.query_limit",
			"Maximum number of results per query (0 = auto).",
			NULL,
			&mkt_query_limit,
			0,
			0,
			100000,
			PGC_USERSET,
			0,
			NULL,
			NULL,
			NULL);

	DefineCustomIntVariable(
			"mkt.leaf_refine_iters",
			"Full-table leaf-centroid refinement passes for memory-bounded "
			"builds.",
			"When maintenance_work_mem caps the k-means sample below the "
			"ideal, "
			"the tree structure is built from the subsample and the leaf "
			"centroids are then refined on the whole table this many passes "
			"(0 disables refinement). Only takes effect when the build is "
			"sample-bounded.",
			&mkt_leaf_refine_iters,
			2,
			0,
			10,
			PGC_USERSET,
			0,
			NULL,
			NULL,
			NULL);

	DefineCustomEnumVariable(
			"mkt.fastscan_bits",
			"Fastscan LUT quantization bits.",
			"8 is faster, 16 is more accurate",
			&mkt_fastscan_bits,
			16,
			mkt_fastscan_bits_options,
			PGC_USERSET,
			0,
			NULL,
			NULL,
			NULL);

	DefineCustomBoolVariable(
			"mkt.rerank",
			"Enable reranking with exact distances.",
			NULL,
			&mkt_rerank,
			true,
			PGC_USERSET,
			0,
			NULL,
			NULL,
			NULL);

	DefineCustomBoolVariable(
			"mkt.log_build_stats",
			"Log per-phase index-build resource statistics.",
			"When on, each build phase logs its elapsed time, build-heap "
			"usage, and CPU/maxrss (via the server's ShowUsage), plus a final "
			"summary, at LOG. Off by default; modeled on the core btree "
			"log_btree_build_stats developer option. The up-front "
			"planned-allocation line is logged regardless of this setting.",
			&mkt_log_build_stats,
			false,
			PGC_SUSET,
			GUC_NOT_IN_SAMPLE,
			NULL,
			NULL,
			NULL);

	DefineCustomRealVariable(
			"mkt.centroid_error_scale",
			"Centroid-search beam width, as a multiple of the RaBitQ "
			"distance-error margin.",
			"During centroid routing each candidate centroid has an "
			"approximate (RaBitQ-quantized) distance plus an error margin; "
			"this multiplies that margin when deciding which centroids the "
			"beam keeps at each tree level. 0 (default) ignores the margin "
			"and "
			"keeps only the closest centroids by point estimate -- the "
			"narrowest and fastest beam. 1 widens the beam to also keep "
			"centroids whose error interval still overlaps the cutoff -- ones "
			"that might rank among the closest once quantization error is "
			"accounted for -- the most recall-conservative setting, at the "
			"cost of scoring more centroids. Larger values widen it further. "
			"Applies to both compressed centroid formats (RaBitQ and "
			"FASTSCAN); float/half centroid pages have exact distances (no "
			"error margin) and are unaffected.",
			&mkt_centroid_error_scale,
			0.0,
			0.0,
			10.0,
			PGC_USERSET,
			0,
			NULL,
			NULL,
			NULL);

	DefineCustomRealVariable(
			"mkt.centroid_beam_scale",
			"Intermediate centroid beam width as a fraction of nprobe.",
			"Sets the beam width at the intermediate tree levels to this "
			"fraction of nprobe; the leaf level always returns the full "
			"nprobe. 0.25 (default) keeps a narrow intermediate beam, scoring "
			"fewer centroids per level during routing -- the leaf level still "
			"returns nprobe because beam_width*fan_out covers the top-nprobe "
			"leaves. 1.0 keeps the full beam (beam_width = nprobe) at every "
			"level -- the widest and most recall-conservative setting. "
			"Smaller "
			"values score fewer centroids and are faster at high nprobe, with "
			"a small recall risk if a near leaf's ancestor falls outside the "
			"narrowed beam.",
			&mkt_centroid_beam_scale,
			0.25,
			0.01,
			1.0,
			PGC_USERSET,
			0,
			NULL,
			NULL,
			NULL);

	DefineCustomBoolVariable(
			"mkt.recent_buffers",
			"Re-pin index pages via a backend-local buffer-id cache.",
			"Skips the shared buffer-mapping hash lookup (a large share of "
			"warm scan CPU) by remembering each block's buffer id and "
			"re-pinning it with ReadRecentBuffer. Stale ids fall back to a "
			"normal read and self-heal. Costs 4 bytes per index block per "
			"backend.",
			&mkt_recent_buffers,
			true,
			PGC_USERSET,
			0,
			NULL,
			mkt_recent_buffers_assign_hook,
			NULL);

	MarkGUCPrefixReserved("mkt");

	mkt_cblas_pin_single_thread();

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
			"Children per tree node (2-255)",
			MKTANN_DEFAULT_FAN_OUT,
			MKTANN_MIN_FAN_OUT,
			MKTANN_MAX_FAN_OUT,
			NoLock);
	add_int_reloption(
			mktann_relopt_kind,
			"nlist",
			"Number of IVF clusters (0 = auto from sqrt(rows))",
			MKTANN_DEFAULT_NLIST,
			MKTANN_MIN_NLIST,
			MKTANN_MAX_NLIST,
			NoLock);
	add_int_reloption(
			mktann_relopt_kind,
			"kmeans_nredo",
			"K-means restarts for cluster quality (1 = no restart)",
			1,
			1,
			20,
			NoLock);
	add_real_reloption(
			mktann_relopt_kind,
			"soar_lambda",
			"SOAR replication lambda (0 = off)",
			0.0,
			0.0,
			100.0,
			NoLock);
	add_real_reloption(
			mktann_relopt_kind,
			"boundary_epsilon",
			"Boundary replication gap threshold (0 = off)",
			0.0,
			0.0,
			100.0,
			NoLock);
	add_enum_reloption(
			mktann_relopt_kind,
			"centroid_compression",
			"RaBitQ compression for centroid pages",
			centroid_compression_relopt_members,
			MKT_CENTROID_COMPRESSION_AUTO,
			"auto compresses for L2/cosine and skips inner product",
			NoLock);
	add_bool_reloption(
			mktann_relopt_kind,
			"fastscan",
			"Use VPSHUFB fastscan posting page format",
			false,
			NoLock);
	add_bool_reloption(
			mktann_relopt_kind,
			"centroid_fastscan",
			"Emit FASTSCAN-format centroid pages "
			"(requires centroid_compression=true)",
			false,
			NoLock);

	mkt_distance_init();
	mkt_rabitq_init_simd();
	mktann_explain_init();
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
 * Build identity
 * ---------------------------------------------------------------- */

/*
 * Return the git commit the extension was built from. The string is
 * baked into the binary via vcs_tag at build time (see meson.build).
 * Rekall and other tooling use this to identify a build for
 * benchmark reports.
 */
PG_FUNCTION_INFO_V1(mkt_git_commit);

Datum
mkt_git_commit(PG_FUNCTION_ARGS)
{
	PG_RETURN_TEXT_P(cstring_to_text(MKT_GIT_COMMIT));
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
