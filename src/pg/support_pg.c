/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * vs_pg.c - pg_vectorsearch PostgreSQL extension entry point
 */

#include <postgres.h>

#include "vs_config.h"

#include <access/reloptions.h>
#include <catalog/namespace.h>
#include <fmgr.h>
#include <utils/builtins.h>
#include <utils/guc.h>

#include "algo/distance.h"
#include "algo/kmeans.h"
#include "explain.h"
#include "git_commit.h"
#include "index/index_build.h"
#include "index/posting_page.h"
#include "index/query_scan.h"
#include "pg/bufstorage.h"
#include "scan.h"
#include "scan_bound.h"
#include "support_pg.h"

PG_MODULE_MAGIC;

/* GUC variables */
int			prism_distance_mode			= VS_DISTANCE_MODE_DEFAULT;
int			prism_nprobe				= 0;
int			prism_query_limit			= 0;
int			prism_fastscan_bits			= 16;
bool		prism_rerank				= true;
bool		prism_log_build_stats		= false;
double		prism_centroid_error_scale	= 0.0;
double		prism_centroid_beam_scale	= 0.5;
int			prism_leaf_refine_threshold = 0;
static bool prism_recent_buffers		= true;
double		prism_probe_expand			= 2.0;
static int	prism_rerank_pool			= 0;

static void
vs_recent_buffers_assign_hook(bool newval, void *extra)
{
	vs_pg_storage_set_recent_buffers(newval);
}

static void
vs_probe_expand_assign_hook(double newval, void *extra)
{
	prism_query_set_probe_expand(newval);
}

static void
vs_rerank_pool_assign_hook(int newval, void *extra)
{
	prism_query_set_rerank_pool((int32_t)newval);
}

static const struct config_enum_entry vs_distance_mode_options[] = {
		{"default", VS_DISTANCE_MODE_DEFAULT, false},
		{"asymmetric", VS_DISTANCE_MODE_ASYMMETRIC, false},
		{"symmetric", VS_DISTANCE_MODE_SYMMETRIC, false},
		{NULL, 0, false},
};

static const struct config_enum_entry vs_fastscan_bits_options[] = {
		{"8", 8, false},
		{"16", 16, false},
		{NULL, 0, false},
};

/* Index reloptions */
relopt_kind prism_relopt_kind;

static relopt_enum_elt_def distance_mode_relopt_members[] = {
		{"asymmetric", VS_DISTANCE_MODE_ASYMMETRIC},
		{"symmetric", VS_DISTANCE_MODE_SYMMETRIC},
		{NULL, 0},
};

/* "true"/"false" are accepted aliases for "on"/"off". */
static relopt_enum_elt_def centroid_compression_relopt_members[] = {
		{"auto", PRISM_CENTROID_COMPRESSION_AUTO},
		{"on", PRISM_CENTROID_COMPRESSION_ON},
		{"true", PRISM_CENTROID_COMPRESSION_ON},
		{"off", PRISM_CENTROID_COMPRESSION_OFF},
		{"false", PRISM_CENTROID_COMPRESSION_OFF},
		{NULL, 0},
};

/* Shared by the `fastscan` and `centroid_fastscan` options. */
static relopt_enum_elt_def fastscan_mode_relopt_members[] = {
		{"auto", PRISM_FASTSCAN_MODE_AUTO},
		{"on", PRISM_FASTSCAN_MODE_ON},
		{"true", PRISM_FASTSCAN_MODE_ON},
		{"off", PRISM_FASTSCAN_MODE_OFF},
		{"false", PRISM_FASTSCAN_MODE_OFF},
		{NULL, 0},
};

void _PG_init(void);

void
_PG_init(void)
{
	DefineCustomEnumVariable(
			VS_GUC_PREFIX ".distance_mode",
			"RaBitQ distance computation mode.",
			"default (use index setting), asymmetric, or symmetric",
			&prism_distance_mode,
			VS_DISTANCE_MODE_DEFAULT,
			vs_distance_mode_options,
			PGC_USERSET,
			0,
			NULL,
			NULL,
			NULL);

	DefineCustomIntVariable(
			VS_GUC_PREFIX ".nprobe",
			"Number of clusters to probe per query.",
			"0 derives it from the index's cluster count "
			"(~0.5*sqrt(nlist), targeting ~0.95 recall). Lower it for "
			"speed, raise it for recall.",
			&prism_nprobe,
			0,
			0,
			10000,
			PGC_USERSET,
			0,
			NULL,
			NULL,
			NULL);

	DefineCustomIntVariable(
			VS_GUC_PREFIX ".query_limit",
			"Caps the top-k an index scan is sized for (0 = no cap).",
			"A scan sizes its top-k from the LIMIT above it, inflated by "
			"the planner's selectivity estimate for any filter the "
			"executor applies above it, and bounded by work_mem -- so a "
			"query wanting more rows than that budget affords is answered "
			"by raising work_mem. With no LIMIT to size from, the query "
			"has asked for every row in order and the scan is sized for "
			"what work_mem affords or the table's estimated row count, "
			"whichever is smaller. Set this to cap that: it lowers the "
			"sizing when it is below what the query asked for and is "
			"ignored otherwise, which bounds a query that has no LIMIT or "
			"one set far above the rows actually read. No scan is sized "
			"below 10 rows.",
			&prism_query_limit,
			0,
			0,
			INT_MAX,
			PGC_USERSET,
			0,
			NULL,
			NULL,
			NULL);

	DefineCustomIntVariable(
			VS_GUC_PREFIX ".leaf_refine_threshold",
			"Sample-per-leaf count below which a subsampled build refines "
			"the leaf centroids on the full table.",
			"The k-means sample trains each leaf's encode reference; the "
			"reference's error shrinks with the leaf's sample count "
			"(stderr ~ spread/sqrt(n)), so with enough samples per leaf the "
			"full-table refine scan buys no recall. Below this many samples "
			"per leaf the sample mean is noisy and the build re-centers the "
			"references from the whole table (one extra scan). 0 (the "
			"default) disables refinement; a large value refines whenever "
			"the sample was bounded below the table.",
			&prism_leaf_refine_threshold,
			0,
			0,
			INT_MAX,
			PGC_USERSET,
			0,
			NULL,
			NULL,
			NULL);

	DefineCustomEnumVariable(
			VS_GUC_PREFIX ".fastscan_bits",
			"Fastscan LUT quantization bits.",
			"8 is faster, 16 is more accurate",
			&prism_fastscan_bits,
			16,
			vs_fastscan_bits_options,
			PGC_USERSET,
			0,
			NULL,
			NULL,
			NULL);

	DefineCustomBoolVariable(
			VS_GUC_PREFIX ".rerank",
			"Enable reranking with exact distances.",
			NULL,
			&prism_rerank,
			true,
			PGC_USERSET,
			0,
			NULL,
			NULL,
			NULL);

	DefineCustomBoolVariable(
			VS_GUC_PREFIX ".log_build_stats",
			"Log per-phase index-build resource statistics.",
			"When on, each build phase logs its elapsed time, build-heap "
			"usage, and CPU/maxrss (via the server's ShowUsage), plus a final "
			"summary, at LOG. Off by default; modeled on the core btree "
			"log_btree_build_stats developer option. The up-front "
			"planned-allocation line is logged regardless of this setting.",
			&prism_log_build_stats,
			false,
			PGC_SUSET,
			GUC_NOT_IN_SAMPLE,
			NULL,
			NULL,
			NULL);

	DefineCustomRealVariable(
			VS_GUC_PREFIX ".centroid_error_scale",
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
			&prism_centroid_error_scale,
			0.0,
			0.0,
			10.0,
			PGC_USERSET,
			0,
			NULL,
			NULL,
			NULL);

	DefineCustomRealVariable(
			VS_GUC_PREFIX ".centroid_beam_scale",
			"Intermediate centroid beam width as a fraction of nprobe.",
			"Sets the beam width at the intermediate tree levels to this "
			"fraction of nprobe; the leaf level always returns the full "
			"nprobe, and the beam never drops below a small floor of "
			"candidates (or nprobe itself, whichever is less) — at small "
			"nprobe a scaled-down beam saves next to nothing and "
			"mis-routes. 0.5 (default) matches the benchmark-tuned "
			"routing shape. 1.0 keeps the full beam (beam_width = nprobe) "
			"at every level -- the widest and most recall-conservative "
			"setting. Smaller values score fewer centroids and are faster "
			"at high nprobe, with a small recall risk if a near leaf's "
			"ancestor falls outside the narrowed beam.",
			&prism_centroid_beam_scale,
			0.5,
			0.01,
			1.0,
			PGC_USERSET,
			0,
			NULL,
			NULL,
			NULL);

	DefineCustomBoolVariable(
			VS_GUC_PREFIX ".recent_buffers",
			"Re-pin index pages via a backend-local buffer-id cache.",
			"Skips the shared buffer-mapping hash lookup (a large share of "
			"warm scan CPU) by remembering each block's buffer id and "
			"re-pinning it with ReadRecentBuffer. Stale ids fall back to a "
			"normal read and self-heal. Costs 4 bytes per index block per "
			"backend.",
			&prism_recent_buffers,
			true,
			PGC_USERSET,
			0,
			NULL,
			vs_recent_buffers_assign_hook,
			NULL);

	DefineCustomRealVariable(
			VS_GUC_PREFIX ".probe_expand",
			"Probe-candidate expansion factor for exact centroid re-rank.",
			"Routes ceil(nprobe * expand) leaf candidates through the "
			"centroid beam, re-ranks them by exact query-centroid distance "
			"(the full-precision rotated centroid on each cluster's first "
			"posting page), and scans only the best nprobe in that order. "
			"Corrects the probe-order noise of compressed (RaBitQ) centroid "
			"routing; skipped automatically for indexes with exact "
			"float/half centroids. 1.0 means no expansion (identity). "
			"Gains saturate around 2 (the default); the extra routed "
			"candidates are capped at 256, which bounds the overhead at "
			"large nprobe while retaining nearly all of the recall gain.",
			&prism_probe_expand,
			2.0,
			1.0,
			16.0,
			PGC_USERSET,
			0,
			NULL,
			vs_probe_expand_assign_hook,
			NULL);

	DefineCustomIntVariable(
			VS_GUC_PREFIX ".rerank_pool",
			"Max candidates to exact-rerank per query (0 = automatic, "
			"-1 = unlimited).",
			"Rerank only the most promising candidates by approximate "
			"distance, bounding the exact-distance heap fetches. 0 (the "
			"default) caps at 3 * k * nprobe^0.15, growing to 1/8th of "
			"the candidate buffer when noisy distance estimates flood "
			"it; -1 reranks every threshold survivor; positive values "
			"set an absolute cap. The effective cap is never below the "
			"query's k, so results are never truncated.",
			&prism_rerank_pool,
			0,
			-1,
			1000000,
			PGC_USERSET,
			0,
			NULL,
			vs_rerank_pool_assign_hook,
			NULL);

	MarkGUCPrefixReserved(VS_GUC_PREFIX);

	vs_cblas_pin_single_thread();

	prism_relopt_kind = add_reloption_kind();
	add_enum_reloption(
			prism_relopt_kind,
			"distance_mode",
			"RaBitQ distance computation mode",
			distance_mode_relopt_members,
			VS_DISTANCE_MODE_ASYMMETRIC,
			"symmetric is faster but has larger estimation error",
			NoLock);
	add_int_reloption(
			prism_relopt_kind,
			"fan_out",
			"Children per tree node (2-255)",
			PRISM_DEFAULT_FAN_OUT,
			PRISM_MIN_FAN_OUT,
			PRISM_MAX_FAN_OUT,
			NoLock);
	add_int_reloption(
			prism_relopt_kind,
			"nlist",
			"Number of IVF clusters (0 = auto from sqrt(rows))",
			PRISM_DEFAULT_NLIST,
			PRISM_MIN_NLIST,
			PRISM_MAX_NLIST,
			NoLock);
	add_int_reloption(
			prism_relopt_kind,
			"target_pages",
			"Posting pages each list rests at (1-64)",
			PRISM_DEFAULT_TARGET_PAGES,
			PRISM_MIN_TARGET_PAGES,
			PRISM_MAX_TARGET_PAGES,
			NoLock);
	add_int_reloption(
			prism_relopt_kind,
			"kmeans_nredo",
			"K-means restarts for cluster quality (1 = no restart)",
			1,
			1,
			20,
			NoLock);
	add_real_reloption(
			prism_relopt_kind,
			"soar_lambda",
			"SOAR replication lambda (0 = off)",
			PRISM_DEFAULT_SOAR_LAMBDA,
			0.0,
			100.0,
			NoLock);
	add_real_reloption(
			prism_relopt_kind,
			"boundary_epsilon",
			"Boundary replication gap threshold (0 = off)",
			PRISM_DEFAULT_BOUNDARY_EPSILON,
			0.0,
			100.0,
			NoLock);
	add_enum_reloption(
			prism_relopt_kind,
			"centroid_compression",
			"RaBitQ compression for centroid pages",
			centroid_compression_relopt_members,
			PRISM_CENTROID_COMPRESSION_AUTO,
			"auto compresses for L2/cosine and skips inner product",
			NoLock);
	add_enum_reloption(
			prism_relopt_kind,
			"fastscan",
			"VPSHUFB fastscan posting page format",
			fastscan_mode_relopt_members,
			PRISM_FASTSCAN_MODE_AUTO,
			"auto uses it up to the dimension where a group fits a page",
			NoLock);
	add_enum_reloption(
			prism_relopt_kind,
			"centroid_fastscan",
			"FASTSCAN-format centroid pages",
			fastscan_mode_relopt_members,
			PRISM_FASTSCAN_MODE_AUTO,
			"auto follows the resolved centroid compression and the "
			"dimension limit; on errors where either is unavailable",
			NoLock);

	vs_distance_init();
	vs_rabitq_init_simd();
	prism_explain_init();
	prism_scan_bound_init();
}

/* ----------------------------------------------------------------
 * Validation helpers
 * ---------------------------------------------------------------- */

void
vs_pg_check_dim_valid(int dim)
{
	if (dim < 1)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("vec32 must have at least 1 dimension")));
	if (dim > VEC32_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("vec32 cannot have more than %d dimensions",
						VEC32_MAX_DIM)));
}

/*
 * rabitq_params_generate() builds a dim x dim orthogonal matrix by
 * Gram-Schmidt -- O(dim^3) in time, O(dim^2) in memory -- and is callable by
 * any role, so an oversized dim is a CPU/memory denial-of-service vector (at
 * the generic vector cap a single call runs for minutes at 100% CPU). Bound
 * it by PRISM_INDEX_MAX_DIM: generating params for a dimension no prism index
 * can hold is pointless, so the largest indexable dimension is the natural
 * ceiling, and it tracks the index limit automatically.
 */
void
vs_pg_check_rabitq_params_dim_valid(int dim)
{
	vs_pg_check_dim_valid(dim);
	if (dim > PRISM_INDEX_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("cannot generate rabitq_params for more than %d "
						"dimensions",
						PRISM_INDEX_MAX_DIM),
				 errhint("Building the transform matrix is O(dim^3); no "
						 "prism index supports more dimensions than this.")));
}

void
vs_pg_check_dims_match(int dim_a, int dim_b)
{
	if (dim_a != dim_b)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("different vec32 dimensions %d and %d",
						dim_a,
						dim_b)));
}

void
vs_pg_check_expected_dim(int actual, int expected)
{
	if (expected != -1 && actual != expected)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("expected %d dimensions, not %d", expected, actual)));
}

void
vs_pg_check_value_finite(float val)
{
	if (isinf(val))
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("infinite value not allowed in vec32")));
	if (isnan(val))
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("NaN value not allowed in vec32")));
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
PG_FUNCTION_INFO_V1(vs_git_commit);

Datum
vs_git_commit(PG_FUNCTION_ARGS)
{
	PG_RETURN_TEXT_P(cstring_to_text(VS_GIT_COMMIT));
}

/*
 * Return the extension version the binary was built as. Comes from
 * meson.build via vs_config.h, the single source of truth for the
 * version string.
 */
PG_FUNCTION_INFO_V1(vs_extension_version);

Datum
vs_extension_version(PG_FUNCTION_ARGS)
{
	PG_RETURN_TEXT_P(cstring_to_text(VS_VERSION));
}

/* ----------------------------------------------------------------
 * Metric identifier support functions (FUNCTION 2 in opclasses)
 * ---------------------------------------------------------------- */

PG_FUNCTION_INFO_V1(prism_metric_l2);

Datum
prism_metric_l2(PG_FUNCTION_ARGS)
{
	PG_RETURN_INT32(DISTANCE_L2);
}

PG_FUNCTION_INFO_V1(prism_metric_ip);

Datum
prism_metric_ip(PG_FUNCTION_ARGS)
{
	PG_RETURN_INT32(DISTANCE_INNER_PRODUCT);
}

PG_FUNCTION_INFO_V1(prism_metric_cosine);

Datum
prism_metric_cosine(PG_FUNCTION_ARGS)
{
	PG_RETURN_INT32(DISTANCE_COSINE);
}
