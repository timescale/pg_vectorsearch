/*
 * mkt_pg.c - Meerkat PostgreSQL extension entry point
 */

#include <postgres.h>

#include "mkt_config.h"

#include <access/reloptions.h>
#include <catalog/namespace.h>
#include <fmgr.h>
#include <utils/builtins.h>
#include <utils/guc.h>

#include "algo/distance.h"
#include "algo/kmeans.h"
#include "git_commit.h"
#include "index/posting_page.h"
#include "index/query_scan.h"
#include "mktann_explain.h"
#include "mktann_storage.h"
#include "support_pg.h"

PG_MODULE_MAGIC;

/* GUC variables */
int			  mkt_distance_mode			= MKT_DISTANCE_MODE_DEFAULT;
int			  mkt_nprobe				= 0;
int			  mkt_query_limit			= 0;
int			  mkt_fastscan_bits			= 16;
bool		  mkt_rerank				= true;
bool		  mkt_log_build_stats		= false;
double		  mkt_centroid_error_scale	= 0.0;
double		  mkt_centroid_beam_scale	= 0.5;
int			  mkt_leaf_refine_threshold = 0;
double		  mkt_rerank_pool_cost_scale = 0.0; /* mkt.rerank_pool_cost_scale */
static bool	  mkt_recent_buffers		= true;
static double mkt_probe_expand			= 2.0;
static int	  mkt_rerank_pool			= 0;

static void
mkt_recent_buffers_assign_hook(bool newval, void *extra)
{
	mktann_storage_set_recent_buffers(newval);
}

static void
mkt_probe_expand_assign_hook(double newval, void *extra)
{
	mkt_query_set_probe_expand(newval);
}

static void
mkt_rerank_pool_assign_hook(int newval, void *extra)
{
	mkt_query_set_rerank_pool((int32_t)newval);
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

/* Shared by the `fastscan` and `centroid_fastscan` options. */
static relopt_enum_elt_def fastscan_mode_relopt_members[] = {
		{"auto", MKT_FASTSCAN_MODE_AUTO},
		{"on", MKT_FASTSCAN_MODE_ON},
		{"true", MKT_FASTSCAN_MODE_ON},
		{"off", MKT_FASTSCAN_MODE_OFF},
		{"false", MKT_FASTSCAN_MODE_OFF},
		{NULL, 0},
};

void _PG_init(void);

void
_PG_init(void)
{
	DefineCustomEnumVariable(
			MKT_EXTENSION_SCHEMA ".distance_mode",
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
			MKT_EXTENSION_SCHEMA ".nprobe",
			"Number of clusters to probe per query.",
			"0 derives it from the index's cluster count "
			"(~0.5*sqrt(nlist), targeting ~0.95 recall). Lower it for "
			"speed, raise it for recall.",
			&mkt_nprobe,
			0,
			0,
			10000,
			PGC_USERSET,
			0,
			NULL,
			NULL,
			NULL);

	DefineCustomIntVariable(
			MKT_EXTENSION_SCHEMA ".query_limit",
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
			MKT_EXTENSION_SCHEMA ".leaf_refine_threshold",
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
			&mkt_leaf_refine_threshold,
			0,
			0,
			INT_MAX,
			PGC_USERSET,
			0,
			NULL,
			NULL,
			NULL);

	DefineCustomEnumVariable(
			MKT_EXTENSION_SCHEMA ".fastscan_bits",
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
			MKT_EXTENSION_SCHEMA ".rerank",
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
			MKT_EXTENSION_SCHEMA ".log_build_stats",
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
			MKT_EXTENSION_SCHEMA ".centroid_error_scale",
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
			MKT_EXTENSION_SCHEMA ".centroid_beam_scale",
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
			&mkt_centroid_beam_scale,
			0.5,
			0.01,
			1.0,
			PGC_USERSET,
			0,
			NULL,
			NULL,
			NULL);

	DefineCustomBoolVariable(
			MKT_EXTENSION_SCHEMA ".recent_buffers",
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

	DefineCustomRealVariable(
			MKT_EXTENSION_SCHEMA ".probe_expand",
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
			&mkt_probe_expand,
			2.0,
			1.0,
			16.0,
			PGC_USERSET,
			0,
			NULL,
			mkt_probe_expand_assign_hook,
			NULL);

	DefineCustomIntVariable(
			MKT_EXTENSION_SCHEMA ".rerank_pool",
			"Max candidates to exact-rerank per query (0 = automatic, "
			"-1 = unlimited).",
			"Rerank only the most promising candidates by approximate "
			"distance, bounding the exact-distance heap fetches. 0 (the "
			"default) caps at 3 * k * nprobe^0.15, growing to 1/8th of "
			"the candidate buffer when noisy distance estimates flood "
			"it; -1 reranks every threshold survivor; positive values "
			"set an absolute cap. The effective cap is never below the "
			"query's k, so results are never truncated.",
			&mkt_rerank_pool,
			0,
			-1,
			1000000,
			PGC_USERSET,
			0,
			NULL,
			mkt_rerank_pool_assign_hook,
			NULL);

	DefineCustomRealVariable(
			MKT_EXTENSION_SCHEMA ".rerank_pool_cost_scale",
			"Relative cost of one exact-rerank candidate, applied to the "
			"automatic rerank_pool only (0 = auto-detect).",
			"The automatic rerank_pool formula (rerank_pool = 0) is fit "
			"assuming a cache-resident rerank: every candidate is a cheap "
			"in-memory fetch. This scale corrects that assumption when it "
			"does not hold -- e.g. the indexed table exceeds available "
			"cache and a candidate is a real disk read instead -- by "
			"dividing the automatic pool by it. 0 (the default) "
			"auto-detects a scale each scan from the ratio of the index's "
			"table+index size to effective_cache_size, floored at 1.0 (the "
			"original fit, unmodified, whenever the working set fits in "
			"cache). A positive value overrides the auto-detection with an "
			"exact scale; 1.0 reproduces the original fit exactly regardless "
			"of relation size. Has no effect when rerank_pool is set "
			"explicitly.",
			&mkt_rerank_pool_cost_scale,
			0.0,
			0.0,
			1000.0,
			PGC_USERSET,
			0,
			NULL,
			NULL,
			NULL);

	MarkGUCPrefixReserved(MKT_EXTENSION_SCHEMA);

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
			MKT_ANN_DEFAULT_FAN_OUT,
			MKT_ANN_MIN_FAN_OUT,
			MKT_ANN_MAX_FAN_OUT,
			NoLock);
	add_int_reloption(
			mktann_relopt_kind,
			"nlist",
			"Number of IVF clusters (0 = auto from sqrt(rows))",
			MKT_ANN_DEFAULT_NLIST,
			MKT_ANN_MIN_NLIST,
			MKT_ANN_MAX_NLIST,
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
			MKT_ANN_DEFAULT_SOAR_LAMBDA,
			0.0,
			100.0,
			NoLock);
	add_real_reloption(
			mktann_relopt_kind,
			"boundary_epsilon",
			"Boundary replication gap threshold (0 = off)",
			MKT_ANN_DEFAULT_BOUNDARY_EPSILON,
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
	add_enum_reloption(
			mktann_relopt_kind,
			"fastscan",
			"VPSHUFB fastscan posting page format",
			fastscan_mode_relopt_members,
			MKT_FASTSCAN_MODE_AUTO,
			"auto uses it up to the dimension where a group fits a page",
			NoLock);
	add_enum_reloption(
			mktann_relopt_kind,
			"centroid_fastscan",
			"FASTSCAN-format centroid pages",
			fastscan_mode_relopt_members,
			MKT_FASTSCAN_MODE_AUTO,
			"auto follows the resolved centroid compression and the "
			"dimension limit; on errors where either is unavailable",
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

/*
 * rabitq_params_generate() builds a dim x dim orthogonal matrix by
 * Gram-Schmidt -- O(dim^3) in time, O(dim^2) in memory -- and is callable by
 * any role, so an oversized dim is a CPU/memory denial-of-service vector (at
 * the generic vector cap a single call runs for minutes at 100% CPU). Bound
 * it by MKT_INDEX_MAX_DIM: generating params for a dimension no meerkat index
 * can hold is pointless, so the largest indexable dimension is the natural
 * ceiling, and it tracks the index limit automatically.
 */
void
mkt_pg_check_rabitq_params_dim_valid(int dim)
{
	mkt_pg_check_dim_valid(dim);
	if (dim > MKT_INDEX_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("cannot generate rabitq_params for more than %d "
						"dimensions",
						MKT_INDEX_MAX_DIM),
				 errhint("Building the transform matrix is O(dim^3); no "
						 "mktann index supports more dimensions than this.")));
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

/*
 * Return the extension version the binary was built as. Comes from
 * meson.build via mkt_config.h, the single source of truth for the
 * version string.
 */
PG_FUNCTION_INFO_V1(mkt_extension_version);

Datum
mkt_extension_version(PG_FUNCTION_ARGS)
{
	PG_RETURN_TEXT_P(cstring_to_text(MKT_VERSION));
}

/*
 * Return the extension name the binary was built as (from meson.build
 * via mkt_config.h). Lets SQL refer to the extension by name without
 * hardcoding it, e.g. in the prerelease install notice.
 */
PG_FUNCTION_INFO_V1(mkt_extension_name);

Datum
mkt_extension_name(PG_FUNCTION_ARGS)
{
	PG_RETURN_TEXT_P(cstring_to_text(MKT_EXTENSION_NAME));
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
