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
#include "index/query_scan.h"
#include "git_commit.h"
#include "mkt_pg.h"
#include "mktann_explain.h"

PG_MODULE_MAGIC;

/* GUC variables */
int	 mkt_distance_mode = MKT_DISTANCE_MODE_DEFAULT;
int	 mkt_nprobe		   = 10;
int	 mkt_query_limit   = 0;
int	 mkt_fastscan_bits = 16;
bool mkt_rerank		   = true;
int	 mkt_rerank_pool   = 0;
bool   mkt_route_ip		 = false;
double mkt_centroid_error_scale = 0.0;
double mkt_centroid_beam_scale  = 1.0;
int	   mkt_centroid_rerank = 0;
bool   mkt_early_terminate = false;
double mkt_term_radius	 = 0.0; /* global max (1-cos(v,centroid)); 0 = off */
double mkt_term_alpha	 = 0.0; /* heuristic stop: d(q,c)^2 > alpha*kth; 0 = off */
static bool mkt_profile = false;

static void
mkt_profile_assign_hook(bool newval, void *extra)
{
	mkt_query_set_profile(newval);
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

#ifdef MKT_HAVE_CBLAS
extern void openblas_set_num_threads(int);
#endif

void
_PG_init(void)
{
#ifdef MKT_HAVE_CBLAS
	/*
	 * Each PG backend (and each parallel build worker) is a separate
	 * process. With OpenBLAS defaulting to one thread per core, every
	 * small BLAS call (e.g. the per-query P^T·q rotation) would spawn
	 * threads-per-core — on a many-core box the thread spawn/join
	 * overhead dwarfs the tiny matrix op, and parallel builds would
	 * oversubscribe (workers × cores). meerkat's runtime BLAS ops are
	 * small; parallelism comes from PG worker processes, not BLAS
	 * threads. Pin BLAS to one thread per process.
	 */
	openblas_set_num_threads(1);
#endif

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

	DefineCustomRealVariable(
			"mkt.centroid_error_scale",
			"Scale on the centroid routing error bound used for topk pruning.",
			"0 (default) prunes the centroid beam by the point estimate only "
			"(far fewer subtree expansions, recall-neutral on tested data). "
			"1.0 restores the legacy conservative error bound.",
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
			"1.0 (default) keeps beam_width=nprobe at every tree level. <1 "
			"keeps a narrower beam at intermediate levels (scores far fewer "
			"centroids at high nprobe); the leaf level still returns nprobe.",
			&mkt_centroid_beam_scale,
			1.0,
			0.01,
			1.0,
			PGC_USERSET,
			0,
			NULL,
			NULL,
			NULL);

	DefineCustomBoolVariable(
			"mkt.route_ip",
			"Route by inner product (keep centroid magnitude) instead of "
			"cosine.",
			"Experimental: scores centroids by <q,c> without normalizing the "
			"centroid, so AVQ centroid magnitude acts as a per-cluster routing "
			"weight. Posting scan and rerank are unaffected.",
			&mkt_route_ip,
			false,
			PGC_USERSET,
			0,
			NULL,
			NULL,
			NULL);

	DefineCustomBoolVariable(
			"mkt.early_terminate",
			"Stop scanning clusters once none can hold a closer neighbor.",
			"Recall-safe: skips a cluster when its best possible member "
			"distance (from the centroid and the global cluster radius) is "
			"already worse than the current k-th neighbor. Needs "
			"mkt.term_radius set. Cosine only.",
			&mkt_early_terminate,
			false,
			PGC_USERSET,
			0,
			NULL,
			NULL,
			NULL);

	DefineCustomRealVariable(
			"mkt.term_radius",
			"Global cluster radius (max 1-cos(vector,centroid)) for early "
			"termination.",
			"The max angular extent of any cluster; used as the triangle-"
			"inequality bound. 0 disables. Must over-estimate the true max to "
			"stay recall-safe.",
			&mkt_term_radius,
			0.0,
			0.0,
			2.0,
			PGC_USERSET,
			0,
			NULL,
			NULL,
			NULL);

	DefineCustomRealVariable(
			"mkt.term_alpha",
			"Heuristic early stop: stop scanning when a cluster's centroid "
			"distance^2 exceeds alpha times the current k-th neighbor distance.",
			"NOT recall-safe (a closer member could exist within the cluster "
			"radius); alpha trades recall for speed (larger = safer/slower). "
			"0 disables. Cosine only. Captures the per-query depth variation "
			"that the recall-safe radius bound cannot.",
			&mkt_term_alpha,
			0.0,
			0.0,
			100.0,
			PGC_USERSET,
			0,
			NULL,
			NULL,
			NULL);

	DefineCustomIntVariable(
			"mkt.centroid_rerank",
			"Two-stage centroid routing: beam shortlist factor (0/1 = off).",
			"The compressed beam selects factor*nprobe candidate clusters, then "
			"re-ranks them by exact full-precision query-centroid distance (from "
			"the posting-head centroid) and scans the top nprobe. Recovers "
			"float-centroid routing accuracy at compressed beam speed.",
			&mkt_centroid_rerank,
			0,
			0,
			64,
			PGC_USERSET,
			0,
			NULL,
			NULL,
			NULL);

	DefineCustomIntVariable(
			"mkt.rerank_pool",
			"Max candidates to rerank with exact distances (0 = unbounded).",
			"Caps full-precision heap fetches. Candidates are reranked in "
			"order of ascending quantized distance, so the pool keeps the "
			"most promising survivors. Never shrinks below k.",
			&mkt_rerank_pool,
			0,
			0,
			1000000,
			PGC_USERSET,
			0,
			NULL,
			NULL,
			NULL);

	DefineCustomBoolVariable(
			"mkt.profile",
			"Record per-phase query timing (centroid/posting/rerank).",
			"Adds clock_gettime calls to the scan path; off by default.",
			&mkt_profile,
			false,
			PGC_USERSET,
			0,
			NULL,
			mkt_profile_assign_hook,
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
			"avq_eta",
			"AVQ anisotropic centroid eta (<=1 = off, e.g. 2-4)",
			0.0,
			0.0,
			1000.0,
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
	add_real_reloption(
			mktann_relopt_kind,
			"soar_ortho_cutoff",
			"Min SOAR secondary orthogonality quality to replicate (0 = keep all)",
			0.0,
			0.0,
			1.0,
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
	add_bool_reloption(
			mktann_relopt_kind,
			"centroid_half",
			"Store uncompressed centroids as fp16 instead of fp32 "
			"(requires centroid_compression=false)",
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
