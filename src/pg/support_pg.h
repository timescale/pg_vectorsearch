/*
 * mkt_pg.h - PostgreSQL-specific macros and helpers for meerkat
 */

#ifndef MKT_PG_H
#define MKT_PG_H

#include <postgres.h>

#include <access/reloptions.h>
#include <fmgr.h>
#include <utils/array.h>
#include <utils/lsyscache.h>
#include <utils/rel.h>

#include "quant/rabitq.h"
#include "types/halfvec.h"
#include "types/vector.h"

/* ----------------------------------------------------------------
 * Support function numbers
 * ---------------------------------------------------------------- */

#define MKT_ANN_DISTANCE_PROC 1 /* distance operator function */
#define MKT_ANN_METRIC_PROC	  2 /* metric identifier function */

/* ----------------------------------------------------------------
 * GUC variables
 * ---------------------------------------------------------------- */

extern int	  mkt_distance_mode;   /* MktDistanceMode */
extern int	  mkt_nprobe;		   /* clusters to probe per query */
extern int	  mkt_query_limit;	   /* max results per query (0=auto) */
extern int	  mkt_fastscan_bits;   /* fastscan LUT bits (8 or 16) */
extern bool	  mkt_rerank;		   /* enable reranking (default: true) */
extern bool	  mkt_log_build_stats; /* log per-phase build stats (def: off) */
extern int	  mkt_leaf_refine_threshold; /* refine when samples/leaf < this */
extern double mkt_centroid_error_scale;	 /* scales centroid pruning error bound
											(1=default, 0=drop) */
extern double mkt_centroid_beam_scale;	 /* intermediate beam width as fraction
											of nprobe (0.25=default) */
extern relopt_kind mktann_relopt_kind;	 /* index reloption kind */

/* ----------------------------------------------------------------
 * Datum conversion macros
 * ---------------------------------------------------------------- */

#define DatumGetMktVector(x)	  ((MktVector *)PG_DETOAST_DATUM(x))
#define PG_GETARG_MKT_VECTOR_P(x) DatumGetMktVector(PG_GETARG_DATUM(x))
#define PG_RETURN_MKT_VECTOR_P(x) PG_RETURN_POINTER(x)

#define DatumGetMktHalfVector(x)   ((MktHalfVector *)PG_DETOAST_DATUM(x))
#define PG_GETARG_MKT_HALFVEC_P(x) DatumGetMktHalfVector(PG_GETARG_DATUM(x))
#define PG_RETURN_MKT_HALFVEC_P(x) PG_RETURN_POINTER(x)

#define DatumGetRaBitQVector(x) ((RaBitQVector *)PG_DETOAST_DATUM(x))
#define PG_GETARG_RABITQ_P(x)	DatumGetRaBitQVector(PG_GETARG_DATUM(x))
#define PG_RETURN_RABITQ_P(x)	PG_RETURN_POINTER(x)

/*
 * RaBitQParamsPG - Orthogonal transform matrix (PostgreSQL varlena)
 *
 * Stores the random orthogonal matrix P and seed for reproducibility.
 * Generated via rabitq_params_generate(dim, seed) and used with
 * rabitq_encode() to quantize vectors.
 *
 * Total size: 16 bytes header + dim * dim * sizeof(float)
 */
typedef struct RaBitQParamsPG
{
	int32_t	 vl_len_;
	int16_t	 dim;
	int16_t	 unused;
	uint64_t seed;
	float	 P[];
} RaBitQParamsPG;

/*
 * The matrix is dim*dim floats. Cast to a fixed 64-bit type before the
 * multiply so the product is computed in 64 bits on every platform: plain
 * int32 overflows once dim exceeds ~46340, and size_t would still be 32-bit
 * on ILP32 targets. Unreachable at today's dimension caps, but keeps the
 * arithmetic robust if a cap is ever raised, matching the sibling
 * MKT_RABITQ_PARAMS_SIZE macro in rabitq.h.
 */
#define MKT_RABITQ_PARAMS_PG_SIZE(dim) \
	(offsetof(RaBitQParamsPG, P) + (uint64_t)(dim) * (dim) * sizeof(float))

#define DatumGetRaBitQParamsPG(x)	 ((RaBitQParamsPG *)PG_DETOAST_DATUM(x))
#define PG_GETARG_RABITQ_PARAMS_P(x) DatumGetRaBitQParamsPG(PG_GETARG_DATUM(x))

/* ----------------------------------------------------------------
 * Index reloptions
 * ---------------------------------------------------------------- */

/*
 * MktCentroidCompression - tri-state control for RaBitQ centroid pages.
 *
 * AUTO compresses where RaBitQ routing is correct (L2, cosine) and falls
 * back to float for inner product, whose ordering cannot be recovered from
 * RaBitQ's L2 distance estimate. ON forces compression and errors at build
 * time for inner product. OFF disables compression.
 */
typedef enum
{
	MKT_CENTROID_COMPRESSION_AUTO = 0,
	MKT_CENTROID_COMPRESSION_ON	  = 1,
	MKT_CENTROID_COMPRESSION_OFF  = 2,
} MktCentroidCompression;

/*
 * MktFastscanMode - tri-state control for the packed FASTSCAN page
 * layouts (posting pages via `fastscan`, centroid pages via
 * `centroid_fastscan`).
 *
 * Both layouts store fixed 32-candidate groups whose byte size grows
 * with the vector dimension, so past a dimension threshold a single
 * group no longer fits a page. AUTO uses the layout where it fits
 * (and, for centroids, where RaBitQ compression is in effect — the
 * FASTSCAN layout exists only over compressed centroids). ON forces
 * the layout and errors at build time where it is unavailable. OFF
 * disables it.
 */
typedef enum
{
	MKT_FASTSCAN_MODE_AUTO = 0,
	MKT_FASTSCAN_MODE_ON   = 1,
	MKT_FASTSCAN_MODE_OFF  = 2,
} MktFastscanMode;

typedef struct MktannOptions
{
	int32  vl_len_;				 /* varlena header (required by reloptions) */
	int	   distance_mode;		 /* MktDistanceMode */
	int	   fan_out;				 /* children per tree node (2-255) */
	int	   nlist;				 /* number of clusters (0 = auto) */
	int	   kmeans_nredo;		 /* k-means restarts (1 = no restart) */
	double soar_lambda;			 /* SOAR replication lambda (0=off) */
	double boundary_epsilon;	 /* boundary replication threshold (0=off) */
	int	   centroid_compression; /* MktCentroidCompression */
	int	   fastscan;			 /* MktFastscanMode, posting pages */
	int	   centroid_fastscan;	 /* MktFastscanMode, centroid pages */
} MktannOptions;

#define MKT_ANN_DEFAULT_FAN_OUT 32
#define MKT_ANN_MIN_FAN_OUT		2
#define MKT_ANN_MAX_FAN_OUT		255

#define MKT_ANN_DEFAULT_NLIST 0
#define MKT_ANN_MIN_NLIST	  0
#define MKT_ANN_MAX_NLIST	  2000000

/*
 * Replication defaults. Both forms of secondary assignment are on by
 * default: SOAR (lambda 1.0) plus a wide boundary band (epsilon 0.35)
 * won the recall/QPS Pareto frontier at every benchmarked scale
 * (10M-100M vectors), at a few percent of index size for the boundary
 * band and a modest build-time cost for SOAR.
 */
#define MKT_ANN_DEFAULT_SOAR_LAMBDA		 1.0
#define MKT_ANN_DEFAULT_BOUNDARY_EPSILON 0.35

/*
 * MktannGetDistanceMode - Resolve effective distance mode for a scan.
 *
 * GUC overrides index relopt when explicitly set (not 'default').
 */
static inline MktDistanceMode
MktannGetDistanceMode(Relation index)
{
	MktannOptions *opts = (MktannOptions *)index->rd_options;
	/* GUC overrides index relopt when explicitly set */
	if (mkt_distance_mode != MKT_DISTANCE_MODE_DEFAULT)
		return (MktDistanceMode)mkt_distance_mode;
	/* Use index relopt, or asymmetric if no options set */
	if (opts != NULL)
		return (MktDistanceMode)opts->distance_mode;
	return MKT_DISTANCE_MODE_ASYMMETRIC;
}

/* ----------------------------------------------------------------
 * Allocation helpers
 * ---------------------------------------------------------------- */

static inline MktVector *
mkt_pg_vector_alloc(int dim)
{
	int		   size = MKT_VECTOR_SIZE(dim);
	MktVector *v	= (MktVector *)palloc0(size);
	SET_VARSIZE(v, size);
	v->dim	  = (int16_t)dim;
	v->unused = 0;
	return v;
}

static inline MktHalfVector *
mkt_pg_halfvec_alloc(int dim)
{
	int			   size = MKT_HALFVEC_SIZE(dim);
	MktHalfVector *v	= (MktHalfVector *)palloc0(size);
	SET_VARSIZE(v, size);
	v->dim	  = (int16_t)dim;
	v->unused = 0;
	return v;
}

static inline RaBitQVector *
mkt_pg_rabitq_alloc(int dim)
{
	int			  size = MKT_RABITQ_VECTOR_SIZE(dim);
	RaBitQVector *v	   = (RaBitQVector *)palloc0(size);
	SET_VARSIZE(v, size);
	v->dim	 = (int16_t)dim;
	v->flags = 0;
	return v;
}

/* ----------------------------------------------------------------
 * Type OID helpers (implemented in mkt_pg.c)
 * ---------------------------------------------------------------- */

Oid mkt_halfvec_type_oid(void);

/* ----------------------------------------------------------------
 * Validation helpers (implemented in mkt_pg.c)
 * ---------------------------------------------------------------- */

void mkt_pg_check_dim_valid(int dim);
void mkt_pg_check_rabitq_params_dim_valid(int dim);
void mkt_pg_check_dims_match(int dim_a, int dim_b);
void mkt_pg_check_expected_dim(int actual, int expected);
void mkt_pg_check_value_finite(float val);

#endif /* MKT_PG_H */
