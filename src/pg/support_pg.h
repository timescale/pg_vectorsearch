/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * vs_pg.h - PostgreSQL-specific macros and helpers for pg_vectorsearch
 */

#ifndef VS_PG_H
#define VS_PG_H

#include <postgres.h>

#include <access/reloptions.h>
#include <fmgr.h>
#include <utils/array.h>
#include <utils/lsyscache.h>
#include <utils/rel.h>

#include "quant/rabitq.h"
#include "types/vec16.h"
#include "types/vec32.h"

/* ----------------------------------------------------------------
 * Support function numbers
 * ---------------------------------------------------------------- */

#define PRISM_DISTANCE_PROC	 1 /* distance operator function */
#define PRISM_METRIC_PROC	 2 /* metric identifier function */
#define PRISM_TYPE_INFO_PROC 3 /* column type descriptor (optional) */

/* ----------------------------------------------------------------
 * GUC variables
 * ---------------------------------------------------------------- */

extern int	prism_distance_mode;   /* VsDistanceMode */
extern int	prism_nprobe;		   /* clusters to probe per query */
extern int	prism_query_limit;	   /* min top-k per scan (0=from LIMIT) */
extern int	prism_fastscan_bits;   /* fastscan LUT bits (8 or 16) */
extern bool prism_rerank;		   /* enable reranking (default: true) */
extern bool prism_log_build_stats; /* log per-phase build stats (def: off) */
extern int	prism_leaf_refine_threshold;  /* refine when samples/leaf < this */
extern double prism_centroid_error_scale; /* scales centroid pruning error
									   bound (1=default, 0=drop) */
extern double prism_probe_expand;		  /* routed clusters / nprobe */
extern double prism_centroid_beam_scale; /* intermediate beam width as fraction
									  of nprobe (0.25=default) */
extern relopt_kind prism_relopt_kind;	 /* index reloption kind */

/* ----------------------------------------------------------------
 * Datum conversion macros
 * ---------------------------------------------------------------- */

#define DatumGetVec32(x)	 ((Vec32 *)PG_DETOAST_DATUM(x))
#define PG_GETARG_VEC32_P(x) DatumGetVec32(PG_GETARG_DATUM(x))
#define PG_RETURN_VEC32_P(x) PG_RETURN_POINTER(x)

#define DatumGetVec16(x)	 ((Vec16 *)PG_DETOAST_DATUM(x))
#define PG_GETARG_VEC16_P(x) DatumGetVec16(PG_GETARG_DATUM(x))
#define PG_RETURN_VEC16_P(x) PG_RETURN_POINTER(x)

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
 * VS_RABITQ_PARAMS_SIZE macro in rabitq.h.
 */
#define VS_RABITQ_PARAMS_PG_SIZE(dim) \
	(offsetof(RaBitQParamsPG, P) + (uint64_t)(dim) * (dim) * sizeof(float))

#define DatumGetRaBitQParamsPG(x)	 ((RaBitQParamsPG *)PG_DETOAST_DATUM(x))
#define PG_GETARG_RABITQ_PARAMS_P(x) DatumGetRaBitQParamsPG(PG_GETARG_DATUM(x))

/* ----------------------------------------------------------------
 * Index reloptions
 * ---------------------------------------------------------------- */

/*
 * PrismCentroidCompression - tri-state control for RaBitQ centroid pages.
 *
 * AUTO compresses where RaBitQ routing is correct (L2, cosine) and falls
 * back to float for inner product, whose ordering cannot be recovered from
 * RaBitQ's L2 distance estimate. ON forces compression and errors at build
 * time for inner product. OFF disables compression.
 */
typedef enum
{
	PRISM_CENTROID_COMPRESSION_AUTO = 0,
	PRISM_CENTROID_COMPRESSION_ON	= 1,
	PRISM_CENTROID_COMPRESSION_OFF	= 2,
} PrismCentroidCompression;

/*
 * PrismFastscanMode - tri-state control for the packed FASTSCAN page
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
	PRISM_FASTSCAN_MODE_AUTO = 0,
	PRISM_FASTSCAN_MODE_ON	 = 1,
	PRISM_FASTSCAN_MODE_OFF	 = 2,
} PrismFastscanMode;

typedef struct PrismOptions
{
	int32  vl_len_;				 /* varlena header (required by reloptions) */
	int	   distance_mode;		 /* VsDistanceMode */
	int	   fan_out;				 /* children per tree node (2-255) */
	int	   nlist;				 /* number of clusters (0 = auto) */
	int	   target_pages;		 /* posting pages per list */
	int	   kmeans_nredo;		 /* k-means restarts (1 = no restart) */
	double soar_lambda;			 /* SOAR replication lambda (0=off) */
	double boundary_epsilon;	 /* boundary replication threshold (0=off) */
	int	   centroid_compression; /* PrismCentroidCompression */
	int	   fastscan;			 /* PrismFastscanMode, posting pages */
	int	   centroid_fastscan;	 /* PrismFastscanMode, centroid pages */
} PrismOptions;

#define PRISM_DEFAULT_FAN_OUT 32
#define PRISM_MIN_FAN_OUT	  2
#define PRISM_MAX_FAN_OUT	  255

#define PRISM_DEFAULT_NLIST 0
#define PRISM_MIN_NLIST		0
#define PRISM_MAX_NLIST		2000000

/*
 * Range of the target_pages setting -- posting pages each list rests at,
 * the denominator of the automatic nlist. Its default is
 * PRISM_DEFAULT_TARGET_PAGES, over in index_build.h with the code that
 * applies it.
 */
#define PRISM_MIN_TARGET_PAGES 1
#define PRISM_MAX_TARGET_PAGES 64

/*
 * Replication defaults. Both forms of secondary assignment are on by
 * default: SOAR (lambda 1.0) plus a wide boundary band (epsilon 0.35)
 * won the recall/QPS Pareto frontier at every benchmarked scale
 * (10M-100M vectors), at a few percent of index size for the boundary
 * band and a modest build-time cost for SOAR.
 */
#define PRISM_DEFAULT_SOAR_LAMBDA	   1.0
#define PRISM_DEFAULT_BOUNDARY_EPSILON 0.35

/*
 * PrismGetDistanceMode - Resolve effective distance mode for a scan.
 *
 * GUC overrides index relopt when explicitly set (not 'default').
 */
static inline VsDistanceMode
PrismGetDistanceMode(Relation index)
{
	PrismOptions *opts = (PrismOptions *)index->rd_options;
	/* GUC overrides index relopt when explicitly set */
	if (prism_distance_mode != VS_DISTANCE_MODE_DEFAULT)
		return (VsDistanceMode)prism_distance_mode;
	/* Use index relopt, or asymmetric if no options set */
	if (opts != NULL)
		return (VsDistanceMode)opts->distance_mode;
	return VS_DISTANCE_MODE_ASYMMETRIC;
}

/* ----------------------------------------------------------------
 * Allocation helpers
 * ---------------------------------------------------------------- */

static inline Vec32 *
vs_pg_vec32_alloc(int dim)
{
	int	   size = VEC32_SIZE(dim);
	Vec32 *v	= (Vec32 *)palloc0(size);
	SET_VARSIZE(v, size);
	v->dim	  = (int16_t)dim;
	v->unused = 0;
	return v;
}

static inline Vec16 *
vs_pg_vec16_alloc(int dim)
{
	int	   size = VEC16_SIZE(dim);
	Vec16 *v	= (Vec16 *)palloc0(size);
	SET_VARSIZE(v, size);
	v->dim	  = (int16_t)dim;
	v->unused = 0;
	return v;
}

/*
 * Allocates from dim, which is then the only record of how many bit bytes
 * follow: readers derive lengths from v->dim via VS_RABITQ_BYTES without
 * re-checking VARSIZE. That holds because every datum comes from here or
 * from the text parser, which allocates the dim it just parsed -- the type
 * has no receive function. A binary input path must validate dim against
 * VARSIZE before the datum reaches those readers.
 */
static inline RaBitQVector *
vs_pg_rabitq_alloc(int dim)
{
	int			  size = VS_RABITQ_VECTOR_SIZE(dim);
	RaBitQVector *v	   = (RaBitQVector *)palloc0(size);
	SET_VARSIZE(v, size);
	v->dim	 = (int16_t)dim;
	v->flags = 0;
	return v;
}

/* ----------------------------------------------------------------
 * Validation helpers (implemented in vs_pg.c)
 * ---------------------------------------------------------------- */

void vs_pg_check_dim_valid(int dim);
void vs_pg_check_rabitq_params_dim_valid(int dim);
void vs_pg_check_dims_match(int dim_a, int dim_b);
void vs_pg_check_expected_dim(int actual, int expected);
void vs_pg_check_value_finite(float val);

#endif /* VS_PG_H */
