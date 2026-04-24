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

#include "mkt_halfvec.h"
#include "mkt_vector.h"
#include "quant/rabitq.h"

/* ----------------------------------------------------------------
 * Support function numbers
 * ---------------------------------------------------------------- */

#define MKTANN_DISTANCE_PROC 1 /* distance operator function */
#define MKTANN_METRIC_PROC	 2 /* metric identifier function */

/* ----------------------------------------------------------------
 * GUC variables
 * ---------------------------------------------------------------- */

extern int		   mkt_distance_mode;  /* MktDistanceMode */
extern int		   mkt_nprobe;		   /* clusters to probe per query */
extern int		   mkt_query_limit;	   /* max results per query (0=auto) */
extern bool		   mkt_rerank;		   /* enable reranking (default: true) */
extern relopt_kind mktann_relopt_kind; /* index reloption kind */

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

#define MKT_RABITQ_PARAMS_PG_SIZE(dim) \
	(offsetof(RaBitQParamsPG, P) + (dim) * (dim) * sizeof(float))

#define DatumGetRaBitQParamsPG(x)	 ((RaBitQParamsPG *)PG_DETOAST_DATUM(x))
#define PG_GETARG_RABITQ_PARAMS_P(x) DatumGetRaBitQParamsPG(PG_GETARG_DATUM(x))

/* ----------------------------------------------------------------
 * Index reloptions
 * ---------------------------------------------------------------- */

typedef struct MktannOptions
{
	int32 vl_len_;				/* varlena header (required by reloptions) */
	int	  distance_mode;		/* MktDistanceMode */
	int	  fan_out;				/* children per tree node (2-255) */
	int	  nlist;				/* number of clusters (0 = auto) */
	bool  centroid_compression; /* use RaBitQ for centroid pages */
} MktannOptions;

#define MKTANN_DEFAULT_FAN_OUT 32
#define MKTANN_MIN_FAN_OUT	   2
#define MKTANN_MAX_FAN_OUT	   255

#define MKTANN_DEFAULT_NLIST 0
#define MKTANN_MIN_NLIST	 0
#define MKTANN_MAX_NLIST	 100000

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
void mkt_pg_check_dims_match(int dim_a, int dim_b);
void mkt_pg_check_expected_dim(int actual, int expected);
void mkt_pg_check_value_finite(float val);

#endif /* MKT_PG_H */
