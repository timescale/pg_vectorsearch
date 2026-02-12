/*
 * mkt_pg.h - PostgreSQL-specific macros and helpers for meerkat
 */

#ifndef MKT_PG_H
#define MKT_PG_H

#include <postgres.h>

#include <fmgr.h>
#include <utils/array.h>
#include <utils/lsyscache.h>

#include "mkt_halfvec.h"
#include "mkt_vector.h"

/* ----------------------------------------------------------------
 * Datum conversion macros
 * ---------------------------------------------------------------- */

#define DatumGetMktVector(x)	  ((MktVector *)PG_DETOAST_DATUM(x))
#define PG_GETARG_MKT_VECTOR_P(x) DatumGetMktVector(PG_GETARG_DATUM(x))
#define PG_RETURN_MKT_VECTOR_P(x) PG_RETURN_POINTER(x)

#define DatumGetMktHalfVector(x)   ((MktHalfVector *)PG_DETOAST_DATUM(x))
#define PG_GETARG_MKT_HALFVEC_P(x) DatumGetMktHalfVector(PG_GETARG_DATUM(x))
#define PG_RETURN_MKT_HALFVEC_P(x) PG_RETURN_POINTER(x)

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

/* ----------------------------------------------------------------
 * Validation helpers (implemented in mkt_pg.c)
 * ---------------------------------------------------------------- */

void mkt_pg_check_dim_valid(int dim);
void mkt_pg_check_dims_match(int dim_a, int dim_b);
void mkt_pg_check_expected_dim(int actual, int expected);
void mkt_pg_check_value_finite(float val);

#endif /* MKT_PG_H */
