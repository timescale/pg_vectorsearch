/*
 * typeinfo.c - type descriptors for prism-indexed columns
 *
 * One descriptor per indexable type, handed to the access method by the
 * opclass. See typeinfo.h for why the opclass is the source.
 */

#include <postgres.h>

#include <access/relation.h>
#include <fmgr.h>
#include <utils/relcache.h>

#include "support_pg.h"
#include "typeinfo.h"
#include "types/vec16.h"
#include "types/vec32.h"

/* ----------------------------------------------------------------
 * Datum unwrapping
 * ---------------------------------------------------------------- */

static const void *
vector_unwrap(Datum d, Dimension *dim)
{
	Vec32 *v = DatumGetVec32(d);

	*dim = (Dimension)v->dim;
	return VEC32_DATA(v);
}

static const void *
halfvec_unwrap(Datum d, Dimension *dim)
{
	Vec16 *v = DatumGetVec16(d);

	*dim = (Dimension)v->dim;
	return VEC16_DATA(v);
}

/* ----------------------------------------------------------------
 * Descriptors
 *
 * vec16 costs one widening pass per value read and buys a heap half the
 * size, which is what an exact rerank reads: at 768d a vector row is 3080
 * bytes and fits 2 to an 8 kB page against vec16's 1544 and 5.
 * ---------------------------------------------------------------- */

static const MktIndexTypeInfo type_info_vector = {
		.name			 = "vec32",
		.max_dimensions	 = VEC32_MAX_DIM,
		.centroid_format = MKT_CENTROID_FMT_FLOAT,
		.ops			 = &mkt_f32_type_ops,
		.unwrap			 = vector_unwrap,
};

static const MktIndexTypeInfo type_info_halfvec = {
		.name = "vec16",
		/* vec16_typmod_in enforces the same ceiling as vec32. */
		.max_dimensions	 = VEC32_MAX_DIM,
		.centroid_format = MKT_CENTROID_FMT_HALF,
		.ops			 = &mkt_f16_type_ops,
		.unwrap			 = halfvec_unwrap,
};

/* ----------------------------------------------------------------
 * Opclass support functions (MKT_ANN_TYPE_INFO_PROC)
 * ---------------------------------------------------------------- */

PG_FUNCTION_INFO_V1(prism_vec32_support);

Datum
prism_vec32_support(PG_FUNCTION_ARGS)
{
	PG_RETURN_POINTER(&type_info_vector);
}

PG_FUNCTION_INFO_V1(prism_vec16_support);

Datum
prism_vec16_support(PG_FUNCTION_ARGS)
{
	PG_RETURN_POINTER(&type_info_halfvec);
}

const MktIndexTypeInfo *
mkt_index_type_info(Relation index)
{
	/*
	 * Optional, pgvector-style: an opclass with no descriptor indexes
	 * `vec32`. That leaves the vec32 opclasses' SQL untouched, and an index
	 * built before this support function existed keeps working.
	 */
	if (!OidIsValid(index_getprocid(index, 1, MKT_ANN_TYPE_INFO_PROC)))
		return &type_info_vector;

	FmgrInfo *procinfo = index_getprocinfo(index, 1, MKT_ANN_TYPE_INFO_PROC);

	return (const MktIndexTypeInfo *)DatumGetPointer(
			FunctionCall0Coll(procinfo, InvalidOid));
}
