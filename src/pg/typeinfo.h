/*
 * typeinfo.h - per-type behaviour for an mktann-indexed column
 *
 * The access method indexes more than one vector type, so every path that
 * reads an indexed value needs to know how to reach the float32 arrays the
 * distance and RaBitQ kernels work on. That knowledge is one descriptor per
 * type, and the descriptor comes from the *opclass* -- support function
 * MKT_ANN_TYPE_INFO_PROC -- not from inspecting the column.
 *
 * Asking the opclass matters beyond tidiness. It is the same authority the
 * planner consulted when it decided the index applied, so the access method
 * cannot disagree with the planner about a column an index was built on.
 * Deriving the type from the column means answering "which type is this?" by
 * name or by OID, and both answers are wrong somewhere: PostgreSQL restricts
 * the search path around CREATE INDEX and index_build, so a name lookup fails
 * during a build -- exactly where a wrong answer corrupts an index -- and OID
 * equality rejects a binary-coercible column the opclass itself accepted, a
 * pgvector `halfvec` column being the case in point. It also makes a new type
 * an opclass plus a descriptor, with no new branch in the access method.
 *
 * The contract is deliberately narrower than pgvector's, which keeps values as
 * opaque Datums and reaches them only through opclass functions. That approach
 * would work here too -- quantization could be a support function taking a
 * Datum, the shape pgvector already uses for normalize -- so this is a choice
 * rather than a limit. Two reasons for it:
 *
 *  - The quantization and distance kernels live in src/index, src/algo and
 *    src/quant and are compiled into the standalone build, which has no fmgr,
 *    no Datum and no type OIDs. A Datum-typed seam would have to be shimmed or
 *    duplicated there, and the point of standalone is to measure the same
 * code.
 *
 *  - The conversion would not disappear, only move. The kernels are element-
 *    wise SIMD over contiguous float32, so a Datum-typed quantize would widen
 *    to float32 inside the callback -- the same work, done per call instead of
 *    once at the boundary, and without the shared Vec32TypeOps to do it.
 *
 * Where this contract would need revisiting is a type for which "widen to
 * dense float32" is the wrong shape -- a sparse or bit-packed vector, both of
 * which pgvector's hnsw does support. Dense float types fit it exactly.
 *
 * The conversion itself is not new here: Vec32TypeOps already covers
 * float32 and float16 for k-means and quantization, and is shared with
 * standalone. This descriptor binds an opclass to one of those vtables and
 * adds the two things only PostgreSQL needs -- how to unwrap a Datum, and
 * which centroid format the type implies. Per docs/development.md, the
 * conversion runs at the "Specialized" tier: one vtable call per vector to
 * obtain a float32 view, then inlined f32 kernels over it.
 *
 * A descriptor describes its opclass's own input type. A cross-type ordering
 * operator whose argument had a different layout would break that; every
 * family member meerkat declares or adds shares its opclass's layout.
 */

#ifndef MKT_TYPEINFO_H
#define MKT_TYPEINFO_H

#include <postgres.h>

#include <utils/rel.h>

#include "core/types.h"
#include "index/centroid_page.h"

typedef struct MktIndexTypeInfo
{
	/* Type name, for error messages. */
	const char *name;

	/* Widest dimension the type carries, before the index's own limit. */
	Dimension max_dimensions;

	/*
	 * Centroid storage the type implies. Centroids are drawn from the indexed
	 * data, so a half-precision column has no use for full-precision
	 * centroids.
	 */
	MktCentroidFormat centroid_format;

	/*
	 * Element kernels and float32 conversion, shared with standalone. f32
	 * returns its input pointer from to_float_block; f16 widens into the
	 * caller's buffer.
	 */
	const Vec32TypeOps *ops;

	/*
	 * Datum -> raw element block, plus the value's dimension. The only part
	 * that touches PostgreSQL's representation; deliberately per-type rather
	 * than one function assuming a shared header layout.
	 */
	const void *(*unwrap)(Datum d, Dimension *dim);
} MktIndexTypeInfo;

/*
 * Descriptor for an index's column type. The support function is optional:
 * an opclass that declares none indexes `vec32`, which is both the original
 * behaviour and what an index built before the function existed still needs.
 */
const MktIndexTypeInfo *mkt_index_type_info(Relation index);

/* Does a float32 view of this type need a caller-provided buffer? */
static inline bool
vec32_needs_buffer(const MktIndexTypeInfo *ti)
{
	return ti->ops->element_size != sizeof(float);
}

/*
 * An index's binding of a type descriptor.
 *
 * MktIndexTypeInfo is a shared static -- one instance per type for the whole
 * backend -- so it holds nothing specific to an index. This is the per-index
 * half: the dimension its column is pinned to, and the buffer a float32 view
 * needs at that dimension. Binding them together means a read cannot be handed
 * a dimension the buffer was not sized for.
 *
 * Resolve once per build / scan / rerank and read many times.
 */
typedef struct Vec32Access
{
	const MktIndexTypeInfo *ti;	 /* shared, per type */
	Dimension				dim; /* this index's column */
	float				   *buf; /* NULL when no conversion is needed */
} Vec32Access;

/*
 * Bind a descriptor to a dimension, allocating the conversion buffer if the
 * type needs one.
 *
 * The context is the caller's because the useful lifetime differs per path:
 * the whole scan for a query vector, the whole build for the scan callbacks,
 * one rerank call for the heap fetches. Pass CurrentMemoryContext for plain
 * palloc semantics.
 */
static inline Vec32Access
vec32_access(const MktIndexTypeInfo *ti, Dimension dim, MemoryContext ctx)
{
	return (Vec32Access){
			.ti	 = ti,
			.dim = dim,
			.buf = vec32_needs_buffer(ti)
						 ? (float *)
								   MemoryContextAlloc(ctx, sizeof(float) * dim)
						 : NULL,
	};
}

/*
 * Float32 view of one indexed value.
 *
 * The dimension is checked before any conversion, so a value wider than the
 * index cannot be converted into a buffer sized for the index. A column's
 * typmod pins its dimension, so a mismatch should be unreachable -- but this
 * reads a varlena into a fixed buffer.
 */
static inline Vec32Ref
vec32_read(const Vec32Access *a, Datum d)
{
	Dimension	dim;
	const void *raw = a->ti->unwrap(d, &dim);

	if (dim != a->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("%s dimension %u does not match index dimension %u",
						a->ti->name,
						dim,
						a->dim)));

	return (Vec32Ref){
			.data = a->ti->ops->to_float_block(raw, a->buf, 1, dim),
			.dim  = dim,
	};
}

#endif /* MKT_TYPEINFO_H */
