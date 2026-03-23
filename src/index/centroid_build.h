/*
 * centroid_build.h - Generic centroid page writer
 *
 * Writes centroid entries to linked pages via MktStorage. Supports
 * all centroid formats (RaBitQ, float32, float16). Standalone-
 * compatible — all dependencies are portable.
 */

#ifndef MKT_CENTROID_BUILD_H
#define MKT_CENTROID_BUILD_H

#include "index/centroid_page.h"
#include "index/storage.h"

/* ----------------------------------------------------------------
 * CentroidEncoder — vtable for encoding entries into page memory
 *
 * Format-specific structs embed CentroidEncoder as their first
 * field, adding context (vectors, params). write_pages calls
 * encode_into() for each entry to write directly into a
 * pre-reserved page data region — no intermediate allocation
 * or copy needed.
 * ---------------------------------------------------------------- */
typedef struct CentroidEncoder CentroidEncoder;

typedef struct CentroidEncoderOps
{
	void (*encode_into)(CentroidEncoder *enc, uint32_t index, void *dest);
} CentroidEncoderOps;

struct CentroidEncoder
{
	const CentroidEncoderOps *ops;
};

/* ----------------------------------------------------------------
 * Built-in encoders for float32, float16, and RaBitQ formats
 *
 * Float:  memcpy from input array into dest
 * Half:   float→half conversion directly into dest
 * RaBitQ: mkt_rabitq_encode_into directly into dest
 * ---------------------------------------------------------------- */

/* Float encoder */
typedef struct
{
	CentroidEncoder base;
	const float	   *vectors;
	Dimension		dim;
} CentroidEncoderFloat;

static inline void
centroid_encode_float(CentroidEncoder *enc, uint32_t index, void *dest)
{
	CentroidEncoderFloat *fe = (CentroidEncoderFloat *)enc;
	memcpy(dest,
		   fe->vectors + (size_t)index * fe->dim,
		   fe->dim * sizeof(float));
}

static const CentroidEncoderOps centroid_encoder_float_ops = {
		.encode_into = centroid_encode_float,
};

/* Half encoder */
typedef struct
{
	CentroidEncoder base;
	const float	   *vectors;
	Dimension		dim;
} CentroidEncoderHalf;

static inline void
centroid_encode_half(CentroidEncoder *enc, uint32_t index, void *dest)
{
	CentroidEncoderHalf *he	 = (CentroidEncoderHalf *)enc;
	const float			*src = he->vectors + (size_t)index * he->dim;
	mkt_float_to_half_array(src, (half *)dest, he->dim);
}

static const CentroidEncoderOps centroid_encoder_half_ops = {
		.encode_into = centroid_encode_half,
};

/* RaBitQ encoder */
typedef struct
{
	CentroidEncoder		base;
	const float		   *vectors;
	Dimension			dim;
	const RaBitQParams *params;
	const float		   *global_mean;
} CentroidEncoderRaBitQ;

static inline void
centroid_encode_rabitq(CentroidEncoder *enc, uint32_t index, void *dest)
{
	CentroidEncoderRaBitQ *re	= (CentroidEncoderRaBitQ *)enc;
	VectorRef			   vref = {
						 .data = re->vectors + (size_t)index * re->dim,
						 .dim  = re->dim,
	 };
	VectorRef mref = {.data = re->global_mean, .dim = re->dim};
	mkt_rabitq_encode_into(re->params, vref, mref, (RaBitQData *)dest);
}

static const CentroidEncoderOps centroid_encoder_rabitq_ops = {
		.encode_into = centroid_encode_rabitq,
};

/* ----------------------------------------------------------------
 * CentroidEncoderState — stack-allocated encoder storage
 *
 * Union of all built-in encoder structs. Callers declare one on
 * the stack and pass it to centroid_encoder_init(), which returns
 * a CentroidEncoder * pointing into the union.
 *
 * RaBitQ params/global_mean are only used when fmt is RABITQ;
 * callers may pass NULL for unused fields.
 * ---------------------------------------------------------------- */
typedef union
{
	CentroidEncoderFloat  f;
	CentroidEncoderHalf	  h;
	CentroidEncoderRaBitQ r;
} CentroidEncoderState;

static inline CentroidEncoder *
centroid_encoder_init(
		CentroidEncoderState *state,
		MktCentroidFormat	  fmt,
		const float			 *vectors,
		Dimension			  dim,
		const RaBitQParams	 *params,
		const float			 *global_mean)
{
	switch (fmt)
	{
	case MKT_CENTROID_FMT_FLOAT:
		state->f = (CentroidEncoderFloat){
				.base.ops = &centroid_encoder_float_ops,
				.vectors  = vectors,
				.dim	  = dim,
		};
		return &state->f.base;
	case MKT_CENTROID_FMT_HALF:
		state->h = (CentroidEncoderHalf){
				.base.ops = &centroid_encoder_half_ops,
				.vectors  = vectors,
				.dim	  = dim,
		};
		return &state->h.base;
	case MKT_CENTROID_FMT_RABITQ:
		state->r = (CentroidEncoderRaBitQ){
				.base.ops	 = &centroid_encoder_rabitq_ops,
				.vectors	 = vectors,
				.dim		 = dim,
				.params		 = params,
				.global_mean = global_mean,
		};
		return &state->r.base;
	}
	return NULL;
}

/*
 * Write centroid entries to linked pages.
 *
 * Creates one or more centroid pages via storage->new_page, filling
 * each to capacity before allocating the next. Pages are linked
 * via next_blkno in the opaque area.
 *
 * Parameters:
 *   level        — tree level for page init (0 = root)
 *   flags        — per-entry flags (e.g. MKT_CENTROID_FLAG_LEAF)
 *   child_count  — uniform child count for all entries
 *   encoder      — vtable producing entry payloads on demand
 *   child_blknos — per-entry child block numbers
 *                   (NULL → InvalidBlockNumber for all entries)
 *
 * Returns the BlockNumber of the first centroid page.
 */
BlockNumber mkt_centroid_write_pages(
		MktStorage		  *storage,
		Dimension		   dim,
		uint32_t		   nlist,
		MktCentroidFormat  fmt,
		uint8_t			   level,
		uint16_t		   flags,
		uint16_t		   child_count,
		CentroidEncoder	  *encoder,
		const BlockNumber *child_blknos);

#endif /* MKT_CENTROID_BUILD_H */
