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
	case MKT_CENTROID_FMT_FASTSCAN:
		/* fastscan uses a group-packed page layout that doesn't fit
		 * this per-vector encoder model. The build path emits these
		 * pages directly (see mkt_centroid_write_fastscan_pages —
		 * future work) or they're produced by an in-place conversion. */
		return NULL;
	}
	return NULL;
}

/*
 * Write centroid entries to linked pages.
 *
 * Fills pages to capacity, linking via next_blkno. When start_blkno
 * is InvalidBlockNumber, appends via new_page. Otherwise writes to
 * pre-reserved blocks starting at start_blkno via write_page.
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
		const BlockNumber *child_blknos,
		const float		  *pt_centroids,
		BlockNumber		   start_blkno);

/*
 * Write centroid entries as FASTSCAN-format pages.
 *
 * Same inputs as mkt_centroid_write_pages but emits 32-vector groups
 * with kPerm0-packed RaBitQ codes instead of per-entry RaBitQData,
 * so the scan path uses mkt_fastscan_accumulate_hacc rather than
 * mkt_rabitq_inner_product_multi at score time.
 *
 * Requires `params` and `global_mean` (the RaBitQ encoding inputs);
 * the float `vectors` array provides the centroids to encode. Each
 * entry's f_error is computed from f_add / f_rescale at emit time
 * so the score path doesn't need to recompute it.
 */
BlockNumber mkt_centroid_write_fastscan_pages(
		MktStorage		   *storage,
		Dimension			dim,
		uint32_t			nlist,
		uint8_t				level,
		uint16_t			flags,
		const RaBitQParams *params,
		const float		   *vectors,
		const float		   *global_mean,
		const BlockNumber  *child_blknos,
		BlockNumber			start_blkno);

#endif /* MKT_CENTROID_BUILD_H */
