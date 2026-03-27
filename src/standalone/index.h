/*
 * index.h - Standalone in-memory index
 *
 * Holds centroid tree (for beam search routing) and per-cluster
 * posting lists (for vector scanning). Used by both the CLI
 * benchmark and Python bindings.
 *
 * The index is built from a flat vector array and supports
 * different centroid formats (RaBitQ, float32, float16) and
 * posting list configurations.
 */

#ifndef MKT_STANDALONE_INDEX_H
#define MKT_STANDALONE_INDEX_H

#include <stdint.h>

#include "core/memory.h"
#include "index/centroid_page.h"
#include "index/storage.h"
#include "mkt_types.h"
#include "quant/rabitq.h"
#include "standalone/vector_source.h"

/* ----------------------------------------------------------------
 * Per-cluster posting list (standalone, contiguous arrays)
 *
 * RaBitQ fields are parallel SoA arrays for cache-friendly
 * quantized scanning. The vectors array is only touched during
 * reranking of survivors (~1% of scanned vectors).
 * ---------------------------------------------------------------- */
typedef struct MktPostingList
{
	uint32_t  count;	/* vectors in this cluster */
	uint32_t  capacity; /* allocated slots */
	uint32_t *ids;		/* [count] original vector IDs */
	float	 *vectors;	/* [count * dim] full-precision */

	/* RaBitQ parallel arrays (NULL if not enabled) */
	float	*f_add;		/* [count] */
	float	*f_rescale; /* [count] */
	float	*f_error;	/* [count] */
	uint8_t *bits;		/* [count * packed_bytes] */
} MktPostingList;

/* ----------------------------------------------------------------
 * Array-backed page storage for centroid pages
 * ---------------------------------------------------------------- */
typedef struct ArrayPageStorage
{
	MktStorage base; /* must be first */
	char	  *pages;
	uint32_t   next_blkno;
	uint32_t   page_cap;
} ArrayPageStorage;

/* ----------------------------------------------------------------
 * Index configuration
 * ---------------------------------------------------------------- */
typedef struct MktIndexConfig
{
	uint32_t		  nlist;		/* 0 = auto: sqrt(nvecs) */
	uint32_t		  fan_out;		/* 0 = auto from nlist */
	MktCentroidFormat centroid_fmt; /* rabitq, float, half */
	DistanceMetric	  metric;
	uint32_t		  km_nredo;		 /* k-means restarts (0 = default) */
	uint32_t		  km_max_iter;	 /* k-means iterations (0 = default) */
	bool			  encode_rabitq; /* encode posting lists with RaBitQ */
} MktIndexConfig;

/* ----------------------------------------------------------------
 * In-memory index
 * ---------------------------------------------------------------- */
typedef struct MktIndex
{
	/* Centroid tree */
	ArrayPageStorage  centroid_storage;
	BlockNumber		  first_centroid;
	uint8_t			  nlevels;
	uint32_t		  fan_out;
	RaBitQParams	 *rq_params;
	float			 *global_mean;
	MktCentroidFormat centroid_fmt;

	/* Posting lists */
	MktPostingList *lists;			/* [nlist] */
	float		   *leaf_centroids; /* [nlist * dim] for per-cluster qstate */
	uint32_t		nlist;
	uint32_t		nvecs; /* total vectors across all lists */

	Dimension	   dim;
	DistanceMetric metric;

	/* Memory context owning all index allocations.
	 * Deleting this frees everything at once. */
	MktMemCtx memctx;
} MktIndex;

/* ----------------------------------------------------------------
 * Build + destroy
 * ---------------------------------------------------------------- */

/*
 * Build an index by streaming vectors from a source.
 *
 * Two-pass build:
 *   1. Samples vectors for hierarchical k-means clustering
 *   2. Resets source, then assigns each vector to its nearest
 *      cluster's posting list
 *
 * If encode_rabitq is set, RaBitQ data is computed per posting
 * list entry after assignment.
 *
 * Returns NULL on failure.
 */
MktIndex *mkt_index_build(MktVectorSource *src, const MktIndexConfig *config);

/*
 * Free the index and all owned memory.
 */
void mkt_index_destroy(MktIndex *idx);

#endif /* MKT_STANDALONE_INDEX_H */
