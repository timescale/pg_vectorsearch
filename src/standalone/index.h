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

#include <pthread.h>
#include <stdint.h>

#include "core/memory.h"
#include "index/centroid_page.h"
#include "index/index_base.h"
#include "index/index_build.h"
#include "index/storage.h"
#include "mkt_types.h"
#include "quant/rabitq.h"
#include "standalone/vector_source.h"

/* ----------------------------------------------------------------
 * Per-cluster vector ID list (for brute-force scan fallback)
 * ---------------------------------------------------------------- */
typedef struct MktClusterList
{
	uint32_t  count;	/* vectors in this cluster */
	uint32_t  capacity; /* allocated slots */
	uint32_t *ids;		/* [count] original vector IDs */
} MktClusterList;

/* ----------------------------------------------------------------
 * Array-backed page storage for centroid pages
 * ---------------------------------------------------------------- */
typedef struct ArrayPageStorage
{
	MktStorage		base; /* must be first */
	char		   *pages;
	uint32_t		next_blkno;
	uint32_t		page_cap;
	MktMemCtx		memctx;		 /* owning context for page growth */
	pthread_mutex_t alloc_mutex; /* protects next_blkno + page growth */
	const float	   *all_vectors; /* for reranking (NULL if not set) */
	uint32_t		nvecs;
	DistanceMetric	metric;
	MktTopK			rerank_topk;	/* pre-allocated, reset per query */
	MktTopKEntry   *rerank_entries; /* pre-allocated extraction buffer */
	uint32_t		rerank_cap;		/* entries buffer capacity */
} ArrayPageStorage;

/* ----------------------------------------------------------------
 * Posting format
 * ---------------------------------------------------------------- */
typedef enum MktPostingFormat
{
	MKT_POSTING_FMT_FLAT  = 0, /* one contiguous buffer per cluster */
	MKT_POSTING_FMT_PAGES = 1, /* chain of BLCKSZ pages in storage */
} MktPostingFormat;

/* ----------------------------------------------------------------
 * Index configuration
 * ---------------------------------------------------------------- */
typedef struct MktIndexConfig
{
	uint32_t		  nlist;		/* 0 = auto: sqrt(nvecs) */
	uint32_t		  fan_out;		/* 0 = auto from nlist */
	MktCentroidFormat centroid_fmt; /* rabitq, float, half */
	DistanceMetric	  metric;
	uint32_t		  km_nredo;			/* k-means restarts (0 = default) */
	uint32_t		  km_max_iter;		/* k-means iterations (0 = default) */
	double			  soar_lambda;		/* SOAR replication (0 = off) */
	double			  boundary_epsilon; /* boundary gate threshold (0 = off) */
	bool			  encode_rabitq;	/* encode posting lists with RaBitQ */
	MktPostingFormat  posting_fmt;		/* flat or pages */
	int				  fastscan;			/* 0=off, 8=uint8 LUT, 16=uint16 LUT */
	int32_t			  nworkers;			/* -1 = auto, 0 = serial */
} MktIndexConfig;

/* ----------------------------------------------------------------
 * In-memory index
 * ---------------------------------------------------------------- */
typedef struct MktIndex
{
	/* Common index descriptor (passed to MktSearchCtx) */
	MktIndexBase base;

	/* Concrete storage (base.centroid_storage/posting_storage
	 * point into these) */
	ArrayPageStorage centroid_storage;
	ArrayPageStorage posting_storage;

	uint32_t fan_out;
	float	*global_mean;

	/* Full-precision vectors for reranking (indexed by vector_id) */
	float *all_vectors; /* [nvecs * dim] */

	/* Posting data for RaBitQ scan (flat or paged) */
	MktPostingFormat posting_fmt;
	BlockNumber		 first_posting;	   /* paged mode: cluster c's head = this + c */
	char		   **flat_pages;	   /* flat mode: [nlist] buffers */
	uint32_t		 max_cluster_size; /* largest cluster entry count */
	bool			 has_posting_data;
	bool			 has_replication;

	/* Per-cluster ID lists for brute-force fallback */
	MktClusterList *clusters; /* [nlist] */

	float	*leaf_centroids; /* [nlist * dim] for per-cluster qstate */
	float	*pt_centroids;	 /* [nlist * dim] P^T * leaf_centroids */
	uint32_t nlist;
	uint32_t nvecs; /* total vectors across all lists */

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
MktIndex *mkt_index_build(
		MktVectorSource		 *src,
		const MktIndexConfig *config,
		MktBuildStats		 *stats);

/*
 * Free the index and all owned memory.
 */
void mkt_index_destroy(MktIndex *idx);

#endif /* MKT_STANDALONE_INDEX_H */
