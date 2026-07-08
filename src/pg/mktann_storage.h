/*
 * mktann_storage.h - PG buffer cache MktStorage implementation
 *
 * MktannStorage embeds MktStorage as its first member and adds PG-
 * specific context (Relation, current buffer). A single static vtable
 * is shared across all instances.
 */

#ifndef MKTANN_STORAGE_H
#define MKTANN_STORAGE_H

#include <postgres.h>

#include <storage/bufmgr.h>
#include <utils/rel.h>

#include "index/centroid_compact.h"
#include "index/storage.h"

/* ----------------------------------------------------------------
 * PG buffer cache storage
 * ---------------------------------------------------------------- */
typedef struct MktannStorage
{
	MktStorage	   base; /* must be first */
	Relation	   index;
	Relation	   rel; /* table relation for rerank (NULL during build) */
	Buffer		   cur_buf;
	Buffer		   ra_buf; /* read-ahead pin held alongside cur_buf */
	DistanceMetric metric;	   /* distance metric for reranking */
	bool		   build_mode; /* skip per-page WAL during build */
	uint32_t	   read_count; /* debug: total page reads */
} MktannStorage;

/*
 * Initialize an MktannStorage backed by the PostgreSQL buffer cache.
 *
 * For write operations, pages are WAL-logged via GenericXLog.
 * For read operations, pages are share-locked.
 *
 * rel is the table relation for reranking (fetching full-precision
 * vectors). Pass NULL during build (no reranking needed).
 */
void mktann_storage_init(
		MktannStorage *s, Relation index, Relation rel, DistanceMetric metric);

/*
 * Lazily set the heap relation after init. Updates the vtable to
 * use the read_stream-optimized rerank path when applicable.
 */
void mktann_storage_set_rel(MktannStorage *s, Relation rel);

/*
 * Per-backend compact centroid cache (FASTSCAN centroids only).
 *
 * The centroid beam search re-reads nearly the entire centroid region on every
 * query, and each page read is a buffer-manager pin (BufTableLookup) that is
 * slow under large shared_buffers. Centroids are immutable until REINDEX/DROP,
 * so we build a dedicated, search-optimized copy once per backend: each tree
 * node's FASTSCAN group sections concatenated back-to-back, addressable by the
 * node's first block number. The beam search scores a node straight from this
 * buffer — no pins, no page headers, no next_blkno walk — and reads are
 * lock-free (immutable data).
 *
 * Keyed by relfilenode, so REINDEX (new relfilenode) rebuilds. An LRU registry
 * across indexes is bounded by mkt.centroid_cache_max_mb; an index whose
 * compact form exceeds the whole budget is not cached.
 *
 * Returns a compact source to set on MktIndexBase.centroid_compact, or NULL if
 * the cache could not be built (non-FASTSCAN, over budget, or read error) — in
 * which case the caller leaves centroid_compact NULL and reads pages.
 * `backing` supplies the buffer-cache storage used to read pages during the
 * one-time build.
 */
MktCentroidCompact *mkt_centroid_compact_get(
		Relation	   index,
		MktannStorage *backing,
		BlockNumber	   first_centroid,
		uint8_t		   nlevels,
		Dimension	   dim);

/* Shared (DSM) centroid cache (implemented in centroid_shmem.c).
 * mkt_centroid_shmem_init() registers the shmem hooks and must be called
 * from _PG_init only when meerkat is in shared_preload_libraries. When not
 * preloaded the cache is unavailable and mkt_centroid_compact_get() returns
 * NULL (page reads). A scan that obtained a compact source must release it
 * with mkt_centroid_shmem_unpin() at end-of-scan. */
void mkt_centroid_shmem_init(void);
bool mkt_centroid_shmem_available(void);
void mkt_centroid_shmem_unpin(MktCentroidCompact *compact);

#endif /* MKTANN_STORAGE_H */
