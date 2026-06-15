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
	/* Read-ahead pin ring (FIFO): pages pinned ahead of cur_buf so their
	 * codes can be prefetched while earlier pages are scanned. */
#define MKT_RA_RING 16
	Buffer	 ra_ring[MKT_RA_RING];
	uint32_t ra_head; /* next to promote/release */
	uint32_t ra_tail; /* next free slot (count = tail - head) */
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
 * Per-backend read-through cache for centroid pages.
 *
 * The centroid beam search re-reads nearly the entire centroid region
 * on every query (~hundreds of pages), and each read is a buffer-manager
 * pin (BufTableLookup) that is slow under large shared_buffers. Centroid
 * pages are immutable after build, so we cache their contents in backend
 * memory (keyed by block number) and serve subsequent reads with a direct
 * pointer — no pin, no content lock, no hash probe of the global buffer
 * table.
 *
 * Returns a MktStorage whose read path is the cache. `backing` supplies
 * the underlying buffer-cache storage used to populate cache misses; it
 * must remain valid for the duration of any scan that triggers a miss.
 * The cache resets automatically when the index relation changes.
 */
MktStorage *mktann_centroid_cache_get(Relation index, MktannStorage *backing);

#endif /* MKTANN_STORAGE_H */
