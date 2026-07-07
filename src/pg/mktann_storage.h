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
	DistanceMetric metric;	   /* distance metric for reranking */
	bool		   build_mode; /* skip per-page WAL during build */
	uint32_t	   read_count; /* debug: total page reads */

	/*
	 * Build-time centroid-page cache: a local copy of the blocks below
	 * first_posting (metadata + centroid tree), which are immutable during
	 * the posting and refine scans. Routing descends these pages on every
	 * row, so serving them from local memory removes the buffer-manager
	 * pin/lock protocol from the scans' hottest loop. NULL when disabled
	 * (reads go through the buffer cache as usual).
	 */
	char	   *cent_cache;
	BlockNumber cent_cache_nblocks;
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
 * Copy blocks [0, nblocks) into a local centroid-page cache so build-scan
 * routing reads them without buffer-manager traffic. Enabled only when the
 * copy fits max_bytes (build memory stays bounded); a no-op otherwise. The
 * caller must guarantee those blocks are immutable while the cache lives —
 * true between the centroid-tree write and the end of the posting scan.
 */
void mktann_storage_cache_centroids(
		MktannStorage *s, BlockNumber nblocks, uint64_t max_bytes);

/* Drop the centroid-page cache (safe if not loaded). */
void mktann_storage_uncache_centroids(MktannStorage *s);

#endif /* MKTANN_STORAGE_H */
