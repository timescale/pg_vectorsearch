/*
 * bufstorage.h - PG buffer cache MktStorage implementation
 *
 * MktannStorage embeds MktStorage as its first member and adds PG-
 * specific context (Relation, current buffer). A single static vtable
 * is shared across all instances.
 */

#ifndef MKT_BUFSTORAGE_H
#define MKT_BUFSTORAGE_H

#include <postgres.h>

#include <storage/bufmgr.h>
#include <utils/rel.h>

#include "index/storage.h"

/* ----------------------------------------------------------------
 * PG buffer cache storage
 * ---------------------------------------------------------------- */

/* Buffer-id cache effectiveness counters (diagnostic; reset and read
 * by the mkt_routing_stats SQL functions). */
extern uint64_t mkt_bufcache_hits;
extern uint64_t mkt_bufcache_cold;
extern uint64_t mkt_bufcache_stale;

typedef struct MktannStorage
{
	MktStorage	   base; /* must be first */
	Relation	   index;
	Relation	   rel; /* table relation for rerank (NULL during build) */
	Buffer		   cur_buf;
	DistanceMetric metric;	   /* distance metric for reranking */
	bool		   build_mode; /* skip per-page WAL during build */
	uint32_t	   read_count; /* debug: total page reads */
	/* Column type, from the opclass; rerank reads the heap attribute through
	 * it. Resolved at init. */
	const struct MktIndexTypeInfo *type_info;
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
 * Storage for an inspection sweep: reads go through the shared page layer,
 * but skip the recent-buffer cache so a sweep neither evicts the slots the
 * query path reuses nor reports itself in the cache counters.
 */
void mktann_storage_init_inspect(MktannStorage *s, Relation index);

/*
 * Toggle the backend-local buffer-id cache (mkt.recent_buffers): re-pin
 * index pages via ReadRecentBuffer instead of a buffer-mapping hash
 * lookup per page. Stale entries self-heal via ReadBuffer fallback.
 */
void mktann_storage_set_recent_buffers(bool enabled);

#endif /* MKT_BUFSTORAGE_H */
