/*
 * centroid_compact.h - Optional compact, in-memory centroid source
 *
 * A dedicated, search-optimized cache of the centroid tree for FASTSCAN
 * centroids. Centroids are immutable after build, so the cache is built once,
 * read lock-free, and invalidated only on REINDEX/DROP (keyed by relfilenode
 * on the PG side).
 *
 * Each tree node (a beam candidate's children) is stored as its own contiguous
 * run of FASTSCAN group sections — exactly the on-page group layout, copied
 * back-to-back across the node's pages — addressable by the node's first block
 * number. The beam search scores a node straight from this buffer (no buffer
 * pins, no page headers, no next_blkno walk).
 *
 * This is a vtable so the core search path (centroid_search.c) can consume it
 * without depending on the PG-specific implementation that builds it.
 */

#ifndef MKT_CENTROID_COMPACT_H
#define MKT_CENTROID_COMPACT_H

#include "mkt_types.h"

/* BlockNumber: from pg_compat (standalone) or PostgreSQL (extension). */
#ifdef MKT_STANDALONE
#include "standalone/pg_compat.h"
#else
#include <postgres.h>

#include <storage/block.h>
#endif

typedef struct MktCentroidCompact MktCentroidCompact;

struct MktCentroidCompact
{
	/*
	 * Look up a node by its first block number. On hit, returns a pointer to
	 * the node's contiguous FASTSCAN group bytes and sets *entry_count to the
	 * node's centroid count; returns NULL on miss (caller falls back to
	 * pages).
	 */
	const char *(*lookup)(
			const MktCentroidCompact *self,
			BlockNumber				  blkno,
			uint32_t				 *entry_count);
};

static inline const char *
mkt_centroid_compact_lookup(
		const MktCentroidCompact *c, BlockNumber blkno, uint32_t *entry_count)
{
	return c->lookup(c, blkno, entry_count);
}

#endif /* MKT_CENTROID_COMPACT_H */
