/*
 * mktann_cache.h - Per-index cached state for mktann
 *
 * MktannIndexCache holds expensive-to-compute state that is constant
 * for the lifetime of an index. Lazily initialized on first access
 * per backend via mktann_cache_get().
 *
 * The rotation matrix P (O(dim³)) lives in a process-local static
 * in CacheMemoryContext, keyed by dim+seed — survives relcache
 * invalidation. The global mean and P^T·global_mean live in
 * rd_amcache (allocated in rd_indexcxt, freed automatically on
 * relcache invalidation) and are cheap to re-populate.
 */

#ifndef MKTANN_CACHE_H
#define MKTANN_CACHE_H

#include <postgres.h>

#include <utils/rel.h>

#include "mkt_types.h"
#include "quant/rabitq.h"

typedef struct MktannIndexCache
{
	RaBitQParams *params;
	const float	 *global_mean;
	const float	 *pt_global_mean;
	Dimension	  dim;
} MktannIndexCache;

/*
 * Get the cached index state, initializing if needed.
 *
 * Safe to call on every beginscan — fast path when already cached.
 * The returned pointers are valid until the next relcache
 * invalidation; do not store them past yield points.
 */
MktannIndexCache mktann_cache_get(Relation index);

#endif /* MKTANN_CACHE_H */
