/*
 * amcache.h - Per-index cached state for mktann
 *
 * Holds the index's build-time-immutable parameters so the hot paths
 * (aminsert, beginscan) don't re-read and re-parse the metadata page on every
 * call. The state is read from the metapage at most once per backend and kept
 * in rd_amcache (allocated in rd_indexcxt, freed automatically on relcache
 * invalidation, e.g. REINDEX).
 *
 * What lives where:
 *   - The rotation matrix P (O(dim³) to build) lives in a dynamically-sized,
 *     reference-counted, process-local hash table in CacheMemoryContext keyed
 *     by dim+seed; it survives relcache invalidation. Every checkout of
 *     base->params from mktann_index_base_init MUST be paired with a
 *     matching mktann_release_params(base->dim, base->rabitq_seed) once the
 *     caller is done with it (scan close / end of insert) — the table only
 *     frees an entry once its checkout count drops to zero, so a live
 *     scan's params pointer is never invalidated out from under it by
 *     another index of a different dim/seed.
 *   - The global mean, P^T·global_mean, the column's type descriptor, and a
 *     fully-populated immutable MktIndexBase template live in rd_amcache.
 *     These are pure data with nothing to release, which is what rd_amcache
 *     can hold: it is freed with rd_indexcxt and has no teardown hook, so
 *     anything needing an explicit release (the params checkout above) has to
 *     stay out of it.
 *
 * mktann_index_base_init fills a caller-owned MktIndexBase from the cache. The
 * only fields it leaves for the caller are the storage pointers
 * (centroid_storage / posting_storage / page_base): those reference a per-call
 * MktannStorage with transient mutable state and must stay per-operation, so
 * they are not cached.
 */

#ifndef MKT_AMCACHE_H
#define MKT_AMCACHE_H

#include <postgres.h>

#include <utils/rel.h>
#include <utils/resowner.h>

#include "index/centroid_page.h"
#include "index/index_base.h"

/*
 * Populate the immutable (build-time) fields of *base from the per-backend
 * cache: params, pt_global_mean, rabitq_seed, dim, nlevels, first_centroid,
 * metric, centroid_format, and fastscan. fastscan is resolved from the
 * mkt.fastscan_bits GUC against the index's stored FASTSCAN flag (so a session
 * SET still takes effect); params is resolved from the process-local rotation
 * cache. The storage pointers (centroid_storage / posting_storage / page_base)
 * are left zeroed for the caller to wire to its own MktannStorage.
 *
 * Pointers placed into *base stay valid until the next relcache invalidation;
 * do not retain them across yield points.
 *
 * base->params is checked out from the process-local rotation-matrix cache
 * and MUST be released with a matching mktann_release_params(base->dim,
 * base->rabitq_seed, owner) when the caller is done with it, where owner is
 * the CurrentResourceOwner observed at this call — see amcache.c. The
 * checkout is registered with that owner, so an error path that skips the
 * release (aborted scan, failed insert) still returns the refcount when the
 * owner is released.
 */
void mktann_index_base_init(Relation index, MktIndexBase *base);

/*
 * Release a params checkout obtained via mktann_index_base_init. dim and seed
 * must be exactly the values observed on the corresponding base->dim /
 * base->rabitq_seed at checkout time; owner is the CurrentResourceOwner that
 * was in effect at the mktann_index_base_init call.
 */
void mktann_release_params(Dimension dim, uint64_t seed, ResourceOwner owner);

/*
 * Test support: one snapshot row per cached RaBitQ rotation matrix in
 * this backend's params cache. Consumed by the test-only module
 * test/pg/src/test_helpers.c; no SQL surface in the extension.
 */
typedef struct MktRabitqCacheStat
{
	int32_t dim;
	int32_t refcount;
	double	usage;
} MktRabitqCacheStat;

int mktann_rabitq_cache_stats(MktRabitqCacheStat *stats, int max_stats);
int mktann_rabitq_cache_clear(void);

/*
 * Immutable dim + distance metric + first posting page from the cache, without
 * forcing the lazy rotation-matrix work that mktann_index_base_init does. For
 * metadata-only callers such as VACUUM's ambulkdelete (which uses
 * first_posting to skip straight past the centroid region).
 */
void mktann_cache_meta(
		Relation		index,
		Dimension	   *dim,
		DistanceMetric *metric,
		BlockNumber	   *first_posting);

/*
 * Build-time scan-planning scalar (immutable), from the same cache. A struct
 * rather than a bare return so a second scalar can join it without touching
 * every caller.
 */
typedef struct MktannScanInfo
{
	uint32_t		  nlist; /* number of leaf centroids */
	Dimension		  dim;
	uint8_t			  nlevels;		 /* centroid tree depth */
	uint8_t			  fan_out;		 /* children per tree node */
	BlockNumber		  first_posting; /* one past the last centroid page */
	MktCentroidFormat centroid_format;
	bool			  has_fastscan; /* built with FASTSCAN posting pages */
} MktannScanInfo;

MktannScanInfo mktann_cache_scan_info(Relation index);

/*
 * The indexed column's type descriptor, from the opclass (see
 * typeinfo.h). Immutable for the life of the relcache entry, so it is
 * resolved once per backend here rather than per call.
 *
 * Only safe once the index has a metadata page: this goes through the same
 * cache as everything else, and populating that cache reads the metapage. The
 * build paths therefore call mkt_index_type_info() directly.
 */
const struct MktIndexTypeInfo *mktann_cache_type_info(Relation index);

#endif /* MKT_AMCACHE_H */
