/*
 * mktann_cache.h - Per-index cached state for mktann
 *
 * Holds the index's build-time-immutable parameters so the hot paths
 * (aminsert, beginscan) don't re-read and re-parse the metadata page on every
 * call. The state is read from the metapage at most once per backend and kept
 * in rd_amcache (allocated in rd_indexcxt, freed automatically on relcache
 * invalidation, e.g. REINDEX).
 *
 * What lives where:
 *   - The rotation matrix P (O(dim³) to build) is a process-local static in
 *     CacheMemoryContext keyed by dim+seed; it survives relcache invalidation.
 *   - The global mean, P^T·global_mean, and a fully-populated immutable
 *     MktIndexBase template live in rd_amcache.
 *
 * mktann_index_base_init fills a caller-owned MktIndexBase from the cache. The
 * only fields it leaves for the caller are the storage pointers
 * (centroid_storage / posting_storage / page_base): those reference a per-call
 * MktannStorage with transient mutable state and must stay per-operation, so
 * they are not cached.
 */

#ifndef MKTANN_CACHE_H
#define MKTANN_CACHE_H

#include <postgres.h>

#include <utils/rel.h>

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
 */
void mktann_index_base_init(Relation index, MktIndexBase *base);

/* Variant for parallel query workers: bind caller-supplied rotation
 * params and rotated global mean (from cluster-shared / scan-shared
 * memory) instead of building them locally. Both must outlive the
 * scan; neither is installed into any process-local cache. */
void mktann_index_base_init_shared(
		Relation	  index,
		MktIndexBase *base,
		RaBitQParams *params,
		const float	 *pt_global_mean);

/* Backend-cached rotation params (leader-side publication source). */
RaBitQParams *mktann_cache_params(Relation index);

/* Cluster-shared rotation params in a postmaster-lifetime named DSM
 * segment, created and filled on first request per (dim, seed). */
RaBitQParams *mktann_params_shared(Dimension dim, uint64_t seed);

/* Rotated global mean from the per-backend cache (computed on first
 * use). Pointer valid until relcache invalidation. */
const float *mktann_cache_pt_global_mean(Relation index);

/* Non-throwing metapage format check for plan-time callers: true if the
 * index's on-disk format is one this build can read. */
bool mktann_cache_format_ok(Relation index);

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
 * Build-time scan-planning scalars (immutable), from the same cache.
 */
typedef struct MktannScanInfo
{
	uint32_t nlist;	  /* number of leaf centroids */
	uint32_t ntuples; /* tuples present at build time */
} MktannScanInfo;

MktannScanInfo mktann_cache_scan_info(Relation index);

#endif /* MKTANN_CACHE_H */
