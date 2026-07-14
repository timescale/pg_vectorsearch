/*
 * mktann_cache.c - Per-index cached state for mktann
 *
 * The rotation matrix P is O(dim³) to generate, so it lives in a
 * process-local static in CacheMemoryContext (keyed by dim+seed)
 * that survives relcache invalidation. The global mean, P^T·global_mean,
 * and an immutable MktIndexBase template live in rd_amcache (allocated in
 * rd_indexcxt) and are cheap to re-populate when invalidated. The metadata
 * page is read at most once per backend, when rd_amcache is first populated.
 */

#include <postgres.h>

#include <storage/bufmgr.h>
#include <storage/dsm_registry.h>
#include <utils/memutils.h>

#include "mkt_pg.h"
#include "mktann_cache.h"
#include "mktann_meta.h"

/* ----------------------------------------------------------------
 * Process-local RaBitQParams cache
 *
 * Single entry — most deployments have one meerkat index per
 * backend. If dim+seed match, reuse; otherwise regenerate.
 * ---------------------------------------------------------------- */

static RaBitQParams *cached_params;
static Dimension	 cached_dim;
static uint64_t		 cached_seed;

static RaBitQParams *
get_or_create_params(Dimension dim, uint64_t seed)
{
	if (cached_params != NULL && cached_dim == dim && cached_seed == seed)
		return cached_params;

	if (cached_params != NULL)
	{
		pfree(cached_params);
		cached_params = NULL;
	}

	MemoryContext old = MemoryContextSwitchTo(CacheMemoryContext);
	cached_params	  = palloc(MKT_RABITQ_PARAMS_SIZE(dim));
	MemoryContextSwitchTo(old);

	mkt_rabitq_init(cached_params, dim, seed);
	cached_dim	= dim;
	cached_seed = seed;

	return cached_params;
}

/* ----------------------------------------------------------------
 * rd_amcache layout
 *
 * A fully-populated immutable MktIndexBase template (params + storage left
 * for the per-call rebind), the scan-planning scalars, and the inline
 * global_mean + pt_global_mean vectors appended after the struct.
 *
 * The metapage-derived scalars (dim, metric, nlist, ...) are filled eagerly on
 * first access. The rotated pt_global_mean — which needs the O(dim³) rotation
 * matrix — is computed lazily (pt_ready) only when a caller actually needs it
 * (the scan / insert path via mktann_index_base_init), so metadata-only
 * consumers like VACUUM's ambulkdelete get dim/metric from the cache without
 * paying for rotation setup.
 * ---------------------------------------------------------------- */

typedef struct AmCacheData
{
	MktIndexBase base;	   /* immutable template; params / fastscan /
							* storage left zeroed (rebound per call) */
	bool	 has_fastscan; /* index built with FASTSCAN posting pages */
	bool	 pt_ready;	   /* pt_global_mean computed (rotation done) */
	uint32_t nlist;
	uint32_t ntuples;
	uint32_t global_mean_off;
	uint32_t pt_global_mean_off;
} AmCacheData;

static inline float *
cache_global_mean(AmCacheData *c)
{
	return (float *)((char *)c + c->global_mean_off);
}

static inline float *
cache_pt_global_mean(AmCacheData *c)
{
	return (float *)((char *)c + c->pt_global_mean_off);
}

/*
 * Non-throwing on-disk format probe. The planner costs every mktann
 * index on the queried table, including ones it will never choose; an
 * index written by an incompatible format must yield "cannot serve"
 * costs there, not abort the whole query. Scan paths keep the loud
 * ERROR in get_cache_data.
 */
bool
mktann_cache_format_ok(Relation index)
{
	if (index->rd_amcache != NULL)
		return true; /* already read and validated */

	Buffer meta_buf = ReadBuffer(index, 0);
	LockBuffer(meta_buf, BUFFER_LOCK_SHARE);
	const MktannMetaPage *meta = (const MktannMetaPage *)PageGetSpecialPointer(
			BufferGetPage(meta_buf));
	bool ok = meta->magic == MKT_META_MAGIC;
	UnlockReleaseBuffer(meta_buf);
	return ok;
}

/*
 * Return the per-backend cache, populating rd_amcache (one metapage read) on
 * first use. pt_global_mean is stored in the cache; params are NOT frozen here
 * (the process-local single-entry cache can evict them) — callers rebind via
 * get_or_create_params.
 */
static AmCacheData *
get_cache_data(Relation index)
{
	if (index->rd_amcache != NULL)
		return (AmCacheData *)index->rd_amcache;

	/* Read the metadata page once. */
	Buffer meta_buf = ReadBuffer(index, 0);
	LockBuffer(meta_buf, BUFFER_LOCK_SHARE);
	Page meta_page = BufferGetPage(meta_buf);

	const MktannMetaPage *meta = (const MktannMetaPage *)PageGetSpecialPointer(
			meta_page);
	/* Reject an index whose metapage was written by an incompatible format
	 * (loud in release too, not just a debug Assert) — its layout would
	 * otherwise be misread. */
	if (meta->magic != MKT_META_MAGIC)
	{
		uint32_t got = meta->magic;
		UnlockReleaseBuffer(meta_buf);
		ereport(ERROR,
				(errcode(ERRCODE_INDEX_CORRUPTED),
				 errmsg("index \"%s\" has an incompatible on-disk format "
						"(metapage magic 0x%08X, expected 0x%08X)",
						RelationGetRelationName(index),
						got,
						(uint32_t)MKT_META_MAGIC),
				 errhint("REINDEX the index to rebuild it in the current "
						 "format.")));
	}

	Dimension dim  = meta->dim;
	uint64_t  seed = meta->rabitq_seed;

	uint32_t gm_off	   = MAXALIGN(sizeof(AmCacheData));
	uint32_t pt_gm_off = gm_off + dim * sizeof(float);
	Size	 total	   = pt_gm_off + dim * sizeof(float);

	MemoryContext old = MemoryContextSwitchTo(index->rd_indexcxt);
	AmCacheData	 *c	  = palloc0(total);
	MemoryContextSwitchTo(old);

	c->global_mean_off	  = gm_off;
	c->pt_global_mean_off = pt_gm_off;
	c->has_fastscan		  = (meta->flags & MKT_META_FLAG_FASTSCAN) != 0;
	c->nlist			  = meta->nlist;
	c->ntuples			  = meta->ntuples;

	/* Immutable base template (storage / params / fastscan rebound per call).
	 */
	c->base.dim				= dim;
	c->base.nlevels			= meta->nlevels;
	c->base.fan_out			= meta->fan_out;
	c->base.nlist			= meta->nlist;
	c->base.first_centroid	= meta->first_centroid;
	c->base.first_posting	= meta->first_posting;
	c->base.metric			= (DistanceMetric)meta->metric;
	c->base.centroid_format = (MktCentroidFormat)meta->centroid_format;
	c->base.rabitq_seed		= seed;
	/* base.pt_global_mean stays NULL until mktann_index_base_init computes it.
	 */

	/* Copy global mean; P^T * global_mean is computed lazily (see
	 * mktann_index_base_init) so metadata-only callers skip the rotation. */
	memcpy(cache_global_mean(c),
		   mktann_meta_global_mean_const(meta),
		   dim * sizeof(float));

	UnlockReleaseBuffer(meta_buf);

	index->rd_amcache = c;
	return c;
}

/* ----------------------------------------------------------------
 * Public API
 * ---------------------------------------------------------------- */

void
mktann_index_base_init(Relation index, MktIndexBase *base)
{
	AmCacheData	 *c = get_cache_data(index);
	RaBitQParams *params =
			get_or_create_params(c->base.dim, c->base.rabitq_seed);

	/* Compute the rotated global mean on first use (needs the rotation
	 * matrix); cached thereafter for the backend. */
	if (!c->pt_ready)
	{
		mkt_rabitq_rotate(
				params, cache_global_mean(c), cache_pt_global_mean(c));
		c->base.pt_global_mean = cache_pt_global_mean(c);
		c->pt_ready			   = true;
	}

	/* Copy the immutable template, then rebind the fields that cannot be
	 * frozen for the backend's lifetime: params (process-local cache may have
	 * evicted them) and fastscan (resolved from the session GUC). Storage
	 * pointers stay zeroed for the caller. */
	*base					   = c->base;
	base->params			   = params;
	base->fastscan			   = c->has_fastscan ? mkt_fastscan_bits : 0;
	base->centroid_error_scale = (float)mkt_centroid_error_scale;
	base->centroid_beam_scale  = (float)mkt_centroid_beam_scale;
}

/*
 * Like mktann_index_base_init, but binds the supplied rotation params
 * instead of the process-local cache. For parallel query workers: the
 * leader publishes its RaBitQParams through the scan's shared memory so
 * a fresh worker process skips the O(dim^3) matrix construction. The
 * caller owns the params' lifetime (they must outlive the scan); they
 * are never installed into the process-local cache.
 */
/* The backend-cached rotation params for this index (built on first
 * use). For the parallel-scan leader to publish into shared memory. */
RaBitQParams *
mktann_cache_params(Relation index)
{
	AmCacheData *c = get_cache_data(index);
	return get_or_create_params(c->base.dim, c->base.rabitq_seed);
}

/* ----------------------------------------------------------------
 * Cluster-wide rotation params (DSM registry)
 *
 * The rotation matrix depends only on (dim, seed), so one copy in a
 * postmaster-lifetime named DSM segment serves every backend and every
 * parallel worker: the alternative -- copying the ~MB flat struct into
 * each parallel scan's per-query DSM -- put the copy on every parallel
 * query's critical path. The segment is created (and filled) by
 * whichever process asks first; parallel workers merely attach.
 * ---------------------------------------------------------------- */

/* Creation-callback inputs (GetNamedDSMSegment callbacks take no
 * arguments; the registry serializes creation, so a static stage is
 * safe). src == NULL means "build from dim+seed" -- the O(dim^3)
 * construction, paid at most once per postmaster lifetime per
 * (dim, seed). */
static struct
{
	const RaBitQParams *src;
	Dimension			dim;
	uint64_t			seed;
} params_segment_stage;

static void
params_segment_init(void *ptr)
{
	if (params_segment_stage.src != NULL)
		memcpy(ptr,
			   params_segment_stage.src,
			   MKT_RABITQ_PARAMS_SIZE(params_segment_stage.dim));
	else
		mkt_rabitq_init(
				(RaBitQParams *)ptr,
				params_segment_stage.dim,
				params_segment_stage.seed);
}

RaBitQParams *
mktann_params_shared(Dimension dim, uint64_t seed)
{
	/* One-entry attach memo, mirroring the process-local params cache:
	 * repeated parallel scans in one backend attach once. */
	static RaBitQParams *attached;
	static Dimension	 attached_dim;
	static uint64_t		 attached_seed;

	if (attached != NULL && attached_dim == dim && attached_seed == seed)
		return attached;

	char name[64];
	snprintf(
			name,
			sizeof(name),
			"meerkat-rabitq-%u-%016" PRIx64,
			(unsigned)dim,
			seed);

	/* Seed the creation callback from the process-local cache when this
	 * backend already built the matrix (the leader always has: sizing
	 * runs get_or_create_params before workers exist). */
	params_segment_stage.src  = (cached_params != NULL && cached_dim == dim &&
								 cached_seed == seed)
									  ? cached_params
									  : NULL;
	params_segment_stage.dim  = dim;
	params_segment_stage.seed = seed;

	bool  found;
	void *ptr = GetNamedDSMSegment(
			name, MKT_RABITQ_PARAMS_SIZE(dim), params_segment_init, &found);

	attached	  = (RaBitQParams *)ptr;
	attached_dim  = dim;
	attached_seed = seed;
	return attached;
}

/* Rotated global mean from the per-backend cache, computing it on
 * first use (needs the rotation matrix). For the parallel-scan leader
 * to publish through the scan's shared memory, sparing each worker the
 * per-query O(dim^2) rotation. */
const float *
mktann_cache_pt_global_mean(Relation index)
{
	AmCacheData *c = get_cache_data(index);

	if (!c->pt_ready)
	{
		RaBitQParams *params =
				get_or_create_params(c->base.dim, c->base.rabitq_seed);
		mkt_rabitq_rotate(
				params, cache_global_mean(c), cache_pt_global_mean(c));
		c->base.pt_global_mean = cache_pt_global_mean(c);
		c->pt_ready			   = true;
	}
	return cache_pt_global_mean(c);
}

void
mktann_index_base_init_shared(
		Relation	  index,
		MktIndexBase *base,
		RaBitQParams *params,
		const float	 *pt_global_mean)
{
	AmCacheData *c = get_cache_data(index);

	*base		 = c->base;
	base->params = params;
	/* Read-only here: the in-place rotate in mkt_index_ensure_rabitq
	 * only runs when params is NULL, and params is always supplied on
	 * this path. */
	base->pt_global_mean	   = unconstify(float *, pt_global_mean);
	base->fastscan			   = c->has_fastscan ? mkt_fastscan_bits : 0;
	base->centroid_error_scale = (float)mkt_centroid_error_scale;
	base->centroid_beam_scale  = (float)mkt_centroid_beam_scale;
}

void
mktann_cache_meta(
		Relation		index,
		Dimension	   *dim,
		DistanceMetric *metric,
		BlockNumber	   *first_posting)
{
	AmCacheData *c = get_cache_data(index);
	*dim		   = c->base.dim;
	*metric		   = c->base.metric;
	*first_posting = c->base.first_posting;
}

MktannScanInfo
mktann_cache_scan_info(Relation index)
{
	AmCacheData *c = get_cache_data(index);
	return (MktannScanInfo){.nlist = c->nlist, .ntuples = c->ntuples};
}
