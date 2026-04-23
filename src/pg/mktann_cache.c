/*
 * mktann_cache.c - Per-index cached state for mktann
 *
 * The rotation matrix P is O(dim³) to generate, so it lives in a
 * process-local static in CacheMemoryContext (keyed by dim+seed)
 * that survives relcache invalidation. The global mean and
 * P^T·global_mean live in rd_amcache (allocated in rd_indexcxt)
 * and are cheap to re-populate when invalidated.
 */

#include <postgres.h>

#include <storage/bufmgr.h>
#include <utils/memutils.h>

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
 * rd_amcache layout (global_mean + pt_global_mean)
 * ---------------------------------------------------------------- */

typedef struct AmCacheData
{
	Dimension dim;
	uint64_t  seed;
	uint32_t  global_mean_off;
	uint32_t  pt_global_mean_off;
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

/* ----------------------------------------------------------------
 * Public API
 * ---------------------------------------------------------------- */

MktannIndexCache
mktann_cache_get(Relation index)
{
	if (index->rd_amcache != NULL)
	{
		AmCacheData *c = (AmCacheData *)index->rd_amcache;
		return (MktannIndexCache){
				.params			= get_or_create_params(c->dim, c->seed),
				.global_mean	= cache_global_mean(c),
				.pt_global_mean = cache_pt_global_mean(c),
				.dim			= c->dim,
		};
	}

	/* Read metadata page */
	Buffer meta_buf = ReadBuffer(index, 0);
	LockBuffer(meta_buf, BUFFER_LOCK_SHARE);
	Page meta_page = BufferGetPage(meta_buf);

	const MktannMetaPage *meta = (const MktannMetaPage *)PageGetSpecialPointer(
			meta_page);
	Assert(meta->magic == MKT_META_MAGIC);

	Dimension dim  = meta->dim;
	uint64_t  seed = meta->rabitq_seed;

	RaBitQParams *params = get_or_create_params(dim, seed);

	/* global_mean + pt_global_mean in rd_amcache */
	uint32_t gm_off	   = MAXALIGN(sizeof(AmCacheData));
	uint32_t pt_gm_off = gm_off + dim * sizeof(float);
	Size	 total	   = pt_gm_off + dim * sizeof(float);

	MemoryContext old = MemoryContextSwitchTo(index->rd_indexcxt);
	AmCacheData	 *c	  = palloc(total);
	MemoryContextSwitchTo(old);

	c->dim				  = dim;
	c->seed				  = seed;
	c->global_mean_off	  = gm_off;
	c->pt_global_mean_off = pt_gm_off;

	/* Copy global mean */
	const float *src_mean = mktann_meta_global_mean_const(meta);
	memcpy(cache_global_mean(c), src_mean, dim * sizeof(float));

	UnlockReleaseBuffer(meta_buf);

	/* Compute P^T * global_mean */
	mkt_rabitq_rotate(params, cache_global_mean(c), cache_pt_global_mean(c));

	index->rd_amcache = c;

	return (MktannIndexCache){
			.params			= params,
			.global_mean	= cache_global_mean(c),
			.pt_global_mean = cache_pt_global_mean(c),
			.dim			= dim,
	};
}
