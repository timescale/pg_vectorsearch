/*
 * mktann_cache.c - Per-index cached state for mktann
 *
 * Note for Darwin: this file calls PostgreSQL's dynahash API
 * (hash_create/hash_search/...), whose bare names collide with unrelated
 * symbols of the same name in Apple's libSystem. Correct resolution here
 * depends on the extension being linked as a bundle with -bundle_loader
 * pointing at the postgres binary (see meson.build and src/pg/meson.build)
 * rather than the flat -undefined dynamic_lookup namespace search meson
 * uses by default for modules on Darwin -- without that, dyld can
 * silently bind these calls to libSystem's versions instead of
 * postgres's, corrupting state on first use.
 *
 * The rotation matrix P is O(dim³) to generate, so it lives in a
 * process-local, reference-counted hash table in CacheMemoryContext (keyed by
 * dim+seed) that survives relcache invalidation — see get_or_create_params /
 * mktann_release_params below. The global mean, P^T·global_mean, and an
 * immutable MktIndexBase template live in rd_amcache (allocated in
 * rd_indexcxt) and are cheap to re-populate when invalidated. The metadata
 * page is read at most once per backend, when rd_amcache is first populated.
 */

#include <postgres.h>

#include <storage/bufmgr.h>
#include <utils/hsearch.h>
#include <utils/memutils.h>

#include "mkt_pg.h"
#include "mktann_cache.h"
#include "mktann_meta.h"

/* ----------------------------------------------------------------
 * Process-local RaBitQParams cache
 *
 * Most deployments have one meerkat index per backend, but a backend can
 * legitimately have several scans of differently-dimensioned (or
 * differently-seeded) indexes open at once (e.g. a join across two
 * vector-indexed tables). A fixed number of slots can't represent that
 * without either dangling a live scan's pointer (freeing an in-use entry to
 * make room for another) or hard-erroring once every slot is checked out.
 *
 * Instead this is a backend-private dynahash table (plain hash_create() in
 * CacheMemoryContext -- not shared memory, so no shared_preload_libraries
 * entry is needed, and nothing here is shared across backends) keyed by
 * dim+seed, sized to grow as needed:
 *
 *   - get_or_create_params() looks up (dim, seed). On a hit, it bumps the
 *     entry's refcount and a decaying usage score and returns the cached
 *     matrix. On a miss, once the table is at or past a soft target size,
 *     it first tries to reclaim one idle (refcount == 0) entry -- the one
 *     with the lowest decayed usage, mirroring pg_stat_statements'
 *     entry_dealloc() -- before generating a new matrix. A held
 *     (refcount > 0) entry is *never* an eviction candidate, so a
 *     checked-out RaBitQParams* stays valid for as long as its owner holds
 *     it, no matter how many other dimensions get checked out around it.
 *     If nothing is currently idle, the table simply grows past the soft
 *     target rather than failing a live caller -- the target only bounds
 *     memory in the common case (few distinct live dimensions per
 *     backend); it is never a hard cap.
 *   - mktann_release_params() checks an entry back in (refcount--).
 * ---------------------------------------------------------------- */

/* Above this many live entries, try to reclaim an idle one before growing
 * further. Not a hard cap -- see above. */
#define MKT_RABITQ_CACHE_TARGET_ENTRIES 8

#define MKT_RABITQ_USAGE_INCREMENT 1.0
#define MKT_RABITQ_USAGE_DECAY	   0.99

typedef struct RaBitQCacheKey
{
	Dimension dim;
	uint64_t  seed;
} RaBitQCacheKey;

typedef struct RaBitQCacheEntry
{
	RaBitQCacheKey key;	   /* hash key; must be first */
	RaBitQParams  *params; /* NULL until fully initialized (see below) */
	int			   refcount;
	double		   usage;
} RaBitQCacheEntry;

static HTAB *rabitq_cache = NULL;

static void
rabitq_cache_init(void)
{
	HASHCTL ctl;

	memset(&ctl, 0, sizeof(ctl));
	ctl.keysize	  = sizeof(RaBitQCacheKey);
	ctl.entrysize = sizeof(RaBitQCacheEntry);
	ctl.hcxt	  = CacheMemoryContext;

	rabitq_cache = hash_create(
			"mktann rabitq params cache",
			MKT_RABITQ_CACHE_TARGET_ENTRIES,
			&ctl,
			HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);
}

/*
 * Decay every entry's usage score and reclaim the least-used entry with no
 * live checkouts, if one exists. A no-op (not an error) when every entry is
 * currently held -- the caller grows the table instead. Modeled on
 * pg_stat_statements' entry_dealloc().
 */
static void
rabitq_cache_evict_one(void)
{
	HASH_SEQ_STATUS	  seq;
	RaBitQCacheEntry *entry;
	RaBitQCacheEntry *victim = NULL;

	hash_seq_init(&seq, rabitq_cache);
	while ((entry = hash_seq_search(&seq)) != NULL)
	{
		entry->usage *= MKT_RABITQ_USAGE_DECAY;
		if (entry->refcount == 0 &&
			(victim == NULL || entry->usage < victim->usage))
			victim = entry;
	}
	/* The scan above always runs to completion (hash_seq_search returned
	 * NULL) before anything below mutates the table -- deleting mid-scan
	 * is only safe for the entry hash_seq_search just returned, which is
	 * why the victim is removed after, not during, the loop. */

	if (victim != NULL)
	{
		pfree(victim->params);
		hash_search(rabitq_cache, &victim->key, HASH_REMOVE, NULL);
	}
}

static RaBitQParams *
get_or_create_params(Dimension dim, uint64_t seed)
{
	RaBitQCacheKey key = {0}; /* zero incl. padding: HASH_BLOBS compares
							   * the whole struct */
	RaBitQCacheEntry *entry;
	bool			  found;

	if (rabitq_cache == NULL)
		rabitq_cache_init();

	key.dim	 = dim;
	key.seed = seed;

	entry = hash_search(rabitq_cache, &key, HASH_FIND, &found);
	if (found && entry->params != NULL)
	{
		entry->usage += MKT_RABITQ_USAGE_INCREMENT;
		entry->refcount++;
		return entry->params;
	}

	/* Miss (or a dead entry left by a failed init below -- params == NULL,
	 * refcount == 0, safe to recreate): try to make room first, but only
	 * when this is about to grow the table with a genuinely new key. */
	if (!found &&
		hash_get_num_entries(rabitq_cache) >= MKT_RABITQ_CACHE_TARGET_ENTRIES)
		rabitq_cache_evict_one();

	entry = hash_search(rabitq_cache, &key, HASH_ENTER, &found);

	/* Mark it dead/pending before any allocation that could throw: a
	 * failure below then leaves a well-defined, idle, re-creatable entry
	 * instead of one that looks live with garbage params. */
	entry->params	= NULL;
	entry->refcount = 0;
	entry->usage	= 0;

	MemoryContext old = MemoryContextSwitchTo(CacheMemoryContext);
	entry->params	  = palloc(MKT_RABITQ_PARAMS_SIZE(dim));
	MemoryContextSwitchTo(old);

	mkt_rabitq_init(entry->params, dim, seed);
	entry->refcount = 1;
	entry->usage	= MKT_RABITQ_USAGE_INCREMENT;

	return entry->params;
}

/*
 * Check an entry back in (public API; see mktann_cache.h). dim+seed must
 * match a currently-held entry exactly as returned by a prior
 * get_or_create_params call; a mismatch indicates a caller bug, not a
 * runtime condition.
 */
void
mktann_release_params(Dimension dim, uint64_t seed)
{
	RaBitQCacheKey	  key = {0};
	RaBitQCacheEntry *entry;
	bool			  found;

	if (rabitq_cache == NULL)
	{
		Assert(false);
		return;
	}

	key.dim	 = dim;
	key.seed = seed;

	entry = hash_search(rabitq_cache, &key, HASH_FIND, &found);
	if (found)
	{
		Assert(entry->refcount > 0);
		if (entry->refcount > 0)
			entry->refcount--;
		return;
	}

	/* Should be unreachable: every release_params call is paired with a
	 * prior successful get_or_create_params for the same dim/seed, and a
	 * held entry is never evicted. */
	Assert(false);
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
 * Return the per-backend cache, populating rd_amcache (one metapage read) on
 * first use. pt_global_mean is stored in the cache; params are NOT frozen
 * here (the process-local params cache is reference-counted per checkout,
 * not per-index) — callers rebind via get_or_create_params /
 * mktann_release_params.
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
	 * frozen for the backend's lifetime: params (checked out from the
	 * refcounted process-local cache; the caller must pair this with a
	 * matching mktann_release_params(base->dim, base->rabitq_seed) once it's
	 * done with base->params) and fastscan (resolved from the session GUC).
	 * Storage pointers stay zeroed for the caller. */
	*base					   = c->base;
	base->params			   = params;
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
