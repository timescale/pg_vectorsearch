/*
 * amcache.c - Per-index cached state for mktann
 *
 * The rotation matrix P is O(dim³) to generate, so it lives in a
 * process-local, reference-counted hash table (keyed by dim+seed) that
 * survives relcache invalidation — see get_or_create_params /
 * mktann_release_params below. The table and the matrices it owns live in a
 * dedicated child context of CacheMemoryContext, so the cache's footprint is
 * visible as its own line in a memory-context dump. The global mean,
 * P^T·global_mean, and an immutable MktIndexBase template live in rd_amcache
 * (allocated in rd_indexcxt) and are cheap to re-populate when invalidated.
 * The metadata page is read at most once per backend, when rd_amcache is first
 * populated.
 */

#include <postgres.h>

#include <storage/bufmgr.h>
#include <utils/hsearch.h>
#include <utils/memutils.h>
#include <utils/resowner.h>

#include "amcache.h"
#include "core/log.h"
#include "meta.h"
#include "support_pg.h"
#include "typeinfo.h"

/* ----------------------------------------------------------------
 * Process-local RaBitQParams cache
 *
 * A backend can have several scans of differently-dimensioned (or
 * differently-seeded) indexes open at once (e.g. a join across two
 * vector-indexed tables). A fixed number of slots can't represent that
 * without either dangling a live scan's pointer (freeing an in-use entry to
 * make room for another) or hard-erroring once every slot is checked out.
 *
 * Instead this is a backend-private dynahash table (plain hash_create() in
 * rabitq_cache_cxt, a child of CacheMemoryContext -- not shared memory, so
 * no shared_preload_libraries entry is needed, and nothing here is shared
 * across backends) keyed by dim+seed, sized to grow as needed:
 *
 *   - get_or_create_params() looks up (dim, seed). On a hit, it bumps the
 *     entry's refcount and a decaying usage score and returns the cached
 *     matrix. On a miss, once the table is at or past a soft target size,
 *     it first tries to reclaim one idle (refcount == 0) entry -- the one
 *     with the lowest decayed usage -- before generating a new matrix. A
 *     held (refcount > 0) entry is *never* an eviction candidate, so a
 *     checked-out RaBitQParams* stays valid for as long as its owner holds
 *     it, no matter how many other dimensions get checked out around it.
 *     If nothing is currently idle, the table simply grows past the soft
 *     target rather than failing a live caller -- the target is not a
 *     hard cap.
 *   - mktann_release_params() checks an entry back in (refcount--).
 *     Checkouts are additionally tracked by the checkout-time resource
 *     owner, so error paths that skip the release still return the
 *     refcount -- see rabitq_params_ref_desc below.
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

/* Owns the hash table and every cached matrix; child of
 * CacheMemoryContext, so it has backend lifetime but shows up as its own
 * line in a memory-context dump. */
static MemoryContext rabitq_cache_cxt = NULL;

/*
 * Error safety for the refcounts: every checkout is also registered
 * with the resource owner current at checkout time. If an error keeps
 * the matching release from running -- an aborted scan's portal is
 * dropped without amendscan, or an insert errors between checkout and
 * release -- the owner's release sweep calls back here and returns the
 * refcount, so no entry is left unevictable. A checkout leaked on the
 * commit path additionally gets PostgreSQL's standard "resource was not
 * closed" warning, turning a pairing bug into a visible failure.
 */
static void rabitq_params_ref_release(Datum res);

static const ResourceOwnerDesc rabitq_params_ref_desc = {
		.name			  = "mktann rabitq params ref",
		.release_phase	  = RESOURCE_RELEASE_BEFORE_LOCKS,
		.release_priority = RELEASE_PRIO_FIRST,
		.ReleaseResource  = rabitq_params_ref_release,
		.DebugPrint		  = NULL,
};

static void
rabitq_params_ref_release(Datum res)
{
	RaBitQCacheEntry *entry = (RaBitQCacheEntry *)DatumGetPointer(res);

	/* Only reached for checkouts no normal release forgot: error paths. */
	Assert(entry->refcount > 0);
	if (entry->refcount > 0)
		entry->refcount--;
}

static void
rabitq_cache_init(void)
{
	HASHCTL ctl;

	/* The context survives mktann_rabitq_cache_clear() resets. */
	if (rabitq_cache_cxt == NULL)
		rabitq_cache_cxt = AllocSetContextCreate(
				CacheMemoryContext,
				"mktann rabitq params cache",
				ALLOCSET_DEFAULT_SIZES);

	memset(&ctl, 0, sizeof(ctl));
	ctl.keysize	  = sizeof(RaBitQCacheKey);
	ctl.entrysize = sizeof(RaBitQCacheEntry);
	ctl.hcxt	  = rabitq_cache_cxt;

	rabitq_cache = hash_create(
			"mktann rabitq params cache",
			MKT_RABITQ_CACHE_TARGET_ENTRIES,
			&ctl,
			HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);
}

/*
 * Decay every entry's usage score and reclaim the least-used entry with no
 * live checkouts, if one exists. A no-op (not an error) when every entry is
 * currently held -- the caller grows the table instead.
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
		/* Enlarge first: it can allocate, and nothing may fail between
		 * the refcount bump and the (no-fail) Remember. */
		ResourceOwnerEnlarge(CurrentResourceOwner);
		entry->usage += MKT_RABITQ_USAGE_INCREMENT;
		entry->refcount++;
		ResourceOwnerRemember(
				CurrentResourceOwner,
				PointerGetDatum(entry),
				&rabitq_params_ref_desc);
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
	 * instead of one that looks live with garbage params. The matrix is
	 * generated into a local pointer and only published once fully
	 * initialized, for the same reason (an error out of mkt_rabitq_init
	 * must not leave a live-looking entry holding a garbage matrix). */
	entry->params	= NULL;
	entry->refcount = 0;
	entry->usage	= 0;

	ResourceOwnerEnlarge(CurrentResourceOwner);

	RaBitQParams *params =
			MemoryContextAlloc(rabitq_cache_cxt, MKT_RABITQ_PARAMS_SIZE(dim));
	mkt_rabitq_init(params, dim, seed);

	entry->params	= params;
	entry->refcount = 1;
	entry->usage	= MKT_RABITQ_USAGE_INCREMENT;
	ResourceOwnerRemember(
			CurrentResourceOwner,
			PointerGetDatum(entry),
			&rabitq_params_ref_desc);

	return entry->params;
}

/*
 * Check an entry back in (public API; see amcache.h). dim+seed must
 * match a currently-held entry exactly as returned by a prior
 * get_or_create_params call; a mismatch indicates a caller bug, not a
 * runtime condition.
 */
void
mktann_release_params(Dimension dim, uint64_t seed, ResourceOwner owner)
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
		/* Forget first: it errors on a pairing bug (wrong owner), and
		 * the refcount must stay consistent with the registrations. */
		ResourceOwnerForget(
				owner, PointerGetDatum(entry), &rabitq_params_ref_desc);
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

/*
 * Test support: snapshot the cache into a caller-provided array and
 * return the number of entries written. A plain exported symbol (no SQL
 * surface): the test-only module test/pg/src/test_helpers.c
 * wraps it in a set-returning function so the regression tests can
 * observe refcounts, usage decay, and eviction. Inert otherwise.
 */
int
mktann_rabitq_cache_stats(MktRabitqCacheStat *stats, int max_stats)
{
	HASH_SEQ_STATUS	  seq;
	RaBitQCacheEntry *entry;
	int				  n = 0;

	if (rabitq_cache == NULL)
		return 0;

	/* The scan must run to completion (see rabitq_cache_evict_one). */
	hash_seq_init(&seq, rabitq_cache);
	while ((entry = hash_seq_search(&seq)) != NULL)
	{
		if (n >= max_stats)
			continue;
		stats[n].dim	  = (int32_t)entry->key.dim;
		stats[n].refcount = entry->refcount;
		stats[n].usage	  = entry->usage;
		n++;
	}
	return n;
}

/*
 * Test support: reset the cache to its initial state, freeing the hash
 * table and every cached matrix in one context reset (they all live in
 * rabitq_cache_cxt). Returns the number of entries dropped. Refuses if
 * any entry is currently checked out -- freeing a held matrix is
 * exactly the use-after-free this cache exists to prevent. Exported
 * for the test-only module test/pg/src/test_helpers.c; inert otherwise.
 */
int
mktann_rabitq_cache_clear(void)
{
	HASH_SEQ_STATUS	  seq;
	RaBitQCacheEntry *entry;
	int				  n = 0;

	if (rabitq_cache == NULL)
		return 0;

	hash_seq_init(&seq, rabitq_cache);
	while ((entry = hash_seq_search(&seq)) != NULL)
	{
		if (entry->refcount > 0)
		{
			hash_seq_term(&seq);
			mkt_error(
					"cannot clear the RaBitQ params cache: entry for "
					"dimension %u has %d live checkout(s)",
					(unsigned)entry->key.dim,
					entry->refcount);
		}
		n++;
	}

	rabitq_cache = NULL;
	MemoryContextReset(rabitq_cache_cxt);
	return n;
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
 *
 * Two separate constraints shape this, and they pull in different directions.
 *
 * ONE ALLOCATION, and that is not stylistic. PostgreSQL frees rd_amcache with
 * a single targeted pfree in RelationInvalidateRelation, which does NOT delete
 * rd_indexcxt — the relation survives with rd_isvalid = false. So a sub-object
 * hung off this blob would leak into rd_indexcxt once per invalidation cycle.
 * Relcache's own caches (RelationGetIndexList and friends) can be lazy
 * per-allocation because relcache wrote per-field cleanup for each of them; an
 * access method gets one line of cleanup written for it, so it gets one
 * allocation. Nor can the blob be grown and swapped later: MktIndexBase
 * .pt_global_mean points into it and a live scan holds that pointer for the
 * duration of the scan.
 *
 * LAZY CONTENT, which the single allocation does not prevent. The metapage
 * read is the floor — dim comes from it and dim sizes the blob — but the work
 * that follows it need not be eager. The rule for a new field: fill it here if
 * computing it is cheap or it needs the pinned metapage buffer (the
 * global_mean memcpy is both); otherwise give it a ready-flag and compute it
 * on first use, as pt_global_mean does. Note that anything living in this blob
 * inherits the metapage read as a precondition even when it would not
 * otherwise need one -- type_info, which comes from the opclass, is the case
 * in point.
 * ---------------------------------------------------------------- */

typedef struct AmCacheData
{
	MktIndexBase base; /* immutable template; params / fastscan /
						* storage left zeroed (rebound per call) */
	const MktIndexTypeInfo *type_info; /* indexed column's type, from the
										* opclass */
	bool	 has_fastscan; /* index built with FASTSCAN posting pages */
	bool	 pt_ready;	   /* pt_global_mean computed (rotation done) */
	uint32_t nlist;
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

	/* Opclass-derived, so no metapage needed -- but resolved here so every
	 * post-build path reads it as a pointer instead of an fmgr call. */
	c->type_info = mkt_index_type_info(index);

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

const MktIndexTypeInfo *
mktann_cache_type_info(Relation index)
{
	return get_cache_data(index)->type_info;
}

MktannScanInfo
mktann_cache_scan_info(Relation index)
{
	AmCacheData *c = get_cache_data(index);
	return (MktannScanInfo){.nlist = c->nlist};
}
