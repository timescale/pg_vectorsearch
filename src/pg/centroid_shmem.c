/*
 * centroid_shmem.c - Shared (DSM) compact FASTSCAN centroid cache
 *
 * A process-shared cache of the compact FASTSCAN centroid tree (see
 * centroid_compact.h / the per-node back-to-back group layout). One copy in a
 * dynamic shared area (DSA) serves all backends.
 *
 * Enabled only when meerkat is in shared_preload_libraries (so the control
 * struct + LWLock can be reserved at postmaster start). Not preloaded → no
 * cache; the beam search reads centroid pages. Standalone elides this file.
 *
 * Concurrency model — centroids are immutable until REINDEX/DROP:
 *   - Slots are keyed by relfilenode; a slot becomes READY once built and is
 *     never mutated, so reads are lock-free. A slot holds the WHOLE immutable
 *     tree, so nodes are addressed by a direct offset index (array keyed by
 *     blkno - base_blkno) — O(1), no search, no per-node miss.
 *   - A scan PINS its slot (refcount) for its duration; eviction only frees
 *     refcount==0 slots, so a reader's memory can't be freed under it.
 *   - The registry LWLock is held only briefly (claim/finalize/evict), never
 *     across page I/O: the compact form is built into backend-local memory
 *     first, then copied into DSA.
 *   - LRU by last_used; total DSA bytes bounded by mkt.centroid_cache_max_mb.
 */

#include <postgres.h>

#include <funcapi.h>
#include <miscadmin.h>
#include <port/atomics.h>
#include <storage/ipc.h>
#include <storage/lwlock.h>
#include <storage/shmem.h>
#include <utils/builtins.h>
#include <utils/dsa.h>
#include <utils/memutils.h>
#include <utils/rel.h>

#include "index/centroid_compact.h"
#include "index/centroid_page.h"
#include "mkt_pg.h"
#include "mktann_storage.h"
#include "quant/fastscan.h"

#define CC_MAX_SLOTS 64
#define CC_TRANCHE	 "mkt_centroid_cache"

typedef enum CcState
{
	CC_FREE = 0,
	CC_BUILDING,
	CC_READY,
} CcState;

/* One node's location within a slot's content buffer (build-time temp). */
typedef struct CcNode
{
	BlockNumber blkno;
	uint32_t	offset;
	uint32_t	entry_count;
} CcNode;

/* Direct index entry: a tree node addressed by (blkno - base_blkno). The slot
 * holds the whole immutable tree, so descent is a plain array index with no
 * search. offset == CC_NO_ENTRY marks a block with no centroid node (a gap in
 * an otherwise dense centroid block range). */
#define CC_NO_ENTRY UINT32_MAX

typedef struct CcEntry
{
	uint32_t offset;	  /* byte offset into the content buffer */
	uint32_t entry_count; /* centroids in this node */
} CcEntry;

typedef struct CcSlot
{
	RelFileNumber	 relfile;
	CcState			 state;
	dsa_pointer		 content;	 /* char[bytes] in DSA */
	dsa_pointer		 index;		 /* CcEntry[index_len] in DSA */
	uint32_t		 index_len;	 /* blocks covered: max_blkno-base_blkno+1 */
	BlockNumber		 base_blkno; /* index[0] is this block */
	Size			 bytes;		 /* content bytes (for the budget) */
	uint64_t		 last_used; /* LRU heuristic (written under shared lock) */
	pg_atomic_uint32 refcount;	/* active scans pinning this slot */
} CcSlot;

typedef struct CcControl
{
	LWLock			*lock; /* exclusive: build/evict; shared: hit/pin */
	dsa_handle		 area; /* DSA shared by all backends */
	int				 dsa_tranche;
	pg_atomic_uint64 tick;
	Size			 total_bytes; /* under exclusive lock */
	CcSlot			 slots[CC_MAX_SLOTS];
} CcControl;

/* Per-backend view of a pinned slot. Lives in the scan's memory context; holds
 * DSA-resolved (backend-local) pointers, valid while the slot is pinned. */
typedef struct CcView
{
	MktCentroidCompact iface; /* must be first (vtable) */
	const char		  *content;
	const CcEntry	  *index;
	uint32_t		   index_len;
	BlockNumber		   base; /* blkno of index[0] */
	int				   slot; /* slot index to unpin */
} CcView;

static CcControl *cc_ctl  = NULL; /* set in startup hook */
static dsa_area	 *cc_area = NULL; /* per-backend attach */
static bool cc_preloaded = false; /* set in init (preloaded), fork-inherited */
static shmem_request_hook_type prev_shmem_request_hook = NULL;
static shmem_startup_hook_type prev_shmem_startup_hook = NULL;

static Size
cc_shmem_size(void)
{
	return MAXALIGN(sizeof(CcControl));
}

static void
cc_shmem_request(void)
{
	if (prev_shmem_request_hook)
		prev_shmem_request_hook();
	RequestAddinShmemSpace(cc_shmem_size());
	RequestNamedLWLockTranche(CC_TRANCHE, 1);
}

static void
cc_shmem_startup(void)
{
	bool found;

	if (prev_shmem_startup_hook)
		prev_shmem_startup_hook();

	LWLockAcquire(AddinShmemInitLock, LW_EXCLUSIVE);
	cc_ctl = ShmemInitStruct("mkt centroid cache", cc_shmem_size(), &found);
	if (!found)
	{
		memset(cc_ctl, 0, cc_shmem_size());
		cc_ctl->lock		= &(GetNamedLWLockTranche(CC_TRANCHE))->lock;
		cc_ctl->area		= DSA_HANDLE_INVALID;
		cc_ctl->dsa_tranche = LWLockNewTrancheId();
		pg_atomic_init_u64(&cc_ctl->tick, 0);
		for (int i = 0; i < CC_MAX_SLOTS; i++)
			pg_atomic_init_u32(&cc_ctl->slots[i].refcount, 0);
	}
	LWLockRelease(AddinShmemInitLock);
}

/* Called from _PG_init only when meerkat is preloaded. */
void
mkt_centroid_shmem_init(void)
{
	cc_preloaded			= true;
	prev_shmem_request_hook = shmem_request_hook;
	shmem_request_hook		= cc_shmem_request;
	prev_shmem_startup_hook = shmem_startup_hook;
	shmem_startup_hook		= cc_shmem_startup;
}

/* Whether the cache can be used: true iff meerkat was preloaded (the control
 * struct + LWLock are only reserved at postmaster start). Set at _PG_init in
 * the postmaster and inherited by forked backends, so it is reliable in the
 * GUC check hook even before the shmem startup hook has run. */
bool
mkt_centroid_shmem_available(void)
{
	return cc_preloaded;
}

/* Attach (or create) the shared DSA for this backend. Caller holds cc_lock. */
static dsa_area *
cc_get_area_locked(void)
{
	if (cc_area != NULL)
		return cc_area;

	/* dsa_create/dsa_attach allocate the dsa_area bookkeeping struct in the
	 * current memory context; we cache it for the backend's life, so build it
	 * in TopMemoryContext (otherwise it dangles when the scan's context is
	 * freed) and dsa_pin_mapping so the mapping survives resource-owner
	 * release. */
	MemoryContext old = MemoryContextSwitchTo(TopMemoryContext);

	LWLockRegisterTranche(cc_ctl->dsa_tranche, CC_TRANCHE);
	if (cc_ctl->area == DSA_HANDLE_INVALID)
	{
		cc_area = dsa_create(cc_ctl->dsa_tranche);
		dsa_pin(cc_area); /* outlive the creating backend */
		cc_ctl->area = dsa_get_handle(cc_area);
	}
	else
	{
		cc_area = dsa_attach(cc_ctl->area);
	}
	dsa_pin_mapping(cc_area); /* keep attached for this backend's life */

	MemoryContextSwitchTo(old);
	return cc_area;
}

/* Evict the least-recently-used unpinned READY slot. Caller holds cc_lock.
 * Returns true if it freed something. */
static bool
cc_evict_lru_locked(void)
{
	int		 victim = -1;
	uint64_t best	= UINT64_MAX;

	for (int i = 0; i < CC_MAX_SLOTS; i++)
	{
		CcSlot *s = &cc_ctl->slots[i];
		if (s->state == CC_READY && pg_atomic_read_u32(&s->refcount) == 0 &&
			s->last_used < best)
		{
			best   = s->last_used;
			victim = i;
		}
	}
	if (victim < 0)
		return false;

	CcSlot *s = &cc_ctl->slots[victim];
	if (DsaPointerIsValid(s->content))
		dsa_free(cc_area, s->content);
	if (DsaPointerIsValid(s->index))
		dsa_free(cc_area, s->index);
	cc_ctl->total_bytes -= s->bytes;
	memset(s, 0, sizeof(*s));
	return true;
}

/* ----------------------------------------------------------------
 * Backend-local build of the compact form (no locks held).
 *
 * Walks the centroid tree from `first_centroid` via `backing`, copying each
 * node's FASTSCAN group sections back-to-back. Returns palloc'd buffers in the
 * current memory context, or false on a non-FASTSCAN page / read issue.
 * ---------------------------------------------------------------- */
static bool
cc_build_local(
		MktannStorage *backing,
		BlockNumber	   first_centroid,
		uint8_t		   nlevels,
		Dimension	   dim,
		Size		   budget,
		char		 **content_out,
		Size		  *used_out,
		CcEntry		 **index_out,
		BlockNumber	  *base_out,
		uint32_t	  *index_len_out)
{
	MktStorage *b	   = &backing->base;
	uint32_t	gbytes = mkt_centroid_fastscan_group_bytes(dim);

	Size		cap		  = 1u << 20;
	char	   *content	  = palloc(cap);
	Size		used	  = 0;
	uint32_t	nnodes	  = 0;
	uint32_t	nodes_cap = 256;
	CcNode	   *nodes	  = palloc(nodes_cap * sizeof(CcNode));
	BlockNumber min_blk	  = InvalidBlockNumber; /* 0xFFFFFFFF */
	BlockNumber max_blk	  = first_centroid;

	uint32_t	 qcap = 256, qhead = 0, qtail = 0;
	BlockNumber *queue = palloc(qcap * sizeof(BlockNumber));
	queue[qtail++]	   = first_centroid;

	bool ok = true;
	while (qhead < qtail && ok)
	{
		BlockNumber start		 = queue[qhead++];
		uint32_t	node_off	 = (uint32_t)used;
		uint32_t	node_entries = 0;
		BlockNumber cb			 = start;

		while (cb != InvalidBlockNumber)
		{
			Page				   page = b->ops->read_page(b, cb);
			MktCentroidPageOpaque *op	= MKT_CENTROID_OPAQUE(page);

			if (mkt_centroid_page_format(page) != MKT_CENTROID_FMT_FASTSCAN)
			{
				b->ops->release_page(b, cb);
				ok = false;
				break;
			}

			uint16_t ec		   = op->entry_count;
			bool	 is_leaf   = (op->level == nlevels - 1);
			uint32_t pg_groups = (ec + MKT_FASTSCAN_GROUP - 1) /
								 MKT_FASTSCAN_GROUP;
			Size  pg_bytes = (Size)pg_groups * gbytes;
			char *pcontent = (char *)PageGetContents(page);

			while (used + pg_bytes > cap)
			{
				cap *= 2;
				content = repalloc(content, cap);
			}
			memcpy(content + used, pcontent, pg_bytes);
			used += pg_bytes;
			node_entries += ec;

			if (!is_leaf)
			{
				for (uint32_t g = 0; g < pg_groups; g++)
				{
					uint32_t g_count = ec - g * MKT_FASTSCAN_GROUP;
					if (g_count > MKT_FASTSCAN_GROUP)
						g_count = MKT_FASTSCAN_GROUP;
					const BlockNumber *child = (const BlockNumber *)
							mkt_centroid_fastscan_group_child(
									pcontent, g, dim);
					for (uint32_t v = 0; v < g_count; v++)
					{
						if (qtail == qcap)
						{
							qcap *= 2;
							queue = repalloc(
									queue, qcap * sizeof(BlockNumber));
						}
						queue[qtail++] = child[v];
					}
				}
			}

			BlockNumber nb = op->next_blkno;
			b->ops->release_page(b, cb);
			cb = nb;
		}
		if (!ok)
			break;

		if (nnodes == nodes_cap)
		{
			nodes_cap *= 2;
			nodes = repalloc(nodes, nodes_cap * sizeof(CcNode));
		}
		nodes[nnodes].blkno		  = start;
		nodes[nnodes].offset	  = node_off;
		nodes[nnodes].entry_count = node_entries;
		nnodes++;
		if (start < min_blk)
			min_blk = start;
		if (start > max_blk)
			max_blk = start;

		if (used > budget) /* would exceed the whole budget on its own */
		{
			ok = false;
			break;
		}
	}

	pfree(queue);
	if (!ok || nnodes == 0)
	{
		pfree(content);
		pfree(nodes);
		return false;
	}

	/* Materialize the direct index: an array keyed by (blkno - base_blkno),
	 * so tree descent resolves a node by plain array indexing (no search).
	 * The centroid block range is dense, so gaps (if any) are rare and marked
	 * CC_NO_ENTRY. */
	uint32_t index_len = (uint32_t)(max_blk - min_blk + 1);
	CcEntry *entries   = palloc(index_len * sizeof(CcEntry));
	for (uint32_t i = 0; i < index_len; i++)
		entries[i].offset = CC_NO_ENTRY;
	for (uint32_t n = 0; n < nnodes; n++)
	{
		uint32_t i			   = nodes[n].blkno - min_blk;
		entries[i].offset	   = nodes[n].offset;
		entries[i].entry_count = nodes[n].entry_count;
	}
	pfree(nodes);

	*content_out   = content;
	*used_out	   = used;
	*index_out	   = entries;
	*base_out	   = min_blk;
	*index_len_out = index_len;
	return true;
}

static const char *
cc_view_lookup(
		const MktCentroidCompact *self,
		BlockNumber				  blkno,
		uint32_t				 *entry_count)
{
	const CcView *v = (const CcView *)self;

	if (blkno < v->base)
		return NULL;
	uint32_t i = (uint32_t)(blkno - v->base);
	if (i >= v->index_len || v->index[i].offset == CC_NO_ENTRY)
		return NULL;
	*entry_count = v->index[i].entry_count;
	return v->content + v->index[i].offset;
}

/* Attach the shared DSA for this backend (once per backend). */
static void
cc_ensure_area(void)
{
	if (cc_area != NULL)
		return;
	LWLockAcquire(cc_ctl->lock, LW_EXCLUSIVE);
	cc_get_area_locked();
	LWLockRelease(cc_ctl->lock);
}

/* Build a per-scan view from already-pinned slot pointers (no lock needed:
 * the slot is pinned so its DSA memory cannot be freed). */
static MktCentroidCompact *
cc_build_view(
		int			slot,
		dsa_pointer dp_content,
		dsa_pointer dp_index,
		uint32_t	index_len,
		BlockNumber base)
{
	CcView *v		= palloc(sizeof(CcView));
	v->iface.lookup = cc_view_lookup;
	v->content		= dsa_get_address(cc_area, dp_content);
	v->index		= dsa_get_address(cc_area, dp_index);
	v->index_len	= index_len;
	v->base			= base;
	v->slot			= slot;
	return &v->iface;
}

/*
 * Get (building if needed) the shared compact cache for `index`, pinning it
 * for the current scan. Returns a per-scan view to set on
 * MktIndexBase.centroid_compact, or NULL to fall back to page reads. The
 * caller must release the pin via mkt_centroid_shmem_unpin().
 *
 * Hits take only a SHARED registry lock and an atomic refcount bump, so many
 * backends read concurrently; the EXCLUSIVE lock is taken only to build a new
 * slot or evict, which is rare (centroids are immutable until REINDEX).
 */
MktCentroidCompact *
mkt_centroid_compact_get(
		Relation	   index,
		MktannStorage *backing,
		BlockNumber	   first_centroid,
		uint8_t		   nlevels,
		Dimension	   dim)
{
	if (cc_ctl == NULL)
		return NULL; /* not preloaded: no cache, read pages */

	RelFileNumber relfile = index->rd_locator.relNumber;
	Size		  budget  = (Size)mkt_centroid_cache_max_mb * 1024 * 1024;

	cc_ensure_area();

	/* Fast path: shared lock, hit on a READY slot. */
	LWLockAcquire(cc_ctl->lock, LW_SHARED);
	for (int i = 0; i < CC_MAX_SLOTS; i++)
	{
		CcSlot *s = &cc_ctl->slots[i];
		if (s->state == CC_FREE || s->relfile != relfile)
			continue;
		if (s->state == CC_READY)
		{
			pg_atomic_fetch_add_u32(&s->refcount, 1); /* pin */
			s->last_used		   = pg_atomic_fetch_add_u64(&cc_ctl->tick, 1);
			dsa_pointer dp_content = s->content;
			dsa_pointer dp_index   = s->index;
			uint32_t	index_len  = s->index_len;
			BlockNumber base	   = s->base_blkno;
			LWLockRelease(cc_ctl->lock);
			return cc_build_view(i, dp_content, dp_index, index_len, base);
		}
		LWLockRelease(cc_ctl->lock); /* BUILDING elsewhere: read pages */
		return NULL;
	}
	LWLockRelease(cc_ctl->lock);

	/* Miss: exclusive lock to claim a slot and build it. */
	LWLockAcquire(cc_ctl->lock, LW_EXCLUSIVE);
	int free_slot = -1;
	for (int i = 0; i < CC_MAX_SLOTS; i++)
	{
		CcSlot *s = &cc_ctl->slots[i];
		if (s->state != CC_FREE && s->relfile == relfile)
		{
			/* Another backend built or claimed it while we waited. */
			if (s->state == CC_READY)
			{
				pg_atomic_fetch_add_u32(&s->refcount, 1);
				s->last_used = pg_atomic_fetch_add_u64(&cc_ctl->tick, 1);
				dsa_pointer dp_content = s->content;
				dsa_pointer dp_index   = s->index;
				uint32_t	index_len  = s->index_len;
				BlockNumber base	   = s->base_blkno;
				LWLockRelease(cc_ctl->lock);
				return cc_build_view(i, dp_content, dp_index, index_len, base);
			}
			LWLockRelease(cc_ctl->lock);
			return NULL;
		}
		if (s->state == CC_FREE && free_slot < 0)
			free_slot = i;
	}
	if (free_slot < 0 && !cc_evict_lru_locked())
	{
		LWLockRelease(cc_ctl->lock); /* full, all pinned */
		return NULL;
	}
	if (free_slot < 0)
		for (int i = 0; i < CC_MAX_SLOTS; i++)
			if (cc_ctl->slots[i].state == CC_FREE)
			{
				free_slot = i;
				break;
			}
	int slot					= free_slot;
	cc_ctl->slots[slot].relfile = relfile;
	cc_ctl->slots[slot].state	= CC_BUILDING;
	LWLockRelease(cc_ctl->lock); /* build without holding the lock */

	char	   *content;
	Size		used;
	CcEntry	   *entries;
	BlockNumber base;
	uint32_t	index_len;
	bool		built = cc_build_local(
			   backing,
			   first_centroid,
			   nlevels,
			   dim,
			   budget,
			   &content,
			   &used,
			   &entries,
			   &base,
			   &index_len);

	LWLockAcquire(cc_ctl->lock, LW_EXCLUSIVE);
	CcSlot *s = &cc_ctl->slots[slot];
	if (!built)
	{
		s->state = CC_FREE;
		LWLockRelease(cc_ctl->lock);
		return NULL;
	}

	while (cc_ctl->total_bytes + used > budget && cc_evict_lru_locked())
		;

	dsa_pointer dp_content = dsa_allocate(cc_area, used);
	dsa_pointer dp_index = dsa_allocate(cc_area, index_len * sizeof(CcEntry));
	memcpy(dsa_get_address(cc_area, dp_content), content, used);
	memcpy(dsa_get_address(cc_area, dp_index),
		   entries,
		   index_len * sizeof(CcEntry));
	pfree(content);
	pfree(entries);

	s->content	  = dp_content;
	s->index	  = dp_index;
	s->index_len  = index_len;
	s->base_blkno = base;
	s->bytes	  = used;
	s->last_used  = pg_atomic_fetch_add_u64(&cc_ctl->tick, 1);
	pg_atomic_init_u32(&s->refcount, 1); /* pinned for this scan */
	s->state = CC_READY;
	cc_ctl->total_bytes += used;
	LWLockRelease(cc_ctl->lock);

	return cc_build_view(slot, dp_content, dp_index, index_len, base);
}

/* Release the pin taken by mkt_centroid_compact_get (lock-free). */
void
mkt_centroid_shmem_unpin(MktCentroidCompact *compact)
{
	if (compact == NULL || cc_ctl == NULL)
		return;
	CcView *v = (CcView *)compact;
	pg_atomic_fetch_sub_u32(&cc_ctl->slots[v->slot].refcount, 1);
}

/* ----------------------------------------------------------------
 * SQL introspection (for tests and observability)
 * ---------------------------------------------------------------- */

static const char *cc_state_names[] = {
		[CC_FREE]	  = "free",
		[CC_BUILDING] = "building",
		[CC_READY]	  = "ready",
};

PG_FUNCTION_INFO_V1(mkt_centroid_cache_available);
PG_FUNCTION_INFO_V1(mkt_centroid_cache_stats);

/* mkt_centroid_cache_available() -> bool: is the shared cache usable (i.e. was
 * meerkat preloaded)? When false, enabling the cache errors and queries read
 * centroid pages. */
Datum
mkt_centroid_cache_available(PG_FUNCTION_ARGS)
{
	PG_RETURN_BOOL(mkt_centroid_shmem_available());
}

/* mkt_centroid_cache_stats() -> one row per occupied (non-FREE) slot. Reads a
 * consistent snapshot under the SHARED registry lock. Returns no rows when the
 * cache is unavailable (not preloaded). relfilenode is reported as an oid so
 * it can be joined to pg_class.relfilenode; an orphan slot (dropped/reindexed
 * index) shows a relfilenode that no longer resolves to a relation. */
Datum
mkt_centroid_cache_stats(PG_FUNCTION_ARGS)
{
	ReturnSetInfo *rsinfo = (ReturnSetInfo *)fcinfo->resultinfo;

	InitMaterializedSRF(fcinfo, 0);

	if (cc_ctl == NULL)
		return (Datum)0; /* not preloaded: no rows */

	LWLockAcquire(cc_ctl->lock, LW_SHARED);
	for (int i = 0; i < CC_MAX_SLOTS; i++)
	{
		CcSlot *s = &cc_ctl->slots[i];
		if (s->state == CC_FREE)
			continue;

		Datum values[7];
		bool  nulls[7] = {0};

		values[0] = Int32GetDatum(i);
		values[1] = ObjectIdGetDatum((Oid)s->relfile);
		values[2] = CStringGetTextDatum(cc_state_names[s->state]);
		values[3] = Int64GetDatum((int64)s->bytes);
		values[4] = Int32GetDatum((int32)s->index_len);
		values[5] = Int64GetDatum((int64)s->last_used);
		values[6] = Int32GetDatum((int32)pg_atomic_read_u32(&s->refcount));

		tuplestore_putvalues(
				rsinfo->setResult, rsinfo->setDesc, values, nulls);
	}
	LWLockRelease(cc_ctl->lock);

	return (Datum)0;
}
