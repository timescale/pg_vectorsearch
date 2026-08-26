/*
 * mktann_maint.c - SQL-callable index maintenance (incremental split)
 *
 * Exposes mkt.split_postinglist(regclass, bigint) and mkt.compact(regclass),
 * which drive the backend-neutral posting-list split (mkt_posting_split) on a
 * live index. Both run in their own transaction, take an exclusive page lock
 * on each cluster they split (serializing against concurrent inserts), and
 * persist the updated leaf count into the metapage.
 *
 * Phase 1 supports flat (nlevels == 1) RaBitQ-centroid indexes; other shapes
 * raise a clear error.
 */

#include <postgres.h>

#include <access/relation.h>
#include <access/table.h>
#include <access/tableam.h>
#include <access/transam.h>
#include <access/xact.h>
#include <catalog/index.h>
#include <catalog/pg_class.h>
#include <executor/tuptable.h>
#include <fmgr.h>
#include <miscadmin.h>
#include <storage/bufmgr.h>
#include <storage/lmgr.h>
#include <utils/acl.h>
#include <utils/inval.h>
#include <utils/lsyscache.h>
#include <utils/rel.h>
#include <utils/snapmgr.h>

#include "index/index_base.h"
#include "index/posting_page.h"
#include "index/posting_split.h"
#include "mktann_cache.h"
#include "mktann_maint.h"
#include "mktann_meta.h"
#include "mktann_storage.h"
#include "support_pg.h"
#include "types/vector.h"

PG_FUNCTION_INFO_V1(mkt_split_postinglist);
PG_FUNCTION_INFO_V1(mkt_merge_postinglist);
PG_FUNCTION_INFO_V1(mkt_compact);

/*
 * These functions mutate the index, so require ownership of the underlying
 * table (mirrors the conversion function in mktann_inspect.c).
 */
static void
require_index_owner(Relation index, LOCKMODE lockmode)
{
	Oid heaprelid = IndexGetRelation(RelationGetRelid(index), false);
	if (!object_ownercheck(RelationRelationId, heaprelid, GetUserId()))
	{
		char	  *relname = get_rel_name(heaprelid);
		ObjectType objtype = get_relkind_objtype(get_rel_relkind(heaprelid));
		relation_close(index, lockmode);
		aclcheck_error(ACLCHECK_NOT_OWNER, objtype, relname);
	}
}

/* ----------------------------------------------------------------
 * Heap vector fetch for re-clustering (the MktSplitEnv seam)
 * ---------------------------------------------------------------- */

typedef struct PgSplitFetchCtx
{
	Relation		heap;
	AttrNumber		attnum;
	TupleTableSlot *slot;
} PgSplitFetchCtx;

static bool
pg_split_fetch_vector(
		void *ctx, ItemPointerData tid, float *out, Dimension dim)
{
	PgSplitFetchCtx *c = (PgSplitFetchCtx *)ctx;

	/* SnapshotAny: the posting entry exists until VACUUM tombstones it, so we
	 * re-cluster against whatever it still points at; a fully-pruned tuple
	 * just drops out of the split (returns false). Same contract as rerank. */
	if (!table_tuple_fetch_row_version(c->heap, &tid, SnapshotAny, c->slot))
		return false;

	bool  isnull;
	Datum val = slot_getattr(c->slot, c->attnum, &isnull);
	bool  ok  = false;
	if (!isnull)
	{
		MktVector *vec = DatumGetMktVector(val);
		if ((Dimension)vec->dim == dim)
		{
			memcpy(out, vec->x, (size_t)dim * sizeof(float));
			ok = true;
		}
	}
	ExecClearTuple(c->slot);
	return ok;
}

/*
 * Retire the split's old chain (the MktSplitEnv seam). Rather than tombstoning
 * it now — which would make scans skip a head a concurrent query may still be
 * about to read from a stale pre-flip pointer — mark each page DELETED and
 * stamp the head with the current next-XID. The chain stays linked and
 * readable; VACUUM physically retires it once that XID clears the global
 * visibility horizon (see reclaim_retired_chain). Scans read DELETED pages
 * (only TOMBSTONED is skipped), so an in-flight scanner still sees the full
 * old list.
 */
static void
pg_retire_chain(void *ctx, MktStorage *posting_storage, BlockNumber head)
{
	(void)ctx;
	/*
	 * Stamp with this transaction's XID — the one that made the chain
	 * unreachable by committing the centroid flip. Any scan that could still
	 * hold a stale pointer took its snapshot no later than this XID, so once
	 * the global horizon passes it no such scan remains and the chain is safe
	 * to reclaim (the invariant btree page deletion uses). The split has
	 * already written WAL, so an XID is assigned.
	 */
	uint64 dxid = U64FromFullTransactionId(GetTopFullTransactionId());

	BlockNumber blk = head;
	while (blk != InvalidBlockNumber)
	{
		Page page = mkt_storage_write_page(posting_storage, blk);
		MktPostingPageOpaque *op   = mkt_posting_opaque(page);
		BlockNumber			  next = op->next_blkno;
		op->flags |= MKT_POSTING_PAGE_DELETED;
		op->delete_xid =
				dxid; /* overlays live_count/tail_blkno (unused now) */
		mkt_storage_commit_page(posting_storage, blk);
		blk = next;
	}
}

/*
 * Physically retire a DELETED chain once no snapshot can still hold a stale
 * pointer into it: tombstone every page so scans skip it and a later
 * page-recycle pass can reclaim the space. Caller has checked the XID gate.
 */
static void
reclaim_retired_chain(MktStorage *posting_storage, BlockNumber head)
{
	BlockNumber blk = head;
	while (blk != InvalidBlockNumber)
	{
		Page page = mkt_storage_write_page(posting_storage, blk);
		MktPostingPageOpaque *op   = mkt_posting_opaque(page);
		BlockNumber			  next = op->next_blkno;
		op->flags |= MKT_POSTING_PAGE_TOMBSTONED;
		mkt_storage_commit_page(posting_storage, blk);
		blk = next;
	}
}

/* ----------------------------------------------------------------
 * Helpers
 * ---------------------------------------------------------------- */

/* Persist an updated leaf count into the metapage (block 0), in place. */
static void
persist_nlist(MktStorage *storage, uint32_t nlist)
{
	Page			page = mkt_storage_write_page(storage, 0);
	MktannMetaPage *meta = (MktannMetaPage *)PageGetSpecialPointer(page);
	meta->nlist			 = nlist;
	mkt_storage_commit_page(storage, 0);
}

static void
require_supported_shape(Relation index, MktIndexBase *base)
{
	if (base->nlevels != 1 || base->centroid_format != MKT_CENTROID_FMT_RABITQ)
	{
		char *name = pstrdup(RelationGetRelationName(index));
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("incremental split is not supported for index \"%s\"",
						name),
				 errdetail(
						 "Only flat (single-level) indexes with RaBitQ "
						 "centroid pages are supported in this release.")));
	}
}

/*
 * Split the posting head at `head` if it is a live first page (and, when
 * threshold > 0, holds at least that many live entries). Serialized against
 * inserts to the same cluster by the head page lock. Returns true and fills
 * *res if a split happened.
 */
static bool
split_one_head(
		Relation		 index,
		MktIndexBase	*base,
		PgSplitFetchCtx *fc,
		BlockNumber		 head,
		uint32_t		 threshold,
		MktSplitResult	*res)
{
	LockPage(index, head, ExclusiveLock);

	Page p = mkt_storage_read_page(base->posting_storage, head);
	const MktPostingPageOpaque *op = mkt_posting_opaque(p);
	bool live_head				   = op->page_id == MKT_POSTING_PAGE_ID &&
					 (op->flags & MKT_POSTING_PAGE_FIRST) &&
					 !(op->flags & MKT_POSTING_PAGE_TOMBSTONED);
	uint32_t live = op->live_count;
	mkt_storage_release_page(base->posting_storage, head);

	if (!live_head || (threshold > 0 && live < threshold))
	{
		UnlockPage(index, head, ExclusiveLock);
		return false;
	}

	MktSplitEnv env =
			{.fetch_vector = pg_split_fetch_vector,
			 .retire_chain = pg_retire_chain,
			 .ctx		   = fc};
	MktSplitConfig cfg = {0};
	int			   rc  = mkt_posting_split(base, head, &cfg, &env, res);
	UnlockPage(index, head, ExclusiveLock);

	return rc == 0 && res->did_split;
}

/* Is `blk` a live posting-list head (first page, not tombstoned/retired)? */
static bool
posting_head_is_live(MktStorage *st, BlockNumber blk)
{
	Page						p	 = mkt_storage_read_page(st, blk);
	const MktPostingPageOpaque *op	 = mkt_posting_opaque(p);
	bool						live = op->page_id == MKT_POSTING_PAGE_ID &&
				(op->flags & MKT_POSTING_PAGE_FIRST) &&
				!(op->flags &
				  (MKT_POSTING_PAGE_TOMBSTONED | MKT_POSTING_PAGE_DELETED));
	mkt_storage_release_page(st, blk);
	return live;
}

/*
 * Merge the undersized list at `head` into its nearest neighbor. Finds the
 * target read-only, then locks both heads in block-number order (so a
 * concurrent op locking the same pair can't deadlock), re-verifies both are
 * still live, and dissolves the source. Returns true and fills *res on merge.
 */
static bool
merge_one_head(
		Relation		 index,
		MktIndexBase	*base,
		PgSplitFetchCtx *fc,
		BlockNumber		 head,
		MktMergeResult	*res)
{
	BlockNumber target = mkt_posting_merge_find_target(base, head);
	if (target == InvalidBlockNumber)
		return false;

	BlockNumber lo = Min(head, target);
	BlockNumber hi = Max(head, target);
	LockPage(index, lo, ExclusiveLock);
	LockPage(index, hi, ExclusiveLock);

	/* State may have changed between find and lock. */
	if (!posting_head_is_live(base->posting_storage, head) ||
		!posting_head_is_live(base->posting_storage, target))
	{
		UnlockPage(index, hi, ExclusiveLock);
		UnlockPage(index, lo, ExclusiveLock);
		return false;
	}

	MktSplitEnv env =
			{.fetch_vector = pg_split_fetch_vector,
			 .retire_chain = pg_retire_chain,
			 .ctx		   = fc};
	int rc = mkt_posting_merge_into(base, head, target, &env, res);

	UnlockPage(index, hi, ExclusiveLock);
	UnlockPage(index, lo, ExclusiveLock);
	return rc == 0 && res->did_merge;
}

/* Common setup: base + storage + heap fetch context. */
typedef struct MaintCtx
{
	MktIndexBase	base;
	MktannStorage	storage;
	Relation		heap;
	PgSplitFetchCtx fetch;
	ResourceOwner	params_owner;
} MaintCtx;

static void
maint_begin(Relation index, MaintCtx *m)
{
	m->params_owner = CurrentResourceOwner;
	mktann_index_base_init(index, &m->base);
	require_supported_shape(index, &m->base);

	mktann_storage_init(&m->storage, index, NULL, m->base.metric);
	m->base.centroid_storage = &m->storage.base;
	m->base.posting_storage	 = &m->storage.base;
	m->base.page_base		 = NULL;

	m->heap			= table_open(index->rd_index->indrelid, AccessShareLock);
	m->fetch.heap	= m->heap;
	m->fetch.attnum = index->rd_index->indkey.values[0];
	m->fetch.slot	= table_slot_create(m->heap, NULL);
}

static void
maint_end(Relation index, MaintCtx *m, bool changed)
{
	if (changed)
	{
		persist_nlist(&m->storage.base, m->base.nlist);
		/* Drop the cached metapage so later queries see the new leaf count. */
		CacheInvalidateRelcache(index);
	}
	ExecDropSingleTupleTableSlot(m->fetch.slot);
	table_close(m->heap, AccessShareLock);
	mktann_release_params(m->base.dim, m->base.rabitq_seed, m->params_owner);
}

/* ----------------------------------------------------------------
 * SQL entry points
 * ---------------------------------------------------------------- */

/*
 * mkt.split_postinglist(index regclass, head_blkno bigint) -> boolean
 *
 * Split the single posting list whose head page is head_blkno. Returns true if
 * a split happened, false if it was declined (not a live head, too few
 * entries, or degenerate data).
 */
Datum
mkt_split_postinglist(PG_FUNCTION_ARGS)
{
	Oid		 indexoid = PG_GETARG_OID(0);
	int64	 blk64	  = PG_GETARG_INT64(1);
	Relation index	  = relation_open(indexoid, RowExclusiveLock);

	if (index->rd_rel->relkind != RELKIND_INDEX)
	{
		relation_close(index, RowExclusiveLock);
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("\"%s\" is not an index",
						RelationGetRelationName(index))));
	}
	require_index_owner(index, RowExclusiveLock);

	BlockNumber nblocks = RelationGetNumberOfBlocks(index);
	if (blk64 < 1 || blk64 >= (int64)nblocks)
	{
		relation_close(index, RowExclusiveLock);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("block number " INT64_FORMAT " out of range", blk64)));
	}

	MaintCtx m;
	maint_begin(index, &m);

	MktSplitResult res;
	bool		   did = split_one_head(
			  index, &m.base, &m.fetch, (BlockNumber)blk64, 0, &res);

	maint_end(index, &m, did);
	relation_close(index, RowExclusiveLock);

	PG_RETURN_BOOL(did);
}

/*
 * mkt.merge_postinglist(index regclass, head_blkno bigint) -> boolean
 *
 * Dissolve the posting list whose head page is head_blkno into its nearest
 * neighbor. Returns true if a merge happened, false if declined (not a live
 * head, or the only leaf).
 */
Datum
mkt_merge_postinglist(PG_FUNCTION_ARGS)
{
	Oid		 indexoid = PG_GETARG_OID(0);
	int64	 blk64	  = PG_GETARG_INT64(1);
	Relation index	  = relation_open(indexoid, RowExclusiveLock);

	if (index->rd_rel->relkind != RELKIND_INDEX)
	{
		relation_close(index, RowExclusiveLock);
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("\"%s\" is not an index",
						RelationGetRelationName(index))));
	}
	require_index_owner(index, RowExclusiveLock);

	BlockNumber nblocks = RelationGetNumberOfBlocks(index);
	if (blk64 < 1 || blk64 >= (int64)nblocks)
	{
		relation_close(index, RowExclusiveLock);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("block number " INT64_FORMAT " out of range", blk64)));
	}

	MaintCtx m;
	maint_begin(index, &m);

	MktMergeResult res;
	bool		   did =
			merge_one_head(index, &m.base, &m.fetch, (BlockNumber)blk64, &res);

	maint_end(index, &m, did);
	relation_close(index, RowExclusiveLock);

	PG_RETURN_BOOL(did);
}

/*
 * Scan an already-open index: split every live head that is flagged
 * MKT_POSTING_PAGE_NEEDS_SPLIT or (when mkt.max_postinglist_size > 0) exceeds
 * that size, and physically retire any DELETED chain whose delete_xid has
 * cleared the global visibility horizon. Returns the number of lists split.
 * Callable from the SQL entry point and from VACUUM cleanup; the caller owns
 * the index lock. New heads created during the pass land past the snapshotted
 * block count and are left for a later call.
 */
int32
mktann_compact_index(Relation index)
{
	MaintCtx m;
	maint_begin(index, &m);

	uint32_t	threshold	= (uint32_t)mkt_max_postinglist_size;
	BlockNumber nblocks		= RelationGetNumberOfBlocks(index);
	BlockNumber start		= Max(m.base.first_posting, 1);
	uint32_t	merge_floor = (uint32_t)mkt_min_postinglist_size;
	int32		nsplits		= 0;
	int32		nmerges		= 0;

	for (BlockNumber blk = start; blk < nblocks; blk++)
	{
		CHECK_FOR_INTERRUPTS();

		Page p = mkt_storage_read_page(&m.storage.base, blk);
		const MktPostingPageOpaque *op = mkt_posting_opaque(p);
		bool is_head				   = op->page_id == MKT_POSTING_PAGE_ID &&
					   (op->flags & MKT_POSTING_PAGE_FIRST) &&
					   !(op->flags & MKT_POSTING_PAGE_TOMBSTONED);
		bool retired	= is_head && (op->flags & MKT_POSTING_PAGE_DELETED);
		bool split_cand = is_head && !retired &&
						  ((op->flags & MKT_POSTING_PAGE_NEEDS_SPLIT) ||
						   (threshold > 0 && op->live_count >= threshold));
		/* live_count is meaningful only on a live (non-retired) head; on a
		 * retired one that slot holds delete_xid. */
		uint32_t live		= (is_head && !retired) ? op->live_count : 0;
		bool	 merge_cand = is_head && !retired && !split_cand &&
						  merge_floor > 0 && live > 0 && live < merge_floor;
		uint64 dxid = retired ? op->delete_xid : 0;
		mkt_storage_release_page(&m.storage.base, blk);

		if (split_cand)
		{
			MktSplitResult res;
			if (split_one_head(index, &m.base, &m.fetch, blk, 0, &res))
				nsplits++;
		}
		else if (merge_cand)
		{
			MktMergeResult res;
			if (merge_one_head(index, &m.base, &m.fetch, blk, &res))
				nmerges++;
		}
		else if (
				retired && GlobalVisCheckRemovableFullXid(
								   m.heap, FullTransactionIdFromU64(dxid)))
		{
			/* No snapshot can still hold a stale pointer into this chain, so
			 * it is safe to physically retire it (its own commits are durable
			 * independent of maint_end, which only persists nlist). */
			reclaim_retired_chain(&m.storage.base, blk);
		}
	}

	maint_end(index, &m, nsplits > 0 || nmerges > 0);
	return nsplits;
}

Datum
mkt_compact(PG_FUNCTION_ARGS)
{
	Oid		 indexoid = PG_GETARG_OID(0);
	Relation index	  = relation_open(indexoid, RowExclusiveLock);

	if (index->rd_rel->relkind != RELKIND_INDEX)
	{
		relation_close(index, RowExclusiveLock);
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("\"%s\" is not an index",
						RelationGetRelationName(index))));
	}
	require_index_owner(index, RowExclusiveLock);

	int32 nsplits = mktann_compact_index(index);

	relation_close(index, RowExclusiveLock);

	PG_RETURN_INT32(nsplits);
}
