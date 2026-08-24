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
#include "mktann_meta.h"
#include "mktann_storage.h"
#include "support_pg.h"
#include "types/vector.h"

PG_FUNCTION_INFO_V1(mkt_split_postinglist);
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

	MktSplitEnv	   env = {pg_split_fetch_vector, fc};
	MktSplitConfig cfg = {0};
	int			   rc  = mkt_posting_split(base, head, &cfg, &env, res);
	UnlockPage(index, head, ExclusiveLock);

	return rc == 0 && res->did_split;
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
 * mkt.compact(index regclass) -> integer
 *
 * Scan the index for posting-list heads flagged for split (or, when
 * mkt.max_postinglist_size > 0, over that size) and split each. Returns the
 * number of lists split. New heads created during the pass land past the
 * snapshotted block count and are left for a later call.
 */
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

	MaintCtx m;
	maint_begin(index, &m);

	uint32_t	threshold = (uint32_t)mkt_max_postinglist_size;
	BlockNumber nblocks	  = RelationGetNumberOfBlocks(index);
	BlockNumber start	  = Max(m.base.first_posting, 1);
	int32		nsplits	  = 0;

	for (BlockNumber blk = start; blk < nblocks; blk++)
	{
		CHECK_FOR_INTERRUPTS();

		Page p = mkt_storage_read_page(&m.storage.base, blk);
		const MktPostingPageOpaque *op = mkt_posting_opaque(p);
		bool candidate				   = op->page_id == MKT_POSTING_PAGE_ID &&
						 (op->flags & MKT_POSTING_PAGE_FIRST) &&
						 !(op->flags & MKT_POSTING_PAGE_TOMBSTONED) &&
						 ((op->flags & MKT_POSTING_PAGE_NEEDS_SPLIT) ||
						  (threshold > 0 && op->live_count >= threshold));
		mkt_storage_release_page(&m.storage.base, blk);

		if (!candidate)
			continue;

		MktSplitResult res;
		if (split_one_head(index, &m.base, &m.fetch, blk, 0, &res))
			nsplits++;
	}

	maint_end(index, &m, nsplits > 0);
	relation_close(index, RowExclusiveLock);

	PG_RETURN_INT32(nsplits);
}
