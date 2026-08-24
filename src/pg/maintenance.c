/*
 * maintenance.c - mutating index maintenance functions
 *
 * SQL-callable operations that modify mktann index pages, and therefore
 * WAL-log them. Kept separate from the strictly read-only inspection
 * functions in inspect.c so the write/WAL paths are grouped where they
 * get the scrutiny persistent mutations need.
 *
 * Functions:
 *   mkt.convert_posting_to_fastscan(regclass, int4) -- convert one cluster's
 *       posting chain from AoS to fastscan format
 *   mkt.split_postinglist(regclass, bigint) -- split one oversized posting
 * list mkt.compact(regclass) -- split every posting list flagged/over size
 */

#include <postgres.h>

#include <access/generic_xlog.h>
#include <access/relation.h>
#include <access/table.h>
#include <access/tableam.h>
#include <catalog/index.h>
#include <catalog/objectaddress.h>
#include <catalog/pg_class.h>
#include <executor/tuptable.h>
#include <fmgr.h>
#include <miscadmin.h>
#include <storage/bufmgr.h>
#include <storage/lmgr.h>
#include <utils/acl.h>
#include <utils/builtins.h>
#include <utils/inval.h>
#include <utils/lsyscache.h>
#include <utils/rel.h>
#include <utils/snapmgr.h>

#include "index/centroid_page.h"
#include "index/index_base.h"
#include "index/posting_convert.h"
#include "index/posting_page.h"
#include "index/posting_split.h"
#include "inspect.h"
#include "maintenance.h"
#include "mktann_cache.h"
#include "mktann_meta.h"
#include "mktann_storage.h"
#include "support_pg.h"
#include "types/vector.h"

PG_FUNCTION_INFO_V1(mkt_convert_posting_to_fastscan);
PG_FUNCTION_INFO_V1(mkt_split_postinglist);
PG_FUNCTION_INFO_V1(mkt_compact);

/*
 * Authorization. convert_posting_to_fastscan mutates the index, so it requires
 * ownership of the *table* the index belongs to -- the same relation model
 * PostgreSQL's pgrowlocks/pgstattuple use for relation-level operations.
 * EXECUTE stays granted to PUBLIC; this runtime check does the per-object
 * authorization a static GRANT cannot express for a regclass argument.
 * Superusers pass automatically.
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

/*
 * Writable pointer to a leaf entry's child block number on a centroid page.
 * The location depends on the page format: an AoS meta array for the
 * meta-based formats, or the packed per-group child array for a fastscan
 * centroid page. This must match how collect_leaf_entries reads the child, or
 * the compare-and-set in update_centroid_posting_head acts on the wrong slot.
 */
static BlockNumber *
centroid_child_ptr(Page page, uint16_t entry_idx, Dimension dim)
{
	if (mkt_centroid_page_format(page) == MKT_CENTROID_FMT_FASTSCAN)
	{
		char	*content = (char *)PageGetContents(page);
		uint32_t g		 = entry_idx / MKT_FASTSCAN_GROUP;
		uint32_t slot	 = entry_idx % MKT_FASTSCAN_GROUP;
		return &mkt_centroid_fastscan_group_child(content, g, dim)[slot];
	}
	return &mkt_centroid_meta_mut(page, entry_idx)->child_blkno;
}

/*
 * Point a centroid leaf entry at new_head via WAL, but only if it still
 * points at expected_old_head. The compare-and-set runs under the same
 * exclusive lock as the write, with no gap, so it is the concurrency gate
 * for conversion: convert_posting_to_fastscan holds only RowExclusiveLock
 * (which does not conflict with itself), so two calls can race on one
 * cluster. The head pointer -- not the posting page's fastscan flag -- is
 * what conversion actually updates, so it is the correct thing to test.
 *
 * Returns true if it performed the update. Returns false if another
 * converter already moved the head, writing the current head to
 * *current_head_out (the caller's freshly built chain is then orphaned,
 * to be reclaimed by a later rebuild/VACUUM).
 */
static bool
update_centroid_posting_head(
		Relation	 index,
		BlockNumber	 centroid_page,
		uint16_t	 entry_idx,
		BlockNumber	 expected_old_head,
		BlockNumber	 new_head,
		Dimension	 dim,
		BlockNumber *current_head_out)
{
	Buffer buf = ReadBuffer(index, centroid_page);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);

	if (*centroid_child_ptr(BufferGetPage(buf), entry_idx, dim) !=
		expected_old_head)
	{
		*current_head_out =
				*centroid_child_ptr(BufferGetPage(buf), entry_idx, dim);
		UnlockReleaseBuffer(buf);
		return false;
	}

	GenericXLogState *state = GenericXLogStart(index);
	Page page = GenericXLogRegisterBuffer(state, buf, GENERIC_XLOG_FULL_IMAGE);
	*centroid_child_ptr(page, entry_idx, dim) = new_head;

	GenericXLogFinish(state);
	UnlockReleaseBuffer(buf);
	return true;
}

/*
 * Set MKT_META_FLAG_FASTSCAN on the metadata page if not already set.
 */
static void
ensure_meta_fastscan_flag(Relation index)
{
	Buffer buf = ReadBuffer(index, 0);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	Page			page = BufferGetPage(buf);
	MktannMetaPage *mp	 = (MktannMetaPage *)PageGetSpecialPointer(page);
	bool			needs_update = !(mp->flags & MKT_META_FLAG_FASTSCAN);
	UnlockReleaseBuffer(buf);

	if (needs_update)
	{
		GenericXLogState *state = GenericXLogStart(index);
		buf						= ReadBuffer(index, 0);
		LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
		page = GenericXLogRegisterBuffer(state, buf, GENERIC_XLOG_FULL_IMAGE);
		mp	 = (MktannMetaPage *)PageGetSpecialPointer(page);
		mp->flags |= MKT_META_FLAG_FASTSCAN;
		GenericXLogFinish(state);
		UnlockReleaseBuffer(buf);
	}
}

/* ----------------------------------------------------------------
 * mkt.convert_posting_to_fastscan(regclass, int4)
 *
 * Converts one cluster's posting chain from AoS to fastscan.
 * Updates the centroid leaf entry and sets the metadata flag.
 * Returns the new posting head block number.
 * ---------------------------------------------------------------- */
Datum
mkt_convert_posting_to_fastscan(PG_FUNCTION_ARGS)
{
	Oid		 indexoid	= PG_GETARG_OID(0);
	int32	 cluster_id = PG_GETARG_INT32(1);
	Relation index		= relation_open(indexoid, RowExclusiveLock);

	if (index->rd_rel->relkind != RELKIND_INDEX)
	{
		relation_close(index, RowExclusiveLock);
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("\"%s\" is not an index",
						RelationGetRelationName(index))));
	}

	require_index_owner(index, RowExclusiveLock);

	/* Read metadata */
	Buffer meta_buf = ReadBuffer(index, 0);
	LockBuffer(meta_buf, BUFFER_LOCK_SHARE);
	Page meta_page = BufferGetPage(meta_buf);

	const MktannMetaPage *meta = (const MktannMetaPage *)PageGetSpecialPointer(
			meta_page);

	if (meta->magic != MKT_META_MAGIC)
	{
		UnlockReleaseBuffer(meta_buf);
		relation_close(index, RowExclusiveLock);
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("\"%s\" is not an mktann index",
						RelationGetRelationName(index))));
	}

	BlockNumber first_centroid = meta->first_centroid;
	Dimension	dim			   = meta->dim;
	uint8_t		nlevels		   = meta->nlevels;
	UnlockReleaseBuffer(meta_buf);

	/*
	 * Find the leaf for this cluster. The argument is the stored cluster_id --
	 * what mkt.posting_pages reports and what callers pass -- not a positional
	 * index. collect_leaf_entries returns leaves in tree-traversal order,
	 * which coincides with cluster_id only for a single-page flat tree; on any
	 * deeper or multi-page tree the two diverge. Match on each leaf's own head
	 * cluster_id so the argument means the same thing everywhere.
	 */
	LeafEntry *leaves;
	int		   nleaves =
			collect_leaf_entries(index, first_centroid, nlevels, dim, &leaves);

	LeafEntry *leaf = NULL;
	for (int i = 0; i < nleaves; i++)
	{
		Buffer hbuf = ReadBuffer(index, leaves[i].posting_head);
		LockBuffer(hbuf, BUFFER_LOCK_SHARE);
		uint32_t cid = mkt_posting_opaque(BufferGetPage(hbuf))->cluster_id;
		UnlockReleaseBuffer(hbuf);
		if (cid == (uint32_t)cluster_id)
		{
			leaf = &leaves[i];
			break;
		}
	}

	if (leaf == NULL)
	{
		pfree(leaves);
		relation_close(index, RowExclusiveLock);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("cluster %d not found or has no posting list",
						cluster_id)));
	}

	BlockNumber old_head	  = leaf->posting_head;
	BlockNumber centroid_page = leaf->centroid_page;
	uint16_t	entry_idx	  = leaf->entry_idx;
	pfree(leaves);

	/* Skip if already fastscan */
	{
		Buffer buf = ReadBuffer(index, old_head);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		Page				  page = BufferGetPage(buf);
		MktPostingPageOpaque *op   = mkt_posting_opaque(page);
		bool already_fastscan = (op->flags & MKT_POSTING_PAGE_FASTSCAN) != 0;
		UnlockReleaseBuffer(buf);

		if (already_fastscan)
		{
			relation_close(index, RowExclusiveLock);
			PG_RETURN_INT32((int32)old_head);
		}
	}

	/* Convert the posting chain */
	MktannStorage storage;
	mktann_storage_init(&storage, index, NULL, DISTANCE_L2);

	/*
	 * Online conversion, unlike a full index build, has no closing
	 * log_newpage_range() to blanket-WAL the new pages. Keep build_mode off
	 * so each fastscan page is WAL-logged as it is committed (per-page
	 * GenericXLog full image). Otherwise the pages would be dirtied but never
	 * shipped, while the centroid repoint below *is* WAL-logged — leaving a
	 * standby whose centroid points at posting heads it never received.
	 */
	storage.build_mode = false;

	BlockNumber new_head =
			mkt_posting_convert_to_fastscan(&storage.base, old_head, dim);

	/*
	 * Publish the new head, but only if a concurrent converter has not
	 * already moved it. If it has, our new_head chain is orphaned and the
	 * winner's head is returned instead.
	 */
	BlockNumber current_head;
	if (!update_centroid_posting_head(
				index,
				centroid_page,
				entry_idx,
				old_head,
				new_head,
				dim,
				&current_head))
	{
		relation_close(index, RowExclusiveLock);
		PG_RETURN_INT32((int32)current_head);
	}

	ensure_meta_fastscan_flag(index);

	relation_close(index, RowExclusiveLock);

	PG_RETURN_INT32((int32)new_head);
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
 * Scan an already-open index and split every posting-list head that is flagged
 * MKT_POSTING_PAGE_NEEDS_SPLIT or (when mkt.max_postinglist_size > 0) exceeds
 * that size. Returns the number of lists split. Callable from the SQL entry
 * point and from VACUUM cleanup; the caller owns the index lock.
 *
 * New heads created during the pass land past the snapshotted block count and
 * are left for a later call.
 */
int32
mktann_compact_index(Relation index)
{
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
	return nsplits;
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

	int32 nsplits = mktann_compact_index(index);

	relation_close(index, RowExclusiveLock);

	PG_RETURN_INT32(nsplits);
}
