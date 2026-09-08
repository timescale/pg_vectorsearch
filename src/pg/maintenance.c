/*
 * maintenance.c - mutating index maintenance functions
 *
 * SQL-callable operations that modify mktann index pages, and therefore
 * WAL-log them. Kept separate from the strictly read-only inspection
 * functions in inspect.c so the write/WAL paths are grouped where they
 * get the scrutiny persistent mutations need.
 *
 * Operations:
 *   mkt.convert_posting_to_fastscan(regclass, int4) -- convert one cluster's
 *       posting chain from AoS to fastscan format (function)
 *   mkt.split_posting_list(regclass, bigint) -- split one list by head block
 *       (procedure)
 *   mkt.rebalance(regclass, int4) -- split every list over a given size
 *       (procedure)
 *
 * split_posting_list and rebalance are procedures, not functions: they are
 * DDL-like mutating maintenance, take no return value (reporting via NOTICE),
 * and a procedure can manage its own transactions, which a function cannot.
 */

#include <postgres.h>

#include <access/generic_xlog.h>
#include <access/htup_details.h>
#include <access/relation.h>
#include <access/reloptions.h>
#include <access/table.h>
#include <access/tableam.h>
#include <access/transam.h>
#include <access/xact.h>
#include <catalog/index.h>
#include <catalog/indexing.h>
#include <catalog/objectaddress.h>
#include <catalog/pg_class.h>
#include <executor/tuptable.h>
#include <fmgr.h>
#include <miscadmin.h>
#include <nodes/makefuncs.h>
#include <nodes/parsenodes.h>
#include <storage/bufmgr.h>
#include <storage/lmgr.h>
#include <utils/acl.h>
#include <utils/builtins.h>
#include <utils/injection_point.h>
#include <utils/inval.h>
#include <utils/lsyscache.h>
#include <utils/rel.h>
#include <utils/snapmgr.h>
#include <utils/syscache.h>

#include "amcache.h"
#include "build.h"
#include "index/centroid_page.h"
#include "index/index_base.h"
#include "index/index_build.h"
#include "index/posting_convert.h"
#include "index/posting_page.h"
#include "index/posting_split.h"
#include "inspect.h"
#include "meta.h"
#include "pg/bufstorage.h"
#include "support_pg.h"
#include "typeinfo.h"
#include "types/vector.h"

/*
 * Lock mode the mutating maintenance entry points take on the index.
 *
 * ShareUpdateExclusiveLock rather than RowExclusiveLock, because it conflicts
 * with itself: two maintenance calls on one index have to serialize. A split
 * mints ids for its new clusters from the leaf count it read (nlist + j) and
 * writes the updated count back when it finishes, so two overlapping passes
 * would hand the same ids to different lists and persist a count too low by
 * one pass's worth -- which then undercounts leaves for the automatic probe
 * count and the cost model, and leaves cluster ids ambiguous for
 * convert_posting_to_fastscan and tids_clusters.
 *
 * Like VACUUM's lock (the same mode, for the same "one housekeeper at a time"
 * reason) it still admits reads and inserts, so maintenance does not block
 * queries.
 *
 * Held until end of transaction, not until the relation is closed: the new
 * leaf count reaches other backends as a relcache invalidation, which is only
 * delivered at commit.
 */
#define MKT_MAINT_LOCK ShareUpdateExclusiveLock

/*
 * Refuse to run inside a caller's transaction.
 *
 * These procedures reorganize the index; they do not change the data it points
 * at. Their page writes are not transactional either -- a ROLLBACK leaves the
 * lists split, the old chains retired and the leaf count raised, while
 * discarding the relcache invalidation that tells other backends the leaf
 * count moved. Letting them run inside a transaction block therefore offers a
 * rollback that does not roll anything back.
 *
 * It is also what lets a pass commit as it goes. Splitting each list in its
 * own transaction -- so a long pass bounds its transaction and can reclaim
 * what its earlier splits retired -- is impossible from inside a caller's
 * transaction, which is the reason a procedure was the right shape for this to
 * begin with.
 *
 * The same test VACUUM makes, in the form a procedure has available: a
 * top-level CALL gets a non-atomic call context, anything else (a transaction
 * block, a function, a DO block) gets an atomic one.
 */
static void
require_own_transaction(FunctionCallInfo fcinfo, const char *procname)
{
	CallContext *cc = (fcinfo->context != NULL &&
					   IsA(fcinfo->context, CallContext))
							? (CallContext *)fcinfo->context
							: NULL;

	if (cc != NULL && !cc->atomic)
		return;

	ereport(ERROR,
			(errcode(ERRCODE_ACTIVE_SQL_TRANSACTION),
			 errmsg("%s cannot run inside a transaction block", procname),
			 errdetail(
					 "It reorganizes the index outside transaction control, "
					 "so a rollback would not undo it."),
			 errhint("Call it on its own, outside BEGIN/COMMIT.")));
}

/*
 * A PROCEDURE cannot be declared STRICT, so a NULL argument arrives here as a
 * zero rather than short-circuiting to a NULL result. Left unchecked, a NULL
 * index reads as OID 0 and fails with "could not open relation with OID 0",
 * and a NULL block number as block 0 -- errors that describe the internal
 * consequence instead of the caller's mistake.
 */
static void
reject_null_arg(FunctionCallInfo fcinfo, int argno, const char *argname)
{
	if (PG_ARGISNULL(argno))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("%s must not be null", argname)));
}

PG_FUNCTION_INFO_V1(mkt_convert_posting_to_fastscan);
PG_FUNCTION_INFO_V1(mkt_split_posting_list);
PG_FUNCTION_INFO_V1(mkt_rebalance);

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
 * exclusive lock as the write, with no gap. The maintenance entry points hold
 * a self-conflicting lock on the index (MKT_MAINT_LOCK), so two
 * of them cannot race on one cluster in the first place -- but the
 * compare-and-set stays as a gate that does not depend on callers agreeing
 * about lock modes. The head pointer -- not the posting page's fastscan flag
 * -- is what conversion actually updates, so it is the correct thing to test.
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
	reject_null_arg(fcinfo, 0, "index_oid");
	reject_null_arg(fcinfo, 1, "cluster_id");

	Oid		 indexoid	= PG_GETARG_OID(0);
	int32	 cluster_id = PG_GETARG_INT32(1);
	Relation index		= relation_open(indexoid, MKT_MAINT_LOCK);

	if (index->rd_rel->relkind != RELKIND_INDEX)
	{
		relation_close(index, MKT_MAINT_LOCK);
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("\"%s\" is not an index",
						RelationGetRelationName(index))));
	}

	require_index_owner(index, MKT_MAINT_LOCK);

	/* Read metadata */
	Buffer meta_buf = ReadBuffer(index, 0);
	LockBuffer(meta_buf, BUFFER_LOCK_SHARE);
	Page meta_page = BufferGetPage(meta_buf);

	const MktannMetaPage *meta = (const MktannMetaPage *)PageGetSpecialPointer(
			meta_page);

	if (meta->magic != MKT_META_MAGIC)
	{
		UnlockReleaseBuffer(meta_buf);
		relation_close(index, MKT_MAINT_LOCK);
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
		relation_close(index, MKT_MAINT_LOCK);
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
			relation_close(index, MKT_MAINT_LOCK);
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
		relation_close(index, MKT_MAINT_LOCK);
		PG_RETURN_INT32((int32)current_head);
	}

	ensure_meta_fastscan_flag(index);

	/*
	 * Keep MKT_MAINT_LOCK until end of transaction (NoLock here releases the
	 * reference, not the lock). maint_end queues a relcache invalidation for
	 * the new leaf count, and that is only delivered at commit -- release the
	 * lock now and the next maintenance call could take it, still holding a
	 * cached index base with the stale count, and mint cluster ids that
	 * collide with the ones just written.
	 */
	relation_close(index, NoLock);

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
	/* How to read the indexed column as float32 -- the index's own type, not
	 * an assumed one. Carries the conversion buffer, so it outlives a fetch.
	 */
	MktVectorAccess access;
	/* For the reserve-nlist seam, which has only this context to work from. */
	MktStorage *storage;
} PgSplitFetchCtx;

static bool
pg_split_fetch_vector(
		void *ctx, ItemPointerData tid, float *out, Dimension dim)
{
	PgSplitFetchCtx *c = (PgSplitFetchCtx *)ctx;

	/* Runs in the walk's per-entry context (see ChainTidCb), which is reset
	 * after every entry -- so the detoast below, and a TOAST reassembly if
	 * the vector is stored out of line, need no cleanup of their own.
	 *
	 * SnapshotAny: the posting entry exists until VACUUM tombstones it, so we
	 * re-cluster against whatever it still points at; a fully-pruned tuple
	 * just drops out of the split (returns false). Same contract as rerank. */
	if (!table_tuple_fetch_row_version(c->heap, &tid, SnapshotAny, c->slot))
		return false;

	bool  isnull;
	Datum val = slot_getattr(c->slot, c->attnum, &isnull);
	bool  ok  = false;
	if (!isnull)
	{
		/*
		 * Read through the index's own type descriptor, as the insert and
		 * rerank paths do. Reading the datum as a float32 vector directly
		 * would be wrong for any other indexed type: a halfvec column passes
		 * the shape check this maintenance requires, and its 16-bit payload
		 * read as `dim` floats runs off the end of the value and feeds the
		 * split whatever follows it.
		 *
		 * Detoast explicitly rather than leaving it to the descriptor, so the
		 * copy can be released here. A vector wide enough to be stored out of
		 * line is copied on every fetch, and this runs once per entry of every
		 * list a pass touches -- keeping them would cost the whole pass's
		 * worth of vector data on top of the list being clustered.
		 */
		struct varlena *raw	 = (struct varlena *)DatumGetPointer(val);
		struct varlena *flat = pg_detoast_datum(raw);
		VectorRef vref = mkt_vector_read(&c->access, PointerGetDatum(flat));

		memcpy(out, vref.data, (size_t)dim * sizeof(float));
		ok = true;

		if (flat != raw)
			pfree(flat);
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
 * visibility horizon (see mkt_rebalance). Scans read DELETED pages
 * (only TOMBSTONED is skipped), so an in-flight scanner still sees the full
 * old list.
 */
/*
 * Start the read for the heap block holding `tid`, one step ahead of the walk
 * that is about to fetch it. PrefetchBuffer does nothing when the block is
 * already resident, so a cached heap pays a buffer-table probe per distinct
 * block and no I/O; a cold one gets the read started while the current entry
 * is still being decoded.
 */
static void
pg_prefetch_vector(void *ctx, ItemPointerData tid)
{
	PgSplitFetchCtx *c = (PgSplitFetchCtx *)ctx;

	PrefetchBuffer(c->heap, MAIN_FORKNUM, ItemPointerGetBlockNumber(&tid));
}

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

/*
 * Persist the leaf count, so the ids counted from it survive a crash that
 * leaves the new leaves reachable -- see MktSplitEnv.reserve_nlist. The
 * storage handle is reached through the split's fetch context, which is the
 * only context the seam carries.
 */
static void
pg_reserve_nlist(void *ctx, uint32_t nlist)
{
	PgSplitFetchCtx *c = (PgSplitFetchCtx *)ctx;

	persist_nlist(c->storage, nlist);
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
 * The memory a split may use, from maintenance_work_mem --
 * the budget an operator already raises for index work, and the same knob the
 * bulk build sizes its clustering sample from. A split streams the list, so
 * this bounds its memory whatever the list's size -- see
 * mkt_split_sample_cap, which turns the budget into a sample size after
 * reserving what clustering costs alongside it.
 */
static uint64_t
maint_memory_budget(void)
{
	return (uint64_t)maintenance_work_mem * UINT64CONST(1024);
}

/*
 * Refuse the pass if the budget cannot pay for a split at this dimension,
 * rather than quietly exceeding it. Checked once per pass: the answer depends
 * only on the dimension and the setting, so it cannot change list by list.
 */
static void
require_memory_budget(Relation index, Dimension dim)
{
	if (mkt_split_sample_cap(maint_memory_budget(), dim) > 0)
		return;

	uint64 need = mkt_split_min_budget_bytes(dim);
	char  *name = pstrdup(RelationGetRelationName(index));

	ereport(ERROR,
			(errcode(ERRCODE_INSUFFICIENT_RESOURCES),
			 errmsg("maintenance_work_mem is too small to split a posting "
					"list of index \"%s\"",
					name),
			 errdetail(
					 "Splitting %u-dimension vectors needs at "
					 "least " UINT64_FORMAT
					 " kB, but maintenance_work_mem is %d kB.",
					 (uint32)dim,
					 (need + 1023) / 1024,
					 maintenance_work_mem),
			 errhint("Increase maintenance_work_mem and retry.")));
}

/*
 * Is this page a posting page at all? Guard on the special-area size before
 * reading the opaque, so a page of another kind -- or one only extended, whose
 * special offset is zero -- is never misread through the posting layout. The
 * scanned range starts past the centroid region, so today nothing else should
 * be in it; the guard costs nothing and does not rely on that staying true.
 * (mktann_bulkdelete guards the same way over the same range.)
 */
static bool
page_is_posting(Page page)
{
	return !PageIsNew(page) &&
		   PageGetSpecialSize(page) == sizeof(MktPostingPageOpaque) &&
		   mkt_posting_opaque(page)->page_id == MKT_POSTING_PAGE_ID;
}

/*
 * Is this posting page a live chain head -- reachable, not retired, not all
 * dead? Written the same way wherever it is asked (the maintenance scan, the
 * re-read under the head lock, vacuum's head recognition), so the three cannot
 * drift.
 */
static bool
posting_head_is_live(const MktPostingPageOpaque *op)
{
	return (op->flags & MKT_POSTING_PAGE_FIRST) &&
		   !(op->flags & MKT_POSTING_PAGE_TOMBSTONED) &&
		   !(op->flags & MKT_POSTING_PAGE_DELETED);
}

/*
 * Split the posting head at `head` if it is a live first page and, when
 * target > 0, holds more entries than the split trigger
 * (mkt_split_trigger). Serialized against inserts to the same
 * cluster by the head page lock. Returns true and fills *res if a split
 * happened.
 *
 * The live_count test here is only a cheap early-out on a page already read.
 * The authoritative check is inside mkt_posting_split, against the entry count
 * after collection: live_count does not account for entries whose vector can
 * no longer be fetched, so a list can look oversized here and turn out not to
 * be.
 *
 * target also sets the width -- round(count / target) parts, so each new list
 * rests at the target with room to grow back to the trigger. With target == 0
 * the split is unconditional and 2-way: the manual escape hatch.
 */
static bool
split_one_head(
		Relation		 index,
		MktIndexBase	*base,
		PgSplitFetchCtx *fc,
		BlockNumber		 head,
		uint32_t		 target,
		MktSplitResult	*res)
{
	LockPage(index, head, ExclusiveLock);

	Page p		   = mkt_storage_read_page(base->posting_storage, head);
	bool live_head = page_is_posting(p) &&
					 posting_head_is_live(mkt_posting_opaque(p));
	/* live_count is meaningful only when the head is live: delete_xid overlays
	 * it once DELETED, which posting_head_is_live excludes. */
	uint32_t live = live_head ? mkt_posting_opaque(p)->live_count : 0;
	mkt_storage_release_page(base->posting_storage, head);

	if (!live_head ||
		(target > 0 && (uint64_t)live <= mkt_split_trigger(target)))
	{
		UnlockPage(index, head, ExclusiveLock);
		return false;
	}

	MktSplitEnv env = {
			.fetch_vector	 = pg_split_fetch_vector,
			.prefetch_vector = pg_prefetch_vector,
			.retire_chain	 = pg_retire_chain,
			.reserve_nlist	 = pg_reserve_nlist,
			.ctx			 = fc,
	};
	/*
	 * Test hook: fires holding the old head's page lock, with the centroid
	 * leaf still pointing at it. That is the window an isolation test needs --
	 * an insert routed now reaches this head, blocks on the page lock, and
	 * once the split below has flipped the leaf and retired the chain it wakes
	 * to find the head retired and has to route again. Firing after the split
	 * instead would prove nothing: the leaf would already point elsewhere, so
	 * the insert would route straight to a new head and never contend.
	 *
	 * No-op unless PG was built with injection points and a test attached an
	 * action.
	 */
	INJECTION_POINT("mktann-split-locked", NULL);

	MktSplitConfig cfg = {
			.target_entries		 = target,
			.sample_budget_bytes = maint_memory_budget(),
	};
	int rc = mkt_posting_split(base, head, &cfg, &env, res);

	UnlockPage(index, head, ExclusiveLock);

	return rc == 0 && res->did_split;
}

/*
 * Resolve the resting list size maintenance should aim at.
 *
 * Derived from the row count alone, deliberately ignoring the nlist
 * reloption. Above MKT_TARGET_ENTRIES_PER_LIST^2 rows that is the flat
 * target; below it the sqrt floor in mkt_auto_nlist makes it smaller, which
 * is the regime where a hardcoded target would fight the build and merge a
 * small index down to too few lists.
 *
 * It used to honour an explicit nlist instead, on the grounds that a
 * rebalanced index should keep the shape the build chose -- the cost model
 * and the automatic nprobe both key off nlist. That is a real property and
 * this gives it up: re-partitioning changes how many lists a query probes.
 *
 * What it bought was worse. With target = rows / nlist the target scales with
 * the table, so an index built with an explicit nlist never splits on growth
 * at all: lists grow in proportion, the trigger is never reached, and a pass
 * correctly reports splitting nothing while every probe scans more entries
 * than the last time. A number typed once at CREATE INDEX would silently
 * switch off maintenance for the life of the index -- and once maintenance is
 * VACUUM-driven, nothing would ever revisit it.
 *
 * A list should rest at the size that keeps a probe's cost flat, which is
 * what MKT_TARGET_ENTRIES_PER_LIST is for. nlist stays what the user asked
 * the *build* for; it is not a maintenance policy.
 */
static uint32_t
resolve_target_entries(Relation heap)
{
	return mkt_target_entries_per_list(mktann_estimate_heap_tuples(heap), 0);
}

/* Common setup: base + storage + heap fetch context. */
typedef struct MaintCtx
{
	MktIndexBase	base;
	MktannStorage	storage;
	Relation		heap;
	PgSplitFetchCtx fetch;
	ResourceOwner	params_owner;
	/*
	 * Scratch for one list's split. Splitting a list holds every one of its
	 * vectors at full precision, so the peak is inherent -- but a pass over
	 * many lists must not stack those peaks. Reset between lists (see
	 * maint_split_done) so the cost is one list's worth, not the pass's.
	 */
	MemoryContext split_ctx;
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
	m->fetch.access = mkt_vector_access(
			mktann_cache_type_info(index), m->base.dim, CurrentMemoryContext);
	m->fetch.storage = &m->storage.base;

	/* Created here, but not switched into: everything above has to outlive the
	 * per-list resets. */
	m->split_ctx = AllocSetContextCreate(
			CurrentMemoryContext, "mktann split", ALLOCSET_DEFAULT_SIZES);
}

/*
 * Run one list's split with its scratch accounted separately, and release that
 * scratch before returning. Splitting is where a maintenance pass allocates,
 * and one list is the logical point at which none of it is needed any more:
 * the result is returned by value and the index state the caller keeps reading
 * lives in the pass's own context.
 */
static bool
split_one_head_in_scratch(
		Relation		index,
		MaintCtx	   *m,
		BlockNumber		head,
		uint32_t		target,
		MktSplitResult *res)
{
	MemoryContext old = MemoryContextSwitchTo(m->split_ctx);
	bool did = split_one_head(index, &m->base, &m->fetch, head, target, res);
	MemoryContextSwitchTo(old);
	MemoryContextReset(m->split_ctx);
	return did;
}

/*
 * Drop the nlist reloption once maintenance has moved the leaf count away
 * from it.
 *
 * A split restructures the index the way an ALTER plus REINDEX would, only
 * incrementally and without the exclusive lock -- so the declaration has to
 * stop claiming a width the index no longer has. Otherwise a later REINDEX
 * rebuilds at the number typed at CREATE INDEX: an index built with
 * nlist = 100, grown a hundredfold and split to match, comes back with 100
 * lists, and no user could reasonably be expected to ALTER the right value in
 * beforehand.
 *
 * Cleared rather than set to the new count. Writing 10000 in would assert the
 * user asked for exactly 10000, which they did not, and it would be a fresh
 * pin that goes stale on the next growth cycle. Removing it lets every later
 * rebuild derive from the rows as they then are.
 *
 * Only when the reloption is set. Left out, auto-derivation already lands
 * where the splits were heading, and writing anything in would convert "size
 * this for my data" into a permanent pin.
 *
 * The tuple lock is required, not decoration: reloptions live in a pg_class
 * column that is also updated in place (ANALYZE's reltuples, say), so a plain
 * heap_update here can lose a concurrent inplace update. PostgreSQL warns
 * about exactly that -- "missing lock for relation ... @ TID" -- and holding
 * AccessExclusiveLock on the index does not satisfy it, because the lock it
 * wants is on the catalog row. The index-level lock the pass already holds
 * (MKT_MAINT_LOCK, ShareUpdateExclusiveLock) is what ALTER INDEX ... SET
 * takes, so no escalation is needed for the index itself.
 */
static void
clear_nlist_reloption(Relation index)
{
	MktannOptions *opts = (MktannOptions *)index->rd_options;

	if (opts == NULL || opts->nlist <= 0)
		return; /* never declared: nothing to clear */

	Relation  pgclass = table_open(RelationRelationId, RowExclusiveLock);
	HeapTuple tuple	  = SearchSysCacheCopy1(
			  RELOID, ObjectIdGetDatum(RelationGetRelid(index)));

	if (!HeapTupleIsValid(tuple))
		elog(ERROR,
			 "cache lookup failed for index %u",
			 RelationGetRelid(index));

	bool  isnull;
	Datum old =
			SysCacheGetAttr(RELOID, tuple, Anum_pg_class_reloptions, &isnull);
	List *reset	  = list_make1(makeDefElem(pstrdup("nlist"), NULL, -1));
	Datum newopts = transformRelOptions(
			isnull ? (Datum)0 : old, reset, NULL, NULL, false, true);

	Datum repl_val[Natts_pg_class]	= {0};
	bool  repl_null[Natts_pg_class] = {false};
	bool  repl_repl[Natts_pg_class] = {false};

	if (PointerIsValid(DatumGetPointer(newopts)))
		repl_val[Anum_pg_class_reloptions - 1] = newopts;
	else
		repl_null[Anum_pg_class_reloptions - 1] = true;
	repl_repl[Anum_pg_class_reloptions - 1] = true;

	HeapTuple newtuple = heap_modify_tuple(
			tuple, RelationGetDescr(pgclass), repl_val, repl_null, repl_repl);

	/*
	 * Keep the old tid: CatalogTupleUpdate rewrites newtuple->t_self to where
	 * the new version lands, so unlocking through it would release a lock on
	 * the wrong row and leave this one held at commit.
	 */
	ItemPointerData otid = newtuple->t_self;

	LockTuple(pgclass, &otid, InplaceUpdateTupleLock);
	CatalogTupleUpdate(pgclass, &otid, newtuple);
	UnlockTuple(pgclass, &otid, InplaceUpdateTupleLock);

	heap_freetuple(newtuple);
	heap_freetuple(tuple);
	table_close(pgclass, RowExclusiveLock);
}

static void
maint_end(Relation index, MaintCtx *m, bool changed)
{
	if (changed)
	{
		persist_nlist(&m->storage.base, m->base.nlist);
		/* The declaration no longer describes the index -- see
		 * clear_nlist_reloption. */
		clear_nlist_reloption(index);
		/* Drop the cached metapage so later queries see the new leaf count. */
		CacheInvalidateRelcache(index);
	}
	MemoryContextDelete(m->split_ctx);
	ExecDropSingleTupleTableSlot(m->fetch.slot);
	table_close(m->heap, AccessShareLock);
	mktann_release_params(m->base.dim, m->base.rabitq_seed, m->params_owner);
}

/* ----------------------------------------------------------------
 * SQL entry points
 * ---------------------------------------------------------------- */

/*
 * CALL mkt.split_posting_list(index regclass, head_blkno bigint)
 *
 * Split the single posting list whose head page is head_blkno. Reports via
 * NOTICE whether a split happened or was declined (not a live head, too few
 * entries, or degenerate data).
 */
Datum
mkt_split_posting_list(PG_FUNCTION_ARGS)
{
	require_own_transaction(fcinfo, "mkt.split_posting_list()");
	reject_null_arg(fcinfo, 0, "index_oid");
	reject_null_arg(fcinfo, 1, "head_blkno");

	Oid		 indexoid = PG_GETARG_OID(0);
	int64	 blk64	  = PG_GETARG_INT64(1);
	Relation index	  = relation_open(indexoid, MKT_MAINT_LOCK);

	if (index->rd_rel->relkind != RELKIND_INDEX)
	{
		relation_close(index, MKT_MAINT_LOCK);
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("\"%s\" is not an index",
						RelationGetRelationName(index))));
	}
	require_index_owner(index, MKT_MAINT_LOCK);

	BlockNumber nblocks = RelationGetNumberOfBlocks(index);
	if (blk64 < 1 || blk64 >= (int64)nblocks)
	{
		relation_close(index, MKT_MAINT_LOCK);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("block number " INT64_FORMAT " out of range", blk64)));
	}

	MaintCtx m;
	maint_begin(index, &m);
	require_memory_budget(index, m.base.dim);

	MktSplitResult res;
	bool		   did =
			split_one_head_in_scratch(index, &m, (BlockNumber)blk64, 0, &res);

	maint_end(index, &m, did);
	/*
	 * Keep MKT_MAINT_LOCK until end of transaction (NoLock here releases the
	 * reference, not the lock). maint_end queues a relcache invalidation for
	 * the new leaf count, and that is only delivered at commit -- release the
	 * lock now and the next maintenance call could take it, still holding a
	 * cached index base with the stale count, and mint cluster ids that
	 * collide with the ones just written.
	 */
	relation_close(index, NoLock);

	if (did)
		ereport(NOTICE,
				(errmsg("split posting list at block " INT64_FORMAT
						" into %u lists",
						blk64,
						res.nparts)));
	else
		/*
		 * split_one_head declines for several reasons -- not a live head, too
		 * few entries, or degenerate data yielding an empty k-means cluster --
		 * so report the outcome without guessing which.
		 */
		ereport(NOTICE,
				(errmsg("posting list at block " INT64_FORMAT " not split",
						blk64)));

	PG_RETURN_VOID();
}

/* ----------------------------------------------------------------
 * CALL mkt.rebalance(index regclass, target_entries int4 DEFAULT NULL)
 *
 * Manual LIRE rebalancing entry point. Scans the index and splits every live
 * posting-list head that has grown past the split trigger into lists of about
 * target_entries entries each, and physically retires any already-DELETED old
 * split chain whose delete_xid has cleared the global visibility horizon.
 * Reports the number of lists split via NOTICE. New heads created during the
 * pass land past the snapshotted block count and are left for a later call.
 *
 * target_entries is the size a list *rests* at, not a bound it never crosses.
 * A list is left alone until it reaches target_entries *
 * MKT_SPLIT_TRIGGER_FACTOR, so the operating band is
 * [target/factor, target*factor] with the target at its centre: room to absorb
 * inserts and deletes, and room for the unevenness of a k-means split.
 * Pinning the trigger at the target instead
 * would leave every fresh list one insert away from splitting again.
 *
 * NULL derives the target from the index itself (see resolve_target_entries),
 * which is what an operator should almost always want; passing a value is an
 * override.
 *
 * Runs in a single transaction, so a chain this pass retires does not become
 * reclaimable within it: the pass's own XID cannot clear the visibility
 * horizon while it is still running. Reclaiming those chains therefore falls
 * to a subsequent call.
 *
 * Splitting is the only rebalancing performed here, and it is driven by the
 * caller -- nothing schedules it.
 * ---------------------------------------------------------------- */
Datum
mkt_rebalance(PG_FUNCTION_ARGS)
{
	require_own_transaction(fcinfo, "mkt.rebalance()");
	reject_null_arg(fcinfo, 0, "index_oid");

	Oid indexoid = PG_GETARG_OID(0);
	/* SQL arg is integer (int4). Unlike the index, a null target is
	 * meaningful: it asks for the size to be derived from the index. */
	bool	 target_given = !PG_ARGISNULL(1);
	int32	 target_arg	  = target_given ? PG_GETARG_INT32(1) : 0;
	Relation index		  = relation_open(indexoid, MKT_MAINT_LOCK);

	if (index->rd_rel->relkind != RELKIND_INDEX)
	{
		relation_close(index, MKT_MAINT_LOCK);
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("\"%s\" is not an index",
						RelationGetRelationName(index))));
	}
	require_index_owner(index, MKT_MAINT_LOCK);
	if (target_given && target_arg < 1)
	{
		relation_close(index, MKT_MAINT_LOCK);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("target_entries must be positive")));
	}

	MaintCtx m;
	maint_begin(index, &m);
	require_memory_budget(index, m.base.dim);

	uint32_t target	 = target_given ? (uint32_t)target_arg
									: resolve_target_entries(m.heap);
	uint64_t trigger = mkt_split_trigger(target);

	BlockNumber nblocks	   = RelationGetNumberOfBlocks(index);
	BlockNumber start	   = Max(m.base.first_posting, 1);
	int32		nsplits	   = 0;
	int32		nreclaimed = 0;

	for (BlockNumber blk = start; blk < nblocks; blk++)
	{
		CHECK_FOR_INTERRUPTS();

		Page p		 = mkt_storage_read_page(&m.storage.base, blk);
		bool posting = page_is_posting(p);
		const MktPostingPageOpaque *op = posting ? mkt_posting_opaque(p)
												 : NULL;

		bool is_head = posting && (op->flags & MKT_POSTING_PAGE_FIRST) &&
					   !(op->flags & MKT_POSTING_PAGE_TOMBSTONED);
		bool retired   = is_head && (op->flags & MKT_POSTING_PAGE_DELETED);
		bool candidate = is_head && !retired &&
						 (uint64_t)op->live_count > trigger;
		/* delete_xid overlays live_count and is meaningful only when DELETED.
		 */
		uint64 dxid = retired ? op->delete_xid : 0;
		mkt_storage_release_page(&m.storage.base, blk);

		if (candidate)
		{
			/* Pass the target so split_one_head re-checks under the head
			 * lock and mkt_posting_split re-checks again after collection --
			 * the list may have shrunk since the unlocked read above, and
			 * live_count does not see unfetchable entries at all. */
			MktSplitResult res;
			if (split_one_head_in_scratch(index, &m, blk, target, &res))
				nsplits++;
		}
		else if (
				retired && GlobalVisCheckRemovableFullXid(
								   m.heap, FullTransactionIdFromU64(dxid)))
		{
			/* No snapshot can still hold a stale pointer into this chain, so
			 * it is safe to physically retire it (its own commits are durable
			 * independent of maint_end, which only persists nlist). */
			mkt_posting_chain_tombstone(&m.storage.base, blk);
			nreclaimed++;
		}
	}

	maint_end(index, &m, nsplits > 0);
	/*
	 * Keep MKT_MAINT_LOCK until end of transaction (NoLock here releases the
	 * reference, not the lock). maint_end queues a relcache invalidation for
	 * the new leaf count, and that is only delivered at commit -- release the
	 * lock now and the next maintenance call could take it, still holding a
	 * cached index base with the stale count, and mint cluster ids that
	 * collide with the ones just written.
	 */
	relation_close(index, NoLock);

	/*
	 * Report the reclaim count as well as the split count. A retired chain is
	 * unreachable from the centroid tree, so nothing that inspects the index
	 * can see it -- this NOTICE is the only way an operator can tell whether a
	 * pass freed the chains an earlier one left behind, or whether a reader's
	 * snapshot is still holding them.
	 */
	ereport(NOTICE,
			(errmsg("rebalance: split %d posting list(s), reclaimed %d "
					"retired chain(s)",
					nsplits,
					nreclaimed)));

	PG_RETURN_VOID();
}
