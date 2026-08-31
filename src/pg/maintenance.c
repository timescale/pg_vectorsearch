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
 */

#include <postgres.h>

#include <access/generic_xlog.h>
#include <access/relation.h>
#include <catalog/index.h>
#include <catalog/objectaddress.h>
#include <catalog/pg_class.h>
#include <miscadmin.h>
#include <storage/bufmgr.h>
#include <utils/acl.h>
#include <utils/builtins.h>
#include <utils/lsyscache.h>
#include <utils/rel.h>

#include "index/centroid_page.h"
#include "index/posting_convert.h"
#include "index/posting_page.h"
#include "inspect.h"
#include "mktann_meta.h"
#include "mktann_storage.h"

PG_FUNCTION_INFO_V1(mkt_convert_posting_to_fastscan);

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
