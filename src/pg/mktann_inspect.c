/*
 * mkt_pg_inspect.c - Index inspection and maintenance functions
 *
 * Provides set-returning functions to inspect the internal structure
 * of mktann indexes via SQL, and utility functions for index
 * maintenance.
 *
 * Functions:
 *   mkt.centroid_pages(regclass) -- centroid tree structure
 *   mkt.posting_pages(regclass)  -- posting list page chains
 *   mkt.convert_posting_to_fastscan(regclass, int4) -- convert one
 *       cluster's posting chain from AoS to fastscan format
 */

#include <postgres.h>

#include <access/generic_xlog.h>
#include <access/relation.h>
#include <catalog/index.h>
#include <catalog/objectaddress.h>
#include <catalog/pg_class.h>
#include <funcapi.h>
#include <miscadmin.h>
#include <storage/bufmgr.h>
#include <utils/acl.h>
#include <utils/array.h>
#include <utils/builtins.h>
#include <utils/lsyscache.h>
#include <utils/rel.h>

#include "index/centroid_page.h"
#include "index/posting_convert.h"
#include "index/posting_page.h"
#include "mktann_meta.h"
#include "mktann_storage.h"

PG_FUNCTION_INFO_V1(mkt_centroid_pages);
PG_FUNCTION_INFO_V1(mkt_posting_pages);
PG_FUNCTION_INFO_V1(mkt_tids_clusters);
PG_FUNCTION_INFO_V1(mkt_convert_posting_to_fastscan);

/*
 * The functions below iterate an on-disk entry_count read straight from a
 * page. Reject a count past the format's real per-page capacity before
 * looping so a corrupt or truncated page can't drive a read past the page.
 */
static void
check_centroid_count(
		BlockNumber		  blkno,
		uint32_t		  count,
		Dimension		  dim,
		MktCentroidFormat fmt)
{
	uint32_t max_entries = mkt_centroid_max_entries_fmt(dim, fmt);
	if (count > max_entries)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("centroid page %u has an invalid entry count "
						"(%u > %u)",
						blkno,
						count,
						max_entries),
				 errhint("The index may be corrupted; REINDEX it.")));
}

static void
check_posting_count(
		BlockNumber blkno, uint32_t count, Dimension dim, bool is_first)
{
	uint32_t max_entries = is_first ? mkt_posting_max_entries_first(dim)
									: mkt_posting_max_entries(dim);
	if (count > max_entries)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("posting page %u has an invalid entry count "
						"(%u > %u)",
						blkno,
						count,
						max_entries),
				 errhint("The index may be corrupted; REINDEX it.")));
}

/*
 * Authorization for the inspection functions. They expose an index's
 * internal structure (block numbers, cluster ids, tombstone state), so
 * gate them on the caller's privileges on the *table* the index belongs
 * to -- the same model PostgreSQL's pgrowlocks/pgstattuple use for
 * relation inspection. EXECUTE stays granted to PUBLIC; these runtime
 * checks do the per-object authorization that a static GRANT cannot
 * express for a regclass argument. Superusers pass automatically.
 *
 * Read-only inspectors require SELECT on the table (its owner has that,
 * so an owner can always inspect their own index). The conversion
 * function mutates the index, so it requires table ownership.
 */
static void
require_index_select(Relation index, LOCKMODE lockmode)
{
	Oid		  heaprelid = IndexGetRelation(RelationGetRelid(index), false);
	AclResult aclresult =
			pg_class_aclcheck(heaprelid, GetUserId(), ACL_SELECT);
	if (aclresult != ACLCHECK_OK)
	{
		char	  *relname = get_rel_name(heaprelid);
		ObjectType objtype = get_relkind_objtype(get_rel_relkind(heaprelid));
		relation_close(index, lockmode);
		aclcheck_error(aclresult, objtype, relname);
	}
}

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
 * Shared helpers
 * ---------------------------------------------------------------- */

/* One leaf centroid entry with its location in the centroid tree */
typedef struct LeafEntry
{
	BlockNumber posting_head;
	BlockNumber centroid_page;
	uint16_t	entry_idx;
} LeafEntry;

/*
 * BFS the centroid tree and collect all leaf entries. Returns
 * the count and fills *out (palloc'd array). Caller must pfree.
 */
static int
collect_leaf_entries(
		Relation	index,
		BlockNumber first_centroid,
		uint8_t		nlevels,
		Dimension	dim,
		LeafEntry **out)
{
	int			 wl_cap	 = 64;
	int			 wl_len	 = 0;
	int			 wl_head = 0;
	BlockNumber *wl		 = palloc(wl_cap * sizeof(BlockNumber));

	int		   leaves_cap = 64;
	int		   leaves_len = 0;
	LeafEntry *leaves	  = palloc(leaves_cap * sizeof(LeafEntry));

	wl[wl_len++] = first_centroid;

	while (wl_head < wl_len)
	{
		BlockNumber blkno = wl[wl_head++];

		Buffer buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		Page page = BufferGetPage(buf);

		const MktCentroidPageOpaque *opaque	  = MKT_CENTROID_OPAQUE(page);
		uint16_t					 nentries = opaque->entry_count;
		MktCentroidFormat			 fmt = (MktCentroidFormat)(opaque->flags &
													   MKT_CENTROID_FMT_MASK);
		bool is_leaf_page				 = (opaque->level == nlevels - 1);

		check_centroid_count(blkno, nentries, dim, fmt);

		if (BlockNumberIsValid(opaque->next_blkno))
		{
			if (wl_len >= wl_cap)
			{
				wl_cap *= 2;
				wl = repalloc(wl, wl_cap * sizeof(BlockNumber));
			}
			wl[wl_len++] = opaque->next_blkno;
		}

		for (uint16_t i = 0; i < nentries; i++)
		{
			BlockNumber child;
			bool		is_leaf;

			if (fmt == MKT_CENTROID_FMT_FASTSCAN)
			{
				/* No per-entry meta: read the child from the packed
				 * group array and take leaf status from the page level. */
				char		*content = (char *)PageGetContents(page);
				uint32_t	 g		 = i / MKT_FASTSCAN_GROUP;
				uint32_t	 slot	 = i % MKT_FASTSCAN_GROUP;
				BlockNumber *grp =
						mkt_centroid_fastscan_group_child(content, g, dim);
				child	= grp[slot];
				is_leaf = is_leaf_page;
			}
			else
			{
				const MktCentroidEntryMeta *entry = mkt_centroid_meta(page, i);
				child							  = entry->child_blkno;
				is_leaf = (entry->flags & MKT_CENTROID_FLAG_LEAF) != 0;
			}

			if (!BlockNumberIsValid(child))
				continue;

			if (is_leaf)
			{
				if (leaves_len >= leaves_cap)
				{
					leaves_cap *= 2;
					leaves = repalloc(leaves, leaves_cap * sizeof(LeafEntry));
				}
				leaves[leaves_len++] = (LeafEntry){
						.posting_head  = child,
						.centroid_page = blkno,
						.entry_idx	   = i,
				};
			}
			else
			{
				if (wl_len >= wl_cap)
				{
					wl_cap *= 2;
					wl = repalloc(wl, wl_cap * sizeof(BlockNumber));
				}
				wl[wl_len++] = child;
			}
		}

		UnlockReleaseBuffer(buf);
	}

	pfree(wl);
	*out = leaves;
	return leaves_len;
}

/* Format name lookup (indexed by MktCentroidFormat) */
static const char *centroid_format_names[] = {
		[MKT_CENTROID_FMT_RABITQ]	= "rabitq",
		[MKT_CENTROID_FMT_FLOAT]	= "float",
		[MKT_CENTROID_FMT_HALF]		= "half",
		[MKT_CENTROID_FMT_FASTSCAN] = "fastscan",
};

/*
 * mkt_centroid_pages(regclass)
 *
 * Returns one row per centroid entry: blkno, entry, level, format,
 * child_blkno, child_count, is_leaf. Traverses the tree via BFS
 * starting from the metapage's first_centroid, following next_blkno
 * chains and child_blkno links.
 */
Datum
mkt_centroid_pages(PG_FUNCTION_ARGS)
{
	Oid			   indexoid = PG_GETARG_OID(0);
	ReturnSetInfo *rsinfo	= (ReturnSetInfo *)fcinfo->resultinfo;

	InitMaterializedSRF(fcinfo, 0);

	/* Open relation and verify it's an index */
	Relation index = relation_open(indexoid, AccessShareLock);

	if (index->rd_rel->relkind != RELKIND_INDEX)
	{
		relation_close(index, AccessShareLock);
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("\"%s\" is not an index",
						RelationGetRelationName(index))));
	}

	require_index_select(index, AccessShareLock);

	/* Read metapage and verify magic */
	Buffer meta_buf = ReadBuffer(index, 0);
	LockBuffer(meta_buf, BUFFER_LOCK_SHARE);
	Page meta_page = BufferGetPage(meta_buf);

	const MktannMetaPage *meta = (const MktannMetaPage *)PageGetSpecialPointer(
			meta_page);

	if (meta->magic != MKT_META_MAGIC)
	{
		UnlockReleaseBuffer(meta_buf);
		relation_close(index, AccessShareLock);
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("\"%s\" is not an mktann index",
						RelationGetRelationName(index))));
	}

	BlockNumber first_centroid = meta->first_centroid;
	Dimension	dim			   = (Dimension)meta->dim;
	uint8_t		nlevels		   = meta->nlevels;

	UnlockReleaseBuffer(meta_buf);

	if (!BlockNumberIsValid(first_centroid))
	{
		relation_close(index, AccessShareLock);
		PG_RETURN_NULL();
	}

	/*
	 * BFS worklist — simple dynamic array of block numbers.
	 * Start with the root centroid page.
	 */
	int			 worklist_cap  = 64;
	int			 worklist_len  = 0;
	int			 worklist_head = 0;
	BlockNumber *worklist	   = palloc(worklist_cap * sizeof(BlockNumber));

	worklist[worklist_len++] = first_centroid;

	while (worklist_head < worklist_len)
	{
		BlockNumber blkno = worklist[worklist_head++];

		Buffer buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		Page page = BufferGetPage(buf);

		const MktCentroidPageOpaque *opaque = MKT_CENTROID_OPAQUE(page);
		MktCentroidFormat			 fmt = (MktCentroidFormat)(opaque->flags &
													   MKT_CENTROID_FMT_MASK);
		/* FASTSCAN doesn't have per-entry flags, so determine leaf
		 * status from tree depth: bottom level holds posting heads. */
		bool is_leaf_page = (opaque->level == nlevels - 1);

		/* Emit one row per entry. FASTSCAN pages have a different
		 * layout (group section instead of per-entry meta + data),
		 * so child_blkno lives in the group array and per-entry
		 * flags don't exist — the leaf bit is page-level. */
		uint16_t nentries = opaque->entry_count;

		check_centroid_count(blkno, nentries, dim, fmt);

		if (fmt == MKT_CENTROID_FMT_FASTSCAN)
		{
			char *content = (char *)PageGetContents(page);

			for (uint16_t i = 0; i < nentries; i++)
			{
				uint32_t	 g	  = i / MKT_FASTSCAN_GROUP;
				uint32_t	 slot = i % MKT_FASTSCAN_GROUP;
				BlockNumber *child =
						mkt_centroid_fastscan_group_child(content, g, dim);

				Datum values[7];
				bool  nulls[7] = {0};

				values[0] = Int32GetDatum((int32)blkno);
				values[1] = Int16GetDatum((int16)i);
				values[2] = Int16GetDatum((int16)opaque->level);
				values[3] = CStringGetTextDatum(centroid_format_names[fmt]);

				if (BlockNumberIsValid(child[slot]))
					values[4] = Int32GetDatum((int32)child[slot]);
				else
					nulls[4] = true;

				/* child_count is not stored per-entry in FASTSCAN — the
				 * page-level leaf flag tells us whether children are
				 * posting heads (child_count = 0) or another centroid
				 * page. Leave as 0 / NULL. */
				nulls[5]  = true;
				values[6] = BoolGetDatum(is_leaf_page);

				tuplestore_putvalues(
						rsinfo->setResult, rsinfo->setDesc, values, nulls);

				/* Enqueue children if this isn't a leaf page */
				if (!is_leaf_page && BlockNumberIsValid(child[slot]))
				{
					if (worklist_len >= worklist_cap)
					{
						worklist_cap *= 2;
						worklist = repalloc(
								worklist, worklist_cap * sizeof(BlockNumber));
					}
					worklist[worklist_len++] = child[slot];
				}
			}
		}
		else
		{
			for (uint16_t i = 0; i < nentries; i++)
			{
				const MktCentroidEntryMeta *entry = mkt_centroid_meta(page, i);
				bool is_leaf = (entry->flags & MKT_CENTROID_FLAG_LEAF) != 0;

				Datum values[7];
				bool  nulls[7] = {0};

				values[0] = Int32GetDatum((int32)blkno);
				values[1] = Int16GetDatum((int16)i);
				values[2] = Int16GetDatum((int16)opaque->level);
				values[3] = CStringGetTextDatum(centroid_format_names[fmt]);

				/* Leaf entries carry their posting-list head block in
				 * child_blkno (formula-derived: first_posting + leaf), so
				 * expose it -- structural tests join it against the posting
				 * heads. */
				if (BlockNumberIsValid(entry->child_blkno))
					values[4] = Int32GetDatum((int32)entry->child_blkno);
				else
					nulls[4] = true;

				values[5] = Int16GetDatum((int16)entry->child_count);
				values[6] = BoolGetDatum(is_leaf);

				tuplestore_putvalues(
						rsinfo->setResult, rsinfo->setDesc, values, nulls);

				if (!is_leaf && BlockNumberIsValid(entry->child_blkno))
				{
					if (worklist_len >= worklist_cap)
					{
						worklist_cap *= 2;
						worklist = repalloc(
								worklist, worklist_cap * sizeof(BlockNumber));
					}
					worklist[worklist_len++] = entry->child_blkno;
				}
			}
		}

		/* Enqueue sibling (next_blkno chain) */
		if (BlockNumberIsValid(opaque->next_blkno))
		{
			if (worklist_len >= worklist_cap)
			{
				worklist_cap *= 2;
				worklist =
						repalloc(worklist, worklist_cap * sizeof(BlockNumber));
			}
			worklist[worklist_len++] = opaque->next_blkno;
		}

		UnlockReleaseBuffer(buf);
	}

	pfree(worklist);
	relation_close(index, AccessShareLock);

	PG_RETURN_NULL();
}

/* ----------------------------------------------------------------
 * mkt.posting_pages(regclass)
 *
 * Returns one row per posting page: blkno, cluster_id, is_first,
 * tombstoned, entry_count, dead_count, max_entries, next_blkno,
 * chain_pos, format. Tombstoned (all-dead, but still linked) pages stay
 * in the output so bloat is visible; filter with WHERE NOT tombstoned
 * for live pages. dead_count is the per-entry DELETED tally for AoS
 * pages and NULL for fastscan pages (no per-entry state).
 *
 * Walks all posting chains by finding leaf centroids (which store
 * posting_head block numbers) and following next_blkno links.
 * ---------------------------------------------------------------- */
Datum
mkt_posting_pages(PG_FUNCTION_ARGS)
{
	Oid			   indexoid = PG_GETARG_OID(0);
	ReturnSetInfo *rsinfo	= (ReturnSetInfo *)fcinfo->resultinfo;

	InitMaterializedSRF(fcinfo, 0);

	Relation index = relation_open(indexoid, AccessShareLock);

	if (index->rd_rel->relkind != RELKIND_INDEX)
	{
		relation_close(index, AccessShareLock);
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("\"%s\" is not an index",
						RelationGetRelationName(index))));
	}

	require_index_select(index, AccessShareLock);

	Buffer meta_buf = ReadBuffer(index, 0);
	LockBuffer(meta_buf, BUFFER_LOCK_SHARE);
	Page meta_page = BufferGetPage(meta_buf);

	const MktannMetaPage *meta = (const MktannMetaPage *)PageGetSpecialPointer(
			meta_page);

	if (meta->magic != MKT_META_MAGIC)
	{
		UnlockReleaseBuffer(meta_buf);
		relation_close(index, AccessShareLock);
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("\"%s\" is not an mktann index",
						RelationGetRelationName(index))));
	}

	BlockNumber first_centroid = meta->first_centroid;
	Dimension	dim			   = meta->dim;
	uint8_t		nlevels		   = meta->nlevels;
	UnlockReleaseBuffer(meta_buf);

	if (!BlockNumberIsValid(first_centroid))
	{
		relation_close(index, AccessShareLock);
		PG_RETURN_NULL();
	}

	LeafEntry *leaves;
	int		   nleaves =
			collect_leaf_entries(index, first_centroid, nlevels, dim, &leaves);

	/* Walk each posting chain */
	for (int c = 0; c < nleaves; c++)
	{
		BlockNumber blkno	  = leaves[c].posting_head;
		int			chain_pos = 0;

		while (BlockNumberIsValid(blkno))
		{
			Buffer buf = ReadBuffer(index, blkno);
			LockBuffer(buf, BUFFER_LOCK_SHARE);
			Page page = BufferGetPage(buf);

			const MktPostingPageOpaque *op = mkt_posting_opaque(page);
			bool is_first	 = (op->flags & MKT_POSTING_PAGE_FIRST) != 0;
			bool tombstoned	 = (op->flags & MKT_POSTING_PAGE_TOMBSTONED) != 0;
			bool is_fastscan = (op->flags & MKT_POSTING_PAGE_FASTSCAN) != 0;

			Datum values[10];
			bool  nulls[10] = {0};

			values[0] = Int32GetDatum((int32)blkno);
			values[1] = Int32GetDatum((int32)op->cluster_id);
			values[2] = BoolGetDatum(is_first);
			values[3] = BoolGetDatum(tombstoned);
			values[4] = Int32GetDatum((int32)op->entry_count);

			if (!is_fastscan)
			{
				/* AoS entries carry a per-entry DELETED flag; count them.
				 * FASTSCAN packs entries into SIMD groups with no per-entry
				 * state (deletion is page-granular there), so dead_count is
				 * NULL for fastscan pages. */
				char *content = is_first ? mkt_posting_content_first(page, dim)
										 : mkt_posting_content(page);
				int32 dead	  = 0;
				check_posting_count(blkno, op->entry_count, dim, is_first);
				for (uint32_t i = 0; i < op->entry_count; i++)
				{
					const MktPostingEntryHeader *h =
							mkt_posting_entry_at(content, i, dim);
					if (h->meta.flags & MKT_POSTING_FLAG_DELETED)
						dead++;
				}
				values[5] = Int32GetDatum(dead);
			}
			else
				nulls[5] = true;

			values[6] = Int32GetDatum((int32)op->max_entries);

			if (BlockNumberIsValid(op->next_blkno))
				values[7] = Int32GetDatum((int32)op->next_blkno);
			else
				nulls[7] = true;

			values[8] = Int32GetDatum(chain_pos);
			values[9] = CStringGetTextDatum(is_fastscan ? "fastscan" : "aos");

			tuplestore_putvalues(
					rsinfo->setResult, rsinfo->setDesc, values, nulls);

			BlockNumber next = op->next_blkno;
			UnlockReleaseBuffer(buf);

			blkno = next;
			chain_pos++;
		}
	}

	pfree(leaves);
	relation_close(index, AccessShareLock);

	PG_RETURN_NULL();
}

/* ----------------------------------------------------------------
 * Centroid tree helpers for posting head updates
 * ---------------------------------------------------------------- */

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
	 * Find the leaf for this cluster by its stored cluster_id -- the value
	 * mkt.posting_pages reports and callers pass -- rather than by position in
	 * the leaf array. Today the two coincide: the build numbers clusters in
	 * the same order collect_leaf_entries traverses them, so leaves[i] always
	 * has cluster_id i. That correspondence is not guaranteed to hold once
	 * split/merge (LIRE) starts minting cluster_ids that no longer match tree
	 * position, so match on the head's own cluster_id to keep the argument
	 * meaning the same thing regardless of how the tree was assembled.
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
 * mkt.tids_clusters(regclass, tid[])
 *
 * Diagnostic: for each input heap TID, return which cluster(s) it is
 * stored in (primary + any SOAR/boundary replica). One pass over all
 * leaf posting lists; emits (tid, cluster_id) for matched TIDs only.
 * Lets a caller compare "where the true nearest neighbors live" against
 * "which clusters a query scans".
 * ---------------------------------------------------------------- */
static int
cmp_u64(const void *a, const void *b)
{
	uint64 x = *(const uint64 *)a, y = *(const uint64 *)b;
	return (x > y) - (x < y);
}

/* Emit (tid, cluster_id) if tid is one of the sorted target keys. */
static void
emit_tid_cluster(
		ReturnSetInfo *rsinfo,
		const uint64  *keys,
		int			   nk,
		ItemPointer	   tid,
		uint32_t	   cluster_id)
{
	uint64 enc = mkt_posting_encode_tid(tid);
	if (bsearch(&enc, keys, nk, sizeof(uint64), cmp_u64) == NULL)
		return;

	Datum		values[2];
	bool		nulls[2] = {0};
	ItemPointer out		 = palloc(sizeof(ItemPointerData));
	*out				 = *tid;
	values[0]			 = PointerGetDatum(out);
	values[1]			 = Int32GetDatum((int32)cluster_id);
	tuplestore_putvalues(rsinfo->setResult, rsinfo->setDesc, values, nulls);
}

Datum
mkt_tids_clusters(PG_FUNCTION_ARGS)
{
	Oid			   indexoid = PG_GETARG_OID(0);
	ArrayType	  *arr		= PG_GETARG_ARRAYTYPE_P(1);
	ReturnSetInfo *rsinfo	= (ReturnSetInfo *)fcinfo->resultinfo;

	InitMaterializedSRF(fcinfo, 0);

	/* Deconstruct tid[] into a sorted array of encoded uint64 keys. */
	Datum *elems;
	bool  *elnulls;
	int	   nelems;
	deconstruct_array(
			arr,
			TIDOID,
			sizeof(ItemPointerData),
			false,
			TYPALIGN_SHORT,
			&elems,
			&elnulls,
			&nelems);
	uint64 *keys = palloc(Max(nelems, 1) * sizeof(uint64));
	int		nk	 = 0;
	for (int i = 0; i < nelems; i++)
	{
		if (elnulls[i])
			continue;
		keys[nk++] = mkt_posting_encode_tid(
				(ItemPointer)DatumGetPointer(elems[i]));
	}
	qsort(keys, nk, sizeof(uint64), cmp_u64);

	Relation index = relation_open(indexoid, AccessShareLock);

	require_index_select(index, AccessShareLock);

	Buffer meta_buf = ReadBuffer(index, 0);
	LockBuffer(meta_buf, BUFFER_LOCK_SHARE);
	const MktannMetaPage *meta = (const MktannMetaPage *)PageGetSpecialPointer(
			BufferGetPage(meta_buf));
	Dimension dim = (Dimension)meta->dim;
	UnlockReleaseBuffer(meta_buf);

	/* Scan every block and pick out posting pages directly (identified by
	 * page_id), rather than walking the centroid tree to find posting
	 * heads — that avoids any dependency on the leaf-enumeration path and
	 * each posting page already carries its cluster_id. */
	BlockNumber nblocks = RelationGetNumberOfBlocks(index);
	for (BlockNumber blkno = 1; nk > 0 && blkno < nblocks; blkno++)
	{
		Buffer buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		Page page = BufferGetPage(buf);

		if (PageGetSpecialSize(page) == MAXALIGN(sizeof(MktPostingPageOpaque)))
		{
			const MktPostingPageOpaque *op = mkt_posting_opaque(page);

			if (op->page_id == MKT_POSTING_PAGE_ID)
			{
				char	*content = (op->flags & MKT_POSTING_PAGE_FIRST)
										 ? mkt_posting_content_first(page, dim)
										 : mkt_posting_content(page);
				uint32_t count	 = op->entry_count;

				check_posting_count(
						blkno,
						count,
						dim,
						(op->flags & MKT_POSTING_PAGE_FIRST) != 0);

				if (op->flags & MKT_POSTING_PAGE_FASTSCAN)
				{
					/* SoA: TIDs packed per group of MKT_FASTSCAN_GROUP. */
					uint32_t ngroups = (count + MKT_FASTSCAN_GROUP - 1) /
									   MKT_FASTSCAN_GROUP;
					for (uint32_t g = 0; g < ngroups; g++)
					{
						ItemPointerData *tids =
								mkt_fastscan_group_tids(content, g, dim);
						uint32_t gc = count - g * MKT_FASTSCAN_GROUP;
						if (gc > MKT_FASTSCAN_GROUP)
							gc = MKT_FASTSCAN_GROUP;
						for (uint32_t v = 0; v < gc; v++)
							emit_tid_cluster(
									rsinfo,
									keys,
									nk,
									&tids[v],
									op->cluster_id);
					}
				}
				else
				{
					/* AoS: the TID is at the head of each entry. */
					for (uint32_t i = 0; i < count; i++)
					{
						MktPostingEntryHeader *e =
								mkt_posting_entry_at(content, i, dim);
						emit_tid_cluster(
								rsinfo,
								keys,
								nk,
								&e->meta.tid,
								op->cluster_id);
					}
				}
			}
		}
		UnlockReleaseBuffer(buf);
	}

	relation_close(index, AccessShareLock);
	PG_RETURN_NULL();
}
