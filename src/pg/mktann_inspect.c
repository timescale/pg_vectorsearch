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
#include <access/table.h>
#include <access/tableam.h>
#include <catalog/pg_type.h>
#include <executor/tuptable.h>
#include <funcapi.h>
#include <storage/bufmgr.h>
#include <utils/array.h>
#include <utils/builtins.h>
#include <utils/rel.h>

#include <math.h>

#include "algo/kmeans.h"
#include "algo/vecops.h"
#include "index/centroid_page.h"
#include "index/posting_convert.h"
#include "index/posting_page.h"
#include "mkt_pg.h"
#include "mkt_vector.h"
#include "mktann_meta.h"
#include "mktann_storage.h"

PG_FUNCTION_INFO_V1(mkt_centroid_pages);
PG_FUNCTION_INFO_V1(mkt_posting_pages);
PG_FUNCTION_INFO_V1(mkt_tids_clusters);
PG_FUNCTION_INFO_V1(mkt_convert_posting_to_fastscan);

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
 *
 * Format-aware: FASTSCAN centroid pages store children in a packed
 * group section (no per-entry meta), and leaf status is page-level
 * (bottom tree level) rather than a per-entry flag. nlevels and dim
 * come from the meta page; dim is only needed to locate group children
 * on FASTSCAN pages.
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
		MktCentroidFormat			 fmt =
				(MktCentroidFormat)(opaque->flags & MKT_CENTROID_FMT_MASK);
		bool is_leaf_page = (opaque->level == nlevels - 1);

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
				char	   *content = (char *) PageGetContents(page);
				uint32_t	g		= i / MKT_FASTSCAN_GROUP;
				uint32_t	slot	= i % MKT_FASTSCAN_GROUP;
				BlockNumber *grp	= mkt_centroid_fastscan_group_child(
						   content, g, dim);
				child	= grp[slot];
				is_leaf = is_leaf_page;
			}
			else
			{
				const MktCentroidEntryMeta *entry = mkt_centroid_meta(page, i);
				child	= entry->child_blkno;
				is_leaf = (entry->flags & MKT_CENTROID_FLAG_LEAF) != 0;
			}

			if (!BlockNumberIsValid(child))
				continue;

			if (is_leaf)
			{
				if (leaves_len >= leaves_cap)
				{
					leaves_cap *= 2;
					leaves =
							repalloc(leaves, leaves_cap * sizeof(LeafEntry));
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
	Dimension	dim			   = (Dimension) meta->dim;
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

		if (fmt == MKT_CENTROID_FMT_FASTSCAN)
		{
			char *content = (char *) PageGetContents(page);

			for (uint16_t i = 0; i < nentries; i++)
			{
				uint32_t	 g	   = i / MKT_FASTSCAN_GROUP;
				uint32_t	 slot  = i % MKT_FASTSCAN_GROUP;
				BlockNumber *child = mkt_centroid_fastscan_group_child(
						content, g, dim);

				Datum values[7];
				bool  nulls[7] = {0};

				values[0] = Int32GetDatum((int32) blkno);
				values[1] = Int16GetDatum((int16) i);
				values[2] = Int16GetDatum((int16) opaque->level);
				values[3] = CStringGetTextDatum(
						centroid_format_names[fmt]);

				if (BlockNumberIsValid(child[slot]))
					values[4] = Int32GetDatum((int32) child[slot]);
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
						worklist	 = repalloc(
								 worklist,
								 worklist_cap * sizeof(BlockNumber));
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

				values[0] = Int32GetDatum((int32) blkno);
				values[1] = Int16GetDatum((int16) i);
				values[2] = Int16GetDatum((int16) opaque->level);
				values[3] = CStringGetTextDatum(centroid_format_names[fmt]);

				if (!is_leaf && BlockNumberIsValid(entry->child_blkno))
					values[4] = Int32GetDatum((int32) entry->child_blkno);
				else
					nulls[4] = true;

				values[5] = Int16GetDatum((int16) entry->child_count);
				values[6] = BoolGetDatum(is_leaf);

				tuplestore_putvalues(
						rsinfo->setResult, rsinfo->setDesc, values, nulls);

				if (!is_leaf && BlockNumberIsValid(entry->child_blkno))
				{
					if (worklist_len >= worklist_cap)
					{
						worklist_cap *= 2;
						worklist	 = repalloc(
								 worklist,
								 worklist_cap * sizeof(BlockNumber));
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
 * mkt.tids_clusters(regclass, tid[])
 *
 * Diagnostic: for each input heap TID, return which cluster(s) it is
 * stored in (primary + any SOAR/boundary replica). One pass over all
 * leaf posting lists; emits (tid, cluster_id) for matched TIDs only.
 * Lets a caller compare "where the true nearest neighbors live" against
 * "which clusters a query scans". Fastscan posting pages only.
 * ---------------------------------------------------------------- */
static int
cmp_u64(const void *a, const void *b)
{
	uint64 x = *(const uint64 *)a, y = *(const uint64 *)b;
	return (x > y) - (x < y);
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
			arr, TIDOID, sizeof(ItemPointerData), false, TYPALIGN_SHORT,
			&elems, &elnulls, &nelems);
	uint64 *keys = palloc(Max(nelems, 1) * sizeof(uint64));
	int		nk	 = 0;
	for (int i = 0; i < nelems; i++)
	{
		if (elnulls[i])
			continue;
		keys[nk++] =
				mkt_posting_encode_tid((ItemPointer)DatumGetPointer(elems[i]));
	}
	qsort(keys, nk, sizeof(uint64), cmp_u64);

	Relation index = relation_open(indexoid, AccessShareLock);

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

		if (PageGetSpecialSize(page) ==
			MAXALIGN(sizeof(MktPostingPageOpaque)))
		{
			const MktPostingPageOpaque *op = mkt_posting_opaque(page);

			if (op->page_id == MKT_POSTING_PAGE_ID &&
				(op->flags & MKT_POSTING_PAGE_FASTSCAN))
			{
				char *content = (op->flags & MKT_POSTING_PAGE_FIRST)
									? mkt_posting_content_first(page, dim)
									: mkt_posting_content(page);
				uint32_t count	 = op->entry_count;
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
					{
						uint64 enc = mkt_posting_encode_tid(&tids[v]);
						if (bsearch(&enc, keys, nk, sizeof(uint64), cmp_u64))
						{
							Datum		values[2];
							bool		nulls[2] = {0};
							ItemPointer out = palloc(sizeof(ItemPointerData));
							*out	  = tids[v];
							values[0] = PointerGetDatum(out);
							values[1] = Int32GetDatum((int32)op->cluster_id);
							tuplestore_putvalues(
									rsinfo->setResult, rsinfo->setDesc, values,
									nulls);
						}
					}
				}
			}
		}
		UnlockReleaseBuffer(buf);
	}

	relation_close(index, AccessShareLock);
	PG_RETURN_NULL();
}

/* ----------------------------------------------------------------
 * mkt.posting_pages(regclass)
 *
 * Returns one row per posting page: blkno, cluster_id, is_first,
 * entry_count, max_entries, next_blkno, chain_pos.
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
	uint8_t		nlevels		   = meta->nlevels;
	Dimension	dim			   = (Dimension) meta->dim;
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
			bool is_fastscan = (op->flags & MKT_POSTING_PAGE_FASTSCAN) != 0;

			Datum values[8];
			bool  nulls[8] = {0};

			values[0] = Int32GetDatum((int32)blkno);
			values[1] = Int32GetDatum((int32)op->cluster_id);
			values[2] = BoolGetDatum(is_first);
			values[3] = Int32GetDatum((int32)op->entry_count);
			values[4] = Int32GetDatum((int32)op->max_entries);

			if (BlockNumberIsValid(op->next_blkno))
				values[5] = Int32GetDatum((int32)op->next_blkno);
			else
				nulls[5] = true;

			values[6] = Int32GetDatum(chain_pos);
			values[7] = CStringGetTextDatum(is_fastscan ? "fastscan" : "aos");

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
 * Update a centroid leaf entry's posting head pointer via WAL.
 */
static void
update_centroid_posting_head(
		Relation	index,
		BlockNumber centroid_page,
		uint16_t	entry_idx,
		BlockNumber new_head)
{
	GenericXLogState *state = GenericXLogStart(index);
	Buffer			  buf	= ReadBuffer(index, centroid_page);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	Page page = GenericXLogRegisterBuffer(state, buf, GENERIC_XLOG_FULL_IMAGE);

	MktCentroidEntryMeta *entry = (MktCentroidEntryMeta *)
			mkt_centroid_meta(page, entry_idx);
	entry->child_blkno = new_head;

	GenericXLogFinish(state);
	UnlockReleaseBuffer(buf);
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

	/* Find the leaf entry for this cluster */
	LeafEntry *leaves;
	int		   nleaves =
			collect_leaf_entries(index, first_centroid, nlevels, dim, &leaves);

	if (cluster_id < 0 || cluster_id >= nleaves)
	{
		pfree(leaves);
		relation_close(index, RowExclusiveLock);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("cluster %d not found or has no posting list",
						cluster_id)));
	}

	LeafEntry  *leaf		  = &leaves[cluster_id];
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
	storage.build_mode = true;

	BlockNumber new_head =
			mkt_posting_convert_to_fastscan(&storage.base, old_head, dim);

	update_centroid_posting_head(index, centroid_page, entry_idx, new_head);
	ensure_meta_fastscan_flag(index);

	relation_close(index, RowExclusiveLock);

	PG_RETURN_INT32((int32)new_head);
}

/* ----------------------------------------------------------------
 * mkt.cluster_subcentroids(regclass, k, max_members)
 *
 * Diagnostic for the sub-centroid routing idea: for each leaf cluster,
 * gather up to max_members of its actual member vectors from the heap,
 * run k-means(k) on them, and return the k sub-centroids. Lets us test
 * (offline, no rebuild) whether routing by nearest sub-centroid instead
 * of the single mean would reduce the clusters scanned at target recall.
 * Read-only. Cosine: vectors are normalized before clustering.
 * ---------------------------------------------------------------- */
PG_FUNCTION_INFO_V1(mkt_cluster_subcentroids);

Datum
mkt_cluster_subcentroids(PG_FUNCTION_ARGS)
{
	Oid			   indexoid	   = PG_GETARG_OID(0);
	int32		   K		   = PG_GETARG_INT32(1);
	int32		   max_members = PG_GETARG_INT32(2);
	ReturnSetInfo *rsinfo	   = (ReturnSetInfo *)fcinfo->resultinfo;

	if (K < 1)
		K = 1;
	if (max_members < K)
		max_members = K;

	InitMaterializedSRF(fcinfo, 0);

	Relation index = relation_open(indexoid, AccessShareLock);

	Buffer meta_buf = ReadBuffer(index, 0);
	LockBuffer(meta_buf, BUFFER_LOCK_SHARE);
	const MktannMetaPage *meta =
			(const MktannMetaPage *)PageGetSpecialPointer(BufferGetPage(meta_buf));
	Dimension	   dim	  = (Dimension)meta->dim;
	uint32_t	   nlist  = meta->nlist;
	DistanceMetric metric = (DistanceMetric)meta->metric;
	UnlockReleaseBuffer(meta_buf);

	/* Per-cluster member-tid buffers (capped at max_members). */
	ItemPointerData **members = palloc0((size_t)nlist * sizeof(ItemPointerData *));
	uint32_t		 *counts  = palloc0((size_t)nlist * sizeof(uint32_t));

	BlockNumber nblocks = RelationGetNumberOfBlocks(index);
	for (BlockNumber blkno = 1; blkno < nblocks; blkno++)
	{
		Buffer buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		Page page = BufferGetPage(buf);

		if (PageGetSpecialSize(page) == MAXALIGN(sizeof(MktPostingPageOpaque)))
		{
			const MktPostingPageOpaque *op = mkt_posting_opaque(page);
			if (op->page_id == MKT_POSTING_PAGE_ID &&
				(op->flags & MKT_POSTING_PAGE_FASTSCAN) &&
				op->cluster_id < nlist)
			{
				uint32_t c = op->cluster_id;
				if (counts[c] < (uint32_t)max_members)
				{
					char *content = (op->flags & MKT_POSTING_PAGE_FIRST)
										? mkt_posting_content_first(page, dim)
										: mkt_posting_content(page);
					uint32_t cnt	 = op->entry_count;
					uint32_t ngroups = (cnt + MKT_FASTSCAN_GROUP - 1) /
									   MKT_FASTSCAN_GROUP;
					for (uint32_t g = 0;
						 g < ngroups && counts[c] < (uint32_t)max_members; g++)
					{
						ItemPointerData *tids =
								mkt_fastscan_group_tids(content, g, dim);
						uint32_t gc = cnt - g * MKT_FASTSCAN_GROUP;
						if (gc > MKT_FASTSCAN_GROUP)
							gc = MKT_FASTSCAN_GROUP;
						for (uint32_t v = 0;
							 v < gc && counts[c] < (uint32_t)max_members; v++)
						{
							if (members[c] == NULL)
								members[c] = palloc(
										(size_t)max_members *
										sizeof(ItemPointerData));
							members[c][counts[c]++] = tids[v];
						}
					}
				}
			}
		}
		UnlockReleaseBuffer(buf);
	}

	/* Heap + a slot to fetch member vectors. */
	Relation		heap	   = table_open(index->rd_index->indrelid,
											AccessShareLock);
	int				vec_attnum = index->rd_index->indkey.values[0];
	TupleTableSlot *slot	   = table_slot_create(heap, NULL);
	float		   *vecs = palloc((size_t)max_members * dim * sizeof(float));

	for (uint32_t c = 0; c < nlist; c++)
	{
		if (counts[c] == 0)
			continue;

		uint32_t n = 0;
		for (uint32_t i = 0; i < counts[c]; i++)
		{
			if (!table_tuple_fetch_row_version(heap, &members[c][i], SnapshotAny,
											   slot))
				continue;
			bool  isnull;
			Datum val = slot_getattr(slot, vec_attnum, &isnull);
			if (!isnull)
			{
				MktVector *vec = DatumGetMktVector(val);
				float	  *dst = vecs + (size_t)n * dim;
				memcpy(dst, vec->x, (size_t)dim * sizeof(float));
				if (metric == DISTANCE_COSINE)
				{
					float nrm = mkt_l2_norm(dst, dim);
					if (nrm > 0.0f)
						for (Dimension d = 0; d < dim; d++)
							dst[d] /= nrm;
				}
				n++;
			}
			ExecClearTuple(slot);
		}
		if (n == 0)
			continue;

		uint32_t	  kk   = (uint32_t)K < n ? (uint32_t)K : n;
		KMeansResult *r	   = mkt_kmeans_f32(vecs, n, dim, kk, metric, NULL);
		if (r == NULL)
			continue;
		for (uint32_t j = 0; j < r->nlist; j++)
		{
			MktVector *sv = mkt_vector_create(dim);
			memcpy(sv->x, r->centroids + (size_t)j * dim,
				   (size_t)dim * sizeof(float));
			Datum values[2];
			bool  nulls[2] = {0};
			values[0]	   = Int32GetDatum((int32)c);
			values[1]	   = PointerGetDatum(sv);
			tuplestore_putvalues(rsinfo->setResult, rsinfo->setDesc, values,
								 nulls);
		}
		mkt_kmeans_result_destroy(r);
	}

	ExecDropSingleTupleTableSlot(slot);
	table_close(heap, AccessShareLock);
	relation_close(index, AccessShareLock);
	PG_RETURN_NULL();
}

/* ----------------------------------------------------------------
 * mkt.cluster_entry_points(regclass, k, max_members)
 *
 * Diagnostic for the multiple-entry-points routing idea: for each leaf
 * cluster, gather up to max_members of its actual member vectors from
 * the heap and select K of them by farthest-point-sampling (FPS) —
 * seed = member farthest from the cluster mean (a boundary point),
 * then greedily add the member maximizing the min-distance to the
 * already-selected set. Returns the K members as routing "entry
 * points", tagged with their FPS order k=1..K so one materialization
 * serves every K (filter k <= K). Lets us test (offline, no rebuild)
 * whether routing by nearest-of-K real member entry points reaches the
 * NN-clusters at a LOWER nprobe than the single centroid (mean) does.
 * Unlike cluster_subcentroids (interior k-means means), these are real
 * boundary-covering points — the variable prior tests never probed.
 * Read-only. Cosine: vectors are normalized before selection.
 * ---------------------------------------------------------------- */
PG_FUNCTION_INFO_V1(mkt_cluster_entry_points);

Datum
mkt_cluster_entry_points(PG_FUNCTION_ARGS)
{
	Oid			   indexoid	   = PG_GETARG_OID(0);
	int32		   K		   = PG_GETARG_INT32(1);
	int32		   max_members = PG_GETARG_INT32(2);
	ReturnSetInfo *rsinfo	   = (ReturnSetInfo *)fcinfo->resultinfo;

	if (K < 1)
		K = 1;
	if (max_members < K)
		max_members = K;

	InitMaterializedSRF(fcinfo, 0);

	Relation index = relation_open(indexoid, AccessShareLock);

	Buffer meta_buf = ReadBuffer(index, 0);
	LockBuffer(meta_buf, BUFFER_LOCK_SHARE);
	const MktannMetaPage *meta =
			(const MktannMetaPage *)PageGetSpecialPointer(BufferGetPage(meta_buf));
	Dimension	   dim	  = (Dimension)meta->dim;
	uint32_t	   nlist  = meta->nlist;
	DistanceMetric metric = (DistanceMetric)meta->metric;
	UnlockReleaseBuffer(meta_buf);

	/* Per-cluster member-tid buffers (capped at max_members). */
	ItemPointerData **members = palloc0((size_t)nlist * sizeof(ItemPointerData *));
	uint32_t		 *counts  = palloc0((size_t)nlist * sizeof(uint32_t));

	BlockNumber nblocks = RelationGetNumberOfBlocks(index);
	for (BlockNumber blkno = 1; blkno < nblocks; blkno++)
	{
		Buffer buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		Page page = BufferGetPage(buf);

		if (PageGetSpecialSize(page) == MAXALIGN(sizeof(MktPostingPageOpaque)))
		{
			const MktPostingPageOpaque *op = mkt_posting_opaque(page);
			if (op->page_id == MKT_POSTING_PAGE_ID &&
				(op->flags & MKT_POSTING_PAGE_FASTSCAN) &&
				op->cluster_id < nlist)
			{
				uint32_t c = op->cluster_id;
				if (counts[c] < (uint32_t)max_members)
				{
					char *content = (op->flags & MKT_POSTING_PAGE_FIRST)
										? mkt_posting_content_first(page, dim)
										: mkt_posting_content(page);
					uint32_t cnt	 = op->entry_count;
					uint32_t ngroups = (cnt + MKT_FASTSCAN_GROUP - 1) /
									   MKT_FASTSCAN_GROUP;
					for (uint32_t g = 0;
						 g < ngroups && counts[c] < (uint32_t)max_members; g++)
					{
						ItemPointerData *tids =
								mkt_fastscan_group_tids(content, g, dim);
						uint32_t gc = cnt - g * MKT_FASTSCAN_GROUP;
						if (gc > MKT_FASTSCAN_GROUP)
							gc = MKT_FASTSCAN_GROUP;
						for (uint32_t v = 0;
							 v < gc && counts[c] < (uint32_t)max_members; v++)
						{
							if (members[c] == NULL)
								members[c] = palloc(
										(size_t)max_members *
										sizeof(ItemPointerData));
							members[c][counts[c]++] = tids[v];
						}
					}
				}
			}
		}
		UnlockReleaseBuffer(buf);
	}

	/* Heap + a slot to fetch member vectors. */
	Relation		heap	   = table_open(index->rd_index->indrelid,
											AccessShareLock);
	int				vec_attnum = index->rd_index->indkey.values[0];
	TupleTableSlot *slot	   = table_slot_create(heap, NULL);
	float		   *vecs = palloc((size_t)max_members * dim * sizeof(float));
	float		   *mean = palloc(dim * sizeof(float));
	float		   *mind = palloc((size_t)max_members * sizeof(float));
	uint32_t	   *sel	 = palloc((size_t)max_members * sizeof(uint32_t));

	for (uint32_t c = 0; c < nlist; c++)
	{
		if (counts[c] == 0)
			continue;

		uint32_t n = 0;
		for (uint32_t i = 0; i < counts[c]; i++)
		{
			if (!table_tuple_fetch_row_version(heap, &members[c][i], SnapshotAny,
											   slot))
				continue;
			bool  isnull;
			Datum val = slot_getattr(slot, vec_attnum, &isnull);
			if (!isnull)
			{
				MktVector *vec = DatumGetMktVector(val);
				float	  *dst = vecs + (size_t)n * dim;
				memcpy(dst, vec->x, (size_t)dim * sizeof(float));
				if (metric == DISTANCE_COSINE)
				{
					float nrm = mkt_l2_norm(dst, dim);
					if (nrm > 0.0f)
						for (Dimension d = 0; d < dim; d++)
							dst[d] /= nrm;
				}
				n++;
			}
			ExecClearTuple(slot);
		}
		if (n == 0)
			continue;

		uint32_t kk = (uint32_t)K < n ? (uint32_t)K : n;

		/* FPS seed: member farthest from the cluster mean (a boundary
		 * point — the kind a near-but-orphaned query approaches). */
		memset(mean, 0, (size_t)dim * sizeof(float));
		for (uint32_t i = 0; i < n; i++)
			for (Dimension d = 0; d < dim; d++)
				mean[d] += vecs[(size_t)i * dim + d];
		for (Dimension d = 0; d < dim; d++)
			mean[d] /= (float)n;

		uint32_t s0	  = 0;
		float	 best = -1.0f;
		for (uint32_t i = 0; i < n; i++)
		{
			float dd = mkt_l2_distance_squared(vecs + (size_t)i * dim, mean, dim);
			if (dd > best)
			{
				best = dd;
				s0	 = i;
			}
		}
		sel[0] = s0;
		for (uint32_t i = 0; i < n; i++)
			mind[i] = mkt_l2_distance_squared(
					vecs + (size_t)i * dim, vecs + (size_t)s0 * dim, dim);

		/* Greedy max-min: each new entry point is the member farthest
		 * from all already-selected entry points. */
		for (uint32_t j = 1; j < kk; j++)
		{
			uint32_t nx	  = 0;
			float	 bb	  = -1.0f;
			for (uint32_t i = 0; i < n; i++)
				if (mind[i] > bb)
				{
					bb = mind[i];
					nx = i;
				}
			sel[j] = nx;
			for (uint32_t i = 0; i < n; i++)
			{
				float dd = mkt_l2_distance_squared(
						vecs + (size_t)i * dim, vecs + (size_t)nx * dim, dim);
				if (dd < mind[i])
					mind[i] = dd;
			}
		}

		for (uint32_t j = 0; j < kk; j++)
		{
			MktVector *sv = mkt_vector_create(dim);
			memcpy(sv->x, vecs + (size_t)sel[j] * dim,
				   (size_t)dim * sizeof(float));
			Datum values[3];
			bool  nulls[3] = {0};
			values[0]	   = Int32GetDatum((int32)c);
			values[1]	   = Int32GetDatum((int32)(j + 1));
			values[2]	   = PointerGetDatum(sv);
			tuplestore_putvalues(rsinfo->setResult, rsinfo->setDesc, values,
								 nulls);
		}
	}

	ExecDropSingleTupleTableSlot(slot);
	table_close(heap, AccessShareLock);
	relation_close(index, AccessShareLock);
	PG_RETURN_NULL();
}

/* ----------------------------------------------------------------
 * mkt.aniso_scanned(regclass, query, nprobe, alpha, max_members)
 *
 * Diagnostic for ANISOTROPIC (direction-aware) cluster routing. Point-
 * based ideas (sub-centroids, entry points) replaced the mean and died
 * to high-D false-near noise. This KEEPS the mean-similarity ranking and
 * adds a smooth aggregate correction crediting a cluster for how much its
 * member residuals SPREAD TOWARD the query:
 *
 *   score(c) = <q, c_hat> + alpha * sqrt( mean_i <q, v_i - c_hat>^2 )
 *              \__ mean __/   \__ RMS residual projection onto q ______/
 *
 * c_hat = unit mean of sampled members (cosine center); v_i = unit
 * members. Ranks clusters by score DESC, returns top-nprobe (rank,
 * cluster_id). alpha=0 == mean-only routing, so alpha=0 vs alpha>0
 * isolates the anisotropic term on identical data. Per-backend cache
 * (built once) amortizes member-gather across a query sweep. Read-only.
 * ---------------------------------------------------------------- */
typedef struct AnisoCache
{
	Oid		  relid;
	uint32_t  nlist;
	Dimension dim;
	uint32_t  M;	   /* members stored per cluster */
	float	 *cmean;   /* nlist*dim, unit-normalized sampled mean */
	float	 *members; /* nlist*M*dim, unit-normalized */
	uint16_t *mcount;
	bool	  valid;
} AnisoCache;

static AnisoCache aniso = {0};

static void
aniso_cache_build(Relation index, uint32_t M)
{
	Buffer meta_buf = ReadBuffer(index, 0);
	LockBuffer(meta_buf, BUFFER_LOCK_SHARE);
	const MktannMetaPage *meta =
			(const MktannMetaPage *)PageGetSpecialPointer(BufferGetPage(meta_buf));
	Dimension	   dim	  = (Dimension)meta->dim;
	uint32_t	   nlist  = meta->nlist;
	DistanceMetric metric = (DistanceMetric)meta->metric;
	UnlockReleaseBuffer(meta_buf);

	MemoryContext old = MemoryContextSwitchTo(TopMemoryContext);
	float	 *cmean	  = palloc0((size_t)nlist * dim * sizeof(float));
	float	 *members = MemoryContextAllocHuge(
			 TopMemoryContext, (size_t)nlist * M * dim * sizeof(float));
	uint16_t *mcount  = palloc0((size_t)nlist * sizeof(uint16_t));
	MemoryContextSwitchTo(old);

	/* Pass 1: gather up to M member tids per cluster. */
	ItemPointerData **tids = palloc0((size_t)nlist * sizeof(ItemPointerData *));
	BlockNumber		  nblocks = RelationGetNumberOfBlocks(index);
	for (BlockNumber blkno = 1; blkno < nblocks; blkno++)
	{
		Buffer buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		Page page = BufferGetPage(buf);
		if (PageGetSpecialSize(page) == MAXALIGN(sizeof(MktPostingPageOpaque)))
		{
			const MktPostingPageOpaque *op = mkt_posting_opaque(page);
			if (op->page_id == MKT_POSTING_PAGE_ID &&
				(op->flags & MKT_POSTING_PAGE_FASTSCAN) && op->cluster_id < nlist)
			{
				uint32_t c = op->cluster_id;
				if (mcount[c] < M)
				{
					char *content = (op->flags & MKT_POSTING_PAGE_FIRST)
										? mkt_posting_content_first(page, dim)
										: mkt_posting_content(page);
					uint32_t cnt	 = op->entry_count;
					uint32_t ngroups = (cnt + MKT_FASTSCAN_GROUP - 1) /
									   MKT_FASTSCAN_GROUP;
					for (uint32_t g = 0; g < ngroups && mcount[c] < M; g++)
					{
						ItemPointerData *gt =
								mkt_fastscan_group_tids(content, g, dim);
						uint32_t gc = cnt - g * MKT_FASTSCAN_GROUP;
						if (gc > MKT_FASTSCAN_GROUP)
							gc = MKT_FASTSCAN_GROUP;
						for (uint32_t v = 0; v < gc && mcount[c] < M; v++)
						{
							if (tids[c] == NULL)
								tids[c] = palloc((size_t)M *
												 sizeof(ItemPointerData));
							tids[c][mcount[c]++] = gt[v];
						}
					}
				}
			}
		}
		UnlockReleaseBuffer(buf);
	}

	/* Pass 2: fetch vectors, normalize, store members, accumulate mean. */
	Relation		heap	   = table_open(index->rd_index->indrelid,
											AccessShareLock);
	int				vec_attnum = index->rd_index->indkey.values[0];
	TupleTableSlot *slot	   = table_slot_create(heap, NULL);
	for (uint32_t c = 0; c < nlist; c++)
	{
		uint32_t got = 0;
		for (uint32_t i = 0; i < mcount[c]; i++)
		{
			if (!table_tuple_fetch_row_version(heap, &tids[c][i], SnapshotAny,
											   slot))
				continue;
			bool  isnull;
			Datum val = slot_getattr(slot, vec_attnum, &isnull);
			if (!isnull)
			{
				MktVector *vec = DatumGetMktVector(val);
				float	  *dst = members + ((size_t)c * M + got) * dim;
				memcpy(dst, vec->x, (size_t)dim * sizeof(float));
				if (metric == DISTANCE_COSINE)
				{
					float nrm = mkt_l2_norm(dst, dim);
					if (nrm > 0.0f)
						for (Dimension d = 0; d < dim; d++)
							dst[d] /= nrm;
				}
				float *cm = cmean + (size_t)c * dim;
				for (Dimension d = 0; d < dim; d++)
					cm[d] += dst[d];
				got++;
			}
			ExecClearTuple(slot);
		}
		mcount[c] = (uint16_t)got;
		if (got > 0)
		{
			float *cm  = cmean + (size_t)c * dim;
			float  nrm = mkt_l2_norm(cm, dim);
			if (nrm > 0.0f)
				for (Dimension d = 0; d < dim; d++)
					cm[d] /= nrm;
		}
	}
	ExecDropSingleTupleTableSlot(slot);
	table_close(heap, AccessShareLock);

	aniso.relid	  = RelationGetRelid(index);
	aniso.nlist	  = nlist;
	aniso.dim	  = dim;
	aniso.M		  = M;
	aniso.cmean	  = cmean;
	aniso.members = members;
	aniso.mcount  = mcount;
	aniso.valid	  = true;
}

typedef struct AnisoScore
{
	float	score;
	int32_t cid;
} AnisoScore;

static int
aniso_cmp_desc(const void *a, const void *b)
{
	float x = ((const AnisoScore *)a)->score;
	float y = ((const AnisoScore *)b)->score;
	return (x < y) - (x > y);
}

PG_FUNCTION_INFO_V1(mkt_aniso_scanned);

Datum
mkt_aniso_scanned(PG_FUNCTION_ARGS)
{
	Oid			   indexoid	   = PG_GETARG_OID(0);
	MktVector	  *q		   = DatumGetMktVector(PG_GETARG_DATUM(1));
	int32		   nprobe	   = PG_GETARG_INT32(2);
	float		   alpha	   = (float)PG_GETARG_FLOAT8(3);
	int32		   M		   = PG_GETARG_INT32(4);
	ReturnSetInfo *rsinfo	   = (ReturnSetInfo *)fcinfo->resultinfo;

	if (M < 1)
		M = 1;
	if (nprobe < 1)
		nprobe = 1;

	InitMaterializedSRF(fcinfo, 0);

	Relation index = relation_open(indexoid, AccessShareLock);

	if (!aniso.valid || aniso.relid != RelationGetRelid(index) ||
		aniso.M != (uint32_t)M)
	{
		/* (Re)build: drop stale cache, then build at the requested M. */
		if (aniso.cmean)
			pfree(aniso.cmean);
		if (aniso.members)
			pfree(aniso.members);
		if (aniso.mcount)
			pfree(aniso.mcount);
		aniso.cmean = aniso.members = NULL;
		aniso.mcount = NULL;
		aniso.valid	 = false;
		aniso_cache_build(index, (uint32_t)M);
	}

	Dimension dim	= aniso.dim;
	uint32_t  nlist = aniso.nlist;

	/* Normalize the query (cosine). */
	float *qn  = palloc((size_t)dim * sizeof(float));
	memcpy(qn, q->x, (size_t)dim * sizeof(float));
	float qnrm = mkt_l2_norm(qn, dim);
	if (qnrm > 0.0f)
		for (Dimension d = 0; d < dim; d++)
			qn[d] /= qnrm;

	AnisoScore *scores = palloc((size_t)nlist * sizeof(AnisoScore));
	uint32_t	nscored = 0;
	for (uint32_t c = 0; c < nlist; c++)
	{
		if (aniso.mcount[c] == 0)
			continue;
		float qc = mkt_dot_product(qn, aniso.cmean + (size_t)c * dim, dim);
		float sumsq = 0.0f;
		uint32_t mc = aniso.mcount[c];
		for (uint32_t i = 0; i < mc; i++)
		{
			float pi = mkt_dot_product(
					qn, aniso.members + ((size_t)c * aniso.M + i) * dim, dim);
			float dd = pi - qc;
			sumsq += dd * dd;
		}
		float var = sumsq / (float)mc;
		scores[nscored].score = qc + alpha * sqrtf(var);
		scores[nscored].cid	  = (int32)c;
		nscored++;
	}

	qsort(scores, nscored, sizeof(AnisoScore), aniso_cmp_desc);

	uint32_t topn = (uint32_t)nprobe < nscored ? (uint32_t)nprobe : nscored;
	for (uint32_t i = 0; i < topn; i++)
	{
		Datum values[2];
		bool  nulls[2] = {0};
		values[0]	   = Int32GetDatum((int32)(i + 1));
		values[1]	   = Int32GetDatum(scores[i].cid);
		tuplestore_putvalues(rsinfo->setResult, rsinfo->setDesc, values, nulls);
	}

	relation_close(index, AccessShareLock);
	PG_RETURN_NULL();
}
