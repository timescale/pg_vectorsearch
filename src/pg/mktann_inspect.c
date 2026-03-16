/*
 * mkt_pg_inspect.c - Index inspection functions
 *
 * Provides set-returning functions to inspect the internal structure
 * of mktann indexes via SQL, useful for debugging and visualization.
 */

#include <postgres.h>

#include <access/relation.h>
#include <funcapi.h>
#include <storage/bufmgr.h>
#include <utils/builtins.h>
#include <utils/rel.h>

#include "index/centroid_page.h"
#include "index/posting_page.h"
#include "mktann_meta.h"

PG_FUNCTION_INFO_V1(mkt_centroid_pages);
PG_FUNCTION_INFO_V1(mkt_posting_pages);

/* Format name lookup (indexed by MktCentroidFormat) */
static const char *centroid_format_names[] = {
		[MKT_CENTROID_FMT_RABITQ] = "rabitq",
		[MKT_CENTROID_FMT_FLOAT]  = "float",
		[MKT_CENTROID_FMT_HALF]	  = "half",
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

		/* Emit one row per entry */
		uint16_t nentries = opaque->entry_count;
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

			if (!is_leaf && BlockNumberIsValid(entry->child_blkno))
				values[4] = Int32GetDatum((int32)entry->child_blkno);
			else
				nulls[4] = true;

			values[5] = Int16GetDatum((int16)entry->child_count);
			values[6] = BoolGetDatum(is_leaf);

			tuplestore_putvalues(
					rsinfo->setResult, rsinfo->setDesc, values, nulls);
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

		/* Enqueue children from entry metadata */
		for (uint16_t i = 0; i < nentries; i++)
		{
			const MktCentroidEntryMeta *entry = mkt_centroid_meta(page, i);

			if (!(entry->flags & MKT_CENTROID_FLAG_LEAF) &&
				BlockNumberIsValid(entry->child_blkno))
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

		UnlockReleaseBuffer(buf);
	}

	pfree(worklist);
	relation_close(index, AccessShareLock);

	PG_RETURN_NULL();
}

/*
 * mkt_posting_pages(regclass)
 *
 * Returns one row per posting page: cluster_id, page_seq (0-based
 * position in chain), blkno, entry_count. Walks all posting chains
 * found via leaf centroid entries.
 */
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

	/* Read metapage */
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
	UnlockReleaseBuffer(meta_buf);

	if (!BlockNumberIsValid(first_centroid))
	{
		relation_close(index, AccessShareLock);
		PG_RETURN_NULL();
	}

	/*
	 * BFS over centroid pages to find leaf entries (posting heads).
	 * Then walk each posting chain.
	 */
	int			 wl_cap	 = 64;
	int			 wl_len	 = 0;
	int			 wl_head = 0;
	BlockNumber *wl		 = palloc(wl_cap * sizeof(BlockNumber));

	wl[wl_len++] = first_centroid;

	uint32_t cluster_id = 0;

	while (wl_head < wl_len)
	{
		BlockNumber cblkno = wl[wl_head++];

		Buffer buf = ReadBuffer(index, cblkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		Page page = BufferGetPage(buf);

		const MktCentroidPageOpaque *opaque	  = MKT_CENTROID_OPAQUE(page);
		uint16_t					 nentries = opaque->entry_count;

		for (uint16_t i = 0; i < nentries; i++)
		{
			const MktCentroidEntryMeta *entry = mkt_centroid_meta(page, i);
			bool is_leaf = (entry->flags & MKT_CENTROID_FLAG_LEAF) != 0;

			if (is_leaf && BlockNumberIsValid(entry->child_blkno))
			{
				/* Walk posting chain */
				BlockNumber pblkno	 = entry->child_blkno;
				int32		page_seq = 0;

				while (BlockNumberIsValid(pblkno))
				{
					Buffer pbuf = ReadBuffer(index, pblkno);
					LockBuffer(pbuf, BUFFER_LOCK_SHARE);
					Page ppage = BufferGetPage(pbuf);

					const MktPostingPageOpaque *pop = MKT_POSTING_OPAQUE(
							ppage);

					Datum values[4];
					bool  nulls[4] = {0};

					values[0] = Int32GetDatum((int32)cluster_id);
					values[1] = Int32GetDatum(page_seq);
					values[2] = Int32GetDatum((int32)pblkno);
					values[3] = Int32GetDatum((int32)pop->entry_count);

					tuplestore_putvalues(
							rsinfo->setResult, rsinfo->setDesc, values, nulls);

					pblkno = pop->next_blkno;
					page_seq++;

					UnlockReleaseBuffer(pbuf);
				}

				cluster_id++;
			}
			else if (!is_leaf && BlockNumberIsValid(entry->child_blkno))
			{
				if (wl_len >= wl_cap)
				{
					wl_cap *= 2;
					wl = repalloc(wl, wl_cap * sizeof(BlockNumber));
				}
				wl[wl_len++] = entry->child_blkno;
			}
		}

		/* Follow centroid page chain */
		if (BlockNumberIsValid(opaque->next_blkno))
		{
			if (wl_len >= wl_cap)
			{
				wl_cap *= 2;
				wl = repalloc(wl, wl_cap * sizeof(BlockNumber));
			}
			wl[wl_len++] = opaque->next_blkno;
		}

		UnlockReleaseBuffer(buf);
	}

	pfree(wl);
	relation_close(index, AccessShareLock);

	PG_RETURN_NULL();
}
