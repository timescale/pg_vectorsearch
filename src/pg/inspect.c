/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * inspect.c - read-only index inspection functions
 *
 * Set-returning functions that expose the internal structure of prism
 * indexes via SQL. All are read-only (AccessShareLock, no WAL); the mutating
 * maintenance operations live in maintenance.c. The centroid-tree leaf walk
 * they share, collect_leaf_entries, is declared in inspect.h.
 *
 * Functions:
 *   prism_centroid_pages(regclass) -- centroid tree structure
 *   prism_posting_pages(regclass)  -- posting list page chains
 *   prism_tids_clusters(regclass, tid[]) -- which cluster(s) hold each TID
 *   prism_index_settings(regclass) -- effective (resolved) settings
 */

#include <postgres.h>

#include <access/relation.h>
#include <catalog/index.h>
#include <catalog/pg_class.h>
#include <catalog/pg_type.h>
#include <funcapi.h>
#include <miscadmin.h>
#include <nodes/parsenodes.h>
#include <storage/bufmgr.h>
#include <utils/acl.h>
#include <utils/array.h>
#include <utils/builtins.h>
#include <utils/lsyscache.h>
#include <utils/rel.h>
#include <utils/syscache.h>
#include <utils/tuplestore.h>

#include "index/centroid_page.h"
#include "index/index_base.h"
#include "index/index_build.h"
#include "index/posting_page.h"
#include "index/query_scan.h"
#include "inspect.h"
#include "meta.h"
#include "pg/bufstorage.h"
#include "support_pg.h"

PG_FUNCTION_INFO_V1(vs_centroid_pages);
PG_FUNCTION_INFO_V1(vs_posting_pages);
PG_FUNCTION_INFO_V1(vs_tids_clusters);
PG_FUNCTION_INFO_V1(vs_index_settings);

/*
 * The functions below iterate an on-disk entry_count read straight from a
 * page. Reject a count past the format's real per-page capacity before
 * looping so a corrupt or truncated page can't drive a read past the page.
 * Posting pages get the same treatment from prism_posting_check_count.
 */
static void
check_centroid_count(
		BlockNumber			blkno,
		uint32_t			count,
		Dimension			dim,
		PrismCentroidFormat fmt)
{
	uint32_t max_entries = prism_centroid_max_entries_fmt(dim, fmt);
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

/*
 * Authorization for the inspection functions. They expose an index's
 * internal structure (block numbers, cluster ids, tombstone state), so
 * gate them on the caller's privileges on the *table* the index belongs
 * to -- the same model PostgreSQL's pgrowlocks/pgstattuple use for
 * relation inspection. EXECUTE stays granted to PUBLIC; these runtime
 * checks do the per-object authorization that a static GRANT cannot
 * express for a regclass argument. Superusers pass automatically.
 *
 * Read-only inspectors require SELECT on the table (its owner has that, so an
 * owner can always inspect their own index). prism_tids_clusters is the
 * exception: it answers which rows live where, so it requires ownership.
 * The mutating maintenance functions check ownership separately.
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

/* Same ownership gate as the maintenance procedures. A SELECT grant is not
 * enough: the function confirms which TIDs the index holds. */
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

/*
 * Grow a doubling array before an append: when len has reached *cap, double
 * *cap and repalloc. Returns the (possibly moved) base pointer; pass
 * sizeof(*arr) as elem_size. A no-op with room to spare, so it is safe to call
 * unconditionally before every append.
 */
static void *
grow_if_full(void *arr, int len, int *cap, Size elem_size)
{
	if (len < *cap)
		return arr;
	*cap *= 2;
	return repalloc(arr, (Size)*cap * elem_size);
}

/*
 * BFS the centroid tree and collect all leaf entries. Returns the count and
 * fills *out (palloc'd array). Caller must pfree. Declared in inspect.h and
 * shared with the maintenance functions; LeafEntry is defined there too.
 */
int
collect_leaf_entries(
		Relation	index,
		BlockNumber first_centroid,
		uint8_t		nlevels,
		Dimension	dim,
		LeafEntry **out)
{
	VsPgStorage store;
	VsStorage  *st = &store.base;

	vs_pg_storage_init_inspect(&store, index);

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

		Page page = vs_storage_read_page(st, blkno);

		const PrismCentroidPageOpaque *opaque	= PRISM_CENTROID_OPAQUE(page);
		uint16_t					   nentries = opaque->entry_count;
		PrismCentroidFormat			   fmt =
				(PrismCentroidFormat)(opaque->flags & PRISM_CENTROID_FMT_MASK);
		bool is_leaf_page = (opaque->level == nlevels - 1);

		check_centroid_count(blkno, nentries, dim, fmt);

		if (BlockNumberIsValid(opaque->next_blkno))
		{
			wl			 = grow_if_full(wl, wl_len, &wl_cap, sizeof(*wl));
			wl[wl_len++] = opaque->next_blkno;
		}

		for (uint16_t i = 0; i < nentries; i++)
		{
			BlockNumber child;
			bool		is_leaf;

			if (fmt == PRISM_CENTROID_FMT_FASTSCAN)
			{
				/* No per-entry meta: read the child from the packed
				 * group array and take leaf status from the page level. */
				char		*content = (char *)PageGetContents(page);
				uint32_t	 g		 = i / VS_FASTSCAN_GROUP;
				uint32_t	 slot	 = i % VS_FASTSCAN_GROUP;
				BlockNumber *grp =
						prism_centroid_fastscan_group_child(content, g, dim);
				child	= grp[slot];
				is_leaf = is_leaf_page;
			}
			else
			{
				const PrismCentroidEntryMeta *entry =
						prism_centroid_meta(page, i);
				child	= entry->child_blkno;
				is_leaf = (entry->flags & PRISM_CENTROID_FLAG_LEAF) != 0;
			}

			if (!BlockNumberIsValid(child))
				continue;

			if (is_leaf)
			{
				leaves = grow_if_full(
						leaves, leaves_len, &leaves_cap, sizeof(*leaves));
				leaves[leaves_len++] = (LeafEntry){
						.posting_head  = child,
						.centroid_page = blkno,
						.entry_idx	   = i,
				};
			}
			else
			{
				wl			 = grow_if_full(wl, wl_len, &wl_cap, sizeof(*wl));
				wl[wl_len++] = child;
			}
		}

		vs_storage_release_page(st, blkno);
	}

	pfree(wl);
	*out = leaves;
	return leaves_len;
}

/* Format name lookup (indexed by PrismCentroidFormat) */
static const char *centroid_format_names[] = {
		[PRISM_CENTROID_FMT_RABITQ]	  = "rabitq",
		[PRISM_CENTROID_FMT_FLOAT]	  = "float",
		[PRISM_CENTROID_FMT_HALF]	  = "half",
		[PRISM_CENTROID_FMT_FASTSCAN] = "fastscan",
};

/*
 * vs_centroid_pages(regclass)
 *
 * Returns one row per centroid entry: blkno, entry, level, format,
 * child_blkno, child_count, is_leaf. Traverses the tree via BFS
 * starting from the metapage's first_centroid, following next_blkno
 * chains and child_blkno links.
 */
Datum
vs_centroid_pages(PG_FUNCTION_ARGS)
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

	VsPgStorage store;
	VsStorage  *st = &store.base;

	vs_pg_storage_init_inspect(&store, index);

	/* Read metapage and verify magic */
	Page meta_page = vs_storage_read_page(st, 0);

	const PrismMetaPage *meta = (const PrismMetaPage *)PageGetSpecialPointer(
			meta_page);

	if (meta->magic != PRISM_META_MAGIC)
	{
		vs_storage_release_page(st, 0);
		relation_close(index, AccessShareLock);
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("\"%s\" is not a prism index",
						RelationGetRelationName(index))));
	}

	BlockNumber first_centroid = meta->first_centroid;
	Dimension	dim			   = (Dimension)meta->dim;
	uint8_t		nlevels		   = meta->nlevels;

	vs_storage_release_page(st, 0);

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

		Page page = vs_storage_read_page(st, blkno);

		const PrismCentroidPageOpaque *opaque = PRISM_CENTROID_OPAQUE(page);
		PrismCentroidFormat			   fmt =
				(PrismCentroidFormat)(opaque->flags & PRISM_CENTROID_FMT_MASK);
		/* FASTSCAN doesn't have per-entry flags, so determine leaf
		 * status from tree depth: bottom level holds posting heads. */
		bool is_leaf_page = (opaque->level == nlevels - 1);

		/* Emit one row per entry. FASTSCAN pages have a different
		 * layout (group section instead of per-entry meta + data),
		 * so child_blkno lives in the group array and per-entry
		 * flags don't exist — the leaf bit is page-level. */
		uint16_t nentries = opaque->entry_count;

		check_centroid_count(blkno, nentries, dim, fmt);

		if (fmt == PRISM_CENTROID_FMT_FASTSCAN)
		{
			char *content = (char *)PageGetContents(page);

			for (uint16_t i = 0; i < nentries; i++)
			{
				uint32_t	 g	  = i / VS_FASTSCAN_GROUP;
				uint32_t	 slot = i % VS_FASTSCAN_GROUP;
				BlockNumber *child =
						prism_centroid_fastscan_group_child(content, g, dim);

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
				const PrismCentroidEntryMeta *entry =
						prism_centroid_meta(page, i);
				bool is_leaf = (entry->flags & PRISM_CENTROID_FLAG_LEAF) != 0;

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

		vs_storage_release_page(st, blkno);
	}

	pfree(worklist);
	relation_close(index, AccessShareLock);

	PG_RETURN_NULL();
}

/*
 * One prism_posting_pages() row per page of a chain.
 *
 * Everything the rows need beyond the page itself -- the tuplestore to emit
 * into, the dimension, the running position in the chain -- rides in the
 * walk's caller state, so the walk itself carries nothing about what it is
 * being walked for.
 */
typedef struct PostingRowCtx
{
	ReturnSetInfo *rsinfo;
	Dimension	   dim;
	int			   chain_pos;
} PostingRowCtx;

static bool
emit_posting_row(PrismPostingChainPos *pos, void *state)
{
	PostingRowCtx				 *ctx = state;
	Dimension					  dim = ctx->dim;
	const PrismPostingPageOpaque *op  = prism_posting_opaque(pos->page);
	bool is_first	 = (op->flags & PRISM_POSTING_PAGE_FIRST) != 0;
	bool tombstoned	 = (op->flags & PRISM_POSTING_PAGE_TOMBSTONED) != 0;
	bool is_fastscan = (op->flags & PRISM_POSTING_PAGE_FASTSCAN) != 0;

	Datum values[10];
	bool  nulls[10] = {0};

	values[0] = Int32GetDatum((int32)pos->blkno);
	values[1] = Int32GetDatum((int32)op->cluster_id);
	values[2] = BoolGetDatum(is_first);
	values[3] = BoolGetDatum(tombstoned);
	values[4] = Int32GetDatum((int32)op->entry_count);

	if (!is_fastscan)
	{
		/* AoS entries carry a per-entry DELETED flag; count them. FASTSCAN
		 * packs entries into SIMD groups with no per-entry state (deletion
		 * is page-granular there), so dead_count is NULL for fastscan
		 * pages. */
		char *content = prism_posting_page_content(pos->page, dim);
		int32 dead	  = 0;

		prism_posting_check_count(pos->blkno, op, dim);
		for (uint32_t i = 0; i < op->entry_count; i++)
		{
			const PrismPostingEntryHeader *h =
					prism_posting_entry_at(content, i, dim);

			if (h->meta.flags & PRISM_POSTING_FLAG_DELETED)
				dead++;
		}
		values[5] = Int32GetDatum(dead);
	}
	else
		nulls[5] = true;

	values[6] = Int32GetDatum((int32)op->max_entries);

	if (BlockNumberIsValid(pos->next))
		values[7] = Int32GetDatum((int32)pos->next);
	else
		nulls[7] = true;

	values[8] = Int32GetDatum(ctx->chain_pos++);
	values[9] = CStringGetTextDatum(is_fastscan ? "fastscan" : "aos");

	tuplestore_putvalues(
			ctx->rsinfo->setResult, ctx->rsinfo->setDesc, values, nulls);
	return true;
}

/* ----------------------------------------------------------------
 * prism_posting_pages(regclass)
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
vs_posting_pages(PG_FUNCTION_ARGS)
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

	VsPgStorage store;
	VsStorage  *st = &store.base;

	vs_pg_storage_init_inspect(&store, index);

	Page meta_page = vs_storage_read_page(st, 0);

	const PrismMetaPage *meta = (const PrismMetaPage *)PageGetSpecialPointer(
			meta_page);

	if (meta->magic != PRISM_META_MAGIC)
	{
		vs_storage_release_page(st, 0);
		relation_close(index, AccessShareLock);
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("\"%s\" is not a prism index",
						RelationGetRelationName(index))));
	}

	BlockNumber first_centroid = meta->first_centroid;
	Dimension	dim			   = meta->dim;
	uint8_t		nlevels		   = meta->nlevels;
	vs_storage_release_page(st, 0);

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
		PostingRowCtx ctx = {.rsinfo = rsinfo, .dim = dim};

		prism_posting_chain_walk(
				st, leaves[c].posting_head, emit_posting_row, &ctx);
	}

	pfree(leaves);
	relation_close(index, AccessShareLock);

	PG_RETURN_NULL();
}

/* ----------------------------------------------------------------
 * prism_tids_clusters(regclass, tid[])
 *
 * Diagnostic: for each input heap TID, return which cluster(s) it is
 * stored in (primary + any SOAR/boundary replica). One pass over all
 * leaf posting lists; emits (tid, cluster_id) for matched TIDs only.
 * Lets a caller compare "where the true nearest neighbors live" against
 * "which clusters a query scans". Owner-only: a SELECT grant must not be
 * enough to confirm which TIDs the index holds.
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
	uint64 enc = prism_posting_encode_tid(tid);
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
vs_tids_clusters(PG_FUNCTION_ARGS)
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
		keys[nk++] = prism_posting_encode_tid(
				(ItemPointer)DatumGetPointer(elems[i]));
	}
	qsort(keys, nk, sizeof(uint64), cmp_u64);

	Relation index = relation_open(indexoid, AccessShareLock);

	require_index_owner(index, AccessShareLock);

	VsPgStorage store;
	VsStorage  *st = &store.base;

	vs_pg_storage_init_inspect(&store, index);

	/*
	 * Through a local: PageGetSpecialPointer is a macro that evaluates its
	 * argument three times, and reading a page is not free of side effects
	 * -- inlining the read pins the buffer once per evaluation.
	 */
	Page				 meta_page = vs_storage_read_page(st, 0);
	const PrismMetaPage *meta = (const PrismMetaPage *)PageGetSpecialPointer(
			meta_page);
	Dimension dim = (Dimension)meta->dim;
	vs_storage_release_page(st, 0);

	/* Scan every block and pick out posting pages directly (identified by
	 * page_id), rather than walking the centroid tree to find posting
	 * heads — that avoids any dependency on the leaf-enumeration path and
	 * each posting page already carries its cluster_id. */
	BlockNumber nblocks = RelationGetNumberOfBlocks(index);
	for (BlockNumber blkno = 1; nk > 0 && blkno < nblocks; blkno++)
	{
		Page page = vs_storage_read_page(st, blkno);

		if (prism_page_is_posting(page))
		{
			const PrismPostingPageOpaque *op = prism_posting_opaque(page);
			char	*content = prism_posting_page_content(page, dim);
			uint32_t count	 = op->entry_count;

			prism_posting_check_count(blkno, op, dim);

			if (op->flags & PRISM_POSTING_PAGE_FASTSCAN)
			{
				/* SoA: TIDs packed per group of VS_FASTSCAN_GROUP. */
				uint32_t ngroups = (count + VS_FASTSCAN_GROUP - 1) /
								   VS_FASTSCAN_GROUP;
				for (uint32_t g = 0; g < ngroups; g++)
				{
					ItemPointerData *tids =
							prism_fastscan_group_tids(content, g, dim);
					uint32_t gc = count - g * VS_FASTSCAN_GROUP;
					if (gc > VS_FASTSCAN_GROUP)
						gc = VS_FASTSCAN_GROUP;
					for (uint32_t v = 0; v < gc; v++)
						emit_tid_cluster(
								rsinfo, keys, nk, &tids[v], op->cluster_id);
				}
			}
			else
			{
				/* AoS: the TID is at the head of each entry. */
				for (uint32_t i = 0; i < count; i++)
				{
					PrismPostingEntryHeader *e =
							prism_posting_entry_at(content, i, dim);
					emit_tid_cluster(
							rsinfo, keys, nk, &e->meta.tid, op->cluster_id);
				}
			}
		}
		vs_storage_release_page(st, blkno);
	}

	relation_close(index, AccessShareLock);
	PG_RETURN_NULL();
}

/* ----------------------------------------------------------------
 * prism_index_settings(regclass)
 *
 * Returns one (name, setting, source) row per effective index
 * setting, with automatic values resolved to what the build (or the
 * current session) actually uses. The source column tells where the
 * value came from:
 *
 *   'option'  -- reloption set explicitly at CREATE INDEX
 *   'auto'    -- resolved from an automatic default (nlist from the
 *                row count, formats from metric/dimension, nprobe
 *                from nlist)
 *   'default' -- reloption default in effect
 *   'column'  -- from the indexed column definition
 *   'opclass' -- from the operator class
 *   'derived' -- computed from other settings
 *   'session' -- a session GUC overriding the index setting
 *
 * Values the build persists (nlist, fan_out, nlevels, formats) come
 * from the metadata page and are authoritative for the index as
 * built. Options only consumed during the build but not persisted
 * (soar_lambda, boundary_epsilon, kmeans_nredo) are read from the
 * catalog, so they reflect the build only as long as they have not
 * been changed with ALTER INDEX ... SET afterwards.
 * ---------------------------------------------------------------- */

/* Metric name lookup (indexed by DistanceMetric); the names match
 * the vector_<metric>_ops opclass names. */
static const char *metric_names[] = {
		[DISTANCE_L2]			 = "l2",
		[DISTANCE_INNER_PRODUCT] = "ip",
		[DISTANCE_COSINE]		 = "cosine",
};

/* The reloptions explicitly set on a relation, as a list of DefElem.
 * Unlike rd_options this excludes defaults, so it tells apart "set
 * to the default value" from "defaulted". Caller must free with
 * list_free_deep. */
static List *
explicit_reloptions(Oid relid)
{
	HeapTuple tuple = SearchSysCache1(RELOID, ObjectIdGetDatum(relid));
	if (!HeapTupleIsValid(tuple))
		elog(ERROR, "cache lookup failed for relation %u", relid);

	bool  isnull;
	Datum datum =
			SysCacheGetAttr(RELOID, tuple, Anum_pg_class_reloptions, &isnull);
	List *options = isnull ? NIL : untransformRelOptions(datum);
	ReleaseSysCache(tuple);
	return options;
}

static bool
reloption_is_set(const List *options, const char *name)
{
	ListCell *lc;
	foreach (lc, options)
	{
		const DefElem *def = lfirst_node(DefElem, lc);
		if (strcmp(def->defname, name) == 0)
			return true;
	}
	return false;
}

static void
settings_row(
		ReturnSetInfo *rsinfo,
		const char	  *name,
		const char	  *setting,
		const char	  *source)
{
	Datum values[3];
	bool  nulls[3] = {0};

	values[0] = CStringGetTextDatum(name);
	values[1] = CStringGetTextDatum(setting);
	values[2] = CStringGetTextDatum(source);
	tuplestore_putvalues(rsinfo->setResult, rsinfo->setDesc, values, nulls);
}

Datum
vs_index_settings(PG_FUNCTION_ARGS)
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

	VsPgStorage store;
	VsStorage  *st = &store.base;

	vs_pg_storage_init_inspect(&store, index);

	/*
	 * Through a local: PageGetSpecialPointer is a macro that evaluates its
	 * argument three times, and reading a page is not free of side effects
	 * -- inlining the read pins the buffer once per evaluation.
	 */
	Page				 meta_page = vs_storage_read_page(st, 0);
	const PrismMetaPage *meta = (const PrismMetaPage *)PageGetSpecialPointer(
			meta_page);

	if (meta->magic != PRISM_META_MAGIC)
	{
		vs_storage_release_page(st, 0);
		relation_close(index, AccessShareLock);
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("\"%s\" is not a prism index",
						RelationGetRelationName(index))));
	}

	Dimension			dim				= meta->dim;
	uint8_t				nlevels			= meta->nlevels;
	PrismCentroidFormat centroid_format = (PrismCentroidFormat)
												  meta->centroid_format;
	uint32_t	   nlist	= meta->nlist;
	DistanceMetric metric	= (DistanceMetric)meta->metric;
	uint8_t		   fan_out	= meta->fan_out;
	bool		   fastscan = (meta->flags & PRISM_META_FLAG_FASTSCAN) != 0;
	uint32_t	   ncentroid_pages = meta->ncentroid_pages;
	BlockNumber	   first_posting   = meta->first_posting;
	vs_storage_release_page(st, 0);

	const PrismOptions *opts = (const PrismOptions *)index->rd_options;
	List			   *set	 = explicit_reloptions(indexoid);

	/* Index definition */
	settings_row(rsinfo, "dim", psprintf("%d", dim), "column");
	settings_row(rsinfo, "metric", metric_names[metric], "opclass");

	/* Build-resolved shape, from the metadata page */
	settings_row(
			rsinfo,
			"nlist",
			psprintf("%u", nlist),
			reloption_is_set(set, "nlist") ? "option" : "auto");
	settings_row(
			rsinfo,
			"fan_out",
			psprintf("%u", (uint32_t)fan_out),
			reloption_is_set(set, "fan_out") ? "option" : "auto");
	settings_row(
			rsinfo, "nlevels", psprintf("%u", (uint32_t)nlevels), "derived");
	/* The resting list size nlist was derived from, and the page target
	 * behind it -- resolved, so the effective default shows rather than the
	 * 0 that means "derive". */
	settings_row(
			rsinfo,
			"target_pages",
			psprintf(
					"%u",
					(opts != NULL && opts->target_pages > 0)
							? (uint32_t)opts->target_pages
							: (uint32_t)PRISM_DEFAULT_TARGET_PAGES),
			reloption_is_set(set, "target_pages") ? "option" : "default");
	settings_row(
			rsinfo,
			"target_entries",
			psprintf(
					"%u",
					prism_target_entries_per_dim(
							dim,
							(opts != NULL) ? (uint32_t)opts->target_pages
										   : 0)),
			"derived");
	settings_row(
			rsinfo,
			"centroid_pages",
			psprintf("%u", ncentroid_pages),
			"maintained");
	settings_row(
			rsinfo, "first_posting", psprintf("%u", first_posting), "derived");
	/*
	 * The posting page count the scan cost estimate prices, from the same
	 * helper it calls and the same page count: get_relation_info fills
	 * IndexOptInfo.pages from RelationGetNumberOfBlocks too, so this and the
	 * planner's figure agree without waiting on ANALYZE.
	 */
	settings_row(
			rsinfo,
			"posting_pages",
			psprintf(
					"%.0f",
					prism_index_posting_pages(
							(double)RelationGetNumberOfBlocks(index),
							(double)ncentroid_pages)),
			"derived");
	settings_row(
			rsinfo,
			"centroid_format",
			centroid_format_names[centroid_format],
			(reloption_is_set(set, "centroid_compression") ||
			 reloption_is_set(set, "centroid_fastscan"))
					? "option"
					: "auto");
	settings_row(
			rsinfo,
			"fastscan",
			fastscan ? "on" : "off",
			reloption_is_set(set, "fastscan") ? "option" : "auto");

	/* Build options not persisted in the index: current catalog
	 * values, with reloption defaults filled in */
	settings_row(
			rsinfo,
			"soar_lambda",
			psprintf(
					"%g",
					(opts != NULL) ? opts->soar_lambda
								   : PRISM_DEFAULT_SOAR_LAMBDA),
			reloption_is_set(set, "soar_lambda") ? "option" : "default");
	settings_row(
			rsinfo,
			"boundary_epsilon",
			psprintf(
					"%g",
					(opts != NULL) ? opts->boundary_epsilon
								   : PRISM_DEFAULT_BOUNDARY_EPSILON),
			reloption_is_set(set, "boundary_epsilon") ? "option" : "default");
	settings_row(
			rsinfo,
			"kmeans_nredo",
			psprintf("%d", (opts != NULL) ? opts->kmeans_nredo : 1),
			reloption_is_set(set, "kmeans_nredo") ? "option" : "default");

	/* Query-time settings whose effective value depends on this
	 * index: the session GUC wins when set, otherwise the index
	 * option or automatic resolution applies. */
	if (prism_distance_mode != VS_DISTANCE_MODE_DEFAULT)
		settings_row(
				rsinfo,
				"distance_mode",
				vs_distance_mode_name((VsDistanceMode)prism_distance_mode),
				"session");
	else
		settings_row(
				rsinfo,
				"distance_mode",
				vs_distance_mode_name(PrismGetDistanceMode(index)),
				reloption_is_set(set, "distance_mode") ? "option" : "default");
	settings_row(
			rsinfo,
			"nprobe",
			psprintf(
					"%u",
					(prism_nprobe > 0) ? (uint32_t)prism_nprobe
									   : prism_auto_nprobe(nlist)),
			(prism_nprobe > 0) ? "session" : "auto");

	list_free_deep(set);
	relation_close(index, AccessShareLock);
	PG_RETURN_NULL();
}
