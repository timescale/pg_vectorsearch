/*
 * mktann.c - meerkat ANN index access method handler
 *
 * Registers the mktann index access method with PostgreSQL. Build and
 * scan callbacks delegate to mktann_build.c and mktann_scan.c; trivial
 * stubs for unimplemented callbacks remain here.
 */

#include <postgres.h>

#include <access/amapi.h>
#include <access/generic_xlog.h>
#include <access/reloptions.h>
#include <access/relscan.h>
#include <commands/vacuum.h>
#include <fmgr.h>
#include <storage/bufmgr.h>
#include <storage/lmgr.h>
#include <utils/float.h>
#include <utils/injection_point.h>
#include <utils/inval.h>
#include <utils/memutils.h>
#include <utils/selfuncs.h>

#include "algo/vecops.h"
#include "index/index_base.h"
#include "index/posting_insert.h"
#include "index/query_scan.h"
#include "mktann_build.h"
#include "mktann_cache.h"
#include "mktann_meta.h"
#include "mktann_scan.h"
#include "mktann_storage.h"
#include "quant/rabitq.h"
#include "support_pg.h"
#include "typeinfo.h"
#include "types/vector.h"

PG_FUNCTION_INFO_V1(mktann_handler);

/* ----------------------------------------------------------------
 * Trivial stubs (no separate file needed)
 * ---------------------------------------------------------------- */

/*
 * ambuildempty populates the init fork of an unlogged index so that crash
 * recovery has a valid image to copy over the main fork. meerkat has no
 * notion of a valid "empty" index image -- even a build over zero rows
 * produces a real single-cluster tree -- so a no-op here would leave the
 * init fork at zero blocks and let recovery wipe the whole index, metadata
 * page included. Rather than seed the init fork with an index shape nothing
 * else ever exercises, refuse unlogged tables outright at CREATE INDEX time.
 */
static void
mktann_buildempty(Relation index)
{
	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("mktann indexes do not support unlogged tables"),
			 errhint("Use a logged table, or run ALTER TABLE ... SET LOGGED "
					 "before creating the index.")));
}

/*
 * Beam width used when routing an inserted vector to its nearest leaf. Wide
 * enough that multi-level tree descent lands the true nearest leaf; we still
 * insert into a single list (results[0]). Phase 0 does no SOAR replication on
 * insert — that's restored in bulk at rebuild / compaction.
 */
#define MKT_INSERT_ROUTE_BEAM 8

static bool
mktann_insert(
		Relation		  index,
		Datum			 *values,
		bool			 *isnull,
		ItemPointer		  heap_tid,
		Relation		  heap,
		IndexUniqueCheck  check_unique,
		bool			  index_unchanged,
		struct IndexInfo *index_info)
{
	(void)heap;
	(void)check_unique;
	(void)index_info;

	/* index_unchanged is deliberately ignored. PostgreSQL sets it when an
	 * UPDATE left the indexed column untouched but still had to place the
	 * new tuple version elsewhere (a non-HOT update); it is a hint for
	 * access methods that can deduplicate against the old version, not a
	 * signal that the old entry still covers the new TID. A HOT update,
	 * where the old entry does still apply, never reaches aminsert at all.
	 * Skipping the insert here left the new version unreachable through
	 * the index. NULL vectors get no entry. */
	if (isnull[0])
		return false;

	/* Per-insert scratch context: beam-search + encode allocations are freed
	 * in one shot and don't accumulate in the inserting transaction. */
	MemoryContext insert_ctx = AllocSetContextCreate(
			CurrentMemoryContext, "mktann insert", ALLOCSET_DEFAULT_SIZES);
	MemoryContext old_ctx = MemoryContextSwitchTo(insert_ctx);

	/* Immutable index parameters from the per-backend cache (metapage read at
	 * most once per backend, not once per insert). The checkout is
	 * registered with the current resource owner; capture it for the
	 * release below (and for the error paths in between, which release
	 * through the owner instead). */
	ResourceOwner params_owner = CurrentResourceOwner;
	MktIndexBase  base;
	mktann_index_base_init(index, &base);
	Dimension dim = base.dim;

	MktannStorage storage;
	mktann_storage_init(&storage, index, NULL, base.metric);
	base.centroid_storage = &storage.base;
	base.posting_storage  = &storage.base;
	base.page_base		  = NULL;

	/* Inserted vector, converted to float32 when the column type is not. */
	MktVectorAccess input = mkt_vector_access(
			mktann_cache_type_info(index), dim, CurrentMemoryContext);
	VectorRef vref = mkt_vector_read(&input, values[0]);

	if (vref.dim != dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("inserted vector dimension %u does not match index "
						"dimension %u",
						vref.dim,
						dim)));

	/* No defined distance under cosine: encoded as unreachable below.
	 * Squared norm -- zero iff the norm is zero, without the sqrt. */
	bool degenerate = base.metric == DISTANCE_COSINE &&
					  mkt_l2_norm_squared(vref.data, dim) == 0.0f;

	/*
	 * Route to the nearest leaf the same way a query does. mkt_query_route
	 * normalizes (cosine) + rotates into qs.pt_query and runs the beam search;
	 * qs.pt_query is then exactly the rotated residual base the encode needs.
	 */
	MktQueryState qs;
	mkt_query_state_init(&qs, &base, 1, MKT_INSERT_ROUTE_BEAM);
	uint32_t n = mkt_query_route(
			&qs,
			vref.data,
			MKT_INSERT_ROUTE_BEAM,
			MKT_DISTANCE_MODE_ASYMMETRIC,
			NULL);

	BlockNumber head = (n > 0) ? qs.beam_results[0].posting_head
							   : InvalidBlockNumber;
	if (head != InvalidBlockNumber)
	{
		RaBitQScratch enc;
		mkt_rabitq_scratch_init(&enc, dim);

		/*
		 * Serialize concurrent inserts into this cluster with a heavyweight
		 * page lock on the head — distinct from the buffer content locks the
		 * insert primitive takes per page, so it doesn't fight the
		 * single-buffer storage model. Released here, not held to xact end.
		 */
		LockPage(index, head, ExclusiveLock);
		/*
		 * Test hook: fires while this insert holds the per-cluster page lock,
		 * so an isolation test can pause here and observe a second insert into
		 * the same cluster block on the lock (proving the serialization and
		 * that it does not deadlock). No-op unless PG was built with injection
		 * points and a test attached an action.
		 */
		INJECTION_POINT("mktann-insert-locked", NULL);
		mkt_posting_insert_one(
				&storage.base,
				base.params,
				dim,
				head,
				*heap_tid,
				qs.pt_query,
				&enc,
				degenerate);
		UnlockPage(index, head, ExclusiveLock);
	}

	MemoryContextSwitchTo(old_ctx);
	MemoryContextDelete(insert_ctx);

	/* Check the RaBitQParams checkout back in — see mktann_index_base_init
	 * above. */
	mktann_release_params(dim, base.rabitq_seed, params_owner);

	/* bool result is only meaningful for unique indexes. */
	return false;
}

/*
 * Adapt PostgreSQL's IndexBulkDeleteCallback (takes ItemPointer) to the shared
 * tombstone predicate (takes ItemPointerData by value).
 */
typedef struct MktannBulkDeleteCtx
{
	IndexBulkDeleteCallback cb;
	void				   *cb_state;
} MktannBulkDeleteCtx;

static bool
tid_is_dead(ItemPointerData tid, void *state)
{
	MktannBulkDeleteCtx *c = (MktannBulkDeleteCtx *)state;
	return c->cb(&tid, c->cb_state);
}

/*
 * VACUUM's dead-tuple removal. Block-scans the index and tombstones each
 * posting chain from its FIRST (head) page via mkt_posting_tombstone_chain —
 * the head walk covers the chain's overflow pages, so only heads are acted on.
 * Tombstoned entries are skipped by later scans; physical reclaim happens at a
 * later compaction/rebuild. This is the cleanup path for both explicit DELETEs
 * and the dead old-version of every vector-column UPDATE.
 */
static IndexBulkDeleteResult *
mktann_bulkdelete(
		IndexVacuumInfo		   *info,
		IndexBulkDeleteResult  *stats,
		IndexBulkDeleteCallback callback,
		void				   *cb_state)
{
	if (stats == NULL)
		stats = palloc0(sizeof(IndexBulkDeleteResult));

	Relation	index	= info->index;
	BlockNumber nblocks = RelationGetNumberOfBlocks(index);
	if (nblocks <= 1)
		return stats; /* block 0 is the metadata page */

	/* dim + metric + first_posting from the per-backend cache (metapage read
	 * at most once per backend). mktann_cache_meta skips the rotation-matrix
	 * work the scan / insert cache path does — VACUUM never needs it. */
	Dimension	   dim;
	DistanceMetric metric;
	BlockNumber	   first_posting;
	mktann_cache_meta(index, &dim, &metric, &first_posting);

	MktannStorage storage;
	mktann_storage_init(&storage, index, NULL, metric);

	MktannBulkDeleteCtx ctx = {.cb = callback, .cb_state = cb_state};

	/*
	 * The index is laid out as: block 0 metadata, then the contiguous centroid
	 * region, then the posting pages. first_posting (from the metapage) is one
	 * past the last centroid page, so start there and skip the whole centroid
	 * region without scanning it.
	 *
	 * Within the posting region we act only on chain heads; overflow pages are
	 * reached via the chain from their head, and new/empty pages are expected
	 * (extension slack). A page that is neither a posting page nor empty is
	 * the only anomaly worth surfacing (corruption or a format bug); count
	 * those and emit a single WARNING after the walk rather than one per page,
	 * so a badly corrupt index can't flood the log (bulkdelete also runs once
	 * per dead-tuple batch, i.e. potentially many times per VACUUM).
	 */
	uint32_t	unrecognized = 0;
	BlockNumber first_bad	 = InvalidBlockNumber;
	BlockNumber start		 = Max(first_posting, 1);

	for (BlockNumber blk = start; blk < nblocks; blk++)
	{
		vacuum_delay_point(false);

		Buffer buf = ReadBuffer(index, blk);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		Page page		= BufferGetPage(buf);
		bool is_head	= false;
		bool recognized = PageIsNew(page); /* an empty page is expected */
		/*
		 * Guard on the special-area size before reading the opaque so a page
		 * of another kind is never misread through the posting layout.
		 */
		if (!recognized &&
			PageGetSpecialSize(page) == sizeof(MktPostingPageOpaque))
		{
			MktPostingPageOpaque *op = mkt_posting_opaque(page);
			if (op->page_id == MKT_POSTING_PAGE_ID)
			{
				recognized = true;
				is_head	   = (op->flags & MKT_POSTING_PAGE_FIRST) != 0;
			}
		}
		UnlockReleaseBuffer(buf);

		if (!recognized)
		{
			if (first_bad == InvalidBlockNumber)
				first_bad = blk;
			unrecognized++;
		}

		if (!is_head)
			continue;

		stats->tuples_removed += mkt_posting_tombstone_chain(
				&storage.base, dim, blk, tid_is_dead, &ctx);

		/* Live tuples remaining: the head's maintained live_count, which the
		 * tombstone pass just decremented (O(1), no rescan). */
		Page hp = mkt_storage_read_page(&storage.base, blk);
		stats->num_index_tuples += mkt_posting_head_live_count(hp);
		mkt_storage_release_page(&storage.base, blk);
	}

	/* One summary line per call, not one per bad page (see loop comment). */
	if (unrecognized > 0)
		ereport(WARNING,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("skipped %u unrecognized page(s) in index \"%s\" "
						"during bulkdelete (first at block %u); index may be "
						"corrupt",
						unrecognized,
						RelationGetRelationName(index),
						first_bad)));

	return stats;
}

/*
 * The tuple count reported here becomes the index's pg_class.reltuples,
 * which the planner reads as indexinfo->tuples. Two things keep it honest.
 *
 * With no bulk delete (a VACUUM that found nothing to remove, or ANALYZE),
 * return NULL so the existing statistics stand. Returning a zeroed result
 * instead would set reltuples to 0 on every such VACUUM -- which is what
 * happened -- leaving the planner believing an index over millions of rows
 * is empty until the next ANALYZE.
 *
 * After a bulk delete, the count bulkdelete accumulated is the number of
 * live posting entries, and that is not the number of indexed heap tuples:
 * SOAR and boundary replication store a vector in more than one list, so
 * the entry count overstates the row count by the replication factor.
 * Report the heap's own live count instead, as nbtree does, and carry
 * estimated_count through so an estimate does not overwrite an exact
 * figure (vacuumlazy skips the update when it is set).
 */
/*
 * Refresh the metapage's indexed-row count. The build writes it once and
 * inserts never touch the metapage (a per-insert metapage write would
 * serialize every insert on one page), so without this the value the
 * per-backend cache hands to scans stays at the build-time count for the
 * life of the index. VACUUM already knows the live row count and already
 * pays a metapage read, so this is the natural refresh point. Written
 * only when it changes, so a VACUUM that removed nothing costs no WAL.
 */
static void
mktann_meta_set_ntuples(Relation index, uint32_t ntuples)
{
	Buffer buf = ReadBuffer(index, 0);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	const MktannMetaPage *meta = (const MktannMetaPage *)PageGetSpecialPointer(
			BufferGetPage(buf));
	bool changed = meta->magic == MKT_META_MAGIC && meta->ntuples != ntuples;
	UnlockReleaseBuffer(buf);

	if (!changed)
		return;

	GenericXLogState *state = GenericXLogStart(index);
	buf						= ReadBuffer(index, 0);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	Page page = GenericXLogRegisterBuffer(state, buf, GENERIC_XLOG_FULL_IMAGE);
	((MktannMetaPage *)PageGetSpecialPointer(page))->ntuples = ntuples;
	GenericXLogFinish(state);
	UnlockReleaseBuffer(buf);

	/* Drop the cached metapage so the next scan reads the new count. */
	CacheInvalidateRelcache(index);
}

static IndexBulkDeleteResult *
mktann_vacuumcleanup(IndexVacuumInfo *info, IndexBulkDeleteResult *stats)
{
	if (info->analyze_only)
		return stats;
	if (stats == NULL)
		return NULL;

	stats->num_pages		= RelationGetNumberOfBlocks(info->index);
	stats->num_index_tuples = info->num_heap_tuples;
	stats->estimated_count	= info->estimated_count;

	/* An exact live count is also the metapage's indexed-row count; an
	 * estimate is not worth overwriting an exact figure with. */
	if (!info->estimated_count && info->num_heap_tuples >= 0)
	{
		double n = Min(info->num_heap_tuples, (double)PG_UINT32_MAX);
		mktann_meta_set_ntuples(info->index, (uint32_t)n);
	}
	return stats;
}

static void
mktann_costestimate(
		PlannerInfo *root,
		IndexPath	*path,
		double		 loop_count,
		Cost		*startup_cost,
		Cost		*total_cost,
		Selectivity *selectivity,
		double		*correlation,
		double		*index_pages)
{
	/* Never use the index without ORDER BY <op> */
	if (path->indexorderbys == NIL)
	{
		*startup_cost			  = get_float8_infinity();
		*total_cost				  = get_float8_infinity();
		*selectivity			  = 0;
		*correlation			  = 0;
		*index_pages			  = 0;
		path->path.disabled_nodes = 2;
		return;
	}

	GenericCosts costs = {0};
	genericcostestimate(root, path, loop_count, &costs);

	*startup_cost = costs.indexStartupCost;
	*total_cost	  = costs.indexTotalCost;
	*selectivity  = costs.indexSelectivity;
	*correlation  = costs.indexCorrelation;
	*index_pages  = costs.numIndexPages;
}

static bytea *
mktann_options(Datum reloptions, bool validate)
{
	static const relopt_parse_elt tab[] = {
			{"distance_mode",
			 RELOPT_TYPE_ENUM,
			 offsetof(MktannOptions, distance_mode)},
			{"fan_out", RELOPT_TYPE_INT, offsetof(MktannOptions, fan_out)},
			{"nlist", RELOPT_TYPE_INT, offsetof(MktannOptions, nlist)},
			{"kmeans_nredo",
			 RELOPT_TYPE_INT,
			 offsetof(MktannOptions, kmeans_nredo)},
			{"soar_lambda",
			 RELOPT_TYPE_REAL,
			 offsetof(MktannOptions, soar_lambda)},
			{"boundary_epsilon",
			 RELOPT_TYPE_REAL,
			 offsetof(MktannOptions, boundary_epsilon)},
			{"centroid_compression",
			 RELOPT_TYPE_ENUM,
			 offsetof(MktannOptions, centroid_compression)},
			{"fastscan", RELOPT_TYPE_ENUM, offsetof(MktannOptions, fastscan)},
			{"centroid_fastscan",
			 RELOPT_TYPE_ENUM,
			 offsetof(MktannOptions, centroid_fastscan)},
	};
	return (bytea *)build_reloptions(
			reloptions,
			validate,
			mktann_relopt_kind,
			sizeof(MktannOptions),
			tab,
			lengthof(tab));
}

static bool
mktann_validate(Oid opclassoid)
{
	return true;
}

/* ----------------------------------------------------------------
 * Handler
 * ---------------------------------------------------------------- */

Datum
mktann_handler(PG_FUNCTION_ARGS)
{
	IndexAmRoutine *amroutine = makeNode(IndexAmRoutine);

	/* Properties */
	amroutine->amstrategies			   = 0;
	amroutine->amsupport			   = 3;
	amroutine->amoptsprocnum		   = 0;
	amroutine->amcanorder			   = false;
	amroutine->amcanorderbyop		   = true;
	amroutine->amcanhash			   = false;
	amroutine->amconsistentequality	   = false;
	amroutine->amconsistentordering	   = false;
	amroutine->amcanbackward		   = false;
	amroutine->amcanunique			   = false;
	amroutine->amcanmulticol		   = false;
	amroutine->amoptionalkey		   = true;
	amroutine->amsearcharray		   = false;
	amroutine->amsearchnulls		   = false;
	amroutine->amstorage			   = false;
	amroutine->amclusterable		   = false;
	amroutine->ampredlocks			   = false;
	amroutine->amcanparallel		   = false;
	amroutine->amcanbuildparallel	   = true;
	amroutine->amcaninclude			   = false;
	amroutine->amusemaintenanceworkmem = false;
	amroutine->amsummarizing		   = false;
	amroutine->amparallelvacuumoptions = VACUUM_OPTION_PARALLEL_BULKDEL;
	amroutine->amkeytype			   = InvalidOid;

	/* Build callbacks */
	amroutine->ambuild			= mktann_build;
	amroutine->ambuildempty		= mktann_buildempty;
	amroutine->ambuildphasename = mktann_buildphasename;

	/* Insert / maintenance */
	amroutine->aminsert		   = mktann_insert;
	amroutine->aminsertcleanup = NULL;
	amroutine->ambulkdelete	   = mktann_bulkdelete;
	amroutine->amvacuumcleanup = mktann_vacuumcleanup;

	/* Cost estimation / validation */
	amroutine->amcanreturn	   = NULL;
	amroutine->amcostestimate  = mktann_costestimate;
	amroutine->amgettreeheight = NULL;
	amroutine->amoptions	   = mktann_options;
	amroutine->amproperty	   = NULL;
	amroutine->amvalidate	   = mktann_validate;
	amroutine->amadjustmembers = NULL;

	/* Scan callbacks */
	amroutine->ambeginscan = mktann_beginscan;
	amroutine->amrescan	   = mktann_rescan;
	amroutine->amgettuple  = mktann_gettuple;
	amroutine->amgetbitmap = NULL;
	amroutine->amendscan   = mktann_endscan;
	amroutine->ammarkpos   = NULL;
	amroutine->amrestrpos  = NULL;

	/* Parallel scan (not supported) */
	amroutine->amestimateparallelscan = NULL;
	amroutine->aminitparallelscan	  = NULL;
	amroutine->amparallelrescan		  = NULL;

	/* Planning */
	amroutine->amtranslatestrategy = NULL;
	amroutine->amtranslatecmptype  = NULL;

	PG_RETURN_POINTER(amroutine);
}
