/*
 * mktann.c - meerkat ANN index access method handler
 *
 * Registers the mktann index access method with PostgreSQL. Build and
 * scan callbacks delegate to mktann_build.c and mktann_scan.c; trivial
 * stubs for unimplemented callbacks remain here.
 */

#include <postgres.h>

#include <access/amapi.h>
#include <access/reloptions.h>
#include <access/relscan.h>
#include <commands/vacuum.h>
#include <fmgr.h>
#include <storage/bufmgr.h>
#include <storage/lmgr.h>
#include <utils/float.h>
#include <utils/injection_point.h>
#include <utils/memutils.h>
#include <utils/selfuncs.h>

#include "index/index_base.h"
#include "index/posting_insert.h"
#include "index/query_scan.h"
#include "mkt_pg.h"
#include "mkt_vector.h"
#include "mktann_build.h"
#include "mktann_cache.h"
#include "mktann_scan.h"
#include "mktann_storage.h"
#include "quant/rabitq.h"

PG_FUNCTION_INFO_V1(mktann_handler);

/* ----------------------------------------------------------------
 * Trivial stubs (no separate file needed)
 * ---------------------------------------------------------------- */

static void
mktann_buildempty(Relation index)
{
	/* nothing to do */
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

	/* HOT / unchanged indexed value: the existing index entry still applies.
	 * NULL vectors get no entry. */
	if (index_unchanged || isnull[0])
		return false;

	/* Per-insert scratch context: beam-search + encode allocations are freed
	 * in one shot and don't accumulate in the inserting transaction. */
	MemoryContext insert_ctx = AllocSetContextCreate(
			CurrentMemoryContext, "mktann insert", ALLOCSET_DEFAULT_SIZES);
	MemoryContext old_ctx = MemoryContextSwitchTo(insert_ctx);

	/* Immutable index parameters from the per-backend cache (metapage read at
	 * most once per backend, not once per insert). */
	MktIndexBase base;
	mktann_index_base_init(index, &base);
	Dimension dim = base.dim;

	MktannStorage storage;
	mktann_storage_init(&storage, index, NULL, base.metric);
	base.centroid_storage = &storage.base;
	base.posting_storage  = &storage.base;
	base.page_base		  = NULL;

	/* Inserted vector. */
	MktVector *vec	= DatumGetMktVector(values[0]);
	VectorRef  vref = MktVectorToRef(vec);
	if (vref.dim != dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("inserted vector dimension %u does not match index "
						"dimension %u",
						vref.dim,
						dim)));

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
				&enc);
		UnlockPage(index, head, ExclusiveLock);
	}

	MemoryContextSwitchTo(old_ctx);
	MemoryContextDelete(insert_ctx);

	/* bool result is only meaningful for unique indexes. */
	return false;
}

static IndexBulkDeleteResult *
mktann_bulkdelete(
		IndexVacuumInfo		   *info,
		IndexBulkDeleteResult  *stats,
		IndexBulkDeleteCallback callback,
		void				   *cb_state)
{
	if (stats == NULL)
		stats = palloc0(sizeof(IndexBulkDeleteResult));
	return stats;
}

static IndexBulkDeleteResult *
mktann_vacuumcleanup(IndexVacuumInfo *info, IndexBulkDeleteResult *stats)
{
	if (stats == NULL)
		stats = palloc0(sizeof(IndexBulkDeleteResult));
	stats->num_pages = RelationGetNumberOfBlocks(info->index);
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
			{"fastscan", RELOPT_TYPE_BOOL, offsetof(MktannOptions, fastscan)},
			{"centroid_fastscan",
			 RELOPT_TYPE_BOOL,
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
	amroutine->amsupport			   = 2;
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
