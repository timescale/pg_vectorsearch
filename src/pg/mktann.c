/*
 * mktann.c - meerkat ANN index access method (shim)
 *
 * Skeleton IAM that registers with PostgreSQL so CREATE INDEX ... USING
 * mktann works.  Every callback is either a no-op or returns an empty
 * result; the real implementation will replace these one at a time.
 */

#include <postgres.h>

#include <access/amapi.h>
#include <access/reloptions.h>
#include <access/relscan.h>
#include <commands/vacuum.h>
#include <fmgr.h>
#include <storage/bufmgr.h>
#include <utils/float.h>
#include <utils/selfuncs.h>

#include "mkt_pg.h"

PG_FUNCTION_INFO_V1(mktann_handler);

/* ----------------------------------------------------------------
 * Build callbacks
 * ---------------------------------------------------------------- */

static IndexBuildResult *
mktann_build(Relation heap, Relation index, struct IndexInfo *index_info)
{
	IndexBuildResult *result = palloc0(sizeof(IndexBuildResult));
	result->heap_tuples		 = 0;
	result->index_tuples	 = 0;
	return result;
}

static void
mktann_buildempty(Relation index)
{
	/* nothing to do */
}

/* ----------------------------------------------------------------
 * Insert / maintenance callbacks
 * ---------------------------------------------------------------- */

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

/* ----------------------------------------------------------------
 * Cost estimation
 * ---------------------------------------------------------------- */

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

/* ----------------------------------------------------------------
 * Options / validation
 * ---------------------------------------------------------------- */

static bytea *
mktann_options(Datum reloptions, bool validate)
{
	static const relopt_parse_elt tab[] = {
			{"distance_mode",
			 RELOPT_TYPE_ENUM,
			 offsetof(MktannOptions, distance_mode)},
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
 * Scan callbacks
 * ---------------------------------------------------------------- */

static IndexScanDesc
mktann_beginscan(Relation index, int nkeys, int norderbys)
{
	IndexScanDesc scan;
	scan		 = RelationGetIndexScan(index, nkeys, norderbys);
	scan->opaque = NULL;
	return scan;
}

static void
mktann_rescan(
		IndexScanDesc scan,
		ScanKey		  keys,
		int			  nkeys,
		ScanKey		  orderbys,
		int			  norderbys)
{
	if (keys && scan->numberOfKeys > 0)
		memcpy(scan->keyData, keys, scan->numberOfKeys * sizeof(ScanKeyData));
	if (orderbys && scan->numberOfOrderBys > 0)
		memcpy(scan->orderByData,
			   orderbys,
			   scan->numberOfOrderBys * sizeof(ScanKeyData));
}

static bool
mktann_gettuple(IndexScanDesc scan, ScanDirection direction)
{
	return false;
}

static void
mktann_endscan(IndexScanDesc scan)
{
	/* nothing to do */
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
	amroutine->amsupport			   = 1;
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
	amroutine->amcanbuildparallel	   = false;
	amroutine->amcaninclude			   = false;
	amroutine->amusemaintenanceworkmem = false;
	amroutine->amsummarizing		   = false;
	amroutine->amparallelvacuumoptions = VACUUM_OPTION_PARALLEL_BULKDEL;
	amroutine->amkeytype			   = InvalidOid;

	/* Required callbacks */
	amroutine->ambuild			= mktann_build;
	amroutine->ambuildempty		= mktann_buildempty;
	amroutine->aminsert			= mktann_insert;
	amroutine->aminsertcleanup	= NULL;
	amroutine->ambulkdelete		= mktann_bulkdelete;
	amroutine->amvacuumcleanup	= mktann_vacuumcleanup;
	amroutine->amcanreturn		= NULL;
	amroutine->amcostestimate	= mktann_costestimate;
	amroutine->amgettreeheight	= NULL;
	amroutine->amoptions		= mktann_options;
	amroutine->amproperty		= NULL;
	amroutine->ambuildphasename = NULL;
	amroutine->amvalidate		= mktann_validate;
	amroutine->amadjustmembers	= NULL;
	amroutine->ambeginscan		= mktann_beginscan;
	amroutine->amrescan			= mktann_rescan;
	amroutine->amgettuple		= mktann_gettuple;
	amroutine->amgetbitmap		= NULL;
	amroutine->amendscan		= mktann_endscan;
	amroutine->ammarkpos		= NULL;
	amroutine->amrestrpos		= NULL;

	/* Parallel scan (not supported) */
	amroutine->amestimateparallelscan = NULL;
	amroutine->aminitparallelscan	  = NULL;
	amroutine->amparallelrescan		  = NULL;

	/* Planning */
	amroutine->amtranslatestrategy = NULL;
	amroutine->amtranslatecmptype  = NULL;

	PG_RETURN_POINTER(amroutine);
}
