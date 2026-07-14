/*
 * mktann_cost.c - Planner cost model for mktann scans
 *
 * The search runs entirely before the first tuple: descent + probe
 * re-rank (inherently serial), the per-cluster posting scan and the
 * exact rerank (both divide across parallel workers). The model prices
 * each phase from the session GUCs (mkt.nprobe is not in the query) and
 * the index metapage, in cpu_operator_cost units.
 *
 * The planner's engagement policy lives entirely here: costs are meant
 * to be honest so PostgreSQL's own machinery (parallel_setup_cost,
 * parallel_tuple_cost, max_parallel_workers_per_gather) discovers the
 * serial/parallel crossover wherever it actually lies -- deliberately
 * no nprobe threshold or other hardcoded engagement rule. The c_*
 * constants below are provisional pending calibration against measured
 * phase timings; their structure (what scales with what) is the part
 * that must be right.
 *
 * Why a pathlist hook: amcostestimate runs before the partial path's
 * worker count exists, and cost_index divides only per-tuple heap cpu
 * by the parallel divisor -- never index startup work. With the whole
 * search in startup, a partial path would always tie the serial path
 * and Gather Merge could never win. The hook re-costs partial mktann
 * paths after compute_parallel_worker has chosen the worker count,
 * dividing the parallel phases by the same divisor formula
 * cost_index uses. Base-relation gather paths are generated after
 * set_rel_pathlist runs its hook (grouping_planner stage), so the
 * re-costed values are what Gather Merge is built from.
 */

#include <postgres.h>

#include <access/relation.h>
#include <commands/defrem.h>
#include <math.h>
#include <optimizer/cost.h>
#include <optimizer/optimizer.h>
#include <optimizer/paths.h>
#include <utils/float.h>
#include <utils/lsyscache.h>
#include <utils/rel.h>
#include <utils/selfuncs.h>

#include "index/posting_page.h"
#include "index/query_scan.h"
#include "mkt_pg.h"
#include "pg/mktann_cache.h"
#include "pg/mktann_cost.h"
#include "pg/mktann_scan.h"

/*
 * Provisional per-unit work factors, in cpu_operator_cost units.
 * Anchored loosely to measured warm-path timings (one cpu_operator_cost
 * ~ one simple operator call); calibration replaces them with fitted
 * values. Structure over precision: entries dominate at scale.
 */
#define MKT_COST_ENTRY		0.2	  /* score one posting entry (fastscan) */
#define MKT_COST_CLUSTER	60.0  /* per-cluster setup (LUT, begin/end) */
#define MKT_COST_ROUTE_PAGE 90.0  /* score one centroid page */
#define MKT_COST_PHASE_A	50.0  /* exact re-rank of one routed cluster */
#define MKT_COST_FETCH		450.0 /* rerank heap fetch + exact distance */

static set_rel_pathlist_hook_type prev_pathlist_hook = NULL;

/* The mktann access method's OID, resolved once per backend. */
static Oid
mktann_am_oid(void)
{
	static Oid am_oid = InvalidOid;

	if (!OidIsValid(am_oid))
		am_oid = get_index_am_oid("mktann", false);
	return am_oid;
}

MktannCosts
mktann_compute_costs(Oid indexoid)
{
	MktannCosts c = {0};

	Relation index = relation_open(indexoid, AccessShareLock);

	Dimension	   dim;
	DistanceMetric metric;
	BlockNumber	   first_posting;
	mktann_cache_meta(index, &dim, &metric, &first_posting);
	MktannScanInfo info = mktann_cache_scan_info(index);

	relation_close(index, AccessShareLock);

	double nlist   = Max(info.nlist, 1);
	double ntuples = Max(info.ntuples, 1);
	double E	   = ntuples / nlist; /* entries per cluster */

	double n = mkt_nprobe > 0 ? (double)mkt_nprobe : 1.0;
	if (n > nlist)
		n = nlist;

	/* Probe expansion mirrors the executor (bounded 2x). */
	double n_route = Min(n * 2.0, Min(n + 256.0, nlist));

	uint32_t k = MKT_DEFAULT_K;
	if (mkt_query_limit > 0 && (uint32_t)mkt_query_limit > k)
		k = (uint32_t)mkt_query_limit;
	double pool = (double)Min(mkt_query_rerank_pool(k), (uint32_t)(n * E));

	double epp			 = Max(mkt_posting_max_entries(dim), 1);
	double cluster_pages = ceil(E / epp);

	/* Descent: roughly one page per beam slot per level, plus the exact
	 * re-rank page read per routed cluster. */
	double levels	  = Max(1.0, ceil(log(nlist) / log(74.0)) + 1);
	double beam_pages = Max(1.0, n_route / 2.0) * levels;

	c.descent = (beam_pages * MKT_COST_ROUTE_PAGE +
				 n_route * MKT_COST_PHASE_A) *
				cpu_operator_cost;
	c.scan = (n * MKT_COST_CLUSTER + n * E * MKT_COST_ENTRY) *
			 cpu_operator_cost;
	c.rerank	  = pool * MKT_COST_FETCH * cpu_operator_cost;
	c.emitted	  = pool;
	c.index_pages = beam_pages + n * cluster_pages;
	c.selectivity = Min(1.0, pool / ntuples);

	return c;
}

void
mktann_costestimate(
		struct PlannerInfo *root,
		struct IndexPath   *path,
		double				loop_count,
		Cost			   *startup_cost,
		Cost			   *total_cost,
		Selectivity		   *selectivity,
		double			   *correlation,
		double			   *index_pages)
{
	(void)root;

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

	MktannCosts c = mktann_compute_costs(path->indexinfo->indexoid);

	/* All search work happens before the first tuple. Repeated inner
	 * scans re-run the whole search (no amrescan shortcut). */
	*startup_cost = (c.descent + c.scan + c.rerank) * Max(loop_count, 1.0);
	*total_cost	  = *startup_cost + c.emitted * cpu_index_tuple_cost;
	*selectivity  = c.selectivity;
	*correlation  = 0;
	*index_pages  = c.index_pages;
}

/* The divisor cost_index applies to parallel paths (costsize.c). */
static double
parallel_divisor(int workers)
{
	double divisor = workers;

	if (parallel_leader_participation)
	{
		double leader = 1.0 - (0.3 * workers);
		if (leader > 0)
			divisor += leader;
	}
	return Max(divisor, 1.0);
}

static void
mktann_cost_pathlist_hook(
		PlannerInfo *root, RelOptInfo *rel, Index rti, RangeTblEntry *rte)
{
	if (prev_pathlist_hook)
		prev_pathlist_hook(root, rel, rti, rte);

	if (rel->reloptkind != RELOPT_BASEREL || rel->partial_pathlist == NIL)
		return;

	ListCell *lc;
	foreach (lc, rel->partial_pathlist)
	{
		Path *path = (Path *)lfirst(lc);

		if (!IsA(path, IndexPath))
			continue;

		IndexPath *ipath = (IndexPath *)path;
		if (ipath->indexinfo->relam != mktann_am_oid() ||
			ipath->indexorderbys == NIL || path->parallel_workers <= 0)
			continue;

		MktannCosts c		= mktann_compute_costs(ipath->indexinfo->indexoid);
		double		divisor = parallel_divisor(path->parallel_workers);

		/* The descent runs once (the rendezvous winner); the scan and
		 * rerank divide across participants. Each participant emits up
		 * to its own pool, but Gather Merge stops pulling at LIMIT --
		 * per-tuple cost is charged on the divided row estimate the
		 * planner already put on the partial path. */
		Cost startup = c.descent + (c.scan + c.rerank) / divisor;

		path->startup_cost = startup;
		path->total_cost   = startup + path->rows * cpu_index_tuple_cost;
	}
}

void
mktann_cost_register_hook(void)
{
	prev_pathlist_hook	  = set_rel_pathlist_hook;
	set_rel_pathlist_hook = mktann_cost_pathlist_hook;
}
