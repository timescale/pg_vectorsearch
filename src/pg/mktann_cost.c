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
#include <optimizer/pathnode.h>
#include <optimizer/paths.h>
#include <storage/buf_internals.h>
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
#include "pg/mktann_storage.h"

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

/*
 * Posting-chain reads are dependent, synchronous, single-page random
 * reads with no prefetch -- the access pattern random_page_cost
 * understates most (it is tuned against scans that overlap I/O).
 * Measured on NVMe: ~475us per cold chained read vs the ~110us the
 * GUC's ratio implies; priced as a multiple of random_page_cost so
 * device-relative tuning still flows through.
 */
#define MKT_COST_SYNC_READ_MULT 4.0

static set_rel_pathlist_hook_type prev_pathlist_hook = NULL;

/*
 * Estimated fraction of the index's posting region resident in shared
 * buffers, from probing the buffer mapping table for a fixed sample of
 * evenly spaced posting blocks (a few microseconds; no pages are read
 * or pinned). This is the signal the stock planner lacks: whether this
 * query's page reads are memory pointer chases or storage I/O. The
 * miss fraction prices an I/O term that is what makes parallel workers
 * win exactly where measurements show they win -- storage-bound scans
 * -- while fully-resident scans keep costs below the parallel setup
 * threshold and stay serial.
 *
 * Pages in the OS page cache but not in shared buffers count as
 * misses, overstating I/O for the warm-OS/cold-buffers case the same
 * way the stock cost model's cache assumptions are approximate; the
 * estimate only needs to separate "mostly resident" from "mostly not".
 */
#define MKT_RESIDENCY_SAMPLES 64

static double
mktann_residency(Relation index, BlockNumber first_posting)
{
	BlockNumber nblocks = RelationGetNumberOfBlocks(index);

	if (nblocks <= first_posting)
		return 1.0;

	uint64 span	   = nblocks - first_posting;
	int	   samples = (int)Min(span, MKT_RESIDENCY_SAMPLES);
	int	   hits	   = 0;

	for (int i = 0; i < samples; i++)
	{
		BlockNumber blkno = first_posting +
							(BlockNumber)((span * (2 * (uint64)i + 1)) /
										  (2 * samples));
		BufferTag tag;
		InitBufferTag(&tag, &index->rd_locator, MAIN_FORKNUM, blkno);

		uint32	hash = BufTableHashCode(&tag);
		LWLock *lock = BufMappingPartitionLock(hash);

		LWLockAcquire(lock, LW_SHARED);
		int buf_id = BufTableLookup(&tag, hash);
		LWLockRelease(lock);

		if (buf_id >= 0)
			hits++;
	}

	double sampled = (double)hits / samples;

	/*
	 * Blend in this backend's own read experience. The global sample
	 * cannot see a partially-warm index whose HOT clusters are
	 * resident (organic warm-up without a full prewarm): queries
	 * there read warm pages while 64 spread samples say "cold". The
	 * recent-buffers counters record what this session's reads
	 * actually hit, so once it has real volume, believe whichever
	 * signal says warmer -- a genuinely cold session keeps a low hit
	 * rate, and eviction pushes the rate back down via stale misses.
	 */
	uint64 reads = mkt_bufcache_hits + mkt_bufcache_cold + mkt_bufcache_stale;
	double session_rate = 0.0;

	if (reads >= 1000)
		session_rate = (double)mkt_bufcache_hits / (double)reads;

	elog(DEBUG1,
		 "mktann residency: %d/%d sampled posting blocks resident, "
		 "session hit rate %.2f over " UINT64_FORMAT " reads "
		 "(first_posting=%u nblocks=%u)",
		 hits,
		 samples,
		 session_rate,
		 reads,
		 first_posting,
		 nblocks);

	return Max(sampled, session_rate);
}

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

	if (!mktann_cache_format_ok(index))
	{
		/* An index this build cannot read must not abort planning for
		 * the whole table; report "cannot serve" and let costestimate
		 * disable the path. */
		relation_close(index, AccessShareLock);
		c.unreadable = true;
		return c;
	}

	Dimension	   dim;
	DistanceMetric metric;
	BlockNumber	   first_posting;
	mktann_cache_meta(index, &dim, &metric, &first_posting);
	MktannScanInfo info		= mktann_cache_scan_info(index);
	double		   resident = mktann_residency(index, first_posting);

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

	/* Storage I/O for the non-resident posting pages this scan will
	 * touch: the per-cluster chains plus Phase A's one first-page read
	 * per routed cluster. Random synchronous reads with
	 * per-participant streams -- the phase divides across workers like
	 * the scan CPU does. */
	c.io = (n * cluster_pages + n_route) * (1.0 - resident) *
		   random_page_cost * MKT_COST_SYNC_READ_MULT;

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

	MktannCosts c = mktann_compute_costs(path->indexinfo->indexoid);

	/* Never use the index without ORDER BY <op>, and never use an index
	 * whose on-disk format this build cannot read. */
	if (path->indexorderbys == NIL || c.unreadable)
	{
		*startup_cost			  = get_float8_infinity();
		*total_cost				  = get_float8_infinity();
		*selectivity			  = 0;
		*correlation			  = 0;
		*index_pages			  = 0;
		path->path.disabled_nodes = 2;
		return;
	}

	/* All search work happens before the first tuple. Repeated inner
	 * scans re-run the whole search (no amrescan shortcut). */
	*startup_cost = (c.descent + c.scan + c.rerank + c.io) *
					Max(loop_count, 1.0);
	*total_cost	 = *startup_cost + c.emitted * cpu_index_tuple_cost;
	*selectivity = c.selectivity;
	*correlation = 0;
	*index_pages = c.index_pages;
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

/* Amdahl re-cost of one partial mktann path: the descent runs once
 * (the rendezvous winner); the scan, rerank and storage I/O divide
 * across participants (each runs its own read stream). Each
 * participant emits up to its own pool, but Gather Merge stops
 * pulling at LIMIT -- per-tuple cost is charged on the (divided) row
 * estimate carried by the partial path. */
static void
recost_partial(IndexPath *ipath, const MktannCosts *c)
{
	double divisor = parallel_divisor(ipath->path.parallel_workers);
	Cost   startup = c->descent + (c->scan + c->rerank + c->io) / divisor;

	elog(DEBUG1,
		 "mktann re-cost: partial path workers=%d startup %.1f -> %.1f",
		 ipath->path.parallel_workers,
		 ipath->path.startup_cost,
		 startup);

	ipath->path.startup_cost = startup;
	ipath->path.total_cost = startup + ipath->path.rows * cpu_index_tuple_cost;
}

/*
 * Core builds a partial index path only when compute_parallel_worker
 * approves, and that gate divides by the estimated HEAP page count --
 * for an ANN scan the heap fetches are just the rerank pool (a few
 * hundred tuples), always below min_parallel_table_scan_size, so the
 * gate rejects parallelism no matter how many index pages the scan
 * reads. cost_index recognizes this exact trap for index-only scans
 * and passes heap_pages = -1 there; an mktann scan has the same shape
 * (index work dominates, heap is an afterthought), so when core made
 * no partial path we build one ourselves, sizing workers from the
 * index side alone.
 */
static void
mktann_build_partial_path(PlannerInfo *root, RelOptInfo *rel)
{
	(void)root;

	if (!rel->consider_parallel || max_parallel_workers_per_gather <= 0)
		return;

	ListCell *lc;

	/* Core already made one (e.g. the table sets parallel_workers)? */
	foreach (lc, rel->partial_pathlist)
	{
		Path *p = (Path *)lfirst(lc);

		if (IsA(p, IndexPath) &&
			((IndexPath *)p)->indexinfo->relam == mktann_am_oid())
			return;
	}

	foreach (lc, rel->pathlist)
	{
		Path *p = (Path *)lfirst(lc);

		if (!IsA(p, IndexPath))
			continue;

		IndexPath *spath = (IndexPath *)p;
		if (spath->indexinfo->relam != mktann_am_oid() ||
			spath->indexorderbys == NIL || spath->path.param_info != NULL ||
			!spath->path.parallel_safe)
			continue;

		MktannCosts c = mktann_compute_costs(spath->indexinfo->indexoid);
		if (c.unreadable)
			continue;

		int workers = compute_parallel_worker(
				rel,
				-1 /* heap pages: not the driver, as for index-only */,
				c.index_pages,
				max_parallel_workers_per_gather);
		if (workers <= 0)
			continue;

		IndexPath *ppath = makeNode(IndexPath);

		*ppath = *spath; /* flat copy; clause lists are shared */
		ppath->path.parallel_aware	 = true;
		ppath->path.parallel_workers = workers;
		ppath->path.rows			 = clamp_row_est(
				spath->path.rows / parallel_divisor(workers));
		recost_partial(ppath, &c);

		add_partial_path(rel, (Path *)ppath);
	}
}

static void
mktann_cost_pathlist_hook(
		PlannerInfo *root, RelOptInfo *rel, Index rti, RangeTblEntry *rte)
{
	if (prev_pathlist_hook)
		prev_pathlist_hook(root, rel, rti, rte);

	if (rel->reloptkind != RELOPT_BASEREL || rel->pathlist == NIL)
		return;

	/* Re-cost the partial mktann paths core built (reloption route). */
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

		MktannCosts c = mktann_compute_costs(ipath->indexinfo->indexoid);
		if (c.unreadable)
			continue;

		recost_partial(ipath, &c);
	}

	mktann_build_partial_path(root, rel);
}

void
mktann_cost_register_hook(void)
{
	prev_pathlist_hook	  = set_rel_pathlist_hook;
	set_rel_pathlist_hook = mktann_cost_pathlist_hook;
}
