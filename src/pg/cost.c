/*
 * cost.c - Cost model for vector index scans
 *
 * Prices the work a scan actually does: the centroid descent, the posting
 * lists it probes, and the rerank fetches. Everything comes from the
 * metapage scalars rd_amcache already holds plus the planner's own
 * statistics, so the estimate reads no index page.
 *
 * Page access is priced by how each page is reached rather than uniformly,
 * because the estimate has to stay comparable against plans that pay for
 * pages in different ways: reads that depend on an earlier read at
 * random_page_cost, reads the scan prefetches as a batch at seq_page_cost,
 * and pages whose contents the per-entry term already prices not at all.
 *
 * The planner resolves top-k, the filter margin and the rerank pool through
 * the same functions the executor uses, so the estimate prices the pool the
 * scan will actually build rather than a second copy of the rules.
 *
 * See docs/cost-model-design.md for the derivation and the measurements the
 * constants in cost.h come from.
 */

#include <postgres.h>

#include <access/heaptoast.h>
#include <access/relation.h>
#include <math.h>
#include <optimizer/cost.h>
#include <optimizer/optimizer.h>
#include <utils/lsyscache.h>
#include <utils/rel.h>
#include <utils/selfuncs.h>
#include <utils/spccache.h>
#include <utils/syscache.h>

#include "amcache.h"
#include "cost.h"
#include "index/posting_page.h"
#include "index/query_scan.h"
#include "scan.h"
#include "scan_bound.h"
#include "support_pg.h"

/*
 * Rows the query will pull from the scan, before any filter inflation.
 *
 * root->limit_tuples is the planner's own view of the LIMIT, which is the
 * same quantity scan_bound.c resolves at execution; where it is absent (no
 * LIMIT, or one the planner could not fold) the scan sizes itself for every
 * row it could return, and passing zero here lets the shared resolver apply
 * that rule rather than restating it.
 */
static uint32_t
row_target(PlannerInfo *root)
{
	if (root->limit_tuples <= 0.0 || !isfinite(root->limit_tuples))
		return 0;
	if (root->limit_tuples >= (double)PG_UINT32_MAX)
		return PG_UINT32_MAX;
	return (uint32_t)ceil(root->limit_tuples);
}

/*
 * True when the indexed column's values live outside the heap tuple, so
 * every rerank candidate costs a toast fetch on top of the heap read.
 *
 * Two things decide it: the column must permit out-of-line storage, and its
 * values must be wide enough to be pushed there. The width has to come from
 * the vector's own dimension rather than from the statistics -- pg_statistic
 * records what is *stored* in the tuple, which for an already-toasted column
 * is the eighteen-byte pointer, so asking it whether the value is wide
 * answers no precisely when the answer is yes.
 */
static bool
indexed_column_is_external(
		PlannerInfo	 *root,
		RelOptInfo	 *baserel,
		IndexOptInfo *info,
		Dimension	  dim)
{
	if (info->nkeycolumns < 1 || info->indexkeys[0] <= 0)
		return false; /* expression index: no column to ask about */

	Oid		  reloid = root->simple_rte_array[baserel->relid]->relid;
	HeapTuple tp	 = SearchSysCache2(
			ATTNUM,
			ObjectIdGetDatum(reloid),
			Int16GetDatum(info->indexkeys[0]));

	if (!HeapTupleIsValid(tp))
		return false;

	char storage = ((Form_pg_attribute)GETSTRUCT(tp))->attstorage;

	ReleaseSysCache(tp);

	if (storage == TYPSTORAGE_PLAIN)
		return false;

	/* varlena header plus one float per dimension */
	return (Size)(VARHDRSZ + 4 + sizeof(float) * dim) > TOAST_TUPLE_THRESHOLD;
}

/* Entries per page for the format the index was built in. */
static double
entries_per_page(Dimension dim)
{
	uint32_t epp = mkt_posting_max_entries(dim);

	return epp > 0 ? (double)epp : 1.0;
}

void
mktann_cost_estimate(
		PlannerInfo *root,
		IndexPath	*path,
		double		 loop_count,
		Cost		*startup_cost,
		Cost		*total_cost,
		Selectivity *selectivity,
		double		*correlation,
		double		*index_pages)
{
	IndexOptInfo *info	  = path->indexinfo;
	RelOptInfo	 *baserel = info->rel;

	/*
	 * Scalars from the metapage, through the per-backend cache: one metapage
	 * read for the first plan in this backend and none afterwards. NoLock
	 * because the planner already holds a lock on the index.
	 */
	Relation	   index = index_open(info->indexoid, NoLock);
	MktannScanInfo si	 = mktann_cache_scan_info(index);

	index_close(index, NoLock);

	uint32_t nlist = si.nlist > 0 ? si.nlist : 1;

	/* --- inputs (§6) --- */

	double N = info->tuples > 0.0 ? info->tuples : baserel->tuples;

	if (N < 1.0)
		N = 1.0;

	/*
	 * Selectivity of the quals the executor applies above the scan. They do
	 * not restrict what the scan returns; they throw rows away afterwards,
	 * which is why they inflate the pool instead of shrinking it.
	 */
	double sel = 1.0;

	if (baserel->baserestrictinfo != NIL)
		sel = clauselist_selectivity(
				root,
				baserel->baserestrictinfo,
				baserel->relid,
				JOIN_INNER,
				NULL);

	/*
	 * The pool the scan will build, resolved by the same functions the
	 * executor uses, so the planner prices what the scan does rather than a
	 * parallel estimate of it.
	 */
	uint32_t k	   = mkt_scan_inflate_for_filter(row_target(root), sel);
	uint32_t k_eff = mkt_scan_resolve_top_k(k, N);

	uint32_t nprobe = mkt_nprobe > 0 ? (uint32_t)mkt_nprobe
									 : mkt_auto_nprobe(nlist);

	if (nprobe > nlist)
		nprobe = nlist;

	double n_route = nprobe;

	if (mkt_probe_expand > 1.0 &&
		si.centroid_format != MKT_CENTROID_FMT_FLOAT &&
		si.centroid_format != MKT_CENTROID_FMT_HALF)
	{
		n_route = (double)nprobe * mkt_probe_expand;
		if (n_route > (double)nlist)
			n_route = (double)nlist;
	}

	double pool = mkt_query_rerank_pool_estimate(k_eff, nprobe);

	/*
	 * Work comes from pages, not from N. Absent the health block of §7 the
	 * posting region is every page past the centroid region, and the list
	 * sizes are taken as even -- an index whose maintenance has not run
	 * costs more than this says, which is the direction that makes the
	 * estimate optimistic rather than unsafe.
	 */
	double posting_pages = (double)info->pages - (double)si.first_posting;

	if (posting_pages < 1.0)
		posting_pages = 1.0;

	double pages_per_list = posting_pages / (double)nlist;
	double epp			  = entries_per_page(si.dim);
	double entries		  = (double)nprobe * pages_per_list * epp;

	/* --- terms (§6) --- */

	double cop = cpu_operator_cost;
	double dim = (double)si.dim;

	double spc_random, spc_seq;

	get_tablespace_page_costs(info->reltablespace, &spc_random, &spc_seq);

	/*
	 * Descent: the beam narrows to nprobe * mkt.centroid_beam_scale with a
	 * fan-out floor, the same rule the scan applies, and phase A re-ranks
	 * every routed cluster's head page exactly.
	 */
	double beam = (double)nprobe * mkt_centroid_beam_scale;

	if (si.fan_out > 0 && beam < (double)si.fan_out)
		beam = (double)si.fan_out;
	if (beam > (double)nlist)
		beam = (double)nlist;

	double descent = (double)si.nlevels * beam * MKT_COST_CENTROID_SLOT * cop +
					 n_route *
							 (MKT_COST_PROBE_HEAD_BASE +
							  MKT_COST_PROBE_HEAD_SLOPE * dim) *
							 cop;

	/* Per-entry scoring; the AoS estimate is the document's, unmeasured. */
	double per_entry = si.has_fastscan
							 ? MKT_COST_ENTRY_BASE + MKT_COST_ENTRY_SLOPE * dim
							 : MKT_COST_AOS_ENTRY_BASE +
									   MKT_COST_AOS_ENTRY_SLOPE * dim;

	/* Every probed list is opened; only fastscan also builds a table. */
	double cluster_setup = MKT_COST_LIST_OPEN +
						   (si.has_fastscan ? MKT_COST_CLUSTER_LUT : 0.0);

	double topk_factor = 1.0;

	if (k_eff > MKT_DEFAULT_K)
		topk_factor += MKT_COST_TOPK_LOG_COEFF *
					   log2((double)k_eff / (double)MKT_DEFAULT_K);

	double scan_cpu = (double)nprobe * cluster_setup * cop +
					  entries * per_entry * topk_factor * cop;

	/*
	 * Index I/O, priced by how each page is actually reached rather than by
	 * charging random_page_cost for all of them. Uniform random pricing puts
	 * roughly 91% of the estimate into seeks a warm scan never performs;
	 * pgvector's IVFFlat makes the same correction crudely, by moving half
	 * its page cost from random to sequential.
	 */

	/*
	 * The descent's centroid pages are seeks: each level's reads depend on
	 * the previous level's result, so they cannot overlap.
	 *
	 * The routed clusters' head pages are not. The scan issues a prefetch
	 * for every one of them before reading any, so they are a batch of
	 * overlapped reads rather than a sequence of seeks, and pricing each at
	 * random_page_cost charges for a latency the scan already hid. They are
	 * priced at seq_page_cost for that reason -- the same correction
	 * PostgreSQL has no general mechanism for, effective_io_concurrency
	 * existing precisely because batched reads do not cost what serial ones
	 * do.
	 */
	double descent_pages = (double)si.nlevels * beam;
	double random_io =
			index_pages_fetched(
					descent_pages, info->pages, (double)info->pages, root) *
			spc_random;
	double head_io = n_route * spc_seq;

	/*
	 * The rest of each probed chain is not charged as I/O. Walking it costs
	 * scoring the entries it holds, which scan_cpu already prices per
	 * entry; charging the page as well counts the same work twice, and the
	 * pages of a list being scanned are resident by the time the scan
	 * reaches them -- the head read brought the chain into cache.
	 */
	double index_io = random_io + head_io;

	/*
	 * Rerank: one exact distance per candidate, plus the heap page it comes
	 * from. The heap fetches are random by construction -- candidates are
	 * ordered by distance, not by TID -- so they go through the same
	 * Mackert-Lohman estimate against the heap's own page count.
	 */
	double per_fetch = MKT_COST_FETCH;

	if (indexed_column_is_external(root, baserel, info, si.dim))
		per_fetch *= MKT_COST_FETCH_DETOAST;

	double rerank_cpu = pool * per_fetch * cop;
	double rerank_io =
			index_pages_fetched(
					pool, baserel->pages, (double)info->pages, root) *
			spc_random;

	double work = descent + scan_cpu + index_io + rerank_cpu + rerank_io;

	/*
	 * All of it is startup work: the scan elects its whole top-k before it
	 * can return the first tuple, so a LIMIT must not discount it. loop_count
	 * multiplies it because a rescan inside a nested loop runs the search
	 * again.
	 */
	*startup_cost = work * (loop_count > 0.0 ? loop_count : 1.0);
	*total_cost	  = *startup_cost + (double)k_eff * cpu_index_tuple_cost;

	/*
	 * The scan emits k_eff rows; the core applies the qual to them and
	 * arrives at about k output rows, and charges heap fetches for k_eff.
	 * That is what happens.
	 */
	*selectivity = Min(1.0, (double)k_eff / N);
	*correlation = 0.0;
	*index_pages = descent_pages + (double)nprobe * pages_per_list;
}
