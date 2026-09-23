/*
 * cost.c - Cost model for vector index scans
 *
 * Prices the work a scan actually does: descending the centroid tree,
 * scanning the posting lists it routes to, and reranking the candidates
 * that survive. Everything comes from the scalars on the index's metapage
 * plus the planner's own statistics, so the estimate reads no index page.
 *
 * Page access is priced by how each page is reached rather than uniformly:
 * dependent reads at random_page_cost, prefetched batches at
 * seq_page_cost.
 *
 * The top-k, the filter margin, the routed cluster count and the rerank
 * pool size are resolved by the same functions in scan.c, scan_bound.c and
 * query_scan.c that the scan itself calls. Constants are in cost.h.
 */

#include <postgres.h>

#include <access/heaptoast.h>
#include <access/relation.h>
#include <catalog/pg_class.h>
#include <math.h>
#include <optimizer/cost.h>
#include <optimizer/optimizer.h>
#include <storage/bufmgr.h>
#include <storage/lmgr.h>
#include <utils/lsyscache.h>
#include <utils/rel.h>
#include <utils/selfuncs.h>
#include <utils/spccache.h>
#include <utils/syscache.h>

#include "amcache.h"
#include "cost.h"
#include "index/centroid_page.h"
#include "index/posting_page.h"
#include "index/query_scan.h"
#include "scan.h"
#include "scan_bound.h"
#include "support_pg.h"

/*
 * Everything the individual terms below need, collected once so each can be
 * read on its own.
 */
typedef struct CostInputs
{
	/* index shape, from the metapage */
	Dimension dim;
	uint32_t  nlevels;		/* centroid tree depth */
	bool	  has_fastscan; /* posting page format the index was built in */
	PrismCentroidFormat centroid_format;

	/* what this particular query will do */
	uint32_t nlist;	 /* posting lists in the index */
	uint32_t nprobe; /* lists it will scan */
	double	 n_route;
	/*
	 * Clusters the descent routes to and scores exactly, of which the best
	 * nprobe are then scanned. Equal to nprobe with exact centroid pages,
	 * larger when compressed ones make probe expansion worthwhile.
	 */
	uint32_t k_eff; /* rows it will elect */
	double	 beam;
	/*
	 * Candidates the beam search keeps at each level -- not the tree's
	 * fan-out, which is how many children a node has. A level scores about
	 * beam * fan_out candidates and keeps beam of them.
	 */

	/* page counts derived from those */
	double centroid_pages; /* the whole centroid region */
	bool   flat_tree;	   /* one level: the descent reads all of it */
	double descent_pages;
	double pages_per_list;
	double probed_pages; /* pages of the lists this query will scan */
	double scanned_pl_entries;
	/*
	 * Entries in the posting lists this query will scan -- not the entries
	 * on a page, in one list, or in the whole index. Multiplied by the
	 * per-entry scoring rate, and the pool size when reranking is uncapped.
	 */

	/* the planner's page prices for this index's tablespace */
	double spc_random;
	double spc_seq;

	/* repetitions of this scan inside a nested loop; 1 when not repeated */
	double loop_count;
} CostInputs;

/*
 * Rows the query will pull from the scan, before any filter inflation.
 *
 * limit_tuples is the best answer but is absent more often than it looks:
 * an unfoldable LIMIT $1 leaves it at -1, and so does a LIMIT above the
 * subquery or CTE holding the ORDER BY. The executor finds a bound in both
 * cases, so treating them as unbounded would size the scan for a full
 * ranking. tuple_fraction covers them.
 *
 * Zero means genuinely unbounded, and prism_scan_resolve_top_k sizes for
 * every row the scan could return.
 *
 * Three cases resolve in the planner's favour, since the plan has to be
 * chosen on the planner's information: WITH TIES, where the executor
 * treats the scan as unbounded; a cursor with no LIMIT, where
 * cursor_tuple_fraction sizes k at a tenth of the rows the executor will
 * rank; and an outer LIMIT above a join or CTE, where both agree on
 * unbounded.
 */
static uint32_t
row_target(PlannerInfo *root, double heap_rows)
{
	double rows = -1.0;

	if (root->limit_tuples > 0.0 && isfinite(root->limit_tuples))
		rows = root->limit_tuples;
	else if (root->tuple_fraction >= 1.0 && isfinite(root->tuple_fraction))
		rows = root->tuple_fraction;
	else if (root->tuple_fraction > 0.0)
		rows = root->tuple_fraction * heap_rows;

	if (rows < 1.0)
		return 0;
	if (rows >= (double)PG_UINT32_MAX)
		return PG_UINT32_MAX;

	return (uint32_t)ceil(rows);
}

/*
 * Page fetches one scan is charged for, given that it touches `pages`
 * distinct pages and will be repeated loop_count times.
 *
 * Repeating a scan does not re-read what is still cached, so the fetches
 * are counted across all the repetitions and then pro-rated back to one
 * scan -- genericcostestimate's shape, and the reason loop_count is passed
 * to an AM at all. Without it a parameterized scan is charged full price
 * for every repetition of pages it read on the first.
 */
static double
pages_fetched_per_scan(
		double		 pages,
		BlockNumber	 rel_pages,
		double		 index_pages,
		double		 loop_count,
		PlannerInfo *root)
{
	if (pages <= 0.0)
		return 0.0;

	if (loop_count > 1.0)
		return index_pages_fetched(
					   pages * loop_count, rel_pages, index_pages, root) /
			   loop_count;

	return index_pages_fetched(pages, rel_pages, index_pages, root);
}

/*
 * True when reranking a candidate has to reach the toast relation for its
 * vector rather than reading it from the heap tuple.
 *
 * Needs a toast relation (heap-family tables only) and a column whose
 * attstorage permits out-of-line storage. Where the values actually live
 * then comes from the statistics rather than the declared width, since
 * whether a value went out of line depends on its compressibility, the
 * other columns in the row and the storage mode -- none of which the
 * dimension reveals. Without statistics the declared width is the only
 * evidence available.
 *
 * An expression index reports inline: it can detoast the same way, but the
 * column to look up is not recoverable from indexkeys.
 */
static bool
indexed_column_uses_toast(
		PlannerInfo	 *root,
		RelOptInfo	 *baserel,
		IndexOptInfo *info,
		Dimension	  dim)
{
	if (info->nkeycolumns < 1 || info->indexkeys[0] <= 0)
		return false; /* expression index: nothing to look up */

	AttrNumber attnum = info->indexkeys[0];
	Oid		   reloid = root->simple_rte_array[baserel->relid]->relid;

	HeapTuple reltup = SearchSysCache1(RELOID, ObjectIdGetDatum(reloid));

	if (!HeapTupleIsValid(reltup))
		return false;

	bool has_toast = OidIsValid(
			((Form_pg_class)GETSTRUCT(reltup))->reltoastrelid);

	ReleaseSysCache(reltup);

	if (!has_toast)
		return false;

	/*
	 * Statistics first, because they say where the values *are* rather than
	 * where new ones would go: SET STORAGE does not rewrite existing rows,
	 * so a column switched to PLAIN can still be full of out-of-line values
	 * that cost a toast fetch. A stored width far below the vector's own is
	 * a pointer; the threshold is half the inline width so that a value
	 * compressed but kept in the tuple counts as inline, paying a
	 * decompress but no toast lookup.
	 */
	Size  inline_width = VARHDRSZ + sizeof(int32) + sizeof(float) * dim;
	int32 avgwidth	   = get_attavgwidth(reloid, attnum);

	if (avgwidth > 0)
		return (Size)avgwidth < inline_width / 2;

	/*
	 * No statistics, so fall back on what the column is declared to allow
	 * and how wide a vector of this dimension would be. attstorage is PLAIN
	 * for a type that cannot be toasted and for a column explicitly set that
	 * way; EXTENDED, EXTERNAL and MAIN all permit out-of-line storage,
	 * differing only in whether compression is tried first and how hard the
	 * value is kept inline.
	 */
	HeapTuple atttup = SearchSysCache2(
			ATTNUM, ObjectIdGetDatum(reloid), Int16GetDatum(attnum));

	if (!HeapTupleIsValid(atttup))
		return false;

	char storage = ((Form_pg_attribute)GETSTRUCT(atttup))->attstorage;

	ReleaseSysCache(atttup);

	if (storage == TYPSTORAGE_PLAIN)
		return false;

	return inline_width > TOAST_TUPLE_THRESHOLD;
}

/*
 * Collect the index's shape and the sizing this query will use.
 */
static void
gather_cost_inputs(
		PlannerInfo *root,
		IndexPath	*path,
		double		 loop_count,
		CostInputs	*in,
		double		*selectivity)
{
	IndexOptInfo *info	  = path->indexinfo;
	RelOptInfo	 *baserel = info->rel;

	/*
	 * The index's own description of itself: vector dimension, number of
	 * posting lists, depth and fan-out of the centroid tree, the page format
	 * it was built in, and where the posting region starts. These live on the
	 * metapage and are cached per backend, so this costs one page read for
	 * the first plan in a backend and nothing afterwards. NoLock because the
	 * planner already holds one, asserted below.
	 */
	Relation index = index_open(info->indexoid, NoLock);

	Assert(CheckRelationLockedByMe(index, AccessShareLock, true));

	PrismScanInfo si = prism_cache_scan_info(index);

	index_close(index, NoLock);

	in->dim				= si.dim;
	in->nlevels			= si.nlevels;
	in->has_fastscan	= si.has_fastscan;
	in->centroid_format = si.centroid_format;
	in->nlist			= si.nlist > 0 ? si.nlist : 1;

	/* Rows in the table, as the index and then the relation see them. */
	double N = info->tuples > 0.0 ? info->tuples : baserel->tuples;

	if (N < 1.0)
		N = 1.0;

	/*
	 * Quals the executor applies above the scan do not restrict what the
	 * scan returns -- they discard rows after it has returned them -- so a
	 * selective filter means the scan has to find proportionally more rows
	 * to leave the requested number standing. Hence the top-k is inflated by
	 * the selectivity, not reduced by it.
	 *
	 * indrestrictinfo rather than baserestrictinfo, because a partial index
	 * has already excluded the rows its predicate rejects: those never reach
	 * the scan, so inflating for them would size the top-k for rows the
	 * index does not contain. It is the same list cost_index uses to charge
	 * its own qual costs. A parameterized path also has ppi_clauses applied
	 * above the scan, which appear in neither list, so a lateral scan with a
	 * join filter is under-inflated.
	 */
	double sel = 1.0;

	if (info->indrestrictinfo != NIL)
		sel = clauselist_selectivity(
				root, info->indrestrictinfo, baserel->relid, JOIN_INNER, NULL);

	/*
	 * Rows the scan will actually elect. prism_scan_resolve_top_k applies the
	 * rules the scan applies: the prism.query_limit ceiling, a floor so a tiny
	 * LIMIT still produces a usable search, and a work_mem-derived cap. With
	 * no LIMIT to size from it uses the row count instead.
	 */
	uint32_t k = prism_scan_inflate_for_filter(row_target(root, N), sel);

	in->k_eff = prism_scan_resolve_top_k(k, N);

	/*
	 * Posting lists this query will scan: pinned by prism.nprobe when that is
	 * set, otherwise derived from the index's list count, and never more
	 * lists than exist.
	 */
	uint32_t nprobe = prism_nprobe > 0 ? (uint32_t)prism_nprobe
									   : prism_auto_nprobe(in->nlist);

	if (nprobe > in->nlist)
		nprobe = in->nlist;

	in->nprobe = nprobe;

	/*
	 * Clusters the descent routes to, which is at least nprobe and can be
	 * more: with compressed centroids the beam's ordering is approximate, so
	 * the scan routes a wider set and phase A re-ranks it on exact centroid
	 * distances before scanning the best nprobe. Each routed cluster costs a
	 * head page read whether or not its list is then scanned.
	 */
	in->n_route = (double)prism_query_routed_clusters(
			nprobe, in->nlist, in->centroid_format);

	/*
	 * Centroid slots the beam keeps per level, from the scan's own rule.
	 */
	in->beam = (double)prism_query_beam_width(
			nprobe, in->nlist, si.fan_out, prism_centroid_beam_scale);

	/*
	 * Taken from the metapage, which the build sets and prism_rebalance keeps
	 * current. It cannot be derived from the block range: a split with no
	 * room on a level-0 page chains the new centroid page past the posting
	 * region, so first_posting stops bounding the count.
	 */
	double centroid_pages = (double)si.ncentroid_pages;

	if (centroid_pages < 1.0)
		centroid_pages = 1.0;

	in->centroid_pages = centroid_pages;
	in->flat_tree	   = in->nlevels <= 1;

	/*
	 * Pages the descent reads, which is not the same as slots it examines. A
	 * single-level tree is read in full, beam width not entering into it.
	 * Below a root, one read per kept slot per level is the right shape,
	 * each level following child_blkno from the last level's survivors --
	 * approximate in both directions, since the root is read whole however
	 * wide the beam, and a parent's children can straddle pages. The region
	 * bounds it either way.
	 */
	if (in->flat_tree)
		in->descent_pages = centroid_pages;
	else
		in->descent_pages =
				Min(centroid_pages, (double)in->nlevels * in->beam);

	/*
	 * The posting region is every page past the centroid region, with the
	 * lists taken as evenly sized -- the metapage records where the region
	 * starts, not which pages inside it are reachable rather than free or
	 * retired.
	 *
	 * This errs both ways. Dead entries on live pages are read and scored
	 * like any other, so counting their pages is right; free and retired
	 * pages are never visited, so a badly bloated index is over-charged.
	 * Uneven lists cut both ways too, the probe set being chosen by the
	 * query rather than uniformly.
	 */
	double posting_pages = (double)info->pages - (double)si.first_posting;

	if (posting_pages < 1.0)
		posting_pages = 1.0;

	in->pages_per_list = posting_pages / (double)in->nlist;

	in->probed_pages = (double)nprobe * in->pages_per_list;

	/*
	 * Entries the probed lists hold: a bound, not a prediction, since how
	 * much of a page survives the error-bound gate moves with the data.
	 *
	 * Rows give the tighter of the two available bounds -- a query sees
	 * nprobe of nlist lists' worth, plus a little for replicas. Page
	 * capacity assumes full pages and badly over-counts a sparse index, so
	 * the smaller is taken.
	 */
	double row_entries = (double)nprobe / (double)in->nlist * N *
						 PRISM_ENTRY_REPLICA_FACTOR;
	/*
	 * Whichever layout packs more entries into a page, since this is an
	 * upper bound and an index built in one format can hold pages of the
	 * other. Taking the AoS figure alone under-bounds a fastscan index at
	 * the dimensions where grouped pages hold more.
	 */
	double page_entries = in->probed_pages *
						  (double)prism_posting_max_entries_any_format(
								  in->dim);

	in->scanned_pl_entries = Min(row_entries, page_entries);

	get_tablespace_page_costs(
			info->reltablespace, &in->spc_random, &in->spc_seq);

	in->loop_count = loop_count > 1.0 ? loop_count : 1.0;

	/*
	 * The scan emits k_eff rows; the core applies the qual to them and
	 * arrives at roughly k output rows, charging heap fetches for k_eff.
	 */
	*selectivity = Min(1.0, (double)in->k_eff / N);
}

/*
 * Descending the centroid tree to decide which posting lists to scan. Two
 * parts, measuring different work.
 *
 * The beam search: `beam` is how many candidates the search keeps per
 * level, not the tree's fan-out. Each level scores the children of the
 * kept parents and keeps the best beam again for the level below.
 *
 * Phase A: with compressed centroid pages the beam's ordering is
 * approximate, so it hands down n_route candidates -- nprobe or more --
 * whose full-precision pt_centroids are scored exactly to decide which
 * nprobe actually get scanned.
 *
 * Both are CPU only; the pages these reads touch are priced in
 * search_page_cost.
 */
/*
 * What scoring one centroid costs, which depends on the format its page
 * was built in. FLOAT and HALF pages hold full-precision vectors and are
 * scored with float kernels; RABITQ and FASTSCAN hold quantized codes, the
 * first scored one vector at a time and the second in interleaved groups.
 */
static double
centroid_score_cost(const CostInputs *in)
{
	switch (in->centroid_format)
	{
	case PRISM_CENTROID_FMT_FLOAT:
	case PRISM_CENTROID_FMT_HALF:
		return PRISM_COST_EXACT_DISTANCE(in->dim);
	case PRISM_CENTROID_FMT_FASTSCAN:
		return PRISM_COST_QUANT_DISTANCE(in->dim);
	default:
		return PRISM_COST_QUANT_DISTANCE(in->dim) *
			   PRISM_COST_UNGROUPED_PENALTY;
	}
}

static double
centroid_descent_cost(const CostInputs *in)
{
	/*
	 * The beam scores the children of every slot it keeps, at each level,
	 * bounded by what the pages it reads can hold.
	 */
	double slots = in->descent_pages * (double)prism_centroid_max_entries_fmt(
											   in->dim, in->centroid_format);

	/*
	 * Page capacity assumes full pages, which badly over-counts a small
	 * tree. Wherever the descent reads the whole region it cannot score
	 * more centroids than the tree holds, so nlist is the better bound --
	 * this covers a two-level tree sharing one page as well as a
	 * single-level one. Where only part of the region is read, the pages
	 * read are the bound.
	 *
	 * nlist counts the leaves; internal nodes add 1/fan_out more, which a
	 * wide fan-out makes negligible here.
	 */
	if (in->descent_pages >= in->centroid_pages && slots > (double)in->nlist)
		slots = (double)in->nlist;

	double beam_work = slots * centroid_score_cost(in);

	/*
	 * Phase A then scores one exact distance per routed cluster, against
	 * the full-precision pt_centroid on its head page.
	 */
	double head_work = in->n_route * PRISM_COST_EXACT_DISTANCE(in->dim);

	return beam_work + head_work;
}

/*
 * Scanning the probed posting lists: opening each one, then scoring every
 * entry in it against the query's quantized code.
 */
static double
posting_cpu_cost(const CostInputs *in)
{
	/*
	 * Opening a list builds the query state against the list's centroid --
	 * a handful of O(dim) passes over the query vector -- priced as one
	 * exact distance for the same shape. The head page it reads is counted
	 * with the pages, not here.
	 */
	double list_open = PRISM_COST_EXACT_DISTANCE(in->dim);

	/* One quantized distance per entry; AoS pages are not grouped. */
	double per_entry = PRISM_COST_QUANT_DISTANCE(in->dim);

	if (!in->has_fastscan)
		per_entry *= PRISM_COST_UNGROUPED_PENALTY;

	/*
	 * Every scored entry is offered to the top-k heap, and a bigger heap
	 * costs more per offer, so the rate rises with the logarithm of k.
	 */
	double topk_factor = 1.0;

	if (in->k_eff > PRISM_DEFAULT_K)
		topk_factor += PRISM_COST_TOPK_LOG_COEFF *
					   log2((double)in->k_eff / (double)PRISM_DEFAULT_K);

	return (double)in->nprobe * list_open +
		   in->scanned_pl_entries * per_entry * topk_factor;
}

/*
 * Reading the index pages a search touches: the descent's centroid pages,
 * the routed lists' head pages, and the rest of each probed chain.
 *
 * Priced by how each page is reached rather than uniformly. The three kinds
 * of page a scan touches are reached in three different ways, and pricing
 * them alike would put most of the estimate into seek latency that most of
 * the reads never wait for.
 */
static double
search_page_cost(const CostInputs *in, PlannerInfo *root, IndexOptInfo *info)
{
	/*
	 * A single-level tree's centroid region is read in full and the build
	 * lays it out contiguously before the posting pages, so those reads are
	 * sequential -- less so once a split has chained a centroid page at the
	 * relation's tail. Below a root each level's reads are chosen by the
	 * level above, so they are dependent seeks.
	 */
	double descent_io = pages_fetched_per_scan(
								in->descent_pages,
								info->pages,
								(double)info->pages,
								in->loop_count,
								root) *
						(in->flat_tree ? in->spc_seq : in->spc_random);

	/*
	 * The scan prefetches every routed head page before reading any, so
	 * these are a batch of overlapped reads rather than seeks and are
	 * priced sequentially -- except where effective_io_concurrency is zero
	 * and the storage layer's prefetch is a no-op, which is the same test
	 * the storage layer applies. No cost function consults
	 * effective_io_concurrency, so switching between the two page costs is
	 * the closest the planner's currency allows.
	 */
	double head_io = pages_fetched_per_scan(
							 in->n_route,
							 info->pages,
							 (double)info->pages,
							 in->loop_count,
							 root) *
					 (effective_io_concurrency > 0 ? in->spc_seq
												   : in->spc_random);

	/*
	 * The rest of each probed chain, the head having been charged above.
	 * Prefetching a head does not bring its overflow pages into shared
	 * buffers, so these are real additional reads -- priced sequentially,
	 * the build laying a chain out contiguously and the scan walking it
	 * forward.
	 */
	double chain_pages = (double)in->nprobe * (in->pages_per_list - 1.0);

	if (chain_pages < 0.0)
		chain_pages = 0.0;

	double chain_io = pages_fetched_per_scan(
							  chain_pages,
							  info->pages,
							  (double)info->pages,
							  in->loop_count,
							  root) *
					  in->spc_seq;

	return descent_io + head_io + chain_io;
}

/*
 * Reranking: settling the top-k on exact distances, and the costliest part
 * of a scan. The posting scan only ever scores quantized codes, so the
 * result is decided by fetching the most promising candidates' full
 * vectors and measuring those -- a heap fetch, a detoast where the vector
 * is out of line, and one exact distance each.
 *
 * The pool has three modes. Reranking off costs nothing. A capped pool
 * uses the estimator's own number. An uncapped pool reranks every survivor
 * of the error-bound gate, and prism_query_rerank_pool_estimate reports that
 * as 0 -- the value the extract step reads as "keep them all", so it must
 * not be taken at face value: here 0 is the widest pool there is. It is
 * charged at every entry the scan scores, an over-estimate, for a setting
 * asking for the slowest and most accurate scan.
 *
 * A capped pool also has a floor the planner cannot see, a fraction of the
 * candidates actually found. Where it binds, this under-estimates.
 *
 * Known error: out-of-line vectors leave a heap of pointers, and
 * baserel->pages is all the planner has -- the toast relation carries no
 * statistics. So such a column is charged for a small heap plus the
 * per-candidate detoast, while the same vectors inline are charged for the
 * large heap they make. At scale the detoast term dominates and the
 * ordering is right; on a small table it is backwards, and a sequential
 * scan over a toasted column can win an estimate it loses badly in
 * practice.
 */
static double
rerank_cost(
		const CostInputs *in,
		PlannerInfo		 *root,
		RelOptInfo		 *baserel,
		IndexOptInfo	 *info)
{
	if (!prism_rerank)
		return 0.0;

	uint32_t capped = prism_query_rerank_pool_estimate(in->k_eff, in->nprobe);
	double	 pool	= capped > 0 ? (double)capped : in->scanned_pl_entries;

	bool external = indexed_column_uses_toast(root, baserel, info, in->dim);
	/*
	 * Reaching one tuple is what cpu_tuple_cost prices; scoring it is one
	 * exact distance over the vector it holds.
	 */
	double per_fetch = cpu_tuple_cost + PRISM_COST_EXACT_DISTANCE(in->dim);

	if (external)
		per_fetch *= PRISM_COST_FETCH_DETOAST;

	/*
	 * Every pool candidate is scored exactly, and where the vectors are out
	 * of line every one of them is also detoasted. That is all index work:
	 * nothing above the scan knows the pool exists.
	 */
	double cpu = pool * per_fetch;

	/*
	 * An index AM does not charge for reaching parent-table rows --
	 * cost_index does that from the selectivity reported here, covering
	 * k_eff tuples. But the pool is wider than k_eff, and those extra reads
	 * are invisible from outside, so they are what this term charges.
	 *
	 * The subtraction has to happen in pages, not candidates:
	 * Mackert-Lohman saturates at the heap's size, so on a small heap the
	 * pool already touches every page and removing candidates removes no
	 * pages, billing cost_index's survivors a second time in full. Pricing
	 * the difference between the pool's pages and the survivors' leaves
	 * only what cost_index has not already paid for.
	 *
	 * Both rerank paths sort into TID order and, on a heap-AM table, fetch
	 * through a streaming read, so these run forward with the gaps' latency
	 * hidden -- the same situation as the routed head pages and priced the
	 * same way.
	 */
	double pool_pages = pages_fetched_per_scan(
			pool, baserel->pages, (double)info->pages, in->loop_count, root);
	double charged_pages = pages_fetched_per_scan(
			(double)in->k_eff,
			baserel->pages,
			(double)info->pages,
			in->loop_count,
			root);
	double extra_pages = pool_pages - charged_pages;

	if (extra_pages < 0.0)
		extra_pages = 0.0;

	double io = extra_pages *
				(effective_io_concurrency > 0 ? in->spc_seq : in->spc_random);

	return cpu + io;
}

void
prism_cost_estimate(
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
	CostInputs	  in;
	double		  sel;

	gather_cost_inputs(root, path, loop_count, &in, &sel);

	double per_scan_cost = PRISM_COST_SCAN_SETUP + centroid_descent_cost(&in) +
						   posting_cpu_cost(&in) +
						   search_page_cost(&in, root, info) +
						   rerank_cost(&in, root, baserel, info);

	/*
	 * All of it is startup work: the scan elects its whole top-k before it
	 * can return the first tuple, so a LIMIT must not discount it.
	 *
	 * The figure is for one scan. cost_nestloop applies loop_count itself,
	 * so multiplying here too would charge the square of the outer row
	 * count; loop_count is used for the cache argument instead, the way
	 * genericcostestimate uses it.
	 *
	 * In practice it is always 1 here: match_clause_to_ordering_op only
	 * accepts an ORDER BY operator whose other operand contains no Var, so
	 * a distance against another relation's column never yields a
	 * parameterized path, and the lateral form's PARAM_EXEC sits inside a
	 * subquery whose own planner sees 1. The handling is what the contract
	 * specifies.
	 */
	*startup_cost = per_scan_cost;
	*total_cost	  = *startup_cost + (double)in.k_eff * cpu_index_tuple_cost;

	*selectivity = sel;

	/* Nothing orders a vector scan's output by heap position. */
	*correlation = 0.0;

	/* The pages search_page_cost prices, so a parallel plan divides the
	 * same count this estimate was built from. */
	*index_pages = in.descent_pages + in.n_route +
				   (double)in.nprobe * (in.pages_per_list - 1.0);
}
