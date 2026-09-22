/*
 * scan_bound.c - row target for a prism scan
 *
 * The index AM API never tells a scan how many rows the query wants: the
 * executor pulls one tuple at a time until the Limit node above it is
 * satisfied. A prism scan computes its whole top-k on the first fetch,
 * so it has to know k up front or it emits a fixed default and a larger
 * LIMIT silently comes up short.
 *
 * Knowing k up front is also what makes the rows it returns correctly
 * ordered: the whole top-k is elected in one pass over the probed
 * clusters, so every row is ranked against every candidate. A scan that
 * instead resumed once its budget ran out could only rank the newcomers
 * against each other, and might find one nearer than a row it has already
 * returned -- ordering the caller cannot rely on.
 *
 * A scan therefore asks, at rescan, whether it runs under a Limit. The
 * hook below only records which query is executing; the search runs
 * inside the scan, where the executor node already points at it and can
 * be matched by identity rather than by position in the plan tree.
 *
 * When the scan carries a filter (a WHERE clause the executor applies
 * above the index), only a fraction of the emitted top-k survives it, so
 * k is inflated by the planner's selectivity estimate.
 */

#include <postgres.h>

#include <access/genam.h>
#include <executor/executor.h>
#include <math.h>
#include <nodes/execnodes.h>
#include <nodes/nodeFuncs.h>
#include <nodes/plannodes.h>
#include <optimizer/optimizer.h>
#include <utils/rel.h>

#include "scan.h"
#include "scan_bound.h"
#include "support_pg.h"

static ExecutorRun_hook_type prev_ExecutorRun_hook = NULL;

/*
 * The query we are executing inside, or NULL when no executor is running
 * (maintenance code opening its own index scans, say). Saved and restored
 * around the run, the way the executor maintains ActivePortal: nesting
 * follows the C stack and names the innermost query, and an error unwinds
 * this along with the executor.
 */
static QueryDesc *mkt_active_query_desc = NULL;

/*
 * True if the expression reads an executor parameter that has no value yet.
 *
 * A PARAM_EXEC is a parameter whose value another plan node produces during
 * execution -- an InitPlan's result, or the current row of the outer
 * relation in a nested loop -- as opposed to a PARAM_EXTERN, which the
 * client binds before execution begins. Until the node that owes it has
 * run, its slot carries the subplan to run rather than a value (execPlan is
 * non-NULL), and evaluating it would execute that subplan from inside a
 * rescan. Decline in that case.
 *
 * Params already produced are fine, which is what asking at rescan buys
 * over asking at executor start.
 */
static bool
has_pending_exec_param(Node *node, void *context)
{
	if (node == NULL)
		return false;

	if (IsA(node, Param))
	{
		Param		*p		  = (Param *)node;
		ExprContext *econtext = (ExprContext *)context;

		if (p->paramkind != PARAM_EXEC)
			return false;
		return econtext == NULL || econtext->ecxt_param_exec_vals == NULL ||
			   p->paramid < 0 ||
			   econtext->ecxt_param_exec_vals[p->paramid].execPlan != NULL;
	}
	return expression_tree_walker(node, has_pending_exec_param, context);
}

/*
 * Evaluate a LIMIT or OFFSET expression.
 *
 * OFFSET counts toward what the scan must produce: the Limit node discards
 * the skipped rows only after the scan has emitted them, so LIMIT 10 OFFSET
 * 100 needs 110 rows out of the index, not 10.
 *
 * Only values fixed for the rest of this scan qualify: constants,
 * parameters already bound, and stable expressions over them. False when
 * the value cannot be known here, or is NULL (which SQL reads as "no
 * limit").
 */
static bool
eval_count_expr(
		Node *expr, ExprState *state, ExprContext *econtext, int64 *value)
{
	if (expr == NULL || state == NULL)
		return false;
	if (has_pending_exec_param(expr, econtext))
		return false;
	if (contain_volatile_functions(expr))
		return false;

	bool  isnull;
	Datum d = ExecEvalExprSwitchContext(state, econtext, &isnull);
	if (isnull)
		return false;

	*value = DatumGetInt64(d);
	return *value >= 0;
}

/*
 * Descend from a Limit's input to the IndexScan feeding it, through nodes
 * that emit exactly one row per input row. Anything that can filter,
 * aggregate, sort or multiply rows breaks the correspondence between the
 * LIMIT and the rows the scan must produce, so the descent stops there.
 */
static IndexScanState *
find_index_scan_state(PlanState *ps)
{
	while (ps != NULL)
	{
		if (IsA(ps, IndexScanState))
			return (IndexScanState *)ps;

		/* A qual on any node between here and the scan filters rows. */
		if (ps->plan->qual != NIL)
			return NULL;

		switch (nodeTag(ps))
		{
		case T_ResultState:
			ps = outerPlanState(ps);
			break;

		case T_WindowAggState:
			if (((WindowAgg *)ps->plan)->runCondition != NIL)
				return NULL;
			ps = outerPlanState(ps);
			break;

		case T_SubqueryScanState:
			ps = ((SubqueryScanState *)ps)->subplan;
			break;

		default:
			return NULL;
		}
	}
	return NULL;
}

/*
 * A LIMIT's count and offset are each clamped to this before being added,
 * so the sum cannot overflow. Purely an arithmetic guard: what limits a
 * scan is work_mem, applied where the top-k is allocated.
 */
#define MKT_LIMIT_SUM_MAX (PG_INT64_MAX / 4)

/*
 * Standard deviations of headroom over the rows a filter is expected to
 * leave.
 *
 * A filter that passes a fraction s of the rows needs about 1/s candidates
 * for s of them to add up to the LIMIT -- but that is a mean, not a
 * guarantee: with a correct s, the survivors among the top-(k/s) are
 * distributed Binomial(k/s, s), so the count has mean k and standard
 * deviation about sqrt(k). Over-fetching by 1 + this/sqrt(k) covers that
 * noise: roughly 1.95x at k = 10, 1.3x at k = 100, 1.09x at k = 1000. A flat
 * factor would have to be sized for the smallest k and would then over-fetch
 * by 1.5x to 2.7x at the larger ones, where the fetches cost most.
 *
 * This covers noise, not a wrong estimate. An s off by an order of magnitude
 * needs an order of magnitude more candidates, which nothing here can
 * anticipate -- raising work_mem, or a partial index on the filter, is the
 * answer to that.
 */
#define MKT_FILTER_NOISE_SIGMAS 3.0

/*
 * Smallest estimated selectivity the sizing will divide by.
 *
 * One row in ten thousand already asks for a top-k ten thousand times the
 * LIMIT, which work_mem will cut down anyway. Anything below this is a
 * planner estimate with no rows behind it -- a default from a missing
 * statistic, or a conjunction of independent guesses -- and dividing by it
 * produces a number, not an estimate.
 */
#define MKT_FILTER_MIN_SELECTIVITY 1e-4

/*
 * Size the scan's top-k for the rows the LIMIT will pull.
 *
 * A filter on the scan is applied by the executor after the index has emitted
 * its top-k, so only an estimated fraction s of the emitted rows survive:
 * multiply the bound by 1/s, with a margin, so that fraction still covers
 * the LIMIT. The estimate is the plan's own row
 * count for the scan node against the relation's statistics; without
 * statistics the plain LIMIT is used, and where the sizing exceeds what
 * work_mem affords a filtered query comes up short.
 *
 * The estimate is only ever a guess -- a filter correlated with vector
 * proximity (a category living in its own region of the space) can still
 * starve the result, and nothing here can know which rows survive, because
 * the pruning gate commits this size before any heap fetch and evaluating a
 * qual needs one. prism.query_limit is the lever for a caller who knows their
 * own selectivity.
 */
static uint32_t
size_top_k(IndexScanState *iss, int64 limit)
{
	uint32_t k	  = (uint32_t)Min(Max(limit, 1), (int64)PG_UINT32_MAX);
	Plan	*plan = iss->ss.ps.plan;
	Relation heap = iss->ss.ss_currentRelation;

	if (plan->qual == NIL || heap == NULL || heap->rd_rel->reltuples <= 0.0)
		return k;

	/*
	 * A row estimate is a planner guess divided by a cached count, so it can
	 * arrive at anything: zero rows from a qual the planner reads as
	 * impossible, more rows than the relation is recorded as holding after a
	 * bulk load, a non-finite value from either being garbage. Only a
	 * fraction strictly inside (0, 1) says anything about filtering; take
	 * the LIMIT unchanged for the rest rather than dividing by them.
	 *
	 * MKT_FILTER_MIN_SELECTIVITY floors the divisor. Without it a plan_rows
	 * of a millionth of a row asks for a top-k a million times the LIMIT,
	 * and while work_mem would refuse to allocate it, the sizing has no
	 * business proposing it: below this the estimate is noise, not a
	 * measurement.
	 */
	return mkt_scan_inflate_for_filter(
			k, plan->plan_rows / heap->rd_rel->reltuples);
}

/*
 * Inflate a row target for a filter the executor applies above the scan.
 *
 * Called by the executor and by the cost model. The executor arrives at the
 * selectivity by dividing its own plan_rows by the relation's row count;
 * the planner passes clauselist_selectivity's fraction for the same thing.
 */
uint32_t
mkt_scan_inflate_for_filter(uint32_t k, double selectivity)
{
	if (k == 0)
		return 0;

	/*
	 * A row estimate is a planner guess divided by a cached count, so it can
	 * arrive at anything: zero rows from a qual the planner reads as
	 * impossible, more rows than the relation is recorded as holding after a
	 * bulk load, a non-finite value from either being garbage. Only a
	 * fraction strictly inside (0, 1) says anything about filtering; take
	 * the target unchanged for the rest rather than dividing by them.
	 *
	 * MKT_FILTER_MIN_SELECTIVITY floors the divisor. Without it a plan_rows
	 * of a millionth of a row asks for a top-k a million times the target,
	 * and while work_mem would refuse to allocate it, the sizing has no
	 * business proposing it: below this the estimate is noise, not a
	 * measurement.
	 */
	if (!isfinite(selectivity) || selectivity >= 1.0 || selectivity <= 0.0)
		return k;
	if (selectivity < MKT_FILTER_MIN_SELECTIVITY)
		selectivity = MKT_FILTER_MIN_SELECTIVITY;

	double margin = 1.0 + MKT_FILTER_NOISE_SIGMAS / sqrt((double)k);
	double sized  = ceil(margin * (double)k / selectivity);

	/*
	 * Clamped only to keep the cast well defined. The ceiling that matters
	 * is work_mem, applied by mkt_scan_resolve_top_k.
	 */
	return (uint32_t)Min(Max(sized, (double)k), (double)PG_UINT32_MAX);
}

/*
 * Rows a Limit will pull: its count plus its offset, each clamped so the
 * sum cannot overflow. False when either cannot be resolved now, when the
 * count is zero, or under WITH TIES -- which keeps pulling past the count
 * for rows tying the last one, so the count is a floor on the rows needed
 * rather than the number of them, and a scan is better left at its default
 * sizing than cut ties off.
 */
static bool
limit_total_rows(LimitState *ls, int64 *total)
{
	Limit		*plan	  = (Limit *)ls->ps.plan;
	ExprContext *econtext = ls->ps.ps_ExprContext;
	int64		 count	  = 0;
	int64		 offset	  = 0;

	/*
	 * WITH TIES pulls past its count for rows tying the last one, so the
	 * count is a floor rather than the total -- but it is the right thing to
	 * size from. Reporting no bound would have resolve_top_k read the query
	 * as asking for every row, and size FETCH FIRST 5 ROWS WITH TIES for the
	 * whole relation. A tie group straddling the top-k's edge can be cut
	 * short instead, which is the same bound every other query is subject to
	 * and vastly cheaper than ranking the table.
	 */
	if (!eval_count_expr(plan->limitCount, ls->limitCount, econtext, &count) ||
		count <= 0)
		return false;
	if (plan->limitOffset != NULL &&
		!eval_count_expr(
				plan->limitOffset, ls->limitOffset, econtext, &offset))
		return false;

	*total = Min(count, MKT_LIMIT_SUM_MAX) + Min(offset, MKT_LIMIT_SUM_MAX);
	return true;
}

/* The scan looking for its Limit; k stays 0 until one is resolved. */
typedef struct LimitSearch
{
	IndexScanDesc scan;
	uint32_t	  k;
} LimitSearch;

/*
 * Look for a Limit whose input is the asking scan. The pairing is by
 * identity -- nodeIndexscan assigns iss_ScanDesc before calling
 * index_rescan -- which is what tells two scans of the same index in one
 * statement apart, where their position in the plan tree cannot.
 */
static bool
limit_walker(PlanState *ps, void *context)
{
	LimitSearch *search = (LimitSearch *)context;

	if (ps == NULL)
		return false;

	if (IsA(ps, LimitState))
	{
		IndexScanState *iss = find_index_scan_state(outerPlanState(ps));
		int64			total;

		if (iss != NULL && iss->iss_ScanDesc == search->scan &&
			iss->iss_NumOrderByKeys > 0 &&
			limit_total_rows((LimitState *)ps, &total))
		{
			search->k = size_top_k(iss, total);
			return true; /* this scan's Limit is resolved */
		}
	}

	return planstate_tree_walker(ps, limit_walker, context);
}

uint32_t
mkt_scan_bound(IndexScanDesc scan)
{
	QueryDesc *qd = mkt_active_query_desc;

	if (qd == NULL || qd->planstate == NULL || qd->estate == NULL)
		return 0; /* no executor above us: keep the default sizing */

	LimitSearch search = {.scan = scan, .k = 0};

	/*
	 * The main tree, then the plan trees hanging off the EState: CTE
	 * bodies, InitPlans and correlated subplans are not reachable from the
	 * root, and a CTE with its own ORDER BY ... LIMIT is the hybrid-search
	 * shape this exists for.
	 */
	if (!limit_walker(qd->planstate, &search))
	{
		ListCell *lc;

		foreach (lc, qd->estate->es_subplanstates)
			if (limit_walker((PlanState *)lfirst(lc), &search))
				break;
	}

	return search.k;
}

/*
 * Record which query is executing, for the duration of its execution.
 *
 * An index AM's callbacks are handed a Relation and an IndexScanDesc and
 * nothing else -- there is no path from a scan back to the executor state
 * it runs under. Noting the running query here is what lets mkt_scan_bound
 * find the plan tree at rescan and, in it, the Limit above this scan.
 *
 * Save, set, restore, exactly as the executor does for ActivePortal.
 * PG_FINALLY does the restoring so that an error -- which leaves this frame
 * by longjmp, not by return -- cannot strand a pointer to a QueryDesc whose
 * memory has gone. Nesting needs no bookkeeping: a query issued from a
 * function body runs its executor inside its caller's, so the C stack keeps
 * the saves in order and the variable always names the innermost query.
 */
static void
prism_executor_run(QueryDesc *queryDesc, ScanDirection direction, uint64 count)
{
	QueryDesc *save = mkt_active_query_desc;

	mkt_active_query_desc = queryDesc;
	PG_TRY();
	{
		if (prev_ExecutorRun_hook)
			prev_ExecutorRun_hook(queryDesc, direction, count);
		else
			standard_ExecutorRun(queryDesc, direction, count);
	}
	PG_FINALLY();
	{
		mkt_active_query_desc = save;
	}
	PG_END_TRY();
}

void
mkt_scan_bound_init(void)
{
	prev_ExecutorRun_hook = ExecutorRun_hook;
	ExecutorRun_hook	  = prism_executor_run;
}
