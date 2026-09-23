/*
 * explain.c - EXPLAIN ANALYZE output for prism scans
 *
 * Injects scan stats into EXPLAIN (ANALYZE, VERBOSE) output via
 * PG's explain_per_node_hook. Shows centroid search, posting scan,
 * and rerank statistics for prism index scans.
 */

#include <postgres.h>

#include "vs_config.h"

#include <commands/defrem.h>
#include <commands/explain.h>
#include <commands/explain_format.h>
#include <commands/explain_state.h>
#include <nodes/execnodes.h>
#include <utils/lsyscache.h>
#include <utils/rel.h>

#include "core/platform.h"
#include "explain.h"
#include "scan.h"

static explain_per_node_hook_type prev_hook = NULL;

static void
prism_explain_hook(
		PlanState	 *planstate,
		List		 *ancestors,
		const char	 *relationship,
		const char	 *plan_name,
		ExplainState *es)
{
	if (prev_hook)
		prev_hook(planstate, ancestors, relationship, plan_name, es);

	if (!es->analyze || !es->verbose)
		return;

	if (!IsA(planstate, IndexScanState))
		return;

	IndexScanState *iss	 = (IndexScanState *)planstate;
	IndexScanDesc	scan = iss->iss_ScanDesc;
	if (scan == NULL)
		return;

	Relation rel	 = scan->indexRelation;
	char	*am_name = get_am_name(rel->rd_rel->relam);
	if (am_name == NULL || strcmp(am_name, VS_AM_NAME) != 0)
	{
		if (am_name != NULL)
			pfree(am_name);
		return;
	}
	pfree(am_name);

	const PrismScanStats *stats = prism_scan_get_stats(scan);
	if (stats == NULL)
		return;

	ExplainOpenGroup("Prism", "Prism", true, es);

	ExplainPropertyInteger("Top-K", NULL, stats->top_k, es);
	ExplainPropertyInteger(
			"Posting Lists Scanned", NULL, stats->clusters_scanned, es);
	ExplainPropertyInteger(
			"Centroid Pages Read", NULL, stats->centroid_pages_read, es);
	ExplainPropertyInteger(
			"Posting Pages Read", NULL, stats->posting_pages_read, es);
	ExplainPropertyInteger(
			"Posting Pages Scanned",
			NULL,
			stats->posting_pages_read - stats->posting_pages_skipped,
			es);
	ExplainPropertyInteger(
			"Posting Dead Pages Skipped",
			NULL,
			stats->posting_pages_skipped,
			es);
	ExplainPropertyInteger(
			"Posting Entries Scanned",
			NULL,
			stats->posting_entries_scanned,
			es);
	ExplainPropertyInteger(
			"Rerank Candidates", NULL, stats->rerank_candidates, es);
	ExplainPropertyInteger("Rerank Results", NULL, stats->rerank_results, es);
	ExplainPropertyInteger("Storage Reads", NULL, stats->storage_reads, es);

	/* Per-phase wall time (ms). Emitted only under EXPLAIN (ANALYZE,
	 * VERBOSE, TIMING ON) so regression output stays stable without it. */
	if (es->timing)
	{
		ExplainPropertyFloat(
				"Centroid Search Time",
				"ms",
				(double)stats->centroid_ns / VS_NS_PER_MS,
				3,
				es);
		ExplainPropertyFloat(
				"Posting Scan Time",
				"ms",
				(double)stats->posting_ns / VS_NS_PER_MS,
				3,
				es);
		ExplainPropertyFloat(
				"Rerank Time",
				"ms",
				(double)stats->rerank_ns / VS_NS_PER_MS,
				3,
				es);
	}

	ExplainCloseGroup("Prism", "Prism", true, es);
}

void
prism_explain_init(void)
{
	prev_hook			  = explain_per_node_hook;
	explain_per_node_hook = prism_explain_hook;
}
