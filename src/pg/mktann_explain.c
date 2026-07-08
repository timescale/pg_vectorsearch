/*
 * mktann_explain.c - EXPLAIN ANALYZE output for mktann scans
 *
 * Injects scan stats into EXPLAIN (ANALYZE, VERBOSE) output via
 * PG's explain_per_node_hook. Shows centroid search, posting scan,
 * and rerank statistics for mktann index scans.
 */

#include <postgres.h>

#include <commands/defrem.h>
#include <commands/explain.h>
#include <commands/explain_format.h>
#include <commands/explain_state.h>
#include <nodes/execnodes.h>
#include <utils/lsyscache.h>
#include <utils/rel.h>

#include "mktann_explain.h"
#include "mktann_scan.h"

static explain_per_node_hook_type prev_hook = NULL;

static void
mktann_explain_hook(
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
	if (am_name == NULL || strcmp(am_name, "mktann") != 0)
	{
		if (am_name != NULL)
			pfree(am_name);
		return;
	}
	pfree(am_name);

	const MktannScanStats *stats = mktann_scan_get_stats(scan);
	if (stats == NULL)
		return;

	ExplainOpenGroup("Mktann", "Mktann", true, es);

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
				(double)stats->centroid_ns / 1e6,
				3,
				es);
		ExplainPropertyFloat(
				"Posting Scan Time",
				"ms",
				(double)stats->posting_ns / 1e6,
				3,
				es);
		ExplainPropertyFloat(
				"Rerank Time", "ms", (double)stats->rerank_ns / 1e6, 3, es);
	}

	ExplainCloseGroup("Mktann", "Mktann", true, es);
}

void
mktann_explain_init(void)
{
	prev_hook			  = explain_per_node_hook;
	explain_per_node_hook = mktann_explain_hook;
}
