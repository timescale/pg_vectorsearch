/*
 * cost.h - Constants for the vector index scan cost estimate
 *
 * Quantities PostgreSQL exposes no cost parameter for.
 */

#ifndef MKT_PG_COST_H
#define MKT_PG_COST_H

#include <postgres.h>

#include <nodes/pathnodes.h>

/*
 * Dimensions of distance work bought by one operator evaluation: eight of
 * float32 multiply-add, or sixty-four of quantized code, a code being one
 * bit per dimension consumed by popcount over machine words.
 */
#define MKT_COST_DIMS_PER_OP	   8.0
#define MKT_COST_DIMS_PER_QUANT_OP 64.0

/* One exact distance over a dim-dimensional float32 vector. */
#define MKT_COST_EXACT_DISTANCE(dim) \
	(((double)(dim) / MKT_COST_DIMS_PER_OP) * cpu_operator_cost)

/* One quantized distance against one entry's code. */
#define MKT_COST_QUANT_DISTANCE(dim) \
	(((double)(dim) / MKT_COST_DIMS_PER_QUANT_OP) * cpu_operator_cost)

/*
 * Codes scored one vector at a time rather than in interleaved groups: an
 * AoS posting page, or a RABITQ-format centroid page. Same bytes as the
 * grouped layout, but more cache lines touched and no group kernel.
 */
#define MKT_COST_UNGROUPED_PENALTY 2.0

/*
 * Per-scan allocations and context setup: about one sequential page fetch
 * at default settings. Without it the estimate approaches zero on a tiny
 * index, where every other term is near zero too.
 */
#define MKT_COST_SCAN_SETUP_OPS 400.0
#define MKT_COST_SCAN_SETUP		(MKT_COST_SCAN_SETUP_OPS * cpu_operator_cost)

/*
 * Top-k maintenance, as a multiplier on per-entry scoring: every scored
 * entry is offered to the heap, so the rate rises with log k. Under-prices
 * a large k, where candidate survival rather than heap maintenance
 * dominates.
 */
#define MKT_COST_TOPK_LOG_COEFF 0.2

/*
 * Reranking a value stored out of line, as a multiple of the inline cost.
 * Standing in for a descent of the toast index, a chunk tuple per two
 * kilobytes of vector, and a decompress of the whole value -- against an
 * inline candidate's one tuple and one distance. A judgement, and charged
 * wholly in CPU currency although much of what it stands for is page
 * access; there are no statistics on a toast relation to price those with.
 */
#define MKT_COST_FETCH_DETOAST 6.0

/*
 * Replicas SOAR and boundary assignment create, so a probed list holds
 * somewhat more entries than the rows it covers.
 */
#define MKT_ENTRY_REPLICA_FACTOR 1.1

void prism_cost_estimate(
		PlannerInfo *root,
		IndexPath	*path,
		double		 loop_count,
		Cost		*startup_cost,
		Cost		*total_cost,
		Selectivity *selectivity,
		double		*correlation,
		double		*index_pages);

#endif /* MKT_PG_COST_H */
