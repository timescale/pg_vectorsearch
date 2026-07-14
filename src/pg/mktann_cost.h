/*
 * mktann_cost.h - Planner cost model for mktann scans
 *
 * One shared model serves amcostestimate (serial paths) and the
 * set_rel_pathlist_hook re-coster (partial paths). The AM does all of
 * its work before returning the first tuple, so the honest shape is
 * all-startup; stock cost_index cannot express "startup work divides
 * across parallel workers", hence the hook.
 */

#ifndef MKTANN_COST_H
#define MKTANN_COST_H

#include <postgres.h>

#include <nodes/pathnodes.h>

/* Per-phase costs of one mktann search at the session's GUCs. */
typedef struct MktannCosts
{
	Cost descent;		 /* centroid descent + probe re-rank (serial part) */
	Cost scan;			 /* posting-cluster scan (divides across workers) */
	Cost rerank;		 /* exact rerank of the candidate pool (divides) */
	Cost io;			 /* storage reads for non-resident posting pages
						  * (divides across workers; ~0 when the index is
						  * resident in shared buffers) */
	double		emitted; /* rows the scan returns (the rerank pool) */
	double		index_pages;
	Selectivity selectivity;
	bool		unreadable; /* on-disk format this build cannot read;
							 * the path must be disabled, not costed */
} MktannCosts;

/* Compute the model for one index at the current GUCs (opens the index
 * relation with AccessShareLock). */
MktannCosts mktann_compute_costs(Oid indexoid);

/* amcostestimate implementation (serial path costing). */
void mktann_costestimate(
		struct PlannerInfo *root,
		struct IndexPath   *path,
		double				loop_count,
		Cost			   *startup_cost,
		Cost			   *total_cost,
		Selectivity		   *selectivity,
		double			   *correlation,
		double			   *index_pages);

/* Install the set_rel_pathlist hook that re-costs partial mktann index
 * paths with the scan and rerank phases divided across the chosen
 * worker count. Called from _PG_init. */
void mktann_cost_register_hook(void);

#endif /* MKTANN_COST_H */
