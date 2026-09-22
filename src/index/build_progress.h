/*
 * build_progress.h - Canonical index-build phase model
 *
 * One ordered set of build phases shared by the serial and parallel builds
 * (and the standalone build), so every driver reports the same phases
 * consistently.
 *
 * The values are the subphase numbers reported to PostgreSQL's
 * pg_stat_progress_create_index via PROGRESS_CREATEIDX_SUBPHASE; value 1
 * matches PostgreSQL's PROGRESS_CREATEIDX_SUBPHASE_INITIALIZE. Values 2-9 are
 * frozen (older releases and the build_progress isolation test depend on
 * them); new phases append at 10+. This header is back-end-neutral (no
 * PostgreSQL or standalone includes) so it is usable from the shared build
 * driver.
 */

#ifndef MKT_BUILD_PROGRESS_H
#define MKT_BUILD_PROGRESS_H

#include <stdbool.h>
#include <stdint.h>

#define MKT_BUILD_PHASE_INITIALIZE \
	1 /* = PROGRESS_CREATEIDX_SUBPHASE_INITIALIZE */
#define MKT_BUILD_PHASE_SAMPLE		  2
#define MKT_BUILD_PHASE_KMEANS		  3 /* root / flat / serial k-means */
#define MKT_BUILD_PHASE_SETUP		  4
#define MKT_BUILD_PHASE_SCAN		  5 /* serial posting scan */
#define MKT_BUILD_PHASE_POSTING		  6 /* posting finalize */
#define MKT_BUILD_PHASE_CENTROID	  7
#define MKT_BUILD_PHASE_WAL			  8
#define MKT_BUILD_PHASE_SCAN_PARALLEL 9	 /* parallel posting scan / drain */
#define MKT_BUILD_PHASE_SUBTREES	  10 /* phase 2c subtree build */
#define MKT_BUILD_PHASE_REFINE		  11 /* full-table leaf refinement */

/* Highest valid phase value (for range checks and exhaustive iteration). */
#define MKT_BUILD_PHASE_MAX 11

/*
 * Human-readable name for a build phase, identical under PostgreSQL and
 * standalone. Returns NULL for a value outside [1, MKT_BUILD_PHASE_MAX].
 */
const char *mkt_build_phase_name(int phase);

/* ----------------------------------------------------------------
 * Build-progress reporting seam
 *
 * One reporting context the serial and parallel build drivers thread through,
 * so both report phases/progress identically. Constructed (stack) and used
 * only by the leader/serial driver — workers never touch it. The bodies are
 * back-end specific: PostgreSQL (src/pg/build_progress.c) updates
 * pg_stat_progress_create_index and emits ResetUsage/ShowUsage + summary lines
 * under the prism.log_build_stats GUC; standalone (src/index/build_progress.c)
 * is a no-op stub for now. Types are back-end-neutral (no PG/standalone
 * headers).
 * ---------------------------------------------------------------- */

struct MktBuildStats; /* index/index_build.h */

typedef struct MktBuildProgress
{
	int		 cur_phase;		 /* phase currently running */
	int64_t	 phase_start_ns; /* monotonic start of cur_phase */
	int64_t	 build_start_ns; /* monotonic start of the whole build */
	bool	 log_stats;		 /* prism.log_build_stats */
	bool	 is_parallel;	 /* selects SCAN vs SCAN_PARALLEL labeling */
	bool	 total_set;		 /* TUPLES_TOTAL already published */
	void	*heap_ctx;	/* PG: MemoryContext for MemoryContextMemAllocated */
	uint64_t dsm_bytes; /* PG parallel: committed DSM total; else 0 */
	double	 tuples_total;		 /* real estimated row count, for the % */
	struct MktBuildStats *stats; /* per-phase ms accumulator; may be NULL */
} MktBuildProgress;

/* Start a build's introspection: baseline timers + initial phase. */
void mkt_build_progress_begin(
		MktBuildProgress	 *p,
		bool				  is_parallel,
		bool				  log_stats,
		void				 *heap_ctx,
		struct MktBuildStats *stats,
		double				  tuples_total);

/* Transition to `phase`: report the previous phase's elapsed/usage (when
 * logging), accumulate it into stats, then set the live phase label. */
void mkt_build_report_phase(MktBuildProgress *p, int phase);

/* Update the current scan phase's progress (publishes TUPLES_TOTAL once). */
void mkt_build_report_progress(MktBuildProgress *p, double done);

/*
 * Batched mid-scan advance of the tuples-done counter, callable from any
 * participant: a parallel worker's increment piggybacks to the leader over
 * the parallel message queue and is applied even while the leader blocks at
 * a barrier. The leader still publishes the exact final count when a scan
 * finishes; these keep the view moving while it runs.
 */
void mkt_build_progress_incr_tuples(int64_t n);

/* Record the committed DSM size (parallel) so per-phase memory lines can add
 * it. */
void mkt_build_report_dsm_bytes(MktBuildProgress *p, uint64_t dsm_bytes);

/* One always-on LOG line naming the planned large allocations (pre-explains an
 * OOM); sizes in bytes, 0 to omit a field. */
void mkt_build_report_planned_alloc(
		MktBuildProgress *p,
		uint64_t		  sample_bytes,
		uint64_t		  centroid_tree_bytes,
		uint64_t		  dsm_total_bytes);

/* Terminal: flush the last phase's timing and emit the final summary. */
void mkt_build_progress_end(MktBuildProgress *p);

#endif /* MKT_BUILD_PROGRESS_H */
