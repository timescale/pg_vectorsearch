/*
 * build_progress.c - PostgreSQL body of the build-progress reporting seam.
 *
 * Live phase + % go to pg_stat_progress_create_index (always). Per-phase
 * resource stats + summary go to the server log only when
 * prism.log_build_stats is on, modeled on core btree's log_btree_build_stats
 * (ResetUsage/ShowUsage). The planned-allocation line is always emitted so an
 * OOM is pre-explained. The standalone no-op bodies live in
 * src/index/build_progress.c.
 */

#include <postgres.h>

#include "mkt_config.h"

#include <commands/progress.h>
#include <miscadmin.h>
#include <pgstat.h>
#include <portability/instr_time.h>
#include <tcop/tcopprot.h> /* ResetUsage / ShowUsage */
#include <utils/backend_progress.h>
#include <utils/injection_point.h>
#include <utils/memutils.h>

#include "index/build_progress.h"
#include "index/index_build.h" /* PrismBuildStats */

static int64
now_ns(void)
{
	instr_time t;

	INSTR_TIME_SET_CURRENT(t);
	return INSTR_TIME_GET_NANOSEC(t);
}

/*
 * Test hook name fired at each phase boundary, so an isolation test can pause
 * a build at any phase and read pg_stat_progress_create_index. SCAN and
 * SCAN_PARALLEL share the long-standing "prism-build-load" name (the
 * build_progress isolation test depends on it); the other phases get their
 * own. Returns NULL for phases without a hook. No-op unless PG was built with
 * injection points and a test attached an action.
 */
static const char *
injection_name_for_phase(int phase)
{
	switch (phase)
	{
	case PRISM_BUILD_PHASE_SAMPLE:
		return "prism-build-sample";
	case PRISM_BUILD_PHASE_KMEANS:
		return "prism-build-kmeans";
	case PRISM_BUILD_PHASE_SUBTREES:
		return "prism-build-subtrees";
	case PRISM_BUILD_PHASE_REFINE:
		return "prism-build-refine";
	case PRISM_BUILD_PHASE_SCAN:
	case PRISM_BUILD_PHASE_SCAN_PARALLEL:
		return "prism-build-load";
	case PRISM_BUILD_PHASE_POSTING:
		return "prism-build-posting";
	default:
		return NULL;
	}
}

/* Add a finished phase's elapsed time to the matching PrismBuildStats field.
 */
static void
accumulate_stats(PrismBuildStats *s, int phase, double ms)
{
	if (s == NULL)
		return;

	switch (phase)
	{
	case PRISM_BUILD_PHASE_SAMPLE:
		s->ms_sample += ms;
		break;
	case PRISM_BUILD_PHASE_KMEANS:
	case PRISM_BUILD_PHASE_SUBTREES:
		s->ms_kmeans += ms;
		break;
	case PRISM_BUILD_PHASE_REFINE:
		s->ms_refine += ms;
		break;
	case PRISM_BUILD_PHASE_SETUP:
		s->ms_setup += ms;
		break;
	case PRISM_BUILD_PHASE_SCAN:
	case PRISM_BUILD_PHASE_SCAN_PARALLEL:
	case PRISM_BUILD_PHASE_POSTING:
		s->ms_posting += ms;
		break;
	case PRISM_BUILD_PHASE_CENTROID:
		s->ms_centroid += ms;
		break;
	default:
		break;
	}
}

/* Emit the just-finished phase's resource line(s) and reset the usage window.
 */
static void
flush_phase(PrismBuildProgress *p)
{
	double ms = (double)(now_ns() - p->phase_start_ns) / 1e6;

	accumulate_stats(p->stats, p->cur_phase, ms);

	if (!p->log_stats || p->cur_phase == 0)
		return;

	double heap_mb = p->heap_ctx ? (double)MemoryContextMemAllocated(
										   (MemoryContext)p->heap_ctx, true) /
										   (1024.0 * 1024.0)
								 : 0.0;

	ereport(LOG,
			(errmsg(MKT_AM_NAME " build: phase \"%s\" done in %.0f ms "
								"(build heap %.1f MB, DSM %.1f MB)",
					prism_build_phase_name(p->cur_phase),
					ms,
					heap_mb,
					(double)p->dsm_bytes / (1024.0 * 1024.0)),
			 errhidestmt(true)));

	/* btree-style CPU + maxrss for the phase, then reset for the next one. */
	ShowUsage(MKT_AM_NAME " build phase resource usage");
	ResetUsage();
}

void
prism_build_progress_begin(
		PrismBuildProgress	   *p,
		bool					is_parallel,
		bool					log_stats,
		void				   *heap_ctx,
		struct PrismBuildStats *stats,
		double					tuples_total)
{
	int64 t = now_ns();

	p->cur_phase	  = 0;
	p->phase_start_ns = t;
	p->build_start_ns = t;
	p->log_stats	  = log_stats;
	p->is_parallel	  = is_parallel;
	p->total_set	  = false;
	p->heap_ctx		  = heap_ctx;
	p->dsm_bytes	  = 0;
	p->tuples_total	  = tuples_total;
	p->stats		  = stats;

	/* Publish the row total up front so percent_complete is meaningful for the
	 * whole build (the parallel path has no leader-side loop to publish it
	 * mid-scan). */
	if (tuples_total > 0)
	{
		pgstat_progress_update_param(
				PROGRESS_CREATEIDX_TUPLES_TOTAL, (int64)tuples_total);
		p->total_set = true;
	}

	if (log_stats)
		ResetUsage();
}

void
prism_build_report_phase(PrismBuildProgress *p, int phase)
{
	flush_phase(p);
	p->cur_phase	  = phase;
	p->phase_start_ns = now_ns();
	pgstat_progress_update_param(PROGRESS_CREATEIDX_SUBPHASE, phase);

	/* Each scan-shaped phase walks the heap from the start, so its
	 * tuples-done count restarts; the phases in between leave the previous
	 * scan's final count standing. */
	if (phase == PRISM_BUILD_PHASE_SAMPLE || phase == PRISM_BUILD_PHASE_SCAN ||
		phase == PRISM_BUILD_PHASE_SCAN_PARALLEL)
		pgstat_progress_update_param(PROGRESS_CREATEIDX_TUPLES_DONE, 0);

	const char *ip = injection_name_for_phase(phase);
	if (ip != NULL)
		INJECTION_POINT(ip, NULL);
}

void
prism_build_progress_incr_tuples(int64_t n)
{
	/* From a worker this piggybacks over the parallel message queue and the
	 * leader applies it (also while blocked at a barrier -- interrupt
	 * processing runs inside its condition-variable sleeps); in the leader
	 * it applies directly. */
	pgstat_progress_parallel_incr_param(PROGRESS_CREATEIDX_TUPLES_DONE, n);

	/* Test hook: lets an isolation test pause a build at its first
	 * mid-scan progress flush and observe the advanced counter. Fires once
	 * per backend so waking the build once suffices. */
#ifdef USE_INJECTION_POINTS
	static bool fired = false;
	if (!fired)
	{
		fired = true;
		INJECTION_POINT("prism-scan-progress", NULL);
	}
#endif
}

void
prism_build_report_progress(PrismBuildProgress *p, double done)
{
	if (!p->total_set && p->tuples_total > 0)
	{
		pgstat_progress_update_param(
				PROGRESS_CREATEIDX_TUPLES_TOTAL, (int64)p->tuples_total);
		p->total_set = true;
	}
	pgstat_progress_update_param(PROGRESS_CREATEIDX_TUPLES_DONE, (int64)done);
}

void
prism_build_report_dsm_bytes(PrismBuildProgress *p, uint64_t dsm_bytes)
{
	p->dsm_bytes = dsm_bytes;
}

void
prism_build_report_planned_alloc(
		PrismBuildProgress *p,
		uint64_t			sample_bytes,
		uint64_t			centroid_tree_bytes,
		uint64_t			dsm_total_bytes)
{
	(void)p;
	const double mb = 1024.0 * 1024.0;

	/* Always emitted (not gated by the GUC) so an OOM in any of these is
	 * pre-explained in the log even on a default-configured server. */
	ereport(LOG,
			(errmsg("prism build: planned allocations — samples %.0f MB, "
					"centroid tree ~%.0f MB, DSM %.0f MB "
					"(maintenance_work_mem %d kB)",
					(double)sample_bytes / mb,
					(double)centroid_tree_bytes / mb,
					(double)dsm_total_bytes / mb,
					maintenance_work_mem),
			 errhidestmt(true)));
}

void
prism_build_progress_end(PrismBuildProgress *p)
{
	flush_phase(p);
	p->cur_phase = 0;

	/* Final per-phase summary, always logged (like the planned-allocation
	 * line), via the shared prism_build_stats_print so the PostgreSQL and
	 * standalone builds report identically. */
	if (p->stats != NULL)
	{
		p->stats->ms_total = (double)(now_ns() - p->build_start_ns) / 1e6;
		prism_build_stats_print(p->stats);
	}
}
