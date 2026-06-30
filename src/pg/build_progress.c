/*
 * build_progress.c - PostgreSQL body of the build-progress reporting seam.
 *
 * Live phase + % go to pg_stat_progress_create_index (always). Per-phase
 * resource stats + summary go to the server log only when mkt.log_build_stats
 * is on, modeled on core btree's log_btree_build_stats (ResetUsage/ShowUsage).
 * The planned-allocation line is always emitted so an OOM is pre-explained.
 * The standalone no-op bodies live in src/index/build_progress.c.
 */

#include <postgres.h>

#include <commands/progress.h>
#include <miscadmin.h>
#include <pgstat.h>
#include <portability/instr_time.h>
#include <tcop/tcopprot.h> /* ResetUsage / ShowUsage */
#include <utils/backend_progress.h>
#include <utils/injection_point.h>
#include <utils/memutils.h>

#include "index/build_progress.h"
#include "index/index_build.h" /* MktBuildStats */

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
 * SCAN_PARALLEL share the long-standing "mktann-build-load" name (the
 * build_progress isolation test depends on it); the other phases get their
 * own. Returns NULL for phases without a hook. No-op unless PG was built with
 * injection points and a test attached an action.
 */
static const char *
injection_name_for_phase(int phase)
{
	switch (phase)
	{
	case MKT_BUILD_PHASE_SAMPLE:
		return "mktann-build-sample";
	case MKT_BUILD_PHASE_KMEANS:
		return "mktann-build-kmeans";
	case MKT_BUILD_PHASE_SUBTREES:
		return "mktann-build-subtrees";
	case MKT_BUILD_PHASE_GRAFT:
		return "mktann-build-graft";
	case MKT_BUILD_PHASE_REFINE:
		return "mktann-build-refine";
	case MKT_BUILD_PHASE_SCAN:
	case MKT_BUILD_PHASE_SCAN_PARALLEL:
		return "mktann-build-load";
	case MKT_BUILD_PHASE_POSTING:
		return "mktann-build-posting";
	default:
		return NULL;
	}
}

/* Add a finished phase's elapsed time to the matching MktBuildStats field. */
static void
accumulate_stats(MktBuildStats *s, int phase, double ms)
{
	if (s == NULL)
		return;

	switch (phase)
	{
	case MKT_BUILD_PHASE_SAMPLE:
		s->ms_sample += ms;
		break;
	case MKT_BUILD_PHASE_KMEANS:
	case MKT_BUILD_PHASE_SUBTREES:
	case MKT_BUILD_PHASE_GRAFT:
		s->ms_kmeans += ms;
		break;
	case MKT_BUILD_PHASE_REFINE:
		s->ms_refine += ms;
		break;
	case MKT_BUILD_PHASE_SETUP:
		s->ms_setup += ms;
		break;
	case MKT_BUILD_PHASE_SCAN:
	case MKT_BUILD_PHASE_SCAN_PARALLEL:
	case MKT_BUILD_PHASE_POSTING:
		s->ms_posting += ms;
		break;
	case MKT_BUILD_PHASE_CENTROID:
		s->ms_centroid += ms;
		break;
	default:
		break;
	}
}

/* Emit the just-finished phase's resource line(s) and reset the usage window.
 */
static void
flush_phase(MktBuildProgress *p)
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
			(errmsg("mktann build: phase \"%s\" done in %.0f ms "
					"(build heap %.1f MB, DSM %.1f MB)",
					mkt_build_phase_name(p->cur_phase),
					ms,
					heap_mb,
					(double)p->dsm_bytes / (1024.0 * 1024.0)),
			 errhidestmt(true)));

	/* btree-style CPU + maxrss for the phase, then reset for the next one. */
	ShowUsage("mktann build phase resource usage");
	ResetUsage();
}

void
mkt_build_progress_begin(
		MktBuildProgress	 *p,
		bool				  is_parallel,
		bool				  log_stats,
		void				 *heap_ctx,
		struct MktBuildStats *stats,
		double				  tuples_total)
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
mkt_build_report_phase(MktBuildProgress *p, int phase)
{
	flush_phase(p);
	p->cur_phase	  = phase;
	p->phase_start_ns = now_ns();
	pgstat_progress_update_param(PROGRESS_CREATEIDX_SUBPHASE, phase);

	const char *ip = injection_name_for_phase(phase);
	if (ip != NULL)
		INJECTION_POINT(ip, NULL);
}

void
mkt_build_report_progress(MktBuildProgress *p, double done)
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
mkt_build_report_dsm_bytes(MktBuildProgress *p, uint64_t dsm_bytes)
{
	p->dsm_bytes = dsm_bytes;
}

void
mkt_build_report_planned_alloc(
		MktBuildProgress *p,
		uint64_t		  sample_bytes,
		uint64_t		  centroid_tree_bytes,
		uint64_t		  pt_centroids_bytes,
		uint64_t		  dsm_total_bytes)
{
	(void)p;
	const double mb = 1024.0 * 1024.0;

	/* Always emitted (not gated by the GUC) so an OOM in any of these is
	 * pre-explained in the log even on a default-configured server. */
	ereport(LOG,
			(errmsg("mktann build: planned allocations — samples %.0f MB, "
					"centroid tree ~%.0f MB, pt_centroids %.0f MB, DSM %.0f "
					"MB "
					"(maintenance_work_mem %d kB)",
					(double)sample_bytes / mb,
					(double)centroid_tree_bytes / mb,
					(double)pt_centroids_bytes / mb,
					(double)dsm_total_bytes / mb,
					maintenance_work_mem),
			 errhidestmt(true)));
}

void
mkt_build_progress_end(MktBuildProgress *p)
{
	flush_phase(p);
	p->cur_phase = 0;

	/* Final per-phase summary, always logged (like the planned-allocation
	 * line), via the shared mkt_build_stats_print so the PostgreSQL and
	 * standalone builds report identically. */
	if (p->stats != NULL)
	{
		p->stats->ms_total = (double)(now_ns() - p->build_start_ns) / 1e6;
		mkt_build_stats_print(p->stats);
	}
}
