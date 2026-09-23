/*
 * build_progress.c - Canonical index-build phase names (see header)
 */

#include <stddef.h>

#include "index/build_progress.h"

/*
 * Phase name table, indexed by phase value. Index 0 is unused (phases start at
 * 1) and any gap stays NULL. These strings are what
 * pg_stat_progress_create_index reports (via prism_buildphasename) and what
 * the build logs print, so the two back-ends can never drift.
 */
static const char *const phase_names[] = {
		[PRISM_BUILD_PHASE_INITIALIZE]	  = "initializing",
		[PRISM_BUILD_PHASE_SAMPLE]		  = "sampling vectors",
		[PRISM_BUILD_PHASE_KMEANS]		  = "clustering (k-means)",
		[PRISM_BUILD_PHASE_SETUP]		  = "preparing RaBitQ encoding",
		[PRISM_BUILD_PHASE_SCAN]		  = "scanning table",
		[PRISM_BUILD_PHASE_POSTING]		  = "finalizing posting lists",
		[PRISM_BUILD_PHASE_CENTROID]	  = "writing centroid pages",
		[PRISM_BUILD_PHASE_WAL]			  = "WAL logging",
		[PRISM_BUILD_PHASE_SCAN_PARALLEL] = "scanning table (parallel)",
		[PRISM_BUILD_PHASE_SUBTREES]	  = "clustering (subtrees)",
		[PRISM_BUILD_PHASE_REFINE]		  = "refining centroids",
};

const char *
prism_build_phase_name(int phase)
{
	if (phase < 0 ||
		(size_t)phase >= sizeof(phase_names) / sizeof(phase_names[0]))
		return NULL;
	return phase_names[phase];
}

#ifdef VS_STANDALONE

/*
 * Standalone build-progress seam: no-op stubs. Standalone introspection is
 * deferred; these exist so the shared build driver links. The PostgreSQL build
 * gets the real bodies from src/pg/build_progress.c instead.
 */
void
prism_build_progress_begin(
		PrismBuildProgress	   *p,
		bool					is_parallel,
		bool					log_stats,
		void				   *heap_ctx,
		struct PrismBuildStats *stats,
		double					tuples_total)
{
	(void)p;
	(void)is_parallel;
	(void)log_stats;
	(void)heap_ctx;
	(void)stats;
	(void)tuples_total;
}

void
prism_build_report_phase(PrismBuildProgress *p, int phase)
{
	(void)p;
	(void)phase;
}

void
prism_build_progress_incr_tuples(int64_t n)
{
	(void)n;
}

void
prism_build_report_progress(PrismBuildProgress *p, double done)
{
	(void)p;
	(void)done;
}

void
prism_build_report_dsm_bytes(PrismBuildProgress *p, uint64_t dsm_bytes)
{
	(void)p;
	(void)dsm_bytes;
}

void
prism_build_report_planned_alloc(
		PrismBuildProgress *p,
		uint64_t			sample_bytes,
		uint64_t			centroid_tree_bytes,
		uint64_t			dsm_total_bytes)
{
	(void)p;
	(void)sample_bytes;
	(void)centroid_tree_bytes;
	(void)dsm_total_bytes;
}

void
prism_build_progress_end(PrismBuildProgress *p)
{
	(void)p;
}

#endif /* VS_STANDALONE */
