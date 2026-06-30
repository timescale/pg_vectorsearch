/*
 * build_progress.c - Canonical index-build phase names (see header)
 */

#include <stddef.h>

#include "index/build_progress.h"

/*
 * Phase name table, indexed by phase value. Index 0 is unused (phases start at
 * 1) and any gap stays NULL. These strings are what
 * pg_stat_progress_create_index reports (via mktann_buildphasename) and what
 * the build logs print, so the two back-ends can never drift.
 */
static const char *const phase_names[] = {
		[MKT_BUILD_PHASE_INITIALIZE]	= "initializing",
		[MKT_BUILD_PHASE_SAMPLE]		= "sampling vectors",
		[MKT_BUILD_PHASE_KMEANS]		= "clustering (k-means)",
		[MKT_BUILD_PHASE_SETUP]			= "preparing RaBitQ encoding",
		[MKT_BUILD_PHASE_SCAN]			= "scanning table",
		[MKT_BUILD_PHASE_POSTING]		= "finalizing posting lists",
		[MKT_BUILD_PHASE_CENTROID]		= "writing centroid pages",
		[MKT_BUILD_PHASE_WAL]			= "WAL logging",
		[MKT_BUILD_PHASE_SCAN_PARALLEL] = "scanning table (parallel)",
		[MKT_BUILD_PHASE_SUBTREES]		= "clustering (subtrees)",
		[MKT_BUILD_PHASE_GRAFT]			= "clustering (assembly)",
		[MKT_BUILD_PHASE_REFINE]		= "refining centroids",
};

const char *
mkt_build_phase_name(int phase)
{
	if (phase < 0 ||
		(size_t)phase >= sizeof(phase_names) / sizeof(phase_names[0]))
		return NULL;
	return phase_names[phase];
}

#ifdef MKT_STANDALONE

/*
 * Standalone build-progress seam: no-op stubs. Standalone introspection is
 * deferred; these exist so the shared build driver links. The PostgreSQL build
 * gets the real bodies from src/pg/build_progress.c instead.
 */
void
mkt_build_progress_begin(
		MktBuildProgress	 *p,
		bool				  is_parallel,
		bool				  log_stats,
		void				 *heap_ctx,
		struct MktBuildStats *stats,
		double				  tuples_total)
{
	(void)p;
	(void)is_parallel;
	(void)log_stats;
	(void)heap_ctx;
	(void)stats;
	(void)tuples_total;
}

void
mkt_build_report_phase(MktBuildProgress *p, int phase)
{
	(void)p;
	(void)phase;
}

void
mkt_build_report_progress(MktBuildProgress *p, double done)
{
	(void)p;
	(void)done;
}

void
mkt_build_report_dsm_bytes(MktBuildProgress *p, uint64_t dsm_bytes)
{
	(void)p;
	(void)dsm_bytes;
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
	(void)sample_bytes;
	(void)centroid_tree_bytes;
	(void)pt_centroids_bytes;
	(void)dsm_total_bytes;
}

void
mkt_build_progress_end(MktBuildProgress *p)
{
	(void)p;
}

#endif /* MKT_STANDALONE */
