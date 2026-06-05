/*
 * mktann_build.h - Index build for mktann
 *
 * Implements ambuild: heap sampling, k-means clustering, RaBitQ
 * encoding, and centroid page writing.
 */

#ifndef MKTANN_BUILD_H
#define MKTANN_BUILD_H

#include <postgres.h>

#include <access/amapi.h>
#include <storage/block.h>
#include <utils/rel.h>

#include "algo/hkmeans.h"
#include "index/index_base.h"
#include "index/parallel_build.h" /* MktBuildConfig */
#include "mkt_types.h"
#include "mktann_storage.h"

/* Build parameters resolved from the index relation/opclass, shared by the
 * serial and parallel build paths. */
typedef struct MktannBuildParams
{
	Dimension		  dim;
	DistanceMetric	  metric;
	MktCentroidFormat centroid_format;
	uint32_t		  nlist;
	uint32_t		  fan_out;
	uint32_t		  kmeans_nredo;
	double			  soar_lambda;
	double			  boundary_epsilon;
	bool			  fastscan;
} MktannBuildParams;

/* Custom build subphases for pg_stat_progress_create_index.
 * Values start at 2 (1 = PROGRESS_CREATEIDX_SUBPHASE_INITIALIZE). */
#define PROGRESS_MKTANN_PHASE_SAMPLE   2
#define PROGRESS_MKTANN_PHASE_KMEANS   3
#define PROGRESS_MKTANN_PHASE_SETUP	   4
#define PROGRESS_MKTANN_PHASE_SCAN	   5
#define PROGRESS_MKTANN_PHASE_POSTING  6
#define PROGRESS_MKTANN_PHASE_CENTROID 7
#define PROGRESS_MKTANN_PHASE_WAL	   8

IndexBuildResult *
mktann_build(Relation heap, Relation index, struct IndexInfo *index_info);

char *mktann_buildphasename(int64 phasenum);

/* do_parallel_build (the shared parallel build entry) is declared in
 * index/parallel_build.h, included above. */

#endif /* MKTANN_BUILD_H */
