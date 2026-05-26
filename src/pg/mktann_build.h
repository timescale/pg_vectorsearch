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
#include <utils/rel.h>

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

#endif /* MKTANN_BUILD_H */
