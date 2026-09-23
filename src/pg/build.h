/*
 * build.h - Index build for prism
 *
 * Implements ambuild: heap sampling, k-means clustering, RaBitQ
 * encoding, and centroid page writing.
 */

#ifndef PRISM_BUILD_H
#define PRISM_BUILD_H

#include <postgres.h>

#include <access/amapi.h>
#include <storage/block.h>
#include <utils/rel.h>

#include "algo/hkmeans.h"
#include "core/types.h"
#include "index/build_progress.h" /* canonical PRISM_BUILD_PHASE_* */
#include "index/index_base.h"
#include "index/parallel_build.h" /* PrismBuildConfig */
#include "pg/bufstorage.h"

/* Build parameters resolved from the index relation/opclass, shared by the
 * serial and parallel build paths. */
typedef struct PrismBuildParams
{
	Dimension			dim;
	DistanceMetric		metric;
	PrismCentroidFormat centroid_format;
	uint32_t			nlist;
	uint32_t			fan_out;
	uint32_t			kmeans_nredo;
	double				soar_lambda;
	double				boundary_epsilon;
	bool				fastscan;
} PrismBuildParams;

IndexBuildResult *
prism_build(Relation heap, Relation index, struct IndexInfo *index_info);

char *prism_buildphasename(int64 phasenum);

/*
 * Estimated live tuples in a heap: reltuples when the relation has been
 * analyzed, otherwise a stride sample of heap pages. Shared with maintenance,
 * which needs the same estimate to resolve the target list size -- see the
 * comment on the definition for why the fallback samples rather than deriving
 * rows from the column width.
 */
double prism_estimate_heap_tuples(Relation heap);

/* do_parallel_build (the shared parallel build entry) is declared in
 * index/parallel_build.h, included above. */

#endif /* PRISM_BUILD_H */
