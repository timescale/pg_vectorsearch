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

/*
 * Build the mktann index. Matches ambuild_function signature.
 *
 * Phases:
 *   1. Determine dimension from index column typmod
 *   2. Sample vectors via BlockSampler + reservoir sampling
 *   3. Run k-means on samples
 *   4. Full heap scan to assign vectors and find medoids
 *   5. RaBitQ-encode centroids relative to global mean
 *   6. Write metadata page (block 0) and centroid pages
 */
IndexBuildResult *
mktann_build(Relation heap, Relation index, struct IndexInfo *index_info);

#endif /* MKTANN_BUILD_H */
