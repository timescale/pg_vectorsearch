/*
 * kmeans_elkan.h - Elkan's accelerated k-means assignment
 *
 * Elkan's algorithm (ICML 2003) maintains per-vector per-centroid
 * lower bounds plus a per-vector upper bound. The triangle inequality
 * prunes distance computations: if the upper bound to the assigned
 * centroid is less than the lower bound to another centroid, that
 * centroid cannot be closer and the computation is skipped.
 *
 * Memory overhead: O(n*k) for lower bounds — much more than Hamerly's
 * O(n), but provides much tighter bounds and higher skip rates.
 *
 * Reference: C. Elkan, "Using the Triangle Inequality to Accelerate
 * k-Means", ICML 2003.
 */

#ifndef MKT_KMEANS_ELKAN_H
#define MKT_KMEANS_ELKAN_H

#include <stdint.h>

#include "algo/kmeans_internal.h"

typedef struct ElkanState ElkanState;

/*
 * Create Elkan state for the given problem size.
 * Only used for DISTANCE_L2 metric.
 *
 * Memory: O(nvecs * nlist) for lower bounds.
 * For 100K vectors and 1000 centroids: ~400 MB.
 */
ElkanState *elkan_create(uint32_t nvecs, uint32_t nlist, uint32_t dim);

/*
 * Free Elkan state.
 */
void elkan_destroy(ElkanState *es);

/*
 * Run one assignment step using Elkan's algorithm.
 *
 * On first call, runs full assignment and initializes all bounds.
 * On subsequent calls, uses bounds to skip distance computations.
 *
 * Updates st->assignments, st->total_cost.
 */
void elkan_assign(KMeansState *st, ElkanState *es);

/*
 * Update bounds after centroid update.
 *
 * Must be called after centroids change and before the next
 * elkan_assign(). Adjusts upper/lower bounds based on how much
 * each centroid moved.
 *
 * old_centroids: centroid positions before the update.
 */
void elkan_update_bounds(
		KMeansState *st, ElkanState *es, const float *old_centroids);

#endif /* MKT_KMEANS_ELKAN_H */
