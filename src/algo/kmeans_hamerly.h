/*
 * kmeans_hamerly.h - Hamerly's accelerated k-means assignment
 *
 * Hamerly's algorithm (ICML 2010) maintains per-vector upper and lower
 * bounds on distances to centroids. Vectors whose assignment cannot
 * possibly change (upper <= lower) are skipped entirely. After a few
 * iterations of convergence, 90%+ of vectors are skipped.
 *
 * Memory overhead: O(n) — much less than Elkan's O(n*k).
 * Complexity: same per-iteration worst case as Lloyd's, but typically
 * 5-10x fewer distance computations in practice.
 *
 * This implementation uses dot product decomposition for distance
 * computation (1 FMA/dim vs 2 for direct L2), combining Hamerly's
 * pruning with the computational efficiency of the BLAS-style approach.
 *
 * Reference: G. Hamerly, "Making k-means even faster", SDM 2010.
 */

#ifndef VS_KMEANS_HAMERLY_H
#define VS_KMEANS_HAMERLY_H

#include <stdbool.h>
#include <stdint.h>

#include "algo/kmeans_internal.h"

typedef struct HamerlyState HamerlyState;

/*
 * Create Hamerly state for the given problem size.
 * Only used for DISTANCE_L2 metric.
 */
HamerlyState *hamerly_create(uint32_t nvecs, uint32_t nlist, uint32_t dim);

/*
 * Free Hamerly state.
 */
void hamerly_destroy(HamerlyState *hs);

/*
 * Run the full assignment step using Hamerly's algorithm.
 *
 * On the first call (bounds not yet valid), runs a full Lloyd's
 * assignment and initializes bounds. On subsequent calls, uses
 * bounds to skip vectors whose assignment cannot change.
 *
 * Updates st->assignments, st->total_cost.
 */
void hamerly_assign(KMeansState *st, HamerlyState *hs);

/*
 * Update bounds after centroid update.
 *
 * Must be called after centroids change (kmeans_update_centroids)
 * and before the next hamerly_assign(). Computes per-centroid
 * movement deltas and adjusts upper/lower bounds accordingly.
 *
 * old_centroids: centroid positions before the update.
 * Must be saved by the caller before calling kmeans_update_centroids.
 */
void hamerly_update_bounds(
		KMeansState *st, HamerlyState *hs, const float *old_centroids);

#endif /* VS_KMEANS_HAMERLY_H */
