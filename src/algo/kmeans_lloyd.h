/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * kmeans_lloyd.h - Lloyd's k-means assignment step
 *
 * Standard brute-force assignment: compute all N*K distances each
 * iteration. Two backends:
 *
 * - CBLAS: uses sgemm for the dot-product matrix, 10-50x faster for
 *   large K due to cache-efficient BLAS kernels.
 * - Builtin: batch dot-product loop with target_clones for AVX-512/AVX2
 *   auto-vectorization. No external dependencies.
 *
 * Both use the decomposition d²(x,c) = ||x||² + ||c||² - 2⟨x,c⟩
 * and process vectors in blocks of KMEANS_BLOCK_SIZE for cache efficiency.
 */

#ifndef VS_KMEANS_LLOYD_H
#define VS_KMEANS_LLOYD_H

#include <stdbool.h>

#include "algo/kmeans_internal.h"

/*
 * Run the full assignment step using Lloyd's algorithm.
 *
 * Assigns every vector to its nearest centroid, computing all N*K
 * distances. Updates st->assignments and st->total_cost.
 *
 * use_cblas: if true and CBLAS is available, use sgemm for the
 * dot-product matrix. Otherwise uses the builtin batch path.
 */
void lloyd_assign(KMeansState *st, bool use_cblas);

/*
 * Fused iterative assign + update using the iterate callback.
 *
 * Each iteration: parallel assignment + per-thread centroid
 * accumulation, then serial merge + convergence check.
 * Workers stay alive across iterations via barrier.
 *
 * Returns the number of iterations completed.
 */
uint32_t
lloyd_iterate(KMeansState *st, bool use_cblas, const KMeansOptions *opts);

#endif /* VS_KMEANS_LLOYD_H */
