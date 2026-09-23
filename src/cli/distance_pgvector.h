/*
 * distance_pgvector.h - pgvector-style auto-vectorized distance
 *
 * Simple loop implementations with GCC's target_clones for compiler
 * auto-vectorization. Used to compare performance against hand-optimized
 * explicit SIMD implementations.
 */

#ifndef DISTANCE_PGVECTOR_H
#define DISTANCE_PGVECTOR_H

#include "algo/distance.h"
#include "core/types.h"

/* Single-pair distance functions */
Distance vs_distance_l2_pgvector(Vec32Ref a, Vec32Ref b);
Distance vs_distance_ip_pgvector(Vec32Ref a, Vec32Ref b);
Distance vs_distance_cosine_pgvector(Vec32Ref a, Vec32Ref b);

/* Batch operations */
int vs_distance_batch_l2_pgvector(
		Vec32Ref	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances);

int vs_distance_batch_ip_pgvector(
		Vec32Ref	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances);

int vs_distance_batch_cosine_pgvector(
		Vec32Ref	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances);

#endif /* DISTANCE_PGVECTOR_H */
