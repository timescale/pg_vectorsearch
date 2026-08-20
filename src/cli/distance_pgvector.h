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
Distance mkt_distance_l2_pgvector(VectorRef a, VectorRef b);
Distance mkt_distance_ip_pgvector(VectorRef a, VectorRef b);
Distance mkt_distance_cosine_pgvector(VectorRef a, VectorRef b);

/* Batch operations */
int mkt_distance_batch_l2_pgvector(
		VectorRef	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances);

int mkt_distance_batch_ip_pgvector(
		VectorRef	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances);

int mkt_distance_batch_cosine_pgvector(
		VectorRef	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances);

#endif /* DISTANCE_PGVECTOR_H */
