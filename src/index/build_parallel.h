/*
 * build_parallel.h - Parallel index build support
 *
 * Shared functions for parallelizing the index build. These work in
 * both standalone (pthreads) and PostgreSQL (parallel workers) contexts.
 * No PostgreSQL dependencies.
 *
 * The build hot path (tree descent + RaBitQ encode + SOAR) is extracted
 * into per-vector functions that can be called from any worker.
 */

#ifndef MKT_BUILD_PARALLEL_H
#define MKT_BUILD_PARALLEL_H

#include <stdbool.h>
#include <stdint.h>

#include "algo/hkmeans.h"
#include "mkt_types.h"

#define MKT_INVALID_CLUSTER UINT32_MAX

/*
 * Build parameters needed by the per-vector processing functions.
 * A subset of MktIndexConfig / MktannBuildParams that's common to
 * both standalone and PG builds.
 */
typedef struct MktBuildParams
{
	Dimension	   dim;
	DistanceMetric metric;
	double		   soar_lambda;
	double		   boundary_epsilon;
} MktBuildParams;

/*
 * Result of assigning a vector to cluster(s).
 *
 * primary: the nearest leaf centroid index (always valid)
 * secondary: the SOAR/boundary secondary centroid, or
 *            MKT_INVALID_CLUSTER if no replication
 * enc_vector: pointer to the vector to encode. For cosine metric,
 *             this points to the normalized copy in norm_buf.
 *             For L2, this points to the original vector.
 */
typedef struct MktBuildAssignment
{
	uint32_t	 primary;
	uint32_t	 secondary;
	const float *enc_vector;
} MktBuildAssignment;

/*
 * Per-worker scratch buffers. Each worker (thread or PG process)
 * must have its own instance — not shared.
 */
typedef struct MktBuildWorkerBufs
{
	float *norm_buf;	 /* [dim] for cosine normalization */
	float *residual_buf; /* [dim] for SOAR residual computation */
} MktBuildWorkerBufs;

/*
 * Allocate per-worker scratch buffers.
 */
MktBuildWorkerBufs mkt_build_worker_bufs_create(Dimension dim);

/*
 * Free per-worker scratch buffers.
 */
void mkt_build_worker_bufs_free(MktBuildWorkerBufs *bufs);

/*
 * Assign a single vector to its primary (and optionally secondary)
 * cluster. This is the core per-vector function extracted from
 * build_callback().
 *
 * Thread-safe: reads only from shared tree/params, writes only to
 * caller-owned bufs and the returned assignment.
 *
 * tree:   hierarchical k-means result (read-only, shared)
 * vec:    raw input vector [dim]
 * params: build parameters (read-only, shared)
 * bufs:   per-worker scratch buffers (caller-owned)
 *
 * Returns: cluster assignment with enc_vector pointing to either
 *          the raw vector (L2) or the normalized copy in bufs (cosine).
 */
MktBuildAssignment mkt_build_assign_vector(
		const HKMeansResult	 *tree,
		const float			 *vec,
		const MktBuildParams *params,
		MktBuildWorkerBufs	 *bufs);

/*
 * Per-vector assignment result stored in flat arrays for batch
 * processing. Used by the threaded build path.
 */
typedef struct MktBatchAssignment
{
	uint32_t *primary;	 /* [count] primary cluster IDs */
	uint32_t *secondary; /* [count] secondary cluster IDs (MKT_INVALID_CLUSTER
							if none) */
	uint32_t count;
} MktBatchAssignment;

MktBatchAssignment mkt_batch_assignment_create(uint32_t count);
void			   mkt_batch_assignment_free(MktBatchAssignment *ba);

/*
 * Assign a batch of vectors to clusters in parallel using pthreads.
 *
 * vectors:    [count * dim] contiguous float array
 * count:      number of vectors
 * tree:       hierarchical k-means result (read-only, shared)
 * params:     build parameters (read-only, shared)
 * nthreads:   number of threads (0 = auto-detect)
 * out:        pre-allocated batch assignment (primary + secondary arrays)
 *
 * Each thread processes a contiguous range of vectors independently.
 * The enc_vector from MktBuildAssignment is NOT stored — the caller
 * should normalize separately if needed, since the normalized vector
 * lives in a per-thread buffer that's freed after the batch.
 */
void mkt_build_assign_batch_parallel(
		const float			 *vectors,
		uint32_t			  count,
		Dimension			  dim,
		const HKMeansResult	 *tree,
		const MktBuildParams *params,
		uint32_t			  nthreads,
		MktBatchAssignment	 *out);

#endif /* MKT_BUILD_PARALLEL_H */
