/*
 * mktann_parallel.h - PG parallel index build for mktann
 *
 * Two-phase parallel build:
 *   Phase 1 (sampling + k-means): workers cooperatively scan the
 *     heap, collect samples, pick initial centroids, then iterate
 *     k-means assignment + accumulation via PG Barrier. The leader
 *     merges per-worker sums between iterations.
 *   Phase 2 (posting): workers cooperatively scan the heap again,
 *     assign vectors to clusters via tree descent, and stream
 *     posting pages to the buffer cache.
 * The leader merges partial pages and writes centroid pages.
 */

#ifndef MKTANN_PARALLEL_H
#define MKTANN_PARALLEL_H

#include <postgres.h>

#include <port/atomics.h>
#include <storage/barrier.h>
#include <storage/block.h>
#include <storage/condition_variable.h>
#include <storage/itemptr.h>
#include <storage/shm_toc.h>
#include <storage/spin.h>

#include "algo/hkmeans.h"
#include "index/posting_build.h"
#include "mkt_types.h"
#include "quant/rabitq.h"

/* ----------------------------------------------------------------
 * DSM table-of-contents keys
 * ---------------------------------------------------------------- */

#define MKTANN_KEY_SHARED		 UINT64CONST(0xB000000000000001)
#define MKTANN_KEY_TREE			 UINT64CONST(0xB000000000000002)
#define MKTANN_KEY_RESERVE		 UINT64CONST(0xB000000000000003)
#define MKTANN_KEY_WORKER_OUTPUT UINT64CONST(0xB000000000000004)
#define MKTANN_KEY_PARTIALS		 UINT64CONST(0xB000000000000005)
#define MKTANN_KEY_WAL_USAGE	 UINT64CONST(0xB000000000000006)
#define MKTANN_KEY_BUFFER_USAGE	 UINT64CONST(0xB000000000000007)
#define MKTANN_KEY_QUERY_TEXT	 UINT64CONST(0xB000000000000008)
#define MKTANN_KEY_BARRIER		 UINT64CONST(0xB000000000000009)
#define MKTANN_KEY_SAMPLES		 UINT64CONST(0xB00000000000000A)
#define MKTANN_KEY_CENTROIDS	 UINT64CONST(0xB00000000000000B)
#define MKTANN_KEY_KM_WORKERS	 UINT64CONST(0xB00000000000000C)
#define MKTANN_KEY_ENTRIES		 UINT64CONST(0xB00000000000000D)
#define MKTANN_KEY_ROOT_ASSIGN	 UINT64CONST(0xB00000000000000E)

/* ----------------------------------------------------------------
 * MktBuildShared — primary shared state in DSM
 *
 * Immutable fields are set by the leader before workers launch.
 * Mutable counters are protected by the spinlock.
 * ParallelTableScanDescData follows at BUFFERALIGN offset.
 * ---------------------------------------------------------------- */

typedef struct MktBuildShared
{
	/* Immutable — set by leader */
	Oid			   heaprelid;
	Oid			   indexrelid;
	int64		   queryid;
	Dimension	   dim;
	DistanceMetric metric;
	uint32_t	   nlist;
	uint32_t	   fan_out;
	double		   soar_lambda;
	double		   boundary_epsilon;
	bool		   fastscan;
	uint64_t	   rabitq_seed;
	int			   nparticipants;

	/* K-means config */
	uint32_t max_samples_per_worker;
	uint32_t km_max_iterations;
	float	 km_tolerance;
	uint32_t km_k; /* root k-means k (= fan_out) */

	/* Mutable — spinlock-protected */
	slock_t			  mutex;
	ConditionVariable workersdonecv;
	int				  nparticipantsdone;
	double			  reltuples;
	double			  indtuples;
	double			  soar_dupes;

	/* K-means convergence — set by leader between barriers */
	bool km_converged;

	/* Child k-means — set by leader between barriers */
	uint32_t current_child;
	uint32_t child_km_k; /* k for current child k-means */
} MktBuildShared;

#define ParallelTableScanFromMktShared(shared)  \
	((ParallelTableScanDesc)((char *)(shared) + \
							 BUFFERALIGN(sizeof(MktBuildShared))))

/* ----------------------------------------------------------------
 * K-means sampling: per-worker sample slots in DSM
 *
 * Layout: [sample_counts[nparticipants]] then
 *         [samples[worker_id][max_per_worker * dim]] packed.
 * ---------------------------------------------------------------- */

typedef struct MktDsmSamples
{
	uint32_t  nparticipants;
	uint32_t  max_per_worker;
	Dimension dim;
} MktDsmSamples;

static inline uint32_t *
mktann_sample_counts(MktDsmSamples *s)
{
	return (uint32_t *)((char *)s + MAXALIGN(sizeof(MktDsmSamples)));
}

static inline float *
mktann_worker_samples(MktDsmSamples *s, int worker_id)
{
	char *base = (char *)mktann_sample_counts(s) +
				 s->nparticipants * sizeof(uint32_t);
	return (float *)(base + (size_t)worker_id * s->max_per_worker * s->dim *
									sizeof(float));
}

static inline Size
mktann_samples_size(int nparticipants, uint32_t max_per_worker, Dimension dim)
{
	Size sz = MAXALIGN(sizeof(MktDsmSamples));
	sz += (Size)nparticipants * sizeof(uint32_t);
	sz += (Size)nparticipants * max_per_worker * dim * sizeof(float);
	return sz;
}

/* ----------------------------------------------------------------
 * Root assignments: per-worker uint32_t[max_per_worker] in DSM
 *
 * After root k-means converges, each worker stores its root
 * assignments here. Used to filter samples during child k-means.
 * ---------------------------------------------------------------- */

typedef struct MktDsmRootAssign
{
	uint32_t nparticipants;
	uint32_t max_per_worker;
} MktDsmRootAssign;

static inline uint32_t *
mktann_root_assignments(MktDsmRootAssign *ra, int worker_id)
{
	char *base = (char *)ra + MAXALIGN(sizeof(MktDsmRootAssign));
	return (uint32_t *)(base + (size_t)worker_id * ra->max_per_worker *
									   sizeof(uint32_t));
}

static inline Size
mktann_root_assign_size(int nparticipants, uint32_t max_per_worker)
{
	Size sz = MAXALIGN(sizeof(MktDsmRootAssign));
	sz += (Size)nparticipants * max_per_worker * sizeof(uint32_t);
	return sz;
}

/* ----------------------------------------------------------------
 * K-means shared centroids in DSM
 *
 * centroids[nlist * dim] + norms_c[nlist]
 * Written by leader between barriers, read by workers during
 * assignment.
 * ---------------------------------------------------------------- */

static inline Size
mktann_centroids_size(uint32_t nlist, Dimension dim)
{
	return (Size)nlist * dim * sizeof(float) + (Size)nlist * sizeof(float);
}

static inline float *
mktann_centroids(char *base)
{
	return (float *)base;
}

static inline float *
mktann_norms_c(char *base, uint32_t nlist, Dimension dim)
{
	return (float *)(base + (size_t)nlist * dim * sizeof(float));
}

/* ----------------------------------------------------------------
 * K-means per-worker accumulators in DSM
 *
 * Per worker: centroid_sums[nlist * dim] + centroid_cnts[nlist]
 *             + cost (1 float)
 * ---------------------------------------------------------------- */

static inline Size
mktann_km_worker_size(uint32_t nlist, Dimension dim)
{
	return (Size)nlist * dim * sizeof(float) + (Size)nlist * sizeof(uint32_t) +
		   sizeof(float);
}

static inline Size
mktann_km_workers_size(int nparticipants, uint32_t nlist, Dimension dim)
{
	return (Size)nparticipants * mktann_km_worker_size(nlist, dim);
}

static inline float *
mktann_km_worker_sums(char *base, uint32_t nlist, Dimension dim, int worker_id)
{
	return (float *)(base +
					 (size_t)worker_id * mktann_km_worker_size(nlist, dim));
}

static inline uint32_t *
mktann_km_worker_cnts(char *base, uint32_t nlist, Dimension dim, int worker_id)
{
	char *slot = base + (size_t)worker_id * mktann_km_worker_size(nlist, dim);
	return (uint32_t *)(slot + (size_t)nlist * dim * sizeof(float));
}

static inline float *
mktann_km_worker_cost(char *base, uint32_t nlist, Dimension dim, int worker_id)
{
	char *slot = base + (size_t)worker_id * mktann_km_worker_size(nlist, dim);
	return (float *)(slot + (size_t)nlist * dim * sizeof(float) +
					 (size_t)nlist * sizeof(uint32_t));
}

/* ----------------------------------------------------------------
 * MktDsmReserve — page reservation with atomics in DSM
 *
 * Layout: header, then starts[nlist], counts[nlist], nexts[nlist]
 * all inline.
 * ---------------------------------------------------------------- */

typedef struct MktDsmReserve
{
	uint32_t	nlist;
	BlockNumber first_posting;
	BlockNumber total_reserved;
} MktDsmReserve;

static inline BlockNumber *
mktann_dsm_reserve_starts(MktDsmReserve *r)
{
	return (BlockNumber *)((char *)r + MAXALIGN(sizeof(MktDsmReserve)));
}

static inline uint32_t *
mktann_dsm_reserve_counts(MktDsmReserve *r)
{
	return (uint32_t *)((char *)mktann_dsm_reserve_starts(r) +
						r->nlist * sizeof(BlockNumber));
}

static inline pg_atomic_uint32 *
mktann_dsm_reserve_nexts(MktDsmReserve *r)
{
	return (pg_atomic_uint32 *)((char *)mktann_dsm_reserve_counts(r) +
								r->nlist * sizeof(uint32_t));
}

static inline Size
mktann_dsm_reserve_size(uint32_t nlist)
{
	Size sz = MAXALIGN(sizeof(MktDsmReserve));
	sz += (Size)nlist * sizeof(BlockNumber);
	sz += (Size)nlist * sizeof(uint32_t);
	sz += (Size)nlist * sizeof(pg_atomic_uint32);
	return sz;
}

/* ----------------------------------------------------------------
 * Per-worker posting output in DSM
 *
 * Flat arrays: heads[nlist] + tails[nlist] + active[nlist]
 * per worker, packed contiguously.
 * ---------------------------------------------------------------- */

static inline Size
mktann_worker_output_size(uint32_t nlist, int nparticipants)
{
	Size per_worker = (Size)nlist * (2 * sizeof(BlockNumber) + sizeof(bool));
	return per_worker * nparticipants;
}

static inline BlockNumber *
mktann_worker_heads(char *base, uint32_t nlist, int worker_id)
{
	Size  per_worker = (Size)nlist * (2 * sizeof(BlockNumber) + sizeof(bool));
	char *slot		 = base + per_worker * worker_id;
	return (BlockNumber *)slot;
}

static inline BlockNumber *
mktann_worker_tails(char *base, uint32_t nlist, int worker_id)
{
	Size  per_worker = (Size)nlist * (2 * sizeof(BlockNumber) + sizeof(bool));
	char *slot		 = base + per_worker * worker_id;
	return (BlockNumber *)(slot + (Size)nlist * sizeof(BlockNumber));
}

static inline bool *
mktann_worker_active(char *base, uint32_t nlist, int worker_id)
{
	Size  per_worker = (Size)nlist * (2 * sizeof(BlockNumber) + sizeof(bool));
	char *slot		 = base + per_worker * worker_id;
	return (bool *)(slot + (Size)nlist * 2 * sizeof(BlockNumber));
}

/* ----------------------------------------------------------------
 * Partials buffer in DSM
 *
 * partials[worker_id * nlist + cluster] = one BLCKSZ page
 * Only allocated when !fastscan.
 * ---------------------------------------------------------------- */

static inline Size
mktann_partials_size(uint32_t nlist, int nparticipants)
{
	return (Size)nparticipants * nlist * BLCKSZ;
}

static inline char *
mktann_worker_partials(char *base, uint32_t nlist, int worker_id)
{
	return base + (Size)worker_id * nlist * BLCKSZ;
}

/* ----------------------------------------------------------------
 * Per-worker encoded entry buffer in DSM
 *
 * Workers encode vectors with RaBitQ and write compact entries
 * here. Leader reads all entries after workers finish and feeds
 * them to serial posting builders. This avoids workers writing
 * to the index relation's buffer pool.
 *
 * Entry layout: cluster_id(4) + tid(6) + pad(2) + f_add(4)
 *               + f_rescale(4) + f_error(4) + bits[rabitq_bytes]
 * ---------------------------------------------------------------- */

typedef struct MktDsmEntryHeader
{
	uint32_t		cluster_id;
	ItemPointerData tid; /* 6 bytes */
	uint16_t		_pad;
	float			f_add;
	float			f_rescale;
	float			f_error;
	/* uint8_t bits[MKT_RABITQ_BYTES(dim)] follows */
} MktDsmEntryHeader;

#define MKT_DSM_ENTRY_SIZE(dim) \
	MAXALIGN(sizeof(MktDsmEntryHeader) + MKT_RABITQ_BYTES(dim))

typedef struct MktDsmEntries
{
	int		  nparticipants;
	uint32_t  max_per_worker;
	uint32_t  entry_size;
	Dimension dim;
} MktDsmEntries;

static inline pg_atomic_uint32 *
mktann_entry_counts(MktDsmEntries *e)
{
	return (pg_atomic_uint32 *)((char *)e + MAXALIGN(sizeof(MktDsmEntries)));
}

static inline char *
mktann_worker_entries(MktDsmEntries *e, int worker_id)
{
	char *base = (char *)mktann_entry_counts(e) +
				 e->nparticipants * sizeof(pg_atomic_uint32);
	return base + (size_t)worker_id * e->max_per_worker * e->entry_size;
}

static inline Size
mktann_entries_size(int nparticipants, uint32_t max_per_worker, Dimension dim)
{
	Size entry_sz = MKT_DSM_ENTRY_SIZE(dim);
	Size sz		  = MAXALIGN(sizeof(MktDsmEntries));
	sz += (Size)nparticipants * sizeof(pg_atomic_uint32);
	sz += (Size)nparticipants * max_per_worker * entry_sz;
	return sz;
}

/* ----------------------------------------------------------------
 * Shared callbacks — used by both leader and workers
 * ---------------------------------------------------------------- */

typedef struct SampleCbState
{
	float		  *samples;
	float		  *centroids;
	uint32_t	   count;
	uint32_t	   max_samples;
	uint32_t	   stride;
	uint32_t	   stride_counter;
	Dimension	   dim;
	DistanceMetric metric;
	uint32_t	   cent_start;
	uint32_t	   cent_end;
	uint32_t	   cents_picked;
} SampleCbState;

extern void mktann_sample_callback(
		Relation	index,
		ItemPointer tid,
		Datum	   *values,
		bool	   *isnull,
		bool		tuple_is_alive,
		void	   *state);

extern void mktann_km_assign_and_accumulate(
		const float	  *samples,
		uint32_t	   nsamples,
		const float	  *centroids,
		const float	  *norms_c,
		uint32_t	   nlist,
		Dimension	   dim,
		DistanceMetric metric,
		float		  *out_sums,
		uint32_t	  *out_cnts,
		float		  *out_cost);

/*
 * Filtered variant: only processes samples where
 * root_assignments[i] == target_child.
 */
extern void mktann_km_assign_and_accumulate_filtered(
		const float	   *samples,
		uint32_t		nsamples,
		const uint32_t *root_assignments,
		uint32_t		target_child,
		const float	   *centroids,
		const float	   *norms_c,
		uint32_t		nlist,
		Dimension		dim,
		DistanceMetric	metric,
		float		   *out_sums,
		uint32_t	   *out_cnts,
		float		   *out_cost);

/* ----------------------------------------------------------------
 * Phase 3: Entry build callback — shared by leader and workers
 *
 * Encodes vectors with RaBitQ and writes compact entries to a
 * DSM buffer. Leader reads all entries after workers finish and
 * feeds them to serial posting builders.
 * ---------------------------------------------------------------- */

typedef struct EntryBuildCbState
{
	const HKMeansResult *tree;
	MktBuildParams		 bp;
	MktBuildWorkerBufs	 bufs;

	/* DSM entry output */
	char			 *entry_buf;
	pg_atomic_uint32 *count;
	uint32_t		  max_entries;
	uint32_t		  entry_size;

	/* RaBitQ encoding scratch */
	const RaBitQParams *params;
	const float		   *leaf_centroids;
	Dimension			dim;
	RaBitQData		   *enc_buf;
	RaBitQScratch		enc_scratch;

	double		  indtuples;
	double		  soar_dupes;
	MemoryContext tmp_ctx;
} EntryBuildCbState;

extern void entry_build_callback(
		Relation	index,
		ItemPointer tid,
		Datum	   *values,
		bool	   *isnull,
		bool		tuple_is_alive,
		void	   *state);

/* ----------------------------------------------------------------
 * Worker entry point — registered with CreateParallelContext
 * ---------------------------------------------------------------- */

extern void mktann_parallel_build_main(dsm_segment *seg, shm_toc *toc);

#endif /* MKTANN_PARALLEL_H */
