/*
 * mktann_parallel.h - PG parallel index build for mktann
 *
 * DSM shared memory structures and worker entry point for parallel
 * index builds using PostgreSQL's parallel worker infrastructure.
 *
 * The leader process runs sampling + k-means, then sets up DSM with
 * the centroid tree, page reservations, and coordination state.
 * Workers cooperatively scan the heap, assign vectors to clusters
 * via tree descent, and stream posting pages to the buffer cache.
 * The leader merges partial pages and writes centroid pages.
 */

#ifndef MKTANN_PARALLEL_H
#define MKTANN_PARALLEL_H

#include <postgres.h>

#include <port/atomics.h>
#include <storage/block.h>
#include <storage/condition_variable.h>
#include <storage/shm_toc.h>
#include <storage/spin.h>

#include "mkt_types.h"

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
	uint32_t	   nlevels;
	int			   nparticipants;

	/* Mutable — spinlock-protected */
	slock_t			  mutex;
	ConditionVariable workersdonecv;
	int				  nparticipantsdone;
	double			  reltuples;
	double			  indtuples;
	double			  soar_dupes;
} MktBuildShared;

#define ParallelTableScanFromMktShared(shared)  \
	((ParallelTableScanDesc)((char *)(shared) + \
							 BUFFERALIGN(sizeof(MktBuildShared))))

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
 * Per-worker output in DSM
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
 * Worker entry point — registered with CreateParallelContext
 * ---------------------------------------------------------------- */

extern void mktann_parallel_build_main(dsm_segment *seg, shm_toc *toc);

#endif /* MKTANN_PARALLEL_H */
