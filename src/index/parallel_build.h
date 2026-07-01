/*
 * parallel_build.h - PG parallel index build (worker + leader shared decls)
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

#ifndef MKT_PARALLEL_BUILD_H
#define MKT_PARALLEL_BUILD_H

/*
 * Back-end primitives the parallel build runs on. The standalone shims live in
 * src/standalone/; a PostgreSQL build uses the real PG headers. This guard is
 * the single place that switch is made — the standalone headers carry no PG
 * branch of their own.
 */
#ifdef MKT_STANDALONE
#include "standalone/barrier.h"
#include "standalone/pg_compat.h" /* Size, BlockNumber, ItemPointerData, Relation */
#include "standalone/shm_mq.h"
#include "standalone/shm_toc.h"
#else
#include <postgres.h>

#include <storage/barrier.h>
#include <storage/block.h>
#include <storage/itemptr.h>
#include <storage/shm_mq.h>
#include <storage/shm_toc.h>
#include <utils/rel.h>
#endif

#include "algo/hkmeans.h"
#include "core/memory.h"
#include "index/centroid_page.h" /* MktCentroidFormat */
#include "index/posting_build.h"
#include "index/posting_build_parallel.h"
#include "index/storage.h" /* MktStorage */
#include "mkt_types.h"
#include "quant/rabitq.h"

/* Forward decl so mkt_build_scan's prototype can reference it without pulling
 * in the executor headers; callers that pass one already have the full type.
 */
struct IndexInfo;

/*
 * Per-vector build scan callback (back-end-neutral): the PG scan adapter
 * unwraps each heap tuple's Datum to the vector's data pointer, and the
 * standalone work-stealing scan points straight into its vector array. The tid
 * is the real heap TID under PG and a reversibly-synthesized one in
 * standalone.
 */
typedef void (*MktBuildScanCb)(
		void *state, ItemPointerData tid, const float *vec);

/* ----------------------------------------------------------------
 * shm_toc region keys (the toc is a DSM segment in PG, a heap arena in
 * standalone). A few regions are PG-only (WAL/buffer usage, query text) but
 * their keys live here with the rest for one contiguous numbering.
 * ---------------------------------------------------------------- */

#define MKT_DSM_KEY_SHARED		   UINT64CONST(0xB000000000000001)
#define MKT_DSM_KEY_TREE		   UINT64CONST(0xB000000000000002)
#define MKT_DSM_KEY_WORKER_OUTPUT  UINT64CONST(0xB000000000000004)
#define MKT_DSM_KEY_PARTIALS	   UINT64CONST(0xB000000000000005)
#define MKT_DSM_KEY_WAL_USAGE	   UINT64CONST(0xB000000000000006)
#define MKT_DSM_KEY_BUFFER_USAGE   UINT64CONST(0xB000000000000007)
#define MKT_DSM_KEY_QUERY_TEXT	   UINT64CONST(0xB000000000000008)
#define MKT_DSM_KEY_BARRIER		   UINT64CONST(0xB000000000000009)
#define MKT_DSM_KEY_SAMPLES		   UINT64CONST(0xB00000000000000A)
#define MKT_DSM_KEY_CENTROIDS	   UINT64CONST(0xB00000000000000B)
#define MKT_DSM_KEY_KM_WORKERS	   UINT64CONST(0xB00000000000000C)
#define MKT_DSM_KEY_ROOT_ASSIGN	   UINT64CONST(0xB00000000000000E)
#define MKT_DSM_KEY_POSTING_QUEUES UINT64CONST(0xB00000000000000F)
#define MKT_DSM_KEY_CHILD_SUBTREES UINT64CONST(0xB000000000000010)
#define MKT_DSM_KEY_REFINE_ACCUM   UINT64CONST(0xB000000000000011)
#define MKT_DSM_KEY_SORTSHARED	   UINT64CONST(0xB000000000000012)
/* Page-backed phase-3 routing: leaf posting-head blocks (head->leaf map) and the
 * global mean, both published by the leader before the tree-ready barrier so
 * workers can route exactly as the query/insert paths do. */
#define MKT_DSM_KEY_POSTING_HEADS  UINT64CONST(0xB000000000000013)
#define MKT_DSM_KEY_GLOBAL_MEAN	   UINT64CONST(0xB000000000000014)

/* ----------------------------------------------------------------
 * MktBuildShared — back-end-neutral shared build state
 *
 * Immutable config set by the leader before the workers start, plus counters
 * the workers update concurrently during the posting scan under a lock the
 * back-end owns (mkt_pbuild_worker_add_counts). Each back-end embeds this as
 * the first member of its own struct carrying the back-end-specific state —
 * for PG, the relation OIDs, query id, the spinlock, and a trailing
 * ParallelTableScanDesc (see MktBuildSharedPg in parallel_backend.c).
 * ---------------------------------------------------------------- */

typedef struct MktBuildShared
{
	/* Immutable — set by the leader before launch */
	Dimension		  dim;
	DistanceMetric	  metric;
	uint32_t		  nlist;
	uint32_t		  fan_out;
	double			  soar_lambda;
	double			  boundary_epsilon;
	bool			  fastscan;
	MktCentroidFormat centroid_format;
	uint64_t		  rabitq_seed;
	int				  nparticipants;
	/* CREATE INDEX CONCURRENTLY: the leader scans with an MVCC snapshot and
	 * participants take weak relation locks. Workers must mark their rebuilt
	 * IndexInfo concurrent too, or heapam's snapshot/OldestXmin check trips.
	 */
	bool concurrent;
	/* Posting sort work budget (KB); PG sets it from maintenance_work_mem,
	 * standalone leaves it 0 (its in-memory sorter ignores it). The shared
	 * phase-3 code reads this instead of the PG-only GUC. */
	int work_mem_kb;

	/* K-means config */
	uint32_t max_samples_per_worker;
	uint32_t km_max_iterations;
	float	 km_tolerance;
	uint32_t km_k; /* root k-means k (= fan_out) */

	/* Full-table leaf-centroid refinement passes (0 = off). Set by the leader
	 * when the sample set is maintenance_work_mem-bounded; standalone leaves
	 * it 0. Both leader and workers gate the refine phase on this, so they run
	 * the same barrier sequence. */
	uint32_t refine_iters;

	/* Counters — updated concurrently under the back-end's lock */
	double reltuples;
	double indtuples;
	double soar_dupes;

	/* K-means convergence — set by leader between barriers */
	bool km_converged;

	/* Per-child subtree blob slot size (bytes) in the child-subtrees region;
	 * set by the leader before launch so workers can index their slot. */
	uint64_t subtree_slot_size;

	/* Page-backed routing knobs (mirror the mkt.centroid_* GUCs), so phase-3
	 * workers build a MktIndexBase that routes identically to the query path.
	 * fastscan_bits is the FASTSCAN centroid bit width (base.fastscan when the
	 * centroid format is FASTSCAN; 0 otherwise). */
	float centroid_error_scale;
	float centroid_beam_scale;
	int	  fastscan_bits;
} MktBuildShared;

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
	/* counts[nparticipants], then the per-worker sample blocks
	 * (samples[worker][max_per_worker * dim]) packed right after. */
	uint32_t counts[];
} MktDsmSamples;

static inline uint32_t *
mkt_dsm_sample_counts(MktDsmSamples *s)
{
	return s->counts;
}

static inline float *
mkt_dsm_worker_samples(MktDsmSamples *s, int worker_id)
{
	/* The sample blocks begin right after counts[nparticipants]. */
	float *samples = (float *)(s->counts + s->nparticipants);
	return samples + (size_t)worker_id * s->max_per_worker * s->dim;
}

static inline Size
mkt_dsm_samples_size(int nparticipants, uint32_t max_per_worker, Dimension dim)
{
	Size sz = offsetof(MktDsmSamples, counts);
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
	/* assignments[worker][max_per_worker] packed contiguously. */
	uint32_t assignments[];
} MktDsmRootAssign;

static inline uint32_t *
mkt_dsm_root_assignments(MktDsmRootAssign *ra, int worker_id)
{
	return ra->assignments + (size_t)worker_id * ra->max_per_worker;
}

static inline Size
mkt_dsm_root_assign_size(int nparticipants, uint32_t max_per_worker)
{
	Size sz = offsetof(MktDsmRootAssign, assignments);
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
mkt_dsm_centroids_size(uint32_t nlist, Dimension dim)
{
	return (Size)nlist * dim * sizeof(float) + (Size)nlist * sizeof(float);
}

static inline float *
mkt_dsm_centroids(char *base)
{
	return (float *)base;
}

/* ----------------------------------------------------------------
 * Child subtrees: per-root-child HKMeansResult blobs in DSM
 *
 * After the root k-means splits the samples into fan_out groups, each group's
 * subtree is built independently (work-partitioned across participants) and
 * written, as a contiguous HKMeansResult, into a fixed-size slot here. The
 * leader then grafts the fan_out subtrees under a fresh root. One slot per
 * root child; slot size is the worst-case blob for a subtree (shared in
 * MktBuildShared.subtree_slot_size).
 * ---------------------------------------------------------------- */

static inline Size
mkt_dsm_child_subtrees_size(uint32_t fan_out, uint64_t slot_size)
{
	return (Size)fan_out * (Size)slot_size;
}

static inline char *
mkt_dsm_child_subtree(char *base, uint32_t child, uint64_t slot_size)
{
	return base + (Size)child * (Size)slot_size;
}

/*
 * Number of hierarchy levels for nlist leaves with the given fan_out. The
 * leader and workers compute this identically to agree on whether the build
 * is flat (1 level, no child subtrees) or hierarchical (>= 2 levels).
 */
static inline uint32_t
mkt_compute_nlevels(uint32_t nlist, uint32_t fan_out)
{
	uint32_t nlevels = 1;
	uint32_t n		 = nlist;
	while (n > fan_out)
	{
		n = (n + fan_out - 1) / fan_out;
		nlevels++;
	}
	return nlevels;
}

/*
 * Worst-case leaf count for a tree targeting `nlist` leaves at this fan_out:
 * fan_out^nlevels (clamped to >= nlist). The exact leaf count isn't known
 * until k-means runs, so the parallel build uses this to size its DSM regions
 * up front, then narrows to the real tree->nleaves afterward.
 */
static inline uint32_t
mkt_max_nlist(uint32_t nlist, uint32_t fan_out)
{
	uint32_t nlevels   = mkt_compute_nlevels(nlist, fan_out);
	uint32_t max_nlist = 1;
	for (uint32_t l = 0; l < nlevels; l++)
		max_nlist *= fan_out;
	return max_nlist < nlist ? nlist : max_nlist;
}

static inline float *
mkt_dsm_norms_c(char *base, uint32_t nlist, Dimension dim)
{
	/* norms_c[nlist] sits right after centroids[nlist * dim]. */
	return mkt_dsm_centroids(base) + (size_t)nlist * dim;
}

/* ----------------------------------------------------------------
 * K-means per-worker accumulators in DSM
 *
 * Per worker: centroid_sums[nlist * dim] + centroid_cnts[nlist]
 *             + cost (1 float)
 * ---------------------------------------------------------------- */

static inline Size
mkt_dsm_km_worker_size(uint32_t nlist, Dimension dim)
{
	return (Size)nlist * dim * sizeof(float) + (Size)nlist * sizeof(uint32_t) +
		   sizeof(float);
}

static inline Size
mkt_dsm_km_workers_size(int nparticipants, uint32_t nlist, Dimension dim)
{
	return (Size)nparticipants * mkt_dsm_km_worker_size(nlist, dim);
}

static inline float *
mkt_dsm_km_worker_sums(
		char *base, uint32_t nlist, Dimension dim, int worker_id)
{
	return (float *)(base +
					 (size_t)worker_id * mkt_dsm_km_worker_size(nlist, dim));
}

static inline uint32_t *
mkt_dsm_km_worker_cnts(
		char *base, uint32_t nlist, Dimension dim, int worker_id)
{
	/* cnts[nlist] sits right after sums[nlist * dim]. */
	return (uint32_t *)(mkt_dsm_km_worker_sums(base, nlist, dim, worker_id) +
						(size_t)nlist * dim);
}

static inline float *
mkt_dsm_km_worker_cost(
		char *base, uint32_t nlist, Dimension dim, int worker_id)
{
	/* cost sits right after cnts[nlist]. */
	return (float *)(mkt_dsm_km_worker_cnts(base, nlist, dim, worker_id) +
					 nlist);
}

/* ----------------------------------------------------------------
 * Per-worker posting output in DSM
 *
 * active[nlist] per worker, packed contiguously: a flag per cluster marking
 * which lists this worker wrote into.
 * ---------------------------------------------------------------- */

static inline Size
mkt_dsm_worker_output_size(uint32_t nlist, int nparticipants)
{
	return (Size)nparticipants * nlist * sizeof(bool);
}

static inline bool *
mkt_dsm_worker_active(char *base, uint32_t nlist, int worker_id)
{
	return (bool *)(base + (Size)worker_id * nlist * sizeof(bool));
}

/* ----------------------------------------------------------------
 * Partials buffer in DSM
 *
 * partials[worker_id * nlist + cluster] = one BLCKSZ page
 * Only allocated when !fastscan.
 * ---------------------------------------------------------------- */

static inline Size
mkt_dsm_partials_size(uint32_t nlist, int nparticipants)
{
	return (Size)nparticipants * nlist * BLCKSZ;
}

static inline char *
mkt_dsm_worker_partials(char *base, uint32_t nlist, int worker_id)
{
	return base + (Size)worker_id * nlist * BLCKSZ;
}

/* ----------------------------------------------------------------
 * Per-worker shm_mq posting-page queues
 *
 * Each worker streams its completed full pages to the leader over a
 * single-reader/single-writer shm_mq (worker = sender, leader =
 * receiver). One message = one BLCKSZ page; the leader reads the
 * cluster id and first/continuation flag from the page header to place
 * it in that list's reserved block range. The ring doubles as the flush
 * buffer: when it fills, the worker's send blocks (backpressure), which
 * is what keeps worker memory bounded to ~one working page per cluster.
 * A worker detaches its queue when done; the leader drains until all
 * queues are detached.
 * ---------------------------------------------------------------- */

#define MKT_DSM_POSTING_QUEUE_PAGES 8

static inline Size
mkt_dsm_posting_queue_bytes(void)
{
	/* Ring large enough for several full-page messages, plus slack for
	 * shm_mq's internal header. */
	return (Size)MKT_DSM_POSTING_QUEUE_PAGES * (BLCKSZ + 64) + 1024;
}

static inline Size
mkt_dsm_posting_queues_size(int nparticipants)
{
	return (Size)nparticipants * MAXALIGN(mkt_dsm_posting_queue_bytes());
}

static inline char *
mkt_dsm_posting_queue(char *base, int worker_id)
{
	return base + (Size)worker_id * MAXALIGN(mkt_dsm_posting_queue_bytes());
}

/* ----------------------------------------------------------------
 * Shared callbacks — used by both leader and workers
 * ---------------------------------------------------------------- */

typedef struct SampleCbState
{
	float		  *samples;
	uint32_t	   count;
	uint32_t	   max_samples;
	uint32_t	   stride;
	uint32_t	   stride_counter;
	Dimension	   dim;
	DistanceMetric metric;
} SampleCbState;

/*
 * Shared sampling logic, called per live tuple with a raw vector pointer (no
 * Datum) so the same code serves both back-ends. mkt_build_scan feeds it: the
 * PG scan unwraps each heap tuple's Datum, the standalone scan passes its
 * in-memory vectors directly.
 */
extern void mkt_sample_cb(void *state, ItemPointerData tid, const float *vec);

extern void mkt_km_assign_and_accumulate(
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
 * Phase 2c: work-partitioned subtree build (shared by leader and workers).
 *
 * After the root k-means, each participant builds the full subtree for the
 * root children it owns (child where child % nparticipants == participant_id):
 * it gathers that child's samples from the pooled DSM sample slots (via the
 * root assignment), runs hierarchical k-means to depth nlevels-1, and writes
 * the resulting contiguous HKMeansResult into the child's fixed-size slot in
 * the child-subtrees region. The children are independent, so there is no
 * synchronization inside this call — only the single barrier the caller issues
 * afterward. The leader then grafts the fan_out subtrees under a fresh root.
 * root_cents (km_k * dim) supplies the centroid for any (degenerate) empty
 * child. slot_size is MktBuildShared.subtree_slot_size.
 */
extern void mkt_subtree_build_partitioned(
		int				  participant_id,
		int				  nparticipants,
		MktDsmSamples	 *dsm_samples,
		MktDsmRootAssign *dsm_ra,
		const float		 *root_cents,
		uint32_t		  km_k,
		uint32_t		  nlist,
		uint32_t		  fan_out,
		Dimension		  dim,
		DistanceMetric	  metric,
		uint32_t		  km_max_iterations,
		char			 *subtrees_base,
		uint64_t		  slot_size);

/*
 * Per-participant execution of phases 1, 2, and 2b, shared by the leader
 * (participant 0) and the workers — see the contract on the definitions in
 * parallel_build_worker.c. Each runs its phase's per-participant work and the
 * phase barrier(s); the leader-only seed/reduce in the k-means phase is gated
 * on participant_id == 0. mkt_pbuild_exec_kmeans returns the iteration count
 * (for the leader's timing log).
 */
extern void mkt_pbuild_exec_sampling(
		int				  participant_id,
		Relation		  heap,
		Relation		  index,
		struct IndexInfo *index_info,
		MktBuildShared	 *shared,
		MktDsmSamples	 *dsm_samples,
		Barrier			 *barrier);

extern uint32_t mkt_pbuild_exec_kmeans(
		int				participant_id,
		MktBuildShared *shared,
		MktDsmSamples  *dsm_samples,
		char		   *centroids_base,
		char		   *km_workers_base,
		Barrier		   *barrier);

extern void mkt_pbuild_exec_root_assign(
		int				  participant_id,
		MktBuildShared	 *shared,
		MktDsmSamples	 *dsm_samples,
		MktDsmRootAssign *dsm_ra,
		char			 *centroids_base,
		Barrier			 *barrier);

/* ----------------------------------------------------------------
 * Full-table leaf-centroid refinement (parallel)
 *
 * A single shared accumulator (sums[nleaves*dim] + counts[nleaves]) holds the
 * per-leaf running mean for all participants -- one copy, independent of the
 * worker count, so it scales to fine nlist where per-worker accumulators would
 * not. Concurrent updates are guarded by a striped lock array owned by the
 * back-end (mkt_pbuild_accum_lock/unlock); contention is low because rows
 * spread across nleaves leaves. sums uses float (means of normalized vectors).
 * ---------------------------------------------------------------- */

#define MKT_REFINE_LOCK_STRIPES 256

typedef struct MktDsmRefineAccum
{
	uint32_t  nleaves;
	Dimension dim;
	/* float sums[nleaves * dim], then uint64 counts[nleaves], packed after. */
	float sums[FLEXIBLE_ARRAY_MEMBER];
} MktDsmRefineAccum;

static inline float *
mkt_dsm_refine_sums(MktDsmRefineAccum *a)
{
	return a->sums;
}

static inline uint64_t *
mkt_dsm_refine_counts(MktDsmRefineAccum *a)
{
	return (uint64_t *)(a->sums + (size_t)a->nleaves * a->dim);
}

static inline Size
mkt_dsm_refine_accum_size(uint32_t nleaves, Dimension dim)
{
	Size sz = offsetof(MktDsmRefineAccum, sums);
	sz += (Size)nleaves * dim * sizeof(float);
	sz += (Size)nleaves * sizeof(uint64_t);
	return sz;
}

/*
 * Leaves per refine tile: the accumulator holds at most this many leaves, so
 * it is a bounded constant (cap_bytes, derived from maintenance_work_mem and
 * MaxAllocSize by the caller) rather than O(nleaves). nleaves above it just
 * means more re-scanned tiles, not a bigger allocation. The DSM region is
 * sized for this (capacity = accum->nleaves); the leader and workers derive
 * the tile count from it identically, so they stay in barrier lockstep.
 */
static inline uint32_t
mkt_refine_tile_leaves(uint32_t nleaves, Dimension dim, uint64_t cap_bytes)
{
	uint64_t per_leaf = (uint64_t)dim * sizeof(float) + sizeof(uint64_t);
	uint64_t t		  = cap_bytes / (per_leaf ? per_leaf : 1);
	if (t < 1)
		t = 1;
	if (t > nleaves)
		t = nleaves;
	return (uint32_t)t;
}

/*
 * Refine the tree's leaf centroids on the whole table: refine_iters streaming
 * passes, each routing every row to its leaf and recomputing per-leaf means.
 * Both leader (participant 0) and workers call it; gated by
 * shared->refine_iters so they run the same barriers. The leader
 * zeroes/divides and reinitializes the scan each pass. tree points at the
 * shared (DSM) tree, updated in place.
 */
extern void mkt_pbuild_exec_refine(
		int				   participant_id,
		Relation		   heap,
		Relation		   index,
		struct IndexInfo  *index_info,
		MktBuildShared	  *shared,
		HKMeansResult	  *tree,
		MktDsmRefineAccum *accum,
		Barrier			  *barrier);

/* Striped lock seam for the refine accumulator (back-end owns the locks). */
extern void mkt_pbuild_accum_lock(MktBuildShared *shared, uint32_t stripe);
extern void mkt_pbuild_accum_unlock(MktBuildShared *shared, uint32_t stripe);

/*
 * Scan every vector cooperatively, invoking cb per live tuple. Back-end seam:
 * the PG implementation (parallel_backend_pg.c) drives a parallel heap scan
 * and unwraps each tuple; the standalone implementation iterates its in-memory
 * vector array. allow_sync/progress are PG table_index_build_scan flags,
 * ignored in standalone.
 */
extern void mkt_build_scan(
		Relation		  heap,
		Relation		  index,
		struct IndexInfo *indexInfo,
		MktBuildShared	 *shared,
		bool			  allow_sync,
		bool			  progress,
		MktBuildScanCb	  cb,
		void			 *state);

/* ----------------------------------------------------------------
 * Worker entry point — registered with CreateParallelContext
 * ---------------------------------------------------------------- */

extern void mkt_parallel_build_main(dsm_segment *seg, shm_toc *toc);

/* ----------------------------------------------------------------
 * Worker lifecycle seam — back-end-specific (parallel_backend.c for PG)
 *
 * Runtime handles for one participant. mkt_pbuild_worker_attach fills it when
 * the worker joins the build; mkt_pbuild_worker_detach tears it down. The PG
 * version opens the heap/index relations and reports instrumentation; the
 * standalone version takes the shared state and vectors directly and joins a
 * thread barrier.
 * ---------------------------------------------------------------- */

typedef struct MktPBuildWorker
{
	MktBuildShared *shared;
	Barrier		   *barrier;
	Relation		heapRel;
	Relation		indexRel;
	int				worker_id;
	Dimension		dim;
} MktPBuildWorker;

extern void mkt_pbuild_worker_attach(shm_toc *toc, MktPBuildWorker *w);
extern void mkt_pbuild_worker_detach(shm_toc *toc, MktPBuildWorker *w);

/*
 * Page-backed phase-3 routing storage seam — back-end-specific. Workers read
 * centroid + posting head pages while routing, so each needs a MktStorage over
 * the index. PG (separate process) opens one on the worker's indexRel;
 * standalone (threads) returns the leader's shared in-memory store published via
 * mkt_pbuild_publish_storage before launch. Release is a no-op for standalone.
 */
extern void mkt_pbuild_publish_storage(MktBuildShared *shared, MktStorage *s);
extern MktStorage *mkt_pbuild_worker_storage(MktPBuildWorker *w);
extern void		   mkt_pbuild_worker_storage_release(MktStorage *s);

/*
 * Accumulate one worker's tuple counts into the shared state under the
 * back-end's lock (the lock lives in the back-end's derived shared struct).
 */
extern void mkt_pbuild_worker_add_counts(
		MktBuildShared *shared, double indtuples, double soar_dupes);

/* ----------------------------------------------------------------
 * Leader launch/teardown seam — back-end-specific (parallel_backend.c for PG)
 *
 * mkt_pbuild_launch starts the workers and blocks until the whole party has
 * attached to the phase barrier (returning false, after teardown, if none
 * started). mkt_pbuild_teardown frees the parallel context. The standalone
 * versions spawn/join threads and free the shared arena. (struct
 * ParallelContext is PostgreSQL's; standalone provides its own definition.)
 * ---------------------------------------------------------------- */

struct ParallelContext;
struct WalUsage;
struct BufferUsage;

extern void mkt_pbuild_teardown(struct ParallelContext *pcxt);
extern bool mkt_pbuild_launch(struct ParallelContext *pcxt, Barrier *barrier);

/* ----------------------------------------------------------------
 * Build configuration — back-end-neutral input to the setup seam
 *
 * The fields the setup seam needs to size and populate the shared state. The
 * PG path fills this from its opclass-resolved MktannBuildParams; the
 * standalone path fills it from its own config. Keeping it neutral lets one
 * setup-seam signature serve both back-ends.
 * ---------------------------------------------------------------- */

typedef struct MktBuildConfig
{
	Dimension		  dim;
	DistanceMetric	  metric;
	MktCentroidFormat centroid_format;
	uint32_t		  nlist;
	uint32_t		  fan_out;
	double			  soar_lambda;
	double			  boundary_epsilon;
	bool			  fastscan;
	/* CREATE INDEX CONCURRENTLY: take weak locks + an MVCC scan snapshot. */
	bool concurrent;
} MktBuildConfig;

/* ----------------------------------------------------------------
 * Leader setup seam — back-end-specific (parallel_backend.c for PG)
 *
 * Leader-side runtime state: the parallel context, the shared regions, and the
 * derived sizes. mkt_pbuild_setup_shared allocates and populates them (a DSM
 * segment + regions in PG, a heap arena in standalone) and fills this struct;
 * the rest of the driver consumes it. Returns false (after teardown) if the
 * back-end could not start. The WalUsage/BufferUsage fields are PG
 * instrumentation, unused in standalone.
 * ---------------------------------------------------------------- */

typedef struct MktPBuildLeader
{
	struct ParallelContext *pcxt;
	MktBuildShared		   *shared;
	Barrier				   *barrier;
	MktDsmSamples		   *dsm_samples;
	char				   *centroids_base;
	float				   *cents;
	char				   *km_workers_base;
	MktDsmRootAssign	   *dsm_ra;
	void				   *dsm_tree;
	char				   *queues_base;
	char				   *dsm_partials;
	char				   *child_subtrees_base; /* per-child subtree blobs */
	struct WalUsage		   *walusage;
	struct BufferUsage	   *bufferusage;
	int						nparticipants;
	uint32_t				km_k;
	uint32_t				max_per_worker;
	Dimension				dim;
	uint32_t				nlist;
	uint64_t				rabitq_seed;
	uint32_t				fan_out;
	Size					max_tree_sz;
	Size					dsm_total; /* committed DSM chunk bytes (for the
										  planned-allocation introspection line) */
} MktPBuildLeader;

extern bool mkt_pbuild_setup_shared(
		MktPBuildLeader		 *lead,
		Relation			  heap,
		Relation			  index,
		const MktBuildConfig *config,
		int					  nworkers);

/*
 * Re-initialize the scan for the posting pass (the sampling pass consumed the
 * first). Back-end seam: resets the PG parallel-scan descriptor or, in
 * standalone, the work-stealing cursor.
 */
extern void mkt_pbuild_rescan(Relation heap, MktBuildShared *shared);

/* ----------------------------------------------------------------
 * Posting sort seam — cluster-keyed external sort of encoded entries.
 *
 * Back-end seam (like mkt_build_scan): the PG back-end implements it with a
 * parallel tuplesort whose shared coordinator (Sharedsort) lives in the build
 * DSM; the standalone back-end with per-participant in-memory arrays merged
 * and sorted by cluster. The shared phase-3 code
 * (parallel_build_worker/leader) calls only this seam, so the posting build
 * stays bounded by maintenance_work_mem without the standalone core depending
 * on PostgreSQL's tuplesort.
 *
 * Lifecycle:
 *   leader, in setup_shared before launch:  size the DSM region with
 *       mkt_pbuild_sort_shared_size(); after launch:
 * mkt_pbuild_sort_shared_init(). each worker:  begin(is_leader=false) ->
 * put... -> performsort -> end. leader:       begin(is_leader=true) ->
 * performsort -> getnext... -> end. Entries are fixed-size opaque blobs sorted
 * by the uint32 cluster key. `seg` and `region` are void* to keep PostgreSQL
 * types out of the shared header (the PG impl casts seg to dsm_segment*);
 * `region` is the MKT_DSM_KEY_SORTSHARED chunk.
 * ---------------------------------------------------------------- */
typedef struct MktSorter MktSorter;

extern Size mkt_pbuild_sort_shared_size(int nparticipants);
extern void
mkt_pbuild_sort_shared_init(void *region, int nparticipants, void *seg);
extern MktSorter *mkt_pbuild_sort_begin(
		void	*region,
		void	*seg,
		int		 participant,
		int		 nparticipants,
		bool	 is_leader,
		uint32_t entry_size,
		int		 work_mem_kb);
extern void
mkt_pbuild_sort_put(MktSorter *sorter, uint32_t cluster, const void *entry);
extern void mkt_pbuild_sort_performsort(MktSorter *sorter);
extern bool mkt_pbuild_sort_getnext(
		MktSorter *sorter, uint32_t *cluster, const void **entry);
extern void mkt_pbuild_sort_end(MktSorter *sorter);

/*
 * Build every cluster's posting list from a populated (not yet performsorted)
 * cluster-keyed sorter. Shared by the serial build and the parallel leader:
 * performsort, then read entries grouped by cluster and write each list with a
 * single resident page builder. Fills posting_heads[nlist] and ends the
 * sorter.
 */
extern void mkt_posting_build_lists(
		MktSorter		   *sorter,
		MktStorage		   *storage,
		uint32_t			nlist,
		Dimension			dim,
		bool				fastscan,
		const RaBitQParams *rq_params,
		const float		   *ref_vecs,
		MktPostingReserve  *reserve,
		BlockNumber			first_posting,
		BlockNumber		   *posting_heads);

/*
 * Route a vector to its refinement leaf (normalizing into scratch for cosine,
 * since the tree is trained in normalized space). Sets *out_v to the vector to
 * accumulate and returns its leaf. Shared by the serial and parallel refine
 * passes so both route identically.
 */
extern uint32_t mkt_refine_assign_leaf(
		const HKMeansResult *tree,
		const float			*vec,
		Dimension			 dim,
		DistanceMetric		 metric,
		float				*scratch,
		const float		   **out_v);

/*
 * Shared pre-posting centroid setup: normalize leaf centroids for cosine,
 * reserve block 0 (metadata) plus the centroid pages, and return the
 * posting-area start block. Shared by the serial build and the parallel
 * leader. The rotated P^T*centroid each posting list needs is computed on the
 * fly, per cluster, in mkt_posting_build_lists (no nlist*dim array).
 */
extern BlockNumber mkt_build_setup_centroid_layout(
		MktStorage		 *storage,
		HKMeansResult	 *tree,
		Dimension		  dim,
		uint32_t		  nlist,
		DistanceMetric	  metric,
		MktCentroidFormat centroid_format);

/* ----------------------------------------------------------------
 * Parallel build entry — shared driver (parallel_build_leader.c)
 *
 * Runs sampling + k-means + the bounded streaming posting build over the heap
 * (its vectors) into storage, returning the centroid tree and per-list posting
 * heads. Returns false if parallelism could not start, so the caller falls
 * back to a serial build. The PG caller fills config from its resolved build
 * params; the standalone caller fills it from its index config.
 *
 * prog is the build-progress reporting seam (phases + % to
 * pg_stat_progress_create_index, per-phase stats under mkt.log_build_stats).
 * The PG caller passes its MktBuildProgress so the parallel phases surface the
 * same way the serial ones do; the standalone caller passes NULL (the seam is
 * a no-op stub there).
 * ---------------------------------------------------------------- */

struct MktBuildProgress; /* index/build_progress.h */

extern bool do_parallel_build(
		Relation				 heap,
		Relation				 index,
		struct IndexInfo		*index_info,
		const MktBuildConfig	*config,
		MktStorage				*storage,
		struct MktBuildProgress *prog,
		HKMeansResult		   **out_tree,
		BlockNumber				*posting_heads,
		double					*out_heap_tuples,
		double					*out_indtuples,
		double					*out_soar_dupes,
		/* The build routes page-backed, so it writes the centroid pages + heads
		 * into `storage` before the scan and computes the global mean. Callers
		 * receive both: *out_global_mean is the (owned) mean used for the centroid
		 * pages (so the caller's metadata write matches), and
		 * *out_centroids_written signals that centroid pages already exist in
		 * `storage` (the caller must not rewrite them there). May be NULL. */
		float				   **out_global_mean,
		bool					*out_centroids_written);

#endif /* MKT_PARALLEL_BUILD_H */
