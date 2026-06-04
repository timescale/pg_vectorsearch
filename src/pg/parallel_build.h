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

#include <postgres.h>

#include <storage/barrier.h>
#include <storage/block.h>
#include <storage/itemptr.h>
#include <storage/shm_mq.h>
#include <storage/shm_toc.h>
#include <storage/spin.h>
#include <utils/rel.h>

#include "algo/hkmeans.h"
#include "core/mkt_build_scan.h"
#include "index/posting_build.h"
#include "index/posting_build_parallel.h"
#include "mkt_types.h"
#include "quant/rabitq.h"

/* Forward decl so mkt_build_scan's prototype can reference it without pulling
 * in the executor headers; callers that pass one already have the full type.
 */
struct IndexInfo;

/* ----------------------------------------------------------------
 * DSM table-of-contents keys
 * ---------------------------------------------------------------- */

#define MKTANN_KEY_SHARED		  UINT64CONST(0xB000000000000001)
#define MKTANN_KEY_TREE			  UINT64CONST(0xB000000000000002)
#define MKTANN_KEY_WORKER_OUTPUT  UINT64CONST(0xB000000000000004)
#define MKTANN_KEY_PARTIALS		  UINT64CONST(0xB000000000000005)
#define MKTANN_KEY_WAL_USAGE	  UINT64CONST(0xB000000000000006)
#define MKTANN_KEY_BUFFER_USAGE	  UINT64CONST(0xB000000000000007)
#define MKTANN_KEY_QUERY_TEXT	  UINT64CONST(0xB000000000000008)
#define MKTANN_KEY_BARRIER		  UINT64CONST(0xB000000000000009)
#define MKTANN_KEY_SAMPLES		  UINT64CONST(0xB00000000000000A)
#define MKTANN_KEY_CENTROIDS	  UINT64CONST(0xB00000000000000B)
#define MKTANN_KEY_KM_WORKERS	  UINT64CONST(0xB00000000000000C)
#define MKTANN_KEY_ROOT_ASSIGN	  UINT64CONST(0xB00000000000000E)
#define MKTANN_KEY_POSTING_QUEUES UINT64CONST(0xB00000000000000F)

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
	slock_t mutex;
	double	reltuples;
	double	indtuples;
	double	soar_dupes;

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
 * Per-worker posting output in DSM
 *
 * active[nlist] per worker, packed contiguously: a flag per cluster marking
 * which lists this worker wrote into.
 * ---------------------------------------------------------------- */

static inline Size
mktann_worker_output_size(uint32_t nlist, int nparticipants)
{
	return (Size)nparticipants * nlist * sizeof(bool);
}

static inline bool *
mktann_worker_active(char *base, uint32_t nlist, int worker_id)
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

#define MKTANN_POSTING_QUEUE_PAGES 8

static inline Size
mktann_posting_queue_bytes(void)
{
	/* Ring large enough for several full-page messages, plus slack for
	 * shm_mq's internal header. */
	return (Size)MKTANN_POSTING_QUEUE_PAGES * (BLCKSZ + 64) + 1024;
}

static inline Size
mktann_posting_queues_size(int nparticipants)
{
	return (Size)nparticipants * MAXALIGN(mktann_posting_queue_bytes());
}

static inline char *
mktann_posting_queue(char *base, int worker_id)
{
	return base + (Size)worker_id * MAXALIGN(mktann_posting_queue_bytes());
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
extern void
mktann_sample_cb(void *state, ItemPointerData tid, const float *vec);

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
 * Phase 3: Posting build callback — shared by leader and workers
 *
 * Uses MktPostingWorkerState in deferred batch mode (storage=NULL).
 * After the heap scan, batch pages are copied to DSM. Leader
 * reconstructs batches and calls mkt_posting_materialize().
 * ---------------------------------------------------------------- */

typedef struct PostingCbState
{
	const HKMeansResult	  *tree;
	MktBuildParams		   bp;
	MktBuildWorkerBufs	   bufs;
	MktPostingWorkerState *ws;
	double				   indtuples;
	double				   soar_dupes;
	MemoryContext		   tmp_ctx;
	MemoryContext		   worker_ctx;

	/* Batched secondary assignment: when replication is on and CBLAS is
	 * available, tuples are buffered and the secondary search runs as a
	 * GEMM over the batch (centroids read once per batch). */
	bool			  use_batch;
	MktSecondaryBatch sb;
	float			 *enc_batch;	   /* [B * dim] */
	ItemPointerData	 *batch_tids;	   /* [B] */
	uint32_t		 *batch_primary;   /* [B] */
	float			 *batch_pdist;	   /* [B] */
	uint32_t		 *batch_secondary; /* [B] */
	uint32_t		  batch_count;
} PostingCbState;

/* Initialize/flush/clean the batch buffers; no-op when batching is off
 * (e.g. no replication or no CBLAS). Call init in the worker context
 * before the scan, flush + cleanup after it. */
void posting_cb_batch_init(PostingCbState *cbs);
void posting_cb_batch_flush(PostingCbState *cbs);
void posting_cb_batch_cleanup(PostingCbState *cbs);

/*
 * Shared posting logic, called per live tuple with a raw vector pointer; fed
 * by mkt_build_scan in both back-ends.
 */
extern void posting_cb(void *state, ItemPointerData tid, const float *vec);

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

extern void mktann_parallel_build_main(dsm_segment *seg, shm_toc *toc);

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
struct MktannBuildParams;

extern void mkt_pbuild_teardown(struct ParallelContext *pcxt);
extern bool mkt_pbuild_launch(struct ParallelContext *pcxt, Barrier *barrier);

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
} MktPBuildLeader;

extern bool mkt_pbuild_setup_shared(
		MktPBuildLeader				   *lead,
		Relation						heap,
		Relation						index,
		const struct MktannBuildParams *params,
		int								nworkers);

#endif /* MKT_PARALLEL_BUILD_H */
