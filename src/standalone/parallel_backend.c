/*
 * parallel_backend.c - Standalone implementations of the build's back-end
 * seams.
 *
 * The build driver (parallel_build_leader.c + parallel_build_worker.c) is
 * shared with the PG extension; the parts that genuinely differ by back-end
 * are expressed as same-named seam functions. This file is the thread-based
 * mirror of src/pg/parallel_backend.c: the shared state lives in a heap arena
 * instead of a DSM segment, workers run on the thread pool instead of
 * background processes, and the scan walks the in-memory vector array instead
 * of a heap relation. Memory bounding is deliberately out of scope here: the
 * standalone engine holds its vectors, samples, and subtree blobs in RAM (it
 * exists to isolate and benchmark the shared build logic), so the
 * maintenance_work_mem-style budgets the PG back-end enforces have no
 * standalone equivalent -- bounded-memory behavior (sample caps, spill,
 * refine tiling) is exercised only through the PG build.
 */

#ifdef VS_STANDALONE

#include "vs_config.h"

#include <pthread.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "algo/hkmeans.h"
#include "core/log.h"
#include "core/memory.h"
#include "index/index_build.h" /* prism_auto_fan_out */
#include "index/parallel_build.h"
#include "quant/rabitq.h"
#include "standalone/parallel_ctx.h"
#include "standalone/parallel_scan.h" /* PrismParallelScan */

/*
 * Standalone shared build state: the neutral PrismBuildShared plus a mutex
 * guarding its counters, the heap/index relations the workers attach to, and
 * the work-stealing scan cursor (the analog of PG's appended
 * ParallelTableScanDesc). The base is the first member, so the
 * PrismBuildShared
 * * the workers look up out of the toc is recovered here as a
 * PrismBuildSharedStandalone *.
 */
typedef struct PrismBuildSharedStandalone
{
	PrismBuildShared  base;
	pthread_mutex_t	  mutex;
	Relation		  heap;
	Relation		  index;
	PrismParallelScan scan;
	/* Leader's in-memory page store, published for phase-3 page-backed
	 * routing; workers are threads so they share the pointer directly. */
	VsStorage *storage;
	/* Per-child subtree ring, allocated by the leader post-assign; workers
	 * are threads and read the pointer directly. */
	char *subtree_ring;
	/* Exact centroid collection, allocated by the leader after the
	 * streaming tree write; workers are threads and read the pointer
	 * directly. */
	char *exact_centroids;
} PrismBuildSharedStandalone;

/*
 * Scan every vector, invoking cb per vector. The shared work-stealing cursor
 * lives in the shared state (initialized by setup_shared from the heap's
 * vector array); every participant runs it cooperatively. The PG flags
 * (allow_sync/progress) and the relation/index-info handles are unused here.
 */
double
prism_build_scan(
		Relation		  heap,
		Relation		  index,
		struct IndexInfo *indexInfo,
		PrismBuildShared *shared,
		bool			  allow_sync,
		bool			  progress,
		PrismBuildScanCb  cb,
		void			 *state)
{
	PrismBuildSharedStandalone *sh = (PrismBuildSharedStandalone *)shared;

	(void)heap;
	(void)index;
	(void)indexInfo;
	(void)allow_sync;
	(void)progress;

	return prism_parallel_scan_run(&sh->scan, cb, state);
}

/*
 * Join the parallel build: look up the shared state and barrier, take the
 * heap/index relations the leader stored, and attach to the phase barrier. No
 * relations to open and no instrumentation to start (single process).
 */
void
prism_pbuild_worker_attach(shm_toc *toc, PrismPBuildWorker *w)
{
	PrismBuildShared *shared =
			shm_toc_lookup(toc, PRISM_DSM_KEY_SHARED, false);
	PrismBuildSharedStandalone *sh = (PrismBuildSharedStandalone *)shared;
	Barrier *barrier = shm_toc_lookup(toc, PRISM_DSM_KEY_BARRIER, false);

	w->shared	 = shared;
	w->barrier	 = barrier;
	w->heapRel	 = sh->heap;
	w->indexRel	 = sh->index;
	w->worker_id = ParallelWorkerNumber + 1;
	w->dim		 = shared->dim;

	BarrierAttach(barrier);
}

/*
 * Leave the parallel build. The barrier was already detached in the worker
 * body at the phase 2->3 transition, the relations are not owned here, and
 * there is no instrumentation to report, so this is a no-op.
 */
void
prism_pbuild_worker_detach(shm_toc *toc, PrismPBuildWorker *w)
{
	(void)toc;
	(void)w;
}

/*
 * Page-backed routing storage seam (see parallel_build.h). Workers are threads
 * in the leader's process, so they share the leader's in-memory page store:
 * the leader publishes it and workers return the same pointer. Release is a
 * no-op.
 */
void
prism_pbuild_publish_storage(PrismBuildShared *shared, VsStorage *s)
{
	((PrismBuildSharedStandalone *)shared)->storage = s;
}

VsStorage *
prism_pbuild_worker_storage(PrismPBuildWorker *w)
{
	return ((PrismBuildSharedStandalone *)w->shared)->storage;
}

void
prism_pbuild_worker_storage_release(VsStorage *s)
{
	(void)s; /* shared with the leader; not owned by the worker */
}

/*
 * Tear the parallel context down (joins the worker threads and frees the
 * arena) and leave parallel mode.
 */
void
prism_pbuild_teardown(ParallelContext *pcxt)
{
	DestroyParallelContext(pcxt);
	ExitParallelMode();
}

/*
 * Leader-only subtree blob store (see parallel_build.h). Standalone is the
 * in-memory engine, so the store is a growing byte buffer with a read
 * cursor; the PG back-end spills through a BufFile temp file instead.
 */
struct PrismBlobStore
{
	char	*data;
	uint64_t size;
	uint64_t cap;
	uint64_t rpos;
	/* The store outlives pass-scoped scratch contexts (the plan pass fills
	 * it, the write pass drains it), so growth allocates in the context the
	 * store was created in, not the caller's current one. */
	VsMemCtx ctx;
};

PrismBlobStore *
prism_pbuild_blobstore_begin(void)
{
	PrismBlobStore *bs = vs_alloc0(sizeof(PrismBlobStore));
	bs->ctx			   = vs_current_memctx; /* the creating context */
	return bs;
}

void
prism_pbuild_blobstore_put(PrismBlobStore *bs, const void *blob, uint64_t size)
{
	uint64_t need = bs->size + sizeof(size) + size;
	if (need > bs->cap)
	{
		uint64_t cap = bs->cap ? bs->cap : (uint64_t)1 << 20;
		while (cap < need)
			cap *= 2;
		VsMemCtx old   = vs_memctx_switch(bs->ctx);
		char	*grown = vs_alloc(cap);
		vs_memctx_switch(old);
		if (bs->data != NULL)
		{
			memcpy(grown, bs->data, bs->size);
			vs_free(bs->data);
		}
		bs->data = grown;
		bs->cap	 = cap;
	}
	memcpy(bs->data + bs->size, &size, sizeof(size));
	bs->size += sizeof(size);
	memcpy(bs->data + bs->size, blob, size);
	bs->size += size;
}

void
prism_pbuild_blobstore_rewind(PrismBlobStore *bs)
{
	bs->rpos = 0;
}

uint64_t
prism_pbuild_blobstore_get(PrismBlobStore *bs, void *buf, uint64_t max_size)
{
	uint64_t size;
	memcpy(&size, bs->data + bs->rpos, sizeof(size));
	bs->rpos += sizeof(size);
	if (size > max_size)
		vs_error("subtree blob larger than its slot");
	memcpy(buf, bs->data + bs->rpos, size);
	bs->rpos += size;
	return size;
}

void
prism_pbuild_blobstore_end(PrismBlobStore *bs)
{
	vs_free(bs->data);
	vs_free(bs);
}

/*
 * Launch the workers on the thread pool and wait until the whole party has
 * attached to the phase barrier. Returns false (after teardown) if none
 * started. The poll uses a 1ms latch timeout rather than a wakeup, since no
 * one sets the leader's latch during attach.
 */
bool
prism_pbuild_launch(
		ParallelContext *pcxt, Barrier *barrier, PrismBuildShared *shared)
{
	LaunchParallelWorkers(pcxt);

	if (pcxt->nworkers_launched == 0)
	{
		WaitForParallelWorkersToFinish(pcxt);
		prism_pbuild_teardown(pcxt);
		return false;
	}

	WaitForParallelWorkersToAttach(pcxt);
	while (BarrierParticipants(barrier) < pcxt->nworkers_launched + 1)
	{
		CHECK_FOR_INTERRUPTS();
		(void)WaitLatch(
				MyLatch,
				WL_LATCH_SET | WL_TIMEOUT | WL_EXIT_ON_PM_DEATH,
				1L,
				WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);
		ResetLatch(MyLatch);
	}

	/* Same contract as the PG back-end: the participant count reflects the
	 * party that attached. The thread pool always starts every planned
	 * worker, so this is the planned count. */
	shared->nparticipants = pcxt->nworkers_launched + 1;
	return true;
}

/*
 * Allocate and populate the parallel build's shared state over a heap arena
 * (the standalone shm_toc): the shared header, barrier, sample/centroid/
 * assignment slots, the tree blob, the per-worker page queues, and the partial
 * pages. The WAL/buffer-usage regions PG carries are replaced by tiny dummies
 * so the leader's instrumentation loop has valid (ignored) pointers. Always
 * succeeds (the arena is plain memory), so it returns true.
 */
bool
prism_pbuild_setup_shared(
		PrismPBuildLeader	   *lead,
		Relation				heap,
		Relation				index,
		const PrismBuildConfig *config,
		int						nworkers)
{
	Dimension  dim			 = config->dim;
	uint32_t   nlist		 = config->nlist;
	int		   nparticipants = nworkers + 1;
	uint64_t   rabitq_seed	 = VS_RABITQ_BUILD_SEED;
	uint32_t   fan_out		 = config->fan_out > 0 ? config->fan_out
												   : prism_auto_fan_out(0, nlist, 0);
	uint32_t   km_k			 = fan_out < nlist ? fan_out : nlist;
	const Size vec_nbytes	 = (Size)dim * sizeof(float);

	uint32_t total_samples = nlist * 256;
	if (total_samples < 10000)
		total_samples = 10000;
	uint32_t max_per_worker = (total_samples + nparticipants - 1) /
							  nparticipants;

	/* The worker entry is resolved by name at launch; register it once. */
	static bool registered = false;
	if (!registered)
	{
		vs_parallel_register_worker(
				"prism_parallel_build_main", prism_parallel_build_main);
		registered = true;
	}

	EnterParallelMode();
	ParallelContext *pcxt = CreateParallelContext(
			VS_MODULE_NAME, "prism_parallel_build_main", nworkers);

	int	 nw_usage	= nworkers > 0 ? nworkers : 1;
	Size usage_sz	= (Size)nw_usage * sizeof(WalUsage);
	Size bufuse_sz	= (Size)nw_usage * sizeof(BufferUsage);
	Size est_shared = BUFFERALIGN(sizeof(PrismBuildSharedStandalone));

	shm_toc_estimate_chunk(&pcxt->estimator, est_shared);
	shm_toc_estimate_chunk(&pcxt->estimator, sizeof(Barrier));
	shm_toc_estimate_chunk(
			&pcxt->estimator,
			prism_dsm_samples_size(nparticipants, max_per_worker, dim));
	shm_toc_estimate_chunk(
			&pcxt->estimator, prism_dsm_centroids_size(km_k, dim));
	shm_toc_estimate_chunk(
			&pcxt->estimator,
			prism_dsm_km_workers_size(nparticipants, km_k, dim));
	shm_toc_estimate_chunk(
			&pcxt->estimator,
			prism_dsm_root_assign_size(nparticipants, max_per_worker));
	shm_toc_estimate_chunk(
			&pcxt->estimator, prism_pbuild_sort_shared_size(nparticipants));
	shm_toc_estimate_chunk(&pcxt->estimator, usage_sz);
	shm_toc_estimate_chunk(&pcxt->estimator, bufuse_sz);
	/* Page-backed routing region (leader fills before the tree-ready barrier):
	 * the global mean. The posting-head base is a scalar in PrismBuildShared.
	 */
	shm_toc_estimate_chunk(&pcxt->estimator, vec_nbytes);
	/* Keyed regions: shared, barrier, samples, centroids, km_workers,
	 * root_assign, sortshared, global_mean (the subtree ring is allocated
	 * by the leader post-assign, outside the toc). */
	shm_toc_estimate_keys(&pcxt->estimator, 8);

	InitializeParallelDSM(pcxt);

	/* ---- Populate shared state ---- */
	PrismBuildSharedStandalone *sh = shm_toc_allocate(pcxt->toc, est_shared);
	PrismBuildShared		   *shared = &sh->base;
	pthread_mutex_init(&sh->mutex, NULL);
	sh->heap  = heap;
	sh->index = index;
	prism_parallel_scan_init(&sh->scan, heap->vectors, heap->nvecs, dim);

	shared->dim				  = dim;
	shared->metric			  = config->metric;
	shared->nlist			  = nlist;
	shared->fan_out			  = fan_out; /* resolved (auto if config 0) */
	shared->subtree_slot_size = 0; /* leader sizes the ring post-assign */
	shared->soar_lambda		  = config->soar_lambda;
	shared->boundary_epsilon  = config->boundary_epsilon;
	shared->fastscan		  = config->fastscan;
	shared->centroid_format	  = config->centroid_format;
	shared->rabitq_seed		  = rabitq_seed;
	shared->nparticipants	  = nparticipants;
	shared->concurrent		  = config->concurrent; /* never set standalone */
	shared->work_mem_kb		  = 0; /* standalone sorter is in-memory */
	shared->max_samples_per_worker = max_per_worker;
	shared->km_max_iterations	   = 20;
	shared->km_tolerance		   = 1e-4f;
	shared->km_k				   = km_k;
	shared->km_converged		   = false;
	shared->refine_threshold	   = 0; /* standalone builds do not refine */
	shared->refine				   = false;
	shared->refine_tile_cap = 0; /* standalone builds are not mem-bounded */
	/* Page-backed routing knobs: route the build scan for accuracy, not
	 * query speed, matching the PG build (see PRISM_BUILD_CENTROID_* in
	 * posting_build.h). */
	shared->centroid_error_scale = PRISM_BUILD_CENTROID_ERROR_SCALE;
	shared->centroid_beam_scale	 = PRISM_BUILD_CENTROID_BEAM_SCALE;
	shared->fastscan_bits		 = 16;
	shared->reltuples			 = 0.0;
	shared->indtuples			 = 0.0;
	shared->soar_dupes			 = 0.0;
	shm_toc_insert(pcxt->toc, PRISM_DSM_KEY_SHARED, shared);

	/* Dynamic barrier (0 parties): the leader and every worker attach as they
	 * start, matching the PG back-end (see do_parallel_build). */
	Barrier *barrier = shm_toc_allocate(pcxt->toc, sizeof(Barrier));
	BarrierInit(barrier, 0);
	shm_toc_insert(pcxt->toc, PRISM_DSM_KEY_BARRIER, barrier);

	Size samp_sz = prism_dsm_samples_size(nparticipants, max_per_worker, dim);
	PrismDsmSamples *dsm_samples = shm_toc_allocate(pcxt->toc, samp_sz);
	memset(dsm_samples, 0, samp_sz);
	dsm_samples->nparticipants	= nparticipants;
	dsm_samples->max_per_worker = max_per_worker;
	dsm_samples->dim			= dim;
	shm_toc_insert(pcxt->toc, PRISM_DSM_KEY_SAMPLES, dsm_samples);

	Size  cent_sz		 = prism_dsm_centroids_size(km_k, dim);
	char *centroids_base = shm_toc_allocate(pcxt->toc, cent_sz);
	memset(centroids_base, 0, cent_sz);
	shm_toc_insert(pcxt->toc, PRISM_DSM_KEY_CENTROIDS, centroids_base);
	float *cents = prism_dsm_centroids(centroids_base);

	Size  km_sz = prism_dsm_km_workers_size(nparticipants, km_k, dim);
	char *km_workers_base = shm_toc_allocate(pcxt->toc, km_sz);
	memset(km_workers_base, 0, km_sz);
	shm_toc_insert(pcxt->toc, PRISM_DSM_KEY_KM_WORKERS, km_workers_base);

	Size ra_sz = prism_dsm_root_assign_size(nparticipants, max_per_worker);
	PrismDsmRootAssign *dsm_ra = shm_toc_allocate(pcxt->toc, ra_sz);
	memset(dsm_ra, 0, ra_sz);
	dsm_ra->nparticipants  = nparticipants;
	dsm_ra->max_per_worker = max_per_worker;
	shm_toc_insert(pcxt->toc, PRISM_DSM_KEY_ROOT_ASSIGN, dsm_ra);

	/* Shared coordinator for the cluster-keyed posting sort (sort seam). */
	Size  sort_sz	 = prism_pbuild_sort_shared_size(nparticipants);
	void *sortshared = shm_toc_allocate(pcxt->toc, sort_sz);
	memset(sortshared, 0, sort_sz);
	shm_toc_insert(pcxt->toc, PRISM_DSM_KEY_SORTSHARED, sortshared);

	/* Page-backed routing region (leader fills before the tree-ready barrier):
	 * the global mean. The posting-head base is a scalar in PrismBuildShared.
	 */
	float *dsm_gmean = shm_toc_allocate(pcxt->toc, vec_nbytes);
	memset(dsm_gmean, 0, vec_nbytes);
	shm_toc_insert(pcxt->toc, PRISM_DSM_KEY_GLOBAL_MEAN, dsm_gmean);

	/* Dummy usage regions so the leader's instrumentation loop is safe. */
	WalUsage	*walusage	 = shm_toc_allocate(pcxt->toc, usage_sz);
	BufferUsage *bufferusage = shm_toc_allocate(pcxt->toc, bufuse_sz);

	lead->pcxt			  = pcxt;
	lead->shared		  = shared;
	lead->barrier		  = barrier;
	lead->dsm_samples	  = dsm_samples;
	lead->sample_seg	  = NULL;
	lead->centroids_base  = centroids_base;
	lead->cents			  = cents;
	lead->km_workers_base = km_workers_base;
	lead->dsm_ra		  = dsm_ra;
	lead->walusage		  = walusage;
	lead->bufferusage	  = bufferusage;
	lead->nparticipants	  = nparticipants;
	lead->km_k			  = km_k;
	lead->max_per_worker  = max_per_worker;
	lead->dim			  = dim;
	lead->nlist			  = nlist;
	lead->rabitq_seed	  = rabitq_seed;
	lead->fan_out		  = fan_out;
	return true;
}

/*
 * Accumulate one worker's tuple counts under the mutex (the analog of PG's
 * spinlock in the derived shared struct).
 */
/*
 * Sample-region seam: standalone keeps the samples in the shared arena for
 * the whole build (it does not bound memory), so attach is a plain lookup
 * and release is a no-op.
 */
PrismDsmSamples *
prism_pbuild_samples_attach(
		shm_toc *toc, PrismBuildShared *shared, void **seg_out)
{
	(void)shared;
	*seg_out = NULL;
	return (PrismDsmSamples *)
			shm_toc_lookup(toc, PRISM_DSM_KEY_SAMPLES, false);
}

void
prism_pbuild_samples_release(PrismDsmSamples *samples, void *seg)
{
	(void)samples;
	(void)seg;
}

/*
 * Subtree-ring seam (thread back-end): one heap allocation the leader makes
 * after root assignment; workers share the pointer. Only the creator gets a
 * non-NULL seg to free at release.
 */
char *
prism_pbuild_subtree_ring_create(
		PrismBuildShared *shared,
		int				  nparticipants,
		uint64_t		  slot_size,
		void			**seg_out)
{
	PrismBuildSharedStandalone *sa = (PrismBuildSharedStandalone *)shared;

	sa->subtree_ring = vs_alloc(
			prism_dsm_child_subtrees_size(nparticipants, slot_size));
	shared->subtree_slot_size = slot_size;
	*seg_out				  = sa->subtree_ring;
	return sa->subtree_ring;
}

char *
prism_pbuild_subtree_ring_attach(PrismBuildShared *shared, void **seg_out)
{
	PrismBuildSharedStandalone *sa = (PrismBuildSharedStandalone *)shared;

	*seg_out = NULL;
	return sa->subtree_ring;
}

void
prism_pbuild_subtree_ring_release(void *seg)
{
	if (seg != NULL)
		vs_free(seg);
}

/*
 * Exact-centroid seam (thread back-end): one heap allocation the leader makes
 * after the streaming tree write; workers share the pointer. Only the
 * creator gets a non-NULL seg to free at release, so the collection outlives
 * every worker's routing (the leader releases last, after the merge).
 */
char *
prism_pbuild_exact_centroids_create(
		PrismBuildShared *shared, uint64_t nbytes, void **seg_out)
{
	PrismBuildSharedStandalone *sa = (PrismBuildSharedStandalone *)shared;

	sa->exact_centroids = vs_alloc(nbytes);
	*seg_out			= sa->exact_centroids;
	return sa->exact_centroids;
}

char *
prism_pbuild_exact_centroids_attach(PrismBuildShared *shared, void **seg_out)
{
	PrismBuildSharedStandalone *sa = (PrismBuildSharedStandalone *)shared;

	*seg_out = NULL;
	if (sa->exact_centroids == NULL)
		vs_error(
				"exact centroid collection attached before the leader "
				"published it");
	return sa->exact_centroids;
}

void
prism_pbuild_exact_centroids_release(void *seg)
{
	if (seg != NULL)
		vs_free(seg);
}

void
prism_pbuild_worker_add_counts(
		PrismBuildShared *shared,
		double			  indtuples,
		double			  soar_dupes,
		double			  heap_tuples)
{
	PrismBuildSharedStandalone *sh = (PrismBuildSharedStandalone *)shared;

	pthread_mutex_lock(&sh->mutex);
	shared->indtuples += indtuples;
	shared->soar_dupes += soar_dupes;
	shared->reltuples += heap_tuples;
	pthread_mutex_unlock(&sh->mutex);
}

/*
 * Re-initialize the scan for the posting pass; the sampling pass consumed the
 * first one. Resets the work-stealing cursor.
 */
void
prism_pbuild_rescan(Relation heap, PrismBuildShared *shared)
{
	PrismBuildSharedStandalone *sh = (PrismBuildSharedStandalone *)shared;

	(void)heap;
	prism_parallel_scan_init(
			&sh->scan, sh->scan.vectors, sh->scan.nvecs, sh->scan.dim);
}

/*
 * Leaf-refinement accumulator lock seam. Standalone never refines
 * (standalone never refines), so these are unused stubs to satisfy the link.
 */
void
prism_pbuild_accum_lock(PrismBuildShared *shared, uint32_t stripe)
{
	(void)shared;
	(void)stripe;
}

void
prism_pbuild_accum_unlock(PrismBuildShared *shared, uint32_t stripe)
{
	(void)shared;
	(void)stripe;
}

/* ----------------------------------------------------------------
 * Posting sort seam (standalone back-end) — in-memory cluster sort
 *
 * Each worker collects fixed-size records ([uint32 cluster][entry]) into a
 * malloc'd buffer and, at performsort, hands ownership to the shared
 * coordinator (one slot per worker). After the phase barrier the leader
 * gathers every worker's records into one buffer, qsorts by cluster, and
 * streams them back via getnext. Standalone is in-memory by design, so this
 * mirrors the PG parallel-tuplesort seam without bounding memory.
 * ---------------------------------------------------------------- */
typedef struct PrismSortShared
{
	int nparticipants;
	/*
	 * Three per-participant arrays follow, in order:
	 *   char	  *bufs[nparticipants];	  worker record buffer (transferred)
	 *   size_t	   counts[nparticipants]; worker record count
	 *   VsMemCtx arenas[nparticipants]; worker arena (ownership transferred)
	 * Each worker allocates its records from its own dedicated arena and, at
	 * performsort, hands the buffer + arena to the leader, which gathers,
	 * sorts, and deletes every worker arena (plus its own) in one go at
	 * sort_end.
	 */
} PrismSortShared;

/*
 * Offset of the per-participant arrays. PrismSortShared is only int-sized, so
 * its size is not a multiple of the pointer alignment; round up so bufs[] (and
 * the size_t/pointer arrays after it) start 8-byte aligned.
 */
#define PRISM_SORTSHARED_HDR (((sizeof(PrismSortShared)) + 7u) & ~(size_t)7u)

static char **
ss_bufs(PrismSortShared *s)
{
	return (char **)((char *)s + PRISM_SORTSHARED_HDR);
}

static size_t *
ss_counts(PrismSortShared *s)
{
	return (size_t *)(ss_bufs(s) + s->nparticipants);
}

static VsMemCtx *
ss_arenas(PrismSortShared *s)
{
	return (VsMemCtx *)(ss_counts(s) + s->nparticipants);
}

struct PrismSorter
{
	bool			 is_leader;
	uint32_t		 entry_size;
	size_t			 stride; /* align4(4 + entry_size) */
	char			*buf;	 /* worker: own records; leader: merged */
	size_t			 n;		 /* record count */
	size_t			 cap;	 /* capacity (records) */
	size_t			 cursor; /* leader getnext position */
	PrismSortShared *sh;
	int				 participant;
	VsMemCtx arena; /* dedicated; holds buf (and merged buf for leader) */
};

static int
prism_sort_cluster_cmp(const void *a, const void *b)
{
	uint32_t ca = *(const uint32_t *)a;
	uint32_t cb = *(const uint32_t *)b;
	return (ca > cb) - (ca < cb);
}

Size
prism_pbuild_sort_shared_size(int nparticipants)
{
	return PRISM_SORTSHARED_HDR +
		   (size_t)nparticipants *
				   (sizeof(char *) + sizeof(size_t) + sizeof(VsMemCtx));
}

void
prism_pbuild_sort_shared_init(void *region, int nparticipants, void *seg)
{
	(void)seg;
	PrismSortShared *s = (PrismSortShared *)region;
	s->nparticipants   = nparticipants;
	memset(ss_bufs(s), 0, (size_t)nparticipants * sizeof(char *));
	memset(ss_counts(s), 0, (size_t)nparticipants * sizeof(size_t));
	memset(ss_arenas(s), 0, (size_t)nparticipants * sizeof(VsMemCtx));
}

PrismSorter *
prism_pbuild_sort_begin(
		void	*region,
		void	*seg,
		int		 participant,
		int		 nparticipants,
		bool	 is_leader,
		uint32_t entry_size,
		int		 work_mem_kb)
{
	(void)seg;
	(void)nparticipants;
	(void)work_mem_kb;
	/* The sorter struct itself stays a plain malloc: it is tiny and freed
	 * deterministically by its own thread at sort_end. Its records go in a
	 * dedicated arena (created here, not the worker's thread-local context,
	 * which is deleted when the worker returns) whose ownership transfers to
	 * the leader at performsort. */
	PrismSorter *s = calloc(1, sizeof(PrismSorter));
	s->is_leader   = is_leader;
	s->entry_size  = entry_size;
	s->stride	   = (sizeof(uint32_t) + entry_size + 3u) & ~(size_t)3u;
	s->sh		   = (PrismSortShared *)region;
	s->participant = participant;
	s->arena	   = vs_memctx_create(NULL, "vs_sort");
	return s;
}

void
prism_pbuild_sort_put(PrismSorter *s, uint32_t cluster, const void *entry)
{
	if (s->n == s->cap)
	{
		/* Grow by doubling. The arena cannot free or grow in place, so the old
		 * buffer is left behind and reclaimed when the arena is deleted — the
		 * cost of arena-backed growth in the in-memory standalone path. */
		size_t newcap = s->cap ? s->cap * 2 : 4096;
		char  *nbuf	  = vs_memctx_alloc(s->arena, newcap * s->stride);
		if (s->buf)
			memcpy(nbuf, s->buf, s->n * s->stride);
		s->buf = nbuf;
		s->cap = newcap;
	}
	char *rec = s->buf + s->n * s->stride;
	memcpy(rec, &cluster, sizeof(uint32_t));
	memcpy(rec + sizeof(uint32_t), entry, s->entry_size);
	s->n++;
}

void
prism_pbuild_sort_performsort(PrismSorter *s)
{
	if (!s->is_leader)
	{
		/* Hand the buffer and its arena to the coordinator; the leader reads
		 * the records and deletes the arena. Drop our references so sort_end
		 * below does not delete the arena out from under the leader. */
		ss_bufs(s->sh)[s->participant]	 = s->buf;
		ss_counts(s->sh)[s->participant] = s->n;
		ss_arenas(s->sh)[s->participant] = s->arena;
		s->buf							 = NULL;
		s->arena						 = NULL;
		return;
	}

	/* Leader: gather every worker's records, then sort by cluster. */
	PrismSortShared *sh	   = s->sh;
	size_t			 total = 0;
	for (int i = 0; i < sh->nparticipants; i++)
		total += ss_counts(sh)[i];
	s->buf	   = total ? vs_memctx_alloc(s->arena, total * s->stride) : NULL;
	size_t off = 0;
	for (int i = 0; i < sh->nparticipants; i++)
	{
		size_t c = ss_counts(sh)[i];
		if (c == 0)
			continue;
		memcpy(s->buf + off * s->stride, ss_bufs(sh)[i], c * s->stride);
		off += c;
	}
	s->n	  = total;
	s->cursor = 0;
	if (total)
		qsort(s->buf, total, s->stride, prism_sort_cluster_cmp);
}

bool
prism_pbuild_sort_getnext(
		PrismSorter *s, uint32_t *cluster, const void **entry)
{
	if (s->cursor >= s->n)
		return false;
	char *rec = s->buf + s->cursor * s->stride;
	memcpy(cluster, rec, sizeof(uint32_t));
	*entry = rec + sizeof(uint32_t);
	s->cursor++;
	return true;
}

void
prism_pbuild_sort_end(PrismSorter *s)
{
	if (s->is_leader)
	{
		/* Delete the worker arenas handed over at performsort (this frees
		 * their record buffers); then our own arena frees the merged buffer
		 * below. */
		PrismSortShared *sh = s->sh;
		for (int i = 0; i < sh->nparticipants; i++)
		{
			if (ss_arenas(sh)[i])
				vs_memctx_delete(ss_arenas(sh)[i]);
			ss_arenas(sh)[i] = NULL;
			ss_bufs(sh)[i]	 = NULL;
		}
	}
	if (s->arena) /* NULL on a worker after performsort transferred ownership
				   */
		vs_memctx_delete(s->arena);
	free(s);
}

#endif /* VS_STANDALONE */
