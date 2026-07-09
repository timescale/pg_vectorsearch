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
 * of a heap relation. The leader still drains the workers' streamed pages over
 * the shm_mq shim exactly as in PG.
 */

#ifdef MKT_STANDALONE

#include <pthread.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "algo/hkmeans.h"
#include "core/memory.h"
#include "index/index_build.h" /* mkt_auto_fan_out */
#include "index/parallel_build.h"
#include "standalone/parallel_ctx.h"
#include "standalone/parallel_scan.h" /* MktParallelScan */

/*
 * Standalone shared build state: the neutral MktBuildShared plus a mutex
 * guarding its counters, the heap/index relations the workers attach to, and
 * the work-stealing scan cursor (the analog of PG's appended
 * ParallelTableScanDesc). The base is the first member, so the MktBuildShared
 * * the workers look up out of the toc is recovered here as a
 * MktBuildSharedStandalone *.
 */
typedef struct MktBuildSharedStandalone
{
	MktBuildShared	base;
	pthread_mutex_t mutex;
	Relation		heap;
	Relation		index;
	MktParallelScan scan;
	/* Leader's in-memory page store, published for phase-3 page-backed
	 * routing; workers are threads so they share the pointer directly. */
	MktStorage *storage;
} MktBuildSharedStandalone;

/*
 * Scan every vector, invoking cb per vector. The shared work-stealing cursor
 * lives in the shared state (initialized by setup_shared from the heap's
 * vector array); every participant runs it cooperatively. The PG flags
 * (allow_sync/progress) and the relation/index-info handles are unused here.
 */
double
mkt_build_scan(
		Relation		  heap,
		Relation		  index,
		struct IndexInfo *indexInfo,
		MktBuildShared	 *shared,
		bool			  allow_sync,
		bool			  progress,
		MktBuildScanCb	  cb,
		void			 *state)
{
	MktBuildSharedStandalone *sh = (MktBuildSharedStandalone *)shared;

	(void)heap;
	(void)index;
	(void)indexInfo;
	(void)allow_sync;
	(void)progress;

	return mkt_parallel_scan_run(&sh->scan, cb, state);
}

/*
 * Join the parallel build: look up the shared state and barrier, take the
 * heap/index relations the leader stored, and attach to the phase barrier. No
 * relations to open and no instrumentation to start (single process).
 */
void
mkt_pbuild_worker_attach(shm_toc *toc, MktPBuildWorker *w)
{
	MktBuildShared *shared = shm_toc_lookup(toc, MKT_DSM_KEY_SHARED, false);
	MktBuildSharedStandalone *sh = (MktBuildSharedStandalone *)shared;
	Barrier *barrier = shm_toc_lookup(toc, MKT_DSM_KEY_BARRIER, false);

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
mkt_pbuild_worker_detach(shm_toc *toc, MktPBuildWorker *w)
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
mkt_pbuild_publish_storage(MktBuildShared *shared, MktStorage *s)
{
	((MktBuildSharedStandalone *)shared)->storage = s;
}

MktStorage *
mkt_pbuild_worker_storage(MktPBuildWorker *w)
{
	return ((MktBuildSharedStandalone *)w->shared)->storage;
}

void
mkt_pbuild_worker_storage_release(MktStorage *s)
{
	(void)s; /* shared with the leader; not owned by the worker */
}

/*
 * Tear the parallel context down (joins the worker threads and frees the
 * arena) and leave parallel mode.
 */
void
mkt_pbuild_teardown(ParallelContext *pcxt)
{
	DestroyParallelContext(pcxt);
	ExitParallelMode();
}

/*
 * Launch the workers on the thread pool and wait until the whole party has
 * attached to the phase barrier. Returns false (after teardown) if none
 * started. The poll uses a 1ms latch timeout rather than a wakeup, since no
 * one sets the leader's latch during attach.
 */
bool
mkt_pbuild_launch(
		ParallelContext *pcxt, Barrier *barrier, MktBuildShared *shared)
{
	LaunchParallelWorkers(pcxt);

	if (pcxt->nworkers_launched == 0)
	{
		WaitForParallelWorkersToFinish(pcxt);
		mkt_pbuild_teardown(pcxt);
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
mkt_pbuild_setup_shared(
		MktPBuildLeader		 *lead,
		Relation			  heap,
		Relation			  index,
		const MktBuildConfig *config,
		int					  nworkers)
{
	Dimension dim			= config->dim;
	uint32_t  nlist			= config->nlist;
	int		  nparticipants = nworkers + 1;
	uint64_t  rabitq_seed	= 42;
	uint32_t  fan_out		= config->fan_out > 0 ? config->fan_out
												  : mkt_auto_fan_out(0, nlist, 0);
	uint32_t  km_k			= fan_out < nlist ? fan_out : nlist;

	/* Per-child subtree blob slot: each root child's subtree targets
	 * ~nlist/fan_out leaves; size the slot for that tree's worst case. */
	uint32_t nlist_c   = (nlist + fan_out - 1) / fan_out;
	uint64_t slot_size = mkt_hkmeans_max_blob_size(nlist_c, fan_out, dim);

	uint32_t total_samples = nlist * 256;
	if (total_samples < 10000)
		total_samples = 10000;
	uint32_t max_per_worker = (total_samples + nparticipants - 1) /
							  nparticipants;

	/* The worker entry is resolved by name at launch; register it once. */
	static bool registered = false;
	if (!registered)
	{
		mkt_parallel_register_worker(
				"mkt_parallel_build_main", mkt_parallel_build_main);
		registered = true;
	}

	EnterParallelMode();
	ParallelContext *pcxt = CreateParallelContext(
			"meerkat", "mkt_parallel_build_main", nworkers);

	int	 nw_usage	 = nworkers > 0 ? nworkers : 1;
	Size usage_sz	 = (Size)nw_usage * sizeof(WalUsage);
	Size bufuse_sz	 = (Size)nw_usage * sizeof(BufferUsage);
	Size est_shared	 = BUFFERALIGN(sizeof(MktBuildSharedStandalone));
	Size max_tree_sz = sizeof(HKMeansResult) +
					   (Size)nlist * 2 * sizeof(HKMeansNode) +
					   (Size)nlist * dim * sizeof(float) * 2;

	shm_toc_estimate_chunk(&pcxt->estimator, est_shared);
	shm_toc_estimate_chunk(&pcxt->estimator, sizeof(Barrier));
	shm_toc_estimate_chunk(
			&pcxt->estimator,
			mkt_dsm_samples_size(nparticipants, max_per_worker, dim));
	shm_toc_estimate_chunk(
			&pcxt->estimator, mkt_dsm_centroids_size(km_k, dim));
	shm_toc_estimate_chunk(
			&pcxt->estimator,
			mkt_dsm_km_workers_size(nparticipants, km_k, dim));
	shm_toc_estimate_chunk(
			&pcxt->estimator,
			mkt_dsm_root_assign_size(nparticipants, max_per_worker));
	shm_toc_estimate_chunk(&pcxt->estimator, max_tree_sz);
	shm_toc_estimate_chunk(
			&pcxt->estimator, mkt_pbuild_sort_shared_size(nparticipants));
	shm_toc_estimate_chunk(
			&pcxt->estimator, mkt_dsm_child_subtrees_size(fan_out, slot_size));
	shm_toc_estimate_chunk(&pcxt->estimator, usage_sz);
	shm_toc_estimate_chunk(&pcxt->estimator, bufuse_sz);
	/* Page-backed routing regions (leader fills before the tree-ready
	 * barrier). */
	shm_toc_estimate_chunk(
			&pcxt->estimator, (Size)nlist * sizeof(BlockNumber));
	shm_toc_estimate_chunk(&pcxt->estimator, (Size)dim * sizeof(float));
	/* Keyed regions: shared, barrier, samples, centroids, km_workers,
	 * root_assign, tree, sortshared, child_subtrees, posting_heads,
	 * global_mean. */
	shm_toc_estimate_keys(&pcxt->estimator, 11);

	InitializeParallelDSM(pcxt);

	/* ---- Populate shared state ---- */
	MktBuildSharedStandalone *sh	 = shm_toc_allocate(pcxt->toc, est_shared);
	MktBuildShared			 *shared = &sh->base;
	pthread_mutex_init(&sh->mutex, NULL);
	sh->heap  = heap;
	sh->index = index;
	mkt_parallel_scan_init(&sh->scan, heap->vectors, heap->nvecs, dim);

	shared->dim				  = dim;
	shared->metric			  = config->metric;
	shared->nlist			  = nlist;
	shared->fan_out			  = fan_out; /* resolved (auto if config 0) */
	shared->subtree_slot_size = slot_size;
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
	shared->refine_iters = 0; /* standalone builds are not mem-bounded */
	/* Page-backed routing knobs: route the build scan for accuracy, not
	 * query speed, matching the PG build (see MKT_BUILD_CENTROID_* in
	 * posting_build.h). */
	shared->centroid_error_scale = MKT_BUILD_CENTROID_ERROR_SCALE;
	shared->centroid_beam_scale	 = MKT_BUILD_CENTROID_BEAM_SCALE;
	shared->fastscan_bits		 = 16;
	shared->reltuples			 = 0.0;
	shared->indtuples			 = 0.0;
	shared->soar_dupes			 = 0.0;
	shm_toc_insert(pcxt->toc, MKT_DSM_KEY_SHARED, shared);

	/* Dynamic barrier (0 parties): the leader and every worker attach as they
	 * start, matching the PG back-end (see do_parallel_build). */
	Barrier *barrier = shm_toc_allocate(pcxt->toc, sizeof(Barrier));
	BarrierInit(barrier, 0);
	shm_toc_insert(pcxt->toc, MKT_DSM_KEY_BARRIER, barrier);

	Size samp_sz = mkt_dsm_samples_size(nparticipants, max_per_worker, dim);
	MktDsmSamples *dsm_samples = shm_toc_allocate(pcxt->toc, samp_sz);
	memset(dsm_samples, 0, samp_sz);
	dsm_samples->nparticipants	= nparticipants;
	dsm_samples->max_per_worker = max_per_worker;
	dsm_samples->dim			= dim;
	shm_toc_insert(pcxt->toc, MKT_DSM_KEY_SAMPLES, dsm_samples);

	Size  cent_sz		 = mkt_dsm_centroids_size(km_k, dim);
	char *centroids_base = shm_toc_allocate(pcxt->toc, cent_sz);
	memset(centroids_base, 0, cent_sz);
	shm_toc_insert(pcxt->toc, MKT_DSM_KEY_CENTROIDS, centroids_base);
	float *cents = mkt_dsm_centroids(centroids_base);

	Size  km_sz			  = mkt_dsm_km_workers_size(nparticipants, km_k, dim);
	char *km_workers_base = shm_toc_allocate(pcxt->toc, km_sz);
	memset(km_workers_base, 0, km_sz);
	shm_toc_insert(pcxt->toc, MKT_DSM_KEY_KM_WORKERS, km_workers_base);

	Size ra_sz = mkt_dsm_root_assign_size(nparticipants, max_per_worker);
	MktDsmRootAssign *dsm_ra = shm_toc_allocate(pcxt->toc, ra_sz);
	memset(dsm_ra, 0, ra_sz);
	dsm_ra->nparticipants  = nparticipants;
	dsm_ra->max_per_worker = max_per_worker;
	shm_toc_insert(pcxt->toc, MKT_DSM_KEY_ROOT_ASSIGN, dsm_ra);

	void *dsm_tree = shm_toc_allocate(pcxt->toc, max_tree_sz);
	memset(dsm_tree, 0, max_tree_sz);
	shm_toc_insert(pcxt->toc, MKT_DSM_KEY_TREE, dsm_tree);

	/* Shared coordinator for the cluster-keyed posting sort (sort seam). */
	Size  sort_sz	 = mkt_pbuild_sort_shared_size(nparticipants);
	void *sortshared = shm_toc_allocate(pcxt->toc, sort_sz);
	memset(sortshared, 0, sort_sz);
	shm_toc_insert(pcxt->toc, MKT_DSM_KEY_SORTSHARED, sortshared);

	/* Per-child subtree blobs (phase 2c, work-partitioned). */
	char *child_subtrees_base = shm_toc_allocate(
			pcxt->toc, mkt_dsm_child_subtrees_size(fan_out, slot_size));
	shm_toc_insert(pcxt->toc, MKT_DSM_KEY_CHILD_SUBTREES, child_subtrees_base);

	/* Page-backed routing regions (leader fills before the tree-ready
	 * barrier). */
	BlockNumber *dsm_heads =
			shm_toc_allocate(pcxt->toc, (Size)nlist * sizeof(BlockNumber));
	memset(dsm_heads, 0, (Size)nlist * sizeof(BlockNumber));
	shm_toc_insert(pcxt->toc, MKT_DSM_KEY_POSTING_HEADS, dsm_heads);

	float *dsm_gmean = shm_toc_allocate(pcxt->toc, (Size)dim * sizeof(float));
	memset(dsm_gmean, 0, (Size)dim * sizeof(float));
	shm_toc_insert(pcxt->toc, MKT_DSM_KEY_GLOBAL_MEAN, dsm_gmean);

	/* Dummy usage regions so the leader's instrumentation loop is safe. */
	WalUsage	*walusage	 = shm_toc_allocate(pcxt->toc, usage_sz);
	BufferUsage *bufferusage = shm_toc_allocate(pcxt->toc, bufuse_sz);

	lead->pcxt				  = pcxt;
	lead->shared			  = shared;
	lead->barrier			  = barrier;
	lead->dsm_samples		  = dsm_samples;
	lead->centroids_base	  = centroids_base;
	lead->cents				  = cents;
	lead->km_workers_base	  = km_workers_base;
	lead->dsm_ra			  = dsm_ra;
	lead->dsm_tree			  = dsm_tree;
	lead->queues_base		  = NULL; /* sort-seam path: no shm_mq queues */
	lead->dsm_partials		  = NULL; /* sort-seam path: no partials region */
	lead->child_subtrees_base = child_subtrees_base;
	lead->walusage			  = walusage;
	lead->bufferusage		  = bufferusage;
	lead->nparticipants		  = nparticipants;
	lead->km_k				  = km_k;
	lead->max_per_worker	  = max_per_worker;
	lead->dim				  = dim;
	lead->nlist				  = nlist;
	lead->rabitq_seed		  = rabitq_seed;
	lead->fan_out			  = fan_out;
	lead->max_tree_sz		  = max_tree_sz;
	return true;
}

/*
 * Accumulate one worker's tuple counts under the mutex (the analog of PG's
 * spinlock in the derived shared struct).
 */
void
mkt_pbuild_worker_add_counts(
		MktBuildShared *shared,
		double			indtuples,
		double			soar_dupes,
		double			heap_tuples)
{
	MktBuildSharedStandalone *sh = (MktBuildSharedStandalone *)shared;

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
mkt_pbuild_rescan(Relation heap, MktBuildShared *shared)
{
	MktBuildSharedStandalone *sh = (MktBuildSharedStandalone *)shared;

	(void)heap;
	mkt_parallel_scan_init(
			&sh->scan, sh->scan.vectors, sh->scan.nvecs, sh->scan.dim);
}

/*
 * Leaf-refinement accumulator lock seam. Standalone never refines
 * (refine_iters is always 0), so these are unused stubs to satisfy the link.
 */
void
mkt_pbuild_accum_lock(MktBuildShared *shared, uint32_t stripe)
{
	(void)shared;
	(void)stripe;
}

void
mkt_pbuild_accum_unlock(MktBuildShared *shared, uint32_t stripe)
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
typedef struct MktSortShared
{
	int nparticipants;
	/*
	 * Three per-participant arrays follow, in order:
	 *   char	  *bufs[nparticipants];	  worker record buffer (transferred)
	 *   size_t	   counts[nparticipants]; worker record count
	 *   MktMemCtx arenas[nparticipants]; worker arena (ownership transferred)
	 * Each worker allocates its records from its own dedicated arena and, at
	 * performsort, hands the buffer + arena to the leader, which gathers,
	 * sorts, and deletes every worker arena (plus its own) in one go at
	 * sort_end.
	 */
} MktSortShared;

/*
 * Offset of the per-participant arrays. MktSortShared is only int-sized, so
 * its size is not a multiple of the pointer alignment; round up so bufs[] (and
 * the size_t/pointer arrays after it) start 8-byte aligned.
 */
#define MKT_SORTSHARED_HDR (((sizeof(MktSortShared)) + 7u) & ~(size_t)7u)

static char **
ss_bufs(MktSortShared *s)
{
	return (char **)((char *)s + MKT_SORTSHARED_HDR);
}

static size_t *
ss_counts(MktSortShared *s)
{
	return (size_t *)(ss_bufs(s) + s->nparticipants);
}

static MktMemCtx *
ss_arenas(MktSortShared *s)
{
	return (MktMemCtx *)(ss_counts(s) + s->nparticipants);
}

struct MktSorter
{
	bool		   is_leader;
	uint32_t	   entry_size;
	size_t		   stride; /* align4(4 + entry_size) */
	char		  *buf;	   /* worker: own records; leader: merged */
	size_t		   n;	   /* record count */
	size_t		   cap;	   /* capacity (records) */
	size_t		   cursor; /* leader getnext position */
	MktSortShared *sh;
	int			   participant;
	MktMemCtx arena; /* dedicated; holds buf (and merged buf for leader) */
};

static int
mkt_sort_cluster_cmp(const void *a, const void *b)
{
	uint32_t ca = *(const uint32_t *)a;
	uint32_t cb = *(const uint32_t *)b;
	return (ca > cb) - (ca < cb);
}

Size
mkt_pbuild_sort_shared_size(int nparticipants)
{
	return MKT_SORTSHARED_HDR +
		   (size_t)nparticipants *
				   (sizeof(char *) + sizeof(size_t) + sizeof(MktMemCtx));
}

void
mkt_pbuild_sort_shared_init(void *region, int nparticipants, void *seg)
{
	(void)seg;
	MktSortShared *s = (MktSortShared *)region;
	s->nparticipants = nparticipants;
	memset(ss_bufs(s), 0, (size_t)nparticipants * sizeof(char *));
	memset(ss_counts(s), 0, (size_t)nparticipants * sizeof(size_t));
	memset(ss_arenas(s), 0, (size_t)nparticipants * sizeof(MktMemCtx));
}

MktSorter *
mkt_pbuild_sort_begin(
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
	MktSorter *s   = calloc(1, sizeof(MktSorter));
	s->is_leader   = is_leader;
	s->entry_size  = entry_size;
	s->stride	   = (sizeof(uint32_t) + entry_size + 3u) & ~(size_t)3u;
	s->sh		   = (MktSortShared *)region;
	s->participant = participant;
	s->arena	   = mkt_memctx_create(NULL, "mkt_sort");
	return s;
}

void
mkt_pbuild_sort_put(MktSorter *s, uint32_t cluster, const void *entry)
{
	if (s->n == s->cap)
	{
		/* Grow by doubling. The arena cannot free or grow in place, so the old
		 * buffer is left behind and reclaimed when the arena is deleted — the
		 * cost of arena-backed growth in the in-memory standalone path. */
		size_t newcap = s->cap ? s->cap * 2 : 4096;
		char  *nbuf	  = mkt_memctx_alloc(s->arena, newcap * s->stride);
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
mkt_pbuild_sort_performsort(MktSorter *s)
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
	MktSortShared *sh	 = s->sh;
	size_t		   total = 0;
	for (int i = 0; i < sh->nparticipants; i++)
		total += ss_counts(sh)[i];
	s->buf	   = total ? mkt_memctx_alloc(s->arena, total * s->stride) : NULL;
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
		qsort(s->buf, total, s->stride, mkt_sort_cluster_cmp);
}

bool
mkt_pbuild_sort_getnext(MktSorter *s, uint32_t *cluster, const void **entry)
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
mkt_pbuild_sort_end(MktSorter *s)
{
	if (s->is_leader)
	{
		/* Delete the worker arenas handed over at performsort (this frees
		 * their record buffers); then our own arena frees the merged buffer
		 * below. */
		MktSortShared *sh = s->sh;
		for (int i = 0; i < sh->nparticipants; i++)
		{
			if (ss_arenas(sh)[i])
				mkt_memctx_delete(ss_arenas(sh)[i]);
			ss_arenas(sh)[i] = NULL;
			ss_bufs(sh)[i]	 = NULL;
		}
	}
	if (s->arena) /* NULL on a worker after performsort transferred ownership
				   */
		mkt_memctx_delete(s->arena);
	free(s);
}

#endif /* MKT_STANDALONE */
