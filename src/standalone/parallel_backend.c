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
#include <string.h>

#include "algo/hkmeans.h"
#include "core/memory.h"
#include "core/parallel_ctx.h"
#include "index/index_build.h" /* mkt_auto_fan_out */
#include "index/parallel_build.h"
#include "quant/matrix.h" /* mkt_random_orthogonal_matrix */

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
} MktBuildSharedStandalone;

/*
 * Scan every vector, invoking cb per vector. The shared work-stealing cursor
 * lives in the shared state (initialized by setup_shared from the heap's
 * vector array); every participant runs it cooperatively. The PG flags
 * (allow_sync/progress) and the relation/index-info handles are unused here.
 */
void
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

	mkt_parallel_scan_run(&sh->scan, cb, state);
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
mkt_pbuild_launch(ParallelContext *pcxt, Barrier *barrier)
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
	shm_toc_estimate_chunk(&pcxt->estimator, mkt_dsm_rabitq_matrix_size(dim));
	shm_toc_estimate_chunk(
			&pcxt->estimator, mkt_dsm_child_cents_size(km_k, fan_out, dim));
	shm_toc_estimate_chunk(
			&pcxt->estimator,
			mkt_dsm_km_workers_size(nparticipants, km_k, dim));
	shm_toc_estimate_chunk(
			&pcxt->estimator,
			mkt_dsm_root_assign_size(nparticipants, max_per_worker));
	shm_toc_estimate_chunk(&pcxt->estimator, max_tree_sz);
	shm_toc_estimate_chunk(
			&pcxt->estimator, mkt_dsm_posting_queues_size(nparticipants));
	shm_toc_estimate_chunk(
			&pcxt->estimator,
			mkt_dsm_worker_output_size(nlist, nparticipants));
	shm_toc_estimate_chunk(
			&pcxt->estimator, mkt_dsm_partials_size(nlist, nparticipants));
	shm_toc_estimate_chunk(&pcxt->estimator, usage_sz);
	shm_toc_estimate_chunk(&pcxt->estimator, bufuse_sz);
	/* Keyed regions: shared, barrier, samples, centroids, rabitq_matrix,
	 * child_centroids, km_workers, root_assign, tree, posting_queues,
	 * worker_output, partials. */
	shm_toc_estimate_keys(&pcxt->estimator, 12);

	InitializeParallelDSM(pcxt);

	/* ---- Populate shared state ---- */
	MktBuildSharedStandalone *sh	 = shm_toc_allocate(pcxt->toc, est_shared);
	MktBuildShared			 *shared = &sh->base;
	pthread_mutex_init(&sh->mutex, NULL);
	sh->heap  = heap;
	sh->index = index;
	mkt_parallel_scan_init(&sh->scan, heap->vectors, heap->nvecs, dim);

	shared->dim					   = dim;
	shared->metric				   = config->metric;
	shared->nlist				   = nlist;
	shared->fan_out				   = fan_out; /* resolved (auto if config 0) */
	shared->soar_lambda			   = config->soar_lambda;
	shared->boundary_epsilon	   = config->boundary_epsilon;
	shared->fastscan			   = config->fastscan;
	shared->centroid_format		   = config->centroid_format;
	shared->rabitq_seed			   = rabitq_seed;
	shared->nparticipants		   = nparticipants;
	shared->max_samples_per_worker = max_per_worker;
	shared->km_max_iterations	   = 20;
	shared->km_tolerance		   = 1e-4f;
	shared->km_k				   = km_k;
	shared->km_converged		   = false;
	shared->reltuples			   = 0.0;
	shared->indtuples			   = 0.0;
	shared->soar_dupes			   = 0.0;
	shm_toc_insert(pcxt->toc, MKT_DSM_KEY_SHARED, shared);

	Barrier *barrier = shm_toc_allocate(pcxt->toc, sizeof(Barrier));
	BarrierInit(barrier, 1);
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

	/* RaBitQ rotation matrix: generated once here, shared by all workers (they
	 * build their RaBitQParams via create_from_matrix instead of regenerating
	 * the identical orthogonal matrix from the seed). */
	float *rabitq_matrix =
			shm_toc_allocate(pcxt->toc, mkt_dsm_rabitq_matrix_size(dim));
	mkt_random_orthogonal_matrix(rabitq_matrix, dim, rabitq_seed);
	shm_toc_insert(pcxt->toc, MKT_DSM_KEY_RABITQ_MATRIX, rabitq_matrix);

	/* Child k-means output region (phase 2c, work-partitioned). */
	char *child_cents_base = shm_toc_allocate(
			pcxt->toc, mkt_dsm_child_cents_size(km_k, fan_out, dim));
	shm_toc_insert(pcxt->toc, MKT_DSM_KEY_CHILD_CENTROIDS, child_cents_base);

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

	/* Per-worker shm_mq posting-page queues: the leader is the receiver of
	 * every queue; the workers attach as senders in phase 3 and stream pages.
	 */
	Size  queues_sz	  = mkt_dsm_posting_queues_size(nparticipants);
	char *queues_base = shm_toc_allocate(pcxt->toc, queues_sz);
	memset(queues_base, 0, queues_sz);
	for (int i = 0; i < nparticipants; i++)
	{
		shm_mq *mq = shm_mq_create(
				mkt_dsm_posting_queue(queues_base, i),
				mkt_dsm_posting_queue_bytes());
		shm_mq_set_receiver(mq, MyProc);
	}
	shm_toc_insert(pcxt->toc, MKT_DSM_KEY_POSTING_QUEUES, queues_base);

	Size  out_sz		= mkt_dsm_worker_output_size(nlist, nparticipants);
	char *worker_output = shm_toc_allocate(pcxt->toc, out_sz);
	memset(worker_output, 0, out_sz);
	shm_toc_insert(pcxt->toc, MKT_DSM_KEY_WORKER_OUTPUT, worker_output);

	Size  part_sz	   = mkt_dsm_partials_size(nlist, nparticipants);
	char *dsm_partials = shm_toc_allocate(pcxt->toc, part_sz);
	memset(dsm_partials, 0, part_sz);
	shm_toc_insert(pcxt->toc, MKT_DSM_KEY_PARTIALS, dsm_partials);

	/* Dummy usage regions so the leader's instrumentation loop is safe. */
	WalUsage	*walusage	 = shm_toc_allocate(pcxt->toc, usage_sz);
	BufferUsage *bufferusage = shm_toc_allocate(pcxt->toc, bufuse_sz);

	lead->pcxt			   = pcxt;
	lead->shared		   = shared;
	lead->barrier		   = barrier;
	lead->dsm_samples	   = dsm_samples;
	lead->centroids_base   = centroids_base;
	lead->cents			   = cents;
	lead->rabitq_matrix	   = rabitq_matrix;
	lead->child_cents_base = child_cents_base;
	lead->km_workers_base  = km_workers_base;
	lead->dsm_ra		   = dsm_ra;
	lead->dsm_tree		   = dsm_tree;
	lead->queues_base	   = queues_base;
	lead->dsm_partials	   = dsm_partials;
	lead->walusage		   = walusage;
	lead->bufferusage	   = bufferusage;
	lead->nparticipants	   = nparticipants;
	lead->km_k			   = km_k;
	lead->max_per_worker   = max_per_worker;
	lead->dim			   = dim;
	lead->nlist			   = nlist;
	lead->rabitq_seed	   = rabitq_seed;
	lead->fan_out		   = fan_out;
	lead->max_tree_sz	   = max_tree_sz;
	return true;
}

/*
 * Accumulate one worker's tuple counts under the mutex (the analog of PG's
 * spinlock in the derived shared struct).
 */
void
mkt_pbuild_worker_add_counts(
		MktBuildShared *shared, double indtuples, double soar_dupes)
{
	MktBuildSharedStandalone *sh = (MktBuildSharedStandalone *)shared;

	pthread_mutex_lock(&sh->mutex);
	shared->indtuples += indtuples;
	shared->soar_dupes += soar_dupes;
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

#endif /* MKT_STANDALONE */
