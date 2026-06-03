/*
 * parallel_build_leader.c - PG parallel index build, leader side
 *
 * The leader sets up the DSM, launches workers, participates in the
 * sampling + k-means phases as worker_id 0, then (phase 3) reserves the
 * posting layout, drains the workers' streamed pages over their shm_mq
 * queues, and finalizes each list (head + trailing-partial merge + chain
 * stitch). do_parallel_build returns the tree + posting heads to
 * mktann_build, which writes the centroid tree and metadata. Returns false
 * if parallelism could not start, so the caller falls back to a serial build.
 */

#include <postgres.h>

#include <access/parallel.h>
#include <access/table.h>
#include <access/tableam.h>
#include <access/xloginsert.h>
#include <catalog/index.h>
#include <commands/progress.h>
#include <common/pg_prng.h>
#include <math.h>
#include <miscadmin.h>
#include <pgstat.h>
#include <tcop/tcopprot.h>
#include <utils/backend_progress.h>
#include <utils/memutils.h>
#include <utils/rel.h>
#include <utils/sampling.h>

#include "algo/distance.h"
#include "algo/hkmeans.h"
#include "algo/kmeans.h"
#include "algo/kmeans_internal.h"
#include "algo/vecops.h"
#include "index/centroid_build.h"
#include "index/centroid_page.h"
#include "index/index_build.h"
#include "index/posting_build.h"
#include "index/posting_build_parallel.h"
#include "index/posting_page.h"
#include "mkt_halfvec.h"
#include "mkt_pg.h"
#include "mkt_vector.h"
#include "mktann_build.h"
#include "mktann_meta.h"
#include "mktann_storage.h"
#include "parallel_build.h"
#include "quant/fastscan.h"

/*
 * Launch the worker participants and wait until they have all attached to the
 * barrier (so the dynamic party reaches launched+1 before the leader advances
 * the first phase). Returns false — after tearing the context down — if no
 * workers started, so the caller falls back to a serial build. This is a
 * coarse back-end seam: a standalone build provides a same-named function that
 * spawns threads and joins them at the barrier instead.
 */
static bool
mkt_pbuild_launch(ParallelContext *pcxt, Barrier *barrier)
{
	LaunchParallelWorkers(pcxt);

	if (pcxt->nworkers_launched == 0)
	{
		WaitForParallelWorkersToFinish(pcxt);
		DestroyParallelContext(pcxt);
		ExitParallelMode();
		return false;
	}

	/*
	 * Workers attach to the barrier dynamically, so the party (1 leader + N
	 * launched) is not final until they all have; if the leader arrived first
	 * it could advance the phase alone and strand late workers.
	 * WaitForParallelWorkersToAttach surfaces a startup failure as an error
	 * rather than a hang; then poll until the live participant count is whole.
	 */
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
 * Leader-side runtime state for the parallel build: the parallel context, the
 * shared DSM regions, and the derived sizes. Produced by
 * mkt_pbuild_setup_shared and consumed by the rest of the driver. (A coarse
 * back-end seam: the standalone build will later populate the same struct from
 * a heap arena instead of DSM.)
 */
typedef struct MktPBuildLeader
{
	ParallelContext	 *pcxt;
	MktBuildShared	 *shared;
	Barrier			 *barrier;
	MktDsmSamples	 *dsm_samples;
	char			 *centroids_base;
	float			 *cents;
	char			 *km_workers_base;
	MktDsmRootAssign *dsm_ra;
	void			 *dsm_tree;
	char			 *queues_base;
	char			 *dsm_partials;
	WalUsage		 *walusage;
	BufferUsage		 *bufferusage;
	int				  nparticipants;
	uint32_t		  km_k;
	uint32_t		  max_per_worker;
	Dimension		  dim;
	uint32_t		  nlist;
	uint64_t		  rabitq_seed;
	uint32_t		  fan_out;
	Size			  max_tree_sz;
} MktPBuildLeader;

/*
 * Allocate and populate the parallel build's shared state: the DSM segment and
 * its regions (shared header, barrier, sample/centroid/assignment slots, the
 * tree blob, the per-worker page queues, usage counters). Does not launch
 * workers; the caller does. Coarse PG block — the standalone back-end provides
 * a same-named function over a heap arena. Returns false (after tearing the
 * parallel context down) if the DSM segment could not be created.
 */
static bool
mkt_pbuild_setup_shared(
		MktPBuildLeader			*lead,
		Relation				 heap,
		Relation				 index,
		const MktannBuildParams *params,
		int						 nworkers)
{
	Dimension dim			= params->dim;
	uint32_t  nlist			= params->nlist;
	int		  nparticipants = nworkers + 1;
	uint64_t  rabitq_seed	= 42;
	uint32_t  fan_out		= params->fan_out > 0 ? params->fan_out
												  : mkt_auto_fan_out(0, nlist, 0);
	uint32_t  km_k			= fan_out < nlist ? fan_out : nlist;

	/* Compute sample budget per worker */
	uint32_t total_samples	= Max(10000, (int)(nlist * 256));
	uint32_t max_per_worker = (total_samples + nparticipants - 1) /
							  nparticipants;

	EnterParallelMode();

	ParallelContext *pcxt = CreateParallelContext(
			"meerkat", "mktann_parallel_build_main", nworkers);

	/* Estimate DSM size for ALL phases */
	Snapshot snapshot	= SnapshotAny;
	Size	 est_shared = add_size(
			BUFFERALIGN(sizeof(MktBuildShared)),
			table_parallelscan_estimate(heap, snapshot));

	shm_toc_estimate_chunk(&pcxt->estimator, est_shared);
	shm_toc_estimate_chunk(&pcxt->estimator, sizeof(Barrier));
	/* Sampling */
	shm_toc_estimate_chunk(
			&pcxt->estimator,
			mktann_samples_size(nparticipants, max_per_worker, dim));
	/* K-means shared centroids + norms (root level, k=km_k) */
	shm_toc_estimate_chunk(&pcxt->estimator, mktann_centroids_size(km_k, dim));
	/* K-means per-worker accumulators */
	shm_toc_estimate_chunk(
			&pcxt->estimator,
			mktann_km_workers_size(nparticipants, km_k, dim));
	/* Root assignments: per-worker uint32_t[max_per_worker] */
	shm_toc_estimate_chunk(
			&pcxt->estimator,
			mktann_root_assign_size(nparticipants, max_per_worker));
	/* Tree blob (placeholder — allocated later by leader, but
	 * we need the max possible size. Use a generous estimate.) */
	Size max_tree_sz = sizeof(HKMeansResult) +
					   (Size)nlist * 2 * sizeof(HKMeansNode) +
					   (Size)nlist * dim * sizeof(float) * 2;
	shm_toc_estimate_chunk(&pcxt->estimator, max_tree_sz);
	/* Bounded streaming posting phase: per-worker shm_mq queues carry full
	 * pages from the workers to the leader, which writes them. */
	shm_toc_estimate_chunk(
			&pcxt->estimator, mktann_posting_queues_size(nparticipants));
	/* Worker output (active flags) */
	shm_toc_estimate_chunk(
			&pcxt->estimator, mktann_worker_output_size(nlist, nparticipants));
	/* Per-worker trailing partial pages (both formats): the leader folds
	 * them into each list's head during finalize. */
	shm_toc_estimate_chunk(
			&pcxt->estimator, mktann_partials_size(nlist, nparticipants));

	shm_toc_estimate_chunk(
			&pcxt->estimator, mul_size(sizeof(WalUsage), pcxt->nworkers));
	shm_toc_estimate_chunk(
			&pcxt->estimator, mul_size(sizeof(BufferUsage), pcxt->nworkers));

	int querylen = 0;
	if (debug_query_string)
	{
		querylen = strlen(debug_query_string);
		shm_toc_estimate_chunk(&pcxt->estimator, querylen + 1);
	}

	/* nkeys: shared, barrier, samples, centroids, km_workers, root_assign,
	 * tree, posting_queues, worker_output, wal, buffer, partials
	 * + optionally query_text */
	int nkeys = 12;
	if (debug_query_string)
		nkeys++;
	shm_toc_estimate_keys(&pcxt->estimator, nkeys);

	InitializeParallelDSM(pcxt);

	if (pcxt->seg == NULL)
	{
		DestroyParallelContext(pcxt);
		ExitParallelMode();
		return false;
	}

	/* ---- Populate shared state ---- */
	MktBuildShared *shared		   = shm_toc_allocate(pcxt->toc, est_shared);
	shared->heaprelid			   = RelationGetRelid(heap);
	shared->indexrelid			   = RelationGetRelid(index);
	shared->queryid				   = pgstat_get_my_query_id();
	shared->dim					   = dim;
	shared->metric				   = params->metric;
	shared->nlist				   = nlist;
	shared->fan_out				   = params->fan_out;
	shared->soar_lambda			   = params->soar_lambda;
	shared->boundary_epsilon	   = params->boundary_epsilon;
	shared->fastscan			   = params->fastscan;
	shared->rabitq_seed			   = rabitq_seed;
	shared->nparticipants		   = nparticipants;
	shared->max_samples_per_worker = max_per_worker;
	shared->km_max_iterations	   = 20;
	shared->km_tolerance		   = 1e-4f;
	shared->km_k				   = km_k;
	shared->km_converged		   = false;
	SpinLockInit(&shared->mutex);
	ConditionVariableInit(&shared->workersdonecv);
	shared->nparticipantsdone = 0;
	shared->reltuples		  = 0.0;
	shared->indtuples		  = 0.0;
	shared->soar_dupes		  = 0.0;
	table_parallelscan_initialize(
			heap, ParallelTableScanFromMktShared(shared), snapshot);
	shm_toc_insert(pcxt->toc, MKTANN_KEY_SHARED, shared);

	/*
	 * Barrier for phase synchronization. Init with party 1 (the leader,
	 * the only statically-known participant). Each launched worker attaches
	 * dynamically (BarrierAttach) on startup, so the party always tracks the
	 * number of participants that actually showed up — PostgreSQL may launch
	 * fewer workers than planned, and a fixed planned-count party would
	 * deadlock waiting on workers that never started.
	 */
	Barrier *barrier = shm_toc_allocate(pcxt->toc, sizeof(Barrier));
	BarrierInit(barrier, 1);
	shm_toc_insert(pcxt->toc, MKTANN_KEY_BARRIER, barrier);

	/* Sample slots */
	Size samp_sz = mktann_samples_size(nparticipants, max_per_worker, dim);
	MktDsmSamples *dsm_samples = shm_toc_allocate(pcxt->toc, samp_sz);
	memset(dsm_samples, 0, samp_sz);
	dsm_samples->nparticipants	= nparticipants;
	dsm_samples->max_per_worker = max_per_worker;
	dsm_samples->dim			= dim;
	shm_toc_insert(pcxt->toc, MKTANN_KEY_SAMPLES, dsm_samples);

	/* Shared centroids + norms (root k-means, k=km_k) */
	Size  cent_sz		 = mktann_centroids_size(km_k, dim);
	char *centroids_base = shm_toc_allocate(pcxt->toc, cent_sz);
	memset(centroids_base, 0, cent_sz);
	shm_toc_insert(pcxt->toc, MKTANN_KEY_CENTROIDS, centroids_base);
	float *cents = mktann_centroids(centroids_base);

	/* Per-worker k-means accumulators */
	Size  km_sz			  = mktann_km_workers_size(nparticipants, km_k, dim);
	char *km_workers_base = shm_toc_allocate(pcxt->toc, km_sz);
	memset(km_workers_base, 0, km_sz);
	shm_toc_insert(pcxt->toc, MKTANN_KEY_KM_WORKERS, km_workers_base);

	/* Root assignment slots */
	Size ra_sz = mktann_root_assign_size(nparticipants, max_per_worker);
	MktDsmRootAssign *dsm_ra = shm_toc_allocate(pcxt->toc, ra_sz);
	memset(dsm_ra, 0, ra_sz);
	dsm_ra->nparticipants  = nparticipants;
	dsm_ra->max_per_worker = max_per_worker;
	shm_toc_insert(pcxt->toc, MKTANN_KEY_ROOT_ASSIGN, dsm_ra);

	/* Tree blob — allocated now, populated after k-means */
	void *dsm_tree = shm_toc_allocate(pcxt->toc, max_tree_sz);
	memset(dsm_tree, 0, max_tree_sz);
	shm_toc_insert(pcxt->toc, MKTANN_KEY_TREE, dsm_tree);

	/* Per-worker shm_mq posting-page queues. The leader is the receiver of
	 * every queue; the launched workers attach as senders in phase 3 and
	 * stream their full pages. Create and register the receiver here,
	 * before launch. */
	Size  queues_sz	  = mktann_posting_queues_size(nparticipants);
	char *queues_base = shm_toc_allocate(pcxt->toc, queues_sz);
	memset(queues_base, 0, queues_sz);
	for (int i = 0; i < nparticipants; i++)
	{
		shm_mq *mq = shm_mq_create(
				mktann_posting_queue(queues_base, i),
				mktann_posting_queue_bytes());
		shm_mq_set_receiver(mq, MyProc);
	}
	shm_toc_insert(pcxt->toc, MKTANN_KEY_POSTING_QUEUES, queues_base);

	/* Worker output (active flags) */
	Size  out_sz		= mktann_worker_output_size(nlist, nparticipants);
	char *worker_output = shm_toc_allocate(pcxt->toc, out_sz);
	memset(worker_output, 0, out_sz);
	shm_toc_insert(pcxt->toc, MKTANN_KEY_WORKER_OUTPUT, worker_output);

	/* Per-worker trailing partial pages (both AoS and fastscan). Each worker
	 * holds at most one partial page per cluster here; the leader folds them
	 * into the list's head during finalize. */
	Size  part_sz	   = mktann_partials_size(nlist, nparticipants);
	char *dsm_partials = shm_toc_allocate(pcxt->toc, part_sz);
	memset(dsm_partials, 0, part_sz);
	shm_toc_insert(pcxt->toc, MKTANN_KEY_PARTIALS, dsm_partials);

	WalUsage *walusage = shm_toc_allocate(
			pcxt->toc, mul_size(sizeof(WalUsage), pcxt->nworkers));
	memset(walusage, 0, mul_size(sizeof(WalUsage), pcxt->nworkers));
	shm_toc_insert(pcxt->toc, MKTANN_KEY_WAL_USAGE, walusage);

	BufferUsage *bufferusage = shm_toc_allocate(
			pcxt->toc, mul_size(sizeof(BufferUsage), pcxt->nworkers));
	memset(bufferusage, 0, mul_size(sizeof(BufferUsage), pcxt->nworkers));
	shm_toc_insert(pcxt->toc, MKTANN_KEY_BUFFER_USAGE, bufferusage);

	if (debug_query_string)
	{
		char *sq = shm_toc_allocate(pcxt->toc, querylen + 1);
		memcpy(sq, debug_query_string, querylen + 1);
		shm_toc_insert(pcxt->toc, MKTANN_KEY_QUERY_TEXT, sq);
	}

	lead->pcxt			  = pcxt;
	lead->shared		  = shared;
	lead->barrier		  = barrier;
	lead->dsm_samples	  = dsm_samples;
	lead->centroids_base  = centroids_base;
	lead->cents			  = cents;
	lead->km_workers_base = km_workers_base;
	lead->dsm_ra		  = dsm_ra;
	lead->dsm_tree		  = dsm_tree;
	lead->queues_base	  = queues_base;
	lead->dsm_partials	  = dsm_partials;
	lead->walusage		  = walusage;
	lead->bufferusage	  = bufferusage;
	lead->nparticipants	  = nparticipants;
	lead->km_k			  = km_k;
	lead->max_per_worker  = max_per_worker;
	lead->dim			  = dim;
	lead->nlist			  = nlist;
	lead->rabitq_seed	  = rabitq_seed;
	lead->fan_out		  = fan_out;
	lead->max_tree_sz	  = max_tree_sz;
	return true;
}

bool
do_parallel_build(
		Relation				 heap,
		Relation				 index,
		struct IndexInfo		*index_info,
		const MktannBuildParams *params,
		MktannStorage			*storage,
		HKMeansResult		   **out_tree,
		BlockNumber				*posting_heads,
		double					*out_heap_tuples,
		double					*out_indtuples,
		double					*out_soar_dupes)
{
	int nworkers = index_info->ii_ParallelWorkers;

	MktPBuildLeader lead;
	if (!mkt_pbuild_setup_shared(&lead, heap, index, params, nworkers))
		return false;

	ParallelContext	 *pcxt			  = lead.pcxt;
	MktBuildShared	 *shared		  = lead.shared;
	Barrier			 *barrier		  = lead.barrier;
	MktDsmSamples	 *dsm_samples	  = lead.dsm_samples;
	char			 *centroids_base  = lead.centroids_base;
	float			 *cents			  = lead.cents;
	char			 *km_workers_base = lead.km_workers_base;
	MktDsmRootAssign *dsm_ra		  = lead.dsm_ra;
	void			 *dsm_tree		  = lead.dsm_tree;
	char			 *queues_base	  = lead.queues_base;
	char			 *dsm_partials	  = lead.dsm_partials;
	WalUsage		 *walusage		  = lead.walusage;
	BufferUsage		 *bufferusage	  = lead.bufferusage;
	int				  nparticipants	  = lead.nparticipants;
	uint32_t		  km_k			  = lead.km_k;
	uint32_t		  max_per_worker  = lead.max_per_worker;
	Dimension		  dim			  = lead.dim;
	uint32_t		  nlist			  = lead.nlist;
	uint64_t		  rabitq_seed	  = lead.rabitq_seed;
	uint32_t		  fan_out		  = lead.fan_out;
	Size			  max_tree_sz	  = lead.max_tree_sz;
	Size			  km_sz = mktann_km_workers_size(nparticipants, km_k, dim);

	instr_time t_launch_start;
	INSTR_TIME_SET_CURRENT(t_launch_start);

	/* ---- Launch workers + wait until they've all attached ---- */
	if (!mkt_pbuild_launch(pcxt, barrier))
		return false;
	/* ==== Leader participates in all phases as worker_id=0 ==== */

	/* ---- Phase 1: Leader samples ---- */
	{
		double est_rows = RelationGetNumberOfBlocks(heap) *
						  (BLCKSZ / (double)(dim * sizeof(float) + 32));
		uint32_t stride = 1;
		if (est_rows / nparticipants > max_per_worker)
			stride = (uint32_t)(est_rows / nparticipants / max_per_worker);
		if (stride < 1)
			stride = 1;

		SampleCbState sc = {
				.samples		= mktann_worker_samples(dsm_samples, 0),
				.count			= 0,
				.max_samples	= max_per_worker,
				.stride			= stride,
				.stride_counter = 0,
				.dim			= dim,
				.metric			= shared->metric,
		};

		TableScanDesc scan = table_beginscan_parallel(
				heap, ParallelTableScanFromMktShared(shared));

		table_index_build_scan(
				heap,
				index,
				index_info,
				true,
				true,
				mktann_sample_callback,
				&sc,
				scan);

		mktann_sample_counts(dsm_samples)[0] = sc.count;
	}

	instr_time t_sample_end;
	INSTR_TIME_SET_CURRENT(t_sample_end);
	INSTR_TIME_SUBTRACT(t_sample_end, t_launch_start);
	elog(LOG,
		 "mktann: phase 1 (sampling) %.1fms",
		 INSTR_TIME_GET_MILLISEC(t_sample_end));

	/* Barrier: all participants done sampling */
	BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

	instr_time t_km_start;
	INSTR_TIME_SET_CURRENT(t_km_start);

	/* ---- Phase 2: Parallel root k-means (k=km_k) ---- */

	/*
	 * Leader-only init: seed all km_k initial centroids from the pooled
	 * samples, spread evenly across the concatenation of every participant's
	 * slot. Centralizing the pick (rather than slicing it by worker index)
	 * makes initialization independent of how many workers launched — a
	 * partial launch can no longer leave centroid slots unseeded — and draws
	 * from every participant's samples, so it's robust to any one of them
	 * being sample-starved. Slots for workers that never launched hold count 0
	 * and are skipped naturally. The k-means iterations below stay parallel;
	 * this seed pick is a few-microsecond copy of km_k vectors.
	 *
	 * Note: the slot contents and their order come from the work-stealing heap
	 * scan, so these seeds (and thus the resulting tree) vary run to run — the
	 * parallel build is not bit-reproducible. See
	 * docs/parallel-build-design.md
	 * ("Determinism").
	 */
	float *norms_c = mktann_norms_c(centroids_base, km_k, dim);
	{
		uint32_t total_ns = 0;
		for (int t = 0; t < nparticipants; t++)
			total_ns += mktann_sample_counts(dsm_samples)[t];

		uint32_t step = (total_ns >= km_k) ? total_ns / km_k : 1;
		for (uint32_t i = 0; i < km_k; i++)
		{
			uint32_t gidx = (total_ns > 0) ? (i * step) % total_ns : 0;

			/* Map the global sample index to its participant slot. */
			int		 t	  = 0;
			uint32_t base = 0;
			while (t < nparticipants &&
				   base + mktann_sample_counts(dsm_samples)[t] <= gidx)
			{
				base += mktann_sample_counts(dsm_samples)[t];
				t++;
			}
			if (t < nparticipants)
				memcpy(cents + (size_t)i * dim,
					   mktann_worker_samples(dsm_samples, t) +
							   (size_t)(gidx - base) * dim,
					   dim * sizeof(float));
		}

		if (shared->metric == DISTANCE_L2)
			for (uint32_t j = 0; j < km_k; j++)
				norms_c[j] = mkt_l2_norm_squared(cents + (size_t)j * dim, dim);
	}

	/*
	 * Barrier: initial centroids + norms are ready. Pairs with the workers'
	 * post-sampling barrier and releases them into the iteration loop without
	 * racing the seed pick above.
	 */
	BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

	float	 *my_sums = mktann_km_worker_sums(km_workers_base, km_k, dim, 0);
	uint32_t *my_cnts = mktann_km_worker_cnts(km_workers_base, km_k, dim, 0);
	float	 *my_cost = mktann_km_worker_cost(km_workers_base, km_k, dim, 0);

	float	*leader_samples	 = mktann_worker_samples(dsm_samples, 0);
	uint32_t leader_nsamples = mktann_sample_counts(dsm_samples)[0];
	float	*old_cents		 = palloc((size_t)km_k * dim * sizeof(float));

	uint32_t km_iters = 0;
	for (uint32_t iter = 0; iter < shared->km_max_iterations; iter++)
	{
		km_iters++;
		mktann_km_assign_and_accumulate(
				leader_samples,
				leader_nsamples,
				cents,
				norms_c,
				km_k,
				dim,
				shared->metric,
				my_sums,
				my_cnts,
				my_cost);

		/* Barrier: all workers done with assignment */
		BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

		memcpy(old_cents, cents, (size_t)km_k * dim * sizeof(float));

		/* Leader reduce */
		const float	   **all_sums = palloc(nparticipants * sizeof(float *));
		const uint32_t **all_cnts = palloc(nparticipants * sizeof(uint32_t *));
		float			*all_costs = palloc(nparticipants * sizeof(float));

		for (int t = 0; t < nparticipants; t++)
		{
			all_sums[t] = mktann_km_worker_sums(km_workers_base, km_k, dim, t);
			all_cnts[t] = mktann_km_worker_cnts(km_workers_base, km_k, dim, t);
			all_costs[t] =
					*mktann_km_worker_cost(km_workers_base, km_k, dim, t);
		}

		float total_cost;
		float shift_sq = kmeans_merge_centroids(
				cents,
				norms_c,
				old_cents,
				all_sums,
				all_cnts,
				all_costs,
				nparticipants,
				km_k,
				dim,
				shared->metric,
				&total_cost);

		float tol_sq		 = shared->km_tolerance * shared->km_tolerance;
		shared->km_converged = (shift_sq < tol_sq);

		memset(km_workers_base, 0, km_sz);

		/* Barrier: workers read updated centroids */
		BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

		pfree(all_sums);
		pfree(all_cnts);
		pfree(all_costs);

		if (shared->km_converged)
			break;
	}

	pfree(old_cents);

	{
		instr_time t_km_elapsed;
		INSTR_TIME_SET_CURRENT(t_km_elapsed);
		INSTR_TIME_SUBTRACT(t_km_elapsed, t_km_start);
		elog(LOG,
			 "mktann: root kmeans %.1fms (%u iters, k=%u)",
			 INSTR_TIME_GET_MILLISEC(t_km_elapsed),
			 km_iters,
			 km_k);
	}

	/*
	 * Compute nlevels to decide parallel vs serial child k-means.
	 * For nlevels == 1, root k-means is the only level. For
	 * nlevels == 2, we do parallel child k-means. For nlevels > 2,
	 * fall back to serial hkmeans.
	 */
	uint32_t nlevels = 1;
	{
		uint32_t n = nlist;
		while (n > fan_out)
		{
			n = (n + fan_out - 1) / fan_out;
			nlevels++;
		}
	}

	HKMeansResult *tree = NULL;

	instr_time t_child_start;
	INSTR_TIME_SET_CURRENT(t_child_start);

	if (nlevels == 2)
	{
		/*
		 * Phase 2b: Root assignment — leader + workers
		 *
		 * Each participant assigns its samples to root centroids.
		 */
		uint32_t *leader_ra = mktann_root_assignments(dsm_ra, 0);
		uint32_t  leader_ns = mktann_sample_counts(dsm_samples)[0];

		kmeans_assign(
				leader_samples,
				0,
				leader_ns,
				cents,
				norms_c,
				km_k,
				dim,
				shared->metric,
				leader_ra);

		/* Barrier: all done with root assignment */
		BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

		/*
		 * Phase 2c: Parallel child k-means
		 *
		 * For each child c, run k-means on samples assigned to c.
		 * The leader picks initial centroids, writes them to the
		 * DSM centroid buffer, then all participants iterate.
		 */

		/* Concatenate all workers' samples + assignments for
		 * leader to pick initial child centroids from */
		uint32_t total_nsamples = 0;
		for (int t = 0; t < nparticipants; t++)
			total_nsamples += mktann_sample_counts(dsm_samples)[t];

		/* nlist * 256 samples * dim can exceed the 1GB palloc limit for
		 * large nlist / high dim (e.g. nlist=2000, dim=768 ~ 1.5GB), so
		 * allow a huge allocation. Freed after child k-means. */
		float *all_samples = palloc_extended(
				(size_t)total_nsamples * dim * sizeof(float), MCXT_ALLOC_HUGE);
		uint32_t *all_root_asgn = palloc(total_nsamples * sizeof(uint32_t));
		uint32_t  soff			= 0;
		for (int t = 0; t < nparticipants; t++)
		{
			uint32_t n = mktann_sample_counts(dsm_samples)[t];
			memcpy(all_samples + (size_t)soff * dim,
				   mktann_worker_samples(dsm_samples, t),
				   (size_t)n * dim * sizeof(float));
			memcpy(all_root_asgn + soff,
				   mktann_root_assignments(dsm_ra, t),
				   n * sizeof(uint32_t));
			soff += n;
		}

		/* Save root centroids — cents buffer will be reused */
		float *root_cents = palloc((size_t)km_k * dim * sizeof(float));
		memcpy(root_cents, cents, (size_t)km_k * dim * sizeof(float));

		/* Per-child result storage */
		float	**child_centroids = palloc(km_k * sizeof(float *));
		uint32_t *child_ks		  = palloc(km_k * sizeof(uint32_t));

		for (uint32_t child = 0; child < km_k; child++)
		{
			/* Count samples for this child */
			uint32_t child_count = 0;
			for (uint32_t i = 0; i < total_nsamples; i++)
			{
				if (all_root_asgn[i] == child)
					child_count++;
			}

			uint32_t child_k = fan_out < child_count ? fan_out : child_count;
			if (child_k < 1)
				child_k = 1;

			shared->child_km_k = child_k;
			child_ks[child]	   = child_k;

			if (child_k <= 1)
			{
				/* Trivial: single centroid = root centroid */
				child_centroids[child] = palloc(dim * sizeof(float));
				memcpy(child_centroids[child],
					   root_cents + (size_t)child * dim,
					   dim * sizeof(float));

				shared->km_converged = true;

				/* Barrier: init done */
				BarrierArriveAndWait(
						barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);
				continue;
			}

			/* Pick initial centroids: first child_k samples
			 * assigned to this child */
			uint32_t picked = 0;
			for (uint32_t i = 0; i < total_nsamples && picked < child_k; i++)
			{
				if (all_root_asgn[i] != child)
					continue;
				memcpy(cents + (size_t)picked * dim,
					   all_samples + (size_t)i * dim,
					   dim * sizeof(float));
				picked++;
			}

			/* Compute norms for child centroids */
			float *child_norms = mktann_norms_c(centroids_base, child_k, dim);
			if (shared->metric == DISTANCE_L2)
				for (uint32_t j = 0; j < child_k; j++)
					child_norms[j] =
							mkt_l2_norm_squared(cents + (size_t)j * dim, dim);

			shared->km_converged = false;

			/* Clear accumulators */
			Size child_km_sz =
					mktann_km_workers_size(nparticipants, child_k, dim);
			memset(km_workers_base, 0, child_km_sz);

			float *child_old_cents = palloc(
					(size_t)child_k * dim * sizeof(float));

			/* Barrier: child centroids written, workers start */
			BarrierArriveAndWait(
					barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

			/* Iterate child k-means */
			for (uint32_t iter = 0; iter < shared->km_max_iterations; iter++)
			{
				/* Leader's assignment + accumulation */
				float *l_sums = mktann_km_worker_sums(
						km_workers_base, child_k, dim, 0);
				uint32_t *l_cnts = mktann_km_worker_cnts(
						km_workers_base, child_k, dim, 0);
				float *l_cost = mktann_km_worker_cost(
						km_workers_base, child_k, dim, 0);

				mktann_km_assign_and_accumulate_filtered(
						leader_samples,
						leader_ns,
						leader_ra,
						child,
						cents,
						child_norms,
						child_k,
						dim,
						shared->metric,
						l_sums,
						l_cnts,
						l_cost);

				/* Barrier: all workers done */
				BarrierArriveAndWait(
						barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

				/* Leader reduce */
				memcpy(child_old_cents,
					   cents,
					   (size_t)child_k * dim * sizeof(float));

				const float **csums = palloc(nparticipants * sizeof(float *));
				const uint32_t **ccnts = palloc(
						nparticipants * sizeof(uint32_t *));
				float *ccosts = palloc(nparticipants * sizeof(float));

				for (int t = 0; t < nparticipants; t++)
				{
					csums[t] = mktann_km_worker_sums(
							km_workers_base, child_k, dim, t);
					ccnts[t] = mktann_km_worker_cnts(
							km_workers_base, child_k, dim, t);
					ccosts[t] = *mktann_km_worker_cost(
							km_workers_base, child_k, dim, t);
				}

				float total_cost;
				float shift_sq = kmeans_merge_centroids(
						cents,
						child_norms,
						child_old_cents,
						csums,
						ccnts,
						ccosts,
						nparticipants,
						child_k,
						dim,
						shared->metric,
						&total_cost);

				float tol_sq = shared->km_tolerance * shared->km_tolerance;
				shared->km_converged = (shift_sq < tol_sq);

				/* Clear accumulators for next iter */
				memset(km_workers_base, 0, child_km_sz);

				/* Barrier: workers read updated centroids */
				BarrierArriveAndWait(
						barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

				pfree(csums);
				pfree(ccnts);
				pfree(ccosts);

				if (shared->km_converged)
					break;
			}

			pfree(child_old_cents);

			/* Save converged child centroids */
			child_centroids[child] = palloc(
					(size_t)child_k * dim * sizeof(float));
			memcpy(child_centroids[child],
				   cents,
				   (size_t)child_k * dim * sizeof(float));
		}

		/* Build 2-level tree from root + child centroids */
		tree = mkt_hkmeans_build_two_level(
				root_cents,
				km_k,
				(const float **)child_centroids,
				child_ks,
				dim);

		/* Cleanup */
		for (uint32_t c = 0; c < km_k; c++)
			pfree(child_centroids[c]);
		pfree(child_centroids);
		pfree(child_ks);
		pfree(root_cents);
		pfree(all_samples);
		pfree(all_root_asgn);

		nlist = tree->nleaves;
	}
	else
	{
		/* nlevels != 2: fall back to serial hkmeans */
		uint32_t total_nsamples = 0;
		for (int t = 0; t < nparticipants; t++)
			total_nsamples += mktann_sample_counts(dsm_samples)[t];

		/* May exceed the 1GB palloc limit for large nlist / high dim. */
		float *all_samples = palloc_extended(
				(size_t)total_nsamples * dim * sizeof(float), MCXT_ALLOC_HUGE);
		uint32_t soff = 0;
		for (int t = 0; t < nparticipants; t++)
		{
			uint32_t n = mktann_sample_counts(dsm_samples)[t];
			memcpy(all_samples + (size_t)soff * dim,
				   mktann_worker_samples(dsm_samples, t),
				   (size_t)n * dim * sizeof(float));
			soff += n;
		}

		KMeansOptions km_opts	  = MKT_KMEANS_OPTIONS_DEFAULT;
		km_opts.max_iterations	  = shared->km_max_iterations;
		km_opts.initial_centroids = cents;

		tree = mkt_hkmeans_f32(
				all_samples,
				total_nsamples,
				NULL,
				dim,
				nlist,
				fan_out,
				shared->metric,
				&km_opts);

		pfree(all_samples);

		/* Still need barriers for root assign + child k-means
		 * that workers are waiting on */
		{
			uint32_t *leader_ra = mktann_root_assignments(dsm_ra, 0);
			uint32_t  ln		= mktann_sample_counts(dsm_samples)[0];
			for (uint32_t i = 0; i < ln; i++)
				leader_ra[i] = 0;

			BarrierArriveAndWait(
					barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);
		}

		for (uint32_t child = 0; child < km_k; child++)
		{
			shared->child_km_k	 = 1;
			shared->km_converged = true;
			BarrierArriveAndWait(
					barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);
		}

		nlist = tree ? tree->nleaves : 0;
	}

	{
		instr_time t_child_elapsed;
		INSTR_TIME_SET_CURRENT(t_child_elapsed);
		INSTR_TIME_SUBTRACT(t_child_elapsed, t_child_start);
		elog(LOG,
			 "mktann: child kmeans %.1fms (nlevels=%u, "
			 "%u children, %u leaves)",
			 INSTR_TIME_GET_MILLISEC(t_child_elapsed),
			 nlevels,
			 km_k,
			 tree ? tree->nleaves : 0);
	}

	if (tree == NULL)
	{
		WaitForParallelWorkersToFinish(pcxt);
		DestroyParallelContext(pcxt);
		ExitParallelMode();
		return false;
	}

	/* Update nlist in shared state (workers read it for phase 3). */
	shared->nlist = nlist;

	/* Copy tree into pre-allocated DSM slot */
	if (tree->total_size > max_tree_sz)
		elog(ERROR,
			 "mktann: tree too large for DSM (%u > %zu)",
			 tree->total_size,
			 max_tree_sz);
	memcpy(dsm_tree, tree, tree->total_size);

	/* Normalize leaf centroids for cosine */
	float *ref_vecs = hk_leaf_centroids(tree);
	if (shared->metric == DISTANCE_COSINE)
		for (uint32_t c = 0; c < nlist; c++)
			mkt_l2_normalize(ref_vecs + (size_t)c * dim, dim);

	/* Compute P^T * centroids */
	RaBitQParams *rq_params	   = mkt_rabitq_create(dim, rabitq_seed);
	float		 *pt_centroids = palloc((size_t)nlist * dim * sizeof(float));
	for (uint32_t c = 0; c < nlist; c++)
		mkt_rabitq_rotate(
				rq_params,
				ref_vecs + (size_t)c * dim,
				pt_centroids + (size_t)c * dim);

	/* Block 0 = metadata page. Extend 1 page so block 0 exists. */
	mkt_storage_extend(&storage->base, 1);

	/* Compute exact centroid page layout from the real tree,
	 * then extend for centroid + posting pages. Same layout
	 * logic as the serial path. */
	uint32_t cent_max_ent =
			mkt_centroid_max_entries_fmt(dim, params->centroid_format);
	BlockNumber *node_first_blkno = palloc(tree->nnodes * sizeof(BlockNumber));
	BlockNumber	 first_centroid	  = 1;
	BlockNumber	 first_posting	  = mkt_compute_centroid_layout(
			tree, cent_max_ent, first_centroid, node_first_blkno);
	uint32_t n_centroid_pages = first_posting - first_centroid;
	mkt_storage_extend(&storage->base, n_centroid_pages);

	instr_time t_km_end;
	INSTR_TIME_SET_CURRENT(t_km_end);
	INSTR_TIME_SUBTRACT(t_km_end, t_km_start);
	elog(LOG,
		 "mktann: tree+setup %.1fms, %u clusters",
		 INSTR_TIME_GET_MILLISEC(t_km_end),
		 nlist);

	/* Re-init parallel scan for posting phase */
	table_parallelscan_reinitialize(
			heap, ParallelTableScanFromMktShared(shared));

	/* Barrier: tree ready, workers can start posting scan. This is the last
	 * barrier; phase 3 (drain) uses the shm_mq queues, not the barrier, so the
	 * leader detaches once released. */
	BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);
	BarrierDetach(barrier);

	instr_time t_scan_start;
	INSTR_TIME_SET_CURRENT(t_scan_start);

	/*
	 * ---- Phase 3: leader pre-reserves layout, then drains worker queues ----
	 *
	 * Bounded streaming build: the launched workers (worker_id 1..N) scan,
	 * assign + RaBitQ-encode, and stream completed full pages over their
	 * shm_mq; the leader does NOT scan here. It estimates each list's size,
	 * reserves a contiguous block range per list, pre-extends the relation,
	 * then drains every queue — placing each page in its list's range and
	 * linking it into the cluster's chain — until all queues detach. The
	 * finalize below writes each list's head (with centroid metadata) and
	 * splices the chain. For both AoS and fastscan, each worker holds its
	 * trailing partial page per cluster and the finalize folds those into the
	 * head, so the head is populated and there is no per-worker under-full
	 * page left in the chain.
	 */
	HKMeansResult *tree_r = (HKMeansResult *)dsm_tree;

	/* Per-cluster page estimate from the sample assignment, extrapolated
	 * to the full table (handles skew; slight over-estimate for headroom). */
	uint32_t *cluster_counts = palloc0((size_t)nlist * sizeof(uint32_t));
	uint32_t  n_est_samples	 = 0;
	for (int w = 0; w < nparticipants; w++)
	{
		float	*sw = mktann_worker_samples(dsm_samples, w);
		uint32_t nw = mktann_sample_counts(dsm_samples)[w];
		n_est_samples += nw;
		for (uint32_t i = 0; i < nw; i++)
		{
			Distance d;
			cluster_counts[mkt_hkmeans_assign(
					tree_r, sw + (size_t)i * dim, params->metric, &d)]++;
		}
	}
	/* Extrapolate per-cluster sample counts to the full table; the reserve
	 * estimator applies the format + replication headroom. */
	bool replicate = params->soar_lambda > 0.0 ||
					 params->boundary_epsilon > 0.0;
	{
		double est_rows = RelationGetNumberOfBlocks(heap) *
						  (BLCKSZ / (double)(dim * sizeof(float) + 32));
		double scale = n_est_samples > 0 ? est_rows / n_est_samples : 1.0;
		for (uint32_t c = 0; c < nlist; c++)
			cluster_counts[c] = (uint32_t)((double)cluster_counts[c] * scale);
	}

	/* Leader-local reserve (0-based ranges; first_posting added on write).
	 * first_posting is the centroid-layout posting start computed above. */
	MktPostingReserve reserve;
	mkt_posting_reserve_init(
			&reserve,
			cluster_counts,
			nlist,
			nparticipants,
			dim,
			params->fastscan,
			replicate);
	pfree(cluster_counts);

	/* Pre-extend the relation to cover the reserved ranges. mkt_storage_extend
	 * handles any backend-specific batching internally (the PG storage chunks
	 * around the ExtendBufferedRelBy pin limit). A cluster that outgrows its
	 * (over-)reservation overflows via on-demand new_page during the drain, so
	 * there is no separate spill region to pre-extend. */
	mkt_storage_extend(&storage->base, reserve.total);

	/* Attach as receiver to each launched worker's queue. */
	int				nq = pcxt->nworkers_launched;
	shm_mq_handle **rh = palloc0(
			(size_t)nparticipants * sizeof(shm_mq_handle *));
	for (int wi = 0; wi < nq; wi++)
	{
		shm_mq *mq = (shm_mq *)mktann_posting_queue(queues_base, wi + 1);
		rh[wi + 1] = shm_mq_attach(mq, pcxt->seg, NULL);
	}

	/*
	 * Drain: place each continuation page at the next offset within its
	 * list's reserved range (offset 0 is the head, written in finalize); a
	 * list that outgrows its (over-)reservation extends the relation on demand
	 * via new_page. cl_used[c] counts placed continuations and drives the
	 * placement; cont_first/cont_last record the actual block numbers so the
	 * chain is linked from real placements rather than assuming the
	 * continuations are contiguous (they aren't, once a list overflows).
	 */
	uint32_t	*cl_used	= palloc0((size_t)nlist * sizeof(uint32_t));
	BlockNumber *cont_first = palloc((size_t)nlist * sizeof(BlockNumber));
	BlockNumber *cont_last	= palloc((size_t)nlist * sizeof(BlockNumber));
	for (uint32_t c = 0; c < nlist; c++)
	{
		cont_first[c] = InvalidBlockNumber;
		cont_last[c]  = InvalidBlockNumber;
	}
	bool *qdone = palloc0((size_t)(nq > 0 ? nq : 1) * sizeof(bool));
	int	  ndone = 0;
	while (ndone < nq)
	{
		bool progressed = false;
		for (int wi = 0; wi < nq; wi++)
		{
			Size		  len;
			void		 *data;
			shm_mq_result res;

			if (qdone[wi])
				continue;
			res = shm_mq_receive(rh[wi + 1], &len, &data, true);
			if (res == SHM_MQ_SUCCESS)
			{
				Page		src = (Page)data;
				uint32_t	c	= mkt_posting_opaque(src)->cluster_id;
				uint32_t	off = ++cl_used[c]; /* 1.. ; 0 = head */
				BlockNumber blk;
				Page		dst;
				if (off < reserve.counts[c])
				{
					/* Within the cluster's reserved (over-estimated) range. */
					blk = first_posting + reserve.starts[c] + off;
					dst = mkt_storage_write_page(&storage->base, blk);
				}
				else
				{
					/* Cluster outgrew its reservation: extend on demand. The
					 * page lands at the end of the relation (non-sequential
					 * for this list, but rare) and is linked in by block
					 * number below. */
					dst = mkt_storage_new_page(&storage->base, &blk);
				}
				memcpy(dst, src, BLCKSZ);
				mkt_storage_commit_page(&storage->base, blk);

				/* Link into this cluster's continuation chain using actual
				 * block numbers. The previous page's next_blkno is fixed up
				 * once its successor's block is known; the final page's link
				 * is set in finalize. */
				if (cont_last[c] == InvalidBlockNumber)
				{
					cont_first[c] = blk;
				}
				else
				{
					Page prev = mkt_storage_write_page(
							&storage->base, cont_last[c]);
					mkt_posting_opaque(prev)->next_blkno = blk;
					mkt_storage_commit_page(&storage->base, cont_last[c]);
				}
				cont_last[c] = blk;
				progressed	 = true;
			}
			else if (res == SHM_MQ_DETACHED)
			{
				qdone[wi] = true;
				ndone++;
				progressed = true;
			}
		}
		if (!progressed)
		{
			WaitLatch(
					MyLatch,
					WL_LATCH_SET | WL_EXIT_ON_PM_DEATH,
					-1L,
					WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);
			ResetLatch(MyLatch);
			CHECK_FOR_INTERRUPTS();
		}
	}
	for (int wi = 0; wi < nq; wi++)
		shm_mq_detach(rh[wi + 1]);
	pfree(rh);
	pfree(qdone);

	WaitForParallelWorkersToFinish(pcxt);

	instr_time t_scan_end;
	INSTR_TIME_SET_CURRENT(t_scan_end);
	INSTR_TIME_SUBTRACT(t_scan_end, t_scan_start);

	for (int i = 0; i < pcxt->nworkers_launched; i++)
		InstrAccumParallelQuery(&bufferusage[i], &walusage[i]);

	*out_heap_tuples = shared->reltuples;
	*out_indtuples	 = shared->indtuples;
	*out_soar_dupes	 = shared->soar_dupes;
	*out_tree		 = tree;

	instr_time t_merge_start;
	INSTR_TIME_SET_CURRENT(t_merge_start);

	/*
	 * Finalize each list: write its head page (the list's reserved offset 0)
	 * with centroid metadata, fold every worker's trailing partial page into
	 * the head (re-packed optimally, overflowing past the continuations / into
	 * spill), then splice the chain head -> continuations -> overflow.
	 */
	uint32_t packed_bytes = (dim + 7) / 8;
	uint8_t *unpack_buf	  = params->fastscan
								  ? palloc(MKT_FASTSCAN_GROUP * packed_bytes)
								  : NULL;
	for (uint32_t c = 0; c < nlist; c++)
	{
		BlockNumber head_blk = first_posting + reserve.starts[c];

		/* Overflow from the head builder is claimed after the
		 * continuations (offsets 1..cl_used are already written). */
		mkt_atomic_init_u32(&reserve.nexts[c], cl_used[c] + 1);

		MktPostingBuilder hb;
		if (params->fastscan)
			mkt_posting_builder_init_fastscan(
					&hb,
					&storage->base,
					rq_params,
					dim,
					c,
					ref_vecs + (size_t)c * dim,
					pt_centroids + (size_t)c * dim);
		else
			mkt_posting_builder_init(
					&hb,
					&storage->base,
					rq_params,
					dim,
					c,
					ref_vecs + (size_t)c * dim,
					pt_centroids + (size_t)c * dim);
		mkt_posting_builder_set_shared_reserve(
				&hb,
				first_posting + reserve.starts[c],
				reserve.counts[c],
				&reserve.nexts[c]);
		mkt_posting_builder_set_first_blkno(&hb, head_blk);

		/*
		 * Fold every worker's trailing partial page for this cluster into the
		 * head builder, which re-packs it optimally. Worker pages are always
		 * continuations (only the leader makes heads), so content lives at the
		 * continuation offset. For fastscan we unpack each group's codes back
		 * to per-vector 1-bit form so add_encoded can re-pack them.
		 */
		for (int w = 0; w < nparticipants; w++)
		{
			Page pg = mktann_worker_partials(dsm_partials, nlist, w) +
					  (size_t)c * BLCKSZ;
			MktPostingPageOpaque *op = mkt_posting_opaque(pg);
			if (op->entry_count == 0)
				continue;
			char	*ct	 = mkt_posting_content(pg);
			uint32_t cnt = op->entry_count;

			if (params->fastscan)
			{
				uint32_t ngroups = (cnt + MKT_FASTSCAN_GROUP - 1) /
								   MKT_FASTSCAN_GROUP;
				for (uint32_t g = 0; g < ngroups; g++)
				{
					uint32_t g_count = cnt - g * MKT_FASTSCAN_GROUP;
					if (g_count > MKT_FASTSCAN_GROUP)
						g_count = MKT_FASTSCAN_GROUP;
					mkt_fastscan_unpack_codes(
							mkt_fastscan_group_codes(ct, g, dim),
							g_count,
							dim,
							unpack_buf);
					ItemPointerData *tids =
							mkt_fastscan_group_tids(ct, g, dim);
					float *fa = mkt_fastscan_group_f_add(ct, g, dim);
					float *fr = mkt_fastscan_group_f_rescale(ct, g, dim);
					float *fe = mkt_fastscan_group_f_error(ct, g, dim);
					for (uint32_t v = 0; v < g_count; v++)
						mkt_posting_builder_add_encoded(
								&hb,
								tids[v],
								fa[v],
								fr[v],
								fe[v],
								unpack_buf + (size_t)v * packed_bytes);
				}
			}
			else
			{
				for (uint32_t e = 0; e < cnt; e++)
				{
					MktPostingEntryHeader *hdr =
							mkt_posting_entry_at(ct, e, dim);
					mkt_posting_builder_add_encoded(
							&hb,
							hdr->meta.tid,
							hdr->f_add,
							hdr->f_rescale,
							hdr->f_error,
							hdr->bits);
				}
			}
		}

		mkt_posting_builder_finish(&hb);
		mkt_posting_builder_cleanup(&hb);

		/* Splice the worker continuation chain (already linked internally
		 * during the drain, in real block order) between the head and the head
		 * builder's own overflow chain: head -> cont_first .. cont_last ->
		 * ov1, where ov1 is whatever the head builder linked to (its overflow,
		 * or InvalidBlockNumber when the head didn't overflow). */
		if (cont_first[c] != InvalidBlockNumber)
		{
			Page		hp	= mkt_storage_write_page(&storage->base, head_blk);
			BlockNumber ov1 = mkt_posting_opaque(hp)->next_blkno;
			mkt_posting_opaque(hp)->next_blkno = cont_first[c];
			mkt_storage_commit_page(&storage->base, head_blk);

			Page lp = mkt_storage_write_page(&storage->base, cont_last[c]);
			mkt_posting_opaque(lp)->next_blkno = ov1;
			mkt_storage_commit_page(&storage->base, cont_last[c]);
		}

		posting_heads[c] = head_blk;
	}

	uint32_t total_pages = RelationGetNumberOfBlocks(index) - first_posting;

	mkt_posting_reserve_free(&reserve);
	pfree(cl_used);
	pfree(cont_first);
	pfree(cont_last);
	if (unpack_buf != NULL)
		pfree(unpack_buf);
	pfree(pt_centroids);

	instr_time t_merge_end;
	INSTR_TIME_SET_CURRENT(t_merge_end);
	INSTR_TIME_SUBTRACT(t_merge_end, t_merge_start);

	elog(LOG,
		 "mktann: parallel streaming build with %d workers, "
		 "%u clusters, %u pages, "
		 "scan+drain %.1fms, finalize %.1fms",
		 pcxt->nworkers_launched,
		 nlist,
		 total_pages,
		 INSTR_TIME_GET_MILLISEC(t_scan_end),
		 INSTR_TIME_GET_MILLISEC(t_merge_end));

	DestroyParallelContext(pcxt);
	ExitParallelMode();

	return true;
}
