/*
 * mktann_parallel.c - PG parallel worker for mktann index build
 *
 * Multi-phase parallel build:
 *   Phase 1 (sampling): cooperative heap scan, each worker collects
 *     samples into its own slot.
 *   Phase 2 (k-means): leader seeds the initial centroids, then
 *     iterative assignment + accumulation on per-worker samples,
 *     barrier-synchronized with leader reduce.
 *   Phase 3 (posting): cooperative heap scan, tree descent +
 *     RaBitQ encode + streaming to posting pages.
 */

#include <postgres.h>

#include <access/parallel.h>
#include <access/table.h>
#include <access/tableam.h>
#include <catalog/index.h>
#include <miscadmin.h>
#include <pgstat.h>
#include <storage/barrier.h>
#include <storage/shm_toc.h>
#include <tcop/tcopprot.h>
#include <utils/memutils.h>
#include <utils/rel.h>

#include "algo/hkmeans.h"
#include "algo/kmeans_internal.h"
#include "algo/vecops.h"
#include "index/posting_build.h"
#include "index/posting_build_parallel.h"
#include "mkt_pg.h"
#include "mkt_vector.h"
#include "mktann_parallel.h"
#include "quant/rabitq.h"

/* ----------------------------------------------------------------
 * Phase 1: Sampling callback
 *
 * Collects vectors into a per-worker sample slot (stride-subsampled,
 * normalized for cosine). Initial centroids are seeded later by the
 * leader from the pooled samples.
 * ---------------------------------------------------------------- */

void
mktann_sample_callback(
		Relation	index,
		ItemPointer tid,
		Datum	   *values,
		bool	   *isnull,
		bool		tuple_is_alive,
		void	   *state)
{
	SampleCbState *sc = (SampleCbState *)state;

	(void)index;
	(void)tid;
	(void)tuple_is_alive;

	if (isnull[0])
		return;

	/* Stride-based subsampling */
	if (sc->stride_counter > 0)
	{
		sc->stride_counter--;
		return;
	}
	sc->stride_counter = sc->stride - 1;

	if (sc->count >= sc->max_samples)
		return;

	MktVector *vec	= DatumGetMktVector(values[0]);
	float	  *src	= MKT_VECTOR_DATA(vec);
	Dimension  dim	= sc->dim;
	float	  *dest = sc->samples + (size_t)sc->count * dim;

	memcpy(dest, src, dim * sizeof(float));

	/* Normalize for cosine */
	if (sc->metric == DISTANCE_COSINE)
	{
		float norm = mkt_l2_norm(dest, dim);
		if (norm > 0.0f)
			mkt_vector_scale(dest, 1.0f / norm, dest, dim);
	}

	sc->count++;
}

/* ----------------------------------------------------------------
 * Phase 2: K-means assignment — thin wrappers around shared kernel
 * ---------------------------------------------------------------- */

void
mktann_km_assign_and_accumulate(
		const float	  *samples,
		uint32_t	   nsamples,
		const float	  *centroids,
		const float	  *norms_c,
		uint32_t	   nlist,
		Dimension	   dim,
		DistanceMetric metric,
		float		  *out_sums,
		uint32_t	  *out_cnts,
		float		  *out_cost)
{
	memset(out_sums, 0, (size_t)nlist * dim * sizeof(float));
	memset(out_cnts, 0, nlist * sizeof(uint32_t));
	*out_cost = 0.0f;
	kmeans_assign_accumulate(
			samples,
			NULL,
			0,
			nsamples,
			centroids,
			norms_c,
			nlist,
			dim,
			metric,
			NULL,
			0,
			out_sums,
			out_cnts,
			out_cost);
}

void
mktann_km_assign_and_accumulate_filtered(
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
		float		   *out_cost)
{
	memset(out_sums, 0, (size_t)nlist * dim * sizeof(float));
	memset(out_cnts, 0, nlist * sizeof(uint32_t));
	*out_cost = 0.0f;
	kmeans_assign_accumulate(
			samples,
			NULL,
			0,
			nsamples,
			centroids,
			norms_c,
			nlist,
			dim,
			metric,
			root_assignments,
			target_child,
			out_sums,
			out_cnts,
			out_cost);
}

/* ----------------------------------------------------------------
 * Phase 3: Posting build callback
 *
 * Uses MktPostingWorkerState in deferred batch mode (storage=NULL).
 * Assigns vectors to clusters via tree descent, encodes with
 * RaBitQ, and adds to per-cluster posting builders. Batch pages
 * accumulate in process memory and are copied to DSM after scan.
 * ---------------------------------------------------------------- */

void
posting_cb_batch_init(PostingCbState *cbs)
{
	cbs->batch_count = 0;
	cbs->use_batch	 = (cbs->bp.soar_lambda > 0.0 ||
						cbs->bp.boundary_epsilon > 0.0) &&
					 mkt_secondary_batch_available();
	if (!cbs->use_batch)
		return;

	uint32_t  B	  = MKT_SECONDARY_BATCH;
	Dimension dim = cbs->bp.dim;

	cbs->enc_batch		 = palloc((size_t)B * dim * sizeof(float));
	cbs->batch_tids		 = palloc(B * sizeof(ItemPointerData));
	cbs->batch_primary	 = palloc(B * sizeof(uint32_t));
	cbs->batch_pdist	 = palloc(B * sizeof(float));
	cbs->batch_secondary = palloc(B * sizeof(uint32_t));
	mkt_secondary_batch_init(
			&cbs->sb,
			hk_leaf_centroids(cbs->tree),
			cbs->tree->nleaves,
			dim,
			B);
}

void
posting_cb_batch_flush(PostingCbState *cbs)
{
	if (!cbs->use_batch || cbs->batch_count == 0)
		return;

	Dimension	  dim	  = cbs->bp.dim;
	MemoryContext old_ctx = MemoryContextSwitchTo(cbs->worker_ctx);

	mkt_secondary_batch_assign(
			&cbs->sb,
			cbs->enc_batch,
			cbs->batch_count,
			cbs->batch_primary,
			cbs->batch_pdist,
			&cbs->bp,
			cbs->batch_secondary);

	for (uint32_t k = 0; k < cbs->batch_count; k++)
	{
		mkt_posting_worker_add_heap(
				cbs->ws,
				cbs->batch_tids[k],
				cbs->enc_batch + (size_t)k * dim,
				cbs->batch_primary[k],
				cbs->batch_secondary[k]);
		if (cbs->batch_secondary[k] != MKT_INVALID_CLUSTER)
			cbs->soar_dupes++;
	}

	cbs->batch_count = 0;
	MemoryContextSwitchTo(old_ctx);
}

void
posting_cb_batch_cleanup(PostingCbState *cbs)
{
	if (!cbs->use_batch)
		return;
	mkt_secondary_batch_free(&cbs->sb);
	pfree(cbs->enc_batch);
	pfree(cbs->batch_tids);
	pfree(cbs->batch_primary);
	pfree(cbs->batch_pdist);
	pfree(cbs->batch_secondary);
}

void
posting_build_callback(
		Relation	index,
		ItemPointer tid,
		Datum	   *values,
		bool	   *isnull,
		bool		tuple_is_alive,
		void	   *state)
{
	PostingCbState *cbs = (PostingCbState *)state;

	(void)index;
	(void)tuple_is_alive;

	if (isnull[0])
		return;

	MktVector *vec	= DatumGetMktVector(values[0]);
	VectorRef  vref = MktVectorToRef(vec);

	if (cbs->use_batch)
	{
		/* Buffer the tuple; the secondary search runs per batch. Primary
		 * assignment (tree descent) writes the encoded vector into the
		 * batch buffer. No per-tuple allocation, so no tmp context. */
		uint32_t k = cbs->batch_count;
		Distance d;
		cbs->batch_primary[k] = mkt_build_assign_primary(
				cbs->tree,
				vref.data,
				&cbs->bp,
				cbs->enc_batch + (size_t)k * cbs->bp.dim,
				&d);
		cbs->batch_pdist[k] = (float)d;
		cbs->batch_tids[k]	= *tid;
		cbs->batch_count++;
		cbs->indtuples++;

		if (cbs->batch_count == MKT_SECONDARY_BATCH)
			posting_cb_batch_flush(cbs);
		return;
	}

	MemoryContext old_ctx = MemoryContextSwitchTo(cbs->tmp_ctx);

	MktBuildAssignment asgn = mkt_build_assign_vector(
			cbs->tree, vref.data, &cbs->bp, &cbs->bufs);

	MemoryContextSwitchTo(cbs->worker_ctx);

	mkt_posting_worker_add_heap(
			cbs->ws, *tid, asgn.enc_vector, asgn.primary, asgn.secondary);

	MemoryContextSwitchTo(old_ctx);

	cbs->indtuples++;
	if (asgn.secondary != MKT_INVALID_CLUSTER)
		cbs->soar_dupes++;

	MemoryContextReset(cbs->tmp_ctx);
}

/*
 * Page sink: stream one completed full page to the leader over this
 * worker's shm_mq. The page carries its own cluster id and first/
 * continuation flag in its header, so only the BLCKSZ page is sent.
 * The send blocks when the ring is full (backpressure) — that is what
 * bounds worker memory to ~one working page per cluster.
 */
static void
mktann_posting_page_sink(void *ctx, uint32_t cluster_id, const char *page)
{
	shm_mq_handle *mqh = (shm_mq_handle *)ctx;
	shm_mq_result  res;

	(void)cluster_id; /* carried in the page header */
	res = shm_mq_send(mqh, BLCKSZ, page, false, true);
	if (res != SHM_MQ_SUCCESS)
		elog(ERROR,
			 "mktann: posting page queue send failed (result %d)",
			 (int)res);
}

/* ----------------------------------------------------------------
 * Worker entry point — multi-phase build
 * ---------------------------------------------------------------- */

void
mktann_parallel_build_main(dsm_segment *seg, shm_toc *toc)
{
	MktBuildShared *shared	= shm_toc_lookup(toc, MKTANN_KEY_SHARED, false);
	Barrier		   *barrier = shm_toc_lookup(toc, MKTANN_KEY_BARRIER, false);

	char *sharedquery  = shm_toc_lookup(toc, MKTANN_KEY_QUERY_TEXT, true);
	debug_query_string = sharedquery;
	pgstat_report_activity(STATE_RUNNING, debug_query_string);
	pgstat_report_query_id(shared->queryid, false);

	Relation heapRel  = table_open(shared->heaprelid, ShareLock);
	Relation indexRel = index_open(shared->indexrelid, AccessExclusiveLock);

	int		  worker_id = ParallelWorkerNumber + 1;
	Dimension dim		= shared->dim;

	InstrStartParallelQuery();

	/*
	 * Attach to the phase barrier. The leader initializes the barrier with
	 * itself as the sole party and waits for every launched worker to attach
	 * before advancing, so attaching here (before the first phase) keeps all
	 * participants in lockstep regardless of how many workers were launched.
	 */
	BarrierAttach(barrier);

	/* ---- Phase 1: Sampling ---- */

	MktDsmSamples *dsm_samples =
			shm_toc_lookup(toc, MKTANN_KEY_SAMPLES, false);
	char  *centroids_base = shm_toc_lookup(toc, MKTANN_KEY_CENTROIDS, false);
	float *cents		  = mktann_centroids(centroids_base);

	uint32_t km_k = shared->km_k;

	/* Compute stride: sample ~max_per_worker from our heap chunk */
	double est_rows_total = RelationGetNumberOfBlocks(heapRel) *
							(BLCKSZ / (double)(dim * sizeof(float) + 32));
	double	 est_per_worker = est_rows_total / shared->nparticipants;
	uint32_t stride			= 1;
	if (est_per_worker > shared->max_samples_per_worker)
		stride = (uint32_t)(est_per_worker / shared->max_samples_per_worker);
	if (stride < 1)
		stride = 1;

	SampleCbState sc = {
			.samples		= mktann_worker_samples(dsm_samples, worker_id),
			.count			= 0,
			.max_samples	= shared->max_samples_per_worker,
			.stride			= stride,
			.stride_counter = 0,
			.dim			= dim,
			.metric			= shared->metric,
	};

	IndexInfo	 *indexInfo = BuildIndexInfo(indexRel);
	TableScanDesc scan		= table_beginscan_parallel(
			 heapRel, ParallelTableScanFromMktShared(shared));

	table_index_build_scan(
			heapRel,
			indexRel,
			indexInfo,
			true,
			false,
			mktann_sample_callback,
			&sc,
			scan);

	*mktann_sample_counts(dsm_samples) = sc.count;
	/* Fix: write to this worker's slot */
	mktann_sample_counts(dsm_samples)[worker_id] = sc.count;

	/* Barrier: all participants done sampling */
	BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

	/*
	 * Barrier: leader has seeded the initial centroids (and their norms) from
	 * the pooled samples. Workers don't touch the centroid buffer until here,
	 * so this also guards the read below against the leader's concurrent
	 * write.
	 */
	BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

	/* ---- Phase 2: K-means iterate (root level, k=km_k) ---- */

	/*
	 * Each participant keeps and processes the samples it gathered itself in
	 * the work-stealing scan above; we deliberately do NOT redistribute them
	 * to equalize per-worker counts. The scan hands out blocks at each
	 * participant's own rate, so a worker's sample count ends up roughly
	 * proportional to its throughput — making the per-iteration k-means cost
	 * (n_samples / throughput) about equal across workers, i.e. self-balanced.
	 * Forcing equal counts would instead hand a slow worker an equal load and
	 * make it the barrier bottleneck. The dominant build phase (posting) is
	 * itself work-stealing, so k-means balance is second-order; revisit only
	 * if it ever proves to be a real bottleneck.
	 */
	char  *km_workers_base = shm_toc_lookup(toc, MKTANN_KEY_KM_WORKERS, false);
	float *my_samples	   = mktann_worker_samples(dsm_samples, worker_id);
	uint32_t my_nsamples   = mktann_sample_counts(dsm_samples)[worker_id];

	float *norms_c = mktann_norms_c(centroids_base, km_k, dim);
	float *my_sums =
			mktann_km_worker_sums(km_workers_base, km_k, dim, worker_id);
	uint32_t *my_cnts =
			mktann_km_worker_cnts(km_workers_base, km_k, dim, worker_id);
	float *my_cost =
			mktann_km_worker_cost(km_workers_base, km_k, dim, worker_id);

	for (uint32_t iter = 0; iter < shared->km_max_iterations; iter++)
	{
		/* Assignment + accumulation on this worker's samples */
		mktann_km_assign_and_accumulate(
				my_samples,
				my_nsamples,
				cents,
				norms_c,
				km_k,
				dim,
				shared->metric,
				my_sums,
				my_cnts,
				my_cost);

		/* Barrier: work done — leader will reduce */
		BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

		/* Barrier: leader done reducing — read updated centroids */
		BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

		if (shared->km_converged)
			break;
	}

	/* ---- Phase 2b: Root assignment ---- */

	MktDsmRootAssign *dsm_ra =
			shm_toc_lookup(toc, MKTANN_KEY_ROOT_ASSIGN, false);
	uint32_t *my_root_asgn = mktann_root_assignments(dsm_ra, worker_id);

	kmeans_assign(
			my_samples,
			0,
			my_nsamples,
			cents,
			norms_c,
			km_k,
			dim,
			shared->metric,
			my_root_asgn);

	/* Barrier: all done with root assignment */
	BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

	/* ---- Phase 2c: Child k-means ---- */

	for (uint32_t child = 0; child < km_k; child++)
	{
		/* Barrier: leader wrote child centroids */
		BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

		uint32_t child_k = shared->child_km_k;

		if (child_k <= 1)
			continue; /* nothing to iterate */

		/*
		 * Reuse the km_workers accumulator buffer. The leader
		 * sizes it for km_k (= fan_out), and child_k <= fan_out,
		 * so the per-worker slot is large enough.
		 */
		float *child_sums = mktann_km_worker_sums(
				km_workers_base, child_k, dim, worker_id);
		uint32_t *child_cnts = mktann_km_worker_cnts(
				km_workers_base, child_k, dim, worker_id);
		float *child_cost = mktann_km_worker_cost(
				km_workers_base, child_k, dim, worker_id);

		float *child_norms_c = mktann_norms_c(centroids_base, child_k, dim);

		for (uint32_t iter = 0; iter < shared->km_max_iterations; iter++)
		{
			mktann_km_assign_and_accumulate_filtered(
					my_samples,
					my_nsamples,
					my_root_asgn,
					child,
					cents,
					child_norms_c,
					child_k,
					dim,
					shared->metric,
					child_sums,
					child_cnts,
					child_cost);

			/* Barrier: all done — leader reduces */
			BarrierArriveAndWait(
					barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

			/* Barrier: leader updated centroids */
			BarrierArriveAndWait(
					barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

			if (shared->km_converged)
				break;
		}
	}

	/* Barrier: leader built tree, set up posting reserve. This is the worker's
	 * last barrier — phase 3 streams pages over the shm_mq, so detach once
	 * released so the leader's drain isn't gated on a stale party count. */
	BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);
	BarrierDetach(barrier);

	/* ---- Phase 3: Posting scan (deferred batch) ---- */

	HKMeansResult *tree = shm_toc_lookup(toc, MKTANN_KEY_TREE, false);
	char *worker_output = shm_toc_lookup(toc, MKTANN_KEY_WORKER_OUTPUT, false);
	char *dsm_partials	= shm_toc_lookup(toc, MKTANN_KEY_PARTIALS, true);
	char *queues_base = shm_toc_lookup(toc, MKTANN_KEY_POSTING_QUEUES, false);

	uint32_t nlist = shared->nlist;

	/* Attach this worker's posting-page queue as the sender; full pages
	 * are streamed to the leader over it as they fill. */
	shm_mq *mq = (shm_mq *)mktann_posting_queue(queues_base, worker_id);
	shm_mq_set_sender(mq, MyProc);
	shm_mq_handle *qhandle = shm_mq_attach(mq, seg, NULL);

	RaBitQParams *rq_params = mkt_rabitq_create(dim, shared->rabitq_seed);

	const float *leaf_cents	  = hk_leaf_centroids(tree);
	float		*pt_centroids = palloc((size_t)nlist * dim * sizeof(float));
	for (uint32_t c = 0; c < nlist; c++)
		mkt_rabitq_rotate(
				rq_params,
				leaf_cents + (size_t)c * dim,
				pt_centroids + (size_t)c * dim);

	char *my_partials =
			(dsm_partials != NULL)
					? mktann_worker_partials(dsm_partials, nlist, worker_id)
					: NULL;

	MemoryContext worker_ctx = AllocSetContextCreate(
			CurrentMemoryContext,
			"mktann worker posting",
			ALLOCSET_DEFAULT_SIZES);
	MemoryContext prev = MemoryContextSwitchTo(worker_ctx);

	MktPostingWorkerState ws;
	mkt_posting_worker_init(
			&ws,
			(uint32_t)worker_id,
			nlist,
			dim,
			shared->fastscan,
			NULL,
			rq_params,
			leaf_cents,
			pt_centroids,
			NULL,
			my_partials,
			worker_ctx);

	/* Stream each completed full page to the leader over the queue
	 * instead of accumulating it; only the trailing partial per cluster
	 * is retained (copied to the partials DSM buffer by worker_finish). */
	mkt_posting_worker_set_page_sink(&ws, mktann_posting_page_sink, qhandle);

	MktBuildWorkerBufs bufs = mkt_build_worker_bufs_create(dim);

	MemoryContextSwitchTo(prev);

	PostingCbState cbs = {
			.tree		= tree,
			.bp			= {.dim				 = dim,
						   .metric			 = shared->metric,
						   .soar_lambda		 = shared->soar_lambda,
						   .boundary_epsilon = shared->boundary_epsilon},
			.bufs		= bufs,
			.ws			= &ws,
			.indtuples	= 0,
			.soar_dupes = 0,
			.tmp_ctx	= AllocSetContextCreate(
					   CurrentMemoryContext,
					   "mktann parallel tuple",
					   ALLOCSET_DEFAULT_SIZES),
			.worker_ctx = worker_ctx,
	};

	MemoryContext batch_ctx = MemoryContextSwitchTo(worker_ctx);
	posting_cb_batch_init(&cbs);
	MemoryContextSwitchTo(batch_ctx);

	/* Second parallel scan for posting build */
	TableScanDesc scan2 = table_beginscan_parallel(
			heapRel, ParallelTableScanFromMktShared(shared));

	table_index_build_scan(
			heapRel,
			indexRel,
			indexInfo,
			true,
			false,
			posting_build_callback,
			&cbs,
			scan2);

	posting_cb_batch_flush(&cbs);
	posting_cb_batch_cleanup(&cbs);

	mkt_posting_worker_finish(&ws);

	/*
	 * All full pages have been streamed to the leader; detach the queue
	 * to signal this worker is done (the leader drains until every queue
	 * detaches). The trailing partial page per cluster was copied to the
	 * partials DSM buffer by worker_finish for the leader to merge.
	 */
	shm_mq_detach(qhandle);

	/* Copy active flags to worker_output */
	bool *wa = mktann_worker_active(worker_output, nlist, worker_id);
	memcpy(wa, ws.active, nlist * sizeof(bool));

	SpinLockAcquire(&shared->mutex);
	shared->nparticipantsdone++;
	shared->indtuples += cbs.indtuples;
	shared->soar_dupes += cbs.soar_dupes;
	SpinLockRelease(&shared->mutex);
	ConditionVariableSignal(&shared->workersdonecv);

	mkt_posting_worker_cleanup(&ws);
	mkt_build_worker_bufs_free(&bufs);
	pfree(pt_centroids);

	BufferUsage *bufferusage =
			shm_toc_lookup(toc, MKTANN_KEY_BUFFER_USAGE, false);
	WalUsage *walusage = shm_toc_lookup(toc, MKTANN_KEY_WAL_USAGE, false);
	InstrEndParallelQuery(
			&bufferusage[ParallelWorkerNumber],
			&walusage[ParallelWorkerNumber]);

	index_close(indexRel, AccessExclusiveLock);
	table_close(heapRel, ShareLock);
}
