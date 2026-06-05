/*
 * parallel_build_worker.c - PG parallel index build, worker side
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

#ifdef MKT_STANDALONE
#include "core/pg_compat.h"
#else
#include <postgres.h>

#include <access/parallel.h>
#include <catalog/index.h>
#include <miscadmin.h>
#include <storage/barrier.h>
#include <storage/bufmgr.h>
#include <storage/shm_toc.h>
#include <utils/rel.h>
#include <utils/wait_event.h>
#endif

#include "algo/hkmeans.h"
#include "algo/kmeans_internal.h"
#include "algo/vecops.h"
#include "core/log.h"
#include "core/memory.h"
#include "core/parallel_ctx.h" /* ParallelWorkerNumber */
#include "index/parallel_build.h"
#include "index/posting_build.h"
#include "index/posting_build_parallel.h"
#include "quant/rabitq.h"

/* ----------------------------------------------------------------
 * Phase 1: Sampling callback
 *
 * Collects vectors into a per-worker sample slot (stride-subsampled,
 * normalized for cosine). Initial centroids are seeded later by the
 * leader from the pooled samples.
 * ---------------------------------------------------------------- */

void
mkt_sample_cb(void *state, ItemPointerData tid, const float *vec)
{
	SampleCbState *sc = (SampleCbState *)state;

	(void)tid;

	/* Stride-based subsampling */
	if (sc->stride_counter > 0)
	{
		sc->stride_counter--;
		return;
	}
	sc->stride_counter = sc->stride - 1;

	if (sc->count >= sc->max_samples)
		return;

	Dimension dim  = sc->dim;
	float	 *dest = sc->samples + (size_t)sc->count * dim;

	memcpy(dest, vec, dim * sizeof(float));

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
mkt_km_assign_and_accumulate(
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
mkt_km_assign_and_accumulate_filtered(
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
 * Phase 2c: work-partitioned child k-means (shared kernel)
 *
 * See the contract on the declaration in parallel_build.h. Replaces the old
 * SPMD-lockstep variant (one child at a time, two barriers per iteration) — at
 * 45 children x ~20 iterations that was ~1800 whole-party barriers gating work
 * that is a few thousand samples per child, so it was dominated by sync and
 * load imbalance. Distributing whole children across participants makes each
 * child's k-means a private, barrier-free serial run.
 * ---------------------------------------------------------------- */

void
mkt_child_kmeans_partitioned(
		int				  participant_id,
		int				  nparticipants,
		MktDsmSamples	 *dsm_samples,
		MktDsmRootAssign *dsm_ra,
		const float		 *root_cents,
		uint32_t		  km_k,
		uint32_t		  fan_out,
		Dimension		  dim,
		DistanceMetric	  metric,
		uint32_t		  km_max_iterations,
		float			  km_tolerance,
		float			 *out_child_cents,
		uint32_t		 *out_child_ks)
{
	/* Sample count per child across all participants (one pass). */
	uint32_t *child_count = mkt_alloc0((size_t)km_k * sizeof(uint32_t));
	for (int t = 0; t < nparticipants; t++)
	{
		const uint32_t *ra = mkt_dsm_root_assignments(dsm_ra, t);
		uint32_t		n  = mkt_dsm_sample_counts(dsm_samples)[t];
		for (uint32_t i = 0; i < n; i++)
			if (ra[i] < km_k)
				child_count[ra[i]]++;
	}

	/* Scratch reused across this participant's children (max k = fan_out). */
	float	 *cents	 = mkt_alloc((size_t)fan_out * dim * sizeof(float));
	float	 *old	 = mkt_alloc((size_t)fan_out * dim * sizeof(float));
	float	 *sums	 = mkt_alloc((size_t)fan_out * dim * sizeof(float));
	uint32_t *cnts	 = mkt_alloc((size_t)fan_out * sizeof(uint32_t));
	float	 *norms	 = mkt_alloc((size_t)fan_out * sizeof(float));
	float	  tol_sq = km_tolerance * km_tolerance;

	for (uint32_t child = (uint32_t)participant_id; child < km_k;
		 child += (uint32_t)nparticipants)
	{
		uint32_t cc		 = child_count[child];
		uint32_t child_k = fan_out < cc ? fan_out : cc;
		if (child_k < 1)
			child_k = 1;
		out_child_ks[child] = child_k;

		float *cout = out_child_cents + (size_t)child * fan_out * dim;

		/* Collapsed child: the single leaf is the root centroid itself. */
		if (child_k <= 1)
		{
			memcpy(cout,
				   root_cents + (size_t)child * dim,
				   (size_t)dim * sizeof(float));
			continue;
		}

		/* Gather this child's samples contiguously (slot order, so the first
		 * child_k rows are the initial centroids — matching the old init). */
		float	*buf = mkt_alloc((size_t)cc * dim * sizeof(float));
		uint32_t g	 = 0;
		for (int t = 0; t < nparticipants; t++)
		{
			const uint32_t *ra = mkt_dsm_root_assignments(dsm_ra, t);
			const float	   *sp = mkt_dsm_worker_samples(dsm_samples, t);
			uint32_t		n  = mkt_dsm_sample_counts(dsm_samples)[t];
			for (uint32_t i = 0; i < n; i++)
				if (ra[i] == child)
				{
					memcpy(buf + (size_t)g * dim,
						   sp + (size_t)i * dim,
						   (size_t)dim * sizeof(float));
					g++;
				}
		}

		memcpy(cents, buf, (size_t)child_k * dim * sizeof(float));
		if (metric == DISTANCE_L2)
			for (uint32_t j = 0; j < child_k; j++)
				norms[j] = mkt_l2_norm_squared(cents + (size_t)j * dim, dim);

		for (uint32_t iter = 0; iter < km_max_iterations; iter++)
		{
			float cost;
			mkt_km_assign_and_accumulate(
					buf,
					cc,
					cents,
					norms,
					child_k,
					dim,
					metric,
					sums,
					cnts,
					&cost);

			memcpy(old, cents, (size_t)child_k * dim * sizeof(float));

			const float *const	  csums[1] = {sums};
			const uint32_t *const ccnts[1] = {cnts};
			float				  total_cost;
			float				  shift_sq = kmeans_merge_centroids(
					cents,
					norms,
					old,
					csums,
					ccnts,
					&cost,
					1,
					child_k,
					dim,
					metric,
					&total_cost);

			if (shift_sq < tol_sq)
				break;
		}

		memcpy(cout, cents, (size_t)child_k * dim * sizeof(float));
		mkt_free(buf);
	}

	mkt_free(norms);
	mkt_free(cnts);
	mkt_free(sums);
	mkt_free(old);
	mkt_free(cents);
	mkt_free(child_count);
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

	cbs->enc_batch		 = mkt_alloc((size_t)B * dim * sizeof(float));
	cbs->batch_tids		 = mkt_alloc(B * sizeof(ItemPointerData));
	cbs->batch_primary	 = mkt_alloc(B * sizeof(uint32_t));
	cbs->batch_pdist	 = mkt_alloc(B * sizeof(float));
	cbs->batch_secondary = mkt_alloc(B * sizeof(uint32_t));
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

	Dimension dim	  = cbs->bp.dim;
	MktMemCtx old_ctx = mkt_memctx_switch(cbs->worker_ctx);

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
	mkt_memctx_switch(old_ctx);
}

void
posting_cb_batch_cleanup(PostingCbState *cbs)
{
	if (!cbs->use_batch)
		return;
	mkt_secondary_batch_free(&cbs->sb);
	mkt_free(cbs->enc_batch);
	mkt_free(cbs->batch_tids);
	mkt_free(cbs->batch_primary);
	mkt_free(cbs->batch_pdist);
	mkt_free(cbs->batch_secondary);
}

void
posting_cb(void *state, ItemPointerData tid, const float *vec)
{
	PostingCbState *cbs = (PostingCbState *)state;

	if (cbs->use_batch)
	{
		/* Buffer the tuple; the secondary search runs per batch. Primary
		 * assignment (tree descent) writes the encoded vector into the
		 * batch buffer. No per-tuple allocation, so no tmp context. */
		uint32_t k = cbs->batch_count;
		Distance d;
		cbs->batch_primary[k] = mkt_build_assign_primary(
				cbs->tree,
				vec,
				&cbs->bp,
				cbs->enc_batch + (size_t)k * cbs->bp.dim,
				&d);
		cbs->batch_pdist[k] = (float)d;
		cbs->batch_tids[k]	= tid;
		cbs->batch_count++;
		cbs->indtuples++;

		if (cbs->batch_count == MKT_SECONDARY_BATCH)
			posting_cb_batch_flush(cbs);
		return;
	}

	MktMemCtx old_ctx = mkt_memctx_switch(cbs->tmp_ctx);

	MktBuildAssignment asgn =
			mkt_build_assign_vector(cbs->tree, vec, &cbs->bp, &cbs->bufs);

	mkt_memctx_switch(cbs->worker_ctx);

	mkt_posting_worker_add_heap(
			cbs->ws, tid, asgn.enc_vector, asgn.primary, asgn.secondary);

	mkt_memctx_switch(old_ctx);

	cbs->indtuples++;
	if (asgn.secondary != MKT_INVALID_CLUSTER)
		cbs->soar_dupes++;

	mkt_memctx_reset(cbs->tmp_ctx);
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
		mkt_error(
				"mktann: posting page queue send failed (result %d)",
				(int)res);
}

/* ----------------------------------------------------------------
 * Worker entry point — multi-phase build
 * ---------------------------------------------------------------- */

void
mkt_parallel_build_main(dsm_segment *seg, shm_toc *toc)
{
	MktPBuildWorker w;
	mkt_pbuild_worker_attach(toc, &w);

	MktBuildShared *shared	  = w.shared;
	Barrier		   *barrier	  = w.barrier;
	Relation		heapRel	  = w.heapRel;
	Relation		indexRel  = w.indexRel;
	int				worker_id = w.worker_id;
	Dimension		dim		  = w.dim;

	/* ---- Phase 1: Sampling ---- */

	MktDsmSamples *dsm_samples =
			shm_toc_lookup(toc, MKT_DSM_KEY_SAMPLES, false);
	char  *centroids_base = shm_toc_lookup(toc, MKT_DSM_KEY_CENTROIDS, false);
	float *cents		  = mkt_dsm_centroids(centroids_base);

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
			.samples		= mkt_dsm_worker_samples(dsm_samples, worker_id),
			.count			= 0,
			.max_samples	= shared->max_samples_per_worker,
			.stride			= stride,
			.stride_counter = 0,
			.dim			= dim,
			.metric			= shared->metric,
	};

	IndexInfo *indexInfo = BuildIndexInfo(indexRel);

	mkt_build_scan(
			heapRel,
			indexRel,
			indexInfo,
			shared,
			true,
			false,
			mkt_sample_cb,
			&sc);

	*mkt_dsm_sample_counts(dsm_samples) = sc.count;
	/* Fix: write to this worker's slot */
	mkt_dsm_sample_counts(dsm_samples)[worker_id] = sc.count;

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
	char *km_workers_base = shm_toc_lookup(toc, MKT_DSM_KEY_KM_WORKERS, false);
	float	*my_samples	  = mkt_dsm_worker_samples(dsm_samples, worker_id);
	uint32_t my_nsamples  = mkt_dsm_sample_counts(dsm_samples)[worker_id];

	float *norms_c = mkt_dsm_norms_c(centroids_base, km_k, dim);
	float *my_sums =
			mkt_dsm_km_worker_sums(km_workers_base, km_k, dim, worker_id);
	uint32_t *my_cnts =
			mkt_dsm_km_worker_cnts(km_workers_base, km_k, dim, worker_id);
	float *my_cost =
			mkt_dsm_km_worker_cost(km_workers_base, km_k, dim, worker_id);

	for (uint32_t iter = 0; iter < shared->km_max_iterations; iter++)
	{
		/* Assignment + accumulation on this worker's samples */
		mkt_km_assign_and_accumulate(
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
			shm_toc_lookup(toc, MKT_DSM_KEY_ROOT_ASSIGN, false);
	uint32_t *my_root_asgn = mkt_dsm_root_assignments(dsm_ra, worker_id);

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

	/* ---- Phase 2c: Child k-means (work-partitioned, barrier-free) ----
	 *
	 * Two-level tree only; for other depths the leader builds the tree
	 * serially and we just wait at the barrier below. Each participant runs
	 * the children it owns to completion against the pooled samples in DSM. */
	if (mkt_compute_nlevels(shared->nlist, shared->fan_out) == 2)
	{
		char *child_base =
				shm_toc_lookup(toc, MKT_DSM_KEY_CHILD_CENTROIDS, false);
		uint32_t fan_out = shared->fan_out;
		mkt_child_kmeans_partitioned(
				worker_id,
				shared->nparticipants,
				dsm_samples,
				dsm_ra,
				cents,
				km_k,
				fan_out,
				dim,
				shared->metric,
				shared->km_max_iterations,
				shared->km_tolerance,
				mkt_dsm_child_cents(child_base),
				mkt_dsm_child_ks(child_base, km_k, fan_out, dim));
	}

	/* Barrier: all participants done child k-means; leader builds the tree. */
	BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

	/* Barrier: leader built tree, set up posting reserve. This is the worker's
	 * last barrier — phase 3 streams pages over the shm_mq, so detach once
	 * released so the leader's drain isn't gated on a stale party count. */
	BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);
	BarrierDetach(barrier);

	/* ---- Phase 3: Posting scan (deferred batch) ---- */

	HKMeansResult *tree = shm_toc_lookup(toc, MKT_DSM_KEY_TREE, false);
	char		  *worker_output =
			shm_toc_lookup(toc, MKT_DSM_KEY_WORKER_OUTPUT, false);
	char *dsm_partials = shm_toc_lookup(toc, MKT_DSM_KEY_PARTIALS, true);
	char *queues_base = shm_toc_lookup(toc, MKT_DSM_KEY_POSTING_QUEUES, false);

	uint32_t nlist = shared->nlist;

	/* Attach this worker's posting-page queue as the sender; full pages
	 * are streamed to the leader over it as they fill. */
	shm_mq *mq = (shm_mq *)mkt_dsm_posting_queue(queues_base, worker_id);
	shm_mq_set_sender(mq, MyProc);
	shm_mq_handle *qhandle = shm_mq_attach(mq, seg, NULL);

	/* Build the RaBitQ params from the leader's shared rotation matrix rather
	 * than regenerating the identical orthogonal matrix per worker. */
	const float *rabitq_matrix =
			shm_toc_lookup(toc, MKT_DSM_KEY_RABITQ_MATRIX, false);
	RaBitQParams *rq_params = mkt_rabitq_create_from_matrix(
			dim, shared->rabitq_seed, rabitq_matrix);

	const float *leaf_cents	  = hk_leaf_centroids(tree);
	float		*pt_centroids = mkt_alloc((size_t)nlist * dim * sizeof(float));
	for (uint32_t c = 0; c < nlist; c++)
		mkt_rabitq_rotate(
				rq_params,
				leaf_cents + (size_t)c * dim,
				pt_centroids + (size_t)c * dim);

	char *my_partials =
			(dsm_partials != NULL)
					? mkt_dsm_worker_partials(dsm_partials, nlist, worker_id)
					: NULL;

	MktMemCtx worker_ctx = mkt_memctx_create(NULL, "mktann worker posting");
	MktMemCtx prev		 = mkt_memctx_switch(worker_ctx);

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
			my_partials);

	/* Stream each completed full page to the leader over the queue
	 * instead of accumulating it; only the trailing partial per cluster
	 * is retained (copied to the partials DSM buffer by worker_finish). */
	mkt_posting_worker_set_page_sink(&ws, mktann_posting_page_sink, qhandle);

	MktBuildWorkerBufs bufs = mkt_build_worker_bufs_create(dim);

	mkt_memctx_switch(prev);

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
			.tmp_ctx	= mkt_memctx_create(NULL, "mktann parallel tuple"),
			.worker_ctx = worker_ctx,
	};

	MktMemCtx batch_ctx = mkt_memctx_switch(worker_ctx);
	posting_cb_batch_init(&cbs);
	mkt_memctx_switch(batch_ctx);

	/* Second parallel scan for posting build */
	mkt_build_scan(
			heapRel,
			indexRel,
			indexInfo,
			shared,
			true,
			false,
			posting_cb,
			&cbs);

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
	bool *wa = mkt_dsm_worker_active(worker_output, nlist, worker_id);
	memcpy(wa, ws.active, nlist * sizeof(bool));

	mkt_pbuild_worker_add_counts(shared, cbs.indtuples, cbs.soar_dupes);

	mkt_posting_worker_cleanup(&ws);
	mkt_build_worker_bufs_free(&bufs);
	mkt_free(pt_centroids);

	mkt_pbuild_worker_detach(toc, &w);
}
