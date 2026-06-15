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
#include "standalone/parallel_ctx.h" /* ParallelWorkerNumber */
#include "standalone/pg_compat.h"
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

/* ----------------------------------------------------------------
 * Phase 2c: work-partitioned subtree build (shared kernel)
 *
 * See the contract on the declaration in parallel_build.h. Each participant
 * builds the full subtree for the root children it owns and writes the
 * contiguous HKMeansResult into that child's DSM slot; the leader grafts them.
 * This generalizes the build to arbitrary depth — the subtree is however many
 * levels nlist/fan_out needs — so the parallel path is not capped at two
 * levels.
 * ---------------------------------------------------------------- */

void
mkt_subtree_build_partitioned(
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
		uint32_t		  km_nredo,
		char			 *subtrees_base,
		uint64_t		  slot_size)
{
	/* Leaves each subtree targets, so its depth is the global depth minus the
	 * root level (all subtrees share it, keeping the grafted tree uniform). */
	uint32_t nlist_c = (nlist + fan_out - 1) / fan_out;

	/* Per-child sample counts across all participants (one pass). */
	uint32_t *child_count = mkt_alloc0((size_t)km_k * sizeof(uint32_t));
	for (int t = 0; t < nparticipants; t++)
	{
		const uint32_t *ra = mkt_dsm_root_assignments(dsm_ra, t);
		uint32_t		n  = mkt_dsm_sample_counts(dsm_samples)[t];
		for (uint32_t i = 0; i < n; i++)
			if (ra[i] < km_k)
				child_count[ra[i]]++;
	}

	KMeansOptions opts	= MKT_KMEANS_OPTIONS_DEFAULT;
	opts.max_iterations = km_max_iterations;
	/* Restarts: each leaf subtree's k-means runs km_nredo independent
	 * attempts (different seeds) and keeps the lowest-cost one — escapes
	 * bad local minima for tighter clusters / better routing. */
	opts.nredo = km_nredo > 0 ? km_nredo : 1;
	/*
	 * Per-child problems are small (~nsamples/fan_out points, k=fan_out) and
	 * run on every participant at once. The Lloyd dot-product kernel beats the
	 * CBLAS sgemm path here — sgemm's per-call overhead dominates at this
	 * size, and avoiding BLAS keeps the concurrent per-participant builds off
	 * a shared library. This is the kernel the pre-subtree child phase used.
	 */
	opts.algorithm = KMEANS_ALGO_LLOYD;

	for (uint32_t child = (uint32_t)participant_id; child < km_k;
		 child += (uint32_t)nparticipants)
	{
		uint32_t cc	  = child_count[child];
		char	*slot = mkt_dsm_child_subtree(subtrees_base, child, slot_size);

		HKMeansResult *sub = NULL;
		opts.initial_centroids =
				NULL; /* default seeding, as the serial path */
		if (cc == 0)
		{
			/* Empty root cluster (k-means reseeding makes this effectively
			 * impossible at nlevels >= 2). Seed the subtree with the root
			 * centroid as a single sample so it still has the same depth as
			 * its siblings and the graft stays uniform. */
			float *seed = mkt_alloc((size_t)dim * sizeof(float));
			memcpy(seed,
				   root_cents + (size_t)child * dim,
				   (size_t)dim * sizeof(float));
			sub = mkt_hkmeans_f32(
					seed, 1, NULL, dim, nlist_c, fan_out, metric, &opts);
			mkt_free(seed);
		}
		else
		{
			/* Index this child's samples in place instead of copying them
			 * into a contiguous [cc * dim] buffer: cc is ~total_samples /
			 * fan_out, which exceeds MaxAllocSize at high nlist. The
			 * per-participant DSM sample blocks form one flat array, so a
			 * sample's global slot is t * max_per_worker + i; hkmeans
			 * clusters via indirect access (cc uint32 indices, not cc
			 * full vectors). */
			const float *vbase = mkt_dsm_worker_samples(dsm_samples, 0);
			uint32_t	 mpw   = dsm_samples->max_per_worker;
			uint32_t	*idx   = mkt_alloc((size_t)cc * sizeof(uint32_t));
			uint32_t	 g	   = 0;
			for (int t = 0; t < nparticipants; t++)
			{
				const uint32_t *ra = mkt_dsm_root_assignments(dsm_ra, t);
				uint32_t		n  = mkt_dsm_sample_counts(dsm_samples)[t];
				for (uint32_t i = 0; i < n; i++)
					if (ra[i] == child)
						idx[g++] = (uint32_t)t * mpw + i;
			}
			sub = mkt_hkmeans_f32(
					vbase, cc, idx, dim, nlist_c, fan_out, metric, &opts);
			mkt_free(idx);
		}

		if (sub != NULL)
		{
			if ((uint64_t)sub->total_size > slot_size)
				mkt_error(
						"mktann: subtree blob %u exceeds slot (%u > %lu)",
						child,
						sub->total_size,
						(unsigned long)slot_size);
			memcpy(slot, sub, sub->total_size);
			mkt_free(sub);
		}
	}

	mkt_free(child_count);
}

/* ----------------------------------------------------------------
 * Phases 1, 2, 2b: per-participant execution, shared by leader and workers
 *
 * The leader runs as participant 0 and calls these exactly as a worker does,
 * rather than duplicating the bodies. The phase barriers live inside each
 * function, so the leader and workers stay in lockstep by construction; the
 * leader-only coordination (seeding the initial centroids, reducing the
 * per-iteration accumulators) is gated on participant_id == 0 and runs at the
 * barrier rendezvous while the other participants wait.
 * ---------------------------------------------------------------- */

void
mkt_pbuild_exec_sampling(
		int				  participant_id,
		Relation		  heap,
		Relation		  index,
		struct IndexInfo *index_info,
		MktBuildShared	 *shared,
		MktDsmSamples	 *dsm_samples,
		Barrier			 *barrier)
{
	Dimension dim = shared->dim;

	/* Stride to subsample ~max_samples_per_worker from this participant's
	 * share of the heap. */
	double	 est_rows = RelationGetNumberOfBlocks(heap) *
						(BLCKSZ / (double)(dim * sizeof(float) + 32));
	double	 est_per  = est_rows / shared->nparticipants;
	uint32_t stride	  = 1;
	if (est_per > shared->max_samples_per_worker)
		stride = (uint32_t)(est_per / shared->max_samples_per_worker);
	if (stride < 1)
		stride = 1;

	SampleCbState sc = {
			.samples	 = mkt_dsm_worker_samples(dsm_samples, participant_id),
			.count		 = 0,
			.max_samples = shared->max_samples_per_worker,
			.stride		 = stride,
			.stride_counter = 0,
			.dim			= dim,
			.metric			= shared->metric,
	};

	/* progress: only the leader (participant 0) drives the PG progress view.
	 */
	mkt_build_scan(
			heap,
			index,
			index_info,
			shared,
			true,
			participant_id == 0,
			mkt_sample_cb,
			&sc);

	mkt_dsm_sample_counts(dsm_samples)[participant_id] = sc.count;

	/* Barrier: all participants done sampling. */
	BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);
}

uint32_t
mkt_pbuild_exec_kmeans(
		int				participant_id,
		MktBuildShared *shared,
		MktDsmSamples  *dsm_samples,
		char		   *centroids_base,
		char		   *km_workers_base,
		Barrier		   *barrier)
{
	Dimension dim			= shared->dim;
	uint32_t  km_k			= shared->km_k;
	int		  nparticipants = shared->nparticipants;
	float	 *cents			= mkt_dsm_centroids(centroids_base);
	float	 *norms_c		= mkt_dsm_norms_c(centroids_base, km_k, dim);

	/*
	 * Leader-only seed: pick km_k initial centroids from the pooled samples,
	 * spread evenly across the concatenation of every participant's slot.
	 * Centralizing it (rather than slicing by participant) keeps init
	 * independent of how many workers launched and robust to a sample-starved
	 * one. The result varies run to run (work-stealing scan order), so the
	 * parallel build is not bit-reproducible.
	 */
	if (participant_id == 0)
	{
		uint32_t total_ns = 0;
		for (int t = 0; t < nparticipants; t++)
			total_ns += mkt_dsm_sample_counts(dsm_samples)[t];

		uint32_t step = (total_ns >= km_k) ? total_ns / km_k : 1;
		for (uint32_t i = 0; i < km_k; i++)
		{
			uint32_t gidx = (total_ns > 0) ? (i * step) % total_ns : 0;

			int		 t	  = 0;
			uint32_t base = 0;
			while (t < nparticipants &&
				   base + mkt_dsm_sample_counts(dsm_samples)[t] <= gidx)
			{
				base += mkt_dsm_sample_counts(dsm_samples)[t];
				t++;
			}
			if (t < nparticipants)
				memcpy(cents + (size_t)i * dim,
					   mkt_dsm_worker_samples(dsm_samples, t) +
							   (size_t)(gidx - base) * dim,
					   dim * sizeof(float));
		}

		if (shared->metric == DISTANCE_L2)
			for (uint32_t j = 0; j < km_k; j++)
				norms_c[j] = mkt_l2_norm_squared(cents + (size_t)j * dim, dim);
	}

	/* Barrier: initial centroids + norms ready; guards the reads below against
	 * the leader's seed write. */
	BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

	float	*my_samples = mkt_dsm_worker_samples(dsm_samples, participant_id);
	uint32_t my_n		= mkt_dsm_sample_counts(dsm_samples)[participant_id];
	float	*my_sums =
			mkt_dsm_km_worker_sums(km_workers_base, km_k, dim, participant_id);
	uint32_t *my_cnts =
			mkt_dsm_km_worker_cnts(km_workers_base, km_k, dim, participant_id);
	float *my_cost =
			mkt_dsm_km_worker_cost(km_workers_base, km_k, dim, participant_id);

	/* Reduce scratch is leader-only. */
	float *old_cents = participant_id == 0
							 ? mkt_alloc((size_t)km_k * dim * sizeof(float))
							 : NULL;
	Size   km_sz	 = mkt_dsm_km_workers_size(nparticipants, km_k, dim);

	uint32_t iters = 0;
	for (uint32_t iter = 0; iter < shared->km_max_iterations; iter++)
	{
		iters++;

		/* Every participant assigns + accumulates over its own samples. */
		mkt_km_assign_and_accumulate(
				my_samples,
				my_n,
				cents,
				norms_c,
				km_k,
				dim,
				shared->metric,
				my_sums,
				my_cnts,
				my_cost);

		/* Barrier: all accumulators written; the leader reduces. */
		BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

		if (participant_id == 0)
		{
			memcpy(old_cents, cents, (size_t)km_k * dim * sizeof(float));

			const float **all_sums = mkt_alloc(
					nparticipants * sizeof(float *));
			const uint32_t **all_cnts = mkt_alloc(
					nparticipants * sizeof(uint32_t *));
			float *all_costs = mkt_alloc(nparticipants * sizeof(float));
			for (int t = 0; t < nparticipants; t++)
			{
				all_sums[t] =
						mkt_dsm_km_worker_sums(km_workers_base, km_k, dim, t);
				all_cnts[t] =
						mkt_dsm_km_worker_cnts(km_workers_base, km_k, dim, t);
				all_costs[t] =
						*mkt_dsm_km_worker_cost(km_workers_base, km_k, dim, t);
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

			mkt_free(all_sums);
			mkt_free(all_cnts);
			mkt_free(all_costs);
		}

		/* Barrier: updated centroids + convergence flag visible to all. */
		BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

		if (shared->km_converged)
			break;
	}

	if (old_cents != NULL)
		mkt_free(old_cents);

	return iters;
}

void
mkt_pbuild_exec_root_assign(
		int				  participant_id,
		MktBuildShared	 *shared,
		MktDsmSamples	 *dsm_samples,
		MktDsmRootAssign *dsm_ra,
		char			 *centroids_base,
		Barrier			 *barrier)
{
	Dimension dim	  = shared->dim;
	uint32_t  km_k	  = shared->km_k;
	float	 *cents	  = mkt_dsm_centroids(centroids_base);
	float	 *norms_c = mkt_dsm_norms_c(centroids_base, km_k, dim);

	kmeans_assign(
			mkt_dsm_worker_samples(dsm_samples, participant_id),
			0,
			mkt_dsm_sample_counts(dsm_samples)[participant_id],
			cents,
			norms_c,
			km_k,
			dim,
			shared->metric,
			mkt_dsm_root_assignments(dsm_ra, participant_id));

	/* Barrier: all root assignments written; subtree gather can read them. */
	BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);
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

	/* ---- Phases 1, 2, 2b: the shared per-participant bodies (the leader runs
	 * the very same code as participant 0). Each call includes its phase
	 * barrier(s). ---- */
	MktDsmSamples *dsm_samples =
			shm_toc_lookup(toc, MKT_DSM_KEY_SAMPLES, false);
	char  *centroids_base = shm_toc_lookup(toc, MKT_DSM_KEY_CENTROIDS, false);
	float *cents		  = mkt_dsm_centroids(centroids_base);
	char *km_workers_base = shm_toc_lookup(toc, MKT_DSM_KEY_KM_WORKERS, false);
	MktDsmRootAssign *dsm_ra =
			shm_toc_lookup(toc, MKT_DSM_KEY_ROOT_ASSIGN, false);
	uint32_t   km_k		 = shared->km_k;
	IndexInfo *indexInfo = BuildIndexInfo(indexRel);

	mkt_pbuild_exec_sampling(
			worker_id,
			heapRel,
			indexRel,
			indexInfo,
			shared,
			dsm_samples,
			barrier);
	mkt_pbuild_exec_kmeans(
			worker_id,
			shared,
			dsm_samples,
			centroids_base,
			km_workers_base,
			barrier);
	mkt_pbuild_exec_root_assign(
			worker_id, shared, dsm_samples, dsm_ra, centroids_base, barrier);

	/* ---- Phase 2c: Child subtrees (work-partitioned, barrier-free) ----
	 *
	 * For a hierarchical tree (>= 2 levels) each participant builds the full
	 * subtree for the root children it owns. A flat (1-level) build has no
	 * children — the leader clusters it serially and we just meet the barrier.
	 * The participant computes nlevels itself so it agrees with the leader. */
	if (mkt_compute_nlevels(shared->nlist, shared->fan_out) >= 2)
	{
		char *subtrees_base =
				shm_toc_lookup(toc, MKT_DSM_KEY_CHILD_SUBTREES, false);
		mkt_subtree_build_partitioned(
				worker_id,
				shared->nparticipants,
				dsm_samples,
				dsm_ra,
				cents,
				km_k,
				shared->nlist,
				shared->fan_out,
				dim,
				shared->metric,
				shared->km_max_iterations,
				shared->km_nredo,
				subtrees_base,
				shared->subtree_slot_size);
	}

	/* Barrier: all participants done building subtrees; the leader grafts the
	 * tree next. */
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

	RaBitQParams *rq_params = mkt_rabitq_create(dim, shared->rabitq_seed);

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

	/* Reclaim the two top-level contexts holding this worker's posting state.
	 * The arena's mkt_free is a no-op (memory is released on context delete),
	 * so the per-tuple temp context and the worker posting context — both
	 * created with no parent above — must be deleted explicitly or every
	 * worker's posting buffers leak. */
	mkt_memctx_delete(cbs.tmp_ctx);
	mkt_memctx_delete(worker_ctx);

	mkt_pbuild_worker_detach(toc, &w);
}
