/*
 * mktann_parallel.c - PG parallel worker for mktann index build
 *
 * Multi-phase parallel build:
 *   Phase 1 (sampling): cooperative heap scan, each worker collects
 *     samples and picks initial centroids.
 *   Phase 2 (k-means): iterative assignment + accumulation on
 *     per-worker samples, barrier-synchronized with leader reduce.
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
#include "algo/vecops.h"
#include "index/posting_build.h"
#include "index/posting_build_parallel.h"
#include "mkt_pg.h"
#include "mkt_vector.h"
#include "mktann_parallel.h"
#include "mktann_storage.h"
#include "quant/rabitq.h"

/* ----------------------------------------------------------------
 * Phase 1: Sampling callback
 *
 * Collects vectors into a per-worker sample slot and picks
 * initial centroids from the first vectors seen.
 * ---------------------------------------------------------------- */

typedef struct SampleCbState
{
	float		  *samples;
	float		  *centroids;
	uint32_t	   count;
	uint32_t	   max_samples;
	uint32_t	   stride;
	uint32_t	   stride_counter;
	Dimension	   dim;
	DistanceMetric metric;

	/* Centroid init: this worker picks centroids [cent_start, cent_end) */
	uint32_t cent_start;
	uint32_t cent_end;
	uint32_t cents_picked;
} SampleCbState;

static void
sample_callback(
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

	/* Pick initial centroids from first vectors */
	uint32_t cent_idx = sc->cent_start + sc->cents_picked;
	if (cent_idx < sc->cent_end)
	{
		memcpy(sc->centroids + (size_t)cent_idx * dim,
			   dest,
			   dim * sizeof(float));
		sc->cents_picked++;
	}

	sc->count++;
}

/* ----------------------------------------------------------------
 * Phase 2: K-means assignment on per-worker samples
 *
 * Each worker computes distances from its samples to all
 * centroids and accumulates per-worker centroid sums.
 * ---------------------------------------------------------------- */

static void
km_assign_and_accumulate(
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
	float cost = 0.0f;

	for (uint32_t i = 0; i < nsamples; i++)
	{
		const float *vec	= samples + (size_t)i * dim;
		float		 best_d = __FLT_MAX__;
		uint32_t	 best_c = 0;

		for (uint32_t c = 0; c < nlist; c++)
		{
			const float *cent = centroids + (size_t)c * dim;
			float		 d;

			switch (metric)
			{
			case DISTANCE_L2:
			{
				float norm_x = mkt_l2_norm_squared(vec, dim);
				float dot	 = mkt_dot_product(vec, cent, dim);
				d			 = norm_x + norms_c[c] - 2.0f * dot;
				if (d < 0.0f)
					d = 0.0f;
				break;
			}
			case DISTANCE_INNER_PRODUCT:
				d = -mkt_dot_product(vec, cent, dim);
				break;
			case DISTANCE_COSINE:
				d = 1.0f - mkt_dot_product(vec, cent, dim);
				break;
			}

			if (d < best_d)
			{
				best_d = d;
				best_c = c;
			}
		}

		cost += best_d;
		out_cnts[best_c]++;
		float *sum = out_sums + (size_t)best_c * dim;
		for (uint32_t d = 0; d < dim; d++)
			sum[d] += vec[d];
	}

	*out_cost = cost;
}

/* ----------------------------------------------------------------
 * Phase 3: Posting build callback
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
} PostingCbState;

static void
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

	MemoryContext old_ctx = MemoryContextSwitchTo(cbs->tmp_ctx);

	MktVector *vec	= DatumGetMktVector(values[0]);
	VectorRef  vref = MktVectorToRef(vec);

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

/* ----------------------------------------------------------------
 * DSM reserve → local MktPostingReserve adapter
 * ---------------------------------------------------------------- */

static void
dsm_reserve_to_local(MktDsmReserve *dsm, MktPostingReserve *local)
{
	local->starts = mktann_dsm_reserve_starts(dsm);
	local->counts = mktann_dsm_reserve_counts(dsm);
	local->nexts  = (mkt_atomic_uint32 *)mktann_dsm_reserve_nexts(dsm);
	local->nlist  = dsm->nlist;
	local->total  = dsm->total_reserved;
}

/* ----------------------------------------------------------------
 * Worker entry point — multi-phase build
 * ---------------------------------------------------------------- */

void
mktann_parallel_build_main(dsm_segment *seg, shm_toc *toc)
{
	(void)seg;

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
	uint32_t  nlist		= shared->nlist;

	InstrStartParallelQuery();

	/* ---- Phase 1: Sampling ---- */

	MktDsmSamples *dsm_samples =
			shm_toc_lookup(toc, MKTANN_KEY_SAMPLES, false);
	char  *centroids_base = shm_toc_lookup(toc, MKTANN_KEY_CENTROIDS, false);
	float *cents		  = mktann_centroids(centroids_base);

	/* Each worker picks K/N initial centroids */
	uint32_t cents_per = nlist / shared->nparticipants;
	uint32_t cents_rem = nlist % shared->nparticipants;
	uint32_t cent_start, cent_end;
	if ((uint32_t)worker_id < cents_rem)
	{
		cent_start = (uint32_t)worker_id * (cents_per + 1);
		cent_end   = cent_start + cents_per + 1;
	}
	else
	{
		cent_start = cents_rem * (cents_per + 1) +
					 ((uint32_t)worker_id - cents_rem) * cents_per;
		cent_end = cent_start + cents_per;
	}

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
			.centroids		= cents,
			.count			= 0,
			.max_samples	= shared->max_samples_per_worker,
			.stride			= stride,
			.stride_counter = 0,
			.dim			= dim,
			.metric			= shared->metric,
			.cent_start		= cent_start,
			.cent_end		= cent_end,
			.cents_picked	= 0,
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
			sample_callback,
			&sc,
			scan);

	*mktann_sample_counts(dsm_samples) = sc.count;
	/* Fix: write to this worker's slot */
	mktann_sample_counts(dsm_samples)[worker_id] = sc.count;

	/* Barrier: all workers done sampling + centroid init */
	BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

	/* ---- Phase 2: K-means iterate ---- */

	char *km_workers_base = shm_toc_lookup(toc, MKTANN_KEY_KM_WORKERS, false);

	float	*my_samples	 = mktann_worker_samples(dsm_samples, worker_id);
	uint32_t my_nsamples = mktann_sample_counts(dsm_samples)[worker_id];

	float *norms_c = mktann_norms_c(centroids_base, nlist, dim);
	float *my_sums =
			mktann_km_worker_sums(km_workers_base, nlist, dim, worker_id);
	uint32_t *my_cnts =
			mktann_km_worker_cnts(km_workers_base, nlist, dim, worker_id);
	float *my_cost =
			mktann_km_worker_cost(km_workers_base, nlist, dim, worker_id);

	for (uint32_t iter = 0; iter < shared->km_max_iterations; iter++)
	{
		/* Assignment + accumulation on this worker's samples */
		km_assign_and_accumulate(
				my_samples,
				my_nsamples,
				cents,
				norms_c,
				nlist,
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

	/* Barrier: leader built tree, set up posting reserve */
	BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

	/* ---- Phase 3: Posting scan ---- */

	HKMeansResult *tree = shm_toc_lookup(toc, MKTANN_KEY_TREE, false);
	MktDsmReserve *dsm_reserve =
			shm_toc_lookup(toc, MKTANN_KEY_RESERVE, false);
	char *worker_output_base =
			shm_toc_lookup(toc, MKTANN_KEY_WORKER_OUTPUT, false);
	char *partials_base = shm_toc_lookup(toc, MKTANN_KEY_PARTIALS, true);

	RaBitQParams *rq_params = mkt_rabitq_create(dim, shared->rabitq_seed);

	MktannStorage storage;
	mktann_storage_init(&storage, indexRel, NULL, shared->metric);
	storage.build_mode = true;

	MktPostingReserve reserve;
	dsm_reserve_to_local(dsm_reserve, &reserve);

	char *my_partials =
			(partials_base != NULL)
					? mktann_worker_partials(partials_base, nlist, worker_id)
					: NULL;

	const float *leaf_cents = hk_leaf_centroids(tree);

	float *pt_centroids = palloc((size_t)nlist * dim * sizeof(float));
	for (uint32_t c = 0; c < nlist; c++)
		mkt_rabitq_rotate(
				rq_params,
				leaf_cents + (size_t)c * dim,
				pt_centroids + (size_t)c * dim);

	MemoryContext worker_ctx = AllocSetContextCreate(
			CurrentMemoryContext,
			"mktann worker posting",
			ALLOCSET_DEFAULT_SIZES);
	MemoryContext prev = MemoryContextSwitchTo(worker_ctx);

	MktPostingWorkerState ws;
	mkt_posting_worker_init(
			&ws,
			worker_id,
			nlist,
			dim,
			shared->fastscan,
			&storage.base,
			rq_params,
			leaf_cents,
			pt_centroids,
			&reserve,
			my_partials);

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

	mkt_posting_worker_finish(&ws);

	BlockNumber *wh =
			mktann_worker_heads(worker_output_base, nlist, worker_id);
	BlockNumber *wt =
			mktann_worker_tails(worker_output_base, nlist, worker_id);
	bool *wa = mktann_worker_active(worker_output_base, nlist, worker_id);
	memcpy(wh, ws.heads, nlist * sizeof(BlockNumber));
	memcpy(wt, ws.tails, nlist * sizeof(BlockNumber));
	memcpy(wa, ws.active, nlist * sizeof(bool));

	SpinLockAcquire(&shared->mutex);
	shared->nparticipantsdone++;
	shared->indtuples += cbs.indtuples;
	shared->soar_dupes += cbs.soar_dupes;
	SpinLockRelease(&shared->mutex);
	ConditionVariableSignal(&shared->workersdonecv);

	BufferUsage *bufferusage =
			shm_toc_lookup(toc, MKTANN_KEY_BUFFER_USAGE, false);
	WalUsage *walusage = shm_toc_lookup(toc, MKTANN_KEY_WAL_USAGE, false);
	InstrEndParallelQuery(
			&bufferusage[ParallelWorkerNumber],
			&walusage[ParallelWorkerNumber]);

	MemoryContextDelete(worker_ctx);
	pfree(pt_centroids);

	index_close(indexRel, AccessExclusiveLock);
	table_close(heapRel, ShareLock);
}
