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
#include "mkt_pg.h"
#include "mkt_vector.h"
#include "mktann_parallel.h"
#include "quant/rabitq.h"

/* ----------------------------------------------------------------
 * Phase 1: Sampling callback
 *
 * Collects vectors into a per-worker sample slot and picks
 * initial centroids from the first vectors seen.
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
 * Phase 2b: Filtered k-means assignment for child k-means
 *
 * Same as mktann_km_assign_and_accumulate but skips samples
 * whose root assignment doesn't match target_child.
 * ---------------------------------------------------------------- */

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
	float cost = 0.0f;

	for (uint32_t i = 0; i < nsamples; i++)
	{
		if (root_assignments[i] != target_child)
			continue;

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
 * Phase 3: Entry build callback
 *
 * Workers encode vectors and write compact entries to a DSM
 * buffer. The leader reads entries after all workers finish and
 * feeds them to serial posting builders. This avoids workers
 * writing to the index relation's buffer pool.
 * ---------------------------------------------------------------- */

static void
write_dsm_entry(
		EntryBuildCbState *cbs,
		uint32_t		   cluster_id,
		ItemPointerData	   tid,
		const float		  *vec)
{
	VectorRef evref = {.data = vec, .dim = cbs->dim};
	VectorRef cref =
			{.data = cbs->leaf_centroids + (size_t)cluster_id * cbs->dim,
			 .dim  = cbs->dim};
	mkt_rabitq_encode_into_ex(
			cbs->params, evref, cref, cbs->enc_buf, &cbs->enc_scratch);

	float f_error = mkt_posting_derive_f_error(
			cbs->enc_buf->f_add, cbs->enc_buf->f_rescale, cbs->dim);

	uint32_t idx = pg_atomic_fetch_add_u32(cbs->count, 1);
	if (idx >= cbs->max_entries)
		return;

	char			  *slot = cbs->entry_buf + (size_t)idx * cbs->entry_size;
	MktDsmEntryHeader *hdr	= (MktDsmEntryHeader *)slot;
	hdr->cluster_id			= cluster_id;
	hdr->tid				= tid;
	hdr->f_add				= cbs->enc_buf->f_add;
	hdr->f_rescale			= cbs->enc_buf->f_rescale;
	hdr->f_error			= f_error;
	memcpy(slot + sizeof(MktDsmEntryHeader),
		   cbs->enc_buf->bits,
		   MKT_RABITQ_BYTES(cbs->dim));
}

void
entry_build_callback(
		Relation	index,
		ItemPointer tid,
		Datum	   *values,
		bool	   *isnull,
		bool		tuple_is_alive,
		void	   *state)
{
	EntryBuildCbState *cbs = (EntryBuildCbState *)state;

	(void)index;
	(void)tuple_is_alive;

	if (isnull[0])
		return;

	MemoryContext old_ctx = MemoryContextSwitchTo(cbs->tmp_ctx);

	MktVector *vec	= DatumGetMktVector(values[0]);
	VectorRef  vref = MktVectorToRef(vec);

	MktBuildAssignment asgn = mkt_build_assign_vector(
			cbs->tree, vref.data, &cbs->bp, &cbs->bufs);

	write_dsm_entry(cbs, asgn.primary, *tid, asgn.enc_vector);
	if (asgn.secondary != MKT_INVALID_CLUSTER)
		write_dsm_entry(cbs, asgn.secondary, *tid, asgn.enc_vector);

	MemoryContextSwitchTo(old_ctx);

	cbs->indtuples++;
	if (asgn.secondary != MKT_INVALID_CLUSTER)
		cbs->soar_dupes++;

	MemoryContextReset(cbs->tmp_ctx);
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

	InstrStartParallelQuery();

	/* ---- Phase 1: Sampling ---- */

	MktDsmSamples *dsm_samples =
			shm_toc_lookup(toc, MKTANN_KEY_SAMPLES, false);
	char  *centroids_base = shm_toc_lookup(toc, MKTANN_KEY_CENTROIDS, false);
	float *cents		  = mktann_centroids(centroids_base);

	/* Each worker picks km_k/N initial centroids */
	uint32_t km_k	   = shared->km_k;
	uint32_t cents_per = km_k / shared->nparticipants;
	uint32_t cents_rem = km_k % shared->nparticipants;
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
			mktann_sample_callback,
			&sc,
			scan);

	*mktann_sample_counts(dsm_samples) = sc.count;
	/* Fix: write to this worker's slot */
	mktann_sample_counts(dsm_samples)[worker_id] = sc.count;

	/* Barrier: all workers done sampling + centroid init */
	BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

	/* ---- Phase 2: K-means iterate (root level, k=km_k) ---- */

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

	for (uint32_t i = 0; i < my_nsamples; i++)
	{
		const float *vec	= my_samples + (size_t)i * dim;
		float		 best_d = __FLT_MAX__;
		uint32_t	 best_c = 0;

		for (uint32_t c = 0; c < km_k; c++)
		{
			const float *cent = cents + (size_t)c * dim;
			float		 d;

			switch (shared->metric)
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

		my_root_asgn[i] = best_c;
	}

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

	/* Barrier: leader built tree, set up posting reserve */
	BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

	/* ---- Phase 3: Posting scan ---- */

	(void)shared->nlist; /* nlist re-read by entry callback */

	HKMeansResult *tree = shm_toc_lookup(toc, MKTANN_KEY_TREE, false);
	MktDsmEntries *dsm_entries =
			shm_toc_lookup(toc, MKTANN_KEY_ENTRIES, false);

	RaBitQParams *rq_params = mkt_rabitq_create(dim, shared->rabitq_seed);

	const float *leaf_cents = hk_leaf_centroids(tree);

	MktBuildWorkerBufs bufs = mkt_build_worker_bufs_create(dim);

	EntryBuildCbState cbs = {
			.tree			= tree,
			.bp				= {.dim				 = dim,
							   .metric			 = shared->metric,
							   .soar_lambda		 = shared->soar_lambda,
							   .boundary_epsilon = shared->boundary_epsilon},
			.bufs			= bufs,
			.entry_buf		= mktann_worker_entries(dsm_entries, worker_id),
			.count			= &mktann_entry_counts(dsm_entries)[worker_id],
			.max_entries	= dsm_entries->max_per_worker,
			.entry_size		= dsm_entries->entry_size,
			.params			= rq_params,
			.leaf_centroids = leaf_cents,
			.dim			= dim,
			.enc_buf		= palloc(MKT_RABITQ_DATA_SIZE(dim)),
			.indtuples		= 0,
			.soar_dupes		= 0,
			.tmp_ctx		= AllocSetContextCreate(
					   CurrentMemoryContext,
					   "mktann parallel tuple",
					   ALLOCSET_DEFAULT_SIZES),
	};
	mkt_rabitq_scratch_init(&cbs.enc_scratch, dim);

	/* Second parallel scan for posting build */
	TableScanDesc scan2 = table_beginscan_parallel(
			heapRel, ParallelTableScanFromMktShared(shared));

	table_index_build_scan(
			heapRel,
			indexRel,
			indexInfo,
			true,
			false,
			entry_build_callback,
			&cbs,
			scan2);

	mkt_rabitq_scratch_cleanup(&cbs.enc_scratch);

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

	mkt_build_worker_bufs_free(&bufs);

	index_close(indexRel, AccessExclusiveLock);
	table_close(heapRel, ShareLock);
}
