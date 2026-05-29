/*
 * mktann_build.c - Index build for mktann
 *
 * Build phases:
 *   1. Determine dimension from index column typmod
 *   2. Resolve distance metric, centroid format, fan_out from relopts
 *   3. Sample vectors for clustering (BlockSampler + reservoir)
 *   4. Run hierarchical k-means (mkt_hkmeans_f32)
 *   5. Compute global mean from leaf centroids
 *
 * Single-pass streaming build:
 *   6. Write metadata page, reserve centroid blocks
 *   7. Single heap scan: assign via tree descent, stream into
 *      posting builders
 *   8. Finish builders, write centroid pages with posting heads
 *   9. Update metadata with tuple count
 *  10. WAL-log all pages
 *
 * Block layout:
 *   Block 0:        Metadata page
 *   Blocks 1..C:    Centroid pages (reserved, written after scan)
 *   Blocks C+1..N:  Posting pages (streamed during scan)
 *
 * Memory layout:
 *   build_ctx  — all build-phase allocations; deleted in one shot
 *     tmp_ctx  — per-tuple scratch; reset after each callback
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
#include "mktann_parallel.h"
#include "mktann_storage.h"
#include "quant/rabitq.h"

/* ----------------------------------------------------------------
 * Build state
 * ---------------------------------------------------------------- */

typedef struct MktannBuildParams
{
	Dimension		  dim;
	DistanceMetric	  metric;
	MktCentroidFormat centroid_format;
	uint32_t		  nlist;
	uint32_t		  fan_out;
	uint32_t		  kmeans_nredo;
	double			  soar_lambda;
	double			  boundary_epsilon;
	bool			  fastscan;
} MktannBuildParams;

typedef struct MktannBuildState
{
	MktannBuildParams params;

	/* Tree for centroid assignment */
	const HKMeansResult *tree;

	double indtuples;  /* total count */
	double soar_dupes; /* replicated SOAR vectors */

	/* Posting list builders (initialized before scan) */
	MktPostingBuilder *builders; /* [nlist] */

	/* Per-worker scratch buffers for parallel-ready assignment */
	MktBuildWorkerBufs worker_bufs;

	/* Sampling */
	float *samples;		/* [max_samples * dim] row-major */
	int	   nsamples;	/* current sample count */
	int	   max_samples; /* target sample count */
	double rowstoskip;	/* reservoir sampling state */

	ReservoirStateData rstate;

	/* PG context */
	Relation		  heap;
	Relation		  index;
	struct IndexInfo *index_info;
	MemoryContext	  build_ctx; /* all build allocations */
	MemoryContext	  tmp_ctx;	 /* per-tuple scratch */
} MktannBuildState;

/* ----------------------------------------------------------------
 * Helpers
 * ---------------------------------------------------------------- */

static void
normalize_in_place(float *v, Dimension dim)
{
	float norm = mkt_l2_norm(v, dim);
	if (norm > 0.0f)
		mkt_vector_scale(v, 1.0f / norm, v, dim);
}

/* ----------------------------------------------------------------
 * Sampling
 * ---------------------------------------------------------------- */

static void
sample_callback(
		Relation	index,
		ItemPointer tid,
		Datum	   *values,
		bool	   *isnull,
		bool		tuple_is_alive,
		void	   *state)
{
	MktannBuildState *bs = (MktannBuildState *)state;

	(void)index;
	(void)tid;
	(void)tuple_is_alive;

	if (isnull[0])
		return;

	MemoryContext old_ctx = MemoryContextSwitchTo(bs->tmp_ctx);

	MktVector *vec = DatumGetMktVector(values[0]);
	float	  *src = MKT_VECTOR_DATA(vec);
	Dimension  dim = bs->params.dim;

	if (bs->nsamples < bs->max_samples)
	{
		memcpy(bs->samples + (size_t)bs->nsamples * dim,
			   src,
			   dim * sizeof(float));
		bs->nsamples++;
	}
	else
	{
		if (bs->rowstoskip < 0)
			bs->rowstoskip = reservoir_get_next_S(
					&bs->rstate, bs->nsamples, bs->max_samples);

		if (bs->rowstoskip <= 0)
		{
			int k = (int)(bs->max_samples *
						  sampler_random_fract(&bs->rstate.randstate));
			Assert(k >= 0 && k < bs->max_samples);
			memcpy(bs->samples + (size_t)k * dim, src, dim * sizeof(float));
		}
		bs->rowstoskip -= 1;
		bs->nsamples++;
	}

	MemoryContextSwitchTo(old_ctx);
	MemoryContextReset(bs->tmp_ctx);
}

static void
sample_rows(MktannBuildState *bs)
{
	BlockNumber		 totalblocks = RelationGetNumberOfBlocks(bs->heap);
	BlockSamplerData bsampler;

	bs->rowstoskip = -1;

	BlockSampler_Init(
			&bsampler,
			totalblocks,
			bs->max_samples,
			pg_prng_uint32(&pg_global_prng_state));
	reservoir_init_selection_state(&bs->rstate, bs->max_samples);

	while (BlockSampler_HasMore(&bsampler))
	{
		BlockNumber targblock = BlockSampler_Next(&bsampler);

		table_index_build_range_scan(
				bs->heap,
				bs->index,
				bs->index_info,
				false,
				true,
				false,
				targblock,
				1,
				sample_callback,
				(void *)bs,
				NULL);
	}
}

/* ----------------------------------------------------------------
 * Build callback — single-pass: assign via tree descent,
 * stream into posting builders
 * ---------------------------------------------------------------- */

static void
build_callback(
		Relation	index,
		ItemPointer tid,
		Datum	   *values,
		bool	   *isnull,
		bool		tuple_is_alive,
		void	   *state)
{
	MktannBuildState *bs = (MktannBuildState *)state;

	(void)index;
	(void)tuple_is_alive;

	if (isnull[0])
		return;

	MemoryContext old_ctx = MemoryContextSwitchTo(bs->tmp_ctx);

	MktVector *vec	= DatumGetMktVector(values[0]);
	VectorRef  vref = MktVectorToRef(vec);

	const MktBuildParams bp = {
			.dim			  = bs->params.dim,
			.metric			  = bs->params.metric,
			.soar_lambda	  = bs->params.soar_lambda,
			.boundary_epsilon = bs->params.boundary_epsilon,
	};

	MktBuildAssignment asgn = mkt_build_assign_vector(
			bs->tree, vref.data, &bp, &bs->worker_bufs);

	mkt_posting_builder_add(
			&bs->builders[asgn.primary], *tid, asgn.enc_vector);
	bs->indtuples++;

	if (((uint64_t)bs->indtuples % 10000) == 0)
		pgstat_progress_update_param(
				PROGRESS_CREATEIDX_TUPLES_DONE, (int64)bs->indtuples);

	if (asgn.secondary != MKT_INVALID_CLUSTER)
	{
		mkt_posting_builder_add(
				&bs->builders[asgn.secondary], *tid, asgn.enc_vector);
		bs->soar_dupes++;
	}

	MemoryContextSwitchTo(old_ctx);
	MemoryContextReset(bs->tmp_ctx);
}

/* ----------------------------------------------------------------
 * Write metadata page
 * ---------------------------------------------------------------- */

static void
write_meta_page(
		MktStorage		 *storage,
		Dimension		  dim,
		uint8_t			  nlevels,
		uint8_t			  fan_out,
		BlockNumber		  first_centroid,
		uint32_t		  ntuples,
		uint32_t		  nlist,
		MktCentroidFormat centroid_format,
		DistanceMetric	  metric,
		uint64_t		  rabitq_seed,
		const float		 *global_mean)
{
	BlockNumber blkno;
	Page		page = mkt_storage_new_page(storage, &blkno);

	Assert(blkno == 0);

	PageInit(page, BLCKSZ, MKT_META_SIZE(dim));

	MktannMetaPage *meta  = (MktannMetaPage *)PageGetSpecialPointer(page);
	meta->magic			  = MKT_META_MAGIC;
	meta->dim			  = dim;
	meta->nlevels		  = nlevels;
	meta->centroid_format = (uint8_t)centroid_format;
	meta->first_centroid  = first_centroid;
	meta->ntuples		  = ntuples;
	meta->nlist			  = nlist;
	meta->metric		  = (uint8_t)metric;
	meta->fan_out		  = (uint8_t)fan_out;
	meta->flags			  = 0;
	meta->reserved		  = 0;
	meta->rabitq_seed	  = rabitq_seed;

	memcpy(mktann_meta_global_mean(meta), global_mean, dim * sizeof(float));

	mkt_storage_commit_page(storage, blkno);
}

/* ----------------------------------------------------------------
 * Helpers: resolve metric and centroid format from opclass/relopts
 * ---------------------------------------------------------------- */

static DistanceMetric
mktann_get_metric(Relation index)
{
	FmgrInfo *procinfo = index_getprocinfo(index, 1, MKTANN_METRIC_PROC);
	return (DistanceMetric)DatumGetInt32(
			FunctionCall1Coll(procinfo, InvalidOid, (Datum)0));
}

static MktCentroidFormat
mktann_resolve_format(Relation index, DistanceMetric metric)
{
	MktannOptions *opts = (MktannOptions *)index->rd_options;
	bool compressed		= (opts != NULL) ? opts->centroid_compression : false;

	if (compressed)
	{
		if (metric == DISTANCE_INNER_PRODUCT)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("centroid_compression is not supported "
							"with vector_ip_ops")));
		return MKT_CENTROID_FMT_RABITQ;
	}

	Oid col_type = TupleDescAttr(index->rd_att, 0)->atttypid;
	if (col_type == mkt_halfvec_type_oid())
		return MKT_CENTROID_FMT_HALF;
	return MKT_CENTROID_FMT_FLOAT;
}

static uint32_t
mktann_get_fan_out(Relation index)
{
	MktannOptions *opts = (MktannOptions *)index->rd_options;
	if (opts != NULL && opts->fan_out >= MKTANN_MIN_FAN_OUT)
		return (uint32_t)opts->fan_out;
	return MKTANN_DEFAULT_FAN_OUT;
}

static void
resolve_build_params(Relation heap, Relation index, MktannBuildParams *p)
{
	Dimension dim = (Dimension)TupleDescAttr(index->rd_att, 0)->atttypmod;
	if (dim == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("column does not have dimensions")));
	if (dim > MKT_VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("column cannot have more than %d dimensions",
						MKT_VECTOR_MAX_DIM)));

	p->dim			   = dim;
	p->metric		   = mktann_get_metric(index);
	p->centroid_format = mktann_resolve_format(index, p->metric);
	p->fan_out		   = mktann_get_fan_out(index);

	/* nlist: use relopt if set, otherwise auto from sqrt(reltuples) */
	MktannOptions *opts		 = (MktannOptions *)index->rd_options;
	uint32_t	   nlist_opt = (opts != NULL) ? (uint32_t)opts->nlist : 0;

	if (nlist_opt > 0)
	{
		p->nlist = nlist_opt;
	}
	else
	{
		double reltuples = (heap->rd_rel->reltuples > 0)
								 ? heap->rd_rel->reltuples
								 : RelationGetNumberOfBlocks(heap) *
										   (BLCKSZ /
											(sizeof(float) * dim + 32));
		p->nlist		 = (uint32_t)sqrt((double)Max(reltuples, 1));
		if (p->nlist < 1)
			p->nlist = 1;
		if (p->nlist > 10000)
			p->nlist = 10000;
	}

	p->fan_out =
			mkt_auto_fan_out(p->fan_out, p->nlist, MKTANN_DEFAULT_FAN_OUT);

	p->kmeans_nredo = (opts != NULL && opts->kmeans_nredo > 0)
							? (uint32_t)opts->kmeans_nredo
							: 1;

	p->soar_lambda		= (opts != NULL) ? opts->soar_lambda : 0.0;
	p->boundary_epsilon = (opts != NULL) ? opts->boundary_epsilon : 0.0;
	p->fastscan			= (opts != NULL) ? opts->fastscan : false;
}

/* ----------------------------------------------------------------
 * Sample and cluster vectors
 * ---------------------------------------------------------------- */

static HKMeansResult *
run_clustering(MktannBuildState *bs, float **out_global_mean)
{
	Dimension dim	= bs->params.dim;
	uint32_t  nlist = bs->params.nlist;

	bs->max_samples = Max(10000, (int)(nlist * 256));
	{
		size_t max_by_mem = MaxAllocSize / (dim * sizeof(float));
		if ((size_t)bs->max_samples > max_by_mem)
			bs->max_samples = (int)max_by_mem;
	}
	bs->nsamples = 0;
	bs->samples	 = palloc((size_t)bs->max_samples * dim * sizeof(float));

	sample_rows(bs);

	if (bs->nsamples > bs->max_samples)
		bs->nsamples = bs->max_samples;

	if (bs->params.metric == DISTANCE_COSINE)
	{
		for (int i = 0; i < bs->nsamples; i++)
			normalize_in_place(bs->samples + (size_t)i * dim, dim);
	}

	if (bs->nsamples == 0)
		return NULL;

	if ((uint32_t)bs->nsamples < nlist)
		nlist = (uint32_t)bs->nsamples;

	pgstat_progress_update_param(
			PROGRESS_CREATEIDX_SUBPHASE, PROGRESS_MKTANN_PHASE_KMEANS);
	pgstat_progress_update_param(PROGRESS_CREATEIDX_TUPLES_TOTAL, 0);
	pgstat_progress_update_param(PROGRESS_CREATEIDX_TUPLES_DONE, 0);

	KMeansOptions km_opts = MKT_KMEANS_OPTIONS_DEFAULT;
	km_opts.algorithm	  = KMEANS_ALGO_LLOYD;
	km_opts.nredo		  = bs->params.kmeans_nredo;

	HKMeansResult *tree = mkt_hkmeans_f32(
			bs->samples,
			(uint32_t)bs->nsamples,
			NULL,
			dim,
			nlist,
			bs->params.fan_out,
			bs->params.metric,
			&km_opts);

	if (tree == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("hierarchical k-means failed")));

	pfree(bs->samples);
	bs->samples = NULL;

	float *global_mean = palloc(dim * sizeof(float));
	mkt_vector_mean(hk_leaf_centroids(tree), tree->nleaves, dim, global_mean);

	if (bs->params.metric == DISTANCE_COSINE)
		normalize_in_place(global_mean, dim);

	*out_global_mean = global_mean;
	return tree;
}

/* Estimate posting pages per cluster from heap size */
static uint32_t
estimate_posting_pages(Relation heap, Dimension dim, uint32_t nlist)
{
	double est_rows = RelationGetNumberOfBlocks(heap) *
					  (BLCKSZ / (double)(dim * sizeof(float) + 32));
	uint32_t est_per_cluster = (uint32_t)ceil(est_rows / nlist);
	uint32_t per_first		 = mkt_posting_max_entries_first(dim);
	uint32_t per_page		 = mkt_posting_max_entries(dim);
	uint32_t npages			 = 1;
	if (est_per_cluster > per_first)
		npages += (est_per_cluster - per_first + per_page - 1) / per_page;
	return npages;
}

/* ----------------------------------------------------------------
 * Parallel build — leader side
 *
 * Sets up DSM, launches workers, participates as worker_id=0,
 * then merges partial pages after all workers complete.
 *
 * Returns true if parallel build succeeded, false if it fell
 * back (no DSM, no workers launched). Caller runs serial build
 * on false.
 * ---------------------------------------------------------------- */

typedef struct LeaderBuildState
{
	MktannBuildState	  *bs;
	MktPostingWorkerState *ws;
	MktBuildParams		   bp;
	MemoryContext		   worker_ctx;
	double				   indtuples;
	double				   soar_dupes;
} LeaderBuildState;

static void
leader_build_callback(
		Relation	index,
		ItemPointer tid,
		Datum	   *values,
		bool	   *isnull,
		bool		tuple_is_alive,
		void	   *state)
{
	LeaderBuildState *ls = (LeaderBuildState *)state;

	(void)index;
	(void)tuple_is_alive;

	if (isnull[0])
		return;

	MemoryContext old_ctx = MemoryContextSwitchTo(ls->bs->tmp_ctx);

	MktVector *vec	= DatumGetMktVector(values[0]);
	VectorRef  vref = MktVectorToRef(vec);

	MktBuildAssignment asgn = mkt_build_assign_vector(
			ls->bs->tree, vref.data, &ls->bp, &ls->bs->worker_bufs);

	MemoryContextSwitchTo(ls->worker_ctx);

	mkt_posting_worker_add_heap(
			ls->ws, *tid, asgn.enc_vector, asgn.primary, asgn.secondary);

	MemoryContextSwitchTo(old_ctx);

	ls->indtuples++;
	if (asgn.secondary != MKT_INVALID_CLUSTER)
		ls->soar_dupes++;

	MemoryContextReset(ls->bs->tmp_ctx);
}

static bool
do_parallel_build(
		Relation		  heap,
		Relation		  index,
		struct IndexInfo *index_info,
		MktannBuildState *bs,
		MktannStorage	 *storage,
		BlockNumber		  first_posting,
		HKMeansResult	**out_tree,
		BlockNumber		 *posting_heads,
		double			 *out_heap_tuples,
		double			 *out_indtuples,
		double			 *out_soar_dupes)
{
	int		  nworkers		= index_info->ii_ParallelWorkers;
	Dimension dim			= bs->params.dim;
	uint32_t  nlist			= bs->params.nlist;
	int		  nparticipants = nworkers + 1;
	uint64_t  rabitq_seed	= 42;

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
	/* K-means shared centroids + norms */
	shm_toc_estimate_chunk(
			&pcxt->estimator, mktann_centroids_size(nlist, dim));
	/* K-means per-worker accumulators */
	shm_toc_estimate_chunk(
			&pcxt->estimator,
			mktann_km_workers_size(nparticipants, nlist, dim));
	/* Tree blob (placeholder — allocated later by leader, but
	 * we need the max possible size. Use a generous estimate.) */
	Size max_tree_sz = sizeof(HKMeansResult) +
					   (Size)nlist * 2 * sizeof(HKMeansNode) +
					   (Size)nlist * dim * sizeof(float) * 2;
	shm_toc_estimate_chunk(&pcxt->estimator, max_tree_sz);
	/* Posting phase */
	shm_toc_estimate_chunk(&pcxt->estimator, mktann_dsm_reserve_size(nlist));
	shm_toc_estimate_chunk(
			&pcxt->estimator, mktann_worker_output_size(nlist, nparticipants));
	if (!bs->params.fastscan)
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

	int nkeys = 12;
	if (!bs->params.fastscan)
		nkeys++;
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
	shared->metric				   = bs->params.metric;
	shared->nlist				   = nlist;
	shared->fan_out				   = bs->params.fan_out;
	shared->soar_lambda			   = bs->params.soar_lambda;
	shared->boundary_epsilon	   = bs->params.boundary_epsilon;
	shared->fastscan			   = bs->params.fastscan;
	shared->rabitq_seed			   = rabitq_seed;
	shared->nparticipants		   = nparticipants;
	shared->max_samples_per_worker = max_per_worker;
	shared->km_max_iterations	   = 20;
	shared->km_tolerance		   = 1e-4f;
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

	/* Barrier for phase synchronization */
	Barrier *barrier = shm_toc_allocate(pcxt->toc, sizeof(Barrier));
	BarrierInit(barrier, nparticipants);
	shm_toc_insert(pcxt->toc, MKTANN_KEY_BARRIER, barrier);

	/* Sample slots */
	Size samp_sz = mktann_samples_size(nparticipants, max_per_worker, dim);
	MktDsmSamples *dsm_samples = shm_toc_allocate(pcxt->toc, samp_sz);
	memset(dsm_samples, 0, samp_sz);
	dsm_samples->nparticipants	= nparticipants;
	dsm_samples->max_per_worker = max_per_worker;
	dsm_samples->dim			= dim;
	shm_toc_insert(pcxt->toc, MKTANN_KEY_SAMPLES, dsm_samples);

	/* Shared centroids + norms */
	Size  cent_sz		 = mktann_centroids_size(nlist, dim);
	char *centroids_base = shm_toc_allocate(pcxt->toc, cent_sz);
	memset(centroids_base, 0, cent_sz);
	shm_toc_insert(pcxt->toc, MKTANN_KEY_CENTROIDS, centroids_base);
	float *cents   = mktann_centroids(centroids_base);
	float *norms_c = mktann_norms_c(centroids_base, nlist, dim);

	/* Per-worker k-means accumulators */
	Size  km_sz			  = mktann_km_workers_size(nparticipants, nlist, dim);
	char *km_workers_base = shm_toc_allocate(pcxt->toc, km_sz);
	memset(km_workers_base, 0, km_sz);
	shm_toc_insert(pcxt->toc, MKTANN_KEY_KM_WORKERS, km_workers_base);

	/* Tree blob — allocated now, populated after k-means */
	void *dsm_tree = shm_toc_allocate(pcxt->toc, max_tree_sz);
	memset(dsm_tree, 0, max_tree_sz);
	shm_toc_insert(pcxt->toc, MKTANN_KEY_TREE, dsm_tree);

	/* Posting phase DSM — allocated now but populated after k-means */
	uint32_t	   alloc_nlist = nlist; /* save allocation-time nlist */
	Size		   res_sz	   = mktann_dsm_reserve_size(alloc_nlist);
	MktDsmReserve *dsm_reserve = shm_toc_allocate(pcxt->toc, res_sz);
	memset(dsm_reserve, 0, res_sz);
	dsm_reserve->nlist = alloc_nlist;
	shm_toc_insert(pcxt->toc, MKTANN_KEY_RESERVE, dsm_reserve);

	Size  out_sz		= mktann_worker_output_size(nlist, nparticipants);
	char *worker_output = shm_toc_allocate(pcxt->toc, out_sz);
	memset(worker_output, 0, out_sz);
	shm_toc_insert(pcxt->toc, MKTANN_KEY_WORKER_OUTPUT, worker_output);

	char *partials = NULL;
	if (!bs->params.fastscan)
	{
		Size part_sz = mktann_partials_size(nlist, nparticipants);
		partials	 = shm_toc_allocate(pcxt->toc, part_sz);
		memset(partials, 0, part_sz);
		shm_toc_insert(pcxt->toc, MKTANN_KEY_PARTIALS, partials);
	}

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

	/* ---- Launch workers ---- */
	LaunchParallelWorkers(pcxt);

	if (pcxt->nworkers_launched == 0)
	{
		WaitForParallelWorkersToFinish(pcxt);
		DestroyParallelContext(pcxt);
		ExitParallelMode();
		return false;
	}

	/* ==== Leader participates in all phases as worker_id=0 ==== */

	/* ---- Phase 1: Leader samples ---- */
	{
		uint32_t cents_per = nlist / nparticipants;
		uint32_t cents_rem = nlist % nparticipants;
		uint32_t cent_end  = (0 < cents_rem) ? (cents_per + 1) : cents_per;

		double est_rows = RelationGetNumberOfBlocks(heap) *
						  (BLCKSZ / (double)(dim * sizeof(float) + 32));
		uint32_t stride = 1;
		if (est_rows / nparticipants > max_per_worker)
			stride = (uint32_t)(est_rows / nparticipants / max_per_worker);
		if (stride < 1)
			stride = 1;

		SampleCbState sc = {
				.samples		= mktann_worker_samples(dsm_samples, 0),
				.centroids		= cents,
				.count			= 0,
				.max_samples	= max_per_worker,
				.stride			= stride,
				.stride_counter = 0,
				.dim			= dim,
				.metric			= shared->metric,
				.cent_start		= 0,
				.cent_end		= cent_end,
				.cents_picked	= 0,
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

	/* Barrier: all participants done sampling */
	BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

	/* ---- Phase 2: Leader runs k-means reduce loop ---- */

	/* Precompute centroid norms for L2 */
	if (shared->metric == DISTANCE_L2)
		for (uint32_t j = 0; j < nlist; j++)
			norms_c[j] = mkt_l2_norm_squared(cents + (size_t)j * dim, dim);

	/* Leader also does assignment on its samples */
	float	 *my_sums = mktann_km_worker_sums(km_workers_base, nlist, dim, 0);
	uint32_t *my_cnts = mktann_km_worker_cnts(km_workers_base, nlist, dim, 0);
	float	 *my_cost = mktann_km_worker_cost(km_workers_base, nlist, dim, 0);

	float	*leader_samples	 = mktann_worker_samples(dsm_samples, 0);
	uint32_t leader_nsamples = mktann_sample_counts(dsm_samples)[0];

	float *old_cents = palloc((size_t)nlist * dim * sizeof(float));

	for (uint32_t iter = 0; iter < shared->km_max_iterations; iter++)
	{
		memcpy(old_cents, cents, (size_t)nlist * dim * sizeof(float));

		/* Leader assignment + accumulation */
		mktann_km_assign_and_accumulate(
				leader_samples,
				leader_nsamples,
				cents,
				norms_c,
				nlist,
				dim,
				shared->metric,
				my_sums,
				my_cnts,
				my_cost);

		/* Barrier: all workers done with assignment */
		BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

		/* Leader reduce: merge per-worker sums → new centroids */
		const float	   **all_sums = palloc(nparticipants * sizeof(float *));
		const uint32_t **all_cnts = palloc(nparticipants * sizeof(uint32_t *));
		float			*all_costs = palloc(nparticipants * sizeof(float));

		for (int t = 0; t < nparticipants; t++)
		{
			all_sums[t] =
					mktann_km_worker_sums(km_workers_base, nlist, dim, t);
			all_cnts[t] =
					mktann_km_worker_cnts(km_workers_base, nlist, dim, t);
			all_costs[t] =
					*mktann_km_worker_cost(km_workers_base, nlist, dim, t);
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
				nlist,
				dim,
				shared->metric,
				&total_cost);

		float tol_sq		 = shared->km_tolerance * shared->km_tolerance;
		shared->km_converged = (shift_sq < tol_sq);

		/* Reset per-worker accumulators for next iteration */
		memset(km_workers_base, 0, km_sz);

		/* Barrier: workers can read updated centroids */
		BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

		if (shared->km_converged)
			break;
	}

	/* ---- Leader: build hkmeans tree from samples + centroids ---- */

	/* Concatenate all workers' samples */
	uint32_t total_nsamples = 0;
	for (int t = 0; t < nparticipants; t++)
		total_nsamples += mktann_sample_counts(dsm_samples)[t];

	float *all_samples = palloc((size_t)total_nsamples * dim * sizeof(float));
	uint32_t off	   = 0;
	for (int t = 0; t < nparticipants; t++)
	{
		uint32_t n = mktann_sample_counts(dsm_samples)[t];
		memcpy(all_samples + (size_t)off * dim,
			   mktann_worker_samples(dsm_samples, t),
			   (size_t)n * dim * sizeof(float));
		off += n;
	}

	KMeansOptions km_opts  = MKT_KMEANS_OPTIONS_DEFAULT;
	km_opts.max_iterations = shared->km_max_iterations;

	HKMeansResult *tree = mkt_hkmeans_f32(
			all_samples,
			total_nsamples,
			NULL,
			dim,
			nlist,
			bs->params.fan_out,
			shared->metric,
			&km_opts);

	pfree(all_samples);
	pfree(old_cents);

	if (tree == NULL)
	{
		WaitForParallelWorkersToFinish(pcxt);
		DestroyParallelContext(pcxt);
		ExitParallelMode();
		return false;
	}

	/* Update nlist from actual tree leaves */
	nlist		  = tree->nleaves;
	shared->nlist = nlist;

	/* Copy tree into pre-allocated DSM slot */
	if (tree->total_size > max_tree_sz)
		elog(ERROR,
			 "mktann: tree too large for DSM (%u > %zu)",
			 tree->total_size,
			 max_tree_sz);
	elog(LOG,
		 "mktann: tree size=%u, dsm slot=%zu, nleaves=%u, nnodes=%u",
		 tree->total_size,
		 max_tree_sz,
		 tree->nleaves,
		 tree->nnodes);
	memcpy(dsm_tree, tree, tree->total_size);

	/* Verify tree is usable from DSM */
	{
		HKMeansResult *t2 = (HKMeansResult *)dsm_tree;
		Assert(t2->nleaves == tree->nleaves);
		Assert(t2->dim == tree->dim);
		float *lc = hk_leaf_centroids(t2);
		Assert(isfinite(lc[0]));
	}

	/* Normalize leaf centroids for cosine */
	float *ref_vecs = hk_leaf_centroids(tree);
	if (shared->metric == DISTANCE_COSINE)
		for (uint32_t c = 0; c < nlist; c++)
			normalize_in_place(ref_vecs + (size_t)c * dim, dim);

	/* Compute P^T * centroids */
	RaBitQParams *rq_params	   = mkt_rabitq_create(dim, rabitq_seed);
	float		 *pt_centroids = palloc((size_t)nlist * dim * sizeof(float));
	for (uint32_t c = 0; c < nlist; c++)
		mkt_rabitq_rotate(
				rq_params,
				ref_vecs + (size_t)c * dim,
				pt_centroids + (size_t)c * dim);

	/* Set up posting page reservation.
	 * Keep dsm_reserve->nlist at the allocated size — the accessor
	 * functions compute array offsets from it. shared->nlist has
	 * the actual count from k-means. */

	uint32_t	 pages_per_cluster = estimate_posting_pages(heap, dim, nlist);
	BlockNumber *starts			   = mktann_dsm_reserve_starts(dsm_reserve);
	uint32_t	*counts			   = mktann_dsm_reserve_counts(dsm_reserve);
	pg_atomic_uint32 *nexts		   = mktann_dsm_reserve_nexts(dsm_reserve);

	BlockNumber total_reserve = 0;
	for (uint32_t c = 0; c < nlist; c++)
		total_reserve += pages_per_cluster;

	BlockNumber actual_start =
			mkt_storage_extend(&storage->base, total_reserve);

	{
		BlockNumber pos = 0;
		for (uint32_t c = 0; c < nlist; c++)
		{
			starts[c] = actual_start + pos;
			counts[c] = pages_per_cluster;
			pos += pages_per_cluster;
		}
	}
	dsm_reserve->total_reserved = total_reserve;
	dsm_reserve->first_posting	= actual_start;

	for (uint32_t c = 0; c < nlist; c++)
		pg_atomic_init_u32(&nexts[c], 1);

	/* Re-init parallel scan for posting phase */
	table_parallelscan_reinitialize(
			heap, ParallelTableScanFromMktShared(shared));

	/* Barrier: tree + reserve ready, workers can start posting */
	BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

	/* ---- Phase 3: Leader does posting scan ---- */
	{
		MktPostingReserve local_reserve = {
				.starts = starts,
				.counts = counts,
				.nexts	= (mkt_atomic_uint32 *)nexts,
				.nlist	= nlist,
				.total	= total_reserve,
		};

		char *my_partials = (partials != NULL)
								  ? mktann_worker_partials(partials, nlist, 0)
								  : NULL;

		MemoryContext worker_ctx = AllocSetContextCreate(
				CurrentMemoryContext,
				"mktann leader worker",
				ALLOCSET_DEFAULT_SIZES);
		MemoryContext prev = MemoryContextSwitchTo(worker_ctx);

		MktPostingWorkerState ws;
		mkt_posting_worker_init(
				&ws,
				0,
				nlist,
				dim,
				bs->params.fastscan,
				&storage->base,
				rq_params,
				ref_vecs,
				pt_centroids,
				&local_reserve,
				my_partials);

		MemoryContextSwitchTo(prev);

		MktBuildParams bp = {
				.dim			  = dim,
				.metric			  = bs->params.metric,
				.soar_lambda	  = bs->params.soar_lambda,
				.boundary_epsilon = bs->params.boundary_epsilon,
		};

		bs->tree		= (HKMeansResult *)dsm_tree;
		bs->worker_bufs = mkt_build_worker_bufs_create(dim);

		LeaderBuildState leader_state = {
				.bs			= bs,
				.ws			= &ws,
				.bp			= bp,
				.worker_ctx = worker_ctx,
				.indtuples	= 0,
				.soar_dupes = 0,
		};

		TableScanDesc scan2 = table_beginscan_parallel(
				heap, ParallelTableScanFromMktShared(shared));

		double leader_reltuples = table_index_build_scan(
				heap,
				index,
				index_info,
				true,
				true,
				leader_build_callback,
				&leader_state,
				scan2);

		mkt_posting_worker_finish(&ws);

		BlockNumber *lh = mktann_worker_heads(worker_output, nlist, 0);
		BlockNumber *lt = mktann_worker_tails(worker_output, nlist, 0);
		bool		*la = mktann_worker_active(worker_output, nlist, 0);
		memcpy(lh, ws.heads, nlist * sizeof(BlockNumber));
		memcpy(lt, ws.tails, nlist * sizeof(BlockNumber));
		memcpy(la, ws.active, nlist * sizeof(bool));

		SpinLockAcquire(&shared->mutex);
		shared->nparticipantsdone++;
		shared->reltuples += leader_reltuples;
		shared->indtuples += leader_state.indtuples;
		shared->soar_dupes += leader_state.soar_dupes;
		SpinLockRelease(&shared->mutex);

		MemoryContextDelete(worker_ctx);
	}

	WaitForParallelWorkersToFinish(pcxt);

	for (int i = 0; i < pcxt->nworkers_launched; i++)
		InstrAccumParallelQuery(&bufferusage[i], &walusage[i]);

	*out_heap_tuples = shared->reltuples;
	*out_indtuples	 = shared->indtuples;
	*out_soar_dupes	 = shared->soar_dupes;
	*out_tree		 = tree;

	/* Merge partial pages and link chains */
	BlockNumber **all_heads	 = palloc(nparticipants * sizeof(BlockNumber *));
	BlockNumber **all_tails	 = palloc(nparticipants * sizeof(BlockNumber *));
	bool		**all_active = palloc(nparticipants * sizeof(bool *));

	for (int t = 0; t < nparticipants; t++)
	{
		all_heads[t]  = mktann_worker_heads(worker_output, nlist, t);
		all_tails[t]  = mktann_worker_tails(worker_output, nlist, t);
		all_active[t] = mktann_worker_active(worker_output, nlist, t);
	}

	MktPostingReserve merge_reserve = {
			.starts = starts,
			.counts = counts,
			.nexts	= (mkt_atomic_uint32 *)nexts,
			.nlist	= nlist,
			.total	= total_reserve,
	};

	MktPostingBuildResult build_result;
	mkt_posting_finalize(
			partials,
			all_heads,
			all_tails,
			all_active,
			nparticipants,
			&storage->base,
			&merge_reserve,
			ref_vecs,
			pt_centroids,
			dim,
			bs->params.fastscan,
			&build_result);

	memcpy(posting_heads, build_result.heads, nlist * sizeof(BlockNumber));
	pfree(build_result.heads);
	pfree(all_heads);
	pfree(all_tails);
	pfree(all_active);
	pfree(pt_centroids);

	elog(LOG,
		 "mktann: parallel build with %d workers, "
		 "%u posting pages, %u merge input, %u merge output, "
		 "reserved %u, relation %u blocks",
		 pcxt->nworkers_launched,
		 build_result.total_pages,
		 build_result.merge_input,
		 build_result.merge_output,
		 total_reserve,
		 RelationGetNumberOfBlocks(index));

	DestroyParallelContext(pcxt);
	ExitParallelMode();

	return true;
}

/* ----------------------------------------------------------------
 * Main build entry point
 * ---------------------------------------------------------------- */

IndexBuildResult *
mktann_build(Relation heap, Relation index, struct IndexInfo *index_info)
{
	MemoryContext caller_ctx = CurrentMemoryContext;
	MemoryContext build_ctx	 = AllocSetContextCreate(
			 CurrentMemoryContext, "mktann build", ALLOCSET_DEFAULT_SIZES);
	MemoryContextSwitchTo(build_ctx);

	/* 1. Initialize build state and resolve parameters */
	MktannBuildState bs = {0};
	bs.heap				= heap;
	bs.index			= index;
	bs.index_info		= index_info;
	bs.build_ctx		= build_ctx;
	bs.tmp_ctx			= AllocSetContextCreate(
			 build_ctx, "mktann build tuple", ALLOCSET_DEFAULT_SIZES);

	resolve_build_params(heap, index, &bs.params);

	const MktannBuildParams *p			 = &bs.params;
	Dimension				 dim		 = p->dim;
	uint64_t				 rabitq_seed = 42;

	MktannStorage storage;
	mktann_storage_init(&storage, index, NULL, p->metric);
	storage.build_mode = true;

	HKMeansResult *tree			 = NULL;
	double		   heap_tuples	 = 0;
	double		   indtuples	 = 0;
	double		   soar_dupes	 = 0;
	BlockNumber	  *posting_heads = NULL;
	float		  *global_mean	 = NULL;

	/* Try parallel build first (sampling + k-means + posting) */
	bool did_parallel = false;
	if (index_info->ii_ParallelWorkers > 0)
	{
		/* Compute nlist estimate. hkmeans with fan_out can produce
		 * up to fan_out^nlevels leaves, so use that as the upper
		 * bound for DSM sizing. */
		uint32_t est_nlist = (uint32_t)sqrt(
				RelationGetNumberOfBlocks(heap) *
				(BLCKSZ / (double)(dim * sizeof(float) + 32)));
		if (est_nlist < 1)
			est_nlist = 1;
		if (p->nlist > 0)
			est_nlist = p->nlist;

		uint32_t fan_out = p->fan_out > 0 ? p->fan_out
										  : mkt_auto_fan_out(0, est_nlist, 0);
		uint32_t nlevels = 1;
		{
			uint32_t n = est_nlist;
			while (n > fan_out)
			{
				n = (n + fan_out - 1) / fan_out;
				nlevels++;
			}
		}
		uint32_t max_nlist = 1;
		for (uint32_t l = 0; l < nlevels; l++)
			max_nlist *= fan_out;
		if (max_nlist < est_nlist)
			max_nlist = est_nlist;

		bs.params.nlist = max_nlist;

		uint32_t est_max_ent =
				mkt_centroid_max_entries_fmt(dim, p->centroid_format);
		BlockNumber first_centroid = 1;
		/* Over-allocate centroid space: max nodes in hkmeans tree
		 * is bounded by 2*nlist, each node's centroids fit in
		 * ceil(fan_out / max_entries_per_page) pages. */
		uint32_t max_nodes			= est_nlist * 2;
		uint32_t est_centroid_pages = (max_nodes + est_max_ent - 1) /
											  est_max_ent +
									  1;
		BlockNumber first_posting = first_centroid + est_centroid_pages;
		mkt_storage_extend(&storage.base, est_centroid_pages);

		posting_heads = palloc(est_nlist * sizeof(BlockNumber));

		did_parallel = do_parallel_build(
				heap,
				index,
				index_info,
				&bs,
				&storage,
				first_posting,
				&tree,
				posting_heads,
				&heap_tuples,
				&indtuples,
				&soar_dupes);

		if (did_parallel && tree != NULL)
		{
			uint32_t nlist	= tree->nleaves;
			bs.params.nlist = nlist;

			global_mean = palloc(dim * sizeof(float));
			mkt_vector_mean(hk_leaf_centroids(tree), nlist, dim, global_mean);
			if (p->metric == DISTANCE_COSINE)
				normalize_in_place(global_mean, dim);
		}
	}

	if (!did_parallel)
	{
		/* Serial fallback: sample, cluster, build */
		pgstat_progress_update_param(
				PROGRESS_CREATEIDX_SUBPHASE, PROGRESS_MKTANN_PHASE_SAMPLE);
		tree = run_clustering(&bs, &global_mean);

		if (tree == NULL)
		{
			MemoryContextSwitchTo(caller_ctx);
			MemoryContextDelete(build_ctx);
			return palloc0(sizeof(IndexBuildResult));
		}

		uint32_t nlist	= tree->nleaves;
		bs.params.nlist = nlist;

		pgstat_progress_update_param(
				PROGRESS_CREATEIDX_SUBPHASE, PROGRESS_MKTANN_PHASE_SETUP);

		RaBitQParams *rq_params = mkt_rabitq_create(dim, rabitq_seed);

		float *ref_vecs = hk_leaf_centroids(tree);
		if (p->metric == DISTANCE_COSINE)
			for (uint32_t c = 0; c < nlist; c++)
				normalize_in_place(ref_vecs + (size_t)c * dim, dim);

		float *pt_centroids = palloc((size_t)nlist * dim * sizeof(float));
		for (uint32_t c = 0; c < nlist; c++)
			mkt_rabitq_rotate(
					rq_params,
					ref_vecs + (size_t)c * dim,
					pt_centroids + (size_t)c * dim);

		uint32_t max_ent =
				mkt_centroid_max_entries_fmt(dim, p->centroid_format);
		BlockNumber *node_first_blkno = palloc(
				tree->nnodes * sizeof(BlockNumber));
		BlockNumber first_centroid = 1;
		BlockNumber first_posting  = mkt_compute_centroid_layout(
				 tree, max_ent, first_centroid, node_first_blkno);

		write_meta_page(
				&storage.base,
				dim,
				(uint8_t)tree->nlevels,
				(uint8_t)p->fan_out,
				first_centroid,
				0,
				nlist,
				p->centroid_format,
				p->metric,
				rabitq_seed,
				global_mean);

		uint32_t n_centroid_pages = first_posting - first_centroid;
		mkt_storage_extend(&storage.base, n_centroid_pages);

		posting_heads = palloc(nlist * sizeof(BlockNumber));

		uint32_t reserve_each = estimate_posting_pages(heap, dim, nlist);

		MktPostingBuilder *builders = palloc(
				nlist * sizeof(MktPostingBuilder));
		for (uint32_t c = 0; c < nlist; c++)
		{
			if (p->fastscan)
				mkt_posting_builder_init_fastscan(
						&builders[c],
						&storage.base,
						rq_params,
						dim,
						c,
						ref_vecs + (size_t)c * dim,
						pt_centroids + (size_t)c * dim);
			else
				mkt_posting_builder_init(
						&builders[c],
						&storage.base,
						rq_params,
						dim,
						c,
						ref_vecs + (size_t)c * dim,
						pt_centroids + (size_t)c * dim);

			BlockNumber start =
					mkt_storage_extend(&storage.base, reserve_each);
			if (start != InvalidBlockNumber)
				mkt_posting_builder_set_reserve(
						&builders[c], start, reserve_each);
		}

		bs.tree		   = tree;
		bs.builders	   = builders;
		bs.worker_bufs = mkt_build_worker_bufs_create(dim);
		bs.indtuples   = 0;
		bs.soar_dupes  = 0;

		heap_tuples = table_index_build_scan(
				heap,
				index,
				index_info,
				true,
				true,
				build_callback,
				(void *)&bs,
				NULL);

		indtuples  = bs.indtuples;
		soar_dupes = bs.soar_dupes;

		/* Finish posting builders */
		pgstat_progress_update_param(
				PROGRESS_CREATEIDX_SUBPHASE, PROGRESS_MKTANN_PHASE_POSTING);
		for (uint32_t c = 0; c < nlist; c++)
		{
			posting_heads[c] = mkt_posting_builder_finish(&builders[c]);
			mkt_posting_builder_cleanup(&builders[c]);
		}
		pfree(builders);
	}

	if (soar_dupes > 0)
		elog(LOG,
			 "mktann: replicated %.0f vectors "
			 "(%.1f%% of %.0f, lambda=%.4g)",
			 soar_dupes,
			 100.0 * soar_dupes / indtuples,
			 indtuples,
			 bs.params.soar_lambda);

	/* Write centroid pages + metadata + WAL */
	{
		uint32_t nlist = tree->nleaves;

		RaBitQParams *rq = mkt_rabitq_create(dim, rabitq_seed);

		uint32_t max_ent =
				mkt_centroid_max_entries_fmt(dim, p->centroid_format);
		BlockNumber *nfb = palloc(tree->nnodes * sizeof(BlockNumber));
		BlockNumber	 fc	 = 1;
		BlockNumber	 fp	 = mkt_compute_centroid_layout(tree, max_ent, fc, nfb);
		(void)fp;

		if (global_mean == NULL)
		{
			global_mean = palloc(dim * sizeof(float));
			mkt_vector_mean(hk_leaf_centroids(tree), nlist, dim, global_mean);
			if (p->metric == DISTANCE_COSINE)
				normalize_in_place(global_mean, dim);
		}

		write_meta_page(
				&storage.base,
				dim,
				(uint8_t)tree->nlevels,
				(uint8_t)p->fan_out,
				fc,
				0,
				nlist,
				p->centroid_format,
				p->metric,
				rabitq_seed,
				global_mean);

		pgstat_progress_update_param(
				PROGRESS_CREATEIDX_SUBPHASE, PROGRESS_MKTANN_PHASE_CENTROID);
		mkt_write_centroid_tree(
				&storage.base,
				tree,
				dim,
				p->fan_out,
				p->centroid_format,
				rq,
				global_mean,
				posting_heads,
				nfb,
				NULL);

		Page			page = mkt_storage_write_page(&storage.base, 0);
		MktannMetaPage *meta = (MktannMetaPage *)PageGetSpecialPointer(page);
		meta->ntuples		 = (uint32_t)indtuples;
		if (p->fastscan)
			meta->flags |= MKT_META_FLAG_FASTSCAN;
		mkt_storage_commit_page(&storage.base, 0);

		pgstat_progress_update_param(
				PROGRESS_CREATEIDX_SUBPHASE, PROGRESS_MKTANN_PHASE_WAL);
		log_newpage_range(
				index,
				MAIN_FORKNUM,
				0,
				RelationGetNumberOfBlocks(index),
				true);

		pfree(nfb);
	}

	/* Cleanup */
	mkt_free(tree);
	pfree(posting_heads);

	MemoryContextSwitchTo(caller_ctx);
	MemoryContextDelete(build_ctx);

	IndexBuildResult *result = palloc0(sizeof(IndexBuildResult));
	result->heap_tuples		 = heap_tuples;
	result->index_tuples	 = indtuples;
	return result;
}

char *
mktann_buildphasename(int64 phasenum)
{
	switch (phasenum)
	{
	case PROGRESS_CREATEIDX_SUBPHASE_INITIALIZE:
		return "initializing";
	case PROGRESS_MKTANN_PHASE_SAMPLE:
		return "sampling vectors";
	case PROGRESS_MKTANN_PHASE_KMEANS:
		return "clustering (k-means)";
	case PROGRESS_MKTANN_PHASE_SETUP:
		return "preparing RaBitQ encoding";
	case PROGRESS_MKTANN_PHASE_SCAN:
		return "scanning table";
	case PROGRESS_MKTANN_PHASE_POSTING:
		return "finalizing posting lists";
	case PROGRESS_MKTANN_PHASE_CENTROID:
		return "writing centroid pages";
	case PROGRESS_MKTANN_PHASE_WAL:
		return "WAL logging";
	default:
		return NULL;
	}
}
