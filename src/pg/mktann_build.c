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
		HKMeansResult	 *tree,
		RaBitQParams	 *rq_params,
		const float		 *ref_vecs,
		const float		 *pt_centroids,
		BlockNumber		  first_posting,
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

	EnterParallelMode();

	ParallelContext *pcxt = CreateParallelContext(
			"meerkat", "mktann_parallel_build_main", nworkers);

	/* Estimate DSM size */
	Snapshot snapshot	= SnapshotAny;
	Size	 est_shared = add_size(
			BUFFERALIGN(sizeof(MktBuildShared)),
			table_parallelscan_estimate(heap, snapshot));

	shm_toc_estimate_chunk(&pcxt->estimator, est_shared);
	shm_toc_estimate_chunk(&pcxt->estimator, tree->total_size);
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

	int nkeys = 7;
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

	/* Populate shared state */
	MktBuildShared *shared	 = shm_toc_allocate(pcxt->toc, est_shared);
	shared->heaprelid		 = RelationGetRelid(heap);
	shared->indexrelid		 = RelationGetRelid(index);
	shared->queryid			 = pgstat_get_my_query_id();
	shared->dim				 = dim;
	shared->metric			 = bs->params.metric;
	shared->nlist			 = nlist;
	shared->fan_out			 = bs->params.fan_out;
	shared->soar_lambda		 = bs->params.soar_lambda;
	shared->boundary_epsilon = bs->params.boundary_epsilon;
	shared->fastscan		 = bs->params.fastscan;
	shared->rabitq_seed		 = rabitq_seed;
	shared->nparticipants	 = nparticipants;
	SpinLockInit(&shared->mutex);
	ConditionVariableInit(&shared->workersdonecv);
	shared->nparticipantsdone = 0;
	shared->reltuples		  = 0.0;
	shared->indtuples		  = 0.0;
	shared->soar_dupes		  = 0.0;
	table_parallelscan_initialize(
			heap, ParallelTableScanFromMktShared(shared), snapshot);
	shm_toc_insert(pcxt->toc, MKTANN_KEY_SHARED, shared);

	/* Copy tree blob into DSM */
	void *dsm_tree = shm_toc_allocate(pcxt->toc, tree->total_size);
	memcpy(dsm_tree, tree, tree->total_size);
	shm_toc_insert(pcxt->toc, MKTANN_KEY_TREE, dsm_tree);

	/* Set up page reservation in DSM */
	Size		   res_sz	   = mktann_dsm_reserve_size(nlist);
	MktDsmReserve *dsm_reserve = shm_toc_allocate(pcxt->toc, res_sz);
	memset(dsm_reserve, 0, res_sz);
	dsm_reserve->nlist		   = nlist;
	dsm_reserve->first_posting = first_posting;

	uint32_t	 pages_per_cluster = estimate_posting_pages(heap, dim, nlist);
	BlockNumber *starts			   = mktann_dsm_reserve_starts(dsm_reserve);
	uint32_t	*counts			   = mktann_dsm_reserve_counts(dsm_reserve);

	BlockNumber total_reserve = 0;
	for (uint32_t c = 0; c < nlist; c++)
	{
		starts[c] = first_posting + total_reserve;
		counts[c] = pages_per_cluster;
		total_reserve += pages_per_cluster;
	}
	dsm_reserve->total_reserved = total_reserve;

	pg_atomic_uint32 *nexts = mktann_dsm_reserve_nexts(dsm_reserve);
	for (uint32_t c = 0; c < nlist; c++)
		pg_atomic_init_u32(&nexts[c], 1);

	shm_toc_insert(pcxt->toc, MKTANN_KEY_RESERVE, dsm_reserve);

	mkt_storage_extend(&storage->base, total_reserve);

	/* Worker output area */
	Size  out_sz		= mktann_worker_output_size(nlist, nparticipants);
	char *worker_output = shm_toc_allocate(pcxt->toc, out_sz);
	memset(worker_output, 0, out_sz);
	shm_toc_insert(pcxt->toc, MKTANN_KEY_WORKER_OUTPUT, worker_output);

	/* Partials buffer (only for AoS, not fastscan) */
	char *partials = NULL;
	if (!bs->params.fastscan)
	{
		Size part_sz = mktann_partials_size(nlist, nparticipants);
		partials	 = shm_toc_allocate(pcxt->toc, part_sz);
		memset(partials, 0, part_sz);
		shm_toc_insert(pcxt->toc, MKTANN_KEY_PARTIALS, partials);
	}

	/* WAL/buffer usage */
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

	/* Launch workers */
	LaunchParallelWorkers(pcxt);

	if (pcxt->nworkers_launched == 0)
	{
		WaitForParallelWorkersToFinish(pcxt);
		DestroyParallelContext(pcxt);
		ExitParallelMode();
		return false;
	}

	/* Leader participates as worker_id = 0 */
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

		LeaderBuildState leader_state = {
				.bs			= bs,
				.ws			= &ws,
				.bp			= bp,
				.worker_ctx = worker_ctx,
				.indtuples	= 0,
				.soar_dupes = 0,
		};

		TableScanDesc scan = table_beginscan_parallel(
				heap, ParallelTableScanFromMktShared(shared));

		double leader_reltuples = table_index_build_scan(
				heap,
				index,
				index_info,
				true,
				true,
				leader_build_callback,
				&leader_state,
				scan);

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

	/* 2. Sample and cluster vectors */
	pgstat_progress_update_param(
			PROGRESS_CREATEIDX_SUBPHASE, PROGRESS_MKTANN_PHASE_SAMPLE);
	float		  *global_mean;
	HKMeansResult *tree = run_clustering(&bs, &global_mean);

	if (tree == NULL)
	{
		MemoryContextSwitchTo(caller_ctx);
		MemoryContextDelete(build_ctx);
		return palloc0(sizeof(IndexBuildResult));
	}

	const MktannBuildParams *p	   = &bs.params;
	Dimension				 dim   = p->dim;
	uint32_t				 nlist = tree->nleaves;
	bs.params.nlist				   = nlist;

	/* 3. Single-pass streaming build */
	pgstat_progress_update_param(
			PROGRESS_CREATEIDX_SUBPHASE, PROGRESS_MKTANN_PHASE_SETUP);
	pgstat_progress_update_param(PROGRESS_CREATEIDX_TUPLES_TOTAL, 0);
	pgstat_progress_update_param(PROGRESS_CREATEIDX_TUPLES_DONE, 0);

	uint64_t	  rabitq_seed = 42;
	RaBitQParams *rq_params	  = mkt_rabitq_create(dim, rabitq_seed);

	MktannStorage storage;
	mktann_storage_init(&storage, index, NULL, p->metric);
	storage.build_mode = true;

	/* Normalize leaf centroids for cosine */
	float *ref_vecs = hk_leaf_centroids(tree);
	if (p->metric == DISTANCE_COSINE)
	{
		for (uint32_t c = 0; c < nlist; c++)
			normalize_in_place(ref_vecs + (size_t)c * dim, dim);
	}

	/* Compute P^T * centroids for posting scan query state */
	float *pt_centroids = palloc((size_t)nlist * dim * sizeof(float));
	for (uint32_t c = 0; c < nlist; c++)
		mkt_rabitq_rotate(
				rq_params,
				ref_vecs + (size_t)c * dim,
				pt_centroids + (size_t)c * dim);

	/* Compute centroid page layout starting at block 1 */
	uint32_t max_ent = mkt_centroid_max_entries_fmt(dim, p->centroid_format);
	BlockNumber *node_first_blkno = palloc(tree->nnodes * sizeof(BlockNumber));
	BlockNumber	 first_centroid	  = 1;
	BlockNumber	 first_posting	  = mkt_compute_centroid_layout(
			tree, max_ent, first_centroid, node_first_blkno);

	/* Write metadata page (block 0) with known centroid layout */
	write_meta_page(
			&storage.base,
			dim,
			(uint8_t)tree->nlevels,
			(uint8_t)p->fan_out,
			first_centroid,
			0, /* ntuples placeholder */
			nlist,
			p->centroid_format,
			p->metric,
			rabitq_seed,
			global_mean);

	/* Reserve centroid blocks so posting pages start after them */
	uint32_t n_centroid_pages = first_posting - first_centroid;
	mkt_storage_extend(&storage.base, n_centroid_pages);

	double		 heap_tuples   = 0;
	double		 indtuples	   = 0;
	double		 soar_dupes	   = 0;
	BlockNumber *posting_heads = palloc(nlist * sizeof(BlockNumber));

	/* 4. Heap scan — parallel or serial */
	pgstat_progress_update_param(
			PROGRESS_CREATEIDX_SUBPHASE, PROGRESS_MKTANN_PHASE_SCAN);
	pgstat_progress_update_param(PROGRESS_CREATEIDX_TUPLES_TOTAL, 0);
	pgstat_progress_update_param(PROGRESS_CREATEIDX_TUPLES_DONE, 0);

	bool did_parallel = false;
	if (index_info->ii_ParallelWorkers > 0)
	{
		bs.tree		   = tree;
		bs.worker_bufs = mkt_build_worker_bufs_create(dim);
		did_parallel   = do_parallel_build(
				  heap,
				  index,
				  index_info,
				  &bs,
				  &storage,
				  tree,
				  rq_params,
				  ref_vecs,
				  pt_centroids,
				  first_posting,
				  posting_heads,
				  &heap_tuples,
				  &indtuples,
				  &soar_dupes);
	}

	if (!did_parallel)
	{
		/* Serial fallback: one builder per cluster */
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

	/* 6. Write centroid pages into reserved blocks */
	pgstat_progress_update_param(
			PROGRESS_CREATEIDX_SUBPHASE, PROGRESS_MKTANN_PHASE_CENTROID);
	pgstat_progress_update_param(PROGRESS_CREATEIDX_TUPLES_TOTAL, 0);
	pgstat_progress_update_param(PROGRESS_CREATEIDX_TUPLES_DONE, 0);
	mkt_write_centroid_tree(
			&storage.base,
			tree,
			dim,
			p->fan_out,
			p->centroid_format,
			rq_params,
			global_mean,
			posting_heads,
			node_first_blkno,
			NULL); /* pt_centroids on posting pages, not here */

	/* Update metadata with final tuple count and flags */
	{
		Page			page = mkt_storage_write_page(&storage.base, 0);
		MktannMetaPage *meta = (MktannMetaPage *)PageGetSpecialPointer(page);
		meta->ntuples		 = (uint32_t)indtuples;
		if (p->fastscan)
			meta->flags |= MKT_META_FLAG_FASTSCAN;
		mkt_storage_commit_page(&storage.base, 0);
	}

	/* 7. WAL-log all pages */
	pgstat_progress_update_param(
			PROGRESS_CREATEIDX_SUBPHASE, PROGRESS_MKTANN_PHASE_WAL);
	pgstat_progress_update_param(PROGRESS_CREATEIDX_TUPLES_TOTAL, 0);
	pgstat_progress_update_param(PROGRESS_CREATEIDX_TUPLES_DONE, 0);
	log_newpage_range(
			index, MAIN_FORKNUM, 0, RelationGetNumberOfBlocks(index), true);

	/* Cleanup */
	mkt_free(tree);
	pfree(node_first_blkno);
	pfree(posting_heads);
	pfree(pt_centroids);

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
