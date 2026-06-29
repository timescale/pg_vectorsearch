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

#include <access/htup_details.h>
#include <access/table.h>
#include <access/tableam.h>
#include <access/xloginsert.h>
#include <catalog/index.h>
#include <catalog/pg_operator_d.h>
#include <catalog/pg_type_d.h>
#include <commands/progress.h>
#include <common/pg_prng.h>
#include <executor/tuptable.h>
#include <math.h>
#include <miscadmin.h>
#include <pgstat.h>
#include <utils/backend_progress.h>
#include <utils/injection_point.h>
#include <utils/memutils.h>
#include <utils/rel.h>
#include <utils/sampling.h>
#include <utils/tuplesort.h>

#include "algo/distance.h"
#include "algo/hkmeans.h"
#include "algo/kmeans.h"
#include "algo/vecops.h"
#include "index/centroid_build.h"
#include "index/centroid_page.h"
#include "index/index_build.h"
#include "index/parallel_build.h"
#include "index/posting_build.h"
#include "index/posting_build_parallel.h"
#include "index/posting_page.h"
#include "mkt_halfvec.h"
#include "mkt_pg.h"
#include "mkt_vector.h"
#include "mktann_build.h"
#include "mktann_meta.h"
#include "mktann_storage.h"
#include "quant/rabitq.h"

/* ----------------------------------------------------------------
 * Build state (MktannBuildParams is in mktann_build.h, shared with the
 * parallel leader)
 * ---------------------------------------------------------------- */

typedef struct MktannBuildState
{
	MktannBuildParams params;

	/* Tree for centroid assignment */
	const HKMeansResult *tree;

	double indtuples;  /* total count */
	double soar_dupes; /* replicated SOAR vectors */

	/*
	 * Posting entries are streamed into a cluster-keyed tuplesort during the
	 * heap scan, then read back grouped by cluster so the build holds only ONE
	 * posting-page builder at a time. This bounds build memory to
	 * maintenance_work_mem (the tuplesort stays in RAM until it exceeds it,
	 * then spills) instead of the old builders[nlist] array (one ~8KB working
	 * page per cluster = O(nlist) = O(N)).
	 */
	Tuplesortstate *post_sort;
	TupleDesc		post_tupdesc;
	TupleTableSlot *post_slot;
	/* Entries are RaBitQ-encoded during the scan (relative to the assigned
	 * cluster centroid) and the compact code is stored in the sort, so the
	 * build phase replays it via add_encoded. */
	RaBitQParams *rq_params;
	const float	 *leaf_cents;	/* [nlist*dim] cluster centroids */
	RaBitQData	 *enc_buf;		/* scratch RaBitQ output */
	RaBitQScratch enc_scratch;	/* scratch encode buffers */
	uint32_t	  packed_bytes; /* (dim+7)/8 = bytes of sign bits */

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

/*
 * Encode one vector relative to its assigned cluster centroid and append the
 * resulting posting entry to the cluster-keyed tuplesort. The payload bytea
 * carries: tid | f_add | f_rescale | f_error | sign-bits[packed_bytes]. The
 * build phase reads tuples back grouped by cluster and replays them with
 * add_encoded, so the encoding is preserved bit-for-bit. Call within a
 * per-tuple context (tuplesort copies the tuple).
 */
static void
post_sort_put(
		MktannBuildState *bs,
		uint32_t		  cluster,
		ItemPointerData	  tid,
		const float		 *vec)
{
	Size   bytes   = mkt_posting_entry_size(bs->params.dim);
	bytea *payload = (bytea *)palloc(VARHDRSZ + bytes);
	SET_VARSIZE(payload, VARHDRSZ + bytes);
	mkt_posting_entry_encode(
			bs->rq_params,
			vec,
			bs->leaf_cents + (size_t)cluster * bs->params.dim,
			bs->params.dim,
			bs->enc_buf,
			&bs->enc_scratch,
			tid,
			VARDATA(payload));

	TupleTableSlot *slot = bs->post_slot;
	ExecClearTuple(slot);
	slot->tts_values[0] = Int32GetDatum((int32)cluster);
	slot->tts_isnull[0] = false;
	slot->tts_values[1] = PointerGetDatum(payload);
	slot->tts_isnull[1] = false;
	ExecStoreVirtualTuple(slot);
	tuplesort_puttupleslot(bs->post_sort, slot);
}

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

	post_sort_put(bs, asgn.primary, *tid, asgn.enc_vector);
	bs->indtuples++;

	if (((uint64_t)bs->indtuples % 10000) == 0)
		pgstat_progress_update_param(
				PROGRESS_CREATEIDX_TUPLES_DONE, (int64)bs->indtuples);

	if (asgn.secondary != MKT_INVALID_CLUSTER)
	{
		post_sort_put(bs, asgn.secondary, *tid, asgn.enc_vector);
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
	/* Block 0 must already exist (extended or new_page'd by caller).
	 * Use write_page to write in-place. */
	Page page = mkt_storage_write_page(storage, 0);

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

	mkt_storage_commit_page(storage, 0);
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
	int			   cc	= (opts != NULL) ? opts->centroid_compression
										 : MKT_CENTROID_COMPRESSION_AUTO;
	bool cfastscan		= (opts != NULL) ? opts->centroid_fastscan : false;

	/*
	 * RaBitQ centroids estimate L2 distance, which routes correctly for
	 * L2 and (normalized) cosine but not inner product. ON forces it and
	 * errors for inner product; AUTO compresses everything except inner
	 * product; OFF disables it.
	 */
	bool compressed;
	switch (cc)
	{
	case MKT_CENTROID_COMPRESSION_ON:
		if (metric == DISTANCE_INNER_PRODUCT)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("centroid_compression=on is not supported "
							"with vector_ip_ops")));
		compressed = true;
		break;
	case MKT_CENTROID_COMPRESSION_OFF:
		compressed = false;
		break;
	default: /* AUTO */
		compressed = (metric != DISTANCE_INNER_PRODUCT);
		break;
	}

	/*
	 * FASTSCAN centroids are a packed layout over the RaBitQ-compressed
	 * representation, so they require compression (and, like RaBitQ
	 * centroids, don't apply to inner product).
	 */
	if (cfastscan)
	{
		if (metric == DISTANCE_INNER_PRODUCT)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("centroid_fastscan is not supported "
							"with vector_ip_ops")));
		if (!compressed)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("centroid_fastscan requires "
							"centroid_compression")));
		return MKT_CENTROID_FMT_FASTSCAN;
	}

	if (compressed)
		return MKT_CENTROID_FMT_RABITQ;

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
		p->nlist		 = mkt_auto_nlist(reltuples);
		if (p->nlist > MKTANN_MAX_NLIST)
			p->nlist = MKTANN_MAX_NLIST;
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

/* ----------------------------------------------------------------
 * Streaming leaf-centroid refinement
 *
 * When maintenance_work_mem forces the k-means sample set below the ideal, the
 * tree structure is trained on that bounded subsample, but the leaf centroids
 * can still be trained on the whole table: a streaming pass routes every row
 * to its leaf (tree descent) and accumulates per-leaf means. Memory is bounded
 * by the tree plus the accumulators -- no large sample buffer -- so it works
 * under the same budget. A few passes act as leaf-level Lloyd iterations over
 * all rows, recovering most of the quality a full in-memory sample would have
 * given.
 * ---------------------------------------------------------------- */

typedef struct RefineState
{
	const HKMeansResult *tree;
	double				*sums;	  /* [nleaves * dim] */
	uint64_t			*cnts;	  /* [nleaves] */
	float				*scratch; /* [dim] normalized copy for cosine */
	Dimension			 dim;
	DistanceMetric		 metric;
} RefineState;

static void
refine_callback(
		Relation	index,
		ItemPointer tid,
		Datum	   *values,
		bool	   *isnull,
		bool		tuple_is_alive,
		void	   *state)
{
	RefineState *rs = (RefineState *)state;

	(void)index;
	(void)tid;
	(void)tuple_is_alive;

	if (isnull[0])
		return;

	Dimension	 dim = rs->dim;
	const float *v	 = MktVectorToRef(DatumGetMktVector(values[0])).data;

	/* The tree is trained in the normalized space for cosine, so route and
	 * accumulate the normalized vector to match. */
	if (rs->metric == DISTANCE_COSINE)
	{
		memcpy(rs->scratch, v, (size_t)dim * sizeof(float));
		mkt_l2_normalize(rs->scratch, dim);
		v = rs->scratch;
	}

	uint32_t leaf = mkt_hkmeans_assign(rs->tree, v, rs->metric, NULL);
	double	*sum  = rs->sums + (size_t)leaf * dim;
	for (Dimension j = 0; j < dim; j++)
		sum[j] += v[j];
	rs->cnts[leaf]++;
}

static void
refine_leaf_centroids(MktannBuildState *bs, HKMeansResult *tree, int iters)
{
	Dimension dim	  = bs->params.dim;
	uint32_t  nleaves = tree->nleaves;
	float	 *cents	  = hk_leaf_centroids(tree);

	RefineState rs = {
			.tree	 = tree,
			.sums	 = palloc((size_t)nleaves * dim * sizeof(double)),
			.cnts	 = palloc((size_t)nleaves * sizeof(uint64_t)),
			.scratch = palloc((size_t)dim * sizeof(float)),
			.dim	 = dim,
			.metric	 = bs->params.metric,
	};

	for (int it = 0; it < iters; it++)
	{
		memset(rs.sums, 0, (size_t)nleaves * dim * sizeof(double));
		memset(rs.cnts, 0, (size_t)nleaves * sizeof(uint64_t));

		table_index_build_scan(
				bs->heap,
				bs->index,
				bs->index_info,
				true,
				false,
				refine_callback,
				(void *)&rs,
				NULL);

		for (uint32_t l = 0; l < nleaves; l++)
		{
			if (rs.cnts[l] == 0)
				continue; /* keep the subsample centroid for an empty leaf */
			double *sum = rs.sums + (size_t)l * dim;
			float  *c	= cents + (size_t)l * dim;
			double	inv = 1.0 / (double)rs.cnts[l];
			for (Dimension j = 0; j < dim; j++)
				c[j] = (float)(sum[j] * inv);
		}
	}

	pfree(rs.sums);
	pfree(rs.cnts);
	pfree(rs.scratch);
}

static HKMeansResult *
run_clustering(MktannBuildState *bs, float **out_global_mean)
{
	Dimension dim	= bs->params.dim;
	uint32_t  nlist = bs->params.nlist;

	/*
	 * Bound the sample buffer by maintenance_work_mem (and MaxAllocSize). When
	 * the budget forces a subsample, the tree structure is built from it and
	 * the leaf centroids are refined on the full table below.
	 */
	uint64_t ideal_samples = Max((uint64_t)10000, (uint64_t)nlist * 256);
	uint64_t budget		   = (uint64_t)maintenance_work_mem * 1024 /
					  (dim * sizeof(float));
	uint64_t alloc_cap = (uint64_t)(MaxAllocSize / (dim * sizeof(float)));
	uint64_t cap	   = Min(budget, alloc_cap);
	if (cap < 10000)
		cap = 10000;
	bool subsampled = ideal_samples > cap;
	bs->max_samples = (int)Min(ideal_samples, cap);

	bs->nsamples = 0;
	bs->samples	 = palloc((size_t)bs->max_samples * dim * sizeof(float));

	sample_rows(bs);

	if (bs->nsamples > bs->max_samples)
		bs->nsamples = bs->max_samples;

	if (bs->params.metric == DISTANCE_COSINE)
	{
		for (int i = 0; i < bs->nsamples; i++)
			mkt_l2_normalize(bs->samples + (size_t)i * dim, dim);
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

	/*
	 * The tree was trained on a budget-bounded subsample. Refine its leaf
	 * centroids on the whole table so they are full-data means, not
	 * subsample means -- bounded memory, a few streaming passes.
	 */
	if (subsampled && mkt_leaf_refine_iters > 0)
	{
		instr_time t_ref_start;
		INSTR_TIME_SET_CURRENT(t_ref_start);

		refine_leaf_centroids(bs, tree, mkt_leaf_refine_iters);

		instr_time t_ref_end;
		INSTR_TIME_SET_CURRENT(t_ref_end);
		INSTR_TIME_SUBTRACT(t_ref_end, t_ref_start);
		elog(LOG,
			 "mktann: leaf refinement %.1fms (%d passes) -- structure from a "
			 "%d-sample subsample (maintenance_work_mem-bounded), %u leaves "
			 "refined on the full table",
			 INSTR_TIME_GET_MILLISEC(t_ref_end),
			 mkt_leaf_refine_iters,
			 bs->max_samples,
			 tree->nleaves);
	}

	float *global_mean = palloc(dim * sizeof(float));
	mkt_vector_mean(hk_leaf_centroids(tree), tree->nleaves, dim, global_mean);

	if (bs->params.metric == DISTANCE_COSINE)
		mkt_l2_normalize(global_mean, dim);

	*out_global_mean = global_mean;
	return tree;
}

/* ----------------------------------------------------------------
 * Serial build
 * ---------------------------------------------------------------- */

/*
 * Serial build fallback, mirroring do_parallel_build's role for the
 * non-parallel path: sample + cluster, reserve the posting page layout, then
 * scan the heap once through build_callback to fill the posting builders.
 *
 * out_posting_heads is allocated here (sized to the resolved tree->nleaves).
 * Returns false with *out_tree == NULL when the heap has no tuples.
 */
static bool
do_serial_build(
		MktannBuildState *bs,
		MktStorage		 *storage,
		uint64_t		  rabitq_seed,
		HKMeansResult	**out_tree,
		float			**out_global_mean,
		BlockNumber		**out_posting_heads,
		double			 *out_heap_tuples,
		double			 *out_indtuples,
		double			 *out_soar_dupes)
{
	const MktannBuildParams *p	 = &bs->params;
	Dimension				 dim = p->dim;

	pgstat_progress_update_param(
			PROGRESS_CREATEIDX_SUBPHASE, PROGRESS_MKTANN_PHASE_SAMPLE);

	float		  *global_mean = NULL;
	HKMeansResult *tree		   = run_clustering(bs, &global_mean);

	if (tree == NULL)
	{
		*out_tree = NULL;
		return false;
	}

	uint32_t nlist	 = tree->nleaves;
	bs->params.nlist = nlist;

	pgstat_progress_update_param(
			PROGRESS_CREATEIDX_SUBPHASE, PROGRESS_MKTANN_PHASE_SETUP);

	RaBitQParams *rq_params = mkt_rabitq_create(dim, rabitq_seed);

	float *ref_vecs = hk_leaf_centroids(tree);
	if (p->metric == DISTANCE_COSINE)
		for (uint32_t c = 0; c < nlist; c++)
			mkt_l2_normalize(ref_vecs + (size_t)c * dim, dim);

	float *pt_centroids = palloc((size_t)nlist * dim * sizeof(float));
	for (uint32_t c = 0; c < nlist; c++)
		mkt_rabitq_rotate(
				rq_params,
				ref_vecs + (size_t)c * dim,
				pt_centroids + (size_t)c * dim);

	uint32_t max_ent = mkt_centroid_max_entries_fmt(dim, p->centroid_format);
	BlockNumber *node_first_blkno = palloc(tree->nnodes * sizeof(BlockNumber));
	BlockNumber	 first_centroid	  = 1;
	BlockNumber	 first_posting	  = mkt_compute_centroid_layout(
			tree, max_ent, first_centroid, node_first_blkno);

	/* Block 0 = metadata page */
	mkt_storage_extend(storage, 1);

	write_meta_page(
			storage,
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
	mkt_storage_extend(storage, n_centroid_pages);

	BlockNumber *posting_heads = palloc(nlist * sizeof(BlockNumber));

	/*
	 * Reserve a contiguous page range per cluster via the shared
	 * reserve_init (same as the parallel and standalone paths). Serial has
	 * no per-cluster sample assignment, so it feeds a uniform estimate:
	 * the heap-size cardinality guess split evenly. reserve_init applies
	 * the format + replication headroom and the page math.
	 */
	bool   replicate = p->soar_lambda > 0.0 || p->boundary_epsilon > 0.0;
	double est_rows	 = RelationGetNumberOfBlocks(bs->heap) *
					  (BLCKSZ / (double)(dim * sizeof(float) + 32));
	uint32_t  base	 = (uint32_t)ceil(est_rows / nlist);
	uint32_t *counts = palloc(nlist * sizeof(uint32_t));
	for (uint32_t c = 0; c < nlist; c++)
		counts[c] = base;

	MktPostingReserve reserve;
	mkt_posting_reserve_init(
			&reserve, counts, nlist, 1, dim, p->fastscan, replicate);
	pfree(counts);
	mkt_storage_extend(storage, reserve.total);

	/*
	 * Cluster-keyed tuplesort: (int4 cluster_id, bytea payload=tid+enc). The
	 * scan streams every posting entry here; the build phase below reads them
	 * back grouped by cluster and builds each list with a single resident page
	 * builder. Memory is bounded by maintenance_work_mem (the sort spills if
	 * exceeded), replacing the old builders[nlist] array (O(N)).
	 */
	bs->rq_params	 = rq_params;
	bs->leaf_cents	 = ref_vecs;
	bs->packed_bytes = (dim + 7) / 8;
	bs->enc_buf		 = palloc(MKT_RABITQ_DATA_SIZE(dim));
	mkt_rabitq_scratch_init(&bs->enc_scratch, dim);
	bs->post_tupdesc = CreateTemplateTupleDesc(2);
	TupleDescInitEntry(bs->post_tupdesc, 1, "cluster", INT4OID, -1, 0);
	TupleDescInitEntry(bs->post_tupdesc, 2, "payload", BYTEAOID, -1, 0);
	{
		AttrNumber attNums[1]	 = {1};
		Oid		   sortOps[1]	 = {Int4LessOperator};
		Oid		   sortColls[1]	 = {InvalidOid};
		bool	   nullsFirst[1] = {false};
		bs->post_sort			 = tuplesort_begin_heap(
				   bs->post_tupdesc,
				   1,
				   attNums,
				   sortOps,
				   sortColls,
				   nullsFirst,
				   maintenance_work_mem,
				   NULL,
				   TUPLESORT_NONE);
	}
	bs->post_slot =
			MakeSingleTupleTableSlot(bs->post_tupdesc, &TTSOpsMinimalTuple);

	bs->tree		= tree;
	bs->worker_bufs = mkt_build_worker_bufs_create(dim);
	bs->indtuples	= 0;
	bs->soar_dupes	= 0;

	pgstat_progress_update_param(
			PROGRESS_CREATEIDX_SUBPHASE, PROGRESS_MKTANN_PHASE_SCAN);
	/* Test hook: lets an isolation test observe an in-progress serial build
	 * (e.g. the progress view's phase). No-op unless PG was built with
	 * injection points and a test has attached an action. */
	INJECTION_POINT("mktann-build-load", NULL);

	instr_time t_serial_start;
	INSTR_TIME_SET_CURRENT(t_serial_start);

	double heap_tuples = table_index_build_scan(
			bs->heap,
			bs->index,
			bs->index_info,
			true,
			true,
			build_callback,
			(void *)bs,
			NULL);

	instr_time t_serial_scan;
	INSTR_TIME_SET_CURRENT(t_serial_scan);
	INSTR_TIME_SUBTRACT(t_serial_scan, t_serial_start);

	/*
	 * Sort entries by cluster, then build each cluster's posting list with a
	 * single resident page builder, in cluster order. Empty clusters still get
	 * an (empty) head page, matching the previous per-cluster behavior.
	 */
	pgstat_progress_update_param(
			PROGRESS_CREATEIDX_SUBPHASE, PROGRESS_MKTANN_PHASE_POSTING);
	tuplesort_performsort(bs->post_sort);

	bool have = tuplesort_gettupleslot(
			bs->post_sort, true, false, bs->post_slot, NULL);
	for (uint32_t c = 0; c < nlist; c++)
	{
		MktPostingBuilder b;
		if (p->fastscan)
			mkt_posting_builder_init_fastscan(
					&b,
					storage,
					rq_params,
					dim,
					c,
					ref_vecs + (size_t)c * dim,
					pt_centroids + (size_t)c * dim);
		else
			mkt_posting_builder_init(
					&b,
					storage,
					rq_params,
					dim,
					c,
					ref_vecs + (size_t)c * dim,
					pt_centroids + (size_t)c * dim);
		mkt_posting_builder_set_shared_reserve(
				&b,
				first_posting + reserve.starts[c],
				reserve.counts[c],
				&reserve.nexts[c]);
		mkt_posting_builder_set_first_blkno(
				&b, first_posting + reserve.starts[c]);

		while (have)
		{
			bool  isnull;
			int32 cl = DatumGetInt32(slot_getattr(bs->post_slot, 1, &isnull));
			if ((uint32_t)cl != c)
				break;
			bytea *pl = DatumGetByteaPP(
					slot_getattr(bs->post_slot, 2, &isnull));
			mkt_posting_entry_add(&b, VARDATA_ANY(pl), dim);
			have = tuplesort_gettupleslot(
					bs->post_sort, true, false, bs->post_slot, NULL);
		}

		posting_heads[c] = mkt_posting_builder_finish(&b);
		mkt_posting_builder_cleanup(&b);
	}

	tuplesort_end(bs->post_sort);
	bs->post_sort = NULL;
	ExecDropSingleTupleTableSlot(bs->post_slot);
	bs->post_slot = NULL;
	FreeTupleDesc(bs->post_tupdesc);
	bs->post_tupdesc = NULL;
	mkt_rabitq_scratch_cleanup(&bs->enc_scratch);
	pfree(bs->enc_buf);
	bs->enc_buf = NULL;
	mkt_posting_reserve_free(&reserve);

	elog(LOG,
		 "mktann: serial build scan %.1fms, "
		 "%.0f tuples, %.0f soar_dupes, %u clusters",
		 INSTR_TIME_GET_MILLISEC(t_serial_scan),
		 bs->indtuples,
		 bs->soar_dupes,
		 tree->nleaves);

	*out_tree		   = tree;
	*out_global_mean   = global_mean;
	*out_posting_heads = posting_heads;
	*out_heap_tuples   = heap_tuples;
	*out_indtuples	   = bs->indtuples;
	*out_soar_dupes	   = bs->soar_dupes;
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
		/*
		 * Estimate the leaf count up front.
		 *
		 * A parallel build allocates its shared-memory (DSM) regions before
		 * the workers run, and DSM segments cannot be resized once created.
		 * Several of those regions are sized per leaf (centroids, posting
		 * heads, per-cluster assignment state), so the leader has to commit to
		 * a leaf count at allocation time — but the real count (tree->nleaves)
		 * is only known after k-means clusters the sample, which happens
		 * inside the workers. hkmeans targets `nlist` leaves at this fan_out
		 * but can produce up to fan_out^nlevels of them, so we size every
		 * per-leaf region for that worst-case upper bound here, then narrow to
		 * the actual tree->nleaves once the tree comes back below.
		 *
		 * nlist and fan_out are already resolved (resolve_build_params).
		 */
		uint32_t max_nlist = mkt_max_nlist(p->nlist, p->fan_out);

		bs.params.nlist = max_nlist;

		posting_heads = palloc(max_nlist * sizeof(BlockNumber));

		MktBuildConfig cfg = {
				.dim			  = bs.params.dim,
				.metric			  = bs.params.metric,
				.centroid_format  = bs.params.centroid_format,
				.nlist			  = bs.params.nlist,
				.fan_out		  = bs.params.fan_out,
				.soar_lambda	  = bs.params.soar_lambda,
				.boundary_epsilon = bs.params.boundary_epsilon,
				.fastscan		  = bs.params.fastscan,
				.concurrent		  = index_info->ii_Concurrent,
		};

		pgstat_progress_update_param(
				PROGRESS_CREATEIDX_SUBPHASE,
				PROGRESS_MKTANN_PHASE_SCAN_PARALLEL);
		/* Test hook: lets an isolation test observe an in-progress parallel
		 * build (e.g. the progress view's phase). No-op unless PG was built
		 * with injection points and a test has attached an action. */
		INJECTION_POINT("mktann-build-load", NULL);

		did_parallel = do_parallel_build(
				heap,
				index,
				index_info,
				&cfg,
				&storage.base,
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
				mkt_l2_normalize(global_mean, dim);
		}
	}

	if (!did_parallel)
	{
		/* Serial fallback: sample, cluster, build. Leaves tree == NULL when
		 * the heap has no indexable tuples. */
		if (!do_serial_build(
					&bs,
					&storage.base,
					rabitq_seed,
					&tree,
					&global_mean,
					&posting_heads,
					&heap_tuples,
					&indtuples,
					&soar_dupes))
			tree = NULL;
	}

	/*
	 * An empty heap (via either path) yields no centroid tree. Refuse the
	 * build rather than emit a centroidless index that cannot route inserts or
	 * be scanned — failing loudly at CREATE INDEX beats a cryptic read error
	 * on the first insert. (A later phase can build a degenerate
	 * single-cluster index so an empty table is indexable.)
	 */
	if (tree == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("cannot build a \"mktann\" index on an empty table"),
				 errhint("Insert data before creating the index, or REINDEX "
						 "once the table has rows.")));

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
				mkt_l2_normalize(global_mean, dim);
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
	case PROGRESS_MKTANN_PHASE_SCAN_PARALLEL:
		return "scanning table (parallel)";
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
