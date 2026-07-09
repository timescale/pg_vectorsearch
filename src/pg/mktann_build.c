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
#include <common/pg_prng.h>
#include <math.h>
#include <miscadmin.h>
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
	/* Posting entries are RaBitQ-encoded during the scan (relative to the
	 * assigned cluster centroid) and fed to a cluster-keyed sorter — the same
	 * MktSorter seam the parallel build uses, here in non-parallel mode (no
	 * coordinate). The shared build loop (mkt_posting_build_lists) reads them
	 * back grouped by cluster and writes one resident page builder at a time.
	 */
	MktSorter	 *sorter;
	char		 *entry; /* scratch, mkt_posting_entry_size(dim) bytes */
	RaBitQParams *rq_params;
	const float	 *leaf_cents;  /* [nlist*dim] cluster centroids */
	RaBitQData	 *enc_buf;	   /* scratch RaBitQ output */
	RaBitQScratch enc_scratch; /* scratch encode buffers */

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

	MktBuildProgress *prog; /* phase/progress reporting seam (serial path) */
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

	if (mkt_posting_emit_assignment(
				bs->sorter,
				&asgn,
				bs->rq_params,
				bs->leaf_cents,
				bs->params.dim,
				*tid,
				bs->enc_buf,
				&bs->enc_scratch,
				bs->entry))
		bs->soar_dupes++;
	bs->indtuples++;

	if (((uint64_t)bs->indtuples % 10000) == 0)
		mkt_build_report_progress(bs->prog, bs->indtuples);

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
		BlockNumber		  first_posting,
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
	meta->first_posting	  = first_posting;
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

/*
 * Estimated heap row count for the progress total. Uses the planner's
 * reltuples when available, else a heap-size guess (same form the nlist
 * auto-tune uses). This is what makes pg_stat_progress_create_index report a
 * meaningful percent_complete during the scan phases.
 */
static double
estimate_heap_tuples(Relation heap, Dimension dim)
{
	if (heap->rd_rel->reltuples > 0)
		return heap->rd_rel->reltuples;
	return RelationGetNumberOfBlocks(heap) *
		   (BLCKSZ / (double)(sizeof(float) * dim + 32));
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
	double				*sums;	  /* [tile * dim], indexed by leaf - tile_lo */
	uint64_t			*cnts;	  /* [tile] */
	float				*scratch; /* [dim] normalized copy for cosine */
	Dimension			 dim;
	DistanceMetric		 metric;
	uint32_t tile_lo; /* accumulate only leaves in [tile_lo, tile_hi) */
	uint32_t tile_hi;
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
	const float *vin = MktVectorToRef(DatumGetMktVector(values[0])).data;

	const float *v;
	uint32_t	 leaf = mkt_refine_assign_leaf(
			rs->tree, vin, dim, rs->metric, rs->scratch, &v);
	/* Only the current tile's leaves are resident in the accumulator. */
	if (leaf < rs->tile_lo || leaf >= rs->tile_hi)
		return;
	double *sum = rs->sums + (size_t)(leaf - rs->tile_lo) * dim;
	for (Dimension j = 0; j < dim; j++)
		sum[j] += v[j];
	rs->cnts[leaf - rs->tile_lo]++;
}

/*
 * Refine leaf centroids on the full table. The accumulator (sums[nleaves*dim]
 * doubles + counts) is O(nleaves*dim) = O(N), so it is bounded by tiling: at
 * most `tile` leaves are resident, and the heap is re-scanned once per tile.
 * `tile` is sized to maintenance_work_mem, so when the whole accumulator fits
 * (the common case — even the large mwm of a big build) there is a single tile
 * and a single scan per iteration, identical to the untiled version. Only a
 * tight mwm relative to nleaves forces multiple tiles (and re-scans); those
 * update centroids in place between tiles (the bounded-memory trade-off).
 */
static void
refine_leaf_centroids(MktannBuildState *bs, HKMeansResult *tree, int iters)
{
	Dimension dim	  = bs->params.dim;
	uint32_t  nleaves = tree->nleaves;
	float	 *cents	  = hk_leaf_centroids(tree);

	/* Tile size: as many leaves as fit a bounded accumulator. Capped by
	 * MaxAllocSize (each palloc stays legal) as well as maintenance_work_mem,
	 * so the accumulator is a constant ceiling independent of nlist — nleaves
	 * above it just means more (re-scanned) tiles, not a bigger allocation. */
	uint64_t cap_bytes =
			Min((uint64_t)maintenance_work_mem * 1024, (uint64_t)MaxAllocSize);
	uint64_t per_leaf = (uint64_t)dim * sizeof(double) + sizeof(uint64_t);
	uint32_t tile	  = (uint32_t)
			Min((uint64_t)nleaves, Max(UINT64CONST(1), cap_bytes / per_leaf));

	RefineState rs = {
			.tree	 = tree,
			.sums	 = palloc((size_t)tile * dim * sizeof(double)),
			.cnts	 = palloc((size_t)tile * sizeof(uint64_t)),
			.scratch = palloc((size_t)dim * sizeof(float)),
			.dim	 = dim,
			.metric	 = bs->params.metric,
	};

	for (int it = 0; it < iters; it++)
	{
		for (uint32_t lo = 0; lo < nleaves; lo += tile)
		{
			uint32_t hi = Min(lo + tile, nleaves);
			rs.tile_lo	= lo;
			rs.tile_hi	= hi;
			memset(rs.sums, 0, (size_t)(hi - lo) * dim * sizeof(double));
			memset(rs.cnts, 0, (size_t)(hi - lo) * sizeof(uint64_t));

			table_index_build_scan(
					bs->heap,
					bs->index,
					bs->index_info,
					true,
					false,
					refine_callback,
					(void *)&rs,
					NULL);

			for (uint32_t l = lo; l < hi; l++)
			{
				if (rs.cnts[l - lo] == 0)
					continue; /* keep subsample centroid for an empty leaf */
				double *sum = rs.sums + (size_t)(l - lo) * dim;
				float  *c	= cents + (size_t)l * dim;
				double	inv = 1.0 / (double)rs.cnts[l - lo];
				for (Dimension j = 0; j < dim; j++)
					c[j] = (float)(sum[j] * inv);
			}
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

	mkt_build_report_phase(bs->prog, MKT_BUILD_PHASE_KMEANS);

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

		mkt_build_report_phase(bs->prog, MKT_BUILD_PHASE_REFINE);
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

	mkt_build_report_phase(bs->prog, MKT_BUILD_PHASE_SAMPLE);

	float		  *global_mean = NULL;
	HKMeansResult *tree		   = run_clustering(bs, &global_mean);

	if (tree == NULL)
	{
		*out_tree = NULL;
		return false;
	}

	uint32_t nlist	 = tree->nleaves;
	bs->params.nlist = nlist;

	mkt_build_report_phase(bs->prog, MKT_BUILD_PHASE_SETUP);

	RaBitQParams *rq_params = mkt_rabitq_create(dim, rabitq_seed);

	/* ref_vecs (the leaf centroids, normalized in place for cosine by the
	 * setup below) is reused as the posting encode reference; its rotated
	 * P^T*centroid is computed per-cluster in mkt_posting_build_lists. */
	float	   *ref_vecs	  = hk_leaf_centroids(tree);
	BlockNumber first_posting = mkt_build_setup_centroid_layout(
			storage, tree, dim, nlist, p->metric, p->centroid_format);

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
	 * Cluster-keyed sorter: the scan streams every posting entry here (keyed
	 * by cluster); mkt_posting_build_lists then reads them back grouped by
	 * cluster and builds each list with a single resident page builder. Memory
	 * is bounded by maintenance_work_mem (the sort spills if exceeded),
	 * replacing the old builders[nlist] array (O(N)).
	 */
	bs->rq_params  = rq_params;
	bs->leaf_cents = ref_vecs;
	bs->enc_buf	   = palloc(MKT_RABITQ_DATA_SIZE(dim));
	mkt_rabitq_scratch_init(&bs->enc_scratch, dim);
	bs->entry = palloc(mkt_posting_entry_size(dim));
	/* region == NULL: a plain, non-parallel cluster-keyed sort bounded by
	 * maintenance_work_mem (spills if exceeded), replacing the old
	 * builders[nlist] array (O(N)). Same seam the parallel leader uses. */
	bs->sorter = mkt_pbuild_sort_begin(
			NULL,
			NULL,
			0,
			0,
			true,
			(uint32_t)mkt_posting_entry_size(dim),
			maintenance_work_mem);

	bs->tree		= tree;
	bs->worker_bufs = mkt_build_worker_bufs_create(dim);
	bs->indtuples	= 0;
	bs->soar_dupes	= 0;

	/* Reports the scan phase and fires the "mktann-build-load" test hook (see
	 * the seam): lets an isolation test observe the in-progress serial build.
	 */
	mkt_build_report_phase(bs->prog, MKT_BUILD_PHASE_SCAN);

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
	mkt_build_report_phase(bs->prog, MKT_BUILD_PHASE_POSTING);
	mkt_posting_build_lists(
			bs->sorter,
			storage,
			nlist,
			dim,
			p->fastscan,
			rq_params,
			ref_vecs,
			&reserve,
			first_posting,
			posting_heads);
	bs->sorter = NULL; /* ended by mkt_posting_build_lists */

	mkt_rabitq_scratch_cleanup(&bs->enc_scratch);
	pfree(bs->enc_buf);
	bs->enc_buf = NULL;
	pfree(bs->entry);
	bs->entry = NULL;
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

	/*
	 * Build introspection: one reporting context the serial and parallel paths
	 * share, so pg_stat_progress_create_index advances through the same phases
	 * either way and (when mkt.log_build_stats is on) per-phase stats land in
	 * the server log.
	 */
	MktBuildStats	 stats = {0};
	MktBuildProgress prog;
	mkt_build_progress_begin(
			&prog,
			index_info->ii_ParallelWorkers > 0,
			mkt_log_build_stats,
			build_ctx,
			&stats,
			estimate_heap_tuples(heap, dim));
	bs.prog = &prog;

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

		/* do_parallel_build reports every phase through the seam (sampling,
		 * k-means, subtrees, graft, refine, setup, scan, posting) and fires
		 * the per-phase test hooks, so no phase is set here. */
		did_parallel = do_parallel_build(
				heap,
				index,
				index_info,
				&cfg,
				&storage.base,
				&prog,
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
				fp,
				0,
				nlist,
				p->centroid_format,
				p->metric,
				rabitq_seed,
				global_mean);

		mkt_build_report_phase(&prog, MKT_BUILD_PHASE_CENTROID);
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

		mkt_build_report_phase(&prog, MKT_BUILD_PHASE_WAL);
		log_newpage_range(
				index,
				MAIN_FORKNUM,
				0,
				RelationGetNumberOfBlocks(index),
				true);

		pfree(nfb);
	}

	/* Flush the final phase timing + emit the build summary (heap_ctx is read
	 * here, so this must run before build_ctx is deleted below). */
	mkt_build_progress_end(&prog);

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
	/* Single source of truth for the phase names (index/build_progress.c),
	 * shared with the build logs so the two never drift. */
	return unconstify(char *, mkt_build_phase_name((int)phasenum));
}
