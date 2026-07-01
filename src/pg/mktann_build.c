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
 *   7. Single heap scan: route each row page-backed (the same
 *      mkt_query_route the query/insert use), stream into posting builders
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
#include "index/index_base.h"
#include "index/index_build.h"
#include "index/parallel_build.h"
#include "index/posting_build.h"
#include "index/posting_build_parallel.h"
#include "index/posting_page.h"
#include "index/query_scan.h"
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
	RaBitQParams *rq_params;

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

	/*
	 * Page-backed assignment (routes each row the same way the query/insert do,
	 * so the in-RAM tree is not needed for the scan). qs holds the routing
	 * state; route is the shared route+encode+emit context (also used by the
	 * parallel posting workers), which references qs and the sorter.
	 */
	MktQueryState	 qs;
	MktBuildRouteCtx route;
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
 * Build callback — single-pass: route each row page-backed,
 * stream encoded entries into the cluster-keyed sorter
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

	/* Route + encode + stream via the shared page-backed helper (the parallel
	 * posting workers use the very same call). */
	mkt_build_route_emit(&bs->route, vref.data, *tid);

	if (((uint64_t)bs->route.indtuples % 10000) == 0)
		mkt_build_report_progress(bs->prog, bs->route.indtuples);

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

/*
 * Draw the maintenance_work_mem-bounded k-means sample into bs->samples (left
 * resident for the caller to cluster + free), normalize it for cosine, resolve
 * the leaf target against the sample size, and compute the global mean (the
 * sample mean — the encoder centering the streaming build needs up front,
 * before any centroid page is written). Returns false (and frees the sample)
 * when the heap yields no indexable rows.
 */
static bool
sample_for_build(
		MktannBuildState *bs, float **out_global_mean, uint32_t *out_nlist)
{
	Dimension dim	= bs->params.dim;
	uint32_t  nlist = bs->params.nlist;

	/* Bound the sample buffer by maintenance_work_mem (and MaxAllocSize). The
	 * tree is trained on this sample; leaf centroids are refined on the full
	 * table by a later (page-backed) refine pass when subsampling loses
	 * quality. */
	uint64_t ideal_samples = Max((uint64_t)10000, (uint64_t)nlist * 256);
	uint64_t budget		   = (uint64_t)maintenance_work_mem * 1024 /
					  (dim * sizeof(float));
	uint64_t alloc_cap = (uint64_t)(MaxAllocSize / (dim * sizeof(float)));
	uint64_t cap	   = Min(budget, alloc_cap);
	if (cap < 10000)
		cap = 10000;
	bs->max_samples = (int)Min(ideal_samples, cap);

	bs->nsamples = 0;
	bs->samples	 = palloc((size_t)bs->max_samples * dim * sizeof(float));

	sample_rows(bs);

	if (bs->nsamples > bs->max_samples)
		bs->nsamples = bs->max_samples;

	if (bs->params.metric == DISTANCE_COSINE)
		for (int i = 0; i < bs->nsamples; i++)
			mkt_l2_normalize(bs->samples + (size_t)i * dim, dim);

	if (bs->nsamples == 0)
	{
		pfree(bs->samples);
		bs->samples = NULL;
		return false;
	}

	if ((uint32_t)bs->nsamples < nlist)
		nlist = (uint32_t)bs->nsamples;

	float *global_mean = palloc(dim * sizeof(float));
	mkt_vector_mean(bs->samples, (uint32_t)bs->nsamples, dim, global_mean);
	if (bs->params.metric == DISTANCE_COSINE)
		mkt_l2_normalize(global_mean, dim);

	*out_global_mean = global_mean;
	*out_nlist		 = nlist;
	return true;
}

/* ----------------------------------------------------------------
 * Serial build
 * ---------------------------------------------------------------- */

/*
 * Per-leaf head-page writer for the streaming build. Fired by
 * mkt_stream_centroid_write once per leaf (during the write pass, while the
 * leaf's float centroid is still resident), it writes that cluster's
 * posting-list head page carrying pt_centroid = P^T*centroid — the exact encode
 * reference the scan reads — at the leaf's reserved head block.
 */
typedef struct SerialHeadCtx
{
	MktStorage		   *storage;
	const RaBitQParams *rq_params;
	Dimension			dim;
	bool				fastscan;
	const BlockNumber  *posting_heads;
	float			   *pt; /* [dim] scratch */
} SerialHeadCtx;

static void
serial_write_head(void *arg, uint32_t leaf, const float *centroid)
{
	SerialHeadCtx *h = (SerialHeadCtx *)arg;
	mkt_rabitq_rotate(h->rq_params, centroid, h->pt);

	MktPostingBuilder hb;
	if (h->fastscan)
		mkt_posting_builder_init_fastscan(
				&hb, h->storage, h->rq_params, h->dim, leaf, centroid, h->pt);
	else
		mkt_posting_builder_init(
				&hb, h->storage, h->rq_params, h->dim, leaf, centroid, h->pt);
	mkt_posting_builder_set_first_blkno(&hb, h->posting_heads[leaf]);
	mkt_posting_builder_finish(&hb);
	mkt_posting_builder_cleanup(&hb);
}

/*
 * Serial build fallback, mirroring do_parallel_build's role for the
 * non-parallel path. Streams the centroid tree straight to pages (no in-RAM
 * tree, no O(nlist*dim) blob), then scans the heap once through build_callback
 * (page-backed routing) to fill the posting lists.
 *
 * *out_tree is a lightweight metadata carrier (leaf/level counts) — the build
 * no longer materializes a tree. out_posting_heads is allocated here. Returns
 * false with *out_tree == NULL when the heap has no tuples.
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
		double			 *out_soar_dupes,
		bool			 *out_finalized)
{
	const MktannBuildParams *p	 = &bs->params;
	Dimension				 dim = p->dim;

	*out_finalized = false;

	mkt_build_report_phase(bs->prog, MKT_BUILD_PHASE_SAMPLE);

	float	*global_mean  = NULL;
	uint32_t target_nlist = 0;
	if (!sample_for_build(bs, &global_mean, &target_nlist))
	{
		*out_tree = NULL;
		return false;
	}

	KMeansOptions km_opts = MKT_KMEANS_OPTIONS_DEFAULT;
	km_opts.algorithm	  = KMEANS_ALGO_LLOYD;
	km_opts.nredo		  = p->kmeans_nredo;

	/*
	 * Plan pass: cluster the sample and discover the tree shape (leaf count,
	 * depth, per-leaf sample counts, centroid page count) without writing. The
	 * write pass below re-clusters the same sample (fixed k-means seed ->
	 * identical tree) and streams the pages, so target_nlist (not the resolved
	 * leaf count) must drive both passes.
	 */
	mkt_build_report_phase(bs->prog, MKT_BUILD_PHASE_KMEANS);
	MktStreamTreePlan plan;
	if (!mkt_stream_centroid_plan(
				bs->samples,
				(uint32_t)bs->nsamples,
				dim,
				target_nlist,
				p->fan_out,
				p->metric,
				p->centroid_format,
				&km_opts,
				&plan))
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("hierarchical k-means failed")));

	uint32_t nlist	 = plan.nleaves;
	bs->params.nlist = nlist;

	mkt_build_report_phase(bs->prog, MKT_BUILD_PHASE_SETUP);
	RaBitQParams *rq_params = mkt_rabitq_create(dim, rabitq_seed);

	/* Centroid area is [first_centroid, first_posting); block 0 is metadata. */
	BlockNumber first_centroid = 1;
	BlockNumber first_posting  = first_centroid + plan.centroid_pages;

	/*
	 * Reserve a contiguous posting range per cluster from the plan's per-leaf
	 * sample counts, extrapolated to the full table (handles skew; slight
	 * over-estimate for headroom). reserve_init applies the format +
	 * replication headroom and the page math.
	 */
	bool   replicate = p->soar_lambda > 0.0 || p->boundary_epsilon > 0.0;
	double est_rows	 = RelationGetNumberOfBlocks(bs->heap) *
					  (BLCKSZ / (double)(dim * sizeof(float) + 32));
	double	  scale	 = bs->nsamples > 0 ? est_rows / (double)bs->nsamples : 1.0;
	uint32_t *counts = palloc(nlist * sizeof(uint32_t));
	for (uint32_t c = 0; c < nlist; c++)
		counts[c] = (uint32_t)((double)plan.leaf_counts[c] * scale);

	MktPostingReserve reserve;
	mkt_posting_reserve_init(
			&reserve, counts, nlist, 1, dim, p->fastscan, replicate);
	pfree(counts);

	BlockNumber *posting_heads = palloc(nlist * sizeof(BlockNumber));
	for (uint32_t c = 0; c < nlist; c++)
		posting_heads[c] = first_posting + reserve.starts[c];

	/*
	 * Pre-extend the relation to cover metadata + centroid + posting areas so
	 * the streaming write can place centroids at reserved blocks and each
	 * leaf's head page (in the far posting area) already exists when the write
	 * pass emits it.
	 */
	mkt_storage_extend(storage, first_posting + reserve.total);

	/*
	 * Write pass: stream the centroid pages (reserved blocks, post-order, root
	 * last) and, per leaf, its head page carrying pt_centroid. Both build and
	 * query then route page-backed over these centroid pages. This path
	 * finalizes centroids + metadata itself (out_finalized).
	 */
	mkt_build_report_phase(bs->prog, MKT_BUILD_PHASE_CENTROID);
	SerialHeadCtx headctx = {
			.storage	   = storage,
			.rq_params	   = rq_params,
			.dim		   = dim,
			.fastscan	   = p->fastscan,
			.posting_heads = posting_heads,
			.pt			   = palloc((size_t)dim * sizeof(float)),
	};
	BlockNumber root = mkt_stream_centroid_write(
			storage,
			bs->samples,
			(uint32_t)bs->nsamples,
			dim,
			target_nlist,
			p->fan_out,
			p->metric,
			p->centroid_format,
			rq_params,
			global_mean,
			&km_opts,
			posting_heads,
			first_centroid,
			serial_write_head,
			&headctx);
	pfree(headctx.pt);
	mkt_free(plan.leaf_counts);
	pfree(bs->samples);
	bs->samples = NULL;
	if (root == InvalidBlockNumber)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("streaming centroid write failed")));

	/*
	 * Page-backed routing: build a MktIndexBase from the just-written index so
	 * the scan routes each row exactly as the query/insert do (mkt_query_route
	 * over the centroid pages). The tree is no longer used for assignment. The
	 * base is stack-local but outlives the scan (all within this function); qs
	 * holds it by pointer until mkt_query_state_cleanup below.
	 */
	MktIndexBase idx_base = {0};
	idx_base.params		   = rq_params;
	idx_base.pt_global_mean = palloc((size_t)dim * sizeof(float));
	mkt_rabitq_rotate(rq_params, global_mean, idx_base.pt_global_mean);
	idx_base.rabitq_seed	 = rabitq_seed;
	idx_base.centroid_storage = storage;
	idx_base.posting_storage  = storage;
	idx_base.page_base		  = NULL;
	idx_base.dim			  = dim;
	idx_base.nlevels		  = (uint8_t)plan.nlevels;
	idx_base.first_centroid	  = root;
	idx_base.metric			  = p->metric;
	idx_base.centroid_format  = p->centroid_format;
	idx_base.fastscan =
			(p->centroid_format == MKT_CENTROID_FMT_FASTSCAN) ? mkt_fastscan_bits
															  : 0;
	idx_base.centroid_error_scale = (float)mkt_centroid_error_scale;
	idx_base.centroid_beam_scale  = (float)mkt_centroid_beam_scale;
	mkt_query_state_init(&bs->qs, &idx_base, 1, MKT_SECONDARY_TOPK);

	/*
	 * Cluster-keyed sorter: the scan streams every posting entry here (keyed
	 * by cluster); mkt_posting_build_lists then reads them back grouped by
	 * cluster and builds each list with a single resident page builder. Memory
	 * is bounded by maintenance_work_mem (the sort spills if exceeded),
	 * replacing the old builders[nlist] array (O(N)).
	 *
	 * region == NULL: a plain, non-parallel cluster-keyed sort. Same seam the
	 * parallel leader uses.
	 */
	bs->rq_params = rq_params;
	bs->sorter	  = mkt_pbuild_sort_begin(
			 NULL,
			 NULL,
			 0,
			 0,
			 true,
			 (uint32_t)mkt_posting_entry_size(dim),
			 maintenance_work_mem);

	/* Shared route+encode+emit context (same helper the parallel workers use). */
	mkt_build_route_ctx_init(
			&bs->route,
			&bs->qs,
			bs->sorter,
			rq_params,
			storage,
			posting_heads,
			nlist,
			dim,
			p->soar_lambda,
			p->boundary_epsilon);

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
			/* ref_vecs = NULL: read each list's pt_centroid from the head page
			 * pre-written above (page-backed), not from the in-RAM tree. */
			NULL,
			&reserve,
			first_posting,
			posting_heads);
	bs->sorter = NULL; /* ended by mkt_posting_build_lists */

	bs->indtuples  = bs->route.indtuples;
	bs->soar_dupes = bs->route.soar_dupes;
	mkt_build_route_ctx_cleanup(&bs->route);
	mkt_query_state_cleanup(&bs->qs);
	mkt_posting_reserve_free(&reserve);

	elog(LOG,
		 "mktann: serial build scan %.1fms, "
		 "%.0f tuples, %.0f soar_dupes, %u clusters",
		 INSTR_TIME_GET_MILLISEC(t_serial_scan),
		 bs->indtuples,
		 bs->soar_dupes,
		 nlist);

	/* Metadata page (needs the final tuple count). first_centroid is the root
	 * block, which the streaming write placed last. This path finalizes the
	 * centroid pages + metadata itself, so the caller's tail does WAL only. */
	write_meta_page(
			storage,
			dim,
			(uint8_t)plan.nlevels,
			(uint8_t)p->fan_out,
			root,
			0,
			nlist,
			p->centroid_format,
			p->metric,
			rabitq_seed,
			global_mean);
	{
		Page			page = mkt_storage_write_page(storage, 0);
		MktannMetaPage *meta = (MktannMetaPage *)PageGetSpecialPointer(page);
		meta->ntuples		 = (uint32_t)bs->indtuples;
		if (p->fastscan)
			meta->flags |= MKT_META_FLAG_FASTSCAN;
		mkt_storage_commit_page(storage, 0);
	}
	*out_finalized = true;

	/* No tree is materialized any more; hand back a lightweight carrier with the
	 * leaf/level counts the caller needs (the empty-table check + metadata). */
	HKMeansResult *meta_tree = palloc0(sizeof(HKMeansResult));
	meta_tree->nleaves		 = nlist;
	meta_tree->nlevels		 = plan.nlevels;
	meta_tree->dim			 = dim;

	*out_tree		   = meta_tree;
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

	/* Set by a build path that writes its own centroid pages + metadata (the
	 * serial page-backed path), so the shared finalize below does WAL only. */
	bool centroids_finalized = false;
	/* Set by the parallel path, which writes the centroid pages itself but not
	 * the metadata page: the finalize still runs (for meta) but skips its own
	 * centroid tree write. */
	bool parallel_centroids = false;

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
				&soar_dupes,
				/* do_parallel_build routes page-backed: it wrote the centroid +
				 * head pages into the index before the scan and returns the
				 * global mean it used (so the metadata write below matches). It
				 * does NOT write the metadata page (needs the final tuple count),
				 * so the finalize below still runs — it only skips the centroid
				 * tree write when parallel_centroids is set. */
				&global_mean,
				&parallel_centroids);

		if (did_parallel && tree != NULL)
			bs.params.nlist = tree->nleaves;
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
					&soar_dupes,
					&centroids_finalized))
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

	/* Finalize: write centroid pages + metadata (unless the build path already
	 * did — the serial page-backed path writes them itself), then WAL-log the
	 * whole index. */
	{
		if (!centroids_finalized)
		{
			uint32_t nlist = tree->nleaves;

			RaBitQParams *rq = mkt_rabitq_create(dim, rabitq_seed);

			uint32_t max_ent =
					mkt_centroid_max_entries_fmt(dim, p->centroid_format);
			BlockNumber *nfb = palloc(tree->nnodes * sizeof(BlockNumber));
			BlockNumber	 fc	 = 1;
			BlockNumber	 fp	 = mkt_compute_centroid_layout(
					 tree, max_ent, fc, nfb);

			if (global_mean == NULL)
			{
				global_mean = palloc(dim * sizeof(float));
				mkt_vector_mean(
						hk_leaf_centroids(tree), nlist, dim, global_mean);
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

			/* The parallel path already wrote the centroid + head pages before
			 * its scan (page-backed routing); only the serial-fallback path
			 * needs the finalize to write them here. */
			if (!parallel_centroids)
			{
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
			}

			Page			page = mkt_storage_write_page(&storage.base, 0);
			MktannMetaPage *meta = (MktannMetaPage *)PageGetSpecialPointer(
					page);
			meta->ntuples = (uint32_t)indtuples;
			if (p->fastscan)
				meta->flags |= MKT_META_FLAG_FASTSCAN;
			mkt_storage_commit_page(&storage.base, 0);

			pfree(nfb);
		}

		mkt_build_report_phase(&prog, MKT_BUILD_PHASE_WAL);
		log_newpage_range(
				index,
				MAIN_FORKNUM,
				0,
				RelationGetNumberOfBlocks(index),
				true);
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
