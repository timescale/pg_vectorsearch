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
		MktannBuildState *bs,
		float		    **out_global_mean,
		uint32_t		 *out_nlist,
		bool			 *out_subsampled)
{
	Dimension dim	= bs->params.dim;
	uint32_t  nlist = bs->params.nlist;

	/* Bound the sample buffer purely by maintenance_work_mem (Option A): a
	 * single huge allocation, not clamped to the 1 GB palloc ceiling, so a large
	 * mwm trains finer centroids on big tables. It is a fixed working set — it
	 * never scales with the row count beyond the mwm cap. The tree is trained on
	 * this sample; leaf centroids are refined on the full table by a later
	 * (page-backed) refine pass when subsampling loses quality.
	 *
	 * Design note — the whole sample is held resident on purpose. Lloyd k-means
	 * makes ~km_max_iterations random-access passes over every sample point, so
	 * one heap read + in-RAM iteration is far cheaper than re-streaming per
	 * iteration; the hierarchical build then partitions this one buffer in place
	 * (index ranges, no per-level copies). The cost is that peak build RAM is the
	 * mwm the operator granted — bounded, but not small.
	 *
	 * Limitation / alternatives if a much smaller footprint is ever wanted (all
	 * trade speed or quality, so deferred):
	 *   - mini-batch / online k-means: stream bounded batches from the table and
	 *     update centroids incrementally (memory O(batch + centroids-in-flight),
	 *     well below mwm) — converges to slightly lower-quality centroids and is
	 *     an algorithmic rewrite of the clustering core;
	 *   - spill the sample to disk (tuplestore/BufFile) and re-stream each
	 *     iteration — bounded to a read buffer, but ~km_max_iterations× the I/O;
	 *     strictly slower whenever the sample fits mwm (the common case). */
	uint64_t ideal_samples = Max((uint64_t)10000, (uint64_t)nlist * 256);
	uint64_t cap		   = (uint64_t)maintenance_work_mem * 1024 /
					  (dim * sizeof(float));
	if (cap < 10000)
		cap = 10000;
	*out_subsampled = ideal_samples > cap;
	uint64_t want = Min(ideal_samples, cap);
	/* nsamples/max_samples index the sample with int (reservoir state), so keep
	 * the count within int range; the mwm budget is the real bound. */
	if (want > (uint64_t)INT_MAX)
		want = (uint64_t)INT_MAX;
	bs->max_samples = (int)want;

	bs->nsamples = 0;
	bs->samples	 = MemoryContextAllocHuge(
			 CurrentMemoryContext, (size_t)bs->max_samples * dim * sizeof(float));

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
	BlockNumber			first_posting; /* leaf c's head = first_posting + c */
	float			   *pt;			   /* [dim] scratch */
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
	mkt_posting_builder_set_first_blkno(&hb, h->first_posting + leaf);
	mkt_posting_builder_finish(&hb);
	mkt_posting_builder_cleanup(&hb);
}

/* ----------------------------------------------------------------
 * Page-backed leaf refinement (streaming build)
 *
 * The streaming write trained the centroids on the maintenance_work_mem-bounded
 * sample. When that subsampled, refine the per-leaf encode reference on the full
 * table: route every row page-backed to its leaf (the same routing the scan +
 * query use), accumulate per-leaf means, and rewrite each leaf's head-page
 * pt_centroid to the full-table mean. This tightens the RaBitQ residuals (the
 * dominant recall factor) for every vector in the leaf. The accumulator is
 * tiled to a bounded ceiling (like the posting reserve), so memory stays
 * O(maintenance_work_mem) regardless of nlist; nleaves above the tile just
 * means more (re-scanned) tiles. Routing stays on the sample-trained centroid
 * pages, so a single pass reaches the fixed point (assignments do not shift).
 * ---------------------------------------------------------------- */

typedef struct RefineHeadState
{
	MktQueryState *qs;
	BlockNumber	   first_posting; /* leaf c's head = first_posting + c */
	uint32_t	   nlist;
	Dimension	   dim;
	bool			   cosine;
	MemoryContext	   tmp_ctx;
	double			  *sums; /* [tile * dim], indexed by leaf - tile_lo */
	uint64_t		  *cnts; /* [tile] */
	float			  *scratch; /* [dim] normalized copy for cosine */
	uint32_t		   tile_lo;
	uint32_t		   tile_hi;
} RefineHeadState;

static void
refine_head_cb(
		Relation	index,
		ItemPointer tid,
		Datum	   *values,
		bool	   *isnull,
		bool		tuple_is_alive,
		void	   *state)
{
	RefineHeadState *rs = (RefineHeadState *)state;

	(void)index;
	(void)tid;
	(void)tuple_is_alive;

	if (isnull[0])
		return;

	MemoryContext old_ctx = MemoryContextSwitchTo(rs->tmp_ctx);

	const float *vin = MktVectorToRef(DatumGetMktVector(values[0])).data;
	uint32_t	 n	 = mkt_query_route(
			 rs->qs, vin, 1, MKT_DISTANCE_MODE_ASYMMETRIC, NULL);
	if (n > 0)
	{
		uint32_t leaf = mkt_route_head_to_leaf(
				rs->first_posting, rs->qs->beam_results[0].posting_head);
		if (leaf >= rs->tile_lo && leaf < rs->tile_hi)
		{
			const float *v = vin;
			if (rs->cosine)
			{
				memcpy(rs->scratch, vin, (size_t)rs->dim * sizeof(float));
				mkt_l2_normalize(rs->scratch, rs->dim);
				v = rs->scratch;
			}
			double *sum = rs->sums + (size_t)(leaf - rs->tile_lo) * rs->dim;
			for (Dimension j = 0; j < rs->dim; j++)
				sum[j] += v[j];
			rs->cnts[leaf - rs->tile_lo]++;
		}
	}

	MemoryContextSwitchTo(old_ctx);
	MemoryContextReset(rs->tmp_ctx);
}

static void
serial_refine_heads(
		MktannBuildState *bs,
		SerialHeadCtx	 *headctx,
		MktQueryState	 *qs,
		BlockNumber		  first_posting,
		uint32_t		  nlist)
{
	Dimension dim = bs->params.dim;

	uint64_t cap_bytes =
			Min((uint64_t)maintenance_work_mem * 1024, (uint64_t)MaxAllocSize);
	uint64_t per_leaf = (uint64_t)dim * sizeof(double) + sizeof(uint64_t);
	uint32_t tile	  = (uint32_t)
			Min((uint64_t)nlist, Max(UINT64CONST(1), cap_bytes / per_leaf));

	RefineHeadState rs = {
			.qs			   = qs,
			.first_posting = first_posting,
			.nlist		   = nlist,
			.dim		   = dim,
			.cosine		   = (bs->params.metric == DISTANCE_COSINE),
			.tmp_ctx	   = bs->tmp_ctx,
			.sums		   = palloc((size_t)tile * dim * sizeof(double)),
			.cnts		   = palloc((size_t)tile * sizeof(uint64_t)),
			.scratch	   = palloc((size_t)dim * sizeof(float)),
	};

	for (uint32_t lo = 0; lo < nlist; lo += tile)
	{
		uint32_t hi = Min(lo + tile, nlist);
		rs.tile_lo	= lo;
		rs.tile_hi	= hi;
		memset(rs.sums, 0, (size_t)(hi - lo) * dim * sizeof(double));
		memset(rs.cnts, 0, (size_t)(hi - lo) * sizeof(uint64_t));

		table_index_build_scan(
				bs->heap, bs->index, bs->index_info, true, false,
				refine_head_cb, (void *)&rs, NULL);

		for (uint32_t l = lo; l < hi; l++)
		{
			if (rs.cnts[l - lo] == 0)
				continue; /* keep the sample-trained head for an empty leaf */
			double *sum = rs.sums + (size_t)(l - lo) * dim;
			double	inv = 1.0 / (double)rs.cnts[l - lo];
			for (Dimension j = 0; j < dim; j++)
				rs.scratch[j] = (float)(sum[j] * inv);
			serial_write_head(headctx, l, rs.scratch);
		}
	}

	pfree(rs.sums);
	pfree(rs.cnts);
	pfree(rs.scratch);
}

/*
 * Serial build fallback, mirroring do_parallel_build's role for the
 * non-parallel path. Streams the centroid tree straight to pages (no in-RAM
 * tree, no O(nlist*dim) blob), then scans the heap once through build_callback
 * (page-backed routing) to fill the posting lists.
 *
 * *out_tree is a lightweight metadata carrier (leaf/level counts) — the build
 * no longer materializes a tree. Posting-list heads are formula-derived
 * (first_posting + leaf), so no head array is returned. Returns false with
 * *out_tree == NULL when the heap has no tuples.
 */
static bool
do_serial_build(
		MktannBuildState *bs,
		MktStorage		 *storage,
		uint64_t		  rabitq_seed,
		HKMeansResult	**out_tree,
		float			**out_global_mean,
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
	bool	 subsampled	  = false;
	if (!sample_for_build(bs, &global_mean, &target_nlist, &subsampled))
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

	mkt_free(plan.leaf_counts);
	plan.leaf_counts = NULL;

	/*
	 * Head blocks are formula-derived: leaf c's head is first_posting + c, a
	 * contiguous head region of nlist pages. Pre-extend the relation to cover
	 * metadata + centroid + head region so the streaming write can place
	 * centroids at reserved blocks and each leaf's head page already exists when
	 * the write pass emits it; continuation pages are appended past the head
	 * region during mkt_posting_build_lists. No O(nlist) reserve arrays.
	 */
	mkt_storage_extend(storage, first_posting + nlist);

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
			.first_posting = first_posting,
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
			first_posting,
			first_centroid,
			serial_write_head,
			&headctx);
	pfree(headctx.pt);
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
	/*
	 * Route the build for accuracy, not query speed: use the build-time
	 * routing constants rather than the query-tuned GUCs (see
	 * MKT_BUILD_CENTROID_* in posting_build.h).
	 */
	idx_base.centroid_error_scale = MKT_BUILD_CENTROID_ERROR_SCALE;
	idx_base.centroid_beam_scale  = MKT_BUILD_CENTROID_BEAM_SCALE;
	mkt_query_state_init(&bs->qs, &idx_base, 1, MKT_SECONDARY_TOPK);

	/*
	 * When the sample was budget-limited, refine each leaf's encode reference on
	 * the full table (page-backed, bounded) before the encode scan, so residuals
	 * are taken against full-table means rather than subsample means.
	 */
	if (subsampled && mkt_leaf_refine_iters > 0)
	{
		instr_time t_ref_start;
		INSTR_TIME_SET_CURRENT(t_ref_start);
		mkt_build_report_phase(bs->prog, MKT_BUILD_PHASE_REFINE);

		SerialHeadCtx rhead = {
				.storage	   = storage,
				.rq_params	   = rq_params,
				.dim		   = dim,
				.fastscan	   = p->fastscan,
				.first_posting = first_posting,
				.pt			   = palloc((size_t)dim * sizeof(float)),
		};
		serial_refine_heads(bs, &rhead, &bs->qs, first_posting, nlist);
		pfree(rhead.pt);

		instr_time t_ref_end;
		INSTR_TIME_SET_CURRENT(t_ref_end);
		INSTR_TIME_SUBTRACT(t_ref_end, t_ref_start);
		elog(LOG,
			 "mktann: page-backed leaf refinement %.1fms -- structure from a "
			 "%d-sample subsample, %u leaf encode references refined on the full "
			 "table",
			 INSTR_TIME_GET_MILLISEC(t_ref_end),
			 bs->max_samples,
			 nlist);
	}

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
			first_posting,
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
			first_posting);
	bs->sorter = NULL; /* ended by mkt_posting_build_lists */

	bs->indtuples  = bs->route.indtuples;
	bs->soar_dupes = bs->route.soar_dupes;
	mkt_build_route_ctx_cleanup(&bs->route);
	mkt_query_state_cleanup(&bs->qs);

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
			first_posting,
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

	*out_tree		 = meta_tree;
	*out_global_mean = global_mean;
	*out_heap_tuples = heap_tuples;
	*out_indtuples	 = bs->indtuples;
	*out_soar_dupes	 = bs->soar_dupes;
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

	HKMeansResult *tree		   = NULL;
	double		   heap_tuples = 0;
	double		   indtuples   = 0;
	double		   soar_dupes  = 0;
	float		  *global_mean = NULL;
	/* Posting-area start block, surfaced by the parallel build so the finalize
	 * can record it in the metadata page (vacuum skips the centroid region). */
	BlockNumber meta_first_posting = InvalidBlockNumber;

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
				&parallel_centroids,
				/* Surfaced for the metadata page: the posting-area start block
				 * lets vacuum skip the centroid region (main's write_meta_page). */
				&meta_first_posting);

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

			BlockNumber fc = 1;

			if (global_mean == NULL)
			{
				global_mean = palloc(dim * sizeof(float));
				mkt_vector_mean(
						hk_leaf_centroids(tree), nlist, dim, global_mean);
				if (p->metric == DISTANCE_COSINE)
					mkt_l2_normalize(global_mean, dim);
			}

			/* Both remaining build paths write their own centroid + head pages
			 * before the scan (page-backed routing); the finalize only needs the
			 * metadata page here. first_posting comes from the build (via
			 * out_first_posting) and is stored so vacuum can skip the centroid
			 * region. (parallel_centroids is always set when this runs — the
			 * serial path sets centroids_finalized and skips it.) */
			(void)parallel_centroids;

			write_meta_page(
					&storage.base,
					dim,
					(uint8_t)tree->nlevels,
					(uint8_t)p->fan_out,
					fc,
					meta_first_posting,
					0,
					nlist,
					p->centroid_format,
					p->metric,
					rabitq_seed,
					global_mean);

			Page			page = mkt_storage_write_page(&storage.base, 0);
			MktannMetaPage *meta = (MktannMetaPage *)PageGetSpecialPointer(
					page);
			meta->ntuples = (uint32_t)indtuples;
			if (p->fastscan)
				meta->flags |= MKT_META_FLAG_FASTSCAN;
			mkt_storage_commit_page(&storage.base, 0);
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
