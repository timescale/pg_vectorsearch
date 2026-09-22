/*
 * build.c - Index build for prism
 *
 * Serial build phases (do_serial_build; the parallel shape lives in
 * parallel_build_leader.c and falls back here when no workers launch):
 *   1. Resolve dimension, metric, centroid format, nlist, fan_out
 *   2. Sample vectors for clustering (BlockSampler + reservoir)
 *   3. Routing-tree PLAN pass: cluster the sample once, recording each
 *      node into a spillable blob store, and size the page layout
 *   4. Pre-extend the relation for the centroid pages + posting heads
 *   5. Routing-tree WRITE pass: replay the recorded nodes and stream
 *      every centroid page and posting-head page (pt_centroid resident)
 *   6. Optional leaf refinement (prism.leaf_refine_threshold) re-centers
 *      the head encode references from the full table
 *   7. Single heap scan: route each row page-backed (the same
 *      mkt_query_route the query/insert use) into the cluster-keyed sort,
 *      then build each posting list
 *   8. Write the metadata page (final tuple count) and WAL-log
 *
 * Block layout:
 *   Block 0:        Metadata page
 *   Blocks 1..C:    Centroid pages (streamed before the scan)
 *   Blocks C+1..H:  Posting-list head pages (leaf c at C+1+c)
 *   Blocks H+1..N:  Posting continuation pages (appended during the scan)
 *
 * Memory layout:
 *   build_ctx  — all build-phase allocations; deleted in one shot
 *     tmp_ctx  — per-tuple scratch; reset after each callback
 */

#include <postgres.h>

#include <access/heaptoast.h>
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
#include "build.h"
#include "core/log.h"
#include "index/centroid_build.h"
#include "index/centroid_page.h"
#include "index/index_base.h"
#include "index/index_build.h"
#include "index/parallel_build.h"
#include "index/posting_build.h"
#include "index/posting_page.h"
#include "index/query_scan.h"
#include "meta.h"
#include "pg/bufstorage.h"
#include "quant/rabitq.h"
#include "support_pg.h"
#include "typeinfo.h"
#include "types/vec16.h"
#include "types/vec32.h"

/* ----------------------------------------------------------------
 * Build state (PrismBuildParams is in build.h, shared with the
 * parallel leader)
 * ---------------------------------------------------------------- */

typedef struct PrismBuildState
{
	PrismBuildParams params;

	double indtuples;  /* total count */
	double soar_dupes; /* replicated SOAR vectors */

	/*
	 * Posting entries are streamed into a cluster-keyed tuplesort during the
	 * heap scan, then read back grouped by cluster so the build holds only ONE
	 * posting-page builder at a time. This bounds build memory to
	 * maintenance_work_mem (the tuplesort stays in RAM until it exceeds it,
	 * then spills); per-cluster resident builders would cost O(nlist) memory.
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
	 * Page-backed assignment (routes each row the same way the query/insert
	 * do, so the in-RAM tree is not needed for the scan). qs holds the routing
	 * state; route is the shared route+encode+emit context (also used by the
	 * parallel posting workers), which references qs and the sorter.
	 */
	MktQueryState	 qs;
	MktBuildRouteCtx route;

	/*
	 * Column type bound to this index's dimension, with the conversion buffer
	 * it needs. Held on the state because both callbacks run under a tmp_ctx
	 * that is reset after every tuple.
	 */
	Vec32Access input;
} PrismBuildState;

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
	PrismBuildState *bs = (PrismBuildState *)state;

	(void)index;
	(void)tid;
	(void)tuple_is_alive;

	if (isnull[0])
		return;

	MemoryContext old_ctx = MemoryContextSwitchTo(bs->tmp_ctx);

	Dimension	 dim		= bs->params.dim;
	Vec32Ref	 vref		= vec32_read(&bs->input, values[0]);
	const float *src		= vref.data;
	const size_t vec_nbytes = (size_t)dim * sizeof(float);

	if (bs->nsamples < bs->max_samples)
	{
		memcpy(bs->samples + (size_t)bs->nsamples * dim, src, vec_nbytes);
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
			memcpy(bs->samples + (size_t)k * dim, src, vec_nbytes);
		}
		bs->rowstoskip -= 1;
		bs->nsamples++;
	}

	MemoryContextSwitchTo(old_ctx);
	MemoryContextReset(bs->tmp_ctx);
}

static void
sample_rows(PrismBuildState *bs)
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
	PrismBuildState *bs = (PrismBuildState *)state;

	(void)index;
	(void)tuple_is_alive;

	if (isnull[0])
		return;

	MemoryContext old_ctx = MemoryContextSwitchTo(bs->tmp_ctx);

	Vec32Ref vref = vec32_read(&bs->input, values[0]);

	/* Route + encode + stream via the shared page-backed helper (the parallel
	 * posting workers use the very same call). tmp_ctx is reset after every
	 * tuple, so nothing reachable from this call may allocate memory that
	 * outlives the callback: the route context's buffers (candidates, batch,
	 * encode scratch) are all preallocated for exactly this reason. */
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
		uint32_t		  ncentroid_pages,
		uint32_t		  nlist,
		MktCentroidFormat centroid_format,
		DistanceMetric	  metric,
		uint64_t		  rabitq_seed,
		bool			  fastscan,
		const float		 *global_mean)
{
	/* Block 0 must already exist (extended or new_page'd by caller).
	 * Use write_page to write in-place. */
	Page page = mkt_storage_write_page(storage, 0);

	PageInit(page, BLCKSZ, MKT_META_SIZE(dim));

	PrismMetaPage *meta	  = (PrismMetaPage *)PageGetSpecialPointer(page);
	meta->magic			  = MKT_META_MAGIC;
	meta->dim			  = dim;
	meta->nlevels		  = nlevels;
	meta->centroid_format = (uint8_t)centroid_format;
	meta->first_centroid  = first_centroid;
	meta->first_posting	  = first_posting;
	meta->ncentroid_pages = ncentroid_pages;
	meta->nlist			  = nlist;
	meta->metric		  = (uint8_t)metric;
	meta->fan_out		  = (uint8_t)fan_out;
	meta->flags			  = fastscan ? MKT_META_FLAG_FASTSCAN : 0;
	meta->reserved		  = 0;
	meta->rabitq_seed	  = rabitq_seed;

	memcpy(prism_meta_global_mean(meta), global_mean, dim * sizeof(float));

	mkt_storage_commit_page(storage, 0);
}

/* ----------------------------------------------------------------
 * Helpers: resolve metric and centroid format from opclass/relopts
 * ---------------------------------------------------------------- */

static DistanceMetric
prism_get_metric(Relation index)
{
	FmgrInfo *procinfo = index_getprocinfo(index, 1, MKT_ANN_METRIC_PROC);
	return (DistanceMetric)DatumGetInt32(
			FunctionCall1Coll(procinfo, InvalidOid, (Datum)0));
}

static MktCentroidFormat
prism_resolve_format(Relation index, DistanceMetric metric, Dimension dim)
{
	PrismOptions *opts = (PrismOptions *)index->rd_options;
	int			  cc   = (opts != NULL) ? opts->centroid_compression
										: MKT_CENTROID_COMPRESSION_AUTO;
	int			  cfs  = (opts != NULL) ? opts->centroid_fastscan
										: MKT_FASTSCAN_MODE_AUTO;

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
							"with vec32_ip_ops")));
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
	 * centroids, don't apply to inner product). The layout stores fixed
	 * 32-candidate groups, so past the dimension where a group no
	 * longer fits a page it cannot be used at all. AUTO uses it exactly
	 * where it is available; ON errors where it is not.
	 */
	bool cfs_fits = mkt_centroid_fastscan_max_groups(dim) > 0;
	if (cfs == MKT_FASTSCAN_MODE_ON)
	{
		if (metric == DISTANCE_INNER_PRODUCT)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("centroid_fastscan=on is not supported "
							"with vec32_ip_ops")));
		if (!compressed)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("centroid_fastscan=on requires "
							"centroid_compression")));
		if (!cfs_fits)
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("centroid_fastscan=on does not support "
							"%d dimensions",
							dim)));
	}
	if (compressed && cfs != MKT_FASTSCAN_MODE_OFF && cfs_fits)
		return MKT_CENTROID_FMT_FASTSCAN;

	if (compressed)
		return MKT_CENTROID_FMT_RABITQ;

	/* Uncompressed centroids follow the column type. Resolved directly rather
	 * than through the per-backend cache: there is no metadata page to
	 * populate that cache from until this build writes one. */
	return mkt_index_type_info(index)->centroid_format;
}

/*
 * Resolve the posting-page format. Like the centroid layout, a
 * fastscan posting group has a dimension-dependent fixed size; the
 * binding constraint is a cluster's first page, which also carries
 * the full-precision centroid reference. AUTO uses fastscan exactly
 * where a group fits; ON errors where it does not.
 */
static bool
prism_resolve_fastscan(Relation index, Dimension dim)
{
	PrismOptions *opts = (PrismOptions *)index->rd_options;
	int	 fs	  = (opts != NULL) ? opts->fastscan : MKT_FASTSCAN_MODE_AUTO;
	bool fits = mkt_fastscan_max_groups(dim, true) > 0;

	switch (fs)
	{
	case MKT_FASTSCAN_MODE_ON:
		if (!fits)
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("fastscan=on does not support %d dimensions",
							dim)));
		return true;
	case MKT_FASTSCAN_MODE_OFF:
		return false;
	default: /* AUTO */
		return fits;
	}
}

static uint32_t
prism_get_fan_out(Relation index)
{
	PrismOptions *opts = (PrismOptions *)index->rd_options;
	if (opts != NULL && opts->fan_out >= MKT_ANN_MIN_FAN_OUT)
		return (uint32_t)opts->fan_out;
	return MKT_ANN_DEFAULT_FAN_OUT;
}

static void
resolve_build_params(
		Relation heap, Relation index, PrismBuildParams *p, double *est_rows)
{
	Dimension dim = (Dimension)TupleDescAttr(index->rd_att, 0)->atttypmod;
	if (dim == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("column does not have dimensions")));
	if (dim > VEC32_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("column cannot have more than %d dimensions",
						VEC32_MAX_DIM)));
	if (dim > MKT_INDEX_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("prism indexes support at most %d dimensions",
						MKT_INDEX_MAX_DIM)));

	p->dim			   = dim;
	p->metric		   = prism_get_metric(index);
	p->centroid_format = prism_resolve_format(index, p->metric, dim);
	p->fan_out		   = prism_get_fan_out(index);

	/* nlist: use relopt if set, otherwise auto from sqrt(reltuples).
	 * The row estimate is computed once and surfaced to the caller (the
	 * progress-reporting total needs it too): the never-analyzed
	 * fallback samples heap pages, and sampling twice would double that
	 * cost and could even disagree with itself on a growing heap. */
	PrismOptions *opts		= (PrismOptions *)index->rd_options;
	uint32_t	  nlist_opt = (opts != NULL) ? (uint32_t)opts->nlist : 0;

	*est_rows = prism_estimate_heap_tuples(heap);

	if (nlist_opt > 0)
	{
		p->nlist = nlist_opt;
	}
	else
	{
		p->nlist = mkt_auto_nlist(*est_rows);
		if (p->nlist > MKT_ANN_MAX_NLIST)
			p->nlist = MKT_ANN_MAX_NLIST;
	}

	p->fan_out =
			mkt_auto_fan_out(p->fan_out, p->nlist, MKT_ANN_DEFAULT_FAN_OUT);

	p->kmeans_nredo = (opts != NULL && opts->kmeans_nredo > 0)
							? (uint32_t)opts->kmeans_nredo
							: 1;

	p->soar_lambda		= (opts != NULL) ? opts->soar_lambda
										 : MKT_ANN_DEFAULT_SOAR_LAMBDA;
	p->boundary_epsilon = (opts != NULL) ? opts->boundary_epsilon
										 : MKT_ANN_DEFAULT_BOUNDARY_EPSILON;
	p->fastscan			= prism_resolve_fastscan(index, dim);

	/* The exact-centroid collection size is known from the resolved
	 * shape alone, so an under-budgeted maintenance_work_mem can be
	 * reported now — at build start, before the expensive phases —
	 * with the setting that would fit. The collector still enforces
	 * the budget against the real size at collection time. */
	if (p->centroid_format == MKT_CENTROID_FMT_RABITQ ||
		p->centroid_format == MKT_CENTROID_FMT_FASTSCAN)
	{
		uint64_t expected =
				mkt_exact_centroid_expected_bytes(p->nlist, p->fan_out, dim);
		uint64_t budget = mkt_exact_centroid_budget(
				(uint64_t)maintenance_work_mem);
		if (expected > budget)
			mkt_warn(
					"maintenance_work_mem is likely too small for exact "
					"centroid collection (about " UINT64_FORMAT
					" kB needed for nlist=%u, fan_out=%u; the budget is "
					"one eighth of maintenance_work_mem): the build will "
					"fall back to estimated internal scoring — raise "
					"maintenance_work_mem to at least " UINT64_FORMAT " kB",
					expected / 1024 + 1,
					p->nlist,
					p->fan_out,
					expected * 8 / 1024 + 1);
	}
}

/*
 * Estimated heap row count. Uses the planner's reltuples when available,
 * else samples real pages and counts normal line pointers (no
 * visibility checks -- dead-but-unpruned rows count, like reltuples
 * right after a bulk delete). Feeds the automatic partition count and
 * the pg_stat_progress_create_index total.
 *
 * The density is measured rather than derived from the vector width
 * because any width formula must assume a storage strategy, and a wrong
 * assumption is catastrophic in both directions: a TOASTed column's
 * main-fork rows are ~64 bytes where the dimension predicts kilobytes
 * (undercounting rows two orders of magnitude), while a PLAIN-stored
 * vector column holds ~2 rows per page where the TOAST assumption
 * predicts ~128 (overcounting 64x -- and the automatic partition count
 * inherits the error, sharding the index into starved clusters).
 */
double
prism_estimate_heap_tuples(Relation heap)
{
	if (heap->rd_rel->reltuples > 0)
		return heap->rd_rel->reltuples;

	BlockNumber nblocks = RelationGetNumberOfBlocks(heap);
	if (nblocks == 0)
		return 0;

	uint32_t samples = Min(nblocks, 32);
	double	 rows	 = 0;

	for (uint32_t i = 0; i < samples; i++)
	{
		BlockNumber blkno = (BlockNumber)(((uint64_t)nblocks *
										   (2 * (uint64_t)i + 1)) /
										  (2 * samples));
		Buffer		buf	  = ReadBuffer(heap, blkno);

		LockBuffer(buf, BUFFER_LOCK_SHARE);
		Page page = BufferGetPage(buf);

		/* An extended-but-unwritten page has pd_lower == 0, where
		 * PageGetMaxOffsetNumber underflows; such pages hold no rows. */
		if (!PageIsNew(page))
		{
			OffsetNumber max = PageGetMaxOffsetNumber(page);
			for (OffsetNumber off = FirstOffsetNumber; off <= max; off++)
			{
				if (ItemIdIsNormal(PageGetItemId(page, off)))
					rows += 1;
			}
		}
		UnlockReleaseBuffer(buf);
	}
	return rows / samples * nblocks;
}

/* ----------------------------------------------------------------
 * Sample and cluster vectors
 * ---------------------------------------------------------------- */

/*
 * Draw the maintenance_work_mem-bounded k-means sample into bs->samples (left
 * resident for the caller to cluster + free), normalize it for cosine, and
 * resolve the leaf target against the sample size. A heap with no indexable
 * rows yields one synthetic sample and a single-cluster target, so the
 * build always proceeds.
 */
static void
sample_for_build(
		PrismBuildState *bs, uint32_t *out_nlist, bool *out_subsampled)
{
	Dimension dim	= bs->params.dim;
	uint32_t  nlist = bs->params.nlist;

	/* Bound the sample buffer by maintenance_work_mem (and MaxAllocSize). The
	 * tree is trained on this sample; leaf centroids are refined on the full
	 * table by a later (page-backed) refine pass when subsampling loses
	 * quality. */
	const uint64_t vec_nbytes = (uint64_t)dim * sizeof(float);

	uint64_t ideal_samples = Max((uint64_t)10000, (uint64_t)nlist * 256);
	uint64_t budget	   = (uint64_t)maintenance_work_mem * 1024 / vec_nbytes;
	uint64_t alloc_cap = MaxAllocSize / vec_nbytes;
	uint64_t cap	   = Min(budget, alloc_cap);
	if (cap < 10000)
		cap = 10000;
	*out_subsampled = ideal_samples > cap;
	bs->max_samples = (int)Min(ideal_samples, cap);

	bs->nsamples = 0;
	bs->samples	 = palloc((size_t)bs->max_samples * vec_nbytes);

	sample_rows(bs);

	if (bs->nsamples > bs->max_samples)
		bs->nsamples = bs->max_samples;

	if (bs->params.metric == DISTANCE_COSINE)
	{
		/*
		 * Normalize the sample onto the unit sphere, dropping zero-norm
		 * vectors: they carry no direction, so under cosine they sit at
		 * the origin -- far from every real point -- and k-means drags
		 * centroids toward them, warping the encode references (and so
		 * the distance estimates) for every vector in the affected
		 * subtree. They stay in the posting lists like any other row;
		 * they just don't get a vote on the clustering.
		 *
		 * Dropped here, by compacting the retained sample, rather than
		 * at collection time: the reservoir offers every live heap row,
		 * so a collection-time check would compute an O(dim) norm per
		 * table row to filter out a tiny fraction, while this pass
		 * touches only the <= max_samples retained rows (and copies
		 * nothing at all when no zeros were sampled). The cost is a
		 * sample budget short by the sampled-zero count -- in expectation
		 * the table's zero fraction, far below k-means run-to-run
		 * variance -- and slots spent on zeros bought nothing before
		 * this pass existed either. nlist is clamped to the compacted
		 * count below, so even a zero-heavy table degrades to fewer
		 * leaves rather than starved ones (all-zero degenerates to the
		 * synthetic single-cluster build).
		 */
		int kept = 0;
		for (int i = 0; i < bs->nsamples; i++)
		{
			float *v	   = bs->samples + (size_t)i * dim;
			float  norm_sq = mkt_l2_norm_squared(v, dim);

			if (norm_sq == 0.0f)
				continue;
			vec32_scale(v, 1.0f / sqrtf(norm_sq), v, dim);
			/* The buffer is a dense row-major matrix consumed directly
			 * by k-means, so a dropped row leaves a hole that must be
			 * closed: shift each kept row down over it. Until the first
			 * drop, kept == i and rows stay normalized in place. */
			if (kept != i)
				memcpy(bs->samples + (size_t)kept * dim,
					   v,
					   (size_t)dim * sizeof(float));
			kept++;
		}
		bs->nsamples = kept;
	}

	if (bs->nsamples == 0)
	{
		/*
		 * No indexable rows (empty table, or every row dead or NULL).
		 * Synthesize one unit-basis sample so the normal machinery
		 * emits a valid single-cluster index: the sample only shapes
		 * the leaf centroid -- the heap scan that fills posting lists
		 * still contributes nothing -- and later inserts route to that
		 * leaf like any single-cluster index (empty posting heads are
		 * a supported shape; every cluster gets one). A unit vector
		 * rather than zeros keeps the centroid safe for cosine
		 * normalization and the RaBitQ norm factors. Clustering
		 * quality after bulk loading comes from REINDEX, exactly as
		 * for any index built far below its final row count.
		 */
		bs->samples		= repalloc(bs->samples, vec_nbytes);
		bs->max_samples = 1;
		memset(bs->samples, 0, vec_nbytes);
		bs->samples[0] = 1.0f;
		bs->nsamples   = 1;
	}

	if ((uint32_t)bs->nsamples < nlist)
		nlist = (uint32_t)bs->nsamples;

	*out_nlist = nlist;
}

/* ----------------------------------------------------------------
 * Serial build
 * ---------------------------------------------------------------- */

/* ----------------------------------------------------------------
 * Page-backed leaf refinement (streaming build)
 *
 * The streaming write trained the centroids on the
 * maintenance_work_mem-bounded sample. When that subsampled, refine the
 * per-leaf encode reference on the full table: route every row page-backed to
 * its leaf (the same routing the scan + query use), accumulate per-leaf means,
 * and rewrite each leaf's head-page pt_centroid to the full-table mean. This
 * tightens the RaBitQ residuals (the dominant recall factor) for every vector
 * in the leaf. The accumulator is tiled to a bounded ceiling (like the posting
 * reserve), so memory stays O(maintenance_work_mem) regardless of nlist;
 * nleaves above the tile just means more (re-scanned) tiles. Routing stays on
 * the sample-trained centroid pages, so a single pass reaches the fixed point
 * (assignments do not shift).
 * ---------------------------------------------------------------- */

typedef struct RefineHeadState
{
	MktQueryState *qs;
	BlockNumber	   first_posting; /* leaf c's head = first_posting + c */
	uint32_t	   nlist;
	Dimension	   dim;
	bool		   cosine;
	MemoryContext  tmp_ctx;
	double		  *sums;	/* [tile * dim], indexed by leaf - tile_lo */
	uint64_t	  *cnts;	/* [tile] */
	float		  *scratch; /* [dim] normalized copy for cosine */
	uint32_t	   tile_lo;
	uint32_t	   tile_hi;
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

	const float *vin = Vec32ToRef(DatumGetVec32(values[0])).data;
	uint32_t	 idx;
	const float *v = mkt_refine_route_row(
			rs->qs,
			rs->first_posting,
			vin,
			rs->dim,
			rs->cosine,
			rs->scratch,
			rs->tile_lo,
			rs->tile_hi,
			&idx);
	if (v != NULL)
	{
		double *sum = rs->sums + (size_t)idx * rs->dim;
		for (Dimension j = 0; j < rs->dim; j++)
			sum[j] += v[j];
		rs->cnts[idx]++;
	}

	MemoryContextSwitchTo(old_ctx);
	MemoryContextReset(rs->tmp_ctx);
}

static void
serial_refine_heads(
		PrismBuildState *bs,
		MktHeadWriteCtx *headctx,
		MktQueryState	*qs,
		BlockNumber		 first_posting,
		uint32_t		 nlist)
{
	Dimension dim = bs->params.dim;

	uint64_t cap_bytes =
			Min((uint64_t)maintenance_work_mem * 1024, (uint64_t)MaxAllocSize);
	uint32_t tile = mkt_refine_tile_leaves(nlist, dim, cap_bytes);

	RefineHeadState rs = {
			.qs			   = qs,
			.first_posting = first_posting,
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
				bs->heap,
				bs->index,
				bs->index_info,
				true,
				false,
				refine_head_cb,
				(void *)&rs,
				NULL);

		mkt_refine_write_means(
				rs.sums,
				rs.cnts,
				lo,
				hi,
				dim,
				rs.scratch,
				mkt_write_leaf_head,
				headctx);
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
 * No in-RAM tree is materialized; the streamed tree's shape (leaf count +
 * depth) comes back through out_nlist/out_tree_nlevels. Posting-list heads
 * are formula-derived (first_posting + leaf), so no head array is returned.
 * A heap with no indexable rows still builds: the sampler substitutes one
 * synthetic sample and the result is a valid single-cluster index with an
 * empty posting head. The serial path writes its own metadata page; the
 * caller must not finalize again.
 */
static void
do_serial_build(
		PrismBuildState *bs,
		MktStorage		*storage,
		uint64_t		 rabitq_seed,
		uint32_t		*out_nlist,
		uint8_t			*out_tree_nlevels,
		float		   **out_global_mean,
		double			*out_heap_tuples,
		double			*out_indtuples,
		double			*out_soar_dupes)
{
	const PrismBuildParams *p	= &bs->params;
	Dimension				dim = p->dim;

	*out_nlist		  = 0;
	*out_tree_nlevels = 0;

	mkt_build_report_phase(bs->prog, MKT_BUILD_PHASE_SAMPLE);

	uint32_t target_nlist = 0;
	bool	 subsampled	  = false;
	sample_for_build(bs, &target_nlist, &subsampled);

	KMeansOptions km_opts = MKT_KMEANS_OPTIONS_DEFAULT;
	km_opts.algorithm	  = KMEANS_ALGO_LLOYD;
	km_opts.nredo		  = p->kmeans_nredo;

	/*
	 * Plan pass: cluster the sample once and discover the tree shape (leaf
	 * count, depth, centroid page count, leaf-centroid mean) without
	 * writing. Each node's clustering is recorded in a spillable blob store
	 * (BufFile-backed, so the resident cost stays one node) and the write
	 * pass below replays it instead of running k-means again -- the same
	 * single-clustering shape as the parallel build's subtree store.
	 * target_nlist (not the resolved leaf count) drives both passes.
	 */
	mkt_build_report_phase(bs->prog, MKT_BUILD_PHASE_KMEANS);
	MktBlobStore	 *node_store = mkt_pbuild_blobstore_begin();
	MktStreamTreePlan plan;
	if (!mkt_routing_tree_plan(
				bs->samples,
				(uint32_t)bs->nsamples,
				dim,
				target_nlist,
				p->fan_out,
				p->metric,
				p->centroid_format,
				&km_opts,
				node_store,
				&plan))
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("hierarchical k-means failed")));
	mkt_pbuild_blobstore_rewind(node_store);

	uint32_t nlist	 = plan.nleaves;
	bs->params.nlist = nlist;

	/*
	 * Re-center the encoder on the leaf-centroid mean the plan pass reported
	 * (the write pass reproduces the identical tree). The in-RAM-tree build
	 * centers on the mean of the leaf centroids, not the per-vector sample
	 * mean, and the quantization quality of every centroid and posting code
	 * depends on this anchor.
	 */
	/* The streamed pages encode against the leaf-centroid mean the plan
	 * pass accumulated. */
	const size_t vec_nbytes = (size_t)dim * sizeof(float);

	float *global_mean = palloc(vec_nbytes);
	memcpy(global_mean, plan.leaf_mean, vec_nbytes);
	if (p->metric == DISTANCE_COSINE)
		mkt_l2_normalize(global_mean, dim);
	mkt_free(plan.leaf_mean);
	plan.leaf_mean = NULL;

	mkt_build_report_phase(bs->prog, MKT_BUILD_PHASE_SETUP);
	RaBitQParams *rq_params = mkt_rabitq_create(dim, rabitq_seed);

	/* Centroid area is [first_centroid, first_posting); block 0 is metadata.
	 */
	BlockNumber first_centroid = MKT_FIRST_CENTROID_BLKNO;
	BlockNumber first_posting  = first_centroid + plan.centroid_pages;

	/*
	 * Head blocks are formula-derived: leaf c's head is first_posting + c, a
	 * contiguous head region of nlist pages. Pre-extend the relation to cover
	 * metadata + centroid + head region so the streaming write can place
	 * centroids at reserved blocks and each leaf's head page already exists
	 * when the write pass emits it; continuation pages are appended past the
	 * head region during mkt_posting_build_lists. No O(nlist) reserve arrays.
	 */
	mkt_build_reserve_layout(storage, first_posting + nlist);

	/*
	 * Write pass: stream the centroid pages (reserved blocks, post-order, root
	 * last) and, per leaf, its head page carrying pt_centroid. Both build and
	 * query then route page-backed over these centroid pages; the metadata
	 * page is written after the scan, when the tuple count is final.
	 */
	mkt_build_report_phase(bs->prog, MKT_BUILD_PHASE_CENTROID);

	/* Exact internal-centroid collection for the build descent (see
	 * MktExactCentroidCollector in index_build.h). */
	MktExactCentroidCollector  exact_centroids = {0};
	MktExactCentroidCollector *collector =
			mkt_exact_centroid_enabled(plan.nlevels, p->centroid_format)
					? &exact_centroids
					: NULL;
	if (collector != NULL)
		mkt_exact_centroid_collector_init(
				collector,
				dim,
				p->centroid_format,
				first_centroid,
				plan.centroid_pages,
				mkt_exact_centroid_budget((uint64_t)maintenance_work_mem),
				mkt_exact_centroid_expected_slots(p->nlist, p->fan_out));

	MktHeadWriteCtx headctx;
	mkt_head_write_ctx_init(
			&headctx, storage, rq_params, dim, p->fastscan, first_posting);
	BlockNumber root = mkt_routing_tree_write(
			storage,
			(uint32_t)bs->nsamples,
			dim,
			p->metric,
			target_nlist,
			p->fan_out,
			p->centroid_format,
			rq_params,
			global_mean,
			node_store,
			first_posting,
			first_centroid,
			mkt_write_leaf_head,
			&headctx,
			collector);
	mkt_pbuild_blobstore_end(node_store);
	mkt_head_write_ctx_cleanup(&headctx);
	pfree(bs->samples);
	bs->samples = NULL;
	if (root == InvalidBlockNumber)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("streaming centroid write failed")));

	/*
	 * Page-backed routing: build a MktIndexBase from the just-written index so
	 * the scan routes each row exactly as the query/insert do (mkt_query_route
	 * over the centroid pages). Assignment uses no in-RAM tree. The
	 * base is stack-local but outlives the scan (all within this function); qs
	 * holds it by pointer until mkt_query_state_cleanup below.
	 */
	MktIndexBase idx_base;
	/* Route the build for accuracy, not query speed: the build-time routing
	 * constants rather than the query-tuned GUCs (see MKT_BUILD_CENTROID_*
	 * in posting_build.h). */
	mkt_build_router_base_init(
			&idx_base,
			rq_params,
			storage,
			dim,
			(uint8_t)plan.nlevels,
			root,
			p->metric,
			p->centroid_format,
			prism_fastscan_bits,
			MKT_BUILD_CENTROID_ERROR_SCALE,
			MKT_BUILD_CENTROID_BEAM_SCALE,
			bs->params.fan_out,
			nlist,
			rabitq_seed,
			global_mean,
			palloc(vec_nbytes));
	/* Build-only accuracy hook: score the internal tree levels against the
	 * exact centroids collected during the streaming write (the query and
	 * insert paths never set this; see MktExactInternalCentroids). */
	MktExactInternalCentroids exact_view;
	if (collector != NULL)
	{
		mkt_exact_centroid_view(collector, &exact_view);
		idx_base.exact_internal = &exact_view;
	}
	mkt_query_state_init(&bs->qs, &idx_base, 1, MKT_SECONDARY_TOPK);

	/*
	 * When the sample was budget-limited AND the leaves are sample-thin
	 * (below prism.leaf_refine_threshold samples per leaf -- a leaf's encode
	 * reference is a sample mean whose error shrinks with its count, so
	 * well-fed leaves gain nothing from the extra scan), refine each leaf's
	 * encode reference on the full table (page-backed, bounded) before the
	 * encode scan.
	 */
	if (subsampled && bs->nsamples >= bs->max_samples &&
		mkt_leaf_refine_threshold > 0 &&
		(uint32_t)bs->nsamples / Max(nlist, 1) <
				(uint32_t)mkt_leaf_refine_threshold)
	{
		instr_time t_ref_start;
		INSTR_TIME_SET_CURRENT(t_ref_start);
		mkt_build_report_phase(bs->prog, MKT_BUILD_PHASE_REFINE);

		MktHeadWriteCtx rhead;
		mkt_head_write_ctx_init(
				&rhead, storage, rq_params, dim, p->fastscan, first_posting);
		serial_refine_heads(bs, &rhead, &bs->qs, first_posting, nlist);
		mkt_head_write_ctx_cleanup(&rhead);

		instr_time t_ref_end;
		INSTR_TIME_SET_CURRENT(t_ref_end);
		INSTR_TIME_SUBTRACT(t_ref_end, t_ref_start);
		elog(LOG,
			 "prism: page-backed leaf refinement %.1fms -- structure from a "
			 "%d-sample subsample, %u leaf encode references refined on the "
			 "full "
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
	 * so no O(nlist) array of resident builders exists.
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

	/* Shared route+encode+emit context (same helper the parallel workers use).
	 */
	mkt_build_route_ctx_init(
			&bs->route,
			&bs->qs,
			bs->sorter,
			rq_params,
			storage,
			first_posting,
			dim,
			p->soar_lambda,
			p->boundary_epsilon);

	/* Reports the scan phase and fires the "prism-build-load" test hook (see
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
			first_posting);
	bs->sorter = NULL; /* ended by mkt_posting_build_lists */

	bs->indtuples  = bs->route.indtuples;
	bs->soar_dupes = bs->route.soar_dupes;
	mkt_build_route_ctx_cleanup(&bs->route);
	mkt_query_state_cleanup(&bs->qs);
	if (collector != NULL)
		mkt_exact_centroid_collector_cleanup(collector);

	elog(LOG,
		 "prism: serial build scan %.1fms, "
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
			plan.centroid_pages,
			nlist,
			p->centroid_format,
			p->metric,
			rabitq_seed,
			p->fastscan,
			global_mean);
	*out_nlist		  = nlist;
	*out_tree_nlevels = (uint8_t)plan.nlevels;
	*out_global_mean  = global_mean;
	*out_heap_tuples  = heap_tuples;
	*out_indtuples	  = bs->indtuples;
	*out_soar_dupes	  = bs->soar_dupes;
}

/* ----------------------------------------------------------------
 * Main build entry point
 * ---------------------------------------------------------------- */

IndexBuildResult *
prism_build(Relation heap, Relation index, struct IndexInfo *index_info)
{
	/*
	 * Refuse expression indexes.
	 *
	 * Rerank fetches the heap tuple and reads the indexed value with
	 * slot_getattr(slot, indkey.values[0]) -- and for an expression index
	 * PostgreSQL stores 0 there, keeping the expression tree in indexprs
	 * instead. attnum 0 reads tts_values[-1] and hands the garbage to
	 * pg_detoast_datum, which segfaults the backend (measured: the crash is
	 * in pg_rerank_readstream, not theoretical).
	 *
	 * Supporting them means doing at rerank time what the build already does
	 * -- evaluating the index expression over the fetched tuple -- which
	 * needs an ExprState and ExprContext live across the whole rerank loop
	 * and an evaluation per candidate, in the hottest path there is. Worth
	 * doing; not worth smuggling in as a crash fix.
	 */
	if (index->rd_index->indkey.values[0] == InvalidAttrNumber)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("prism indexes do not support index expressions"),
				 errhint("Index the column directly, or materialize the "
						 "expression into a column and index that.")));

	MemoryContext caller_ctx = CurrentMemoryContext;
	MemoryContext build_ctx	 = AllocSetContextCreate(
			 CurrentMemoryContext, "prism build", ALLOCSET_DEFAULT_SIZES);
	MemoryContextSwitchTo(build_ctx);

	/* 1. Initialize build state and resolve parameters */
	PrismBuildState bs = {0};
	bs.heap			   = heap;
	bs.index		   = index;
	bs.index_info	   = index_info;
	bs.build_ctx	   = build_ctx;
	bs.tmp_ctx		   = AllocSetContextCreate(
			build_ctx, "prism build tuple", ALLOCSET_DEFAULT_SIZES);

	double est_rows;
	resolve_build_params(heap, index, &bs.params, &est_rows);

	const PrismBuildParams *p			= &bs.params;
	Dimension				dim			= p->dim;
	uint64_t				rabitq_seed = MKT_RABITQ_BUILD_SEED;

	/* Resolved once, directly -- no metadata page exists yet for the
	 * per-backend cache to read. The per-tuple callbacks follow the pointer.
	 */
	bs.input = vec32_access(mkt_index_type_info(index), dim, build_ctx);

	/*
	 * Build introspection: one reporting context the serial and parallel paths
	 * share, so pg_stat_progress_create_index advances through the same phases
	 * either way and (when prism.log_build_stats is on) per-phase stats land
	 * in the server log.
	 */
	MktBuildStats	 stats = {0};
	MktBuildProgress prog;
	mkt_build_progress_begin(
			&prog,
			index_info->ii_ParallelWorkers > 0,
			prism_log_build_stats,
			build_ctx,
			&stats,
			est_rows);
	bs.prog = &prog;

	MktPgStorage storage;
	mkt_pg_storage_init(&storage, index, NULL, p->metric);
	storage.build_mode = true;

	double heap_tuples = 0;
	double indtuples   = 0;
	double soar_dupes  = 0;
	float *global_mean = NULL;
	/* Posting-area start block, surfaced by the parallel build so the finalize
	 * can record it in the metadata page (vacuum skips the centroid region).
	 */
	BlockNumber meta_first_posting = InvalidBlockNumber;

	/* The streamed tree's shape, from whichever build path ran (0 leaves =
	 * empty heap). The serial path writes its own metadata page; the
	 * parallel path leaves it for the finalize below (it needs the final
	 * tuple count). */
	uint32_t built_nlist   = 0;
	uint8_t	 built_nlevels = 0;

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
		 * a leaf count at allocation time — but the real count is only known
		 * after k-means clusters the sample, which happens inside the
		 * workers. hkmeans targets `nlist` leaves at this fan_out but can
		 * produce up to fan_out^nlevels of them, so we size every per-leaf
		 * region for that worst-case upper bound here, then narrow to the
		 * actual built leaf count below.
		 *
		 * nlist and fan_out are already resolved (resolve_build_params).
		 */
		uint32_t requested_nlist = p->nlist;
		uint32_t max_nlist		 = mkt_max_nlist(p->nlist, p->fan_out);

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
		 * k-means, subtrees, refine, setup, scan, posting) and fires
		 * the per-phase test hooks, so no phase is set here. */
		did_parallel = do_parallel_build(
				heap,
				index,
				index_info,
				&cfg,
				&storage.base,
				&prog,
				&built_nlist,
				&built_nlevels,
				&heap_tuples,
				&indtuples,
				&soar_dupes,
				/* do_parallel_build routes page-backed: it wrote the centroid
				 * + head pages into the index before the scan and returns the
				 * global mean it used (so the metadata write below matches).
				 * It does NOT write the metadata page (needs the final tuple
				 * count), so the finalize below still runs. */
				&global_mean,
				/* Surfaced for the metadata page: the posting-area start block
				 * lets vacuum skip the centroid region without scanning it. */
				&meta_first_posting);

		if (did_parallel && built_nlist == 0)
		{
			/*
			 * The parallel path found no indexable rows (a non-empty heap
			 * whose rows are all dead or NULL can reach it). It wrote no
			 * pages for the zero-leaf layout, so the serial fallback below
			 * -- whose sampler substitutes a synthetic sample and emits a
			 * valid single-cluster index -- starts from a clean relation.
			 * Enforced at runtime: falling back onto a relation the
			 * parallel path already wrote to would corrupt the index.
			 */
			if (RelationGetNumberOfBlocks(index) > 1)
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("parallel index build reported no leaves "
								"but wrote %u pages",
								RelationGetNumberOfBlocks(index))));

			/* The zero-leaf return leaves out_global_mean NULL today;
			 * free defensively so a future change to that contract
			 * cannot leak the parallel mean when the serial build
			 * replaces it. */
			if (global_mean != NULL)
			{
				pfree(global_mean);
				global_mean = NULL;
			}
			did_parallel = false;
		}

		if (did_parallel && built_nlist > 0)
			bs.params.nlist = built_nlist;
		else if (!did_parallel)
		{
			/* The worst-case bound exists only to size the parallel DSM
			 * regions. The serial fallback clusters from scratch, so it must
			 * target the requested partition count, not the sizing bound. */
			bs.params.nlist = requested_nlist;
		}
	}

	if (!did_parallel)
	{
		/* Serial fallback: sample, cluster, build. Always emits at least
		 * one leaf, even when the heap has no indexable tuples. */
		do_serial_build(
				&bs,
				&storage.base,
				rabitq_seed,
				&built_nlist,
				&built_nlevels,
				&global_mean,
				&heap_tuples,
				&indtuples,
				&soar_dupes);
	}

	/* Both paths produce at least one leaf now: a heap with no indexable
	 * rows builds a single-cluster index around a synthetic centroid (see
	 * sample_for_build), so inserts route and scans return empty.
	 * Enforced at runtime: finalizing a zero-leaf layout would produce a
	 * broken index. */
	if (built_nlist == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("index build produced no clusters")));

	if (soar_dupes > 0)
		elog(LOG,
			 "prism: replicated %.0f vectors "
			 "(%.1f%% of %.0f, lambda=%.4g)",
			 soar_dupes,
			 100.0 * soar_dupes / indtuples,
			 indtuples,
			 bs.params.soar_lambda);

	/* Finalize: write centroid pages + metadata (unless the build path already
	 * did — the serial page-backed path writes them itself), then WAL-log the
	 * whole index. */
	{
		if (did_parallel)
		{
			/* The parallel build wrote its centroid + head pages before the
			 * scan but left the metadata page for here (it needs the final
			 * tuple count); the serial path wrote its own. first_posting
			 * comes from the build and lets vacuum skip the centroid region
			 * without scanning it. */
			BlockNumber fc = 1;

			Assert(global_mean != NULL);

			write_meta_page(
					&storage.base,
					dim,
					built_nlevels,
					(uint8_t)p->fan_out,
					fc,
					meta_first_posting,
					meta_first_posting - MKT_FIRST_CENTROID_BLKNO,
					built_nlist,
					p->centroid_format,
					p->metric,
					rabitq_seed,
					p->fastscan,
					global_mean);
		}
		mkt_build_report_phase(&prog, MKT_BUILD_PHASE_WAL);
		/*
		 * WAL-log the built pages only when the relation is WAL-logged, the
		 * same guard every core index AM (GiST/GIN/SP-GiST) applies here.
		 * Temp relations and same-transaction builds under wal_level=minimal
		 * skip this: emitting WAL for them is wasted volume, and doing so
		 * against a session-local temp relfilenode is something no core AM
		 * does. (Unlogged tables never reach this path -- their build is
		 * rejected up front in prism_buildempty.)
		 *
		 * page_std = false, and it has to be. The flag promises the standard
		 * page layout, which lets the full-page image omit everything between
		 * pd_lower and pd_upper as free space. prism keeps its entries there
		 * -- pd_lower stays at SizeOfPageHeaderData and pd_upper at pd_special
		 * on a posting page -- so a standard image would carry the header and
		 * the opaque and nothing else. The primary would be fine, since its
		 * pages are already on disk; a streaming standby, which has only the
		 * WAL, would restore every page with its entries zeroed while the
		 * opaque still reported them present. See t/001_replication.pl.
		 *
		 * The insert path solves the same problem the other way, by covering
		 * the hole (pd_lower = pd_upper) before GenericXLog; that is not an
		 * option here because centroid pages use pd_lower for their own
		 * capacity accounting.
		 */
		if (RelationNeedsWAL(index))
			log_newpage_range(
					index,
					MAIN_FORKNUM,
					0,
					RelationGetNumberOfBlocks(index),
					false);
	}

	/* Flush the final phase timing + emit the build summary (heap_ctx is read
	 * here, so this must run before build_ctx is deleted below). */
	mkt_build_progress_end(&prog);

	/* Cleanup */

	MemoryContextSwitchTo(caller_ctx);
	MemoryContextDelete(build_ctx);

	IndexBuildResult *result = palloc0(sizeof(IndexBuildResult));
	result->heap_tuples		 = heap_tuples;
	result->index_tuples	 = indtuples;
	return result;
}

char *
prism_buildphasename(int64 phasenum)
{
	/* Single source of truth for the phase names (index/build_progress.c),
	 * shared with the build logs so the two never drift. */
	return unconstify(char *, mkt_build_phase_name((int)phasenum));
}
