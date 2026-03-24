/*
 * mktann_build.c - Index build for mktann
 *
 * Build phases:
 *   1. Determine dimension from index column typmod
 *   2. Resolve distance metric, centroid format, fan_out from relopts
 *   3. Sample vectors for clustering (BlockSampler + reservoir)
 *   4. Run hierarchical k-means (mkt_hkmeans_f32)
 *   5. Compute global mean from leaf centroids
 *   6. Full heap scan to assign vectors to leaf centroids
 *   7. Pre-compute block numbers for BFS tree nodes
 *   8. Encode + write centroid pages per BFS node
 *   9. Write metadata page (block 0) with tree parameters
 *  10. WAL-log all pages
 *
 * Memory layout:
 *   build_ctx  — all build-phase allocations; deleted in one shot
 *     tmp_ctx  — per-tuple scratch; reset after each callback
 */

#include <postgres.h>

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
#include "mkt_halfvec.h"
#include "mkt_pg.h"
#include "mkt_vector.h"
#include "mktann_build.h"
#include "mktann_meta.h"
#include "mktann_storage.h"
#include "quant/rabitq.h"

/* ----------------------------------------------------------------
 * Build state
 * ---------------------------------------------------------------- */

typedef struct MktannBuildState
{
	/* Shared across workers (read-only after k-means) */
	float		  *centroids; /* [nlist * dim], row-major */
	Dimension	   dim;
	uint32_t	   nlist;
	DistanceMetric metric;

	/* Per-worker accumulators */
	double indtuples; /* count */

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
	Dimension  dim = bs->dim;

	if (bs->nsamples < bs->max_samples)
	{
		/* Fill phase */
		memcpy(bs->samples + (size_t)bs->nsamples * dim,
			   src,
			   dim * sizeof(float));
		bs->nsamples++;
	}
	else
	{
		/* Reservoir replacement */
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
 * Full scan callback — assign vectors to leaf centroids
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
	(void)tid;
	(void)values;
	(void)tuple_is_alive;

	if (isnull[0])
		return;

	/* TODO: assign vector to nearest centroid for posting list build */
	bs->indtuples++;
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

	/* Initialize as empty page with enough special space */
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
	meta->fan_out		  = fan_out;
	memset(meta->reserved, 0, sizeof(meta->reserved));
	meta->rabitq_seed = rabitq_seed;

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

	/* Uncompressed: format matches heap column type */
	Oid col_type = TupleDescAttr(index->rd_att, 0)->atttypid;
	if (col_type == mkt_halfvec_type_oid())
		return MKT_CENTROID_FMT_HALF;
	return MKT_CENTROID_FMT_FLOAT;
}

/* ----------------------------------------------------------------
 * Helpers: resolve fan_out from reloptions
 * ---------------------------------------------------------------- */

static uint32_t
mktann_get_fan_out(Relation index)
{
	MktannOptions *opts = (MktannOptions *)index->rd_options;
	if (opts != NULL && opts->fan_out >= MKTANN_MIN_FAN_OUT)
		return (uint32_t)opts->fan_out;
	return MKTANN_DEFAULT_FAN_OUT;
}

/* ----------------------------------------------------------------
 * Main build entry point
 * ---------------------------------------------------------------- */

IndexBuildResult *
mktann_build(Relation heap, Relation index, struct IndexInfo *index_info)
{
	MemoryContext caller_ctx = CurrentMemoryContext;

	/* All build-phase allocations go into build_ctx.
	 * MemoryContextDelete(build_ctx) frees everything at the end. */
	MemoryContext build_ctx = AllocSetContextCreate(
			CurrentMemoryContext, "mktann build", ALLOCSET_DEFAULT_SIZES);
	MemoryContextSwitchTo(build_ctx);

	MktannBuildState bs = {0};

	/* 1. Determine dimension from typmod */
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

	/* 2. Resolve metric, centroid format, and fan_out */
	DistanceMetric	  metric		  = mktann_get_metric(index);
	MktCentroidFormat centroid_format = mktann_resolve_format(index, metric);
	uint32_t		  fan_out		  = mktann_get_fan_out(index);

	bs.dim		  = dim;
	bs.metric	  = metric;
	bs.heap		  = heap;
	bs.index	  = index;
	bs.index_info = index_info;
	bs.build_ctx  = build_ctx;
	bs.tmp_ctx	  = AllocSetContextCreate(
			   build_ctx, "mktann build tuple", ALLOCSET_DEFAULT_SIZES);

	/* 3. Compute nlist = sqrt(ntuples), at least 1 */
	double reltuples = RelationGetNumberOfBlocks(heap) *
					   (BLCKSZ / (sizeof(float) * dim + 32));
	uint32_t nlist = (uint32_t)sqrt((double)Max(reltuples, 1));
	if (nlist < 1)
		nlist = 1;
	if (nlist > 10000)
		nlist = 10000;
	bs.nlist = nlist;

	/* 4. Sample vectors for k-means */
	bs.max_samples = Max(10000, (int)(nlist * 50));
	bs.nsamples	   = 0;
	bs.samples	   = palloc((size_t)bs.max_samples * dim * sizeof(float));

	sample_rows(&bs);

	if (bs.nsamples == 0)
	{
		/* Empty table — nothing to index */
		MemoryContextSwitchTo(caller_ctx);
		MemoryContextDelete(build_ctx);

		IndexBuildResult *result = palloc0(sizeof(IndexBuildResult));
		return result;
	}

	/* Adjust nlist if we have too few samples */
	if ((uint32_t)bs.nsamples < nlist)
	{
		nlist	 = (uint32_t)bs.nsamples;
		bs.nlist = nlist;
	}

	/* 5. Run hierarchical k-means
	 *
	 * hkmeans allocates via mkt_alloc (= palloc), so all its
	 * internal state lands in build_ctx and gets freed with it. */
	KMeansOptions km_opts = MKT_KMEANS_OPTIONS_DEFAULT;
	km_opts.seed		  = 42;
	km_opts.nredo		  = 1;
	km_opts.algorithm	  = KMEANS_ALGO_LLOYD;

	HKMeansResult *tree = mkt_hkmeans_f32(
			bs.samples,
			(uint32_t)bs.nsamples,
			dim,
			nlist,
			fan_out,
			metric,
			&km_opts);

	if (tree == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("hierarchical k-means failed")));

	nlist	 = tree->nleaves;
	bs.nlist = nlist;

	/* Samples no longer needed — reclaim memory */
	pfree(bs.samples);
	bs.samples = NULL;

	/* 6. Use leaf centroids from tree and compute global mean */
	bs.centroids = tree->leaf_centroids;

	float *global_mean = palloc(dim * sizeof(float));
	mkt_vector_mean(tree->leaf_centroids, nlist, dim, global_mean);

	/* 7. Full heap scan — assign vectors to leaf centroids */
	bs.indtuples = 0;

	double heap_tuples = table_index_build_scan(
			heap,
			index,
			index_info,
			true,
			true,
			build_callback,
			(void *)&bs,
			NULL);

	/* 8. Pre-compute block numbers for each BFS node.
	 *
	 * ReadBufferExtended(P_NEW) allocates blocks sequentially.
	 * BFS order naturally matches. Block 0 is the meta page;
	 * centroid pages start at block 1. */
	uint32_t max_ent = mkt_centroid_max_entries_fmt(dim, centroid_format);

	BlockNumber *node_first_blkno = palloc(tree->nnodes * sizeof(BlockNumber));

	mkt_compute_centroid_layout(
			tree, max_ent, 1 /* block 0 = meta */, node_first_blkno);

	/* 9. Write index pages */
	uint64_t	  rabitq_seed = 42;
	RaBitQParams *rq_params	  = NULL;
	if (centroid_format == MKT_CENTROID_FMT_RABITQ)
		rq_params = mkt_rabitq_create(dim, rabitq_seed);

	MktannStorage storage;
	mktann_storage_init(&storage, index, NULL, metric);

	/* Block 0: metadata */
	write_meta_page(
			&storage.base,
			dim,
			(uint8_t)tree->nlevels,
			(uint8_t)fan_out,
			node_first_blkno[0],
			(uint32_t)bs.indtuples,
			nlist,
			centroid_format,
			metric,
			rabitq_seed,
			global_mean);

	/* Write centroid pages for all BFS nodes */
	mkt_write_centroid_tree(
			&storage.base,
			tree,
			dim,
			fan_out,
			centroid_format,
			rq_params,
			global_mean,
			NULL, /* no posting heads on main yet */
			node_first_blkno);

	/* 10. WAL-log all pages */
	log_newpage_range(
			index, MAIN_FORKNUM, 0, RelationGetNumberOfBlocks(index), true);

	/* 11. Cleanup — delete build context, return in caller ctx */
	double indtuples = bs.indtuples;

	mkt_hkmeans_result_destroy(tree);
	pfree(node_first_blkno);

	MemoryContextSwitchTo(caller_ctx);
	MemoryContextDelete(build_ctx);

	IndexBuildResult *result = palloc0(sizeof(IndexBuildResult));
	result->heap_tuples		 = heap_tuples;
	result->index_tuples	 = indtuples;
	return result;
}
