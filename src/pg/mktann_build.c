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
 *   6. Write placeholder metadata page, initialize posting builders
 *   7. Single heap scan: assign via tree descent, stream into
 *      posting builders
 *   8. Finish builders, write centroid pages with posting heads
 *   9. Update metadata page with final block layout
 *  10. WAL-log all pages
 *
 * Posting lists are always RaBitQ-encoded regardless of centroid
 * format. The centroid format only affects centroid page encoding
 * (exact float/half for routing, or RaBitQ for compressed routing).
 *
 * Block layout:
 *   Block 0:        Metadata page
 *   Blocks 1..N:    Posting pages (streamed during scan)
 *   Blocks N+1..C:  Centroid pages (written after scan)
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
#include "index/posting_build.h"
#include "index/posting_page.h"
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

typedef struct MktannBuildParams
{
	Dimension		  dim;
	DistanceMetric	  metric;
	MktCentroidFormat centroid_format;
	uint32_t		  nlist;
	uint32_t		  fan_out;
} MktannBuildParams;

typedef struct MktannBuildState
{
	MktannBuildParams params;

	/* Tree for centroid assignment */
	const HKMeansResult *tree;

	double indtuples; /* total count */

	/* Posting list builders (initialized before scan) */
	MktPostingBuilder *builders; /* [nlist] */
	float			  *norm_buf; /* [dim] reusable for cosine normalize */

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
	Dimension  dim = bs->params.dim;

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

	const MktannBuildParams *p = &bs->params;

	MktVector *vec	= DatumGetMktVector(values[0]);
	VectorRef  vref = MktVectorToRef(vec);
	Dimension  dim	= p->dim;

	/* Find nearest centroid via tree descent O(fan_out * depth) */
	Distance min_dist;
	uint32_t best_c =
			mkt_hkmeans_assign(bs->tree, vref.data, p->metric, &min_dist);

	/* Normalize for cosine so RaBitQ encodes in L2-equivalent space */
	if (p->metric == DISTANCE_COSINE)
	{
		memcpy(bs->norm_buf, vref.data, dim * sizeof(float));
		mkt_normalize(bs->norm_buf, dim);
		vref = (VectorRef){.data = bs->norm_buf, .dim = dim};
	}

	/* Stream directly into the cluster's posting builder */
	mkt_posting_builder_add(
			&bs->builders[best_c],
			vref,
			ItemPointerGetBlockNumber(tid),
			ItemPointerGetOffsetNumber(tid));

	bs->indtuples++;

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
		uint16_t		  fan_out,
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
	meta->flags			  = 0;
	meta->fan_out		  = fan_out;
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
 * Resolve build parameters from index definition and reloptions
 * ---------------------------------------------------------------- */

static void
resolve_build_params(Relation heap, Relation index, MktannBuildParams *p)
{
	/* Dimension from column typmod */
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

	/* Resolve nlist: use reloption if set, else sqrt(ntuples) */
	MktannOptions *opts = (MktannOptions *)index->rd_options;

	if (opts != NULL && opts->nlist > 0)
	{
		p->nlist = (uint32_t)opts->nlist;
	}
	else
	{
		double reltuples = RelationGetNumberOfBlocks(heap) *
						   (BLCKSZ / (sizeof(float) * dim + 32));
		p->nlist = (uint32_t)sqrt((double)Max(reltuples, 1));
		if (p->nlist < 1)
			p->nlist = 1;
		if (p->nlist > 10000)
			p->nlist = 10000;
	}

	/* Derive fan_out from nlist so hkmeans produces ~nlist leaves.
	 * Only override when fan_out is at its default. */
	if (p->fan_out == MKTANN_DEFAULT_FAN_OUT && p->nlist > p->fan_out)
	{
		uint32_t f = (uint32_t)ceil(sqrt((double)p->nlist));
		if (f > 256)
			f = (uint32_t)ceil(cbrt((double)p->nlist));
		p->fan_out = f;
	}
	else if (p->nlist <= p->fan_out)
	{
		p->fan_out = p->nlist;
	}
}

/* ----------------------------------------------------------------
 * Sample and cluster vectors
 * ---------------------------------------------------------------- */

static HKMeansResult *
run_clustering(MktannBuildState *bs, float **out_global_mean)
{
	Dimension dim	= bs->params.dim;
	uint32_t  nlist = bs->params.nlist;

	/* Sample vectors (capped to fit in MaxAllocSize) */
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

	/* Normalize samples for cosine so k-means operates in L2 space */
	if (bs->params.metric == DISTANCE_COSINE)
	{
		for (int i = 0; i < bs->nsamples; i++)
			mkt_normalize(bs->samples + (size_t)i * dim, dim);
	}

	if (bs->nsamples == 0)
		return NULL;

	/* Adjust nlist if too few samples */
	if ((uint32_t)bs->nsamples < nlist)
		nlist = (uint32_t)bs->nsamples;

	/* Run hierarchical k-means */
	KMeansOptions km_opts = MKT_KMEANS_OPTIONS_DEFAULT;
	km_opts.seed		  = 42;
	km_opts.nredo		  = 1;
	km_opts.algorithm	  = KMEANS_ALGO_LLOYD;

	HKMeansResult *tree = mkt_hkmeans_f32(
			bs->samples,
			(uint32_t)bs->nsamples,
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

	/* Compute global mean from leaf centroids */
	float *global_mean = palloc(dim * sizeof(float));
	mkt_vector_mean(tree->leaf_centroids, tree->nleaves, dim, global_mean);

	if (bs->params.metric == DISTANCE_COSINE)
		mkt_normalize(global_mean, dim);

	*out_global_mean = global_mean;
	return tree;
}

/* ----------------------------------------------------------------
 * Write centroid pages for all BFS nodes
 * ---------------------------------------------------------------- */

static void
write_centroid_tree(
		MktStorage				*storage,
		const HKMeansResult		*tree,
		const MktannBuildParams *params,
		const RaBitQParams		*rq_params,
		const float				*global_mean,
		const BlockNumber		*posting_heads,
		const BlockNumber		*node_first_blkno)
{
	Dimension dim = params->dim;

	for (uint32_t i = 0; i < tree->nnodes; i++)
	{
		HKMeansNode *node	 = &tree->nodes[i];
		bool		 is_leaf = (node->level == tree->nlevels - 1);

		uint16_t flags		 = is_leaf ? MKT_CENTROID_FLAG_LEAF : 0;
		uint16_t child_count = is_leaf ? 0 : (uint16_t)params->fan_out;

		CentroidEncoderState enc_state;
		CentroidEncoder		*encoder = centroid_encoder_init(
				&enc_state,
				params->centroid_format,
				node->centroids,
				dim,
				rq_params,
				global_mean);

		const BlockNumber *child_blks;
		if (is_leaf && posting_heads != NULL)
			child_blks = &posting_heads[node->first_leaf];
		else if (!is_leaf)
			child_blks = &node_first_blkno[node->first_child];
		else
			child_blks = NULL;

		mkt_centroid_write_pages(
				storage,
				dim,
				node->nchildren,
				params->centroid_format,
				(uint8_t)node->level,
				flags,
				child_count,
				encoder,
				child_blks);
	}
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
	float		  *global_mean;
	HKMeansResult *tree = run_clustering(&bs, &global_mean);

	if (tree == NULL)
	{
		MemoryContextSwitchTo(caller_ctx);
		MemoryContextDelete(build_ctx);
		return palloc0(sizeof(IndexBuildResult));
	}

	/* hkmeans may produce fewer leaves than requested */
	const MktannBuildParams *p	   = &bs.params;
	Dimension				 dim   = p->dim;
	uint32_t				 nlist = tree->nleaves;
	bs.params.nlist				   = nlist;

	/* 3. Single-pass streaming build: assign via tree descent,
	 * stream into posting builders — all in one heap scan. */
	uint64_t	  rabitq_seed = 42;
	RaBitQParams *rq_params	  = mkt_rabitq_create(dim, rabitq_seed);

	MktannStorage storage;
	mktann_storage_init(&storage, index, NULL, p->metric);
	storage.build_mode = true;

	/* Normalize leaf centroids for cosine. K-means ran on normalized
	 * samples so centroids are approximately unit length already;
	 * re-normalize for exactness. */
	float *ref_vecs = tree->leaf_centroids;
	if (p->metric == DISTANCE_COSINE)
	{
		for (uint32_t c = 0; c < nlist; c++)
			mkt_normalize(ref_vecs + (size_t)c * dim, dim);
	}

	/* Write placeholder metadata page (block 0) */
	write_meta_page(
			&storage.base,
			dim,
			(uint8_t)tree->nlevels,
			(uint16_t)p->fan_out,
			0,
			0, /* placeholders: first_centroid, ntuples */
			nlist,
			p->centroid_format,
			p->metric,
			rabitq_seed,
			global_mean);

	/* Initialize posting builders with contiguous page reservations.
	 *
	 * Estimate pages per cluster assuming uniform distribution, with
	 * some slack (1.2x) for skew. Pre-extend the relation so each
	 * cluster's posting pages are laid out contiguously for
	 * sequential scan and prefetch performance. Overflow pages (from
	 * clusters larger than estimated) fall back to new_page. */
	uint32_t per_page = mkt_posting_max_entries(dim);
	double	 est_rows = RelationGetNumberOfBlocks(heap) *
					  (BLCKSZ / (double)(dim * sizeof(float) + 32));
	uint32_t est_per_cluster = (uint32_t)ceil(est_rows / nlist);
	uint32_t reserve_each	 = (uint32_t)ceil(
			   (double)est_per_cluster / per_page * 1.2);
	if (reserve_each < 1)
		reserve_each = 1;

	MktPostingBuilder *builders = palloc(nlist * sizeof(MktPostingBuilder));
	for (uint32_t c = 0; c < nlist; c++)
	{
		mkt_posting_builder_init(
				&builders[c],
				&storage.base,
				rq_params,
				dim,
				c,
				ref_vecs + (size_t)c * dim);

		BlockNumber start = mkt_storage_extend(&storage.base, reserve_each);
		if (start != InvalidBlockNumber)
			mkt_posting_builder_set_reserve(&builders[c], start, reserve_each);
	}

	/* Prepare build state for scan */
	bs.tree		 = tree;
	bs.builders	 = builders;
	bs.norm_buf	 = (p->metric == DISTANCE_COSINE) ? palloc(dim * sizeof(float))
												  : NULL;
	bs.indtuples = 0;

	/* 4. Single heap scan */
	double heap_tuples = table_index_build_scan(
			heap,
			index,
			index_info,
			true,
			true,
			build_callback,
			(void *)&bs,
			NULL);

	double indtuples = bs.indtuples;

	/* 5. Finish posting builders */
	BlockNumber *posting_heads = palloc(nlist * sizeof(BlockNumber));
	for (uint32_t c = 0; c < nlist; c++)
	{
		posting_heads[c] = mkt_posting_builder_finish(&builders[c]);
		mkt_posting_builder_cleanup(&builders[c]);
	}
	pfree(builders);

	/* 6. Compute centroid page layout and update metadata */
	uint32_t max_ent = mkt_centroid_max_entries_fmt(dim, p->centroid_format);

	uint32_t	*node_pages		  = palloc(tree->nnodes * sizeof(uint32_t));
	BlockNumber *node_first_blkno = palloc(tree->nnodes * sizeof(BlockNumber));

	for (uint32_t i = 0; i < tree->nnodes; i++)
		node_pages[i] = (tree->nodes[i].nchildren + max_ent - 1) / max_ent;

	BlockNumber first_centroid = RelationGetNumberOfBlocks(index);
	for (uint32_t i = 0; i < tree->nnodes; i++)
	{
		node_first_blkno[i] = first_centroid;
		first_centroid += node_pages[i];
	}

	/* Update metadata page with final layout */
	{
		Page			page = mkt_storage_write_page(&storage.base, 0);
		MktannMetaPage *meta = (MktannMetaPage *)PageGetSpecialPointer(page);
		meta->first_centroid = node_first_blkno[0];
		meta->ntuples		 = (uint32_t)indtuples;
		mkt_storage_commit_page(&storage.base, 0);
	}

	/* 7. Write centroid pages */
	write_centroid_tree(
			&storage.base,
			tree,
			p,
			rq_params,
			global_mean,
			posting_heads,
			node_first_blkno);

	/* Cleanup */
	mkt_hkmeans_result_destroy(tree);
	pfree(node_pages);
	pfree(node_first_blkno);
	pfree(posting_heads);

	log_newpage_range(
			index, MAIN_FORKNUM, 0, RelationGetNumberOfBlocks(index), true);

	MemoryContextSwitchTo(caller_ctx);
	MemoryContextDelete(build_ctx);

	IndexBuildResult *result = palloc0(sizeof(IndexBuildResult));
	result->heap_tuples		 = heap_tuples;
	result->index_tuples	 = indtuples;
	return result;
}
