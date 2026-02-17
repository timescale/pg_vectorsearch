/*
 * mktann_build.c - Index build for mktann
 *
 * Build phases:
 *   1. Determine dimension from index column typmod
 *   2. Resolve distance metric from opclass, centroid format from relopt
 *   3. Sample vectors for k-means (BlockSampler + reservoir)
 *   4. Run k-means clustering
 *   5. Compute global mean of centroids
 *   6. Full heap scan to assign vectors and find medoids
 *   7. Encode centroids (RaBitQ, float32, or float16)
 *   8. Write metadata page (block 0) and centroid pages
 *   9. WAL-log all pages
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
#include "algo/kmeans.h"
#include "index/centroid_build.h"
#include "index/centroid_page.h"
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
	ItemPointerData *medoid_tids;  /* [nlist] best medoid per cluster */
	float			*medoid_dists; /* [nlist] distance of best medoid */
	double			 indtuples;	   /* count */

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
 * Full scan callback — assign vectors, find medoids
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
	Dimension  dim	= bs->dim;

	/* Find nearest centroid */
	float	 min_dist = INFINITY;
	uint32_t best_c	  = 0;

	for (uint32_t c = 0; c < bs->nlist; c++)
	{
		VectorRef cref = {
				.data = bs->centroids + (size_t)c * dim,
				.dim  = dim,
		};
		float d = mkt_distance(vref, cref, bs->metric);
		if (d < min_dist)
		{
			min_dist = d;
			best_c	 = c;
		}
	}

	/* Track medoid (closest vector to centroid) — only for RaBitQ */
	if (bs->medoid_tids != NULL && min_dist < bs->medoid_dists[best_c])
	{
		bs->medoid_dists[best_c] = min_dist;
		bs->medoid_tids[best_c]	 = *tid;
	}

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

	/* 2. Resolve metric and centroid format */
	DistanceMetric	  metric		  = mktann_get_metric(index);
	MktCentroidFormat centroid_format = mktann_resolve_format(index, metric);

	bs.dim		  = dim;
	bs.metric	  = metric;
	bs.heap		  = heap;
	bs.index	  = index;
	bs.index_info = index_info;
	bs.build_ctx  = build_ctx;
	bs.tmp_ctx	  = AllocSetContextCreate(
			   build_ctx, "mktann build tuple", ALLOCSET_DEFAULT_SIZES);

	/* 2. Compute nlist = sqrt(ntuples), at least 1 */
	double reltuples = RelationGetNumberOfBlocks(heap) *
					   (BLCKSZ / (sizeof(float) * dim + 32));
	uint32_t nlist = (uint32_t)sqrt((double)Max(reltuples, 1));
	if (nlist < 1)
		nlist = 1;
	if (nlist > 10000)
		nlist = 10000;
	bs.nlist = nlist;

	/* 3. Sample vectors for k-means */
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

	/* 4. Run k-means
	 *
	 * kmeans allocates via mkt_alloc (= palloc), so all its internal
	 * state lands in build_ctx and gets freed with it. */
	KMeansOptions opts = MKT_KMEANS_OPTIONS_DEFAULT;
	opts.seed		   = 42;
	opts.nredo		   = 1;
	opts.algorithm	   = KMEANS_ALGO_LLOYD;

	KMeansResult *km = mkt_kmeans_f32(
			bs.samples, (uint32_t)bs.nsamples, dim, nlist, metric, &opts);

	bs.centroids = km->centroids;
	bs.nlist	 = km->nlist;
	nlist		 = km->nlist;

	/* Samples no longer needed — reclaim memory */
	pfree(bs.samples);
	bs.samples = NULL;

	/* 5. Compute global mean of centroids */
	float *global_mean = palloc0(dim * sizeof(float));
	for (uint32_t c = 0; c < nlist; c++)
	{
		const float *cent = bs.centroids + (size_t)c * dim;
		for (Dimension d = 0; d < dim; d++)
			global_mean[d] += cent[d];
	}
	for (Dimension d = 0; d < dim; d++)
		global_mean[d] /= (float)nlist;

	/* 6. Full heap scan — assign vectors, find medoids */
	if (centroid_format == MKT_CENTROID_FMT_RABITQ)
	{
		bs.medoid_tids	= palloc0(nlist * sizeof(ItemPointerData));
		bs.medoid_dists = palloc(nlist * sizeof(float));
		for (uint32_t c = 0; c < nlist; c++)
			bs.medoid_dists[c] = INFINITY;
	}
	else
	{
		bs.medoid_tids	= NULL;
		bs.medoid_dists = NULL;
	}
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

	/* 7. Encode centroids based on format */
	uint64_t	 rabitq_seed = 42;
	const void **encoded	 = palloc(nlist * sizeof(void *));

	switch (centroid_format)
	{
	case MKT_CENTROID_FMT_RABITQ:
	{
		RaBitQParams *params   = mkt_rabitq_create(dim, rabitq_seed);
		VectorRef	  mean_ref = {.data = global_mean, .dim = dim};
		for (uint32_t c = 0; c < nlist; c++)
		{
			VectorRef cref = {
					.data = bs.centroids + (size_t)c * dim,
					.dim  = dim,
			};
			encoded[c] = mkt_rabitq_encode(params, cref, mean_ref);
		}
		break;
	}
	case MKT_CENTROID_FMT_FLOAT:
	{
		for (uint32_t c = 0; c < nlist; c++)
			encoded[c] = bs.centroids + (size_t)c * dim;
		break;
	}
	case MKT_CENTROID_FMT_HALF:
	{
		for (uint32_t c = 0; c < nlist; c++)
		{
			half *hvec = palloc(dim * sizeof(half));
			mkt_float_to_half_array(bs.centroids + (size_t)c * dim, hvec, dim);
			encoded[c] = hvec;
		}
		break;
	}
	}

	/* 8. Write index pages */
	MktannStorage storage;
	mktann_storage_init(&storage, index, NULL, metric);

	/* Block 0: metadata */
	write_meta_page(
			&storage.base,
			dim,
			1, /* nlevels=1 for flat */
			1, /* first_centroid = block 1 (written next) */
			(uint32_t)bs.indtuples,
			nlist,
			centroid_format,
			metric,
			rabitq_seed,
			global_mean);

	/* Blocks 1+: centroid pages */
	BlockNumber first_centroid = mkt_centroid_write_pages(
			&storage.base,
			dim,
			nlist,
			centroid_format,
			0,
			MKT_CENTROID_FLAG_LEAF,
			0,
			encoded,
			bs.medoid_tids,
			NULL);

	/* Update meta page if first_centroid != 1 (shouldn't happen
	 * but be safe) */
	if (first_centroid != 1)
	{
		Page			page = mkt_storage_write_page(&storage.base, 0);
		MktannMetaPage *meta = (MktannMetaPage *)PageGetSpecialPointer(page);
		meta->first_centroid = first_centroid;
		mkt_storage_commit_page(&storage.base, 0);
	}

	/* 9. WAL-log all pages */
	log_newpage_range(
			index, MAIN_FORKNUM, 0, RelationGetNumberOfBlocks(index), true);

	/* 10. Cleanup — delete build context, return result in caller ctx */
	double indtuples = bs.indtuples;

	MemoryContextSwitchTo(caller_ctx);
	MemoryContextDelete(build_ctx);

	IndexBuildResult *result = palloc0(sizeof(IndexBuildResult));
	result->heap_tuples		 = heap_tuples;
	result->index_tuples	 = indtuples;
	return result;
}
