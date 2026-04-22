/*
 * search_ctx.c - Pre-allocated search context for ANN queries
 *
 * Shared between standalone and PG builds. All buffers are
 * allocated once in init; the per-query execute path does zero
 * allocations (except rare topk candidate buffer growth).
 */

#include <math.h>
#include <string.h>

#include "algo/topk.h"
#include "algo/vecops.h"
#include "core/memory.h"
#include "index/centroid_search.h"
#include "index/posting_page.h"
#include "index/posting_scan.h"
#include "index/query_scan.h"
#include "index/search_ctx.h"
#include "quant/rabitq.h"

/* ----------------------------------------------------------------
 * Init / cleanup
 * ---------------------------------------------------------------- */

void
mkt_search_ctx_init(MktSearchCtx *ctx, const MktSearchCtxConfig *cfg)
{
	memset(ctx, 0, sizeof(*ctx));

	/* Copy configuration */
	ctx->params			  = cfg->params;
	ctx->pt_global_mean	  = cfg->pt_global_mean;
	ctx->dim			  = cfg->dim;
	ctx->nlevels		  = cfg->nlevels;
	ctx->first_centroid	  = cfg->first_centroid;
	ctx->max_k			  = cfg->max_k;
	ctx->max_nprobe		  = cfg->max_nprobe;
	ctx->metric			  = cfg->metric;
	ctx->centroid_format  = cfg->centroid_format;
	ctx->centroid_storage = cfg->centroid_storage;
	ctx->posting_storage  = cfg->posting_storage;
	ctx->page_base		  = cfg->page_base;

	Dimension dim		   = cfg->dim;
	uint32_t  packed_bytes = MKT_RABITQ_BYTES(dim);

	/* Query buffers */
	ctx->query_buf			 = mkt_alloc(dim * sizeof(float));
	ctx->pt_query			 = mkt_alloc_aligned(dim * sizeof(float), 64);
	ctx->beam_transformed	 = mkt_alloc_aligned(dim * sizeof(float), 64);
	ctx->cluster_transformed = mkt_alloc_aligned(dim * sizeof(float), 64);
	ctx->beam_query_bits	 = mkt_alloc_aligned(packed_bytes, 64);
	ctx->cluster_query_bits	 = mkt_alloc_aligned(packed_bytes, 64);

	/* Wire up query state buffers */
	ctx->beam_qs.transformed	= ctx->beam_transformed;
	ctx->beam_qs.query_bits		= ctx->beam_query_bits;
	ctx->cluster_qs.transformed = ctx->cluster_transformed;
	ctx->cluster_qs.query_bits	= ctx->cluster_query_bits;

	mkt_rabitq_init_query_constants(&ctx->beam_qs, dim);
	mkt_rabitq_init_query_constants(&ctx->cluster_qs, dim);

	/* Beam search results */
	ctx->beam_results = mkt_alloc(cfg->max_nprobe * sizeof(MktCentroidResult));

	/* Top-K */
	mkt_topk_init(&ctx->topk, cfg->max_k);

	/* Candidate extraction buffer */
	ctx->cand_cap	= cfg->max_k * 16;
	ctx->candidates = mkt_alloc(ctx->cand_cap * sizeof(MktTopKEntry));

	/* Posting scan iterator */
	uint32_t max_entries = mkt_posting_max_entries(dim);
	mkt_posting_scan_init(
			&ctx->pscan,
			cfg->posting_storage,
			cfg->page_base,
			cfg->params,
			dim,
			max_entries);
}

void
mkt_search_ctx_cleanup(MktSearchCtx *ctx)
{
	if (ctx == NULL)
		return;

	/* Release any pinned page still held by the posting scan.
	 * All memory is freed by the caller's context deletion — no
	 * individual mkt_free calls needed. */
	mkt_posting_scan_cleanup(&ctx->pscan);
	mkt_topk_cleanup(&ctx->topk);
}

/* ----------------------------------------------------------------
 * Per-query execution
 * ---------------------------------------------------------------- */

static const float *
prepare_query(MktSearchCtx *ctx, const float *query)
{
	if (ctx->metric != DISTANCE_COSINE)
		return query;

	Dimension dim = ctx->dim;
	memcpy(ctx->query_buf, query, dim * sizeof(float));
	float norm = mkt_l2_norm(ctx->query_buf, dim);
	if (norm > 0.0f)
		mkt_vector_scale(ctx->query_buf, 1.0f / norm, ctx->query_buf, dim);
	return ctx->query_buf;
}

static uint32_t
search_centroids(
		MktSearchCtx		   *ctx,
		const float			   *qvec,
		MktDistanceMode			mode,
		MktCentroidSearchStats *beam_stats)
{
	Dimension dim = ctx->dim;

	/* Rotate query: P^T * query */
	mkt_rabitq_rotate(ctx->params, qvec, ctx->pt_query);

	/* Beam search query state (RaBitQ centroids only) */
	RaBitQQueryState *qs = NULL;
	if (ctx->centroid_format == MKT_CENTROID_FMT_RABITQ)
	{
		mkt_rabitq_init_query_state(
				&ctx->beam_qs, ctx->pt_query, ctx->pt_global_mean, dim, mode);
		qs = &ctx->beam_qs;
	}

	MktCentroidSearchState search = {
			.qstate		= qs,
			.query		= qvec,
			.storage	= ctx->centroid_storage,
			.beam_width = ctx->max_nprobe,
			.nprobe		= ctx->max_nprobe,
			.dim		= dim,
			.metric		= ctx->metric,
	};

	return mkt_centroid_beam_search(
			&search,
			ctx->first_centroid,
			ctx->nlevels,
			ctx->beam_results,
			NULL,
			beam_stats);
}

static uint32_t
extract_candidates(MktSearchCtx *ctx)
{
	uint32_t ncands;

	if (ctx->topk.cand_count <= ctx->cand_cap)
	{
		mkt_topk_extract_sorted(&ctx->topk, ctx->candidates, &ncands);
	}
	else
	{
		MktTopKEntry *tmp = mkt_alloc(
				ctx->topk.cand_count * sizeof(MktTopKEntry));
		mkt_topk_extract_sorted(&ctx->topk, tmp, &ncands);
		uint32_t copy = ncands < ctx->cand_cap ? ncands : ctx->cand_cap;
		memcpy(ctx->candidates, tmp, copy * sizeof(MktTopKEntry));
		ncands = copy;
		mkt_free(tmp);
	}

	ctx->ncandidates = ncands;
	return ncands;
}

uint32_t
mkt_search_execute(
		MktSearchCtx   *ctx,
		const float	   *query,
		uint32_t		k,
		uint32_t		nprobe,
		MktDistanceMode mode,
		MktSearchStats *stats)
{
	if (k > ctx->max_k)
		k = ctx->max_k;
	if (nprobe > ctx->max_nprobe)
		nprobe = ctx->max_nprobe;

	/* Temporarily adjust nprobe for this query */
	uint32_t saved_nprobe = ctx->max_nprobe;
	ctx->max_nprobe		  = nprobe;

	/* Reset top-K for this query */
	mkt_topk_reset(&ctx->topk);
	ctx->topk.k = k;

	/* 1. Normalize query (cosine) */
	const float *qvec = prepare_query(ctx, query);

	/* 2. Beam search over centroids */
	MktCentroidSearchStats beam_stats = {0};
	uint32_t ncentroids = search_centroids(ctx, qvec, mode, &beam_stats);

	/* 3. Scan posting lists */
	ctx->pscan.storage = ctx->posting_storage;

	MktQueryScanParams scan_params = {
			.posting_scan = &ctx->pscan,
			.cluster_qs	  = &ctx->cluster_qs,
			.pt_query	  = ctx->pt_query,
			.dim		  = ctx->dim,
			.mode		  = mode,
	};
	mkt_query_scan_clusters(
			&scan_params, ctx->beam_results, ncentroids, &ctx->topk);

	ctx->pscan.storage = NULL;

	/* 4. Extract candidates */
	uint32_t ncands = extract_candidates(ctx);

	/* 5. Stats */
	if (stats != NULL)
	{
		stats->centroid_pages_read	   = beam_stats.pages_read;
		stats->posting_pages_read	   = scan_params.total_posting_pages;
		stats->posting_entries_scanned = scan_params.total_posting_entries;
		stats->clusters_scanned		   = ncentroids;
	}

	ctx->max_nprobe = saved_nprobe;
	return ncands;
}
