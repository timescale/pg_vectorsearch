/*
 * query.c - Zero-allocation query execution
 *
 * All buffers are pre-allocated in MktQueryCtx. The query hot path
 * uses only pre-allocated memory and arena reset (no malloc/free).
 *
 * For paged posting lists, delegates to MktQueryState (shared with
 * PG). Flat posting lists and brute-force remain standalone-only.
 *
 * Memory layout:
 *   memctx (long-lived) — owns all query context buffers
 *     └── arena (child) — transient per-query allocations, reset each query
 */

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "algo/topk.h"
#include "algo/vecops.h"
#include "core/memory.h"
#include "index/posting_page.h"
#include "index/posting_scan.h"
#include "index/query_scan.h"
#include "quant/rabitq.h"
#include "standalone/query.h"

/* ----------------------------------------------------------------
 * Query context internals
 * ---------------------------------------------------------------- */

struct MktQueryCtx
{
	MktIndex *idx;

	/* Shared search context (paged mode) */
	MktQueryState search;
	bool		  has_query_state;

	/* Reranking (standalone-specific) */
	MktTopK		  rerank_topk;
	MktTopKEntry *rerank_buf;
	uint32_t	  rerank_cap;

	/* Flat/brute-force fallback buffers */
	float			   *query_buf;
	MktCentroidResult  *beam_results;
	MktCentroidScratch *centroid_scratch;
	MktTopK				topk;
	MktPostingScan		posting_scan;
	float			   *pt_query;
	float			   *pt_cents_buf;
	RaBitQQueryState	beam_qs;
	RaBitQQueryState	cluster_qs;
	float			   *beam_transformed;
	float			   *cluster_transformed;
	uint8_t			   *beam_query_bits;
	uint8_t			   *cluster_query_bits;

	/* Long-lived memory context for all query context buffers.
	 * Deleting this frees everything at once (no individual frees). */
	MktMemCtx memctx;

	/* Child arena for transient per-query allocations (beam search
	 * internals). Reset per query — no create/delete overhead. */
	MktMemCtx arena;

	/* Limits */
	uint32_t max_k;
	uint32_t max_nprobe;
};

/* ----------------------------------------------------------------
 * Create / destroy
 * ---------------------------------------------------------------- */

MktQueryCtx *
mkt_query_ctx_create(MktIndex *idx, uint32_t max_k, uint32_t max_nprobe)
{
	if (idx == NULL || max_k == 0 || max_nprobe == 0)
		return NULL;

	MktMemCtx memctx  = mkt_memctx_create(NULL, "query_ctx");
	MktMemCtx old_ctx = mkt_memctx_switch(memctx);

	MktQueryCtx *ctx = mkt_alloc0(sizeof(MktQueryCtx));
	ctx->idx		 = idx;
	ctx->max_k		 = max_k;
	ctx->max_nprobe	 = max_nprobe;
	ctx->memctx		 = memctx;

	Dimension dim = idx->base.dim;

	/* Paged mode: use shared MktQueryState */
	if (idx->has_posting_data && idx->base.params != NULL &&
		idx->posting_fmt == MKT_POSTING_FMT_PAGES)
	{
		mkt_query_state_init(&ctx->search, &idx->base, max_k, max_nprobe);
		ctx->has_query_state = true;

		if (idx->base.fastscan)
			mkt_posting_scan_enable_fastscan(
					&ctx->search.pscan, idx->base.fastscan);

		if (idx->has_replication)
		{
			/* Size for max load at ~50%. Generation counter
			 * avoids per-query memset. */
			uint32_t est = max_nprobe * (idx->nvecs / idx->nlist) * 2;
			uint32_t cap = 1024;
			while (cap < est * 2)
				cap *= 2;
			ctx->search.dedup_set  = mkt_alloc(cap * sizeof(uint64_t));
			ctx->search.dedup_gens = mkt_alloc0(cap * sizeof(uint32_t));
			ctx->search.dedup_cap  = cap;
			ctx->search.dedup_gen  = 0;
		}
	}
	else
	{
		/* Flat/brute-force: allocate standalone buffers */
		uint32_t packed_bytes = MKT_RABITQ_BYTES(dim);

		ctx->query_buf		  = mkt_alloc(dim * sizeof(float));
		ctx->beam_results	  = mkt_alloc(max_nprobe * sizeof(MktCentroidResult));
		ctx->centroid_scratch = mkt_centroid_scratch_create(dim, max_nprobe);
		mkt_topk_init(&ctx->topk, max_k);

		if (idx->has_posting_data)
		{
			MktStorage *storage	  = (idx->posting_fmt == MKT_POSTING_FMT_PAGES)
										  ? &idx->posting_storage.base
										  : NULL;
			char	   *page_base = (idx->posting_fmt == MKT_POSTING_FMT_PAGES)
										  ? idx->posting_storage.pages
										  : NULL;
			uint32_t max_per_page = (idx->posting_fmt == MKT_POSTING_FMT_PAGES)
										  ? mkt_posting_max_entries(dim)
										  : idx->max_cluster_size;

			mkt_posting_scan_init(
					&ctx->posting_scan,
					storage,
					page_base,
					idx->base.params,
					dim,
					max_per_page);

			if (idx->base.fastscan)
				mkt_posting_scan_enable_fastscan(
						&ctx->posting_scan, idx->base.fastscan);
		}

		ctx->pt_cents_buf		 = mkt_alloc(max_nprobe * dim * sizeof(float));
		ctx->pt_query			 = mkt_alloc_aligned(dim * sizeof(float), 64);
		ctx->beam_transformed	 = mkt_alloc_aligned(dim * sizeof(float), 64);
		ctx->cluster_transformed = mkt_alloc_aligned(dim * sizeof(float), 64);
		ctx->beam_query_bits	 = mkt_alloc_aligned(packed_bytes, 64);
		ctx->cluster_query_bits	 = mkt_alloc_aligned(packed_bytes, 64);

		ctx->beam_qs.transformed	= ctx->beam_transformed;
		ctx->beam_qs.query_bits		= ctx->beam_query_bits;
		ctx->cluster_qs.transformed = ctx->cluster_transformed;
		ctx->cluster_qs.query_bits	= ctx->cluster_query_bits;

		mkt_rabitq_init_query_constants(&ctx->beam_qs, dim);
		mkt_rabitq_init_query_constants(&ctx->cluster_qs, dim);
	}

	/* Reranking buffers (used for both paged and flat) */
	mkt_topk_init(&ctx->rerank_topk, max_k);
	ctx->rerank_cap = max_k * 16;
	ctx->rerank_buf = mkt_alloc(ctx->rerank_cap * sizeof(MktTopKEntry));

	/* Child arena for transient per-query allocations */
	ctx->arena = mkt_memctx_create(memctx, "query_arena");

	mkt_memctx_switch(old_ctx);
	return ctx;
}

void
mkt_query_ctx_destroy(MktQueryCtx *ctx)
{
	if (ctx == NULL)
		return;

	if (ctx->has_query_state)
		mkt_query_state_cleanup(&ctx->search);
	else
	{
		if (ctx->idx->has_posting_data)
			mkt_posting_scan_cleanup(&ctx->posting_scan);
		mkt_topk_cleanup(&ctx->topk);
		mkt_centroid_scratch_free(ctx->centroid_scratch);
		ctx->centroid_scratch = NULL;
	}

	mkt_topk_cleanup(&ctx->rerank_topk);

	/* Deleting memctx frees ctx itself, all buffers, and the arena */
	mkt_memctx_delete(ctx->memctx);
}

/* ----------------------------------------------------------------
 * Paged query (via shared MktQueryState)
 * ---------------------------------------------------------------- */

static uint32_t
exec_paged(
		MktQueryCtx	   *ctx,
		const float	   *query,
		uint32_t		k,
		uint32_t		nprobe,
		MktDistanceMode mode,
		bool			rerank,
		uint32_t	   *result_ids)
{
	MktMemCtx old = mkt_memctx_switch(ctx->memctx);
	mkt_query_execute(&ctx->search, query, k, nprobe, mode, rerank, 0, NULL);
	mkt_memctx_switch(old);

	uint32_t nresults = ctx->search.nresults;
	for (uint32_t i = 0; i < nresults; i++)
	{
		uint32_t ci	  = ctx->search.result_order[i];
		result_ids[i] = mkt_posting_decode_vector_id(
				ctx->search.candidates[ci].id);
	}

	return nresults;
}

/* ----------------------------------------------------------------
 * Flat/brute-force query (standalone-only fallback)
 * ---------------------------------------------------------------- */

static uint32_t
exec_fallback(
		MktQueryCtx	   *ctx,
		const float	   *query,
		uint32_t		k,
		uint32_t		nprobe,
		MktDistanceMode mode,
		bool			rerank,
		uint32_t	   *result_ids)
{
	MktIndex *idx = ctx->idx;
	Dimension dim = idx->base.dim;

	if (k > ctx->max_k)
		k = ctx->max_k;
	if (nprobe > ctx->max_nprobe)
		nprobe = ctx->max_nprobe;

	MktMemCtx old_ctx = mkt_memctx_switch(ctx->arena);
	mkt_memctx_reset(ctx->arena);

	/* Normalize query */
	const float *qvec = query;
	if (idx->base.metric == DISTANCE_COSINE)
	{
		memcpy(ctx->query_buf, query, dim * sizeof(float));
		float norm = mkt_l2_norm(ctx->query_buf, dim);
		if (norm > 0.0f)
			mkt_vector_scale(ctx->query_buf, 1.0f / norm, ctx->query_buf, dim);
		qvec = ctx->query_buf;
	}

	/* Rotate query */
	RaBitQQueryState *qs = NULL;
	if (idx->base.params != NULL)
	{
		mkt_rabitq_rotate(idx->base.params, qvec, ctx->pt_query);

		if (idx->base.centroid_format == MKT_CENTROID_FMT_RABITQ)
		{
			mkt_rabitq_init_query_state(
					&ctx->beam_qs,
					ctx->pt_query,
					idx->base.pt_global_mean,
					dim,
					mode);
			qs = &ctx->beam_qs;
		}
	}

	/* Beam search */
	MktCentroidSearchState state = {
			.qstate		= qs,
			.query		= qvec,
			.storage	= &idx->centroid_storage.base,
			.beam_width = nprobe,
			.nprobe		= nprobe,
			.dim		= dim,
			.metric		= idx->base.metric,
			.scratch	= ctx->centroid_scratch,
	};

	MktCentroidSearchStats beam_stats = {0};
	uint32_t			   n_results  = mkt_centroid_beam_search(
			   &state,
			   idx->base.first_centroid,
			   idx->base.nlevels,
			   ctx->beam_results,
			   NULL,
			   &beam_stats);

	mkt_memctx_switch(old_ctx);

	/* Reset top-K */
	mkt_topk_reset(&ctx->topk);
	ctx->topk.k = k;

	/* Flat mode or brute-force */
	if (idx->has_posting_data && idx->base.params != NULL)
	{
		for (uint32_t j = 0; j < n_results; j++)
		{
			uint32_t li = (uint32_t)ctx->beam_results[j].posting_head;
			if (li >= idx->nlist)
				continue;

			const float *pt_cent = idx->pt_centroids + (size_t)li * dim;
			mkt_rabitq_init_query_state(
					&ctx->cluster_qs, ctx->pt_query, pt_cent, dim, mode);

			mkt_posting_scan_begin_flat(
					&ctx->posting_scan, &ctx->cluster_qs, idx->flat_pages[li]);
			mkt_posting_scan_cluster(&ctx->posting_scan, &ctx->topk);
			mkt_posting_scan_end_cluster(&ctx->posting_scan);
		}
	}
	else
	{
		for (uint32_t j = 0; j < n_results; j++)
		{
			uint32_t li = (uint32_t)ctx->beam_results[j].posting_head;
			if (li >= idx->nlist)
				continue;

			MktClusterList *cl = &idx->clusters[li];
			for (uint32_t vi = 0; vi < cl->count; vi++)
			{
				uint32_t	 vid = cl->ids[vi];
				const float *vec = idx->all_vectors + (size_t)vid * dim;
				Distance	 d	 = mkt_l2_distance_squared(qvec, vec, dim);
				mkt_topk_insert(&ctx->topk, d, 0.0f, vid);
			}
		}
	}

	/* Extract results */
	uint32_t count;
	if (rerank)
	{
		if (ctx->topk.cand_count > ctx->rerank_cap)
		{
			ctx->rerank_cap = ctx->topk.cand_count;
			ctx->rerank_buf = mkt_realloc(
					ctx->rerank_buf, ctx->rerank_cap * sizeof(MktTopKEntry));
		}

		uint32_t n_cands;
		mkt_topk_extract_sorted(&ctx->topk, ctx->rerank_buf, &n_cands);

		mkt_topk_reset(&ctx->rerank_topk);
		ctx->rerank_topk.k = k;

		for (uint32_t i = 0; i < n_cands; i++)
		{
			uint32_t vid = mkt_posting_decode_vector_id(ctx->rerank_buf[i].id);
			const float *vec = idx->all_vectors + (size_t)vid * dim;
			Distance	 d	 = mkt_l2_distance_squared(qvec, vec, dim);
			mkt_topk_insert(&ctx->rerank_topk, d, 0.0f, ctx->rerank_buf[i].id);
		}

		mkt_topk_extract_sorted(&ctx->rerank_topk, ctx->rerank_buf, &count);
	}
	else
	{
		if (ctx->topk.cand_count > ctx->rerank_cap)
		{
			ctx->rerank_cap = ctx->topk.cand_count;
			ctx->rerank_buf = mkt_realloc(
					ctx->rerank_buf, ctx->rerank_cap * sizeof(MktTopKEntry));
		}

		mkt_topk_extract_sorted(&ctx->topk, ctx->rerank_buf, &count);
	}

	uint32_t out = count < k ? count : k;
	for (uint32_t i = 0; i < out; i++)
		result_ids[i] = mkt_posting_decode_vector_id(ctx->rerank_buf[i].id);

	return out;
}

/* ----------------------------------------------------------------
 * Query execution — dispatch to paged or fallback
 * ---------------------------------------------------------------- */

uint32_t
mkt_query_exec(
		MktQueryCtx	   *ctx,
		const float	   *query,
		uint32_t		k,
		uint32_t		nprobe,
		MktDistanceMode mode,
		bool			rerank,
		uint32_t	   *result_ids)
{
	if (ctx == NULL || query == NULL || result_ids == NULL || k == 0)
		return 0;

	if (ctx->has_query_state)
		return exec_paged(ctx, query, k, nprobe, mode, rerank, result_ids);
	else
		return exec_fallback(ctx, query, k, nprobe, mode, rerank, result_ids);
}
