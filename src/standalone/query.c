/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * query.c - Zero-allocation query execution
 *
 * All buffers are pre-allocated in PrismQueryCtx. The query hot path
 * uses only pre-allocated memory and arena reset (no malloc/free).
 *
 * For paged posting lists, delegates to PrismQueryState (shared with
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

struct PrismQueryCtx
{
	PrismIndex *idx;

	/* Shared search context (paged mode) */
	PrismQueryState search;
	bool			has_query_state;

	/* Reranking (standalone-specific) */
	VsTopK		 rerank_topk;
	VsTopKEntry *rerank_buf;
	uint32_t	 rerank_cap;

	/* Flat/brute-force fallback buffers */
	float				 *query_buf;
	PrismCentroidResult	 *beam_results;
	PrismCentroidScratch *centroid_scratch;
	VsTopK				  topk;
	PrismPostingScan	  posting_scan;
	float				 *pt_query;
	float				 *pt_cents_buf;
	RaBitQQueryState	  beam_qs;
	RaBitQQueryState	  cluster_qs;
	float				 *beam_transformed;
	float				 *cluster_transformed;
	uint8_t				 *beam_query_bits;
	uint8_t				 *cluster_query_bits;

	/* Long-lived memory context for all query context buffers.
	 * Deleting this frees everything at once (no individual frees). */
	VsMemCtx memctx;

	/* Child arena for transient per-query allocations (beam search
	 * internals). Reset per query — no create/delete overhead. */
	VsMemCtx arena;

	/* Limits */
	uint32_t max_k;
	uint32_t max_nprobe;
};

/* ----------------------------------------------------------------
 * Create / destroy
 * ---------------------------------------------------------------- */

PrismQueryCtx *
prism_query_ctx_create(PrismIndex *idx, uint32_t max_k, uint32_t max_nprobe)
{
	if (idx == NULL || max_k == 0 || max_nprobe == 0)
		return NULL;

	VsMemCtx memctx	 = vs_memctx_create(NULL, "query_ctx");
	VsMemCtx old_ctx = vs_memctx_switch(memctx);

	PrismQueryCtx *ctx = vs_alloc0(sizeof(PrismQueryCtx));
	ctx->idx		   = idx;
	ctx->max_k		   = max_k;
	ctx->max_nprobe	   = max_nprobe;
	ctx->memctx		   = memctx;

	Dimension dim = idx->base.dim;

	/* Paged mode: use shared PrismQueryState */
	if (idx->has_posting_data && idx->base.params != NULL &&
		idx->posting_fmt == PRISM_POSTING_FMT_PAGES)
	{
		prism_query_state_init(&ctx->search, &idx->base, max_k, max_nprobe);
		ctx->has_query_state = true;

		if (idx->base.fastscan)
			prism_posting_scan_enable_fastscan(
					&ctx->search.pscan, idx->base.fastscan);
	}
	else
	{
		/* Flat/brute-force: allocate standalone buffers */
		uint32_t packed_bytes = VS_RABITQ_BYTES(dim);

		ctx->query_buf	  = vs_alloc(dim * sizeof(float));
		ctx->beam_results = vs_alloc(max_nprobe * sizeof(PrismCentroidResult));
		ctx->centroid_scratch = prism_centroid_scratch_create(dim, max_nprobe);
		vs_topk_init(&ctx->topk, max_k);

		if (idx->has_posting_data)
		{
			VsStorage *storage = (idx->posting_fmt == PRISM_POSTING_FMT_PAGES)
									   ? &idx->posting_storage.base
									   : NULL;
			char	*page_base = (idx->posting_fmt == PRISM_POSTING_FMT_PAGES)
									   ? idx->posting_storage.pages
									   : NULL;
			uint32_t max_per_page = (idx->posting_fmt ==
									 PRISM_POSTING_FMT_PAGES)
										  ? prism_posting_max_entries(dim)
										  : idx->max_cluster_size;

			prism_posting_scan_init(
					&ctx->posting_scan,
					storage,
					page_base,
					idx->base.params,
					dim,
					max_per_page);

			if (idx->base.fastscan)
				prism_posting_scan_enable_fastscan(
						&ctx->posting_scan, idx->base.fastscan);
		}

		ctx->pt_cents_buf		 = vs_alloc(max_nprobe * dim * sizeof(float));
		ctx->pt_query			 = vs_alloc_aligned(dim * sizeof(float), 64);
		ctx->beam_transformed	 = vs_alloc_aligned(dim * sizeof(float), 64);
		ctx->cluster_transformed = vs_alloc_aligned(dim * sizeof(float), 64);
		ctx->beam_query_bits	 = vs_alloc_aligned(packed_bytes, 64);
		ctx->cluster_query_bits	 = vs_alloc_aligned(packed_bytes, 64);

		ctx->beam_qs.transformed	= ctx->beam_transformed;
		ctx->beam_qs.query_bits		= ctx->beam_query_bits;
		ctx->cluster_qs.transformed = ctx->cluster_transformed;
		ctx->cluster_qs.query_bits	= ctx->cluster_query_bits;

		vs_rabitq_init_query_constants(&ctx->beam_qs, dim);
		vs_rabitq_init_query_constants(&ctx->cluster_qs, dim);
	}

	/* Reranking buffers (used for both paged and flat) */
	vs_topk_init(&ctx->rerank_topk, max_k);
	ctx->rerank_cap = max_k * 16;
	ctx->rerank_buf = vs_alloc(ctx->rerank_cap * sizeof(VsTopKEntry));

	/* Child arena for transient per-query allocations */
	ctx->arena = vs_memctx_create(memctx, "query_arena");

	vs_memctx_switch(old_ctx);
	return ctx;
}

void
prism_query_ctx_destroy(PrismQueryCtx *ctx)
{
	if (ctx == NULL)
		return;

	if (ctx->has_query_state)
		prism_query_state_cleanup(&ctx->search);
	else
	{
		if (ctx->idx->has_posting_data)
			prism_posting_scan_cleanup(&ctx->posting_scan);
		vs_topk_cleanup(&ctx->topk);
		prism_centroid_scratch_free(ctx->centroid_scratch);
		ctx->centroid_scratch = NULL;
	}

	vs_topk_cleanup(&ctx->rerank_topk);

	/* Deleting memctx frees ctx itself, all buffers, and the arena */
	vs_memctx_delete(ctx->memctx);
}

/* ----------------------------------------------------------------
 * Paged query (via shared PrismQueryState)
 * ---------------------------------------------------------------- */

static uint32_t
exec_paged(
		PrismQueryCtx *ctx,
		const float	  *query,
		uint32_t	   k,
		uint32_t	   nprobe,
		VsDistanceMode mode,
		bool		   rerank,
		uint32_t	  *result_ids)
{
	VsMemCtx old = vs_memctx_switch(ctx->memctx);
	prism_query_execute(&ctx->search, query, k, nprobe, mode, rerank, NULL);
	vs_memctx_switch(old);

	uint32_t nresults = ctx->search.nresults;
	for (uint32_t i = 0; i < nresults; i++)
	{
		uint32_t ci	  = ctx->search.result_order[i];
		result_ids[i] = prism_posting_decode_vector_id(
				ctx->search.candidates[ci].id);
	}

	return nresults;
}

/* ----------------------------------------------------------------
 * Flat/brute-force query (standalone-only fallback)
 * ---------------------------------------------------------------- */

static uint32_t
exec_fallback(
		PrismQueryCtx *ctx,
		const float	  *query,
		uint32_t	   k,
		uint32_t	   nprobe,
		VsDistanceMode mode,
		bool		   rerank,
		uint32_t	  *result_ids)
{
	PrismIndex *idx = ctx->idx;
	Dimension	dim = idx->base.dim;

	if (k > ctx->max_k)
		k = ctx->max_k;
	if (nprobe > ctx->max_nprobe)
		nprobe = ctx->max_nprobe;

	VsMemCtx old_ctx = vs_memctx_switch(ctx->arena);
	vs_memctx_reset(ctx->arena);

	/* Normalize query */
	const float *qvec = query;
	if (idx->base.metric == DISTANCE_COSINE)
	{
		memcpy(ctx->query_buf, query, dim * sizeof(float));
		float norm = vs_l2_norm(ctx->query_buf, dim);
		if (norm > 0.0f)
			vec32_scale(ctx->query_buf, 1.0f / norm, ctx->query_buf, dim);
		qvec = ctx->query_buf;
	}

	/* Rotate query */
	RaBitQQueryState *qs = NULL;
	if (idx->base.params != NULL)
	{
		vs_rabitq_rotate(idx->base.params, qvec, ctx->pt_query);

		if (idx->base.centroid_format == PRISM_CENTROID_FMT_RABITQ)
		{
			vs_rabitq_init_query_state(
					&ctx->beam_qs,
					ctx->pt_query,
					idx->base.pt_global_mean,
					dim,
					mode);
			qs = &ctx->beam_qs;
		}
	}

	/* Beam search */
	PrismCentroidSearchState state = {
			.qstate		 = qs,
			.query		 = qvec,
			.storage	 = &idx->centroid_storage.base,
			.beam_width	 = nprobe,
			.nprobe		 = nprobe,
			.dim		 = dim,
			.metric		 = idx->base.metric,
			.error_scale = 0.0f,
			.scratch	 = ctx->centroid_scratch,
	};

	PrismCentroidSearchStats beam_stats = {0};
	uint32_t				 n_results	= prism_centroid_beam_search(
			 &state,
			 idx->base.first_centroid,
			 idx->base.nlevels,
			 ctx->beam_results,
			 NULL,
			 &beam_stats);

	vs_memctx_switch(old_ctx);

	/* Reset top-K */
	vs_topk_reset_to_k(&ctx->topk, k);

	/* Flat mode or brute-force */
	if (idx->has_posting_data && idx->base.params != NULL)
	{
		for (uint32_t j = 0; j < n_results; j++)
		{
			uint32_t li = (uint32_t)ctx->beam_results[j].posting_head;
			if (li >= idx->nlist)
				continue;

			const float *pt_cent = idx->pt_centroids + (size_t)li * dim;
			vs_rabitq_init_query_state(
					&ctx->cluster_qs, ctx->pt_query, pt_cent, dim, mode);

			prism_posting_scan_begin_flat(
					&ctx->posting_scan, &ctx->cluster_qs, idx->flat_pages[li]);
			prism_posting_scan_cluster(&ctx->posting_scan, &ctx->topk);
			prism_posting_scan_end_cluster(&ctx->posting_scan);
		}
	}
	else
	{
		for (uint32_t j = 0; j < n_results; j++)
		{
			uint32_t li = (uint32_t)ctx->beam_results[j].posting_head;
			if (li >= idx->nlist)
				continue;

			PrismClusterList *cl = &idx->clusters[li];
			for (uint32_t vi = 0; vi < cl->count; vi++)
			{
				uint32_t	 vid = cl->ids[vi];
				const float *vec = idx->all_vectors + (size_t)vid * dim;
				Distance	 d	 = vs_l2_distance_squared(qvec, vec, dim);
				vs_topk_insert(&ctx->topk, d, 0.0f, vid);
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
			ctx->rerank_buf = vs_realloc(
					ctx->rerank_buf, ctx->rerank_cap * sizeof(VsTopKEntry));
		}

		uint32_t n_cands;
		vs_topk_extract_sorted(&ctx->topk, ctx->rerank_buf, &n_cands);

		vs_topk_reset_to_k(&ctx->rerank_topk, k);

		for (uint32_t i = 0; i < n_cands; i++)
		{
			uint32_t vid = prism_posting_decode_vector_id(
					ctx->rerank_buf[i].id);
			const float *vec = idx->all_vectors + (size_t)vid * dim;
			Distance	 d	 = vs_l2_distance_squared(qvec, vec, dim);
			vs_topk_insert(&ctx->rerank_topk, d, 0.0f, ctx->rerank_buf[i].id);
		}

		vs_topk_extract_sorted(&ctx->rerank_topk, ctx->rerank_buf, &count);
	}
	else
	{
		if (ctx->topk.cand_count > ctx->rerank_cap)
		{
			ctx->rerank_cap = ctx->topk.cand_count;
			ctx->rerank_buf = vs_realloc(
					ctx->rerank_buf, ctx->rerank_cap * sizeof(VsTopKEntry));
		}

		vs_topk_extract_sorted(&ctx->topk, ctx->rerank_buf, &count);
	}

	uint32_t out = count < k ? count : k;
	for (uint32_t i = 0; i < out; i++)
		result_ids[i] = prism_posting_decode_vector_id(ctx->rerank_buf[i].id);

	return out;
}

/* ----------------------------------------------------------------
 * Query execution — dispatch to paged or fallback
 * ---------------------------------------------------------------- */

uint32_t
prism_query_exec(
		PrismQueryCtx *ctx,
		const float	  *query,
		uint32_t	   k,
		uint32_t	   nprobe,
		VsDistanceMode mode,
		bool		   rerank,
		uint32_t	  *result_ids)
{
	if (ctx == NULL || query == NULL || result_ids == NULL || k == 0)
		return 0;

	if (ctx->has_query_state)
		return exec_paged(ctx, query, k, nprobe, mode, rerank, result_ids);
	else
		return exec_fallback(ctx, query, k, nprobe, mode, rerank, result_ids);
}
