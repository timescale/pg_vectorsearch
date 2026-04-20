/*
 * query.c - Zero-allocation query execution
 *
 * All buffers are pre-allocated in MktQueryCtx. The query hot path
 * uses only pre-allocated memory and arena reset (no malloc/free).
 *
 * Key optimization: P^T * centroid is precomputed at index build time.
 * At query time, P^T * query is computed once, then per-cluster state
 * is derived via O(dim) vector subtraction instead of O(dim²) matrix
 * multiply.
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

	/* Pre-allocated buffers (owned by memctx) */
	float			  *query_buf;	 /* [dim] for normalization */
	MktCentroidResult *beam_results; /* [max_nprobe] */
	MktTopK			   topk;		 /* approximate candidates */
	MktTopK			   rerank_topk;	 /* exact reranked results */
	MktTopKEntry	  *extract_buf;	 /* [extract_cap] for extraction */
	uint32_t		   extract_cap;

	/* Posting scan */
	MktPostingScan posting_scan;
	float		  *pt_cents_buf; /* [max_nprobe * dim] for shared scan */

	/* Pre-allocated RaBitQ query state (avoids per-cluster alloc) */
	float			*pt_query;			  /* [dim] P^T * query */
	RaBitQQueryState beam_qs;			  /* for beam search */
	RaBitQQueryState cluster_qs;		  /* for cluster scan (reused) */
	float			*beam_transformed;	  /* [dim] scratch */
	float			*cluster_transformed; /* [dim] scratch */
	uint8_t			*beam_query_bits;	  /* [packed_bytes] */
	uint8_t			*cluster_query_bits;  /* [packed_bytes] */

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

	/* All allocations go into this long-lived context.
	 * Destroying it frees everything at once. */
	MktMemCtx memctx  = mkt_memctx_create(NULL, "query_ctx");
	MktMemCtx old_ctx = mkt_memctx_switch(memctx);

	MktQueryCtx *ctx = mkt_alloc0(sizeof(MktQueryCtx));
	ctx->idx		 = idx;
	ctx->max_k		 = max_k;
	ctx->max_nprobe	 = max_nprobe;
	ctx->memctx		 = memctx;

	Dimension dim		   = idx->dim;
	uint32_t  packed_bytes = MKT_RABITQ_BYTES(dim);

	/* Query normalization buffer */
	ctx->query_buf = mkt_alloc(dim * sizeof(float));

	/* Beam search results */
	ctx->beam_results = mkt_alloc(max_nprobe * sizeof(MktCentroidResult));

	/* Top-K: approximate candidates + exact reranked results */
	mkt_topk_init(&ctx->topk, max_k);
	mkt_topk_init(&ctx->rerank_topk, max_k);

	/* Extraction buffer — sized for typical case */
	ctx->extract_cap = max_k * 16;
	ctx->extract_buf = mkt_alloc(ctx->extract_cap * sizeof(MktTopKEntry));

	/* Initialize posting scan iterator if index has posting data */
	if (idx->has_posting_data)
	{
		MktStorage *storage		 = (idx->posting_fmt == MKT_POSTING_FMT_PAGES)
										 ? &idx->posting_storage.base
										 : NULL;
		char	   *page_base	 = (idx->posting_fmt == MKT_POSTING_FMT_PAGES)
										 ? idx->posting_storage.pages
										 : NULL;
		uint32_t	max_per_page = (idx->posting_fmt == MKT_POSTING_FMT_PAGES)
										 ? mkt_posting_max_entries(dim)
										 : idx->max_cluster_size;

		mkt_posting_scan_init(
				&ctx->posting_scan,
				storage,
				page_base,
				idx->rq_params,
				dim,
				max_per_page);
	}

	/* Buffer for gathering pt_centroids per query (shared scan) */
	ctx->pt_cents_buf = mkt_alloc(max_nprobe * dim * sizeof(float));

	/* Pre-allocated RaBitQ buffers */
	ctx->pt_query			 = mkt_alloc_aligned(dim * sizeof(float), 64);
	ctx->beam_transformed	 = mkt_alloc_aligned(dim * sizeof(float), 64);
	ctx->cluster_transformed = mkt_alloc_aligned(dim * sizeof(float), 64);
	ctx->beam_query_bits	 = mkt_alloc_aligned(packed_bytes, 64);
	ctx->cluster_query_bits	 = mkt_alloc_aligned(packed_bytes, 64);

	/* Wire up the pre-allocated buffers to the query states */
	ctx->beam_qs.transformed	= ctx->beam_transformed;
	ctx->beam_qs.query_bits		= ctx->beam_query_bits;
	ctx->cluster_qs.transformed = ctx->cluster_transformed;
	ctx->cluster_qs.query_bits	= ctx->cluster_query_bits;

	/* Set dim-dependent constants once (avoid recomputing per cluster) */
	mkt_rabitq_init_query_constants(&ctx->beam_qs, dim);
	mkt_rabitq_init_query_constants(&ctx->cluster_qs, dim);

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

	/* Clean up posting scan buffers */
	if (ctx->idx->has_posting_data)
		mkt_posting_scan_cleanup(&ctx->posting_scan);

	/* TopK has its own heap that may have been grown outside memctx */
	mkt_topk_cleanup(&ctx->topk);
	mkt_topk_cleanup(&ctx->rerank_topk);

	/* Deleting memctx frees ctx itself, all buffers, and the arena */
	mkt_memctx_delete(ctx->memctx);
}

/* ----------------------------------------------------------------
 * Query execution — zero malloc on hot path
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

	MktIndex *idx = ctx->idx;
	Dimension dim = idx->dim;

	if (k > ctx->max_k)
		k = ctx->max_k;
	if (nprobe > ctx->max_nprobe)
		nprobe = ctx->max_nprobe;

	/* Reset arena for transient allocations */
	MktMemCtx old_ctx = mkt_memctx_switch(ctx->arena);
	mkt_memctx_reset(ctx->arena);

	/* Normalize query (into pre-allocated buffer) */
	const float *qvec = query;
	if (idx->metric == DISTANCE_COSINE)
	{
		memcpy(ctx->query_buf, query, dim * sizeof(float));
		float norm = mkt_l2_norm(ctx->query_buf, dim);
		if (norm > 0.0f)
			mkt_vector_scale(ctx->query_buf, 1.0f / norm, ctx->query_buf, dim);
		qvec = ctx->query_buf;
	}

	/* Compute P^T * query once (O(dim²) — but only once per query).
	 * Needed when posting lists use RaBitQ encoding, and also for
	 * RaBitQ centroid routing. */
	RaBitQQueryState *qs = NULL;
	if (idx->rq_params != NULL)
	{
		mkt_rabitq_rotate(idx->rq_params, qvec, ctx->pt_query);

		/* Beam search query state — only for RaBitQ centroids */
		if (idx->centroid_fmt == MKT_CENTROID_FMT_RABITQ)
		{
			mkt_rabitq_init_query_state(
					&ctx->beam_qs,
					ctx->pt_query,
					idx->pt_global_mean,
					dim,
					mode);
			qs = &ctx->beam_qs;
		}
	}

	/* Beam search (uses arena for internal scratch) */
	MktCentroidSearchState state = {
			.qstate		= qs,
			.query		= qvec,
			.storage	= &idx->centroid_storage.base,
			.beam_width = nprobe,
			.nprobe		= nprobe,
			.dim		= dim,
			.metric		= idx->metric,
	};

	MktCentroidSearchStats beam_stats = {0};
	uint32_t			   n_results  = mkt_centroid_beam_search(
			   &state,
			   idx->first_centroid,
			   idx->nlevels,
			   ctx->beam_results,
			   NULL,
			   &beam_stats);

	/* Gather pt_centroids for winning clusters from pre-computed
	 * array. Beam results' posting_head maps to cluster index
	 * (paged: actual block number, flat: cluster index). */
	for (uint32_t j = 0; j < n_results; j++)
	{
		uint32_t li = (uint32_t)ctx->beam_results[j].posting_head;
		if (li < idx->nlist)
			memcpy(ctx->pt_cents_buf + (size_t)j * dim,
				   idx->pt_centroids + (size_t)li * dim,
				   dim * sizeof(float));
	}

	/* Switch away from arena so topk growth allocations
	 * (if any) use the default allocator, not the arena
	 * that gets reset per query. */
	mkt_memctx_switch(old_ctx);

	/* Reset top-K (pre-allocated, no alloc) */
	mkt_topk_reset(&ctx->topk);
	ctx->topk.k = k;

	/* Scan posting lists */
	if (idx->has_posting_data && idx->rq_params != NULL &&
		idx->posting_fmt == MKT_POSTING_FMT_PAGES)
	{
		/* Paged mode: shared scan reads pt_centroid from pages */
		MktQueryScanParams scan_params = {
				.posting_scan		   = &ctx->posting_scan,
				.cluster_qs			   = &ctx->cluster_qs,
				.pt_query			   = ctx->pt_query,
				.dim				   = dim,
				.mode				   = mode,
				.total_posting_pages   = 0,
				.total_posting_entries = 0,
		};

		mkt_query_scan_clusters(
				&scan_params, ctx->beam_results, n_results, &ctx->topk);

		/* Print page stats on first query */
		static bool stats_printed = false;
		if (!stats_printed)
		{
			fprintf(stderr,
					"[query stats] centroid_pages=%u "
					"posting_pages=%u posting_entries=%u "
					"nprobe=%u\n",
					beam_stats.pages_read,
					scan_params.total_posting_pages,
					scan_params.total_posting_entries,
					nprobe);
			stats_printed = true;
		}
	}
	else if (idx->has_posting_data && idx->rq_params != NULL)
	{
		/* Flat mode: pt_centroid from idx->pt_centroids */
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
		/* No posting data: brute-force L2 per cluster */
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

	/* Phase 2: Rerank candidates with exact L2 (optional) */
	uint32_t count;
	if (rerank)
	{
		uint32_t	  n_cands;
		MktTopKEntry *cand_entries;
		if (ctx->topk.cand_count <= ctx->extract_cap)
			cand_entries = ctx->extract_buf;
		else
			cand_entries = mkt_alloc(
					ctx->topk.cand_count * sizeof(MktTopKEntry));
		mkt_topk_extract_sorted(&ctx->topk, cand_entries, &n_cands);

		mkt_topk_reset(&ctx->rerank_topk);
		ctx->rerank_topk.k = k;

		for (uint32_t i = 0; i < n_cands; i++)
		{
			uint32_t vid = mkt_posting_decode_vector_id(cand_entries[i].id);
			const float *vec = idx->all_vectors + (size_t)vid * dim;
			Distance	 d	 = mkt_l2_distance_squared(qvec, vec, dim);
			mkt_topk_insert(&ctx->rerank_topk, d, 0.0f, cand_entries[i].id);
		}

		if (cand_entries != ctx->extract_buf)
			mkt_free(cand_entries);

		mkt_topk_extract_sorted(&ctx->rerank_topk, ctx->extract_buf, &count);
	}
	else
	{
		/* No rerank: return approximate results directly */
		if (ctx->topk.cand_count <= ctx->extract_cap)
		{
			mkt_topk_extract_sorted(&ctx->topk, ctx->extract_buf, &count);
		}
		else
		{
			MktTopKEntry *tmp = mkt_alloc(
					ctx->topk.cand_count * sizeof(MktTopKEntry));
			mkt_topk_extract_sorted(&ctx->topk, tmp, &count);
			if (count > ctx->extract_cap)
				count = ctx->extract_cap;
			memcpy(ctx->extract_buf, tmp, count * sizeof(MktTopKEntry));
			mkt_free(tmp);
		}
	}

	uint32_t out = count < k ? count : k;
	for (uint32_t i = 0; i < out; i++)
		result_ids[i] = mkt_posting_decode_vector_id(ctx->extract_buf[i].id);

	return out;
}
