/*
 * query.c - Zero-allocation query execution
 *
 * All buffers are pre-allocated in MktQueryCtx. The query hot path
 * uses only pre-allocated memory and arena reset (no malloc/free).
 *
 * Memory layout:
 *   memctx (long-lived) — owns all query context buffers
 *     └── arena (child) — transient per-query allocations, reset each query
 */

#include <math.h>
#include <string.h>

#include "algo/topk.h"
#include "algo/vecops.h"
#include "core/memory.h"
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
	MktTopK			   topk;		 /* pre-initialized, reset per query */
	MktTopKEntry	  *extract_buf;	 /* [extract_cap] for extraction */
	uint32_t		   extract_cap;

	/* Posting list scan scratch (for RaBitQ batch distance) */
	Distance *scan_distances;	 /* [max_per_cluster] */
	Distance *scan_lower_bounds; /* [max_per_cluster] */
	float	 *scan_scratch;		 /* [max_per_cluster] float scratch */
	uint32_t  max_per_cluster;

	/* Long-lived memory context for all query context buffers.
	 * Deleting this frees everything at once (no individual frees). */
	MktMemCtx memctx;

	/* Child arena for transient per-query allocations (beam search
	 * internals, rabitq query state). Reset per query — no
	 * create/delete overhead. */
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

	/* Query normalization buffer */
	ctx->query_buf = mkt_alloc(idx->dim * sizeof(float));

	/* Beam search results */
	ctx->beam_results = mkt_alloc(max_nprobe * sizeof(MktCentroidResult));

	/* Top-K: init once, reset per query */
	mkt_topk_init(&ctx->topk, max_k);

	/* Extraction buffer — sized for typical case */
	ctx->extract_cap = max_k * 16;
	ctx->extract_buf = mkt_alloc(ctx->extract_cap * sizeof(MktTopKEntry));

	/* Posting scan scratch — sized for the largest cluster */
	uint32_t max_cluster = 0;
	for (uint32_t c = 0; c < idx->nlist; c++)
		if (idx->lists[c].count > max_cluster)
			max_cluster = idx->lists[c].count;
	if (max_cluster < 1024)
		max_cluster = 1024;
	ctx->max_per_cluster   = max_cluster;
	ctx->scan_distances	   = mkt_alloc(max_cluster * sizeof(Distance));
	ctx->scan_lower_bounds = mkt_alloc(max_cluster * sizeof(Distance));
	ctx->scan_scratch	   = mkt_alloc(max_cluster * sizeof(float));

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

	/* TopK has its own heap that may have been grown outside memctx */
	mkt_topk_cleanup(&ctx->topk);

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

	/* Prepare RaBitQ query state (uses arena — freed on reset) */
	RaBitQQueryState *qs = NULL;
	if (idx->centroid_fmt == MKT_CENTROID_FMT_RABITQ)
	{
		VectorRef qref = {.data = qvec, .dim = dim};
		VectorRef cref = {.data = idx->global_mean, .dim = dim};
		qs = mkt_rabitq_prepare_query_ex(idx->rq_params, qref, cref, mode);
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

	uint32_t n_results = mkt_centroid_beam_search(
			&state,
			idx->first_centroid,
			idx->nlevels,
			ctx->beam_results,
			NULL,
			NULL);

	/* Switch away from arena so topk growth allocations
	 * (if any) use the default allocator, not the arena
	 * that gets reset per query. */
	mkt_memctx_switch(old_ctx);

	/* Reset top-K (pre-allocated, no alloc) */
	mkt_topk_reset(&ctx->topk);
	ctx->topk.k = k;

	/* Scan posting lists in selected clusters */
	uint32_t packed_bytes = MKT_RABITQ_BYTES(dim);

	for (uint32_t j = 0; j < n_results; j++)
	{
		uint32_t li = (uint32_t)ctx->beam_results[j].posting_head;
		if (li >= idx->nlist)
			continue;

		MktPostingList *pl = &idx->lists[li];

		if (pl->bits != NULL && qs != NULL)
		{
			/* RaBitQ quantized scan with error bounds.
			 *
			 * Prepare per-cluster query state, then batch
			 * distance + lower bound computation. Only vectors
			 * whose lower bound passes the top-K threshold get
			 * reranked with exact distance. */
			const float *cent = idx->leaf_centroids + (size_t)li * dim;
			VectorRef	 qref = {.data = qvec, .dim = dim};
			VectorRef	 cref = {.data = cent, .dim = dim};

			/* Per-cluster qstate (arena-allocated) */
			MktMemCtx		  prev		 = mkt_memctx_switch(ctx->arena);
			RaBitQQueryState *cluster_qs = mkt_rabitq_prepare_query_ex(
					idx->rq_params, qref, cref, mode);
			mkt_memctx_switch(prev);

			/* Batch distance computation */
			mkt_rabitq_distance_batch_multi_with_bound(
					cluster_qs,
					pl->f_add,
					pl->f_rescale,
					pl->bits,
					packed_bytes,
					pl->count,
					dim,
					ctx->scan_distances,
					ctx->scan_lower_bounds,
					ctx->scan_scratch);

			/* Two-stage: prune via lower bound, rerank
			 * survivors with exact L2 distance. */
			Distance threshold = mkt_topk_threshold(&ctx->topk);
			float	 g_error   = cluster_qs->g_error;

			for (uint32_t vi = 0; vi < pl->count; vi++)
			{
				Distance lb = ctx->scan_distances[vi] -
							  pl->f_error[vi] * g_error;

				if (lb >= threshold)
					continue;

				/* Rerank with exact distance */
				const float *vec = pl->vectors + (size_t)vi * dim;
				Distance	 d	 = mkt_l2_distance_squared(qvec, vec, dim);
				mkt_topk_insert(&ctx->topk, d, 0.0f, pl->ids[vi]);
				threshold = mkt_topk_threshold(&ctx->topk);
			}

			/* cluster_qs freed on arena reset */
		}
		else
		{
			/* Brute-force L2 scan (no RaBitQ data) */
			for (uint32_t vi = 0; vi < pl->count; vi++)
			{
				const float *vec = pl->vectors + (size_t)vi * dim;
				Distance	 d	 = mkt_l2_distance_squared(qvec, vec, dim);
				mkt_topk_insert(&ctx->topk, d, 0.0f, pl->ids[vi]);
			}
		}
	}

	/* Extract results into pre-allocated buffer */
	uint32_t count;
	if (ctx->topk.cand_count <= ctx->extract_cap)
	{
		mkt_topk_extract_sorted(&ctx->topk, ctx->extract_buf, &count);
	}
	else
	{
		/* Rare: more candidates than pre-sized. */
		MktTopKEntry *tmp = mkt_alloc(
				ctx->topk.cand_count * sizeof(MktTopKEntry));
		mkt_topk_extract_sorted(&ctx->topk, tmp, &count);
		if (count > ctx->extract_cap)
			count = ctx->extract_cap;
		memcpy(ctx->extract_buf, tmp, count * sizeof(MktTopKEntry));
		mkt_free(tmp);
	}

	uint32_t out = count < k ? count : k;
	for (uint32_t i = 0; i < out; i++)
		result_ids[i] = (uint32_t)ctx->extract_buf[i].id;

	return out;
}
