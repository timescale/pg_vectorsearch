/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * posting_build.c - Posting list construction
 *
 * Vector-to-cluster assignment (tree descent, SOAR, boundary
 * replication) and streaming page builder. The page format (AoS or
 * fastscan) is selected at init time via page_ops callbacks.
 */

#include "vs_config.h"

#include <math.h>
#include <string.h>

#ifdef VS_HAVE_CBLAS
/* See matrix.c for the rationale on the Apple branch. */
#ifdef __APPLE__
#include <vecLib/cblas_new.h>
#else
#include <cblas.h>
#endif
#endif

#include "algo/distance.h"
#include "algo/vecops.h"
#include "core/memory.h"
#include "index/index_build.h"
#include "index/parallel_build.h"
#include "index/posting_build.h"
#include "index/query_scan.h"
#include "quant/fastscan.h"

/* ================================================================
 * Vector-to-cluster assignment
 *
 * Building a posting list assigns each vector to a PRIMARY cluster and,
 * when replication is enabled, to a SECONDARY (replica) cluster. These
 * two use deliberately different methods, because they are different
 * problems:
 *
 *   PRIMARY  -> tree descent (vs_hkmeans_assign).
 *     The nearest centroid via a greedy descent of the k-means tree:
 *     O(fan_out * nlevels) distance evaluations, a single root-to-leaf
 *     path. Two reasons it is the right tool here:
 *       1. It is far cheaper than scanning every leaf. For primary we
 *          only need the (approximate) nearest, and a greedy descent
 *          finds it in ~log(nlist) work.
 *       2. It MATCHES the query path. Queries route to clusters by
 *          descending the same tree, so assigning vectors the same way
 *          makes a vector and a nearby query land in the same cluster.
 *          Assigning by exact-nearest instead would create a
 *          build/query routing mismatch — some vectors would sit in
 *          their exact-nearest leaf that a nearby query's greedy descent
 *          never reaches — which measurably lowers recall. (Measured:
 *          exact brute-force primary was both slower and slightly lower
 *          recall than tree descent.)
 *
 *   SECONDARY (SOAR + boundary) -> per-vector path in
 *     prism_build_assign_vector: boundary via a wider tree beam
 *     (vs_hkmeans_assign_topk), SOAR via a per-vector SIMD scan
 *     (prism_find_soar_secondary).
 * ================================================================ */

static void
normalize_vec(float *out, const float *in, Dimension dim)
{
	float norm = 0.0f;
	for (Dimension d = 0; d < dim; d++)
		norm += in[d] * in[d];

	if (norm > 0.0f)
	{
		float inv = 1.0f / sqrtf(norm);
		for (Dimension d = 0; d < dim; d++)
			out[d] = in[d] * inv;
	}
	else
	{
		memcpy(out, in, dim * sizeof(float));
	}
}

PrismBuildWorkerBufs
prism_build_worker_bufs_create(Dimension dim)
{
	return (PrismBuildWorkerBufs){
			.norm_buf	  = vs_alloc(dim * sizeof(float)),
			.residual_buf = vs_alloc(dim * sizeof(float)),
			.cand_leaves  = vs_alloc(PRISM_SECONDARY_TOPK * sizeof(uint32_t)),
			.cand_dists	  = vs_alloc(PRISM_SECONDARY_TOPK * sizeof(Distance)),
	};
}

void
prism_build_worker_bufs_free(PrismBuildWorkerBufs *bufs)
{
	vs_free(bufs->norm_buf);
	vs_free(bufs->residual_buf);
	vs_free(bufs->cand_leaves);
	vs_free(bufs->cand_dists);
	bufs->norm_buf	   = NULL;
	bufs->residual_buf = NULL;
	bufs->cand_leaves  = NULL;
	bufs->cand_dists   = NULL;
}

PrismBuildAssignment
prism_build_assign_vector(
		const HKMeansResult		*tree,
		const float				*vec,
		const PrismAssignParams *params,
		PrismBuildWorkerBufs	*bufs)
{
	const Dimension dim = params->dim;

	Distance min_dist;
	uint32_t best_c = vs_hkmeans_assign(tree, vec, params->metric, &min_dist);

	const float *enc_vec = vec;
	if (params->metric == DISTANCE_COSINE)
	{
		normalize_vec(bufs->norm_buf, vec, dim);
		enc_vec = bufs->norm_buf;
	}

	uint32_t secondary = PRISM_INVALID_CLUSTER;

	bool has_soar	  = params->soar_lambda > 0.0;
	bool has_boundary = params->boundary_epsilon > 0.0;

	if (has_soar || has_boundary)
	{
		const float *leaves = hk_leaf_centroids(tree);

		/* Beam-descend once for the nearest leaves. These candidates serve
		 * both the boundary test (2nd-nearest) and the SOAR search: the SOAR
		 * objective is minimized by a near leaf, so the secondary is always
		 * among them — avoiding an O(nleaves) full scan per vector, which does
		 * not scale to large nlist. */
		uint32_t ncand = vs_hkmeans_assign_topk(
				tree,
				enc_vec,
				params->metric,
				PRISM_SECONDARY_TOPK,
				PRISM_SECONDARY_BEAM_WIDTH,
				bufs->cand_leaves,
				bufs->cand_dists);

		uint32_t boundary_c2 = best_c;
		if (has_boundary)
			boundary_c2 = prism_find_secondary_cluster(
					bufs->cand_leaves,
					bufs->cand_dists,
					ncand,
					best_c,
					min_dist,
					params->boundary_epsilon);

		bool should_replicate = has_boundary ? (boundary_c2 != best_c) : true;

		if (should_replicate)
		{
			if (has_soar)
			{
				const float *cent = leaves + (size_t)best_c * dim;
				float		*r	  = bufs->residual_buf;
				float		 norm = 0.0f;
				for (Dimension d = 0; d < dim; d++)
				{
					r[d] = enc_vec[d] - cent[d];
					norm += r[d] * r[d];
				}
				if (norm > 1e-7f)
				{
					float inv = 1.0f / sqrtf(norm);
					for (Dimension d = 0; d < dim; d++)
						r[d] *= inv;
				}
				secondary = prism_find_soar_secondary(
						enc_vec,
						leaves,
						bufs->cand_leaves,
						ncand,
						dim,
						best_c,
						r,
						params->soar_lambda);
			}
			else
			{
				secondary = boundary_c2;
			}

			if (secondary == best_c)
				secondary = PRISM_INVALID_CLUSTER;
		}
	}

	return (PrismBuildAssignment){
			.primary	= best_c,
			.secondary	= secondary,
			.enc_vector = enc_vec,
	};
}

/* ----------------------------------------------------------------
 * Compact posting entry (cluster-sorted build path) — see posting_build.h.
 * ---------------------------------------------------------------- */

Size
prism_posting_entry_size(Dimension dim)
{
	return sizeof(ItemPointerData) + 3 * sizeof(float) + (Size)((dim + 7) / 8);
}

void
prism_posting_entry_encode(
		const RaBitQParams *params,
		const float		   *vec,
		const float		   *centroid,
		Dimension			dim,
		RaBitQData		   *enc_buf,
		RaBitQScratch	   *scratch,
		ItemPointerData		tid,
		void			   *out_entry)
{
	Vec32Ref vref = {.data = vec, .dim = dim};
	Vec32Ref cref = {.data = centroid, .dim = dim};
	vs_rabitq_encode_into_ex(params, vref, cref, enc_buf, scratch);
	float f_error =
			vs_rabitq_derive_f_error(enc_buf->f_add, enc_buf->f_rescale, dim);

	char *p = (char *)out_entry;
	memcpy(p, &tid, sizeof(ItemPointerData));
	p += sizeof(ItemPointerData);
	memcpy(p, &enc_buf->f_add, sizeof(float));
	p += sizeof(float);
	memcpy(p, &enc_buf->f_rescale, sizeof(float));
	p += sizeof(float);
	memcpy(p, &f_error, sizeof(float));
	p += sizeof(float);
	memcpy(p, enc_buf->bits, (size_t)((dim + 7) / 8));
}

void
prism_posting_entry_encode_from_pt(
		const RaBitQParams *params,
		const float		   *pt_residual,
		Dimension			dim,
		RaBitQData		   *enc_buf,
		RaBitQScratch	   *scratch,
		ItemPointerData		tid,
		void			   *out_entry)
{
	/* Page-backed twin of prism_posting_entry_encode: the caller already has
	 * the rotated residual (pt_query - pt_centroid), e.g. from routing + the
	 * head page's pt_centroid, so no float centroid is needed. */
	vs_rabitq_encode_from_pt(params, pt_residual, enc_buf, scratch);
	float f_error =
			vs_rabitq_derive_f_error(enc_buf->f_add, enc_buf->f_rescale, dim);

	char *p = (char *)out_entry;
	memcpy(p, &tid, sizeof(ItemPointerData));
	p += sizeof(ItemPointerData);
	memcpy(p, &enc_buf->f_add, sizeof(float));
	p += sizeof(float);
	memcpy(p, &enc_buf->f_rescale, sizeof(float));
	p += sizeof(float);
	memcpy(p, &f_error, sizeof(float));
	p += sizeof(float);
	memcpy(p, enc_buf->bits, (size_t)((dim + 7) / 8));
}

void
prism_build_route_ctx_init(
		PrismBuildRouteCtx *ctx,
		PrismQueryState	   *qs,
		PrismSorter		   *sorter,
		const RaBitQParams *rq_params,
		VsStorage		   *storage,
		BlockNumber			first_posting,
		Dimension			dim,
		double				soar_lambda,
		double				boundary_epsilon)
{
	ctx->qs				  = qs;
	ctx->sorter			  = sorter;
	ctx->rq_params		  = rq_params;
	ctx->storage		  = storage;
	ctx->first_posting	  = first_posting;
	ctx->dim			  = dim;
	ctx->soar_lambda	  = soar_lambda;
	ctx->boundary_epsilon = boundary_epsilon;

	ctx->cand_pt = vs_alloc(
			(size_t)PRISM_SECONDARY_TOPK * dim * sizeof(float));
	ctx->cand_leaf = vs_alloc(PRISM_SECONDARY_TOPK * sizeof(uint32_t));
	ctx->cand_dist = vs_alloc(PRISM_SECONDARY_TOPK * sizeof(Distance));
	ctx->pt_r	   = vs_alloc((size_t)dim * sizeof(float));
	ctx->enc_buf   = vs_alloc(VS_RABITQ_DATA_SIZE(dim));
	vs_rabitq_scratch_init(&ctx->enc_scratch, dim);
	ctx->entry = vs_alloc(prism_posting_entry_size(dim));

	ctx->indtuples	= 0;
	ctx->soar_dupes = 0;
}

void
prism_build_route_ctx_cleanup(PrismBuildRouteCtx *ctx)
{
	vs_free(ctx->cand_pt);
	vs_free(ctx->cand_leaf);
	vs_free(ctx->cand_dist);
	vs_free(ctx->pt_r);
	vs_free(ctx->enc_buf);
	vs_rabitq_scratch_cleanup(&ctx->enc_scratch);
	vs_free(ctx->entry);
}

/* Swap two gathered route candidates (leaf, exact distance, and the
 * pt_centroid row, via the ctx's pt_r scratch). */
static void
route_swap_candidates(
		PrismBuildRouteCtx *ctx, uint32_t a, uint32_t b, Dimension dim)
{
	uint32_t leaf = ctx->cand_leaf[a];
	Distance dist = ctx->cand_dist[a];

	ctx->cand_leaf[a] = ctx->cand_leaf[b];
	ctx->cand_dist[a] = ctx->cand_dist[b];
	ctx->cand_leaf[b] = leaf;
	ctx->cand_dist[b] = dist;

	float *pa	  = ctx->cand_pt + (size_t)a * dim;
	float *pb	  = ctx->cand_pt + (size_t)b * dim;
	size_t nbytes = (size_t)dim * sizeof(float);
	memcpy(ctx->pt_r, pa, nbytes);
	memcpy(pa, pb, nbytes);
	memcpy(pb, ctx->pt_r, nbytes);
}

/*
 * Stamp a serialized posting entry as unreachable: estimated distance
 * +inf with zero error, so every scan prunes it before it can enter
 * the top-k threshold heap or the candidate buffer. Used for vectors
 * with no defined distance under the index metric (a zero-norm vector
 * under cosine): their exact distance is NaN, but their quantized
 * estimate would look mid-range -- an impostor entering the top-k
 * upper-bound heap of every query and falsely tightening its pruning
 * threshold. The row stays indexed; it just can never be a result.
 */
static void
mark_entry_unreachable(void *entry)
{
	/* Serialized layout (see prism_posting_entry_encode_from_pt):
	 * [tid][f_add][f_rescale][f_error][bits]. p starts at f_add; the
	 * advance steps over the just-written f_add plus the untouched
	 * f_rescale to land exactly on f_error. */
	char	   *p	 = (char *)entry + sizeof(ItemPointerData);
	const float inf	 = INFINITY;
	const float zero = 0.0f;

	memcpy(p, &inf, sizeof(float));
	p += 2 * sizeof(float);
	memcpy(p, &zero, sizeof(float));
}

bool
prism_build_route_emit(
		PrismBuildRouteCtx *ctx, const float *vec, ItemPointerData tid)
{
	Dimension dim = ctx->dim;

	/* No defined distance under cosine: encode as unreachable below.
	 * Squared norm -- zero iff the norm is zero -- skips the sqrt on
	 * this per-tuple build path. */
	bool degenerate = ctx->qs->index->metric == DISTANCE_COSINE &&
					  vs_l2_norm_squared(vec, dim) == 0.0f;

	/*
	 * Route page-backed, exactly as the query/insert do: descend the centroid
	 * pages, giving the nearest leaves' posting-head blocks + qs->pt_query
	 * (the rotated vector). No in-RAM tree.
	 */
	uint32_t n = prism_query_route(
			ctx->qs,
			vec,
			PRISM_SECONDARY_TOPK,
			VS_DISTANCE_MODE_ASYMMETRIC,
			NULL);
	if (n == 0)
		return false;

	bool has_soar	  = ctx->soar_lambda > 0.0;
	bool has_boundary = ctx->boundary_epsilon > 0.0;

	/* Gather the beam candidates: leaf index and pt_centroid (read from each
	 * head page -- the full-precision float encode reference). The gather is
	 * unconditional: the exact primary re-rank below needs every finalist's
	 * pt_centroid, secondary selection or not. */
	for (uint32_t i = 0; i < n; i++)
	{
		BlockNumber h	  = ctx->qs->beam_results[i].posting_head;
		ctx->cand_leaf[i] = prism_route_head_to_leaf(ctx->first_posting, h);
		Page hp			  = vs_storage_read_page(ctx->storage, h);
		memcpy(ctx->cand_pt + (size_t)i * dim,
			   prism_posting_pt_centroid(hp),
			   (size_t)dim * sizeof(float));
		vs_storage_release_page(ctx->storage, h);
	}

	/*
	 * Exact re-rank of the leaf finalists: the beam's leaf-level distances
	 * are 1-bit estimates whose noise misfiles rows, but each finalist's
	 * full-precision rotated centroid was just gathered, so one O(dim)
	 * distance per candidate recovers the true order (P^T preserves both
	 * L2 and dot products). The re-rank must use the index metric so the
	 * primary matches query-time routing: inner product ranks by -dot,
	 * while L2 and cosine rank by squared L2 (cosine queries and encode
	 * references are unit-normalized, making the two orders identical).
	 * Only the first two positions are order-sensitive -- the primary and,
	 * for the boundary gate, the exact 2nd-nearest -- and SOAR scans all
	 * candidates order-independently, so a top-2 selection suffices.
	 */
	bool rank_by_dot = ctx->qs->index->metric == DISTANCE_INNER_PRODUCT;
	for (uint32_t i = 0; i < n; i++)
	{
		const float *cand = ctx->cand_pt + (size_t)i * dim;
		ctx->cand_dist[i] =
				rank_by_dot
						? -vs_dot_product(ctx->qs->pt_query, cand, dim)
						: vs_l2_distance_squared(ctx->qs->pt_query, cand, dim);
	}

	uint32_t top = (n < 2) ? n : 2;
	for (uint32_t r = 0; r < top; r++)
	{
		uint32_t best = r;
		for (uint32_t i = r + 1; i < n; i++)
			if (ctx->cand_dist[i] < ctx->cand_dist[best])
				best = i;
		if (best != r)
			route_swap_candidates(ctx, r, best, dim);
	}

	/* Primary: encode pt_query - pt_centroid[0] and stream to the sorter. */
	uint32_t primary = ctx->cand_leaf[0];
	for (Dimension d = 0; d < dim; d++)
		ctx->pt_r[d] = ctx->qs->pt_query[d] - ctx->cand_pt[d];
	prism_posting_entry_encode_from_pt(
			ctx->rq_params,
			ctx->pt_r,
			dim,
			ctx->enc_buf,
			&ctx->enc_scratch,
			tid,
			ctx->entry);
	if (degenerate)
		mark_entry_unreachable(ctx->entry);
	prism_pbuild_sort_put(ctx->sorter, primary, ctx->entry);
	ctx->indtuples++;

	/* Secondary (SOAR / boundary), in rotated space over the gathered
	 * candidates (positions index cand_pt). OA scoring and distances are
	 * norm-preserving under P^T, so this matches the float-space result. */
	if (!has_soar && !has_boundary)
		return false;

	bool boundary_repl = false;
	if (has_boundary && n > 1)
	{
		double d0 = (double)ctx->cand_dist[0];
		double gr = (d0 != 0.0) ? ((double)ctx->cand_dist[1] - d0) / fabs(d0)
								: INFINITY;
		boundary_repl = gr <= ctx->boundary_epsilon;
	}
	bool	 should	 = has_boundary ? boundary_repl : true;
	uint32_t sec_pos = 0; /* 0 = primary position = no secondary */
	if (should && has_soar)
	{
		float norm = 0.0f;
		for (Dimension d = 0; d < dim; d++)
			norm += ctx->pt_r[d] * ctx->pt_r[d];
		if (norm > 1e-7f)
		{
			float inv = 1.0f / sqrtf(norm);
			for (Dimension d = 0; d < dim; d++)
				ctx->pt_r[d] *= inv;
		}
		/* cand_leaves = NULL -> candidate position is the index; primary is
		 * position 0. leaf_centroids = cand_pt (rotated); vec = pt_query. */
		sec_pos = prism_find_soar_secondary(
				ctx->qs->pt_query,
				ctx->cand_pt,
				NULL,
				n,
				dim,
				0,
				ctx->pt_r,
				ctx->soar_lambda);
	}
	else if (should && n > 1)
	{
		sec_pos = 1; /* boundary-only: the 2nd-nearest candidate */
	}

	if (sec_pos != 0 && ctx->cand_leaf[sec_pos] != primary)
	{
		for (Dimension d = 0; d < dim; d++)
			ctx->pt_r[d] = ctx->qs->pt_query[d] -
						   ctx->cand_pt[(size_t)sec_pos * dim + d];
		prism_posting_entry_encode_from_pt(
				ctx->rq_params,
				ctx->pt_r,
				dim,
				ctx->enc_buf,
				&ctx->enc_scratch,
				tid,
				ctx->entry);
		if (degenerate)
			mark_entry_unreachable(ctx->entry);
		prism_pbuild_sort_put(
				ctx->sorter, ctx->cand_leaf[sec_pos], ctx->entry);
		ctx->soar_dupes++;
		return true;
	}
	return false;
}

void
prism_posting_entry_add(
		PrismPostingBuilder *builder, const void *entry, Dimension dim)
{
	const char	   *p = (const char *)entry;
	ItemPointerData tid;
	float			f_add, f_rescale, f_error;
	memcpy(&tid, p, sizeof(ItemPointerData));
	p += sizeof(ItemPointerData);
	memcpy(&f_add, p, sizeof(float));
	p += sizeof(float);
	memcpy(&f_rescale, p, sizeof(float));
	p += sizeof(float);
	memcpy(&f_error, p, sizeof(float));
	p += sizeof(float);
	(void)dim;
	prism_posting_builder_add_encoded(
			builder, tid, f_add, f_rescale, f_error, (const uint8_t *)p);
}

/* ----------------------------------------------------------------
 * Shared helpers
 * ---------------------------------------------------------------- */

/*
 * Flush the in-memory page: write to storage and link
 * into the chain. Prefers reserved contiguous blocks.
 */
static void
flush_page(PrismPostingBuilder *builder)
{
	/*
	 * The first page of a chain (the posting-list head) must always be
	 * materialized, even with no entries: it carries the cluster's centroid
	 * metadata and pt_centroid, anchors the chain, and is where a scan begins.
	 * An empty head arises in the parallel bounded build, where the leader
	 * synthesizes the head while the workers stream the continuations (for
	 * fastscan nothing is merged into the head, so it stays empty). Later
	 * pages are still skipped when they hold no new entries.
	 */
	if (!builder->page_dirty && !builder->is_first)
		return;

	BlockNumber blkno;
	Page		spage;
	if (builder->is_first && builder->fixed_first_blkno != InvalidBlockNumber)
	{
		blkno = builder->fixed_first_blkno;
		spage = vs_storage_write_page(builder->storage, blkno);
	}
	else
	{
		spage = vs_storage_new_page(builder->storage, &blkno);
	}

	memcpy(spage, builder->mem_page, BLCKSZ);
	vs_storage_commit_page(builder->storage, blkno);

	if (builder->is_first)
	{
		builder->head_blkno = blkno;
		builder->is_first	= false;
	}

	if (builder->prev_blkno != InvalidBlockNumber)
	{
		Page prev =
				vs_storage_write_page(builder->storage, builder->prev_blkno);
		prism_posting_opaque(prev)->next_blkno = blkno;
		vs_storage_commit_page(builder->storage, builder->prev_blkno);
	}
	builder->prev_blkno = blkno;

	builder->page_ops->reinit_page(builder);
	builder->page_dirty = false;
}

/*
 * Common init for both formats. When pt_centroid is NULL, initializes
 * a continuation page (no pt_centroid header, overflow flags).
 */
static void
builder_init_common(
		PrismPostingBuilder		  *builder,
		VsStorage				  *storage,
		const RaBitQParams		  *params,
		Dimension				   dim,
		uint32_t				   cluster_id,
		const float				  *centroid,
		const float				  *pt_centroid,
		const PrismPostingPageOps *ops,
		uint16_t				   first_page_flags)
{
	memset(builder, 0, sizeof(*builder));
	builder->storage	= storage;
	builder->params		= params;
	builder->dim		= dim;
	builder->cluster_id = cluster_id;
	builder->centroid	= centroid;
	builder->head_blkno = InvalidBlockNumber;
	builder->prev_blkno = InvalidBlockNumber;
	builder->is_first	= true;
	builder->owns_head	= (first_page_flags & PRISM_POSTING_PAGE_FIRST) != 0;
	builder->n_entries	= 0;
	builder->page_ops	= ops;

	builder->fixed_first_blkno = InvalidBlockNumber;

	prism_posting_page_init(
			builder->mem_page, cluster_id, dim, first_page_flags);
	if (pt_centroid != NULL)
		memcpy(prism_posting_pt_centroid_mut(builder->mem_page),
			   pt_centroid,
			   dim * sizeof(float));

	builder->enc_buf = params ? vs_alloc(VS_RABITQ_DATA_SIZE(dim)) : NULL;
	if (params)
		vs_rabitq_scratch_init(&builder->enc_scratch, dim);
}

/* ----------------------------------------------------------------
 * AoS page ops — one entry per call
 * ---------------------------------------------------------------- */

static bool
aos_write_entry(
		PrismPostingBuilder *builder,
		ItemPointerData		 tid,
		float				 f_add,
		float				 f_rescale,
		float				 f_error,
		const uint8_t		*bits)
{
	if (!prism_posting_page_has_room(builder->mem_page))
		return false;

	prism_posting_page_add(
			builder->mem_page,
			builder->dim,
			tid,
			f_add,
			f_rescale,
			f_error,
			bits,
			0);
	return true;
}

static void
aos_reinit_page(PrismPostingBuilder *builder)
{
	prism_posting_page_init(
			builder->mem_page,
			builder->cluster_id,
			builder->dim,
			PRISM_POSTING_PAGE_OVERFLOW);
}

static void
aos_finalize(PrismPostingBuilder *builder)
{
	(void)builder;
}

static void
aos_cleanup(PrismPostingBuilder *builder)
{
	(void)builder;
}

static const PrismPostingPageOps aos_page_ops = {
		.write_entry = aos_write_entry,
		.reinit_page = aos_reinit_page,
		.finalize	 = aos_finalize,
		.cleanup	 = aos_cleanup,
};

/* ----------------------------------------------------------------
 * Fastscan page ops — buffer 32 entries, pack group
 * ---------------------------------------------------------------- */

/*
 * Pack the current group into fastscan SoA layout and write it
 * to the in-memory page.
 */
static void
fs_write_group(PrismPostingBuilder *builder)
{
	FsGroupStage *grp = &builder->fs.grp;
	if (grp->count == 0)
		return;

	Dimension dim = builder->dim;

	PrismPostingPageOpaque *op = prism_posting_opaque(builder->mem_page);
	char *content = prism_posting_page_content(builder->mem_page, dim);

	uint32_t g = builder->fs.groups_on_page;

	memcpy(prism_fastscan_group_tids(content, g, dim),
		   grp->tids,
		   VS_FASTSCAN_GROUP * sizeof(ItemPointerData));
	memcpy(prism_fastscan_group_f_add(content, g, dim),
		   grp->f_add,
		   VS_FASTSCAN_GROUP * sizeof(float));
	memcpy(prism_fastscan_group_f_rescale(content, g, dim),
		   grp->f_rescale,
		   VS_FASTSCAN_GROUP * sizeof(float));
	memcpy(prism_fastscan_group_f_error(content, g, dim),
		   grp->f_error,
		   VS_FASTSCAN_GROUP * sizeof(float));

	vs_fastscan_pack_codes(
			builder->fs.bits_buf, grp->count, dim, builder->fs.codes_buf);
	memcpy(prism_fastscan_group_codes(content, g, dim),
		   builder->fs.codes_buf,
		   VS_FASTSCAN_GROUP_BYTES(dim));

	op->entry_count += (uint16_t)grp->count;
	builder->fs.groups_on_page++;
	builder->page_dirty = true;

	memset(grp, 0, sizeof(*grp));
	memset(builder->fs.bits_buf,
		   0,
		   (size_t)VS_FASTSCAN_GROUP * VS_RABITQ_BYTES(dim));
}

static bool
fs_write_entry(
		PrismPostingBuilder *builder,
		ItemPointerData		 tid,
		float				 f_add,
		float				 f_rescale,
		float				 f_error,
		const uint8_t		*bits)
{
	FsGroupStage *grp		   = &builder->fs.grp;
	uint32_t	  idx		   = grp->count;
	uint32_t	  packed_bytes = VS_RABITQ_BYTES(builder->dim);

	grp->tids[idx]		= tid;
	grp->f_add[idx]		= f_add;
	grp->f_rescale[idx] = f_rescale;
	grp->f_error[idx]	= f_error;
	memcpy(builder->fs.bits_buf + (size_t)idx * packed_bytes,
		   bits,
		   packed_bytes);
	grp->count++;

	if (grp->count == VS_FASTSCAN_GROUP)
	{
		if (builder->fs.groups_on_page >= builder->fs.max_groups)
			flush_page(builder);
		fs_write_group(builder);
	}

	return true;
}

static void
fs_reinit_page(PrismPostingBuilder *builder)
{
	prism_posting_page_init(
			builder->mem_page,
			builder->cluster_id,
			builder->dim,
			PRISM_POSTING_PAGE_OVERFLOW | PRISM_POSTING_PAGE_FASTSCAN);
	builder->fs.groups_on_page = 0;
	builder->fs.max_groups = prism_fastscan_max_groups(builder->dim, false);
}

static void
fs_finalize(PrismPostingBuilder *builder)
{
	if (builder->fs.groups_on_page >= builder->fs.max_groups)
		flush_page(builder);
	fs_write_group(builder);
}

static void
fs_cleanup(PrismPostingBuilder *builder)
{
	vs_free(builder->fs.bits_buf);
	builder->fs.bits_buf = NULL;
	vs_free(builder->fs.codes_buf);
	builder->fs.codes_buf = NULL;
}

static const PrismPostingPageOps fs_page_ops = {
		.write_entry = fs_write_entry,
		.reinit_page = fs_reinit_page,
		.finalize	 = fs_finalize,
		.cleanup	 = fs_cleanup,
};

/* ----------------------------------------------------------------
 * Public API — init
 * ---------------------------------------------------------------- */

void
prism_posting_builder_init(
		PrismPostingBuilder *builder,
		VsStorage			*storage,
		const RaBitQParams	*params,
		Dimension			 dim,
		uint32_t			 cluster_id,
		const float			*centroid,
		const float			*pt_centroid)
{
	builder_init_common(
			builder,
			storage,
			params,
			dim,
			cluster_id,
			centroid,
			pt_centroid,
			&aos_page_ops,
			PRISM_POSTING_PAGE_FIRST);
}

void
prism_posting_builder_init_fastscan(
		PrismPostingBuilder *builder,
		VsStorage			*storage,
		const RaBitQParams	*params,
		Dimension			 dim,
		uint32_t			 cluster_id,
		const float			*centroid,
		const float			*pt_centroid)
{
	builder_init_common(
			builder,
			storage,
			params,
			dim,
			cluster_id,
			centroid,
			pt_centroid,
			&fs_page_ops,
			PRISM_POSTING_PAGE_FIRST | PRISM_POSTING_PAGE_FASTSCAN);

	uint32_t packed_bytes = VS_RABITQ_BYTES(dim);
	builder->fs.bits_buf  = vs_alloc((size_t)VS_FASTSCAN_GROUP * packed_bytes);
	builder->fs.codes_buf = vs_alloc(VS_FASTSCAN_GROUP_BYTES(dim));
	builder->fs.max_groups = prism_fastscan_max_groups(dim, true);
}

void
prism_posting_builder_init_fmt(
		PrismPostingBuilder *builder,
		VsStorage			*storage,
		const RaBitQParams	*params,
		Dimension			 dim,
		uint32_t			 cluster_id,
		const float			*centroid,
		const float			*pt_centroid,
		bool				 fastscan)
{
	if (fastscan)
		prism_posting_builder_init_fastscan(
				builder,
				storage,
				params,
				dim,
				cluster_id,
				centroid,
				pt_centroid);
	else
		prism_posting_builder_init(
				builder,
				storage,
				params,
				dim,
				cluster_id,
				centroid,
				pt_centroid);
}

void
prism_posting_builder_adopt_head(
		PrismPostingBuilder *builder,
		VsStorage			*storage,
		const RaBitQParams	*params,
		Dimension			 dim,
		uint32_t			 cluster_id,
		BlockNumber			 head_blk,
		bool				 fastscan)
{
	memset(builder, 0, sizeof(*builder));
	builder->storage	= storage;
	builder->params		= params;
	builder->dim		= dim;
	builder->cluster_id = cluster_id;
	builder->centroid	= NULL; /* entries arrive pre-encoded */
	builder->head_blkno = InvalidBlockNumber;
	builder->prev_blkno = InvalidBlockNumber;
	builder->is_first	= true;
	builder->owns_head	= true;
	builder->page_ops	= fastscan ? &fs_page_ops : &aos_page_ops;

	builder->fixed_first_blkno = head_blk;
	builder->adopted_head	   = true;

	/* Start from the pre-written head itself: it already carries the
	 * cluster's header and pt_centroid. It must be empty — the pre-scan
	 * writer never adds entries. */
	Page hp = vs_storage_read_page(storage, head_blk);
	Assert(prism_posting_opaque(hp)->live_count == 0);
	memcpy(builder->mem_page, hp, BLCKSZ);
	vs_storage_release_page(storage, head_blk);

	/* No encoder scratch: an adopted head only ever receives pre-encoded
	 * entries (prism_posting_entry_add from the cluster-keyed sort), and this
	 * runs once per posting list -- per-list encoder allocations would be
	 * O(nlist) churn for buffers the raw-vector add path never touches. */
	builder->enc_buf = NULL;

	if (fastscan)
	{
		uint32_t packed_bytes = VS_RABITQ_BYTES(dim);
		builder->fs.bits_buf  = vs_alloc(
				 (size_t)VS_FASTSCAN_GROUP * packed_bytes);
		builder->fs.codes_buf  = vs_alloc(VS_FASTSCAN_GROUP_BYTES(dim));
		builder->fs.max_groups = prism_fastscan_max_groups(dim, true);
	}
}

void
prism_posting_builder_set_first_blkno(
		PrismPostingBuilder *builder, BlockNumber blkno)
{
	builder->fixed_first_blkno = blkno;
}

void
prism_posting_builder_add_encoded(
		PrismPostingBuilder *builder,
		ItemPointerData		 tid,
		float				 f_add,
		float				 f_rescale,
		float				 f_error,
		const uint8_t		*bits)
{
	if (!builder->page_ops
				 ->write_entry(builder, tid, f_add, f_rescale, f_error, bits))
	{
		flush_page(builder);
		builder->page_ops
				->write_entry(builder, tid, f_add, f_rescale, f_error, bits);
	}
	builder->page_dirty = true;
	builder->n_entries++;
}

void
prism_posting_builder_add(
		PrismPostingBuilder *builder, ItemPointerData tid, const float *vector)
{
	prism_posting_builder_add_ex(builder, tid, vector, false);
}

void
prism_posting_builder_add_ex(
		PrismPostingBuilder *builder,
		ItemPointerData		 tid,
		const float			*vector,
		bool				 unreachable)
{
	/* Adopted heads carry no encoder (their entries arrive pre-encoded). */
	Assert(builder->enc_buf != NULL);

	Vec32Ref vref = {.data = vector, .dim = builder->dim};
	Vec32Ref cref = {.data = builder->centroid, .dim = builder->dim};
	vs_rabitq_encode_into_ex(
			builder->params,
			vref,
			cref,
			builder->enc_buf,
			&builder->enc_scratch);

	float f_add	  = builder->enc_buf->f_add;
	float f_error = vs_rabitq_derive_f_error(
			f_add, builder->enc_buf->f_rescale, builder->dim);

	/* Same stamp the build and insert paths apply -- see the header. */
	if (unreachable)
	{
		f_add	= INFINITY;
		f_error = 0.0f;
	}

	prism_posting_builder_add_encoded(
			builder,
			tid,
			f_add,
			builder->enc_buf->f_rescale,
			f_error,
			builder->enc_buf->bits);
}

BlockNumber
prism_posting_builder_finish(PrismPostingBuilder *builder)
{
	builder->page_ops->finalize(builder);

	/* An adopted head that received no entries is already final on disk
	 * (the pre-scan writer stamped it empty): skip the rewrite and the
	 * restamp below, which would clobber its tail_blkno with an invalid
	 * chain tail. */
	if (builder->adopted_head && builder->is_first && !builder->page_dirty)
	{
		builder->head_blkno = builder->fixed_first_blkno;
		return builder->head_blkno;
	}

	flush_page(builder);

	/*
	 * Stamp the head's per-cluster metadata from state the builder already
	 * tracked, so the runtime insert path finds the chain tail in O(1) with no
	 * walk. prev_blkno is the last page flushed (the chain tail) and n_entries
	 * is the exact entry count; for a head builder that holds the whole
	 * chain these values are final.
	 */
	if (builder->owns_head && builder->head_blkno != InvalidBlockNumber)
	{
		Page hp = vs_storage_write_page(builder->storage, builder->head_blkno);
		PrismPostingPageOpaque *op = prism_posting_opaque(hp);
		op->tail_blkno			   = builder->prev_blkno;
		op->live_count			   = builder->n_entries;
		vs_storage_commit_page(builder->storage, builder->head_blkno);
	}

	return builder->head_blkno;
}

void
prism_posting_builder_cleanup(PrismPostingBuilder *builder)
{
	builder->page_ops->cleanup(builder);
	if (builder->enc_buf != NULL)
	{
		vs_rabitq_scratch_cleanup(&builder->enc_scratch);
		vs_free(builder->enc_buf);
		builder->enc_buf = NULL;
	}
}

/* ----------------------------------------------------------------
 * Flat builder — one buffer per cluster (unchanged)
 * ---------------------------------------------------------------- */

void
prism_flat_posting_builder_init(
		PrismFlatPostingBuilder *builder,
		const RaBitQParams		*params,
		Dimension				 dim,
		uint32_t				 cluster_id,
		const float				*centroid,
		uint32_t				 count)
{
	builder->params		 = params;
	builder->dim		 = dim;
	builder->cluster_id	 = cluster_id;
	builder->centroid	 = centroid;
	builder->max_entries = count;

	size_t buf_size = prism_posting_flat_page_size(dim, count);
	builder->buf	= vs_alloc(buf_size);
	memset(builder->buf, 0, buf_size);
	prism_posting_flat_init(builder->buf, count, cluster_id);

	builder->enc_buf = vs_alloc(VS_RABITQ_DATA_SIZE(dim));
}

void
prism_flat_posting_builder_add(
		PrismFlatPostingBuilder *builder,
		ItemPointerData			 tid,
		const float				*vector)
{
	Dimension dim = builder->dim;

	Vec32Ref vref = {.data = vector, .dim = dim};
	Vec32Ref cref = {.data = builder->centroid, .dim = dim};
	vs_rabitq_encode_into(builder->params, vref, cref, builder->enc_buf);

	float f_error = vs_rabitq_derive_f_error(
			builder->enc_buf->f_add, builder->enc_buf->f_rescale, dim);

	prism_posting_flat_add(
			builder->buf,
			dim,
			tid,
			builder->enc_buf->f_add,
			builder->enc_buf->f_rescale,
			f_error,
			builder->enc_buf->bits,
			0);
}

char *
prism_flat_posting_builder_finish(PrismFlatPostingBuilder *builder)
{
	return builder->buf;
}

void
prism_flat_posting_builder_cleanup(PrismFlatPostingBuilder *builder)
{
	if (builder->enc_buf != NULL)
	{
		vs_free(builder->enc_buf);
		builder->enc_buf = NULL;
	}
}

/* See posting_build.h: the route/filter/normalize half of the refine pass,
 * shared verbatim by the serial and parallel builds. */
const float *
prism_refine_route_row(
		struct PrismQueryState *qs,
		BlockNumber				first_posting,
		const float			   *vec,
		Dimension				dim,
		bool					cosine,
		float				   *scratch,
		uint32_t				tile_lo,
		uint32_t				tile_hi,
		uint32_t			   *out_idx)
{
	uint32_t n =
			prism_query_route(qs, vec, 1, VS_DISTANCE_MODE_ASYMMETRIC, NULL);
	if (n == 0)
		return NULL;

	uint32_t leaf = prism_route_head_to_leaf(
			first_posting, qs->beam_results[0].posting_head);
	if (leaf < tile_lo || leaf >= tile_hi)
		return NULL;
	*out_idx = leaf - tile_lo;

	/* Cosine centroids are trained in normalized space, so the mean must
	 * average the normalized vectors. */
	if (cosine)
	{
		memcpy(scratch, vec, (size_t)dim * sizeof(float));
		vs_l2_normalize(scratch, dim);
		return scratch;
	}
	return vec;
}

void
prism_refine_write_means(
		const double	*sums,
		const uint64_t	*counts,
		uint32_t		 lo,
		uint32_t		 hi,
		Dimension		 dim,
		float			*scratch,
		PrismLeafWriteFn write_head,
		void			*write_head_ctx)
{
	for (uint32_t l = lo; l < hi; l++)
	{
		if (counts[l - lo] == 0)
			continue; /* keep the sample-trained head for an empty leaf */
		const double *sum = sums + (size_t)(l - lo) * dim;
		double		  inv = 1.0 / (double)counts[l - lo];
		for (Dimension j = 0; j < dim; j++)
			scratch[j] = (float)(sum[j] * inv);
		write_head(write_head_ctx, l, scratch);
	}
}

/* See posting_build.h: the one leaf-head writer both builds share. */
void
prism_write_leaf_head(void *arg, uint32_t leaf, const float *centroid)
{
	PrismHeadWriteCtx *h = (PrismHeadWriteCtx *)arg;
	vs_rabitq_rotate(h->rq_params, centroid, h->pt);

	PrismPostingBuilder hb;
	prism_posting_builder_init_fmt(
			&hb,
			h->storage,
			h->rq_params,
			h->dim,
			leaf,
			centroid,
			h->pt,
			h->fastscan);
	prism_posting_builder_set_first_blkno(&hb, h->first_posting + leaf);
	prism_posting_builder_finish(&hb);
	prism_posting_builder_cleanup(&hb);
}
