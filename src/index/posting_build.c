/*
 * posting_build.c - Posting list construction
 *
 * Vector-to-cluster assignment (tree descent, SOAR, boundary
 * replication) and streaming page builder. The page format (AoS or
 * fastscan) is selected at init time via page_ops callbacks.
 */

#include "mkt_config.h"

#include <math.h>
#include <string.h>

#ifdef MKT_HAVE_CBLAS
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
 *   PRIMARY  -> tree descent (mkt_hkmeans_assign).
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
 *     mkt_build_assign_vector: boundary via a wider tree beam
 *     (mkt_hkmeans_assign_topk), SOAR via a per-vector SIMD scan
 *     (mkt_find_soar_secondary).
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

MktBuildWorkerBufs
mkt_build_worker_bufs_create(Dimension dim)
{
	return (MktBuildWorkerBufs){
			.norm_buf	  = mkt_alloc(dim * sizeof(float)),
			.residual_buf = mkt_alloc(dim * sizeof(float)),
			.cand_leaves  = mkt_alloc(MKT_SECONDARY_TOPK * sizeof(uint32_t)),
			.cand_dists	  = mkt_alloc(MKT_SECONDARY_TOPK * sizeof(Distance)),
	};
}

void
mkt_build_worker_bufs_free(MktBuildWorkerBufs *bufs)
{
	mkt_free(bufs->norm_buf);
	mkt_free(bufs->residual_buf);
	mkt_free(bufs->cand_leaves);
	mkt_free(bufs->cand_dists);
	bufs->norm_buf	   = NULL;
	bufs->residual_buf = NULL;
	bufs->cand_leaves  = NULL;
	bufs->cand_dists   = NULL;
}

MktBuildAssignment
mkt_build_assign_vector(
		const HKMeansResult	 *tree,
		const float			 *vec,
		const MktBuildParams *params,
		MktBuildWorkerBufs	 *bufs)
{
	const Dimension dim = params->dim;

	Distance min_dist;
	uint32_t best_c = mkt_hkmeans_assign(tree, vec, params->metric, &min_dist);

	const float *enc_vec = vec;
	if (params->metric == DISTANCE_COSINE)
	{
		normalize_vec(bufs->norm_buf, vec, dim);
		enc_vec = bufs->norm_buf;
	}

	uint32_t secondary = MKT_INVALID_CLUSTER;

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
		uint32_t ncand = mkt_hkmeans_assign_topk(
				tree,
				enc_vec,
				params->metric,
				MKT_SECONDARY_TOPK,
				MKT_SECONDARY_BEAM_WIDTH,
				bufs->cand_leaves,
				bufs->cand_dists);

		uint32_t boundary_c2 = best_c;
		if (has_boundary)
			boundary_c2 = mkt_find_secondary_cluster(
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
				secondary = mkt_find_soar_secondary(
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
				secondary = MKT_INVALID_CLUSTER;
		}
	}

	return (MktBuildAssignment){
			.primary	= best_c,
			.secondary	= secondary,
			.enc_vector = enc_vec,
	};
}

/* ----------------------------------------------------------------
 * Compact posting entry (cluster-sorted build path) — see posting_build.h.
 * ---------------------------------------------------------------- */

Size
mkt_posting_entry_size(Dimension dim)
{
	return sizeof(ItemPointerData) + 3 * sizeof(float) + (Size)((dim + 7) / 8);
}

void
mkt_posting_entry_encode(
		const RaBitQParams *params,
		const float		   *vec,
		const float		   *centroid,
		Dimension			dim,
		RaBitQData		   *enc_buf,
		RaBitQScratch	   *scratch,
		ItemPointerData		tid,
		void			   *out_entry)
{
	VectorRef vref = {.data = vec, .dim = dim};
	VectorRef cref = {.data = centroid, .dim = dim};
	mkt_rabitq_encode_into_ex(params, vref, cref, enc_buf, scratch);
	float f_error =
			mkt_rabitq_derive_f_error(enc_buf->f_add, enc_buf->f_rescale, dim);

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
mkt_posting_entry_encode_from_pt(
		const RaBitQParams *params,
		const float		   *pt_residual,
		Dimension			dim,
		RaBitQData		   *enc_buf,
		RaBitQScratch	   *scratch,
		ItemPointerData		tid,
		void			   *out_entry)
{
	/* Page-backed twin of mkt_posting_entry_encode: the caller already has the
	 * rotated residual (pt_query - pt_centroid), e.g. from routing + the head
	 * page's pt_centroid, so no float centroid is needed. */
	mkt_rabitq_encode_from_pt(params, pt_residual, enc_buf, scratch);
	float f_error =
			mkt_rabitq_derive_f_error(enc_buf->f_add, enc_buf->f_rescale, dim);

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
mkt_build_route_ctx_init(
		MktBuildRouteCtx   *ctx,
		MktQueryState	   *qs,
		MktSorter		   *sorter,
		const RaBitQParams *rq_params,
		MktStorage		   *storage,
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

	ctx->cand_pt = mkt_alloc((size_t)MKT_SECONDARY_TOPK * dim * sizeof(float));
	ctx->cand_leaf = mkt_alloc(MKT_SECONDARY_TOPK * sizeof(uint32_t));
	ctx->cand_dist = mkt_alloc(MKT_SECONDARY_TOPK * sizeof(Distance));
	ctx->pt_r	   = mkt_alloc((size_t)dim * sizeof(float));
	ctx->enc_buf   = mkt_alloc(MKT_RABITQ_DATA_SIZE(dim));
	mkt_rabitq_scratch_init(&ctx->enc_scratch, dim);
	ctx->entry = mkt_alloc(mkt_posting_entry_size(dim));

	ctx->indtuples	= 0;
	ctx->soar_dupes = 0;
}

void
mkt_build_route_ctx_cleanup(MktBuildRouteCtx *ctx)
{
	mkt_free(ctx->cand_pt);
	mkt_free(ctx->cand_leaf);
	mkt_free(ctx->cand_dist);
	mkt_free(ctx->pt_r);
	mkt_free(ctx->enc_buf);
	mkt_rabitq_scratch_cleanup(&ctx->enc_scratch);
	mkt_free(ctx->entry);
}

bool
mkt_build_route_emit(
		MktBuildRouteCtx *ctx, const float *vec, ItemPointerData tid)
{
	Dimension dim = ctx->dim;

	/*
	 * Route page-backed, exactly as the query/insert do: descend the centroid
	 * pages, giving the nearest leaves' posting-head blocks + qs->pt_query
	 * (the rotated vector). No in-RAM tree.
	 */
	uint32_t n = mkt_query_route(
			ctx->qs,
			vec,
			MKT_SECONDARY_TOPK,
			MKT_DISTANCE_MODE_ASYMMETRIC,
			NULL);
	if (n == 0)
		return false;

	/* Gather the beam candidates: leaf index, distance, and pt_centroid (read
	 * from each head page -- the float encode reference). */
	for (uint32_t i = 0; i < n; i++)
	{
		BlockNumber h	  = ctx->qs->beam_results[i].posting_head;
		ctx->cand_leaf[i] = mkt_route_head_to_leaf(ctx->first_posting, h);
		ctx->cand_dist[i] = ctx->qs->beam_results[i].distance;
		Page hp			  = mkt_storage_read_page(ctx->storage, h);
		memcpy(ctx->cand_pt + (size_t)i * dim,
			   mkt_posting_pt_centroid(hp),
			   (size_t)dim * sizeof(float));
		mkt_storage_release_page(ctx->storage, h);
	}

	/* Primary: encode pt_query - pt_centroid[0] and stream to the sorter. */
	uint32_t primary = ctx->cand_leaf[0];
	for (Dimension d = 0; d < dim; d++)
		ctx->pt_r[d] = ctx->qs->pt_query[d] - ctx->cand_pt[d];
	mkt_posting_entry_encode_from_pt(
			ctx->rq_params,
			ctx->pt_r,
			dim,
			ctx->enc_buf,
			&ctx->enc_scratch,
			tid,
			ctx->entry);
	mkt_pbuild_sort_put(ctx->sorter, primary, ctx->entry);
	ctx->indtuples++;

	/* Secondary (SOAR / boundary), in rotated space over the gathered
	 * candidates (positions index cand_pt). OA scoring and distances are
	 * norm-preserving under P^T, so this matches the float-space result. */
	bool has_soar	  = ctx->soar_lambda > 0.0;
	bool has_boundary = ctx->boundary_epsilon > 0.0;
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
		sec_pos = mkt_find_soar_secondary(
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
		mkt_posting_entry_encode_from_pt(
				ctx->rq_params,
				ctx->pt_r,
				dim,
				ctx->enc_buf,
				&ctx->enc_scratch,
				tid,
				ctx->entry);
		mkt_pbuild_sort_put(ctx->sorter, ctx->cand_leaf[sec_pos], ctx->entry);
		ctx->soar_dupes++;
		return true;
	}
	return false;
}

void
mkt_posting_entry_add(
		MktPostingBuilder *builder, const void *entry, Dimension dim)
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
	mkt_posting_builder_add_encoded(
			builder, tid, f_add, f_rescale, f_error, (const uint8_t *)p);
}

uint32_t
mkt_build_assign_primary(
		const HKMeansResult	 *tree,
		const float			 *vec,
		const MktBuildParams *params,
		float				 *enc_out,
		Distance			 *out_dist)
{
	uint32_t best_c = mkt_hkmeans_assign(tree, vec, params->metric, out_dist);

	if (params->metric == DISTANCE_COSINE)
		normalize_vec(enc_out, vec, params->dim);
	else
		memcpy(enc_out, vec, (size_t)params->dim * sizeof(float));

	return best_c;
}

/* ----------------------------------------------------------------
 * Shared helpers
 * ---------------------------------------------------------------- */

/*
 * Flush the in-memory page.
 *
 * Deferred mode (storage == NULL): streams the full page to the page sink
 * (the leader writes it). No block numbers, no chain linking.
 *
 * Direct mode (storage != NULL): writes to storage and links
 * into the chain. Prefers reserved contiguous blocks.
 */
static void
flush_page(MktPostingBuilder *builder)
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

	if (builder->storage == NULL)
	{
		/*
		 * Deferred mode: stream the full page to the sink — the leader writes
		 * it — for bounded memory.
		 */
		builder->page_sink(
				builder->sink_ctx, builder->cluster_id, builder->mem_page);
		builder->is_first = false;
		builder->page_ops->reinit_page(builder);
		builder->page_dirty = false;
		return;
	}

	BlockNumber blkno;
	Page		spage;
	if (builder->is_first && builder->fixed_first_blkno != InvalidBlockNumber)
	{
		blkno = builder->fixed_first_blkno;
		spage = mkt_storage_write_page(builder->storage, blkno);
	}
	else
	{
		spage = mkt_storage_new_page(builder->storage, &blkno);
	}

	memcpy(spage, builder->mem_page, BLCKSZ);
	mkt_storage_commit_page(builder->storage, blkno);

	if (builder->is_first)
	{
		builder->head_blkno = blkno;
		builder->is_first	= false;
	}

	if (builder->prev_blkno != InvalidBlockNumber)
	{
		Page prev =
				mkt_storage_write_page(builder->storage, builder->prev_blkno);
		mkt_posting_opaque(prev)->next_blkno = blkno;
		mkt_storage_commit_page(builder->storage, builder->prev_blkno);
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
		MktPostingBuilder		*builder,
		MktStorage				*storage,
		const RaBitQParams		*params,
		Dimension				 dim,
		uint32_t				 cluster_id,
		const float				*centroid,
		const float				*pt_centroid,
		const MktPostingPageOps *ops,
		uint16_t				 first_page_flags)
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
	builder->owns_head	= (first_page_flags & MKT_POSTING_PAGE_FIRST) != 0;
	builder->n_entries	= 0;
	builder->page_ops	= ops;

	builder->fixed_first_blkno	 = InvalidBlockNumber;

	mkt_posting_page_init(
			builder->mem_page, cluster_id, dim, first_page_flags);
	if (pt_centroid != NULL)
		memcpy(mkt_posting_pt_centroid_mut(builder->mem_page),
			   pt_centroid,
			   dim * sizeof(float));

	builder->enc_buf = params ? mkt_alloc(MKT_RABITQ_DATA_SIZE(dim)) : NULL;
	if (params)
		mkt_rabitq_scratch_init(&builder->enc_scratch, dim);
}

/* ----------------------------------------------------------------
 * AoS page ops — one entry per call
 * ---------------------------------------------------------------- */

static bool
aos_write_entry(
		MktPostingBuilder *builder,
		ItemPointerData	   tid,
		float			   f_add,
		float			   f_rescale,
		float			   f_error,
		const uint8_t	  *bits)
{
	if (!mkt_posting_page_has_room(builder->mem_page))
		return false;

	mkt_posting_page_add(
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
aos_reinit_page(MktPostingBuilder *builder)
{
	mkt_posting_page_init(
			builder->mem_page,
			builder->cluster_id,
			builder->dim,
			MKT_POSTING_PAGE_OVERFLOW);
}

static void
aos_finalize(MktPostingBuilder *builder)
{
	(void)builder;
}

static void
aos_cleanup(MktPostingBuilder *builder)
{
	(void)builder;
}

static const MktPostingPageOps aos_page_ops = {
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
fs_write_group(MktPostingBuilder *builder)
{
	FsGroupStage *grp = &builder->fs.grp;
	if (grp->count == 0)
		return;

	Dimension dim = builder->dim;

	MktPostingPageOpaque *op	  = mkt_posting_opaque(builder->mem_page);
	char				 *content = (op->flags & MKT_POSTING_PAGE_FIRST)
										  ? mkt_posting_content_first(builder->mem_page, dim)
										  : mkt_posting_content(builder->mem_page);

	uint32_t g = builder->fs.groups_on_page;

	memcpy(mkt_fastscan_group_tids(content, g, dim),
		   grp->tids,
		   MKT_FASTSCAN_GROUP * sizeof(ItemPointerData));
	memcpy(mkt_fastscan_group_f_add(content, g, dim),
		   grp->f_add,
		   MKT_FASTSCAN_GROUP * sizeof(float));
	memcpy(mkt_fastscan_group_f_rescale(content, g, dim),
		   grp->f_rescale,
		   MKT_FASTSCAN_GROUP * sizeof(float));
	memcpy(mkt_fastscan_group_f_error(content, g, dim),
		   grp->f_error,
		   MKT_FASTSCAN_GROUP * sizeof(float));

	mkt_fastscan_pack_codes(
			builder->fs.bits_buf, grp->count, dim, builder->fs.codes_buf);
	memcpy(mkt_fastscan_group_codes(content, g, dim),
		   builder->fs.codes_buf,
		   MKT_FASTSCAN_GROUP_BYTES(dim));

	op->entry_count += (uint16_t)grp->count;
	builder->fs.groups_on_page++;
	builder->page_dirty = true;

	memset(grp, 0, sizeof(*grp));
	memset(builder->fs.bits_buf,
		   0,
		   (size_t)MKT_FASTSCAN_GROUP * MKT_RABITQ_BYTES(dim));
}

static bool
fs_write_entry(
		MktPostingBuilder *builder,
		ItemPointerData	   tid,
		float			   f_add,
		float			   f_rescale,
		float			   f_error,
		const uint8_t	  *bits)
{
	FsGroupStage *grp		   = &builder->fs.grp;
	uint32_t	  idx		   = grp->count;
	uint32_t	  packed_bytes = MKT_RABITQ_BYTES(builder->dim);

	grp->tids[idx]		= tid;
	grp->f_add[idx]		= f_add;
	grp->f_rescale[idx] = f_rescale;
	grp->f_error[idx]	= f_error;
	memcpy(builder->fs.bits_buf + (size_t)idx * packed_bytes,
		   bits,
		   packed_bytes);
	grp->count++;

	if (grp->count == MKT_FASTSCAN_GROUP)
	{
		if (builder->fs.groups_on_page >= builder->fs.max_groups)
			flush_page(builder);
		fs_write_group(builder);
	}

	return true;
}

static void
fs_reinit_page(MktPostingBuilder *builder)
{
	mkt_posting_page_init(
			builder->mem_page,
			builder->cluster_id,
			builder->dim,
			MKT_POSTING_PAGE_OVERFLOW | MKT_POSTING_PAGE_FASTSCAN);
	builder->fs.groups_on_page = 0;
	builder->fs.max_groups	   = mkt_fastscan_max_groups(builder->dim, false);
}

static void
fs_finalize(MktPostingBuilder *builder)
{
	if (builder->fs.groups_on_page >= builder->fs.max_groups)
		flush_page(builder);
	fs_write_group(builder);
}

static void
fs_cleanup(MktPostingBuilder *builder)
{
	mkt_free(builder->fs.bits_buf);
	builder->fs.bits_buf = NULL;
	mkt_free(builder->fs.codes_buf);
	builder->fs.codes_buf = NULL;
}

static const MktPostingPageOps fs_page_ops = {
		.write_entry = fs_write_entry,
		.reinit_page = fs_reinit_page,
		.finalize	 = fs_finalize,
		.cleanup	 = fs_cleanup,
};

/* ----------------------------------------------------------------
 * Public API — init
 * ---------------------------------------------------------------- */

void
mkt_posting_builder_init(
		MktPostingBuilder  *builder,
		MktStorage		   *storage,
		const RaBitQParams *params,
		Dimension			dim,
		uint32_t			cluster_id,
		const float		   *centroid,
		const float		   *pt_centroid)
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
			MKT_POSTING_PAGE_FIRST);
}

void
mkt_posting_builder_init_fastscan(
		MktPostingBuilder  *builder,
		MktStorage		   *storage,
		const RaBitQParams *params,
		Dimension			dim,
		uint32_t			cluster_id,
		const float		   *centroid,
		const float		   *pt_centroid)
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
			MKT_POSTING_PAGE_FIRST | MKT_POSTING_PAGE_FASTSCAN);

	uint32_t packed_bytes = MKT_RABITQ_BYTES(dim);
	builder->fs.bits_buf  = mkt_alloc(
			 (size_t)MKT_FASTSCAN_GROUP * packed_bytes);
	builder->fs.codes_buf  = mkt_alloc(MKT_FASTSCAN_GROUP_BYTES(dim));
	builder->fs.max_groups = mkt_fastscan_max_groups(dim, true);
}

void
mkt_posting_builder_init_fmt(
		MktPostingBuilder  *builder,
		MktStorage		   *storage,
		const RaBitQParams *params,
		Dimension			dim,
		uint32_t			cluster_id,
		const float		   *centroid,
		const float		   *pt_centroid,
		bool				fastscan)
{
	if (fastscan)
		mkt_posting_builder_init_fastscan(
				builder,
				storage,
				params,
				dim,
				cluster_id,
				centroid,
				pt_centroid);
	else
		mkt_posting_builder_init(
				builder,
				storage,
				params,
				dim,
				cluster_id,
				centroid,
				pt_centroid);
}

void
mkt_posting_builder_adopt_head(
		MktPostingBuilder  *builder,
		MktStorage		   *storage,
		const RaBitQParams *params,
		Dimension			dim,
		uint32_t			cluster_id,
		BlockNumber			head_blk,
		bool				fastscan)
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

	builder->fixed_first_blkno	 = head_blk;
	builder->adopted_head		 = true;

	/* Start from the pre-written head itself: it already carries the
	 * cluster's header and pt_centroid. It must be empty — the pre-scan
	 * writer never adds entries. */
	Page hp = mkt_storage_read_page(storage, head_blk);
	Assert(mkt_posting_opaque(hp)->live_count == 0);
	memcpy(builder->mem_page, hp, BLCKSZ);
	mkt_storage_release_page(storage, head_blk);

	builder->enc_buf = params ? mkt_alloc(MKT_RABITQ_DATA_SIZE(dim)) : NULL;
	if (params)
		mkt_rabitq_scratch_init(&builder->enc_scratch, dim);

	if (fastscan)
	{
		uint32_t packed_bytes = MKT_RABITQ_BYTES(dim);
		builder->fs.bits_buf  = mkt_alloc(
				 (size_t)MKT_FASTSCAN_GROUP * packed_bytes);
		builder->fs.codes_buf  = mkt_alloc(MKT_FASTSCAN_GROUP_BYTES(dim));
		builder->fs.max_groups = mkt_fastscan_max_groups(dim, true);
	}
}

void
mkt_posting_builder_init_continuation(
		MktPostingBuilder  *builder,
		MktStorage		   *storage,
		const RaBitQParams *params,
		Dimension			dim,
		uint32_t			cluster_id,
		const float		   *centroid)
{
	builder_init_common(
			builder,
			storage,
			params,
			dim,
			cluster_id,
			centroid,
			NULL,
			&aos_page_ops,
			MKT_POSTING_PAGE_OVERFLOW);
}

void
mkt_posting_builder_init_continuation_fastscan(
		MktPostingBuilder  *builder,
		MktStorage		   *storage,
		const RaBitQParams *params,
		Dimension			dim,
		uint32_t			cluster_id,
		const float		   *centroid)
{
	builder_init_common(
			builder,
			storage,
			params,
			dim,
			cluster_id,
			centroid,
			NULL,
			&fs_page_ops,
			MKT_POSTING_PAGE_OVERFLOW | MKT_POSTING_PAGE_FASTSCAN);

	uint32_t packed_bytes = MKT_RABITQ_BYTES(dim);
	builder->fs.bits_buf  = mkt_alloc(
			 (size_t)MKT_FASTSCAN_GROUP * packed_bytes);
	builder->fs.codes_buf  = mkt_alloc(MKT_FASTSCAN_GROUP_BYTES(dim));
	builder->fs.max_groups = mkt_fastscan_max_groups(dim, false);
}


void
mkt_posting_builder_set_first_blkno(
		MktPostingBuilder *builder, BlockNumber blkno)
{
	builder->fixed_first_blkno = blkno;
}

void
mkt_posting_builder_add_encoded(
		MktPostingBuilder *builder,
		ItemPointerData	   tid,
		float			   f_add,
		float			   f_rescale,
		float			   f_error,
		const uint8_t	  *bits)
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
mkt_posting_builder_add(
		MktPostingBuilder *builder, ItemPointerData tid, const float *vector)
{
	VectorRef vref = {.data = vector, .dim = builder->dim};
	VectorRef cref = {.data = builder->centroid, .dim = builder->dim};
	mkt_rabitq_encode_into_ex(
			builder->params,
			vref,
			cref,
			builder->enc_buf,
			&builder->enc_scratch);

	float f_error = mkt_rabitq_derive_f_error(
			builder->enc_buf->f_add,
			builder->enc_buf->f_rescale,
			builder->dim);

	mkt_posting_builder_add_encoded(
			builder,
			tid,
			builder->enc_buf->f_add,
			builder->enc_buf->f_rescale,
			f_error,
			builder->enc_buf->bits);
}

BlockNumber
mkt_posting_builder_finish(MktPostingBuilder *builder)
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
	 * is the exact entry count. Skipped in deferred mode (storage == NULL),
	 * where the parallel leader writes and stamps the head itself; for a head
	 * builder that holds the whole chain (serial build, and the leader's head
	 * builder when no continuations were streamed) these values are final.
	 */
	if (builder->owns_head && builder->storage != NULL &&
		builder->head_blkno != InvalidBlockNumber)
	{
		Page hp =
				mkt_storage_write_page(builder->storage, builder->head_blkno);
		MktPostingPageOpaque *op = mkt_posting_opaque(hp);
		op->tail_blkno			 = builder->prev_blkno;
		op->live_count			 = builder->n_entries;
		mkt_storage_commit_page(builder->storage, builder->head_blkno);
	}

	return builder->head_blkno;
}

BlockNumber
mkt_posting_builder_finish_partial(MktPostingBuilder *builder)
{
	builder->page_ops->finalize(builder);
	return builder->head_blkno;
}

void
mkt_posting_builder_set_page_sink(
		MktPostingBuilder *builder,
		void (*sink)(void *ctx, uint32_t cluster_id, const char *page),
		void *sink_ctx)
{
	builder->page_sink = sink;
	builder->sink_ctx  = sink_ctx;
}

void
mkt_posting_builder_cleanup(MktPostingBuilder *builder)
{
	builder->page_ops->cleanup(builder);
	if (builder->enc_buf != NULL)
	{
		mkt_rabitq_scratch_cleanup(&builder->enc_scratch);
		mkt_free(builder->enc_buf);
		builder->enc_buf = NULL;
	}
}

/* ----------------------------------------------------------------
 * Flat builder — one buffer per cluster (unchanged)
 * ---------------------------------------------------------------- */

void
mkt_flat_posting_builder_init(
		MktFlatPostingBuilder *builder,
		const RaBitQParams	  *params,
		Dimension			   dim,
		uint32_t			   cluster_id,
		const float			  *centroid,
		uint32_t			   count)
{
	builder->params		 = params;
	builder->dim		 = dim;
	builder->cluster_id	 = cluster_id;
	builder->centroid	 = centroid;
	builder->max_entries = count;

	size_t buf_size = mkt_posting_flat_page_size(dim, count);
	builder->buf	= mkt_alloc(buf_size);
	memset(builder->buf, 0, buf_size);
	mkt_posting_flat_init(builder->buf, count, cluster_id);

	builder->enc_buf = mkt_alloc(MKT_RABITQ_DATA_SIZE(dim));
}

void
mkt_flat_posting_builder_add(
		MktFlatPostingBuilder *builder,
		ItemPointerData		   tid,
		const float			  *vector)
{
	Dimension dim = builder->dim;

	VectorRef vref = {.data = vector, .dim = dim};
	VectorRef cref = {.data = builder->centroid, .dim = dim};
	mkt_rabitq_encode_into(builder->params, vref, cref, builder->enc_buf);

	float f_error = mkt_rabitq_derive_f_error(
			builder->enc_buf->f_add, builder->enc_buf->f_rescale, dim);

	mkt_posting_flat_add(
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
mkt_flat_posting_builder_finish(MktFlatPostingBuilder *builder)
{
	return builder->buf;
}

void
mkt_flat_posting_builder_cleanup(MktFlatPostingBuilder *builder)
{
	if (builder->enc_buf != NULL)
	{
		mkt_free(builder->enc_buf);
		builder->enc_buf = NULL;
	}
}

/* See posting_build.h: the route/filter/normalize half of the refine pass,
 * shared verbatim by the serial and parallel builds. */
const float *
mkt_refine_route_row(
		struct MktQueryState *qs,
		BlockNumber			  first_posting,
		const float			 *vec,
		Dimension			  dim,
		bool				  cosine,
		float				 *scratch,
		uint32_t			  tile_lo,
		uint32_t			  tile_hi,
		uint32_t			 *out_idx)
{
	uint32_t n = mkt_query_route(qs, vec, 1, MKT_DISTANCE_MODE_ASYMMETRIC,
								 NULL);
	if (n == 0)
		return NULL;

	uint32_t leaf =
			mkt_route_head_to_leaf(first_posting, qs->beam_results[0].posting_head);
	if (leaf < tile_lo || leaf >= tile_hi)
		return NULL;
	*out_idx = leaf - tile_lo;

	/* Cosine centroids are trained in normalized space, so the mean must
	 * average the normalized vectors. */
	if (cosine)
	{
		memcpy(scratch, vec, (size_t)dim * sizeof(float));
		mkt_l2_normalize(scratch, dim);
		return scratch;
	}
	return vec;
}

void
mkt_refine_write_means(
		const double  *sums,
		const uint64_t *counts,
		uint32_t		lo,
		uint32_t		hi,
		Dimension		dim,
		float		   *scratch,
		MktLeafWriteFn	write_head,
		void		   *write_head_ctx)
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
mkt_write_leaf_head(void *arg, uint32_t leaf, const float *centroid)
{
	MktHeadWriteCtx *h = (MktHeadWriteCtx *)arg;
	mkt_rabitq_rotate(h->rq_params, centroid, h->pt);

	MktPostingBuilder hb;
	mkt_posting_builder_init_fmt(
			&hb,
			h->storage,
			h->rq_params,
			h->dim,
			leaf,
			centroid,
			h->pt,
			h->fastscan);
	mkt_posting_builder_set_first_blkno(&hb, h->first_posting + leaf);
	mkt_posting_builder_finish(&hb);
	mkt_posting_builder_cleanup(&hb);
}
