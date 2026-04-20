/*
 * posting_build.c - Streaming posting list builder
 *
 * Encodes and writes posting entries one at a time with O(1) memory.
 * A single RaBitQData buffer is reused for every entry, eliminating
 * per-cluster allocations proportional to cluster size.
 *
 * Each builder embeds an in-memory page buffer (BLCKSZ). When full,
 * the buffer is flushed to storage (new_page -> memcpy -> commit),
 * and the previous page is linked. We build into local memory
 * rather than directly into storage-allocated pages because in PG
 * mode each storage page holds a pinned buffer-cache lock. With
 * nlist builders active simultaneously, that would exceed the
 * per-backend pin limit. The memcpy at flush time is the cost of
 * needing only one buffer pin at a time.
 */

#include <math.h>
#include <string.h>

#include "core/memory.h"
#include "index/posting_build.h"

/* Derive f_error from f_add and f_rescale */
static float
derive_f_error(float f_add, float f_rescale, Dimension dim)
{
	float f_rsq = f_rescale * f_rescale;
	if (f_rsq > f_add && dim > 1)
	{
		float c_err = 2.0f * MKT_RABITQ_EPSILON / sqrtf((float)(dim - 1));
		return c_err * sqrtf(f_rsq - f_add);
	}
	return 2e-4f * sqrtf(f_add);
}

/* ----------------------------------------------------------------
 * Flush in-memory page to storage and link previous page
 *
 * Uses reserved contiguous blocks when available for sequential
 * layout; falls back to new_page which may interleave with other
 * builders' pages.
 * ---------------------------------------------------------------- */

static void
flush_page(MktPostingBuilder *builder)
{
	if (!builder->page_dirty)
		return;

	MktStorage *storage = builder->storage;

	/* Allocate storage page — prefer reserved range */
	BlockNumber blkno;
	Page		spage;
	if (builder->reserve_used < builder->reserve_count)
	{
		blkno = builder->reserve_start + builder->reserve_used;
		builder->reserve_used++;
		spage = mkt_storage_write_page(storage, blkno);
	}
	else
	{
		spage = mkt_storage_new_page(storage, &blkno);
	}

	/* Copy in-memory page to storage */
	memcpy(spage, builder->mem_page, BLCKSZ);
	mkt_storage_commit_page(storage, blkno);

	/* Track head of chain */
	if (builder->is_first)
	{
		builder->head_blkno = blkno;
		builder->is_first	= false;
	}

	/* Link previous page to this one */
	if (builder->prev_blkno != InvalidBlockNumber)
	{
		Page prev = mkt_storage_write_page(storage, builder->prev_blkno);
		mkt_posting_opaque(prev)->next_blkno = blkno;
		mkt_storage_commit_page(storage, builder->prev_blkno);
	}
	builder->prev_blkno = blkno;

	/* Re-init in-memory buffer for next page */
	mkt_posting_page_init(
			builder->mem_page,
			builder->cluster_id,
			builder->dim,
			MKT_POSTING_PAGE_OVERFLOW);
	builder->page_dirty = false;
}

/* ----------------------------------------------------------------
 * Init / cleanup
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
	builder->storage	 = storage;
	builder->params		 = params;
	builder->dim		 = dim;
	builder->cluster_id	 = cluster_id;
	builder->centroid	 = centroid;
	builder->pt_centroid = pt_centroid;
	builder->head_blkno	 = InvalidBlockNumber;
	builder->prev_blkno	 = InvalidBlockNumber;
	builder->is_first	 = true;
	builder->page_dirty	 = false;

	builder->reserve_start = InvalidBlockNumber;
	builder->reserve_count = 0;
	builder->reserve_used  = 0;

	/* Initialize in-memory first page and write pt_centroid */
	mkt_posting_page_init(
			builder->mem_page, cluster_id, dim, MKT_POSTING_PAGE_FIRST);
	memcpy(mkt_posting_pt_centroid_mut(builder->mem_page),
		   pt_centroid,
		   dim * sizeof(float));

	/* Allocate reusable encode buffer */
	builder->enc_buf = mkt_alloc(MKT_RABITQ_DATA_SIZE(dim));
}

void
mkt_posting_builder_set_reserve(
		MktPostingBuilder *builder, BlockNumber start, uint32_t count)
{
	builder->reserve_start = start;
	builder->reserve_count = count;
	builder->reserve_used  = 0;
}

void
mkt_posting_builder_add(
		MktPostingBuilder *builder, ItemPointerData tid, const float *vector)
{
	Dimension dim = builder->dim;

	/* Encode with RaBitQ relative to cluster centroid */
	VectorRef vref = {.data = vector, .dim = dim};
	VectorRef cref = {.data = builder->centroid, .dim = dim};
	mkt_rabitq_encode_into(builder->params, vref, cref, builder->enc_buf);

	/* Derive f_error */
	float f_error = derive_f_error(
			builder->enc_buf->f_add, builder->enc_buf->f_rescale, dim);

	/* Flush current page if full */
	if (!mkt_posting_page_has_room(builder->mem_page))
		flush_page(builder);

	/* Add to in-memory page */
	mkt_posting_page_add(
			builder->mem_page,
			dim,
			tid,
			builder->enc_buf->f_add,
			builder->enc_buf->f_rescale,
			f_error,
			builder->enc_buf->bits,
			0);
	builder->page_dirty = true;
}

BlockNumber
mkt_posting_builder_finish(MktPostingBuilder *builder)
{
	flush_page(builder);
	return builder->head_blkno;
}

void
mkt_posting_builder_cleanup(MktPostingBuilder *builder)
{
	if (builder->enc_buf != NULL)
	{
		mkt_free(builder->enc_buf);
		builder->enc_buf = NULL;
	}
}

/* ----------------------------------------------------------------
 * Flat builder — one buffer per cluster
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

	/* Allocate and init flat page buffer */
	size_t buf_size = mkt_posting_flat_page_size(dim, count);
	builder->buf	= mkt_alloc(buf_size);
	memset(builder->buf, 0, buf_size);
	mkt_posting_flat_init(builder->buf, count, cluster_id);

	/* Reusable encode buffer */
	builder->enc_buf = mkt_alloc(MKT_RABITQ_DATA_SIZE(dim));
}

void
mkt_flat_posting_builder_add(
		MktFlatPostingBuilder *builder,
		ItemPointerData		   tid,
		const float			  *vector)
{
	Dimension dim = builder->dim;

	/* Encode with RaBitQ */
	VectorRef vref = {.data = vector, .dim = dim};
	VectorRef cref = {.data = builder->centroid, .dim = dim};
	mkt_rabitq_encode_into(builder->params, vref, cref, builder->enc_buf);

	float f_error = derive_f_error(
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
