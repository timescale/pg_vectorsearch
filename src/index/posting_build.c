/*
 * posting_build.c - Streaming posting list builder
 *
 * Encodes and writes posting entries one at a time with O(1) memory.
 * A single RaBitQData buffer is reused for every entry, eliminating
 * per-cluster allocations proportional to cluster size.
 *
 * Each builder embeds an in-memory page buffer (BLCKSZ). When full,
 * the buffer is flushed to storage (new_page → memcpy → commit),
 * and the previous page is linked. We build into local memory
 * rather than directly into storage-allocated pages because in PG
 * mode each storage page holds a pinned buffer-cache lock. With
 * nlist builders active simultaneously, that would exceed the
 * per-backend pin limit. The memcpy at flush time is the cost of
 * needing only one buffer pin at a time.
 */

#include <assert.h>

#include "core/memory.h"
#include "index/posting_build.h"

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
		const float		   *centroid)
{
	builder->storage	= storage;
	builder->params		= params;
	builder->dim		= dim;
	builder->cluster_id = cluster_id;
	builder->centroid	= centroid;

	builder->head_blkno = InvalidBlockNumber;
	builder->prev_blkno = InvalidBlockNumber;
	builder->is_first	= true;

	mkt_posting_page_init(
			builder->mem_page, cluster_id, MKT_POSTING_PAGE_FIRST);
	builder->page_dirty = false;

	builder->reserve_start = InvalidBlockNumber;
	builder->reserve_count = 0;
	builder->reserve_used  = 0;

	builder->enc_buf = mkt_alloc(MKT_RABITQ_DATA_SIZE(dim));
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

void
mkt_posting_builder_set_reserve(
		MktPostingBuilder *builder, BlockNumber start, uint32_t count)
{
	builder->reserve_start = start;
	builder->reserve_count = count;
	builder->reserve_used  = 0;
}

/* ----------------------------------------------------------------
 * Flush in-memory page to storage and link previous page
 * ---------------------------------------------------------------- */
static void
builder_flush_page(MktPostingBuilder *builder)
{
	MktStorage *storage = builder->storage;

	/*
	 * Allocate a storage page and copy in-memory content.
	 * Use reserved contiguous blocks when available for sequential
	 * layout; fall back to new_page (which may interleave with
	 * other builders' pages).
	 */
	BlockNumber new_blkno;
	Page		pg_page;
	if (builder->reserve_used < builder->reserve_count)
	{
		new_blkno = builder->reserve_start + builder->reserve_used;
		builder->reserve_used++;
		pg_page = mkt_storage_write_page(storage, new_blkno);
	}
	else
	{
		pg_page = mkt_storage_new_page(storage, &new_blkno);
	}
	memcpy(pg_page, builder->mem_page, BLCKSZ);
	mkt_storage_commit_page(storage, new_blkno);

	/* Track head of chain */
	if (builder->is_first)
	{
		builder->head_blkno = new_blkno;
		builder->is_first	= false;
	}

	/* Link previous page to this one */
	if (builder->prev_blkno != InvalidBlockNumber)
	{
		Page prev = mkt_storage_write_page(storage, builder->prev_blkno);
		MktPostingPageOpaque *prev_opaque = MKT_POSTING_OPAQUE(prev);
		prev_opaque->next_blkno			  = new_blkno;
		mkt_storage_commit_page(storage, builder->prev_blkno);
	}

	builder->prev_blkno = new_blkno;

	/* Re-init in-memory buffer for next page */
	mkt_posting_page_init(
			builder->mem_page, builder->cluster_id, MKT_POSTING_PAGE_OVERFLOW);
	builder->page_dirty = false;
}

/* ----------------------------------------------------------------
 * Add one entry
 * ---------------------------------------------------------------- */
void
mkt_posting_builder_add(
		MktPostingBuilder *builder,
		VectorRef		   vec,
		BlockNumber		   block,
		OffsetNumber	   offset)
{
	/* Encode vector with IVF residual from cluster centroid */
	VectorRef cref = {.data = builder->centroid, .dim = builder->dim};
	mkt_rabitq_encode_into(builder->params, vec, cref, builder->enc_buf);

	/* Flush to storage if the in-memory page is full */
	if (!mkt_posting_page_has_room(builder->mem_page, builder->dim))
		builder_flush_page(builder);

	/* Write entry to in-memory page */
	mkt_posting_page_add(
			builder->mem_page,
			builder->dim,
			block,
			offset,
			builder->enc_buf,
			0);
	builder->page_dirty = true;
}

/* ----------------------------------------------------------------
 * Finish: flush last page, return head block number
 * ---------------------------------------------------------------- */
BlockNumber
mkt_posting_builder_finish(MktPostingBuilder *builder)
{
	if (builder->page_dirty)
		builder_flush_page(builder);

	return builder->head_blkno;
}
