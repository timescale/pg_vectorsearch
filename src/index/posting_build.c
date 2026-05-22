/*
 * posting_build.c - Streaming posting list builder
 *
 * Encodes and writes posting entries one at a time with O(1) memory.
 * The page format (AoS or fastscan) is selected at init time via
 * page_ops callbacks. Everything else — encode, flush, chain link,
 * reserve — is shared.
 */

#include <math.h>
#include <string.h>

#include "core/memory.h"
#include "index/posting_build.h"
#include "quant/fastscan.h"

/* ----------------------------------------------------------------
 * Shared helpers
 * ---------------------------------------------------------------- */

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

/*
 * Flush the in-memory page to storage and link it into the chain.
 * Prefers reserved contiguous blocks for sequential layout.
 */
static void
flush_page(MktPostingBuilder *builder)
{
	if (!builder->page_dirty)
		return;

	BlockNumber blkno;
	Page		spage;
	if (builder->is_first && builder->fixed_first_blkno != InvalidBlockNumber)
	{
		blkno = builder->fixed_first_blkno;
		spage = mkt_storage_write_page(builder->storage, blkno);
	}
	else if (builder->shared_reserve_next != NULL)
	{
		uint32_t slot = atomic_fetch_add(builder->shared_reserve_next, 1);
		if (slot < builder->reserve_count)
		{
			blkno = builder->reserve_start + slot;
			spage = mkt_storage_write_page(builder->storage, blkno);
		}
		else
		{
			spage = mkt_storage_new_page(builder->storage, &blkno);
		}
	}
	else if (builder->reserve_used < builder->reserve_count)
	{
		blkno = builder->reserve_start + builder->reserve_used;
		builder->reserve_used++;
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
	builder->page_ops	= ops;

	builder->reserve_start		 = InvalidBlockNumber;
	builder->shared_reserve_next = NULL;
	builder->fixed_first_blkno	 = InvalidBlockNumber;

	mkt_posting_page_init(
			builder->mem_page, cluster_id, dim, first_page_flags);
	if (pt_centroid != NULL)
		memcpy(mkt_posting_pt_centroid_mut(builder->mem_page),
			   pt_centroid,
			   dim * sizeof(float));

	builder->enc_buf = params ? mkt_alloc(MKT_RABITQ_DATA_SIZE(dim)) : NULL;
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

/* ----------------------------------------------------------------
 * Public API — add / finish / cleanup / reserve
 * ---------------------------------------------------------------- */

void
mkt_posting_builder_set_reserve(
		MktPostingBuilder *builder, BlockNumber start, uint32_t count)
{
	builder->reserve_start = start;
	builder->reserve_count = count;
	builder->reserve_used  = 0;
}

void
mkt_posting_builder_set_shared_reserve(
		MktPostingBuilder *builder,
		BlockNumber		   start,
		uint32_t		   count,
		_Atomic(uint32_t) *next)
{
	builder->reserve_start		 = start;
	builder->reserve_count		 = count;
	builder->shared_reserve_next = next;
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
}

void
mkt_posting_builder_add(
		MktPostingBuilder *builder, ItemPointerData tid, const float *vector)
{
	VectorRef vref = {.data = vector, .dim = builder->dim};
	VectorRef cref = {.data = builder->centroid, .dim = builder->dim};
	mkt_rabitq_encode_into(builder->params, vref, cref, builder->enc_buf);

	float f_error = derive_f_error(
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
	flush_page(builder);
	return builder->head_blkno;
}

BlockNumber
mkt_posting_builder_finish_partial(MktPostingBuilder *builder)
{
	builder->page_ops->finalize(builder);
	return builder->head_blkno;
}

void
mkt_posting_builder_cleanup(MktPostingBuilder *builder)
{
	builder->page_ops->cleanup(builder);
	if (builder->enc_buf != NULL)
	{
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
