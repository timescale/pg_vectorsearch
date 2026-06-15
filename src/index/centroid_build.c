/*
 * centroid_build.c - Generic centroid page writer
 *
 * Writes centroid entries to linked pages via MktStorage. Format-
 * agnostic: page format determines metadata and data sizes.
 */

#include <assert.h>
#include <math.h>
#include <string.h>

#include "core/memory.h"
#include "index/centroid_build.h"
#include "quant/fastscan.h"
#include "quant/rabitq.h"

BlockNumber
mkt_centroid_write_pages(
		MktStorage		  *storage,
		Dimension		   dim,
		uint32_t		   nlist,
		MktCentroidFormat  fmt,
		uint8_t			   level,
		uint16_t		   flags,
		uint16_t		   child_count,
		CentroidEncoder	  *encoder,
		const BlockNumber *child_blknos,
		const float		  *pt_centroids,
		BlockNumber		   start_blkno)
{
	bool		reserved	= (start_blkno != InvalidBlockNumber);
	BlockNumber first_blkno = reserved ? start_blkno : InvalidBlockNumber;
	BlockNumber prev_blkno	= InvalidBlockNumber;
	BlockNumber next_blkno	= start_blkno;
	Page		cur_page	= NULL;
	BlockNumber cur_blkno	= InvalidBlockNumber;

	(void)pt_centroids; /* pt_centroids stored on posting pages, not here */

	for (uint32_t i = 0; i < nlist; i++)
	{
		/* Allocate a new page if needed */
		if (cur_page == NULL ||
			!mkt_centroid_page_has_room(cur_page, dim, false))
		{
			/* Commit the previous page if any */
			if (cur_page != NULL)
				mkt_storage_commit_page(storage, cur_blkno);

			if (reserved)
			{
				cur_blkno = next_blkno++;
				cur_page  = mkt_storage_write_page(storage, cur_blkno);
			}
			else
			{
				cur_page = mkt_storage_new_page(storage, &cur_blkno);
			}
			mkt_centroid_page_init_fmt(cur_page, level, fmt);

			if (first_blkno == InvalidBlockNumber)
				first_blkno = cur_blkno;

			/* Link previous page to this one.  The storage layer
			 * holds at most one buffer, so we commit the new page
			 * first, reopen the previous page to set next_blkno,
			 * then reopen the new page. */
			if (prev_blkno != InvalidBlockNumber)
			{
				mkt_storage_commit_page(storage, cur_blkno);

				Page prev_page = mkt_storage_write_page(storage, prev_blkno);
				MKT_CENTROID_OPAQUE(prev_page)->next_blkno = cur_blkno;
				mkt_storage_commit_page(storage, prev_blkno);

				cur_page = mkt_storage_write_page(storage, cur_blkno);
			}

			prev_blkno = cur_blkno;
		}

		BlockNumber entry_child = child_blknos != NULL ? child_blknos[i]
													   : InvalidBlockNumber;

		void *dest = mkt_centroid_page_add_entry_begin(
				cur_page, dim, entry_child, child_count, flags);
		assert(dest != NULL);
		encoder->ops->encode_into(encoder, i, dest);
	}

	/* Commit last page */
	if (cur_page != NULL)
		mkt_storage_commit_page(storage, cur_blkno);

	return first_blkno;
}

/* ----------------------------------------------------------------
 * FASTSCAN centroid writer
 *
 * Walks the input centroid list in 32-vector groups. For each group:
 *   1. Encode each centroid with mkt_rabitq_encode_into to get
 *      f_add, f_rescale, and 1-bit packed code bytes.
 *   2. Derive f_error from (f_add, f_rescale).
 *   3. Repack the bits into the kPerm0 layout that
 *      mkt_fastscan_accumulate_hacc expects, and place the per-
 *      group f_add / f_rescale / f_error / child_blkno arrays in
 *      the group section.
 *
 * One page holds `groups_per_page` groups; once a page is full we
 * commit it and chain via next_blkno, the same way the RABITQ
 * writer above does. Partial trailing group is padded with
 * InvalidBlockNumber and zero codes.
 * ---------------------------------------------------------------- */

BlockNumber
mkt_centroid_write_fastscan_pages(
		MktStorage		   *storage,
		Dimension			dim,
		uint32_t			nlist,
		uint8_t				level,
		uint16_t			flags,
		const RaBitQParams *params,
		const float		   *vectors,
		const float		   *global_mean,
		const BlockNumber  *child_blknos,
		BlockNumber			start_blkno)
{
	(void)flags; /* per-entry flags not stored in FASTSCAN format */

	uint32_t packed_bytes	 = (dim + 7) / 8;
	uint32_t groups_per_page = mkt_centroid_fastscan_max_groups(dim);
	if (groups_per_page == 0)
		groups_per_page = 1; /* defensive — bigger dims may need split */

	uint32_t ngroups = (nlist + MKT_FASTSCAN_GROUP - 1) / MKT_FASTSCAN_GROUP;

	/* Scratch for one group's RaBitQ data (32 entries). */
	size_t	 rdata_size = MKT_RABITQ_DATA_SIZE(dim);
	uint8_t *rdata_buf	= mkt_alloc((size_t)MKT_FASTSCAN_GROUP * rdata_size);
	uint8_t *bits_buf	= mkt_alloc((size_t)MKT_FASTSCAN_GROUP * packed_bytes);

	VectorRef mref = {.data = global_mean, .dim = dim};

	bool		reserved	= (start_blkno != InvalidBlockNumber);
	BlockNumber first_blkno = reserved ? start_blkno : InvalidBlockNumber;
	BlockNumber prev_blkno	= InvalidBlockNumber;
	BlockNumber next_blkno	= start_blkno;
	Page		cur_page	= NULL;
	BlockNumber cur_blkno	= InvalidBlockNumber;
	uint32_t	cur_ngroups = 0;

	for (uint32_t g = 0; g < ngroups; g++)
	{
		/* Open a fresh page when needed */
		if (cur_page == NULL || cur_ngroups == groups_per_page)
		{
			if (cur_page != NULL)
				mkt_storage_commit_page(storage, cur_blkno);

			if (reserved)
			{
				cur_blkno = next_blkno++;
				cur_page  = mkt_storage_write_page(storage, cur_blkno);
			}
			else
				cur_page = mkt_storage_new_page(storage, &cur_blkno);

			mkt_centroid_page_init_fmt(
					cur_page, level, MKT_CENTROID_FMT_FASTSCAN);

			if (first_blkno == InvalidBlockNumber)
				first_blkno = cur_blkno;

			if (prev_blkno != InvalidBlockNumber)
			{
				mkt_storage_commit_page(storage, cur_blkno);
				Page prev = mkt_storage_write_page(storage, prev_blkno);
				MKT_CENTROID_OPAQUE(prev)->next_blkno = cur_blkno;
				mkt_storage_commit_page(storage, prev_blkno);
				cur_page = mkt_storage_write_page(storage, cur_blkno);
			}

			prev_blkno	= cur_blkno;
			cur_ngroups = 0;
		}

		/* Encode this group's 32 entries (or fewer for the last). */
		uint32_t g_start = g * MKT_FASTSCAN_GROUP;
		uint32_t g_count = nlist - g_start;
		if (g_count > MKT_FASTSCAN_GROUP)
			g_count = MKT_FASTSCAN_GROUP;

		/* Get per-entry RaBitQ encoding into rdata_buf, copy bits
		 * out into a flat packed_bytes-stride array for the packer. */
		for (uint32_t v = 0; v < g_count; v++)
		{
			VectorRef vref = {
					.data = vectors + (size_t)(g_start + v) * dim,
					.dim  = dim,
			};
			RaBitQData *d = (RaBitQData *)(rdata_buf + (size_t)v * rdata_size);
			mkt_rabitq_encode_into(params, vref, mref, d);
			memcpy(bits_buf + (size_t)v * packed_bytes, d->bits, packed_bytes);
		}

		/* Write per-entry scalars + child_blkno arrays into the
		 * group section, then pack codes. */
		uint32_t cur_group_idx = cur_ngroups;
		char	*content	   = (char *)PageGetContents(cur_page);

		BlockNumber *child =
				mkt_centroid_fastscan_group_child(content, cur_group_idx, dim);
		float *f_add_arr =
				mkt_centroid_fastscan_group_f_add(content, cur_group_idx, dim);
		float *f_rescale_arr = mkt_centroid_fastscan_group_f_rescale(
				content, cur_group_idx, dim);
		float *f_error_arr = mkt_centroid_fastscan_group_f_error(
				content, cur_group_idx, dim);
		uint8_t *codes =
				mkt_centroid_fastscan_group_codes(content, cur_group_idx, dim);

		for (uint32_t v = 0; v < MKT_FASTSCAN_GROUP; v++)
		{
			if (v < g_count)
			{
				const RaBitQData *d = (const RaBitQData *)(rdata_buf +
														   (size_t)v *
																   rdata_size);
				child[v]	 = child_blknos != NULL ? child_blknos[g_start + v]
													: InvalidBlockNumber;
				f_add_arr[v] = d->f_add;
				f_rescale_arr[v] = d->f_rescale;
				f_error_arr[v] =
						mkt_rabitq_derive_f_error(d->f_add, d->f_rescale, dim);
			}
			else
			{
				child[v]		 = InvalidBlockNumber;
				f_add_arr[v]	 = 0.0f;
				f_rescale_arr[v] = 0.0f;
				f_error_arr[v]	 = 0.0f;
			}
		}

		mkt_fastscan_pack_codes(bits_buf, g_count, dim, codes);

		MktCentroidPageOpaque *op = MKT_CENTROID_OPAQUE(cur_page);
		op->entry_count += (uint16_t)g_count;
		cur_ngroups++;

		/* Bump pd_lower so PostgreSQL's hole-compression preserves
		 * our writes. FASTSCAN doesn't use the backward data region
		 * — everything lives in the forward content area. */
		PageHeader header = (PageHeader)cur_page;
		header->pd_lower += mkt_centroid_fastscan_group_bytes(dim);
	}

	if (cur_page != NULL)
		mkt_storage_commit_page(storage, cur_blkno);

	mkt_free(rdata_buf);
	mkt_free(bits_buf);

	return first_blkno;
}
