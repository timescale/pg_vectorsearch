/*
 * centroid_build.c - Generic centroid page writer
 *
 * Writes centroid entries to linked pages via MktStorage. Format-
 * agnostic: page format determines metadata and data sizes.
 */

#include <assert.h>

#include "index/centroid_build.h"

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
