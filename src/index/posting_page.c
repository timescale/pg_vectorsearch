/*
 * posting_page.c - Posting list page operations
 */

#include "index/posting_page.h"

/* ----------------------------------------------------------------
 * Paged mode (BLCKSZ pages with PG header + opaque)
 * ---------------------------------------------------------------- */

void
mkt_posting_page_init(
		Page page, uint32_t cluster_id, Dimension dim, uint16_t flags)
{
	PageInit(page, BLCKSZ, sizeof(MktPostingPageOpaque));

	/* First pages have reduced capacity due to pt_centroid */
	uint32_t			  max	 = (flags & MKT_POSTING_PAGE_FIRST)
										 ? mkt_posting_max_entries_first(dim)
										 : mkt_posting_max_entries(dim);
	MktPostingPageOpaque *opaque = mkt_posting_opaque(page);
	opaque->next_blkno			 = InvalidBlockNumber;
	opaque->cluster_id			 = cluster_id;
	opaque->entry_count			 = 0;
	opaque->flags				 = flags;
	opaque->page_id				 = MKT_POSTING_PAGE_ID;
	opaque->max_entries			 = (uint16_t)max;
}

bool
mkt_posting_page_add(
		Page			page,
		Dimension		dim,
		ItemPointerData tid,
		float			f_add,
		float			f_rescale,
		float			f_error,
		const uint8_t  *bits,
		uint8_t			entry_flags)
{
	if (!mkt_posting_page_has_room(page))
		return false;

	MktPostingPageOpaque *opaque  = mkt_posting_opaque(page);
	uint32_t			  max	  = opaque->max_entries;
	uint32_t			  i		  = opaque->entry_count;
	char				 *content = (opaque->flags & MKT_POSTING_PAGE_FIRST)
										  ? mkt_posting_content_first(page, dim)
										  : mkt_posting_content(page);

	/* Write metadata */
	MktPostingEntryMeta *meta = &mkt_posting_metas_at(content)[i];
	meta->tid				  = tid;
	meta->flags				  = entry_flags;
	meta->reserved			  = 0;

	/* Write scalar arrays */
	mkt_posting_f_add_at(content, max)[i]	  = f_add;
	mkt_posting_f_rescale_at(content, max)[i] = f_rescale;
	mkt_posting_f_error_at(content, max)[i]	  = f_error;

	/* Write bits */
	uint32_t packed_bytes = MKT_RABITQ_BYTES(dim);
	memcpy(mkt_posting_entry_bits_at(content, max, dim, i),
		   bits,
		   packed_bytes);

	opaque->entry_count = i + 1;
	return true;
}

/* ----------------------------------------------------------------
 * Flat mode (one buffer per cluster, custom header)
 * ---------------------------------------------------------------- */

void
mkt_posting_flat_init(char *buf, uint32_t max_entries, uint32_t cluster_id)
{
	MktFlatPostingHeader *hdr = mkt_flat_posting_header(buf);
	hdr->max_entries		  = max_entries;
	hdr->entry_count		  = 0;
	hdr->cluster_id			  = cluster_id;
	hdr->_pad				  = 0;
}

bool
mkt_posting_flat_add(
		char		   *buf,
		Dimension		dim,
		ItemPointerData tid,
		float			f_add,
		float			f_rescale,
		float			f_error,
		const uint8_t  *bits,
		uint8_t			entry_flags)
{
	MktFlatPostingHeader *hdr = mkt_flat_posting_header(buf);
	if (hdr->entry_count >= hdr->max_entries)
		return false;

	uint32_t max	 = hdr->max_entries;
	uint32_t i		 = hdr->entry_count;
	char	*content = mkt_flat_posting_content(buf);

	/* Write metadata */
	MktPostingEntryMeta *meta = &mkt_posting_metas_at(content)[i];
	meta->tid				  = tid;
	meta->flags				  = entry_flags;
	meta->reserved			  = 0;

	/* Write scalar arrays */
	mkt_posting_f_add_at(content, max)[i]	  = f_add;
	mkt_posting_f_rescale_at(content, max)[i] = f_rescale;
	mkt_posting_f_error_at(content, max)[i]	  = f_error;

	/* Write bits */
	uint32_t packed_bytes = MKT_RABITQ_BYTES(dim);
	memcpy(mkt_posting_entry_bits_at(content, max, dim, i),
		   bits,
		   packed_bytes);

	hdr->entry_count = i + 1;
	return true;
}
