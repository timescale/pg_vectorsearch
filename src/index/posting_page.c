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
	uint32_t			  i		  = opaque->entry_count;
	char				 *content = (opaque->flags & MKT_POSTING_PAGE_FIRST)
										  ? mkt_posting_content_first(page, dim)
										  : mkt_posting_content(page);

	/* Write the entry header (meta + factors) + bits in one contiguous
	 * block at content + i * entry_size. */
	MktPostingEntryHeader *hdr = mkt_posting_entry_at(content, i, dim);
	hdr->meta.tid			   = tid;
	hdr->meta.flags			   = entry_flags;
	hdr->meta.reserved		   = 0;
	hdr->f_add				   = f_add;
	hdr->f_rescale			   = f_rescale;
	hdr->f_error			   = f_error;
	memcpy(hdr->bits, bits, MKT_RABITQ_BYTES(dim));

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
	MktFlatPostingHeader *flat_hdr = mkt_flat_posting_header(buf);
	if (flat_hdr->entry_count >= flat_hdr->max_entries)
		return false;

	uint32_t i		 = flat_hdr->entry_count;
	char	*content = mkt_flat_posting_content(buf);

	MktPostingEntryHeader *hdr = mkt_posting_entry_at(content, i, dim);
	hdr->meta.tid			   = tid;
	hdr->meta.flags			   = entry_flags;
	hdr->meta.reserved		   = 0;
	hdr->f_add				   = f_add;
	hdr->f_rescale			   = f_rescale;
	hdr->f_error			   = f_error;
	memcpy(hdr->bits, bits, MKT_RABITQ_BYTES(dim));

	flat_hdr->entry_count = i + 1;
	return true;
}
