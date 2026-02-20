/*
 * centroid_page.c - Centroid tree page operations
 *
 * Implements page initialization and entry insertion for centroid pages
 * using bidirectional growth: metadata forward, vector data backward.
 * The data format (RaBitQ, float32, float16) is stored in the page
 * opaque flags and determines per-entry data size.
 */

#include "index/centroid_page.h"

void
mkt_centroid_page_init_fmt(Page page, uint8_t level, MktCentroidFormat fmt)
{
	PageInit(page, BLCKSZ, sizeof(MktCentroidPageOpaque));

	/* Initialize the opaque area */
	MktCentroidPageOpaque *opaque = MKT_CENTROID_OPAQUE(page);
	opaque->next_blkno			  = InvalidBlockNumber;
	opaque->entry_count			  = 0;
	opaque->level				  = level;
	opaque->flags				  = fmt & MKT_CENTROID_FMT_MASK;
	opaque->page_id				  = MKT_CENTROID_PAGE_ID;
}

bool
mkt_centroid_page_add_entry(
		Page				   page,
		Dimension			   dim,
		BlockNumber			   child_blkno,
		uint16_t			   child_count,
		uint16_t			   flags,
		const ItemPointerData *medoid_tid,
		const void			  *data)
{
	if (!mkt_centroid_page_has_room(page, dim))
		return false;

	PageHeader			   header	 = (PageHeader)page;
	MktCentroidPageOpaque *opaque	 = MKT_CENTROID_OPAQUE(page);
	MktCentroidFormat	   fmt		 = mkt_centroid_page_format(page);
	uint16_t			   index	 = opaque->entry_count;
	uint32_t			   meta_size = mkt_centroid_meta_size(fmt);

	/* Write metadata (forward region) */
	MktCentroidEntryMeta *meta = mkt_centroid_meta_mut(page, index);
	meta->child_blkno		   = child_blkno;
	meta->child_count		   = child_count;
	meta->flags				   = flags;

	if (fmt == MKT_CENTROID_FMT_RABITQ)
	{
		MktCentroidEntryMetaRaBitQ *rmeta = (MktCentroidEntryMetaRaBitQ *)meta;
		if (medoid_tid != NULL)
			rmeta->medoid_tid = *medoid_tid;
		else
			memset(&rmeta->medoid_tid, 0, sizeof(rmeta->medoid_tid));
		rmeta->reserved = 0;
	}

	/* Write data (backward region) */
	uint32_t data_size = mkt_centroid_data_size(dim, fmt);
	header->pd_upper -= data_size;
	memcpy(page + header->pd_upper, data, data_size);

	header->pd_lower += meta_size;
	opaque->entry_count = index + 1;
	return true;
}
