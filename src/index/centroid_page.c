/*
 * centroid_page.c - Centroid tree page operations
 *
 * Implements page initialization and entry insertion for centroid pages
 * using bidirectional growth: metadata forward, RaBitQData backward.
 */

#include "index/centroid_page.h"

void
mkt_centroid_page_init(Page page, uint8_t level)
{
	PageInit(page, BLCKSZ, sizeof(MktCentroidPageOpaque));

	/* Initialize the opaque area */
	MktCentroidPageOpaque *opaque = MKT_CENTROID_OPAQUE(page);
	opaque->next_blkno			  = InvalidBlockNumber;
	opaque->entry_count			  = 0;
	opaque->level				  = level;
	opaque->flags				  = 0;
	opaque->page_id				  = MKT_CENTROID_PAGE_ID;
}

bool
mkt_centroid_page_add(
		Page				   page,
		Dimension			   dim,
		BlockNumber			   child_blkno,
		uint16_t			   child_count,
		uint16_t			   flags,
		const ItemPointerData *medoid_tid,
		const RaBitQData	  *data)
{
	if (!mkt_centroid_page_has_room(page, dim))
		return false;

	PageHeader			   header = (PageHeader)page;
	MktCentroidPageOpaque *opaque = MKT_CENTROID_OPAQUE(page);
	uint16_t			   index  = opaque->entry_count;

	/* Write metadata (forward region) */
	MktCentroidEntryMeta *meta = mkt_centroid_meta_mut(page, index);
	meta->child_blkno		   = child_blkno;
	meta->child_count		   = child_count;
	meta->flags				   = flags;
	meta->medoid_tid		   = *medoid_tid;
	meta->reserved			   = 0;

	/* Write RaBitQData (backward region) */
	uint32_t data_size = MKT_RABITQ_DATA_SIZE(dim);
	header->pd_upper -= data_size;
	memcpy(page + header->pd_upper, data, data_size);

	header->pd_lower += sizeof(MktCentroidEntryMeta);
	opaque->entry_count = index + 1;
	return true;
}
