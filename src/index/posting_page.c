/*
 * posting_page.c - Posting list page operations
 *
 * Implements page initialization and entry insertion for posting pages
 * using bidirectional growth: metadata forward, RaBitQData backward.
 */

#include <assert.h>

#include "index/posting_page.h"

void
mkt_posting_page_init(Page page, uint32_t cluster_id, uint16_t flags)
{
	PageInit(page, BLCKSZ, sizeof(MktPostingPageOpaque));

	MktPostingPageOpaque *opaque = MKT_POSTING_OPAQUE(page);
	opaque->next_blkno			 = InvalidBlockNumber;
	opaque->cluster_id			 = cluster_id;
	opaque->entry_count			 = 0;
	opaque->flags				 = flags;
	opaque->page_id				 = MKT_POSTING_PAGE_ID;
	opaque->reserved			 = 0;
}

bool
mkt_posting_page_add(
		Page			  page,
		Dimension		  dim,
		BlockNumber		  block,
		OffsetNumber	  offset,
		const RaBitQData *data,
		uint8_t			  flags)
{
	if (!mkt_posting_page_has_room(page, dim))
		return false;

	PageHeader			  header = (PageHeader)page;
	MktPostingPageOpaque *opaque = MKT_POSTING_OPAQUE(page);
	uint16_t			  index	 = opaque->entry_count;

	/* Write metadata (forward region) */
	MktPostingEntryMeta *meta = mkt_posting_meta_mut(page, index);
	meta->flags				  = flags;
	ItemPointerSet(&meta->tid, block, offset);
	meta->reserved = 0;

	/* Verify TID was written correctly */
	assert(ItemPointerGetBlockNumber(&meta->tid) == block);
	assert(ItemPointerGetOffsetNumber(&meta->tid) == offset);

	/* Write RaBitQData (backward region) */
	uint32_t data_size = MKT_RABITQ_DATA_SIZE(dim);
	header->pd_upper -= data_size;
	assert(header->pd_lower <= header->pd_upper);
	memcpy(page + header->pd_upper, data, data_size);

	/* Verify TID not clobbered by RaBitQ write */
	assert(ItemPointerGetBlockNumber(&meta->tid) == block);
	assert(ItemPointerGetOffsetNumber(&meta->tid) == offset);

	header->pd_lower += sizeof(MktPostingEntryMeta);
	opaque->entry_count = index + 1;
	return true;
}
