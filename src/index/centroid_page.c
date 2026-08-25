/*
 * centroid_page.c - Centroid tree page operations
 *
 * Implements page initialization and entry insertion for centroid pages
 * using bidirectional growth: metadata forward, vector data backward.
 * The data format (RaBitQ, float32, float16) is stored in the page
 * opaque flags and determines per-entry data size.
 */

#include <math.h>

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

void *
mkt_centroid_page_add_entry_begin(
		Page		page,
		Dimension	dim,
		BlockNumber child_blkno,
		uint16_t	child_count,
		uint16_t	flags)
{
	if (!mkt_centroid_page_has_room(page, dim, false))
		return NULL;

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

	/* Reserve data space (backward region) */
	uint32_t data_size = mkt_centroid_data_size(dim, fmt);
	header->pd_upper -= data_size;

	header->pd_lower += meta_size;
	opaque->entry_count = index + 1;

	return page + header->pd_upper;
}

bool
mkt_centroid_page_add_entry(
		Page		page,
		Dimension	dim,
		BlockNumber child_blkno,
		uint16_t	child_count,
		uint16_t	flags,
		const void *data)
{
	MktCentroidFormat fmt		= mkt_centroid_page_format(page);
	uint32_t		  data_size = mkt_centroid_data_size(dim, fmt);

	void *dest = mkt_centroid_page_add_entry_begin(
			page, dim, child_blkno, child_count, flags);
	if (dest == NULL)
		return false;

	memcpy(dest, data, data_size);
	return true;
}

void
mkt_centroid_page_set_child(Page page, uint32_t index, BlockNumber child_blkno)
{
	/* Repoint a leaf at a new posting head, leaving its routing centroid and
	 * everything else intact (used when a list is rewritten in place, e.g. by
	 * reassignment, keeping the same centroid). */
	mkt_centroid_meta_mut(page, index)->child_blkno = child_blkno;
}

void
mkt_centroid_page_poison_entry(Page page, Dimension dim, uint32_t index)
{
	/*
	 * Make a leaf entry unreachable to routing without removing it: set its
	 * RaBitQ f_add to +inf so scoring always estimates +inf distance and it is
	 * never selected. Used when a list is dissolved (merge) — the entry stays
	 * in place (a later centroid compaction reclaims the slot) but no query is
	 * routed to the now-empty list. RaBitQ centroid pages only.
	 */
	MktCentroidFormat fmt = mkt_centroid_page_format(page);
	if (fmt != MKT_CENTROID_FMT_RABITQ)
		return;
	uint32_t	data_size = mkt_centroid_data_size(dim, fmt);
	RaBitQData *d = (RaBitQData *)((char *)PageGetSpecialPointer(page) -
								   (size_t)(index + 1) * data_size);
	d->f_add	  = INFINITY;
}

void
mkt_centroid_page_overwrite_entry(
		Page		page,
		Dimension	dim,
		uint32_t	index,
		BlockNumber child_blkno,
		const void *data)
{
	MktCentroidFormat fmt		= mkt_centroid_page_format(page);
	uint32_t		  data_size = mkt_centroid_data_size(dim, fmt);

	/*
	 * In-place replacement of an existing entry: rewrite the fixed-size
	 * metadata slot (forward region) and the fixed-size data slot (backward
	 * region) without touching pd_lower/pd_upper or entry_count, so the page
	 * layout is unchanged and no relayout is needed. child_count/flags are
	 * preserved (a leaf entry stays a leaf). Used by the incremental split to
	 * repoint a leaf at its first child list and update its routing centroid.
	 */
	MktCentroidEntryMeta *meta = mkt_centroid_meta_mut(page, index);
	meta->child_blkno		   = child_blkno;

	void *dest = (char *)PageGetSpecialPointer(page) -
				 (size_t)(index + 1) * data_size;
	memcpy(dest, data, data_size);
}
