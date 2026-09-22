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
prism_centroid_page_init_fmt(Page page, uint8_t level, PrismCentroidFormat fmt)
{
	PageInit(page, BLCKSZ, sizeof(PrismCentroidPageOpaque));

	/* Initialize the opaque area */
	PrismCentroidPageOpaque *opaque = PRISM_CENTROID_OPAQUE(page);
	opaque->next_blkno				= InvalidBlockNumber;
	opaque->entry_count				= 0;
	opaque->level					= level;
	opaque->flags					= fmt & PRISM_CENTROID_FMT_MASK;
	opaque->page_id					= PRISM_CENTROID_PAGE_ID;
}

void *
prism_centroid_page_add_entry_begin(
		Page		page,
		Dimension	dim,
		BlockNumber child_blkno,
		uint16_t	child_count,
		uint16_t	flags)
{
	if (!prism_centroid_page_has_room(page, dim, false))
		return NULL;

	PageHeader				 header = (PageHeader)page;
	PrismCentroidPageOpaque *opaque = PRISM_CENTROID_OPAQUE(page);
	PrismCentroidFormat		 fmt	= prism_centroid_page_format(page);
	uint16_t				 index	= opaque->entry_count;

	/* Write metadata (forward region) */
	PrismCentroidEntryMeta *meta = prism_centroid_meta_mut(page, index);
	meta->child_blkno			 = child_blkno;
	meta->child_count			 = child_count;
	meta->flags					 = flags;

	/* Reserve data space (backward region) */
	uint32_t data_size = prism_centroid_data_size(dim, fmt);
	header->pd_upper -= data_size;

	opaque->entry_count = index + 1;

	/*
	 * Recompute rather than advance: a page committed outside index build
	 * comes back with pd_lower covering the hole (see
	 * prism_centroid_meta_end), so incrementing it would leave the page
	 * looking permanently full.
	 */
	header->pd_lower = (LocationIndex)
			prism_centroid_meta_end(page, opaque->entry_count);

	return page + header->pd_upper;
}

bool
prism_centroid_page_add_entry(
		Page		page,
		Dimension	dim,
		BlockNumber child_blkno,
		uint16_t	child_count,
		uint16_t	flags,
		const void *data)
{
	PrismCentroidFormat fmt		  = prism_centroid_page_format(page);
	uint32_t			data_size = prism_centroid_data_size(dim, fmt);

	void *dest = prism_centroid_page_add_entry_begin(
			page, dim, child_blkno, child_count, flags);
	if (dest == NULL)
		return false;

	memcpy(dest, data, data_size);
	return true;
}

void
prism_centroid_page_overwrite_entry(
		Page		page,
		Dimension	dim,
		uint32_t	index,
		BlockNumber child_blkno,
		const void *data)
{
	PrismCentroidFormat fmt		  = prism_centroid_page_format(page);
	uint32_t			data_size = prism_centroid_data_size(dim, fmt);

	/*
	 * In-place replacement of an existing entry: rewrite the fixed-size
	 * metadata slot (forward region) and the fixed-size data slot (backward
	 * region) without touching pd_lower/pd_upper or entry_count, so the page
	 * layout is unchanged and no relayout is needed. child_count/flags are
	 * preserved (a leaf entry stays a leaf). Used by the incremental split to
	 * repoint a leaf at its first child list and update its routing centroid.
	 */
	PrismCentroidEntryMeta *meta = prism_centroid_meta_mut(page, index);
	meta->child_blkno			 = child_blkno;

	void *dest = (char *)PageGetSpecialPointer(page) -
				 (size_t)(index + 1) * data_size;
	memcpy(dest, data, data_size);
}
