/*
 * centroid_page.c - Centroid tree page operations
 *
 * Implements page initialization and entry insertion for centroid pages.
 * The same code runs in standalone and PostgreSQL mode — only the I/O
 * layer (MktStorage) differs.
 */

#include "index/centroid_page.h"

void
mkt_centroid_page_init(Page page, uint8_t level, uint16_t first_global_idx)
{
	/* Zero the entire page */
	memset(page, 0, BLCKSZ);

	/* Initialize the opaque area */
	MktCentroidPageOpaque *opaque = MKT_CENTROID_OPAQUE(page);
	opaque->next_blkno			  = InvalidBlockNumber;
	opaque->entry_count			  = 0;
	opaque->level				  = level;
	opaque->flags				  = 0;
	opaque->page_id				  = MKT_CENTROID_PAGE_ID;
	opaque->first_global_idx	  = first_global_idx;
	opaque->reserved			  = 0;
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
	MktCentroidPageOpaque *opaque = MKT_CENTROID_OPAQUE(page);
	uint32_t			   max	  = mkt_centroid_max_entries(dim);
	uint16_t			   index  = opaque->entry_count;

	if (index >= max)
		return false;

	uint32_t packed_bytes = MKT_RABITQ_BYTES(dim);

	/* Write metadata */
	MktCentroidEntryMeta *meta = MKT_CENTROID_META(page);
	meta[index].child_blkno	   = child_blkno;
	meta[index].child_count	   = child_count;
	meta[index].flags		   = flags;
	meta[index].medoid_tid	   = *medoid_tid;
	meta[index].reserved	   = 0;

	/* Write RaBitQ factors */
	float *f_add	 = MKT_CENTROID_F_ADD(page, max);
	float *f_rescale = MKT_CENTROID_F_RESCALE(page, max);
	f_add[index]	 = data->f_add;
	f_rescale[index] = data->f_rescale;

	/* Write RaBitQ bits */
	uint8_t *bits = MKT_CENTROID_BITS(page, max);
	memcpy(bits + (size_t)index * packed_bytes, data->bits, packed_bytes);

	opaque->entry_count = index + 1;
	return true;
}
