/*
 * posting_convert.c - Convert AoS posting pages to fastscan format
 *
 * Walks an AoS posting chain, reads pre-encoded entries, and feeds
 * them to MktFsPostingBuilder via _add_encoded(). The builder
 * handles group packing, page layout, and chain linking.
 */

#include <string.h>

#include "core/memory.h"
#include "index/posting_build.h"
#include "index/posting_convert.h"
#include "quant/fastscan.h"
#include "quant/rabitq.h"

BlockNumber
mkt_posting_convert_to_fastscan(
		MktStorage *storage, BlockNumber aos_head, Dimension dim)
{
	if (aos_head == InvalidBlockNumber)
		return InvalidBlockNumber;

	uint32_t packed_bytes = MKT_RABITQ_BYTES(dim);

	/* Read pt_centroid from AoS first page */
	Page   first_page  = mkt_storage_read_page(storage, aos_head);
	float *pt_centroid = mkt_alloc(dim * sizeof(float));
	memcpy(pt_centroid,
		   mkt_posting_pt_centroid(first_page),
		   dim * sizeof(float));

	MktPostingPageOpaque *first_op	 = mkt_posting_opaque(first_page);
	uint32_t			  cluster_id = first_op->cluster_id;
	mkt_storage_release_page(storage, aos_head);

	MktPostingBuilder builder;
	mkt_posting_builder_init_fastscan(
			&builder, storage, NULL, dim, cluster_id, NULL, pt_centroid);

	/*
	 * Walk AoS chain and feed entries to builder. We stage all
	 * entries first because the PG storage layer tracks only one
	 * pinned buffer at a time — interleaving reads and writes
	 * would clobber the buffer reference.
	 */
	typedef struct StagedEntry
	{
		ItemPointerData tid;
		float			f_add;
		float			f_rescale;
		float			f_error;
	} StagedEntry;

	uint32_t	 total_entries = 0;
	uint32_t	 entries_cap   = 256;
	StagedEntry *staged		   = mkt_alloc(entries_cap * sizeof(StagedEntry));
	uint8_t		*all_bits	   = mkt_alloc(entries_cap * (size_t)packed_bytes);

	BlockNumber blkno = aos_head;
	while (blkno != InvalidBlockNumber)
	{
		Page				  page = mkt_storage_read_page(storage, blkno);
		MktPostingPageOpaque *op   = mkt_posting_opaque(page);

		char *content = (op->flags & MKT_POSTING_PAGE_FIRST)
							  ? mkt_posting_content_first(page, dim)
							  : mkt_posting_content(page);

		for (uint32_t i = 0; i < op->entry_count; i++)
		{
			if (total_entries >= entries_cap)
			{
				entries_cap *= 2;
				staged =
						mkt_realloc(staged, entries_cap * sizeof(StagedEntry));
				all_bits = mkt_realloc(
						all_bits, entries_cap * (size_t)packed_bytes);
			}
			MktPostingEntryHeader *e  = mkt_posting_entry_at(content, i, dim);
			staged[total_entries].tid = e->meta.tid;
			staged[total_entries].f_add		= e->f_add;
			staged[total_entries].f_rescale = e->f_rescale;
			staged[total_entries].f_error	= e->f_error;
			memcpy(all_bits + (size_t)total_entries * packed_bytes,
				   e->bits,
				   packed_bytes);
			total_entries++;
		}

		BlockNumber next = op->next_blkno;
		mkt_storage_release_page(storage, blkno);
		blkno = next;
	}

	/* Feed staged entries to the fastscan builder */
	for (uint32_t i = 0; i < total_entries; i++)
	{
		mkt_posting_builder_add_encoded(
				&builder,
				staged[i].tid,
				staged[i].f_add,
				staged[i].f_rescale,
				staged[i].f_error,
				all_bits + (size_t)i * packed_bytes);
	}

	mkt_free(staged);
	mkt_free(all_bits);

	BlockNumber result = mkt_posting_builder_finish(&builder);
	mkt_posting_builder_cleanup(&builder);
	mkt_free(pt_centroid);

	return result;
}
