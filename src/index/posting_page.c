/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * posting_page.c - Posting list page operations
 */

#include "index/posting_page.h"

/* ----------------------------------------------------------------
 * Paged mode (BLCKSZ pages with PG header + opaque)
 * ---------------------------------------------------------------- */

void
prism_posting_page_init(
		Page page, uint32_t cluster_id, Dimension dim, uint16_t flags)
{
	PageInit(page, BLCKSZ, sizeof(PrismPostingPageOpaque));

	/* First pages have reduced capacity due to pt_centroid */
	uint32_t				max	   = (flags & PRISM_POSTING_PAGE_FIRST)
										   ? prism_posting_max_entries_first(dim)
										   : prism_posting_max_entries(dim);
	PrismPostingPageOpaque *opaque = prism_posting_opaque(page);
	opaque->next_blkno			   = InvalidBlockNumber;
	opaque->cluster_id			   = cluster_id;
	opaque->entry_count			   = 0;
	opaque->flags				   = flags;
	opaque->page_id				   = PRISM_POSTING_PAGE_ID;
	opaque->max_entries			   = (uint16_t)max;
	/* Head metadata starts "not computed"; the first insert fills it (the
	 * fields are only consulted on the FIRST page). */
	opaque->live_count = 0;
	opaque->tail_blkno = InvalidBlockNumber;
}

bool
prism_posting_page_add(
		Page			page,
		Dimension		dim,
		ItemPointerData tid,
		float			f_add,
		float			f_rescale,
		float			f_error,
		const uint8_t  *bits,
		uint8_t			entry_flags)
{
	if (!prism_posting_page_has_room(page))
		return false;

	PrismPostingPageOpaque *opaque	= prism_posting_opaque(page);
	uint32_t				i		= opaque->entry_count;
	char				   *content = prism_posting_page_content(page, dim);

	/* Write the entry header (meta + factors) + bits in one contiguous
	 * block at content + i * entry_size. */
	PrismPostingEntryHeader *hdr = prism_posting_entry_at(content, i, dim);
	hdr->meta.tid				 = tid;
	hdr->meta.flags				 = entry_flags;
	hdr->meta.reserved			 = 0;
	hdr->f_add					 = f_add;
	hdr->f_rescale				 = f_rescale;
	hdr->f_error				 = f_error;
	memcpy(hdr->bits, bits, VS_RABITQ_BYTES(dim));

	opaque->entry_count = i + 1;
	return true;
}

/* ----------------------------------------------------------------
 * Chain walks
 * ---------------------------------------------------------------- */

void
prism_posting_chain_walk(
		VsStorage		   *storage,
		BlockNumber			head,
		PrismPostingChainCb cb,
		void			   *state)
{
	PrismPostingChainPos pos = {.storage = storage, .first = true};
	BlockNumber			 blk = head;

	while (blk != InvalidBlockNumber)
	{
		pos.blkno = blk;
		pos.page  = vs_storage_read_page(storage, blk);
		pos.next  = prism_posting_opaque(pos.page)->next_blkno;

		bool keep_going = cb(&pos, state);

		/* No-op when the callback released it already. */
		prism_posting_chain_release(&pos);

		if (!keep_going)
			return;

		pos.first = false;
		blk		  = pos.next;
	}
}

void
prism_posting_chain_mutate(
		VsStorage				 *storage,
		BlockNumber				  head,
		PrismPostingChainMutateCb cb,
		void					 *state)
{
	BlockNumber blk = head;

	while (blk != InvalidBlockNumber)
	{
		Page					page = vs_storage_write_page(storage, blk);
		PrismPostingPageOpaque *op	 = prism_posting_opaque(page);
		BlockNumber				next = op->next_blkno;

		cb(op, state);

		vs_storage_commit_page(storage, blk);
		blk = next;
	}
}

/* ----------------------------------------------------------------
 * Flat mode (one buffer per cluster, custom header)
 * ---------------------------------------------------------------- */

void
prism_posting_flat_init(char *buf, uint32_t max_entries, uint32_t cluster_id)
{
	PrismFlatPostingHeader *hdr = prism_flat_posting_header(buf);
	hdr->max_entries			= max_entries;
	hdr->entry_count			= 0;
	hdr->cluster_id				= cluster_id;
	hdr->_pad					= 0;
}

bool
prism_posting_flat_add(
		char		   *buf,
		Dimension		dim,
		ItemPointerData tid,
		float			f_add,
		float			f_rescale,
		float			f_error,
		const uint8_t  *bits,
		uint8_t			entry_flags)
{
	PrismFlatPostingHeader *flat_hdr = prism_flat_posting_header(buf);
	if (flat_hdr->entry_count >= flat_hdr->max_entries)
		return false;

	uint32_t i		 = flat_hdr->entry_count;
	char	*content = prism_flat_posting_content(buf);

	PrismPostingEntryHeader *hdr = prism_posting_entry_at(content, i, dim);
	hdr->meta.tid				 = tid;
	hdr->meta.flags				 = entry_flags;
	hdr->meta.reserved			 = 0;
	hdr->f_add					 = f_add;
	hdr->f_rescale				 = f_rescale;
	hdr->f_error				 = f_error;
	memcpy(hdr->bits, bits, VS_RABITQ_BYTES(dim));

	flat_hdr->entry_count = i + 1;
	return true;
}
