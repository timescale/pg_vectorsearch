/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * posting_page.c - Posting list page operations
 */

#include "vs_config.h"

#include "core/log.h"
#include "index/posting_page.h"

#ifndef VS_STANDALONE
#include <miscadmin.h> /* CHECK_FOR_INTERRUPTS */
#endif

void
prism_posting_check_count(
		BlockNumber blkno, const PrismPostingPageOpaque *op, Dimension dim)
{
	uint32_t cap = prism_posting_page_cap(op, dim);

	if (op->entry_count > cap)
		vs_error(
				VS_EXTENSION_NAME
				": posting page %u has an invalid entry count (%u > %u); "
				"the index may be corrupted -- REINDEX it",
				blkno,
				(unsigned)op->entry_count,
				cap);
}

/* ----------------------------------------------------------------
 * Paged mode (BLCKSZ pages with PG header + opaque)
 * ---------------------------------------------------------------- */

void
prism_posting_page_init(
		Page page, uint32_t cluster_id, Dimension dim, uint16_t flags)
{
	PageInit(page, BLCKSZ, sizeof(PrismPostingPageOpaque));

	/* An inline first page gives up room to pt_centroid; a REFERENCE page
	 * holds a chunk of it and no entries at all. */
	uint32_t max;
	if (flags & PRISM_POSTING_PAGE_REFERENCE)
		max = 0;
	else if (flags & PRISM_POSTING_PAGE_FIRST)
		max = prism_posting_max_entries_first(dim);
	else
		max = prism_posting_max_entries(dim);
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

	/* A live entry means the page is no longer wholly dead, so drop the
	 * scan-side shortcut -- see PRISM_POSTING_PAGE_TOMBSTONED. */
	if ((entry_flags & PRISM_POSTING_FLAG_DELETED) == 0)
		opaque->flags &= (uint16_t)~PRISM_POSTING_PAGE_TOMBSTONED;

	return true;
}

/* ----------------------------------------------------------------
 * Encode reference, inline or spilled
 * ---------------------------------------------------------------- */

void
prism_posting_reference_page_fill(
		Page		 page,
		uint32_t	 cluster_id,
		Dimension	 dim,
		uint32_t	 chunk,
		const float *ref,
		uint16_t	 extra_flags)
{
	uint16_t flags = (chunk == 0 ? PRISM_POSTING_PAGE_FIRST
								 : PRISM_POSTING_PAGE_OVERFLOW) |
					 PRISM_POSTING_PAGE_REFERENCE | extra_flags;
	uint32_t start	 = chunk * (uint32_t)PRISM_POSTING_REF_CHUNK_FLOATS;
	uint32_t nfloats = prism_posting_ref_chunk_floats(dim, chunk);

	prism_posting_page_init(page, cluster_id, dim, flags);
	memcpy(prism_posting_pt_centroid_mut(page),
		   ref + start,
		   (size_t)nfloats * sizeof(float));
}

/* Chunks [first_chunk, npages) of a reference chain starting at blk. */
static BlockNumber
reference_read_chain(
		VsStorage  *storage,
		BlockNumber head,
		BlockNumber blk,
		uint32_t	first_chunk,
		uint32_t	cluster_id,
		Dimension	dim,
		float	   *out)
{
	uint32_t npages = prism_posting_ref_pages(dim);

	for (uint32_t chunk = first_chunk; chunk < npages; chunk++)
	{
		if (blk == InvalidBlockNumber)
			vs_error(
					VS_EXTENSION_NAME
					": posting list %u ends after %u of %u reference "
					"pages; the index may be corrupted -- REINDEX it",
					head,
					chunk,
					npages);

		Page page = vs_storage_read_page(storage, blk);
		if (!prism_page_is_posting(page))
		{
			vs_storage_release_page(storage, blk);
			vs_error(
					VS_EXTENSION_NAME
					": block %u of posting list %u is not a posting page; "
					"the index may be corrupted -- REINDEX it",
					blk,
					head);
		}

		const PrismPostingPageOpaque *op = prism_posting_opaque(page);
		bool spilled = (op->flags & PRISM_POSTING_PAGE_REFERENCE) != 0;
		bool first	 = (op->flags & PRISM_POSTING_PAGE_FIRST) != 0;

		/* The head says which layout the list has; every later chunk must
		 * agree with it and belong to the same list. */
		if (chunk == 0)
			cluster_id = op->cluster_id;
		if (first != (chunk == 0) || spilled != prism_posting_spilled(dim) ||
			op->cluster_id != cluster_id)
		{
			vs_storage_release_page(storage, blk);
			vs_error(
					VS_EXTENSION_NAME
					": block %u is not reference page %u of posting list "
					"%u (flags 0x%04X, cluster %u); the index may be "
					"corrupted -- REINDEX it",
					blk,
					chunk,
					head,
					op->flags,
					op->cluster_id);
		}

		uint32_t start	 = chunk * (uint32_t)PRISM_POSTING_REF_CHUNK_FLOATS;
		uint32_t nfloats = spilled ? prism_posting_ref_chunk_floats(dim, chunk)
								   : dim;
		memcpy(out + start,
			   prism_posting_pt_centroid(page),
			   (size_t)nfloats * sizeof(float));

		BlockNumber next = op->next_blkno;
		vs_storage_release_page(storage, blk);

		/* An inline head holds its own entries; a spilled chain's entries
		 * begin past the last chunk. */
		if (!spilled)
			return blk;
		blk = next;
	}
	return blk;
}

BlockNumber
prism_posting_reference_read(
		VsStorage *storage, BlockNumber head, Dimension dim, float *out)
{
	return reference_read_chain(storage, head, head, 0, 0, dim, out);
}

BlockNumber
prism_posting_reference_read_rest(
		VsStorage  *storage,
		BlockNumber head,
		BlockNumber next,
		uint32_t	cluster_id,
		Dimension	dim,
		float	   *out)
{
	if (!prism_posting_spilled(dim))
		return next;
	return reference_read_chain(storage, head, next, 1, cluster_id, dim, out);
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
		CHECK_FOR_INTERRUPTS();

		pos.blkno = blk;
		pos.page  = vs_storage_read_page(storage, blk);

		/*
		 * Reading next_blkno means casting the special area, so establish
		 * that it is a posting page first -- the same guard every other
		 * page read applies. A chain that leaves posting pages is a
		 * corrupt chain, and ends the walk rather than being followed.
		 */
		if (!prism_page_is_posting(pos.page))
		{
			prism_posting_chain_release(&pos);
			return;
		}

		pos.next = prism_posting_opaque(pos.page)->next_blkno;

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
