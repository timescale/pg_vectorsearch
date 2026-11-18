/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * posting_page.c - Posting list page operations
 */

#include "vs_config.h"

#include <math.h>

#include "core/log.h"
#include "index/posting_page.h"
#include "types/vec16.h"

#ifndef VS_STANDALONE
#include <miscadmin.h> /* CHECK_FOR_INTERRUPTS */
#endif

void
prism_fastscan_check_page(
		BlockNumber					  blkno,
		const PrismPostingPageOpaque *op,
		uint32_t					  cap,
		uint32_t					  space)
{
	uint32_t groups	 = prism_fastscan_page_groups(op);
	uint32_t col_len = VS_FASTSCAN_GROUP_BYTES(8); /* one 8-dim column */

	if (op->entry_count > cap ||
		(size_t)groups * PRISM_FASTSCAN_META_BYTES + op->code_bytes > space ||
		op->code_bytes % col_len != 0)
		vs_error(
				VS_EXTENSION_NAME
				": fastscan posting page %u does not fit its page "
				"(%u entries, %u code bytes); "
				"the index may be corrupted -- REINDEX it",
				blkno,
				(unsigned)op->entry_count,
				(unsigned)op->code_bytes);
}

void
prism_posting_check_count(BlockNumber blkno, Page page, Dimension dim)
{
	const PrismPostingPageOpaque *op  = prism_posting_opaque(page);
	uint32_t					  cap = prism_posting_page_cap(op, dim);

	if (op->flags & PRISM_POSTING_PAGE_FASTSCAN)
	{
		bool prefix = prism_posting_has_ref_prefix(op->flags, dim);

		prism_fastscan_check_page(
				blkno,
				op,
				cap,
				prism_fastscan_page_space(
						dim,
						prefix,
						prism_posting_flags_ref_format(op->flags)));
		return;
	}

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

	/* A list's first entry page gives up room to the reference prefix, and
	 * a page that holds only reference has no room for entries. A fastscan
	 * page keeps its code length in the same field, and starts with none. */
	uint32_t max = 0;
	if (!(flags & PRISM_POSTING_PAGE_FASTSCAN) &&
		!prism_posting_is_reference_only_page(flags, dim))
		max = prism_posting_has_ref_prefix(flags, dim)
					? prism_posting_max_entries_first(
							  dim, prism_posting_flags_ref_format(flags))
					: prism_posting_max_entries(dim);
	PrismPostingPageOpaque *opaque = prism_posting_opaque(page);
	opaque->next_blkno			   = InvalidBlockNumber;
	opaque->cluster_id			   = cluster_id;
	opaque->entry_count			   = 0;
	opaque->flags				   = flags;
	opaque->page_id				   = PRISM_POSTING_PAGE_ID;
	opaque->max_entries			   = (uint16_t)max;
	/* Head metadata starts "not computed"; the first insert fills it (the
	 * fields are only consulted on the START page). */
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

/*
 * The flags of head page i of a list (0 = START): its part of the head, and
 * PRISM_POSTING_PAGE_REF16 for a float16 reference.
 */
static uint16_t
reference_page_kind(
		Dimension dim, uint32_t i, PrismPostingReferenceFormat ref_format)
{
	uint16_t ref = prism_posting_ref_format_flag(ref_format);

	if (i == 0)
		return PRISM_POSTING_HEAD_START | ref;
	if (i == prism_posting_reference_pages(dim, ref_format) - 1)
		return PRISM_POSTING_HEAD_END | ref;
	return PRISM_POSTING_HEAD_MIDDLE | ref;
}

void
prism_posting_reference_load(Page page, uint32_t n, float *out)
{
	if (prism_posting_flags_ref_format(prism_posting_opaque(page)->flags) ==
		PRISM_POSTING_REFERENCE_FLOAT16)
		vs_half_to_float_array((const half *)PageGetContents(page), out, n);
	else
		memcpy(out, PageGetContents(page), (size_t)n * sizeof(float));
}

void
prism_posting_reference_round16(float *ref, Dimension dim)
{
	for (Dimension d = 0; d < dim; d++)
	{
		if (!(fabsf(ref[d]) <= 65504.0f))
			vs_error(
					VS_EXTENSION_NAME
					": a posting-list reference does not fit in float16, "
					"whose largest value is 65504; build the index with "
					"posting_reference_format = float32");
		ref[d] = vs_half_to_float(vs_float_to_half(ref[d]));
	}
}

void
prism_posting_reference_store(Page page, const float *ref, uint32_t n)
{
	if (prism_posting_flags_ref_format(prism_posting_opaque(page)->flags) ==
		PRISM_POSTING_REFERENCE_FLOAT16)
		vs_float_to_half_array(ref, (half *)PageGetContents(page), n);
	else
		memcpy(PageGetContents(page), ref, (size_t)n * sizeof(float));
}

void
prism_posting_reference_page_fill(
		Page		 page,
		uint32_t	 cluster_id,
		Dimension	 dim,
		uint32_t	 i,
		const float *ref,
		uint16_t	 extra_flags)
{
	PrismPostingReferenceFormat ref_format = prism_posting_flags_ref_format(
			extra_flags);
	uint16_t flags = reference_page_kind(dim, i, ref_format) | extra_flags;
	uint32_t start = i * prism_posting_reference_page_capacity(ref_format);
	uint32_t n	   = prism_posting_spilled(dim, ref_format)
						   ? prism_posting_reference_page_elems(dim, i, ref_format)
						   : dim;

	prism_posting_page_init(page, cluster_id, dim, flags);
	prism_posting_reference_store(page, ref + start, n);
}

/*
 * Whether a page read while gathering a list's reference is the head page
 * expected at position i: START first, then MIDDLE pages, then END, all of
 * the START page's cluster, and with the reference width the index uses
 * (ref_format). `opaque` is the page's opaque. The START page (i = 0) sets
 * *cluster_id for the later pages to match. A page that does not match
 * means the chain is corrupt, and the caller reports it rather than read
 * its contents as reference.
 */
static bool
reference_page_matches(
		const PrismPostingPageOpaque *opaque,
		uint32_t					  i,
		uint32_t					 *cluster_id,
		Dimension					  dim,
		PrismPostingReferenceFormat	  ref_format)
{
	uint16_t want = reference_page_kind(dim, i, ref_format);

	if (i == 0)
		*cluster_id = opaque->cluster_id;
	return prism_posting_head_part(opaque->flags) ==
				   prism_posting_head_part(want) &&
		   prism_posting_flags_ref_format(opaque->flags) == ref_format &&
		   opaque->cluster_id == *cluster_id;
}

/* Copy head page i's part of the reference into out, as floats. */
static void
reference_copy_page(
		Page						page,
		uint32_t					i,
		Dimension					dim,
		PrismPostingReferenceFormat ref_format,
		float					   *out)
{
	uint32_t start = i * prism_posting_reference_page_capacity(ref_format);
	uint32_t n	   = prism_posting_spilled(dim, ref_format)
						   ? prism_posting_reference_page_elems(dim, i, ref_format)
						   : dim;

	prism_posting_reference_load(page, n, out + start);
}

static void
reference_page_error(
		BlockNumber head,
		BlockNumber blk,
		uint32_t	i,
		uint16_t	flags,
		uint32_t	cluster_id)
{
	vs_error(
			VS_EXTENSION_NAME
			": block %u is not reference page %u of posting list "
			"%u (flags 0x%04X, cluster %u); the index may be "
			"corrupted -- REINDEX it",
			blk,
			i,
			head,
			flags,
			cluster_id);
}

static void
not_posting_page_error(BlockNumber head, BlockNumber blk)
{
	vs_error(
			VS_EXTENSION_NAME
			": block %u of posting list %u is not a posting page; "
			"the index may be corrupted -- REINDEX it",
			blk,
			head);
}

/*
 * Copy the reference from head pages [first, end) of a list, starting at
 * blk. Returns the page the list's entries start on once the chain reaches
 * it (an inline START page or an END page), else the next page to read.
 */
static BlockNumber
posting_reference_read_chain(
		VsStorage				   *storage,
		BlockNumber					head,
		BlockNumber					blk,
		uint32_t					first,
		uint32_t					end,
		uint32_t				   *cluster_id,
		Dimension					dim,
		PrismPostingReferenceFormat ref_format,
		float					   *out)
{
	uint32_t npages = prism_posting_reference_pages(dim, ref_format);

	for (uint32_t i = first; i < end; i++)
	{
		if (blk == InvalidBlockNumber)
			vs_error(
					VS_EXTENSION_NAME
					": posting list %u ends after %u of %u reference "
					"pages; the index may be corrupted -- REINDEX it",
					head,
					i,
					npages);

		Page page = vs_storage_read_page(storage, blk);
		if (!prism_page_is_posting(page))
		{
			vs_storage_release_page(storage, blk);
			not_posting_page_error(head, blk);
		}

		const PrismPostingPageOpaque *op = prism_posting_opaque(page);
		if (!reference_page_matches(op, i, cluster_id, dim, ref_format))
		{
			uint16_t flags = op->flags;
			uint32_t cid   = op->cluster_id;

			vs_storage_release_page(storage, blk);
			reference_page_error(head, blk, i, flags, cid);
		}

		reference_copy_page(page, i, dim, ref_format, out);
		BlockNumber next = op->next_blkno;
		vs_storage_release_page(storage, blk);

		if (!prism_posting_is_reference_only_page(
					reference_page_kind(dim, i, ref_format), dim))
			return blk;
		blk = next;
	}
	return blk;
}

BlockNumber
prism_posting_reference_read(
		VsStorage				   *storage,
		BlockNumber					head,
		Dimension					dim,
		PrismPostingReferenceFormat ref_format,
		float					   *out)
{
	uint32_t cluster_id = 0;

	return posting_reference_read_chain(
			storage,
			head,
			head,
			0,
			prism_posting_reference_pages(dim, ref_format),
			&cluster_id,
			dim,
			ref_format,
			out);
}

BlockNumber
prism_posting_reference_read_to_end(
		VsStorage				   *storage,
		BlockNumber					head,
		Dimension					dim,
		PrismPostingReferenceFormat ref_format,
		float					   *out,
		uint32_t				   *cluster_id)
{
	uint32_t npages = prism_posting_reference_pages(dim, ref_format);
	uint32_t end	= prism_posting_spilled(dim, ref_format) ? npages - 1
															 : npages;

	return posting_reference_read_chain(
			storage, head, head, 0, end, cluster_id, dim, ref_format, out);
}

void
prism_posting_reference_read_end(
		BlockNumber					head,
		BlockNumber					blk,
		Page						page,
		uint32_t					cluster_id,
		Dimension					dim,
		PrismPostingReferenceFormat ref_format,
		float					   *out)
{
	uint32_t last = prism_posting_reference_pages(dim, ref_format) - 1;

	Assert(prism_posting_spilled(dim, ref_format));
	if (!prism_page_is_posting(page))
		not_posting_page_error(head, blk);

	const PrismPostingPageOpaque *op = prism_posting_opaque(page);
	if (!reference_page_matches(op, last, &cluster_id, dim, ref_format))
		reference_page_error(head, blk, last, op->flags, op->cluster_id);
	reference_copy_page(page, last, dim, ref_format, out);
}

BlockNumber
prism_posting_reference_read_rest(
		VsStorage				   *storage,
		BlockNumber					head,
		BlockNumber					next,
		uint32_t					cluster_id,
		Dimension					dim,
		PrismPostingReferenceFormat ref_format,
		float					   *out)
{
	if (!prism_posting_spilled(dim, ref_format))
		return next;
	return posting_reference_read_chain(
			storage,
			head,
			next,
			1,
			prism_posting_reference_pages(dim, ref_format),
			&cluster_id,
			dim,
			ref_format,
			out);
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
