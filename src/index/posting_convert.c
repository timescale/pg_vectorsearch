/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * posting_convert.c - Convert AoS posting pages to fastscan format
 *
 * Walks an AoS posting chain, reads pre-encoded entries, and feeds
 * them to PrismPostingBuilder via _add_encoded(). The builder
 * handles group packing, page layout, and chain linking.
 */

#include "vs_config.h"

#include <string.h>

#include "core/log.h"
#include "core/memory.h"
#include "index/posting_build.h"
#include "index/posting_convert.h"
#include "quant/fastscan.h"
#include "quant/rabitq.h"

/*
 * Entries staged out of the AoS chain before any of them is written back.
 *
 * The PG storage layer holds one pinned buffer at a time, so reads and
 * writes cannot interleave: appending to the builder mid-walk would take
 * over the buffer the walk is reading from. Staging everything first is what
 * keeps the two apart.
 */
typedef struct StagedEntry
{
	ItemPointerData tid;
	float			f_add;
	float			f_rescale;
	float			f_error;
} StagedEntry;

typedef struct StageCtx
{
	Dimension	 dim;
	uint32_t	 packed_bytes;
	uint32_t	 total_entries;
	uint32_t	 entries_cap;
	StagedEntry *staged;
	uint8_t		*all_bits;
	BlockNumber	 head;
	bool		 aos_seen;		/* an AoS entry page came first */
	bool		 fastscan_base; /* the list starts with fastscan pages */
} StageCtx;

static bool
stage_page(PrismPostingChainPos *pos, void *state)
{
	StageCtx					 *ctx = state;
	const PrismPostingPageOpaque *op  = prism_posting_opaque(pos->page);
	char *content = prism_posting_page_content(pos->page, ctx->dim);

	prism_posting_check_count(pos->blkno, pos->page, ctx->dim);

	/*
	 * A list built, converted or split as fastscan starts with its fastscan
	 * pages, and inserts only ever append AoS pages after them. So the first
	 * fastscan page ends the walk: before any AoS page it means the list is
	 * already converted, and after one the chain is not one we write.
	 */
	if (op->flags & PRISM_POSTING_PAGE_FASTSCAN)
	{
		if (ctx->aos_seen)
			vs_error(
					VS_EXTENSION_NAME
					": posting list %u has a fastscan page (block %u) after "
					"AoS ones; the index may be corrupted -- REINDEX it",
					ctx->head,
					pos->blkno);
		ctx->fastscan_base = true;
		return false;
	}
	if (!prism_posting_is_reference_only_page(op->flags, ctx->dim))
		ctx->aos_seen = true;

	for (uint32_t i = 0; i < op->entry_count; i++)
	{
		PrismPostingEntryHeader *src =
				prism_posting_entry_at(content, i, ctx->dim);

		/*
		 * Leave behind the entries VACUUM has marked dead. Nothing here
		 * carries a dead AoS entry's state into the fresh fastscan chain --
		 * the destination's tombstone_mask words start at zero, same as any
		 * freshly built page -- so copying the entry would resurrect it as
		 * live. The head's live_count, which the builder stamps from what it
		 * was given, would be wrong by the same number.
		 */
		if (src->meta.flags & PRISM_POSTING_FLAG_DELETED)
			continue;

		if (ctx->total_entries >= ctx->entries_cap)
		{
			ctx->entries_cap *= 2;
			ctx->staged = vs_realloc(
					ctx->staged, ctx->entries_cap * sizeof(StagedEntry));
			ctx->all_bits = vs_realloc(
					ctx->all_bits,
					ctx->entries_cap * (size_t)ctx->packed_bytes);
		}

		StagedEntry *dst = &ctx->staged[ctx->total_entries];

		dst->tid	   = src->meta.tid;
		dst->f_add	   = src->f_add;
		dst->f_rescale = src->f_rescale;
		dst->f_error   = src->f_error;
		memcpy(ctx->all_bits + (size_t)ctx->total_entries * ctx->packed_bytes,
			   src->bits,
			   ctx->packed_bytes);
		ctx->total_entries++;
	}

	return true;
}

BlockNumber
prism_posting_convert_to_fastscan(
		VsStorage  *storage,
		BlockNumber aos_head,
		Dimension	dim,
		bool	   *has_fastscan_base)
{
	*has_fastscan_base = false;
	if (aos_head == InvalidBlockNumber)
		return InvalidBlockNumber;
	uint32_t packed_bytes = VS_RABITQ_BYTES(dim);

	/* The cluster id from the AoS head, then its encode reference. */
	Page	 first_page = vs_storage_read_page(storage, aos_head);
	uint32_t cluster_id = prism_posting_opaque(first_page)->cluster_id;
	PrismPostingReferenceFormat ref_format = prism_posting_flags_ref_format(
			prism_posting_opaque(first_page)->flags);
	vs_storage_release_page(storage, aos_head);

	float *pt_centroid = vs_alloc(dim * sizeof(float));
	prism_posting_reference_read(
			storage, aos_head, dim, ref_format, pt_centroid);

	PrismPostingBuilder builder;
	prism_posting_builder_init_ref(
			&builder,
			storage,
			NULL,
			dim,
			cluster_id,
			NULL,
			pt_centroid,
			true,
			ref_format);

	StageCtx ctx = {
			.dim		  = dim,
			.packed_bytes = packed_bytes,
			.entries_cap  = 256,
			.head		  = aos_head,
	};
	ctx.staged	 = vs_alloc(ctx.entries_cap * sizeof(StagedEntry));
	ctx.all_bits = vs_alloc(ctx.entries_cap * (size_t)packed_bytes);

	prism_posting_chain_walk(storage, aos_head, stage_page, &ctx);

	/* The builder writes nothing until it gets entries, so leaving it
	 * unfinished leaves the index untouched. */
	if (ctx.fastscan_base)
	{
		*has_fastscan_base = true;
		vs_free(ctx.staged);
		vs_free(ctx.all_bits);
		prism_posting_builder_cleanup(&builder);
		vs_free(pt_centroid);
		return InvalidBlockNumber;
	}

	/* Feed staged entries to the fastscan builder */
	for (uint32_t i = 0; i < ctx.total_entries; i++)
	{
		prism_posting_builder_add_encoded(
				&builder,
				ctx.staged[i].tid,
				ctx.staged[i].f_add,
				ctx.staged[i].f_rescale,
				ctx.staged[i].f_error,
				ctx.all_bits + (size_t)i * packed_bytes);
	}

	vs_free(ctx.staged);
	vs_free(ctx.all_bits);

	BlockNumber result = prism_posting_builder_finish(&builder);
	prism_posting_builder_cleanup(&builder);
	vs_free(pt_centroid);

	return result;
}
