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

#include <string.h>

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
} StageCtx;

static bool
stage_page(PrismPostingChainPos *pos, void *state)
{
	StageCtx					 *ctx = state;
	const PrismPostingPageOpaque *op  = prism_posting_opaque(pos->page);
	char *content = prism_posting_page_content(pos->page, ctx->dim);

	prism_posting_check_count(pos->blkno, op, ctx->dim);

	for (uint32_t i = 0; i < op->entry_count; i++)
	{
		PrismPostingEntryHeader *src =
				prism_posting_entry_at(content, i, ctx->dim);

		/*
		 * Leave behind the entries VACUUM has marked dead. A fastscan page
		 * packs codes with no per-entry flag -- deletion there is
		 * page-granular -- so a dead entry copied into one comes back as
		 * live and can never be marked again: a later VACUUM can only
		 * tombstone the page once *every* entry on it is dead, which a page
		 * holding live entries never is. The head's live_count, which the
		 * builder stamps from what it was given, would be wrong by the same
		 * number.
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
		VsStorage *storage, BlockNumber aos_head, Dimension dim)
{
	if (aos_head == InvalidBlockNumber)
		return InvalidBlockNumber;

	uint32_t packed_bytes = VS_RABITQ_BYTES(dim);

	/* Read pt_centroid from AoS first page */
	Page   first_page  = vs_storage_read_page(storage, aos_head);
	float *pt_centroid = vs_alloc(dim * sizeof(float));
	memcpy(pt_centroid,
		   prism_posting_pt_centroid(first_page),
		   dim * sizeof(float));

	PrismPostingPageOpaque *first_op   = prism_posting_opaque(first_page);
	uint32_t				cluster_id = first_op->cluster_id;
	vs_storage_release_page(storage, aos_head);

	PrismPostingBuilder builder;
	prism_posting_builder_init_fastscan(
			&builder, storage, NULL, dim, cluster_id, NULL, pt_centroid);

	StageCtx ctx = {
			.dim		  = dim,
			.packed_bytes = packed_bytes,
			.entries_cap  = 256,
	};
	ctx.staged	 = vs_alloc(ctx.entries_cap * sizeof(StagedEntry));
	ctx.all_bits = vs_alloc(ctx.entries_cap * (size_t)packed_bytes);

	prism_posting_chain_walk(storage, aos_head, stage_page, &ctx);

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
