/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * posting_insert.c - Runtime insert primitives (see header)
 */

#include <math.h>

#include "core/memory.h"
#include "index/posting_insert.h"

bool
prism_posting_insert_one(
		VsStorage		   *storage,
		const RaBitQParams *params,
		Dimension			dim,
		BlockNumber			head_blkno,
		ItemPointerData		tid,
		const float		   *pt_input,
		RaBitQScratch	   *scratch,
		bool				unreachable,
		bool			   *head_retired)
{
	if (head_retired != NULL)
		*head_retired = false;

	if (storage == NULL || params == NULL || pt_input == NULL ||
		scratch == NULL || head_blkno == InvalidBlockNumber)
		return false;

	/*
	 * Phase A: read the head — compute the rotated residual against its
	 * pt_centroid and snapshot the head metadata, then release it (the PG
	 * backing holds only one buffer at a time).
	 */
	Page head = vs_storage_read_page(storage, head_blkno);
	const PrismPostingPageOpaque *hop = prism_posting_opaque(head);

	/*
	 * If the head was retired since it was routed to (split away by a
	 * concurrent rebalance), don't insert into a dead chain — report it so the
	 * caller re-routes. Checked here, on the read we already do, so the caller
	 * needn't re-read the head just to test these flags.
	 */
	if (hop->flags &
		(PRISM_POSTING_PAGE_TOMBSTONED | PRISM_POSTING_PAGE_DELETED))
	{
		vs_storage_release_page(storage, head_blkno);
		if (head_retired != NULL)
			*head_retired = true;
		return false;
	}

	uint32_t	 cluster_id = hop->cluster_id;
	BlockNumber	 tail_blkno = hop->tail_blkno;
	uint32_t	 live		= hop->live_count;
	const float *pt_cent	= prism_posting_pt_centroid(head);

	/* pt_residual = pt_input - pt_centroid (P^T linear). Reuse scratch
	 * (encode_from_pt only touches scratch->xu_cb). */
	float *pt_residual = scratch->transformed;
	for (Dimension i = 0; i < dim; i++)
		pt_residual[i] = pt_input[i] - pt_cent[i];

	vs_storage_release_page(storage, head_blkno);

	/* Encode relative to the leaf centroid. */
	RaBitQData *rdata = vs_alloc(VS_RABITQ_DATA_SIZE(dim));
	vs_rabitq_encode_from_pt(params, pt_residual, rdata, scratch);
	float f_error =
			vs_rabitq_derive_f_error(rdata->f_add, rdata->f_rescale, dim);

	/* No defined distance under the index metric: estimated distance
	 * +inf with zero error, so scans prune the entry before it can
	 * enter the top-k threshold heap (see mark_entry_unreachable in
	 * posting_build.c for the full rationale). */
	if (unreachable)
	{
		rdata->f_add = INFINITY;
		f_error		 = 0.0f;
	}

	/*
	 * tail_blkno / live are read straight from the head: the build stamps both
	 * for every cluster (see prism_posting_builder_finish and the parallel
	 * leader's finalize), and inserts maintain them, so a built head always
	 * has a valid tail — no chain walk on the insert path.
	 */

	/*
	 * Phase B: append to the tail. Read it first to check for room (released
	 * before any write); the caller's per-cluster lock prevents a concurrent
	 * fill in between.
	 */
	Page tcheck = vs_storage_read_page(storage, tail_blkno);
	/*
	 * Only append into an AoS tail that has room. A FASTSCAN page packs codes
	 * in immutable 32-vector groups, so an insert can never append to it — it
	 * starts a fresh AoS overflow page instead (the scan merges the mixed
	 * chain by per-page format).
	 */
	bool tail_is_aos = (prism_posting_opaque(tcheck)->flags &
						PRISM_POSTING_PAGE_FASTSCAN) == 0;
	bool has_room	 = tail_is_aos && prism_posting_page_has_room(tcheck);
	vs_storage_release_page(storage, tail_blkno);

	BlockNumber new_tail	 = tail_blkno;
	bool		head_updated = false;

	if (has_room)
	{
		Page wp = vs_storage_write_page(storage, tail_blkno);
		prism_posting_page_add(
				wp,
				dim,
				tid,
				rdata->f_add,
				rdata->f_rescale,
				f_error,
				rdata->bits,
				0);
		if (tail_blkno == head_blkno)
		{
			/* Tail is the head — fold the metadata update into this write so
			 * single-page clusters cost one head write, not two. */
			PrismPostingPageOpaque *op = prism_posting_opaque(wp);
			op->tail_blkno			   = new_tail;
			op->live_count			   = live + 1;
			head_updated			   = true;
		}
		vs_storage_commit_page(storage, tail_blkno);
	}
	else
	{
		/* Tail full: build the new overflow page fully and commit it BEFORE
		 * linking, so a crash never points the chain at an uninitialized
		 * page (it leaves at worst an unreferenced page). */
		BlockNumber t2;
		Page		np = vs_storage_new_page(storage, &t2);
		prism_posting_page_init(
				np, cluster_id, dim, PRISM_POSTING_PAGE_OVERFLOW);
		prism_posting_page_add(
				np,
				dim,
				tid,
				rdata->f_add,
				rdata->f_rescale,
				f_error,
				rdata->bits,
				0);
		vs_storage_commit_page(storage, t2);

		Page lp = vs_storage_write_page(storage, tail_blkno);
		prism_posting_opaque(lp)->next_blkno = t2;
		if (tail_blkno == head_blkno)
		{
			PrismPostingPageOpaque *op = prism_posting_opaque(lp);
			op->tail_blkno			   = t2;
			op->live_count			   = live + 1;
			head_updated			   = true;
		}
		vs_storage_commit_page(storage, tail_blkno);
		new_tail = t2;
	}

	/* Phase C: update the head metadata when it wasn't folded above. */
	if (!head_updated)
	{
		Page hw = vs_storage_write_page(storage, head_blkno);
		PrismPostingPageOpaque *op = prism_posting_opaque(hw);
		op->tail_blkno			   = new_tail;
		op->live_count			   = live + 1;
		vs_storage_commit_page(storage, head_blkno);
	}

	return true;
}

/*
 * Contract is in the header. Implementation notes not stated there: each page
 * is read-scanned first and only write-locked / WAL-logged when it actually
 * has a dead entry to mark, so vacuuming a chain with no dead tuples dirties
 * nothing. Marking is idempotent — an already-deleted entry is skipped, not
 * recounted — so a repeated VACUUM over the same dead TIDs is a no-op.
 */
uint32_t
prism_posting_tombstone_chain(
		VsStorage  *storage,
		Dimension	dim,
		BlockNumber head_blkno,
		bool (*is_dead)(ItemPointerData tid, void *state),
		void *state)
{
	if (storage == NULL || is_dead == NULL || head_blkno == InvalidBlockNumber)
		return 0;

	uint32_t	total_marked = 0;
	BlockNumber blk			 = head_blkno;

	while (blk != InvalidBlockNumber)
	{
		/*
		 * Phase 1: under a SHARE lock, scan the page's entries to find whether
		 * it has any dead entry that still needs marking. This is only a probe
		 * — nothing is mutated — so a page with no dead tuples is never
		 * dirtied or WAL-logged. The AoS scan early-breaks at the first such
		 * entry (it only needs to know "is there work?"); the FASTSCAN scan
		 * instead checks whether the whole page is dead, since packed entries
		 * can't be flagged individually.
		 */
		Page						  p	 = vs_storage_read_page(storage, blk);
		const PrismPostingPageOpaque *op = prism_posting_opaque(p);
		BlockNumber					  next	= op->next_blkno;
		uint16_t					  flags = op->flags;
		uint32_t					  n		= op->entry_count;

		prism_posting_check_count(blk, op, dim);

		/* Already fully tombstoned: nothing to mark, and skip so its entries
		 * aren't counted into the live_count decrement twice. */
		if (flags & PRISM_POSTING_PAGE_TOMBSTONED)
		{
			vs_storage_release_page(storage, blk);
			blk = next;
			continue;
		}

		bool  is_fastscan = (flags & PRISM_POSTING_PAGE_FASTSCAN) != 0;
		char *content	  = prism_posting_page_content(p, dim);
		bool  needs_mark  = false; /* AoS: has a not-yet-deleted dead entry */
		bool  fs_all_dead = false; /* FASTSCAN: every entry is dead */

		if (!is_fastscan)
		{
			for (uint32_t i = 0; i < n; i++)
			{
				PrismPostingEntryHeader *h =
						prism_posting_entry_at(content, i, dim);
				if (!(h->meta.flags & PRISM_POSTING_FLAG_DELETED) &&
					is_dead(h->meta.tid, state))
				{
					needs_mark = true;
					break;
				}
			}
		}
		else if (n > 0)
		{
			/* FASTSCAN entries can't be flagged individually, but a wholly
			 * dead page can be tombstoned at page granularity. Early-exit on
			 * the first live TID, so live pages cost little. */
			fs_all_dead		 = true;
			uint32_t ngroups = (n + VS_FASTSCAN_GROUP - 1) / VS_FASTSCAN_GROUP;
			for (uint32_t g = 0; g < ngroups && fs_all_dead; g++)
			{
				uint32_t g_count = n - g * VS_FASTSCAN_GROUP;
				if (g_count > VS_FASTSCAN_GROUP)
					g_count = VS_FASTSCAN_GROUP;
				ItemPointerData *tids =
						prism_fastscan_group_tids(content, g, dim);
				for (uint32_t v = 0; v < g_count; v++)
					if (!is_dead(tids[v], state))
					{
						fs_all_dead = false;
						break;
					}
			}
		}
		vs_storage_release_page(storage, blk);

		/*
		 * Phase 2: only if phase 1 found work, take the EXCLUSIVE write lock
		 * (which WAL-logs the page on commit) and mark the dead entries. The
		 * share lock was dropped above and PG has no atomic lock upgrade, so
		 * the page may have changed; re-derive everything from scratch here
		 * (re-read entry_count, re-test is_dead and the DELETED flag) rather
		 * than trusting phase 1's findings. This makes marking idempotent.
		 */
		if (needs_mark)
		{
			Page					wp	= vs_storage_write_page(storage, blk);
			PrismPostingPageOpaque *wop = prism_posting_opaque(wp);
			char				   *c	= prism_posting_page_content(wp, dim);
			uint32_t				deleted_on_page = 0;
			for (uint32_t i = 0; i < wop->entry_count; i++)
			{
				PrismPostingEntryHeader *h = prism_posting_entry_at(c, i, dim);
				if (h->meta.flags & PRISM_POSTING_FLAG_DELETED)
				{
					deleted_on_page++;
					continue;
				}
				if (is_dead(h->meta.tid, state))
				{
					h->meta.flags |= PRISM_POSTING_FLAG_DELETED;
					total_marked++;
					deleted_on_page++;
				}
			}
			/* Whole page now dead: flag it so the scan skips its scoring. */
			if (wop->entry_count > 0 && deleted_on_page == wop->entry_count)
				wop->flags |= PRISM_POSTING_PAGE_TOMBSTONED;
			vs_storage_commit_page(storage, blk);
		}
		else if (fs_all_dead)
		{
			Page					wp	= vs_storage_write_page(storage, blk);
			PrismPostingPageOpaque *wop = prism_posting_opaque(wp);
			wop->flags |= PRISM_POSTING_PAGE_TOMBSTONED;
			total_marked += n; /* FASTSCAN entries weren't otherwise counted */
			vs_storage_commit_page(storage, blk);
		}

		blk = next;
	}

	/* Keep the head's live_count current (one head write per cluster). It is
	 * stamped at build and maintained by inserts, so it is always meaningful
	 * here; clamp defensively. */
	if (total_marked > 0)
	{
		Page hw = vs_storage_write_page(storage, head_blkno);
		PrismPostingPageOpaque *op = prism_posting_opaque(hw);
		op->live_count			   = (op->live_count >= total_marked)
										   ? op->live_count - total_marked
										   : 0;
		vs_storage_commit_page(storage, head_blkno);
	}

	return total_marked;
}
