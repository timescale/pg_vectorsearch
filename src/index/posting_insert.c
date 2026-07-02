/*
 * posting_insert.c - Runtime insert primitives (see header)
 */

#include "core/memory.h"
#include "index/posting_insert.h"

bool
mkt_posting_insert_one(
		MktStorage		   *storage,
		const RaBitQParams *params,
		Dimension			dim,
		BlockNumber			head_blkno,
		ItemPointerData		tid,
		const float		   *pt_input,
		RaBitQScratch	   *scratch)
{
	if (storage == NULL || params == NULL || pt_input == NULL ||
		scratch == NULL || head_blkno == InvalidBlockNumber)
		return false;

	/*
	 * Phase A: read the head — compute the rotated residual against its
	 * pt_centroid and snapshot the head metadata, then release it (the PG
	 * backing holds only one buffer at a time).
	 */
	Page head = mkt_storage_read_page(storage, head_blkno);
	const MktPostingPageOpaque *hop		   = mkt_posting_opaque(head);
	uint32_t					cluster_id = hop->cluster_id;
	BlockNumber					tail_blkno = hop->tail_blkno;
	uint32_t					live	   = hop->live_count;
	const float				   *pt_cent	   = mkt_posting_pt_centroid(head);

	/* pt_residual = pt_input - pt_centroid (P^T linear). Reuse scratch
	 * (encode_from_pt only touches scratch->xu_cb). */
	float *pt_residual = scratch->transformed;
	for (Dimension i = 0; i < dim; i++)
		pt_residual[i] = pt_input[i] - pt_cent[i];

	mkt_storage_release_page(storage, head_blkno);

	/* Encode relative to the leaf centroid. */
	RaBitQData *rdata = mkt_alloc(MKT_RABITQ_DATA_SIZE(dim));
	mkt_rabitq_encode_from_pt(params, pt_residual, rdata, scratch);
	float f_error =
			mkt_rabitq_derive_f_error(rdata->f_add, rdata->f_rescale, dim);

	/*
	 * tail_blkno / live are read straight from the head: the build stamps both
	 * for every cluster (see mkt_posting_builder_finish and the parallel
	 * leader's finalize), and inserts maintain them, so a built head always
	 * has a valid tail — no chain walk on the insert path.
	 */

	/*
	 * Phase B: append to the tail. Read it first to check for room (released
	 * before any write); the caller's per-cluster lock prevents a concurrent
	 * fill in between.
	 */
	Page tcheck = mkt_storage_read_page(storage, tail_blkno);
	/*
	 * Only append into an AoS tail that has room. A FASTSCAN page packs codes
	 * in immutable 32-vector groups, so an insert can never append to it — it
	 * starts a fresh AoS overflow page instead (the scan merges the mixed
	 * chain by per-page format).
	 */
	bool tail_is_aos = (mkt_posting_opaque(tcheck)->flags &
						MKT_POSTING_PAGE_FASTSCAN) == 0;
	bool has_room	 = tail_is_aos && mkt_posting_page_has_room(tcheck);
	mkt_storage_release_page(storage, tail_blkno);

	BlockNumber new_tail	 = tail_blkno;
	bool		head_updated = false;

	if (has_room)
	{
		Page wp = mkt_storage_write_page(storage, tail_blkno);
		mkt_posting_page_add(
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
			MktPostingPageOpaque *op = mkt_posting_opaque(wp);
			op->tail_blkno			 = new_tail;
			op->live_count			 = live + 1;
			head_updated			 = true;
		}
		mkt_storage_commit_page(storage, tail_blkno);
	}
	else
	{
		/* Tail full: build the new overflow page fully and commit it BEFORE
		 * linking, so a crash never points the chain at an uninitialized
		 * page (it leaves at worst an unreferenced page). */
		BlockNumber t2;
		Page		np = mkt_storage_new_page(storage, &t2);
		mkt_posting_page_init(np, cluster_id, dim, MKT_POSTING_PAGE_OVERFLOW);
		mkt_posting_page_add(
				np,
				dim,
				tid,
				rdata->f_add,
				rdata->f_rescale,
				f_error,
				rdata->bits,
				0);
		mkt_storage_commit_page(storage, t2);

		Page lp = mkt_storage_write_page(storage, tail_blkno);
		mkt_posting_opaque(lp)->next_blkno = t2;
		if (tail_blkno == head_blkno)
		{
			MktPostingPageOpaque *op = mkt_posting_opaque(lp);
			op->tail_blkno			 = t2;
			op->live_count			 = live + 1;
			head_updated			 = true;
		}
		mkt_storage_commit_page(storage, tail_blkno);
		new_tail = t2;
	}

	/* Phase C: update the head metadata when it wasn't folded above. */
	if (!head_updated)
	{
		Page				  hw = mkt_storage_write_page(storage, head_blkno);
		MktPostingPageOpaque *op = mkt_posting_opaque(hw);
		op->tail_blkno			 = new_tail;
		op->live_count			 = live + 1;
		mkt_storage_commit_page(storage, head_blkno);
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
mkt_posting_tombstone_chain(
		MktStorage *storage,
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
		Page						p	 = mkt_storage_read_page(storage, blk);
		const MktPostingPageOpaque *op	 = mkt_posting_opaque(p);
		BlockNumber					next = op->next_blkno;
		uint16_t					flags = op->flags;
		uint32_t					n	  = op->entry_count;

		/* Already fully tombstoned: nothing to mark, and skip so its entries
		 * aren't counted into the live_count decrement twice. */
		if (flags & MKT_POSTING_PAGE_TOMBSTONED)
		{
			mkt_storage_release_page(storage, blk);
			blk = next;
			continue;
		}

		bool  is_fastscan = (flags & MKT_POSTING_PAGE_FASTSCAN) != 0;
		bool  first		  = (flags & MKT_POSTING_PAGE_FIRST) != 0;
		char *content	  = first ? mkt_posting_content_first(p, dim)
								  : mkt_posting_content(p);
		bool  needs_mark  = false; /* AoS: has a not-yet-deleted dead entry */
		bool  fs_all_dead = false; /* FASTSCAN: every entry is dead */

		if (!is_fastscan)
		{
			for (uint32_t i = 0; i < n; i++)
			{
				MktPostingEntryHeader *h =
						mkt_posting_entry_at(content, i, dim);
				if (!(h->meta.flags & MKT_POSTING_FLAG_DELETED) &&
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
			uint32_t ngroups = (n + MKT_FASTSCAN_GROUP - 1) /
							   MKT_FASTSCAN_GROUP;
			for (uint32_t g = 0; g < ngroups && fs_all_dead; g++)
			{
				uint32_t g_count = n - g * MKT_FASTSCAN_GROUP;
				if (g_count > MKT_FASTSCAN_GROUP)
					g_count = MKT_FASTSCAN_GROUP;
				ItemPointerData *tids =
						mkt_fastscan_group_tids(content, g, dim);
				for (uint32_t v = 0; v < g_count; v++)
					if (!is_dead(tids[v], state))
					{
						fs_all_dead = false;
						break;
					}
			}
		}
		mkt_storage_release_page(storage, blk);

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
			Page				  wp  = mkt_storage_write_page(storage, blk);
			MktPostingPageOpaque *wop = mkt_posting_opaque(wp);
			char				 *c	  = (wop->flags & MKT_POSTING_PAGE_FIRST)
											  ? mkt_posting_content_first(wp, dim)
											  : mkt_posting_content(wp);
			uint32_t			  deleted_on_page = 0;
			for (uint32_t i = 0; i < wop->entry_count; i++)
			{
				MktPostingEntryHeader *h = mkt_posting_entry_at(c, i, dim);
				if (h->meta.flags & MKT_POSTING_FLAG_DELETED)
				{
					deleted_on_page++;
					continue;
				}
				if (is_dead(h->meta.tid, state))
				{
					h->meta.flags |= MKT_POSTING_FLAG_DELETED;
					total_marked++;
					deleted_on_page++;
				}
			}
			/* Whole page now dead: flag it so the scan skips its scoring. */
			if (wop->entry_count > 0 && deleted_on_page == wop->entry_count)
				wop->flags |= MKT_POSTING_PAGE_TOMBSTONED;
			mkt_storage_commit_page(storage, blk);
		}
		else if (fs_all_dead)
		{
			Page				  wp  = mkt_storage_write_page(storage, blk);
			MktPostingPageOpaque *wop = mkt_posting_opaque(wp);
			wop->flags |= MKT_POSTING_PAGE_TOMBSTONED;
			total_marked += n; /* FASTSCAN entries weren't otherwise counted */
			mkt_storage_commit_page(storage, blk);
		}

		blk = next;
	}

	/* Keep the head's live_count current (one head write per cluster). It is
	 * stamped at build and maintained by inserts, so it is always meaningful
	 * here; clamp defensively. */
	if (total_marked > 0)
	{
		Page				  hw = mkt_storage_write_page(storage, head_blkno);
		MktPostingPageOpaque *op = mkt_posting_opaque(hw);
		op->live_count			 = (op->live_count >= total_marked)
										 ? op->live_count - total_marked
										 : 0;
		mkt_storage_commit_page(storage, head_blkno);
	}

	return total_marked;
}
