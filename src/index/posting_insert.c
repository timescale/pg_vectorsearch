/*
 * posting_insert.c - Runtime insert + delete primitives (see header)
 */

#include "core/memory.h"
#include "index/posting_insert.h"

/*
 * Walk the cluster chain once: report the tail block and the live
 * (non-deleted) entry count. Used to lazily fill the head metadata and by
 * mkt_posting_chain_count.
 */
static void
walk_chain(
		MktStorage	*storage,
		Dimension	 dim,
		BlockNumber	 head_blkno,
		BlockNumber *tail_out,
		uint32_t	*live_out)
{
	BlockNumber blk	 = head_blkno;
	BlockNumber tail = head_blkno;
	uint32_t	live = 0;

	while (blk != InvalidBlockNumber)
	{
		Page						p	 = mkt_storage_read_page(storage, blk);
		const MktPostingPageOpaque *op	 = mkt_posting_opaque(p);
		BlockNumber					next = op->next_blkno;
		uint32_t					n	 = op->entry_count;

		tail = blk;

		if (op->flags & MKT_POSTING_PAGE_FASTSCAN)
		{
			/* FASTSCAN dead entries aren't marked; count them all live. */
			live += n;
		}
		else
		{
			char *content = (op->flags & MKT_POSTING_PAGE_FIRST)
								  ? mkt_posting_content_first(p, dim)
								  : mkt_posting_content(p);
			for (uint32_t i = 0; i < n; i++)
			{
				MktPostingEntryHeader *h =
						mkt_posting_entry_at(content, i, dim);
				if (!(h->meta.flags & MKT_POSTING_FLAG_DELETED))
					live++;
			}
		}

		mkt_storage_release_page(storage, blk);
		blk = next;
	}

	*tail_out = tail;
	*live_out = live;
}

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

	/* Lazily resolve the tail + live baseline on first use. */
	if (tail_blkno == InvalidBlockNumber)
		walk_chain(storage, dim, head_blkno, &tail_blkno, &live);

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
		/* Read-scan first so clean pages aren't dirtied/WAL-logged. */
		Page						p	 = mkt_storage_read_page(storage, blk);
		const MktPostingPageOpaque *op	 = mkt_posting_opaque(p);
		BlockNumber					next = op->next_blkno;
		bool	 is_fastscan = (op->flags & MKT_POSTING_PAGE_FASTSCAN) != 0;
		bool	 first		 = (op->flags & MKT_POSTING_PAGE_FIRST) != 0;
		uint32_t n			 = op->entry_count;
		bool	 needs_mark	 = false;

		if (!is_fastscan)
		{
			char *content = first ? mkt_posting_content_first(p, dim)
								  : mkt_posting_content(p);
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
		mkt_storage_release_page(storage, blk);

		if (needs_mark)
		{
			Page				  wp  = mkt_storage_write_page(storage, blk);
			MktPostingPageOpaque *wop = mkt_posting_opaque(wp);
			char *content			  = (wop->flags & MKT_POSTING_PAGE_FIRST)
											  ? mkt_posting_content_first(wp, dim)
											  : mkt_posting_content(wp);
			for (uint32_t i = 0; i < wop->entry_count; i++)
			{
				MktPostingEntryHeader *h =
						mkt_posting_entry_at(content, i, dim);
				if (!(h->meta.flags & MKT_POSTING_FLAG_DELETED) &&
					is_dead(h->meta.tid, state))
				{
					h->meta.flags |= MKT_POSTING_FLAG_DELETED;
					total_marked++;
				}
			}
			mkt_storage_commit_page(storage, blk);
		}

		blk = next;
	}

	/* Maintain the head live_count (only when it has been computed; otherwise
	 * a later first insert recomputes it). One head write per cluster.
	 * Read-check the sentinel first so an uncomputed head isn't dirtied. */
	if (total_marked > 0)
	{
		Page rp		  = mkt_storage_read_page(storage, head_blkno);
		bool computed = mkt_posting_opaque(rp)->tail_blkno !=
						InvalidBlockNumber;
		mkt_storage_release_page(storage, head_blkno);

		if (computed)
		{
			Page hw = mkt_storage_write_page(storage, head_blkno);
			MktPostingPageOpaque *op = mkt_posting_opaque(hw);
			op->live_count			 = (op->live_count >= total_marked)
											 ? op->live_count - total_marked
											 : 0;
			mkt_storage_commit_page(storage, head_blkno);
		}
	}

	return total_marked;
}

uint32_t
mkt_posting_chain_count(
		MktStorage *storage, Dimension dim, BlockNumber head_blkno)
{
	if (storage == NULL || head_blkno == InvalidBlockNumber)
		return 0;

	BlockNumber tail;
	uint32_t	live;
	walk_chain(storage, dim, head_blkno, &tail, &live);
	return live;
}
