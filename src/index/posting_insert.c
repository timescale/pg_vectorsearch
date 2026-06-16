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
	Page tcheck	  = mkt_storage_read_page(storage, tail_blkno);
	bool has_room = mkt_posting_page_has_room(tcheck);
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
