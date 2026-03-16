/*
 * posting_scan.c - Volcano-model iterator for posting list scan
 *
 * Pull-based iterator that walks posting page chains with two-stage
 * RaBitQ filtering. Each next() call yields at most one result.
 *
 * Write path fills posting pages to capacity and chains them,
 * returning the head block number for the centroid leaf.
 */

#include <assert.h>
#include <math.h>

#include "index/posting_scan.h"

/* ----------------------------------------------------------------
 * Init / cleanup
 * ---------------------------------------------------------------- */
void
mkt_posting_scan_init(
		MktPostingScan	   *scan,
		MktStorage		   *storage,
		const RaBitQParams *params,
		Dimension			dim)
{
	scan->storage	= storage;
	scan->params	= params;
	scan->dim		= dim;
	scan->threshold = NULL;
	scan->qstate	= NULL;
	scan->cur_blkno = InvalidBlockNumber;
	scan->cur_page	= NULL;
	scan->cur_entry = 0;
	scan->cur_count = 0;

	scan->pages_read	  = 0;
	scan->entries_scanned = 0;
	scan->entries_pruned  = 0;
}

void
mkt_posting_scan_set_threshold(MktPostingScan *scan, const Distance *threshold)
{
	scan->threshold = threshold;
}

void
mkt_posting_scan_cleanup(MktPostingScan *scan)
{
	/* Release pinned page if still held */
	if (scan->cur_page != NULL)
	{
		mkt_storage_release_page(scan->storage, scan->cur_blkno);
		scan->cur_page = NULL;
	}

	if (scan->qstate != NULL)
	{
		mkt_rabitq_free_query(scan->qstate);
		scan->qstate = NULL;
	}
	scan->cur_blkno = InvalidBlockNumber;
}

/* ----------------------------------------------------------------
 * Per-cluster iteration
 * ---------------------------------------------------------------- */
void
mkt_posting_scan_begin_cluster(
		MktPostingScan *scan,
		VectorRef		query,
		VectorRef		centroid,
		BlockNumber		posting_head)
{
	/* Prepare RaBitQ query state for this cluster's centroid */
	scan->qstate = mkt_rabitq_prepare_query(scan->params, query, centroid);

	scan->cur_blkno = posting_head;
	scan->cur_page	= NULL;
	scan->cur_entry = 0;
	scan->cur_count = 0;

	/* Pin+lock the first page and keep it for iteration */
	if (posting_head != InvalidBlockNumber)
	{
		Page page = mkt_storage_read_page(scan->storage, posting_head);
		const MktPostingPageOpaque *opaque = MKT_POSTING_OPAQUE(page);
		assert(opaque->page_id == MKT_POSTING_PAGE_ID);
		scan->cur_page	= page;
		scan->cur_count = opaque->entry_count;
		scan->pages_read++;
	}
}

bool
mkt_posting_scan_next(MktPostingScan *scan, MktPostingScanResult *result)
{
	Dimension dim = scan->dim;

	for (;;)
	{
		/* Advance to next page if current is exhausted */
		while (scan->cur_entry >= scan->cur_count)
		{
			if (scan->cur_page == NULL)
				return false;

			/* Follow chain: read next_blkno, release current */
			const MktPostingPageOpaque *opaque = MKT_POSTING_OPAQUE(
					scan->cur_page);
			BlockNumber next = opaque->next_blkno;
			mkt_storage_release_page(scan->storage, scan->cur_blkno);
			scan->cur_page = NULL;

			scan->cur_blkno = next;
			scan->cur_entry = 0;
			scan->cur_count = 0;

			if (next != InvalidBlockNumber)
			{
				Page np = mkt_storage_read_page(scan->storage, next);
				const MktPostingPageOpaque *nop = MKT_POSTING_OPAQUE(np);
				assert(nop->page_id == MKT_POSTING_PAGE_ID);
				scan->cur_page	= np;
				scan->cur_count = nop->entry_count;
				scan->pages_read++;
			}
		}

		if (scan->cur_page == NULL)
			return false;

		/* Process current entry from the pinned page */
		uint16_t i = scan->cur_entry++;

		/* Check metadata for deleted entries */
		const MktPostingEntryMeta *meta = mkt_posting_meta(scan->cur_page, i);
		if (meta->flags & MKT_POSTING_FLAG_DELETED)
			continue;

		/* Two-stage RaBitQ filtering */
		const RaBitQData *data = mkt_posting_data(scan->cur_page, i, dim);
		Distance		  est_dist;
		Distance		  lower_bound;
		mkt_rabitq_distance_with_bound(
				scan->qstate, data, dim, &est_dist, &lower_bound);

		scan->entries_scanned++;

		/* Threshold pruning */
		if (scan->threshold != NULL && lower_bound >= *scan->threshold)
		{
			scan->entries_pruned++;
			continue;
		}

		/* Populate result */
		result->tid		 = meta->tid;
		result->distance = est_dist;
		result->error	 = est_dist - lower_bound;

		return true;
	}
}

void
mkt_posting_scan_end_cluster(MktPostingScan *scan)
{
	/* Release pinned page if still held */
	if (scan->cur_page != NULL)
	{
		mkt_storage_release_page(scan->storage, scan->cur_blkno);
		scan->cur_page = NULL;
	}

	if (scan->qstate != NULL)
	{
		mkt_rabitq_free_query(scan->qstate);
		scan->qstate = NULL;
	}
	scan->cur_blkno = InvalidBlockNumber;
	scan->cur_entry = 0;
	scan->cur_count = 0;
}

/* ----------------------------------------------------------------
 * Write a posting list
 * ---------------------------------------------------------------- */
BlockNumber
mkt_posting_write_list(
		MktStorage				   *storage,
		Dimension					dim,
		uint32_t					cluster_id,
		const MktPostingWriteEntry *entries,
		uint32_t					count)
{
	if (count == 0)
		return InvalidBlockNumber;

	BlockNumber head_blkno = InvalidBlockNumber;
	BlockNumber prev_blkno = InvalidBlockNumber;
	uint32_t	idx		   = 0;

	while (idx < count)
	{
		BlockNumber blkno;
		Page		page = mkt_storage_new_page(storage, &blkno);

		uint16_t flags = 0;
		if (head_blkno == InvalidBlockNumber)
		{
			flags	   = MKT_POSTING_PAGE_FIRST;
			head_blkno = blkno;
		}
		else
		{
			flags = MKT_POSTING_PAGE_OVERFLOW;
		}

		mkt_posting_page_init(page, cluster_id, flags);

		/* Fill page to capacity */
		while (idx < count && mkt_posting_page_add(
									  page,
									  dim,
									  entries[idx].block,
									  entries[idx].offset,
									  entries[idx].data,
									  0))
		{
			idx++;
		}

		/* Link previous page to this one */
		if (prev_blkno != InvalidBlockNumber)
		{
			Page prev = mkt_storage_write_page(storage, prev_blkno);
			MktPostingPageOpaque *prev_opaque = MKT_POSTING_OPAQUE(prev);
			prev_opaque->next_blkno			  = blkno;
			mkt_storage_commit_page(storage, prev_blkno);
		}

		mkt_storage_commit_page(storage, blkno);
		prev_blkno = blkno;
	}

	return head_blkno;
}
