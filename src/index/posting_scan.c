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

	/* Allocate batch buffers for symmetric mode */
	uint32_t max_per_page	 = mkt_posting_max_entries(dim);
	scan->batch_distances	 = mkt_alloc(max_per_page * sizeof(Distance));
	scan->batch_lower_bounds = mkt_alloc(max_per_page * sizeof(Distance));
	scan->batch_hamming		 = mkt_alloc(max_per_page * sizeof(uint32_t));
	scan->batch_count		 = 0;
	scan->batch_pos			 = 0;

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

	if (scan->batch_distances != NULL)
	{
		mkt_free(scan->batch_distances);
		mkt_free(scan->batch_lower_bounds);
		mkt_free(scan->batch_hamming);
		scan->batch_distances	 = NULL;
		scan->batch_lower_bounds = NULL;
		scan->batch_hamming		 = NULL;
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

/*
 * Score all entries on the current page in batch (symmetric mode).
 * Computes hamming distances for all entries at once, then converts
 * to estimated distances and lower bounds.
 */
static void
score_page_batch(MktPostingScan *scan)
{
	Dimension dim		   = scan->dim;
	uint16_t  count		   = scan->cur_count;
	uint32_t  packed_bytes = MKT_RABITQ_BYTES(dim);

	/* Batch hamming: data entries grow backward on the page, so
	 * the last entry (count-1) is at the lowest address. Start
	 * there and stride forward by data_size. Results are in
	 * reverse page order (batch_hamming[0] = page entry count-1). */
	uint32_t		  data_size = MKT_RABITQ_DATA_SIZE(dim);
	const RaBitQData *last = mkt_posting_data(scan->cur_page, count - 1, dim);
	mkt_rabitq_hamming_distance_multi(
			scan->qstate->query_bits,
			last->bits,
			data_size,
			packed_bytes,
			count,
			scan->batch_hamming);

	/* Convert hamming to distances + lower bounds */
	float g_add		 = scan->qstate->g_add;
	float g_scale	 = scan->qstate->g_scale;
	float inv_sqrt_d = scan->qstate->inv_sqrt_d;
	float g_error	 = scan->qstate->g_error;
	float c_error	 = scan->qstate->c_error;
	float mult		 = scan->qstate->error_multiplier;

	for (uint16_t i = 0; i < count; i++)
	{
		/* batch_hamming is in reverse order (lowest-address first),
		 * so batch_hamming[k] corresponds to page entry count-1-k. */
		uint16_t		  page_idx = count - 1 - i;
		const RaBitQData *data =
				mkt_posting_data(scan->cur_page, page_idx, dim);

		float sym_dot	= (float)((int32_t)dim -
								  2 * (int32_t)scan->batch_hamming[i]);
		float final_dot = sym_dot * inv_sqrt_d;
		float est_dist	= data->f_add + g_add -
						 2.0f * data->f_rescale * g_scale * final_dot;

		/* Derive f_error from f_add and f_rescale */
		float f_rsq	  = data->f_rescale * data->f_rescale;
		float f_error = (f_rsq > data->f_add && dim > 1)
							  ? c_error * sqrtf(f_rsq - data->f_add)
							  : 2e-4f * sqrtf(data->f_add);

		float err_margin = mult * f_error * g_error;
		float fp_margin	 = 1e-5f * fabsf(est_dist);

		scan->batch_distances[page_idx]	   = est_dist;
		scan->batch_lower_bounds[page_idx] = est_dist - err_margin - fp_margin;
	}

	scan->batch_count = count;
	scan->batch_pos	  = 0;
}

bool
mkt_posting_scan_next(MktPostingScan *scan, MktPostingScanResult *result)
{
	Dimension dim = scan->dim;
	bool	  use_batch =
			(scan->qstate != NULL &&
			 scan->qstate->mode == MKT_DISTANCE_MODE_SYMMETRIC);

	for (;;)
	{
		/* Yield from batch buffer if available */
		if (use_batch && scan->batch_pos < scan->batch_count)
		{
			uint16_t i = scan->batch_pos++;

			const MktPostingEntryMeta *meta =
					mkt_posting_meta(scan->cur_page, i);
			if (meta->flags & MKT_POSTING_FLAG_DELETED)
				continue;

			scan->entries_scanned++;

			Distance lower_bound = scan->batch_lower_bounds[i];
			if (scan->threshold != NULL && lower_bound >= *scan->threshold)
			{
				scan->entries_pruned++;
				continue;
			}

			result->tid		 = meta->tid;
			result->distance = scan->batch_distances[i];
			result->error	 = scan->batch_distances[i] - lower_bound;
			return true;
		}

		/* Reset batch when exhausted */
		scan->batch_count = 0;
		scan->batch_pos	  = 0;

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

		/* Batch score entire page in symmetric mode */
		if (use_batch)
		{
			score_page_batch(scan);
			scan->cur_entry = scan->cur_count; /* mark page consumed */
			continue; /* yield from batch buffer on next iteration */
		}

		/* Per-entry path (asymmetric / non-batched) */
		uint16_t i = scan->cur_entry++;

		const MktPostingEntryMeta *meta = mkt_posting_meta(scan->cur_page, i);
		if (meta->flags & MKT_POSTING_FLAG_DELETED)
			continue;

		const RaBitQData *data = mkt_posting_data(scan->cur_page, i, dim);
		Distance		  est_dist;
		Distance		  lower_bound;
		mkt_rabitq_distance_with_bound(
				scan->qstate, data, dim, &est_dist, &lower_bound);

		scan->entries_scanned++;

		if (scan->threshold != NULL && lower_bound >= *scan->threshold)
		{
			scan->entries_pruned++;
			continue;
		}

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
