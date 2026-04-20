/*
 * posting_scan.c - Posting list scan with fused cluster function
 *
 * Scans posting data (paged or flat), computes batch RaBitQ
 * distances, prunes via error bounds, and reranks survivors
 * in a single function call per cluster.
 *
 * Optimizations:
 * - Inline page access via stored base pointer (no vtable dispatch)
 * - Score + prune + rerank fused into one function (no iterator
 *   overhead, compiler can optimize across all phases)
 * - IP count padded to multiple of 4 (avoids tail kernel)
 */

#include <string.h>

#include "core/memory.h"
#include "index/posting_scan.h"

/* ----------------------------------------------------------------
 * Init / cleanup
 * ---------------------------------------------------------------- */

void
mkt_posting_scan_init(
		MktPostingScan	   *scan,
		MktStorage		   *storage,
		char			   *page_base,
		const RaBitQParams *params,
		Dimension			dim,
		uint32_t			max_entries_per_page)
{
	memset(scan, 0, sizeof(*scan));
	scan->storage	   = storage;
	scan->page_base	   = page_base;
	scan->params	   = params;
	scan->dim		   = dim;
	scan->packed_bytes = MKT_RABITQ_BYTES(dim);
	scan->cur_blkno	   = InvalidBlockNumber;

	/* Pre-allocate batch buffers — pad to multiple of 4 for IP kernel */
	uint32_t padded		 = (max_entries_per_page + 3) & ~3u;
	scan->page_distances = mkt_alloc(padded * sizeof(Distance));
	scan->page_scratch	 = mkt_alloc(padded * sizeof(float));
}

void
mkt_posting_scan_cleanup(MktPostingScan *scan)
{
	/* Release pinned page if still held */
	if (scan->cur_page != NULL && scan->storage != NULL &&
		scan->page_base == NULL)
	{
		mkt_storage_release_page(scan->storage, scan->cur_blkno);
		scan->cur_page = NULL;
	}

	mkt_free(scan->page_distances);
	mkt_free(scan->page_scratch);
	scan->page_distances = NULL;
	scan->page_scratch	 = NULL;
}

static bool advance_page(MktPostingScan *scan);

/* ----------------------------------------------------------------
 * Per-cluster begin / end
 * ---------------------------------------------------------------- */

void
mkt_posting_scan_begin_cluster(
		MktPostingScan	 *scan,
		RaBitQQueryState *qstate,
		BlockNumber		  posting_head)
{
	scan->qstate		  = qstate;
	scan->cur_blkno		  = posting_head;
	scan->cur_page		  = NULL;
	scan->cur_content	  = NULL;
	scan->cur_max_entries = 0;
	scan->cur_count		  = 0;
	scan->pages_read	  = 0;
	scan->entries_scanned = 0;
	scan->entries_pruned  = 0;

	/* Eagerly read the first page so pt_centroid is accessible
	 * via mkt_posting_pt_centroid(scan->cur_page) before _cluster
	 * is called. */
	if (posting_head != InvalidBlockNumber)
		advance_page(scan);
}

/*
 * Return pt_centroid from the first page (must be called after
 * begin_cluster). Returns NULL if no page was loaded.
 */
const float *
mkt_posting_scan_pt_centroid(const MktPostingScan *scan)
{
	if (scan->cur_page == NULL)
		return NULL;
	return mkt_posting_pt_centroid(scan->cur_page);
}

void
mkt_posting_scan_begin_flat(
		MktPostingScan *scan, RaBitQQueryState *qstate, char *flat_buf)
{
	MktFlatPostingHeader *hdr = mkt_flat_posting_header(flat_buf);

	scan->qstate		  = qstate;
	scan->cur_blkno		  = InvalidBlockNumber;
	scan->cur_page		  = flat_buf;
	scan->cur_content	  = mkt_flat_posting_content(flat_buf);
	scan->cur_max_entries = hdr->max_entries;
	scan->cur_count		  = hdr->entry_count;
	scan->pages_read	  = 1;
	scan->entries_scanned = 0;
	scan->entries_pruned  = 0;
}

void
mkt_posting_scan_end_cluster(MktPostingScan *scan)
{
	/* Release current page if held via storage vtable */
	if (scan->cur_page != NULL && scan->storage != NULL &&
		scan->page_base == NULL)
	{
		mkt_storage_release_page(scan->storage, scan->cur_blkno);
	}

	scan->cur_page	  = NULL;
	scan->cur_content = NULL;
	scan->qstate	  = NULL;
}

/* ----------------------------------------------------------------
 * Advance to next page in chain
 * ---------------------------------------------------------------- */

static bool
advance_page(MktPostingScan *scan)
{
	/* Follow chain from current page */
	if (scan->cur_page != NULL)
	{
		if (scan->storage != NULL)
		{
			scan->cur_blkno = mkt_posting_opaque(scan->cur_page)->next_blkno;
		}
		else
		{
			/* Flat mode: single page, no chain */
			scan->cur_blkno = InvalidBlockNumber;
		}
		scan->cur_page	  = NULL;
		scan->cur_content = NULL;
	}

	if (scan->cur_blkno == InvalidBlockNumber)
		return false;

	/* Read next page — inline for ArrayPageStorage */
	if (scan->page_base != NULL)
		scan->cur_page = scan->page_base + (size_t)scan->cur_blkno * BLCKSZ;
	else
		scan->cur_page = mkt_storage_read_page(scan->storage, scan->cur_blkno);

	MktPostingPageOpaque *opaque = mkt_posting_opaque(scan->cur_page);
	scan->cur_content =
			(opaque->flags & MKT_POSTING_PAGE_FIRST)
					? mkt_posting_content_first(scan->cur_page, scan->dim)
					: mkt_posting_content(scan->cur_page);
	scan->cur_max_entries = opaque->max_entries;
	scan->cur_count		  = opaque->entry_count;
	scan->pages_read++;
	return true;
}

/* ----------------------------------------------------------------
 * Score + prune a full cluster into topk
 *
 * Inserts survivors with approximate (estimated) distances and
 * error bounds. No exact reranking — the caller does that after
 * extracting candidates from topk.
 * ---------------------------------------------------------------- */

void
mkt_posting_scan_cluster(MktPostingScan *scan, MktTopK *topk)
{
	Dimension dim	 = scan->dim;
	uint32_t  packed = scan->packed_bytes;

	float g_add		 = scan->qstate->g_add;
	float sum_t		 = scan->qstate->sum_transformed;
	float inv_sqrt_d = scan->qstate->inv_sqrt_d;
	float g_error	 = scan->qstate->g_error;

	Distance *distances = scan->page_distances;
	float	 *scratch	= scan->page_scratch;

	/* Process pages until chain is exhausted.
	 * For flat mode: single iteration (one page, no chain). */
	for (;;)
	{
		/* Advance to next page */
		if (scan->cur_page == NULL)
		{
			if (!advance_page(scan))
				break;
		}

		char	*content = scan->cur_content;
		uint32_t max_ent = scan->cur_max_entries;
		uint32_t count	 = scan->cur_count;

		/* --- Score: batch IP + distance conversion --- */
		const float	  *f_add	 = mkt_posting_f_add_at(content, max_ent);
		const float	  *f_rescale = mkt_posting_f_rescale_at(content, max_ent);
		const uint8_t *bits		 = mkt_posting_bits_at(content, max_ent);

		uint32_t padded = (count + 3) & ~3u;
		mkt_rabitq_inner_product_multi(
				scan->qstate->transformed, bits, packed, dim, padded, scratch);

		for (uint32_t i = 0; i < count; i++)
		{
			float final_dot = (2.0f * scratch[i] - sum_t) * inv_sqrt_d;
			distances[i] = f_add[i] + g_add - 2.0f * f_rescale[i] * final_dot;
		}

		/* --- Prune + insert approximate distances --- */
		const float *f_error = mkt_posting_f_error_at(content, max_ent);
		const MktPostingEntryMeta *metas	 = mkt_posting_metas_at(content);
		Distance				   threshold = mkt_topk_threshold(topk);

		for (uint32_t i = 0; i < count; i++)
		{
			scan->entries_scanned++;

			if (metas[i].flags & MKT_POSTING_FLAG_DELETED)
			{
				scan->entries_pruned++;
				continue;
			}

			Distance est = distances[i];
			Distance err = f_error[i] * g_error;
			Distance lb	 = est - err;

			if (lb >= threshold)
			{
				scan->entries_pruned++;
				continue;
			}

			uint64_t id = mkt_posting_encode_tid(&metas[i].tid);
			mkt_topk_insert(topk, est, err, id);
			threshold = mkt_topk_threshold(topk);
		}

		/* Follow chain: read next_blkno, then release current */
		BlockNumber prev_blkno = scan->cur_blkno;

		if (scan->storage != NULL)
			scan->cur_blkno = mkt_posting_opaque(scan->cur_page)->next_blkno;
		else
			scan->cur_blkno = InvalidBlockNumber;

		if (scan->storage != NULL && scan->page_base == NULL)
			mkt_storage_release_page(scan->storage, prev_blkno);

		scan->cur_page	  = NULL;
		scan->cur_content = NULL;
	}
}
