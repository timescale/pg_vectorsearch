/*
 * posting_scan.h - Posting list scan with RaBitQ scoring and pruning
 *
 * Scans posting data (paged or flat), computes batch RaBitQ
 * distances, prunes via error bounds, and inserts survivors into
 * a MktTopK with approximate distances. Reranking with exact
 * distances is the caller's responsibility.
 *
 * Usage (paged mode):
 *   MktPostingScan scan;
 *   mkt_posting_scan_init(&scan, storage, page_base, params, dim,
 *                         mkt_posting_max_entries(dim));
 *   mkt_posting_scan_begin_cluster(&scan, qstate, posting_head);
 *   mkt_posting_scan_cluster(&scan, &topk);
 *   mkt_posting_scan_end_cluster(&scan);
 *
 * Usage (flat mode):
 *   mkt_posting_scan_begin_flat(&scan, qstate, flat_buf);
 *   mkt_posting_scan_cluster(&scan, &topk);
 *   mkt_posting_scan_end_cluster(&scan);
 *
 *   mkt_posting_scan_cleanup(&scan);
 */

#ifndef MKT_POSTING_SCAN_H
#define MKT_POSTING_SCAN_H

#include "algo/topk.h"
#include "index/posting_page.h"
#include "index/storage.h"
#include "quant/rabitq.h"

/* ----------------------------------------------------------------
 * Scan state
 * ---------------------------------------------------------------- */

typedef struct MktPostingScan
{
	/* Configuration (set at init, constant across clusters) */
	MktStorage		   *storage;   /* NULL for flat mode */
	char			   *page_base; /* direct page pointer (bypass vtable) */
	const RaBitQParams *params;
	Dimension			dim;
	uint32_t			packed_bytes; /* MKT_RABITQ_BYTES(dim) */

	/* Per-cluster state (set at begin_cluster/begin_flat) */
	RaBitQQueryState *qstate;

	/* Page iteration — paged mode uses cur_blkno/cur_page,
	 * flat mode sets cur_page to the flat buffer */
	BlockNumber cur_blkno;
	Page		cur_page;  /* current page buffer (from storage or flat) */
	uint32_t	cur_count; /* entries on current page */

	/* Content-based access (header-agnostic) */
	char	*cur_content;	  /* start of SoA data for current page */
	uint32_t cur_max_entries; /* capacity of current page */

	/* Per-page batch results */
	Distance *page_distances; /* [max_entries_per_page] */
	float	 *page_scratch;	  /* [max_entries_per_page] for IP scratch */

	/* TID dedup for replicated vectors (NULL = no dedup).
	 * Open-addressing hash with generation counter — no memset
	 * needed per query, just bump seen_gen. */
	uint64_t *seen_tids;	 /* hash set: TID per slot */
	uint32_t *seen_gens;	 /* generation per slot */
	uint32_t  seen_tids_cap; /* must be power of 2 */
	uint32_t  seen_gen;		 /* current generation (bumped per query) */

	/* Stats */
	uint32_t pages_read;
	uint32_t entries_scanned;
	uint32_t entries_pruned;
} MktPostingScan;

/* ----------------------------------------------------------------
 * API
 * ---------------------------------------------------------------- */

/*
 * Initialize scan. Allocates scratch buffers sized for
 * max_entries_per_page.
 *
 * page_base: direct pointer to page array for inline access
 *            (bypasses storage vtable). NULL to use storage vtable.
 */
void mkt_posting_scan_init(
		MktPostingScan	   *scan,
		MktStorage		   *storage,
		char			   *page_base,
		const RaBitQParams *params,
		Dimension			dim,
		uint32_t			max_entries_per_page);

/*
 * Begin scanning a cluster's posting list (paged mode).
 * qstate must be valid for the duration of the scan.
 */
void mkt_posting_scan_begin_cluster(
		MktPostingScan	 *scan,
		RaBitQQueryState *qstate,
		BlockNumber		  posting_head);

/*
 * Begin scanning a flat posting page.
 */
void mkt_posting_scan_begin_flat(
		MktPostingScan *scan, RaBitQQueryState *qstate, char *flat_buf);

/*
 * End cluster scan. Releases the current page (paged mode).
 */
void mkt_posting_scan_end_cluster(MktPostingScan *scan);

/*
 * Return P^T * centroid from the first posting page.
 * Call after begin_cluster. Returns NULL if no page loaded.
 */
const float *mkt_posting_scan_pt_centroid(const MktPostingScan *scan);

/*
 * Score and prune a full cluster.
 *
 * Processes all pages (or the flat buffer). Survivors are inserted
 * into topk with approximate (estimated) distances and error bounds
 * via mkt_posting_encode_tid(). The caller extracts candidates
 * afterward and reranks with exact distances.
 *
 * The threshold updates progressively as better candidates are
 * found, providing dynamic pruning within the cluster.
 */
void mkt_posting_scan_cluster(MktPostingScan *scan, MktTopK *topk);

/*
 * Free scratch buffers.
 */
void mkt_posting_scan_cleanup(MktPostingScan *scan);

#endif /* MKT_POSTING_SCAN_H */
