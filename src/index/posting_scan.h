/*
 * posting_scan.h - Volcano-model iterator for posting list scan
 *
 * Pull-based iterator that walks posting page chains with two-stage
 * RaBitQ filtering. Matches PG's pull-based execution model: the
 * scan yields one result at a time, with top-K as a separate operator
 * above.
 *
 * Per-cluster granularity: each MktPostingScan instance is fully
 * independent with no shared mutable state. Multiple workers can
 * each own a scan instance, scanning separate clusters in parallel
 * and merging results into a shared top-K.
 *
 * Usage:
 *   MktPostingScan scan;
 *   mkt_posting_scan_init(&scan, storage, params, dim);
 *   mkt_posting_scan_set_threshold(&scan, &topk_threshold);
 *
 *   for each cluster:
 *       mkt_posting_scan_begin_cluster(&scan, query, centroid, head);
 *       MktPostingScanResult result;
 *       while (mkt_posting_scan_next(&scan, &result))
 *           topk_insert(..., result.distance, result.error, ...);
 *       mkt_posting_scan_end_cluster(&scan);
 *
 *   mkt_posting_scan_cleanup(&scan);
 */

#ifndef MKT_POSTING_SCAN_H
#define MKT_POSTING_SCAN_H

#include <stdint.h>

#include "index/posting_page.h"
#include "index/storage.h"
#include "mkt_types.h"
#include "quant/rabitq.h"

/* ----------------------------------------------------------------
 * Scan result (returned from next())
 * ---------------------------------------------------------------- */
typedef struct MktPostingScanResult
{
	ItemPointerData tid;
	Distance		distance; /* estimated distance */
	Distance		error;	  /* symmetric error margin */
} MktPostingScanResult;

/* ----------------------------------------------------------------
 * Scan state — volcano-model iterator
 *
 * No shared mutable state: each instance is fully independent.
 * ---------------------------------------------------------------- */
typedef struct MktPostingScan
{
	/* Set once at init */
	MktStorage		   *storage;
	const RaBitQParams *params;
	Dimension			dim;

	/* Push-down filter: scan skips entries where
	 * lower_bound >= *threshold. NULL = no pruning.
	 * Caller owns the pointee and updates it. */
	const Distance *threshold;

	/* Per-cluster state (set by begin_cluster) */
	RaBitQQueryState *qstate;	 /* prepared for current cluster */
	BlockNumber		  cur_blkno; /* current page in chain */
	Page			  cur_page;	 /* pinned+locked current page */
	uint16_t		  cur_entry; /* position on current page */
	uint16_t		  cur_count; /* entries on current page */

	/* Batch scoring buffer for symmetric mode.
	 * When non-NULL, all entries on a page are scored at once and
	 * results yielded from the buffer. */
	Distance *batch_distances;	  /* [max_entries_per_page] */
	Distance *batch_lower_bounds; /* [max_entries_per_page] */
	uint32_t *batch_hamming;	  /* [max_entries_per_page] scratch */
	uint16_t  batch_count;		  /* entries scored in current batch */
	uint16_t  batch_pos;		  /* next entry to yield from batch */

	/* Scan counters (accumulated across clusters) */
	uint32_t pages_read;
	uint32_t entries_scanned;
	uint32_t entries_pruned;
} MktPostingScan;

/* ----------------------------------------------------------------
 * Iterator API
 * ---------------------------------------------------------------- */

/*
 * Init: set storage, params, dim. No iteration state yet.
 */
void mkt_posting_scan_init(
		MktPostingScan	   *scan,
		MktStorage		   *storage,
		const RaBitQParams *params,
		Dimension			dim);

/*
 * Begin scanning one cluster's posting list. Prepares per-cluster
 * qstate from (query, centroid).
 */
void mkt_posting_scan_begin_cluster(
		MktPostingScan *scan,
		VectorRef		query,
		VectorRef		centroid,
		BlockNumber		posting_head);

/*
 * Pull next surviving entry from current cluster. Returns true
 * if a result was produced, false when chain exhausted.
 */
bool mkt_posting_scan_next(MktPostingScan *scan, MktPostingScanResult *result);

/*
 * End current cluster. Frees qstate.
 */
void mkt_posting_scan_end_cluster(MktPostingScan *scan);

/*
 * Set push-down threshold pointer. Caller updates *threshold
 * (e.g., from top-K operator); scan reads it per entry.
 * NULL means no pruning.
 */
void mkt_posting_scan_set_threshold(
		MktPostingScan *scan, const Distance *threshold);

/*
 * Cleanup: release any remaining resources.
 */
void mkt_posting_scan_cleanup(MktPostingScan *scan);

/* ----------------------------------------------------------------
 * Write API (build path)
 * ---------------------------------------------------------------- */

#endif /* MKT_POSTING_SCAN_H */
