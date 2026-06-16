/*
 * posting_insert.h - Runtime (post-build) insert + delete primitives
 *
 * Shared between the PostgreSQL index AM (aminsert / ambulkdelete) and the
 * standalone build so the route/encode/append/tombstone logic is exercised
 * by unit tests without PostgreSQL.
 *
 * All operations go through the MktStorage vtable one page at a time (the PG
 * backing holds a single buffer), so a caller MUST serialize concurrent
 * inserts to the SAME cluster externally — the PG glue takes a per-cluster
 * lock; single-threaded callers (standalone) need none.
 */

#ifndef MKT_POSTING_INSERT_H
#define MKT_POSTING_INSERT_H

#include <stdbool.h>
#include <stdint.h>

#include "index/posting_page.h"
#include "index/storage.h"
#include "quant/rabitq.h"

/*
 * Append one vector to a cluster's posting chain.
 *
 * pt_input is P^T * v — the inserted vector already rotated by the caller (the
 * same rotation build/scan apply). The vector is encoded relative to the
 * cluster head's stored pt_centroid via mkt_rabitq_encode_from_pt (so no raw
 * centroid is needed) and appended as an AoS entry to the chain tail, growing
 * a new overflow page when the tail is full. On FASTSCAN indexes the appended
 * page is still plain AoS (packed groups can't take an in-place append); the
 * scan already merges mixed chains by per-page format.
 *
 * The head's tail_blkno / live_count are filled lazily on first use (one chain
 * walk when tail_blkno == InvalidBlockNumber) and maintained thereafter.
 *
 * scratch is used for the encode (xu_cb) and to hold the rotated residual.
 * Returns true on success.
 */
bool mkt_posting_insert_one(
		MktStorage		   *storage,
		const RaBitQParams *params,
		Dimension			dim,
		BlockNumber			head_blkno,
		ItemPointerData		tid,
		const float		   *pt_input,
		RaBitQScratch	   *scratch);

/*
 * Mark every AoS entry in the cluster chain whose TID is_dead() reports dead
 * with MKT_POSTING_FLAG_DELETED (the scan skips such entries). FASTSCAN base
 * pages are left untouched — their dead entries stay correct via MVCC
 * visibility recheck and are physically reclaimed at compaction. Decrements
 * the head live_count by the number newly marked. Returns the count marked.
 */
uint32_t mkt_posting_tombstone_chain(
		MktStorage *storage,
		Dimension	dim,
		BlockNumber head_blkno,
		bool (*is_dead)(ItemPointerData tid, void *state),
		void *state);

/*
 * Count live (non-deleted) entries across the chain. AoS pages skip
 * DELETED-flagged entries; FASTSCAN pages contribute their full entry_count
 * (their dead entries aren't marked — see decision above). Used to lazily
 * initialize the head live_count and as a test/validation helper.
 */
uint32_t mkt_posting_chain_count(
		MktStorage *storage, Dimension dim, BlockNumber head_blkno);

#endif /* MKT_POSTING_INSERT_H */
