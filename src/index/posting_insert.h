/*
 * posting_insert.h - Runtime (post-build) insert primitives
 *
 * Shared between the PostgreSQL index AM (aminsert) and the standalone build
 * so the route/encode/append logic is exercised by unit tests without
 * PostgreSQL.
 *
 * All operations go through the VsStorage vtable one page at a time (the PG
 * backing holds a single buffer), so a caller MUST serialize concurrent
 * inserts to the SAME cluster externally — the PG glue takes a per-cluster
 * lock; single-threaded callers (standalone) need none.
 */

#ifndef PRISM_POSTING_INSERT_H
#define PRISM_POSTING_INSERT_H

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
 * cluster head's stored pt_centroid via vs_rabitq_encode_from_pt (so no raw
 * centroid is needed) and appended as an AoS entry to the chain tail, growing
 * a new overflow page when the tail is full. On FASTSCAN indexes the appended
 * page is still plain AoS (packed groups can't take an in-place append); the
 * scan already merges mixed chains by per-page format.
 *
 * The head's tail_blkno / live_count are filled lazily on first use (one chain
 * walk when tail_blkno == InvalidBlockNumber) and maintained thereafter.
 *
 * scratch is used for the encode (xu_cb) and to hold the rotated residual.
 *
 * If the head has been retired (TOMBSTONED or DELETED) — e.g. split away by a
 * concurrent rebalance between routing and this call — nothing is inserted;
 * the caller should re-route to the current head. This is reported via
 * *head_retired (may be NULL), detected during the head read this does anyway,
 * so the caller needn't read the head a second time to check.
 *
 * Returns true on success.
 */
bool prism_posting_insert_one(
		VsStorage		   *storage,
		const RaBitQParams *params,
		Dimension			dim,
		BlockNumber			head_blkno,
		ItemPointerData		tid,
		const float		   *pt_input,
		RaBitQScratch	   *scratch,
		bool				unreachable,
		bool			   *head_retired);

/*
 * Tombstone every AoS entry in the cluster chain whose TID is_dead() reports
 * dead by setting PRISM_POSTING_FLAG_DELETED, so later scans skip it. FASTSCAN
 * base pages are left untouched: their packed groups cannot be edited in
 * place, so their dead entries stay correct via the executor's MVCC visibility
 * recheck and are physically reclaimed only at compaction/rebuild. The head's
 * live_count (stamped at build, maintained by inserts) is decremented by the
 * number newly marked. Returns the count marked.
 *
 * Shared between the PG ambulkdelete (VACUUM) path and the standalone build so
 * the tombstone logic is unit-tested without PostgreSQL. The caller must
 * serialize against concurrent inserts to the same cluster (the PG glue holds
 * the per-cluster page lock).
 */
uint32_t prism_posting_tombstone_chain(
		VsStorage  *storage,
		Dimension	dim,
		BlockNumber head_blkno,
		bool (*is_dead)(ItemPointerData tid, void *state),
		void *state);

#endif /* PRISM_POSTING_INSERT_H */
