/*
 * posting_insert.h - Runtime (post-build) insert primitives
 *
 * Shared between the PostgreSQL index AM (aminsert) and the standalone build
 * so the route/encode/append logic is exercised by unit tests without
 * PostgreSQL.
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
 * split_threshold: when > 0 and the post-insert live_count reaches it, the
 * head is flagged MKT_POSTING_PAGE_NEEDS_SPLIT (advisory; consumed by the
 * split maintenance paths). 0 disables the check with zero extra I/O.
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
		RaBitQScratch	   *scratch,
		bool				unreachable,
		uint32_t			split_threshold);

/*
 * Tombstone every AoS entry in the cluster chain whose TID is_dead() reports
 * dead by setting MKT_POSTING_FLAG_DELETED, so later scans skip it. FASTSCAN
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
uint32_t mkt_posting_tombstone_chain(
		MktStorage *storage,
		Dimension	dim,
		BlockNumber head_blkno,
		bool (*is_dead)(ItemPointerData tid, void *state),
		void *state);

#endif /* MKT_POSTING_INSERT_H */
