/*
 * query_parallel.h - Backend-neutral core for parallel query scans
 *
 * A parallel query runs one-time upfront coordination -- some
 * participant produces the probe list ("the plan") -- after which every
 * participant executes independently, as if running its own query over
 * the clusters it claims, meeting again only at the backend's merge
 * (PostgreSQL: Gather Merge over per-worker sorted streams; standalone:
 * a leader merge into one top-k).
 *
 * Shared state is deliberately minimal: read-only data (the published
 * probe list, sizing constants), and exactly two non-blocking
 * primitives -- an atomic work cursor and the SOAR claim table. No
 * participant ever waits on another during execution.
 *
 * The cursor hands out batches of probe ranks (ramp-down chunking) and
 * is preferred over a fixed upfront partition for correctness: under
 * PostgreSQL, worker launch can fall short of plan and the AM cannot
 * observe it, so a fixed slice of a never-launched worker would
 * silently go unscanned. The cursor drains completely for any subset
 * of participants that shows up.
 *
 * The claim table exists because SOAR replicates entries across
 * clusters: replicas of one TID can be scanned by different
 * participants, and per-participant dedup cannot see the pair. A
 * participant claims a TID before emitting (and before paying its
 * exact-rerank heap fetch); the loser drops its copy. Claiming is a
 * lock-free CAS into an open-addressed table of encoded TIDs (id 0 is
 * never a valid encoded TID -- block 0 is the metapage -- so 0 marks
 * empty slots).
 *
 * Ownership pattern mirrors parallel_build.h: this neutral struct is
 * embedded first in a backend-derived struct; the backend owns the
 * memory (a DSM segment under PostgreSQL, plain memory standalone),
 * lays out the trailing regions, and records their offsets here.
 */

#ifndef MKT_QUERY_PARALLEL_H
#define MKT_QUERY_PARALLEL_H

#include "mkt_config.h"

#ifdef MKT_STANDALONE
#include "standalone/pg_compat.h"
#else
#include <postgres.h>

#include <storage/block.h>
#endif

#include "algo/topk.h"
#include "core/atomics.h"
#include "index/query_scan.h"

/* ----------------------------------------------------------------
 * Shared state (embedded first in a backend-derived struct)
 * ---------------------------------------------------------------- */

typedef struct MktQueryShared
{
	/* Sizing, frozen by the backend at init. */
	uint32_t max_nscan;		/* capacity of the heads[] region */
	uint32_t nparticipants; /* sized (planned) participant count */
	uint32_t claim_mask;	/* claim slots - 1 (power of two) */

	/* Published by the probe-list producer; participants treat it as
	 * read-only afterwards. The backend orders the publication (PG:
	 * spinlock + condition variable; standalone: publish-then-launch). */
	uint32_t nprobes;

	/* Work cursor: next probe rank to hand out. */
	mkt_atomic_uint32 cursor;

	/* Cooperative cancellation: participants poll between clusters. */
	mkt_atomic_uint32 cancel;

	/* Set once when the claim table overflows (see mkt_pquery_claim). */
	mkt_atomic_uint32 claims_full;

	/* Trailing-region offsets in bytes from the START of this struct
	 * (i.e. of the backend-derived allocation), set by the backend:
	 *   heads_off  -> BlockNumber heads[max_nscan]
	 *   claims_off -> mkt_atomic_uint64 claims[claim_mask + 1]
	 */
	uint64_t heads_off;
	uint64_t claims_off;
} MktQueryShared;

static inline BlockNumber *
mkt_pquery_heads(MktQueryShared *shared)
{
	return (BlockNumber *)((char *)shared + shared->heads_off);
}

static inline mkt_atomic_uint64 *
mkt_pquery_claims(MktQueryShared *shared)
{
	return (mkt_atomic_uint64 *)((char *)shared + shared->claims_off);
}

/* ----------------------------------------------------------------
 * Sizing helpers (the backend lays out the full allocation)
 * ---------------------------------------------------------------- */

/* Claim-table slot count for the sized participants and per-participant
 * rerank pool: at least 2x the worst-case claim volume so open
 * addressing stays short, with a floor that keeps tiny configurations
 * collision-free. Always a power of two. */
uint32_t mkt_pquery_claim_slots(uint32_t nparticipants, uint32_t pool);

static inline uint64_t
mkt_pquery_heads_size(uint32_t max_nscan)
{
	return (uint64_t)max_nscan * sizeof(BlockNumber);
}

static inline uint64_t
mkt_pquery_claims_size(uint32_t nslots)
{
	return (uint64_t)nslots * sizeof(mkt_atomic_uint64);
}

/* ----------------------------------------------------------------
 * Lifecycle
 * ---------------------------------------------------------------- */

/* Initialize the embedded shared struct. heads_off/claims_off are
 * offsets of the backend-placed trailing regions; nclaim_slots must
 * come from mkt_pquery_claim_slots (power of two). */
void mkt_pquery_shared_init(
		MktQueryShared *shared,
		uint32_t		max_nscan,
		uint32_t		nparticipants,
		uint32_t		nclaim_slots,
		uint64_t		heads_off,
		uint64_t		claims_off);

/* Reset per-query state (cursor, cancel, claims, nprobes) for a new
 * execution of the same scan (rescan). Not concurrency-safe: the
 * backend guarantees no participant is running. */
void mkt_pquery_begin(MktQueryShared *shared);

/* Publish the probe list (heads[0..n) in final scan-rank order; entries
 * may be InvalidBlockNumber). The caller is the rendezvous winner; the
 * backend orders this publication against readers. */
void mkt_pquery_publish_probes(
		MktQueryShared *shared, const BlockNumber *heads, uint32_t n);

/* ----------------------------------------------------------------
 * Execution
 * ---------------------------------------------------------------- */

/*
 * Claim the next batch of probe ranks. Returns the number of ranks
 * claimed (0 = list drained) and stores the first rank in *start.
 * Batches ramp down toward the tail so the last clusters spread across
 * participants; batching amortizes cursor cacheline traffic when
 * clusters are small.
 */
uint32_t mkt_pquery_next_batch(MktQueryShared *shared, uint32_t *start);

/*
 * Scan every cluster this participant can claim into topk, using the
 * participant's own query state (own posting scan, own storage, own
 * top-k). Returns false if cancelled. Checks for backend interrupts
 * between clusters (under PostgreSQL an interrupt throws; standalone
 * observes the cancel flag).
 */
bool mkt_pquery_scan(
		MktQueryState  *qs,
		MktQueryShared *shared,
		MktDistanceMode mode,
		MktTopK		   *topk,
		MktQueryStats  *stats);

/*
 * Claim ownership of an encoded TID. Returns true if this participant
 * owns it (first claimer, or the table is full -- emitting a potential
 * duplicate is preferred over dropping a result; the overflow is
 * recorded in shared->claims_full for the backend to warn about).
 * Returns false if another participant already claimed it.
 */
bool mkt_pquery_claim(MktQueryShared *shared, uint64_t id);

/* Filter a candidate array in place, keeping entries this participant
 * claims; returns the new count. Used by the per-participant-emission
 * topology before the exact rerank, so a claim loser also skips its
 * heap fetches. */
uint32_t mkt_pquery_filter_claims(
		MktQueryShared *shared, MktTopKEntry *cands, uint32_t n);

/* Request cooperative cancellation (any participant / the backend). */
void mkt_pquery_cancel(MktQueryShared *shared);

/* ----------------------------------------------------------------
 * Merge (leader-merge topology; standalone and tests)
 * ---------------------------------------------------------------- */

/* Replay a participant's collected candidates into dst. Replaying every
 * participant's buffer into one top-k reproduces exactly the serial
 * scan's survivor set: each participant's local pruning threshold is
 * conservative relative to the global one, and the replay re-runs the
 * duplicate-id dedup across participants. */
void mkt_pquery_merge_candidates(
		MktTopK *dst, const MktTopKEntry *cands, uint32_t n);

#endif /* MKT_QUERY_PARALLEL_H */
