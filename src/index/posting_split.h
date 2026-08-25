/*
 * posting_split.h - Incremental posting-list split (SPFresh/LIRE)
 *
 * Splits one oversized posting list into two balanced lists, keeping
 * partition quality close to a from-scratch build without a global rebuild.
 * The core is backend-neutral (operates on MktIndexBase + MktStorage), so the
 * standalone engine and the PostgreSQL extension share it.
 *
 * Algorithm (one cluster, caller holds an exclusive lock on it):
 *   1. Snapshot the list's live entries and fetch their full-precision
 *      vectors (via MktSplitEnv — heap fetch in PG, in-RAM copy standalone).
 *   2. 2-means over those vectors -> two centroids c0, c1.
 *   3. Allocate two fresh posting heads; re-encode each entry's RaBitQ
 *      residual against its assigned new centroid and append it.
 *   4. Flip the centroid tree: overwrite the old leaf entry to point at the
 *      first new head (routing centroid c0) and append a new leaf entry for
 *      the second head (c1).
 *   5. Tombstone the old chain for later reclaim; bump nlist.
 *
 * Crash-safety (PG): pages are committed in that order, so any crash leaks
 * unreachable pages but never corrupts — before the flip the old list is
 * authoritative; after it the old chain is unreachable garbage.
 *
 * Phase-1 limitations: flat tree (nlevels == 1) and RaBitQ centroid format
 * only. FLOAT/HALF/FASTSCAN centroid formats and multi-level trees are
 * handled in later phases.
 */

#ifndef MKT_POSTING_SPLIT_H
#define MKT_POSTING_SPLIT_H

#include <stdbool.h>
#include <stdint.h>

#include "core/types.h"
#include "index/index_base.h"

#ifdef MKT_STANDALONE
#include "standalone/pg_compat.h"
#else
#include <postgres.h>

#include <storage/itemptr.h>
#endif

/*
 * Context-specific vector access. The split re-clusters on full-precision
 * vectors, which live outside the index (heap in PG, in-RAM store
 * standalone). Fill out[dim] for the given entry TID.
 *
 * Returns true on success; false if the vector is unavailable (e.g. a dead
 * heap tuple), in which case the entry is dropped from the split and left for
 * VACUUM to reclaim.
 */
typedef struct MktSplitEnv
{
	bool (*fetch_vector)(
			void *ctx, ItemPointerData tid, float *out, Dimension dim);
	/*
	 * Retire the split's old chain (unreachable via the tree once the flip
	 * commits). NULL means the core tombstones it immediately — correct where
	 * no concurrent snapshot-holding scanner can still reach it (standalone).
	 * A backend with MVCC snapshots supplies this to defer reclaim behind an
	 * XID gate, keeping the chain readable for in-flight scanners meanwhile.
	 */
	void (*retire_chain)(
			void *ctx, MktStorage *posting_storage, BlockNumber head);
	void *ctx;
} MktSplitEnv;

/* Split tuning. Zero-initialize for defaults. */
typedef struct MktSplitConfig
{
	uint32_t min_split_entries; /* refuse to split below this (0 -> 2) */
	uint32_t km_max_iter;		/* k-means iterations (0 -> default) */
	uint64_t km_seed;			/* k-means seed (0 -> default) */
	/*
	 * LIRE boundary reassignment: after the split, examine this many nearest
	 * neighbor leaves and pull in any entry now closer to a new centroid,
	 * restoring the nearest-partition invariant across the new boundary. 0
	 * disables it. Each affected neighbor is rewritten in place (see
	 * posting_split.c); the caller must hold the right locks where concurrent
	 * (PG wiring is not yet enabled — used from standalone).
	 */
	uint32_t reassign_neighbors;
} MktSplitConfig;

typedef struct MktSplitResult
{
	bool did_split;			/* false if the split was declined (too few
							 * entries, or 2-means produced an empty cluster) */
	BlockNumber head0;		/* first new posting head (former leaf slot) */
	BlockNumber head1;		/* second new posting head (appended leaf) */
	uint32_t	count0;		/* live entries routed to head0 */
	uint32_t	count1;		/* live entries routed to head1 */
	uint32_t	new_nlist;	/* leaf count after the split */
	uint32_t	reassigned; /* entries pulled in from neighbors (LIRE) */
} MktSplitResult;

/*
 * Split the posting list whose head page is `head` into two. The caller must
 * hold an exclusive lock on the cluster (no-op in standalone). cfg and out may
 * be NULL. Returns 0 on success (out->did_split reports whether a split
 * happened), or a negative value on error.
 */
int mkt_posting_split(
		MktIndexBase		 *base,
		BlockNumber			  head,
		const MktSplitConfig *cfg,
		const MktSplitEnv	 *env,
		MktSplitResult		 *out);

typedef struct MktMergeResult
{
	bool		did_merge; /* false if declined (no other leaf, empty list) */
	BlockNumber target;	   /* neighbor head that absorbed the list */
	uint32_t	moved;	   /* entries moved into the neighbor */
	uint32_t	new_nlist; /* leaf count after the merge */
} MktMergeResult;

/*
 * Dissolve the posting list whose head is `head` into its nearest neighbor
 * leaf: re-encode and append its entries there, poison its (now empty) leaf so
 * routing skips it, and retire its chain. The counterpart to split, for
 * undersized lists; the caller decides when a list is small enough and must
 * hold an exclusive lock on it (no-op in standalone). Flat tree + RaBitQ
 * centroid only. Returns 0 on success (out->did_merge reports whether a merge
 * happened), negative on error.
 */
int mkt_posting_merge(
		MktIndexBase	  *base,
		BlockNumber		   head,
		const MktSplitEnv *env,
		MktMergeResult	  *out);

#endif /* MKT_POSTING_SPLIT_H */
