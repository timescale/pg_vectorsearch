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
	void *ctx;
} MktSplitEnv;

/* Split tuning. Zero-initialize for defaults. */
typedef struct MktSplitConfig
{
	uint32_t min_split_entries; /* refuse to split below this (0 -> 2) */
	uint32_t km_max_iter;		/* k-means iterations (0 -> default) */
	uint64_t km_seed;			/* k-means seed (0 -> default) */
} MktSplitConfig;

typedef struct MktSplitResult
{
	bool did_split;		   /* false if the split was declined (too few
							* entries, or 2-means produced an empty cluster) */
	BlockNumber head0;	   /* first new posting head (former leaf slot) */
	BlockNumber head1;	   /* second new posting head (appended leaf) */
	uint32_t	count0;	   /* live entries routed to head0 */
	uint32_t	count1;	   /* live entries routed to head1 */
	uint32_t	new_nlist; /* leaf count after the split */
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

#endif /* MKT_POSTING_SPLIT_H */
