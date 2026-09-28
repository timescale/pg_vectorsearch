/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * posting_split.h - Incremental posting-list split (SPFresh/LIRE)
 *
 * Splits one oversized posting list into two or more balanced lists, keeping
 * partition quality close to a from-scratch build without a global rebuild.
 * The core is backend-neutral (operates on PrismIndexBase + VsStorage), so
 * the standalone engine and the PostgreSQL extension share it.
 *
 * Splits into k >= 2 partitions. With PrismSplitConfig.target_entries the
 * width is derived from the counted entries, so a list far past its trigger is
 * right-sized in one pass rather than by repeated 2-way bisection -- one flat
 * k-means also beats greedy hierarchical halving. Without a target,
 * PrismSplitConfig.nparts decides, defaulting to 2.
 *
 * The list is streamed, not held: holding every vector at full precision costs
 * entries * dim * 4 bytes, which would make splitting an oversized list fail
 * in proportion to how oversized it is. Two passes over the chain, and the
 * memory is a sample the caller's budget pays for -- see
 * PrismSplitConfig.sample_budget_bytes and prism_split_sample_cap.
 *
 * Algorithm (one cluster, caller holds an exclusive lock on it):
 *   1. Walk the chain, counting the entries whose vectors can still be
 *      fetched (via PrismSplitEnv — heap fetch in PG, in-RAM copy standalone)
 *      and reservoir-sampling them into the clustering buffer. That count,
 *      not the head's live_count, decides whether and how far to split.
 *   2. k-means over the sample -> k centroids, then drop those too small to
 *      be worth a list of their own, widening once or twice rather than
 *      declining if that would leave fewer than two.
 *   3. Walk the chain again, appending each entry to the builder for the
 *      surviving centroid nearest it -- so dropping a centroid is what folds
 *      its entries away, and an entry lands in the list whose stored centroid
 *      is nearest, which is what a query reproduces at scan time. The builders
 *      write in the index's page format, so a split also upgrades a list that
 *      had drifted to AoS back to fastscan.
 *   4. Flip the centroid tree to point the leaf at the k new heads. When the
 *      old leaf and the chain tail share one page and the k-1 extra leaves fit
 *      on it (the common flat single-page tree), the whole flip is one atomic
 *      page write, so a concurrent scan never sees old and new heads together.
 *      Otherwise (tail elsewhere, or the page is full) it falls back to
 *      appending the k-1 extra leaves first, then overwriting the old leaf
 *      last: the old list stays authoritative until that overwrite, so a scan
 *      never misses entries; it may briefly reach a new head and the old head
 *      both, and the duplicate vectors are dropped by the top-k's id dedup
 * (the same path that dedups SOAR replicas), so results stay correct.
 *   5. Retire the old chain for later reclaim; bump nlist by k-1.
 *
 * Crash-safety (PG): pages are committed in that order, so any crash leaks
 * unreachable pages but never corrupts — before the flip the old list is
 * authoritative; after it the old chain is unreachable garbage.
 *
 * Phase-1 limitations: flat tree (nlevels == 1) and RaBitQ centroid format
 * only. FLOAT/HALF/FASTSCAN centroid formats and multi-level trees are
 * handled in later phases.
 */

#ifndef PRISM_POSTING_SPLIT_H
#define PRISM_POSTING_SPLIT_H

#include <stdbool.h>
#include <stdint.h>

#include "core/types.h"
#include "index/index_base.h"

#ifdef VS_STANDALONE
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
typedef struct PrismSplitEnv
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
			void *ctx, VsStorage *posting_storage, BlockNumber head);
	/*
	 * Persist the leaf count before the ids counted from it are used, if the
	 * backend keeps one durably. New cluster ids are counted from nlist and
	 * the new leaves are written before anything raises it, so a backend that
	 * only persists the count afterwards has a window where a crash leaves the
	 * leaves reachable and the count stale -- and the next split then hands
	 * the same ids out again. Called once per split, before the new lists are
	 * written. May be NULL where the count is not persisted separately.
	 */
	void (*reserve_nlist)(void *ctx, uint32_t nlist);
	/*
	 * Start the read for a tid the walk is about to reach, so the fetch that
	 * follows can find it resident. Both passes fetch every entry of the
	 * list, in a chain whose tids are already roughly ascending, so there is
	 * a next block worth starting early. Called at most once per distinct
	 * block, one tid ahead of the callback. May be NULL where fetching does
	 * no I/O (standalone holds its vectors in memory).
	 */
	void (*prefetch_vector)(void *ctx, ItemPointerData tid);

	void *ctx;
} PrismSplitEnv;

/* Upper bound on the number of partitions one split may produce. A very
 * oversized list is split toward this many at once; anything beyond is left
 * for a later split. Bounds the on-stack result arrays and the per-split work.
 */
#define PRISM_SPLIT_MAX_PARTS 32

/*
 * How far above the target size a list must grow before it is worth splitting.
 * The target is the resting size a split aims each new list at; the trigger is
 * that times this factor, so a list has room to absorb inserts (and deletes)
 * without immediately splitting again, and a fresh part sits at the geometric
 * centre of the operating band [target/factor, target*factor].
 *
 * Keep it an integer: split width is round(count / target), which at the
 * trigger is the factor itself, so each new list lands on the target. A
 * fractional factor would round the width up and land every new part below the
 * target, and the target would stop being where lists rest. At 2 the
 * steady-state split is a bisection, matching the SPFresh/LIRE protocol, and a
 * wider split only happens when a batch pass meets a neglected list.
 */
#define PRISM_SPLIT_TRIGGER_FACTOR 2

/*
 * How many times a split may widen by one partition when dropping the
 * undersized clusters cannot leave two standing. Small on purpose: one extra
 * partition resolves the common case (a dense region plus a straggler), and
 * data that resists more than that is better declined than re-clustered
 * repeatedly.
 */
#define PRISM_SPLIT_WIDEN_ATTEMPTS 2

/*
 * The size a list must exceed before it is worth splitting. One definition,
 * because the trigger is tested in three places -- the scan that picks
 * candidates, the re-check under the head lock, and the authoritative re-check
 * after collection -- and they have to agree. They drifted once already, when
 * the two re-checks disagreed on `<` versus `<=`.
 */
static inline uint64_t
prism_split_trigger(uint32_t target_entries)
{
	return (uint64_t)target_entries * PRISM_SPLIT_TRIGGER_FACTOR;
}

/*
 * Memory budget for a split, in bytes, when the caller does not state one.
 * Deliberately modest -- a caller with a maintenance memory budget of its own
 * should say so through PrismSplitConfig.sample_budget_bytes.
 */
#define PRISM_SPLIT_SAMPLE_BUDGET_BYTES (32u * 1024u * 1024u)

/*
 * Ceiling on the sample's own allocation, whatever budget the caller states.
 * A single allocation has an upper limit in a PostgreSQL backend
 * (MaxAllocSize, just under 1 GB) and asking for more is an error, not a
 * smaller sample -- so a generous maintenance_work_mem must not turn into a
 * failed split. Well inside that limit, and far more sample than clustering
 * into at most PRISM_SPLIT_MAX_PARTS partitions can use.
 */
#define PRISM_SPLIT_MAX_SAMPLE_BYTES (256u * 1024u * 1024u)

/*
 * Sample points a partition needs before its share of the sample says anything
 * about its real size. Below this the tally is noise, and a cluster that is
 * really a fair size looks small enough to drop -- so a split asks for no more
 * partitions than its sample can speak for, and leaves the rest to the pass
 * after it. With the default budget this never binds; it protects a caller
 * that sets a tight one.
 */
#define PRISM_SPLIT_MIN_SAMPLE_PER_PART 32u

/* Seed for the reservoir draws when the caller states no k-means seed, so a
 * split samples the same way on a re-run. */
#define PRISM_SPLIT_SAMPLE_SEED UINT64_C(0x5EED5A1717C0FFEE)

/*
 * What clustering costs per sampled point, on top of the point itself: k-means
 * keeps an assignment, an L2 norm and an initialisation distance per point,
 * Elkan keeps an upper bound per point and a lower bound per point *per
 * centroid*, and the result carries its own copy of the assignments. The
 * per-centroid part is sized for the widest split, because the width is not
 * known until the list has been counted.
 *
 * At a wide dimension this is a few percent of the point; at a narrow one it
 * is several times the point, which is why the budget cannot simply be divided
 * by the vector size.
 */
#define PRISM_SPLIT_SAMPLE_POINT_OVERHEAD \
	(5u * (uint32_t)sizeof(float) +       \
	 (uint32_t)sizeof(float) * PRISM_SPLIT_MAX_PARTS)

/*
 * What clustering costs regardless of how many points are sampled: several
 * sets of centroids, and k-means' blocked distance and vector scratch. Sized
 * for the widest split and the full block, so it over-reserves for a narrow
 * one rather than under-reserving.
 */
static inline uint64_t
prism_split_fixed_bytes(Dimension dim)
{
	const uint64_t block	 = 4096; /* KMEANS_BLOCK_SIZE */
	const uint64_t centroids = 5u * PRISM_SPLIT_MAX_PARTS;
	uint64_t per_dim	= (centroids + block) * (uint64_t)dim * sizeof(float);
	uint64_t dist_block = block * PRISM_SPLIT_MAX_PARTS * sizeof(float);

	/* Slack for the many small allocations neither term names. */
	return per_dim + dist_block + 64u * 1024u;
}

/*
 * Sampled points a budget can pay for, or 0 if it cannot pay for a split at
 * all. A caller that gets 0 should say so rather than proceed: exceeding the
 * budget it was given is not its decision to make.
 */
static inline uint32_t
prism_split_sample_cap(uint64_t budget_bytes, Dimension dim)
{
	uint64_t fixed	   = prism_split_fixed_bytes(dim);
	uint64_t per_point = (uint64_t)dim * sizeof(float) +
						 PRISM_SPLIT_SAMPLE_POINT_OVERHEAD;
	uint64_t vec_cap = PRISM_SPLIT_MAX_SAMPLE_BYTES /
					   ((uint64_t)dim * sizeof(float));

	if (budget_bytes <= fixed)
		return 0;

	uint64_t n = (budget_bytes - fixed) / per_point;
	if (n > vec_cap)
		n = vec_cap;
	/* Too few points to say anything about even two partitions. */
	if (n < 2u * PRISM_SPLIT_MIN_SAMPLE_PER_PART)
		return 0;
	return (uint32_t)n;
}

/* The smallest budget prism_split_sample_cap will accept, for a caller that
 * wants to say by how much its own is short. */
static inline uint64_t
prism_split_min_budget_bytes(Dimension dim)
{
	uint64_t per_point = (uint64_t)dim * sizeof(float) +
						 PRISM_SPLIT_SAMPLE_POINT_OVERHEAD;
	return prism_split_fixed_bytes(dim) +
		   2u * PRISM_SPLIT_MIN_SAMPLE_PER_PART * per_point;
}

/* Split tuning. Zero-initialize for defaults. */
typedef struct PrismSplitConfig
{
	uint32_t min_split_entries; /* refuse to split below this (0 -> 2) */
	/*
	 * Resting size to aim each new list at. When set, the split declines
	 * unless the collected entries still exceed
	 * target_entries * PRISM_SPLIT_TRIGGER_FACTOR, and otherwise splits
	 * round(count / target_entries) ways -- rounded, not ceiled, so that at
	 * the trigger the width is exactly the factor and each new list lands on
	 * the target rather than below it.
	 *
	 * The check is deliberately against the *collected* count, not the head's
	 * live_count: collection drops entries whose vector cannot be fetched,
	 * and those are not reflected in live_count, so a list can look oversized
	 * and turn out not to be. Re-verifying after collection is also what the
	 * LIRE protocol does (SPFresh SS4.2.1: garbage-collect, re-check against
	 * the split limit, complete without splitting if it now fits).
	 *
	 * 0 disables both behaviours: the split is unconditional and nparts (or
	 * its default of 2) decides the width. That is the manual escape hatch.
	 */
	uint32_t target_entries;
	/* Number of partitions to split into, clamped to
	 * [2, PRISM_SPLIT_MAX_PARTS] and to the live entry count (0 -> 2). Ignored
	 * when target_entries is set, which derives the width instead. Lets a
	 * caller right-size a hugely oversized list in one pass instead of
	 * repeatedly bisecting. */
	uint32_t nparts;
	/*
	 * Memory a split may use, in bytes. It streams the list rather than
	 * holding it, and sizes its clustering sample so that the sample and the
	 * clustering working set together fit here -- see prism_split_sample_cap.
	 * 0 takes PRISM_SPLIT_SAMPLE_BUDGET_BYTES. A backend with a maintenance
	 * memory budget should pass it through.
	 */
	uint64_t sample_budget_bytes;
	uint32_t km_max_iter; /* k-means iterations (0 -> default) */
	uint64_t km_seed;	  /* k-means seed (0 -> default) */
} PrismSplitConfig;

typedef struct PrismSplitResult
{
	/*
	 * false when the split declined: too few entries, a list already inside
	 * its operating band, or no two partitions clearing the floor even after
	 * widening.
	 */
	bool		did_split;
	uint32_t	nparts; /* number of new lists produced (0 if declined) */
	BlockNumber head[PRISM_SPLIT_MAX_PARTS]; /* new posting heads; head[0]
											  * reuses the former leaf slot */
	uint32_t count[PRISM_SPLIT_MAX_PARTS];	 /* live entries per new head */
	uint32_t new_nlist;						 /* leaf count after the split */
	/*
	 * Centroid pages this split appended because a level-0 page had no room.
	 * The caller persists the new total; the split cannot, the metapage
	 * being a PostgreSQL detail the shared layer does not reach.
	 */
	uint32_t new_centroid_pages;
} PrismSplitResult;

/*
 * Split the posting list whose head page is `head`. The caller must hold an
 * exclusive lock on the cluster (no-op in standalone). cfg and out may be
 * NULL. Returns 0 on success (out->did_split reports whether a split happened
 * -- declining is not an error), or a negative value on error, which includes
 * a memory budget too small to split at this dimension.
 */
int prism_posting_split(
		PrismIndexBase		   *base,
		BlockNumber				head,
		const PrismSplitConfig *cfg,
		const PrismSplitEnv	   *env,
		PrismSplitResult	   *out);

/*
 * Tombstone every page of a posting chain starting at `head`, so scans skip it
 * and a later recycle pass can reclaim the space. Retires a split's old chain
 * once no scanner can still reach it: the immediate standalone path, and the
 * PG XID-gated reclaim once the deletion horizon has passed.
 */
void prism_posting_chain_tombstone(VsStorage *storage, BlockNumber head);

#endif /* PRISM_POSTING_SPLIT_H */
