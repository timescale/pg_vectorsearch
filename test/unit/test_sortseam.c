/*
 * test_sortseam.c - Unit tests for the posting sort seam (standalone impl)
 *
 * The parallel posting build routes per-cluster entries through the MktSorter
 * seam: each worker puts its (cluster, entry) records and performsorts; the
 * leader merges every worker's records and streams them back grouped by
 * cluster, so the build writes one posting list at a time. This drives the
 * standalone implementation directly through the real seam lifecycle
 * (shared_init -> workers put/performsort -> leader performsort/getnext) and
 * checks the merged stream is ordered by cluster with every entry, its
 * cluster, and its payload preserved.
 */

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "core/memory.h"
#include "index/parallel_build.h"
#include "mkt_test.h"

TEST_GROUP(SortSeam);

/* The shared region (coordinator metadata) is allocated from the per-test
 * arena. The sort's record buffers live in their own dedicated arenas managed
 * by the seam (the leader deletes them at sort_end), independent of this one.
 */
TEST_MEMCTX_FIXTURE();

/* Deterministic PRNG so failures reproduce. */
static uint32_t g_rng;

static uint32_t
rnd(void)
{
	g_rng = g_rng * 1664525u + 1013904223u;
	return g_rng >> 8;
}

/*
 * Multiple workers put entries with interleaved cluster ids; the leader merge
 * must return them sorted by cluster, each exactly once, with its cluster and
 * payload intact, and with the per-cluster counts preserved.
 */
TEST(sort_seam_merges_sorted_by_cluster)
{
	enum
	{
		W		   = 3,
		PER_WORKER = 120,
		NCLUSTERS  = 9,
		TOTAL	   = W * PER_WORKER,
	};

	const uint32_t entry_size = sizeof(uint32_t); /* payload = unique seq id */

	void *region = mkt_alloc(mkt_pbuild_sort_shared_size(W));
	mkt_pbuild_sort_shared_init(region, W, NULL);

	uint32_t cluster_of[TOTAL];
	uint32_t expect_per_cluster[NCLUSTERS] = {0};
	uint32_t seq						   = 0;

	g_rng = 0x1234567u;
	for (int w = 0; w < W; w++)
	{
		MktSorter *s = mkt_pbuild_sort_begin(
				region, NULL, w, W, false, entry_size, 0);
		for (uint32_t i = 0; i < PER_WORKER; i++)
		{
			uint32_t cluster = rnd() % NCLUSTERS;
			cluster_of[seq]	 = cluster;
			expect_per_cluster[cluster]++;
			mkt_pbuild_sort_put(s, cluster, &seq);
			seq++;
		}
		mkt_pbuild_sort_performsort(s);
		mkt_pbuild_sort_end(s);
	}
	ASSERT_EQ((uint32_t)TOTAL, seq, "all entries were put");

	MktSorter *lead =
			mkt_pbuild_sort_begin(region, NULL, 0, W, true, entry_size, 0);
	mkt_pbuild_sort_performsort(lead);

	bool	 seen[TOTAL];
	uint32_t got_per_cluster[NCLUSTERS] = {0};
	memset(seen, 0, sizeof(seen));
	uint32_t	got	 = 0;
	uint32_t	prev = 0;
	uint32_t	cluster;
	const void *entry;
	while (mkt_pbuild_sort_getnext(lead, &cluster, &entry))
	{
		uint32_t s_id;
		memcpy(&s_id, entry, sizeof(uint32_t));
		ASSERT_TRUE(cluster >= prev, "stream must be ordered by cluster");
		ASSERT_TRUE(s_id < (uint32_t)TOTAL, "payload seq in range");
		ASSERT_FALSE(seen[s_id], "no entry returned twice");
		ASSERT_EQ(
				cluster_of[s_id],
				cluster,
				"entry keeps the cluster it was put under");
		seen[s_id] = true;
		got_per_cluster[cluster]++;
		prev = cluster;
		got++;
	}
	ASSERT_EQ((uint32_t)TOTAL, got, "every entry returned exactly once");
	for (uint32_t c = 0; c < NCLUSTERS; c++)
		ASSERT_EQ(
				expect_per_cluster[c],
				got_per_cluster[c],
				"per-cluster counts preserved");

	mkt_pbuild_sort_end(lead);
}

/* Workers that put nothing must merge to an empty stream (no entries, no
 * crash) — the build's empty-table / empty-cluster path. */
TEST(sort_seam_empty)
{
	const uint32_t entry_size = sizeof(uint32_t);
	void		  *region	  = mkt_alloc(mkt_pbuild_sort_shared_size(2));
	mkt_pbuild_sort_shared_init(region, 2, NULL);

	for (int w = 0; w < 2; w++)
	{
		MktSorter *s = mkt_pbuild_sort_begin(
				region, NULL, w, 2, false, entry_size, 0);
		mkt_pbuild_sort_performsort(s);
		mkt_pbuild_sort_end(s);
	}

	MktSorter *lead =
			mkt_pbuild_sort_begin(region, NULL, 0, 2, true, entry_size, 0);
	mkt_pbuild_sort_performsort(lead);
	uint32_t	cluster;
	const void *entry;
	ASSERT_FALSE(
			mkt_pbuild_sort_getnext(lead, &cluster, &entry),
			"an empty sort yields no entries");
	mkt_pbuild_sort_end(lead);
}
