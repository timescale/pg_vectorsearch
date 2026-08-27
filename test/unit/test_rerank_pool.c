/*
 * test_rerank_pool.c - mkt.rerank_pool resolution
 *
 * The pool cap governs how many candidates get exact distances, and so how
 * many random heap fetches a query issues. Its arithmetic is tested here
 * rather than through a PG regression fixture because the cap only becomes
 * observable once the survivor population exceeds it: a 500-row table
 * yields ~16 survivors, well under either the current 4 * k or the former
 * 16 * k, which makes the two indistinguishable at that size.
 */

#include "index/query_scan.h"
#include "mkt_test.h"

TEST_GROUP(RerankPool);

TEST(auto_is_four_k)
{
	/* The multiplier, pinned. Lowered from 16 after measurement showed
	 * 16 * k slower at the same recall on both a cached and an uncached
	 * heap -- see the comment on MKT_RERANK_POOL_AUTO_MULT. */
	ASSERT_EQ(40u, mkt_auto_rerank_pool(0, 10, 0), "auto pool at k=10");
	ASSERT_EQ(4u, mkt_auto_rerank_pool(0, 1, 0), "auto pool at k=1");
	ASSERT_EQ(400u, mkt_auto_rerank_pool(0, 100, 0), "auto pool at k=100");
}

TEST(noise_term_takes_over_when_survivors_flood)
{
	/* max(4 * k, cand_count / 8): with accurate estimates the buffer stays
	 * small and the floor applies; noisy estimates flood it and a flat
	 * floor would cap recall below what the probed clusters hold. */
	ASSERT_EQ(
			40u, mkt_auto_rerank_pool(0, 10, 320), "floor still wins at 320");
	ASSERT_EQ(50u, mkt_auto_rerank_pool(0, 10, 400), "noise term wins at 400");
	ASSERT_EQ(1250u, mkt_auto_rerank_pool(0, 10, 10000), "noise term at 10k");
}

TEST(explicit_setting_is_absolute)
{
	ASSERT_EQ(12u, mkt_auto_rerank_pool(12, 10, 0), "explicit 12");
	/* An explicit value ignores the noise term entirely -- it is a
	 * deliberate cost ceiling, not a hint. */
	ASSERT_EQ(
			12u, mkt_auto_rerank_pool(12, 10, 100000), "explicit beats noise");
}

TEST(never_below_k)
{
	/* A cap below k would truncate the result set. */
	ASSERT_EQ(10u, mkt_auto_rerank_pool(3, 10, 0), "explicit 3 floored to k");
	ASSERT_EQ(10u, mkt_auto_rerank_pool(1, 10, 0), "explicit 1 floored to k");
}

TEST(negative_means_unlimited)
{
	/* 0 downstream means "no cap": rerank every threshold survivor. */
	ASSERT_EQ(0u, mkt_auto_rerank_pool(-1, 10, 0), "-1 is unlimited");
	ASSERT_EQ(
			0u,
			mkt_auto_rerank_pool(-1, 10, 99999),
			"unlimited ignores count");
}
