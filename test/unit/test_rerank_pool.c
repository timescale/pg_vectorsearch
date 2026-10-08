/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * test_rerank_pool.c - prism.rerank_pool resolution
 *
 * The pool cap decides how many candidates get exact distances, and so how
 * many random heap fetches a query issues -- the difference between a query
 * that touches the heap a few dozen times and one that touches it thousands
 * of times.
 *
 * It is tested here rather than through a regression fixture because the cap
 * only becomes observable once the survivor population exceeds it, and a
 * small table yields far fewer survivors than any of these values: at that
 * size every candidate formula returns the same answer, so a fixture cannot
 * tell them apart. What a fixture can cover -- that the GUC reaches the scan
 * and never truncates below k -- is in test/pg/sql/rerank.sql.
 *
 * The scan adds one term the estimate cannot: a floor at an eighth of the
 * candidate buffer, which measures estimate noise and exists only once the
 * clusters have been scanned. That term is exercised by the scan itself.
 */

#include "index/query_scan.h"
#include "vs_test.h"

TEST_GROUP(RerankPool);

/* The estimate reads the GUC through a setter; leave it at the default. */
static uint32_t
pool_for(int32_t setting, uint32_t k, uint32_t nprobe)
{
	prism_query_set_rerank_pool(setting);
	uint32_t pool = prism_query_rerank_pool_estimate(k, nprobe);
	prism_query_set_rerank_pool(0);
	return pool;
}

TEST(auto_pool_scales_with_nprobe)
{
	/*
	 * 3 * k * nprobe^0.15, pinned at k=10. The exponent is what matters:
	 * a wider probe returns more candidates whose estimates are worse, so
	 * the pool has to grow -- but sub-linearly, since rerank cost is what
	 * the cap exists to bound. Fitted against a recall-vs-pool sweep; see
	 * PRISM_RERANK_POOL_AUTO_COEFF.
	 */
	ASSERT_EQ(30u, pool_for(0, 10, 1), "nprobe=1 gives 3 * k");
	ASSERT_EQ(42u, pool_for(0, 10, 10), "nprobe=10");
	ASSERT_EQ(50u, pool_for(0, 10, 32), "nprobe=32");
	ASSERT_EQ(64u, pool_for(0, 10, 160), "nprobe=160");

	/*
	 * Growth is slow enough to stay useful: a 16x wider probe raises the
	 * pool by well under 2x. A flat multiplier (this formula's
	 * predecessor) had to be sized for the widest probe and was therefore
	 * oversized at every narrower one.
	 */
	ASSERT_TRUE(
			pool_for(0, 10, 160) < 2 * pool_for(0, 10, 10),
			"16x the probe costs less than 2x the pool");
}

TEST(auto_pool_scales_with_k)
{
	/* Linear in k: the pool is a multiple of the result set, so asking for
	 * ten times the rows reranks ten times as many candidates. */
	ASSERT_EQ(4u, pool_for(0, 1, 10), "k=1");
	ASSERT_EQ(42u, pool_for(0, 10, 10), "k=10");
	ASSERT_EQ(424u, pool_for(0, 100, 10), "k=100");
}

TEST(explicit_setting_is_absolute)
{
	/* A positive setting is a deliberate cost ceiling, not a hint: it
	 * ignores the formula. */
	ASSERT_EQ(12u, pool_for(12, 10, 32), "explicit 12");
	ASSERT_EQ(12u, pool_for(12, 10, 160), "and does not scale with nprobe");
}

TEST(never_below_k)
{
	/* A cap under k would truncate the result set, so k is the floor -- for
	 * an explicit setting, and for a formula that lands low. */
	ASSERT_EQ(10u, pool_for(3, 10, 32), "explicit 3 floors to k");
	ASSERT_EQ(10u, pool_for(1, 10, 32), "explicit 1 floors to k");
	ASSERT_EQ(
			10u,
			pool_for(0, 10, 0),
			"a zero probe count floors to k rather than to zero");
}

TEST(negative_means_unlimited)
{
	/* 0 downstream means "no cap": rerank every threshold survivor. */
	ASSERT_EQ(0u, pool_for(-1, 10, 32), "-1 is unlimited");
	ASSERT_EQ(0u, pool_for(-1, 100, 160), "regardless of k and nprobe");
}
