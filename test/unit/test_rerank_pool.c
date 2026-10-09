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

/* The estimate reads the GUCs through setters; restore the defaults. */
static uint32_t
pool_for(int32_t setting, double cost_scale, uint32_t k, uint32_t nprobe)
{
	prism_query_set_rerank_pool(setting);
	prism_query_set_rerank_cost_scale(cost_scale);
	uint32_t pool = prism_query_rerank_pool_estimate(k, nprobe);
	prism_query_set_rerank_pool(0);
	prism_query_set_rerank_cost_scale(1.0);
	return pool;
}

TEST(auto_pool_scales_with_nprobe)
{
	/*
	 * 3 * k * nprobe^0.15 at the default cost scale of 1. The exponent is
	 * what matters: a wider probe returns more candidates whose estimates
	 * are worse, so the pool has to grow -- but sub-linearly, since rerank
	 * cost is what the cap exists to bound. Fitted against a
	 * recall-vs-pool sweep; see PRISM_RERANK_POOL_AUTO_COEFF.
	 */
	ASSERT_EQ(30u, pool_for(0, 1.0, 10, 1), "nprobe=1 gives 3 * k");
	ASSERT_EQ(42u, pool_for(0, 1.0, 10, 10), "nprobe=10");
	ASSERT_EQ(50u, pool_for(0, 1.0, 10, 32), "nprobe=32");
	ASSERT_EQ(64u, pool_for(0, 1.0, 10, 160), "nprobe=160");

	/*
	 * Growth is slow enough to stay useful: a 16x wider probe raises the
	 * pool by well under 2x. A flat multiplier (this formula's
	 * predecessor) had to be sized for the widest probe and was therefore
	 * oversized at every narrower one.
	 */
	ASSERT_TRUE(
			pool_for(0, 1.0, 10, 160) < 2 * pool_for(0, 1.0, 10, 10),
			"16x the probe costs less than 2x the pool");
}

TEST(auto_pool_scales_with_k)
{
	/* Linear in k: the pool is a multiple of the result set, so asking for
	 * ten times the rows reranks ten times as many candidates. */
	ASSERT_EQ(4u, pool_for(0, 1.0, 1, 10), "k=1");
	ASSERT_EQ(42u, pool_for(0, 1.0, 10, 10), "k=10");
	ASSERT_EQ(424u, pool_for(0, 1.0, 100, 10), "k=100");
}

TEST(cost_scale_shrinks_pool_but_not_below_recall_floor)
{
	/*
	 * Dividing the base fit by the cost scale is the point of the scale;
	 * the recall floor 1.2 * k * nprobe^0.3 is what the division may not
	 * cross. At nprobe=10 with scale 4.4: base 42 -> 42/4.4 = 9.5, but
	 * the floor is 1.2 * 10 * 10^0.3 = 23.9 -> 24. At nprobe=640 the
	 * floor (83) clears the fit (79), so the pool is the full unscaled
	 * fit: deep probes always rerank the full fit. Measured on cohere-1m
	 * 768d: the recall ceiling needs pool ~42..79 across nprobe 10..640,
	 * so a uniformly scaled fit caps recall far below what the probed
	 * lists contain.
	 */
	ASSERT_EQ(24u, pool_for(0, 4.4, 10, 10), "scaled base 9 floors to 24");
	ASSERT_EQ(36u, pool_for(0, 4.4, 10, 40), "nprobe=40");
	ASSERT_EQ(55u, pool_for(0, 4.4, 10, 160), "nprobe=160");
	ASSERT_EQ(
			79u, pool_for(0, 4.4, 10, 640), "nprobe=640 reranks the full fit");

	/* Between the two regimes the scaled base itself can win: at
	 * nprobe=80, base 58 / 1.5 = 38.7 vs floor 1.2 * 10 * 80^0.3 = 44.6
	 * -- the floor binds here, so 45. At scale 1.2: 58 / 1.2 = 48.3 vs
	 * floor 44.6, the scaled base wins. */
	ASSERT_EQ(45u, pool_for(0, 1.5, 10, 80), "floor binds above scaled base");
	ASSERT_EQ(48u, pool_for(0, 1.2, 10, 80), "scaled base above floor");
}

TEST(recall_floor_never_exceeds_unscaled_fit)
{
	/*
	 * The floor restores what scaling removed, no more: at very wide
	 * probes k * nprobe^0.3 outgrows the 3 * k * nprobe^0.15 fit, and the
	 * pool must not follow it past the fit. nprobe=2560: base 97, floor
	 * 105 -> capped at 97 even under scale 2.
	 */
	ASSERT_EQ(97u, pool_for(0, 2.0, 10, 2560), "floor capped at the fit");
	ASSERT_EQ(97u, pool_for(0, 1.0, 10, 2560), "scale 1 unaffected");
}

TEST(explicit_setting_is_absolute)
{
	/* A positive setting is a deliberate cost ceiling, not a hint: it
	 * ignores the formula and the cost scale. */
	ASSERT_EQ(12u, pool_for(12, 1.0, 10, 32), "explicit 12");
	ASSERT_EQ(12u, pool_for(12, 4.4, 10, 32), "and ignores the scale");
	ASSERT_EQ(
			12u, pool_for(12, 1.0, 10, 160), "and does not scale with nprobe");
}

TEST(never_below_k)
{
	/* A cap under k would truncate the result set, so k is the floor -- for
	 * an explicit setting, and for a formula that lands low. */
	ASSERT_EQ(10u, pool_for(3, 1.0, 10, 32), "explicit 3 floors to k");
	ASSERT_EQ(10u, pool_for(1, 1.0, 10, 32), "explicit 1 floors to k");
	ASSERT_EQ(
			12u,
			pool_for(0, 4.4, 10, 1),
			"auto never drops below the recall floor (1.2 * k at nprobe=1)");
}

TEST(negative_means_unlimited)
{
	/* 0 downstream means "no cap": rerank every threshold survivor. */
	ASSERT_EQ(0u, pool_for(-1, 1.0, 10, 32), "-1 is unlimited");
	ASSERT_EQ(
			0u, pool_for(-1, 4.4, 100, 160), "regardless of k, nprobe, scale");
}

TEST(cost_scale_has_a_minimum)
{
	/* A near-zero scale must not blow the pool up toward unbounded: the
	 * setter clamps to PRISM_RERANK_COST_SCALE_MIN. */
	uint32_t pool = pool_for(0, 0.0, 10, 10);

	ASSERT_TRUE(pool < 10000u, "clamped scale keeps the pool bounded");
}
