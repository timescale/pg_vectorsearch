/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * test_kv_cmp.c - Noise-aware metric compare
 */

#include <math.h>

#include "cli/kv.h"
#include "cli/kv_cmp.h"
#include "core/memory.h"
#include "vs_test.h"

TEST_GROUP(KvCmp);
TEST_MEMCTX_FIXTURE();

static KvSample
S(double value, double stddev, double n)
{
	return (KvSample){value, stddev, n};
}

TEST(no_stats_fail)
{
	KvCmp r;

	kv_cmp_values("vec/s", S(1e7, 0, 0), S(6.5e6, 0, 0), &r);
	ASSERT_EQ(KV_CMP_FAIL, r.kind, "35% drop fails");
	ASSERT_TRUE(r.worse, "worse");
}

TEST(no_stats_ignore)
{
	KvCmp r;

	kv_cmp_values("vec/s", S(1e7, 0, 0), S(9.8e6, 0, 0), &r);
	ASSERT_EQ(KV_CMP_OK, r.kind, "2% drop ignored");
}

TEST(no_stats_warn)
{
	KvCmp r;

	kv_cmp_values("vec/s", S(1e7, 0, 0), S(8e6, 0, 0), &r);
	ASSERT_EQ(KV_CMP_WARN, r.kind, "20% drop warns");
}

TEST(improvement_ok)
{
	KvCmp r;

	kv_cmp_values("vec/s", S(1e7, 0, 0), S(2e7, 0, 0), &r);
	ASSERT_EQ(KV_CMP_OK, r.kind, "faster is ok");
	ASSERT_TRUE(!r.worse, "not worse");
}

TEST(time_higher_is_worse)
{
	KvCmp r;

	kv_cmp_values("us", S(100, 0, 0), S(120, 0, 0), &r);
	ASSERT_EQ(KV_CMP_WARN, r.kind, "20% slower warns");
	ASSERT_TRUE(r.worse, "worse");
}

TEST(ratio_abs_fail)
{
	KvCmp r;

	kv_cmp_values("ratio", S(0.99, 0, 0), S(0.96, 0, 0), &r);
	ASSERT_EQ(KV_CMP_FAIL, r.kind, "0.03 recall drop fails");
}

TEST(ratio_abs_ignore)
{
	KvCmp r;

	kv_cmp_values("ratio", S(0.99, 0, 0), S(0.989, 0, 0), &r);
	ASSERT_EQ(KV_CMP_OK, r.kind, "0.001 recall drop ignored");
}

TEST(noisy_small_drop_ok)
{
	KvCmp r;

	/* 20% drop, but σ=4e6 n=5 → SE_Δ ≈ 2.5e6, k·SE/mean ≈ 0.51 */
	kv_cmp_values("vec/s", S(1e7, 4e6, 5), S(8e6, 4e6, 5), &r);
	ASSERT_EQ(KV_CMP_OK, r.kind, "inside combined SE");
	ASSERT_TRUE(r.inside_noise, "tagged noise");
}

TEST(noisy_fail_grade_drop_warns)
{
	KvCmp r;

	/* A fail-grade drop is softened by noise, never silenced: a
	 * wild stddev in a result file must not blank the gate. */
	kv_cmp_values("vec/s", S(1e7, 4e6, 5), S(6.5e6, 4e6, 5), &r);
	ASSERT_EQ(KV_CMP_WARN, r.kind, "35% drop warns inside SE");
	ASSERT_TRUE(r.inside_noise, "tagged noise");
	kv_cmp_values("vec/s", S(1e7, 4e7, 5), S(5e6, 4e7, 5), &r);
	ASSERT_EQ(KV_CMP_WARN, r.kind, "50% drop warns under wild stddev");
}

TEST(stable_drop_outside_se)
{
	KvCmp r;

	/* 35% drop, σ=1e5 n=9 → SE tiny relative to 30% bar */
	kv_cmp_values("vec/s", S(1e7, 1e5, 9), S(6.5e6, 1e5, 9), &r);
	ASSERT_EQ(KV_CMP_FAIL, r.kind, "outside SE and past fail bar");
	ASSERT_TRUE(!r.inside_noise, "not noise");
}

TEST(zero_base_fail)
{
	KvCmp r;

	/* No relative scale and no SE band to hide in. */
	kv_cmp_values("us", S(0, 0, 0), S(100, 0, 0), &r);
	ASSERT_EQ(KV_CMP_FAIL, r.kind, "regression from zero fails");
	ASSERT_TRUE(r.worse, "worse");
}

TEST(zero_base_inside_se)
{
	KvCmp r;

	/* σ=5 n=9 each → SE_Δ ≈ 2.36, band ≈ 4.7 covers a delta of 1 */
	kv_cmp_values("us", S(0, 5, 9), S(1, 5, 9), &r);
	ASSERT_EQ(KV_CMP_OK, r.kind, "inside SE of a zero base");
	ASSERT_TRUE(r.inside_noise, "tagged noise");
}

TEST(zero_base_improvement_ok)
{
	KvCmp r;

	kv_cmp_values("vec/s", S(0, 0, 0), S(100, 0, 0), &r);
	ASSERT_EQ(KV_CMP_OK, r.kind, "improvement from zero is ok");
}

TEST(slots_higher_is_worse)
{
	KvCmp r;

	kv_cmp_values("slots", S(100, 0, 0), S(120, 0, 0), &r);
	ASSERT_EQ(KV_CMP_WARN, r.kind, "20% longer run warns");
	ASSERT_TRUE(r.worse, "worse");
}

TEST(nonfinite_value_fails)
{
	KvCmp r;

	/* NaN slips past every comparison as not-worse; in-memory trees
	 * can carry it even though JSON cannot. */
	kv_cmp_values("vec/s", S(1e7, 0, 0), S(NAN, 0, 0), &r);
	ASSERT_EQ(KV_CMP_FAIL, r.kind, "NaN head fails");
	ASSERT_TRUE(r.worse, "worse");
	kv_cmp_values("vec/s", S(NAN, 0, 0), S(1e7, 0, 0), &r);
	ASSERT_EQ(KV_CMP_FAIL, r.kind, "NaN base fails");
	kv_cmp_values("us", S(100, 0, 0), S(INFINITY, 0, 0), &r);
	ASSERT_EQ(KV_CMP_FAIL, r.kind, "infinite head fails");
}

TEST(abs_bar_roundoff_consistent)
{
	KvCmp r;

	/* Equal recall drops must not be split by their binary
	 * representation landing a hair under a bar. */
	kv_cmp_values("ratio", S(0.94, 0, 0), S(0.92, 0, 0), &r);
	ASSERT_EQ(KV_CMP_FAIL, r.kind, "0.02 drop from 0.94 fails");
	kv_cmp_values("ratio", S(0.99, 0, 0), S(0.97, 0, 0), &r);
	ASSERT_EQ(KV_CMP_FAIL, r.kind, "0.02 drop from 0.99 fails");
	kv_cmp_values("ratio", S(0.94, 0, 0), S(0.935, 0, 0), &r);
	ASSERT_EQ(KV_CMP_WARN, r.kind, "0.005 drop from 0.94 warns");
	kv_cmp_values("ratio", S(0.99, 0, 0), S(0.985, 0, 0), &r);
	ASSERT_EQ(KV_CMP_WARN, r.kind, "0.005 drop from 0.99 warns");
}

TEST(count_is_informational)
{
	KvCmp r;

	/* Buffered-candidate tallies have no better direction. */
	kv_cmp_values("count", S(100, 0, 0), S(65, 0, 0), &r);
	ASSERT_EQ(KV_CMP_OK, r.kind, "fewer survivors never fail");
	ASSERT_TRUE(!r.worse, "not judged worse");
	kv_cmp_values("count", S(100, 0, 0), S(200, 0, 0), &r);
	ASSERT_EQ(KV_CMP_OK, r.kind, "more survivors never fail");
	ASSERT_EQ(1.0, r.rel, "the change is still reported");
}
