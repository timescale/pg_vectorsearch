/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * test_kv_cmp.c - Noise-aware metric compare
 */

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

TEST(noisy_drop_inside_se)
{
	KvCmp r;

	/* 35% drop, but σ=4e6 n=5 → SE_Δ ≈ 2.5e6, k·SE/mean ≈ 0.51 */
	kv_cmp_values("vec/s", S(1e7, 4e6, 5), S(6.5e6, 4e6, 5), &r);
	ASSERT_EQ(KV_CMP_OK, r.kind, "inside combined SE");
	ASSERT_TRUE(r.inside_noise, "tagged noise");
}

TEST(stable_drop_outside_se)
{
	KvCmp r;

	/* 35% drop, σ=1e5 n=9 → SE tiny relative to 30% bar */
	kv_cmp_values("vec/s", S(1e7, 1e5, 9), S(6.5e6, 1e5, 9), &r);
	ASSERT_EQ(KV_CMP_FAIL, r.kind, "outside SE and past fail bar");
	ASSERT_TRUE(!r.inside_noise, "not noise");
}

TEST(from_metric_json)
{
	Kv *b = kv_from_json(
			"{\"name\":\"qps\",\"unit\":\"1/s\",\"value\":100,"
			"\"stddev\":2,\"n\":8}");
	Kv *h = kv_from_json(
			"{\"name\":\"qps\",\"unit\":\"1/s\",\"value\":60,"
			"\"stddev\":2,\"n\":8}");
	KvCmp r;

	kv_cmp_metrics(b, h, &r);
	ASSERT_EQ(KV_CMP_FAIL, r.kind, "stable 40% QPS drop");
}
