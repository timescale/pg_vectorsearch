/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * test_bench.c - The bench_run timing harness
 *
 * The tests script the harness clock: each pass advances a fake time
 * by a fixed cost, so window filling, batch growth, and the per-pass
 * sample are exact regardless of how loaded the test runner is.
 */

#include <stdint.h>

#include "cli/bench.h"
#include "cli/kv_cmp.h"
#include "core/memory.h"
#include "vs_test.h"

TEST_GROUP(BenchRun);
TEST_MEMCTX_FIXTURE();

/* A pass that counts calls and advances the fake clock. */
typedef struct
{
	uint64_t passes;
	uint64_t pre_passes;
	uint64_t pass_ns; /* what one pass costs on the fake clock */
	double	 avg;	  /* copied out by the report callback */
	uint32_t n;		  /* samples seen by the report callback */
} HarnessCtx;

static uint64_t g_fake_ns;

static uint64_t
fake_now(void)
{
	return g_fake_ns;
}

static int
counting_pre_pass(void *ctx, const BenchCase *bench_case)
{
	HarnessCtx *h = ctx;

	(void)bench_case;
	h->pre_passes++;
	return 0;
}

static int
counting_pass(void *ctx, const BenchCase *bench_case)
{
	HarnessCtx *h = ctx;

	(void)bench_case;
	h->passes++;
	g_fake_ns += h->pass_ns;
	return 0;
}

/* The stats are freed when bench_run returns, so copy them out. */
static void
copy_stats_report(void *ctx, const BenchCase *bench_case, uint32_t runs)
{
	HarnessCtx *h = ctx;

	(void)runs;
	h->avg = bench_case->stats.avg;
	h->n   = bench_case->stats.count;
}

static int
run_one(HarnessCtx *h, uint32_t runs, uint32_t warmup, uint32_t min_ms)
{
	BenchCase cases[] = {{.name = "one"}};
	BenchSpec spec	  = {
			   .runs		   = runs,
			   .warmup_passes  = warmup,
			   .min_runtime_ms = min_ms,
			   .now_ns		   = fake_now,
			   .pre_pass	   = counting_pre_pass,
			   .pass		   = counting_pass,
			   .report		   = copy_stats_report,
			   .ctx			   = h,
	   };

	g_fake_ns = 0;
	return bench_run(&spec, cases, 1);
}

/* Seconds per pass as bench_run computes one sample. */
static double
sample_of(uint64_t window_ns, uint64_t passes)
{
	return (double)window_ns / 1e9 / (double)passes;
}

TEST(zero_window_times_single_passes)
{
	HarnessCtx h = {.pass_ns = 1000};

	ASSERT_EQ(0, run_one(&h, 3, 2, 0), "runs");
	/* 2 warmup passes plus exactly one pass per timed sample. */
	ASSERT_EQ(5, h.passes, "pass count");
	ASSERT_EQ(5, h.pre_passes, "one pre_pass per pass");
	ASSERT_EQ(3, h.n, "samples");
	ASSERT_TRUE(h.avg == sample_of(1000, 1), "seconds per pass");
}

TEST(window_repeats_cheap_passes)
{
	HarnessCtx h = {.pass_ns = 1000};

	ASSERT_EQ(0, run_one(&h, 1, 0, 5), "runs");
	/*
	 * At 1 us per pass, the batch doubles from 1 to 512 while a
	 * batch stays under the 1 ms clock target (1023 passes), holds
	 * at 1024 once a batch reaches it, and the 5 ms window closes
	 * after four full batches: 1023 + 4 * 1024.
	 */
	ASSERT_EQ(5119, h.passes, "batch growth");
	ASSERT_EQ(1, h.pre_passes, "one pre_pass per sample");
	ASSERT_TRUE(h.avg == sample_of(5119000, 5119), "seconds per pass");
}

TEST(window_spans_slow_passes)
{
	HarnessCtx h = {.pass_ns = 3000000};

	ASSERT_EQ(0, run_one(&h, 1, 0, 5), "runs");
	/* A 3 ms pass already exceeds the clock target, so the batch
	 * never grows and the second pass closes the 5 ms window. */
	ASSERT_EQ(2, h.passes, "window spanned");
	ASSERT_TRUE(h.avg == sample_of(6000000, 2), "seconds per pass");
}

TEST(stats_use_sample_variance)
{
	BenchStats s;

	bench_stats_init(&s, 2);
	bench_stats_add(&s, 1.0);
	bench_stats_add(&s, 3.0);
	bench_stats_compute(&s);
	/* Bessel: variance over n - 1 = 2, not over n = 1. */
	ASSERT_EQ(2.0, s.avg, "avg");
	ASSERT_TRUE(s.stddev * s.stddev > 1.9, "sample variance");
	bench_stats_free(&s);

	bench_stats_init(&s, 1);
	bench_stats_add(&s, 1.0);
	bench_stats_compute(&s);
	ASSERT_EQ(0.0, s.stddev, "one sample has no spread");
	bench_stats_free(&s);
}

/* A case whose stats carry the given per-pass samples. */
static void
case_with_samples(BenchCase *bc, const double *samples, uint32_t n)
{
	bench_stats_init(&bc->stats, n);
	for (uint32_t i = 0; i < n; i++)
		bench_stats_add(&bc->stats, samples[i]);
	bench_stats_compute(&bc->stats);
}

TEST(record_rate_attaches_stats)
{
	Kv		 *metrics	 = kv_arr();
	BenchCase bc		 = {.name = "one"};
	double	  samples[2] = {1.0, 3.0};
	const Kv *m;

	case_with_samples(&bc, samples, 2);
	bench_record_rate(metrics, "enc", &bc, "vec/s", 100.0, 2);

	m = kv_at(metrics, 0);
	ASSERT_STR_EQ("enc/one", kv_as_str(kv_get(m, "name")), "name");
	ASSERT_EQ(2.0, kv_num(kv_get(m, "n")), "runs recorded");
	/* The samples' spread scaled into the rate's own unit. */
	ASSERT_EQ(
			100.0 * bc.stats.stddev / bc.stats.avg,
			kv_num(kv_get(m, "stddev")),
			"rate-scaled stddev");
	bench_stats_free(&bc.stats);
}

TEST(record_rate_single_run_no_stddev)
{
	Kv		 *metrics	 = kv_arr();
	BenchCase bc		 = {.name = "one"};
	double	  samples[1] = {2.0};
	const Kv *m;

	case_with_samples(&bc, samples, 1);
	bench_record_rate(metrics, "enc", &bc, "vec/s", 100.0, 1);
	bench_stats_free(&bc.stats);

	m = kv_at(metrics, 0);
	ASSERT_EQ(1.0, kv_num(kv_get(m, "n")), "single run recorded");
	ASSERT_NULL(kv_get(m, "stddev"), "one run has no spread");
}

TEST(record_rate_feeds_noise_gate)
{
	Kv		 *metrics	 = kv_arr();
	BenchCase bc		 = {.name = "one"};
	double	  samples[3] = {1.0, 1.0, 2.0};
	KvMetric  bm;
	KvMetric  hm;
	KvCmp	  r;

	/* The same noisy samples on both sides: rsd ≈ 43%. Record two
	 * rates with distinct prefixes, as two runs of a bench would. */
	case_with_samples(&bc, samples, 3);
	bench_record_rate(metrics, "base", &bc, "vec/s", 1e7, 3);
	bench_record_rate(metrics, "head", &bc, "vec/s", 8e6, 3);
	bench_stats_free(&bc.stats);

	/* A 20% drop, inside the SE built from the recorded stats; with
	 * n or stddev dropped from the record it would WARN instead. */
	kv_metric_load(kv_at(metrics, 0), &bm);
	kv_metric_load(kv_at(metrics, 1), &hm);
	kv_cmp_values(
			"vec/s",
			(KvSample){bm.value, bm.stddev, bm.n},
			(KvSample){hm.value, hm.stddev, hm.n},
			&r);
	ASSERT_EQ(KV_CMP_OK, r.kind, "noisy drop inside the gate");
	ASSERT_TRUE(r.inside_noise, "noise came from the recorded stats");
}
