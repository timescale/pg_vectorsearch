/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * test_spinup.c - Spin-up plateau detector
 */

#include <stdio.h>
#include <string.h>

#include "cli/bench_spinup.h"
#include "cli/kv.h"
#include "core/memory.h"
#include "vs_test.h"

TEST_GROUP(Spinup);
TEST_MEMCTX_FIXTURE();

/* Feed n chunks of the same time; return true if any feed signaled. */
static bool
feed_n(SpinupDetect *d, double chunk_s, int n)
{
	bool hit = false;

	for (int i = 0; i < n; i++)
		hit |= spinup_detect_feed(d, chunk_s);
	return hit;
}

TEST(first_chunk_never_plateaus)
{
	SpinupDetect d;

	spinup_detect_init(&d);
	ASSERT_TRUE(!spinup_detect_feed(&d, 1.0), "first chunk");
}

TEST(stable_plateaus_after_window)
{
	SpinupDetect d;

	spinup_detect_init(&d);
	spinup_detect_feed(&d, 1.0);
	ASSERT_TRUE(!feed_n(&d, 1.0, SPINUP_WINDOW - 1), "window open");
	ASSERT_TRUE(spinup_detect_feed(&d, 1.0), "window full");
}

TEST(improvement_resets_window)
{
	SpinupDetect d;

	spinup_detect_init(&d);
	spinup_detect_feed(&d, 1.0);
	feed_n(&d, 1.0, SPINUP_WINDOW - 1);
	/* A clearly faster chunk reopens the window. */
	ASSERT_TRUE(!spinup_detect_feed(&d, 0.8), "improvement");
	ASSERT_TRUE(!feed_n(&d, 0.8, SPINUP_WINDOW - 1), "window reopened");
	ASSERT_TRUE(spinup_detect_feed(&d, 0.8), "window full again");
}

TEST(decay_counts_as_stable)
{
	SpinupDetect d;
	bool		 hit = false;

	spinup_detect_init(&d);
	spinup_detect_feed(&d, 1.0);
	/* Clock decaying from peak: times get worse, never better. */
	for (int i = 1; i <= SPINUP_WINDOW; i++)
		hit |= spinup_detect_feed(&d, 1.0 + 0.05 * i);
	ASSERT_TRUE(hit, "decay plateaus");
}

TEST(jitter_within_eps_is_stable)
{
	SpinupDetect d;
	bool		 hit = false;

	spinup_detect_init(&d);
	spinup_detect_feed(&d, 1.0);
	/* Sub-epsilon wobble around the best must not keep it spinning. */
	hit |= spinup_detect_feed(&d, 0.995);
	hit |= spinup_detect_feed(&d, 1.004);
	hit |= spinup_detect_feed(&d, 0.996);
	hit |= spinup_detect_feed(&d, 1.002);
	hit |= spinup_detect_feed(&d, 0.997);
	ASSERT_TRUE(hit, "jitter plateaus");
}

/*
 * Scripted work and clock for cmd_spinup_run. Each chunk call takes
 * the next scripted duration (the last one repeats) and advances the
 * fake clock by exactly that much, so floor, cap, and plateau exits
 * happen at exact, assertable times.
 */
static const double *g_chunks;
static size_t		 g_nchunks;
static size_t		 g_chunk_pos;
static uint64_t		 g_now;

static double
fake_chunk(uint32_t iters)
{
	size_t i = g_chunk_pos < g_nchunks ? g_chunk_pos : g_nchunks - 1;
	double s = g_chunks[i];

	(void)iters;
	g_chunk_pos++;
	g_now += (uint64_t)(s * 1e9);
	return s;
}

static uint64_t
fake_now(void)
{
	return g_now;
}

static Kv *
run_scripted_spinup(const double *chunks, size_t n, uint32_t cap_ms)
{
	SpinupOps ops = {.chunk = fake_chunk, .now = fake_now};

	g_chunks	= chunks;
	g_nchunks	= n;
	g_chunk_pos = 0;
	g_now		= 0;
	cmd_spinup_run(&ops, cap_ms, true);
	return cmd_spinup_kv();
}

TEST(plateau_holds_until_floor)
{
	KvBuild *kb = kv_build_open();

	/* Identical 7.8125 ms chunks plateau after the detector window,
	 * at 54.7 ms, but the floor keeps the loop spinning: 13 chunks,
	 * calibration included, reach 101.5625 ms before it may exit. */
	static const double chunks[] = {0.0078125};

	char *json = kv_to_json(run_scripted_spinup(chunks, 1, 2000));

	ASSERT_TRUE(strstr(json, "\"capped\":false") != NULL, "plateaued");
	ASSERT_TRUE(strstr(json, "\"ms\":101.5625") != NULL, "exits at floor");
	vs_free(json);
	kv_build_close(kb);
}

TEST(cap_stops_continued_improvement)
{
	KvBuild *kb = kv_build_open();

	/* Every chunk beats the best by over the epsilon, so the plateau
	 * never comes and only the cap can end the spin. */
	static const double chunks[] = {
			0.0080,
			0.0077,
			0.0074,
			0.0071,
			0.0068,
			0.0065,
			0.0062,
			0.0059,
			0.0056,
			0.0053,
	};

	char *json = kv_to_json(run_scripted_spinup(chunks, 10, 50));

	ASSERT_TRUE(strstr(json, "\"capped\":true") != NULL, "capped");
	vs_free(json);
	kv_build_close(kb);
}

TEST(budget_tracks_observed_chunk_time)
{
	KvBuild *kb = kv_build_open();

	/* Chunks run twice as long as calibration predicted. Once that
	 * is observed, a third chunk no longer fits the 45 ms budget:
	 * the spin ends at 39.0625 ms instead of overshooting the cap. */
	static const double chunks[] = {0.0078125, 0.015625};

	char *json = kv_to_json(run_scripted_spinup(chunks, 2, 45));

	ASSERT_TRUE(strstr(json, "\"capped\":true") != NULL, "capped");
	ASSERT_TRUE(strstr(json, "\"ms\":39.0625") != NULL, "cap held");
	vs_free(json);
	kv_build_close(kb);
}

TEST(decaying_clock_keeps_calibration_in_stats)
{
	KvBuild *kb = kv_build_open();

	/* Calibration lands on peak boost (125 ns per distance) and every
	 * full chunk after it is slower. The fastest observed sample is
	 * the calibration chunk, so the ramp must stay 1, not dip below. */
	static const double chunks[] = {0.00025, 0.0078125};

	char *json = kv_to_json(run_scripted_spinup(chunks, 2, 2000));

	ASSERT_TRUE(strstr(json, "\"capped\":false") != NULL, "plateaued");
	ASSERT_TRUE(strstr(json, "\"ramp\":1,") != NULL, "calibration counts");
	vs_free(json);
	kv_build_close(kb);
}

TEST(tiny_cap_caps_with_finite_stats)
{
	KvBuild *kb = kv_build_open();

	/* A 1 ms cap cannot fit a single work chunk, so only the
	 * calibration chunk runs and it must carry the stats. */
	cmd_spinup(1, true);

	Kv *o = cmd_spinup_kv();

	ASSERT_TRUE(o != NULL, "spin-up ran");

	char *json = kv_to_json(o);

	/* The JSON encoder writes non-finite doubles as null, so finite
	 * stats means the fields are present and none encode as null. */
	ASSERT_TRUE(strstr(json, "\"capped\":true") != NULL, "capped");
	ASSERT_TRUE(strstr(json, "\"ms\":") != NULL, "ms present");
	ASSERT_TRUE(strstr(json, "\"ramp\":") != NULL, "ramp present");
	ASSERT_TRUE(strstr(json, "\"vps\":") != NULL, "vps present");
	ASSERT_TRUE(strstr(json, "null") == NULL, "finite stats");
	vs_free(json);
	kv_build_close(kb);
}

TEST(gradual_ramp_reopens_window)
{
	SpinupDetect d;

	spinup_detect_init(&d);
	spinup_detect_feed(&d, 1.0);
	/* Steps under the epsilon do not move the reference, so the
	 * third accumulates to 1.2% below it and reopens the window. */
	ASSERT_TRUE(!spinup_detect_feed(&d, 0.996), "sub-eps step");
	ASSERT_TRUE(!spinup_detect_feed(&d, 0.992), "sub-eps step");
	ASSERT_TRUE(!spinup_detect_feed(&d, 0.988), "cumulative reset");
	/* The window restarts from the new reference. */
	ASSERT_TRUE(!feed_n(&d, 0.988, SPINUP_WINDOW - 1), "window open");
	ASSERT_TRUE(spinup_detect_feed(&d, 0.988), "plateau reached");
}

TEST(gradual_ramp_defers_plateau)
{
	KvBuild *kb = kv_build_open();

	/* Chunks improve 0.78% each: no step clears the epsilon on its
	 * own, but every second one does measured from the held
	 * reference, so the window stays open until the ramp flattens:
	 * calibration, 12 ramping chunks, then a window of settled ones
	 * at the final speed. */
	static const double chunks[] = {
			0.0078125,
			0.00775146484375,
			0.0076904296875,
			0.00762939453125,
			0.007568359375,
			0.00750732421875,
			0.0074462890625,
			0.00738525390625,
			0.00732421875,
			0.00726318359375,
			0.0072021484375,
			0.00714111328125,
			0.007080078125,
	};

	char *json = kv_to_json(run_scripted_spinup(chunks, 13, 2000));

	/* The expected exit time counts those chunks on the fake clock:
	 * the final settled stretch is SPINUP_WINDOW long, the ramping
	 * chunks resetting the window keep it from ending earlier. */
	uint64_t ns = 0;

	for (int k = 0; k < 13; k++)
		ns += (uint64_t)(chunks[k] * 1e9);
	ns += (SPINUP_WINDOW - 1) * (uint64_t)(chunks[12] * 1e9);

	char expect[32];

	snprintf(expect, sizeof(expect), "\"ms\":%.17g", (double)ns / 1e6);
	ASSERT_TRUE(strstr(json, "\"capped\":false") != NULL, "plateaued");
	ASSERT_TRUE(strstr(json, expect) != NULL, "spins through the ramp");
	vs_free(json);
	kv_build_close(kb);
}
