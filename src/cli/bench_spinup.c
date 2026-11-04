/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * bench_spinup.c - Adaptive CPU spin-up before benchmarks
 *
 * Exists so bench results are comparable across invocations. A bench
 * started on an idle core rides the frequency ramp: early timed runs
 * measure a slower clock than late ones, and two invocations of the
 * same binary disagree for no reason in the code under test. The
 * clock cannot be pinned from here, and CI runners expose no
 * frequency controls at all, so instead the harness burns the ramp
 * off before anything is timed.
 *
 * To do that, it runs the dispatched distance kernel over an
 * L1-resident working set in fixed-size chunks and times each chunk.
 * Chunk throughput tracks the core clock, so the plateau detector
 * exits once the ramp from idle is over. A floor keeps a lucky first
 * chunk from exiting immediately and a cap bounds machines that never
 * settle.
 */

#include <stdio.h>

#include "algo/distance.h"
#include "bench.h"
#include "bench_spinup.h"

#define SPINUP_DIM		   768
#define SPINUP_NDB		   8
#define SPINUP_FLOOR_MS	   100
#define SPINUP_CHUNK_MS	   5
#define SPINUP_CALIB_ITERS 2000

typedef struct SpinupInfo
{
	bool   ran;
	bool   capped;
	double ms;		 /* wall time spent spinning */
	double ramp;	 /* first chunk time over best, per distance */
	double best_vps; /* distances per second in the best chunk */
} SpinupInfo;

static SpinupInfo g_spinup;

static float g_query[SPINUP_DIM];
static float g_db[SPINUP_NDB][SPINUP_DIM];

/* Sink so the kernel calls cannot be optimized away. */
static volatile float g_sink;

/* Reset the detector so the next chunk counts as the first. */
void
spinup_detect_init(SpinupDetect *d)
{
	d->ref	  = 0.0;
	d->best	  = 0.0;
	d->stable = 0;
}

/*
 * Feed one chunk time; returns true once the plateau is reached. A
 * chunk beating the reference by more than SPINUP_EPS moves it and
 * reopens the window; anything else, slower chunks included, counts
 * toward it. The reference holds still across sub-epsilon steps, so
 * a gradual ramp accumulates against it rather than hiding below the
 * epsilon chunk by chunk. The best is tracked separately for the
 * reported statistics.
 */
bool
spinup_detect_feed(SpinupDetect *d, double chunk_s)
{
	bool improved = d->ref == 0.0 || chunk_s < d->ref * (1.0 - SPINUP_EPS);

	if (d->best == 0.0 || chunk_s < d->best)
		d->best = chunk_s;
	if (improved)
	{
		d->ref	  = chunk_s;
		d->stable = 0;
		return false;
	}
	d->stable++;
	return d->stable >= SPINUP_WINDOW;
}

/* Fill the query and database vectors with fixed pseudo-random data. */
static void
spinup_fill(void)
{
	uint32_t state = 0x9e3779b9;

	for (uint32_t i = 0; i < SPINUP_DIM; i++)
	{
		state	   = state * 1664525u + 1013904223u;
		g_query[i] = (float)(state >> 8) / (float)(1u << 24);
	}
	for (uint32_t v = 0; v < SPINUP_NDB; v++)
	{
		for (uint32_t i = 0; i < SPINUP_DIM; i++)
		{
			state	   = state * 1664525u + 1013904223u;
			g_db[v][i] = (float)(state >> 8) / (float)(1u << 24);
		}
	}
}

/* Time iters distance calls; returns seconds. */
static double
spinup_chunk(uint32_t iters)
{
	Vec32Ref query = {.data = g_query, .dim = SPINUP_DIM};
	float	 acc   = 0.0f;
	uint64_t start = bench_now_ns();

	for (uint32_t i = 0; i < iters; i++)
	{
		Vec32Ref vec = {
				.data = g_db[i % SPINUP_NDB],
				.dim  = SPINUP_DIM,
		};

		acc += vs_distance(query, vec, DISTANCE_L2);
	}
	g_sink = acc;
	return (double)(bench_now_ns() - start) / 1e9;
}

/*
 * Spin until the clock settles, on the given work and clock sources.
 * A calibration chunk sizes the work so later chunks take about
 * SPINUP_CHUNK_MS each, then chunks run until the detector reports a
 * plateau, no earlier than the floor and no later than cap_ms. What
 * the spin-up saw is kept for cmd_spinup_kv.
 */
void
cmd_spinup_run(const SpinupOps *ops, uint32_t cap_ms, bool quiet)
{
	SpinupDetect d;

	spinup_detect_init(&d);

	uint64_t t0		  = ops->now();
	uint64_t floor_ns = (uint64_t)SPINUP_FLOOR_MS * 1000000ull;
	uint64_t cap_ns	  = (uint64_t)cap_ms * 1000000ull;

	if (floor_ns > cap_ns)
		floor_ns = cap_ns;

	/* Calibrate the chunk size to roughly SPINUP_CHUNK_MS. The
	 * calibration chunk is also the coldest sample, kept for the
	 * ramp ratio. */
	double	 calib	   = ops->chunk(SPINUP_CALIB_ITERS);
	double	 first_per = calib / SPINUP_CALIB_ITERS;
	uint32_t iters	   = (uint32_t)((double)SPINUP_CALIB_ITERS *
								(SPINUP_CHUNK_MS / 1000.0) / calib);

	if (iters < SPINUP_CALIB_ITERS)
		iters = SPINUP_CALIB_ITERS;

	/* What a chunk costs: start from what calibration implies for
	 * iters and grow when a slower chunk shows up, so a chunk
	 * already known not to fit the budget is not started. */
	uint64_t est_ns = (uint64_t)(first_per * (double)iters * 1e9);

	bool plateau = false;
	bool capped	 = false;

	for (;;)
	{
		uint64_t elapsed = ops->now() - t0;

		if (plateau && elapsed >= floor_ns)
			break;
		/* Do not start a chunk the remaining budget cannot fit.
		 * Chunks stay the same size so the detector compares like
		 * with like; the cap trims how many run, not how big. */
		if (elapsed >= cap_ns || cap_ns - elapsed < est_ns)
		{
			capped = true;
			break;
		}

		double	 chunk_s  = ops->chunk(iters);
		uint64_t chunk_ns = (uint64_t)(chunk_s * 1e9);

		if (chunk_ns > est_ns)
			est_ns = chunk_ns;
		plateau = spinup_detect_feed(&d, chunk_s);
	}

	/* Best time per distance, calibration included: a clock already
	 * decaying from boost can make the calibration chunk the fastest
	 * sample. The plateau detector still compares only equal-sized
	 * full chunks. */
	double best_per = d.best > 0.0 ? d.best / iters : first_per;

	if (first_per < best_per)
		best_per = first_per;

	g_spinup.ran	  = true;
	g_spinup.capped	  = capped;
	g_spinup.ms		  = (double)(ops->now() - t0) / 1e6;
	g_spinup.ramp	  = first_per / best_per;
	g_spinup.best_vps = 1.0 / best_per;

	if (!quiet)
		printf("Spin-up: %.0f ms, ramp %.2f%s\n",
			   g_spinup.ms,
			   g_spinup.ramp,
			   capped ? " (capped)" : "");
}

/* Spin up on the real kernel and clock; cap_ms 0 skips entirely. */
void
cmd_spinup(uint32_t cap_ms, bool quiet)
{
	static const SpinupOps real_ops = {
			.chunk = spinup_chunk,
			.now   = bench_now_ns,
	};

	if (cap_ms == 0)
		return;

	spinup_fill();
	cmd_spinup_run(&real_ops, cap_ms, quiet);
}

/*
 * Facts from the spin-up as a Kv object: wall time in ms, the ramp
 * ratio (first chunk over best, so 1.0 means the clock was already
 * settled), best throughput, and whether the cap cut it short.
 * Returns NULL when no spin-up ran.
 */
Kv *
cmd_spinup_kv(void)
{
	Kv *o;

	if (!g_spinup.ran)
		return NULL;
	o = kv_obj();
	kv_obj_put(o, "ms", kv_dbl(g_spinup.ms));
	kv_obj_put(o, "ramp", kv_dbl(g_spinup.ramp));
	kv_obj_put(o, "vps", kv_dbl(g_spinup.best_vps));
	kv_obj_put(o, "capped", kv_bool(g_spinup.capped));
	return o;
}
