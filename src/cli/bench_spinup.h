/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * bench_spinup.h - Adaptive CPU spin-up before benchmarks
 *
 * Keeps bench results comparable across invocations of the same
 * binary: busy-runs a SIMD kernel in fixed work chunks until chunk
 * throughput stops improving, so timed runs start at a settled clock
 * instead of riding the post-idle frequency ramp.
 */

#ifndef VS_CLI_BENCH_SPINUP_H
#define VS_CLI_BENCH_SPINUP_H

#include <stdbool.h>
#include <stdint.h>

#include "kv.h"

/*
 * Plateau detector over chunk times. A chunk improves when it beats
 * the reference by more than SPINUP_EPS; the plateau is reached after
 * SPINUP_WINDOW consecutive chunks without such an improvement. The
 * reference only moves on a significant improvement, so a gradual
 * ramp accumulates until it crosses the epsilon and reopens the
 * window instead of passing as settled. Chunks that get slower also
 * count as stable, so a clock decaying from peak boost exits rather
 * than waiting for a new maximum; the cap bounds a ramp that never
 * ends.
 */
#define SPINUP_EPS	  0.01
#define SPINUP_WINDOW 5

typedef struct SpinupDetect
{
	double	 ref;	 /* baseline an improvement must beat (0 = none) */
	double	 best;	 /* lowest chunk seconds seen, for the stats */
	uint32_t stable; /* chunks since the reference last moved */
} SpinupDetect;

void spinup_detect_init(SpinupDetect *d);
bool spinup_detect_feed(SpinupDetect *d, double chunk_s);

/*
 * Work and clock sources for the spin-up loop. cmd_spinup wires the
 * real SIMD kernel and monotonic clock; tests inject scripted ones to
 * drive the floor, cap, and plateau paths deterministically.
 */
typedef struct SpinupOps
{
	double (*chunk)(uint32_t iters); /* run iters distances; seconds */
	uint64_t (*now)(void);			 /* monotonic ns */
} SpinupOps;

/* Spin until the clock plateaus or cap_ms elapses (0 = skip). */
void cmd_spinup(uint32_t cap_ms, bool quiet);

/* The loop behind cmd_spinup, on the given sources. cap_ms > 0. */
void cmd_spinup_run(const SpinupOps *ops, uint32_t cap_ms, bool quiet);

/* Spin-up facts for the JSON root; NULL when no spin-up ran. */
Kv *cmd_spinup_kv(void);

#endif /* VS_CLI_BENCH_SPINUP_H */
