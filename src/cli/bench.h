/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * bench.h - Timing harness shared by the bench subcommands
 *
 * A benchmark supplies the cases to compare and a function that runs
 * one pass of one case. bench_run owns the clock: it prepares and
 * warms every case, then times them interleaved -- runs outer, cases
 * inner -- and hands back the samples.
 *
 * The nesting alternates rather than finishing one case before
 * starting the next, so every case is sampled over the same stretch
 * of wall clock. Absolute numbers are a separate matter: they depend
 * on how warm the machine already is, so they compare within one
 * invocation rather than against an earlier one.
 */

#ifndef VS_CLI_BENCH_H
#define VS_CLI_BENCH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "core/types.h"
#include "kv.h"

/*
 * An implementation a benchmark can offer to compare, as listed in its
 * own table of what this build and architecture support. Benchmarks
 * filter that table with bench_impl_selected and turn what survives
 * into BenchCases.
 */
typedef struct
{
	const char *name;
	uint32_t	mask; /* SIMD capability bits, or SIMD_NONE */
} ImplSpec;

/*
 * Whether --impls selected this implementation. A NULL filter selects
 * everything; otherwise the filter is a comma-separated list of names.
 */
bool bench_impl_selected(const char *impl_name, const char *filter);

/* "L2", "IP", "Cosine". */
const char *bench_metric_name(DistanceMetric metric);

/*
 * Fill n floats with values in [-1, 1) stepped by 1/100, advancing rng.
 *
 * Quantized to hundredths so a value survives a float round trip
 * unchanged, which keeps results comparable across implementations
 * that differ in intermediate precision.
 */
void bench_fill_random(float *data, size_t n, uint32_t *rng);

/* Samples collected for one case, and their summary. */
typedef struct
{
	double	*samples; /* one per run, in the caller's own unit */
	uint32_t count;
	double	 avg;
	double	 min;
	double	 max;
	double	 stddev;
} BenchStats;

/*
 * One case under comparison. `variant` and `state` are the benchmark's
 * own: `variant` tags which implementation this is (a SIMD mask, an
 * enum, anything), and `state` carries whatever prepare() set up for
 * pass() to use.
 */
typedef struct
{
	const char *name;
	uint32_t	variant;
	void	   *state;
	BenchStats	stats;
} BenchCase;

/*
 * Set up anything pass() needs that must not be timed, such as
 * resolving a kernel or building a query state. Called once per case,
 * before any timing. Returns 0 on success.
 */
typedef int (*BenchPrepareFn)(void *ctx, BenchCase *bench_case);

/*
 * Point global state at this case, for a selection pass() cannot be
 * handed -- a SIMD dispatch override, say. Called before every pass,
 * warmup included, outside the timing window. Returns 0 on success.
 */
typedef int (*BenchPrePassFn)(void *ctx, const BenchCase *bench_case);

/* Exactly one pass of one case. No loop, no clock. Returns 0 on success. */
typedef int (*BenchPassFn)(void *ctx, const BenchCase *bench_case);

/* Print one result line. Called for every case once timing is done. */
typedef void (*BenchReportFn)(
		void *ctx, const BenchCase *bench_case, uint32_t runs);

/* Release what prepare() allocated. Optional. */
typedef void (*BenchTeardownFn)(void *ctx, BenchCase *bench_case);

typedef struct
{
	uint32_t runs;			/* timed passes per case */
	uint32_t warmup_passes; /* untimed passes per case, before any timing */

	BenchPrepareFn	prepare;  /* optional */
	BenchPrePassFn	pre_pass; /* optional */
	BenchPassFn		pass;	  /* required */
	BenchReportFn	report;	  /* required */
	BenchTeardownFn teardown; /* optional */

	void *ctx; /* passed through to every callback */
} BenchSpec;

/*
 * Monotonic clock, and the stats accumulator bench_run fills.
 *
 * Exposed for the benchmarks that time a single variant, where there
 * is nothing to alternate. Anything comparing variants goes through
 * bench_run instead.
 */
uint64_t bench_now_ns(void);
double	 bench_ns_to_ms(uint64_t ns);

void bench_stats_init(BenchStats *stats, uint32_t capacity);
void bench_stats_add(BenchStats *stats, double value);
void bench_stats_compute(BenchStats *stats);
void bench_stats_free(BenchStats *stats);

/*
 * Prepare, warm, and time every case; then report each one.
 *
 * Each sample is passes-per-second as measured, so a benchmark whose
 * pass covers `count` items scales by count in its report callback.
 * Returns 0, or -1 once a callback has failed and said so.
 */
int bench_run(const BenchSpec *spec, BenchCase *cases, uint32_t ncases);

/*
 * Seconds a single pass took, averaged over the runs. The usual
 * starting point for a report callback.
 */
double bench_seconds_per_pass(const BenchCase *bench_case);

/* Relative standard deviation, as a percentage. */
double bench_rsd_percent(const BenchCase *bench_case);

/*
 * Record a rate under "<prefix>/<case name>" in a --json result tree.
 * No-op when metrics is NULL. stddev is attached when runs > 1.
 */
void bench_record_rate(
		Kv				*metrics,
		const char		*prefix,
		const BenchCase *bench_case,
		const char		*unit,
		double			 rate,
		uint32_t		 runs);

#endif /* VS_CLI_BENCH_H */
