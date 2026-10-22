/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * bench.c - Timing harness shared by the bench subcommands
 *
 * See bench.h for the loop nesting and what the callbacks are for.
 */

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "bench.h"
#include "cmd.h"
#include "core/memory.h"

bool
bench_impl_selected(const char *impl_name, const char *filter)
{
	if (filter == NULL)
		return true;

	/* The filter is a comma-separated list; match a whole element. */
	const char *start = filter;
	while (*start)
	{
		const char *end = strchr(start, ',');
		size_t		len = end ? (size_t)(end - start) : strlen(start);

		if (strncmp(start, impl_name, len) == 0 && impl_name[len] == '\0')
			return true;

		if (end == NULL)
			break;
		start = end + 1;
	}

	return false;
}

const char *
bench_metric_name(DistanceMetric metric)
{
	switch (metric)
	{
	case DISTANCE_L2:
		return "L2";
	case DISTANCE_INNER_PRODUCT:
		return "IP";
	case DISTANCE_COSINE:
		return "Cosine";
	}

	return "Unknown";
}

void
bench_fill_random(float *data, size_t n, uint32_t *rng)
{
	for (size_t i = 0; i < n; i++)
	{
		int val = (int)(cmd_rand(rng) % 200) - 100; /* -100 to +99 */
		data[i] = (float)val / 100.0f;
	}
}

uint64_t
bench_now_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

double
bench_ns_to_ms(uint64_t ns)
{
	return (double)ns / 1e6;
}

void
bench_stats_init(BenchStats *stats, uint32_t capacity)
{
	stats->samples = vs_alloc(capacity * sizeof(double));
	stats->count   = 0;
	stats->avg	   = 0.0;
	stats->min	   = 0.0;
	stats->max	   = 0.0;
	stats->stddev  = 0.0;
}

void
bench_stats_add(BenchStats *stats, double value)
{
	stats->samples[stats->count++] = value;
}

void
bench_stats_compute(BenchStats *stats)
{
	if (stats->count == 0)
		return;

	double sum = 0.0;
	for (uint32_t i = 0; i < stats->count; i++)
		sum += stats->samples[i];
	stats->avg = sum / stats->count;

	stats->min = stats->samples[0];
	stats->max = stats->samples[0];
	for (uint32_t i = 1; i < stats->count; i++)
	{
		if (stats->samples[i] < stats->min)
			stats->min = stats->samples[i];
		if (stats->samples[i] > stats->max)
			stats->max = stats->samples[i];
	}

	double var_sum = 0.0;
	for (uint32_t i = 0; i < stats->count; i++)
	{
		double diff = stats->samples[i] - stats->avg;
		var_sum += diff * diff;
	}
	stats->stddev = sqrt(var_sum / stats->count);
}

void
bench_stats_free(BenchStats *stats)
{
	vs_free(stats->samples);
	stats->samples = NULL;
}

static void
free_all(
		const BenchSpec *spec,
		BenchCase		*cases,
		uint32_t		 ncases,
		uint32_t		 prepared)
{
	for (uint32_t i = 0; i < ncases; i++)
		bench_stats_free(&cases[i].stats);

	if (spec->teardown != NULL)
	{
		for (uint32_t i = 0; i < prepared; i++)
			spec->teardown(spec->ctx, &cases[i]);
	}
}

int
bench_run(const BenchSpec *spec, BenchCase *cases, uint32_t ncases)
{
	if (spec == NULL || cases == NULL || ncases == 0 || spec->pass == NULL ||
		spec->report == NULL || spec->runs == 0)
		return -1;

	/* Cases prepared so far, and so the ones cleanup must tear down. */
	uint32_t prepared = (spec->prepare == NULL) ? ncases : 0;
	int		 result	  = -1;

	for (uint32_t i = 0; i < ncases; i++)
		bench_stats_init(&cases[i].stats, spec->runs);

	/* Prepare every case before timing any of them. */
	while (prepared < ncases)
	{
		if (spec->prepare(spec->ctx, &cases[prepared]) != 0)
			goto cleanup;
		prepared++;
	}

	/* Warm every case before timing any of them. */
	for (uint32_t i = 0; i < ncases; i++)
	{
		for (uint32_t w = 0; w < spec->warmup_passes; w++)
		{
			if (spec->pre_pass != NULL &&
				spec->pre_pass(spec->ctx, &cases[i]) != 0)
				goto cleanup;

			if (spec->pass(spec->ctx, &cases[i]) != 0)
				goto cleanup;
		}
	}

	/*
	 * Runs outer, cases inner, so the cases alternate.
	 */
	for (uint32_t run = 0; run < spec->runs; run++)
	{
		for (uint32_t i = 0; i < ncases; i++)
		{
			if (spec->pre_pass != NULL &&
				spec->pre_pass(spec->ctx, &cases[i]) != 0)
				goto cleanup;

			uint64_t start = bench_now_ns();
			int		 ret   = spec->pass(spec->ctx, &cases[i]);
			uint64_t end   = bench_now_ns();

			if (ret != 0)
				goto cleanup;

			bench_stats_add(&cases[i].stats, (double)(end - start) / 1e9);
		}
	}

	for (uint32_t i = 0; i < ncases; i++)
	{
		bench_stats_compute(&cases[i].stats);
		spec->report(spec->ctx, &cases[i], spec->runs);
	}

	result = 0;

cleanup:
	free_all(spec, cases, ncases, prepared);
	return result;
}

double
bench_seconds_per_pass(const BenchCase *bench_case)
{
	return bench_case->stats.avg;
}

double
bench_rsd_percent(const BenchCase *bench_case)
{
	if (bench_case->stats.avg == 0.0)
		return 0.0;
	return (bench_case->stats.stddev / bench_case->stats.avg) * 100.0;
}

void
bench_record_rate(
		Kv				*metrics,
		const char		*prefix,
		const BenchCase *bench_case,
		const char		*unit,
		double			 rate,
		uint32_t		 runs)
{
	if (metrics == NULL)
		return;

	Kv *m = kv_arr_add_metric(
			metrics, unit, rate, "%s/%s", prefix, bench_case->name);

	if (m != NULL && runs > 1)
	{
		/*
		 * The samples are seconds per pass, so scale the deviation into
		 * the reported rate's own unit before publishing it.
		 */
		kv_obj_put(
				m,
				"stddev",
				kv_dbl(rate * bench_rsd_percent(bench_case) / 100.0));
	}
}
