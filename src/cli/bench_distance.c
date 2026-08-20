/*
 * mkt bench distance
 *
 * Benchmark distance computations across SIMD implementations.
 */

/* Must be first - defines MKT_SIMD_FULL used by distance.h */
#include "mkt_config.h"

#include <getopt.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "algo/distance.h"
#include "cmd.h"
#include "core/memory.h"
#include "core/platform.h"
#include "core/types.h"
#include "distance_pgvector.h"
#include "types/vector.h"

/* Default parameters */
#define DEFAULT_DIM	  768
#define DEFAULT_COUNT 10000

/* SIMD implementations to test */
typedef struct
{
	const char *name;
	uint32_t	mask;
} ImplSpec;

/*
 * Available implementations depend on build mode and architecture.
 * - simd=full: compiler + hand-optimized (avx2/avx512 on x86, neon on ARM)
 * - simd=compiler or simd=none: compiler only
 */
static const ImplSpec impls[] = {
		{"compiler", SIMD_NONE},
#if defined(MKT_SIMD_FULL) && (defined(__x86_64__) || defined(_M_X64))
		{"avx2", SIMD_AVX2},
		{"avx512", MKT_SIMD_AVX512_DQ},
#endif
#if defined(MKT_SIMD_FULL) && (defined(__aarch64__) || defined(_M_ARM64))
		{"neon", SIMD_NEON},
#endif
		{NULL, 0},
};

/* Metric filter bitmask */
#define METRIC_L2	  (1 << 0)
#define METRIC_IP	  (1 << 1)
#define METRIC_COSINE (1 << 2)
#define METRIC_ALL	  (METRIC_L2 | METRIC_IP | METRIC_COSINE)

/* Benchmark configuration */
typedef struct
{
	Dimension	dim;
	uint32_t	count;
	uint32_t	runs;		   /* Number of iterations to average */
	uint32_t	metric_filter; /* Bitmask of metrics to run */
	const char *impl_filter;   /* NULL = all, or "scalar,avx2,..." */
	bool		help;
} BenchConfig;

/* Timing helpers */
static inline uint64_t
get_time_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static inline double
ns_to_ms(uint64_t ns)
{
	return (double)ns / 1e6;
}

/* Timing statistics */
typedef struct
{
	double	*samples; /* Array of vec/s measurements */
	uint32_t count;
	double	 avg;
	double	 min;
	double	 max;
	double	 stddev;
} BenchStats;

static void
bench_stats_init(BenchStats *stats, uint32_t capacity)
{
	stats->samples = mkt_alloc(capacity * sizeof(double));
	stats->count   = 0;
	stats->avg	   = 0.0;
	stats->min	   = 0.0;
	stats->max	   = 0.0;
	stats->stddev  = 0.0;
}

static void
bench_stats_add(BenchStats *stats, double value)
{
	stats->samples[stats->count++] = value;
}

static void
bench_stats_compute(BenchStats *stats)
{
	if (stats->count == 0)
		return;

	/* Calculate average */
	double sum = 0.0;
	for (uint32_t i = 0; i < stats->count; i++)
		sum += stats->samples[i];
	stats->avg = sum / stats->count;

	/* Find min/max */
	stats->min = stats->samples[0];
	stats->max = stats->samples[0];
	for (uint32_t i = 1; i < stats->count; i++)
	{
		if (stats->samples[i] < stats->min)
			stats->min = stats->samples[i];
		if (stats->samples[i] > stats->max)
			stats->max = stats->samples[i];
	}

	/* Calculate standard deviation */
	double var_sum = 0.0;
	for (uint32_t i = 0; i < stats->count; i++)
	{
		double diff = stats->samples[i] - stats->avg;
		var_sum += diff * diff;
	}
	stats->stddev = sqrt(var_sum / stats->count);
}

static void
bench_stats_free(BenchStats *stats)
{
	mkt_free(stats->samples);
}

/*
 * parse_metrics - Parse comma-separated list of metric names
 * Returns bitmask of metrics to run, or METRIC_ALL if "all" specified
 */
static uint32_t
parse_metrics(const char *names)
{
	if (strcmp(names, "all") == 0)
		return METRIC_ALL;

	uint32_t	mask  = 0;
	const char *start = names;

	while (*start)
	{
		const char *end = strchr(start, ',');
		size_t		len = end ? (size_t)(end - start) : strlen(start);

		if (len == 2 && strncmp(start, "l2", 2) == 0)
			mask |= METRIC_L2;
		else if (len == 2 && strncmp(start, "ip", 2) == 0)
			mask |= METRIC_IP;
		else if (len == 6 && strncmp(start, "cosine", 6) == 0)
			mask |= METRIC_COSINE;

		if (end == NULL)
			break;
		start = end + 1;
	}

	return mask ? mask : METRIC_ALL; /* Default to all if empty/invalid */
}

/*
 * metric_name - Get metric name for display
 */
static const char *
metric_name(DistanceMetric metric)
{
	switch (metric)
	{
	case DISTANCE_L2:
		return "L2";
	case DISTANCE_INNER_PRODUCT:
		return "IP";
	case DISTANCE_COSINE:
		return "Cosine";
	default:
		return "Unknown";
	}
}

/*
 * should_test_impl - Check if implementation should be tested
 */
static bool
should_test_impl(const char *impl_name, const char *filter)
{
	if (filter == NULL)
		return true;

	/* Check if impl_name appears in comma-separated filter */
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

/*
 * generate_random_vector - Fill vector with random floats in [-1, 1]
 */
static void
generate_random_vector(float *data, Dimension dim)
{
	for (Dimension i = 0; i < dim; i++)
	{
		/* Use random integers to ensure consistent floating-point
		 * representation */
		int val = rand() % 200 - 100; /* -100 to +99 */
		data[i] = (float)val / 100.0f;
	}
}

/*
 * benchmark_impl_loop - Benchmark with single-pair loop (pgvector-style)
 * (currently unused - kept for debugging)
 */
__attribute__((unused)) static void
benchmark_impl_loop(
		const ImplSpec *impl,
		Dimension		dim,
		uint32_t		count,
		uint32_t		runs,
		DistanceMetric	metric,
		const float	   *query_data,
		const float	   *db_data)
{
	VectorRef query = {.data = query_data, .dim = dim};

	/* Force re-initialization with this SIMD implementation */
	mkt_simd_set_override(impl->mask);
	mkt_simd_reset_cache();
	mkt_distance_force_reinit();

	/* Warm up (ensure code is in cache) */
	for (int i = 0; i < 10; i++)
	{
		VectorRef vec = {.data = db_data, .dim = dim};
		(void)mkt_distance(query, vec, metric);
	}

	/* Benchmark: loop over single-pair calls (pgvector approach) */
	Distance *distances = mkt_alloc(count * sizeof(Distance));

	BenchStats stats;
	bench_stats_init(&stats, runs);

	for (uint32_t run = 0; run < runs; run++)
	{
		uint64_t start = get_time_ns();

		for (uint32_t i = 0; i < count; i++)
		{
			VectorRef vec = {.data = db_data + i * dim, .dim = dim};
			distances[i]  = mkt_distance(query, vec, metric);
		}

		uint64_t end		 = get_time_ns();
		double	 elapsed_ms	 = ns_to_ms(end - start);
		double	 vec_per_sec = (double)count / (elapsed_ms / 1000.0);

		bench_stats_add(&stats, vec_per_sec);
	}

	mkt_free(distances);

	/* Compute statistics */
	bench_stats_compute(&stats);

	double elapsed_ms = (double)count / (stats.avg / 1000.0);

	/* FLOPS calculation: metric-dependent operation count per vector
	 * L2: 3 ops/elem (1 sub, 1 mul, 1 add)
	 * IP: 2 ops/elem (1 mul, 1 add)
	 * Cosine: 6 ops/elem (3 mul, 3 add) + sqrt/div
	 */
	uint64_t ops_per_vec;
	switch (metric)
	{
	case DISTANCE_L2:
		ops_per_vec = (uint64_t)dim * 3;
		break;
	case DISTANCE_INNER_PRODUCT:
		ops_per_vec = (uint64_t)dim * 2;
		break;
	case DISTANCE_COSINE:
		ops_per_vec = (uint64_t)dim * 6 + 2; /* +2 for sqrt and div */
		break;
	default:
		ops_per_vec = (uint64_t)dim;
	}
	double gflops = (stats.avg * ops_per_vec) / 1e9;

	if (runs == 1)
	{
		printf("  %-10s %8.1f ms  (%7.1fK vec/s, %.2f GFLOPS) [loop]\n",
			   impl->name,
			   elapsed_ms,
			   stats.avg / 1000.0,
			   gflops);
	}
	else
	{
		printf("  %-10s %8.1f ms  (%7.1fK vec/s, %.2f GFLOPS) [loop] "
			   "(±%.1f%%, n=%u)\n",
			   impl->name,
			   elapsed_ms,
			   stats.avg / 1000.0,
			   gflops,
			   (stats.stddev / stats.avg) * 100.0,
			   runs);
	}

	bench_stats_free(&stats);

	/* Reset to auto-detection */
	mkt_simd_set_override(0xFFFFFFFF);
	mkt_simd_reset_cache();
}

/*
 * benchmark_impl_batch - Benchmark with batch operation (direct call, no
 * dispatch)
 */
static void
benchmark_impl_batch(
		const ImplSpec *impl,
		Dimension		dim,
		uint32_t		count,
		uint32_t		runs,
		DistanceMetric	metric,
		const float	   *query_data,
		const float	   *db_data)
{
	VectorRef query = {.data = query_data, .dim = dim};

	/* No reinitialization needed - we call implementations directly */

	/* Allocate distances array */
	Distance *distances = mkt_alloc(count * sizeof(Distance));

	/* Warm up with actual batch function (5 iterations) */
	for (int warmup = 0; warmup < 5; warmup++)
	{
		if (impl->mask == SIMD_NONE)
		{
			switch (metric)
			{
			case DISTANCE_L2:
				mkt_distance_batch_l2_compiler(
						query, db_data, count, dim, distances);
				break;
			case DISTANCE_INNER_PRODUCT:
				mkt_distance_batch_ip_compiler(
						query, db_data, count, dim, distances);
				break;
			case DISTANCE_COSINE:
				mkt_distance_batch_cosine_compiler(
						query, db_data, count, dim, distances);
				break;
			}
		}
#if defined(MKT_SIMD_FULL) && (defined(__x86_64__) || defined(_M_X64))
		else if (impl->mask == SIMD_AVX2)
		{
			switch (metric)
			{
			case DISTANCE_L2:
				mkt_distance_batch_l2_avx2(
						query, db_data, count, dim, distances);
				break;
			case DISTANCE_INNER_PRODUCT:
				mkt_distance_batch_ip_avx2(
						query, db_data, count, dim, distances);
				break;
			case DISTANCE_COSINE:
				mkt_distance_batch_cosine_avx2(
						query, db_data, count, dim, distances);
				break;
			}
		}
		else if (impl->mask == MKT_SIMD_AVX512_DQ)
		{
			switch (metric)
			{
			case DISTANCE_L2:
				mkt_distance_batch_l2_avx512(
						query, db_data, count, dim, distances);
				break;
			case DISTANCE_INNER_PRODUCT:
				mkt_distance_batch_ip_avx512(
						query, db_data, count, dim, distances);
				break;
			case DISTANCE_COSINE:
				mkt_distance_batch_cosine_avx512(
						query, db_data, count, dim, distances);
				break;
			}
		}
#endif
	}

	BenchStats stats;
	bench_stats_init(&stats, runs);

	for (uint32_t run = 0; run < runs; run++)
	{
		uint64_t start = get_time_ns();

		/* Call implementation directly based on impl->mask */
		int ret = -1;
		if (impl->mask == SIMD_NONE)
		{
			/* Scalar */
			switch (metric)
			{
			case DISTANCE_L2:
				ret = mkt_distance_batch_l2_compiler(
						query, db_data, count, dim, distances);
				break;
			case DISTANCE_INNER_PRODUCT:
				ret = mkt_distance_batch_ip_compiler(
						query, db_data, count, dim, distances);
				break;
			case DISTANCE_COSINE:
				ret = mkt_distance_batch_cosine_compiler(
						query, db_data, count, dim, distances);
				break;
			}
		}
#if defined(MKT_SIMD_FULL) && (defined(__x86_64__) || defined(_M_X64))
		else if (impl->mask == SIMD_AVX2)
		{
			switch (metric)
			{
			case DISTANCE_L2:
				ret = mkt_distance_batch_l2_avx2(
						query, db_data, count, dim, distances);
				break;
			case DISTANCE_INNER_PRODUCT:
				ret = mkt_distance_batch_ip_avx2(
						query, db_data, count, dim, distances);
				break;
			case DISTANCE_COSINE:
				ret = mkt_distance_batch_cosine_avx2(
						query, db_data, count, dim, distances);
				break;
			}
		}
		else if (impl->mask == MKT_SIMD_AVX512_DQ)
		{
			switch (metric)
			{
			case DISTANCE_L2:
				ret = mkt_distance_batch_l2_avx512(
						query, db_data, count, dim, distances);
				break;
			case DISTANCE_INNER_PRODUCT:
				ret = mkt_distance_batch_ip_avx512(
						query, db_data, count, dim, distances);
				break;
			case DISTANCE_COSINE:
				ret = mkt_distance_batch_cosine_avx512(
						query, db_data, count, dim, distances);
				break;
			}
		}
#endif
#if defined(MKT_SIMD_FULL) && (defined(__aarch64__) || defined(_M_ARM64))
		else if (impl->mask == SIMD_NEON)
		{
			switch (metric)
			{
			case DISTANCE_L2:
				ret = mkt_distance_batch_l2_neon(
						query, db_data, count, dim, distances);
				break;
			case DISTANCE_INNER_PRODUCT:
				ret = mkt_distance_batch_ip_neon(
						query, db_data, count, dim, distances);
				break;
			case DISTANCE_COSINE:
				ret = mkt_distance_batch_cosine_neon(
						query, db_data, count, dim, distances);
				break;
			}
		}
#endif

		if (ret != 0)
		{
			fprintf(stderr, "Distance computation failed\n");
			mkt_free(distances);
			return;
		}

		uint64_t end		 = get_time_ns();
		double	 elapsed_ms	 = ns_to_ms(end - start);
		double	 vec_per_sec = (double)count / (elapsed_ms / 1000.0);

		bench_stats_add(&stats, vec_per_sec);
	}

	mkt_free(distances);

	/* Compute statistics */
	bench_stats_compute(&stats);

	double elapsed_ms = (double)count / (stats.avg / 1000.0);

	/* FLOPS calculation: metric-dependent operation count per vector
	 * L2: 3 ops/elem (1 sub, 1 mul, 1 add)
	 * IP: 2 ops/elem (1 mul, 1 add)
	 * Cosine: 6 ops/elem (3 mul, 3 add) + sqrt/div
	 */
	uint64_t ops_per_vec;
	switch (metric)
	{
	case DISTANCE_L2:
		ops_per_vec = (uint64_t)dim * 3;
		break;
	case DISTANCE_INNER_PRODUCT:
		ops_per_vec = (uint64_t)dim * 2;
		break;
	case DISTANCE_COSINE:
		ops_per_vec = (uint64_t)dim * 6 + 2; /* +2 for sqrt and div */
		break;
	default:
		ops_per_vec = (uint64_t)dim;
	}
	double gflops = (stats.avg * ops_per_vec) / 1e9;

	if (runs == 1)
	{
		printf("  %-10s %8.1f ms  (%7.1fK vec/s, %.2f GFLOPS) [batch]\n",
			   impl->name,
			   elapsed_ms,
			   stats.avg / 1000.0,
			   gflops);
	}
	else
	{
		printf("  %-10s %8.1f ms  (%7.1fK vec/s, %.2f GFLOPS) [batch] "
			   "(±%.1f%%, n=%u)\n",
			   impl->name,
			   elapsed_ms,
			   stats.avg / 1000.0,
			   gflops,
			   (stats.stddev / stats.avg) * 100.0,
			   runs);
	}

	bench_stats_free(&stats);

	/* Reset to auto-detection */
	mkt_simd_set_override(0xFFFFFFFF);
	mkt_simd_reset_cache();
}

/*
 * benchmark_pgvector - Benchmark pgvector-style auto-vectorized implementation
 */
static void
benchmark_pgvector(
		Dimension	   dim,
		uint32_t	   count,
		uint32_t	   runs,
		DistanceMetric metric,
		const float	  *query_data,
		const float	  *db_data)
{
	VectorRef query = {.data = query_data, .dim = dim};

	/* Allocate distances array */
	Distance *distances = mkt_alloc(count * sizeof(Distance));

	/* Warm up with actual batch function (5 iterations) */
	for (int warmup = 0; warmup < 5; warmup++)
	{
		switch (metric)
		{
		case DISTANCE_L2:
			mkt_distance_batch_l2_pgvector(
					query, db_data, count, dim, distances);
			break;
		case DISTANCE_INNER_PRODUCT:
			mkt_distance_batch_ip_pgvector(
					query, db_data, count, dim, distances);
			break;
		case DISTANCE_COSINE:
			mkt_distance_batch_cosine_pgvector(
					query, db_data, count, dim, distances);
			break;
		}
	}

	BenchStats stats;
	bench_stats_init(&stats, runs);

	for (uint32_t run = 0; run < runs; run++)
	{
		uint64_t start = get_time_ns();

		switch (metric)
		{
		case DISTANCE_L2:
			mkt_distance_batch_l2_pgvector(
					query, db_data, count, dim, distances);
			break;
		case DISTANCE_INNER_PRODUCT:
			mkt_distance_batch_ip_pgvector(
					query, db_data, count, dim, distances);
			break;
		case DISTANCE_COSINE:
			mkt_distance_batch_cosine_pgvector(
					query, db_data, count, dim, distances);
			break;
		}

		uint64_t end		 = get_time_ns();
		double	 elapsed_ms	 = ns_to_ms(end - start);
		double	 vec_per_sec = (double)count / (elapsed_ms / 1000.0);

		bench_stats_add(&stats, vec_per_sec);
	}

	mkt_free(distances);

	/* Compute statistics */
	bench_stats_compute(&stats);

	double elapsed_ms = (double)count / (stats.avg / 1000.0);

	/* FLOPS calculation: metric-dependent operation count per vector
	 * L2: 3 ops/elem (1 sub, 1 mul, 1 add)
	 * IP: 2 ops/elem (1 mul, 1 add)
	 * Cosine: 6 ops/elem (3 mul, 3 add) + sqrt/div
	 */
	uint64_t ops_per_vec;
	switch (metric)
	{
	case DISTANCE_L2:
		ops_per_vec = (uint64_t)dim * 3;
		break;
	case DISTANCE_INNER_PRODUCT:
		ops_per_vec = (uint64_t)dim * 2;
		break;
	case DISTANCE_COSINE:
		ops_per_vec = (uint64_t)dim * 6 + 2; /* +2 for sqrt and div */
		break;
	default:
		ops_per_vec = (uint64_t)dim;
	}
	double gflops = (stats.avg * ops_per_vec) / 1e9;

	if (runs == 1)
	{
		printf("  %-10s %8.1f ms  (%7.1fK vec/s, %.2f GFLOPS) "
			   "[batch]\n",
			   "pgvector",
			   elapsed_ms,
			   stats.avg / 1000.0,
			   gflops);
	}
	else
	{
		printf("  %-10s %8.1f ms  (%7.1fK vec/s, %.2f GFLOPS) "
			   "[batch] (±%.1f%%, n=%u)\n",
			   "pgvector",
			   elapsed_ms,
			   stats.avg / 1000.0,
			   gflops,
			   (stats.stddev / stats.avg) * 100.0,
			   runs);
	}

	bench_stats_free(&stats);
}

/*
 * run_benchmark - Run distance benchmark for a single metric
 */
static int
run_benchmark(const BenchConfig *config, DistanceMetric metric)
{
	Dimension dim	= config->dim;
	uint32_t  count = config->count;

	/* Check for unreasonable memory requirements */
	uint64_t query_bytes = (uint64_t)dim * sizeof(float);
	uint64_t db_bytes	 = (uint64_t)count * dim * sizeof(float);
	uint64_t total_bytes = query_bytes + db_bytes;

	/* Warn if > 8GB */
	if (total_bytes > 8ULL * 1024 * 1024 * 1024)
	{
		fprintf(stderr,
				"Warning: This will allocate %.1f GB of memory\n",
				(double)total_bytes / (1024.0 * 1024.0 * 1024.0));
		fprintf(stderr,
				"  Query: %.1f MB, Database: %.1f MB\n",
				(double)query_bytes / (1024.0 * 1024.0),
				(double)db_bytes / (1024.0 * 1024.0));
	}

	/* Hard limit at 64GB to prevent obvious mistakes */
	if (total_bytes > 64ULL * 1024 * 1024 * 1024)
	{
		fprintf(stderr,
				"Error: Memory requirement (%.1f GB) exceeds 64GB limit\n",
				(double)total_bytes / (1024.0 * 1024.0 * 1024.0));
		fprintf(stderr,
				"For large-scale benchmarks, use smaller --count values\n");
		fprintf(stderr,
				"Suggested maximum for dim=%u: --count %u (1GB)\n",
				dim,
				(uint32_t)(1024 * 1024 * 1024 / (dim * sizeof(float))));
		return 1;
	}

	/* Calculate and display dataset size */
	double size_mb = (double)(count * dim * sizeof(float)) / (1024.0 * 1024.0);
	const char *size_unit;
	double		size_value;

	if (size_mb < 1.0)
	{
		size_value = size_mb * 1024.0;
		size_unit  = "KB";
	}
	else if (size_mb < 1024.0)
	{
		size_value = size_mb;
		size_unit  = "MB";
	}
	else
	{
		size_value = size_mb / 1024.0;
		size_unit  = "GB";
	}

	printf("%s distance (dim=%u, count=%u, %.1f %s):\n",
		   metric_name(metric),
		   dim,
		   count,
		   size_value,
		   size_unit);

	/* Generate random test vectors */
	float *query_data = mkt_alloc(dim * sizeof(float));
	if (query_data == NULL)
	{
		fprintf(stderr, "Error: Failed to allocate query vector\n");
		return 1;
	}

	float *db_data = mkt_alloc(count * dim * sizeof(float));
	if (db_data == NULL)
	{
		fprintf(stderr,
				"Error: Failed to allocate database vectors (%.1f GB)\n",
				(double)db_bytes / (1024.0 * 1024.0 * 1024.0));
		mkt_free(query_data);
		return 1;
	}

	srand(42); /* Deterministic */
	generate_random_vector(query_data, dim);
	for (uint32_t i = 0; i < count; i++)
	{
		generate_random_vector(db_data + i * dim, dim);
	}

	/*
	 * CPU warmup: run a throwaway benchmark pass first.
	 * The first implementation measured always appears ~14% slower due to
	 * CPU warmup effects (µop cache, branch predictor). Running a full
	 * benchmark pass (not just a few iterations) as throwaway fixes this.
	 */
	{
		Distance *throwaway = mkt_alloc(count * sizeof(Distance));
		if (throwaway != NULL)
		{
			VectorRef q = {.data = query_data, .dim = dim};

			/* Run full benchmark iterations as throwaway */
			for (uint32_t run = 0; run < config->runs; run++)
			{
				switch (metric)
				{
				case DISTANCE_L2:
					mkt_distance_batch_l2_compiler(
							q, db_data, count, dim, throwaway);
					break;
				case DISTANCE_INNER_PRODUCT:
					mkt_distance_batch_ip_compiler(
							q, db_data, count, dim, throwaway);
					break;
				case DISTANCE_COSINE:
					mkt_distance_batch_cosine_compiler(
							q, db_data, count, dim, throwaway);
					break;
				}
			}
			mkt_free(throwaway);
		}
	}

	/* Benchmark each implementation */
	for (const ImplSpec *impl = impls; impl->name; impl++)
	{
		/* Skip if not in filter */
		if (!should_test_impl(impl->name, config->impl_filter))
			continue;

		/* Skip if not supported by current CPU */
		uint32_t caps = mkt_detect_simd();
		if (impl->mask != SIMD_NONE && (caps & impl->mask) == 0)
		{
			printf("  %-10s (not supported on this CPU)\n", impl->name);
			continue;
		}

		/* Benchmark batch approach (loop approach hidden by default - doesn't
		 * compare well to pgvector since it uses dispatch overhead)
		 */
		benchmark_impl_batch(
				impl, dim, count, config->runs, metric, query_data, db_data);
	}

	/* Benchmark pgvector-style auto-vectorized implementation */
	if (should_test_impl("pgvector", config->impl_filter))
	{
		benchmark_pgvector(
				dim, count, config->runs, metric, query_data, db_data);
	}

	mkt_free(query_data);
	mkt_free(db_data);

	return 0;
}

/*
 * print_usage - Show usage information
 */
static void
print_usage(CmdContext *ctx)
{
	CMD_USAGE_HEADER(ctx, "bench distance");
	printf("Benchmark distance computation performance across SIMD "
		   "implementations.\n\n");
	printf("Options:\n");
	printf("  --dim <int>        Vector dimension (default: %d)\n",
		   DEFAULT_DIM);
	printf("  --count <int>      Number of distance computations (default: "
		   "%d)\n",
		   DEFAULT_COUNT);
	printf("  --runs <int>       Number of benchmark iterations (default: "
		   "3)\n");
	printf("  --metric <list>    Comma-separated metrics: l2, ip, cosine, "
		   "all\n");
	printf("                     (default: all)\n");
	printf("  --impls <list>     Comma-separated implementations to test\n");
	printf("                     Available: compiler, avx2, avx512, neon, "
		   "pgvector\n");
	printf("                     (default: all supported)\n");
	printf("  --help             Show this help message\n");
	printf("\n");
	printf("Examples:\n");
	CMD_USAGE_EXAMPLE(ctx, "bench distance", "--dim 768 --count 10000");
	CMD_USAGE_EXAMPLE(
			ctx, "bench distance", "--metric l2,ip --impls scalar,avx512");
}

/*
 * cmd_bench_distance - Entry point for distance benchmark
 */
int
cmd_bench_distance(CmdContext *ctx)
{
	BenchConfig config = {
			.dim		   = DEFAULT_DIM,
			.count		   = DEFAULT_COUNT,
			.runs		   = 3, /* Default: 3 runs for stable averages */
			.metric_filter = METRIC_ALL, /* Default: run all metrics */
			.impl_filter   = NULL,
			.help		   = false,
	};

	/* Option parsing with getopt_long */
	static struct option long_options[] =
			{{"dim", required_argument, 0, 'd'},
			 {"count", required_argument, 0, 'c'},
			 {"runs", required_argument, 0, 'r'},
			 {"metric", required_argument, 0, 'm'},
			 {"impls", required_argument, 0, 'i'},
			 {"help", no_argument, 0, 'h'},
			 {0, 0, 0, 0}};

	/* Reset getopt for this command */
	optind = 1;

	int opt;
	while ((opt = getopt_long(
					ctx->argc,
					ctx->argv,
					"d:c:r:m:i:h",
					long_options,
					NULL)) != -1)
	{
		switch (opt)
		{
		case 'd':
			config.dim = (Dimension)atoi(optarg);
			if (config.dim == 0)
			{
				fprintf(stderr,
						"Error: Invalid dimension %s (must be > 0)\n",
						optarg);
				return 1;
			}
			break;
		case 'c':
			config.count = (uint32_t)atoi(optarg);
			if (config.count == 0)
			{
				fprintf(stderr,
						"Error: Invalid count %s (must be > 0)\n",
						optarg);
				return 1;
			}
			break;
		case 'r':
			config.runs = (uint32_t)atoi(optarg);
			if (config.runs == 0)
			{
				fprintf(stderr,
						"Error: Invalid runs %s (must be > 0)\n",
						optarg);
				return 1;
			}
			break;
		case 'm':
			config.metric_filter = parse_metrics(optarg);
			break;
		case 'i':
			config.impl_filter = optarg;
			break;
		case 'h':
			config.help = true;
			break;
		default:
			print_usage(ctx);
			return 1;
		}
	}

	if (config.help)
	{
		print_usage(ctx);
		return 0;
	}

	/* Run benchmarks for each metric in the filter */
	int ret = 0;

	if (config.metric_filter & METRIC_L2)
	{
		ret = run_benchmark(&config, DISTANCE_L2);
		if (ret != 0)
			return ret;
		if (config.metric_filter != METRIC_L2)
			printf("\n"); /* Blank line between metrics */
	}

	if (config.metric_filter & METRIC_IP)
	{
		ret = run_benchmark(&config, DISTANCE_INNER_PRODUCT);
		if (ret != 0)
			return ret;
		if (config.metric_filter & METRIC_COSINE)
			printf("\n"); /* Blank line before next metric */
	}

	if (config.metric_filter & METRIC_COSINE)
	{
		ret = run_benchmark(&config, DISTANCE_COSINE);
		if (ret != 0)
			return ret;
	}

	return 0;
}
