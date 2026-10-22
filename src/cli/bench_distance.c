/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * vectorsearch bench distance
 *
 * Benchmark distance computations across SIMD implementations.
 */

/* Must be first - defines VS_SIMD_FULL used by distance.h */
#include "vs_config.h"

#include <getopt.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "algo/distance.h"
#include "bench.h"
#include "cmd.h"
#include "core/memory.h"
#include "core/platform.h"
#include "core/types.h"
#include "distance_pgvector.h"
#include "kv.h"
#include "types/vec32.h"

/* Default parameters */
#define DEFAULT_DIM	  768
#define DEFAULT_COUNT 10000

/*
 * Available implementations depend on build mode and architecture.
 * - simd=full: compiler + hand-optimized (avx2/avx512 on x86, neon on ARM)
 * - simd=compiler or simd=none: compiler only
 */
static const ImplSpec impls[] = {
		{"compiler", SIMD_NONE},
#if defined(VS_SIMD_FULL) && (defined(__x86_64__) || defined(_M_X64))
		{"avx2", SIMD_AVX2},
		{"avx512", VS_SIMD_AVX512_DQ},
#endif
#if defined(VS_SIMD_FULL) && (defined(__aarch64__) || defined(_M_ARM64))
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
	Kv		   *metrics; /* NULL unless --json */
} BenchConfig;

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

static const char *
metric_key(DistanceMetric metric)
{
	switch (metric)
	{
	case DISTANCE_L2:
		return "l2";
	case DISTANCE_INNER_PRODUCT:
		return "ip";
	case DISTANCE_COSINE:
		return "cosine";
	default:
		return "unknown";
	}
}

/* Seeded once per command, below. */
static uint32_t g_rng;

/*
 * Marks the pgvector-style reference among the SIMD masks. Not a
 * capability bit: SIMD_NONE already means the compiler build, so the
 * reference needs a value of its own that no capability can take.
 */
#define MASK_PGVECTOR 0xFFFFFFFEu

/*
 * batch_call - Run one batch pass for a target
 *
 * Called directly rather than through vs_distance dispatch, so what is
 * timed is the kernel and not the dispatch.
 */
static int
batch_call(
		uint32_t	   mask,
		DistanceMetric metric,
		Vec32Ref	   query,
		const float	  *db_data,
		uint32_t	   count,
		Dimension	   dim,
		Distance	  *distances)
{
	if (mask == MASK_PGVECTOR)
	{
		switch (metric)
		{
		case DISTANCE_L2:
			return vs_distance_batch_l2_pgvector(
					query, db_data, count, dim, distances);
		case DISTANCE_INNER_PRODUCT:
			return vs_distance_batch_ip_pgvector(
					query, db_data, count, dim, distances);
		case DISTANCE_COSINE:
			return vs_distance_batch_cosine_pgvector(
					query, db_data, count, dim, distances);
		}
		return -1;
	}

	if (mask == SIMD_NONE)
	{
		switch (metric)
		{
		case DISTANCE_L2:
			return vs_distance_batch_l2_compiler(
					query, db_data, count, dim, distances);
		case DISTANCE_INNER_PRODUCT:
			return vs_distance_batch_ip_compiler(
					query, db_data, count, dim, distances);
		case DISTANCE_COSINE:
			return vs_distance_batch_cosine_compiler(
					query, db_data, count, dim, distances);
		}
		return -1;
	}

#if defined(VS_SIMD_FULL) && (defined(__x86_64__) || defined(_M_X64))
	if (mask == SIMD_AVX2)
	{
		switch (metric)
		{
		case DISTANCE_L2:
			return vs_distance_batch_l2_avx2(
					query, db_data, count, dim, distances);
		case DISTANCE_INNER_PRODUCT:
			return vs_distance_batch_ip_avx2(
					query, db_data, count, dim, distances);
		case DISTANCE_COSINE:
			return vs_distance_batch_cosine_avx2(
					query, db_data, count, dim, distances);
		}
		return -1;
	}
	if (mask == VS_SIMD_AVX512_DQ)
	{
		switch (metric)
		{
		case DISTANCE_L2:
			return vs_distance_batch_l2_avx512(
					query, db_data, count, dim, distances);
		case DISTANCE_INNER_PRODUCT:
			return vs_distance_batch_ip_avx512(
					query, db_data, count, dim, distances);
		case DISTANCE_COSINE:
			return vs_distance_batch_cosine_avx512(
					query, db_data, count, dim, distances);
		}
		return -1;
	}
#endif
#if defined(VS_SIMD_FULL) && (defined(__aarch64__) || defined(_M_ARM64))
	if (mask == SIMD_NEON)
	{
		switch (metric)
		{
		case DISTANCE_L2:
			return vs_distance_batch_l2_neon(
					query, db_data, count, dim, distances);
		case DISTANCE_INNER_PRODUCT:
			return vs_distance_batch_ip_neon(
					query, db_data, count, dim, distances);
		case DISTANCE_COSINE:
			return vs_distance_batch_cosine_neon(
					query, db_data, count, dim, distances);
		}
		return -1;
	}
#endif

	return -1;
}

/*
 * ops_per_vector - Operation count per vector, for the GFLOPS column
 *
 * L2: 3 ops/elem (sub, mul, add). IP: 2 (mul, add).
 * Cosine: 6 (3 mul, 3 add) plus a sqrt and a divide.
 */
static uint64_t
ops_per_vector(DistanceMetric metric, Dimension dim)
{
	switch (metric)
	{
	case DISTANCE_L2:
		return (uint64_t)dim * 3;
	case DISTANCE_INNER_PRODUCT:
		return (uint64_t)dim * 2;
	case DISTANCE_COSINE:
		return (uint64_t)dim * 6 + 2;
	default:
		return (uint64_t)dim;
	}
}

/*
 * Everything the pass and report callbacks need. The query, the
 * vectors and the output buffer are shared by every case: the cases
 * differ only in which kernel runs over them.
 */
typedef struct
{
	Vec32Ref	   query;
	const float	  *db_data;
	uint32_t	   count;
	Dimension	   dim;
	DistanceMetric metric;
	Distance	  *distances;
	Kv			  *metrics;
} DistanceCtx;

static int
distance_pass(void *vctx, const BenchCase *bench_case)
{
	DistanceCtx *ctx = vctx;

	return batch_call(
			bench_case->variant,
			ctx->metric,
			ctx->query,
			ctx->db_data,
			ctx->count,
			ctx->dim,
			ctx->distances);
}

static void
distance_report(void *vctx, const BenchCase *bench_case, uint32_t runs)
{
	DistanceCtx *ctx	 = vctx;
	double		 seconds = bench_seconds_per_pass(bench_case);
	double		 vps	 = seconds > 0.0 ? (double)ctx->count / seconds : 0.0;
	double		 gflops	 = (vps * ops_per_vector(ctx->metric, ctx->dim)) / 1e9;

	if (runs == 1)
	{
		printf("  %-10s %8.1f ms  (%7.1fK vec/s, %.2f GFLOPS) [batch]\n",
			   bench_case->name,
			   seconds * 1000.0,
			   vps / 1000.0,
			   gflops);
	}
	else
	{
		printf("  %-10s %8.1f ms  (%7.1fK vec/s, %.2f GFLOPS) [batch] "
			   "(±%.1f%%, n=%u)\n",
			   bench_case->name,
			   seconds * 1000.0,
			   vps / 1000.0,
			   gflops,
			   bench_rsd_percent(bench_case),
			   runs);
	}

	bench_record_rate(
			ctx->metrics,
			metric_key(ctx->metric),
			bench_case,
			"vec/s",
			vps,
			runs);
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
	double size_mb = (double)((size_t)count * dim * sizeof(float)) / 1024.0 /
					 1024.0;
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
		   bench_metric_name(metric),
		   dim,
		   count,
		   size_value,
		   size_unit);

	/* Generate random test vectors */
	float *query_data = vs_alloc(dim * sizeof(float));
	if (query_data == NULL)
	{
		fprintf(stderr, "Error: Failed to allocate query vector\n");
		return 1;
	}

	float *db_data = vs_alloc((size_t)count * dim * sizeof(float));
	if (db_data == NULL)
	{
		fprintf(stderr,
				"Error: Failed to allocate database vectors (%.1f GB)\n",
				(double)db_bytes / (1024.0 * 1024.0 * 1024.0));
		vs_free(query_data);
		return 1;
	}

	g_rng = 42; /* Deterministic */
	bench_fill_random(query_data, dim, &g_rng);
	for (uint32_t i = 0; i < count; i++)
	{
		bench_fill_random(db_data + i * dim, dim, &g_rng);
	}

	/* Collect the cases to compare; bench_run owns the timing. */
	BenchCase cases[8];
	uint32_t  ncases = 0;

	for (const ImplSpec *impl = impls; impl->name; impl++)
	{
		if (!bench_impl_selected(impl->name, config->impl_filter))
			continue;

		/* Skip if not supported by current CPU */
		uint32_t caps = vs_detect_simd();
		if (impl->mask != SIMD_NONE && (caps & impl->mask) == 0)
		{
			printf("  %-10s (not supported on this CPU)\n", impl->name);
			continue;
		}

		cases[ncases].name	  = impl->name;
		cases[ncases].variant = impl->mask;
		ncases++;
	}

	if (bench_impl_selected("pgvector", config->impl_filter))
	{
		cases[ncases].name	  = "pgvector";
		cases[ncases].variant = MASK_PGVECTOR;
		ncases++;
	}

	if (ncases > 0)
	{
		Distance *distances = vs_alloc(count * sizeof(Distance));

		if (distances == NULL)
		{
			fprintf(stderr, "Error: Failed to allocate distances\n");
			vs_free(query_data);
			vs_free(db_data);
			return 1;
		}

		DistanceCtx ctx = {
				.query	   = {.data = query_data, .dim = dim},
				.db_data   = db_data,
				.count	   = count,
				.dim	   = dim,
				.metric	   = metric,
				.distances = distances,
				.metrics   = config->metrics,
		};

		BenchSpec spec = {
				.runs		   = config->runs,
				.warmup_passes = 5,
				.pass		   = distance_pass,
				.report		   = distance_report,
				.ctx		   = &ctx,
		};

		int ret = bench_run(&spec, cases, ncases);

		vs_free(distances);

		/* Reset to auto-detection */
		vs_simd_set_override(0xFFFFFFFF);
		vs_simd_reset_cache();

		if (ret != 0)
		{
			fprintf(stderr, "Distance computation failed\n");
			vs_free(query_data);
			vs_free(db_data);
			return 1;
		}
	}

	vs_free(query_data);
	vs_free(db_data);

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
	CMD_USAGE_JSON();
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

	KvBuild *kb	  = NULL;
	Kv		*root = NULL;
	if (ctx->json_path != NULL)
	{
		kb	 = kv_build_open();
		root = kv_obj();
		kv_obj_put(root, "bench", kv_str("distance"));
		kv_obj_put(root, "dim", kv_int(config.dim));
		kv_obj_put(root, "count", kv_int(config.count));
		kv_obj_put(root, "runs", kv_int(config.runs));
		config.metrics = kv_arr();
		kv_obj_put(root, "metrics", config.metrics);
	}

	/* Run benchmarks for each metric in the filter */
	int ret = 0;

	if (config.metric_filter & METRIC_L2)
	{
		ret = run_benchmark(&config, DISTANCE_L2);
		if (ret != 0)
		{
			kv_build_close(kb);
			return ret;
		}
		if (config.metric_filter != METRIC_L2)
			printf("\n"); /* Blank line between metrics */
	}

	if (config.metric_filter & METRIC_IP)
	{
		ret = run_benchmark(&config, DISTANCE_INNER_PRODUCT);
		if (ret != 0)
		{
			kv_build_close(kb);
			return ret;
		}
		if (config.metric_filter & METRIC_COSINE)
			printf("\n"); /* Blank line before next metric */
	}

	if (config.metric_filter & METRIC_COSINE)
	{
		ret = run_benchmark(&config, DISTANCE_COSINE);
		if (ret != 0)
		{
			kv_build_close(kb);
			return ret;
		}
	}

	int rc = cmd_kv_print(root, ctx->json_path);
	kv_build_close(kb);
	return rc;
}
