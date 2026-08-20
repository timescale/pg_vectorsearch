/*
 * mkt bench page-score
 *
 * Benchmark page-level RaBitQ scoring: sequential vs vertical
 * (multi-candidate) inner product. Compares batch (separate arrays)
 * and AoS (contiguous RaBitQData) distance computation layouts.
 */

#include "mkt_config.h"

#include <getopt.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cmd.h"
#include "core/memory.h"
#include "core/platform.h"
#include "core/types.h"
#include "quant/rabitq.h"
#include "types/vector.h"

/* Default parameters */
#define DEFAULT_DIM	  768
#define DEFAULT_COUNT 200
#define DEFAULT_RUNS  1000

/* SIMD implementations */
typedef struct
{
	const char *name;
	uint32_t	mask;
} ImplSpec;

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

/* Benchmark configuration */
typedef struct
{
	Dimension	dim;
	uint32_t	count;
	uint32_t	runs;
	const char *impl_filter;
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

/* Timing statistics */
typedef struct
{
	double	*samples;
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

static void
bench_stats_free(BenchStats *stats)
{
	mkt_free(stats->samples);
}

/*
 * should_test_impl - Check if implementation should be tested
 */
static bool
should_test_impl(const char *impl_name, const char *filter)
{
	if (filter == NULL)
		return true;

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
		int val = rand() % 200 - 100;
		data[i] = (float)val / 100.0f;
	}
}

/*
 * Helper to re-initialize RaBitQ SIMD dispatch with specific override.
 */
static void
reinit_rabitq_with_simd(uint32_t simd_mask)
{
	mkt_simd_set_override(simd_mask);
	mkt_simd_reset_cache();
	mkt_rabitq_force_reinit();
	mkt_rabitq_init_simd();
}

/*
 * run_one_bench - Time a single function for N runs
 */
static void
run_one_bench(BenchStats *stats, uint32_t runs, void (*fn)(void *), void *arg)
{
	/* Warmup */
	for (int w = 0; w < 5; w++)
		fn(arg);

	for (uint32_t run = 0; run < runs; run++)
	{
		uint64_t start = get_time_ns();
		fn(arg);
		uint64_t end = get_time_ns();
		bench_stats_add(stats, (double)(end - start));
	}
	bench_stats_compute(stats);
}

/* Callback context for batch (separate arrays) benchmarks */
typedef struct
{
	RaBitQQueryState *qstate;
	const float		 *f_add;
	const float		 *f_rescale;
	const uint8_t	 *bits;
	uint32_t		  count;
	Dimension		  dim;
	Distance		 *distances;
} BatchBenchCtx;

static void
bench_batch_sequential(void *arg)
{
	BatchBenchCtx *c = arg;
	mkt_rabitq_distance_batch(
			c->qstate,
			c->f_add,
			c->f_rescale,
			c->bits,
			c->count,
			c->dim,
			c->distances);
}

static void
bench_batch_vertical(void *arg)
{
	BatchBenchCtx *c = arg;
	mkt_rabitq_distance_batch_multi(
			c->qstate,
			c->f_add,
			c->f_rescale,
			c->bits,
			c->count,
			c->dim,
			c->distances);
}

/* Callback context for AoS benchmarks */
typedef struct
{
	RaBitQQueryState *qstate;
	uint8_t			 *aos_data;	  /* packed AoS entries */
	uint32_t		  entry_size; /* bytes per AoS entry */
	uint32_t		  count;
	Dimension		  dim;
	Distance		 *distances;
} AoSBenchCtx;

static void
bench_aos_sequential(void *arg)
{
	AoSBenchCtx *c = arg;
	for (uint32_t i = 0; i < c->count; i++)
	{
		RaBitQData *d	= (RaBitQData *)(c->aos_data +
										 (size_t)i * c->entry_size);
		c->distances[i] = mkt_rabitq_distance(c->qstate, d, c->dim);
	}
}

static void
bench_aos_vertical(void *arg)
{
	AoSBenchCtx	  *c			= arg;
	uint32_t	   packed_bytes = MKT_RABITQ_BYTES(c->dim);
	uint32_t	   bits_offset	= offsetof(RaBitQData, bits);
	const uint8_t *bits_base	= c->aos_data + bits_offset;

	/* Vertical inner product with AoS stride */
	float *ips = mkt_alloc_aligned(c->count * sizeof(float), 64);
	mkt_rabitq_inner_product_multi(
			c->qstate->transformed,
			bits_base,
			c->entry_size,
			c->dim,
			c->count,
			ips);

	/* Distance formula with strided f_add/f_rescale access */
	float g_add		 = c->qstate->g_add;
	float sum_t		 = c->qstate->sum_transformed;
	float inv_sqrt_d = c->qstate->inv_sqrt_d;

	for (uint32_t i = 0; i < c->count; i++)
	{
		RaBitQData *d		  = (RaBitQData *)(c->aos_data +
									   (size_t)i * c->entry_size);
		float		final_dot = (2.0f * ips[i] - sum_t) * inv_sqrt_d;
		c->distances[i] = d->f_add + g_add - 2.0f * d->f_rescale * final_dot;
	}

	mkt_free_aligned(ips);
	(void)packed_bytes;
}

/*
 * run_benchmark - Run page-score benchmark for a single SIMD impl
 */
static void
run_benchmark(
		const BenchConfig *config,
		const ImplSpec	  *impl,
		const float		  *f_add,
		const float		  *f_rescale,
		const uint8_t	  *bits,
		uint8_t			  *aos_data,
		uint32_t		   aos_entry_size,
		RaBitQQueryState  *qstate,
		Distance		  *ref_distances)
{
	uint32_t  count = config->count;
	Dimension dim	= config->dim;

	(void)ref_distances;

	/* Select SIMD implementation */
	reinit_rabitq_with_simd(impl->mask);

	Distance *distances = mkt_alloc_aligned(count * sizeof(Distance), 64);

	/* Batch (separate arrays) benchmark contexts */
	BatchBenchCtx batch_ctx = {
			.qstate	   = qstate,
			.f_add	   = f_add,
			.f_rescale = f_rescale,
			.bits	   = bits,
			.count	   = count,
			.dim	   = dim,
			.distances = distances,
	};

	/* AoS benchmark contexts */
	AoSBenchCtx aos_ctx = {
			.qstate		= qstate,
			.aos_data	= aos_data,
			.entry_size = aos_entry_size,
			.count		= count,
			.dim		= dim,
			.distances	= distances,
	};

	/* Run all 4 benchmarks */
	BenchStats batch_seq, batch_vert, aos_seq, aos_vert;
	bench_stats_init(&batch_seq, config->runs);
	bench_stats_init(&batch_vert, config->runs);
	bench_stats_init(&aos_seq, config->runs);
	bench_stats_init(&aos_vert, config->runs);

	run_one_bench(
			&batch_seq, config->runs, bench_batch_sequential, &batch_ctx);
	run_one_bench(&batch_vert, config->runs, bench_batch_vertical, &batch_ctx);
	run_one_bench(&aos_seq, config->runs, bench_aos_sequential, &aos_ctx);
	run_one_bench(&aos_vert, config->runs, bench_aos_vertical, &aos_ctx);

	/* Report results */
	printf("  %-10s  batch seq: %7.0f ns  "
		   "batch vert: %7.0f ns  "
		   "AoS seq: %7.0f ns  "
		   "AoS vert: %7.0f ns",
		   impl->name,
		   batch_seq.avg,
		   batch_vert.avg,
		   aos_seq.avg,
		   aos_vert.avg);

	if (config->runs > 1)
		printf("  (n=%u)", config->runs);

	printf("\n");

	bench_stats_free(&aos_vert);
	bench_stats_free(&aos_seq);
	bench_stats_free(&batch_vert);
	bench_stats_free(&batch_seq);
	mkt_free_aligned(distances);
}

/*
 * print_usage - Show usage information
 */
static void
print_usage(CmdContext *ctx)
{
	CMD_USAGE_HEADER(ctx, "bench page-score");
	printf("Benchmark page-level RaBitQ scoring: sequential vs "
		   "vertical SIMD.\n\n");
	printf("Options:\n");
	printf("  --dim <int>        Vector dimension (default: %d)\n",
		   DEFAULT_DIM);
	printf("  --count <int>      Centroids per page (default: %d)\n",
		   DEFAULT_COUNT);
	printf("  --runs <int>       Benchmark iterations "
		   "(default: %d)\n",
		   DEFAULT_RUNS);
	printf("  --impls <list>     Comma-separated implementations "
		   "to test\n");
	printf("                     Available: compiler, avx2, avx512, "
		   "neon\n");
	printf("                     (default: all supported)\n");
	printf("  --help             Show this help message\n");
	printf("\n");
	printf("Examples:\n");
	CMD_USAGE_EXAMPLE(ctx, "bench page-score", "--dim 768");
	CMD_USAGE_EXAMPLE(
			ctx, "bench page-score", "--dim 768 --count 100 --impls avx512");
}

/*
 * cmd_bench_page_score - Entry point for page-score benchmark
 */
int
cmd_bench_page_score(CmdContext *ctx)
{
	BenchConfig config = {
			.dim		 = DEFAULT_DIM,
			.count		 = DEFAULT_COUNT,
			.runs		 = DEFAULT_RUNS,
			.impl_filter = NULL,
			.help		 = false,
	};

	static struct option long_options[] =
			{{"dim", required_argument, 0, 'd'},
			 {"count", required_argument, 0, 'c'},
			 {"runs", required_argument, 0, 'r'},
			 {"impls", required_argument, 0, 'i'},
			 {"help", no_argument, 0, 'h'},
			 {0, 0, 0, 0}};

	optind = 1;

	int opt;
	while ((opt = getopt_long(
					ctx->argc, ctx->argv, "d:c:r:i:h", long_options, NULL)) !=
		   -1)
	{
		switch (opt)
		{
		case 'd':
			config.dim = (Dimension)atoi(optarg);
			if (config.dim == 0)
			{
				fprintf(stderr, "Error: Invalid dimension %s\n", optarg);
				return 1;
			}
			break;
		case 'c':
			config.count = (uint32_t)atoi(optarg);
			if (config.count == 0)
			{
				fprintf(stderr, "Error: Invalid count %s\n", optarg);
				return 1;
			}
			break;
		case 'r':
			config.runs = (uint32_t)atoi(optarg);
			if (config.runs == 0)
			{
				fprintf(stderr, "Error: Invalid runs %s\n", optarg);
				return 1;
			}
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

	Dimension dim	= config.dim;
	uint32_t  count = config.count;

	uint32_t packed_bytes = MKT_RABITQ_BYTES(dim);
	double	 bits_kb	  = (double)count * packed_bytes / 1024.0;

	/* Generate random test data */
	srand(42);

	/* Create RaBitQ params and encode vectors into separate arrays */
	RaBitQParams *params = mkt_rabitq_create(dim, 42);
	if (params == NULL)
	{
		fprintf(stderr, "Error: Failed to create RaBitQ params\n");
		return 1;
	}

	/* Generate random centroid and vectors */
	float *centroid = mkt_alloc(dim * sizeof(float));
	generate_random_vector(centroid, dim);
	VectorRef cent_ref = {.data = centroid, .dim = dim};

	float *vectors = mkt_alloc((size_t)count * dim * sizeof(float));
	for (uint32_t i = 0; i < count; i++)
		generate_random_vector(vectors + i * dim, dim);

	/* Encode into separate arrays */
	float	*f_add	   = mkt_alloc_aligned(count * sizeof(float), 64);
	float	*f_rescale = mkt_alloc_aligned(count * sizeof(float), 64);
	uint8_t *bits	   = mkt_alloc_aligned((size_t)count * packed_bytes, 64);

	int ret = mkt_rabitq_encode_batch(
			params,
			vectors,
			MKT_VEC_F32,
			cent_ref,
			f_add,
			f_rescale,
			bits,
			(uint16_t)(count > UINT16_MAX ? UINT16_MAX : count));

	if (ret != 0)
	{
		fprintf(stderr, "Error: Batch encoding failed\n");
		mkt_free(vectors);
		mkt_free(centroid);
		mkt_rabitq_destroy(params);
		return 1;
	}

	/* Generate random query and prepare query state */
	float *query = mkt_alloc(dim * sizeof(float));
	generate_random_vector(query, dim);
	VectorRef query_ref = {.data = query, .dim = dim};

	RaBitQQueryState *qstate =
			mkt_rabitq_prepare_query(params, query_ref, cent_ref);
	if (qstate == NULL)
	{
		fprintf(stderr, "Error: Failed to prepare query state\n");
		mkt_free_aligned(bits);
		mkt_free_aligned(f_rescale);
		mkt_free_aligned(f_add);
		mkt_free(vectors);
		mkt_free(query);
		mkt_free(centroid);
		mkt_rabitq_destroy(params);
		return 1;
	}

	/* Build AoS array from separate arrays */
	uint32_t aos_entry_size = MKT_RABITQ_DATA_SIZE(dim);
	uint8_t *aos_data = mkt_alloc_aligned((size_t)count * aos_entry_size, 64);
	for (uint32_t i = 0; i < count; i++)
	{
		RaBitQData *entry = (RaBitQData *)(aos_data +
										   (size_t)i * aos_entry_size);
		entry->f_add	  = f_add[i];
		entry->f_rescale  = f_rescale[i];
		memcpy(entry->bits, bits + (size_t)i * packed_bytes, packed_bytes);
	}

	double aos_kb = (double)count * aos_entry_size / 1024.0;
	printf("Page score (dim=%u, count=%u, "
		   "batch: %.1f KB bits, AoS: %.1f KB, %u B/vec):\n",
		   dim,
		   count,
		   bits_kb,
		   aos_kb,
		   packed_bytes);

	/* Compute reference distances for correctness checks */
	Distance *ref_distances = mkt_alloc_aligned(count * sizeof(Distance), 64);
	mkt_rabitq_distance_batch(
			qstate, f_add, f_rescale, bits, count, dim, ref_distances);

	/*
	 * CPU warmup: run a throwaway benchmark pass to stabilize
	 * CPU frequency and fill branch predictor / µop cache.
	 */
	{
		Distance *throwaway = mkt_alloc_aligned(count * sizeof(Distance), 64);
		if (throwaway != NULL)
		{
			for (uint32_t w = 0; w < config.runs; w++)
			{
				mkt_rabitq_distance_batch(
						qstate, f_add, f_rescale, bits, count, dim, throwaway);
			}
			mkt_free_aligned(throwaway);
		}
	}

	/* Detect actual CPU capabilities before benchmarking */
	reinit_rabitq_with_simd(0xFFFFFFFF);
	uint32_t cpu_caps = mkt_detect_simd();

	/* Benchmark each SIMD implementation */
	for (const ImplSpec *impl = impls; impl->name; impl++)
	{
		if (!should_test_impl(impl->name, config.impl_filter))
			continue;

		if (impl->mask != SIMD_NONE && (cpu_caps & impl->mask) == 0)
		{
			printf("  %-10s  (not supported on this CPU)\n", impl->name);
			continue;
		}

		run_benchmark(
				&config,
				impl,
				f_add,
				f_rescale,
				bits,
				aos_data,
				aos_entry_size,
				qstate,
				ref_distances);
	}

	/* Cleanup */
	mkt_free_aligned(aos_data);
	mkt_free_aligned(ref_distances);
	mkt_rabitq_free_query(qstate);
	mkt_free_aligned(bits);
	mkt_free_aligned(f_rescale);
	mkt_free_aligned(f_add);
	mkt_free(vectors);
	mkt_free(query);
	mkt_free(centroid);
	mkt_rabitq_destroy(params);

	/* Reset SIMD to auto-detection */
	reinit_rabitq_with_simd(0xFFFFFFFF);

	return 0;
}
