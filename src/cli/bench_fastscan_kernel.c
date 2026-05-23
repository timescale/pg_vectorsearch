/*
 * mkt bench fastscan-kernel
 *
 * Benchmark VPSHUFB fastscan kernel components:
 * - LUT construction (uint8 and uint16 hacc)
 * - Code packing
 * - Accumulate kernel (the hot inner loop)
 * - Full distance batch (LUT + accumulate + distance formula)
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
#include "mkt_types.h"
#include "quant/fastscan.h"
#include "quant/rabitq.h"

#define DEFAULT_DIM	  768
#define DEFAULT_COUNT 10000
#define DEFAULT_RUNS  5

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

typedef struct
{
	Dimension	dim;
	uint32_t	count;
	uint32_t	runs;
	const char *simd;
	bool		help;
} BenchConfig;

static void
fill_random(void *buf, size_t bytes, uint32_t seed)
{
	srand(seed);
	uint8_t *p = buf;
	for (size_t i = 0; i < bytes; i++)
		p[i] = (uint8_t)(rand() & 0xFF);
}

static void
fill_random_floats(float *buf, uint32_t n, uint32_t seed)
{
	srand(seed);
	for (uint32_t i = 0; i < n; i++)
		buf[i] = (float)(rand() % 10000 - 5000) / 5000.0f;
}

/* ----------------------------------------------------------------
 * LUT construction benchmark
 * ---------------------------------------------------------------- */

static void
benchmark_lut_build(const BenchConfig *config)
{
	Dimension dim	  = config->dim;
	uint32_t  queries = 1000;

	printf("LUT construction (dim=%u, %u queries):\n", dim, queries);

	float *transformed = mkt_alloc(dim * sizeof(float));

	uint32_t lut_bytes = MKT_FASTSCAN_LUT_BYTES(dim);
	uint8_t *lut	   = mkt_alloc_aligned(lut_bytes, 64);
	float	 scale, bias;

	uint32_t lut_hacc_bytes = MKT_FASTSCAN_LUT_HACC_BYTES(dim);
	uint8_t *lut_hacc		= mkt_alloc_aligned(lut_hacc_bytes, 64);

	/* Benchmark uint8 LUT */
	double best_vps = 0.0;
	for (uint32_t run = 0; run < config->runs; run++)
	{
		uint64_t start = get_time_ns();
		for (uint32_t q = 0; q < queries; q++)
		{
			fill_random_floats(transformed, dim, 42 + q);
			mkt_fastscan_build_lut(transformed, dim, lut, &scale, &bias);
		}
		uint64_t end = get_time_ns();
		double	 vps = (double)queries / (ns_to_ms(end - start) / 1000.0);
		if (vps > best_vps)
			best_vps = vps;
	}
	printf("  uint8 LUT:   %7.0fK builds/s  (%u nsq, %u bytes)\n",
		   best_vps / 1000.0,
		   MKT_FASTSCAN_NSQ(dim),
		   lut_bytes);

	/* Benchmark uint16 hacc LUT */
	best_vps = 0.0;
	for (uint32_t run = 0; run < config->runs; run++)
	{
		uint64_t start = get_time_ns();
		for (uint32_t q = 0; q < queries; q++)
		{
			fill_random_floats(transformed, dim, 42 + q);
			mkt_fastscan_build_lut_hacc(
					transformed, dim, lut_hacc, &scale, &bias);
		}
		uint64_t end = get_time_ns();
		double	 vps = (double)queries / (ns_to_ms(end - start) / 1000.0);
		if (vps > best_vps)
			best_vps = vps;
	}
	printf("  uint16 LUT:  %7.0fK builds/s  (%u nsq, %u bytes)\n",
		   best_vps / 1000.0,
		   MKT_FASTSCAN_NSQ(dim),
		   lut_hacc_bytes);

	mkt_free_aligned(lut_hacc);
	mkt_free_aligned(lut);
	mkt_free(transformed);
}

/* ----------------------------------------------------------------
 * Accumulate kernel benchmark
 * ---------------------------------------------------------------- */

static void
benchmark_accumulate(const BenchConfig *config)
{
	Dimension dim		   = config->dim;
	uint32_t  count		   = config->count;
	uint32_t  packed_bytes = (dim + 7) / 8;
	uint32_t  ngroups = (count + MKT_FASTSCAN_GROUP - 1) / MKT_FASTSCAN_GROUP;
	uint32_t  group_bytes = MKT_FASTSCAN_GROUP_BYTES(dim);

	printf("\nAccumulate kernel (dim=%u, %u vectors, %u groups):\n",
		   dim,
		   count,
		   ngroups);

	float *transformed = mkt_alloc(dim * sizeof(float));
	fill_random_floats(transformed, dim, 42);

	uint8_t *bits = mkt_alloc((size_t)count * packed_bytes);
	fill_random(bits, (size_t)count * packed_bytes, 99);

	/* Build LUT */
	uint32_t lut_bytes = MKT_FASTSCAN_LUT_BYTES(dim);
	uint8_t *lut	   = mkt_alloc_aligned(lut_bytes, 64);
	float	 scale, bias;
	mkt_fastscan_build_lut(transformed, dim, lut, &scale, &bias);

	/* Pack codes */
	uint32_t codes_size = mkt_fastscan_codes_size(count, dim);
	uint8_t *codes		= mkt_alloc_aligned(codes_size, 64);
	mkt_fastscan_pack_codes(bits, count, dim, codes);

	uint16_t accum[MKT_FASTSCAN_GROUP];

	/* Warmup */
	for (int w = 0; w < 3; w++)
		for (uint32_t g = 0; g < ngroups; g++)
			mkt_fastscan_accumulate(codes + g * group_bytes, lut, accum, dim);

	/* Benchmark uint8 accumulate */
	double best_vps = 0.0;
	for (uint32_t run = 0; run < config->runs; run++)
	{
		uint64_t start = get_time_ns();
		for (uint32_t g = 0; g < ngroups; g++)
			mkt_fastscan_accumulate(codes + g * group_bytes, lut, accum, dim);
		uint64_t end = get_time_ns();
		double	 vps = (double)count / (ns_to_ms(end - start) / 1000.0);
		if (vps > best_vps)
			best_vps = vps;
	}
	printf("  uint8:   %7.1fM vec/s  (%5.2f ms/%uk)\n",
		   best_vps / 1e6,
		   (double)count / best_vps * 1000.0,
		   count / 1000);

	/* Benchmark uint16 hacc accumulate */
	uint32_t lut_hacc_bytes = MKT_FASTSCAN_LUT_HACC_BYTES(dim);
	uint8_t *lut_hacc		= mkt_alloc_aligned(lut_hacc_bytes, 64);
	mkt_fastscan_build_lut_hacc(transformed, dim, lut_hacc, &scale, &bias);
	int32_t accum_hacc[MKT_FASTSCAN_GROUP];

	best_vps = 0.0;
	for (uint32_t run = 0; run < config->runs; run++)
	{
		uint64_t start = get_time_ns();
		for (uint32_t g = 0; g < ngroups; g++)
			mkt_fastscan_accumulate_hacc(
					codes + g * group_bytes, lut_hacc, accum_hacc, dim);
		uint64_t end = get_time_ns();
		double	 vps = (double)count / (ns_to_ms(end - start) / 1000.0);
		if (vps > best_vps)
			best_vps = vps;
	}
	printf("  uint16:  %7.1fM vec/s  (%5.2f ms/%uk)\n",
		   best_vps / 1e6,
		   (double)count / best_vps * 1000.0,
		   count / 1000);

	mkt_free_aligned(lut_hacc);
	mkt_free_aligned(codes);
	mkt_free_aligned(lut);
	mkt_free(bits);
	mkt_free(transformed);
}

/* ----------------------------------------------------------------
 * Distance throughput: fastscan vs rabitq asymmetric
 *
 * Uses properly encoded RaBitQ vectors so both kernels operate
 * on the same data. Output matches bench rabitq-kernel format.
 * ---------------------------------------------------------------- */

static void
benchmark_distance_throughput(const BenchConfig *config)
{
	Dimension dim		   = config->dim;
	uint32_t  count		   = config->count;
	uint32_t  packed_bytes = (dim + 7) / 8;
	uint32_t  ngroups = (count + MKT_FASTSCAN_GROUP - 1) / MKT_FASTSCAN_GROUP;

	printf("\nDistance throughput (dim=%u, count=%u):\n", dim, count);

	/* Encode vectors with RaBitQ */
	RaBitQParams *params   = mkt_rabitq_create(dim, 42);
	float		 *centroid = mkt_alloc0(dim * sizeof(float));
	VectorRef	  cent_ref = {.data = centroid, .dim = dim};

	float *vectors = mkt_alloc((size_t)count * dim * sizeof(float));
	for (uint32_t i = 0; i < count; i++)
		fill_random_floats(vectors + (size_t)i * dim, dim, i);

	float	*f_add	   = mkt_alloc(count * sizeof(float));
	float	*f_rescale = mkt_alloc(count * sizeof(float));
	uint8_t *bits	   = mkt_alloc((size_t)count * packed_bytes);

	mkt_rabitq_encode_batch(
			params,
			vectors,
			MKT_VEC_F32,
			cent_ref,
			f_add,
			f_rescale,
			bits,
			(uint16_t)(count > 65535 ? 65535 : count));

	float *query = mkt_alloc(dim * sizeof(float));
	fill_random_floats(query, dim, 99999);
	VectorRef query_ref = {.data = query, .dim = dim};

	RaBitQQueryState *state =
			mkt_rabitq_prepare_query(params, query_ref, cent_ref);

	float *distances = mkt_alloc(count * sizeof(float));

	/* --- RaBitQ per-vector asymmetric --- */
	size_t	 data_size = MKT_RABITQ_DATA_SIZE(dim);
	uint8_t *data_buf  = mkt_alloc(data_size);

	RaBitQData *temp_data = (RaBitQData *)data_buf;
	for (int w = 0; w < 3; w++)
		for (uint32_t i = 0; i < count; i++)
		{
			temp_data->f_add	 = f_add[i];
			temp_data->f_rescale = f_rescale[i];
			memcpy(temp_data->bits,
				   bits + (size_t)i * packed_bytes,
				   packed_bytes);
			distances[i] = mkt_rabitq_distance(state, temp_data, dim);
		}

	double rabitq_best = 0.0;
	for (uint32_t run = 0; run < config->runs; run++)
	{
		uint64_t start = get_time_ns();
		for (uint32_t i = 0; i < count; i++)
		{
			temp_data->f_add	 = f_add[i];
			temp_data->f_rescale = f_rescale[i];
			memcpy(temp_data->bits,
				   bits + (size_t)i * packed_bytes,
				   packed_bytes);
			distances[i] = mkt_rabitq_distance(state, temp_data, dim);
		}
		uint64_t end = get_time_ns();
		double	 vps = (double)count / (ns_to_ms(end - start) / 1000.0);
		if (vps > rabitq_best)
			rabitq_best = vps;
	}

	/* --- Fastscan uint8 --- */
	uint32_t codes_size = mkt_fastscan_codes_size(count, dim);
	uint8_t *codes		= mkt_alloc_aligned(codes_size, 64);
	mkt_fastscan_pack_codes(bits, count, dim, codes);

	uint8_t	 *lut_buf = mkt_alloc_aligned(MKT_FASTSCAN_LUT_BYTES(dim), 64);
	uint16_t *accum_buf =
			mkt_alloc_aligned(MKT_FASTSCAN_GROUP * sizeof(uint16_t), 64);

	for (int w = 0; w < 3; w++)
		mkt_fastscan_distance_batch(
				state,
				f_add,
				f_rescale,
				codes,
				ngroups,
				count,
				dim,
				distances,
				lut_buf,
				accum_buf);

	double fs8_best = 0.0;
	for (uint32_t run = 0; run < config->runs; run++)
	{
		uint64_t start = get_time_ns();
		mkt_fastscan_distance_batch(
				state,
				f_add,
				f_rescale,
				codes,
				ngroups,
				count,
				dim,
				distances,
				lut_buf,
				accum_buf);
		uint64_t end = get_time_ns();
		double	 vps = (double)count / (ns_to_ms(end - start) / 1000.0);
		if (vps > fs8_best)
			fs8_best = vps;
	}

	printf("  asymmetric: %7.1fK vec/s  (%5.2f ms/%uk)\n",
		   rabitq_best / 1000.0,
		   (double)count / (rabitq_best / 1000.0),
		   count / 1000);
	printf("  fastscan:   %7.1fK vec/s  (%5.2f ms/%uk)\n",
		   fs8_best / 1000.0,
		   (double)count / (fs8_best / 1000.0),
		   count / 1000);
	printf("  speedup:    %5.1fx\n", fs8_best / rabitq_best);

	mkt_free_aligned(accum_buf);
	mkt_free_aligned(lut_buf);
	mkt_free_aligned(codes);
	mkt_free(data_buf);
	mkt_free(distances);
	mkt_rabitq_free_query(state);
	mkt_free(query);
	mkt_free(bits);
	mkt_free(f_rescale);
	mkt_free(f_add);
	mkt_free(vectors);
	mkt_free(centroid);
	mkt_rabitq_destroy(params);
}

/* ----------------------------------------------------------------
 * Entry point
 * ---------------------------------------------------------------- */

static void
print_usage(CmdContext *ctx)
{
	CMD_USAGE_HEADER(ctx, "bench fastscan-kernel");
	printf("Benchmark VPSHUFB fastscan kernel components.\n\n");
	printf("Options:\n");
	printf("  --dim <int>     Vector dimension (default: %d)\n", DEFAULT_DIM);
	printf("  --count <int>   Number of vectors (default: %d)\n",
		   DEFAULT_COUNT);
	printf("  --runs <int>    Benchmark iterations (default: %d)\n",
		   DEFAULT_RUNS);
	printf("  --simd <str>    SIMD level: avx512, avx2, scalar "
		   "(default: all)\n");
	printf("  --help          Show this help message\n");
	printf("\n");
	printf("Examples:\n");
	CMD_USAGE_EXAMPLE(
			ctx, "bench fastscan-kernel", "--dim 768 --count 100000");
	CMD_USAGE_EXAMPLE(ctx, "bench fastscan-kernel", "--simd avx2 --runs 10");
}

int
cmd_bench_fastscan_kernel(CmdContext *ctx)
{
	BenchConfig config = {
			.dim   = DEFAULT_DIM,
			.count = DEFAULT_COUNT,
			.runs  = DEFAULT_RUNS,
			.help  = false,
	};

	static struct option long_options[] = {
			{"dim", required_argument, 0, 'd'},
			{"count", required_argument, 0, 'c'},
			{"runs", required_argument, 0, 'r'},
			{"simd", required_argument, 0, 's'},
			{"help", no_argument, 0, 'h'},
			{0, 0, 0, 0},
	};

	optind = 1;
	int opt;
	while ((opt = getopt_long(
					ctx->argc, ctx->argv, "d:c:r:s:h", long_options, NULL)) !=
		   -1)
	{
		switch (opt)
		{
		case 'd':
			config.dim = (Dimension)atoi(optarg);
			break;
		case 'c':
			config.count = (uint32_t)atoi(optarg);
			break;
		case 'r':
			config.runs = (uint32_t)atoi(optarg);
			break;
		case 's':
			config.simd = optarg;
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

	printf("Fastscan kernel benchmark\n");
	printf("=========================\n\n");

	SimdCapability caps = mkt_detect_simd();

#ifdef MKT_SIMD_FULL
	/* Benchmark each available SIMD level (widest first) */
	typedef struct
	{
		uint32_t	cap;
		const char *name;
		bool		run_distance;
	} SimdLevel;

#if defined(__x86_64__) || defined(_M_X64)
	SimdLevel levels[] = {
			{SIMD_AVX512F, "avx512", true},
			{SIMD_AVX2, "avx2", false},
			{0, "scalar", false},
	};
#elif defined(__aarch64__) || defined(_M_ARM64)
	SimdLevel levels[] = {
			{SIMD_SVE2, "sve2", true},
			{SIMD_NEON, "neon", true},
			{0, "scalar", false},
	};
#else
	SimdLevel levels[] = {
			{0, "scalar", true},
	};
#endif
	uint32_t nlevels = sizeof(levels) / sizeof(levels[0]);

	for (uint32_t l = 0; l < nlevels; l++)
	{
		if (levels[l].cap != 0 && !(caps & levels[l].cap))
			continue;
		if (config.simd != NULL && strcmp(config.simd, levels[l].name) != 0)
			continue;

		mkt_simd_set_override(levels[l].cap ? levels[l].cap : SIMD_NONE);
		mkt_fastscan_reset_simd();
		mkt_fastscan_init_simd();

		printf("--- %s ---\n\n", levels[l].name);
		benchmark_lut_build(&config);
		benchmark_accumulate(&config);
		if (levels[l].run_distance)
			benchmark_distance_throughput(&config);
		printf("\n");
	}

	mkt_simd_set_override(0xFFFFFFFF);
	mkt_fastscan_reset_simd();
	mkt_fastscan_init_simd();
#else
	(void)caps;
	mkt_fastscan_init_simd();
	benchmark_lut_build(&config);
	benchmark_accumulate(&config);
	benchmark_distance_throughput(&config);
	printf("\n");
#endif

	return 0;
}
