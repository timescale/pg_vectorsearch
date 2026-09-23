/*
 * vectorsearch bench rabitq-kernel
 *
 * Benchmark asymmetric vs symmetric RaBitQ distance kernels.
 * Compares:
 * - Inner product throughput (mask+add vs xor+popcount)
 * - Full distance computation (asymmetric vs symmetric)
 * - Accuracy against ground truth L2 distance
 */

#include "vs_config.h"

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

/* Default parameters */
#define DEFAULT_DIM	  768
#define DEFAULT_COUNT 10000
#define DEFAULT_RUNS  5

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

/* Benchmark configuration */
typedef struct
{
	Dimension dim;
	uint32_t  count;
	uint32_t  runs;
	bool	  help;
} BenchConfig;

/*
 * Generate deterministic random vector
 */
static void
generate_vector(float *data, Dimension dim, int seed)
{
	for (Dimension i = 0; i < dim; i++)
		data[i] = (float)(((int)i * 37 + seed * 13) % 200 - 100) / 50.0f;
}

/*
 * Compute true L2 squared distance
 */
static float
true_l2_sq(const float *a, const float *b, Dimension dim)
{
	float sum = 0.0f;
	for (Dimension i = 0; i < dim; i++)
	{
		float d = a[i] - b[i];
		sum += d * d;
	}
	return sum;
}

/*
 * benchmark_hamming_throughput - Raw hamming kernel throughput
 */
static void
benchmark_hamming_throughput(const BenchConfig *config)
{
	Dimension dim		   = config->dim;
	uint32_t  count		   = config->count;
	uint32_t  packed_bytes = VS_RABITQ_BYTES(dim);

	printf("Hamming kernel (dim=%u, %u bytes, count=%u):\n",
		   dim,
		   packed_bytes,
		   count);
	printf("  implementation: %s\n", vs_rabitq_hamming_impl_name());

	/* Allocate bit vectors */
	uint8_t	 *query_bits = vs_alloc_aligned(packed_bytes, 64);
	uint8_t	 *data_bits	 = vs_alloc_aligned((size_t)count * packed_bytes, 64);
	uint32_t *results	 = vs_alloc(count * sizeof(uint32_t));

	/* Fill with deterministic patterns */
	for (uint32_t i = 0; i < packed_bytes; i++)
		query_bits[i] = (uint8_t)((i * 37 + 13) & 0xFF);

	for (uint32_t v = 0; v < count; v++)
		for (uint32_t i = 0; i < packed_bytes; i++)
			data_bits[v * packed_bytes + i] = (uint8_t)((v * 53 + i * 37 + 7) &
														0xFF);

	/* Warmup */
	for (int w = 0; w < 5; w++)
		for (uint32_t i = 0; i < count; i++)
			results[i] = vs_rabitq_hamming_distance(
					query_bits, data_bits + i * packed_bytes, packed_bytes);

	/* Benchmark */
	double best_vps = 0.0;
	for (uint32_t run = 0; run < config->runs; run++)
	{
		uint64_t start = get_time_ns();

		for (uint32_t i = 0; i < count; i++)
			results[i] = vs_rabitq_hamming_distance(
					query_bits, data_bits + i * packed_bytes, packed_bytes);

		uint64_t end = get_time_ns();
		double	 ms	 = ns_to_ms(end - start);
		double	 vps = (double)count / (ms / 1000.0);

		if (vps > best_vps)
			best_vps = vps;
	}

	printf("  hamming:     %7.1fK vec/s  (%5.2f ms/%uk)\n",
		   best_vps / 1000.0,
		   (double)count / (best_vps / 1000.0),
		   count / 1000);

	vs_free(results);
	vs_free_aligned(data_bits);
	vs_free_aligned(query_bits);
}

/*
 * benchmark_distance_throughput - Asymmetric vs symmetric distance
 */
static void
benchmark_distance_throughput(const BenchConfig *config)
{
	Dimension dim	= config->dim;
	uint32_t  count = config->count;

	printf("\nDistance throughput (dim=%u, count=%u):\n", dim, count);
	printf("  inner product impl: %s\n", vs_rabitq_impl_name());
	printf("  hamming impl:       %s\n", vs_rabitq_hamming_impl_name());

	/* Create RaBitQ params and test data */
	RaBitQParams *params   = vs_rabitq_create(dim, 42);
	float		 *centroid = vs_alloc(dim * sizeof(float));

	for (Dimension i = 0; i < dim; i++)
		centroid[i] = 0.0f;

	Vec32Ref cent_ref = {.data = centroid, .dim = dim};

	/* Generate and encode vectors */
	float	*vectors	  = vs_alloc((size_t)count * dim * sizeof(float));
	uint32_t packed_bytes = VS_RABITQ_BYTES(dim);
	float	*f_add		  = vs_alloc(count * sizeof(float));
	float	*f_rescale	  = vs_alloc(count * sizeof(float));
	uint8_t *bits		  = vs_alloc((size_t)count * packed_bytes);

	for (uint32_t i = 0; i < count; i++)
		generate_vector(vectors + i * dim, dim, (int)i);

	vs_rabitq_encode_batch(
			params,
			vectors,
			VS_VEC_F32,
			cent_ref,
			f_add,
			f_rescale,
			bits,
			(uint16_t)(count > 65535 ? 65535 : count));

	/* Prepare query */
	float *query = vs_alloc(dim * sizeof(float));
	generate_vector(query, dim, 99999);
	Vec32Ref query_ref = {.data = query, .dim = dim};

	RaBitQQueryState *state =
			vs_rabitq_prepare_query(params, query_ref, cent_ref);

	Distance *distances = vs_alloc(count * sizeof(Distance));

	/* Pre-build RaBitQData structs to avoid alloc overhead in loop */
	size_t		data_size = VS_RABITQ_DATA_SIZE(dim);
	uint8_t	   *data_buf  = vs_alloc(data_size);
	RaBitQData *temp_data = (RaBitQData *)data_buf;

	/* Benchmark asymmetric distance */
	double best_asym_vps = 0.0;
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
			distances[i] = vs_rabitq_distance(state, temp_data, dim);
		}

		uint64_t end = get_time_ns();
		double	 ms	 = ns_to_ms(end - start);
		double	 vps = (double)count / (ms / 1000.0);

		if (vps > best_asym_vps)
			best_asym_vps = vps;
	}

	/* Benchmark symmetric distance (batch) */
	double best_sym_vps = 0.0;
	for (uint32_t run = 0; run < config->runs; run++)
	{
		uint64_t start = get_time_ns();

		vs_rabitq_distance_batch_symmetric(
				state, f_add, f_rescale, bits, count, dim, distances);

		uint64_t end = get_time_ns();
		double	 ms	 = ns_to_ms(end - start);
		double	 vps = (double)count / (ms / 1000.0);

		if (vps > best_sym_vps)
			best_sym_vps = vps;
	}

	printf("  asymmetric:  %7.1fK vec/s  (%5.2f ms/%uk)\n",
		   best_asym_vps / 1000.0,
		   (double)count / (best_asym_vps / 1000.0),
		   count / 1000);
	printf("  symmetric:   %7.1fK vec/s  (%5.2f ms/%uk)\n",
		   best_sym_vps / 1000.0,
		   (double)count / (best_sym_vps / 1000.0),
		   count / 1000);
	printf("  speedup:     %.1fx\n", best_sym_vps / best_asym_vps);

	vs_free(data_buf);
	vs_free(distances);
	vs_rabitq_free_query(state);
	vs_free(query);
	vs_free(bits);
	vs_free(f_rescale);
	vs_free(f_add);
	vs_free(vectors);
	vs_free(centroid);
	vs_rabitq_destroy(params);
}

/*
 * benchmark_accuracy - Compare distance estimates against ground truth
 */
static void
benchmark_accuracy(const BenchConfig *config)
{
	Dimension dim	= config->dim;
	uint32_t  count = config->count;

	printf("\nAccuracy (dim=%u, count=%u):\n", dim, count);

	RaBitQParams *params   = vs_rabitq_create(dim, 42);
	float		 *centroid = vs_alloc(dim * sizeof(float));
	for (Dimension i = 0; i < dim; i++)
		centroid[i] = 0.0f;
	Vec32Ref cent_ref = {.data = centroid, .dim = dim};

	/* Generate vectors */
	float *vectors = vs_alloc((size_t)count * dim * sizeof(float));
	for (uint32_t i = 0; i < count; i++)
		generate_vector(vectors + i * dim, dim, (int)i);

	/* Encode */
	uint32_t packed_bytes = VS_RABITQ_BYTES(dim);
	float	*f_add		  = vs_alloc(count * sizeof(float));
	float	*f_rescale	  = vs_alloc(count * sizeof(float));
	uint8_t *bits		  = vs_alloc((size_t)count * packed_bytes);

	vs_rabitq_encode_batch(
			params,
			vectors,
			VS_VEC_F32,
			cent_ref,
			f_add,
			f_rescale,
			bits,
			(uint16_t)(count > 65535 ? 65535 : count));

	/* Prepare query */
	float *query = vs_alloc(dim * sizeof(float));
	generate_vector(query, dim, 99999);
	Vec32Ref query_ref = {.data = query, .dim = dim};

	RaBitQQueryState *state =
			vs_rabitq_prepare_query(params, query_ref, cent_ref);

	/* Compute distances and compare */
	double asym_abs_err_sum = 0.0;
	double sym_abs_err_sum	= 0.0;
	double asym_rel_err_sum = 0.0;
	double sym_rel_err_sum	= 0.0;
	float  asym_max_rel_err = 0.0f;
	float  sym_max_rel_err	= 0.0f;

	for (uint32_t i = 0; i < count; i++)
	{
		float true_dist = true_l2_sq(query, vectors + i * dim, dim);

		/* Build RaBitQData on stack */
		size_t		data_size = VS_RABITQ_DATA_SIZE(dim);
		uint8_t	   *buf		  = vs_alloc(data_size);
		RaBitQData *data	  = (RaBitQData *)buf;
		data->f_add			  = f_add[i];
		data->f_rescale		  = f_rescale[i];
		memcpy(data->bits, bits + (size_t)i * packed_bytes, packed_bytes);

		float asym_dist = vs_rabitq_distance(state, data, dim);
		float sym_dist	= vs_rabitq_distance_symmetric(state, data, dim);

		float asym_err = fabsf(asym_dist - true_dist);
		float sym_err  = fabsf(sym_dist - true_dist);

		asym_abs_err_sum += (double)asym_err;
		sym_abs_err_sum += (double)sym_err;

		if (true_dist > 1e-6f)
		{
			float asym_rel = asym_err / true_dist;
			float sym_rel  = sym_err / true_dist;
			asym_rel_err_sum += (double)asym_rel;
			sym_rel_err_sum += (double)sym_rel;

			if (asym_rel > asym_max_rel_err)
				asym_max_rel_err = asym_rel;
			if (sym_rel > sym_max_rel_err)
				sym_max_rel_err = sym_rel;
		}

		vs_free(buf);
	}

	printf("  %-14s %12s %12s\n", "", "asymmetric", "symmetric");
	printf("  %-14s %12.2f %12.2f\n",
		   "mean abs err:",
		   asym_abs_err_sum / count,
		   sym_abs_err_sum / count);
	printf("  %-14s %12.4f %12.4f\n",
		   "mean rel err:",
		   asym_rel_err_sum / count,
		   sym_rel_err_sum / count);
	printf("  %-14s %12.4f %12.4f\n",
		   "max rel err:",
		   (double)asym_max_rel_err,
		   (double)sym_max_rel_err);

	/* Recall@K: how many of the true top-K are in each approximate top-K */
	const int K = 10;
	if ((int)count >= K)
	{
		/* Find true top-K indices */
		float *true_dists = vs_alloc(count * sizeof(float));
		int	  *true_idx	  = vs_alloc(count * sizeof(int));
		for (uint32_t i = 0; i < count; i++)
		{
			true_dists[i] = true_l2_sq(query, vectors + i * dim, dim);
			true_idx[i]	  = (int)i;
		}

		/* Simple selection sort for top-K */
		for (int k = 0; k < K; k++)
		{
			for (uint32_t j = (uint32_t)k + 1; j < count; j++)
			{
				if (true_dists[j] < true_dists[k])
				{
					float tmp	  = true_dists[k];
					true_dists[k] = true_dists[j];
					true_dists[j] = tmp;
					int ti		  = true_idx[k];
					true_idx[k]	  = true_idx[j];
					true_idx[j]	  = ti;
				}
			}
		}

		/* Compute asymmetric distances and find top-K */
		float *asym_dists = vs_alloc(count * sizeof(float));
		int	  *asym_idx	  = vs_alloc(count * sizeof(int));
		for (uint32_t i = 0; i < count; i++)
		{
			size_t		data_size = VS_RABITQ_DATA_SIZE(dim);
			uint8_t	   *buf		  = vs_alloc(data_size);
			RaBitQData *data	  = (RaBitQData *)buf;
			data->f_add			  = f_add[i];
			data->f_rescale		  = f_rescale[i];
			memcpy(data->bits, bits + (size_t)i * packed_bytes, packed_bytes);
			asym_dists[i] = vs_rabitq_distance(state, data, dim);
			asym_idx[i]	  = (int)i;
			vs_free(buf);
		}

		for (int k = 0; k < K; k++)
			for (uint32_t j = (uint32_t)k + 1; j < count; j++)
				if (asym_dists[j] < asym_dists[k])
				{
					float tmp	  = asym_dists[k];
					asym_dists[k] = asym_dists[j];
					asym_dists[j] = tmp;
					int ti		  = asym_idx[k];
					asym_idx[k]	  = asym_idx[j];
					asym_idx[j]	  = ti;
				}

		/* Compute symmetric distances and find top-K */
		float *sym_dists = vs_alloc(count * sizeof(float));
		int	  *sym_idx	 = vs_alloc(count * sizeof(int));
		for (uint32_t i = 0; i < count; i++)
		{
			size_t		data_size = VS_RABITQ_DATA_SIZE(dim);
			uint8_t	   *buf		  = vs_alloc(data_size);
			RaBitQData *data	  = (RaBitQData *)buf;
			data->f_add			  = f_add[i];
			data->f_rescale		  = f_rescale[i];
			memcpy(data->bits, bits + (size_t)i * packed_bytes, packed_bytes);
			sym_dists[i] = vs_rabitq_distance_symmetric(state, data, dim);
			sym_idx[i]	 = (int)i;
			vs_free(buf);
		}

		for (int k = 0; k < K; k++)
			for (uint32_t j = (uint32_t)k + 1; j < count; j++)
				if (sym_dists[j] < sym_dists[k])
				{
					float tmp	 = sym_dists[k];
					sym_dists[k] = sym_dists[j];
					sym_dists[j] = tmp;
					int ti		 = sym_idx[k];
					sym_idx[k]	 = sym_idx[j];
					sym_idx[j]	 = ti;
				}

		/* Count recall */
		int asym_recall = 0;
		int sym_recall	= 0;
		for (int k = 0; k < K; k++)
			for (int j = 0; j < K; j++)
			{
				if (asym_idx[k] == true_idx[j])
					asym_recall++;
				if (sym_idx[k] == true_idx[j])
					sym_recall++;
			}

		printf("  %-14s %11d%% %11d%%\n",
			   "recall@10:",
			   asym_recall * 100 / K,
			   sym_recall * 100 / K);

		vs_free(sym_idx);
		vs_free(sym_dists);
		vs_free(asym_idx);
		vs_free(asym_dists);
		vs_free(true_idx);
		vs_free(true_dists);
	}

	vs_rabitq_free_query(state);
	vs_free(query);
	vs_free(bits);
	vs_free(f_rescale);
	vs_free(f_add);
	vs_free(vectors);
	vs_free(centroid);
	vs_rabitq_destroy(params);
}

/*
 * print_usage - Show usage information
 */
static void
print_usage(CmdContext *ctx)
{
	CMD_USAGE_HEADER(ctx, "bench rabitq-kernel");
	printf("Benchmark asymmetric vs symmetric RaBitQ distance kernels.\n\n");
	printf("Options:\n");
	printf("  --dim <int>     Vector dimension (default: %d)\n", DEFAULT_DIM);
	printf("  --count <int>   Number of vectors (default: %d)\n",
		   DEFAULT_COUNT);
	printf("  --runs <int>    Benchmark iterations (default: %d)\n",
		   DEFAULT_RUNS);
	printf("  --help          Show this help message\n");
	printf("\n");
	printf("Examples:\n");
	CMD_USAGE_EXAMPLE(ctx, "bench rabitq-kernel", "--dim 768 --count 10000");
	CMD_USAGE_EXAMPLE(ctx, "bench rabitq-kernel", "--dim 128 --runs 10");
}

/*
 * cmd_bench_rabitq_kernel - Entry point
 */
int
cmd_bench_rabitq_kernel(CmdContext *ctx)
{
	BenchConfig config = {
			.dim   = DEFAULT_DIM,
			.count = DEFAULT_COUNT,
			.runs  = DEFAULT_RUNS,
			.help  = false,
	};

	static struct option long_options[] =
			{{"dim", required_argument, 0, 'd'},
			 {"count", required_argument, 0, 'c'},
			 {"runs", required_argument, 0, 'r'},
			 {"help", no_argument, 0, 'h'},
			 {0, 0, 0, 0}};

	optind = 1;
	int opt;
	while ((opt = getopt_long(
					ctx->argc, ctx->argv, "d:c:r:h", long_options, NULL)) !=
		   -1)
	{
		switch (opt)
		{
		case 'd':
			config.dim = (Dimension)atoi(optarg);
			if (config.dim == 0)
			{
				fprintf(stderr, "Error: Invalid dimension\n");
				return 1;
			}
			break;
		case 'c':
			config.count = (uint32_t)atoi(optarg);
			if (config.count == 0)
			{
				fprintf(stderr, "Error: Invalid count\n");
				return 1;
			}
			break;
		case 'r':
			config.runs = (uint32_t)atoi(optarg);
			if (config.runs == 0)
			{
				fprintf(stderr, "Error: Invalid runs\n");
				return 1;
			}
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

	vs_rabitq_init_simd();

	benchmark_hamming_throughput(&config);
	benchmark_distance_throughput(&config);
	benchmark_accuracy(&config);

	return 0;
}
