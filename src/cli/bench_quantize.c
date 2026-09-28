/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * vectorsearch bench quantize
 *
 * Benchmark RaBitQ quantization performance across operations and SIMD.
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
#include "quant/matrix.h"
#include "quant/rabitq.h"
#include "types/vec16.h"

#ifdef VS_HAVE_FAISS
#include <RaBitQuantizer_c.h>
#include <error_c.h>
#endif

/* Default parameters */
#define DEFAULT_DIM	  768
#define DEFAULT_COUNT 10000
#define DEFAULT_RUNS  3

/* SIMD implementations for distance computation */
typedef struct
{
	const char *name;
	uint32_t	mask;
} ImplSpec;

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

/* Operation filter bitmask */
#define OP_ENCODE_SINGLE (1 << 0)
#define OP_ENCODE_BATCH	 (1 << 1)
#define OP_DISTANCE		 (1 << 2)
#define OP_ALL			 (OP_ENCODE_SINGLE | OP_ENCODE_BATCH | OP_DISTANCE)

/* Benchmark configuration */
typedef struct
{
	Dimension	dim;
	uint32_t	count;
	uint32_t	runs;
	uint32_t	op_filter;
	const char *impl_filter;
	const char *type; /* "f32" or "f16" */
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
	stats->samples = vs_alloc(capacity * sizeof(double));
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
	vs_free(stats->samples);
}

/*
 * Generate random test vectors
 */
static void
generate_random_vectors(float *data, uint32_t count, Dimension dim)
{
	for (uint32_t i = 0; i < count; i++)
	{
		for (Dimension j = 0; j < dim; j++)
		{
			int val			  = rand() % 200 - 100;
			data[i * dim + j] = (float)val / 100.0f;
		}
	}
}

static void
generate_random_halfvecs(half *data, uint32_t count, Dimension dim)
{
	for (uint32_t i = 0; i < count; i++)
	{
		for (Dimension j = 0; j < dim; j++)
		{
			int val			  = rand() % 200 - 100;
			data[i * dim + j] = vs_float_to_half((float)val / 100.0f);
		}
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
 * Benchmark single-vector encoding
 */
static void
benchmark_encode_single(
		RaBitQParams *params,
		const float	 *vectors,
		const float	 *centroid,
		uint32_t	  count,
		Dimension	  dim,
		uint32_t	  runs)
{
	printf("Single-vector encoding (dim=%u, count=%u):\n", dim, count);

	Vec32Ref cent_ref = {.data = centroid, .dim = dim};

	/* Allocate output buffers */
	size_t	 data_size = VS_RABITQ_DATA_SIZE(dim);
	uint8_t *outputs   = vs_alloc(count * data_size);
	if (outputs == NULL)
	{
		fprintf(stderr, "Failed to allocate output buffer\n");
		return;
	}

	/* CPU warmup: throwaway passes to stabilize measurements */
	for (uint32_t warmup = 0; warmup < runs; warmup++)
	{
		for (uint32_t i = 0; i < count; i++)
		{
			Vec32Ref	vec_ref = {.data = vectors + i * dim, .dim = dim};
			RaBitQData *out		= (RaBitQData *)(outputs + i * data_size);
			vs_rabitq_encode_into(params, vec_ref, cent_ref, out);
		}
	}

	BenchStats stats;
	bench_stats_init(&stats, runs);

	for (uint32_t run = 0; run < runs; run++)
	{
		uint64_t start = get_time_ns();

		for (uint32_t i = 0; i < count; i++)
		{
			Vec32Ref	vec_ref = {.data = vectors + i * dim, .dim = dim};
			RaBitQData *out		= (RaBitQData *)(outputs + i * data_size);
			vs_rabitq_encode_into(params, vec_ref, cent_ref, out);
		}

		uint64_t end		 = get_time_ns();
		double	 elapsed_ms	 = ns_to_ms(end - start);
		double	 vec_per_sec = (double)count / (elapsed_ms / 1000.0);

		bench_stats_add(&stats, vec_per_sec);
	}

	bench_stats_compute(&stats);
	double elapsed_ms = (double)count / (stats.avg / 1000.0);

	if (runs == 1)
	{
		printf("  %-12s %8.1f ms  (%7.1fK vec/s)\n",
			   "single",
			   elapsed_ms,
			   stats.avg / 1000.0);
	}
	else
	{
		printf("  %-12s %8.1f ms  (%7.1fK vec/s) (±%.1f%%, n=%u)\n",
			   "single",
			   elapsed_ms,
			   stats.avg / 1000.0,
			   (stats.stddev / stats.avg) * 100.0,
			   runs);
	}

	bench_stats_free(&stats);
	vs_free(outputs);
}

/*
 * Benchmark batch encoding
 */
static void
benchmark_encode_batch(
		RaBitQParams *params,
		const void	 *vectors,
		VecType		  vec_type,
		const float	 *centroid,
		uint32_t	  count,
		Dimension	  dim,
		uint32_t	  runs)
{
	printf("Batch encoding [%s] (dim=%u, count=%u):\n",
		   vs_vec_type_name(vec_type),
		   dim,
		   count);

	Vec32Ref cent_ref = {.data = centroid, .dim = dim};

	/* Allocate batch output buffers */
	uint32_t packed_bytes = VS_RABITQ_BYTES(dim);
	float	*f_add		  = vs_alloc(count * sizeof(float));
	float	*f_rescale	  = vs_alloc(count * sizeof(float));
	uint8_t *bits		  = vs_alloc((size_t)count * packed_bytes);
	if (f_add == NULL || f_rescale == NULL || bits == NULL)
	{
		fprintf(stderr, "Failed to allocate output buffer\n");
		if (f_add)
			vs_free(f_add);
		if (f_rescale)
			vs_free(f_rescale);
		if (bits)
			vs_free(bits);
		return;
	}

	/* CPU warmup: throwaway passes to stabilize measurements */
	for (uint32_t warmup = 0; warmup < runs; warmup++)
		vs_rabitq_encode_batch(
				params,
				vectors,
				vec_type,
				cent_ref,
				f_add,
				f_rescale,
				bits,
				count);

	BenchStats stats;
	bench_stats_init(&stats, runs);

	for (uint32_t run = 0; run < runs; run++)
	{
		uint64_t start = get_time_ns();

		vs_rabitq_encode_batch(
				params,
				vectors,
				vec_type,
				cent_ref,
				f_add,
				f_rescale,
				bits,
				count);

		uint64_t end		 = get_time_ns();
		double	 elapsed_ms	 = ns_to_ms(end - start);
		double	 vec_per_sec = (double)count / (elapsed_ms / 1000.0);

		bench_stats_add(&stats, vec_per_sec);
	}

	bench_stats_compute(&stats);
	double elapsed_ms = (double)count / (stats.avg / 1000.0);

	if (runs == 1)
	{
		printf("  %-12s %8.1f ms  (%7.1fK vec/s)\n",
			   "batch",
			   elapsed_ms,
			   stats.avg / 1000.0);
	}
	else
	{
		printf("  %-12s %8.1f ms  (%7.1fK vec/s) (±%.1f%%, n=%u)\n",
			   "batch",
			   elapsed_ms,
			   stats.avg / 1000.0,
			   (stats.stddev / stats.avg) * 100.0,
			   runs);
	}

	bench_stats_free(&stats);
	vs_free(bits);
	vs_free(f_rescale);
	vs_free(f_add);
}

/*
 * Benchmark distance computation with different SIMD implementations
 */
static void
benchmark_distance(
		RaBitQParams *params,
		RaBitQData	**encoded,
		const float	 *query,
		const float	 *centroid,
		uint32_t	  count,
		Dimension	  dim,
		uint32_t	  runs,
		const char	 *impl_filter)
{
	printf("Distance computation (dim=%u, count=%u):\n", dim, count);

	Vec32Ref query_ref = {.data = query, .dim = dim};
	Vec32Ref cent_ref  = {.data = centroid, .dim = dim};

	/* Allocate distances array */
	Distance *distances = vs_alloc(count * sizeof(Distance));
	if (distances == NULL)
	{
		fprintf(stderr, "Failed to allocate distances\n");
		return;
	}

	/* Get CPU capabilities once before the loop */
	vs_simd_set_override(0xFFFFFFFF); /* Ensure auto-detect mode */
	vs_simd_reset_cache();
	uint32_t caps = vs_detect_simd();

	/* Test each SIMD implementation */
	for (const ImplSpec *impl = impls; impl->name; impl++)
	{
		if (!should_test_impl(impl->name, impl_filter))
			continue;

		/* Check if supported */
		if (impl->mask != SIMD_NONE && (caps & impl->mask) == 0)
		{
			printf("  %-12s (not supported on this CPU)\n", impl->name);
			continue;
		}

		/* Set SIMD override */
		vs_simd_set_override(impl->mask);
		vs_simd_reset_cache();
		vs_rabitq_force_reinit();

		/* Prepare query state */
		RaBitQQueryState *state =
				vs_rabitq_prepare_query(params, query_ref, cent_ref);
		if (state == NULL)
		{
			fprintf(stderr, "Failed to prepare query state\n");
			continue;
		}

		/* CPU warmup: throwaway passes to stabilize measurements */
		for (uint32_t warmup = 0; warmup < runs; warmup++)
		{
			for (uint32_t i = 0; i < count; i++)
				distances[i] = vs_rabitq_distance(state, encoded[i], dim);
		}

		BenchStats stats;
		bench_stats_init(&stats, runs);

		for (uint32_t run = 0; run < runs; run++)
		{
			uint64_t start = get_time_ns();

			for (uint32_t i = 0; i < count; i++)
				distances[i] = vs_rabitq_distance(state, encoded[i], dim);

			uint64_t end		 = get_time_ns();
			double	 elapsed_ms	 = ns_to_ms(end - start);
			double	 vec_per_sec = (double)count / (elapsed_ms / 1000.0);

			bench_stats_add(&stats, vec_per_sec);
		}

		bench_stats_compute(&stats);
		double elapsed_ms = (double)count / (stats.avg / 1000.0);

		if (runs == 1)
		{
			printf("  %-12s %8.1f ms  (%7.1fK vec/s)\n",
				   impl->name,
				   elapsed_ms,
				   stats.avg / 1000.0);
		}
		else
		{
			printf("  %-12s %8.1f ms  (%7.1fK vec/s) (±%.1f%%, n=%u)\n",
				   impl->name,
				   elapsed_ms,
				   stats.avg / 1000.0,
				   (stats.stddev / stats.avg) * 100.0,
				   runs);
		}

		bench_stats_free(&stats);
		vs_rabitq_free_query(state);
	}

	/* Reset to auto-detection */
	vs_simd_set_override(0xFFFFFFFF);
	vs_simd_reset_cache();
	vs_rabitq_force_reinit();

	vs_free(distances);
}

#ifdef VS_HAVE_FAISS
/*
 * Benchmark FAISS encoding
 *
 * Note: FAISS assumes random rotation is done externally.
 * "faiss-sign" measures sign bit extraction only (what FAISS does).
 * pg_vectorsearch includes rotation in encoding, so compare with batch
 * for full picture.
 */
static void
benchmark_faiss_encode(
		RaBitQParams *params,
		const float	 *vectors,
		const float	 *centroid,
		uint32_t	  count,
		Dimension	  dim,
		uint32_t	  runs)
{
	printf("FAISS encoding (dim=%u, count=%u):\n", dim, count);

	/* Allocate buffer for transformed vectors */
	float *transformed = vs_alloc((size_t)count * dim * sizeof(float));
	if (transformed == NULL)
	{
		fprintf(stderr, "Failed to allocate transformed buffer\n");
		return;
	}

	/* Create FAISS RaBitQuantizer (1-bit, L2 metric) */
	FaissRaBitQuantizer *faiss_rq = NULL;
	if (faiss_RaBitQuantizer_new_with(&faiss_rq, dim, METRIC_L2, 1) != 0)
	{
		fprintf(stderr,
				"Error: Failed to create FAISS RaBitQ: %s\n",
				faiss_get_last_error());
		vs_free(transformed);
		return;
	}

	size_t	 code_size = faiss_RaBitQuantizer_code_size(faiss_rq);
	uint8_t *codes	   = vs_alloc(count * code_size);
	if (codes == NULL)
	{
		fprintf(stderr, "Failed to allocate FAISS code buffer\n");
		faiss_RaBitQuantizer_free(faiss_rq);
		vs_free(transformed);
		return;
	}

	/* CPU warmup - include rotation in timing */
	for (uint32_t warmup = 0; warmup < runs; warmup++)
	{
		/* Transform: residual then P^T multiply */
		for (uint32_t i = 0; i < count; i++)
		{
			float *residual = transformed + i * dim;
			for (Dimension j = 0; j < dim; j++)
				residual[j] = vectors[i * dim + j] - centroid[j];
		}
		for (uint32_t i = 0; i < count; i++)
			vs_rabitq_rotate(
					params,
					transformed + (size_t)i * dim,
					transformed + (size_t)i * dim);
		faiss_RaBitQuantizer_compute_codes(
				faiss_rq, transformed, codes, count);
	}

	BenchStats stats;
	bench_stats_init(&stats, runs);

	for (uint32_t run = 0; run < runs; run++)
	{
		uint64_t start = get_time_ns();

		/* Transform: residual then P^T multiply (same as pg_vectorsearch) */
		for (uint32_t i = 0; i < count; i++)
		{
			float *residual = transformed + i * dim;
			for (Dimension j = 0; j < dim; j++)
				residual[j] = vectors[i * dim + j] - centroid[j];
		}
		for (uint32_t i = 0; i < count; i++)
			vs_rabitq_rotate(
					params,
					transformed + (size_t)i * dim,
					transformed + (size_t)i * dim);
		faiss_RaBitQuantizer_compute_codes(
				faiss_rq, transformed, codes, count);

		uint64_t end		 = get_time_ns();
		double	 elapsed_ms	 = ns_to_ms(end - start);
		double	 vec_per_sec = (double)count / (elapsed_ms / 1000.0);

		bench_stats_add(&stats, vec_per_sec);
	}

	bench_stats_compute(&stats);
	double elapsed_ms = (double)count / (stats.avg / 1000.0);

	if (runs == 1)
	{
		printf("  %-12s %8.1f ms  (%7.1fK vec/s)\n",
			   "faiss",
			   elapsed_ms,
			   stats.avg / 1000.0);
	}
	else
	{
		printf("  %-12s %8.1f ms  (%7.1fK vec/s) (±%.1f%%, n=%u)\n",
			   "faiss",
			   elapsed_ms,
			   stats.avg / 1000.0,
			   (stats.stddev / stats.avg) * 100.0,
			   runs);
	}

	bench_stats_free(&stats);
	vs_free(codes);
	vs_free(transformed);
	faiss_RaBitQuantizer_free(faiss_rq);
}

/*
 * Benchmark FAISS distance computation
 *
 * Tests both qb=0 (scalar, no query quantization) and qb=8 (SIMD with SQ8).
 * qb=8 is the fair comparison since it uses SIMD bit operations.
 */
static void
benchmark_faiss_distance(
		const float *vectors,
		const float *query,
		uint32_t	 count,
		Dimension	 dim,
		uint32_t	 runs)
{
	printf("FAISS distance (dim=%u, count=%u):\n", dim, count);

	/* Create FAISS RaBitQuantizer */
	FaissRaBitQuantizer *faiss_rq = NULL;
	if (faiss_RaBitQuantizer_new_with(&faiss_rq, dim, METRIC_L2, 1) != 0)
	{
		fprintf(stderr,
				"Error: Failed to create FAISS RaBitQ: %s\n",
				faiss_get_last_error());
		return;
	}

	/* Encode all vectors */
	size_t	 code_size = faiss_RaBitQuantizer_code_size(faiss_rq);
	uint8_t *codes	   = vs_alloc(count * code_size);
	if (codes == NULL)
	{
		fprintf(stderr, "Failed to allocate FAISS code buffer\n");
		faiss_RaBitQuantizer_free(faiss_rq);
		return;
	}

	faiss_RaBitQuantizer_compute_codes(faiss_rq, vectors, codes, count);

	/* Allocate distances array */
	float *distances = vs_alloc(count * sizeof(float));
	if (distances == NULL)
	{
		fprintf(stderr, "Failed to allocate distances\n");
		vs_free(codes);
		faiss_RaBitQuantizer_free(faiss_rq);
		return;
	}

	/* Test with qb=8 (SIMD with quantized query) - fair comparison */
	uint8_t		qb_values[] = {8, 0};
	const char *qb_names[]	= {"faiss-q8", "faiss-q0"};

	for (int qi = 0; qi < 2; qi++)
	{
		uint8_t qb = qb_values[qi];

		/* Get distance computer with specified qb */
		FaissRaBitQDistanceComputer *faiss_dc = NULL;
		if (faiss_RaBitQuantizer_get_distance_computer(
					faiss_rq, &faiss_dc, qb, NULL, 0) != 0)
		{
			fprintf(stderr,
					"Error: Failed to get FAISS distance computer (qb=%d): "
					"%s\n",
					qb,
					faiss_get_last_error());
			continue;
		}

		if (faiss_RaBitQDistanceComputer_set_query(faiss_dc, query) != 0)
		{
			fprintf(stderr,
					"Error: Failed to set FAISS query: %s\n",
					faiss_get_last_error());
			faiss_RaBitQDistanceComputer_free(faiss_dc);
			continue;
		}

		/* CPU warmup */
		for (uint32_t warmup = 0; warmup < runs; warmup++)
		{
			for (uint32_t i = 0; i < count; i++)
			{
				faiss_RaBitQDistanceComputer_distance_to_code(
						faiss_dc, codes + i * code_size, &distances[i]);
			}
		}

		BenchStats stats;
		bench_stats_init(&stats, runs);

		for (uint32_t run = 0; run < runs; run++)
		{
			uint64_t start = get_time_ns();

			for (uint32_t i = 0; i < count; i++)
			{
				faiss_RaBitQDistanceComputer_distance_to_code(
						faiss_dc, codes + i * code_size, &distances[i]);
			}

			uint64_t end		 = get_time_ns();
			double	 elapsed_ms	 = ns_to_ms(end - start);
			double	 vec_per_sec = (double)count / (elapsed_ms / 1000.0);

			bench_stats_add(&stats, vec_per_sec);
		}

		bench_stats_compute(&stats);
		double elapsed_ms = (double)count / (stats.avg / 1000.0);

		if (runs == 1)
		{
			printf("  %-12s %8.1f ms  (%7.1fK vec/s)\n",
				   qb_names[qi],
				   elapsed_ms,
				   stats.avg / 1000.0);
		}
		else
		{
			printf("  %-12s %8.1f ms  (%7.1fK vec/s) (±%.1f%%, n=%u)\n",
				   qb_names[qi],
				   elapsed_ms,
				   stats.avg / 1000.0,
				   (stats.stddev / stats.avg) * 100.0,
				   runs);
		}

		bench_stats_free(&stats);
		faiss_RaBitQDistanceComputer_free(faiss_dc);
	}

	vs_free(distances);
	vs_free(codes);
	faiss_RaBitQuantizer_free(faiss_rq);
}

/*
 * Compute L2 squared distance between two vectors
 */
static float
l2_distance_sq(const float *a, const float *b, Dimension dim)
{
	float sum = 0.0f;
	for (Dimension i = 0; i < dim; i++)
	{
		float diff = a[i] - b[i];
		sum += diff * diff;
	}
	return sum;
}

/*
 * Compare pg_vectorsearch vs FAISS distance correctness
 *
 * Both implementations approximate L2 distance using different formulas.
 * We compare each against true L2 distance, and check bit pattern match.
 */
static void
compare_correctness(
		RaBitQParams *params,
		const float	 *vectors,
		const float	 *query,
		const float	 *centroid,
		uint32_t	  count,
		Dimension	  dim)
{
	printf("\nCorrectness comparison (vs True L2):\n");

	Vec32Ref query_ref = {.data = query, .dim = dim};
	Vec32Ref cent_ref  = {.data = centroid, .dim = dim};

	/* Transform vectors with P^T for FAISS (FAISS expects rotation externally)
	 */
	float *transformed		 = vs_alloc(count * dim * sizeof(float));
	float *query_transformed = vs_alloc(dim * sizeof(float));
	if (transformed == NULL || query_transformed == NULL)
	{
		fprintf(stderr, "Failed to allocate transformed vectors\n");
		vs_free(transformed);
		vs_free(query_transformed);
		return;
	}

	/* Transform all vectors: rotated = P^T * (v - centroid) */
	for (uint32_t i = 0; i < count; i++)
	{
		/* Subtract centroid first */
		float *v_centered = vs_alloc(dim * sizeof(float));
		for (Dimension j = 0; j < dim; j++)
			v_centered[j] = vectors[i * dim + j] - centroid[j];

		vs_rabitq_rotate(params, v_centered, transformed + i * dim);
		vs_free(v_centered);
	}

	/* Transform query */
	float *q_centered = vs_alloc(dim * sizeof(float));
	for (Dimension j = 0; j < dim; j++)
		q_centered[j] = query[j] - centroid[j];
	vs_rabitq_rotate(params, q_centered, query_transformed);
	vs_free(q_centered);

	/* Create FAISS quantizer */
	FaissRaBitQuantizer *faiss_rq = NULL;
	if (faiss_RaBitQuantizer_new_with(&faiss_rq, dim, METRIC_L2, 1) != 0)
	{
		fprintf(stderr, "Error: Failed to create FAISS RaBitQ\n");
		vs_free(transformed);
		vs_free(query_transformed);
		return;
	}

	/* Encode with FAISS (using transformed vectors) */
	size_t	 faiss_code_size = faiss_RaBitQuantizer_code_size(faiss_rq);
	uint8_t *faiss_codes	 = vs_alloc(count * faiss_code_size);
	faiss_RaBitQuantizer_compute_codes(
			faiss_rq, transformed, faiss_codes, count);

	/* Get FAISS distance computer (qb=0 for no query quantization) */
	FaissRaBitQDistanceComputer *faiss_dc = NULL;
	faiss_RaBitQuantizer_get_distance_computer(
			faiss_rq, &faiss_dc, 0, NULL, 0);
	faiss_RaBitQDistanceComputer_set_query(faiss_dc, query_transformed);

	/* Encode with pg_vectorsearch and compute distances */
	RaBitQQueryState *vs_state =
			vs_rabitq_prepare_query(params, query_ref, cent_ref);

	/* Statistics */
	float  vs_max_err = 0.0f, vs_sum_err = 0.0f;
	float  faiss_max_err = 0.0f, faiss_sum_err = 0.0f;
	int	   bits_match = 0;
	size_t nbytes	  = (dim + 7) / 8;

	for (uint32_t i = 0; i < count; i++)
	{
		Vec32Ref vec_ref = {.data = vectors + i * dim, .dim = dim};

		/* True L2 squared distance */
		float true_dist = l2_distance_sq(query, vectors + i * dim, dim);

		/* pg_vectorsearch encoding and distance */
		RaBitQData *vs_enc	= vs_rabitq_encode(params, vec_ref, cent_ref);
		float		vs_dist = vs_rabitq_distance(vs_state, vs_enc, dim);

		/* FAISS distance */
		float faiss_dist;
		faiss_RaBitQDistanceComputer_distance_to_code(
				faiss_dc, faiss_codes + i * faiss_code_size, &faiss_dist);

		/* Compute relative errors vs true distance */
		if (true_dist > 1e-6f)
		{
			float vs_err	= fabsf(vs_dist - true_dist) / true_dist;
			float faiss_err = fabsf(faiss_dist - true_dist) / true_dist;

			if (vs_err > vs_max_err)
				vs_max_err = vs_err;
			if (faiss_err > faiss_max_err)
				faiss_max_err = faiss_err;
			vs_sum_err += vs_err;
			faiss_sum_err += faiss_err;
		}

		/* Check bit pattern match */
		if (memcmp(vs_enc->bits, faiss_codes + i * faiss_code_size, nbytes) ==
			0)
			bits_match++;

		vs_free(vs_enc);
	}

	printf("  Vectors compared: %u\n", count);
	printf("  Bit patterns match: %d/%u (%.1f%%)\n",
		   bits_match,
		   count,
		   100.0f * bits_match / count);
	printf("\n  Distance error vs True L2:\n");
	printf("    pg_vectorsearch: max=%.2f%%, mean=%.2f%%\n",
		   vs_max_err * 100.0f,
		   (vs_sum_err / count) * 100.0f);
	printf("    FAISS:           max=%.2f%%, mean=%.2f%%\n",
		   faiss_max_err * 100.0f,
		   (faiss_sum_err / count) * 100.0f);

	if (bits_match == (int)count)
		printf("  Result: PASS (bit-for-bit identical encoding)\n");
	else if ((float)bits_match / count > 0.99f)
		printf("  Result: PASS (>99%% bit match, minor tie-breaking diffs)\n");
	else
		printf("  Result: CHECK (%.1f%% bit match)\n",
			   100.0f * bits_match / count);

	/* Cleanup */
	vs_rabitq_free_query(vs_state);
	faiss_RaBitQDistanceComputer_free(faiss_dc);
	vs_free(faiss_codes);
	faiss_RaBitQuantizer_free(faiss_rq);
	vs_free(query_transformed);
	vs_free(transformed);
}
#endif /* VS_HAVE_FAISS */

/*
 * Parse operation filter from comma-separated list
 */
static uint32_t
parse_ops(const char *names)
{
	if (strcmp(names, "all") == 0)
		return OP_ALL;

	uint32_t	mask  = 0;
	const char *start = names;

	while (*start)
	{
		const char *end = strchr(start, ',');
		size_t		len = end ? (size_t)(end - start) : strlen(start);

		if (len == 6 && strncmp(start, "single", 6) == 0)
			mask |= OP_ENCODE_SINGLE;
		else if (len == 5 && strncmp(start, "batch", 5) == 0)
			mask |= OP_ENCODE_BATCH;
		else if (len == 8 && strncmp(start, "distance", 8) == 0)
			mask |= OP_DISTANCE;
		else if (len == 6 && strncmp(start, "encode", 6) == 0)
			mask |= OP_ENCODE_SINGLE | OP_ENCODE_BATCH;

		if (end == NULL)
			break;
		start = end + 1;
	}

	return mask ? mask : OP_ALL;
}

/*
 * Print usage information
 */
static void
print_usage(CmdContext *ctx)
{
	CMD_USAGE_HEADER(ctx, "bench quantize");
	printf("Benchmark RaBitQ quantization performance.\n\n");
	printf("Options:\n");
	printf("  --dim <int>        Vector dimension (default: %d)\n",
		   DEFAULT_DIM);
	printf("  --count <int>      Number of vectors (default: %d)\n",
		   DEFAULT_COUNT);
	printf("  --runs <int>       Number of benchmark iterations (default: "
		   "%d)\n",
		   DEFAULT_RUNS);
	printf("  --ops <list>       Comma-separated operations:\n");
	printf("                     single, batch, distance, encode, all\n");
	printf("                     (default: all)\n");
	printf("  --type <str>       f32 or f16 (default: f32)\n");
	printf("  --impls <list>     SIMD implementations for distance:\n");
	printf("                     compiler, avx2, avx512, neon\n");
	printf("                     (default: all supported)\n");
	printf("  --help             Show this help message\n");
	printf("\n");
#ifdef VS_HAVE_FAISS
	printf("FAISS comparison: enabled (built with -Dfaiss=true)\n\n");
#else
	printf("FAISS comparison: disabled (rebuild with -Dfaiss=true)\n\n");
#endif
	printf("Examples:\n");
	CMD_USAGE_EXAMPLE(ctx, "bench quantize", "--dim 768 --count 10000");
	CMD_USAGE_EXAMPLE(ctx, "bench quantize", "--ops encode --dim 128");
	CMD_USAGE_EXAMPLE(
			ctx, "bench quantize", "--ops distance --impls compiler,avx512");
}

/*
 * Benchmark and compare single vs batch encoding
 */
static void
benchmark_encode_comparison(
		RaBitQParams *params,
		const void	 *vectors,
		VecType		  vec_type,
		const float	 *f32_vectors,
		const float	 *centroid,
		uint32_t	  count,
		Dimension	  dim,
		uint32_t	  runs)
{
	printf("Encoding comparison [%s] (dim=%u, count=%u):\n",
		   vs_vec_type_name(vec_type),
		   dim,
		   count);

	Vec32Ref cent_ref	  = {.data = centroid, .dim = dim};
	size_t	 data_size	  = VS_RABITQ_DATA_SIZE(dim);
	uint32_t packed_bytes = VS_RABITQ_BYTES(dim);

	/* Allocate single-vector output buffer (always f32) */
	uint8_t *single_outputs = vs_alloc(count * data_size);
	if (single_outputs == NULL)
	{
		fprintf(stderr, "Failed to allocate output buffer\n");
		return;
	}

	/* Allocate batch output buffers */
	float	*batch_f_add	 = vs_alloc(count * sizeof(float));
	float	*batch_f_rescale = vs_alloc(count * sizeof(float));
	uint8_t *batch_bits		 = vs_alloc((size_t)count * packed_bytes);
	if (batch_f_add == NULL || batch_f_rescale == NULL || batch_bits == NULL)
	{
		fprintf(stderr, "Failed to allocate batch output buffers\n");
		vs_free(single_outputs);
		if (batch_f_add)
			vs_free(batch_f_add);
		if (batch_f_rescale)
			vs_free(batch_f_rescale);
		if (batch_bits)
			vs_free(batch_bits);
		return;
	}

	/*
	 * CPU warmup: run full throwaway passes first.
	 * The first measurement always appears slower due to CPU warmup effects
	 * (µop cache, branch predictor). Running full benchmark iterations as
	 * throwaway stabilizes the measurements.
	 */
	for (uint32_t warmup = 0; warmup < runs; warmup++)
	{
		/* Throwaway single-vector pass (always f32) */
		for (uint32_t i = 0; i < count; i++)
		{
			Vec32Ref	vec_ref = {.data = f32_vectors + i * dim, .dim = dim};
			RaBitQData *out = (RaBitQData *)(single_outputs + i * data_size);
			vs_rabitq_encode_into(params, vec_ref, cent_ref, out);
		}
		/* Throwaway batch pass (typed) */
		vs_rabitq_encode_batch(
				params,
				vectors,
				vec_type,
				cent_ref,
				batch_f_add,
				batch_f_rescale,
				batch_bits,
				count);
	}

	/* Benchmark single-vector encoding (always f32) */
	BenchStats single_stats;
	bench_stats_init(&single_stats, runs);

	for (uint32_t run = 0; run < runs; run++)
	{
		uint64_t start = get_time_ns();
		for (uint32_t i = 0; i < count; i++)
		{
			Vec32Ref	vec_ref = {.data = f32_vectors + i * dim, .dim = dim};
			RaBitQData *out = (RaBitQData *)(single_outputs + i * data_size);
			vs_rabitq_encode_into(params, vec_ref, cent_ref, out);
		}
		uint64_t end		 = get_time_ns();
		double	 elapsed_ms	 = ns_to_ms(end - start);
		double	 vec_per_sec = (double)count / (elapsed_ms / 1000.0);
		bench_stats_add(&single_stats, vec_per_sec);
	}
	bench_stats_compute(&single_stats);

	/* Benchmark batch encoding with current implementation (typed) */
	BenchStats batch_stats;
	bench_stats_init(&batch_stats, runs);

	for (uint32_t run = 0; run < runs; run++)
	{
		uint64_t start = get_time_ns();
		vs_rabitq_encode_batch(
				params,
				vectors,
				vec_type,
				cent_ref,
				batch_f_add,
				batch_f_rescale,
				batch_bits,
				count);
		uint64_t end		 = get_time_ns();
		double	 elapsed_ms	 = ns_to_ms(end - start);
		double	 vec_per_sec = (double)count / (elapsed_ms / 1000.0);
		bench_stats_add(&batch_stats, vec_per_sec);
	}
	bench_stats_compute(&batch_stats);

#ifdef VS_HAVE_CBLAS
	/* Benchmark batch encoding with builtin (non-CBLAS) for comparison */
	BenchStats builtin_stats;
	bench_stats_init(&builtin_stats, runs);

	vs_matrix_set_use_cblas(false); /* Switch to builtin */
	for (uint32_t run = 0; run < runs; run++)
	{
		uint64_t start = get_time_ns();
		vs_rabitq_encode_batch(
				params,
				vectors,
				vec_type,
				cent_ref,
				batch_f_add,
				batch_f_rescale,
				batch_bits,
				count);
		uint64_t end		 = get_time_ns();
		double	 elapsed_ms	 = ns_to_ms(end - start);
		double	 vec_per_sec = (double)count / (elapsed_ms / 1000.0);
		bench_stats_add(&builtin_stats, vec_per_sec);
	}
	bench_stats_compute(&builtin_stats);
	vs_matrix_set_use_cblas(true); /* Restore CBLAS */
#endif

	/* Print results with comparison */
	double single_ms = (double)count / (single_stats.avg / 1000.0);
	double batch_ms	 = (double)count / (batch_stats.avg / 1000.0);
	double speedup = single_stats.avg > 0 ? batch_stats.avg / single_stats.avg
										  : 0.0;

	printf("  %-12s %8.1f ms  (%7.1fK vec/s) (±%.1f%%, n=%u) [f32]\n",
		   "single",
		   single_ms,
		   single_stats.avg / 1000.0,
		   (single_stats.stddev / single_stats.avg) * 100.0,
		   runs);

	printf("  %-12s %8.1f ms  (%7.1fK vec/s) (±%.1f%%, n=%u) [%s]\n",
		   "batch",
		   batch_ms,
		   batch_stats.avg / 1000.0,
		   (batch_stats.stddev / batch_stats.avg) * 100.0,
		   runs,
		   vs_matrix_impl_name());

#ifdef VS_HAVE_CBLAS
	double builtin_ms = (double)count / (builtin_stats.avg / 1000.0);
	printf("  %-12s %8.1f ms  (%7.1fK vec/s) (±%.1f%%, n=%u) [builtin]\n",
		   "batch",
		   builtin_ms,
		   builtin_stats.avg / 1000.0,
		   (builtin_stats.stddev / builtin_stats.avg) * 100.0,
		   runs);
	bench_stats_free(&builtin_stats);
#endif

	printf("  %-12s %.2fx faster\n", "speedup", speedup);

	bench_stats_free(&single_stats);
	bench_stats_free(&batch_stats);
	vs_free(batch_bits);
	vs_free(batch_f_rescale);
	vs_free(batch_f_add);
	vs_free(single_outputs);
}

/*
 * Entry point
 */
int
cmd_bench_quantize(CmdContext *ctx)
{
	BenchConfig config = {
			.dim		 = DEFAULT_DIM,
			.count		 = DEFAULT_COUNT,
			.runs		 = DEFAULT_RUNS,
			.op_filter	 = OP_ALL,
			.impl_filter = NULL,
			.type		 = "f32",
			.help		 = false,
	};

	static struct option long_options[] =
			{{"dim", required_argument, 0, 'd'},
			 {"count", required_argument, 0, 'c'},
			 {"runs", required_argument, 0, 'r'},
			 {"ops", required_argument, 0, 'o'},
			 {"impls", required_argument, 0, 'i'},
			 {"type", required_argument, 0, 't'},
			 {"help", no_argument, 0, 'h'},
			 {0, 0, 0, 0}};

	optind = 1;

	int opt;
	while ((opt = getopt_long(
					ctx->argc,
					ctx->argv,
					"d:c:r:o:i:t:h",
					long_options,
					NULL)) != -1)
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
		case 'o':
			config.op_filter = parse_ops(optarg);
			break;
		case 'i':
			config.impl_filter = optarg;
			break;
		case 't':
			config.type = optarg;
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

	/* Validate type */
	bool use_f16 = strcmp(config.type, "f16") == 0;
	if (!use_f16 && strcmp(config.type, "f32") != 0)
	{
		fprintf(stderr, "Error: --type must be f32 or f16\n");
		return 1;
	}

	VecType vec_type;
	if (use_f16)
	{
#if defined(VS_F16C_SUPPORT) && !defined(VS_SIMD_NONE)
		vec_type = (vs_detect_simd() & SIMD_AVX2) ? VS_VEC_F16C : VS_VEC_F16;
#else
		vec_type = VS_VEC_F16;
#endif
	}
	else
		vec_type = VS_VEC_F32;
	size_t elem_size = vs_vec_element_size(vec_type);

	/* Memory calculation */
	size_t vectors_size = (size_t)count * dim * elem_size;
	size_t encoded_size = (size_t)count * VS_RABITQ_DATA_SIZE(dim);
	size_t total_mb		= (vectors_size + encoded_size) / (1024 * 1024);

	printf("RaBitQ Quantization Benchmark\n");
	printf("  Dimension: %u\n", dim);
	printf("  Vectors:   %u\n", count);
	printf("  Type:      %s\n", vs_vec_type_name(vec_type));
	printf("  Memory:    ~%zu MB\n\n", total_mb);

	/* Create RaBitQ parameters */
	RaBitQParams *params = vs_rabitq_create(dim, 42);
	if (params == NULL)
	{
		fprintf(stderr, "Error: Failed to create RaBitQ params\n");
		return 1;
	}

	/* Generate test data */
	srand(42);

	/* Generate typed vectors (native f16 or f32) */
	void *vectors = vs_alloc(vectors_size);
	if (vectors == NULL)
	{
		fprintf(stderr, "Error: Failed to allocate vectors\n");
		vs_rabitq_destroy(params);
		return 1;
	}
	if (use_f16)
		generate_random_halfvecs((half *)vectors, count, dim);
	else
		generate_random_vectors((float *)vectors, count, dim);

	/* f32 copy for single-encode and distance paths (Vec32Ref is f32) */
	float *f32_vectors = NULL;
	if (use_f16)
	{
		f32_vectors = vs_alloc((size_t)count * dim * sizeof(float));
		if (f32_vectors == NULL)
		{
			fprintf(stderr, "Error: Failed to allocate f32 vectors\n");
			vs_free(vectors);
			vs_rabitq_destroy(params);
			return 1;
		}
		vs_half_to_float_array(
				(const half *)vectors, f32_vectors, (uint32_t)count * dim);
	}
	else
	{
		f32_vectors = (float *)vectors;
	}

	float *centroid = vs_alloc(dim * sizeof(float));
	if (centroid == NULL)
	{
		fprintf(stderr, "Error: Failed to allocate centroid\n");
		if (use_f16)
			vs_free(f32_vectors);
		vs_free(vectors);
		vs_rabitq_destroy(params);
		return 1;
	}
	generate_random_vectors(centroid, 1, dim);

	float *query = vs_alloc(dim * sizeof(float));
	if (query == NULL)
	{
		fprintf(stderr, "Error: Failed to allocate query\n");
		vs_free(centroid);
		if (use_f16)
			vs_free(f32_vectors);
		vs_free(vectors);
		vs_rabitq_destroy(params);
		return 1;
	}
	generate_random_vectors(query, 1, dim);

	/* Run encoding benchmarks */
	bool run_single = (config.op_filter & OP_ENCODE_SINGLE) != 0;
	bool run_batch	= (config.op_filter & OP_ENCODE_BATCH) != 0;

	if (run_single && run_batch)
	{
		/* Run comparison benchmark showing both with speedup */
		benchmark_encode_comparison(
				params,
				vectors,
				vec_type,
				f32_vectors,
				centroid,
				count,
				dim,
				config.runs);
#ifdef VS_HAVE_FAISS
		benchmark_faiss_encode(
				params, f32_vectors, centroid, count, dim, config.runs);
#endif
		printf("\n");
	}
	else
	{
		if (run_single)
		{
			benchmark_encode_single(
					params, f32_vectors, centroid, count, dim, config.runs);
			printf("\n");
		}

		if (run_batch)
		{
			benchmark_encode_batch(
					params,
					vectors,
					vec_type,
					centroid,
					count,
					dim,
					config.runs);
#ifdef VS_HAVE_FAISS
			benchmark_faiss_encode(
					params, f32_vectors, centroid, count, dim, config.runs);
#endif
			printf("\n");
		}
	}

	/* For distance benchmark, we need pre-encoded vectors (always f32) */
	if (config.op_filter & OP_DISTANCE)
	{
		/* Encode all vectors */
		RaBitQData **encoded = vs_alloc(count * sizeof(void *));
		if (encoded == NULL)
		{
			fprintf(stderr, "Error: Failed to allocate encoded array\n");
		}
		else
		{
			Vec32Ref cent_ref = {.data = centroid, .dim = dim};
			bool	 ok		  = true;

			for (uint32_t i = 0; i < count && ok; i++)
			{
				Vec32Ref vec_ref = {.data = f32_vectors + i * dim, .dim = dim};
				encoded[i]		 = vs_rabitq_encode(params, vec_ref, cent_ref);
				if (encoded[i] == NULL)
				{
					fprintf(stderr, "Error: Failed to encode vector %u\n", i);
					ok = false;
				}
			}

			if (ok)
			{
				benchmark_distance(
						params,
						encoded,
						query,
						centroid,
						count,
						dim,
						config.runs,
						config.impl_filter);
#ifdef VS_HAVE_FAISS
				benchmark_faiss_distance(
						f32_vectors, query, count, dim, config.runs);
				compare_correctness(
						params, f32_vectors, query, centroid, count, dim);
#endif
			}

			/* Cleanup encoded vectors */
			for (uint32_t i = 0; i < count; i++)
			{
				if (encoded[i])
					vs_free(encoded[i]);
			}
			vs_free(encoded);
		}
	}

	/* Cleanup */
	vs_free(query);
	vs_free(centroid);
	if (use_f16)
		vs_free(f32_vectors);
	vs_free(vectors);
	vs_rabitq_destroy(params);

	return 0;
}
