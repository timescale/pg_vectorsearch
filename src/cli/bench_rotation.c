/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * vectorsearch bench rotation
 *
 * Benchmark the two RaBitQ rotations at each dimension: generating the
 * parameters from a seed, and applying them to a vector. The dense rotation
 * is a dim x dim random orthogonal matrix and works at any dimension; the
 * fast one is a randomized Hadamard transform and only works where the
 * dimension factors suitably. RaBitQ uses the fast one wherever it can.
 */

#include "vs_config.h"

#include <getopt.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "bench.h"
#include "cmd.h"
#include "core/memory.h"
#include "kv.h"
#include "quant/fast_rotate.h"
#include "quant/matrix.h"
#include "quant/rabitq.h"

/* Dimensions only the dense rotation supports, including the largest one an
 * index accepts, plus a few the fast rotation covers for comparison. */
#define DEFAULT_DIMS	  "100,300,768,960,1968,4096,8192"
#define DEFAULT_RUNS	  3
#define DEFAULT_DENSE_MAX 1968
#define MAX_DIMS		  32
#define MAX_RUNS		  1000
#define SEED			  42

typedef enum RotationCase
{
	ROTATION_DENSE_GENERATE = 0,
	ROTATION_DENSE_APPLY,
	ROTATION_FAST_GENERATE,
	ROTATION_FAST_APPLY,
	ROTATION_NUM_CASES,
} RotationCase;

static const char *rotation_case_names[] = {
		[ROTATION_DENSE_GENERATE] = "dense/generate",
		[ROTATION_DENSE_APPLY]	  = "dense/apply",
		[ROTATION_FAST_GENERATE]  = "fast/generate",
		[ROTATION_FAST_APPLY]	  = "fast/apply",
};

typedef struct
{
	uint32_t dims[MAX_DIMS];
	uint32_t ndims;
	uint32_t runs;
	uint32_t dense_max;
	uint32_t min_runtime;
	bool	 help;
} BenchConfig;

/* One dimension's cases. P is the dense matrix both dense cases use. */
typedef struct
{
	Dimension dim;
	float	 *P;
	Kv		 *metrics;
} RotationCtx;

/* What one case's passes touch, set up untimed by rotation_prepare. */
typedef struct
{
	VsFastRotateParams fr;
	RaBitQParams	  *params;
	float			  *in;
	float			  *out;
} RotationState;

static bool
is_generate(const BenchCase *bench_case)
{
	return bench_case->variant == ROTATION_DENSE_GENERATE ||
		   bench_case->variant == ROTATION_FAST_GENERATE;
}

static int
rotation_prepare(void *vctx, BenchCase *bench_case)
{
	RotationCtx	  *ctx	 = vctx;
	RotationState *state = vs_alloc0(sizeof(RotationState));

	bench_case->state = state;
	if (is_generate(bench_case))
		return 0;

	/* vs_rabitq_create picks the fast rotation wherever it is supported. */
	state->params =
			bench_case->variant == ROTATION_DENSE_APPLY
					? vs_rabitq_create_from_matrix(ctx->dim, SEED, ctx->P)
					: vs_rabitq_create(ctx->dim, SEED);
	state->in  = vs_alloc((size_t)ctx->dim * sizeof(float));
	state->out = vs_alloc((size_t)ctx->dim * sizeof(float));

	uint32_t rng = SEED;
	bench_fill_random(state->in, ctx->dim, &rng);
	return state->params == NULL ? -1 : 0;
}

static int
rotation_pass(void *vctx, const BenchCase *bench_case)
{
	RotationCtx	  *ctx	 = vctx;
	RotationState *state = bench_case->state;

	switch ((RotationCase)bench_case->variant)
	{
	case ROTATION_DENSE_GENERATE:
		return vs_random_orthogonal_matrix(ctx->P, ctx->dim, SEED);
	case ROTATION_FAST_GENERATE:
		vs_fast_rotate_init(&state->fr, ctx->dim, SEED);
		return 0;
	case ROTATION_DENSE_APPLY:
	case ROTATION_FAST_APPLY:
		vs_rabitq_rotate(state->params, state->in, state->out);
		return 0;
	case ROTATION_NUM_CASES:
		break;
	}
	return -1;
}

static void
rotation_report(void *vctx, const BenchCase *bench_case, uint32_t runs)
{
	RotationCtx *ctx	 = vctx;
	bool		 gen	 = is_generate(bench_case);
	const char	*unit	 = gen ? "ms" : "us";
	double		 per_sec = gen ? 1e3 : 1e6;
	char		 prefix[16];

	printf("%6u %-16s %14.3f %-2s",
		   ctx->dim,
		   bench_case->name,
		   bench_seconds_per_pass(bench_case) * per_sec,
		   unit);
	if (runs > 1)
		printf("  (±%.1f%%, n=%u)", bench_rsd_percent(bench_case), runs);
	printf("\n");

	snprintf(prefix, sizeof(prefix), "dim%u", ctx->dim);
	bench_record_time(ctx->metrics, prefix, bench_case, unit, per_sec, runs);
}

static void
rotation_teardown(void *vctx, BenchCase *bench_case)
{
	RotationState *state = bench_case->state;

	(void)vctx;
	if (state == NULL)
		return;
	if (state->params != NULL)
		vs_rabitq_destroy(state->params);
	vs_free(state->in);
	vs_free(state->out);
	vs_free(state);
	bench_case->state = NULL;
}

/* Print a row for each case from first to last, which is not timed. */
static void
report_untimed(
		Dimension	 dim,
		RotationCase first,
		RotationCase last,
		const char	*reason)
{
	for (uint32_t v = first; v <= last; v++)
		printf("%6u %-16s %17s\n", dim, rotation_case_names[v], reason);
}

/* Time both rotations at one dimension, the dense one only up to dense_max.
 * A case that is not timed still gets a row saying why. */
static int
bench_dimension(const BenchConfig *config, Dimension dim, Kv *metrics)
{
	bool		dense = dim <= config->dense_max;
	bool		fast  = vs_fast_rotate_supported(dim);
	BenchCase	cases[ROTATION_NUM_CASES];
	uint32_t	ncases = 0;
	RotationCtx ctx	   = {.dim = dim, .metrics = metrics};
	int			ret	   = 0;

	for (uint32_t v = 0; v < ROTATION_NUM_CASES; v++)
	{
		if (v >= ROTATION_FAST_GENERATE ? !fast : !dense)
			continue;
		cases[ncases++] = (BenchCase){
				.name	 = rotation_case_names[v],
				.variant = v,
		};
	}

	/* Generated once untimed, for the apply case to build its params from.
	 * The generate case rewrites it with the same matrix every pass.
	 *
	 * The harness reports every case once all are timed, so a skipped
	 * dense row goes before them and an unsupported fast row after. */
	if (dense)
	{
		ctx.P = vs_alloc((size_t)dim * dim * sizeof(float));
		vs_random_orthogonal_matrix(ctx.P, dim, SEED);
	}
	else
		report_untimed(
				dim, ROTATION_DENSE_GENERATE, ROTATION_DENSE_APPLY, "skipped");

	if (ncases > 0)
	{
		/* No warmup, since a dense generate pass can take seconds. */
		BenchSpec spec = {
				.runs			= config->runs,
				.warmup_passes	= 0,
				.min_runtime_ms = config->min_runtime,
				.prepare		= rotation_prepare,
				.pass			= rotation_pass,
				.report			= rotation_report,
				.teardown		= rotation_teardown,
				.ctx			= &ctx,
		};
		ret = bench_run(&spec, cases, ncases);
	}

	if (!fast)
		report_untimed(
				dim,
				ROTATION_FAST_GENERATE,
				ROTATION_FAST_APPLY,
				"not supported");

	vs_free(ctx.P);
	return ret;
}

static bool
parse_dim_list(const char *arg, BenchConfig *config)
{
	config->ndims = 0;
	while (*arg != '\0')
	{
		char *end;
		long  d = strtol(arg, &end, 10);

		if (end == arg || d < 1 || d > VS_FAST_ROTATE_MAX_DIM ||
			(*end != ',' && *end != '\0') || config->ndims == MAX_DIMS)
			return false;
		config->dims[config->ndims++] = (uint32_t)d;
		arg							  = (*end == ',') ? end + 1 : end;
	}
	return config->ndims > 0;
}

/* Parse a whole decimal number from min to max into *out. */
static bool
parse_uint(const char *arg, long min, long max, uint32_t *out)
{
	char *end;
	long  v = strtol(arg, &end, 10);

	if (end == arg || *end != '\0' || v < min || v > max)
		return false;
	*out = (uint32_t)v;
	return true;
}

static void
print_usage(CmdContext *ctx)
{
	CMD_USAGE_HEADER(ctx, "bench rotation");
	printf("Benchmark generating and applying the RaBitQ rotations.\n\n");
	printf("Options:\n");
	printf("  --dims <list>     Comma-separated dimensions (default: %s)\n",
		   DEFAULT_DIMS);
	printf("  --runs <n>        Timed samples per case (default: %d)\n",
		   DEFAULT_RUNS);
	printf("  --dense-max <d>   Largest dimension to time the dense rotation "
		   "at;\n"
		   "                    it costs O(dim^3) to generate "
		   "(default: %d)\n",
		   DEFAULT_DENSE_MAX);
	printf("  --help            Show this help message\n");
	CMD_USAGE_JSON();
	printf("\n");
	printf("Examples:\n");
	CMD_USAGE_EXAMPLE(ctx, "bench rotation", "");
	CMD_USAGE_EXAMPLE(ctx, "bench rotation", "--dims 1968 --runs 1");
}

int
cmd_bench_rotation(CmdContext *ctx)
{
	BenchConfig config = {
			.runs		 = DEFAULT_RUNS,
			.dense_max	 = DEFAULT_DENSE_MAX,
			.min_runtime = ctx->min_runtime,
			.help		 = false,
	};

	parse_dim_list(DEFAULT_DIMS, &config);

	static struct option long_options[] = {
			{"dims", required_argument, 0, 'd'},
			{"runs", required_argument, 0, 'r'},
			{"dense-max", required_argument, 0, 'm'},
			{"help", no_argument, 0, 'h'},
			{0, 0, 0, 0},
	};

	optind = 1;
	int opt;
	while ((opt = getopt_long(
					ctx->argc, ctx->argv, "d:r:m:h", long_options, NULL)) !=
		   -1)
	{
		switch (opt)
		{
		case 'd':
			if (!parse_dim_list(optarg, &config))
			{
				fprintf(stderr,
						"Error: --dims takes up to %d dimensions from 1 to "
						"%d\n",
						MAX_DIMS,
						VS_FAST_ROTATE_MAX_DIM);
				return 1;
			}
			break;
		case 'r':
			if (!parse_uint(optarg, 1, MAX_RUNS, &config.runs))
			{
				fprintf(stderr,
						"Error: --runs takes a number from 1 to %d\n",
						MAX_RUNS);
				return 1;
			}
			break;
		case 'm':
			if (!parse_uint(
						optarg, 0, VS_FAST_ROTATE_MAX_DIM, &config.dense_max))
			{
				fprintf(stderr,
						"Error: --dense-max takes a dimension from 0 to %d\n",
						VS_FAST_ROTATE_MAX_DIM);
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

	printf("Rotation benchmark\n");
	printf("==================\n\n");
	printf("The dense rotation is timed up to %u dimensions.\n\n",
		   config.dense_max);

	Kv *metrics = NULL;
	if (ctx->json_path != NULL)
	{
		kv_obj_put(ctx->json_root, "bench", kv_str("rotation"));
		kv_obj_put(ctx->json_root, "runs", kv_int(config.runs));
		kv_obj_put(ctx->json_root, "dense_max", kv_int(config.dense_max));
		metrics = kv_arr();
		kv_obj_put(ctx->json_root, "metrics", metrics);
	}

	printf("%6s %-16s %17s\n", "dim", "case", "time");

	for (uint32_t i = 0; i < config.ndims; i++)
	{
		if (bench_dimension(&config, config.dims[i], metrics) != 0)
		{
			fprintf(stderr, "Error: rotation benchmark failed\n");
			return 1;
		}
	}

	return 0;
}
