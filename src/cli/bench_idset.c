/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * vectorsearch bench idset
 *
 * Benchmark the id set the candidate extract deduplicates with. Each run
 * adds n posting-encoded TIDs, a share of them repeats of earlier ids as
 * SOAR replicas produce, to a set sized for n. Besides the time, it
 * reports the longest run of occupied slots, the most slots any lookup
 * walks. That figure depends on the hash, not on the machine.
 */

#include "vs_config.h"

#include <getopt.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "bench.h"
#include "cmd.h"
#include "core/idset.h"
#include "core/memory.h"
#include "kv.h"

#define DEFAULT_N		 "1000,10000,100000"
#define DEFAULT_RUNS	 5
#define DEFAULT_REPLICAS 0.2
#define MAX_N_VALUES	 16

/* Heap rows per page for 128-dim vectors: TID offsets run 1..14. */
#define ROWS_PER_PAGE 14

typedef struct
{
	uint32_t ns[MAX_N_VALUES];
	uint32_t nns;
	uint32_t runs;
	double	 replicas;
	bool	 help;
} BenchConfig;

static uint64_t
random_tid(uint32_t *rng)
{
	uint64_t block	= 1 + cmd_rand(rng) % (1u << 22);
	uint64_t offset = 1 + cmd_rand(rng) % ROWS_PER_PAGE;

	return (block << 16) | offset;
}

static void
fill_ids(uint64_t *ids, uint32_t n, double replicas, uint32_t seed)
{
	uint32_t rng = seed;

	for (uint32_t i = 0; i < n; i++)
	{
		bool repeat = i > 0 && cmd_rand(&rng) / 4294967296.0 < replicas;

		ids[i] = repeat ? ids[cmd_rand(&rng) % i] : random_tid(&rng);
	}
}

static uint32_t
longest_run(const VsIdSet *set)
{
	uint64_t nslots	 = (uint64_t)set->mask + 1;
	uint32_t longest = 0;
	uint32_t run	 = 0;

	/* Two passes over the slots, so a run that wraps is measured whole. */
	for (uint64_t i = 0; i < 2 * nslots; i++)
	{
		if (set->slots[i & set->mask] == 0)
			run = 0;
		else if (++run > longest)
			longest = run;
	}
	return longest;
}

static bool
parse_n_list(const char *arg, BenchConfig *config)
{
	config->nns = 0;
	while (*arg != '\0')
	{
		char *end;
		long  n = strtol(arg, &end, 10);

		if (end == arg || n < 1 || n > (1L << 30) ||
			(*end != ',' && *end != '\0') || config->nns == MAX_N_VALUES)
			return false;
		config->ns[config->nns++] = (uint32_t)n;
		arg						  = (*end == ',') ? end + 1 : end;
	}
	return config->nns > 0;
}

static void
print_usage(CmdContext *ctx)
{
	CMD_USAGE_HEADER(ctx, "bench idset");
	printf("Benchmark the id set the candidate extract deduplicates "
		   "with.\n\n");
	printf("Options:\n");
	printf("  --n <list>        Comma-separated id counts (default: %s)\n",
		   DEFAULT_N);
	printf("  --runs <n>        Runs; the fastest is reported "
		   "(default: %d)\n",
		   DEFAULT_RUNS);
	printf("  --replicas <f>    Share of ids that repeat an earlier id "
		   "(default: %.1f)\n",
		   DEFAULT_REPLICAS);
	printf("  --help            Show this help message\n");
	CMD_USAGE_JSON();
	printf("\n");
	printf("Examples:\n");
	CMD_USAGE_EXAMPLE(ctx, "bench idset", "");
	CMD_USAGE_EXAMPLE(ctx, "bench idset", "--n 5000,50000 --runs 10");
}

int
cmd_bench_idset(CmdContext *ctx)
{
	BenchConfig config = {
			.runs	  = DEFAULT_RUNS,
			.replicas = DEFAULT_REPLICAS,
			.help	  = false,
	};

	parse_n_list(DEFAULT_N, &config);

	static struct option long_options[] = {
			{"n", required_argument, 0, 'n'},
			{"runs", required_argument, 0, 'r'},
			{"replicas", required_argument, 0, 'p'},
			{"help", no_argument, 0, 'h'},
			{0, 0, 0, 0},
	};

	optind = 1;
	int opt;
	while ((opt = getopt_long(
					ctx->argc, ctx->argv, "n:r:p:h", long_options, NULL)) !=
		   -1)
	{
		switch (opt)
		{
		case 'n':
			if (!parse_n_list(optarg, &config))
			{
				fprintf(stderr,
						"Error: --n takes up to %d counts from 1 to 2^30\n",
						MAX_N_VALUES);
				return 1;
			}
			break;
		case 'r':
			config.runs = (uint32_t)atoi(optarg);
			break;
		case 'p':
			config.replicas = atof(optarg);
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

	if (config.runs == 0)
	{
		fprintf(stderr, "Error: --runs must be > 0\n");
		return 1;
	}

	printf("Id set benchmark\n");
	printf("================\n\n");
	printf("Posting-encoded TIDs, %.0f%% replicas, best of %u runs. "
		   "Time is per set built.\n\n",
		   config.replicas * 100.0,
		   config.runs);

	KvBuild *kb		 = NULL;
	Kv		*root	 = NULL;
	Kv		*metrics = NULL;
	if (ctx->json_path != NULL)
	{
		kb	 = kv_build_open();
		root = kv_obj();
		kv_obj_put(root, "bench", kv_str("idset"));
		kv_obj_put(root, "runs", kv_int(config.runs));
		kv_obj_put(root, "replicas", kv_dbl(config.replicas));
		metrics = kv_arr();
		kv_obj_put(root, "metrics", metrics);
	}

	printf("%10s %10s %12s %10s %12s\n",
		   "ids",
		   "distinct",
		   "time us",
		   "ns per id",
		   "longest run");

	for (uint32_t i = 0; i < config.nns; i++)
	{
		uint32_t  n		   = config.ns[i];
		uint64_t *ids	   = vs_alloc(n * sizeof(uint64_t));
		uint64_t  best	   = UINT64_MAX;
		uint32_t  distinct = 0;
		uint32_t  longest  = 0;

		/* Small sets are rebuilt until a run covers about a million ids,
		 * so the time is well above the clock's resolution. */
		uint32_t reps = n < 1000000 ? 1000000 / n : 1;

		fill_ids(ids, n, config.replicas, 42 + n);

		for (uint32_t run = 0; run < config.runs; run++)
		{
			uint64_t t0 = bench_now_ns();

			for (uint32_t rep = 0; rep < reps; rep++)
			{
				VsIdSet set;

				distinct = 0;
				vs_idset_init(&set, n);
				for (uint32_t j = 0; j < n; j++)
					distinct += vs_idset_test_add(&set, ids[j]);
				if (rep == reps - 1)
					longest = longest_run(&set);
				vs_idset_cleanup(&set);
			}

			uint64_t elapsed = bench_now_ns() - t0;
			if (elapsed < best)
				best = elapsed;
		}

		double per_set_ns = cmd_ratio((double)best, (double)reps);

		printf("%10u %10u %12.1f %10.1f %12u\n",
			   n,
			   distinct,
			   per_set_ns / 1e3,
			   cmd_ratio(per_set_ns, (double)n),
			   longest);
		kv_arr_add_metric(metrics, "us", per_set_ns / 1e3, "n%u/build", n);
		kv_arr_add_metric(metrics, "slots", longest, "n%u/longest_run", n);
		vs_free(ids);
	}

	int rc = cmd_kv_print(root, ctx->json_path);
	kv_build_close(kb);
	return rc;
}
