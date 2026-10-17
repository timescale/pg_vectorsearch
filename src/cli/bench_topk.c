/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * vectorsearch bench topk
 *
 * Benchmark the top-K collection a posting scan builds per query. Each
 * query feeds one top-K a stream of candidates the way the scan does:
 * prune on the lower bound, then vs_topk_insert, which keeps the better
 * bound when an id repeats (a SOAR replica). Ids are posting-encoded
 * TIDs. The insert and the deduplicating extract are timed separately.
 * A third column feeds unique ids through vs_topk_insert_unique, the
 * path centroid search and the rerank use.
 */

#include "vs_config.h"

#include <getopt.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "algo/topk.h"
#include "cmd.h"
#include "core/memory.h"
#include "kv.h"

#define DEFAULT_K		 "10,100,1000,5000"
#define DEFAULT_COUNT	 50000
#define DEFAULT_QUERIES	 20
#define DEFAULT_RUNS	 5
#define DEFAULT_REPLICAS 0.2
#define MAX_K_VALUES	 16

/* Heap rows per page for 128-dim vectors: TID offsets run 1..14. */
#define ROWS_PER_PAGE 14

typedef struct
{
	uint32_t ks[MAX_K_VALUES];
	uint32_t nks;
	uint32_t count;
	uint32_t queries;
	uint32_t runs;
	double	 replicas;
	bool	 help;
} BenchConfig;

/* One query's candidate stream. */
typedef struct
{
	Distance *dist;
	Distance *err;
	uint64_t *ids;
	uint64_t *unique_ids;
} Stream;

static inline uint64_t
get_time_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static inline double
unit_rand(uint32_t *rng)
{
	return (double)cmd_rand(rng) / 4294967296.0;
}

static uint64_t
random_tid(uint32_t *rng)
{
	uint64_t block	= 1 + cmd_rand(rng) % (1u << 22);
	uint64_t offset = 1 + cmd_rand(rng) % ROWS_PER_PAGE;

	return (block << 16) | offset;
}

/*
 * Distances near 1.0 with error margins of a few percent, so part of
 * the stream overlaps the threshold. A share of the ids repeat an
 * earlier id, the second copy of a vector stored in two posting lists.
 */
static void
fill_stream(Stream *s, uint32_t count, double replicas, uint32_t seed)
{
	uint32_t rng = seed;

	for (uint32_t i = 0; i < count; i++)
	{
		double u = unit_rand(&rng) + 1e-12;
		double v = unit_rand(&rng);

		s->dist[i]		 = (Distance)(1.0 + 0.12 * sqrt(-2.0 * log(u)) *
												cos(2.0 * M_PI * v));
		s->err[i]		 = (Distance)(0.03 + 0.02 * unit_rand(&rng));
		s->ids[i]		 = (i > 0 && unit_rand(&rng) < replicas)
								 ? s->ids[cmd_rand(&rng) % i]
								 : random_tid(&rng);
		s->unique_ids[i] = random_tid(&rng);
	}
}

typedef struct
{
	double insert_us;
	double extract_us;
	double unique_us;
	double survivors;
} KResult;

/*
 * The unique inserts get a top-K of their own. A top-K that has seen a
 * deduplicating insert keeps its id map current, while the ones centroid
 * search and the rerank fill never build one.
 */
static void
run_k(const BenchConfig *config,
	  const Stream		*streams,
	  VsTopK			*topk,
	  VsTopK			*unique_topk,
	  VsTopKEntry		*out,
	  uint32_t			 k,
	  KResult			*res)
{
	uint64_t best_insert  = UINT64_MAX;
	uint64_t best_extract = UINT64_MAX;
	uint64_t best_unique  = UINT64_MAX;
	uint64_t survivors	  = 0;

	for (uint32_t run = 0; run < config->runs; run++)
	{
		uint64_t insert_ns	= 0;
		uint64_t extract_ns = 0;
		uint64_t unique_ns	= 0;

		survivors = 0;
		for (uint32_t q = 0; q < config->queries; q++)
		{
			const Stream *s = &streams[q];
			uint32_t	  nout;

			vs_topk_reset_to_k(topk, k);
			uint64_t t0 = get_time_ns();
			for (uint32_t i = 0; i < config->count; i++)
			{
				if (!(s->dist[i] - s->err[i] < vs_topk_threshold(topk)))
					continue;
				vs_topk_insert(topk, s->dist[i], s->err[i], s->ids[i]);
			}
			uint64_t t1 = get_time_ns();
			vs_topk_extract_sorted(topk, out, &nout);
			uint64_t t2 = get_time_ns();

			insert_ns += t1 - t0;
			extract_ns += t2 - t1;
			survivors += topk->cand_count;

			vs_topk_reset_to_k(unique_topk, k);
			t0 = get_time_ns();
			for (uint32_t i = 0; i < config->count; i++)
			{
				if (!(s->dist[i] - s->err[i] < vs_topk_threshold(unique_topk)))
					continue;
				vs_topk_insert_unique(
						unique_topk, s->dist[i], s->err[i], s->unique_ids[i]);
			}
			unique_ns += get_time_ns() - t0;
		}

		if (insert_ns < best_insert)
			best_insert = insert_ns;
		if (extract_ns < best_extract)
			best_extract = extract_ns;
		if (unique_ns < best_unique)
			best_unique = unique_ns;
	}

	double per_query_us = 1e-3 / (double)config->queries;

	res->insert_us	= (double)best_insert * per_query_us;
	res->extract_us = (double)best_extract * per_query_us;
	res->unique_us	= (double)best_unique * per_query_us;
	res->survivors	= cmd_ratio((double)survivors, (double)config->queries);
}

static bool
parse_k_list(const char *arg, BenchConfig *config)
{
	config->nks = 0;
	while (*arg != '\0')
	{
		char *end;
		long  k = strtol(arg, &end, 10);

		if (end == arg || k < 1 || k > (long)UINT32_MAX ||
			(*end != ',' && *end != '\0') || config->nks == MAX_K_VALUES)
			return false;
		config->ks[config->nks++] = (uint32_t)k;
		arg						  = (*end == ',') ? end + 1 : end;
	}
	return config->nks > 0;
}

static void
print_usage(CmdContext *ctx)
{
	CMD_USAGE_HEADER(ctx, "bench topk");
	printf("Benchmark the top-K insert and deduplicating extract.\n\n");
	printf("Options:\n");
	printf("  --k <list>        Comma-separated k values "
		   "(default: %s)\n",
		   DEFAULT_K);
	printf("  --count <n>       Candidates per query (default: %d)\n",
		   DEFAULT_COUNT);
	printf("  --queries <n>     Queries per run (default: %d)\n",
		   DEFAULT_QUERIES);
	printf("  --runs <n>        Runs; the fastest is reported "
		   "(default: %d)\n",
		   DEFAULT_RUNS);
	printf("  --replicas <f>    Share of candidates that repeat an "
		   "earlier id (default: %.1f)\n",
		   DEFAULT_REPLICAS);
	printf("  --help            Show this help message\n");
	CMD_USAGE_JSON();
	printf("\n");
	printf("Examples:\n");
	CMD_USAGE_EXAMPLE(ctx, "bench topk", "");
	CMD_USAGE_EXAMPLE(ctx, "bench topk", "--k 200,2000 --count 100000");
}

int
cmd_bench_topk(CmdContext *ctx)
{
	BenchConfig config = {
			.count	  = DEFAULT_COUNT,
			.queries  = DEFAULT_QUERIES,
			.runs	  = DEFAULT_RUNS,
			.replicas = DEFAULT_REPLICAS,
			.help	  = false,
	};

	parse_k_list(DEFAULT_K, &config);

	static struct option long_options[] = {
			{"k", required_argument, 0, 'k'},
			{"count", required_argument, 0, 'c'},
			{"queries", required_argument, 0, 'q'},
			{"runs", required_argument, 0, 'r'},
			{"replicas", required_argument, 0, 'p'},
			{"help", no_argument, 0, 'h'},
			{0, 0, 0, 0},
	};

	optind = 1;
	int opt;
	while ((opt = getopt_long(
					ctx->argc,
					ctx->argv,
					"k:c:q:r:p:h",
					long_options,
					NULL)) != -1)
	{
		switch (opt)
		{
		case 'k':
			if (!parse_k_list(optarg, &config))
			{
				fprintf(stderr,
						"Error: --k takes up to %d positive integers\n",
						MAX_K_VALUES);
				return 1;
			}
			break;
		case 'c':
			config.count = (uint32_t)atoi(optarg);
			break;
		case 'q':
			config.queries = (uint32_t)atoi(optarg);
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

	if (config.count == 0 || config.queries == 0 || config.runs == 0)
	{
		fprintf(stderr, "Error: --count, --queries and --runs must be > 0\n");
		return 1;
	}

	uint32_t max_k = 0;
	for (uint32_t i = 0; i < config.nks; i++)
		if (config.ks[i] > max_k)
			max_k = config.ks[i];

	Stream *streams = vs_alloc(config.queries * sizeof(Stream));
	for (uint32_t q = 0; q < config.queries; q++)
	{
		streams[q].dist		  = vs_alloc(config.count * sizeof(Distance));
		streams[q].err		  = vs_alloc(config.count * sizeof(Distance));
		streams[q].ids		  = vs_alloc(config.count * sizeof(uint64_t));
		streams[q].unique_ids = vs_alloc(config.count * sizeof(uint64_t));
		fill_stream(&streams[q], config.count, config.replicas, 42 + q);
	}
	VsTopKEntry *out = vs_alloc(config.count * sizeof(VsTopKEntry));

	VsTopK topk;
	VsTopK unique_topk;
	vs_topk_init(&topk, max_k);
	vs_topk_init(&unique_topk, max_k);

	printf("Top-K benchmark\n");
	printf("===============\n\n");
	printf("%u candidates per query, %.0f%% replicas, %u queries, "
		   "best of %u runs. Times are microseconds per query.\n\n",
		   config.count,
		   config.replicas * 100.0,
		   config.queries,
		   config.runs);

	KvBuild *kb		 = NULL;
	Kv		*root	 = NULL;
	Kv		*metrics = NULL;
	if (ctx->json_path != NULL)
	{
		kb	 = kv_build_open();
		root = kv_obj();
		kv_obj_put(root, "bench", kv_str("topk"));
		kv_obj_put(root, "count", kv_int(config.count));
		kv_obj_put(root, "queries", kv_int(config.queries));
		kv_obj_put(root, "runs", kv_int(config.runs));
		kv_obj_put(root, "replicas", kv_dbl(config.replicas));
		metrics = kv_arr();
		kv_obj_put(root, "metrics", metrics);
	}

	printf("%8s %10s %12s %12s %12s\n",
		   "k",
		   "survivors",
		   "insert us",
		   "extract us",
		   "unique us");

	for (uint32_t i = 0; i < config.nks; i++)
	{
		KResult res;

		run_k(&config, streams, &topk, &unique_topk, out, config.ks[i], &res);
		printf("%8u %10.0f %12.1f %12.1f %12.1f\n",
			   config.ks[i],
			   res.survivors,
			   res.insert_us,
			   res.extract_us,
			   res.unique_us);
		kv_arr_add_metric(
				metrics, "us", res.insert_us, "k%u/insert", config.ks[i]);
		kv_arr_add_metric(
				metrics, "us", res.extract_us, "k%u/extract", config.ks[i]);
		kv_arr_add_metric(
				metrics, "us", res.unique_us, "k%u/unique", config.ks[i]);
	}

	vs_topk_cleanup(&unique_topk);
	vs_topk_cleanup(&topk);
	vs_free(out);
	for (uint32_t q = 0; q < config.queries; q++)
	{
		vs_free(streams[q].dist);
		vs_free(streams[q].err);
		vs_free(streams[q].ids);
		vs_free(streams[q].unique_ids);
	}
	vs_free(streams);

	int rc = cmd_kv_print(root, ctx->json_path);
	kv_build_close(kb);
	return rc;
}
