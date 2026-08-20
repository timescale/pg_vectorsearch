/*
 * mkt bench cluster
 *
 * Benchmark K-means clustering algorithms.
 */

#include "mkt_config.h"

#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "algo/kmeans.h"
#include "algo/vecops.h"
#include "cmd.h"
#include "core/memory.h"
#include "core/platform.h"
#include "core/types.h"
#include "kmeans_pgvector.h"
#include "types/halfvec.h"

/* Default parameters */
#define DEFAULT_DIM	  128
#define DEFAULT_NVECS 100000
#define DEFAULT_NLIST 1000

/* Timing helper */
static inline uint64_t
get_time_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* Benchmark configuration */
typedef struct
{
	Dimension	   dim;
	uint32_t	   nvecs;
	uint32_t	   nlist;
	uint32_t	   iters;
	uint32_t	   nredo;
	uint64_t	   seed;
	DistanceMetric metric;
	const char	  *impl;
	const char	  *file;
	const char	  *type; /* "f32" or "f16" */
	bool		   help;
	bool		   verify;
} ClusterBenchConfig;

static DistanceMetric
parse_metric(const char *s)
{
	if (strcmp(s, "ip") == 0)
		return DISTANCE_INNER_PRODUCT;
	if (strcmp(s, "cosine") == 0)
		return DISTANCE_COSINE;
	return DISTANCE_L2;
}

static const char *
metric_name(DistanceMetric m)
{
	switch (m)
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

/*
 * Generate random vectors using simple LCG for reproducibility.
 */
static void
generate_random_vectors(
		float *data, uint32_t nvecs, Dimension dim, uint64_t seed)
{
	uint32_t rng = (uint32_t)(seed ^ (seed >> 32));
	for (size_t i = 0; i < (size_t)nvecs * dim; i++)
	{
		rng		= rng * 1103515245 + 12345;
		data[i] = ((float)(rng >> 16) / 32768.0f) - 1.0f;
	}
}

/*
 * Normalize vectors to unit length (for cosine metric).
 */
static void
normalize_vectors(float *data, uint32_t nvecs, Dimension dim)
{
	for (uint32_t i = 0; i < nvecs; i++)
	{
		float *v	= data + (size_t)i * dim;
		float  norm = mkt_l2_norm(v, dim);
		if (norm > 1e-10f)
			mkt_vector_scale(v, 1.0f / norm, v, dim);
	}
}

/*
 * Load vectors from DEFAULT binary format files.
 *
 * Format: [int32 count][int32 dim] header in first file, then raw int8
 * data spanning one or more chunk files. Used by SPTAG's SPACEV1B dataset.
 *
 * Single file: path points directly to the .bin file.
 * Multi-chunk directory: path points to directory containing vectors_N.bin
 * files (numbered 1..N, header only in vectors_1.bin).
 *
 * Vectors are converted from int8 to float32. If max_nvecs > 0 and is less
 * than the file's count, only max_nvecs vectors are loaded.
 *
 * Sets *out_nvecs and *out_dim on success. Returns allocated float array.
 */
static float *
load_vectors_from_file(
		const char *path,
		uint32_t	max_nvecs,
		uint32_t   *out_nvecs,
		Dimension  *out_dim)
{
	/* Determine if path is a directory or single file */
	struct stat st;
	if (stat(path, &st) != 0)
	{
		fprintf(stderr, "Error: cannot access '%s'\n", path);
		return NULL;
	}

	bool  is_dir = S_ISDIR(st.st_mode);
	FILE *fp;

	if (is_dir)
	{
		char first_chunk[4096];
		snprintf(first_chunk, sizeof(first_chunk), "%s/vectors_1.bin", path);
		fp = fopen(first_chunk, "rb");
		if (fp == NULL)
		{
			fprintf(stderr,
					"Error: cannot open "
					"'%s/vectors_1.bin'\n",
					path);
			return NULL;
		}
	}
	else
	{
		fp = fopen(path, "rb");
		if (fp == NULL)
		{
			fprintf(stderr, "Error: cannot open '%s'\n", path);
			return NULL;
		}
	}

	/* Read header: [int32 count][int32 dim] */
	int32_t count, dim;
	if (fread(&count, 4, 1, fp) != 1 || fread(&dim, 4, 1, fp) != 1)
	{
		fprintf(stderr, "Error: failed to read header from '%s'\n", path);
		fclose(fp);
		return NULL;
	}

	if (count <= 0 || dim <= 0 || dim > 65536)
	{
		fprintf(stderr, "Error: invalid header count=%d dim=%d\n", count, dim);
		fclose(fp);
		return NULL;
	}

	uint32_t nvecs = (uint32_t)count;
	if (max_nvecs > 0 && max_nvecs < nvecs)
		nvecs = max_nvecs;

	printf("Loading %u vectors (dim=%d) from '%s'", nvecs, dim, path);
	if (nvecs < (uint32_t)count)
		printf(" (of %d total)", count);
	printf("...\n");

	/* Allocate output float array */
	size_t float_bytes = (size_t)nvecs * dim * sizeof(float);
	float *data		   = mkt_alloc(float_bytes);
	if (data == NULL)
	{
		fprintf(stderr,
				"Error: failed to allocate %.1f MB\n",
				(double)float_bytes / (1024.0 * 1024.0));
		fclose(fp);
		return NULL;
	}

	/* Read int8 data in chunks and convert to float32 */
	size_t	total_bytes = (size_t)nvecs * dim;
	size_t	bytes_read	= 0;
	int8_t *buf			= mkt_alloc(1024 * 1024); /* 1MB read buffer */
	if (buf == NULL)
	{
		fprintf(stderr, "Error: failed to allocate read buffer\n");
		mkt_free(data);
		fclose(fp);
		return NULL;
	}

	int chunk_idx = 1; /* current chunk file number */

	while (bytes_read < total_bytes)
	{
		size_t remaining = total_bytes - bytes_read;
		size_t to_read	 = remaining < 1024 * 1024 ? remaining : 1024 * 1024;
		size_t got		 = fread(buf, 1, to_read, fp);

		if (got == 0)
		{
			fclose(fp);
			fp = NULL;

			if (!is_dir)
				break; /* single file exhausted */

			/* Try next chunk */
			chunk_idx++;
			char next_chunk[4096];
			snprintf(
					next_chunk,
					sizeof(next_chunk),
					"%s/vectors_%d.bin",
					path,
					chunk_idx);
			fp = fopen(next_chunk, "rb");
			if (fp == NULL)
				break; /* no more chunks */
			continue;
		}

		/* Convert int8 -> float32 */
		for (size_t i = 0; i < got; i++)
			data[bytes_read + i] = (float)buf[i];
		bytes_read += got;
	}

	mkt_free(buf);
	if (fp != NULL)
		fclose(fp);

	if (bytes_read < total_bytes)
	{
		fprintf(stderr,
				"Warning: only read %zu of %zu bytes "
				"(loaded %zu vectors)\n",
				bytes_read,
				total_bytes,
				bytes_read / dim);
		nvecs = (uint32_t)(bytes_read / dim);
	}

	*out_nvecs = nvecs;
	*out_dim   = (Dimension)dim;
	return data;
}

/*
 * Recompute exact total cost from assignments (no bound approximation).
 */
static double
recompute_exact_cost(
		const float		   *data,
		const KMeansResult *res,
		uint32_t			nvecs,
		Dimension			dim,
		DistanceMetric		metric)
{
	double cost = 0.0;
	for (uint32_t i = 0; i < nvecs; i++)
	{
		const float *v = data + (size_t)i * dim;
		const float *c = res->centroids + (size_t)res->assignments[i] * dim;
		switch (metric)
		{
		case DISTANCE_L2:
			cost += (double)mkt_l2_distance_squared(v, c, dim);
			break;
		case DISTANCE_INNER_PRODUCT:
			cost += (double)(-mkt_dot_product(v, c, dim));
			break;
		case DISTANCE_COSINE:
			cost += (double)(1.0f - mkt_dot_product(v, c, dim));
			break;
		}
	}
	return cost;
}

/*
 * Compare a result against a reference (Lloyd's).
 * Prints: assignment match %, max centroid distance, cost comparison.
 */
static void
verify_against_ref(
		const KMeansResult *ref,
		double				ref_cost,
		const KMeansResult *res,
		double				res_cost,
		const char		   *name,
		uint32_t			nvecs)
{
	uint32_t  nlist = ref->nlist;
	Dimension dim	= ref->dim;

	/* Assignment match rate */
	uint32_t matches = 0;
	for (uint32_t i = 0; i < nvecs; i++)
	{
		if (ref->assignments[i] == res->assignments[i])
			matches++;
	}
	double match_pct = 100.0 * matches / nvecs;

	/* Max centroid L2 distance (corresponding indices) */
	float  max_cdist = 0.0f;
	double sum_cdist = 0.0;
	for (uint32_t j = 0; j < nlist; j++)
	{
		float d = mkt_l2_distance_squared(
				ref->centroids + (size_t)j * dim,
				res->centroids + (size_t)j * dim,
				dim);
		if (d > max_cdist)
			max_cdist = d;
		sum_cdist += (double)d;
	}

	/* Cost relative error */
	double cost_err = 0.0;
	if (ref_cost > 0.0)
		cost_err = (res_cost - ref_cost) / ref_cost * 100.0;

	printf("    %-14s assign_match=%.2f%%  "
		   "max_centroid_d=%.4f  mean=%.4f  "
		   "cost_err=%+.4f%%\n",
		   name,
		   match_pct,
		   sqrtf(max_cdist),
		   sqrtf((float)(sum_cdist / nlist)),
		   cost_err);
}

/*
 * Run one benchmark and print results.
 * If out_res is non-NULL, the result is returned (caller must free).
 */
static KMeansResult *
run_bench(
		const ClusterBenchConfig *cfg,
		const void				 *data,
		MktVecType				  vec_type,
		KMeansAlgorithm			  algo)
{
	const char	 *name	= mkt_kmeans_algo_name(algo);
	KMeansOptions opts	= MKT_KMEANS_OPTIONS_DEFAULT;
	opts.max_iterations = cfg->iters;
	opts.seed			= cfg->seed;
	opts.nredo			= cfg->nredo;
	opts.algorithm		= algo;

	uint64_t	  start = get_time_ns();
	KMeansResult *res	= mkt_kmeans(
			  data,
			  NULL,
			  vec_type,
			  cfg->nvecs,
			  cfg->dim,
			  cfg->nlist,
			  cfg->metric,
			  &opts);
	uint64_t end = get_time_ns();

	if (res == NULL)
	{
		fprintf(stderr, "K-means failed (%s)\n", name);
		return NULL;
	}

	double elapsed_ms = (double)(end - start) / 1e6;
	double vecs_per_s = (double)cfg->nvecs / (elapsed_ms / 1000.0);

	/* Cluster size stats */
	uint32_t min_size = res->cluster_sizes[0];
	uint32_t max_size = res->cluster_sizes[0];
	uint32_t empty	  = 0;
	for (uint32_t j = 0; j < cfg->nlist; j++)
	{
		if (res->cluster_sizes[j] < min_size)
			min_size = res->cluster_sizes[j];
		if (res->cluster_sizes[j] > max_size)
			max_size = res->cluster_sizes[j];
		if (res->cluster_sizes[j] == 0)
			empty++;
	}

	printf("  %-14s %8.1f ms  cost=%.2f  "
		   "cluster_sizes=[%u..%u] empty=%u  "
		   "(%.0f vec/s)\n",
		   name,
		   elapsed_ms,
		   res->total_cost,
		   min_size,
		   max_size,
		   empty,
		   vecs_per_s);

	return res;
}

/*
 * Run pgvector's Elkan k-means benchmark.
 */
static KMeansResult *
run_bench_pgvector(const ClusterBenchConfig *cfg, const float *data)
{
	KMeansOptions opts	= MKT_KMEANS_OPTIONS_DEFAULT;
	opts.max_iterations = cfg->iters;
	opts.seed			= cfg->seed;
	opts.nredo			= cfg->nredo;

	uint64_t	  start = get_time_ns();
	KMeansResult *res	= pgvector_kmeans(
			  data, cfg->nvecs, cfg->dim, cfg->nlist, cfg->metric, &opts);
	uint64_t end = get_time_ns();

	if (res == NULL)
	{
		fprintf(stderr, "K-means failed (pgvector)\n");
		return NULL;
	}

	double elapsed_ms = (double)(end - start) / 1e6;
	double vecs_per_s = (double)cfg->nvecs / (elapsed_ms / 1000.0);

	/* Cluster size stats */
	uint32_t min_size = res->cluster_sizes[0];
	uint32_t max_size = res->cluster_sizes[0];
	uint32_t empty	  = 0;
	for (uint32_t j = 0; j < cfg->nlist; j++)
	{
		if (res->cluster_sizes[j] < min_size)
			min_size = res->cluster_sizes[j];
		if (res->cluster_sizes[j] > max_size)
			max_size = res->cluster_sizes[j];
		if (res->cluster_sizes[j] == 0)
			empty++;
	}

	printf("  %-14s %8.1f ms  cost=%.2f  "
		   "cluster_sizes=[%u..%u] empty=%u  "
		   "(%.0f vec/s)\n",
		   "elkan(pgvec)",
		   elapsed_ms,
		   res->total_cost,
		   min_size,
		   max_size,
		   empty,
		   vecs_per_s);

	return res;
}

static void
print_usage(CmdContext *ctx)
{
	CMD_USAGE_HEADER(ctx, "bench cluster");
	printf("Benchmark K-means clustering performance.\n\n");
	printf("Options:\n");
	printf("  --dim <int>        Vector dimension (default: %d)\n",
		   DEFAULT_DIM);
	printf("  --nvecs <int>      Number of vectors (default: %d)\n",
		   DEFAULT_NVECS);
	printf("  --nlist <int>      Number of clusters (default: %d)\n",
		   DEFAULT_NLIST);
	printf("  --iters <int>      Max iterations (default: 20)\n");
	printf("  --nredo <int>      Number of restarts (default: 1)\n");
	printf("  --seed <int>       Random seed (default: 42)\n");
	printf("  --metric <str>     l2, ip, or cosine (default: l2)\n");
	printf("  --file <path>      Load vectors from file or directory "
		   "(DEFAULT int8 format)\n");
	printf("  --type <str>       f32 or f16 (default: f32)\n");
	printf("  --impl <str>       all, cblas, lloyd, hamerly, "
		   "elkan, or pgvector (default: all)\n");
	printf("  --verify           Compare all algos against lloyd "
		   "(reference)\n");
	printf("  --help             Show this help\n");
	printf("\n");
	printf("Examples:\n");
	CMD_USAGE_EXAMPLE(
			ctx, "bench cluster", "--dim 128 --nvecs 10000 --nlist 100");
	CMD_USAGE_EXAMPLE(ctx, "bench cluster", "--impl all --metric cosine");
	CMD_USAGE_EXAMPLE(
			ctx,
			"bench cluster",
			"--file path/to/vectors.bin --nvecs 1000000");
}

int
cmd_bench_cluster(CmdContext *ctx)
{
	ClusterBenchConfig cfg = {
			.dim	= DEFAULT_DIM,
			.nvecs	= DEFAULT_NVECS,
			.nlist	= DEFAULT_NLIST,
			.iters	= 20,
			.nredo	= 1,
			.seed	= 42,
			.metric = DISTANCE_L2,
			.impl	= NULL,
			.type	= "f32",
			.help	= false,
	};

	static struct option long_options[] = {
			{"dim", required_argument, 0, 'd'},
			{"nvecs", required_argument, 0, 'n'},
			{"nlist", required_argument, 0, 'k'},
			{"iters", required_argument, 0, 'i'},
			{"nredo", required_argument, 0, 'r'},
			{"seed", required_argument, 0, 's'},
			{"metric", required_argument, 0, 'm'},
			{"impl", required_argument, 0, 'p'},
			{"file", required_argument, 0, 'f'},
			{"type", required_argument, 0, 't'},
			{"verify", no_argument, 0, 'v'},
			{"help", no_argument, 0, 'h'},
			{0, 0, 0, 0},
	};

	optind = 1;

	int opt;
	while ((opt = getopt_long(
					ctx->argc,
					ctx->argv,
					"d:n:k:i:r:s:m:p:f:t:vh",
					long_options,
					NULL)) != -1)
	{
		switch (opt)
		{
		case 'd':
			cfg.dim = (Dimension)atoi(optarg);
			break;
		case 'n':
			cfg.nvecs = (uint32_t)atoi(optarg);
			break;
		case 'k':
			cfg.nlist = (uint32_t)atoi(optarg);
			break;
		case 'i':
			cfg.iters = (uint32_t)atoi(optarg);
			break;
		case 'r':
			cfg.nredo = (uint32_t)atoi(optarg);
			break;
		case 's':
			cfg.seed = (uint64_t)atoll(optarg);
			break;
		case 'm':
			cfg.metric = parse_metric(optarg);
			break;
		case 'p':
			cfg.impl = optarg;
			break;
		case 'f':
			cfg.file = optarg;
			break;
		case 't':
			cfg.type = optarg;
			break;
		case 'v':
			cfg.verify = true;
			break;
		case 'h':
			cfg.help = true;
			break;
		default:
			print_usage(ctx);
			return 1;
		}
	}

	if (cfg.help)
	{
		print_usage(ctx);
		return 0;
	}

	/* Validate */
	if (cfg.nlist == 0)
	{
		fprintf(stderr, "Error: nlist must be > 0\n");
		return 1;
	}

	float *data;

	if (cfg.file != NULL)
	{
		/* Load from file */
		data = load_vectors_from_file(
				cfg.file, cfg.nvecs, &cfg.nvecs, &cfg.dim);
		if (data == NULL)
			return 1;
	}
	else
	{
		/* Generate random data */
		if (cfg.dim == 0 || cfg.nvecs == 0)
		{
			fprintf(stderr, "Error: dim and nvecs must be > 0\n");
			return 1;
		}

		uint64_t data_bytes = (uint64_t)cfg.nvecs * cfg.dim * sizeof(float);
		if (data_bytes > 8ULL * 1024 * 1024 * 1024)
		{
			fprintf(stderr,
					"Error: data would require %.1f GB\n",
					(double)data_bytes / (1024.0 * 1024.0 * 1024.0));
			return 1;
		}

		data = mkt_alloc(data_bytes);
		if (data == NULL)
		{
			fprintf(stderr, "Error: failed to allocate data\n");
			return 1;
		}

		generate_random_vectors(data, cfg.nvecs, cfg.dim, cfg.seed);
	}

	if (cfg.metric == DISTANCE_COSINE)
		normalize_vectors(data, cfg.nvecs, cfg.dim);

	/* Determine type and optionally convert to f16 */
	bool use_f16 = cfg.type != NULL && strcmp(cfg.type, "f16") == 0;
	if (cfg.type != NULL && strcmp(cfg.type, "f32") != 0 && !use_f16)
	{
		fprintf(stderr, "Error: --type must be f32 or f16\n");
		mkt_free(data);
		return 1;
	}

	MktVecType	vec_type   = MKT_VEC_F32;
	const void *bench_data = data;
	half	   *data_f16   = NULL;

	if (use_f16)
	{
		size_t n_elems = (size_t)cfg.nvecs * cfg.dim;
		data_f16	   = mkt_alloc(n_elems * sizeof(half));
		mkt_float_to_half_array(data, data_f16, (uint32_t)n_elems);
		bench_data = data_f16;
#if defined(MKT_F16C_SUPPORT) && !defined(MKT_SIMD_NONE)
		vec_type = (mkt_detect_simd() & SIMD_AVX2) ? MKT_VEC_F16C
												   : MKT_VEC_F16;
#else
		vec_type = MKT_VEC_F16;
#endif
	}

	size_t elem_size = mkt_vec_element_size(vec_type);
	double size_mb	 = (double)cfg.nvecs * cfg.dim * elem_size /
					 (1024.0 * 1024.0);
	printf("K-means clustering (%s, %s, dim=%u, nvecs=%u, nlist=%u, "
		   "%.1f MB):\n",
		   metric_name(cfg.metric),
		   mkt_vec_type_name(vec_type),
		   cfg.dim,
		   cfg.nvecs,
		   cfg.nlist,
		   size_mb);

	/* Run benchmarks */
	bool run_all = cfg.impl == NULL || strcmp(cfg.impl, "all") == 0;

	/* Struct to hold results for verify mode */
	struct
	{
		KMeansResult *res;
		const char	 *name;
	} results[5];

	int nresults = 0;

	if (run_all || (cfg.impl != NULL && strcmp(cfg.impl, "cblas") == 0))
	{
		KMeansResult *r =
				run_bench(&cfg, bench_data, vec_type, KMEANS_ALGO_CBLAS);
		if (cfg.verify && r != NULL)
			results[nresults++] = (typeof(results[0])){r, "lloyd(cblas)"};
		else
			mkt_kmeans_result_destroy(r);
	}

	/* Lloyd's: always run in verify mode (reference), else on demand */
	KMeansResult *ref	   = NULL;
	double		  ref_cost = 0.0;
	if (cfg.verify || run_all ||
		(cfg.impl != NULL && strcmp(cfg.impl, "lloyd") == 0))
	{
		ref = run_bench(&cfg, bench_data, vec_type, KMEANS_ALGO_LLOYD);
		if (cfg.verify && ref != NULL)
			ref_cost = recompute_exact_cost(
					data, ref, cfg.nvecs, cfg.dim, cfg.metric);
		else if (!cfg.verify)
		{
			mkt_kmeans_result_destroy(ref);
			ref = NULL;
		}
	}

	if (run_all || (cfg.impl != NULL && strcmp(cfg.impl, "hamerly") == 0))
	{
		KMeansResult *r =
				run_bench(&cfg, bench_data, vec_type, KMEANS_ALGO_HAMERLY);
		if (cfg.verify && r != NULL)
			results[nresults++] = (typeof(results[0])){r, "hamerly"};
		else
			mkt_kmeans_result_destroy(r);
	}

	if (run_all || (cfg.impl != NULL && strcmp(cfg.impl, "elkan") == 0))
	{
		KMeansResult *r =
				run_bench(&cfg, bench_data, vec_type, KMEANS_ALGO_ELKAN);
		if (cfg.verify && r != NULL)
			results[nresults++] = (typeof(results[0])){r, "elkan"};
		else
			mkt_kmeans_result_destroy(r);
	}

	if (run_all || (cfg.impl != NULL && strcmp(cfg.impl, "pgvector") == 0))
	{
		KMeansResult *r = run_bench_pgvector(&cfg, data);
		if (cfg.verify && r != NULL)
			results[nresults++] = (typeof(results[0])){r, "elkan(pgvec)"};
		else
			mkt_kmeans_result_destroy(r);
	}

	/* Verification: compare each result against Lloyd's */
	if (cfg.verify && ref != NULL)
	{
		printf("\nVerification (reference: lloyd, exact_cost=%.2f):\n",
			   ref_cost);

		for (int i = 0; i < nresults; i++)
		{
			double cost = recompute_exact_cost(
					data, results[i].res, cfg.nvecs, cfg.dim, cfg.metric);
			verify_against_ref(
					ref,
					ref_cost,
					results[i].res,
					cost,
					results[i].name,
					cfg.nvecs);
			mkt_kmeans_result_destroy(results[i].res);
		}

		mkt_kmeans_result_destroy(ref);
	}

	mkt_free(data_f16);
	mkt_free(data);
	return 0;
}
