/*
 * vectorsearch bench search
 *
 * Benchmark index build and query through the bindings API — the
 * same code path used by Python ctypes and ann-benchmarks.
 *
 * Supports both HDF5 datasets and synthetic random vectors.
 */

#include "vs_config.h"

#include <getopt.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifdef VS_HAVE_HDF5
#include <hdf5.h>

#include "standalone/hdf5_source.h"
#endif

#include "cmd.h"
#include "index/query_scan.h"
#include "standalone/api.h"
#include "standalone/vec32_source.h"

/* ----------------------------------------------------------------
 * Defaults
 * ---------------------------------------------------------------- */

#define DEFAULT_DIM		768
#define DEFAULT_NLIST	0 /* auto */
#define DEFAULT_NPROBE	0 /* 0 = auto from nlist */
#define DEFAULT_K		10
#define DEFAULT_QUERIES 100
#define DEFAULT_RUNS	5
#define DEFAULT_WARMUP	100

/* ----------------------------------------------------------------
 * Configuration
 * ---------------------------------------------------------------- */

typedef struct
{
	uint32_t	dim;
	uint32_t	nlist;
	uint32_t	fan_out;
	uint32_t	nprobe;
	uint32_t	k;
	uint32_t	queries;
	uint32_t	runs;
	uint32_t	warmup;
	uint32_t	nredo;
	uint32_t	km_iter;
	double		soar_lambda;
	double		boundary_epsilon;
	const char *hdf5_path;
	const char *metric;
	const char *centroid_fmt;
	const char *posting_fmt;
	const char *posting_layout;
	const char *distance_mode;
	bool		no_rerank;
	int			fastscan; /* 0=off, 8=uint8 LUT, 16=uint16 LUT (hacc) */
	int32_t		nworkers; /* -1=auto, 0=serial */
	bool		wait_profile;
	bool		help;
} BenchConfig;

/* ----------------------------------------------------------------
 * Timing
 * ---------------------------------------------------------------- */

static inline uint64_t
get_time_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* ----------------------------------------------------------------
 * HDF5 loading
 * ---------------------------------------------------------------- */

#ifdef VS_HAVE_HDF5
static float *
load_hdf5_float(
		const char *path, const char *dataset, hsize_t *rows, hsize_t *cols)
{
	hid_t file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0)
	{
		fprintf(stderr, "Error: cannot open HDF5 '%s'\n", path);
		return NULL;
	}
	hid_t dset = H5Dopen2(file, dataset, H5P_DEFAULT);
	if (dset < 0)
	{
		H5Fclose(file);
		return NULL;
	}
	hid_t	space = H5Dget_space(dset);
	hsize_t dims[2];
	H5Sget_simple_extent_dims(space, dims, NULL);
	*rows		= dims[0];
	*cols		= dims[1];
	float *data = malloc(*rows * *cols * sizeof(float));
	H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
	H5Sclose(space);
	H5Dclose(dset);
	H5Fclose(file);
	return data;
}

static int64_t *
load_hdf5_int64(
		const char *path, const char *dataset, hsize_t *rows, hsize_t *cols)
{
	H5Eset_auto(H5E_DEFAULT, NULL, NULL);
	hid_t file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0)
		return NULL;
	hid_t dset = H5Dopen2(file, dataset, H5P_DEFAULT);
	if (dset < 0)
	{
		H5Fclose(file);
		H5Eset_auto(H5E_DEFAULT, (H5E_auto_t)H5Eprint, stderr);
		return NULL;
	}
	hid_t	space = H5Dget_space(dset);
	hsize_t dims[2];
	H5Sget_simple_extent_dims(space, dims, NULL);
	*rows		  = dims[0];
	*cols		  = dims[1];
	int64_t *data = malloc(*rows * *cols * sizeof(int64_t));
	H5Dread(dset, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
	H5Sclose(space);
	H5Dclose(dset);
	H5Fclose(file);
	H5Eset_auto(H5E_DEFAULT, (H5E_auto_t)H5Eprint, stderr);
	return data;
}
#endif /* VS_HAVE_HDF5 */

/* ----------------------------------------------------------------
 * Synthetic vector generation
 * ---------------------------------------------------------------- */

static float
rand_normal(void)
{
	float u1 = ((float)(rand() % 10000) + 1.0f) / 10001.0f;
	float u2 = ((float)(rand() % 10000)) / 10000.0f;
	return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

static float *
generate_random_vectors(uint32_t nvecs, uint32_t dim)
{
	float *data = malloc((size_t)nvecs * dim * sizeof(float));
	for (uint32_t i = 0; i < nvecs; i++)
		for (uint32_t d = 0; d < dim; d++)
			data[(size_t)i * dim + d] = rand_normal();
	return data;
}

/* ----------------------------------------------------------------
 * Usage
 * ---------------------------------------------------------------- */

static void
print_usage(CmdContext *ctx)
{
	CMD_USAGE_HEADER(ctx, "bench search");
	printf("Benchmark centroid routing and cluster scan.\n\n");
	printf("Options:\n");
	printf("  --dim <int>        Vector dimension (default: %d)\n",
		   DEFAULT_DIM);
	printf("  --nlist <int>      Clusters (0=auto)\n");
	printf("  --fan-out <int>    Tree branching factor (0=auto)\n");
	printf("  --nprobe <int>     Probes (0 = auto from nlist; default: %d)\n",
		   DEFAULT_NPROBE);
	printf("  -k <int>           Top-k (default: %d)\n", DEFAULT_K);
	printf("  --queries <int>    Query count (default: %d)\n",
		   DEFAULT_QUERIES);
	printf("  --runs <int>       Runs per query (default: %d)\n",
		   DEFAULT_RUNS);
	printf("  --centroid-fmt <s> float32, float16, rabitq "
		   "(default: float32)\n");
	printf("  --posting-fmt <s>  native, rabitq, int8 "
		   "(default: rabitq)\n");
	printf("  --posting-layout   pages, flat (default: pages)\n");
	printf("  --mode <str>       asymmetric, symmetric\n");
	printf("  --warmup <int>     Warmup queries (default: %d)\n",
		   DEFAULT_WARMUP);
	printf("  --nredo <int>      K-means restarts\n");
	printf("  --km-iter <int>    K-means iterations\n");
	printf("  --boundary-epsilon <float>  Boundary replication "
		   "threshold (0=off)\n");
	printf("  --soar-lambda <float>  SOAR replication lambda "
		   "(0=off)\n");
	printf("  --no-rerank        Skip reranking (return approximate)\n");
	printf("  --fastscan <8|16>  Use VPSHUFB fastscan (8=fast, "
		   "16=accurate)\n");
	printf("  --workers <int>    Build parallelism "
		   "(-1=auto, 0=serial)\n");
	printf("  --wait-profile     Pause before queries (print PID for perf "
		   "attach)\n");
#ifdef VS_HAVE_HDF5
	printf("  --hdf5 <path>      HDF5 dataset\n");
	printf("  --metric <str>     angular, euclidean\n");
#endif
	printf("  --help\n");
}

/* ----------------------------------------------------------------
 * Entry point
 * ---------------------------------------------------------------- */

int
cmd_bench_search(CmdContext *ctx)
{
	BenchConfig config = {
			.dim			= DEFAULT_DIM,
			.nprobe			= DEFAULT_NPROBE,
			.k				= DEFAULT_K,
			.queries		= DEFAULT_QUERIES,
			.runs			= DEFAULT_RUNS,
			.warmup			= DEFAULT_WARMUP,
			.metric			= "euclidean",
			.centroid_fmt	= "float32",
			.posting_fmt	= "rabitq",
			.posting_layout = "pages",
			.distance_mode	= "asymmetric",
			.nworkers		= -1,
	};

	static struct option long_options[] = {
			{"dim", required_argument, 0, 'd'},
			{"nlist", required_argument, 0, 'n'},
			{"fan-out", required_argument, 0, 'f'},
			{"nprobe", required_argument, 0, 'p'},
			{"queries", required_argument, 0, 'q'},
			{"runs", required_argument, 0, 'r'},
			{"centroid-fmt", required_argument, 0, 'F'},
			{"fmt", required_argument, 0, 'F'}, /* backward compat */
			{"mode", required_argument, 0, 'M'},
			{"warmup", required_argument, 0, 'W'},
			{"nredo", required_argument, 0, 'R'},
			{"km-iter", required_argument, 0, 'I'},
			{"hdf5", required_argument, 0, 'H'},
			{"metric", required_argument, 0, 'm'},
			{"posting-fmt", required_argument, 0, 'T'},
			{"posting-layout", required_argument, 0, 'P'},
			{"no-rerank", no_argument, 0, 'N'},
			{"fastscan", required_argument, 0, 'X'},
			{"wait-profile", no_argument, 0, 'Z'},
			{"boundary-epsilon", required_argument, 0, 'B'},
			{"soar-lambda", required_argument, 0, 'S'},
			{"workers", required_argument, 0, 'w'},
			{"help", no_argument, 0, 'h'},
			{0, 0, 0, 0},
	};

	optind = 1;
	int opt;
	while ((opt = getopt_long(
					ctx->argc,
					ctx->argv,
					"d:n:f:p:q:r:k:F:M:R:I:H:m:h",
					long_options,
					NULL)) != -1)
	{
		switch (opt)
		{
		case 'd':
			config.dim = (uint32_t)atoi(optarg);
			break;
		case 'n':
			config.nlist = (uint32_t)atoi(optarg);
			break;
		case 'f':
			config.fan_out = (uint32_t)atoi(optarg);
			break;
		case 'p':
			config.nprobe = (uint32_t)atoi(optarg);
			break;
		case 'q':
			config.queries = (uint32_t)atoi(optarg);
			break;
		case 'r':
			config.runs = (uint32_t)atoi(optarg);
			break;
		case 'k':
			config.k = (uint32_t)atoi(optarg);
			break;
		case 'F':
			config.centroid_fmt = optarg;
			break;
		case 'M':
			config.distance_mode = optarg;
			break;
		case 'T':
			config.posting_fmt = optarg;
			break;
		case 'P':
			config.posting_layout = optarg;
			break;
		case 'W':
			config.warmup = (uint32_t)atoi(optarg);
			break;
		case 'R':
			config.nredo = (uint32_t)atoi(optarg);
			break;
		case 'I':
			config.km_iter = (uint32_t)atoi(optarg);
			break;
		case 'H':
			config.hdf5_path = optarg;
			break;
		case 'm':
			config.metric = optarg;
			break;
		case 'N':
			config.no_rerank = true;
			break;
		case 'X':
			config.fastscan = atoi(optarg);
			if (config.fastscan != 8 && config.fastscan != 16)
				config.fastscan = 16;
			break;
		case 'Z':
			config.wait_profile = true;
			break;
		case 'S':
			config.soar_lambda = atof(optarg);
			break;
		case 'B':
			config.boundary_epsilon = atof(optarg);
			break;
		case 'w':
			config.nworkers = (int32_t)atoi(optarg);
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

	/* --------------------------------------------------------
	 * Load data and build index
	 * -------------------------------------------------------- */

	float	*query_vecs	  = NULL;
	int64_t *gt_neighbors = NULL;
	uint32_t nqueries	  = 0;
	uint32_t gt_k		  = 0;

	PrismBuildInfo info;
	VsHandle	  *handle = NULL;

#ifdef VS_HAVE_HDF5
	if (config.hdf5_path != NULL)
	{
		/* Open train dataset as streaming source */
		VsHdf5Source train_src;
		if (vs_hdf5_source_open(&train_src, config.hdf5_path, "train") != 0)
		{
			fprintf(stderr, "Error: cannot open HDF5 train dataset\n");
			return 1;
		}

		config.dim = train_src.base.dim;

		/* Load queries into memory (small, need random access) */
		hsize_t n_test, dim_test;
		query_vecs =
				load_hdf5_float(config.hdf5_path, "test", &n_test, &dim_test);
		if (query_vecs == NULL)
		{
			vs_hdf5_source_close(&train_src);
			return 1;
		}
		nqueries = (uint32_t)n_test;

		/* Load ground truth */
		hsize_t n_gt, k_gt_h;
		gt_neighbors =
				load_hdf5_int64(config.hdf5_path, "neighbors", &n_gt, &k_gt_h);
		if (gt_neighbors != NULL)
			gt_k = (uint32_t)k_gt_h;

		/* Detect metric from filename if not specified */
		if (strcmp(config.metric, "euclidean") == 0 &&
			(strstr(config.hdf5_path, "angular") != NULL ||
			 strstr(config.hdf5_path, "cosine") != NULL))
		{
			config.metric = "angular";
		}

		if (config.queries > nqueries)
			config.queries = nqueries;

		printf("Dataset: %s\n", config.hdf5_path);
		printf("  %u x %u (%s)\n",
			   train_src.base.nvecs,
			   config.dim,
			   config.metric);

		/* Build index by streaming from HDF5 */
		printf("Building index (nlist=%u, centroid=%s, "
			   "posting=%s, layout=%s, workers=%d)...\n",
			   config.nlist,
			   config.centroid_fmt,
			   config.posting_fmt,
			   config.posting_layout,
			   config.nworkers);

		uint64_t t0 = get_time_ns();
		handle		= vs_handle_create(
				 &train_src.base,
				 config.nlist,
				 config.fan_out,
				 config.metric,
				 config.centroid_fmt,
				 config.posting_layout,
				 config.nredo,
				 config.km_iter,
				 config.soar_lambda,
				 config.boundary_epsilon,
				 config.fastscan,
				 config.nworkers,
				 &info);
		double build_ms = (double)(get_time_ns() - t0) / 1e6;

		vs_hdf5_source_close(&train_src);

		if (handle == NULL)
		{
			fprintf(stderr, "Error: index build failed\n");
			free(query_vecs);
			free(gt_neighbors);
			return 1;
		}

		printf("  Built in %.1fs: %u clusters, %u levels\n",
			   build_ms / 1000.0,
			   info.nlist,
			   info.nlevels);
		prism_build_stats_print(&info.stats);
	}
	else
#endif
	{
		/* Synthetic vectors */
		srand(42);
		uint32_t nvecs	  = 10000;
		nqueries		  = config.queries;
		float *train_vecs = generate_random_vectors(nvecs, config.dim);
		query_vecs		  = generate_random_vectors(nqueries, config.dim);
		printf("Synthetic: %u x %u (L2)\n", nvecs, config.dim);

		printf("Building index (nlist=%u, centroid=%s, "
			   "posting=%s, layout=%s)...\n",
			   config.nlist,
			   config.centroid_fmt,
			   config.posting_fmt,
			   config.posting_layout);

		uint64_t t0 = get_time_ns();
		handle		= vs_handle_create_from_array(
				 train_vecs,
				 nvecs,
				 config.dim,
				 config.nlist,
				 config.fan_out,
				 config.metric,
				 config.centroid_fmt,
				 config.posting_layout,
				 config.nredo,
				 config.km_iter,
				 config.soar_lambda,
				 config.boundary_epsilon,
				 config.fastscan,
				 config.nworkers,
				 &info);
		double build_ms = (double)(get_time_ns() - t0) / 1e6;
		free(train_vecs);

		if (handle == NULL)
		{
			fprintf(stderr, "Error: index build failed\n");
			free(query_vecs);
			return 1;
		}

		printf("  Built in %.1fs: %u clusters, %u levels\n",
			   build_ms / 1000.0,
			   info.nlist,
			   info.nlevels);
		prism_build_stats_print(&info.stats);
	}

	uint32_t avg = info.nvecs / info.nlist;
	printf("  Clusters: avg=%u, min=%u, max=%u\n",
		   avg,
		   info.min_cluster,
		   info.max_cluster);
	if (info.max_cluster > avg * 10)
		printf("  WARNING: cluster imbalance (max/avg=%.0fx)\n",
			   (double)info.max_cluster / avg);

	/* --nprobe 0: derive from the built cluster count, same rule as the
	 * extension's prism.nprobe = 0. */
	if (config.nprobe == 0)
		config.nprobe = prism_auto_nprobe(info.nlist);

	/* --------------------------------------------------------
	 * Run queries via bindings API
	 * -------------------------------------------------------- */

	if (config.queries > 0 && config.queries < nqueries)
		nqueries = config.queries;

	uint32_t *result_ids = malloc(config.k * sizeof(uint32_t));
	double	  recall_sum = 0.0;
	double	  lat_sum	 = 0.0;
	double	  lat_min	 = 1e30;
	double	  lat_max	 = 0.0;

	printf("Querying: %u queries x %u runs, k=%u, nprobe=%u, "
		   "centroid=%s, posting=%s, layout=%s, mode=%s\n",
		   nqueries,
		   config.runs,
		   config.k,
		   config.nprobe,
		   config.centroid_fmt,
		   config.posting_fmt,
		   config.posting_layout,
		   config.distance_mode);

	if (config.wait_profile)
	{
		/* Write PID to a signal file so a profiler can attach,
		 * then wait for the file to be removed as the trigger. */
		const char *tmpdir = getenv("TMPDIR");
		char		sig_path[256];
		snprintf(
				sig_path,
				sizeof(sig_path),
				"%s/vectorsearch-bench-ready",
				tmpdir ? tmpdir : "/tmp");
		const char *sig_file = sig_path;
		FILE	   *f		 = fopen(sig_file, "w");
		if (f)
		{
			fprintf(f, "%d\n", (int)getpid());
			fclose(f);
		}
		printf("PID %d ready. Waiting for %s to be removed...\n",
			   (int)getpid(),
			   sig_file);
		fflush(stdout);
		while (access(sig_file, F_OK) == 0)
			usleep(10000); /* 10ms poll */
		printf("Starting queries.\n");
		fflush(stdout);
	}

	/* Warmup */
	for (uint32_t w = 0; w < config.warmup; w++)
	{
		uint32_t qi = w % nqueries;
		vs_handle_query(
				handle,
				query_vecs + (size_t)qi * config.dim,
				config.k,
				config.nprobe,
				config.distance_mode,
				!config.no_rerank,
				result_ids);
	}

	/* Timed runs */
	uint32_t total_queries = nqueries * config.runs;
	for (uint32_t r = 0; r < config.runs; r++)
	{
		for (uint32_t q = 0; q < nqueries; q++)
		{
			const float *qvec = query_vecs + (size_t)q * config.dim;

			uint64_t t_start = get_time_ns();
			uint32_t count	 = vs_handle_query(
					  handle,
					  qvec,
					  config.k,
					  config.nprobe,
					  config.distance_mode,
					  !config.no_rerank,
					  result_ids);
			uint64_t t_end = get_time_ns();

			double lat_us = (double)(t_end - t_start) / 1000.0;
			lat_sum += lat_us;
			if (lat_us < lat_min)
				lat_min = lat_us;
			if (lat_us > lat_max)
				lat_max = lat_us;

			/* Recall (only on first run) */
			if (r == 0 && gt_neighbors != NULL && count > 0)
			{
				uint32_t hits  = 0;
				uint32_t res_k = count < config.k ? count : config.k;
				uint32_t g_k   = config.k < gt_k ? config.k : gt_k;
				for (uint32_t i = 0; i < res_k; i++)
				{
					for (uint32_t g = 0; g < g_k; g++)
					{
						if (result_ids[i] ==
							(uint32_t)gt_neighbors[q * gt_k + g])
						{
							hits++;
							break;
						}
					}
				}
				recall_sum += (double)hits / g_k;
			}
		}
	}

	/* --------------------------------------------------------
	 * Report
	 * -------------------------------------------------------- */

	double avg_lat = lat_sum / total_queries;
	double qps	   = 1e6 / avg_lat;
	double recall  = nqueries > 0 ? recall_sum / nqueries : 0.0;

	printf("\nResults:\n");
	printf("  centroid=%-8s posting=%-8s layout=%-6s mode=%-12s "
		   "fastscan=%-5s nprobe=%u k=%u\n",
		   config.centroid_fmt,
		   config.posting_fmt,
		   config.posting_layout,
		   config.distance_mode,
		   config.fastscan ? (config.fastscan == 8 ? "8bit" : "16bit") : "off",
		   config.nprobe,
		   config.k);
	if (gt_neighbors != NULL)
		printf("  Recall@%u:   %.4f\n", config.k, recall);
	printf("  Avg latency: %.2f us\n", avg_lat);
	printf("  Min latency: %.2f us\n", lat_min);
	printf("  Max latency: %.2f us\n", lat_max);
	printf("  QPS:         %.0f\n", qps);

	/* Cleanup */
	free(result_ids);
	vs_handle_destroy(handle);
	free(query_vecs);
	free(gt_neighbors);

	return 0;
}
