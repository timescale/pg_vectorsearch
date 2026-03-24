/*
 * mkt bench search
 *
 * Benchmark beam search over a synthetic centroid tree. Generates
 * random vectors, runs hierarchical k-means to build a realistic
 * cluster structure, and compares centroid page formats (RaBitQ,
 * float32, float16) for latency, recall, and storage efficiency.
 *
 * Uses mkt_centroid_beam_search — the same code path as the
 * production index — ensuring benchmark results reflect real
 * search performance.
 *
 * Useful for:
 * - Comparing page formats to choose production defaults
 * - Measuring recall with realistic cluster structure
 * - Validating constant memory usage under valgrind/massif
 * - Tuning beam_width and nprobe parameters
 */

#include "mkt_config.h"

#include <float.h>
#include <getopt.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef MKT_HAVE_HDF5
#include <hdf5.h>
#endif

#include "algo/hkmeans.h"
#include "algo/kmeans.h"
#include "algo/topk.h"
#include "algo/vecops.h"
#include "cmd.h"
#include "core/memory.h"
#include "index/centroid_build.h"
#include "index/centroid_page.h"
#include "index/centroid_search.h"
#include "index/index_build.h"
#include "mkt_halfvec.h"
#include "mkt_types.h"
#include "quant/rabitq.h"

/* Default parameters */
#define DEFAULT_DIM		   768
#define DEFAULT_NLEVELS	   2
#define DEFAULT_FAN_OUT	   32
#define DEFAULT_BEAM_WIDTH 4
#define DEFAULT_NPROBE	   10
#define DEFAULT_QUERIES	   100
#define DEFAULT_RUNS	   10
#define WARMUP_RUNS		   3

/* ----------------------------------------------------------------
 * Benchmark configuration
 * ---------------------------------------------------------------- */

typedef struct
{
	Dimension	dim;
	uint32_t	nlevels;
	uint32_t	fan_out;
	uint32_t	beam_width;
	uint32_t	nprobe;
	uint32_t	queries;
	uint32_t	runs;
	uint32_t	nlist;		/* 0 = auto (HDF5 mode only) */
	const char *hdf5_path;	/* NULL = synthetic mode */
	const char *metric_str; /* NULL = auto-detect from HDF5 */
	bool		help;
} BenchConfig;

/* ----------------------------------------------------------------
 * Timing helpers (same pattern as bench_page_score.c)
 * ---------------------------------------------------------------- */

static inline uint64_t
get_time_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

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

/* ----------------------------------------------------------------
 * Test storage: flat page array (same pattern as test_centroid_search.c)
 * ---------------------------------------------------------------- */

typedef struct TestStorage
{
	MktStorage base; /* must be first */
	char	  *pages;
	uint32_t   next_blkno;	/* for write path (new_page counter) */
	float	  *rerank_vecs; /* [total_slots * dim] flat array */
	uint32_t   max_entries; /* entries per page (for TID decode) */
	Dimension  dim;

	/* Preallocated rerank scratch buffers (avoid per-call allocs) */
	void	 *rerank_lb;	/* LBEntry array */
	Distance *rerank_exact; /* exact distance array */
	uint32_t  rerank_cap;	/* allocated capacity */
} TestStorage;

static Page
test_read_page(MktStorage *self, BlockNumber blkno)
{
	TestStorage *ts = (TestStorage *)self;
	return ts->pages + (size_t)blkno * BLCKSZ;
}

static void
test_release_page(MktStorage *self, BlockNumber blkno)
{
	(void)self;
	(void)blkno;
}

static Page
test_new_page(MktStorage *self, BlockNumber *blkno_out)
{
	TestStorage *ts = (TestStorage *)self;
	*blkno_out		= ts->next_blkno++;
	return ts->pages + (size_t)*blkno_out * BLCKSZ;
}

static Page
test_write_page(MktStorage *self, BlockNumber blkno)
{
	TestStorage *ts = (TestStorage *)self;
	return ts->pages + (size_t)blkno * BLCKSZ;
}

static void
test_commit_page(MktStorage *self, BlockNumber blkno)
{
	(void)self;
	(void)blkno;
}

/*
 * bench_rerank - Rerank candidates using stored medoid vectors.
 *
 * Decodes TID -> flat index (blkno * max_entries + offset - 1),
 * looks up medoid vector, computes exact L2 distance. Uses
 * lower-bound ordering with threshold pruning.
 *
 * Uses preallocated scratch buffers in TestStorage to avoid
 * per-call allocations in the hot path.
 */

typedef struct
{
	Distance lower_bound;
	uint32_t idx;
} LBEntry;

static uint32_t
bench_rerank(
		MktStorage			  *self,
		Datum				   query,
		Dimension			   dim,
		const ItemPointerData *tids,
		const Distance		  *distances,
		const Distance		  *errors,
		uint32_t			   count,
		uint32_t			   keep,
		uint32_t			  *out_indices,
		Distance			  *out_distances)
{
	TestStorage *ts		 = (TestStorage *)self;
	const float *query_f = (const float *)DatumGetPointer(query);

	if (ts->rerank_vecs == NULL || query_f == NULL || count == 0)
		return 0;

	/*
	 * Grow scratch buffers if needed. Use malloc/free (not
	 * mkt_alloc/mkt_free) because mkt_alloc goes through the
	 * memory context system — beam search destroys its context
	 * after each call, which would free these buffers while
	 * TestStorage still holds pointers to them.
	 */
	if (count > ts->rerank_cap)
	{
		free(ts->rerank_lb);
		free(ts->rerank_exact);
		ts->rerank_lb	 = malloc(count * sizeof(LBEntry));
		ts->rerank_exact = malloc(count * sizeof(Distance));
		ts->rerank_cap	 = count;
	}

	LBEntry	 *lb	= ts->rerank_lb;
	Distance *exact = ts->rerank_exact;

	for (uint32_t i = 0; i < count; i++)
	{
		lb[i].lower_bound = distances[i] - errors[i];
		lb[i].idx		  = i;
	}

	/* Sort by lower_bound ascending (insertion sort, small N) */
	for (uint32_t i = 1; i < count; i++)
	{
		LBEntry	 tmp = lb[i];
		uint32_t j	 = i;
		while (j > 0 && lb[j - 1].lower_bound > tmp.lower_bound)
		{
			lb[j] = lb[j - 1];
			j--;
		}
		lb[j] = tmp;
	}

	/* Iterate in lower_bound order with threshold pruning */
	Distance threshold = FLT_MAX;
	uint32_t nresults  = 0;

	for (uint32_t i = 0; i < count; i++)
	{
		uint32_t idx = lb[i].idx;

		if (nresults >= keep && lb[i].lower_bound >= threshold)
			break;

		Distance d;
		if (errors[idx] == 0.0f)
		{
			d = distances[idx];
		}
		else
		{
			BlockNumber	 blk  = ItemPointerGetBlockNumber(&tids[idx]);
			OffsetNumber off  = ItemPointerGetOffsetNumber(&tids[idx]);
			uint32_t	 midx = blk * ts->max_entries + (uint32_t)(off - 1);
			const float *vec  = ts->rerank_vecs + (size_t)midx * dim;
			d				  = mkt_l2_distance_squared(query_f, vec, dim);
		}
		exact[idx] = d;

		/* Insert maintaining sorted order */
		uint32_t pos = nresults;
		while (pos > 0 && exact[out_indices[pos - 1]] > d)
		{
			if (pos < keep)
			{
				out_indices[pos]   = out_indices[pos - 1];
				out_distances[pos] = out_distances[pos - 1];
			}
			pos--;
		}
		if (pos < keep)
		{
			out_indices[pos]   = idx;
			out_distances[pos] = d;
			if (nresults < keep)
				nresults++;
			threshold = out_distances[nresults - 1];
		}
	}

	return nresults;
}

static const MktStorageOps test_storage_ops = {
		.read_page	  = test_read_page,
		.release_page = test_release_page,
		.write_page	  = test_write_page,
		.new_page	  = test_new_page,
		.commit_page  = test_commit_page,
		.rerank		  = bench_rerank,
};

/* ----------------------------------------------------------------
 * Vector generation
 * ---------------------------------------------------------------- */

/* Box-Muller: two uniform randoms -> one standard normal */
static float
rand_normal(void)
{
	float u1 = ((float)(rand() % 10000) + 1.0f) / 10001.0f;
	float u2 = ((float)(rand() % 10000)) / 10000.0f;
	return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

static void
generate_random_vector(float *data, Dimension dim)
{
	for (Dimension i = 0; i < dim; i++)
		data[i] = rand_normal();
}

/* ----------------------------------------------------------------
 * Clustered tree construction
 *
 * 1. Generate a random vector pool
 * 2. Run mkt_hkmeans_f32() to build hierarchical tree
 * 3. Build centroid pages from the tree structure
 *
 * Block numbers assigned sequentially per BFS node:
 * each node gets ceil(nchildren / max_entries) pages.
 * ---------------------------------------------------------------- */

/* Vectors per leaf centroid for k-means training */
#define VECS_PER_LEAF 20
#define DEFAULT_TOPK  10

static uint32_t
power_u32(uint32_t base, uint32_t exp)
{
	uint32_t result = 1;
	for (uint32_t i = 0; i < exp; i++)
		result *= base;
	return result;
}

/*
 * Generate a hierarchically-clustered vector pool.
 *
 * Builds a tree of centers: at each level, fan_out child centers
 * are generated around each parent center with decreasing noise.
 * Data vectors are sampled around the leaf centers.
 *
 * This creates spatial structure that matches the tree topology:
 * medoids in the same subtree are genuinely closer together than
 * medoids in different subtrees, giving meaningful recall numbers.
 *
 * To avoid the curse of dimensionality (where independent random
 * vectors concentrate at equal distance in high dims), cluster
 * center offsets are generated in a low-dimensional subspace
 * (min(dim, 32) active dimensions). This simulates real-world
 * embedding data where intrinsic dimensionality is much lower
 * than the vector dimension, creating natural cluster overlap
 * that makes beam search routing non-trivial.
 *
 * Scale hierarchy (each level 2x tighter):
 *   Level 0: centers in subspace (inter-center ~sqrt(2*subdim))
 *   Level l: parent + scale * noise  (scale *= 0.5 per level)
 *   Data:    leaf + data_scale * noise (full dim)
 */
static float *
generate_hierarchical_vectors(
		Dimension dim,
		uint32_t  fan_out,
		uint32_t  nlevels,
		uint32_t  vecs_per_leaf,
		uint32_t *nvecs_out)
{
	uint32_t leaf_count = power_u32(fan_out, nlevels);
	uint32_t nvecs		= leaf_count * vecs_per_leaf;
	float	*vectors	= mkt_alloc((size_t)nvecs * dim * sizeof(float));

	/*
	 * Subspace dimension for cluster centers. Real embeddings
	 * have intrinsic dim << vector dim. Using a small subspace
	 * makes inter-cluster distances comparable to within-cluster
	 * distances, so beam_width genuinely affects recall.
	 */
	Dimension subdim = dim < 32 ? dim : 32;

	/*
	 * Build leaf centers by iteratively expanding the tree.
	 * Start with 1 virtual root, expand fan_out children per
	 * parent at each level. Only keep current + next level.
	 */
	uint32_t curr_count	  = 1;
	float	*curr_centers = mkt_alloc(dim * sizeof(float));
	memset(curr_centers, 0, dim * sizeof(float));

	float scale = 1.0f;
	for (uint32_t l = 0; l < nlevels; l++)
	{
		uint32_t next_count = curr_count * fan_out;
		float	*next = mkt_alloc((size_t)next_count * dim * sizeof(float));

		for (uint32_t p = 0; p < curr_count; p++)
		{
			const float *pc = curr_centers + (size_t)p * dim;
			for (uint32_t c = 0; c < fan_out; c++)
			{
				float *nc = next + (size_t)(p * fan_out + c) * dim;
				/* Copy parent, then perturb in subspace */
				memcpy(nc, pc, dim * sizeof(float));
				for (Dimension d = 0; d < subdim; d++)
					nc[d] += scale * rand_normal();
			}
		}

		mkt_free(curr_centers);
		curr_centers = next;
		curr_count	 = next_count;
		scale *= 0.5f;
	}

	/* Generate data vectors around each leaf center
	 * (full-dimensional noise for realistic data spread) */
	float data_scale = scale;
	for (uint32_t i = 0; i < leaf_count; i++)
	{
		const float *center = curr_centers + (size_t)i * dim;
		for (uint32_t v = 0; v < vecs_per_leaf; v++)
		{
			float *vec = vectors + (size_t)(i * vecs_per_leaf + v) * dim;
			for (Dimension d = 0; d < dim; d++)
				vec[d] = center[d] + data_scale * rand_normal();
		}
	}

	mkt_free(curr_centers);
	*nvecs_out = nvecs;
	return vectors;
}

#ifdef MKT_HAVE_HDF5
static float *
load_hdf5_float(
		const char *path, const char *dataset, hsize_t *rows, hsize_t *cols)
{
	hid_t file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0)
	{
		fprintf(stderr, "Error: cannot open HDF5 file '%s'\n", path);
		return NULL;
	}

	hid_t dset = H5Dopen2(file, dataset, H5P_DEFAULT);
	if (dset < 0)
	{
		fprintf(stderr,
				"Error: dataset '%s' not found in '%s'\n",
				dataset,
				path);
		H5Fclose(file);
		return NULL;
	}

	hid_t	space = H5Dget_space(dset);
	hsize_t dims[2];
	H5Sget_simple_extent_dims(space, dims, NULL);
	*rows = dims[0];
	*cols = dims[1];

	float *data = malloc(*rows * *cols * sizeof(float));
	if (data == NULL)
	{
		H5Sclose(space);
		H5Dclose(dset);
		H5Fclose(file);
		return NULL;
	}

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
	hid_t file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0)
		return NULL;

	hid_t dset = H5Dopen2(file, dataset, H5P_DEFAULT);
	if (dset < 0)
	{
		H5Fclose(file);
		return NULL;
	}

	hid_t	space = H5Dget_space(dset);
	hsize_t dims[2];
	H5Sget_simple_extent_dims(space, dims, NULL);
	*rows = dims[0];
	*cols = dims[1];

	int64_t *data = malloc(*rows * *cols * sizeof(int64_t));
	if (data == NULL)
	{
		H5Sclose(space);
		H5Dclose(dset);
		H5Fclose(file);
		return NULL;
	}

	H5Dread(dset, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
	H5Sclose(space);
	H5Dclose(dset);
	H5Fclose(file);
	return data;
}

static void
normalize_vectors(float *data, uint32_t nvecs, Dimension dim)
{
	for (uint32_t i = 0; i < nvecs; i++)
	{
		float *v	= data + (size_t)i * dim;
		float  norm = mkt_l2_norm(v, dim);
		if (norm > 0.0f)
			mkt_vector_scale(v, 1.0f / norm, v, dim);
	}
}
#endif /* MKT_HAVE_HDF5 */

/* ----------------------------------------------------------------
 * build_tree - Build centroid tree via hierarchical k-means
 *
 * Clusters vectors (synthetic or external) hierarchically and
 * builds centroid pages in the specified format. All RaBitQ
 * entries are encoded relative to the global centroid.
 *
 * When ext_vectors is NULL, generates random vectors internally.
 * When ext_vectors is provided, uses them directly (caller owns).
 *
 * Returns the allocated page buffer, or NULL on failure.
 * ---------------------------------------------------------------- */
static char *
build_tree(
		const BenchConfig  *config,
		const RaBitQParams *params,
		uint8_t				fmt,
		const float		   *ext_vectors,
		uint32_t			ext_nvecs,
		DistanceMetric		metric,
		uint32_t		   *total_pages_out,
		uint32_t		   *total_centroids_out,
		float			   *global_centroid_out,
		float			  **query_seeds_out,
		uint32_t		   *nseeds_out,
		float			  **rerank_vecs_out,
		float			  **vectors_out,
		uint32_t		  **leaf_idx_out,
		uint32_t		  **leaf_offsets_out,
		uint32_t		   *nvecs_out,
		bool				verbose)
{
	Dimension dim	  = config->dim;
	uint32_t  fan_out = config->fan_out;

	uint32_t max_entries = mkt_centroid_max_entries_fmt(dim, fmt);

	/*
	 * Get vectors: external dataset or synthetic generation.
	 */
	uint32_t nvecs;
	float	*vectors;
	uint32_t nlist;

	if (ext_vectors != NULL)
	{
		nvecs = ext_nvecs;

		nlist = config->nlist;
		if (nlist == 0)
		{
			nlist = (uint32_t)sqrt((double)nvecs);
			if (nlist < 1)
				nlist = 1;
			if (nlist > 10000)
				nlist = 10000;
		}

		/* Sample for clustering (stride sampling).
		 * Use all vectors when nlist is explicitly set and larger
		 * datasets need full coverage. */
		uint32_t max_samples = nvecs < 256000 ? nvecs : 256000;
		vectors = mkt_alloc((size_t)max_samples * dim * sizeof(float));
		if (max_samples == nvecs)
		{
			memcpy(vectors, ext_vectors, (size_t)nvecs * dim * sizeof(float));
		}
		else
		{
			uint32_t stride = nvecs / max_samples;
			for (uint32_t i = 0; i < max_samples; i++)
				memcpy(vectors + (size_t)i * dim,
					   ext_vectors + (size_t)(i * stride) * dim,
					   dim * sizeof(float));
		}
		/* k-means runs on the sample; nvecs is reset to sample
		 * count for clustering only. The full dataset is used
		 * for vector assignment later via ext_vectors. */
		nvecs = max_samples;

		if (verbose)
			printf("  Loaded %u vectors (sample=%u), nlist=%u\n",
				   ext_nvecs,
				   max_samples,
				   nlist);
	}
	else
	{
		uint32_t nlevels = config->nlevels;
		nvecs			 = 0;
		vectors			 = generate_hierarchical_vectors(
				 dim, fan_out, nlevels, VECS_PER_LEAF, &nvecs);

		nlist = power_u32(fan_out, nlevels);

		if (verbose)
			printf("  Generated %u vectors (%u leaf clusters x %u "
				   "vecs)\n",
				   nvecs,
				   nlist,
				   VECS_PER_LEAF);
	}

	/* Compute global centroid (mean of all vectors) */
	mkt_vector_mean(vectors, nvecs, dim, global_centroid_out);

	/*
	 * Run hierarchical k-means using the shared module.
	 */
	KMeansOptions km_opts = MKT_KMEANS_OPTIONS_DEFAULT;

	HKMeansResult *tree = mkt_hkmeans_f32(
			vectors, nvecs, dim, nlist, fan_out, metric, &km_opts);
	if (tree == NULL)
	{
		fprintf(stderr, "Error: hierarchical k-means failed\n");
		mkt_free(vectors);
		return NULL;
	}

	if (verbose)
		printf("  Tree: %u nodes, %u levels, %u leaves\n",
			   tree->nnodes,
			   tree->nlevels,
			   tree->nleaves);

	/*
	 * Pre-compute block numbers for each BFS node.
	 * Block numbers are assigned sequentially.
	 */
	BlockNumber *node_first_blkno = mkt_alloc(
			tree->nnodes * sizeof(BlockNumber));
	BlockNumber next_blkno = mkt_compute_centroid_layout(
			tree, max_entries, 0, node_first_blkno);
	uint32_t total_pages = (uint32_t)next_blkno;

	uint32_t total_centroids = 0;
	for (uint32_t i = 0; i < tree->nnodes; i++)
		total_centroids += tree->nodes[i].nchildren;

	/* Allocate page buffer */
	char *pages = mkt_alloc((size_t)total_pages * BLCKSZ);
	memset(pages, 0, (size_t)total_pages * BLCKSZ);

	/* Build-time storage for mkt_centroid_write_pages */
	TestStorage build_storage = {
			.base.ops	= &test_storage_ops,
			.pages		= pages,
			.next_blkno = 0,
	};

	/*
	 * Allocate rerank vector storage: centroid vectors indexed by
	 * (page_blkno * max_entries + entry_offset) for exact distance
	 * reranking during beam search.
	 */
	float *rerank_vecs = mkt_alloc(
			(size_t)total_pages * max_entries * dim * sizeof(float));
	memset(rerank_vecs,
		   0,
		   (size_t)total_pages * max_entries * dim * sizeof(float));

	/* Populate rerank_vecs from tree centroids by page slot */
	for (uint32_t i = 0; i < tree->nnodes; i++)
	{
		HKMeansNode *node = &tree->nodes[i];
		for (uint32_t c = 0; c < node->nchildren; c++)
		{
			BlockNumber blk	 = node_first_blkno[i] + c / max_entries;
			uint32_t	slot = blk * max_entries + (c % max_entries);
			memcpy(rerank_vecs + (size_t)slot * dim,
				   node->centroids + (size_t)c * dim,
				   dim * sizeof(float));
		}
	}

	/*
	 * Leaf arrays: query seeds and leaf index mapping.
	 */
	uint32_t leaf_centroids = tree->nleaves;

	float *seeds = mkt_alloc((size_t)leaf_centroids * dim * sizeof(float));
	memcpy(seeds,
		   tree->leaf_centroids,
		   (size_t)leaf_centroids * dim * sizeof(float));

	/* Assign all vectors (not just the clustering sample) to
	 * leaf centroids via tree descent. For external datasets,
	 * use the full original vectors; for synthetic, vectors
	 * already contains all of them. */
	const float *assign_vecs = ext_vectors != NULL ? ext_vectors : vectors;
	uint32_t	 assign_n	 = ext_vectors != NULL ? ext_nvecs : nvecs;

	uint32_t *leaf_offsets = mkt_alloc(
			(leaf_centroids + 1) * sizeof(uint32_t));
	uint32_t *leaf_idx = mkt_alloc(assign_n * sizeof(uint32_t));
	memset(leaf_offsets, 0, (leaf_centroids + 1) * sizeof(uint32_t));

	uint32_t *assignments = mkt_alloc(assign_n * sizeof(uint32_t));
	for (uint32_t v = 0; v < assign_n; v++)
	{
		assignments[v] = mkt_hkmeans_assign(
				tree, assign_vecs + (size_t)v * dim, metric, NULL);
		leaf_offsets[assignments[v] + 1]++;
	}

	for (uint32_t i = 1; i <= leaf_centroids; i++)
		leaf_offsets[i] += leaf_offsets[i - 1];

	uint32_t *pos = mkt_alloc(leaf_centroids * sizeof(uint32_t));
	memcpy(pos, leaf_offsets, leaf_centroids * sizeof(uint32_t));

	for (uint32_t v = 0; v < assign_n; v++)
		leaf_idx[pos[assignments[v]]++] = v;

	mkt_free(pos);
	mkt_free(assignments);

	/* Build leaf index array as posting_heads so beam search
	 * results carry the leaf cluster index in posting_head. */
	BlockNumber *leaf_heads = mkt_alloc(tree->nleaves * sizeof(BlockNumber));
	for (uint32_t i = 0; i < tree->nleaves; i++)
		leaf_heads[i] = (BlockNumber)i;

	/* Write centroid pages using shared builder */
	mkt_write_centroid_tree(
			&build_storage.base,
			tree,
			dim,
			fan_out,
			fmt,
			params,
			global_centroid_out,
			leaf_heads,
			node_first_blkno);
	mkt_free(leaf_heads);

	mkt_free(node_first_blkno);
	mkt_hkmeans_result_destroy(tree);

	*total_pages_out	 = total_pages;
	*total_centroids_out = total_centroids;
	*query_seeds_out	 = seeds;
	*nseeds_out			 = leaf_centroids;
	*rerank_vecs_out	 = rerank_vecs;
	*leaf_idx_out		 = leaf_idx;
	*leaf_offsets_out	 = leaf_offsets;

	/* For external datasets, the caller owns the full vectors;
	 * free the clustering sample and return NULL. For synthetic
	 * data, vectors is the full dataset. */
	if (ext_vectors != NULL)
	{
		mkt_free(vectors);
		*vectors_out = NULL;
		*nvecs_out	 = ext_nvecs;
	}
	else
	{
		*vectors_out = vectors;
		*nvecs_out	 = nvecs;
	}
	return pages;
}

/* ----------------------------------------------------------------
 * Usage / help
 * ---------------------------------------------------------------- */

static void
print_usage(CmdContext *ctx)
{
	CMD_USAGE_HEADER(ctx, "bench search");
	printf("Benchmark beam search over a synthetic centroid "
		   "tree.\n\n");
	printf("Options:\n");
	printf("  --dim <int>        Vector dimension "
		   "(default: %d)\n",
		   DEFAULT_DIM);
	printf("  --nlevels <int>    Tree depth "
		   "(default: %d)\n",
		   DEFAULT_NLEVELS);
	printf("  --fan-out <int>    Children per node "
		   "(default: %d)\n",
		   DEFAULT_FAN_OUT);
	printf("  --beam-width <int> Beam width "
		   "(default: %d)\n",
		   DEFAULT_BEAM_WIDTH);
	printf("  --nprobe <int>     Final result count "
		   "(default: %d)\n",
		   DEFAULT_NPROBE);
	printf("  --queries <int>    Number of queries "
		   "(default: %d)\n",
		   DEFAULT_QUERIES);
	printf("  --runs <int>       Runs per query "
		   "(default: %d)\n",
		   DEFAULT_RUNS);
#ifdef MKT_HAVE_HDF5
	printf("  --hdf5 <path>      Load vectors from HDF5 file\n");
	printf("  --nlist <int>      Number of clusters "
		   "(0=auto, HDF5 only)\n");
	printf("  --metric <str>     Distance metric override "
		   "(angular, euclidean)\n");
#endif
	printf("  --help             Show this help message\n");
	printf("\n");
	printf("Examples:\n");
	CMD_USAGE_EXAMPLE(ctx, "bench search", "--dim 768");
	CMD_USAGE_EXAMPLE(
			ctx, "bench search", "--dim 128 --nlevels 3 --fan-out 16");
}

/* ----------------------------------------------------------------
 * Benchmark variants and per-format state
 * ---------------------------------------------------------------- */

typedef struct BenchVariant
{
	uint8_t			fmt;
	MktDistanceMode mode; /* only used for RaBitQ */
	const char	   *name;
} BenchVariant;

static const BenchVariant variants[] = {
		{MKT_CENTROID_FMT_RABITQ, MKT_DISTANCE_MODE_ASYMMETRIC, "rabitq-asym"},
		{MKT_CENTROID_FMT_RABITQ, MKT_DISTANCE_MODE_SYMMETRIC, "rabitq-sym"},
		{MKT_CENTROID_FMT_FLOAT, 0, "float32"},
		{MKT_CENTROID_FMT_HALF, 0, "float16"},
};
#define NUM_VARIANTS 4
#define NUM_FORMATS	 3

typedef struct TreeState
{
	char	   *pages;
	float	   *rerank_vecs;
	TestStorage storage;
	uint32_t	total_pages;
	uint32_t	total_centroids;
	uint32_t	max_entries;
} TreeState;

static int
format_index(uint8_t fmt)
{
	switch (fmt)
	{
	case MKT_CENTROID_FMT_RABITQ:
		return 0;
	case MKT_CENTROID_FMT_FLOAT:
		return 1;
	case MKT_CENTROID_FMT_HALF:
		return 2;
	default:
		return -1;
	}
}

/* ----------------------------------------------------------------
 * Entry point
 * ---------------------------------------------------------------- */

int
cmd_bench_search(CmdContext *ctx)
{
	BenchConfig config = {
			.dim		= DEFAULT_DIM,
			.nlevels	= DEFAULT_NLEVELS,
			.fan_out	= DEFAULT_FAN_OUT,
			.beam_width = DEFAULT_BEAM_WIDTH,
			.nprobe		= DEFAULT_NPROBE,
			.queries	= DEFAULT_QUERIES,
			.runs		= DEFAULT_RUNS,
			.help		= false,
	};

	static struct option long_options[] = {
			{"dim", required_argument, 0, 'd'},
			{"nlevels", required_argument, 0, 'l'},
			{"fan-out", required_argument, 0, 'f'},
			{"beam-width", required_argument, 0, 'b'},
			{"nprobe", required_argument, 0, 'p'},
			{"queries", required_argument, 0, 'q'},
			{"runs", required_argument, 0, 'r'},
			{"nlist", required_argument, 0, 'n'},
			{"hdf5", required_argument, 0, 'H'},
			{"metric", required_argument, 0, 'm'},
			{"help", no_argument, 0, 'h'},
			{0, 0, 0, 0},
	};

	optind = 1;

	int opt;
	while ((opt = getopt_long(
					ctx->argc,
					ctx->argv,
					"d:l:f:b:p:q:r:n:H:m:h",
					long_options,
					NULL)) != -1)
	{
		switch (opt)
		{
		case 'd':
			config.dim = (Dimension)atoi(optarg);
			break;
		case 'l':
			config.nlevels = (uint32_t)atoi(optarg);
			break;
		case 'f':
			config.fan_out = (uint32_t)atoi(optarg);
			break;
		case 'b':
			config.beam_width = (uint32_t)atoi(optarg);
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
		case 'n':
			config.nlist = (uint32_t)atoi(optarg);
			break;
		case 'H':
			config.hdf5_path = optarg;
			break;
		case 'm':
			config.metric_str = optarg;
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

	/* Load HDF5 data if requested */
	float		  *hdf5_train	  = NULL;
	float		  *hdf5_test	  = NULL;
	int64_t		  *hdf5_neighbors = NULL;
	uint32_t	   hdf5_nvecs	  = 0;
	uint32_t	   hdf5_gt_k	  = 0;
	DistanceMetric metric		  = DISTANCE_L2;

#ifndef MKT_HAVE_HDF5
	if (config.hdf5_path != NULL)
	{
		fprintf(stderr,
				"Error: --hdf5 requires HDF5 support. "
				"Install libhdf5-dev and reconfigure.\n");
		return 1;
	}
#else
	if (config.hdf5_path != NULL)
	{
		hsize_t n_train, dim_h, n_test, dim_test;
		hdf5_train =
				load_hdf5_float(config.hdf5_path, "train", &n_train, &dim_h);
		hdf5_test =
				load_hdf5_float(config.hdf5_path, "test", &n_test, &dim_test);

		if (hdf5_train == NULL || hdf5_test == NULL)
		{
			fprintf(stderr, "Error: failed to load HDF5 datasets\n");
			free(hdf5_train);
			free(hdf5_test);
			return 1;
		}

		config.dim = (Dimension)dim_h;
		hdf5_nvecs = (uint32_t)n_train;

		/* Load ground truth neighbors (optional, suppress HDF5
		 * errors since the dataset may not exist) */
		hsize_t n_gt, k_gt;
		H5Eset_auto(H5E_DEFAULT, NULL, NULL);
		hdf5_neighbors =
				load_hdf5_int64(config.hdf5_path, "neighbors", &n_gt, &k_gt);
		H5Eset_auto(H5E_DEFAULT, (H5E_auto_t)H5Eprint, stderr);
		if (hdf5_neighbors != NULL)
			hdf5_gt_k = (uint32_t)k_gt;

		/* Detect metric from HDF5 "distance" attribute */
		const char *hdf5_metric = NULL;
		{
			H5Eset_auto(H5E_DEFAULT, NULL, NULL);
			hid_t file =
					H5Fopen(config.hdf5_path, H5F_ACC_RDONLY, H5P_DEFAULT);
			if (file >= 0 && H5Aexists(file, "distance"))
			{
				hid_t attr	= H5Aopen(file, "distance", H5P_DEFAULT);
				hid_t atype = H5Aget_type(attr);

				if (H5Tis_variable_str(atype))
				{
					/* Variable-length string */
					char *vstr	  = NULL;
					hid_t memtype = H5Tcopy(H5T_C_S1);
					H5Tset_size(memtype, H5T_VARIABLE);
					H5Aread(attr, memtype, &vstr);
					if (vstr != NULL)
						hdf5_metric = strdup(vstr);
					H5free_memory(vstr);
					H5Tclose(memtype);
				}
				else
				{
					/* Fixed-length string */
					size_t sz	   = H5Tget_size(atype);
					char  *buf	   = malloc(sz + 1);
					hid_t  memtype = H5Tcopy(H5T_C_S1);
					H5Tset_size(memtype, sz + 1);
					H5Aread(attr, memtype, buf);
					buf[sz]		= '\0';
					hdf5_metric = buf;
					H5Tclose(memtype);
				}

				H5Tclose(atype);
				H5Aclose(attr);
			}
			if (file >= 0)
				H5Fclose(file);
			H5Eset_auto(H5E_DEFAULT, (H5E_auto_t)H5Eprint, stderr);
		}

		/* Resolve metric: CLI override > HDF5 attribute > filename */
		const char *detected = hdf5_metric;
		if (detected == NULL)
		{
			/* Fall back to filename heuristic */
			if (strstr(config.hdf5_path, "angular") != NULL ||
				strstr(config.hdf5_path, "cosine") != NULL)
				detected = "angular";
		}

		if (config.metric_str != NULL)
		{
			if (detected != NULL && strcmp(config.metric_str, detected) != 0)
			{
				fprintf(stderr,
						"Warning: --metric '%s' overrides "
						"dataset metric '%s'\n",
						config.metric_str,
						detected);
			}
			detected = config.metric_str;
		}

		if (detected != NULL && (strcmp(detected, "angular") == 0 ||
								 strcmp(detected, "cosine") == 0))
		{
			metric = DISTANCE_COSINE;
			normalize_vectors(hdf5_train, hdf5_nvecs, config.dim);
			normalize_vectors(hdf5_test, (uint32_t)n_test, config.dim);
		}

		free((void *)hdf5_metric);

		/* Auto-derive tree shape from dataset */
		if (config.nlist == 0)
		{
			config.nlist = (uint32_t)sqrt((double)hdf5_nvecs);
			if (config.nlist < 1)
				config.nlist = 1;
			if (config.nlist > 10000)
				config.nlist = 10000;
		}
		config.fan_out = mkt_auto_fan_out(0, config.nlist, 0);
		/* nlevels from fan_out and nlist */
		config.nlevels = 1;
		{
			uint32_t leaves = config.fan_out;
			while (leaves < config.nlist)
			{
				leaves *= config.fan_out;
				config.nlevels++;
			}
		}

		if (config.queries > (uint32_t)n_test)
			config.queries = (uint32_t)n_test;

		printf("Search benchmark (HDF5: %s):\n", config.hdf5_path);
		printf("  Vectors: %u x %u (%s)\n",
			   hdf5_nvecs,
			   config.dim,
			   metric == DISTANCE_COSINE ? "cosine" : "L2");
	}
	else
#endif /* MKT_HAVE_HDF5 */
	{
		if (config.dim == 0 || config.nlevels == 0 || config.fan_out == 0)
		{
			fprintf(stderr, "Error: dim, nlevels, fan_out must be > 0\n");
			return 1;
		}
		printf("Search benchmark (dim=%u, nlevels=%u, "
			   "fan_out=%u):\n",
			   config.dim,
			   config.nlevels,
			   config.fan_out);
	}

	if (config.queries == 0 || config.runs == 0)
	{
		fprintf(stderr, "Error: queries and runs must be > 0\n");
		return 1;
	}

	/* Create RaBitQ params (needed for RaBitQ format) */
	srand(42);
	RaBitQParams *params = mkt_rabitq_create(config.dim, 42);
	if (params == NULL)
	{
		fprintf(stderr, "Error: failed to create RaBitQ params\n");
		return 1;
	}

	/* Build one tree per format (same srand → same clustering) */
	static const uint8_t fmts[NUM_FORMATS] = {
			MKT_CENTROID_FMT_RABITQ,
			MKT_CENTROID_FMT_FLOAT,
			MKT_CENTROID_FMT_HALF,
	};
	static const char *fmt_names[NUM_FORMATS] = {
			"rabitq",
			"float32",
			"float16",
	};

	float	 *centroid	   = mkt_alloc(config.dim * sizeof(float));
	float	 *query_seeds  = NULL;
	uint32_t  nseeds	   = 0;
	float	 *all_vectors  = NULL;
	uint32_t *leaf_idx	   = NULL;
	uint32_t *leaf_offsets = NULL;
	uint32_t  nvecs		   = 0;
	TreeState trees[NUM_FORMATS];
	bool	  build_ok = true;

	for (int f = 0; f < NUM_FORMATS; f++)
	{
		srand(42); /* same seed → identical clustering */
		float	 *seeds	   = NULL;
		uint32_t  ns	   = 0;
		float	 *vecs_tmp = NULL;
		uint32_t *lidx_tmp = NULL;
		uint32_t *loff_tmp = NULL;
		uint32_t  nv_tmp   = 0;

		trees[f].max_entries =
				mkt_centroid_max_entries_fmt(config.dim, fmts[f]);

		trees[f].pages = build_tree(
				&config,
				params,
				fmts[f],
				hdf5_train,
				hdf5_nvecs,
				metric,
				&trees[f].total_pages,
				&trees[f].total_centroids,
				centroid,
				&seeds,
				&ns,
				&trees[f].rerank_vecs,
				&vecs_tmp,
				&lidx_tmp,
				&loff_tmp,
				&nv_tmp,
				f == 0); /* verbose only for first */

		if (trees[f].pages == NULL)
		{
			build_ok = false;
			mkt_free(seeds);
			mkt_free(vecs_tmp);
			mkt_free(lidx_tmp);
			mkt_free(loff_tmp);
			break;
		}

		/*
		 * Save seeds and leaf data from first build
		 * (identical across formats due to same srand).
		 */
		if (f == 0)
		{
			query_seeds	 = seeds;
			nseeds		 = ns;
			all_vectors	 = vecs_tmp != NULL ? vecs_tmp : hdf5_train;
			leaf_idx	 = lidx_tmp;
			leaf_offsets = loff_tmp;
			nvecs		 = nv_tmp;
		}
		else
		{
			mkt_free(seeds);
			mkt_free(vecs_tmp);
			mkt_free(lidx_tmp);
			mkt_free(loff_tmp);
		}

		trees[f].storage = (TestStorage){
				.base.ops	 = &test_storage_ops,
				.pages		 = trees[f].pages,
				.rerank_vecs = trees[f].rerank_vecs,
				.max_entries = trees[f].max_entries,
				.dim		 = config.dim,
		};
	}

	if (!build_ok)
	{
		for (int f = 0; f < NUM_FORMATS; f++)
		{
			mkt_free(trees[f].rerank_vecs);
			mkt_free(trees[f].pages);
		}
		mkt_free(leaf_offsets);
		mkt_free(leaf_idx);
		if (all_vectors != hdf5_train)
			mkt_free(all_vectors);
		mkt_free(query_seeds);
		mkt_free(centroid);
		mkt_rabitq_destroy(params);
		free(hdf5_train);
		free(hdf5_test);
		return 1;
	}

	printf("  Trees:\n");
	for (int f = 0; f < NUM_FORMATS; f++)
	{
		double kb = (double)trees[f].total_pages * BLCKSZ / 1024.0;
		printf("    %-8s %4u pages (%7.1f KB)\n",
			   fmt_names[f],
			   trees[f].total_pages,
			   kb);
	}

	printf("  Params: beam_width=%u, nprobe=%u\n\n",
		   config.beam_width,
		   config.nprobe);

	/* Pre-generate all queries (same across variants) */
	uint32_t nq		 = config.queries;
	uint32_t nprobe	 = config.nprobe;
	float	*queries = mkt_alloc((size_t)nq * config.dim * sizeof(float));

	if (hdf5_test != NULL)
	{
		/* Use HDF5 test vectors as queries */
		memcpy(queries, hdf5_test, (size_t)nq * config.dim * sizeof(float));
	}
	else
	{
		srand(12345);
		for (uint32_t q = 0; q < nq; q++)
		{
			float *qvec = queries + (size_t)q * config.dim;
			if (nseeds > 0)
			{
				Dimension	 subdim	  = config.dim < 32 ? config.dim : 32;
				uint32_t	 seed_idx = (uint32_t)rand() % nseeds;
				const float *seed	  = query_seeds +
									(size_t)seed_idx * config.dim;
				for (Dimension d = 0; d < config.dim; d++)
					qvec[d] = seed[d] + 0.05f * rand_normal();
				for (Dimension d = 0; d < subdim; d++)
					qvec[d] += 1.5f * rand_normal();
			}
			else
			{
				generate_random_vector(qvec, config.dim);
			}
		}
	}

	/*
	 * Ground truth: top-K nearest neighbors per query.
	 * Use HDF5 neighbors dataset when available, otherwise
	 * compute by brute-force over all vectors.
	 */
	uint32_t  topk = DEFAULT_TOPK;
	uint32_t *gt   = mkt_alloc((size_t)nq * topk * sizeof(uint32_t));

	if (hdf5_neighbors != NULL)
	{
		for (uint32_t q = 0; q < nq; q++)
		{
			uint32_t gt_k = topk < hdf5_gt_k ? topk : hdf5_gt_k;
			for (uint32_t i = 0; i < gt_k; i++)
				gt[q * topk + i] = (uint32_t)hdf5_neighbors[q * hdf5_gt_k + i];
			for (uint32_t i = gt_k; i < topk; i++)
				gt[q * topk + i] = UINT32_MAX;
		}
	}
	else
	{
		for (uint32_t q = 0; q < nq; q++)
		{
			const float *qvec = queries + (size_t)q * config.dim;

			MktTopK tk;
			mkt_topk_init(&tk, topk);

			for (uint32_t v = 0; v < nvecs; v++)
			{
				float d2 = mkt_l2_distance_squared(
						qvec,
						all_vectors + (size_t)v * config.dim,
						config.dim);
				mkt_topk_insert(&tk, d2, 0.0f, v);
			}

			MktTopKEntry *entries = mkt_alloc(
					tk.cand_count * sizeof(MktTopKEntry));
			uint32_t count;
			mkt_topk_extract_sorted(&tk, entries, &count);

			uint32_t gt_k = count < topk ? count : topk;
			for (uint32_t i = 0; i < gt_k; i++)
				gt[q * topk + i] = (uint32_t)entries[i].id;
			for (uint32_t i = gt_k; i < topk; i++)
				gt[q * topk + i] = UINT32_MAX;

			mkt_free(entries);
			mkt_topk_cleanup(&tk);
		}
	}

	/*
	 * Centroid ground truth: brute-force top-nprobe leaf centroids
	 * by exact L2 distance. Used to measure routing recall
	 * (how well RaBitQ selects the same centroids as exact search).
	 */
	uint32_t *centroid_gt = mkt_alloc((size_t)nq * nprobe * sizeof(uint32_t));

	for (uint32_t q = 0; q < nq; q++)
	{
		const float *qvec = queries + (size_t)q * config.dim;

		MktTopK ctk;
		mkt_topk_init(&ctk, nprobe);

		for (uint32_t c = 0; c < nseeds; c++)
		{
			float d2 = mkt_l2_distance_squared(
					qvec, query_seeds + (size_t)c * config.dim, config.dim);
			mkt_topk_insert(&ctk, d2, 0.0f, c);
		}

		MktTopKEntry *centries = mkt_alloc(
				ctk.cand_count * sizeof(MktTopKEntry));
		uint32_t ccount;
		mkt_topk_extract_sorted(&ctk, centries, &ccount);

		uint32_t cgt_k = ccount < nprobe ? ccount : nprobe;
		for (uint32_t i = 0; i < cgt_k; i++)
			centroid_gt[q * nprobe + i] = (uint32_t)centries[i].id;
		for (uint32_t i = cgt_k; i < nprobe; i++)
			centroid_gt[q * nprobe + i] = UINT32_MAX;

		mkt_free(centries);
		mkt_topk_cleanup(&ctk);
	}

	/* Benchmark each variant */
	uint32_t   total_samples = nq * config.runs;
	BenchStats var_stats[NUM_VARIANTS];
	double	   recall_sum[NUM_VARIANTS];
	double	   routing_recall_sum[NUM_VARIANTS];

	for (int v = 0; v < NUM_VARIANTS; v++)
	{
		bench_stats_init(&var_stats[v], total_samples);
		recall_sum[v]		  = 0.0;
		routing_recall_sum[v] = 0.0;
	}

	MktCentroidResult *results = mkt_alloc(nprobe * sizeof(MktCentroidResult));

	MktCentroidSearchStats var_stats_search[NUM_VARIANTS];

	for (int v = 0; v < NUM_VARIANTS; v++)
	{
		int		   fi = format_index(variants[v].fmt);
		TreeState *ts = &trees[fi];

		var_stats_search[v] = (MktCentroidSearchStats){0};

		for (uint32_t q = 0; q < nq; q++)
		{
			const float *qvec	   = queries + (size_t)q * config.dim;
			VectorRef	 query_ref = {.data = qvec, .dim = config.dim};

			/* Prepare RaBitQ query state if needed */
			RaBitQQueryState *qs = NULL;
			if (variants[v].fmt == MKT_CENTROID_FMT_RABITQ)
			{
				VectorRef cent_ref = {.data = centroid, .dim = config.dim};
				qs				   = mkt_rabitq_prepare_query_ex(
						params, query_ref, cent_ref, variants[v].mode);
			}

			MktCentroidSearchState state = {
					.qstate		= qs,
					.query		= qvec,
					.storage	= &ts->storage.base,
					.beam_width = config.beam_width,
					.nprobe		= nprobe,
					.dim		= config.dim,
			};

			/* Warmup */
			for (int w = 0; w < WARMUP_RUNS; w++)
				mkt_centroid_beam_search(
						&state,
						0,
						(uint8_t)config.nlevels,
						results,
						NULL,
						NULL);

			/* Timed runs */
			for (uint32_t r = 0; r < config.runs; r++)
			{
				uint64_t start = get_time_ns();
				mkt_centroid_beam_search(
						&state,
						0,
						(uint8_t)config.nlevels,
						results,
						NULL,
						NULL);
				uint64_t end = get_time_ns();
				bench_stats_add(&var_stats[v], (double)(end - start) / 1000.0);
			}

			/* Quality run (accumulate search stats) */
			uint32_t n_results = mkt_centroid_beam_search(
					&state,
					0,
					(uint8_t)config.nlevels,
					results,
					NULL,
					&var_stats_search[v]);

			/*
			 * Routing recall: how many of the beam search's
			 * selected centroids match the exact top-nprobe?
			 */
			uint32_t routing_hits = 0;
			for (uint32_t j = 0; j < n_results; j++)
			{
				uint32_t li = (uint32_t)results[j].posting_head;
				for (uint32_t g = 0; g < nprobe; g++)
				{
					if (centroid_gt[q * nprobe + g] == li)
					{
						routing_hits++;
						break;
					}
				}
			}
			routing_recall_sum[v] += (double)routing_hits /
									 (n_results > 0 ? n_results : 1);

			/*
			 * End-to-end recall: scan vectors in selected
			 * clusters, find top-K, compare to ground truth.
			 */
			MktTopK scan;
			mkt_topk_init(&scan, topk);

			for (uint32_t j = 0; j < n_results; j++)
			{
				uint32_t li = (uint32_t)results[j].posting_head;
				for (uint32_t i = leaf_offsets[li]; i < leaf_offsets[li + 1];
					 i++)
				{
					uint32_t vi = leaf_idx[i];
					float	 d2 = mkt_l2_distance_squared(
							   qvec,
							   all_vectors + (size_t)vi * config.dim,
							   config.dim);
					mkt_topk_insert(&scan, d2, 0.0f, vi);
				}
			}

			MktTopKEntry *scan_entries = mkt_alloc(
					scan.cand_count * sizeof(MktTopKEntry));
			uint32_t scan_count;
			mkt_topk_extract_sorted(&scan, scan_entries, &scan_count);

			/* Count hits: scan results in ground truth set */
			uint32_t hits = 0;
			for (uint32_t i = 0; i < scan_count && i < topk; i++)
			{
				uint32_t sid = (uint32_t)scan_entries[i].id;
				for (uint32_t g = 0; g < topk; g++)
				{
					if (gt[q * topk + g] == sid)
					{
						hits++;
						break;
					}
				}
			}
			recall_sum[v] += (double)hits / topk;

			mkt_free(scan_entries);
			mkt_topk_cleanup(&scan);

			if (qs)
				mkt_rabitq_free_query(qs);
		}
	}

	/* Print results */
	char recall_hdr[32];
	snprintf(recall_hdr, sizeof(recall_hdr), "R@%u", topk);

	printf("  %-14s %9s %9s %9s %9s %9s %9s %9s\n",
		   "Variant",
		   "Avg (us)",
		   "Min (us)",
		   "Max (us)",
		   "Stddev",
		   recall_hdr,
		   "Route R",
		   "QPS");

	for (int v = 0; v < NUM_VARIANTS; v++)
	{
		bench_stats_compute(&var_stats[v]);
		double recall_pct = nq > 0 ? 100.0 * recall_sum[v] / nq : 0.0;
		double route_pct  = nq > 0 ? 100.0 * routing_recall_sum[v] / nq : 0.0;
		double qps = var_stats[v].avg > 0 ? 1e6 / var_stats[v].avg : 0.0;

		printf("  %-14s %9.2f %9.2f %9.2f %9.2f %8.1f%% %8.1f%% %9.0f\n",
			   variants[v].name,
			   var_stats[v].avg,
			   var_stats[v].min,
			   var_stats[v].max,
			   var_stats[v].stddev,
			   recall_pct,
			   route_pct,
			   qps);
	}

	/* Print routing distance stats */
	printf("\n  Routing distance computations"
		   " (avg per query):\n");
	printf("  %-14s %8s\n", "Variant", "Scored");
	for (int v = 0; v < NUM_VARIANTS; v++)
	{
		MktCentroidSearchStats *ss		 = &var_stats_search[v];
		double					avg_dist = (double)ss->dist_calcs / nq;
		printf("  %-14s %8.1f\n", variants[v].name, avg_dist);
	}

	/* Cleanup */
	for (int v = 0; v < NUM_VARIANTS; v++)
		bench_stats_free(&var_stats[v]);
	mkt_free(results);
	mkt_free(centroid_gt);
	mkt_free(gt);
	mkt_free(queries);
	mkt_free(leaf_offsets);
	mkt_free(leaf_idx);
	if (all_vectors != hdf5_train)
		mkt_free(all_vectors);
	mkt_free(query_seeds);

	for (int f = 0; f < NUM_FORMATS; f++)
	{
		free(trees[f].storage.rerank_lb);
		free(trees[f].storage.rerank_exact);
		mkt_free(trees[f].rerank_vecs);
		mkt_free(trees[f].pages);
	}

	mkt_free(centroid);
	mkt_rabitq_destroy(params);
	free(hdf5_train);
	free(hdf5_test);
	free(hdf5_neighbors);

	return 0;
}
