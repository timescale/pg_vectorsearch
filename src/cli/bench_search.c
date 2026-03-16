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

#include "algo/hkmeans.h"
#include "algo/kmeans.h"
#include "algo/topk.h"
#include "algo/vecops.h"
#include "cmd.h"
#include "core/memory.h"
#include "index/centroid_build.h"
#include "index/centroid_page.h"
#include "index/centroid_search.h"
#include "index/posting_build.h"
#include "index/posting_scan.h"
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
	Dimension dim;
	uint32_t  nlevels;
	uint32_t  fan_out;
	uint32_t  beam_width;
	uint32_t  nprobe;
	uint32_t  queries;
	uint32_t  runs;
	bool	  help;
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
	float	  *medoid_vecs; /* [total_slots * dim] flat array */
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

	if (ts->medoid_vecs == NULL || query_f == NULL || count == 0)
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
			const float *vec  = ts->medoid_vecs + (size_t)midx * dim;
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

/* Encode TID as uint64 for top-K storage */
static inline uint64_t
encode_tid(BlockNumber block, OffsetNumber offset)
{
	return ((uint64_t)block << 16) | (uint64_t)offset;
}

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

/*
 * find_medoid - Find the actual vector closest to a centroid
 *
 * Brute-force scan of all vectors, returning a pointer to the
 * one with minimum squared L2 distance to the centroid. Falls
 * back to the centroid itself if nvecs == 0.
 */
static const float *
find_medoid(
		const float *all_vecs,
		uint32_t	 nvecs,
		const float *centroid,
		Dimension	 dim)
{
	const float *best	   = centroid;
	float		 best_dist = FLT_MAX;

	for (uint32_t v = 0; v < nvecs; v++)
	{
		const float *vec  = all_vecs + (size_t)v * dim;
		float		 dist = 0.0f;
		for (Dimension d = 0; d < dim; d++)
		{
			float diff = vec[d] - centroid[d];
			dist += diff * diff;
		}

		if (dist < best_dist)
		{
			best_dist = dist;
			best	  = vec;
		}
	}

	return best;
}

/* ----------------------------------------------------------------
 * build_tree - Build centroid tree + posting pages
 *
 * Generates nvecs random vectors, clusters them hierarchically
 * using mkt_hkmeans_f32(), builds centroid pages in the specified
 * format, then encodes each leaf cluster's vectors with IVF
 * residual encoding and writes them into posting pages.
 *
 * Returns the allocated page buffer, or NULL on failure.
 * ---------------------------------------------------------------- */
static char *
build_tree(
		const BenchConfig  *config,
		const RaBitQParams *params,
		uint8_t				fmt,
		uint32_t		   *total_pages_out,
		uint32_t		   *centroid_pages_out,
		uint32_t		   *total_centroids_out,
		float			   *global_centroid_out,
		float			  **query_seeds_out,
		BlockNumber		  **seed_posting_heads_out,
		uint32_t		   *nseeds_out,
		float			  **medoid_vecs_out,
		float			  **vectors_out,
		float			  **leaf_centroids_out,
		BlockNumber		  **leaf_posting_heads_out,
		uint32_t		   *nleaves_out,
		uint32_t		   *nvecs_out,
		bool				verbose)
{
	Dimension dim	  = config->dim;
	uint32_t  fan_out = config->fan_out;
	uint32_t  nlevels = config->nlevels;

	uint32_t max_entries = mkt_centroid_max_entries_fmt(dim, fmt);

	/*
	 * Generate hierarchically-clustered vectors.
	 */
	uint32_t nvecs	 = 0;
	float	*vectors = generate_hierarchical_vectors(
			  dim, fan_out, nlevels, VECS_PER_LEAF, &nvecs);

	uint32_t nlist = power_u32(fan_out, nlevels);

	if (verbose)
		printf("  Generated %u vectors (%u leaf clusters x %u "
			   "vecs)\n",
			   nvecs,
			   nlist,
			   VECS_PER_LEAF);

	/* Compute global centroid (mean of all vectors) */
	mkt_vector_mean(vectors, nvecs, dim, global_centroid_out);

	/*
	 * Run hierarchical k-means using the shared module.
	 */
	KMeansOptions km_opts = MKT_KMEANS_OPTIONS_DEFAULT;

	HKMeansResult *tree = mkt_hkmeans_f32(
			vectors, nvecs, dim, nlist, fan_out, DISTANCE_L2, &km_opts);
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
	 * Pre-compute centroid page block numbers for each BFS node.
	 */
	BlockNumber *node_first_blkno = mkt_alloc(
			tree->nnodes * sizeof(BlockNumber));
	BlockNumber next_blkno		= 0;
	uint32_t	centroid_npages = 0;

	for (uint32_t i = 0; i < tree->nnodes; i++)
	{
		uint32_t n			  = tree->nodes[i].nchildren;
		uint32_t pages_needed = (n + max_entries - 1) / max_entries;
		if (pages_needed == 0)
			pages_needed = 1;
		node_first_blkno[i] = next_blkno;
		next_blkno += pages_needed;
		centroid_npages += pages_needed;
	}

	uint32_t total_centroids = 0;
	for (uint32_t i = 0; i < tree->nnodes; i++)
		total_centroids += tree->nodes[i].nchildren;

	/*
	 * Leaf arrays: query seeds and leaf index mapping.
	 * Build leaf_offsets/leaf_idx before page allocation so we
	 * can pre-compute posting page counts.
	 */
	uint32_t nleaves = tree->nleaves;

	float		*seeds	  = mkt_alloc((size_t)nleaves * dim * sizeof(float));
	BlockNumber *seed_phs = mkt_alloc(nleaves * sizeof(BlockNumber));
	uint32_t	 nsaved	  = 0;

	uint32_t *leaf_offsets = mkt_alloc((nleaves + 1) * sizeof(uint32_t));
	uint32_t *leaf_idx	   = mkt_alloc(nvecs * sizeof(uint32_t));
	memset(leaf_offsets, 0, (nleaves + 1) * sizeof(uint32_t));

	const float *leaf_cents = tree->leaf_centroids;

	for (uint32_t v = 0; v < nvecs; v++)
	{
		const float *vec	= vectors + (size_t)v * dim;
		float		 best_d = FLT_MAX;
		uint32_t	 best_c = 0;
		for (uint32_t c = 0; c < nleaves; c++)
		{
			float d = mkt_l2_distance_squared(
					vec, leaf_cents + (size_t)c * dim, dim);
			if (d < best_d)
			{
				best_d = d;
				best_c = c;
			}
		}
		leaf_offsets[best_c + 1]++;
	}

	for (uint32_t i = 1; i <= nleaves; i++)
		leaf_offsets[i] += leaf_offsets[i - 1];

	uint32_t *pos = mkt_alloc(nleaves * sizeof(uint32_t));
	memcpy(pos, leaf_offsets, nleaves * sizeof(uint32_t));

	for (uint32_t v = 0; v < nvecs; v++)
	{
		const float *vec	= vectors + (size_t)v * dim;
		float		 best_d = FLT_MAX;
		uint32_t	 best_c = 0;
		for (uint32_t c = 0; c < nleaves; c++)
		{
			float d = mkt_l2_distance_squared(
					vec, leaf_cents + (size_t)c * dim, dim);
			if (d < best_d)
			{
				best_d = d;
				best_c = c;
			}
		}
		leaf_idx[pos[best_c]++] = v;
	}

	mkt_free(pos);

	/*
	 * Pre-compute posting page block numbers per leaf cluster.
	 * Posting pages are allocated sequentially after centroid pages.
	 */
	uint32_t	 posting_max		= mkt_posting_max_entries(dim);
	BlockNumber *leaf_posting_heads = mkt_alloc(nleaves * sizeof(BlockNumber));
	uint32_t	 posting_npages		= 0;

	for (uint32_t c = 0; c < nleaves; c++)
	{
		uint32_t count = leaf_offsets[c + 1] - leaf_offsets[c];
		if (count == 0)
		{
			leaf_posting_heads[c] = InvalidBlockNumber;
			continue;
		}
		uint32_t pages		  = (count + posting_max - 1) / posting_max;
		leaf_posting_heads[c] = centroid_npages + posting_npages;
		posting_npages += pages;
	}

	uint32_t total_pages = centroid_npages + posting_npages;

	/* Save a copy of leaf centroids before tree is destroyed */
	float *saved_leaf_cents = mkt_alloc((size_t)nleaves * dim * sizeof(float));
	memcpy(saved_leaf_cents,
		   tree->leaf_centroids,
		   (size_t)nleaves * dim * sizeof(float));

	/*
	 * Allocate combined page buffer (centroid + posting pages).
	 */
	char *pages = mkt_alloc((size_t)total_pages * BLCKSZ);
	memset(pages, 0, (size_t)total_pages * BLCKSZ);

	TestStorage build_storage = {
			.base.ops	= &test_storage_ops,
			.pages		= pages,
			.next_blkno = 0,
	};

	/* Allocate medoid vector storage for centroid reranking */
	float *medoid_vecs = mkt_alloc(
			(size_t)centroid_npages * max_entries * dim * sizeof(float));
	memset(medoid_vecs,
		   0,
		   (size_t)centroid_npages * max_entries * dim * sizeof(float));

	/*
	 * Write centroid pages for each BFS node.
	 *
	 * Two passes per node: (1) find medoids and collect metadata,
	 * (2) create encoder pointing at medoid buffer, call write_pages.
	 */
	float *medoid_buf = mkt_alloc((size_t)fan_out * dim * sizeof(float));

	bool ok = true;

	for (uint32_t i = 0; i < tree->nnodes && ok; i++)
	{
		HKMeansNode *node	 = &tree->nodes[i];
		bool		 is_leaf = (node->level == tree->nlevels - 1);
		uint32_t	 k		 = node->nchildren;

		uint16_t entry_flags	   = is_leaf ? MKT_CENTROID_FLAG_LEAF : 0;
		uint16_t entry_child_count = is_leaf ? 0 : (uint16_t)fan_out;

		BlockNumber		*leaf_children = NULL;
		ItemPointerData *node_tids	   = NULL;

		if (is_leaf)
			leaf_children = mkt_alloc(k * sizeof(BlockNumber));
		if (fmt == MKT_CENTROID_FMT_RABITQ)
			node_tids = mkt_alloc(k * sizeof(ItemPointerData));

		for (uint32_t c = 0; c < k; c++)
		{
			const float *centroid = node->centroids + (size_t)c * dim;
			const float *medoid	  = find_medoid(vectors, nvecs, centroid, dim);

			memcpy(medoid_buf + (size_t)c * dim, medoid, dim * sizeof(float));

			if (is_leaf)
			{
				leaf_children[c] = leaf_posting_heads[node->first_leaf + c];

				if (nsaved < nleaves)
				{
					memcpy(seeds + (size_t)nsaved * dim,
						   medoid,
						   dim * sizeof(float));
					seed_phs[nsaved] = leaf_children[c];
					nsaved++;
				}
			}

			BlockNumber entry_blkno = node_first_blkno[i] + c / max_entries;
			uint16_t	entry_in_pg = (uint16_t)(c % max_entries);

			if (node_tids != NULL)
				ItemPointerSet(
						&node_tids[c],
						entry_blkno,
						(OffsetNumber)(entry_in_pg + 1));

			uint32_t midx = entry_blkno * max_entries + entry_in_pg;
			memcpy(medoid_vecs + (size_t)midx * dim,
				   medoid,
				   dim * sizeof(float));
		}

		CentroidEncoderState enc_state;
		CentroidEncoder		*encoder = centroid_encoder_init(
				&enc_state, fmt, medoid_buf, dim, params, global_centroid_out);

		const BlockNumber *child_blks =
				is_leaf ? leaf_children : &node_first_blkno[node->first_child];

		build_storage.next_blkno = node_first_blkno[i];
		mkt_centroid_write_pages(
				&build_storage.base,
				dim,
				k,
				fmt,
				(uint8_t)node->level,
				entry_flags,
				entry_child_count,
				encoder,
				node_tids,
				child_blks);

		mkt_free(leaf_children);
		mkt_free(node_tids);
	}

	mkt_free(medoid_buf);
	mkt_free(node_first_blkno);
	mkt_hkmeans_result_destroy(tree);

	if (!ok)
	{
		mkt_free(leaf_idx);
		mkt_free(leaf_offsets);
		mkt_free(leaf_posting_heads);
		mkt_free(saved_leaf_cents);
		mkt_free(medoid_vecs);
		mkt_free(seed_phs);
		mkt_free(seeds);
		mkt_free(pages);
		mkt_free(vectors);
		return NULL;
	}

	/*
	 * Encode and write posting pages. Per leaf cluster: stream
	 * vectors through MktPostingBuilder which encodes with IVF
	 * residual and writes pages with O(1) memory per cluster.
	 */
	for (uint32_t c = 0; c < nleaves; c++)
	{
		uint32_t start = leaf_offsets[c];
		uint32_t count = leaf_offsets[c + 1] - start;
		if (count == 0)
			continue;

		MktPostingBuilder pb;
		build_storage.next_blkno = leaf_posting_heads[c];
		mkt_posting_builder_init(
				&pb,
				&build_storage.base,
				params,
				dim,
				c,
				saved_leaf_cents + (size_t)c * dim);

		for (uint32_t i = 0; i < count; i++)
		{
			uint32_t  vi   = leaf_idx[start + i];
			VectorRef vref = {
					.data = vectors + (size_t)vi * dim,
					.dim  = dim,
			};
			mkt_posting_builder_add(&pb, vref, (BlockNumber)vi, 1);
		}

		mkt_posting_builder_finish(&pb);
		mkt_posting_builder_cleanup(&pb);
	}

	mkt_free(leaf_idx);
	mkt_free(leaf_offsets);

	if (verbose)
		printf("  Pages: %u centroid + %u posting = %u total\n",
			   centroid_npages,
			   posting_npages,
			   total_pages);

	*total_pages_out		= total_pages;
	*centroid_pages_out		= centroid_npages;
	*total_centroids_out	= total_centroids;
	*query_seeds_out		= seeds;
	*seed_posting_heads_out = seed_phs;
	*nseeds_out				= nsaved;
	*medoid_vecs_out		= medoid_vecs;
	*vectors_out			= vectors;
	*leaf_centroids_out		= saved_leaf_cents;
	*leaf_posting_heads_out = leaf_posting_heads;
	*nleaves_out			= nleaves;
	*nvecs_out				= nvecs;
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
	char		*pages;
	float		*medoid_vecs;
	BlockNumber *leaf_posting_heads; /* posting head per leaf */
	uint32_t	 nleaves;
	TestStorage	 storage;
	uint32_t	 total_pages;
	uint32_t	 centroid_pages;
	uint32_t	 total_centroids;
	uint32_t	 max_entries;
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
			{"help", no_argument, 0, 'h'},
			{0, 0, 0, 0},
	};

	optind = 1;

	int opt;
	while ((opt = getopt_long(
					ctx->argc,
					ctx->argv,
					"d:l:f:b:p:q:r:h",
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

	if (config.dim == 0 || config.nlevels == 0 || config.fan_out == 0 ||
		config.queries == 0 || config.runs == 0)
	{
		fprintf(stderr, "Error: all numeric parameters must be > 0\n");
		return 1;
	}

	printf("Search benchmark (dim=%u, nlevels=%u, "
		   "fan_out=%u):\n",
		   config.dim,
		   config.nlevels,
		   config.fan_out);

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

	float		*centroid			= mkt_alloc(config.dim * sizeof(float));
	float		*query_seeds		= NULL;
	BlockNumber *seed_posting_heads = NULL;
	uint32_t	 nseeds				= 0;
	float		*all_vectors		= NULL;
	float		*leaf_cents			= NULL;
	uint32_t	 nvecs				= 0;
	TreeState	 trees[NUM_FORMATS];
	bool		 build_ok = true;

	for (int f = 0; f < NUM_FORMATS; f++)
	{
		srand(42); /* same seed -> identical clustering */
		float		*seeds		= NULL;
		BlockNumber *seed_phs	= NULL;
		uint32_t	 ns			= 0;
		float		*vecs_tmp	= NULL;
		float		*lcents_tmp = NULL;
		BlockNumber *lph_tmp	= NULL;
		uint32_t	 nl_tmp		= 0;
		uint32_t	 nv_tmp		= 0;

		trees[f].max_entries =
				mkt_centroid_max_entries_fmt(config.dim, fmts[f]);

		trees[f].pages = build_tree(
				&config,
				params,
				fmts[f],
				&trees[f].total_pages,
				&trees[f].centroid_pages,
				&trees[f].total_centroids,
				centroid,
				&seeds,
				&seed_phs,
				&ns,
				&trees[f].medoid_vecs,
				&vecs_tmp,
				&lcents_tmp,
				&lph_tmp,
				&nl_tmp,
				&nv_tmp,
				f == 0); /* verbose only for first */

		if (trees[f].pages == NULL)
		{
			build_ok = false;
			mkt_free(seed_phs);
			mkt_free(seeds);
			mkt_free(vecs_tmp);
			mkt_free(lcents_tmp);
			mkt_free(lph_tmp);
			break;
		}

		trees[f].leaf_posting_heads = lph_tmp;
		trees[f].nleaves			= nl_tmp;

		/*
		 * Save seeds and leaf data from first build
		 * (identical across formats due to same srand).
		 */
		if (f == 0)
		{
			query_seeds		   = seeds;
			seed_posting_heads = seed_phs;
			nseeds			   = ns;
			all_vectors		   = vecs_tmp;
			leaf_cents		   = lcents_tmp;
			nvecs			   = nv_tmp;
		}
		else
		{
			mkt_free(seed_phs);
			mkt_free(seeds);
			mkt_free(vecs_tmp);
			mkt_free(lcents_tmp);
		}

		trees[f].storage = (TestStorage){
				.base.ops	 = &test_storage_ops,
				.pages		 = trees[f].pages,
				.medoid_vecs = trees[f].medoid_vecs,
				.max_entries = trees[f].max_entries,
				.dim		 = config.dim,
		};
	}

	if (!build_ok)
	{
		for (int f = 0; f < NUM_FORMATS; f++)
		{
			mkt_free(trees[f].leaf_posting_heads);
			mkt_free(trees[f].medoid_vecs);
			mkt_free(trees[f].pages);
		}
		mkt_free(leaf_cents);
		mkt_free(all_vectors);
		mkt_free(seed_posting_heads);
		mkt_free(query_seeds);
		mkt_free(centroid);
		mkt_rabitq_destroy(params);
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
	srand(12345);
	uint32_t nq		 = config.queries;
	uint32_t nprobe	 = config.nprobe;
	float	*queries = mkt_alloc((size_t)nq * config.dim * sizeof(float));

	for (uint32_t q = 0; q < nq; q++)
	{
		float *qvec = queries + (size_t)q * config.dim;
		if (nseeds > 0)
		{
			Dimension	 subdim	  = config.dim < 32 ? config.dim : 32;
			uint32_t	 seed_idx = (uint32_t)rand() % nseeds;
			const float *seed = query_seeds + (size_t)seed_idx * config.dim;
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

	/*
	 * Ground truth: brute-force top-K nearest neighbors over
	 * all vectors. gt[q * topk + i] = global vector index of
	 * the i-th nearest neighbor for query q.
	 */
	uint32_t  topk = DEFAULT_TOPK;
	uint32_t *gt   = mkt_alloc((size_t)nq * topk * sizeof(uint32_t));

	for (uint32_t q = 0; q < nq; q++)
	{
		const float *qvec = queries + (size_t)q * config.dim;

		MktTopK tk;
		mkt_topk_init(&tk, topk);

		for (uint32_t v = 0; v < nvecs; v++)
		{
			float d2 = mkt_l2_distance_squared(
					qvec, all_vectors + (size_t)v * config.dim, config.dim);
			mkt_topk_insert(&tk, d2, 0.0f, v);
		}

		MktTopKEntry *entries = mkt_alloc(
				tk.cand_count * sizeof(MktTopKEntry));
		uint32_t count;
		mkt_topk_extract_sorted(&tk, entries, &count);

		uint32_t gt_k = count < topk ? count : topk;
		for (uint32_t i = 0; i < gt_k; i++)
			gt[q * topk + i] = (uint32_t)entries[i].id;
		/* Pad remaining slots (if count < topk) */
		for (uint32_t i = gt_k; i < topk; i++)
			gt[q * topk + i] = UINT32_MAX;

		mkt_free(entries);
		mkt_topk_cleanup(&tk);
	}

	/* Benchmark each variant */
	uint32_t   total_samples = nq * config.runs;
	BenchStats var_stats[NUM_VARIANTS];
	double	   recall_sum[NUM_VARIANTS];

	for (int v = 0; v < NUM_VARIANTS; v++)
	{
		bench_stats_init(&var_stats[v], total_samples);
		recall_sum[v] = 0.0;
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
					.qstate		 = qs,
					.query		 = qvec,
					.query_datum = PointerGetDatum(qvec),
					.storage	 = &ts->storage.base,
					.beam_width	 = config.beam_width,
					.nprobe		 = nprobe,
					.dim		 = config.dim,
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
			 * End-to-end recall: posting scan with two-stage
			 * RaBitQ filtering + rerank, compare to ground truth.
			 */
			MktTopK tk;
			mkt_topk_init(&tk, topk);
			Distance topk_thresh = INFINITY;

			MktPostingScan pscan;
			mkt_posting_scan_init(
					&pscan, &ts->storage.base, params, config.dim);
			mkt_posting_scan_set_threshold(&pscan, &topk_thresh);

			for (uint32_t j = 0; j < n_results; j++)
			{
				BlockNumber ph = results[j].posting_head;

				/* Find leaf index for this posting head */
				uint32_t li = 0;
				for (uint32_t c = 0; c < ts->nleaves; c++)
				{
					if (ts->leaf_posting_heads[c] == ph)
					{
						li = c;
						break;
					}
				}

				VectorRef cref = {
						.data = leaf_cents + (size_t)li * config.dim,
						.dim  = config.dim,
				};
				mkt_posting_scan_begin_cluster(&pscan, query_ref, cref, ph);

				MktPostingScanResult pr;
				while (mkt_posting_scan_next(&pscan, &pr))
				{
					uint64_t id = encode_tid(
							ItemPointerGetBlockNumber(&pr.tid),
							ItemPointerGetOffsetNumber(&pr.tid));
					mkt_topk_insert(&tk, pr.distance, pr.error, id);
					topk_thresh = mkt_topk_threshold(&tk);
				}
				mkt_posting_scan_end_cluster(&pscan);
			}
			mkt_posting_scan_cleanup(&pscan);

			/* Extract candidates for reranking */
			uint32_t	  cand_max = tk.cand_count;
			MktTopKEntry *cands	   = mkt_alloc(
					   (cand_max > 0 ? cand_max : 1) * sizeof(MktTopKEntry));
			uint32_t cand_count;
			mkt_topk_extract_sorted(&tk, cands, &cand_count);

			/*
			 * Rerank: candidates are sorted by estimated
			 * distance. Compute exact L2 for those with
			 * error > 0, collect top-K by exact distance.
			 */
			MktTopK rerank_tk;
			mkt_topk_init(&rerank_tk, topk);

			for (uint32_t i = 0; i < cand_count; i++)
			{
				BlockNumber blk = (BlockNumber)(cands[i].id >> 16);
				Distance	d;
				if (cands[i].error > 0.0f)
				{
					const float *vec = all_vectors + (size_t)blk * config.dim;
					d = mkt_l2_distance_squared(qvec, vec, config.dim);
				}
				else
				{
					d = cands[i].distance;
				}
				mkt_topk_insert(&rerank_tk, d, 0.0f, (uint64_t)blk);
			}

			uint32_t	  rr_max = rerank_tk.cand_count;
			MktTopKEntry *rr	 = mkt_alloc(
					(rr_max > 0 ? rr_max : 1) * sizeof(MktTopKEntry));
			uint32_t rr_count;
			mkt_topk_extract_sorted(&rerank_tk, rr, &rr_count);

			uint32_t hits = 0;
			for (uint32_t i = 0; i < rr_count && i < topk; i++)
			{
				uint32_t vid = (uint32_t)rr[i].id;
				for (uint32_t g = 0; g < topk; g++)
				{
					if (gt[q * topk + g] == vid)
					{
						hits++;
						break;
					}
				}
			}
			recall_sum[v] += (double)hits / topk;

			mkt_free(rr);
			mkt_topk_cleanup(&rerank_tk);
			mkt_free(cands);
			mkt_topk_cleanup(&tk);

			if (qs)
				mkt_rabitq_free_query(qs);
		}
	}

	/* Print results */
	char recall_hdr[32];
	snprintf(recall_hdr, sizeof(recall_hdr), "R@%u", topk);

	printf("  %-14s %9s %9s %9s %9s %9s\n",
		   "Variant",
		   "Avg (us)",
		   "Min (us)",
		   "Max (us)",
		   "Stddev",
		   recall_hdr);

	for (int v = 0; v < NUM_VARIANTS; v++)
	{
		bench_stats_compute(&var_stats[v]);
		double recall_pct = nq > 0 ? 100.0 * recall_sum[v] / nq : 0.0;

		printf("  %-14s %9.2f %9.2f %9.2f %9.2f %8.1f%%\n",
			   variants[v].name,
			   var_stats[v].avg,
			   var_stats[v].min,
			   var_stats[v].max,
			   var_stats[v].stddev,
			   recall_pct);
	}

	/* Print routing distance stats */
	bool has_rerank = false;
	for (int v = 0; v < NUM_VARIANTS; v++)
		has_rerank |= (var_stats_search[v].reranked > 0);

	if (has_rerank)
	{
		printf("\n  Routing distance computations"
			   " (avg per query):\n");
		printf("  %-14s %8s %8s %8s\n",
			   "Variant",
			   "Scored",
			   "Rerank",
			   "Total");
		for (int v = 0; v < NUM_VARIANTS; v++)
		{
			MktCentroidSearchStats *ss		   = &var_stats_search[v];
			double					avg_dist   = (double)ss->dist_calcs / nq;
			double					avg_rerank = (double)ss->reranked / nq;
			printf("  %-14s %8.1f %8.1f %8.1f\n",
				   variants[v].name,
				   avg_dist,
				   avg_rerank,
				   avg_dist + avg_rerank);
		}
	}

	/* Cleanup */
	for (int v = 0; v < NUM_VARIANTS; v++)
		bench_stats_free(&var_stats[v]);
	mkt_free(results);
	mkt_free(gt);
	mkt_free(queries);
	mkt_free(leaf_cents);
	mkt_free(all_vectors);
	mkt_free(seed_posting_heads);
	mkt_free(query_seeds);

	for (int f = 0; f < NUM_FORMATS; f++)
	{
		free(trees[f].storage.rerank_lb);
		free(trees[f].storage.rerank_exact);
		mkt_free(trees[f].leaf_posting_heads);
		mkt_free(trees[f].medoid_vecs);
		mkt_free(trees[f].pages);
	}

	mkt_free(centroid);
	mkt_rabitq_destroy(params);

	return 0;
}
