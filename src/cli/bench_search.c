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

#include "algo/kmeans.h"
#include "algo/topk.h"
#include "algo/vecops.h"
#include "cmd.h"
#include "core/memory.h"
#include "index/centroid_build.h"
#include "index/centroid_page.h"
#include "index/centroid_search.h"
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
 * 2. Run k-means (K = fan_out) to get level-0 centroids
 * 3. For each cluster, run k-means again for level-1, etc.
 * 4. Build centroid pages from real cluster structure
 *
 * Block numbers assigned sequentially with overflow pages:
 * each tree node gets ceil(fan_out / max_entries) pages.
 * ---------------------------------------------------------------- */

/* Vectors per leaf centroid for k-means training */
#define VECS_PER_LEAF	  20
#define POSTING_HEAD_BASE 1000000
#define DEFAULT_TOPK	  10

static uint32_t
power_u32(uint32_t base, uint32_t exp)
{
	uint32_t result = 1;
	for (uint32_t i = 0; i < exp; i++)
		result *= base;
	return result;
}

static BlockNumber
level_start_blkno(uint32_t level, uint32_t fan_out, uint32_t pages_per_node)
{
	BlockNumber start = 0;
	for (uint32_t l = 0; l < level; l++)
		start += power_u32(fan_out, l) * pages_per_node;
	return start;
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
 * Scans vectors assigned to cluster c and returns a pointer to
 * the one with minimum squared L2 distance to the centroid.
 * Falls back to the centroid itself if no vectors are assigned.
 */
static const float *
find_medoid(
		const float	   *vecs,
		const uint32_t *assignments,
		uint32_t		count,
		uint32_t		c,
		const float	   *centroid,
		Dimension		dim)
{
	const float *best	   = centroid;
	float		 best_dist = FLT_MAX;

	for (uint32_t v = 0; v < count; v++)
	{
		if (assignments[v] != c)
			continue;

		const float *vec  = vecs + (size_t)v * dim;
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
 * build_tree - Build centroid tree via hierarchical k-means
 *
 * Generates nvecs random vectors, clusters them hierarchically,
 * and builds centroid pages in the specified format. All RaBitQ
 * entries are encoded relative to the global centroid.
 *
 * Supports overflow pages: when fan_out > max entries per page,
 * nodes chain multiple pages via next_blkno.
 *
 * Returns the allocated page buffer, or NULL on failure.
 * ---------------------------------------------------------------- */
static char *
build_tree(
		const BenchConfig  *config,
		const RaBitQParams *params,
		uint8_t				fmt,
		uint32_t		   *total_pages_out,
		uint32_t		   *total_centroids_out,
		float			   *global_centroid_out,
		float			  **query_seeds_out,
		BlockNumber		  **seed_posting_heads_out,
		uint32_t		   *nseeds_out,
		float			  **medoid_vecs_out,
		float			  **vectors_out,
		uint32_t		  **leaf_idx_out,
		uint32_t		  **leaf_offsets_out,
		uint32_t		   *nvecs_out,
		bool				verbose)
{
	Dimension dim	  = config->dim;
	uint32_t  fan_out = config->fan_out;
	uint32_t  nlevels = config->nlevels;

	uint32_t max_entries	= mkt_centroid_max_entries_fmt(dim, fmt);
	uint32_t pages_per_node = (fan_out + max_entries - 1) / max_entries;
	if (pages_per_node == 0)
		pages_per_node = 1;

	/* Count total pages and centroids */
	uint32_t total_pages	 = 0;
	uint32_t total_centroids = 0;
	for (uint32_t l = 0; l < nlevels; l++)
	{
		uint32_t nodes = power_u32(fan_out, l);
		total_pages += nodes * pages_per_node;
		total_centroids += nodes * fan_out;
	}

	uint32_t leaf_centroids = power_u32(fan_out, nlevels);

	/* Query seeds: leaf medoids + posting heads */
	float *seeds = mkt_alloc((size_t)leaf_centroids * dim * sizeof(float));
	BlockNumber *seed_phs = mkt_alloc(leaf_centroids * sizeof(BlockNumber));
	uint32_t	 nsaved	  = 0;

	/*
	 * Generate hierarchically-clustered vectors.
	 */
	uint32_t nvecs	 = 0;
	float	*vectors = generate_hierarchical_vectors(
			  dim, fan_out, nlevels, VECS_PER_LEAF, &nvecs);
	if (verbose)
		printf("  Generated %u vectors (%u leaf clusters x %u "
			   "vecs)\n",
			   nvecs,
			   leaf_centroids,
			   VECS_PER_LEAF);

	/* Compute global centroid (mean of all vectors) */
	memset(global_centroid_out, 0, dim * sizeof(float));
	for (uint32_t i = 0; i < nvecs; i++)
	{
		const float *v = vectors + (size_t)i * dim;
		for (Dimension d = 0; d < dim; d++)
			global_centroid_out[d] += v[d];
	}
	for (Dimension d = 0; d < dim; d++)
		global_centroid_out[d] /= (float)nvecs;

	KMeansOptions km_opts = MKT_KMEANS_OPTIONS_DEFAULT;

	/* Allocate page buffer */
	char *pages = mkt_alloc((size_t)total_pages * BLCKSZ);
	memset(pages, 0, (size_t)total_pages * BLCKSZ);

	/* Build-time storage for mkt_centroid_write_pages */
	TestStorage build_storage = {
			.base.ops	= &test_storage_ops,
			.pages		= pages,
			.next_blkno = 0,
	};

	/* Allocate medoid vector storage for reranking */
	float *medoid_vecs = mkt_alloc(
			(size_t)total_pages * max_entries * dim * sizeof(float));
	memset(medoid_vecs,
		   0,
		   (size_t)total_pages * max_entries * dim * sizeof(float));

	/*
	 * Leaf index arrays: map each leaf cluster to global vector
	 * indices. leaf_offsets is a prefix-sum array of length
	 * leaf_centroids+1, and leaf_idx holds nvecs global indices.
	 *
	 * Built in two phases: count during tree build, fill after.
	 */
	uint32_t *leaf_offsets = mkt_alloc(
			(leaf_centroids + 1) * sizeof(uint32_t));
	uint32_t *leaf_idx = mkt_alloc(nvecs * sizeof(uint32_t));
	memset(leaf_offsets, 0, (leaf_centroids + 1) * sizeof(uint32_t));

	/* Saved leaf work items for second pass (index filling) */
	typedef struct LeafItem
	{
		uint32_t *indices;	   /* global vector indices */
		uint32_t *assignments; /* k-means cluster assignments */
		uint32_t  count;
		uint32_t  idx_in_level;
	} LeafItem;

	uint32_t  leaf_item_cap	  = power_u32(fan_out, nlevels - 1);
	LeafItem *leaf_items	  = mkt_alloc(leaf_item_cap * sizeof(LeafItem));
	uint32_t  leaf_item_count = 0;

	/* Work queue entry: a subset of vectors to cluster */
	typedef struct WorkItem
	{
		float	   *vecs;	 /* [count * dim] (owned if level>0) */
		uint32_t   *indices; /* global vector indices (owned if level>0) */
		uint32_t	count;
		BlockNumber first_blkno; /* first page of this node */
		uint32_t	level;
		uint32_t	idx_in_level;
	} WorkItem;

	uint32_t  max_queue = total_pages;
	WorkItem *queue		= mkt_alloc(max_queue * sizeof(WorkItem));
	uint32_t  q_head = 0, q_tail = 0;

	/* Root indices: [0, 1, ..., nvecs-1] */
	uint32_t *root_indices = mkt_alloc(nvecs * sizeof(uint32_t));
	for (uint32_t i = 0; i < nvecs; i++)
		root_indices[i] = i;

	/* Seed: the full vector pool at level 0 */
	queue[q_tail++] = (WorkItem){
			.vecs		  = vectors,
			.indices	  = root_indices,
			.count		  = nvecs,
			.first_blkno  = 0,
			.level		  = 0,
			.idx_in_level = 0,
	};

	bool ok = true;

	while (q_head < q_tail && ok)
	{
		WorkItem item = queue[q_head++];

		bool is_leaf = (item.level == nlevels - 1);

		uint32_t k = fan_out;
		if (item.count < k)
			k = item.count;

		if (item.level == 0 && verbose)
		{
			printf("  Clustering %u vectors -> %u "
				   "level-0 centroids\n",
				   item.count,
				   k);
		}

		KMeansResult *km = mkt_kmeans_f32(
				item.vecs, item.count, dim, k, DISTANCE_L2, &km_opts);
		if (km == NULL)
		{
			fprintf(stderr, "Error: k-means failed at level %u\n", item.level);
			ok = false;
			break;
		}

		BlockNumber next_start =
				level_start_blkno(item.level + 1, fan_out, pages_per_node);

		/*
		 * At leaf level, count vectors per cluster and save
		 * assignments for the fill pass after the main loop.
		 */
		if (is_leaf)
		{
			for (uint32_t v = 0; v < item.count; v++)
			{
				uint32_t c	= km->assignments[v];
				uint32_t li = item.idx_in_level * fan_out + c;
				leaf_offsets[li + 1]++;
			}

			/* Save for second pass */
			uint32_t *saved_asgn = mkt_alloc(item.count * sizeof(uint32_t));
			memcpy(saved_asgn, km->assignments, item.count * sizeof(uint32_t));

			uint32_t *saved_idx = mkt_alloc(item.count * sizeof(uint32_t));
			memcpy(saved_idx, item.indices, item.count * sizeof(uint32_t));

			leaf_items[leaf_item_count++] = (LeafItem){
					.indices	  = saved_idx,
					.assignments  = saved_asgn,
					.count		  = item.count,
					.idx_in_level = item.idx_in_level,
			};
		}

		/*
		 * Pre-encode entries and build per-entry arrays for
		 * mkt_centroid_write_pages.
		 */
		const void	   **node_data	   = mkt_alloc(k * sizeof(void *));
		BlockNumber		*node_children = mkt_alloc(k * sizeof(BlockNumber));
		ItemPointerData *node_tids	   = NULL;
		if (fmt == MKT_CENTROID_FMT_RABITQ)
			node_tids = mkt_alloc(k * sizeof(ItemPointerData));

		uint16_t entry_flags	   = is_leaf ? MKT_CENTROID_FLAG_LEAF : 0;
		uint16_t entry_child_count = is_leaf ? 0 : (uint16_t)fan_out;

		for (uint32_t c = 0; c < k; c++)
		{
			const float *centroid = km->centroids + (size_t)c * dim;
			const float *medoid	  = find_medoid(
					  item.vecs, km->assignments, item.count, c, centroid, dim);

			/* Per-entry child block number */
			if (is_leaf)
			{
				node_children[c] = POSTING_HEAD_BASE +
								   item.idx_in_level * fan_out + c;

				if (nsaved < leaf_centroids)
				{
					memcpy(seeds + (size_t)nsaved * dim,
						   medoid,
						   dim * sizeof(float));
					seed_phs[nsaved] = node_children[c];
					nsaved++;
				}
			}
			else
			{
				uint32_t child_idx = item.idx_in_level * fan_out + c;
				node_children[c]   = next_start + child_idx * pages_per_node;
			}

			/* Encode data for write_pages */
			switch (fmt)
			{
			case MKT_CENTROID_FMT_RABITQ:
			{
				VectorRef vec_ref  = {.data = medoid, .dim = dim};
				VectorRef cent_ref = {.data = global_centroid_out, .dim = dim};
				node_data[c] = mkt_rabitq_encode(params, vec_ref, cent_ref);
				if (node_data[c] == NULL)
				{
					fprintf(stderr,
							"Error: RaBitQ encode failed "
							"at level %u entry %u\n",
							item.level,
							c);
					ok = false;
				}
				break;
			}
			case MKT_CENTROID_FMT_FLOAT:
				node_data[c] = medoid;
				break;
			case MKT_CENTROID_FMT_HALF:
			{
				half *hvec = mkt_alloc(dim * sizeof(half));
				mkt_float_to_half_array(medoid, hvec, dim);
				node_data[c] = hvec;
				break;
			}
			}

			if (!ok)
				break;

			/* Compute TID for reranking lookup */
			BlockNumber entry_blkno = item.first_blkno + c / max_entries;
			uint16_t	entry_in_pg = (uint16_t)(c % max_entries);

			if (node_tids != NULL)
				ItemPointerSet(
						&node_tids[c],
						entry_blkno,
						(OffsetNumber)(entry_in_pg + 1));

			/* Save medoid for reranking */
			uint32_t midx = entry_blkno * max_entries + entry_in_pg;
			memcpy(medoid_vecs + (size_t)midx * dim,
				   medoid,
				   dim * sizeof(float));

			/* Enqueue children (non-leaf) */
			if (!is_leaf)
			{
				uint32_t sub_n = 0;
				for (uint32_t v = 0; v < item.count; v++)
				{
					if (km->assignments[v] == c)
						sub_n++;
				}
				if (sub_n > 0)
				{
					float *sub_vecs = mkt_alloc(
							(size_t)sub_n * dim * sizeof(float));
					uint32_t *sub_idx = mkt_alloc(sub_n * sizeof(uint32_t));

					uint32_t idx = 0;
					for (uint32_t v = 0; v < item.count; v++)
					{
						if (km->assignments[v] == c)
						{
							memcpy(sub_vecs + (size_t)idx * dim,
								   item.vecs + (size_t)v * dim,
								   dim * sizeof(float));
							sub_idx[idx] = item.indices[v];
							idx++;
						}
					}

					uint32_t	child_il	= item.idx_in_level * fan_out + c;
					BlockNumber child_first = next_start +
											  child_il * pages_per_node;

					queue[q_tail++] = (WorkItem){
							.vecs		  = sub_vecs,
							.indices	  = sub_idx,
							.count		  = sub_n,
							.first_blkno  = child_first,
							.level		  = item.level + 1,
							.idx_in_level = child_il,
					};
				}
			}
		}

		/* Write pages using production code path */
		if (ok)
		{
			build_storage.next_blkno = item.first_blkno;
			mkt_centroid_write_pages(
					&build_storage.base,
					dim,
					k,
					fmt,
					(uint8_t)item.level,
					entry_flags,
					entry_child_count,
					node_data,
					node_tids,
					node_children);
		}

		/* Free encoded data */
		for (uint32_t c = 0; c < k; c++)
		{
			if (fmt == MKT_CENTROID_FMT_RABITQ || fmt == MKT_CENTROID_FMT_HALF)
			{
				if (node_data[c] != NULL)
					mkt_free((void *)node_data[c]);
			}
		}
		mkt_free(node_data);
		mkt_free(node_children);
		mkt_free(node_tids);

		mkt_kmeans_result_destroy(km);

		if (item.level > 0)
		{
			mkt_free(item.vecs);
			mkt_free(item.indices);
		}
	}

	/* Free any unprocessed queue items */
	for (uint32_t i = q_head; i < q_tail; i++)
	{
		if (queue[i].level > 0)
		{
			mkt_free(queue[i].vecs);
			mkt_free(queue[i].indices);
		}
	}
	mkt_free(queue);
	mkt_free(root_indices);

	/*
	 * Build leaf_idx: convert histogram in leaf_offsets to a
	 * prefix sum, then fill leaf_idx using saved leaf items.
	 */
	for (uint32_t i = 1; i <= leaf_centroids; i++)
		leaf_offsets[i] += leaf_offsets[i - 1];

	/* Write positions (copy of offsets, advanced during fill) */
	uint32_t *pos = mkt_alloc(leaf_centroids * sizeof(uint32_t));
	memcpy(pos, leaf_offsets, leaf_centroids * sizeof(uint32_t));

	for (uint32_t li = 0; li < leaf_item_count; li++)
	{
		LeafItem *lf = &leaf_items[li];
		for (uint32_t v = 0; v < lf->count; v++)
		{
			uint32_t c			  = lf->assignments[v];
			uint32_t cidx		  = lf->idx_in_level * fan_out + c;
			leaf_idx[pos[cidx]++] = lf->indices[v];
		}
		mkt_free(lf->indices);
		mkt_free(lf->assignments);
	}
	mkt_free(pos);
	mkt_free(leaf_items);

	if (!ok)
	{
		mkt_free(leaf_idx);
		mkt_free(leaf_offsets);
		mkt_free(medoid_vecs);
		mkt_free(seed_phs);
		mkt_free(seeds);
		mkt_free(pages);
		mkt_free(vectors);
		return NULL;
	}

	*total_pages_out		= total_pages;
	*total_centroids_out	= total_centroids;
	*query_seeds_out		= seeds;
	*seed_posting_heads_out = seed_phs;
	*nseeds_out				= nsaved;
	*medoid_vecs_out		= medoid_vecs;
	*vectors_out			= vectors;
	*leaf_idx_out			= leaf_idx;
	*leaf_offsets_out		= leaf_offsets;
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
	char	   *pages;
	float	   *medoid_vecs;
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
	uint32_t	*leaf_idx			= NULL;
	uint32_t	*leaf_offsets		= NULL;
	uint32_t	 nvecs				= 0;
	TreeState	 trees[NUM_FORMATS];
	bool		 build_ok = true;

	for (int f = 0; f < NUM_FORMATS; f++)
	{
		srand(42); /* same seed → identical clustering */
		float		*seeds	  = NULL;
		BlockNumber *seed_phs = NULL;
		uint32_t	 ns		  = 0;
		float		*vecs_tmp = NULL;
		uint32_t	*lidx_tmp = NULL;
		uint32_t	*loff_tmp = NULL;
		uint32_t	 nv_tmp	  = 0;

		trees[f].max_entries =
				mkt_centroid_max_entries_fmt(config.dim, fmts[f]);

		trees[f].pages = build_tree(
				&config,
				params,
				fmts[f],
				&trees[f].total_pages,
				&trees[f].total_centroids,
				centroid,
				&seeds,
				&seed_phs,
				&ns,
				&trees[f].medoid_vecs,
				&vecs_tmp,
				&lidx_tmp,
				&loff_tmp,
				&nv_tmp,
				f == 0); /* verbose only for first */

		if (trees[f].pages == NULL)
		{
			build_ok = false;
			mkt_free(seed_phs);
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
			query_seeds		   = seeds;
			seed_posting_heads = seed_phs;
			nseeds			   = ns;
			all_vectors		   = vecs_tmp;
			leaf_idx		   = lidx_tmp;
			leaf_offsets	   = loff_tmp;
			nvecs			   = nv_tmp;
		}
		else
		{
			mkt_free(seed_phs);
			mkt_free(seeds);
			mkt_free(vecs_tmp);
			mkt_free(lidx_tmp);
			mkt_free(loff_tmp);
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
			mkt_free(trees[f].medoid_vecs);
			mkt_free(trees[f].pages);
		}
		mkt_free(leaf_offsets);
		mkt_free(leaf_idx);
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
						&state, 0, (uint8_t)config.nlevels, results, NULL);

			/* Timed runs */
			for (uint32_t r = 0; r < config.runs; r++)
			{
				uint64_t start = get_time_ns();
				mkt_centroid_beam_search(
						&state, 0, (uint8_t)config.nlevels, results, NULL);
				uint64_t end = get_time_ns();
				bench_stats_add(&var_stats[v], (double)(end - start) / 1000.0);
			}

			/* Quality run (accumulate search stats) */
			uint32_t n_results = mkt_centroid_beam_search(
					&state,
					0,
					(uint8_t)config.nlevels,
					results,
					&var_stats_search[v]);

			/*
			 * End-to-end recall: scan vectors in selected
			 * clusters, find top-K, compare to ground truth.
			 */
			MktTopK scan;
			mkt_topk_init(&scan, topk);

			for (uint32_t j = 0; j < n_results; j++)
			{
				uint32_t li = results[j].posting_head - POSTING_HEAD_BASE;
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
	mkt_free(leaf_offsets);
	mkt_free(leaf_idx);
	mkt_free(all_vectors);
	mkt_free(seed_posting_heads);
	mkt_free(query_seeds);

	for (int f = 0; f < NUM_FORMATS; f++)
	{
		free(trees[f].storage.rerank_lb);
		free(trees[f].storage.rerank_exact);
		mkt_free(trees[f].medoid_vecs);
		mkt_free(trees[f].pages);
	}

	mkt_free(centroid);
	mkt_rabitq_destroy(params);

	return 0;
}
