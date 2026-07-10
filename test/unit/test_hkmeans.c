/*
 * test_hkmeans.c - Tests for hierarchical k-means tree builder
 */

#include <math.h>
#include <string.h>

#include "algo/hkmeans.h"
#include "algo/vecops.h"
#include "core/memory.h"
#include "mkt_test.h"

TEST_GROUP(HKMeans);
TEST_MEMCTX_FIXTURE();

/*
 * Helper: generate clustered data with well-separated groups.
 *
 * Each cluster has a large value in dimension (c % dim) = 10.0,
 * with small random noise. Returns [nclusters * per_cluster * dim].
 */
static float *
make_clustered_data(
		uint32_t nclusters, uint32_t per_cluster, Dimension dim, uint64_t seed)
{
	uint32_t nvecs = nclusters * per_cluster;
	float	*data  = mkt_alloc((size_t)nvecs * dim * sizeof(float));

	uint32_t rng = (uint32_t)seed;
	for (uint32_t c = 0; c < nclusters; c++)
	{
		for (uint32_t i = 0; i < per_cluster; i++)
		{
			uint32_t idx = c * per_cluster + i;
			for (Dimension d = 0; d < dim; d++)
			{
				rng			 = rng * 1103515245 + 12345;
				float noise	 = ((float)(rng % 1000) / 500.0f - 1.0f) * 0.1f;
				float center = (d == c % dim) ? 10.0f : 0.0f;
				data[(size_t)idx * dim + d] = center + noise;
			}
		}
	}

	return data;
}

/*
 * Flat tree: nlist <= fan_out gives nlevels=1.
 */
TEST(flat_single_level)
{
	uint32_t  nlist	  = 4;
	uint32_t  fan_out = 8;
	uint32_t  nvecs	  = 200;
	Dimension dim	  = 16;
	float	 *data	  = make_clustered_data(4, 50, dim, 42);

	KMeansOptions opts = MKT_KMEANS_OPTIONS_DEFAULT;

	HKMeansResult *tree = mkt_hkmeans_f32(
			data, nvecs, NULL, dim, nlist, fan_out, DISTANCE_L2, &opts);

	ASSERT_NOT_NULL(tree, "should return result");
	ASSERT_EQ(1, tree->nlevels, "flat tree: 1 level");
	ASSERT_EQ(1, tree->nnodes, "flat tree: 1 node");
	ASSERT_EQ(nlist, tree->nleaves, "nleaves = nlist");
	ASSERT_EQ(nlist, hk_nodes(tree)[0].nchildren, "root has nlist children");
	ASSERT_EQ(0, hk_nodes(tree)[0].level, "root at level 0");
	ASSERT_EQ(
			HKMEANS_NO_CHILD,
			hk_nodes(tree)[0].first_child,
			"leaf: no children");

	/* Leaf centroids should be valid */
	ASSERT_NOT_NULL(hk_leaf_centroids(tree), "leaf centroids allocated");

	for (uint32_t i = 0; i < nlist * dim; i++)
		ASSERT_TRUE(
				isfinite(hk_leaf_centroids(tree)[i]),
				"centroids should be finite");

	mkt_free(tree);
	mkt_free(data);
}

/*
 * Two-level tree: nlist > fan_out.
 */
TEST(two_level_tree)
{
	uint32_t  fan_out = 4;
	uint32_t  nlist	  = 16;
	uint32_t  nvecs	  = 16 * 62;
	Dimension dim	  = 16;
	float	 *data	  = make_clustered_data(16, 62, dim, 42);

	KMeansOptions opts = MKT_KMEANS_OPTIONS_DEFAULT;

	HKMeansResult *tree = mkt_hkmeans_f32(
			data, nvecs, NULL, dim, nlist, fan_out, DISTANCE_L2, &opts);

	ASSERT_NOT_NULL(tree, "should return result");
	ASSERT_EQ(2, tree->nlevels, "2-level tree");
	ASSERT_TRUE(tree->nnodes > 1, "multiple nodes");

	/* Root should be at level 0 */
	ASSERT_EQ(0, hk_nodes(tree)[0].level, "root at level 0");
	ASSERT_TRUE(
			hk_nodes(tree)[0].nchildren <= fan_out,
			"root nchildren <= fan_out");
	ASSERT_TRUE(
			hk_nodes(tree)[0].nchildren >= 2, "root has at least 2 children");

	/* Root should have first_child set */
	ASSERT_TRUE(
			hk_nodes(tree)[0].first_child != HKMEANS_NO_CHILD,
			"root has children");

	/* All non-root nodes should be at level 1 (leaf) */
	for (uint32_t i = 1; i < tree->nnodes; i++)
	{
		ASSERT_EQ(1, hk_nodes(tree)[i].level, "child at level 1");
		ASSERT_EQ(
				HKMEANS_NO_CHILD,
				hk_nodes(tree)[i].first_child,
				"leaf has no children");
	}

	/* Total leaf centroids should be reasonable */
	ASSERT_TRUE(tree->nleaves > 0, "has leaf centroids");
	ASSERT_TRUE(tree->nleaves <= fan_out * fan_out, "nleaves <= fan_out^2");

	/* Leaf centroids should be valid */
	ASSERT_NOT_NULL(hk_leaf_centroids(tree), "leaf centroids allocated");
	for (uint32_t i = 0; i < tree->nleaves * dim; i++)
		ASSERT_TRUE(
				isfinite(hk_leaf_centroids(tree)[i]), "leaf centroids finite");

	mkt_free(tree);
	mkt_free(data);
}

/*
 * Three-level tree: nlist > fan_out^2.
 */
TEST(three_level_tree)
{
	uint32_t  fan_out = 4;
	uint32_t  nlist	  = 64;
	Dimension dim	  = 8;
	uint32_t  nvecs	  = 8 * 80;
	float	 *data	  = make_clustered_data(8, 80, dim, 42);

	KMeansOptions opts = MKT_KMEANS_OPTIONS_DEFAULT;

	HKMeansResult *tree = mkt_hkmeans_f32(
			data, nvecs, NULL, dim, nlist, fan_out, DISTANCE_L2, &opts);

	ASSERT_NOT_NULL(tree, "should return result");
	ASSERT_EQ(3, tree->nlevels, "3-level tree");

	/* Verify levels */
	uint32_t level_counts[4] = {0};
	for (uint32_t i = 0; i < tree->nnodes; i++)
	{
		ASSERT_TRUE(hk_nodes(tree)[i].level < 3, "level in range");
		level_counts[hk_nodes(tree)[i].level]++;
	}
	ASSERT_EQ(1, level_counts[0], "1 root node");
	ASSERT_TRUE(level_counts[1] > 0, "has level-1 nodes");
	ASSERT_TRUE(level_counts[2] > 0, "has level-2 nodes");

	/* first_child set for internal, not for leaves */
	for (uint32_t i = 0; i < tree->nnodes; i++)
	{
		if (hk_nodes(tree)[i].level < 2)
			ASSERT_TRUE(
					hk_nodes(tree)[i].first_child != HKMEANS_NO_CHILD,
					"internal nodes have children");
		else
			ASSERT_EQ(
					HKMEANS_NO_CHILD,
					hk_nodes(tree)[i].first_child,
					"leaf nodes have no children");
	}

	ASSERT_TRUE(tree->nleaves > 0, "has leaf centroids");

	mkt_free(tree);
	mkt_free(data);
}

/*
 * build_flat assembles a one-level tree directly from leaf centroids, matching
 * the shape mkt_hkmeans_f32 produces for a flat (nlist <= fan_out) build.
 */
TEST(build_flat_from_centroids)
{
	Dimension dim	  = 8;
	uint32_t  nleaves = 5;
	uint32_t  fan_out = 16;

	/* Distinct centroids: row i has value (i+1) in dim 0. */
	float *cents = mkt_alloc((size_t)nleaves * dim * sizeof(float));
	memset(cents, 0, (size_t)nleaves * dim * sizeof(float));
	for (uint32_t i = 0; i < nleaves; i++)
		cents[(size_t)i * dim] = (float)(i + 1);

	HKMeansResult *tree = mkt_hkmeans_build_flat(cents, nleaves, fan_out, dim);

	ASSERT_NOT_NULL(tree, "build_flat returns a tree");
	ASSERT_EQ(1, tree->nlevels, "flat: 1 level");
	ASSERT_EQ(1, tree->nnodes, "flat: 1 node");
	ASSERT_EQ(nleaves, tree->nleaves, "nleaves preserved");
	ASSERT_EQ(fan_out, tree->fan_out, "fan_out preserved");
	ASSERT_EQ(0u, hk_nodes(tree)[0].level, "root at level 0");
	ASSERT_EQ(
			nleaves, hk_nodes(tree)[0].nchildren, "root has nleaves children");
	ASSERT_EQ(
			HKMEANS_NO_CHILD,
			hk_nodes(tree)[0].first_child,
			"root is a leaf-parent");

	/* Leaf centroids copied verbatim. */
	const float *lc = hk_leaf_centroids(tree);
	for (uint32_t i = 0; i < nleaves * dim; i++)
		ASSERT_TRUE(lc[i] == cents[i], "leaf centroids match input");

	/* A query nearest to centroid i routes to leaf i. */
	float *q = mkt_alloc((size_t)dim * sizeof(float));
	for (uint32_t i = 0; i < nleaves; i++)
	{
		memset(q, 0, (size_t)dim * sizeof(float));
		q[0]		  = (float)(i + 1);
		uint32_t leaf = mkt_hkmeans_assign(tree, q, DISTANCE_L2, NULL);
		ASSERT_EQ(i, leaf, "query routes to its nearest leaf");
	}

	mkt_free(q);
	mkt_free(tree);
	mkt_free(cents);
}

/*
 * Structural invariants of a two-level tree.
 */
TEST(structural_invariants)
{
	uint32_t  fan_out = 4;
	uint32_t  nlist	  = 16;
	uint32_t  nvecs	  = 400;
	Dimension dim	  = 8;
	float	 *data	  = make_clustered_data(8, 50, dim, 42);

	KMeansOptions opts = MKT_KMEANS_OPTIONS_DEFAULT;

	HKMeansResult *tree = mkt_hkmeans_f32(
			data, nvecs, NULL, dim, nlist, fan_out, DISTANCE_L2, &opts);

	ASSERT_NOT_NULL(tree, "should return result");
	ASSERT_EQ(2, tree->nlevels, "2-level tree");

	/* All nodes have nchildren <= fan_out */
	for (uint32_t i = 0; i < tree->nnodes; i++)
		ASSERT_TRUE(
				hk_nodes(tree)[i].nchildren <= fan_out,
				"nchildren <= fan_out");

	/* Leaf centroid count matches nleaves */
	uint32_t leaf_sum = 0;
	for (uint32_t i = 0; i < tree->nnodes; i++)
	{
		if (hk_nodes(tree)[i].level == tree->nlevels - 1)
			leaf_sum += hk_nodes(tree)[i].nchildren;
	}
	ASSERT_EQ(tree->nleaves, leaf_sum, "nleaves matches leaf sum");

	/* Leaf centroids should be valid */
	ASSERT_NOT_NULL(hk_leaf_centroids(tree), "leaf centroids allocated");
	for (uint32_t i = 0; i < tree->nleaves * dim; i++)
		ASSERT_TRUE(
				isfinite(hk_leaf_centroids(tree)[i]), "leaf centroids finite");

	mkt_free(tree);
	mkt_free(data);
}

/*
 * NULL/invalid inputs return NULL.
 */
TEST(null_inputs)
{
	float		  data[] = {1.0f, 2.0f, 3.0f, 4.0f};
	KMeansOptions opts	 = MKT_KMEANS_OPTIONS_DEFAULT;

	ASSERT_NULL(
			mkt_hkmeans_f32(NULL, 2, NULL, 2, 1, 4, DISTANCE_L2, &opts),
			"NULL vectors");
	ASSERT_NULL(
			mkt_hkmeans_f32(data, 0, NULL, 2, 1, 4, DISTANCE_L2, &opts),
			"nvecs=0");
	ASSERT_NULL(
			mkt_hkmeans_f32(data, 2, NULL, 0, 1, 4, DISTANCE_L2, &opts),
			"dim=0");
	ASSERT_NULL(
			mkt_hkmeans_f32(data, 2, NULL, 2, 0, 4, DISTANCE_L2, &opts),
			"nlist=0");
	ASSERT_NULL(
			mkt_hkmeans_f32(data, 2, NULL, 2, 1, 1, DISTANCE_L2, &opts),
			"fan_out=1");
}

/*
 * Destroy NULL should not crash.
 */
TEST(destroy_null)
{
	mkt_free(NULL);
	ASSERT_TRUE(true, "no crash");
}

/*
 * Deterministic: same seed gives same result.
 */
TEST(deterministic)
{
	uint32_t  nvecs = 200;
	Dimension dim	= 8;
	float	 *data	= make_clustered_data(4, 50, dim, 42);

	KMeansOptions opts = MKT_KMEANS_OPTIONS_DEFAULT;
	opts.seed		   = 42;

	HKMeansResult *t1 =
			mkt_hkmeans_f32(data, nvecs, NULL, dim, 4, 4, DISTANCE_L2, &opts);
	HKMeansResult *t2 =
			mkt_hkmeans_f32(data, nvecs, NULL, dim, 4, 4, DISTANCE_L2, &opts);

	ASSERT_NOT_NULL(t1, "first run");
	ASSERT_NOT_NULL(t2, "second run");
	ASSERT_EQ(t1->nnodes, t2->nnodes, "same node count");
	ASSERT_EQ(t1->nleaves, t2->nleaves, "same leaf count");

	for (uint32_t i = 0; i < t1->nnodes; i++)
	{
		ASSERT_EQ(
				hk_nodes(t1)[i].nchildren,
				hk_nodes(t2)[i].nchildren,
				"same nchildren");
	}

	mkt_free(t1);
	mkt_free(t2);
	mkt_free(data);
}

/*
 * Few vectors (< fan_out): K gets clamped.
 */
TEST(few_vectors)
{
	float data[] = {
			1.0f,
			0.0f,
			0.0f,
			1.0f,
			-1.0f,
			0.0f,
			0.0f,
			-1.0f,
	};

	KMeansOptions opts = MKT_KMEANS_OPTIONS_DEFAULT;

	HKMeansResult *tree =
			mkt_hkmeans_f32(data, 4, NULL, 2, 2, 32, DISTANCE_L2, &opts);

	ASSERT_NOT_NULL(tree, "should succeed with few vectors");
	ASSERT_EQ(1, tree->nlevels, "flat when nlist <= fan_out");
	ASSERT_EQ(2, tree->nleaves, "2 leaf centroids");

	mkt_free(tree);
}

/*
 * nlist == 1: single cluster, single level.
 */
TEST(single_cluster)
{
	float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

	KMeansOptions opts = MKT_KMEANS_OPTIONS_DEFAULT;

	HKMeansResult *tree =
			mkt_hkmeans_f32(data, 3, NULL, 2, 1, 4, DISTANCE_L2, &opts);

	ASSERT_NOT_NULL(tree, "should succeed");
	ASSERT_EQ(1, tree->nlevels, "1 level");
	ASSERT_EQ(1, tree->nleaves, "1 leaf centroid");

	ASSERT_FLOAT_EQ(3.0f, hk_leaf_centroids(tree)[0], 1e-3f, "mean x");
	ASSERT_FLOAT_EQ(4.0f, hk_leaf_centroids(tree)[1], 1e-3f, "mean y");

	mkt_free(tree);
}

/* Brute-force nearest leaf (squared L2) for comparison. */
static uint32_t
bruteforce_nearest(
		const float *leaves,
		uint32_t	 nleaves,
		const float *vec,
		Dimension	 dim,
		double		*out_dist)
{
	uint32_t best  = 0;
	double	 bestd = INFINITY;
	for (uint32_t l = 0; l < nleaves; l++)
	{
		double d = 0.0;
		for (Dimension dd = 0; dd < dim; dd++)
		{
			double diff = (double)vec[dd] -
						  (double)leaves[(size_t)l * dim + dd];
			d += diff * diff;
		}
		if (d < bestd)
		{
			bestd = d;
			best  = l;
		}
	}
	*out_dist = bestd;
	return best;
}

/*
 * A wide beam (>= fan_out) explores every subtree, so the nearest leaf
 * it returns must equal the brute-force nearest, and distances must be
 * ascending.
 */
TEST(assign_topk_wide_beam_is_exact)
{
	uint32_t  fan_out = 4;
	uint32_t  nlist	  = 16;
	Dimension dim	  = 16;
	uint32_t  nvecs	  = 16 * 62;
	float	 *data	  = make_clustered_data(16, 62, dim, 7);

	KMeansOptions  opts = MKT_KMEANS_OPTIONS_DEFAULT;
	HKMeansResult *tree = mkt_hkmeans_f32(
			data, nvecs, NULL, dim, nlist, fan_out, DISTANCE_L2, &opts);
	ASSERT_NOT_NULL(tree, "tree built");
	ASSERT_EQ(2, tree->nlevels, "2-level tree");

	const float *leaves	 = hk_leaf_centroids(tree);
	uint32_t	 nleaves = tree->nleaves;

	for (uint32_t q = 0; q < nvecs; q += 37)
	{
		const float *vec = data + (size_t)q * dim;

		double	 bf_dist;
		uint32_t bf_best =
				bruteforce_nearest(leaves, nleaves, vec, dim, &bf_dist);

		uint32_t out_leaves[8];
		Distance out_dists[8];
		uint32_t n = mkt_hkmeans_assign_topk(
				tree,
				vec,
				DISTANCE_L2,
				4,
				MKT_HK_MAX_TOPK,
				out_leaves,
				out_dists);

		ASSERT_TRUE(n >= 1, "returns at least one leaf");
		ASSERT_TRUE(n <= 4, "returns at most k leaves");
		ASSERT_EQ(bf_best, out_leaves[0], "wide beam nearest == brute force");
		for (uint32_t i = 1; i < n; i++)
			ASSERT_TRUE(
					out_dists[i] >= out_dists[i - 1], "distances ascending");
		/* Returned leaf indices must be distinct. */
		for (uint32_t i = 0; i < n; i++)
			for (uint32_t j = i + 1; j < n; j++)
				ASSERT_TRUE(out_leaves[i] != out_leaves[j], "leaves distinct");
	}

	mkt_free(tree);
	mkt_free(data);
}

/*
 * beam_width = 1 follows the single greedy path, so its nearest leaf
 * must match mkt_hkmeans_assign().
 */
TEST(assign_topk_beam1_equals_greedy)
{
	uint32_t  fan_out = 4;
	uint32_t  nlist	  = 16;
	Dimension dim	  = 16;
	uint32_t  nvecs	  = 16 * 62;
	float	 *data	  = make_clustered_data(16, 62, dim, 11);

	KMeansOptions  opts = MKT_KMEANS_OPTIONS_DEFAULT;
	HKMeansResult *tree = mkt_hkmeans_f32(
			data, nvecs, NULL, dim, nlist, fan_out, DISTANCE_L2, &opts);
	ASSERT_NOT_NULL(tree, "tree built");

	for (uint32_t q = 0; q < nvecs; q += 53)
	{
		const float *vec = data + (size_t)q * dim;

		Distance greedy_dist;
		uint32_t greedy =
				mkt_hkmeans_assign(tree, vec, DISTANCE_L2, &greedy_dist);

		uint32_t out_leaves[4];
		Distance out_dists[4];
		uint32_t n = mkt_hkmeans_assign_topk(
				tree, vec, DISTANCE_L2, 2, 1, out_leaves, out_dists);

		ASSERT_TRUE(n >= 1, "returns at least one leaf");
		ASSERT_EQ(greedy, out_leaves[0], "beam-1 nearest == greedy assign");
		ASSERT_FLOAT_EQ(
				(float)greedy_dist,
				(float)out_dists[0],
				1e-3f,
				"beam-1 distance == greedy distance");
	}

	mkt_free(tree);
	mkt_free(data);
}

/*
 * The capped blob bound must dominate any tree actually built from
 * max_leaves vectors, whatever the nlist target -- including the chain
 * shape, where few vectors under a deep target produce one narrow node per
 * level. The parallel build sizes its subtree ring slots with this bound.
 */
TEST(max_blob_size_capped_bounds_actual)
{
	struct
	{
		uint32_t nvecs;
		uint32_t nlist;
		uint32_t fan_out;
	} cases[] = {
			{5, 1000, 10},	/* chains: 3 deep levels, 5 vectors */
			{50, 1000, 10}, /* partial width at every level */
			{200, 64, 4},	/* deeper than wide */
			{300, 16, 8},	/* two full levels */
			{7, 100000, 4}, /* extreme target, tiny data */
	};

	Dimension dim = 16;

	for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
	{
		uint32_t nvecs	 = cases[i].nvecs;
		uint32_t nlist	 = cases[i].nlist;
		uint32_t fan_out = cases[i].fan_out;
		float	*data =
				make_clustered_data(4, (nvecs + 3) / 4, dim, 42 + (unsigned)i);

		KMeansOptions  opts = MKT_KMEANS_OPTIONS_DEFAULT;
		HKMeansResult *tree = mkt_hkmeans_f32(
				data, nvecs, NULL, dim, nlist, fan_out, DISTANCE_L2, &opts);
		ASSERT_NOT_NULL(tree, "tree builds");

		size_t capped =
				mkt_hkmeans_max_blob_size_capped(nlist, fan_out, dim, nvecs);
		size_t uncapped = mkt_hkmeans_max_blob_size(nlist, fan_out, dim);

		ASSERT_TRUE(
				(size_t)tree->total_size <= capped,
				"actual blob within the capped bound");
		ASSERT_TRUE(capped <= uncapped, "cap never exceeds the worst case");

		mkt_free(tree);
		mkt_free(data);
	}
}
