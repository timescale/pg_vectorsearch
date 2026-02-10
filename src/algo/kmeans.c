/*
 * kmeans.c - K-means clustering orchestration
 *
 * Common infrastructure shared by all k-means variants:
 * - k-means++ initialization (D²-weighted sampling)
 * - Centroid update step (mean of assigned vectors)
 * - Empty cluster handling (largest-cluster splitting)
 * - Convergence checking (max centroid shift)
 * - State management and public API
 *
 * Algorithm-specific assignment steps live in separate files:
 * - kmeans_lloyd.c:   brute-force (CBLAS sgemm or builtin batch)
 * - kmeans_hamerly.c: Hamerly's single-bound acceleration
 * - kmeans_elkan.c:   Elkan's per-centroid bounds
 */

#include "mkt_config.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "algo/kmeans_elkan.h"
#include "algo/kmeans_hamerly.h"
#include "algo/kmeans_internal.h"
#include "algo/kmeans_lloyd.h"
#include "algo/vecops.h"
#include "core/memory.h"

/*
 * Xoshiro256** PRNG (same as matrix.c - duplicated to keep kmeans.c
 * self-contained without exposing PRNG in a shared header).
 */
typedef struct
{
	uint64_t s[4];
} Xoshiro256State;

static inline uint64_t
xo_rotl(uint64_t x, int k)
{
	return (x << k) | (x >> (64 - k));
}

static uint64_t
xo_next(Xoshiro256State *state)
{
	uint64_t *s		 = state->s;
	uint64_t  result = xo_rotl(s[1] * 5, 7) * 9;
	uint64_t  t		 = s[1] << 17;

	s[2] ^= s[0];
	s[3] ^= s[1];
	s[1] ^= s[2];
	s[0] ^= s[3];
	s[2] ^= t;
	s[3] = xo_rotl(s[3], 45);

	return result;
}

static void
xo_seed(Xoshiro256State *state, uint64_t seed)
{
	for (int i = 0; i < 4; i++)
	{
		seed += 0x9e3779b97f4a7c15ULL;
		uint64_t z	= seed;
		z			= (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
		z			= (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
		state->s[i] = z ^ (z >> 31);
	}
}

/* Random double in [0, 1) */
static double
xo_uniform(Xoshiro256State *state)
{
	uint64_t x = xo_next(state) >> 11;
	return (double)x / (double)(1ULL << 53);
}

/* CBLAS runtime toggle (same pattern as matrix.c) */
static bool g_use_cblas = true;

void
mkt_kmeans_set_use_cblas(bool use_cblas)
{
	g_use_cblas = use_cblas;
}

bool
mkt_kmeans_get_use_cblas(void)
{
#ifdef MKT_HAVE_CBLAS
	return g_use_cblas;
#else
	return false;
#endif
}

const char *
mkt_kmeans_impl_name(void)
{
#ifdef MKT_HAVE_CBLAS
	return g_use_cblas ? "cblas" : "builtin";
#else
	return "builtin";
#endif
}

static const char *algo_names[] = {
		[KMEANS_ALGO_AUTO]	  = "auto",
		[KMEANS_ALGO_LLOYD]	  = "lloyd",
		[KMEANS_ALGO_HAMERLY] = "hamerly",
		[KMEANS_ALGO_ELKAN]	  = "elkan",
		[KMEANS_ALGO_CBLAS]	  = "lloyd(cblas)",
};

const char *
mkt_kmeans_algo_name(KMeansAlgorithm algo)
{
	if ((unsigned)algo <= KMEANS_ALGO_CBLAS)
		return algo_names[algo];
	return "unknown";
}

bool
mkt_cblas_is_single_threaded(void)
{
	const char *v = getenv("OMP_NUM_THREADS");
	return v != NULL && v[0] == '1' && v[1] == '\0';
}

/*
 * Precompute ||x||^2 for all input vectors (L2 only).
 */
static void
precompute_norms_x(KMeansState *st)
{
	for (uint32_t i = 0; i < st->nvecs; i++)
		st->norms_x[i] = mkt_l2_norm_squared(
				st->vectors + (size_t)i * st->dim, st->dim);
}

/*
 * Compute distance from a single vector to a single centroid.
 */
static float
vector_centroid_distance(
		const float	  *vec,
		const float	  *centroid,
		Dimension	   dim,
		DistanceMetric metric)
{
	switch (metric)
	{
	case DISTANCE_L2:
		return mkt_l2_distance_squared(vec, centroid, dim);
	case DISTANCE_INNER_PRODUCT:
		return -mkt_dot_product(vec, centroid, dim);
	case DISTANCE_COSINE:
		return 1.0f - mkt_dot_product(vec, centroid, dim);
	}
	return FLT_MAX;
}

/*
 * k-means++ initialization.
 *
 * 1. Pick first centroid uniformly at random.
 * 2. For each subsequent centroid, pick with probability proportional
 *    to squared distance to nearest existing centroid.
 */
static void
kmeans_init_plusplus(KMeansState *st, uint64_t seed)
{
	Xoshiro256State rng;
	xo_seed(&rng, seed);

	uint32_t dim   = st->dim;
	uint32_t nvecs = st->nvecs;
	uint32_t nlist = st->nlist;
	float	*dists = mkt_alloc(nvecs * sizeof(float));

	/* 1. First centroid: random vector */
	uint32_t idx = (uint32_t)(xo_uniform(&rng) * nvecs);
	if (idx >= nvecs)
		idx = nvecs - 1;
	memcpy(st->centroids,
		   st->vectors + (size_t)idx * dim,
		   dim * sizeof(float));

	/* Initialize distances to first centroid */
	for (uint32_t i = 0; i < nvecs; i++)
		dists[i] = vector_centroid_distance(
				st->vectors + (size_t)i * dim, st->centroids, dim, st->metric);

	/* 2. Pick remaining centroids */
	for (uint32_t k = 1; k < nlist; k++)
	{
		/* Compute cumulative distribution */
		double total = 0.0;
		for (uint32_t i = 0; i < nvecs; i++)
			total += (double)dists[i];

		/* Sample from distribution */
		double	 r		= xo_uniform(&rng) * total;
		double	 cum	= 0.0;
		uint32_t chosen = 0;
		for (uint32_t i = 0; i < nvecs; i++)
		{
			cum += (double)dists[i];
			if (cum >= r)
			{
				chosen = i;
				break;
			}
		}

		/* Copy chosen vector as new centroid */
		memcpy(st->centroids + (size_t)k * dim,
			   st->vectors + (size_t)chosen * dim,
			   dim * sizeof(float));

		/* Update min distances */
		for (uint32_t i = 0; i < nvecs; i++)
		{
			float d = vector_centroid_distance(
					st->vectors + (size_t)i * dim,
					st->centroids + (size_t)k * dim,
					dim,
					st->metric);
			if (d < dists[i])
				dists[i] = d;
		}
	}

	mkt_free(dists);
}

/*
 * Update step: recompute centroids as the mean of assigned vectors.
 *
 * For cosine: also normalizes centroids to unit length after averaging.
 */
static void
kmeans_update_centroids(KMeansState *st)
{
	uint32_t nlist = st->nlist;
	uint32_t dim   = st->dim;

	/* Zero accumulators */
	memset(st->new_centroids, 0, (size_t)nlist * dim * sizeof(float));
	memset(st->cluster_sizes, 0, nlist * sizeof(uint32_t));

	/* Accumulate */
	for (uint32_t i = 0; i < st->nvecs; i++)
	{
		ClusterId c = st->assignments[i];
		st->cluster_sizes[c]++;
		const float *vec  = st->vectors + (size_t)i * dim;
		float		*cent = st->new_centroids + (size_t)c * dim;
		for (uint32_t d = 0; d < dim; d++)
			cent[d] += vec[d];
	}

	/* Divide by cluster size */
	for (uint32_t j = 0; j < nlist; j++)
	{
		if (st->cluster_sizes[j] == 0)
			continue;

		float inv_size = 1.0f / (float)st->cluster_sizes[j];
		mkt_vector_scale(
				st->new_centroids + (size_t)j * dim,
				inv_size,
				st->new_centroids + (size_t)j * dim,
				dim);
	}

	/* For cosine: normalize centroids to unit length */
	if (st->metric == DISTANCE_COSINE)
	{
		for (uint32_t j = 0; j < nlist; j++)
		{
			if (st->cluster_sizes[j] == 0)
				continue;
			float *cent = st->new_centroids + (size_t)j * dim;
			float  norm = mkt_l2_norm(cent, dim);
			if (norm > 1e-10f)
				mkt_vector_scale(cent, 1.0f / norm, cent, dim);
		}
	}

	/* Swap new centroids into place */
	float *tmp		  = st->centroids;
	st->centroids	  = st->new_centroids;
	st->new_centroids = tmp;
}

/*
 * Handle empty clusters by splitting the largest cluster.
 *
 * FAISS approach: replace empty centroid with a perturbation of the
 * largest cluster's centroid.
 */
static void
kmeans_handle_empty_clusters(KMeansState *st)
{
	uint32_t dim   = st->dim;
	uint32_t nlist = st->nlist;

	for (uint32_t j = 0; j < nlist; j++)
	{
		if (st->cluster_sizes[j] > 0)
			continue;

		/* Find largest cluster */
		uint32_t largest	  = 0;
		uint32_t largest_size = st->cluster_sizes[0];
		for (uint32_t k = 1; k < nlist; k++)
		{
			if (st->cluster_sizes[k] > largest_size)
			{
				largest		 = k;
				largest_size = st->cluster_sizes[k];
			}
		}

		if (largest_size <= 1)
			continue; /* Can't split a cluster of size 1 */

		/* Perturb: empty = largest * (1 + eps), largest *= (1 - eps) */
		float *c_empty	 = st->centroids + (size_t)j * dim;
		float *c_largest = st->centroids + (size_t)largest * dim;

		for (uint32_t d = 0; d < dim; d++)
		{
			float val	 = c_largest[d];
			c_empty[d]	 = val * (1.0f + 1e-4f);
			c_largest[d] = val * (1.0f - 1e-4f);
		}

		/* Split size estimate (roughly half each) */
		uint32_t half		 = largest_size / 2;
		st->cluster_sizes[j] = half;
		st->cluster_sizes[largest] -= half;
	}
}

/*
 * Check convergence: max centroid shift.
 *
 * Returns the maximum squared L2 distance any centroid moved.
 */
static float
kmeans_max_centroid_shift(const KMeansState *st)
{
	float max_shift = 0.0f;

	for (uint32_t j = 0; j < st->nlist; j++)
	{
		float shift = mkt_l2_distance_squared(
				st->centroids + (size_t)j * st->dim,
				st->new_centroids + (size_t)j * st->dim,
				st->dim);
		if (shift > max_shift)
			max_shift = shift;
	}

	return max_shift;
}

/*
 * Allocate working state for one k-means run.
 */
static KMeansState *
kmeans_state_create(
		const float	  *vectors,
		uint32_t	   nvecs,
		Dimension	   dim,
		uint32_t	   nlist,
		DistanceMetric metric)
{
	KMeansState *st = mkt_alloc0(sizeof(KMeansState));

	st->vectors = vectors;
	st->nvecs	= nvecs;
	st->nlist	= nlist;
	st->dim		= dim;
	st->metric	= metric;

	st->centroids	  = mkt_alloc((size_t)nlist * dim * sizeof(float));
	st->assignments	  = mkt_alloc(nvecs * sizeof(ClusterId));
	st->cluster_sizes = mkt_alloc(nlist * sizeof(uint32_t));
	st->new_centroids = mkt_alloc((size_t)nlist * dim * sizeof(float));

	uint32_t block = KMEANS_BLOCK_SIZE;
	if (block > nvecs)
		block = nvecs;
	st->dist_block = mkt_alloc((size_t)block * nlist * sizeof(float));

	if (metric == DISTANCE_L2)
	{
		st->norms_x = mkt_alloc(nvecs * sizeof(float));
		st->norms_c = mkt_alloc(nlist * sizeof(float));
		precompute_norms_x(st);
	}

	return st;
}

static void
kmeans_state_destroy(KMeansState *st)
{
	if (st == NULL)
		return;
	mkt_free(st->centroids);
	mkt_free(st->assignments);
	mkt_free(st->cluster_sizes);
	mkt_free(st->new_centroids);
	mkt_free(st->dist_block);
	mkt_free(st->norms_x);
	mkt_free(st->norms_c);
	mkt_free(st);
}

/*
 * Algorithm vtable for the k-means iteration loop.
 *
 * Each variant provides: create/destroy for per-algorithm state,
 * assign for the assignment step, and optionally update_bounds
 * for bound-accelerated algorithms (Hamerly, Elkan).
 */
typedef struct
{
	void *(*create)(const KMeansState *st);
	void (*destroy)(void *algo_state);
	void (*assign)(KMeansState *st, void *algo_state);
	void (*update_bounds)(
			KMeansState *st, void *algo_state, const float *old_cents);
} KMeansAlgoOps;

/* Lloyd wrappers (no per-algorithm state) */

static void
lloyd_plain_assign(KMeansState *st, void *state)
{
	(void)state;
	lloyd_assign(st, false);
}

static void
lloyd_cblas_assign(KMeansState *st, void *state)
{
	(void)state;
	lloyd_assign(st, true);
}

static const KMeansAlgoOps lloyd_ops = {
		.assign = lloyd_plain_assign,
};

static const KMeansAlgoOps lloyd_cblas_ops = {
		.assign = lloyd_cblas_assign,
};

/* Hamerly wrappers */

static void *
hamerly_wrap_create(const KMeansState *st)
{
	return hamerly_create(st->nvecs, st->nlist, st->dim);
}

static void
hamerly_wrap_assign(KMeansState *st, void *state)
{
	hamerly_assign(st, state);
}

static void
hamerly_wrap_bounds(KMeansState *st, void *state, const float *old_cents)
{
	hamerly_update_bounds(st, state, old_cents);
}

static void
hamerly_wrap_destroy(void *state)
{
	hamerly_destroy(state);
}

static const KMeansAlgoOps hamerly_ops = {
		.create		   = hamerly_wrap_create,
		.destroy	   = hamerly_wrap_destroy,
		.assign		   = hamerly_wrap_assign,
		.update_bounds = hamerly_wrap_bounds,
};

/* Elkan wrappers */

static void *
elkan_wrap_create(const KMeansState *st)
{
	return elkan_create(st->nvecs, st->nlist, st->dim);
}

static void
elkan_wrap_assign(KMeansState *st, void *state)
{
	elkan_assign(st, state);
}

static void
elkan_wrap_bounds(KMeansState *st, void *state, const float *old_cents)
{
	elkan_update_bounds(st, state, old_cents);
}

static void
elkan_wrap_destroy(void *state)
{
	elkan_destroy(state);
}

static const KMeansAlgoOps elkan_ops = {
		.create		   = elkan_wrap_create,
		.destroy	   = elkan_wrap_destroy,
		.assign		   = elkan_wrap_assign,
		.update_bounds = elkan_wrap_bounds,
};

/*
 * Run one complete k-means attempt (init + iterate to convergence).
 *
 * Unified loop for all algorithms. The ops vtable determines the
 * assignment step and optional bound maintenance.
 */
static void
kmeans_run_one(
		KMeansState			*st,
		const KMeansOptions *opts,
		uint64_t			 seed,
		const KMeansAlgoOps *ops)
{
	size_t cent_bytes = (size_t)st->nlist * st->dim * sizeof(float);
	void  *algo_state = ops->create ? ops->create(st) : NULL;
	float *old_cents  = ops->update_bounds ? mkt_alloc(cent_bytes) : NULL;

	kmeans_init_plusplus(st, seed);

	for (uint32_t iter = 0; iter < opts->max_iterations; iter++)
	{
		/* Save old centroids for convergence check */
		memcpy(st->new_centroids, st->centroids, cent_bytes);

		/* Save separately for bounds update (Hamerly/Elkan) */
		if (old_cents)
			memcpy(old_cents, st->centroids, cent_bytes);

		ops->assign(st, algo_state);
		kmeans_update_centroids(st);
		kmeans_handle_empty_clusters(st);

		if (ops->update_bounds)
			ops->update_bounds(st, algo_state, old_cents);

		float shift = kmeans_max_centroid_shift(st);

		if (opts->verbose)
		{
			fprintf(stderr,
					"  iter %u: cost=%.4f, max_shift=%.6f\n",
					iter,
					st->total_cost,
					shift);
		}

		st->total_cost = 0.0f;

		if (shift < opts->tolerance)
		{
			ops->assign(st, algo_state);
			if (ops->destroy)
				ops->destroy(algo_state);
			mkt_free(old_cents);
			return;
		}
	}

	/* Final assignment after max iterations */
	ops->assign(st, algo_state);
	if (ops->destroy)
		ops->destroy(algo_state);
	mkt_free(old_cents);
}

/*
 * Public API
 */

KMeansResult *
mkt_kmeans(
		const float			*vectors,
		uint32_t			 nvecs,
		Dimension			 dim,
		uint32_t			 nlist,
		DistanceMetric		 metric,
		const KMeansOptions *options)
{
	if (vectors == NULL || nvecs == 0 || dim == 0 || nlist == 0)
		return NULL;

	/* Clamp nlist to nvecs */
	if (nlist > nvecs)
		nlist = nvecs;

	KMeansOptions opts = MKT_KMEANS_OPTIONS_DEFAULT;
	if (options != NULL)
		opts = *options;
	if (opts.nredo == 0)
		opts.nredo = 1;

	KMeansResult *best		= NULL;
	float		  best_cost = FLT_MAX;

	/* Resolve AUTO to a concrete algorithm */
	KMeansAlgorithm algo = opts.algorithm;
	if (algo == KMEANS_ALGO_AUTO)
	{
#ifdef MKT_HAVE_CBLAS
		algo = g_use_cblas ? KMEANS_ALGO_CBLAS : KMEANS_ALGO_LLOYD;
#else
		algo = KMEANS_ALGO_LLOYD;
#endif
	}

	/* Hamerly/Elkan only support L2 — fall back to Lloyd for others */
	if ((algo == KMEANS_ALGO_HAMERLY || algo == KMEANS_ALGO_ELKAN) &&
		metric != DISTANCE_L2)
		algo = KMEANS_ALGO_LLOYD;

	/* Select algorithm vtable */
	const KMeansAlgoOps *ops;
	switch (algo)
	{
	case KMEANS_ALGO_HAMERLY:
		ops = &hamerly_ops;
		break;
	case KMEANS_ALGO_ELKAN:
		ops = &elkan_ops;
		break;
	case KMEANS_ALGO_CBLAS:
		ops = &lloyd_cblas_ops;
		break;
	case KMEANS_ALGO_LLOYD:
	default:
		ops = &lloyd_ops;
		break;
	}

	for (uint32_t redo = 0; redo < opts.nredo; redo++)
	{
		KMeansState *st =
				kmeans_state_create(vectors, nvecs, dim, nlist, metric);

		uint64_t seed = opts.seed + redo;

		kmeans_run_one(st, &opts, seed, ops);

		if (opts.verbose && opts.nredo > 1)
		{
			fprintf(stderr,
					"redo %u/%u: cost=%.4f%s\n",
					redo + 1,
					opts.nredo,
					st->total_cost,
					st->total_cost < best_cost ? " (best)" : "");
		}

		if (st->total_cost < best_cost)
		{
			/* Save as best */
			if (best != NULL)
				mkt_kmeans_result_destroy(best);

			best			 = mkt_alloc0(sizeof(KMeansResult));
			best->nlist		 = nlist;
			best->dim		 = dim;
			best->total_cost = st->total_cost;

			/* Transfer ownership of arrays */
			best->centroids = st->centroids;
			st->centroids	= NULL;

			best->assignments = st->assignments;
			st->assignments	  = NULL;

			best->cluster_sizes = st->cluster_sizes;
			st->cluster_sizes	= NULL;

			best_cost = st->total_cost;
		}

		kmeans_state_destroy(st);
	}

	return best;
}

void
mkt_kmeans_result_destroy(KMeansResult *result)
{
	if (result == NULL)
		return;
	mkt_free(result->centroids);
	mkt_free(result->assignments);
	mkt_free(result->cluster_sizes);
	mkt_free(result);
}
