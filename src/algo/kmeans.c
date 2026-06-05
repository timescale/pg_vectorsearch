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
#include <stdlib.h>
#include <string.h>

#include "algo/kmeans_elkan.h"
#include "algo/kmeans_hamerly.h"
#include "algo/kmeans_internal.h"
#include "algo/kmeans_lloyd.h"
#include "algo/vecops.h"
#include "core/log.h"
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
 * Pin BLAS to single-threaded operation via the vendor's runtime API.
 *
 * Only symbols for the BLAS vendor detected at configure time are
 * referenced. Declaring all vendors' externs and relying on weak
 * symbols doesn't survive macOS's linker (Mach-O has no portable
 * "undefined weak reference resolves to NULL" that holds up under
 * LTO).
 *
 * Supported vendors: OpenBLAS (openblas_set_num_threads),
 * BLIS/AOCL (bli_thread_set_num_threads). Others (Accelerate,
 * MKL, ARM PL) must be pinned via env vars.
 */
#ifdef MKT_BLAS_OPENBLAS
extern void openblas_set_num_threads(int);
#endif
#ifdef MKT_BLAS_BLIS
extern void bli_thread_set_num_threads(long);
#endif

void
mkt_cblas_pin_single_thread(void)
{
#ifdef MKT_BLAS_OPENBLAS
	openblas_set_num_threads(1);
#endif
#ifdef MKT_BLAS_BLIS
	bli_thread_set_num_threads(1);
#endif
}

/*
 * Precompute ||x||^2 for all input vectors.
 */
__attribute__((always_inline)) static inline void
precompute_norms_x_impl(KMeansState *st, const MktVectorTypeOps *ops)
{
	size_t esz = ops->element_size;
	for (uint32_t i = 0; i < st->nvecs; i++)
		st->norms_x[i] = ops->norm_sq(km_get_vector(st, i, esz), st->dim);
}

/*
 * Compute distance from a typed vector to a float32 centroid.
 */
__attribute__((always_inline)) static inline float
vector_centroid_distance_impl(
		DistanceMetric			metric,
		const void			   *vec,
		const float			   *centroid,
		Dimension				dim,
		const MktVectorTypeOps *ops)
{
	switch (metric)
	{
	case DISTANCE_L2:
		return ops->l2_squared(vec, centroid, dim);
	case DISTANCE_INNER_PRODUCT:
		return -ops->dot_product(vec, centroid, dim);
	case DISTANCE_COSINE:
		return 1.0f - ops->dot_product(vec, centroid, dim);
	}
	return FLT_MAX;
}

/*
 * k-means++ initialization.
 */
__attribute__((always_inline)) static inline void
kmeans_init_plusplus_impl(
		KMeansState *st, uint64_t seed, const MktVectorTypeOps *ops)
{
	Xoshiro256State rng;
	xo_seed(&rng, seed);

	Dimension dim	= st->dim;
	uint32_t  nvecs = st->nvecs;
	uint32_t  nlist = st->nlist;
	size_t	  esz	= ops->element_size;
	float	 *dists = mkt_alloc(nvecs * sizeof(float));

	/* 1. First centroid: random vector */
	uint32_t idx = (uint32_t)(xo_uniform(&rng) * nvecs);
	if (idx >= nvecs)
		idx = nvecs - 1;
	ops->to_float_one(km_get_vector(st, idx, esz), st->centroids, dim);

	/* Initialize distances to first centroid */
	for (uint32_t i = 0; i < nvecs; i++)
		dists[i] = vector_centroid_distance_impl(
				st->metric,
				km_get_vector(st, i, esz),
				st->centroids,
				dim,
				ops);

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

		/* Copy chosen vector as new centroid (convert to f32) */
		ops->to_float_one(
				km_get_vector(st, chosen, esz),
				st->centroids + (size_t)k * dim,
				dim);

		/* Update min distances */
		for (uint32_t i = 0; i < nvecs; i++)
		{
			float d = vector_centroid_distance_impl(
					st->metric,
					km_get_vector(st, i, esz),
					st->centroids + (size_t)k * dim,
					dim,
					ops);
			if (d < dists[i])
				dists[i] = d;
		}
	}

	mkt_free(dists);
}

/*
 * Update step: recompute centroids as mean of assigned vectors.
 */
__attribute__((always_inline)) static inline void
kmeans_update_centroids_impl(KMeansState *st, const MktVectorTypeOps *ops)
{
	uint32_t  nlist = st->nlist;
	Dimension dim	= st->dim;
	size_t	  esz	= ops->element_size;

	/* Zero accumulators */
	memset(st->new_centroids, 0, (size_t)nlist * dim * sizeof(float));
	memset(st->cluster_sizes, 0, nlist * sizeof(uint32_t));

	/* Accumulate */
	for (uint32_t i = 0; i < st->nvecs; i++)
	{
		ClusterId c = st->assignments[i];
		st->cluster_sizes[c]++;
		const void *vec	 = km_get_vector(st, i, esz);
		float	   *cent = st->new_centroids + (size_t)c * dim;
		ops->sum_to_float(vec, cent, dim);
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
 * Max centroid movement between two centroid arrays.
 * Returns the maximum squared L2 shift.
 */
float
kmeans_max_centroid_shift_between(
		const float *a, const float *b, uint32_t nlist, Dimension dim)
{
	float max_shift = 0.0f;

	for (uint32_t j = 0; j < nlist; j++)
	{
		float shift = mkt_l2_distance_squared(
				a + (size_t)j * dim, b + (size_t)j * dim, dim);
		if (shift > max_shift)
			max_shift = shift;
	}

	return max_shift;
}

MKT_TARGET_CLONES void
kmeans_assign_accumulate(
		const float	   *vectors,
		const uint32_t *indices,
		uint32_t		start,
		uint32_t		end,
		const float	   *centroids,
		const float	   *norms_c,
		uint32_t		k,
		Dimension		dim,
		DistanceMetric	metric,
		const uint32_t *filter,
		uint32_t		filter_val,
		float		   *out_sums,
		uint32_t	   *out_cnts,
		float		   *out_cost)
{
	float cost = 0.0f;

	for (uint32_t i = start; i < end; i++)
	{
		if (filter != NULL && filter[i] != filter_val)
			continue;

		uint32_t	 idx	= indices ? indices[i] : i;
		const float *vec	= vectors + (size_t)idx * dim;
		float		 best_d = __FLT_MAX__;
		uint32_t	 best_c = 0;

		for (uint32_t c = 0; c < k; c++)
		{
			const float *cent = centroids + (size_t)c * dim;
			float		 d;

			switch (metric)
			{
			case DISTANCE_L2:
			{
				float nx  = mkt_l2_norm_squared(vec, dim);
				float dot = mkt_dot_product(vec, cent, dim);
				d		  = nx + norms_c[c] - 2.0f * dot;
				if (d < 0.0f)
					d = 0.0f;
				break;
			}
			case DISTANCE_INNER_PRODUCT:
				d = -mkt_dot_product(vec, cent, dim);
				break;
			case DISTANCE_COSINE:
				d = 1.0f - mkt_dot_product(vec, cent, dim);
				break;
			}

			if (d < best_d)
			{
				best_d = d;
				best_c = c;
			}
		}

		cost += best_d;
		out_cnts[best_c]++;
		float *sum = out_sums + (size_t)best_c * dim;
		for (uint32_t d = 0; d < dim; d++)
			sum[d] += vec[d];
	}

	*out_cost += cost;
}

MKT_TARGET_CLONES void
kmeans_assign(
		const float	  *vectors,
		uint32_t	   start,
		uint32_t	   end,
		const float	  *centroids,
		const float	  *norms_c,
		uint32_t	   k,
		Dimension	   dim,
		DistanceMetric metric,
		uint32_t	  *out_assignments)
{
	for (uint32_t i = start; i < end; i++)
	{
		const float *vec	= vectors + (size_t)i * dim;
		float		 best_d = __FLT_MAX__;
		uint32_t	 best_c = 0;

		for (uint32_t c = 0; c < k; c++)
		{
			const float *cent = centroids + (size_t)c * dim;
			float		 d;

			switch (metric)
			{
			case DISTANCE_L2:
			{
				float nx  = mkt_l2_norm_squared(vec, dim);
				float dot = mkt_dot_product(vec, cent, dim);
				d		  = nx + norms_c[c] - 2.0f * dot;
				if (d < 0.0f)
					d = 0.0f;
				break;
			}
			case DISTANCE_INNER_PRODUCT:
				d = -mkt_dot_product(vec, cent, dim);
				break;
			case DISTANCE_COSINE:
				d = 1.0f - mkt_dot_product(vec, cent, dim);
				break;
			}

			if (d < best_d)
			{
				best_d = d;
				best_c = c;
			}
		}

		out_assignments[i] = best_c;
	}
}

float
kmeans_merge_centroids(
		float				  *centroids,
		float				  *norms_c,
		const float			  *old_cents,
		const float *const	  *worker_sums,
		const uint32_t *const *worker_cnts,
		const float			  *worker_costs,
		uint32_t			   nworkers,
		uint32_t			   nlist,
		Dimension			   dim,
		DistanceMetric		   metric,
		float				  *out_total_cost)
{
	/* Sum costs */
	float total_cost = 0.0f;
	for (uint32_t t = 0; t < nworkers; t++)
		total_cost += worker_costs[t];
	*out_total_cost = total_cost;

	/* Merge per-worker accumulators */
	float *new_cents = centroids;
	memset(new_cents, 0, (size_t)nlist * dim * sizeof(float));

	uint32_t *sizes = (uint32_t *)alloca(nlist * sizeof(uint32_t));
	memset(sizes, 0, nlist * sizeof(uint32_t));

	for (uint32_t t = 0; t < nworkers; t++)
	{
		const float	   *sums = worker_sums[t];
		const uint32_t *cnts = worker_cnts[t];

		for (uint32_t c = 0; c < nlist; c++)
		{
			sizes[c] += cnts[c];
			float		*dst = new_cents + (size_t)c * dim;
			const float *src = sums + (size_t)c * dim;
			for (uint32_t d = 0; d < dim; d++)
				dst[d] += src[d];
		}
	}

	/* Divide by cluster size to get mean */
	for (uint32_t c = 0; c < nlist; c++)
	{
		if (sizes[c] == 0)
			continue;
		float  inv	= 1.0f / (float)sizes[c];
		float *cent = new_cents + (size_t)c * dim;
		for (uint32_t d = 0; d < dim; d++)
			cent[d] *= inv;
	}

	/* Normalize for cosine metric */
	if (metric == DISTANCE_COSINE)
	{
		for (uint32_t c = 0; c < nlist; c++)
		{
			if (sizes[c] == 0)
				continue;
			float *cent = new_cents + (size_t)c * dim;
			float  norm = mkt_l2_norm(cent, dim);
			if (norm > 1e-10f)
				mkt_vector_scale(cent, 1.0f / norm, cent, dim);
		}
	}

	/* Convergence: max centroid shift (squared) */
	float shift_sq = kmeans_max_centroid_shift_between(
			centroids, old_cents, nlist, dim);

	/* Precompute centroid norms for next iteration */
	if (norms_c != NULL && metric == DISTANCE_L2)
	{
		for (uint32_t j = 0; j < nlist; j++)
			norms_c[j] = mkt_l2_norm_squared(centroids + (size_t)j * dim, dim);
	}

	return shift_sq;
}

/*
 * Allocate working state for one k-means run.
 *
 * Creates an arena context and allocates everything (including the
 * struct itself) within it. kmeans_state_destroy() bulk-frees all
 * memory by deleting the arena.
 */
static KMeansState *
kmeans_state_create(
		const void	   *vectors,
		const uint32_t *indices,
		MktVecType		vec_type,
		uint32_t		nvecs,
		Dimension		dim,
		uint32_t		nlist,
		DistanceMetric	metric)
{
	MktMemCtx ctx	  = mkt_memctx_create(NULL, "kmeans_state");
	MktMemCtx old_ctx = mkt_memctx_switch(ctx);

	KMeansState *st = mkt_alloc0(sizeof(KMeansState));

	st->memctx	 = ctx;
	st->vectors	 = vectors;
	st->indices	 = indices;
	st->vec_type = vec_type;
	st->nvecs	 = nvecs;
	st->nlist	 = nlist;
	st->dim		 = dim;
	st->metric	 = metric;

	st->centroids	  = mkt_alloc((size_t)nlist * dim * sizeof(float));
	st->assignments	  = mkt_alloc(nvecs * sizeof(ClusterId));
	st->cluster_sizes = mkt_alloc(nlist * sizeof(uint32_t));
	st->new_centroids = mkt_alloc((size_t)nlist * dim * sizeof(float));

	uint32_t block = KMEANS_BLOCK_SIZE;
	if (block > nvecs)
		block = nvecs;
	st->dist_block = mkt_alloc((size_t)block * nlist * sizeof(float));

	/*
	 * Allocate conversion buffer for non-f32 types, or for f32 with
	 * indices (Lloyd gather needs a contiguous block).
	 */
	if (mkt_vec_element_size(vec_type) != sizeof(float) || indices != NULL)
		st->vec_block = mkt_alloc((size_t)block * dim * sizeof(float));

	if (metric == DISTANCE_L2)
	{
		st->norms_x = mkt_alloc(nvecs * sizeof(float));
		st->norms_c = mkt_alloc(nlist * sizeof(float));
	}

	mkt_memctx_switch(old_ctx);
	return st;
}

static void
kmeans_state_destroy(KMeansState *st)
{
	if (st == NULL)
		return;
	MktMemCtx ctx = (MktMemCtx)st->memctx;
	mkt_memctx_delete(ctx); /* st is now invalid */
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
 * always_inline — the specialized wrappers below pass a static const
 * MktVectorTypeOps from the header, so the compiler inlines through
 * every vtable function pointer. MKT_TARGET_CLONES on the wrappers
 * generates AVX2/AVX-512 variants of the entire inlined body.
 */
__attribute__((always_inline)) static inline void
kmeans_run_one_impl(
		KMeansState			   *st,
		const KMeansOptions	   *opts,
		uint64_t				seed,
		const KMeansAlgoOps	   *algo,
		const MktVectorTypeOps *ops)
{
	/* Precompute input vector norms for L2 */
	if (st->metric == DISTANCE_L2)
		precompute_norms_x_impl(st, ops);

	size_t cent_bytes = (size_t)st->nlist * st->dim * sizeof(float);
	void  *algo_state = algo->create ? algo->create(st) : NULL;
	float *old_cents  = algo->update_bounds ? mkt_alloc(cent_bytes) : NULL;

	if (opts->initial_centroids != NULL)
	{
		memcpy(st->centroids,
			   opts->initial_centroids,
			   (size_t)st->nlist * st->dim * sizeof(float));
	}
	else
	{
		kmeans_init_plusplus_impl(st, seed, ops);
	}

	/* Use fused iterate path for Lloyd with f32 vectors.
	 * Works with or without a thread pool — serial fallback
	 * calls work+reduce in a plain loop.
	 * Hamerly/Elkan use the generic loop below (needs bounds). */
	bool use_iterate = st->vec_type == MKT_VEC_F32 &&
					   algo->update_bounds == NULL;

	if (use_iterate)
	{
		bool use_cblas = (algo == &lloyd_cblas_ops);
		lloyd_iterate(st, use_cblas, opts);
		kmeans_handle_empty_clusters(st);
		if (algo->destroy)
			algo->destroy(algo_state);
		if (old_cents)
			mkt_free(old_cents);
		return;
	}

	for (uint32_t iter = 0; iter < opts->max_iterations; iter++)
	{
		/* Save old centroids for convergence check */
		memcpy(st->new_centroids, st->centroids, cent_bytes);

		/* Save separately for bounds update (Hamerly/Elkan) */
		if (old_cents)
			memcpy(old_cents, st->centroids, cent_bytes);

		algo->assign(st, algo_state);
		kmeans_update_centroids_impl(st, ops);
		kmeans_handle_empty_clusters(st);

		if (algo->update_bounds)
			algo->update_bounds(st, algo_state, old_cents);

		float shift_sq = kmeans_max_centroid_shift_between(
				st->centroids, st->new_centroids, st->nlist, st->dim);

		if (opts->verbose)
			mkt_log("  iter %u: cost=%.4f, max_shift=%.6f\n",
					iter,
					st->total_cost,
					sqrtf(shift_sq));

		st->total_cost = 0.0f;

		float tol_sq = opts->tolerance * opts->tolerance;
		if (shift_sq < tol_sq)
		{
			algo->assign(st, algo_state);
			if (algo->destroy)
				algo->destroy(algo_state);
			if (old_cents)
				mkt_free(old_cents);
			return;
		}
	}

	/* Final assignment after max iterations */
	algo->assign(st, algo_state);
	if (algo->destroy)
		algo->destroy(algo_state);
	if (old_cents)
		mkt_free(old_cents);
}

/* Specialized wrappers — MKT_TARGET_CLONES generates SIMD variants */

MKT_TARGET_CLONES static void
kmeans_run_one_f32(
		KMeansState			*st,
		const KMeansOptions *opts,
		uint64_t			 seed,
		const KMeansAlgoOps *algo)
{
	kmeans_run_one_impl(st, opts, seed, algo, &mkt_f32_type_ops);
}

MKT_TARGET_CLONES static void
kmeans_run_one_f16(
		KMeansState			*st,
		const KMeansOptions *opts,
		uint64_t			 seed,
		const KMeansAlgoOps *algo)
{
	kmeans_run_one_impl(st, opts, seed, algo, &mkt_f16_type_ops);
}

#if defined(MKT_F16C_SUPPORT) && !defined(MKT_SIMD_NONE)
MKT_TARGET_F16C_AVX2 static void
kmeans_run_one_f16c(
		KMeansState			*st,
		const KMeansOptions *opts,
		uint64_t			 seed,
		const KMeansAlgoOps *algo)
{
	kmeans_run_one_impl(st, opts, seed, algo, &mkt_f16c_type_ops);
}
#endif

/* Single dispatch point — selects the inline vtable once */
static void
kmeans_run_one(
		KMeansState			*st,
		const KMeansOptions *opts,
		uint64_t			 seed,
		const KMeansAlgoOps *algo)
{
	switch (st->vec_type)
	{
	case MKT_VEC_F32:
		kmeans_run_one_f32(st, opts, seed, algo);
		break;
#if defined(MKT_F16C_SUPPORT) && !defined(MKT_SIMD_NONE)
	case MKT_VEC_F16C:
		kmeans_run_one_f16c(st, opts, seed, algo);
		break;
#endif
	default:
		kmeans_run_one_f16(st, opts, seed, algo);
		break;
	}
}

/*
 * Public API
 */

KMeansResult *
mkt_kmeans(
		const void			*vectors,
		const uint32_t		*indices,
		MktVecType			 vec_type,
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
	const KMeansAlgoOps *algo_ops;
	switch (algo)
	{
	case KMEANS_ALGO_HAMERLY:
		algo_ops = &hamerly_ops;
		break;
	case KMEANS_ALGO_ELKAN:
		algo_ops = &elkan_ops;
		break;
	case KMEANS_ALGO_CBLAS:
		algo_ops = &lloyd_cblas_ops;
		break;
	case KMEANS_ALGO_LLOYD:
	default:
		algo_ops = &lloyd_ops;
		break;
	}

	size_t cent_sz	 = (size_t)nlist * dim * sizeof(float);
	size_t assign_sz = nvecs * sizeof(ClusterId);
	size_t clsize_sz = nlist * sizeof(uint32_t);

	for (uint32_t redo = 0; redo < opts.nredo; redo++)
	{
		KMeansState *st = kmeans_state_create(
				vectors, indices, vec_type, nvecs, dim, nlist, metric);

		uint64_t seed = opts.seed + redo;

		/* Run in arena context so per-iteration temps land there */
		MktMemCtx run_ctx = mkt_memctx_switch((MktMemCtx)st->memctx);
		kmeans_run_one(st, &opts, seed, algo_ops);
		mkt_memctx_switch(run_ctx);

		if (opts.verbose && opts.nredo > 1)
			mkt_log("redo %u/%u: cost=%.4f%s\n",
					redo + 1,
					opts.nredo,
					st->total_cost,
					st->total_cost < best_cost ? " (best)" : "");

		if (st->total_cost < best_cost)
		{
			/* Save as best — copy out of arena into caller ctx */
			if (best != NULL)
				mkt_kmeans_result_destroy(best);

			best			 = mkt_alloc0(sizeof(KMeansResult));
			best->nlist		 = nlist;
			best->dim		 = dim;
			best->total_cost = st->total_cost;

			best->centroids = mkt_alloc(cent_sz);
			memcpy(best->centroids, st->centroids, cent_sz);

			best->assignments = mkt_alloc(assign_sz);
			memcpy(best->assignments, st->assignments, assign_sz);

			best->cluster_sizes = mkt_alloc(clsize_sz);
			memcpy(best->cluster_sizes, st->cluster_sizes, clsize_sz);

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
