/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * kmeans_elkan.c - Elkan's accelerated k-means assignment
 *
 * Elkan's algorithm maintains K lower bounds per vector (one per
 * centroid) plus a single upper bound on the distance to the assigned
 * centroid. The triangle inequality enables three pruning rules:
 *
 * 1. If upper[i] <= s[assigned[i]], skip entirely.
 *    (s[j] = min half-distance from centroid j to any other centroid)
 *
 * 2. If upper[i] <= lower[i][k], skip centroid k.
 *    (lower bound is per-centroid, so tighter than Hamerly's global)
 *
 * 3. If upper[i] <= halfcdist[assigned[i]][k], skip centroid k.
 *    (centroid-centroid distance provides a tighter filter)
 *
 * After centroid update, bounds are adjusted per-centroid:
 *   lower[i][k] -= movement[k]   (only loosened by that centroid)
 *   upper[i]    += movement[assigned[i]]
 *
 * This is the key advantage over Hamerly: each lower bound is only
 * affected by its own centroid's movement, not by the global maximum.
 *
 * Distance computation uses dot product decomposition:
 *   d(x, c) = sqrt(||x||^2 + ||c||^2 - 2 * dot(x, c))
 */

#include "vs_config.h"

#include <math.h>
#include <string.h>

#include "algo/kmeans_elkan.h"
#include "algo/vecops.h"
#include "core/memory.h"

struct ElkanState
{
	float	*lower;		/* [nvecs * nlist] per-centroid lower bounds */
	float	*upper;		/* [nvecs] upper bound on assigned distance */
	float	*halfcdist; /* [nlist * nlist] half centroid-centroid dist */
	float	*s;			/* [nlist] min half-dist to any other centroid */
	float	*cdist;		/* [nlist] centroid movement distances */
	uint32_t nvecs;
	uint32_t nlist;
	uint32_t dim;
	bool	 bounds_valid;
};

ElkanState *
elkan_create(uint32_t nvecs, uint32_t nlist, uint32_t dim)
{
	ElkanState *es = vs_alloc0(sizeof(ElkanState));
	es->nvecs	   = nvecs;
	es->nlist	   = nlist;
	es->dim		   = dim;
	es->lower	   = vs_alloc0((size_t)nvecs * nlist * sizeof(float));
	es->upper	   = vs_alloc(nvecs * sizeof(float));
	es->halfcdist  = vs_alloc((size_t)nlist * nlist * sizeof(float));
	es->s		   = vs_alloc(nlist * sizeof(float));
	es->cdist	   = vs_alloc(nlist * sizeof(float));
	return es;
}

void
elkan_destroy(ElkanState *es)
{
	if (es == NULL)
		return;
	vs_free(es->lower);
	vs_free(es->upper);
	vs_free(es->halfcdist);
	vs_free(es->s);
	vs_free(es->cdist);
	vs_free(es);
}

/*
 * Compute dot product between two float32 vectors.
 * Used for centroid-centroid distances (always float32).
 */
VS_TARGET_CLONES static float
dot_product(const float *a, const float *b, uint32_t dim)
{
	float dot = 0.0f;
	for (uint32_t d = 0; d < dim; d++)
		dot += a[d] * b[d];
	return dot;
}

/*
 * Compute squared L2 distance using dot product decomposition.
 * Requires precomputed norms: norm_a = ||a||^2, norm_b = ||b||^2.
 */
static inline float
l2_sq_from_dot(float norm_a, float norm_b, float dot)
{
	float d_sq = norm_a + norm_b - 2.0f * dot;
	return d_sq > 0.0f ? d_sq : 0.0f;
}

/*
 * Compute Euclidean distance using dot product decomposition.
 */
static inline float
l2_dist_from_dot(float norm_a, float norm_b, float dot)
{
	return sqrtf(l2_sq_from_dot(norm_a, norm_b, dot));
}

/*
 * Compute centroid-centroid half-distances and min separations.
 *
 * halfcdist[j][k] = 0.5 * d(c_j, c_k)
 * s[j] = min_k(halfcdist[j][k])  for k != j
 */
static void
compute_centroid_dists(ElkanState *es, const KMeansState *st)
{
	uint32_t	 nlist = st->nlist;
	uint32_t	 dim   = st->dim;
	const float *cents = st->centroids;
	float		*hcd   = es->halfcdist;

	/* Initialize s to infinity */
	for (uint32_t j = 0; j < nlist; j++)
		es->s[j] = FLT_MAX;

	for (uint32_t j = 0; j < nlist; j++)
	{
		hcd[(size_t)j * nlist + j] = 0.0f;
		const float *cj			   = cents + (size_t)j * dim;
		float		 norm_j		   = st->norms_c[j];

		for (uint32_t k = j + 1; k < nlist; k++)
		{
			float dot = dot_product(cj, cents + (size_t)k * dim, dim);
			float d	  = l2_dist_from_dot(norm_j, st->norms_c[k], dot);
			float hd  = 0.5f * d;

			hcd[(size_t)j * nlist + k] = hd;
			hcd[(size_t)k * nlist + j] = hd;

			if (hd < es->s[j])
				es->s[j] = hd;
			if (hd < es->s[k])
				es->s[k] = hd;
		}
	}
}

/*
 * Precompute centroid norms (||c||^2) into st->norms_c.
 */
static void
precompute_norms_c(KMeansState *st)
{
	for (uint32_t j = 0; j < st->nlist; j++)
		st->norms_c[j] = vs_l2_norm_squared(
				st->centroids + (size_t)j * st->dim, st->dim);
}

/*
 * Full initial assignment — always_inline, specialized by ops vtable.
 */
__attribute__((always_inline)) static inline void
elkan_initial_assign_impl(
		KMeansState *st, ElkanState *es, const Vec32TypeOps *ops)
{
	uint32_t	 nvecs = st->nvecs;
	uint32_t	 nlist = st->nlist;
	Dimension	 dim   = st->dim;
	size_t		 esz   = ops->element_size;
	const float *cents = st->centroids;

	for (uint32_t i = 0; i < nvecs; i++)
	{
		const void *v  = km_get_vector(st, i, esz);
		float		nx = st->norms_x[i];
		float	   *lb = es->lower + (size_t)i * nlist;

		float	 best_d_sq = FLT_MAX;
		float	 best_d	   = FLT_MAX;
		uint32_t best_j	   = 0;

		for (uint32_t j = 0; j < nlist; j++)
		{
			float dp   = ops->dot_product(v, cents + (size_t)j * dim, dim);
			float d_sq = l2_sq_from_dot(nx, st->norms_c[j], dp);
			float d	   = sqrtf(d_sq);
			lb[j]	   = d;

			if (d_sq < best_d_sq)
			{
				best_d_sq = d_sq;
				best_d	  = d;
				best_j	  = j;
			}
		}

		st->assignments[i] = best_j;
		es->upper[i]	   = best_d;
		st->total_cost += best_d_sq;
	}
}

/*
 * Main Elkan assignment loop — always_inline, specialized by ops vtable.
 *
 * Centroid-centroid distances use the separate dot_product() function
 * (always float32 × float32). Only vector-centroid uses the typed ops.
 */
__attribute__((always_inline)) static inline void
elkan_assign_impl(KMeansState *st, ElkanState *es, const Vec32TypeOps *ops)
{
	uint32_t	 nvecs = st->nvecs;
	uint32_t	 nlist = st->nlist;
	Dimension	 dim   = st->dim;
	size_t		 esz   = ops->element_size;
	const float *cents = st->centroids;

	if (!es->bounds_valid)
	{
		elkan_initial_assign_impl(st, es, ops);
		compute_centroid_dists(es, st);
		es->bounds_valid = true;
		return;
	}

	/* Compute centroid-centroid distances for this iteration */
	compute_centroid_dists(es, st);

	for (uint32_t i = 0; i < nvecs; i++)
	{
		float	 ub	  = es->upper[i];
		uint32_t asgn = st->assignments[i];
		float	*lb	  = es->lower + (size_t)i * nlist;

		/* Step 1: skip if upper bound <= min half-centroid-distance */
		if (ub <= es->s[asgn])
		{
			st->total_cost += ub * ub;
			continue;
		}

		const void *v		  = km_get_vector(st, i, esz);
		float		nx		  = st->norms_x[i];
		float		ub_sq	  = ub * ub;
		bool		recompute = true;

		for (uint32_t k = 0; k < nlist; k++)
		{
			if (k == asgn)
				continue;

			/* Step 3: skip centroid via lower bound or half-dist */
			if (ub <= lb[k])
				continue;

			float *hcd_row = es->halfcdist + (size_t)asgn * nlist;
			if (ub <= hcd_row[k])
				continue;

			/* Step 3a: tighten upper bound if not yet done */
			if (recompute)
			{
				float dp_a =
						ops->dot_product(v, cents + (size_t)asgn * dim, dim);
				float d_sq_a = l2_sq_from_dot(nx, st->norms_c[asgn], dp_a);
				float d_a	 = sqrtf(d_sq_a);
				lb[asgn]	 = d_a;
				ub			 = d_a;
				ub_sq		 = d_sq_a;
				es->upper[i] = d_a;
				recompute	 = false;

				/* Re-check with tightened upper bound */
				if (ub <= lb[k])
					continue;
				if (ub <= hcd_row[k])
					continue;
			}

			/* Step 3b: compute actual distance to centroid k */
			float dp_k	 = ops->dot_product(v, cents + (size_t)k * dim, dim);
			float d_sq_k = l2_sq_from_dot(nx, st->norms_c[k], dp_k);
			float d_k	 = sqrtf(d_sq_k);
			lb[k]		 = d_k;

			/* Compare squared distances (matches Lloyd/Hamerly) */
			if (d_sq_k < ub_sq)
			{
				asgn			   = k;
				ub				   = d_k;
				ub_sq			   = d_sq_k;
				es->upper[i]	   = d_k;
				st->assignments[i] = k;
			}
		}

		st->total_cost += ub_sq;
	}
}

/*
 * Preconvert f16 variants: convert each f16 vector to f32 once before
 * the centroid loop, then use the f32 dot product kernel. Saves
 * O(K*dim) f16→f32 conversions per vector (one conversion vs K).
 * Vectors pruned by bounds are never converted.
 */
__attribute__((always_inline)) static inline void
elkan_initial_assign_preconvert_impl(
		KMeansState *st, ElkanState *es, size_t esz)
{
	uint32_t	 nvecs = st->nvecs;
	uint32_t	 nlist = st->nlist;
	Dimension	 dim   = st->dim;
	const float *cents = st->centroids;
	float		*buf   = st->vec_block;

	const Vec32TypeOps *f32ops = &vs_f32_type_ops;

	for (uint32_t i = 0; i < nvecs; i++)
	{
		const void *raw = km_get_vector(st, i, esz);
		vs_half_to_float_array((const half *)raw, buf, dim);
		float  nx = st->norms_x[i];
		float *lb = es->lower + (size_t)i * nlist;

		float	 best_d_sq = FLT_MAX;
		float	 best_d	   = FLT_MAX;
		uint32_t best_j	   = 0;

		for (uint32_t j = 0; j < nlist; j++)
		{
			float dp = f32ops->dot_product(buf, cents + (size_t)j * dim, dim);
			float d_sq = l2_sq_from_dot(nx, st->norms_c[j], dp);
			float d	   = sqrtf(d_sq);
			lb[j]	   = d;

			if (d_sq < best_d_sq)
			{
				best_d_sq = d_sq;
				best_d	  = d;
				best_j	  = j;
			}
		}

		st->assignments[i] = best_j;
		es->upper[i]	   = best_d;
		st->total_cost += best_d_sq;
	}
}

__attribute__((always_inline)) static inline void
elkan_assign_preconvert_impl(KMeansState *st, ElkanState *es, size_t esz)
{
	uint32_t	 nvecs = st->nvecs;
	uint32_t	 nlist = st->nlist;
	Dimension	 dim   = st->dim;
	const float *cents = st->centroids;
	float		*buf   = st->vec_block;

	const Vec32TypeOps *f32ops = &vs_f32_type_ops;

	if (!es->bounds_valid)
	{
		elkan_initial_assign_preconvert_impl(st, es, esz);
		compute_centroid_dists(es, st);
		es->bounds_valid = true;
		return;
	}

	compute_centroid_dists(es, st);

	for (uint32_t i = 0; i < nvecs; i++)
	{
		float	 ub	  = es->upper[i];
		uint32_t asgn = st->assignments[i];
		float	*lb	  = es->lower + (size_t)i * nlist;

		if (ub <= es->s[asgn])
		{
			st->total_cost += ub * ub;
			continue;
		}

		/* Convert once — only for vectors that pass the s-bound */
		const void *raw = km_get_vector(st, i, esz);
		vs_half_to_float_array((const half *)raw, buf, dim);

		float nx		= st->norms_x[i];
		float ub_sq		= ub * ub;
		bool  recompute = true;

		for (uint32_t k = 0; k < nlist; k++)
		{
			if (k == asgn)
				continue;

			if (ub <= lb[k])
				continue;

			float *hcd_row = es->halfcdist + (size_t)asgn * nlist;
			if (ub <= hcd_row[k])
				continue;

			if (recompute)
			{
				float dp_a = f32ops->dot_product(
						buf, cents + (size_t)asgn * dim, dim);
				float d_sq_a = l2_sq_from_dot(nx, st->norms_c[asgn], dp_a);
				float d_a	 = sqrtf(d_sq_a);
				lb[asgn]	 = d_a;
				ub			 = d_a;
				ub_sq		 = d_sq_a;
				es->upper[i] = d_a;
				recompute	 = false;

				if (ub <= lb[k])
					continue;
				if (ub <= hcd_row[k])
					continue;
			}

			float dp_k =
					f32ops->dot_product(buf, cents + (size_t)k * dim, dim);
			float d_sq_k = l2_sq_from_dot(nx, st->norms_c[k], dp_k);
			float d_k	 = sqrtf(d_sq_k);
			lb[k]		 = d_k;

			if (d_sq_k < ub_sq)
			{
				asgn			   = k;
				ub				   = d_k;
				ub_sq			   = d_sq_k;
				es->upper[i]	   = d_k;
				st->assignments[i] = k;
			}
		}

		st->total_cost += ub_sq;
	}
}

/* Specialized wrappers — VS_TARGET_CLONES generates SIMD variants */
VS_TARGET_CLONES static void
elkan_assign_f32(KMeansState *st, ElkanState *es)
{
	elkan_assign_impl(st, es, &vs_f32_type_ops);
}

VS_TARGET_CLONES static void
elkan_assign_f16(KMeansState *st, ElkanState *es)
{
	elkan_assign_preconvert_impl(st, es, sizeof(half));
}

/* Public entry — dispatches once based on type */
void
elkan_assign(KMeansState *st, ElkanState *es)
{
	st->total_cost = 0.0f;
	precompute_norms_c(st);

	switch (st->vec_type)
	{
	case VS_VEC_F32:
		elkan_assign_f32(st, es);
		break;
	default:
		elkan_assign_f16(st, es);
		break;
	}
}

void
elkan_update_bounds(
		KMeansState *st, ElkanState *es, const float *old_centroids)
{
	uint32_t nlist = st->nlist;
	uint32_t dim   = st->dim;
	uint32_t nvecs = st->nvecs;

	/* Compute per-centroid movement distances */
	for (uint32_t j = 0; j < nlist; j++)
	{
		float d_sq = vs_l2_distance_squared(
				st->centroids + (size_t)j * dim,
				old_centroids + (size_t)j * dim,
				dim);
		es->cdist[j] = sqrtf(d_sq);
	}

	/* Step 5: update lower bounds (loosen per-centroid) */
	for (uint32_t i = 0; i < nvecs; i++)
	{
		float *lb = es->lower + (size_t)i * nlist;
		for (uint32_t k = 0; k < nlist; k++)
		{
			lb[k] -= es->cdist[k];
			if (lb[k] < 0.0f)
				lb[k] = 0.0f;
		}
	}

	/* Step 6: update upper bounds (loosen by assigned movement) */
	for (uint32_t i = 0; i < nvecs; i++)
		es->upper[i] += es->cdist[st->assignments[i]];
}
