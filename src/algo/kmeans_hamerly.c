/*
 * kmeans_hamerly.c - Hamerly's accelerated k-means assignment
 *
 * Hamerly's algorithm maintains two bounds per vector:
 *   upper[i] = upper bound on d(x_i, c_assigned)
 *   lower[i] = lower bound on d(x_i, second_nearest_centroid)
 *
 * After centroid update, bounds are loosened:
 *   upper[i] += delta[assigned[i]]   (assigned centroid may have moved away)
 *   lower[i] -= delta_max            (closest other centroid may have moved
 * closer)
 *
 * If upper[i] <= lower[i], the assignment cannot change and the vector
 * is skipped entirely. This is the key optimization — in later iterations,
 * 90%+ of vectors are skipped.
 *
 * For vectors that can't be skipped, we first tighten the upper bound
 * by computing the exact distance to the assigned centroid. If still
 * upper > lower, we do a full search over all centroids.
 *
 * Distance computation uses dot product decomposition:
 *   d^2(x, c) = ||x||^2 + ||c||^2 - 2 * dot(x, c)
 * This needs only 1 FMA per dimension (vs 2 for direct L2).
 */

#include "mkt_config.h"

#include <math.h>
#include <string.h>

#include "algo/kmeans_hamerly.h"
#include "algo/vecops.h"
#include "core/memory.h"

struct HamerlyState
{
	float	*upper_bound; /* [nvecs] d(x, assigned centroid) */
	float	*lower_bound; /* [nvecs] d(x, 2nd nearest centroid) */
	uint32_t nvecs;
	uint32_t nlist;
	uint32_t dim;
	bool	 bounds_valid;
};

HamerlyState *
hamerly_create(uint32_t nvecs, uint32_t nlist, uint32_t dim)
{
	HamerlyState *hs = mkt_alloc0(sizeof(HamerlyState));
	hs->nvecs		 = nvecs;
	hs->nlist		 = nlist;
	hs->dim			 = dim;
	hs->upper_bound	 = mkt_alloc(nvecs * sizeof(float));
	hs->lower_bound	 = mkt_alloc(nvecs * sizeof(float));
	return hs;
}

void
hamerly_destroy(HamerlyState *hs)
{
	if (hs == NULL)
		return;
	mkt_free(hs->upper_bound);
	mkt_free(hs->lower_bound);
	mkt_free(hs);
}

/*
 * Compute L2 squared distance using dot product decomposition.
 * Requires precomputed norms: norm_a = ||a||^2, norm_b = ||b||^2.
 */
static inline float
l2_from_dot(float norm_a, float norm_b, float dot)
{
	float d = norm_a + norm_b - 2.0f * dot;
	return d > 0.0f ? d : 0.0f;
}

/*
 * Full assignment for one vector: find nearest and second-nearest.
 */
__attribute__((always_inline)) static inline void
assign_full_impl(
		const void			   *vec,
		float					norm_x,
		const KMeansState	   *st,
		uint32_t			   *out_j1,
		float				   *out_dist1,
		float				   *out_dist2,
		const MktVectorTypeOps *ops)
{
	uint32_t	 nlist = st->nlist;
	Dimension	 dim   = st->dim;
	const float *cents = st->centroids;

	float	 d1 = FLT_MAX, d2 = FLT_MAX;
	uint32_t j1 = 0;

	for (uint32_t j = 0; j < nlist; j++)
	{
		float dp = ops->dot_product(vec, cents + (size_t)j * dim, dim);
		float d	 = l2_from_dot(norm_x, st->norms_c[j], dp);

		if (d < d1)
		{
			d2 = d1;
			d1 = d;
			j1 = j;
		}
		else if (d < d2)
		{
			d2 = d;
		}
	}

	*out_j1	   = j1;
	*out_dist1 = d1;
	*out_dist2 = d2;
}

/*
 * Precompute centroid norms (||c||^2) into st->norms_c.
 */
static void
precompute_norms_c(KMeansState *st)
{
	for (uint32_t j = 0; j < st->nlist; j++)
		st->norms_c[j] = mkt_l2_norm_squared(
				st->centroids + (size_t)j * st->dim, st->dim);
}

/*
 * Main Hamerly assignment loop.
 *
 * always_inline — the specialized wrappers below pass a static const
 * MktVectorTypeOps from the header, so the compiler inlines through
 * every vtable function pointer. MKT_TARGET_CLONES on the wrappers
 * generates AVX2/AVX-512 variants of the entire inlined body.
 */
__attribute__((always_inline)) static inline void
hamerly_assign_impl(
		KMeansState *st, HamerlyState *hs, const MktVectorTypeOps *ops)
{
	uint32_t	 nvecs = st->nvecs;
	Dimension	 dim   = st->dim;
	size_t		 esz   = ops->element_size;
	const float *cents = st->centroids;

	if (!hs->bounds_valid)
	{
		/*
		 * First call: full assignment, initialize bounds.
		 * Find nearest and second-nearest for each vector.
		 */
		for (uint32_t i = 0; i < nvecs; i++)
		{
			const void *v  = km_get_vector(st, i, esz);
			float		nx = st->norms_x[i];

			uint32_t j1;
			float	 d1, d2;
			assign_full_impl(v, nx, st, &j1, &d1, &d2, ops);

			st->assignments[i] = j1;
			st->total_cost += d1;
			hs->upper_bound[i] = sqrtf(d1);
			hs->lower_bound[i] = sqrtf(d2);
		}

		hs->bounds_valid = true;
		return;
	}

	/*
	 * Subsequent calls: use bounds to skip vectors.
	 */
	for (uint32_t i = 0; i < nvecs; i++)
	{
		float ub = hs->upper_bound[i];
		float lb = hs->lower_bound[i];

		/* Step 1: bound check — can we skip this vector? */
		if (ub <= lb)
		{
			/* Assignment can't change. Accumulate cost from bound. */
			st->total_cost += ub * ub;
			continue;
		}

		/* Step 2: tighten upper bound with exact distance */
		const void *v	 = km_get_vector(st, i, esz);
		float		nx	 = st->norms_x[i];
		uint32_t	prev = st->assignments[i];
		float dot_p = ops->dot_product(v, cents + (size_t)prev * dim, dim);
		float d_sq	= l2_from_dot(nx, st->norms_c[prev], dot_p);
		ub			= sqrtf(d_sq);

		hs->upper_bound[i] = ub;

		/* Check again with tightened bound */
		if (ub <= lb)
		{
			st->total_cost += d_sq;
			continue;
		}

		/* Step 3: full search required */
		uint32_t j1;
		float	 d1, d2;
		assign_full_impl(v, nx, st, &j1, &d1, &d2, ops);

		st->assignments[i] = j1;
		st->total_cost += d1;
		hs->upper_bound[i] = sqrtf(d1);
		hs->lower_bound[i] = sqrtf(d2);
	}
}

/*
 * Preconvert f16 variant: converts each f16 vector to f32 once before
 * the centroid loop, then uses the f32 dot product kernel. Saves
 * O(K*dim) f16→f32 conversions per vector (one conversion vs K).
 * Vectors skipped by bounds check are never converted.
 */
__attribute__((always_inline)) static inline void
hamerly_assign_preconvert_impl(KMeansState *st, HamerlyState *hs, size_t esz)
{
	uint32_t	 nvecs = st->nvecs;
	Dimension	 dim   = st->dim;
	const float *cents = st->centroids;
	float		*buf   = st->vec_block;

	const MktVectorTypeOps *f32ops = &mkt_f32_type_ops;

	if (!hs->bounds_valid)
	{
		for (uint32_t i = 0; i < nvecs; i++)
		{
			const void *raw = km_get_vector(st, i, esz);
			mkt_half_to_float_array((const half *)raw, buf, dim);
			float nx = st->norms_x[i];

			uint32_t j1;
			float	 d1, d2;
			assign_full_impl(buf, nx, st, &j1, &d1, &d2, f32ops);

			st->assignments[i] = j1;
			st->total_cost += d1;
			hs->upper_bound[i] = sqrtf(d1);
			hs->lower_bound[i] = sqrtf(d2);
		}

		hs->bounds_valid = true;
		return;
	}

	for (uint32_t i = 0; i < nvecs; i++)
	{
		float ub = hs->upper_bound[i];
		float lb = hs->lower_bound[i];

		if (ub <= lb)
		{
			st->total_cost += ub * ub;
			continue;
		}

		/* Convert once — only for vectors that need distance computation */
		const void *raw = km_get_vector(st, i, esz);
		mkt_half_to_float_array((const half *)raw, buf, dim);

		float	 nx	  = st->norms_x[i];
		uint32_t prev = st->assignments[i];
		float	 dot_p =
				f32ops->dot_product(buf, cents + (size_t)prev * dim, dim);
		float d_sq = l2_from_dot(nx, st->norms_c[prev], dot_p);
		ub		   = sqrtf(d_sq);

		hs->upper_bound[i] = ub;

		if (ub <= lb)
		{
			st->total_cost += d_sq;
			continue;
		}

		uint32_t j1;
		float	 d1, d2;
		assign_full_impl(buf, nx, st, &j1, &d1, &d2, f32ops);

		st->assignments[i] = j1;
		st->total_cost += d1;
		hs->upper_bound[i] = sqrtf(d1);
		hs->lower_bound[i] = sqrtf(d2);
	}
}

/* Specialized wrappers — MKT_TARGET_CLONES generates SIMD variants */
MKT_TARGET_CLONES static void
hamerly_assign_f32(KMeansState *st, HamerlyState *hs)
{
	hamerly_assign_impl(st, hs, &mkt_f32_type_ops);
}

MKT_TARGET_CLONES static void
hamerly_assign_f16(KMeansState *st, HamerlyState *hs)
{
	hamerly_assign_preconvert_impl(st, hs, sizeof(half));
}

/* Public entry — dispatches once based on type */
void
hamerly_assign(KMeansState *st, HamerlyState *hs)
{
	st->total_cost = 0.0f;
	precompute_norms_c(st);

	switch (st->vec_type)
	{
	case MKT_VEC_F32:
		hamerly_assign_f32(st, hs);
		break;
	default:
		hamerly_assign_f16(st, hs);
		break;
	}
}

void
hamerly_update_bounds(
		KMeansState *st, HamerlyState *hs, const float *old_centroids)
{
	uint32_t nlist = st->nlist;
	uint32_t dim   = st->dim;
	uint32_t nvecs = st->nvecs;

	/*
	 * Compute per-centroid movement (Euclidean distance) and
	 * find the two largest movements: delta_max and delta_second.
	 *
	 * delta_max is used to loosen all lower bounds.
	 * We track which centroid moved the most (p_max) so that
	 * vectors assigned to it use delta_second instead.
	 * (Hamerly's original paper optimization.)
	 */
	float	 delta_max = 0.0f, delta_second = 0.0f;
	uint32_t p_max = 0;

	for (uint32_t j = 0; j < nlist; j++)
	{
		float d_sq = mkt_l2_distance_squared(
				st->centroids + (size_t)j * dim,
				old_centroids + (size_t)j * dim,
				dim);
		float d = sqrtf(d_sq);

		if (d >= delta_max)
		{
			delta_second = delta_max;
			delta_max	 = d;
			p_max		 = j;
		}
		else if (d > delta_second)
		{
			delta_second = d;
		}
	}

	/*
	 * Also need per-centroid delta for the upper bound update.
	 * We compute these on the fly to avoid an extra allocation.
	 * Store them temporarily in norms_c (will be recomputed at
	 * next assign call anyway).
	 */
	float *delta = st->norms_c; /* temporary reuse */
	for (uint32_t j = 0; j < nlist; j++)
	{
		float d_sq = mkt_l2_distance_squared(
				st->centroids + (size_t)j * dim,
				old_centroids + (size_t)j * dim,
				dim);
		delta[j] = sqrtf(d_sq);
	}

	/* Update bounds for each vector */
	for (uint32_t i = 0; i < nvecs; i++)
	{
		uint32_t a = st->assignments[i];

		hs->upper_bound[i] += delta[a];

		/*
		 * Use delta_second if this vector's assigned centroid is
		 * the one that moved the most. Otherwise use delta_max.
		 */
		float lb_delta = (a == p_max) ? delta_second : delta_max;
		hs->lower_bound[i] -= lb_delta;
		if (hs->lower_bound[i] < 0.0f)
			hs->lower_bound[i] = 0.0f;
	}
}
