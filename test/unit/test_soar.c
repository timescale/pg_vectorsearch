/*
 * test_soar.c - Unit tests for SOAR secondary-cluster selection
 *
 * The build picks a SOAR (orthogonality-amplified) secondary cluster for
 * replicated vectors. prism_find_soar_secondary searches a candidate set
 * (cand_leaves, the beam-descent candidates — O(count)) in production, or
 * every leaf when cand_leaves is NULL (a full scan). These tests pin it down:
 *
 *  - The candidate set spanning every leaf returns exactly what the NULL full
 *    scan returns (same objective, same domain) — so the only thing that
 *    changes in production is which leaves are offered as candidates.
 *  - Over an arbitrary candidate set it returns the candidate (other than the
 *    primary) that minimizes the SOAR objective.
 *  - It never returns the primary, and it skips a primary in the candidate
 * list.
 */

#include <math.h>
#include <string.h>

#include "algo/vecops.h"
#include "core/memory.h"
#include "index/index_build.h"
#include "vs_test.h"

TEST_GROUP(SoarSecondary);
TEST_MEMCTX_FIXTURE();

/* Deterministic PRNG so failures reproduce. */
static uint32_t g_rng;

static void
rng_seed(uint32_t s)
{
	g_rng = s;
}

static float
frand(void)
{
	g_rng = g_rng * 1103515245u + 12345u;
	return (float)((g_rng >> 8) & 0xffff) / 65535.0f - 0.5f;
}

static void
fill_random(float *p, size_t n)
{
	for (size_t i = 0; i < n; i++)
		p[i] = frand();
}

static uint32_t
nearest_leaf(
		const float *vec, const float *leaves, uint32_t nleaves, uint32_t dim)
{
	uint32_t best = 0;
	float	 bd	  = INFINITY;
	for (uint32_t i = 0; i < nleaves; i++)
	{
		float d = vs_l2_distance_squared(vec, leaves + (size_t)i * dim, dim);
		if (d < bd)
		{
			bd	 = d;
			best = i;
		}
	}
	return best;
}

/* Normalized residual of vec from the primary centroid (as the build does). */
static void
make_residual(float *r, const float *vec, const float *cent, uint32_t dim)
{
	float norm = 0.0f;
	for (uint32_t d = 0; d < dim; d++)
	{
		r[d] = vec[d] - cent[d];
		norm += r[d] * r[d];
	}
	if (norm > 1e-7f)
	{
		float inv = 1.0f / sqrtf(norm);
		for (uint32_t d = 0; d < dim; d++)
			r[d] *= inv;
	}
}

/* Reference SOAR objective, argmin over the candidate set (excluding primary).
 */
static uint32_t
ref_soar_min(
		const float	   *vec,
		const float	   *leaves,
		const uint32_t *cand,
		uint32_t		ncand,
		uint32_t		dim,
		uint32_t		primary,
		const float	   *r,
		float			lambda)
{
	float	 qrv	= vs_dot_product(r, vec, dim);
	float	 best	= INFINITY;
	uint32_t best_c = primary;
	for (uint32_t k = 0; k < ncand; k++)
	{
		uint32_t i = cand[k];
		if (i == primary)
			continue;
		const float *c	 = leaves + (size_t)i * dim;
		float		 l2	 = vs_l2_distance_squared(vec, c, dim);
		float		 gap = qrv - vs_dot_product(r, c, dim);
		float		 oa	 = l2 + lambda * gap * gap;
		if (oa < best)
		{
			best   = oa;
			best_c = i;
		}
	}
	return best_c;
}

/*
 * Over the full leaf set, the candidate variant must equal the full scan: same
 * objective, same domain, so the result is identical. This is the contract
 * that lets production swap the full scan for a candidate search without
 * changing what gets chosen — only which leaves are offered changes.
 */
TEST(soar_cand_equals_full_scan_over_all_leaves)
{
	const uint32_t nleaves = 200;
	const uint32_t dim	   = 32;
	float		  *leaves  = vs_alloc((size_t)nleaves * dim * sizeof(float));
	float		  *vec	   = vs_alloc((size_t)dim * sizeof(float));
	float		  *r	   = vs_alloc((size_t)dim * sizeof(float));
	uint32_t	  *all	   = vs_alloc((size_t)nleaves * sizeof(uint32_t));
	for (uint32_t i = 0; i < nleaves; i++)
		all[i] = i;

	rng_seed(0xC0FFEEu);
	for (int trial = 0; trial < 300; trial++)
	{
		fill_random(leaves, (size_t)nleaves * dim);
		fill_random(vec, dim);
		uint32_t primary = nearest_leaf(vec, leaves, nleaves, dim);
		make_residual(r, vec, leaves + (size_t)primary * dim, dim);

		uint32_t full = prism_find_soar_secondary(
				vec, leaves, NULL, nleaves, dim, primary, r, 1.0);
		uint32_t cand = prism_find_soar_secondary(
				vec, leaves, all, nleaves, dim, primary, r, 1.0);
		ASSERT_EQ(
				full,
				cand,
				"candidate SOAR over all leaves must match the full scan");
	}
}

/*
 * Over an arbitrary candidate set, the function must return the SOAR-objective
 * argmin (excluding primary), matching an independent reference computation.
 */
TEST(soar_cand_picks_objective_min)
{
	const uint32_t nleaves = 128;
	const uint32_t dim	   = 16;
	const uint32_t ncand   = 8;
	float		  *leaves  = vs_alloc((size_t)nleaves * dim * sizeof(float));
	float		  *vec	   = vs_alloc((size_t)dim * sizeof(float));
	float		  *r	   = vs_alloc((size_t)dim * sizeof(float));
	uint32_t	  *cand	   = vs_alloc((size_t)ncand * sizeof(uint32_t));

	rng_seed(0x1234u);
	for (int trial = 0; trial < 300; trial++)
	{
		fill_random(leaves, (size_t)nleaves * dim);
		fill_random(vec, dim);
		uint32_t primary = nearest_leaf(vec, leaves, nleaves, dim);
		make_residual(r, vec, leaves + (size_t)primary * dim, dim);

		/* An arbitrary candidate set (distinct leaf ids). */
		for (uint32_t k = 0; k < ncand; k++)
			cand[k] = (uint32_t)((g_rng = g_rng * 1103515245u + 12345u) >> 9) %
					  nleaves;

		uint32_t got = prism_find_soar_secondary(
				vec, leaves, cand, ncand, dim, primary, r, 1.5);
		uint32_t expect =
				ref_soar_min(vec, leaves, cand, ncand, dim, primary, r, 1.5f);
		ASSERT_EQ(
				expect,
				got,
				"candidate SOAR must return the objective argmin");
		ASSERT_NEQ(primary, got, "SOAR secondary must never be the primary");
	}
}

/* A primary appearing in the candidate list is skipped; with only the primary
 * offered, the result is the primary (no replication). */
TEST(soar_cand_skips_primary)
{
	const uint32_t nleaves = 32;
	const uint32_t dim	   = 8;
	float		  *leaves  = vs_alloc((size_t)nleaves * dim * sizeof(float));
	float		  *vec	   = vs_alloc((size_t)dim * sizeof(float));
	float		  *r	   = vs_alloc((size_t)dim * sizeof(float));

	rng_seed(0x55u);
	fill_random(leaves, (size_t)nleaves * dim);
	fill_random(vec, dim);
	uint32_t primary = nearest_leaf(vec, leaves, nleaves, dim);
	make_residual(r, vec, leaves + (size_t)primary * dim, dim);

	uint32_t only_primary[1] = {primary};
	uint32_t got			 = prism_find_soar_secondary(
			vec, leaves, only_primary, 1, dim, primary, r, 1.0);
	ASSERT_EQ(
			primary,
			got,
			"with only the primary offered, the result is the primary");

	/* primary mixed with one other candidate -> the other one wins. */
	uint32_t other	  = (primary + 1) % nleaves;
	uint32_t mixed[2] = {primary, other};
	uint32_t got2	  = prism_find_soar_secondary(
			vec, leaves, mixed, 2, dim, primary, r, 1.0);
	ASSERT_EQ(other, got2, "primary in the candidate list must be skipped");
}
