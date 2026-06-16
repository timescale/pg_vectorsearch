/*
 * test_avq.c - Tests for the AVQ (anisotropic) partition center
 */

#include <math.h>
#include <string.h>

#include "algo/avq.h"
#include "core/memory.h"
#include "mkt_test.h"

TEST_GROUP(AVQ);
TEST_MEMCTX_FIXTURE();

/* Fill pts[n*dim] with deterministic pseudo-random unit vectors. */
static void
make_unit_vectors(float *pts, uint32_t n, uint32_t dim, uint64_t seed)
{
	uint64_t rng = seed * 2862933555777941757ULL + 3037000493ULL;
	for (uint32_t i = 0; i < n; i++)
	{
		float *x	 = pts + (size_t)i * dim;
		float  norm	 = 0.0f;
		for (uint32_t d = 0; d < dim; d++)
		{
			rng	 = rng * 6364136223846793005ULL + 1442695040888963407ULL;
			float v = (float)((rng >> 33) % 2000) / 1000.0f - 1.0f;
			x[d] = v;
			norm += v * v;
		}
		norm = sqrtf(norm);
		if (norm < 1e-12f)
			norm = 1.0f;
		for (uint32_t d = 0; d < dim; d++)
			x[d] /= norm;
	}
}

/* eta <= 1 must return the plain mean. */
TEST(eta_one_is_mean)
{
	const uint32_t n = 50, dim = 16;
	float		  *pts = mkt_alloc((size_t)n * dim * sizeof(float));
	make_unit_vectors(pts, n, dim, 7);

	float *c	= mkt_alloc(dim * sizeof(float));
	float *mean = mkt_alloc(dim * sizeof(float));
	memset(mean, 0, dim * sizeof(float));
	for (uint32_t i = 0; i < n; i++)
		for (uint32_t d = 0; d < dim; d++)
			mean[d] += pts[(size_t)i * dim + d];
	for (uint32_t d = 0; d < dim; d++)
		mean[d] /= (float)n;

	ASSERT_EQ(0, mkt_avq_center(pts, n, dim, 1.0f, c), "eta=1 ok");
	for (uint32_t d = 0; d < dim; d++)
		ASSERT_FLOAT_EQ(mean[d], c[d], 1e-5f, "eta=1 equals mean");

	mkt_free(pts);
	mkt_free(c);
	mkt_free(mean);
}

/* A single unit point: the AVQ center equals that point (for any eta). */
TEST(single_point_is_itself)
{
	const uint32_t dim = 8;
	float		  *x   = mkt_alloc(dim * sizeof(float));
	make_unit_vectors(x, 1, dim, 99);
	float *c = mkt_alloc(dim * sizeof(float));

	ASSERT_EQ(0, mkt_avq_center(x, 1, dim, 4.0f, c), "single point ok");
	for (uint32_t d = 0; d < dim; d++)
		ASSERT_FLOAT_EQ(x[d], c[d], 1e-4f, "single point center == point");

	mkt_free(x);
	mkt_free(c);
}

/*
 * The defining property: the AVQ center solves the normal equation
 *   (N*I + (eta-1) X^T X) c = eta * sum_i x_i.
 * Verify the residual directly (independent of the closed-form derivation).
 */
TEST(solves_normal_equation)
{
	const uint32_t n = 40, dim = 12;
	const float	   eta = 3.0f;
	float		  *pts = mkt_alloc((size_t)n * dim * sizeof(float));
	make_unit_vectors(pts, n, dim, 1234);
	float *c = mkt_alloc(dim * sizeof(float));
	ASSERT_EQ(0, mkt_avq_center(pts, n, dim, eta, c), "avq ok");

	/* rhs = eta * sum_x */
	float *rhs = mkt_alloc(dim * sizeof(float));
	memset(rhs, 0, dim * sizeof(float));
	for (uint32_t i = 0; i < n; i++)
		for (uint32_t d = 0; d < dim; d++)
			rhs[d] += pts[(size_t)i * dim + d];
	for (uint32_t d = 0; d < dim; d++)
		rhs[d] *= eta;

	/* lhs = (N*I + (eta-1) X^T X) c, computed as N*c + (eta-1) X^T (X c) */
	float *xc = mkt_alloc(n * sizeof(float)); /* X c */
	for (uint32_t i = 0; i < n; i++)
	{
		float s = 0.0f;
		for (uint32_t d = 0; d < dim; d++)
			s += pts[(size_t)i * dim + d] * c[d];
		xc[i] = s;
	}
	float *lhs = mkt_alloc(dim * sizeof(float));
	for (uint32_t d = 0; d < dim; d++)
	{
		float g = 0.0f; /* (X^T (X c))_d */
		for (uint32_t i = 0; i < n; i++)
			g += pts[(size_t)i * dim + d] * xc[i];
		lhs[d] = (float)n * c[d] + (eta - 1.0f) * g;
	}

	for (uint32_t d = 0; d < dim; d++)
		ASSERT_FLOAT_EQ(rhs[d], lhs[d], 1e-2f, "normal equation residual");

	mkt_free(pts);
	mkt_free(c);
	mkt_free(rhs);
	mkt_free(xc);
	mkt_free(lhs);
}

/*
 * Anisotropy direction check: along a dominant axis (high variance), eta>1
 * should push the center's projection BEYOND the mean's, recovering more of
 * the parallel component. Build points tightly around +e0 with spread on e1.
 */
TEST(shifts_toward_dominant_direction)
{
	const uint32_t n = 60, dim = 4;
	float		  *pts = mkt_alloc((size_t)n * dim * sizeof(float));
	uint64_t	   rng = 555;
	for (uint32_t i = 0; i < n; i++)
	{
		rng		   = rng * 6364136223846793005ULL + 1ULL;
		float	t  = (float)((rng >> 33) % 2000) / 1000.0f - 1.0f; /* [-1,1] */
		float  *x  = pts + (size_t)i * dim;
		x[0]	   = 1.0f;	   /* dominant */
		x[1]	   = 0.6f * t;  /* spread orthogonal */
		x[2]	   = 0.0f;
		x[3]	   = 0.0f;
		float norm = sqrtf(x[0] * x[0] + x[1] * x[1]);
		for (uint32_t d = 0; d < dim; d++)
			x[d] /= norm;
	}

	float *cm = mkt_alloc(dim * sizeof(float));
	float *ca = mkt_alloc(dim * sizeof(float));
	mkt_avq_center(pts, n, dim, 1.0f, cm); /* mean */
	mkt_avq_center(pts, n, dim, 5.0f, ca); /* anisotropic */

	/* The anisotropic center should have a LARGER component along e0 than the
	 * mean (it sacrifices the orthogonal e1 spread to preserve the parallel
	 * direction). */
	ASSERT_TRUE(ca[0] > cm[0], "AVQ center pushed along dominant axis");

	mkt_free(pts);
	mkt_free(cm);
	mkt_free(ca);
}
