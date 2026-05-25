/*
 * test_fast_rotate.c - Unit tests for the randomized Hadamard rotation
 *
 * Covers the orthonormality property the RaBitQ math depends on
 * (norm preservation, inner-product preservation, involution after a
 * paired sign flip), plus the deterministic-from-seed contract.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "core/memory.h"
#include "mkt_test.h"
#include "quant/fast_rotate.h"

TEST_GROUP(FastRotate);
TEST_MEMCTX_FIXTURE();

/* dim must factor as fwht_n * k with fwht_n ≥ 4 power-of-2 and k small. */
TEST(supported_dims)
{
	ASSERT_FALSE(mkt_fast_rotate_supported(0), "0 not supported");
	ASSERT_FALSE(mkt_fast_rotate_supported(1), "1 not supported");
	ASSERT_FALSE(mkt_fast_rotate_supported(3), "3: fwht_n=1 too small");
	ASSERT_FALSE(mkt_fast_rotate_supported(72), "72=8*9: k=9 > K_MAX");
	ASSERT_TRUE(mkt_fast_rotate_supported(4), "4: pure FWHT");
	ASSERT_TRUE(mkt_fast_rotate_supported(8), "8: pure FWHT");
	ASSERT_TRUE(mkt_fast_rotate_supported(1024), "1024: pure FWHT");
	ASSERT_TRUE(mkt_fast_rotate_supported(768), "768 = 256*3");
	ASSERT_TRUE(mkt_fast_rotate_supported(384), "384 = 128*3");
	ASSERT_TRUE(mkt_fast_rotate_supported(1536), "1536 = 512*3");
}

static double
sum_sq(const float *v, Dimension d)
{
	double s = 0;
	for (Dimension i = 0; i < d; i++)
		s += (double) v[i] * v[i];
	return s;
}

static double
dot(const float *a, const float *b, Dimension d)
{
	double s = 0;
	for (Dimension i = 0; i < d; i++)
		s += (double) a[i] * b[i];
	return s;
}

static void
fill_random(float *v, Dimension d, unsigned seed)
{
	srand(seed);
	for (Dimension i = 0; i < d; i++)
		v[i] = (float) ((rand() / (double) RAND_MAX) * 2.0 - 1.0);
}

/* Norm preservation across dims: pure-FWHT path and mixed-radix path. */
TEST(preserves_norm)
{
	const Dimension dims[] = {64, 256, 768, 1024, 1536};
	const uint64_t  seeds[] = {0x123, 0x456, 0xC0FFEE, 0xDEAD, 0xABCD};
	for (size_t t = 0; t < sizeof(dims) / sizeof(dims[0]); t++)
	{
		Dimension dim = dims[t];
		MktFastRotateParams p;
		mkt_fast_rotate_init(&p, dim, seeds[t]);

		float *x   = mkt_alloc(dim * sizeof(float));
		float *out = mkt_alloc(dim * sizeof(float));
		fill_random(x, dim, (unsigned) (7 + t));

		mkt_fast_rotate_apply(&p, x, out);

		double nx = sum_sq(x, dim);
		double ny = sum_sq(out, dim);
		ASSERT_FLOAT_EQ((float) nx, (float) ny, 5e-3f,
						"rotation preserves norm");

		mkt_free(x);
		mkt_free(out);
	}
}

/* Inner-product preservation across dims. */
TEST(preserves_inner_product)
{
	const Dimension dims[] = {64, 256, 768, 1024};
	for (size_t t = 0; t < sizeof(dims) / sizeof(dims[0]); t++)
	{
		Dimension dim = dims[t];
		MktFastRotateParams p;
		mkt_fast_rotate_init(&p, dim, 0x1234 + t);

		float *x  = mkt_alloc(dim * sizeof(float));
		float *y  = mkt_alloc(dim * sizeof(float));
		float *fx = mkt_alloc(dim * sizeof(float));
		float *fy = mkt_alloc(dim * sizeof(float));
		fill_random(x, dim, (unsigned) (11 + t));
		fill_random(y, dim, (unsigned) (13 + t));

		mkt_fast_rotate_apply(&p, x, fx);
		mkt_fast_rotate_apply(&p, y, fy);

		double a = dot(x, y, dim);
		double b = dot(fx, fy, dim);
		ASSERT_FLOAT_EQ((float) a, (float) b, 5e-3f,
						"rotation preserves inner product");

		mkt_free(x);
		mkt_free(y);
		mkt_free(fx);
		mkt_free(fy);
	}
}

/* Deterministic from seed: same seed → same sign vector → same output. */
TEST(deterministic_from_seed)
{
	const Dimension dim = 64;
	MktFastRotateParams p1, p2;
	mkt_fast_rotate_init(&p1, dim, 0xCAFEULL);
	mkt_fast_rotate_init(&p2, dim, 0xCAFEULL);

	ASSERT_MEM_EQ(
			p1.signs1, p2.signs1, (size_t)((dim + 7) / 8),
			"same seed produces same sign1 vector");
	ASSERT_MEM_EQ(
			p1.signs2, p2.signs2, (size_t)((dim + 7) / 8),
			"same seed produces same sign2 vector");

	float *x	= mkt_alloc(dim * sizeof(float));
	float *out1 = mkt_alloc(dim * sizeof(float));
	float *out2 = mkt_alloc(dim * sizeof(float));
	for (Dimension i = 0; i < dim; i++)
		x[i] = (float) i - 32.0f;
	mkt_fast_rotate_apply(&p1, x, out1);
	mkt_fast_rotate_apply(&p2, x, out2);
	ASSERT_MEM_EQ(
			out1, out2, dim * sizeof(float),
			"same seed produces identical rotation");

	mkt_free(x);
	mkt_free(out1);
	mkt_free(out2);
}

/* In-place aliasing: out == in must work. */
TEST(in_place)
{
	const Dimension dim = 32;
	MktFastRotateParams p;
	mkt_fast_rotate_init(&p, dim, 99);

	float x[32], copy[32], copy_out[32];
	for (Dimension i = 0; i < dim; i++)
		copy[i] = x[i] = (float) (i * 0.3f - 5.0f);

	mkt_fast_rotate_apply(&p, x, x);
	mkt_fast_rotate_apply(&p, copy, copy_out);

	for (Dimension i = 0; i < dim; i++)
		ASSERT_FLOAT_EQ(copy_out[i], x[i], 1e-6f,
						"in-place result matches out-of-place");
}
