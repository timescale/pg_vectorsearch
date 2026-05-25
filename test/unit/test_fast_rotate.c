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

/* Reference: dim must be power of two, ≥ 4. */
TEST(supported_dims)
{
	ASSERT_FALSE(mkt_fast_rotate_supported(0), "0 not supported");
	ASSERT_FALSE(mkt_fast_rotate_supported(1), "1 not supported");
	ASSERT_FALSE(mkt_fast_rotate_supported(3), "3 not power of two");
	ASSERT_FALSE(mkt_fast_rotate_supported(768), "768 not power of two");
	ASSERT_TRUE(mkt_fast_rotate_supported(4), "4 supported");
	ASSERT_TRUE(mkt_fast_rotate_supported(8), "8 supported");
	ASSERT_TRUE(mkt_fast_rotate_supported(1024), "1024 supported");
}

/* Norm preservation: ||F(x)|| should equal ||x|| (orthonormal). */
TEST(preserves_norm)
{
	const Dimension dim = 1024;
	uint8_t *signs = mkt_alloc((dim + 7) / 8);
	MktFastRotateParams p;
	mkt_fast_rotate_init(&p, dim, 0x123456789ABCDEF0ULL, signs);

	float *x   = mkt_alloc(dim * sizeof(float));
	float *out = mkt_alloc(dim * sizeof(float));
	srand(7);
	double nx = 0;
	for (Dimension i = 0; i < dim; i++)
	{
		x[i] = (float) ((rand() / (double) RAND_MAX) * 2.0 - 1.0);
		nx += (double) x[i] * x[i];
	}

	mkt_fast_rotate_apply(&p, x, out);

	double ny = 0;
	for (Dimension i = 0; i < dim; i++)
		ny += (double) out[i] * out[i];

	ASSERT_FLOAT_EQ((float) nx, (float) ny, 1e-3f,
					"orthonormal rotation preserves norm");

	mkt_free(x);
	mkt_free(out);
	mkt_free(signs);
}

/* Inner-product preservation: <F(x), F(y)> should equal <x, y>. */
TEST(preserves_inner_product)
{
	const Dimension dim = 256;
	uint8_t *signs = mkt_alloc((dim + 7) / 8);
	MktFastRotateParams p;
	mkt_fast_rotate_init(&p, dim, 42, signs);

	float *x  = mkt_alloc(dim * sizeof(float));
	float *y  = mkt_alloc(dim * sizeof(float));
	float *fx = mkt_alloc(dim * sizeof(float));
	float *fy = mkt_alloc(dim * sizeof(float));
	srand(11);
	double dot_orig = 0;
	for (Dimension i = 0; i < dim; i++)
	{
		x[i] = (float) ((rand() / (double) RAND_MAX) * 2.0 - 1.0);
		y[i] = (float) ((rand() / (double) RAND_MAX) * 2.0 - 1.0);
		dot_orig += (double) x[i] * y[i];
	}

	mkt_fast_rotate_apply(&p, x, fx);
	mkt_fast_rotate_apply(&p, y, fy);

	double dot_rot = 0;
	for (Dimension i = 0; i < dim; i++)
		dot_rot += (double) fx[i] * fy[i];

	ASSERT_FLOAT_EQ((float) dot_orig, (float) dot_rot, 1e-3f,
					"orthonormal rotation preserves inner product");

	mkt_free(x);
	mkt_free(y);
	mkt_free(fx);
	mkt_free(fy);
	mkt_free(signs);
}

/* Deterministic from seed: same seed → same sign vector → same output. */
TEST(deterministic_from_seed)
{
	const Dimension dim	   = 64;
	uint8_t		   *signs1 = mkt_alloc((dim + 7) / 8);
	uint8_t		   *signs2 = mkt_alloc((dim + 7) / 8);
	MktFastRotateParams p1, p2;
	mkt_fast_rotate_init(&p1, dim, 0xCAFEULL, signs1);
	mkt_fast_rotate_init(&p2, dim, 0xCAFEULL, signs2);

	ASSERT_MEM_EQ(
			signs1, signs2, (size_t)((dim + 7) / 8),
			"same seed produces same sign vector");

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
	mkt_free(signs1);
	mkt_free(signs2);
}

/* In-place aliasing: out == in must work. */
TEST(in_place)
{
	const Dimension dim = 32;
	uint8_t *signs = mkt_alloc((dim + 7) / 8);
	MktFastRotateParams p;
	mkt_fast_rotate_init(&p, dim, 99, signs);

	float x[32], copy[32], copy_out[32];
	for (Dimension i = 0; i < dim; i++)
		copy[i] = x[i] = (float) (i * 0.3f - 5.0f);

	mkt_fast_rotate_apply(&p, x, x);
	mkt_fast_rotate_apply(&p, copy, copy_out);

	for (Dimension i = 0; i < dim; i++)
		ASSERT_FLOAT_EQ(copy_out[i], x[i], 1e-6f,
						"in-place result matches out-of-place");

	mkt_free(signs);
}
