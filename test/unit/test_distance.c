/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * test_distance.c - Comprehensive tests for distance computation
 *
 * Tests cover:
 * - Known value tests (hand-calculated expected results)
 * - SIMD vs scalar comparison (bit-exact matching)
 * - Edge cases (dimension mismatch, zero vectors, unaligned access)
 * - Batch vs single-pair consistency
 * - All metrics (L2, IP, cosine)
 * - All SIMD implementations (using override mechanism)
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "algo/distance.h"
#include "core/memory.h"
#include "core/platform.h"
#include "test_config.h"
#include "vs_test.h"

TEST_GROUP(Distance);

/*
 * Helper to re-initialize distance system with specific SIMD override.
 * Used to test all SIMD implementations.
 */
static void
reinit_distance_with_simd(uint32_t simd_mask)
{
	/* Set SIMD override and reset caches */
	vs_simd_set_override(simd_mask);
	vs_simd_reset_cache();
	vs_distance_force_reinit();

	/* Re-initialize with new SIMD setting */
	int ret = vs_distance_init();
	if (ret != 0)
	{
		fprintf(stderr, "distance init failed: %d\n", ret);
		abort();
	}
}

static void
group_setup(void)
{
	/* Start with auto-detection */
	reinit_distance_with_simd(0xFFFFFFFF);
}

static void
group_teardown(void)
{
	/* Restore auto-detection */
	reinit_distance_with_simd(0xFFFFFFFF);
}

GROUP_FIXTURE(group_setup, group_teardown);
TEST_MEMCTX_FIXTURE();

/*
 * Helper: allocate and fill test vectors with integer values for exact
 * float representation
 */
static float *
alloc_test_vector(Dimension dim, int offset)
{
	float *data = vs_alloc(dim * sizeof(float));
	for (Dimension i = 0; i < dim; i++)
		data[i] = (float)((i + offset) % 100);
	return data;
}

/*
 * Known Value Tests - hand-calculated expected results
 */

TEST(l2_known_values)
{
	float	 a_data[] = {1.0f, 2.0f, 3.0f};
	float	 b_data[] = {4.0f, 5.0f, 6.0f};
	Vec32Ref a		  = {.data = a_data, .dim = 3};
	Vec32Ref b		  = {.data = b_data, .dim = 3};

	/* (1-4)² + (2-5)² + (3-6)² = 9+9+9 = 27 */
	Distance d = vs_distance_l2(a, b);
	ASSERT_FLOAT_EQ(27.0f, d, 1e-5f, "L2 distance should be 27");
}

TEST(l2_identical_vectors)
{
	float	 a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
	Vec32Ref a		  = {.data = a_data, .dim = 4};
	Vec32Ref b		  = {.data = a_data, .dim = 4};

	Distance d = vs_distance_l2(a, b);
	ASSERT_FLOAT_EQ(0.0f, d, 1e-6f, "identical vectors have zero distance");
}

TEST(ip_known_values)
{
	float	 a_data[] = {1.0f, 2.0f, 3.0f};
	float	 b_data[] = {4.0f, 5.0f, 6.0f};
	Vec32Ref a		  = {.data = a_data, .dim = 3};
	Vec32Ref b		  = {.data = b_data, .dim = 3};

	/* 1*4 + 2*5 + 3*6 = 4+10+18 = 32, negated = -32 */
	Distance d = vs_distance_ip(a, b);
	ASSERT_FLOAT_EQ(-32.0f, d, 1e-5f, "IP distance should be -32");
}

TEST(ip_orthogonal)
{
	float	 a_data[] = {1.0f, 0.0f, 0.0f};
	float	 b_data[] = {0.0f, 1.0f, 0.0f};
	Vec32Ref a		  = {.data = a_data, .dim = 3};
	Vec32Ref b		  = {.data = b_data, .dim = 3};

	Distance d = vs_distance_ip(a, b);
	ASSERT_FLOAT_EQ(0.0f, d, 1e-6f, "orthogonal vectors");
}

TEST(cosine_identical_normalized)
{
	float	 a_data[] = {0.6f, 0.8f, 0.0f};
	Vec32Ref a		  = {.data = a_data, .dim = 3};
	Vec32Ref b		  = {.data = a_data, .dim = 3};

	Distance d = vs_distance_cosine(a, b);
	ASSERT_FLOAT_EQ(0.0f, d, 1e-6f, "identical vectors have zero distance");
}

TEST(cosine_orthogonal)
{
	float	 a_data[] = {1.0f, 0.0f, 0.0f};
	float	 b_data[] = {0.0f, 1.0f, 0.0f};
	Vec32Ref a		  = {.data = a_data, .dim = 3};
	Vec32Ref b		  = {.data = b_data, .dim = 3};

	Distance d = vs_distance_cosine(a, b);
	ASSERT_FLOAT_EQ(1.0f, d, 1e-6f, "orthogonal vectors");
}

TEST(cosine_opposite)
{
	float	 a_data[] = {1.0f, 0.0f, 0.0f};
	float	 b_data[] = {-1.0f, 0.0f, 0.0f};
	Vec32Ref a		  = {.data = a_data, .dim = 3};
	Vec32Ref b		  = {.data = b_data, .dim = 3};

	Distance d = vs_distance_cosine(a, b);
	ASSERT_FLOAT_EQ(2.0f, d, 1e-6f, "opposite vectors");
}

TEST(cosine_zero_vectors)
{
	float	 a_data[] = {0.0f, 0.0f, 0.0f};
	float	 b_data[] = {1.0f, 2.0f, 3.0f};
	Vec32Ref a		  = {.data = a_data, .dim = 3};
	Vec32Ref b		  = {.data = b_data, .dim = 3};

	Distance d = vs_distance_cosine(a, b);
	ASSERT_FLOAT_EQ(1.0f, d, 1e-6f, "zero vector has max distance");
}

#if HAVE_GSL
/*
 * GSL Reference Tests
 *
 * These tests validate our implementations against GSL (GNU Scientific
 * Library). GSL is a well-tested, industry-standard library for scientific
 * computing.
 */

#include <gsl/gsl_blas.h>
#include <gsl/gsl_vector.h>

/*
 * Helper to compute reference distances using GSL
 */
static void
compute_gsl_reference(
		const float *a_data,
		const float *b_data,
		Dimension	 dim,
		float		*ref_l2,
		float		*ref_ip,
		float		*ref_cosine)
{
	/* Create GSL vectors */
	gsl_vector *a = gsl_vector_alloc(dim);
	gsl_vector *b = gsl_vector_alloc(dim);

	for (Dimension i = 0; i < dim; i++)
	{
		gsl_vector_set(a, i, a_data[i]);
		gsl_vector_set(b, i, b_data[i]);
	}

	/* L2 squared distance: ||a - b||² */
	gsl_vector *diff = gsl_vector_alloc(dim);
	gsl_vector_memcpy(diff, a);
	gsl_vector_sub(diff, b);
	double l2_sq;
	gsl_blas_ddot(diff, diff, &l2_sq);
	*ref_l2 = (float)l2_sq;

	/* Inner product: -a·b */
	double dot;
	gsl_blas_ddot(a, b, &dot);
	*ref_ip = (float)(-dot);

	/* Cosine distance: 1 - (a·b)/(||a|| ||b||) */
	double norm_a = gsl_blas_dnrm2(a);
	double norm_b = gsl_blas_dnrm2(b);

	if (norm_a < 1e-8 || norm_b < 1e-8)
	{
		*ref_cosine = 1.0f; /* Max distance for zero vectors */
	}
	else
	{
		double cos_sim = dot / (norm_a * norm_b);
		*ref_cosine	   = (float)(1.0 - cos_sim);
	}

	gsl_vector_free(a);
	gsl_vector_free(b);
	gsl_vector_free(diff);
}

TEST(gsl_reference_simple_integers)
{
	float	 a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
	float	 b_data[] = {5.0f, 6.0f, 7.0f, 8.0f};
	Vec32Ref a		  = {.data = a_data, .dim = 4};
	Vec32Ref b		  = {.data = b_data, .dim = 4};

	float ref_l2, ref_ip, ref_cosine;
	compute_gsl_reference(a_data, b_data, 4, &ref_l2, &ref_ip, &ref_cosine);

	Distance d_l2	  = vs_distance_l2(a, b);
	Distance d_ip	  = vs_distance_ip(a, b);
	Distance d_cosine = vs_distance_cosine(a, b);

	ASSERT_FLOAT_EQ(ref_l2, d_l2, 1e-5f, "L2 vs GSL");
	ASSERT_FLOAT_EQ(ref_ip, d_ip, 1e-5f, "IP vs GSL");
	ASSERT_FLOAT_EQ(ref_cosine, d_cosine, 1e-5f, "Cosine vs GSL");
}

TEST(gsl_reference_random_vectors)
{
	/* Test with various dimensions */
	const Dimension dims[] = {8, 16, 32, 64, 128, 256};

	for (size_t d = 0; d < sizeof(dims) / sizeof(dims[0]); d++)
	{
		Dimension dim = dims[d];

		float *a_data = vs_alloc(dim * sizeof(float));
		float *b_data = vs_alloc(dim * sizeof(float));

		/* Fill with pseudo-random but deterministic values */
		for (Dimension i = 0; i < dim; i++)
		{
			a_data[i] = (float)((i * 17 + 23) % 100 - 50) / 10.0f;
			b_data[i] = (float)((i * 13 + 47) % 100 - 50) / 10.0f;
		}

		Vec32Ref a = {.data = a_data, .dim = dim};
		Vec32Ref b = {.data = b_data, .dim = dim};

		float ref_l2, ref_ip, ref_cosine;
		compute_gsl_reference(
				a_data, b_data, dim, &ref_l2, &ref_ip, &ref_cosine);

		Distance d_l2	  = vs_distance_l2(a, b);
		Distance d_ip	  = vs_distance_ip(a, b);
		Distance d_cosine = vs_distance_cosine(a, b);

		/* Looser tolerance for higher dimensions due to floating point
		 * accumulation, and never tighter than a few ulps of the value
		 * itself: a different summation order (SIMD width, or none at all)
		 * legitimately moves the last bit. */
		float abs_tol	= (dim >= 128) ? 2e-4f : (dim >= 64) ? 1e-4f : 1e-5f;
		float tolerance = fmaxf(abs_tol, fabsf(ref_l2) * 1e-6f);

		char msg[64];
		snprintf(msg, sizeof(msg), "dim=%u L2", dim);
		ASSERT_FLOAT_EQ(ref_l2, d_l2, tolerance, msg);

		tolerance = fmaxf(abs_tol, fabsf(ref_ip) * 1e-6f);
		snprintf(msg, sizeof(msg), "dim=%u IP", dim);
		ASSERT_FLOAT_EQ(ref_ip, d_ip, tolerance, msg);

		snprintf(msg, sizeof(msg), "dim=%u Cosine", dim);
		ASSERT_FLOAT_EQ(ref_cosine, d_cosine, tolerance, msg);
	}
}

#endif /* HAVE_GSL */

/*
 * SIMD vs Scalar Comparison Tests
 *
 * Test that SIMD implementations match scalar exactly using integer-based
 * test vectors (for exact float representation).
 */

/*
 * Helper to get SIMD mask and expected name from variant string.
 * Returns true if the variant is available, false if it should be skipped.
 */
static bool
get_simd_mask_for_variant(
		const char *variant, uint32_t *simd_mask, const char **expected_name)
{
	if (strcmp(variant, "compiler") == 0)
	{
		*expected_name = "compiler";
		*simd_mask	   = SIMD_NONE;
		return true; /* Always available */
	}
	else if (strcmp(variant, "avx2") == 0)
	{
#if !(defined(__x86_64__) || defined(_M_X64))
		return false; /* The AVX2 kernels are built for x86-64 only */
#endif
		SimdCapability caps = vs_detect_simd();
		if (!(caps & SIMD_AVX2))
			return false; /* Not available */

		*expected_name = "avx2";
		*simd_mask	   = SIMD_AVX2;
		return true;
	}
	else if (strcmp(variant, "avx512") == 0)
	{
#if !(defined(__x86_64__) || defined(_M_X64))
		return false; /* The AVX-512 kernels are built for x86-64 only */
#endif
		SimdCapability caps = vs_detect_simd();
		if ((caps & VS_SIMD_AVX512_DQ) != VS_SIMD_AVX512_DQ)
			return false; /* Not available */

		*expected_name = "avx512";
		*simd_mask	   = VS_SIMD_AVX512_DQ;
		return true;
	}
	else if (strcmp(variant, "neon") == 0)
	{
		SimdCapability caps = vs_detect_simd();
		if (!(caps & SIMD_NEON))
			return false; /* Not available */

		*expected_name = "neon";
		*simd_mask	   = SIMD_NEON;
		return true;
	}

	/* Unknown variant - use auto-detect */
	*expected_name = NULL;
	*simd_mask	   = 0xFFFFFFFF;
	return true;
}

/*
 * Macro to skip test if SIMD variant is not available.
 * Declares simd_mask and expected_name variables for use in the test.
 */
#define SKIP_IF_SIMD_VARIANT_NOT_AVAILABLE(variant)                      \
	const char *expected_name;                                           \
	uint32_t	simd_mask;                                               \
	if (!get_simd_mask_for_variant(variant, &simd_mask, &expected_name)) \
	{                                                                    \
		TEST_PRINT("%s not available, skipping\n", variant);             \
		return;                                                          \
	}

TEST_PARAMETERIZED(l2_simd_variant, "compiler", "avx2", "avx512", "neon")
{
	SKIP_IF_SIMD_VARIANT_NOT_AVAILABLE(param);

	/* Test dimensions that exercise different code paths */
	const Dimension dims[] = {1,   3,	7,	 8,	  15,  16,	17,	 31,  32,
							  33,  63,	64,	 65,  127, 128, 129, 255, 256,
							  257, 383, 384, 385, 767, 768, 769};

	reinit_distance_with_simd(simd_mask);
	const char *impl_name = vs_distance_impl_name();

	/* Verify we got the expected implementation */
	if (expected_name != NULL)
		ASSERT_STR_EQ(expected_name, impl_name, "should use expected impl");

	/* Get scalar reference for comparison (unless we ARE testing scalar) */
	reinit_distance_with_simd(SIMD_NONE);

	for (size_t t = 0; t < sizeof(dims) / sizeof(dims[0]); t++)
	{
		Dimension dim = dims[t];

		float *a_data = alloc_test_vector(dim, 0);
		float *b_data = alloc_test_vector(dim, 17);

		Vec32Ref a = {.data = a_data, .dim = dim};
		Vec32Ref b = {.data = b_data, .dim = dim};

		Distance d_scalar = vs_distance_l2(a, b);

		/* Now test the variant */
		reinit_distance_with_simd(simd_mask);
		Distance d_variant = vs_distance_l2(a, b);

		char msg[128];
		snprintf(
				msg,
				sizeof(msg),
				"dim=%u: %s should match scalar",
				dim,
				param);
		ASSERT_FLOAT_EQ(d_scalar, d_variant, 1e-5f, msg);
	}

	/* Restore auto-detection */
	reinit_distance_with_simd(0xFFFFFFFF);
}

TEST_PARAMETERIZED(ip_simd_variant, "compiler", "avx2", "avx512", "neon")
{
	SKIP_IF_SIMD_VARIANT_NOT_AVAILABLE(param);

	const Dimension dims[] =
			{8, 16, 17, 32, 33, 64, 65, 128, 129, 256, 257, 768, 769};

	reinit_distance_with_simd(simd_mask);
	const char *impl_name = vs_distance_impl_name();

	if (expected_name != NULL)
		ASSERT_STR_EQ(expected_name, impl_name, "should use expected impl");

	reinit_distance_with_simd(SIMD_NONE);

	for (size_t t = 0; t < sizeof(dims) / sizeof(dims[0]); t++)
	{
		Dimension dim = dims[t];

		float *a_data = alloc_test_vector(dim, 0);
		float *b_data = alloc_test_vector(dim, 23);

		Vec32Ref a = {.data = a_data, .dim = dim};
		Vec32Ref b = {.data = b_data, .dim = dim};

		Distance d_scalar = vs_distance_ip(a, b);

		reinit_distance_with_simd(simd_mask);
		Distance d_variant = vs_distance_ip(a, b);

		char msg[128];
		snprintf(
				msg,
				sizeof(msg),
				"dim=%u: %s should match scalar",
				dim,
				param);
		ASSERT_FLOAT_EQ(d_scalar, d_variant, 1e-4f, msg);
	}

	reinit_distance_with_simd(0xFFFFFFFF);
}

TEST_PARAMETERIZED(cosine_simd_variant, "compiler", "avx2", "avx512", "neon")
{
	SKIP_IF_SIMD_VARIANT_NOT_AVAILABLE(param);

	const Dimension dims[] =
			{8, 16, 17, 32, 33, 64, 65, 128, 129, 256, 257, 768};

	reinit_distance_with_simd(simd_mask);
	const char *impl_name = vs_distance_impl_name();

	if (expected_name != NULL)
		ASSERT_STR_EQ(expected_name, impl_name, "should use expected impl");

	reinit_distance_with_simd(SIMD_NONE);

	for (size_t t = 0; t < sizeof(dims) / sizeof(dims[0]); t++)
	{
		Dimension dim = dims[t];

		float *a_data = alloc_test_vector(dim, 1);
		float *b_data = alloc_test_vector(dim, 13);

		Vec32Ref a = {.data = a_data, .dim = dim};
		Vec32Ref b = {.data = b_data, .dim = dim};

		Distance d_scalar = vs_distance_cosine(a, b);

		reinit_distance_with_simd(simd_mask);
		Distance d_variant = vs_distance_cosine(a, b);

		char msg[128];
		snprintf(
				msg,
				sizeof(msg),
				"dim=%u: %s should match scalar",
				dim,
				param);
		ASSERT_FLOAT_EQ(d_scalar, d_variant, 1e-5f, msg);
	}

	reinit_distance_with_simd(0xFFFFFFFF);
}

/*
 * Edge Case Tests
 */

TEST(dimension_mismatch)
{
	float	 a_data[] = {1.0f, 2.0f, 3.0f};
	float	 b_data[] = {4.0f, 5.0f};
	Vec32Ref a		  = {.data = a_data, .dim = 3};
	Vec32Ref b		  = {.data = b_data, .dim = 2};

	Distance d = vs_distance_l2(a, b);
	ASSERT_FLOAT_EQ(-1.0f, d, 1e-6f, "should return error");
}

TEST(null_pointer_a)
{
	float	 b_data[] = {1.0f, 2.0f, 3.0f};
	Vec32Ref a		  = {.data = NULL, .dim = 3};
	Vec32Ref b		  = {.data = b_data, .dim = 3};

	Distance d = vs_distance_l2(a, b);
	ASSERT_FLOAT_EQ(-1.0f, d, 1e-6f, "should return error");
}

TEST(null_pointer_b)
{
	float	 a_data[] = {1.0f, 2.0f, 3.0f};
	Vec32Ref a		  = {.data = a_data, .dim = 3};
	Vec32Ref b		  = {.data = NULL, .dim = 3};

	Distance d = vs_distance_l2(a, b);
	ASSERT_FLOAT_EQ(-1.0f, d, 1e-6f, "should return error");
}

TEST(zero_dimension)
{
	float	 a_data[] = {1.0f};
	float	 b_data[] = {2.0f};
	Vec32Ref a		  = {.data = a_data, .dim = 0};
	Vec32Ref b		  = {.data = b_data, .dim = 0};

	Distance d = vs_distance_l2(a, b);
	ASSERT_FLOAT_EQ(-1.0f, d, 1e-6f, "should return error");
}

TEST(unaligned_access)
{
	/* Allocate with offset to force misalignment */
	float *storage = vs_alloc(128 * sizeof(float) + 4);
	ASSERT_NOT_NULL(storage, "malloc should succeed");

	float *a_data = (float *)((char *)storage + 1);
	float *b_data = a_data + 64;

	/* Use memcpy with byte-level pointer arithmetic to avoid UBSan warnings */
	for (int i = 0; i < 64; i++)
	{
		float val_a = (float)i;
		float val_b = (float)(i + 1);
		memcpy((char *)a_data + i * sizeof(float), &val_a, sizeof(float));
		memcpy((char *)b_data + i * sizeof(float), &val_b, sizeof(float));
	}

	Vec32Ref a = {.data = a_data, .dim = 64};
	Vec32Ref b = {.data = b_data, .dim = 64};

	Distance d = vs_distance_l2(a, b);
	ASSERT_FLOAT_EQ(64.0f, d, 1e-5f, "unaligned should work");
}

TEST(null_pointer_ip)
{
	float	 a_data[] = {1.0f, 2.0f, 3.0f};
	Vec32Ref a		  = {.data = a_data, .dim = 3};
	Vec32Ref b		  = {.data = NULL, .dim = 3};

	Distance d = vs_distance_ip(a, b);
	ASSERT_FLOAT_EQ(-1.0f, d, 1e-6f, "IP should return error for null");
}

TEST(null_pointer_cosine)
{
	float	 a_data[] = {1.0f, 2.0f, 3.0f};
	Vec32Ref a		  = {.data = NULL, .dim = 3};
	Vec32Ref b		  = {.data = a_data, .dim = 3};

	Distance d = vs_distance_cosine(a, b);
	ASSERT_FLOAT_EQ(-1.0f, d, 1e-6f, "cosine should return error for null");
}

TEST(cosine_zero_vector)
{
	float	 a_data[] = {0.0f, 0.0f, 0.0f};
	float	 b_data[] = {1.0f, 2.0f, 3.0f};
	Vec32Ref a		  = {.data = a_data, .dim = 3};
	Vec32Ref b		  = {.data = b_data, .dim = 3};

	/* Zero vector should return maximum distance (1.0) */
	Distance d = vs_distance_cosine(a, b);
	ASSERT_FLOAT_EQ(1.0f, d, 1e-6f, "zero vector should give max distance");
}

TEST(cosine_both_zero_vectors)
{
	float	 data[] = {0.0f, 0.0f, 0.0f};
	Vec32Ref a		= {.data = data, .dim = 3};
	Vec32Ref b		= {.data = data, .dim = 3};

	Distance d = vs_distance_cosine(a, b);
	ASSERT_FLOAT_EQ(1.0f, d, 1e-6f, "both zero should give max distance");
}

TEST(invalid_metric_enum)
{
	float		   a_data[] = {1.0f, 2.0f, 3.0f};
	float		   b_data[] = {4.0f, 5.0f, 6.0f};
	Vec32Ref	   a		= {.data = a_data, .dim = 3};
	Vec32Ref	   b		= {.data = b_data, .dim = 3};
	DistanceMetric invalid	= (DistanceMetric)999;

	Distance d = vs_distance(a, b, invalid);
	ASSERT_FLOAT_EQ(-1.0f, d, 1e-6f, "invalid metric should return error");
}

/*
 * Batch Tests - verify batch operations match single-pair
 */

TEST(batch_l2_matches_single_pair)
{
	Dimension dim	= 128;
	uint32_t  count = 10;

	float	 *query		= alloc_test_vector(dim, 0);
	float	 *vectors	= vs_alloc(count * dim * sizeof(float));
	Distance *distances = vs_alloc(count * sizeof(Distance));

	for (uint32_t v = 0; v < count; v++)
		for (Dimension i = 0; i < dim; i++)
			vectors[v * dim + i] = (float)(v * 10 + i);

	Vec32Ref q = {.data = query, .dim = dim};

	int ret = vs_distance_batch_l2(q, vectors, count, dim, distances);
	ASSERT_EQ(0, ret, "batch should succeed");

	for (uint32_t v = 0; v < count; v++)
	{
		Vec32Ref vec	  = {.data = vectors + v * dim, .dim = dim};
		Distance d_single = vs_distance_l2(q, vec);

		char msg[128];
		snprintf(msg, sizeof(msg), "batch[%u] should match single-pair", v);
		ASSERT_FLOAT_EQ(d_single, distances[v], 1e-5f, msg);
	}
}

TEST(batch_ip_matches_single_pair)
{
	Dimension dim	= 64;
	uint32_t  count = 5;

	float	 *query		= alloc_test_vector(dim, 5);
	float	 *vectors	= vs_alloc(count * dim * sizeof(float));
	Distance *distances = vs_alloc(count * sizeof(Distance));

	for (uint32_t v = 0; v < count; v++)
		for (Dimension i = 0; i < dim; i++)
			vectors[v * dim + i] = (float)(v * 5 + i + 1);

	Vec32Ref q = {.data = query, .dim = dim};

	int ret = vs_distance_batch_ip(q, vectors, count, dim, distances);
	ASSERT_EQ(0, ret, "batch should succeed");

	for (uint32_t v = 0; v < count; v++)
	{
		Vec32Ref vec	  = {.data = vectors + v * dim, .dim = dim};
		Distance d_single = vs_distance_ip(q, vec);

		char msg[128];
		snprintf(msg, sizeof(msg), "batch[%u] should match single-pair", v);
		ASSERT_FLOAT_EQ(d_single, distances[v], 1e-4f, msg);
	}
}

TEST(batch_cosine_matches_single_pair)
{
	Dimension dim	= 32;
	uint32_t  count = 8;

	float	 *query		= alloc_test_vector(dim, 3);
	float	 *vectors	= vs_alloc(count * dim * sizeof(float));
	Distance *distances = vs_alloc(count * sizeof(Distance));

	for (uint32_t v = 0; v < count; v++)
		for (Dimension i = 0; i < dim; i++)
			vectors[v * dim + i] = (float)(v * 3 + i + 2);

	Vec32Ref q = {.data = query, .dim = dim};

	int ret = vs_distance_batch_cosine(q, vectors, count, dim, distances);
	ASSERT_EQ(0, ret, "batch should succeed");

	for (uint32_t v = 0; v < count; v++)
	{
		Vec32Ref vec	  = {.data = vectors + v * dim, .dim = dim};
		Distance d_single = vs_distance_cosine(q, vec);

		char msg[128];
		snprintf(msg, sizeof(msg), "batch[%u] should match single-pair", v);
		ASSERT_FLOAT_EQ(d_single, distances[v], 1e-5f, msg);
	}
}

TEST(batch_dimension_mismatch)
{
	float	 query_data[]	= {1.0f, 2.0f, 3.0f};
	float	 vectors_data[] = {4.0f, 5.0f, 6.0f, 7.0f};
	Distance distances[2];

	Vec32Ref q = {.data = query_data, .dim = 3};

	int ret = vs_distance_batch_l2(q, vectors_data, 2, 2, distances);
	ASSERT_EQ(-1, ret, "should return error on dimension mismatch");
}

TEST(batch_null_query_data)
{
	float	 vectors_data[] = {1.0f, 2.0f, 3.0f};
	Distance distances[1];
	Vec32Ref q = {.data = NULL, .dim = 3};

	int ret = vs_distance_batch_l2(q, vectors_data, 1, 3, distances);
	ASSERT_EQ(-1, ret, "should return error for null query data");
}

TEST(batch_null_vectors)
{
	float	 query_data[] = {1.0f, 2.0f, 3.0f};
	Distance distances[1];
	Vec32Ref q = {.data = query_data, .dim = 3};

	int ret = vs_distance_batch_l2(q, NULL, 1, 3, distances);
	ASSERT_EQ(-1, ret, "should return error for null vectors");
}

TEST(batch_null_distances)
{
	float	 query_data[]	= {1.0f, 2.0f, 3.0f};
	float	 vectors_data[] = {4.0f, 5.0f, 6.0f};
	Vec32Ref q				= {.data = query_data, .dim = 3};

	int ret = vs_distance_batch_l2(q, vectors_data, 1, 3, NULL);
	ASSERT_EQ(-1, ret, "should return error for null distances array");
}

TEST(batch_ip_null_vectors)
{
	float	 query_data[] = {1.0f, 2.0f, 3.0f};
	Distance distances[1];
	Vec32Ref q = {.data = query_data, .dim = 3};

	int ret = vs_distance_batch_ip(q, NULL, 1, 3, distances);
	ASSERT_EQ(-1, ret, "IP batch should return error for null vectors");
}

TEST(batch_cosine_null_distances)
{
	float	 query_data[]	= {1.0f, 2.0f, 3.0f};
	float	 vectors_data[] = {4.0f, 5.0f, 6.0f};
	Vec32Ref q				= {.data = query_data, .dim = 3};

	int ret = vs_distance_batch_cosine(q, vectors_data, 1, 3, NULL);
	ASSERT_EQ(-1, ret, "cosine batch should return error for null distances");
}

TEST(batch_odd_dimension_l2)
{
	/* Dimension 17 is not a multiple of 16 (AVX-512) or 8 (AVX2) */
	Dimension dim	= 17;
	uint32_t  count = 5;

	float	 *query		= alloc_test_vector(dim, 0);
	float	 *vectors	= vs_alloc(count * dim * sizeof(float));
	Distance *distances = vs_alloc(count * sizeof(Distance));

	for (uint32_t v = 0; v < count; v++)
		for (Dimension i = 0; i < dim; i++)
			vectors[v * dim + i] = (float)(v * 7 + i);

	Vec32Ref q = {.data = query, .dim = dim};

	int ret = vs_distance_batch_l2(q, vectors, count, dim, distances);
	ASSERT_EQ(0, ret, "batch with odd dimension should succeed");

	/* Verify against single-pair */
	for (uint32_t v = 0; v < count; v++)
	{
		Vec32Ref vec	  = {.data = vectors + v * dim, .dim = dim};
		Distance d_single = vs_distance_l2(q, vec);

		char msg[128];
		snprintf(msg, sizeof(msg), "batch[%u] odd dim should match", v);
		ASSERT_FLOAT_EQ(d_single, distances[v], 1e-5f, msg);
	}
}

TEST(batch_odd_dimension_ip)
{
	/* Dimension 13 exercises tail loops differently */
	Dimension dim	= 13;
	uint32_t  count = 3;

	float	 *query		= alloc_test_vector(dim, 1);
	float	 *vectors	= vs_alloc(count * dim * sizeof(float));
	Distance *distances = vs_alloc(count * sizeof(Distance));

	for (uint32_t v = 0; v < count; v++)
		for (Dimension i = 0; i < dim; i++)
			vectors[v * dim + i] = (float)(v * 5 + i + 2);

	Vec32Ref q = {.data = query, .dim = dim};

	int ret = vs_distance_batch_ip(q, vectors, count, dim, distances);
	ASSERT_EQ(0, ret, "batch with odd dimension should succeed");

	for (uint32_t v = 0; v < count; v++)
	{
		Vec32Ref vec	  = {.data = vectors + v * dim, .dim = dim};
		Distance d_single = vs_distance_ip(q, vec);

		char msg[128];
		snprintf(msg, sizeof(msg), "batch[%u] odd dim IP should match", v);
		ASSERT_FLOAT_EQ(d_single, distances[v], 1e-4f, msg);
	}
}

TEST(batch_odd_dimension_cosine)
{
	/* Dimension 25 for cosine */
	Dimension dim	= 25;
	uint32_t  count = 4;

	float	 *query		= alloc_test_vector(dim, 2);
	float	 *vectors	= vs_alloc(count * dim * sizeof(float));
	Distance *distances = vs_alloc(count * sizeof(Distance));

	for (uint32_t v = 0; v < count; v++)
		for (Dimension i = 0; i < dim; i++)
			vectors[v * dim + i] = (float)(v * 3 + i + 1);

	Vec32Ref q = {.data = query, .dim = dim};

	int ret = vs_distance_batch_cosine(q, vectors, count, dim, distances);
	ASSERT_EQ(0, ret, "batch with odd dimension should succeed");

	for (uint32_t v = 0; v < count; v++)
	{
		Vec32Ref vec	  = {.data = vectors + v * dim, .dim = dim};
		Distance d_single = vs_distance_cosine(q, vec);

		char msg[128];
		snprintf(msg, sizeof(msg), "batch[%u] odd dim cosine should match", v);
		ASSERT_FLOAT_EQ(d_single, distances[v], 1e-5f, msg);
	}
}

TEST(scalar_null_pointer_checks)
{
	/* Force scalar implementation to test its error paths */
	reinit_distance_with_simd(SIMD_NONE);

	float	 a_data[] = {1.0f, 2.0f, 3.0f};
	Vec32Ref a		  = {.data = a_data, .dim = 3};
	Vec32Ref b_null	  = {.data = NULL, .dim = 3};

	Distance d = vs_distance_l2(a, b_null);
	ASSERT_FLOAT_EQ(-1.0f, d, 1e-6f, "scalar L2 should error on null");

	d = vs_distance_ip(a, b_null);
	ASSERT_FLOAT_EQ(-1.0f, d, 1e-6f, "scalar IP should error on null");

	d = vs_distance_cosine(a, b_null);
	ASSERT_FLOAT_EQ(-1.0f, d, 1e-6f, "scalar cosine should error on null");

	/* Restore auto-detection */
	reinit_distance_with_simd(0xFFFFFFFF);
}

TEST(scalar_cosine_zero_vector)
{
	/* Force scalar to test zero vector handling in cosine */
	reinit_distance_with_simd(SIMD_NONE);

	float	 zero[] = {0.0f, 0.0f, 0.0f};
	float	 norm[] = {1.0f, 2.0f, 3.0f};
	Vec32Ref a		= {.data = zero, .dim = 3};
	Vec32Ref b		= {.data = norm, .dim = 3};

	Distance d = vs_distance_cosine(a, b);
	ASSERT_FLOAT_EQ(1.0f, d, 1e-6f, "scalar cosine zero vector");

	/* Restore auto-detection */
	reinit_distance_with_simd(0xFFFFFFFF);
}

/*
 * Generic distance function tests
 */

TEST(generic_distance_l2)
{
	float	 a_data[] = {1.0f, 2.0f, 3.0f};
	float	 b_data[] = {4.0f, 5.0f, 6.0f};
	Vec32Ref a		  = {.data = a_data, .dim = 3};
	Vec32Ref b		  = {.data = b_data, .dim = 3};

	Distance d		  = vs_distance(a, b, DISTANCE_L2);
	Distance d_direct = vs_distance_l2(a, b);

	ASSERT_FLOAT_EQ(d_direct, d, 1e-6f, "generic should match direct");
}

TEST(generic_distance_ip)
{
	float	 a_data[] = {1.0f, 2.0f, 3.0f};
	float	 b_data[] = {4.0f, 5.0f, 6.0f};
	Vec32Ref a		  = {.data = a_data, .dim = 3};
	Vec32Ref b		  = {.data = b_data, .dim = 3};

	Distance d		  = vs_distance(a, b, DISTANCE_INNER_PRODUCT);
	Distance d_direct = vs_distance_ip(a, b);

	ASSERT_FLOAT_EQ(d_direct, d, 1e-6f, "generic should match direct");
}

TEST(generic_distance_cosine)
{
	float	 a_data[] = {1.0f, 2.0f, 3.0f};
	float	 b_data[] = {4.0f, 5.0f, 6.0f};
	Vec32Ref a		  = {.data = a_data, .dim = 3};
	Vec32Ref b		  = {.data = b_data, .dim = 3};

	Distance d		  = vs_distance(a, b, DISTANCE_COSINE);
	Distance d_direct = vs_distance_cosine(a, b);

	ASSERT_FLOAT_EQ(d_direct, d, 1e-6f, "generic should match direct");
}

/*
 * Implementation name test
 */

TEST(impl_name_is_valid)
{
	const char *name = vs_distance_impl_name();
	ASSERT_NOT_NULL(name, "implementation name should not be null");

	/* Should be one of the known implementations */
	int valid =
			(strcmp(name, "avx512") == 0 || strcmp(name, "avx2") == 0 ||
			 strcmp(name, "neon") == 0 || strcmp(name, "compiler") == 0 ||
			 strcmp(name, "none") == 0);

	char msg[128];
	snprintf(msg, sizeof(msg), "unknown implementation: %s", name);
	ASSERT_TRUE(valid, msg);
}

/*
 * Explicit SIMD Implementation Tests
 *
 * These tests force specific SIMD implementations using the override
 * mechanism, ensuring all code paths are tested regardless of CPU.
 */

TEST(force_compiler_implementation)
{
	/* Force compiler mode */
	reinit_distance_with_simd(SIMD_NONE);

	const char *name = vs_distance_impl_name();
	ASSERT_STR_EQ("compiler", name, "should use compiler implementation");

	/* Test basic functionality in scalar mode */
	float	 a_data[] = {1.0f, 2.0f, 3.0f};
	float	 b_data[] = {4.0f, 5.0f, 6.0f};
	Vec32Ref a		  = {.data = a_data, .dim = 3};
	Vec32Ref b		  = {.data = b_data, .dim = 3};

	Distance d = vs_distance_l2(a, b);
	ASSERT_FLOAT_EQ(27.0f, d, 1e-5f, "scalar L2 should work");

	/* Test batch operations with scalar (exercises fallback path) */
	const uint32_t	count	  = 5;
	const Dimension dim		  = 8;
	float		   *query	  = alloc_test_vector(dim, 0);
	float		   *vectors	  = vs_alloc(count * dim * sizeof(float));
	Distance	   *distances = vs_alloc(count * sizeof(Distance));

	for (uint32_t i = 0; i < count; i++)
		for (Dimension j = 0; j < dim; j++)
			vectors[i * dim + j] = (float)(i * 2 + j);

	Vec32Ref q = {.data = query, .dim = dim};

	/* Test all batch operations in scalar mode */
	int ret = vs_distance_batch_l2(q, vectors, count, dim, distances);
	ASSERT_EQ(0, ret, "scalar batch L2 should succeed");

	ret = vs_distance_batch_ip(q, vectors, count, dim, distances);
	ASSERT_EQ(0, ret, "scalar batch IP should succeed");

	ret = vs_distance_batch_cosine(q, vectors, count, dim, distances);
	ASSERT_EQ(0, ret, "scalar batch cosine should succeed");

	/* Restore auto-detection */
	reinit_distance_with_simd(0xFFFFFFFF);
}

#if defined(__x86_64__) || defined(_M_X64)

TEST(force_avx2_implementation)
{
	/* Check if AVX2 is available */
	SimdCapability caps = vs_detect_simd();
	if (!(caps & SIMD_AVX2))
	{
		TEST_PRINT("AVX2 not available, skipping test\n");
		return;
	}

	/* Force AVX2 mode (exclude AVX-512) */
	reinit_distance_with_simd(SIMD_AVX2);

	const char *name = vs_distance_impl_name();
	ASSERT_STR_EQ("avx2", name, "should use AVX2 implementation");

	/* Test all metrics with AVX2 - various dimensions to exercise all paths */
	/* Dimension 8: exact AVX2 width */
	float	 a_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
	float	 b_data[] = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
	Vec32Ref a		  = {.data = a_data, .dim = 8};
	Vec32Ref b		  = {.data = b_data, .dim = 8};

	Distance d_l2 = vs_distance_l2(a, b);
	ASSERT_FLOAT_EQ(8.0f, d_l2, 1e-5f, "AVX2 L2 should work");

	/* IP: 1*2 + 2*3 + 3*4 + 4*5 + 5*6 + 6*7 + 7*8 + 8*9 = 240 */
	Distance d_ip = vs_distance_ip(a, b);
	ASSERT_FLOAT_EQ(-240.0f, d_ip, 1e-5f, "AVX2 IP should work");

	Distance d_cos = vs_distance_cosine(a, b);
	ASSERT_TRUE(d_cos >= 0.0f && d_cos <= 1.0f, "AVX2 cosine in range");

	/* Test batch operations with AVX2 (this is where coverage was missing) */
	const uint32_t	count	  = 10;
	const Dimension dim		  = 16;
	float		   *query	  = alloc_test_vector(dim, 0);
	float		   *vectors	  = vs_alloc(count * dim * sizeof(float));
	Distance	   *distances = vs_alloc(count * sizeof(Distance));

	for (uint32_t i = 0; i < count; i++)
		for (Dimension j = 0; j < dim; j++)
			vectors[i * dim + j] = (float)(i + j);

	Vec32Ref q = {.data = query, .dim = dim};

	/* Test batch L2 */
	int ret = vs_distance_batch_l2(q, vectors, count, dim, distances);
	ASSERT_EQ(0, ret, "AVX2 batch L2 should succeed");

	/* Test batch IP */
	ret = vs_distance_batch_ip(q, vectors, count, dim, distances);
	ASSERT_EQ(0, ret, "AVX2 batch IP should succeed");

	/* Test batch cosine */
	ret = vs_distance_batch_cosine(q, vectors, count, dim, distances);
	ASSERT_EQ(0, ret, "AVX2 batch cosine should succeed");

	/* Restore */
	reinit_distance_with_simd(0xFFFFFFFF);
}

TEST(force_avx512_implementation)
{
	/* Check if AVX-512 is available */
	SimdCapability caps = vs_detect_simd();
	if ((caps & VS_SIMD_AVX512_DQ) != VS_SIMD_AVX512_DQ)
	{
		TEST_PRINT("AVX-512 not available, skipping test\n");
		return;
	}

	/* Force AVX-512 mode */
	reinit_distance_with_simd(VS_SIMD_AVX512_DQ);

	const char *name = vs_distance_impl_name();
	ASSERT_STR_EQ("avx512", name, "should use AVX-512 implementation");

	/* Test all metrics with AVX-512 */
	float a_data[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
	float b_data[16] =
			{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
	Vec32Ref a = {.data = a_data, .dim = 16};
	Vec32Ref b = {.data = b_data, .dim = 16};

	Distance d_l2 = vs_distance_l2(a, b);
	ASSERT_FLOAT_EQ(16.0f, d_l2, 1e-5f, "AVX-512 L2 should work");

	/* IP: sum(i * (i+1)) for i=1..16 = 1632 */
	Distance d_ip = vs_distance_ip(a, b);
	ASSERT_FLOAT_EQ(-1632.0f, d_ip, 1e-5f, "AVX-512 IP should work");

	Distance d_cos = vs_distance_cosine(a, b);
	ASSERT_TRUE(d_cos >= 0.0f && d_cos <= 1.0f, "AVX-512 cosine in range");

	/* Test batch operations with AVX-512 */
	const uint32_t	count	  = 10;
	const Dimension dim		  = 32;
	float		   *query	  = alloc_test_vector(dim, 0);
	float		   *vectors	  = vs_alloc(count * dim * sizeof(float));
	Distance	   *distances = vs_alloc(count * sizeof(Distance));

	for (uint32_t i = 0; i < count; i++)
		for (Dimension j = 0; j < dim; j++)
			vectors[i * dim + j] = (float)(i + j);

	Vec32Ref q = {.data = query, .dim = dim};

	/* Test batch L2 */
	int ret = vs_distance_batch_l2(q, vectors, count, dim, distances);
	ASSERT_EQ(0, ret, "AVX-512 batch L2 should succeed");

	/* Test batch IP */
	ret = vs_distance_batch_ip(q, vectors, count, dim, distances);
	ASSERT_EQ(0, ret, "AVX-512 batch IP should succeed");

	/* Test batch cosine */
	ret = vs_distance_batch_cosine(q, vectors, count, dim, distances);
	ASSERT_EQ(0, ret, "AVX-512 batch cosine should succeed");

	/* Restore */
	reinit_distance_with_simd(0xFFFFFFFF);
}

TEST(test_all_avx_dimensions)
{
	/* Test dimensions that exercise different code paths */
	const Dimension dims[] = {
			1,	 /* Scalar tail only */
			8,	 /* Exact AVX2 width */
			9,	 /* AVX2 + tail */
			16,	 /* Exact AVX-512 width */
			17,	 /* AVX-512 + tail */
			32,	 /* Multiple AVX-512 iterations */
			128, /* Typical embedding size */
			768	 /* Large embedding (BERT) */
	};

	for (size_t i = 0; i < sizeof(dims) / sizeof(dims[0]); i++)
	{
		Dimension dim = dims[i];

		float	*a	= alloc_test_vector(dim, 0);
		float	*b	= alloc_test_vector(dim, 1);
		Vec32Ref va = {.data = a, .dim = dim};
		Vec32Ref vb = {.data = b, .dim = dim};

		/* Test with each available SIMD level */
		SimdCapability caps = vs_detect_simd();

		/* Scalar */
		reinit_distance_with_simd(SIMD_NONE);
		Distance d_scalar = vs_distance_l2(va, vb);

		/* AVX2 (if available) */
		if (caps & SIMD_AVX2)
		{
			reinit_distance_with_simd(SIMD_AVX2);
			Distance d_avx2 = vs_distance_l2(va, vb);

			char msg[128];
			snprintf(
					msg,
					sizeof(msg),
					"AVX2 should match scalar at dim=%u",
					dim);
			ASSERT_FLOAT_EQ(d_scalar, d_avx2, 1e-4f, msg);
		}

		/* AVX-512 (if available) */
		if ((caps & VS_SIMD_AVX512_DQ) == VS_SIMD_AVX512_DQ)
		{
			reinit_distance_with_simd(VS_SIMD_AVX512_DQ);
			Distance d_avx512 = vs_distance_l2(va, vb);

			char msg[128];
			snprintf(
					msg,
					sizeof(msg),
					"AVX-512 should match scalar at dim=%u",
					dim);
			ASSERT_FLOAT_EQ(d_scalar, d_avx512, 1e-4f, msg);
		}
	}

	/* Restore auto-detection */
	reinit_distance_with_simd(0xFFFFFFFF);
}

#endif /* x86-64 */
