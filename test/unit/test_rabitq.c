/*
 * test_rabitq.c - Comprehensive tests for RaBitQ quantization
 *
 * Tests cover:
 * - Matrix generation and orthogonality
 * - Encoding correctness
 * - Distance estimation accuracy
 * - Lower bound guarantees
 * - SIMD implementation equivalence
 * - Edge cases
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "algo/distance.h"
#include "core/memory.h"
#include "core/platform.h"
#include "mkt_test.h"
#include "mkt_types.h"
#include "quant/rabitq.h"
#include "test_config.h"

TEST_GROUP(RaBitQ);

/*
 * Helper to re-initialize RaBitQ SIMD dispatch with specific override.
 */
static void
reinit_rabitq_with_simd(uint32_t simd_mask)
{
	mkt_simd_set_override(simd_mask);
	mkt_simd_reset_cache();
	mkt_rabitq_force_reinit();
	mkt_rabitq_init_simd();
}

static void
group_setup(void)
{
	reinit_rabitq_with_simd(0xFFFFFFFF);
	mkt_distance_init();
}

static void
group_teardown(void)
{
	reinit_rabitq_with_simd(0xFFFFFFFF);
}

GROUP_FIXTURE(group_setup, group_teardown);
TEST_MEMCTX_FIXTURE();

/*
 * Helper: allocate test vector with deterministic values
 */
static float *
alloc_test_vector(Dimension dim, int seed)
{
	float *data = mkt_alloc(dim * sizeof(float));
	for (Dimension i = 0; i < dim; i++)
		data[i] = (float)((i * 17 + seed) % 100 - 50) / 10.0f;
	return data;
}

/*
 * Helper: compute true L2 squared distance
 */
static float
true_l2_distance(VectorRef a, VectorRef b)
{
	float sum = 0.0f;
	for (Dimension i = 0; i < a.dim; i++)
	{
		float diff = a.data[i] - b.data[i];
		sum += diff * diff;
	}
	return sum;
}

/*
 * Matrix Tests
 */

/*
 * Test rotation-property tests via mkt_rabitq_rotate so they exercise
 * whichever orthonormal transform mkt_rabitq_init picked for the dim
 * (dense matrix when dim is unusual, randomized Hadamard otherwise).
 */
static double
sq_norm(const float *v, Dimension d)
{
	double s = 0;
	for (Dimension i = 0; i < d; i++)
		s += (double)v[i] * v[i];
	return s;
}

static void
fill_rotation_input(float *v, Dimension dim, int seed_offset)
{
	for (Dimension i = 0; i < dim; i++)
		v[i] = (float)(((i * 131 + seed_offset) % 211) - 105) * 0.1f;
}

TEST(rotation_preserves_norm_small)
{
	RaBitQParams *params = mkt_rabitq_create(8, 12345);
	ASSERT_NOT_NULL(params, "params should be created");

	float x[8], y[8];
	fill_rotation_input(x, 8, 1);
	mkt_rabitq_rotate(params, x, y);
	ASSERT_FLOAT_EQ(
			(float)sq_norm(x, 8),
			(float)sq_norm(y, 8),
			1e-3f,
			"rotation preserves ||x||²");

	mkt_rabitq_destroy(params);
}

TEST(rotation_preserves_norm_medium)
{
	RaBitQParams *params = mkt_rabitq_create(64, 54321);
	ASSERT_NOT_NULL(params, "params should be created");

	float x[64], y[64];
	fill_rotation_input(x, 64, 2);
	mkt_rabitq_rotate(params, x, y);
	ASSERT_FLOAT_EQ(
			(float)sq_norm(x, 64),
			(float)sq_norm(y, 64),
			1e-2f,
			"rotation preserves ||x||²");

	mkt_rabitq_destroy(params);
}

TEST(rotation_reproducibility)
{
	/* Same seed must produce the same rotation regardless of which
	 * kind (DENSE vs HADAMARD) the dim selects. */
	RaBitQParams *p1 = mkt_rabitq_create(16, 99999);
	RaBitQParams *p2 = mkt_rabitq_create(16, 99999);
	ASSERT_NOT_NULL(p1, "p1 should be created");
	ASSERT_NOT_NULL(p2, "p2 should be created");

	float x[16], y1[16], y2[16];
	fill_rotation_input(x, 16, 3);
	mkt_rabitq_rotate(p1, x, y1);
	mkt_rabitq_rotate(p2, x, y2);
	ASSERT_MEM_EQ(
			y1, y2, sizeof(y1), "same seed produces identical rotated output");

	mkt_rabitq_destroy(p1);
	mkt_rabitq_destroy(p2);
}

TEST(rotation_different_seeds)
{
	RaBitQParams *p1 = mkt_rabitq_create(16, 11111);
	RaBitQParams *p2 = mkt_rabitq_create(16, 22222);
	ASSERT_NOT_NULL(p1, "p1 should be created");
	ASSERT_NOT_NULL(p2, "p2 should be created");

	float x[16], y1[16], y2[16];
	fill_rotation_input(x, 16, 4);
	mkt_rabitq_rotate(p1, x, y1);
	mkt_rabitq_rotate(p2, x, y2);

	int different = 0;
	for (int i = 0; i < 16; i++)
		if (y1[i] != y2[i])
		{
			different = 1;
			break;
		}

	ASSERT_TRUE(different, "different seeds produce different rotations");

	mkt_rabitq_destroy(p1);
	mkt_rabitq_destroy(p2);
}

/* matrix_cblas_vs_builtin removed: with the dense sgemv path gone,
 * there is no cblas-vs-builtin code path to compare. */

/*
 * Encoding Tests
 */

TEST(encode_basic)
{
	Dimension	  dim	   = 16;
	RaBitQParams *params   = mkt_rabitq_create(dim, 42);
	float		 *input	   = alloc_test_vector(dim, 0);
	float		 *centroid = alloc_test_vector(dim, 50);

	ASSERT_NOT_NULL(params, "params should be created");

	VectorRef input_ref	   = {.data = input, .dim = dim};
	VectorRef centroid_ref = {.data = centroid, .dim = dim};

	RaBitQData *encoded = mkt_rabitq_encode(params, input_ref, centroid_ref);
	ASSERT_NOT_NULL(encoded, "encoding should succeed");

	/* Check that factors are finite */
	ASSERT_TRUE(isfinite(encoded->f_add), "f_add should be finite");
	ASSERT_TRUE(isfinite(encoded->f_rescale), "f_rescale should be finite");

	mkt_free(encoded);
	mkt_rabitq_destroy(params);
}

TEST(encode_into_preallocated)
{
	Dimension	  dim	   = 32;
	RaBitQParams *params   = mkt_rabitq_create(dim, 123);
	float		 *input	   = alloc_test_vector(dim, 10);
	float		 *centroid = alloc_test_vector(dim, 20);

	ASSERT_NOT_NULL(params, "params should be created");

	VectorRef input_ref	   = {.data = input, .dim = dim};
	VectorRef centroid_ref = {.data = centroid, .dim = dim};

	/* Allocate output buffer */
	size_t		size   = MKT_RABITQ_DATA_SIZE(dim);
	RaBitQData *output = mkt_alloc(size);
	ASSERT_NOT_NULL(output, "output should be allocated");

	int ret = mkt_rabitq_encode_into(params, input_ref, centroid_ref, output);
	ASSERT_EQ(0, ret, "encode_into should succeed");

	/* Check that factors are finite */
	ASSERT_TRUE(isfinite(output->f_add), "f_add should be finite");
	ASSERT_TRUE(isfinite(output->f_rescale), "f_rescale should be finite");

	mkt_free(output);
	mkt_rabitq_destroy(params);
}

TEST(encode_null_inputs)
{
	Dimension	  dim	   = 8;
	RaBitQParams *params   = mkt_rabitq_create(dim, 1);
	float		 *input	   = alloc_test_vector(dim, 0);
	float		 *centroid = alloc_test_vector(dim, 0);

	VectorRef input_ref	   = {.data = input, .dim = dim};
	VectorRef centroid_ref = {.data = centroid, .dim = dim};
	VectorRef null_ref	   = {.data = NULL, .dim = dim};

	/* Test null params */
	RaBitQData *enc = mkt_rabitq_encode(NULL, input_ref, centroid_ref);
	ASSERT_NULL(enc, "null params should return null");

	/* Test null input */
	enc = mkt_rabitq_encode(params, null_ref, centroid_ref);
	ASSERT_NULL(enc, "null input should return null");

	/* Test null centroid */
	enc = mkt_rabitq_encode(params, input_ref, null_ref);
	ASSERT_NULL(enc, "null centroid should return null");

	mkt_rabitq_destroy(params);
}

TEST(encode_dimension_mismatch)
{
	RaBitQParams *params   = mkt_rabitq_create(16, 1);
	float		 *input	   = alloc_test_vector(16, 0);
	float		 *centroid = alloc_test_vector(8, 0);

	VectorRef input_ref	   = {.data = input, .dim = 16};
	VectorRef centroid_ref = {.data = centroid, .dim = 8};

	RaBitQData *enc = mkt_rabitq_encode(params, input_ref, centroid_ref);
	ASSERT_NULL(enc, "dimension mismatch should return null");

	mkt_rabitq_destroy(params);
}

TEST(encode_batch_matches_single)
{
	/* Verify batch encoding produces identical results to single-vector */
	Dimension	  dim	   = 64;
	const int	  count	   = 8;
	RaBitQParams *params   = mkt_rabitq_create(dim, 42);
	float		 *centroid = alloc_test_vector(dim, 0);
	VectorRef	  cent_ref = {.data = centroid, .dim = dim};

	/* Allocate vectors */
	float *vectors = mkt_alloc((size_t)count * dim * sizeof(float));
	for (int i = 0; i < count; i++)
	{
		float *v = vectors + i * dim;
		for (Dimension j = 0; j < dim; j++)
			v[j] = (float)((i * 17 + j * 13) % 100 - 50) / 10.0f;
	}

	/* Encode with batch function into separate arrays */
	uint32_t packed_bytes	 = MKT_RABITQ_BYTES(dim);
	float	*batch_f_add	 = mkt_alloc(count * sizeof(float));
	float	*batch_f_rescale = mkt_alloc(count * sizeof(float));
	uint8_t *batch_bits		 = mkt_alloc((size_t)count * packed_bytes);
	int		 ret			 = mkt_rabitq_encode_batch(
			 params,
			 vectors,
			 MKT_VEC_F32,
			 cent_ref,
			 batch_f_add,
			 batch_f_rescale,
			 batch_bits,
			 count);
	ASSERT_EQ(0, ret, "batch encode should succeed");

	/* Encode individually and compare */
	for (int i = 0; i < count; i++)
	{
		VectorRef vec_ref = {.data = vectors + i * dim, .dim = dim};

		RaBitQData *single = mkt_rabitq_encode(params, vec_ref, cent_ref);
		ASSERT_NOT_NULL(single, "single encode should succeed");

		/* Compare factors (with tolerance for floating point variation)
		 * The FAISS-style dp_multiplier formula involves L1 norm computation
		 * which can have order-of-operations differences between single and
		 * batch encoding, leading to small variations. With CBLAS, these
		 * differences can be larger (~5e-4) due to optimized matrix ops.
		 */
		char msg[128];
		snprintf(msg, sizeof(msg), "vec %d: f_add should match", i);
		ASSERT_FLOAT_EQ(single->f_add, batch_f_add[i], 5e-4f, msg);
		snprintf(msg, sizeof(msg), "vec %d: f_rescale should match", i);
		ASSERT_FLOAT_EQ(single->f_rescale, batch_f_rescale[i], 5e-4f, msg);

		/* Compare bits */
		size_t bytes	  = MKT_RABITQ_BYTES(dim);
		int	   bits_match = memcmp(single->bits,
								   batch_bits + i * packed_bytes,
								   bytes) == 0;
		snprintf(msg, sizeof(msg), "vec %d: bits should match", i);
		ASSERT_TRUE(bits_match, msg);

		mkt_free(single);
	}

	mkt_free(batch_bits);
	mkt_free(batch_f_rescale);
	mkt_free(batch_f_add);
	mkt_free(vectors);
	mkt_rabitq_destroy(params);
}

TEST(encode_batch_null_inputs)
{
	Dimension	  dim	   = 16;
	RaBitQParams *params   = mkt_rabitq_create(dim, 1);
	float		 *vectors  = alloc_test_vector(dim * 4, 0);
	float		 *centroid = alloc_test_vector(dim, 0);
	VectorRef	  cent_ref = {.data = centroid, .dim = dim};

	uint32_t packed_bytes = MKT_RABITQ_BYTES(dim);
	float	*f_add		  = mkt_alloc(4 * sizeof(float));
	float	*f_rescale	  = mkt_alloc(4 * sizeof(float));
	uint8_t *bits		  = mkt_alloc(4 * packed_bytes);

	/* Test null params */
	int ret = mkt_rabitq_encode_batch(
			NULL, vectors, MKT_VEC_F32, cent_ref, f_add, f_rescale, bits, 4);
	ASSERT_EQ(-1, ret, "null params should fail");

	/* Test null vectors */
	ret = mkt_rabitq_encode_batch(
			params, NULL, MKT_VEC_F32, cent_ref, f_add, f_rescale, bits, 4);
	ASSERT_EQ(-1, ret, "null vectors should fail");

	/* Test null centroid */
	VectorRef null_cent = {.data = NULL, .dim = dim};
	ret					= mkt_rabitq_encode_batch(
			params,
			vectors,
			MKT_VEC_F32,
			null_cent,
			f_add,
			f_rescale,
			bits,
			4);
	ASSERT_EQ(-1, ret, "null centroid should fail");

	/* Test null f_add */
	ret = mkt_rabitq_encode_batch(
			params, vectors, MKT_VEC_F32, cent_ref, NULL, f_rescale, bits, 4);
	ASSERT_EQ(-1, ret, "null f_add should fail");

	/* Test null f_rescale */
	ret = mkt_rabitq_encode_batch(
			params, vectors, MKT_VEC_F32, cent_ref, f_add, NULL, bits, 4);
	ASSERT_EQ(-1, ret, "null f_rescale should fail");

	/* Test null bits */
	ret = mkt_rabitq_encode_batch(
			params, vectors, MKT_VEC_F32, cent_ref, f_add, f_rescale, NULL, 4);
	ASSERT_EQ(-1, ret, "null bits should fail");

	/* Test zero count */
	ret = mkt_rabitq_encode_batch(
			params, vectors, MKT_VEC_F32, cent_ref, f_add, f_rescale, bits, 0);
	ASSERT_EQ(-1, ret, "zero count should fail");

	mkt_free(bits);
	mkt_free(f_rescale);
	mkt_free(f_add);
	mkt_rabitq_destroy(params);
}

/*
 * Query Preparation Tests
 */

TEST(prepare_query_basic)
{
	Dimension	  dim	   = 16;
	RaBitQParams *params   = mkt_rabitq_create(dim, 42);
	float		 *query	   = alloc_test_vector(dim, 100);
	float		 *centroid = alloc_test_vector(dim, 50);

	VectorRef query_ref	   = {.data = query, .dim = dim};
	VectorRef centroid_ref = {.data = centroid, .dim = dim};

	RaBitQQueryState *state =
			mkt_rabitq_prepare_query(params, query_ref, centroid_ref);
	ASSERT_NOT_NULL(state, "prepare_query should succeed");

	ASSERT_EQ(dim, state->dim, "dimension should match");
	ASSERT_TRUE(isfinite(state->g_add), "g_add should be finite");
	ASSERT_TRUE(state->g_add >= 0, "g_add should be non-negative");
	ASSERT_TRUE(isfinite(state->g_error), "g_error should be finite");

	mkt_rabitq_free_query(state);
	mkt_rabitq_destroy(params);
}

TEST(prepare_query_null_inputs)
{
	Dimension	  dim	   = 8;
	RaBitQParams *params   = mkt_rabitq_create(dim, 1);
	float		 *query	   = alloc_test_vector(dim, 0);
	float		 *centroid = alloc_test_vector(dim, 0);

	VectorRef query_ref	   = {.data = query, .dim = dim};
	VectorRef centroid_ref = {.data = centroid, .dim = dim};
	VectorRef null_ref	   = {.data = NULL, .dim = dim};

	RaBitQQueryState *state =
			mkt_rabitq_prepare_query(NULL, query_ref, centroid_ref);
	ASSERT_NULL(state, "null params should return null");

	state = mkt_rabitq_prepare_query(params, null_ref, centroid_ref);
	ASSERT_NULL(state, "null query should return null");

	state = mkt_rabitq_prepare_query(params, query_ref, null_ref);
	ASSERT_NULL(state, "null centroid should return null");

	mkt_rabitq_destroy(params);
}

/*
 * Distance Computation Tests
 */

TEST(distance_basic)
{
	Dimension	  dim	   = 16;
	RaBitQParams *params   = mkt_rabitq_create(dim, 42);
	float		 *input	   = alloc_test_vector(dim, 0);
	float		 *query	   = alloc_test_vector(dim, 30);
	float		 *centroid = alloc_test_vector(dim, 50);

	VectorRef input_ref	   = {.data = input, .dim = dim};
	VectorRef query_ref	   = {.data = query, .dim = dim};
	VectorRef centroid_ref = {.data = centroid, .dim = dim};

	RaBitQData *encoded = mkt_rabitq_encode(params, input_ref, centroid_ref);
	ASSERT_NOT_NULL(encoded, "encoding should succeed");

	TEST_PRINT(
			"Encoded: f_add=%.4f, f_rescale=%.4f\n",
			encoded->f_add,
			encoded->f_rescale);

	RaBitQQueryState *state =
			mkt_rabitq_prepare_query(params, query_ref, centroid_ref);
	ASSERT_NOT_NULL(state, "prepare_query should succeed");

	TEST_PRINT(
			"Query: g_add=%.4f, g_error=%.4f, sum_transformed=%.4f\n",
			state->g_add,
			state->g_error,
			state->sum_transformed);

	Distance est_dist = mkt_rabitq_distance(state, encoded, dim);
	ASSERT_TRUE(isfinite(est_dist), "distance should be finite");

	/* Estimated distance should be reasonable (not wildly different from true)
	 */
	Distance true_dist = true_l2_distance(input_ref, query_ref);
	TEST_PRINT("True distance: %.4f, Estimated: %.4f\n", true_dist, est_dist);

	mkt_rabitq_free_query(state);
	mkt_free(encoded);
	mkt_rabitq_destroy(params);
}

TEST(distance_with_bound)
{
	Dimension	  dim	   = 32;
	RaBitQParams *params   = mkt_rabitq_create(dim, 42);
	float		 *input	   = alloc_test_vector(dim, 0);
	float		 *query	   = alloc_test_vector(dim, 30);
	float		 *centroid = alloc_test_vector(dim, 50);

	VectorRef input_ref	   = {.data = input, .dim = dim};
	VectorRef query_ref	   = {.data = query, .dim = dim};
	VectorRef centroid_ref = {.data = centroid, .dim = dim};

	RaBitQData *encoded = mkt_rabitq_encode(params, input_ref, centroid_ref);
	RaBitQQueryState *state =
			mkt_rabitq_prepare_query(params, query_ref, centroid_ref);

	Distance est_dist, lower_bound;
	mkt_rabitq_distance_with_bound(
			state, encoded, dim, &est_dist, &lower_bound);

	ASSERT_TRUE(isfinite(est_dist), "estimated distance should be finite");
	ASSERT_TRUE(isfinite(lower_bound), "lower bound should be finite");
	ASSERT_TRUE(
			lower_bound <= est_dist,
			"lower bound should be <= estimated distance");
	/* Note: lower bound can be negative when estimate is negative.
	 * In practice, use max(0, lower_bound) for distance comparisons. */

	/* Lower bound should be <= true distance (this is the key guarantee) */
	Distance true_dist = true_l2_distance(input_ref, query_ref);
	TEST_PRINT(
			"True: %.4f, Est: %.4f, Lower: %.4f\n",
			true_dist,
			est_dist,
			lower_bound);

	/* The lower bound guarantee: lower_bound <= true_dist */
	/* Allow small tolerance for numerical issues */
	ASSERT_TRUE(
			lower_bound <= true_dist + 1e-3f,
			"lower bound should be <= true distance");

	mkt_rabitq_free_query(state);
	mkt_free(encoded);
	mkt_rabitq_destroy(params);
}

TEST(distance_null_inputs)
{
	Dimension	  dim	   = 8;
	RaBitQParams *params   = mkt_rabitq_create(dim, 1);
	float		 *input	   = alloc_test_vector(dim, 0);
	float		 *query	   = alloc_test_vector(dim, 10);
	float		 *centroid = alloc_test_vector(dim, 5);

	VectorRef input_ref	   = {.data = input, .dim = dim};
	VectorRef query_ref	   = {.data = query, .dim = dim};
	VectorRef centroid_ref = {.data = centroid, .dim = dim};

	RaBitQData *encoded = mkt_rabitq_encode(params, input_ref, centroid_ref);
	RaBitQQueryState *state =
			mkt_rabitq_prepare_query(params, query_ref, centroid_ref);

	/* Test null inputs */
	Distance d = mkt_rabitq_distance(NULL, encoded, dim);
	ASSERT_FLOAT_EQ(-1.0f, d, 1e-6f, "null state should return error");

	d = mkt_rabitq_distance(state, NULL, dim);
	ASSERT_FLOAT_EQ(-1.0f, d, 1e-6f, "null data should return error");

	/* Test dim mismatch */
	d = mkt_rabitq_distance(state, encoded, 32);
	ASSERT_FLOAT_EQ(-1.0f, d, 1e-6f, "dim mismatch should return error");

	mkt_rabitq_free_query(state);
	mkt_free(encoded);
	mkt_rabitq_destroy(params);
}

/*
 * Accuracy Tests - verify estimation quality
 */

TEST(accuracy_correlation)
{
	/* Test that estimated distances correlate well with true distances */
	Dimension	  dim	   = 64;
	RaBitQParams *params   = mkt_rabitq_create(dim, 42);
	float		 *centroid = alloc_test_vector(dim, 0);

	VectorRef centroid_ref = {.data = centroid, .dim = dim};

	const int num_vectors = 20;
	const int num_queries = 5;

	/* Encode multiple vectors */
	RaBitQData **encoded = mkt_alloc(num_vectors * sizeof(void *));
	float	   **vectors = mkt_alloc(num_vectors * sizeof(void *));

	for (int v = 0; v < num_vectors; v++)
	{
		vectors[v]		  = alloc_test_vector(dim, v * 7);
		VectorRef vec_ref = {.data = vectors[v], .dim = dim};
		encoded[v]		  = mkt_rabitq_encode(params, vec_ref, centroid_ref);
		ASSERT_NOT_NULL(encoded[v], "encoding should succeed");
	}

	/* Test with multiple queries */
	for (int q = 0; q < num_queries; q++)
	{
		float	 *query		= alloc_test_vector(dim, 100 + q * 13);
		VectorRef query_ref = {.data = query, .dim = dim};

		RaBitQQueryState *state =
				mkt_rabitq_prepare_query(params, query_ref, centroid_ref);
		ASSERT_NOT_NULL(state, "prepare_query should succeed");

		int lower_bound_violations = 0;

		for (int v = 0; v < num_vectors; v++)
		{
			VectorRef vec_ref	= {.data = vectors[v], .dim = dim};
			Distance  true_dist = true_l2_distance(vec_ref, query_ref);
			Distance  est, lower_bound;
			mkt_rabitq_distance_with_bound(
					state, encoded[v], dim, &est, &lower_bound);

			/* Check lower bound guarantee */
			if (lower_bound > true_dist + 1e-3f)
			{
				lower_bound_violations++;
				TEST_PRINT(
						"  violation: q=%d v=%d true=%.4f est=%.4f "
						"lower=%.4f gap=%.4f\n",
						q,
						v,
						true_dist,
						est,
						lower_bound,
						lower_bound - true_dist);
				TEST_PRINT(
						"    f_add=%.4f f_rescale=%.4f\n",
						encoded[v]->f_add,
						encoded[v]->f_rescale);
				TEST_PRINT(
						"    g_add=%.4f g_error=%.4f sum_transformed=%.4f\n",
						state->g_add,
						state->g_error,
						state->sum_transformed);
			}
		}

		char msg[64];
		snprintf(msg, sizeof(msg), "query %d: lower bound violations", q);
		ASSERT_EQ(0, lower_bound_violations, msg);

		mkt_rabitq_free_query(state);
	}

	/* Cleanup */
	for (int v = 0; v < num_vectors; v++)
	{
		mkt_free(encoded[v]);
	}
	mkt_free(encoded);
	mkt_free(vectors);
	mkt_rabitq_destroy(params);
}

/*
 * SIMD Implementation Equivalence Tests
 */

static bool
get_rabitq_simd_mask(
		const char *variant, uint32_t *simd_mask, const char **expected_name)
{
	if (strcmp(variant, "compiler") == 0)
	{
		*expected_name = "compiler";
		*simd_mask	   = SIMD_NONE;
		return true;
	}
	else if (strcmp(variant, "avx2") == 0)
	{
		SimdCapability caps = mkt_detect_simd();
		if (!(caps & SIMD_AVX2))
			return false;
		*expected_name = "avx2";
		*simd_mask	   = SIMD_AVX2;
		return true;
	}
	else if (strcmp(variant, "avx512") == 0)
	{
		SimdCapability caps = mkt_detect_simd();
		if (!(caps & SIMD_AVX512F))
			return false;
		*expected_name = "avx512";
		*simd_mask	   = SIMD_AVX512F;
		return true;
	}
	else if (strcmp(variant, "neon") == 0)
	{
		SimdCapability caps = mkt_detect_simd();
		if (!(caps & SIMD_NEON))
			return false;
		*expected_name = "neon";
		*simd_mask	   = SIMD_NEON;
		return true;
	}
	*expected_name = NULL;
	*simd_mask	   = 0xFFFFFFFF;
	return true;
}

#define SKIP_IF_RABITQ_SIMD_NOT_AVAILABLE(variant)                  \
	const char *expected_name;                                      \
	uint32_t	simd_mask;                                          \
	if (!get_rabitq_simd_mask(variant, &simd_mask, &expected_name)) \
	{                                                               \
		TEST_PRINT("%s not available, skipping\n", variant);        \
		return;                                                     \
	}

TEST_PARAMETERIZED(
		simd_distance_equivalence, "compiler", "avx2", "avx512", "neon")
{
	SKIP_IF_RABITQ_SIMD_NOT_AVAILABLE(param);

	const Dimension dims[] = {8, 16, 17, 32, 33, 64, 65, 128, 256, 768};

	/* Get scalar reference results first */
	reinit_rabitq_with_simd(SIMD_NONE);

	for (size_t d = 0; d < sizeof(dims) / sizeof(dims[0]); d++)
	{
		Dimension dim = dims[d];

		RaBitQParams *params   = mkt_rabitq_create(dim, 42);
		float		 *input	   = alloc_test_vector(dim, 0);
		float		 *query	   = alloc_test_vector(dim, 30);
		float		 *centroid = alloc_test_vector(dim, 50);

		VectorRef input_ref	   = {.data = input, .dim = dim};
		VectorRef query_ref	   = {.data = query, .dim = dim};
		VectorRef centroid_ref = {.data = centroid, .dim = dim};

		RaBitQData *encoded =
				mkt_rabitq_encode(params, input_ref, centroid_ref);

		/* Compute reference distance with scalar */
		reinit_rabitq_with_simd(SIMD_NONE);
		RaBitQQueryState *state =
				mkt_rabitq_prepare_query(params, query_ref, centroid_ref);
		Distance ref_dist = mkt_rabitq_distance(state, encoded, dim);
		mkt_rabitq_free_query(state);

		/* Compute with variant */
		reinit_rabitq_with_simd(simd_mask);
		state = mkt_rabitq_prepare_query(params, query_ref, centroid_ref);
		Distance var_dist = mkt_rabitq_distance(state, encoded, dim);

		char msg[128];
		snprintf(
				msg,
				sizeof(msg),
				"dim=%u: %s should match scalar",
				dim,
				param);
		/* Allow relative tolerance for SIMD floating point differences.
		 * AVX2 uses different reduction patterns that accumulate small
		 * variations, especially in high dimensions. Use 1e-4 relative
		 * tolerance with minimum absolute tolerance for small values.
		 */
		float tolerance = fabsf(ref_dist) * 1e-4f;
		if (tolerance < 1e-3f)
			tolerance = 1e-3f; /* Minimum absolute tolerance */
		ASSERT_FLOAT_EQ(ref_dist, var_dist, tolerance, msg);

		mkt_rabitq_free_query(state);
		mkt_free(encoded);
		mkt_rabitq_destroy(params);
	}

	reinit_rabitq_with_simd(0xFFFFFFFF);
}

/*
 * Edge Cases
 */

TEST(identical_vectors)
{
	/* Test with zero centroid where self-distance works correctly.
	 * With non-zero centroids, the RaBitQ formula can produce negative
	 * estimates when the centroid dominates the residual in certain
	 * directions. This is acceptable since the lower bound still guarantees
	 * correctness.
	 */
	Dimension	  dim	 = 16;
	RaBitQParams *params = mkt_rabitq_create(dim, 42);
	float		 *data	 = alloc_test_vector(dim, 0);

	/* Zero centroid for clean self-distance test */
	float *centroid = mkt_alloc(dim * sizeof(float));
	for (Dimension i = 0; i < dim; i++)
		centroid[i] = 0.0f;

	VectorRef data_ref	   = {.data = data, .dim = dim};
	VectorRef centroid_ref = {.data = centroid, .dim = dim};

	RaBitQData *encoded = mkt_rabitq_encode(params, data_ref, centroid_ref);
	RaBitQQueryState *state =
			mkt_rabitq_prepare_query(params, data_ref, centroid_ref);

	Distance est_dist = mkt_rabitq_distance(state, encoded, dim);

	/* With zero centroid, self-distance should be close to 0.
	 * Since ip_cent_xucb = 0, f_add = l2_sqr and the formulas simplify. */
	ASSERT_TRUE(est_dist >= -1e-3f, "self-distance should be non-negative");
	ASSERT_TRUE(est_dist < 10.0f, "self-distance should be small");

	mkt_free(centroid);

	mkt_rabitq_free_query(state);
	mkt_free(encoded);
	mkt_rabitq_destroy(params);
}

TEST(zero_centroid)
{
	Dimension	  dim	 = 16;
	RaBitQParams *params = mkt_rabitq_create(dim, 42);
	float		 *input	 = alloc_test_vector(dim, 0);
	float		 *query	 = alloc_test_vector(dim, 30);

	/* Zero centroid */
	float *centroid = mkt_alloc(dim * sizeof(float));
	for (Dimension i = 0; i < dim; i++)
		centroid[i] = 0.0f;

	VectorRef input_ref	   = {.data = input, .dim = dim};
	VectorRef query_ref	   = {.data = query, .dim = dim};
	VectorRef centroid_ref = {.data = centroid, .dim = dim};

	RaBitQData *encoded = mkt_rabitq_encode(params, input_ref, centroid_ref);
	ASSERT_NOT_NULL(encoded, "encoding with zero centroid should succeed");

	RaBitQQueryState *state =
			mkt_rabitq_prepare_query(params, query_ref, centroid_ref);
	ASSERT_NOT_NULL(state, "prepare_query with zero centroid should succeed");

	Distance est_dist = mkt_rabitq_distance(state, encoded, dim);
	ASSERT_TRUE(isfinite(est_dist), "distance should be finite");

	mkt_rabitq_free_query(state);
	mkt_free(encoded);
	mkt_rabitq_destroy(params);
}

TEST(small_dimension)
{
	/* Test minimum dimension */
	Dimension dim = 8;

	RaBitQParams *params = mkt_rabitq_create(dim, 42);
	ASSERT_NOT_NULL(params, "params should be created");

	float *input	= alloc_test_vector(dim, 0);
	float *query	= alloc_test_vector(dim, 10);
	float *centroid = alloc_test_vector(dim, 5);

	VectorRef input_ref	   = {.data = input, .dim = dim};
	VectorRef query_ref	   = {.data = query, .dim = dim};
	VectorRef centroid_ref = {.data = centroid, .dim = dim};

	RaBitQData *encoded = mkt_rabitq_encode(params, input_ref, centroid_ref);
	ASSERT_NOT_NULL(encoded, "encoding should succeed");

	RaBitQQueryState *state =
			mkt_rabitq_prepare_query(params, query_ref, centroid_ref);
	ASSERT_NOT_NULL(state, "prepare_query should succeed");

	Distance est_dist = mkt_rabitq_distance(state, encoded, dim);
	ASSERT_TRUE(isfinite(est_dist), "distance should be finite");

	mkt_rabitq_free_query(state);
	mkt_free(encoded);
	mkt_rabitq_destroy(params);
}

TEST(non_byte_aligned_dimension)
{
	/* Dimension not divisible by 8 (but factorable for Hadamard).
	 * dim=20 = 4 * 5 → fwht_n=4, k=5, supported. */
	Dimension dim = 20;

	RaBitQParams *params = mkt_rabitq_create(dim, 42);
	ASSERT_NOT_NULL(params, "params should be created");

	float *input	= alloc_test_vector(dim, 0);
	float *query	= alloc_test_vector(dim, 10);
	float *centroid = alloc_test_vector(dim, 5);

	VectorRef input_ref	   = {.data = input, .dim = dim};
	VectorRef query_ref	   = {.data = query, .dim = dim};
	VectorRef centroid_ref = {.data = centroid, .dim = dim};

	RaBitQData *encoded = mkt_rabitq_encode(params, input_ref, centroid_ref);
	ASSERT_NOT_NULL(encoded, "encoding should succeed");

	/* Check packed bytes calculation */
	ASSERT_EQ(3, MKT_RABITQ_BYTES(20), "20 bits needs 3 bytes");

	RaBitQQueryState *state =
			mkt_rabitq_prepare_query(params, query_ref, centroid_ref);
	Distance est_dist = mkt_rabitq_distance(state, encoded, dim);
	ASSERT_TRUE(isfinite(est_dist), "distance should be finite");

	Distance lower_bound;
	Distance dummy_est;
	mkt_rabitq_distance_with_bound(
			state, encoded, dim, &dummy_est, &lower_bound);

	Distance true_dist = true_l2_distance(input_ref, query_ref);
	ASSERT_TRUE(
			lower_bound <= true_dist + 1e-3f,
			"lower bound should be <= true distance");

	mkt_rabitq_free_query(state);
	mkt_free(encoded);
	mkt_rabitq_destroy(params);
}

TEST(unsupported_dimension_fails)
{
	/* dim=17 is prime — Hadamard can't factor it; mkt_rabitq_create
	 * must fail rather than silently returning broken params. */
	ASSERT_NULL(mkt_rabitq_create(17, 42), "prime dim should be rejected");
}

/*
 * Lifecycle Edge Cases
 */

TEST(destroy_null)
{
	/* Should not crash */
	mkt_rabitq_destroy(NULL);
	ASSERT_TRUE(1, "destroy null should not crash");
}

TEST(cleanup_null)
{
	/* Should not crash */
	mkt_rabitq_cleanup(NULL);
	ASSERT_TRUE(1, "cleanup null should not crash");
}

TEST(cleanup_null_params)
{
	/* Cleanup with NULL should be safe */
	mkt_rabitq_cleanup(NULL);
	ASSERT_TRUE(1, "cleanup with null should not crash");
}

TEST(free_query_null)
{
	/* Should not crash */
	mkt_rabitq_free_query(NULL);
	ASSERT_TRUE(1, "free_query null should not crash");
}

TEST(init_null_params)
{
	int ret = mkt_rabitq_init(NULL, 16, 42);
	ASSERT_EQ(-1, ret, "init with null params should fail");
}

TEST(init_zero_dim)
{
	/* Need buffer for flexible array, but dim=0 should fail before
	 * accessing P, so a minimal alloc suffices. */
	RaBitQParams *params = mkt_alloc(sizeof(RaBitQParams));
	int			  ret	 = mkt_rabitq_init(params, 0, 42);
	ASSERT_EQ(-1, ret, "init with zero dim should fail");
	mkt_free(params);
}

/*
 * Encoding Edge Cases
 */

TEST(encode_into_null_output)
{
	Dimension	  dim	   = 8;
	RaBitQParams *params   = mkt_rabitq_create(dim, 1);
	float		 *input	   = alloc_test_vector(dim, 0);
	float		 *centroid = alloc_test_vector(dim, 5);

	VectorRef input_ref	   = {.data = input, .dim = dim};
	VectorRef centroid_ref = {.data = centroid, .dim = dim};

	int ret = mkt_rabitq_encode_into(params, input_ref, centroid_ref, NULL);
	ASSERT_EQ(-1, ret, "encode_into with null output should fail");

	mkt_rabitq_destroy(params);
}

TEST(encode_into_dim_mismatch_input)
{
	Dimension	  dim	   = 16;
	RaBitQParams *params   = mkt_rabitq_create(dim, 1);
	float		 *input	   = alloc_test_vector(8, 0);
	float		 *centroid = alloc_test_vector(dim, 5);

	VectorRef input_ref	   = {.data = input, .dim = 8};
	VectorRef centroid_ref = {.data = centroid, .dim = dim};

	size_t		size   = MKT_RABITQ_DATA_SIZE(dim);
	RaBitQData *output = mkt_alloc(size);

	int ret = mkt_rabitq_encode_into(params, input_ref, centroid_ref, output);
	ASSERT_EQ(-1, ret, "encode_into dim mismatch input should fail");

	mkt_free(output);
	mkt_rabitq_destroy(params);
}

TEST(encode_into_dim_mismatch_centroid)
{
	Dimension	  dim	   = 16;
	RaBitQParams *params   = mkt_rabitq_create(dim, 1);
	float		 *input	   = alloc_test_vector(dim, 0);
	float		 *centroid = alloc_test_vector(8, 5);

	VectorRef input_ref	   = {.data = input, .dim = dim};
	VectorRef centroid_ref = {.data = centroid, .dim = 8};

	size_t		size   = MKT_RABITQ_DATA_SIZE(dim);
	RaBitQData *output = mkt_alloc(size);

	int ret = mkt_rabitq_encode_into(params, input_ref, centroid_ref, output);
	ASSERT_EQ(-1, ret, "encode_into dim mismatch centroid should fail");

	mkt_free(output);
	mkt_rabitq_destroy(params);
}

TEST(encode_batch_dim_mismatch)
{
	Dimension	  dim	   = 16;
	RaBitQParams *params   = mkt_rabitq_create(dim, 1);
	float		 *vectors  = alloc_test_vector(dim * 2, 0);
	float		 *centroid = alloc_test_vector(8, 0);
	VectorRef	  cent_ref = {.data = centroid, .dim = 8};

	uint32_t packed_bytes = MKT_RABITQ_BYTES(dim);
	float	*f_add		  = mkt_alloc(2 * sizeof(float));
	float	*f_rescale	  = mkt_alloc(2 * sizeof(float));
	uint8_t *bits		  = mkt_alloc(2 * packed_bytes);

	int ret = mkt_rabitq_encode_batch(
			params, vectors, MKT_VEC_F32, cent_ref, f_add, f_rescale, bits, 2);
	ASSERT_EQ(-1, ret, "batch encode dim mismatch should fail");

	mkt_free(bits);
	mkt_free(f_rescale);
	mkt_free(f_add);
	mkt_rabitq_destroy(params);
}

/*
 * Distance Edge Cases
 */

TEST(distance_dim_mismatch)
{
	/* Create two different-dimension setups */
	Dimension	  dim16	   = 16;
	Dimension	  dim32	   = 32;
	RaBitQParams *params16 = mkt_rabitq_create(dim16, 42);
	RaBitQParams *params32 = mkt_rabitq_create(dim32, 42);
	float		 *input16  = alloc_test_vector(dim16, 0);
	float		 *query32  = alloc_test_vector(dim32, 30);
	float		 *cent16   = alloc_test_vector(dim16, 50);
	float		 *cent32   = alloc_test_vector(dim32, 50);

	VectorRef input16_ref = {.data = input16, .dim = dim16};
	VectorRef query32_ref = {.data = query32, .dim = dim32};
	VectorRef cent16_ref  = {.data = cent16, .dim = dim16};
	VectorRef cent32_ref  = {.data = cent32, .dim = dim32};

	RaBitQData *encoded = mkt_rabitq_encode(params16, input16_ref, cent16_ref);
	RaBitQQueryState *state =
			mkt_rabitq_prepare_query(params32, query32_ref, cent32_ref);

	/* Query state dim=32 but passing dim=16 triggers mismatch */
	Distance d = mkt_rabitq_distance(state, encoded, dim16);
	ASSERT_FLOAT_EQ(-1.0f, d, 1e-6f, "dim mismatch should return -1");

	mkt_rabitq_free_query(state);
	mkt_free(encoded);
	mkt_rabitq_destroy(params16);
	mkt_rabitq_destroy(params32);
}

TEST(distance_with_bound_null_est_dist)
{
	Dimension	  dim	   = 8;
	RaBitQParams *params   = mkt_rabitq_create(dim, 1);
	float		 *input	   = alloc_test_vector(dim, 0);
	float		 *query	   = alloc_test_vector(dim, 10);
	float		 *centroid = alloc_test_vector(dim, 5);

	VectorRef input_ref	   = {.data = input, .dim = dim};
	VectorRef query_ref	   = {.data = query, .dim = dim};
	VectorRef centroid_ref = {.data = centroid, .dim = dim};

	RaBitQData *encoded = mkt_rabitq_encode(params, input_ref, centroid_ref);
	RaBitQQueryState *state =
			mkt_rabitq_prepare_query(params, query_ref, centroid_ref);

	/* Test with null state */
	Distance lower_bound;
	mkt_rabitq_distance_with_bound(NULL, encoded, dim, NULL, &lower_bound);
	ASSERT_FLOAT_EQ(-1.0f, lower_bound, 1e-6f, "null state should set lb=-1");

	/* Test with null data */
	Distance est_dist;
	mkt_rabitq_distance_with_bound(state, NULL, dim, &est_dist, NULL);
	ASSERT_FLOAT_EQ(-1.0f, est_dist, 1e-6f, "null data should set est=-1");

	mkt_rabitq_free_query(state);
	mkt_free(encoded);
	mkt_rabitq_destroy(params);
}

TEST(prepare_query_dim_mismatch)
{
	Dimension	  dim	   = 16;
	RaBitQParams *params   = mkt_rabitq_create(dim, 42);
	float		 *query	   = alloc_test_vector(8, 0);
	float		 *centroid = alloc_test_vector(dim, 5);

	VectorRef query_ref	   = {.data = query, .dim = 8};
	VectorRef centroid_ref = {.data = centroid, .dim = dim};

	RaBitQQueryState *state =
			mkt_rabitq_prepare_query(params, query_ref, centroid_ref);
	ASSERT_NULL(state, "dim mismatch query should return null");

	mkt_rabitq_destroy(params);
}

TEST(prepare_query_centroid_dim_mismatch)
{
	Dimension	  dim	   = 16;
	RaBitQParams *params   = mkt_rabitq_create(dim, 42);
	float		 *query	   = alloc_test_vector(dim, 0);
	float		 *centroid = alloc_test_vector(8, 5);

	VectorRef query_ref	   = {.data = query, .dim = dim};
	VectorRef centroid_ref = {.data = centroid, .dim = 8};

	RaBitQQueryState *state =
			mkt_rabitq_prepare_query(params, query_ref, centroid_ref);
	ASSERT_NULL(state, "dim mismatch centroid should return null");

	mkt_rabitq_destroy(params);
}

/*
 * Accuracy with high dimensions
 */

TEST(high_dimension_lower_bound)
{
	/* Test lower bound guarantee holds at higher dimensions */
	Dimension	  dim	   = 256;
	RaBitQParams *params   = mkt_rabitq_create(dim, 42);
	float		 *centroid = alloc_test_vector(dim, 0);

	VectorRef centroid_ref = {.data = centroid, .dim = dim};

	const int count = 10;
	for (int i = 0; i < count; i++)
	{
		float	 *input		= alloc_test_vector(dim, i * 7);
		float	 *query		= alloc_test_vector(dim, 100 + i * 13);
		VectorRef input_ref = {.data = input, .dim = dim};
		VectorRef query_ref = {.data = query, .dim = dim};

		RaBitQData *encoded =
				mkt_rabitq_encode(params, input_ref, centroid_ref);
		RaBitQQueryState *state =
				mkt_rabitq_prepare_query(params, query_ref, centroid_ref);

		Distance est, lower_bound;
		mkt_rabitq_distance_with_bound(
				state, encoded, dim, &est, &lower_bound);

		Distance true_dist = true_l2_distance(input_ref, query_ref);

		char msg[128];
		snprintf(
				msg,
				sizeof(msg),
				"vec %d: lower bound (%.4f) should be "
				"<= true dist (%.4f)",
				i,
				lower_bound,
				true_dist);
		ASSERT_TRUE(lower_bound <= true_dist + 1e-3f, msg);

		mkt_rabitq_free_query(state);
		mkt_free(encoded);
	}

	mkt_rabitq_destroy(params);
}

/*
 * Compact RaBitQData Tests
 */

TEST(data_lower_bound_multi_dim)
{
	/*
	 * Verify that derived f_error lower bounds hold across many
	 * vectors and dimensions (the key safety guarantee).
	 */
	const Dimension dims[] = {16, 64, 128, 256};

	for (size_t d = 0; d < sizeof(dims) / sizeof(dims[0]); d++)
	{
		Dimension	  dim	   = dims[d];
		RaBitQParams *params   = mkt_rabitq_create(dim, 42);
		float		 *centroid = alloc_test_vector(dim, 0);
		VectorRef	  cent_ref = {.data = centroid, .dim = dim};

		int violations = 0;

		for (int v = 0; v < 10; v++)
		{
			float	 *input		= alloc_test_vector(dim, v * 7);
			VectorRef input_ref = {.data = input, .dim = dim};

			RaBitQData *enc = mkt_rabitq_encode(params, input_ref, cent_ref);

			for (int q = 0; q < 5; q++)
			{
				float	 *query		= alloc_test_vector(dim, 100 + q * 13);
				VectorRef query_ref = {.data = query, .dim = dim};

				RaBitQQueryState *state =
						mkt_rabitq_prepare_query(params, query_ref, cent_ref);

				Distance est, lb;
				mkt_rabitq_distance_with_bound(state, enc, dim, &est, &lb);

				Distance true_dist = true_l2_distance(input_ref, query_ref);

				if (lb > true_dist + 1e-3f)
				{
					violations++;
					TEST_PRINT(
							"  dim=%u v=%d q=%d: lb=%.4f "
							"true=%.4f\n",
							dim,
							v,
							q,
							lb,
							true_dist);
				}

				mkt_rabitq_free_query(state);
			}

			mkt_free(enc);
		}

		/* RaBitQ's `lb <= true_dist` bound is probabilistic — the
		 * MKT_RABITQ_EPSILON constant gives 99% confidence in the
		 * Gaussian-rotation analysis. At 50 (q × v) trials per dim we
		 * therefore allow ≤ 1 violation; tighter check (==0) is too
		 * strict for the randomized-Hadamard rotation, which has
		 * slightly higher tail variance than fully-random orthonormal. */
		int	 max_allowed = 1;
		char msg[128];
		snprintf(
				msg,
				sizeof(msg),
				"dim=%u: %d lower bound violations (allowed %d)",
				dims[d],
				violations,
				max_allowed);
		ASSERT_TRUE(violations <= max_allowed, msg);

		mkt_rabitq_destroy(params);
	}
}

TEST(data_distance_null_inputs)
{
	Dimension	  dim	   = 8;
	RaBitQParams *params   = mkt_rabitq_create(dim, 1);
	float		 *input	   = alloc_test_vector(dim, 0);
	float		 *query	   = alloc_test_vector(dim, 10);
	float		 *centroid = alloc_test_vector(dim, 5);

	VectorRef input_ref	   = {.data = input, .dim = dim};
	VectorRef query_ref	   = {.data = query, .dim = dim};
	VectorRef centroid_ref = {.data = centroid, .dim = dim};

	RaBitQData *encoded = mkt_rabitq_encode(params, input_ref, centroid_ref);
	RaBitQQueryState *state =
			mkt_rabitq_prepare_query(params, query_ref, centroid_ref);

	/* Test null state */
	Distance d = mkt_rabitq_distance(NULL, encoded, dim);
	ASSERT_FLOAT_EQ(-1.0f, d, 1e-6f, "null state should return -1");

	/* Test null data */
	d = mkt_rabitq_distance(state, NULL, dim);
	ASSERT_FLOAT_EQ(-1.0f, d, 1e-6f, "null data should return -1");

	/* Test dim mismatch */
	d = mkt_rabitq_distance(state, encoded, 32);
	ASSERT_FLOAT_EQ(-1.0f, d, 1e-6f, "dim mismatch should return -1");

	/* Test null inputs for distance_with_bound */
	Distance est, lb;
	mkt_rabitq_distance_with_bound(NULL, encoded, dim, &est, &lb);
	ASSERT_FLOAT_EQ(-1.0f, lb, 1e-6f, "null state should set lb=-1");

	mkt_rabitq_distance_with_bound(state, NULL, dim, &est, &lb);
	ASSERT_FLOAT_EQ(-1.0f, est, 1e-6f, "null data should set est=-1");

	mkt_rabitq_free_query(state);
	mkt_free(encoded);
	mkt_rabitq_destroy(params);
}

/*
 * Batch distance tests
 */

TEST(batch_matches_single)
{
	Dimension dim	= 128;
	const int count = 8;

	RaBitQParams *params = mkt_rabitq_create(dim, 42);
	ASSERT_NOT_NULL(params, "params created");

	float	 *centroid = alloc_test_vector(dim, 0);
	VectorRef cent_ref = {.data = centroid, .dim = dim};

	/* Encode vectors into separate arrays */
	uint32_t packed_bytes = MKT_RABITQ_BYTES(dim);
	float	*f_add		  = mkt_alloc(count * sizeof(float));
	float	*f_rescale	  = mkt_alloc(count * sizeof(float));
	uint8_t *bits		  = mkt_alloc((size_t)count * packed_bytes);

	RaBitQData **encodings = mkt_alloc(count * sizeof(void *));

	for (int i = 0; i < count; i++)
	{
		float	 *vec	  = alloc_test_vector(dim, i * 7);
		VectorRef vec_ref = {.data = vec, .dim = dim};
		encodings[i]	  = mkt_rabitq_encode(params, vec_ref, cent_ref);
		ASSERT_NOT_NULL(encodings[i], "encoding succeeded");

		f_add[i]	 = encodings[i]->f_add;
		f_rescale[i] = encodings[i]->f_rescale;
		memcpy(bits + (size_t)i * packed_bytes,
			   encodings[i]->bits,
			   packed_bytes);
	}

	float			 *query		= alloc_test_vector(dim, 100);
	VectorRef		  query_ref = {.data = query, .dim = dim};
	RaBitQQueryState *qstate =
			mkt_rabitq_prepare_query(params, query_ref, cent_ref);
	ASSERT_NOT_NULL(qstate, "query state created");

	/* Compute batch distances */
	Distance *batch_dists = mkt_alloc(count * sizeof(Distance));
	mkt_rabitq_distance_batch(
			qstate, f_add, f_rescale, bits, count, dim, batch_dists);

	/* Compare with single-entry distances */
	for (int i = 0; i < count; i++)
	{
		Distance single_dist = mkt_rabitq_distance(qstate, encodings[i], dim);

		char msg[128];
		snprintf(
				msg,
				sizeof(msg),
				"vec %d: batch=%.6f single=%.6f",
				i,
				batch_dists[i],
				single_dist);
		ASSERT_FLOAT_EQ(single_dist, batch_dists[i], 1e-6f, msg);
	}

	mkt_free(batch_dists);
	mkt_rabitq_free_query(qstate);
	for (int i = 0; i < count; i++)
		mkt_free(encodings[i]);
	mkt_free(encodings);
	mkt_free(bits);
	mkt_free(f_rescale);
	mkt_free(f_add);
	mkt_rabitq_destroy(params);
}

TEST(batch_null_inputs)
{
	Dimension	  dim	   = 64;
	RaBitQParams *params   = mkt_rabitq_create(dim, 42);
	float		 *centroid = alloc_test_vector(dim, 0);
	VectorRef	  cent_ref = {.data = centroid, .dim = dim};

	float			 *query		= alloc_test_vector(dim, 1);
	VectorRef		  query_ref = {.data = query, .dim = dim};
	RaBitQQueryState *qstate =
			mkt_rabitq_prepare_query(params, query_ref, cent_ref);

	float	 f_add[1]	  = {1.0f};
	float	 f_rescale[1] = {1.0f};
	uint8_t	 bits[16]	  = {0};
	Distance dists[1];

	/* Should not crash with null inputs */
	mkt_rabitq_distance_batch(NULL, f_add, f_rescale, bits, 1, dim, dists);
	mkt_rabitq_distance_batch(qstate, NULL, f_rescale, bits, 1, dim, dists);
	mkt_rabitq_distance_batch(qstate, f_add, NULL, bits, 1, dim, dists);
	mkt_rabitq_distance_batch(qstate, f_add, f_rescale, NULL, 1, dim, dists);
	mkt_rabitq_distance_batch(qstate, f_add, f_rescale, bits, 0, dim, dists);
	mkt_rabitq_distance_batch(qstate, f_add, f_rescale, bits, 1, dim, NULL);

	ASSERT_TRUE(1, "null inputs should not crash");

	mkt_rabitq_free_query(qstate);
	mkt_rabitq_destroy(params);
}

/*
 * Multi-candidate inner product tests
 */

TEST_PARAMETERIZED(multi_matches_single, "compiler", "avx2", "avx512", "neon")
{
	SKIP_IF_RABITQ_SIMD_NOT_AVAILABLE(param);

	const Dimension dims[] = {64, 128, 768};

	for (size_t d = 0; d < sizeof(dims) / sizeof(dims[0]); d++)
	{
		Dimension dim	= dims[d];
		const int count = 16;

		RaBitQParams *params   = mkt_rabitq_create(dim, 42);
		float		 *centroid = alloc_test_vector(dim, 0);
		VectorRef	  cent_ref = {.data = centroid, .dim = dim};

		/* Generate and encode test vectors */
		float *vectors = mkt_alloc((size_t)count * dim * sizeof(float));
		for (int i = 0; i < count; i++)
		{
			float *v = vectors + i * dim;
			for (Dimension j = 0; j < dim; j++)
				v[j] = (float)((i * 17 + j * 13) % 100 - 50) / 10.0f;
		}

		uint32_t packed_bytes = MKT_RABITQ_BYTES(dim);
		float	*f_add		  = mkt_alloc(count * sizeof(float));
		float	*f_rescale	  = mkt_alloc(count * sizeof(float));
		uint8_t *bits		  = mkt_alloc((size_t)count * packed_bytes);
		int		 ret		  = mkt_rabitq_encode_batch(
				  params,
				  vectors,
				  MKT_VEC_F32,
				  cent_ref,
				  f_add,
				  f_rescale,
				  bits,
				  count);
		ASSERT_EQ(0, ret, "batch encode should succeed");

		/* Generate query and prepare state */
		float	 *query		= alloc_test_vector(dim, 100);
		VectorRef query_ref = {.data = query, .dim = dim};

		reinit_rabitq_with_simd(simd_mask);

		RaBitQQueryState *qstate =
				mkt_rabitq_prepare_query(params, query_ref, cent_ref);
		ASSERT_NOT_NULL(qstate, "prepare_query should succeed");

		/* Compute sequential distances */
		Distance *seq_dists = mkt_alloc(count * sizeof(Distance));
		mkt_rabitq_distance_batch(
				qstate, f_add, f_rescale, bits, count, dim, seq_dists);

		/* Compute multi distances */
		Distance *multi_dists = mkt_alloc(count * sizeof(Distance));
		mkt_rabitq_distance_batch_multi(
				qstate, f_add, f_rescale, bits, count, dim, multi_dists);

		/* Compare results */
		for (int i = 0; i < count; i++)
		{
			char msg[128];
			snprintf(
					msg,
					sizeof(msg),
					"dim=%u %s vec %d: seq=%.6f multi=%.6f",
					dim,
					param,
					i,
					seq_dists[i],
					multi_dists[i]);
			float tolerance = fabsf(seq_dists[i]) * 1e-4f;
			if (tolerance < 1e-3f)
				tolerance = 1e-3f;
			ASSERT_FLOAT_EQ(seq_dists[i], multi_dists[i], tolerance, msg);
		}

		mkt_free(multi_dists);
		mkt_free(seq_dists);
		mkt_rabitq_free_query(qstate);
		mkt_free(bits);
		mkt_free(f_rescale);
		mkt_free(f_add);
		mkt_free(vectors);
		mkt_rabitq_destroy(params);
	}

	reinit_rabitq_with_simd(0xFFFFFFFF);
}

TEST_PARAMETERIZED(multi_tail_handling, "compiler", "avx2", "avx512", "neon")
{
	SKIP_IF_RABITQ_SIMD_NOT_AVAILABLE(param);

	Dimension dim = 128;

	/* Test counts that are not multiples of 4 (the SIMD group size) */
	const int counts[] = {1, 2, 3, 5, 7};

	for (size_t c = 0; c < sizeof(counts) / sizeof(counts[0]); c++)
	{
		int count = counts[c];

		RaBitQParams *params   = mkt_rabitq_create(dim, 42);
		float		 *centroid = alloc_test_vector(dim, 0);
		VectorRef	  cent_ref = {.data = centroid, .dim = dim};

		float *vectors = mkt_alloc((size_t)count * dim * sizeof(float));
		for (int i = 0; i < count; i++)
		{
			float *v = vectors + i * dim;
			for (Dimension j = 0; j < dim; j++)
				v[j] = (float)((i * 17 + j * 13) % 100 - 50) / 10.0f;
		}

		uint32_t packed_bytes = MKT_RABITQ_BYTES(dim);
		float	*f_add		  = mkt_alloc(count * sizeof(float));
		float	*f_rescale	  = mkt_alloc(count * sizeof(float));
		uint8_t *bits		  = mkt_alloc((size_t)count * packed_bytes);
		int		 ret		  = mkt_rabitq_encode_batch(
				  params,
				  vectors,
				  MKT_VEC_F32,
				  cent_ref,
				  f_add,
				  f_rescale,
				  bits,
				  count);
		ASSERT_EQ(0, ret, "batch encode should succeed");

		float	 *query		= alloc_test_vector(dim, 100);
		VectorRef query_ref = {.data = query, .dim = dim};

		reinit_rabitq_with_simd(simd_mask);

		RaBitQQueryState *qstate =
				mkt_rabitq_prepare_query(params, query_ref, cent_ref);
		ASSERT_NOT_NULL(qstate, "prepare_query should succeed");

		Distance *seq_dists	  = mkt_alloc(count * sizeof(Distance));
		Distance *multi_dists = mkt_alloc(count * sizeof(Distance));

		mkt_rabitq_distance_batch(
				qstate, f_add, f_rescale, bits, count, dim, seq_dists);
		mkt_rabitq_distance_batch_multi(
				qstate, f_add, f_rescale, bits, count, dim, multi_dists);

		for (int i = 0; i < count; i++)
		{
			char msg[128];
			snprintf(
					msg,
					sizeof(msg),
					"count=%d %s vec %d: seq=%.6f multi=%.6f",
					count,
					param,
					i,
					seq_dists[i],
					multi_dists[i]);
			float tolerance = fabsf(seq_dists[i]) * 1e-4f;
			if (tolerance < 1e-3f)
				tolerance = 1e-3f;
			ASSERT_FLOAT_EQ(seq_dists[i], multi_dists[i], tolerance, msg);
		}

		mkt_free(multi_dists);
		mkt_free(seq_dists);
		mkt_rabitq_free_query(qstate);
		mkt_free(bits);
		mkt_free(f_rescale);
		mkt_free(f_add);
		mkt_free(vectors);
		mkt_rabitq_destroy(params);
	}

	reinit_rabitq_with_simd(0xFFFFFFFF);
}

/*
 * Implementation name test
 */

TEST(impl_name_is_valid)
{
	const char *name = mkt_rabitq_impl_name();
	ASSERT_NOT_NULL(name, "implementation name should not be null");

	int valid =
			(strcmp(name, "avx512") == 0 || strcmp(name, "avx2") == 0 ||
			 strcmp(name, "neon") == 0 || strcmp(name, "compiler") == 0);

	char msg[128];
	snprintf(msg, sizeof(msg), "unknown implementation: %s", name);
	ASSERT_TRUE(valid, msg);
}

TEST(hamming_impl_name_is_valid)
{
	const char *name = mkt_rabitq_hamming_impl_name();
	ASSERT_NOT_NULL(name, "hamming impl name should not be null");

	int valid =
			(strcmp(name, "avx512-vpopcntdq") == 0 ||
			 strcmp(name, "avx2") == 0 || strcmp(name, "compiler") == 0);

	char msg[128];
	snprintf(msg, sizeof(msg), "unknown hamming implementation: %s", name);
	ASSERT_TRUE(valid, msg);
}

/*
 * Hamming Distance Tests
 */

TEST(hamming_distance_basic)
{
	/* Known bit patterns: 0xFF vs 0x00 -> 8 bits differ per byte */
	uint8_t a[] = {0xFF, 0xFF, 0x00, 0xAA};
	uint8_t b[] = {0x00, 0x00, 0x00, 0x55};

	uint32_t dist = mkt_rabitq_hamming_distance(a, b, 4);

	/* 0xFF^0x00=0xFF (8 bits) + 0xFF^0x00=0xFF (8 bits) +
	 * 0x00^0x00=0x00 (0 bits) + 0xAA^0x55=0xFF (8 bits) = 24 */
	ASSERT_EQ(24, dist, "hamming distance should be 24");
}

TEST(hamming_distance_all_same)
{
	uint8_t a[] = {0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE};
	uint8_t b[] = {0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE};

	uint32_t dist = mkt_rabitq_hamming_distance(a, b, 8);
	ASSERT_EQ(0, dist, "identical vectors should have hamming distance 0");
}

TEST(hamming_distance_all_different)
{
	uint8_t a[16];
	uint8_t b[16];

	memset(a, 0xFF, 16);
	memset(b, 0x00, 16);

	uint32_t dist = mkt_rabitq_hamming_distance(a, b, 16);
	ASSERT_EQ(128, dist, "all-different should have hamming distance 128");
}

TEST(hamming_distance_large)
{
	/* Test with 96 bytes (768 dimensions) to exercise SIMD paths */
	const uint32_t bytes = 96;
	uint8_t		  *a	 = mkt_alloc(bytes);
	uint8_t		  *b	 = mkt_alloc(bytes);

	/* Generate deterministic pattern */
	for (uint32_t i = 0; i < bytes; i++)
	{
		a[i] = (uint8_t)((i * 37 + 13) & 0xFF);
		b[i] = (uint8_t)((i * 53 + 7) & 0xFF);
	}

	/* Compute reference with scalar popcount */
	uint32_t expected = 0;
	for (uint32_t i = 0; i < bytes; i++)
		expected += (uint32_t)__builtin_popcount(a[i] ^ b[i]);

	uint32_t actual = mkt_rabitq_hamming_distance(a, b, bytes);
	ASSERT_EQ(expected, actual, "large hamming distance should match scalar");

	mkt_free(a);
	mkt_free(b);
}

TEST(hamming_distance_multi_basic)
{
	const uint32_t packed_bytes = 8;
	const uint32_t count		= 4;

	uint8_t query[8];
	memset(query, 0xAA, packed_bytes);

	uint8_t data[4 * 8];
	memset(data + 0 * packed_bytes, 0xAA, packed_bytes); /* same */
	memset(data + 1 * packed_bytes, 0x55, packed_bytes); /* opposite */
	memset(data + 2 * packed_bytes, 0xFF, packed_bytes); /* half diff */
	memset(data + 3 * packed_bytes, 0x00, packed_bytes); /* half diff */

	uint32_t results[4];
	mkt_rabitq_hamming_distance_multi(
			query, data, packed_bytes, packed_bytes, count, results);

	ASSERT_EQ(0, results[0], "identical should be 0");
	ASSERT_EQ(64, results[1], "opposite should be 64");
	/* 0xAA ^ 0xFF = 0x55 -> 4 bits per byte * 8 bytes = 32 */
	ASSERT_EQ(32, results[2], "0xAA vs 0xFF should be 32");
	/* 0xAA ^ 0x00 = 0xAA -> 4 bits per byte * 8 bytes = 32 */
	ASSERT_EQ(32, results[3], "0xAA vs 0x00 should be 32");
}

TEST_PARAMETERIZED(simd_hamming_equivalence, "compiler", "avx2", "avx512")
{
	SKIP_IF_RABITQ_SIMD_NOT_AVAILABLE(param);

	/* For hamming, "avx512" needs VPOPCNTDQ */
	if (strcmp(param, "avx512") == 0)
	{
		SimdCapability caps = mkt_detect_simd();
		if (!(caps & SIMD_AVX512_VPOPCNTDQ))
		{
			TEST_PRINT("VPOPCNTDQ not available, skipping\n");
			return;
		}
	}

	const uint32_t test_bytes[] = {1, 4, 8, 12, 16, 32, 64, 96, 128, 192};

	/* Get scalar reference first */
	reinit_rabitq_with_simd(SIMD_NONE);

	for (size_t t = 0; t < sizeof(test_bytes) / sizeof(test_bytes[0]); t++)
	{
		uint32_t bytes = test_bytes[t];
		uint8_t *a	   = mkt_alloc(bytes);
		uint8_t *b	   = mkt_alloc(bytes);

		for (uint32_t i = 0; i < bytes; i++)
		{
			a[i] = (uint8_t)((i * 37 + 13) & 0xFF);
			b[i] = (uint8_t)((i * 53 + 7) & 0xFF);
		}

		/* Reference with compiler */
		reinit_rabitq_with_simd(SIMD_NONE);
		uint32_t ref = mkt_rabitq_hamming_distance(a, b, bytes);

		/* Variant */
		reinit_rabitq_with_simd(simd_mask);
		uint32_t var = mkt_rabitq_hamming_distance(a, b, bytes);

		char msg[128];
		snprintf(
				msg,
				sizeof(msg),
				"bytes=%u: %s should match scalar",
				bytes,
				param);
		ASSERT_EQ(ref, var, msg);

		mkt_free(a);
		mkt_free(b);
	}

	reinit_rabitq_with_simd(0xFFFFFFFF);
}

/*
 * Symmetric Distance Tests
 */

TEST(symmetric_distance_basic)
{
	Dimension	  dim	   = 64;
	RaBitQParams *params   = mkt_rabitq_create(dim, 42);
	float		 *input	   = alloc_test_vector(dim, 0);
	float		 *query	   = alloc_test_vector(dim, 30);
	float		 *centroid = alloc_test_vector(dim, 50);

	VectorRef input_ref	   = {.data = input, .dim = dim};
	VectorRef query_ref	   = {.data = query, .dim = dim};
	VectorRef centroid_ref = {.data = centroid, .dim = dim};

	RaBitQData *encoded = mkt_rabitq_encode(params, input_ref, centroid_ref);
	ASSERT_NOT_NULL(encoded, "encoding should succeed");

	RaBitQQueryState *state =
			mkt_rabitq_prepare_query(params, query_ref, centroid_ref);
	ASSERT_NOT_NULL(state, "prepare_query should succeed");

	Distance sym_dist = mkt_rabitq_distance_symmetric(state, encoded, dim);
	ASSERT_TRUE(isfinite(sym_dist), "symmetric distance should be finite");

	Distance asym_dist = mkt_rabitq_distance(state, encoded, dim);

	TEST_PRINT("Asymmetric: %.4f, Symmetric: %.4f\n", asym_dist, sym_dist);

	mkt_rabitq_free_query(state);
	mkt_free(encoded);
	mkt_rabitq_destroy(params);
}

TEST(symmetric_distance_vs_asymmetric)
{
	/* Symmetric should correlate with asymmetric (not exact) */
	Dimension	  dim	   = 128;
	RaBitQParams *params   = mkt_rabitq_create(dim, 42);
	float		 *centroid = alloc_test_vector(dim, 0);

	VectorRef centroid_ref = {.data = centroid, .dim = dim};

	const int num_vectors = 10;
	float	  asym_dists[10];
	float	  sym_dists[10];

	float	 *query		= alloc_test_vector(dim, 100);
	VectorRef query_ref = {.data = query, .dim = dim};

	RaBitQQueryState *state =
			mkt_rabitq_prepare_query(params, query_ref, centroid_ref);

	for (int i = 0; i < num_vectors; i++)
	{
		float	 *input		= alloc_test_vector(dim, i * 7);
		VectorRef input_ref = {.data = input, .dim = dim};

		RaBitQData *encoded =
				mkt_rabitq_encode(params, input_ref, centroid_ref);

		asym_dists[i] = mkt_rabitq_distance(state, encoded, dim);
		sym_dists[i]  = mkt_rabitq_distance_symmetric(state, encoded, dim);

		mkt_free(encoded);
	}

	/* Check rank correlation: for most pairs, if asym[i] < asym[j]
	 * then sym[i] < sym[j] (allow some inversions) */
	int concordant = 0;
	int discordant = 0;
	for (int i = 0; i < num_vectors; i++)
	{
		for (int j = i + 1; j < num_vectors; j++)
		{
			float asym_diff = asym_dists[i] - asym_dists[j];
			float sym_diff	= sym_dists[i] - sym_dists[j];
			if (asym_diff * sym_diff > 0)
				concordant++;
			else if (asym_diff * sym_diff < 0)
				discordant++;
		}
	}

	int total = concordant + discordant;
	if (total > 0)
	{
		float kendall_tau = (float)(concordant - discordant) / (float)total;
		TEST_PRINT(
				"Kendall tau: %.4f (concordant=%d discordant=%d)\n",
				kendall_tau,
				concordant,
				discordant);
		/* Expect reasonable correlation (tau > 0.3) */
		ASSERT_TRUE(
				kendall_tau > 0.3f,
				"symmetric should correlate with asymmetric");
	}

	mkt_rabitq_free_query(state);
	mkt_rabitq_destroy(params);
}

TEST(symmetric_lower_bound_valid)
{
	/* Verify symmetric lower bound <= true L2 distance */
	Dimension	  dim	   = 64;
	RaBitQParams *params   = mkt_rabitq_create(dim, 42);
	float		 *centroid = alloc_test_vector(dim, 0);

	VectorRef centroid_ref = {.data = centroid, .dim = dim};

	int violations = 0;

	for (int v = 0; v < 10; v++)
	{
		float	 *input		= alloc_test_vector(dim, v * 7);
		VectorRef input_ref = {.data = input, .dim = dim};

		RaBitQData *encoded =
				mkt_rabitq_encode(params, input_ref, centroid_ref);

		for (int q = 0; q < 5; q++)
		{
			float	 *query		= alloc_test_vector(dim, 100 + q * 13);
			VectorRef query_ref = {.data = query, .dim = dim};

			RaBitQQueryState *state =
					mkt_rabitq_prepare_query(params, query_ref, centroid_ref);

			Distance est, lb;
			mkt_rabitq_distance_symmetric_with_bound(
					state, encoded, dim, &est, &lb);

			Distance true_dist = true_l2_distance(input_ref, query_ref);

			if (lb > true_dist + 1e-2f)
			{
				violations++;
				TEST_PRINT(
						"  violation: v=%d q=%d true=%.4f "
						"est=%.4f lb=%.4f\n",
						v,
						q,
						true_dist,
						est,
						lb);
			}

			mkt_rabitq_free_query(state);
		}

		mkt_free(encoded);
	}

	char msg[128];
	snprintf(msg, sizeof(msg), "%d lower bound violations", violations);
	ASSERT_EQ(0, violations, msg);

	mkt_rabitq_destroy(params);
}

TEST(symmetric_batch_matches_single)
{
	Dimension	  dim	   = 64;
	const int	  count	   = 8;
	RaBitQParams *params   = mkt_rabitq_create(dim, 42);
	float		 *centroid = alloc_test_vector(dim, 0);
	VectorRef	  cent_ref = {.data = centroid, .dim = dim};

	/* Encode vectors */
	float	*vectors	  = mkt_alloc((size_t)count * dim * sizeof(float));
	uint32_t packed_bytes = MKT_RABITQ_BYTES(dim);
	float	*f_add		  = mkt_alloc(count * sizeof(float));
	float	*f_rescale	  = mkt_alloc(count * sizeof(float));
	uint8_t *bits		  = mkt_alloc((size_t)count * packed_bytes);

	for (int i = 0; i < count; i++)
	{
		float *v = vectors + i * dim;
		for (Dimension j = 0; j < dim; j++)
			v[j] = (float)((i * 17 + j * 13) % 100 - 50) / 10.0f;
	}

	mkt_rabitq_encode_batch(
			params,
			vectors,
			MKT_VEC_F32,
			cent_ref,
			f_add,
			f_rescale,
			bits,
			count);

	/* Prepare query */
	float	 *query		= alloc_test_vector(dim, 100);
	VectorRef query_ref = {.data = query, .dim = dim};

	RaBitQQueryState *state =
			mkt_rabitq_prepare_query(params, query_ref, cent_ref);

	/* Batch symmetric distance */
	Distance batch_dists[8];
	mkt_rabitq_distance_batch_symmetric(
			state, f_add, f_rescale, bits, count, dim, batch_dists);

	/* Compare with single-vector symmetric distance */
	for (int i = 0; i < count; i++)
	{
		RaBitQData *data = (RaBitQData *)(void *)mkt_alloc(
				MKT_RABITQ_DATA_SIZE(dim));
		data->f_add		= f_add[i];
		data->f_rescale = f_rescale[i];
		memcpy(data->bits, bits + (size_t)i * packed_bytes, packed_bytes);

		Distance single = mkt_rabitq_distance_symmetric(state, data, dim);

		char msg[128];
		snprintf(
				msg,
				sizeof(msg),
				"vec %d: batch (%.4f) should match single (%.4f)",
				i,
				batch_dists[i],
				single);
		ASSERT_FLOAT_EQ(single, batch_dists[i], 1e-6f, msg);

		mkt_free(data);
	}

	mkt_rabitq_free_query(state);
	mkt_free(bits);
	mkt_free(f_rescale);
	mkt_free(f_add);
	mkt_free(vectors);
	mkt_rabitq_destroy(params);
}

TEST(symmetric_distance_null_inputs)
{
	Dimension	  dim	   = 8;
	RaBitQParams *params   = mkt_rabitq_create(dim, 1);
	float		 *input	   = alloc_test_vector(dim, 0);
	float		 *query	   = alloc_test_vector(dim, 10);
	float		 *centroid = alloc_test_vector(dim, 5);

	VectorRef input_ref	   = {.data = input, .dim = dim};
	VectorRef query_ref	   = {.data = query, .dim = dim};
	VectorRef centroid_ref = {.data = centroid, .dim = dim};

	RaBitQData *encoded = mkt_rabitq_encode(params, input_ref, centroid_ref);
	RaBitQQueryState *state =
			mkt_rabitq_prepare_query(params, query_ref, centroid_ref);

	Distance d = mkt_rabitq_distance_symmetric(NULL, encoded, dim);
	ASSERT_FLOAT_EQ(-1.0f, d, 1e-6f, "null state should return -1");

	d = mkt_rabitq_distance_symmetric(state, NULL, dim);
	ASSERT_FLOAT_EQ(-1.0f, d, 1e-6f, "null data should return -1");

	d = mkt_rabitq_distance_symmetric(state, encoded, 32);
	ASSERT_FLOAT_EQ(-1.0f, d, 1e-6f, "dim mismatch should return -1");

	mkt_rabitq_free_query(state);
	mkt_free(encoded);
	mkt_rabitq_destroy(params);
}

/*
 * Distance Mode Dispatch Tests
 */

TEST(distance_mode_name)
{
	ASSERT_STR_EQ(
			"asymmetric",
			mkt_distance_mode_name(MKT_DISTANCE_MODE_ASYMMETRIC),
			"asymmetric name");
	ASSERT_STR_EQ(
			"symmetric",
			mkt_distance_mode_name(MKT_DISTANCE_MODE_SYMMETRIC),
			"symmetric name");
}

TEST(prepare_query_default_is_asymmetric)
{
	Dimension	  dim	   = 32;
	RaBitQParams *params   = mkt_rabitq_create(dim, 42);
	float		 *query	   = alloc_test_vector(dim, 100);
	float		 *centroid = alloc_test_vector(dim, 50);

	VectorRef query_ref	   = {.data = query, .dim = dim};
	VectorRef centroid_ref = {.data = centroid, .dim = dim};

	RaBitQQueryState *state =
			mkt_rabitq_prepare_query(params, query_ref, centroid_ref);
	ASSERT_NOT_NULL(state, "prepare_query should succeed");

	ASSERT_EQ(
			MKT_DISTANCE_MODE_ASYMMETRIC,
			state->mode,
			"default mode should be asymmetric");
	ASSERT_NOT_NULL(
			(void *)(uintptr_t)state->distance_fn,
			"distance_fn should be set");
	ASSERT_NOT_NULL(
			(void *)(uintptr_t)state->distance_with_bound_fn,
			"distance_with_bound_fn should be set");

	mkt_rabitq_free_query(state);
	mkt_rabitq_destroy(params);
}

TEST(prepare_query_ex_asymmetric)
{
	/* Dispatch through _ex with asymmetric should match direct call */
	Dimension	  dim	   = 64;
	RaBitQParams *params   = mkt_rabitq_create(dim, 42);
	float		 *input	   = alloc_test_vector(dim, 0);
	float		 *query	   = alloc_test_vector(dim, 30);
	float		 *centroid = alloc_test_vector(dim, 50);

	VectorRef input_ref	   = {.data = input, .dim = dim};
	VectorRef query_ref	   = {.data = query, .dim = dim};
	VectorRef centroid_ref = {.data = centroid, .dim = dim};

	RaBitQData *encoded = mkt_rabitq_encode(params, input_ref, centroid_ref);

	RaBitQQueryState *state = mkt_rabitq_prepare_query_ex(
			params, query_ref, centroid_ref, MKT_DISTANCE_MODE_ASYMMETRIC);
	ASSERT_NOT_NULL(state, "prepare_query_ex should succeed");

	Distance direct	  = mkt_rabitq_distance(state, encoded, dim);
	Distance dispatch = mkt_rabitq_distance_dispatch(state, encoded, dim);

	ASSERT_FLOAT_EQ(
			direct, dispatch, 1e-6f, "dispatch should match direct call");

	mkt_rabitq_free_query(state);
	mkt_free(encoded);
	mkt_rabitq_destroy(params);
}

TEST(prepare_query_ex_symmetric)
{
	/* Dispatch through _ex with symmetric should match direct call */
	Dimension	  dim	   = 64;
	RaBitQParams *params   = mkt_rabitq_create(dim, 42);
	float		 *input	   = alloc_test_vector(dim, 0);
	float		 *query	   = alloc_test_vector(dim, 30);
	float		 *centroid = alloc_test_vector(dim, 50);

	VectorRef input_ref	   = {.data = input, .dim = dim};
	VectorRef query_ref	   = {.data = query, .dim = dim};
	VectorRef centroid_ref = {.data = centroid, .dim = dim};

	RaBitQData *encoded = mkt_rabitq_encode(params, input_ref, centroid_ref);

	RaBitQQueryState *state = mkt_rabitq_prepare_query_ex(
			params, query_ref, centroid_ref, MKT_DISTANCE_MODE_SYMMETRIC);
	ASSERT_NOT_NULL(state, "prepare_query_ex should succeed");

	Distance direct	  = mkt_rabitq_distance_symmetric(state, encoded, dim);
	Distance dispatch = mkt_rabitq_distance_dispatch(state, encoded, dim);

	ASSERT_FLOAT_EQ(
			direct, dispatch, 1e-6f, "dispatch should match direct call");

	mkt_rabitq_free_query(state);
	mkt_free(encoded);
	mkt_rabitq_destroy(params);
}

TEST(dispatch_with_bound_both_modes)
{
	/* Both modes should produce valid bounds via dispatch */
	Dimension	  dim	   = 64;
	RaBitQParams *params   = mkt_rabitq_create(dim, 42);
	float		 *input	   = alloc_test_vector(dim, 0);
	float		 *query	   = alloc_test_vector(dim, 30);
	float		 *centroid = alloc_test_vector(dim, 0);

	VectorRef input_ref	   = {.data = input, .dim = dim};
	VectorRef query_ref	   = {.data = query, .dim = dim};
	VectorRef centroid_ref = {.data = centroid, .dim = dim};

	RaBitQData *encoded	  = mkt_rabitq_encode(params, input_ref, centroid_ref);
	Distance	true_dist = true_l2_distance(input_ref, query_ref);

	/* Asymmetric dispatch */
	RaBitQQueryState *asym = mkt_rabitq_prepare_query_ex(
			params, query_ref, centroid_ref, MKT_DISTANCE_MODE_ASYMMETRIC);

	Distance est_a, lb_a;
	mkt_rabitq_distance_dispatch_with_bound(asym, encoded, dim, &est_a, &lb_a);

	ASSERT_TRUE(isfinite(est_a), "asymmetric est should be finite");
	ASSERT_TRUE(
			lb_a <= true_dist + 1e-3f, "asymmetric lower bound should hold");

	/* Symmetric dispatch */
	RaBitQQueryState *sym = mkt_rabitq_prepare_query_ex(
			params, query_ref, centroid_ref, MKT_DISTANCE_MODE_SYMMETRIC);

	Distance est_s, lb_s;
	mkt_rabitq_distance_dispatch_with_bound(sym, encoded, dim, &est_s, &lb_s);

	ASSERT_TRUE(isfinite(est_s), "symmetric est should be finite");
	ASSERT_TRUE(
			lb_s <= true_dist + 1e-2f, "symmetric lower bound should hold");

	TEST_PRINT(
			"Asymmetric: est=%.4f lb=%.4f | "
			"Symmetric: est=%.4f lb=%.4f | True: %.4f\n",
			est_a,
			lb_a,
			est_s,
			lb_s,
			true_dist);

	mkt_rabitq_free_query(asym);
	mkt_rabitq_free_query(sym);
	mkt_free(encoded);
	mkt_rabitq_destroy(params);
}

/*
 * Batch with_bound tests
 */

TEST(batch_multi_with_bound_lower_bounds_valid)
{
	Dimension	  dim	   = 64;
	const int	  count	   = 8;
	RaBitQParams *params   = mkt_rabitq_create(dim, 42);
	float		 *centroid = alloc_test_vector(dim, 0);
	VectorRef	  cent_ref = {.data = centroid, .dim = dim};

	/* Generate and encode vectors */
	float *vectors = mkt_alloc((size_t)count * dim * sizeof(float));
	for (int i = 0; i < count; i++)
	{
		float *v = vectors + i * dim;
		for (Dimension j = 0; j < dim; j++)
			v[j] = (float)((i * 17 + j * 13) % 100 - 50) / 10.0f;
	}

	uint32_t packed_bytes = MKT_RABITQ_BYTES(dim);
	float	*f_add		  = mkt_alloc(count * sizeof(float));
	float	*f_rescale	  = mkt_alloc(count * sizeof(float));
	uint8_t *bits		  = mkt_alloc((size_t)count * packed_bytes);
	int		 ret		  = mkt_rabitq_encode_batch(
			  params,
			  vectors,
			  MKT_VEC_F32,
			  cent_ref,
			  f_add,
			  f_rescale,
			  bits,
			  count);
	ASSERT_EQ(0, ret, "batch encode should succeed");

	/* Prepare query */
	float	 *query		= alloc_test_vector(dim, 100);
	VectorRef query_ref = {.data = query, .dim = dim};

	RaBitQQueryState *qstate =
			mkt_rabitq_prepare_query(params, query_ref, cent_ref);
	ASSERT_NOT_NULL(qstate, "prepare_query should succeed");

	/* Compute batch distances with bounds */
	Distance dists[8];
	Distance lower_bounds[8];
	float	 scratch[8];
	mkt_rabitq_distance_batch_multi_with_bound(
			qstate,
			f_add,
			f_rescale,
			bits,
			packed_bytes,
			count,
			dim,
			dists,
			lower_bounds,
			scratch);

	/* Compute reference distances without bounds */
	Distance ref_dists[8];
	mkt_rabitq_distance_batch_multi(
			qstate, f_add, f_rescale, bits, count, dim, ref_dists);

	/* Verify distances match and lower bounds are valid */
	int violations = 0;
	for (int i = 0; i < count; i++)
	{
		char msg[128];

		/* Distances should match the non-bound variant */
		snprintf(
				msg,
				sizeof(msg),
				"vec %d: with_bound (%.4f) vs multi (%.4f)",
				i,
				dists[i],
				ref_dists[i]);
		ASSERT_FLOAT_EQ(ref_dists[i], dists[i], 1e-6f, msg);

		/* Lower bound must be <= estimated distance */
		snprintf(
				msg,
				sizeof(msg),
				"vec %d: lb (%.4f) should be <= est (%.4f)",
				i,
				lower_bounds[i],
				dists[i]);
		ASSERT_TRUE(lower_bounds[i] <= dists[i], msg);

		/* Lower bound must be <= true L2 distance */
		VectorRef vec_ref	= {.data = vectors + i * dim, .dim = dim};
		Distance  true_dist = true_l2_distance(vec_ref, query_ref);
		if (lower_bounds[i] > true_dist + 1e-3f)
			violations++;
	}

	char msg[128];
	snprintf(msg, sizeof(msg), "%d lower bound violations", violations);
	ASSERT_EQ(0, violations, msg);

	mkt_rabitq_free_query(qstate);
	mkt_free(bits);
	mkt_free(f_rescale);
	mkt_free(f_add);
	mkt_free(vectors);
	mkt_rabitq_destroy(params);
}

TEST(batch_symmetric_with_bound_lower_bounds_valid)
{
	Dimension	  dim	   = 64;
	const int	  count	   = 8;
	RaBitQParams *params   = mkt_rabitq_create(dim, 42);
	float		 *centroid = alloc_test_vector(dim, 0);
	VectorRef	  cent_ref = {.data = centroid, .dim = dim};

	/* Generate and encode vectors */
	float *vectors = mkt_alloc((size_t)count * dim * sizeof(float));
	for (int i = 0; i < count; i++)
	{
		float *v = vectors + i * dim;
		for (Dimension j = 0; j < dim; j++)
			v[j] = (float)((i * 17 + j * 13) % 100 - 50) / 10.0f;
	}

	uint32_t packed_bytes = MKT_RABITQ_BYTES(dim);
	float	*f_add		  = mkt_alloc(count * sizeof(float));
	float	*f_rescale	  = mkt_alloc(count * sizeof(float));
	uint8_t *bits		  = mkt_alloc((size_t)count * packed_bytes);
	int		 ret		  = mkt_rabitq_encode_batch(
			  params,
			  vectors,
			  MKT_VEC_F32,
			  cent_ref,
			  f_add,
			  f_rescale,
			  bits,
			  count);
	ASSERT_EQ(0, ret, "batch encode should succeed");

	/* Prepare symmetric query */
	float	 *query		= alloc_test_vector(dim, 100);
	VectorRef query_ref = {.data = query, .dim = dim};

	RaBitQQueryState *qstate = mkt_rabitq_prepare_query_ex(
			params, query_ref, cent_ref, MKT_DISTANCE_MODE_SYMMETRIC);
	ASSERT_NOT_NULL(qstate, "prepare_query_ex should succeed");

	/* Compute batch distances with bounds */
	Distance dists[8];
	Distance lower_bounds[8];
	uint32_t scratch[8];
	mkt_rabitq_distance_batch_symmetric_with_bound(
			qstate,
			f_add,
			f_rescale,
			bits,
			packed_bytes,
			count,
			dim,
			dists,
			lower_bounds,
			scratch);

	/* Compute reference distances without bounds */
	Distance ref_dists[8];
	mkt_rabitq_distance_batch_symmetric(
			qstate, f_add, f_rescale, bits, count, dim, ref_dists);

	/* Verify distances match and lower bounds are valid */
	int violations = 0;
	for (int i = 0; i < count; i++)
	{
		char msg[128];

		/* Distances should match the non-bound variant */
		snprintf(
				msg,
				sizeof(msg),
				"vec %d: with_bound (%.4f) vs batch_sym (%.4f)",
				i,
				dists[i],
				ref_dists[i]);
		ASSERT_FLOAT_EQ(ref_dists[i], dists[i], 1e-6f, msg);

		/* Lower bound must be <= estimated distance */
		snprintf(
				msg,
				sizeof(msg),
				"vec %d: lb (%.4f) should be <= est (%.4f)",
				i,
				lower_bounds[i],
				dists[i]);
		ASSERT_TRUE(lower_bounds[i] <= dists[i], msg);

		/* Lower bound must be <= true L2 distance (wider tolerance
		 * for symmetric since both query and data are quantized) */
		VectorRef vec_ref	= {.data = vectors + i * dim, .dim = dim};
		Distance  true_dist = true_l2_distance(vec_ref, query_ref);
		if (lower_bounds[i] > true_dist + 1e-2f)
			violations++;
	}

	char msg[128];
	snprintf(msg, sizeof(msg), "%d lower bound violations", violations);
	ASSERT_EQ(0, violations, msg);

	mkt_rabitq_free_query(qstate);
	mkt_free(bits);
	mkt_free(f_rescale);
	mkt_free(f_add);
	mkt_free(vectors);
	mkt_rabitq_destroy(params);
}
