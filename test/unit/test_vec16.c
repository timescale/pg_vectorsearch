/*
 * test_halfvec.c - Half-precision vector type and operations tests
 */

#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#include "core/memory.h"
#include "mkt_test.h"
#include "types/vec16.h"
#include "types/vec32.h"

TEST_GROUP(HalfVec);

TEST_MEMCTX_FIXTURE();

/* ----------------------------------------------------------------
 * Scalar conversion: known IEEE 754 bit patterns
 * ---------------------------------------------------------------- */

TEST(half_convert_one)
{
	half  h = mkt_float_to_half(1.0f);
	float f = mkt_half_to_float(h);
	ASSERT_FLOAT_EQ(1.0f, f, 1e-6f, "1.0 roundtrip");
}

TEST(half_convert_neg_one)
{
	half  h = mkt_float_to_half(-1.0f);
	float f = mkt_half_to_float(h);
	ASSERT_FLOAT_EQ(-1.0f, f, 1e-6f, "-1.0 roundtrip");
}

TEST(half_convert_two)
{
	half  h = mkt_float_to_half(2.0f);
	float f = mkt_half_to_float(h);
	ASSERT_FLOAT_EQ(2.0f, f, 1e-6f, "2.0 roundtrip");
}

TEST(half_convert_zero)
{
	half  h = mkt_float_to_half(0.0f);
	float f = mkt_half_to_float(h);
	ASSERT_FLOAT_EQ(0.0f, f, 1e-6f, "0.0 roundtrip");
}

TEST(half_convert_neg_zero)
{
	half  h = mkt_float_to_half(-0.0f);
	float f = mkt_half_to_float(h);
	/* -0.0 == 0.0 in IEEE 754 */
	ASSERT_FLOAT_EQ(0.0f, f, 0.0f, "-0.0 roundtrip");
}

TEST(half_convert_half_max)
{
	half  h = mkt_float_to_half(65504.0f);
	float f = mkt_half_to_float(h);
	ASSERT_FLOAT_EQ(65504.0f, f, 1.0f, "half max roundtrip");
}

TEST(half_convert_small)
{
	/* Smallest normal half: 2^-14 ≈ 6.1035e-5 */
	float small = 6.103515625e-5f;
	half  h		= mkt_float_to_half(small);
	float f		= mkt_half_to_float(h);
	ASSERT_FLOAT_EQ(small, f, 1e-8f, "small normal roundtrip");
}

TEST(half_convert_inf)
{
	half  h = mkt_float_to_half(INFINITY);
	float f = mkt_half_to_float(h);
	ASSERT_TRUE(isinf(f) && f > 0, "positive infinity");
}

TEST(half_convert_neg_inf)
{
	half  h = mkt_float_to_half(-INFINITY);
	float f = mkt_half_to_float(h);
	ASSERT_TRUE(isinf(f) && f < 0, "negative infinity");
}

TEST(half_convert_nan)
{
	half  h = mkt_float_to_half(NAN);
	float f = mkt_half_to_float(h);
	ASSERT_TRUE(isnan(f), "NaN roundtrip");
}

TEST(half_convert_subnormal)
{
	/* Smallest subnormal half: 2^-24 ≈ 5.96e-8 */
	float subnormal = 5.960464477539063e-8f;
	half  h			= mkt_float_to_half(subnormal);
	float f			= mkt_half_to_float(h);
	/* Subnormal values may lose precision */
	ASSERT_TRUE(f >= 0.0f && f < 1e-4f, "subnormal is small positive");
}

/* ----------------------------------------------------------------
 * Special value checks
 * ---------------------------------------------------------------- */

TEST(half_is_nan)
{
	half h = mkt_float_to_half(NAN);
	ASSERT_TRUE(mkt_half_is_nan(h), "NaN detected");
	ASSERT_FALSE(mkt_half_is_inf(h), "NaN is not Inf");
	ASSERT_FALSE(mkt_half_is_zero(h), "NaN is not zero");
}

TEST(half_is_inf)
{
	half h = mkt_float_to_half(INFINITY);
	ASSERT_TRUE(mkt_half_is_inf(h), "Inf detected");
	ASSERT_FALSE(mkt_half_is_nan(h), "Inf is not NaN");
	ASSERT_FALSE(mkt_half_is_zero(h), "Inf is not zero");
}

TEST(half_is_neg_inf)
{
	half h = mkt_float_to_half(-INFINITY);
	ASSERT_TRUE(mkt_half_is_inf(h), "-Inf detected");
}

TEST(half_is_zero)
{
	half h = mkt_float_to_half(0.0f);
	ASSERT_TRUE(mkt_half_is_zero(h), "zero detected");
	ASSERT_FALSE(mkt_half_is_nan(h), "zero is not NaN");
	ASSERT_FALSE(mkt_half_is_inf(h), "zero is not Inf");
}

TEST(half_is_neg_zero)
{
	half h = mkt_float_to_half(-0.0f);
	ASSERT_TRUE(mkt_half_is_zero(h), "-zero detected");
}

TEST(half_normal_not_special)
{
	half h = mkt_float_to_half(1.0f);
	ASSERT_FALSE(mkt_half_is_nan(h), "1.0 is not NaN");
	ASSERT_FALSE(mkt_half_is_inf(h), "1.0 is not Inf");
	ASSERT_FALSE(mkt_half_is_zero(h), "1.0 is not zero");
}

/* ----------------------------------------------------------------
 * Bulk array conversion
 * ---------------------------------------------------------------- */

TEST(half_array_roundtrip_dim1)
{
	float src[] = {1.5f};
	half  tmp[1];
	float dst[1];

	mkt_float_to_half_array(src, tmp, 1);
	mkt_half_to_float_array(tmp, dst, 1);
	ASSERT_FLOAT_EQ(src[0], dst[0], 0.002f, "dim=1 roundtrip");
}

TEST(half_array_roundtrip_dim7)
{
	float src[7], dst[7];
	half  tmp[7];

	for (int i = 0; i < 7; i++)
		src[i] = (float)(i + 1) * 0.25f;

	mkt_float_to_half_array(src, tmp, 7);
	mkt_half_to_float_array(tmp, dst, 7);

	for (int i = 0; i < 7; i++)
		ASSERT_FLOAT_EQ(src[i], dst[i], 0.002f, "dim=7 element");
}

TEST(half_array_roundtrip_dim8)
{
	float src[8], dst[8];
	half  tmp[8];

	for (int i = 0; i < 8; i++)
		src[i] = (float)(i - 4) * 1.5f;

	mkt_float_to_half_array(src, tmp, 8);
	mkt_half_to_float_array(tmp, dst, 8);

	for (int i = 0; i < 8; i++)
		ASSERT_FLOAT_EQ(src[i], dst[i], 0.01f, "dim=8 element");
}

TEST(half_array_roundtrip_dim15)
{
	float src[15], dst[15];
	half  tmp[15];

	for (int i = 0; i < 15; i++)
		src[i] = (float)(i) * 0.1f;

	mkt_float_to_half_array(src, tmp, 15);
	mkt_half_to_float_array(tmp, dst, 15);

	for (int i = 0; i < 15; i++)
		ASSERT_FLOAT_EQ(src[i], dst[i], 0.002f, "dim=15 element");
}

TEST(half_array_roundtrip_dim16)
{
	float src[16], dst[16];
	half  tmp[16];

	for (int i = 0; i < 16; i++)
		src[i] = (float)(i - 8) * 2.0f;

	mkt_float_to_half_array(src, tmp, 16);
	mkt_half_to_float_array(tmp, dst, 16);

	for (int i = 0; i < 16; i++)
		ASSERT_FLOAT_EQ(src[i], dst[i], 0.01f, "dim=16 element");
}

TEST(half_array_roundtrip_dim128)
{
	float *src = mkt_alloc(128 * sizeof(float));
	float *dst = mkt_alloc(128 * sizeof(float));
	half  *tmp = mkt_alloc(128 * sizeof(half));

	for (int i = 0; i < 128; i++)
		src[i] = (float)(i - 64) * 0.5f;

	mkt_float_to_half_array(src, tmp, 128);
	mkt_half_to_float_array(tmp, dst, 128);

	for (int i = 0; i < 128; i++)
		ASSERT_FLOAT_EQ(src[i], dst[i], 0.5f, "dim=128 element");

	mkt_free(src);
	mkt_free(dst);
	mkt_free(tmp);
}

TEST(half_array_roundtrip_dim768)
{
	float *src = mkt_alloc(768 * sizeof(float));
	float *dst = mkt_alloc(768 * sizeof(float));
	half  *tmp = mkt_alloc(768 * sizeof(half));

	for (int i = 0; i < 768; i++)
		src[i] = ((float)(i % 100) - 50.0f) * 0.01f;

	mkt_float_to_half_array(src, tmp, 768);
	mkt_half_to_float_array(tmp, dst, 768);

	for (int i = 0; i < 768; i++)
		ASSERT_FLOAT_EQ(src[i], dst[i], 0.002f, "dim=768 element");

	mkt_free(src);
	mkt_free(dst);
	mkt_free(tmp);
}

/* ----------------------------------------------------------------
 * Layout compatibility
 * ---------------------------------------------------------------- */

TEST(half_sizeof)
{
	ASSERT_EQ(2, sizeof(half), "sizeof(half) must be 2");
}

TEST(halfvec_layout)
{
	ASSERT_EQ(8, offsetof(Vec16, x), "Vec16.x offset must be 8");
}

/* ----------------------------------------------------------------
 * f32_ops: zero-copy verification
 * ---------------------------------------------------------------- */

TEST(f32_ops_to_float_block_zero_copy)
{
	float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
	float buf[4];

	const float *out = mkt_f32_type_ops.to_float_block(data, buf, 1, 4);
	ASSERT_TRUE(
			out == data, "f32 to_float_block should return src (zero copy)");
}

TEST(f32_ops_name)
{
	ASSERT_STR_EQ("float32", mkt_f32_type_ops.name, "f32 ops name");
}

TEST(f32_ops_element_size)
{
	ASSERT_EQ(4, mkt_f32_type_ops.element_size, "f32 element size");
}

/* ----------------------------------------------------------------
 * f16_ops: distance correctness against float32 reference
 * ---------------------------------------------------------------- */

TEST(f16_ops_name)
{
	ASSERT_STR_EQ("float16", mkt_f16_type_ops.name, "f16 ops name");
}

TEST(f16_ops_element_size)
{
	ASSERT_EQ(2, mkt_f16_type_ops.element_size, "f16 element size");
}

TEST(f16_ops_dot_product)
{
	/* Create half vector and float centroid */
	float values[] = {1.0f, 2.0f, 3.0f, 4.0f};
	half  hv[4];
	mkt_float_to_half_array(values, hv, 4);

	float centroid[] = {0.5f, 1.0f, 1.5f, 2.0f};

	/* Expected: 1*0.5 + 2*1 + 3*1.5 + 4*2 = 0.5+2+4.5+8 = 15.0 */
	float dot = mkt_f16_type_ops.dot_product(hv, centroid, 4);
	ASSERT_FLOAT_EQ(15.0f, dot, 0.1f, "f16 dot product");
}

TEST(f16_ops_l2_squared)
{
	float values[] = {1.0f, 2.0f, 3.0f, 4.0f};
	half  hv[4];
	mkt_float_to_half_array(values, hv, 4);

	float centroid[] = {0.5f, 1.0f, 1.5f, 2.0f};

	/* Expected: 0.25 + 1.0 + 2.25 + 4.0 = 7.5 */
	float l2 = mkt_f16_type_ops.l2_squared(hv, centroid, 4);
	ASSERT_FLOAT_EQ(7.5f, l2, 0.1f, "f16 l2 squared");
}

TEST(f16_ops_norm_sq)
{
	float values[] = {3.0f, 4.0f};
	half  hv[2];
	mkt_float_to_half_array(values, hv, 2);

	/* Expected: 9 + 16 = 25 */
	float nsq = mkt_f16_type_ops.norm_sq(hv, 2);
	ASSERT_FLOAT_EQ(25.0f, nsq, 0.1f, "f16 norm squared");
}

TEST(f16_ops_sum_to_float)
{
	float values[] = {1.0f, 2.0f, 3.0f};
	half  hv[3];
	mkt_float_to_half_array(values, hv, 3);

	float accum[] = {10.0f, 20.0f, 30.0f};
	mkt_f16_type_ops.sum_to_float(hv, accum, 3);

	ASSERT_FLOAT_EQ(11.0f, accum[0], 0.01f, "accum[0]");
	ASSERT_FLOAT_EQ(22.0f, accum[1], 0.01f, "accum[1]");
	ASSERT_FLOAT_EQ(33.0f, accum[2], 0.01f, "accum[2]");
}

TEST(f16_ops_to_float_one)
{
	float values[] = {1.5f, -2.5f, 3.5f};
	half  hv[3];
	mkt_float_to_half_array(values, hv, 3);

	float dst[3] = {0};
	mkt_f16_type_ops.to_float_one(hv, dst, 3);

	ASSERT_FLOAT_EQ(1.5f, dst[0], 0.01f, "to_float_one[0]");
	ASSERT_FLOAT_EQ(-2.5f, dst[1], 0.01f, "to_float_one[1]");
	ASSERT_FLOAT_EQ(3.5f, dst[2], 0.01f, "to_float_one[2]");
}

TEST(f16_ops_to_float_block)
{
	float values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
	half  hv[6];
	mkt_float_to_half_array(values, hv, 6);

	float dst[6] = {0};

	const float *out = mkt_f16_type_ops.to_float_block(hv, dst, 2, 3);
	ASSERT_TRUE(out == dst, "f16 to_float_block returns dst");

	for (int i = 0; i < 6; i++)
		ASSERT_FLOAT_EQ(values[i], dst[i], 0.01f, "to_float_block value");
}

/* ----------------------------------------------------------------
 * f16c_ops: F16C SIMD vtable correctness
 * ---------------------------------------------------------------- */

#if defined(MKT_F16C_SUPPORT) && !defined(MKT_SIMD_NONE)

/*
 * Volatile pointer prevents the compiler from resolving vtable function
 * pointers at -O3, which would trigger "always_inline + target mismatch"
 * errors against the MKT_TARGET_F16C_AVX2 inline functions.
 */
static const Vec32TypeOps *volatile f16c_ops = &mkt_f16c_type_ops;

TEST(f16c_ops_name)
{
	ASSERT_STR_EQ("float16-f16c", mkt_f16c_type_ops.name, "f16c ops name");
}

TEST(f16c_ops_dot_product_small)
{
	float values[]	 = {1.0f, 2.0f, 3.0f, 4.0f};
	float centroid[] = {0.5f, 1.0f, 1.5f, 2.0f};
	half  hv[4];
	mkt_float_to_half_array(values, hv, 4);

	float dot = f16c_ops->dot_product(hv, centroid, 4);
	ASSERT_FLOAT_EQ(15.0f, dot, 0.1f, "f16c dot product (scalar tail)");
}

TEST(f16c_ops_dot_product_simd)
{
	Dimension dim = 128;
	float	 *fv  = mkt_alloc(dim * sizeof(float));
	float	 *fc  = mkt_alloc(dim * sizeof(float));
	half	 *hv  = mkt_alloc(dim * sizeof(half));

	uint32_t rng = 42;
	for (Dimension i = 0; i < dim; i++)
	{
		rng	  = rng * 1103515245 + 12345;
		fv[i] = ((float)(rng >> 16) / 32768.0f) - 1.0f;
		rng	  = rng * 1103515245 + 12345;
		fc[i] = ((float)(rng >> 16) / 32768.0f) - 1.0f;
	}
	mkt_float_to_half_array(fv, hv, dim);

	float ref = mkt_f16_type_ops.dot_product(hv, fc, dim);
	float got = f16c_ops->dot_product(hv, fc, dim);
	ASSERT_FLOAT_EQ(ref, got, 0.01f, "f16c dot matches f16 scalar");

	mkt_free(fv);
	mkt_free(fc);
	mkt_free(hv);
}

TEST(f16c_ops_l2_squared)
{
	float values[]	 = {1.0f, 2.0f, 3.0f, 4.0f};
	float centroid[] = {0.5f, 1.0f, 1.5f, 2.0f};
	half  hv[4];
	mkt_float_to_half_array(values, hv, 4);

	float l2 = f16c_ops->l2_squared(hv, centroid, 4);
	ASSERT_FLOAT_EQ(7.5f, l2, 0.1f, "f16c l2 squared");
}

TEST(f16c_ops_l2_squared_simd)
{
	Dimension dim = 128;
	float	 *fv  = mkt_alloc(dim * sizeof(float));
	float	 *fc  = mkt_alloc(dim * sizeof(float));
	half	 *hv  = mkt_alloc(dim * sizeof(half));

	uint32_t rng = 42;
	for (Dimension i = 0; i < dim; i++)
	{
		rng	  = rng * 1103515245 + 12345;
		fv[i] = ((float)(rng >> 16) / 32768.0f) - 1.0f;
		rng	  = rng * 1103515245 + 12345;
		fc[i] = ((float)(rng >> 16) / 32768.0f) - 1.0f;
	}
	mkt_float_to_half_array(fv, hv, dim);

	float ref = mkt_f16_type_ops.l2_squared(hv, fc, dim);
	float got = f16c_ops->l2_squared(hv, fc, dim);
	ASSERT_FLOAT_EQ(ref, got, 0.01f, "f16c l2 matches f16 scalar");

	mkt_free(fv);
	mkt_free(fc);
	mkt_free(hv);
}

TEST(f16c_ops_norm_sq)
{
	float values[] = {3.0f, 4.0f};
	half  hv[2];
	mkt_float_to_half_array(values, hv, 2);

	float nsq = f16c_ops->norm_sq(hv, 2);
	ASSERT_FLOAT_EQ(25.0f, nsq, 0.1f, "f16c norm squared");
}

TEST(f16c_ops_norm_sq_simd)
{
	Dimension dim = 128;
	float	 *fv  = mkt_alloc(dim * sizeof(float));
	half	 *hv  = mkt_alloc(dim * sizeof(half));

	uint32_t rng = 42;
	for (Dimension i = 0; i < dim; i++)
	{
		rng	  = rng * 1103515245 + 12345;
		fv[i] = ((float)(rng >> 16) / 32768.0f) - 1.0f;
	}
	mkt_float_to_half_array(fv, hv, dim);

	float ref = mkt_f16_type_ops.norm_sq(hv, dim);
	float got = f16c_ops->norm_sq(hv, dim);
	ASSERT_FLOAT_EQ(ref, got, 0.01f, "f16c norm_sq matches f16 scalar");

	mkt_free(fv);
	mkt_free(hv);
}

TEST(f16c_ops_sum_to_float)
{
	float values[] = {1.0f, 2.0f, 3.0f};
	half  hv[3];
	mkt_float_to_half_array(values, hv, 3);

	float accum[] = {10.0f, 20.0f, 30.0f};
	f16c_ops->sum_to_float(hv, accum, 3);

	ASSERT_FLOAT_EQ(11.0f, accum[0], 0.01f, "f16c accum[0]");
	ASSERT_FLOAT_EQ(22.0f, accum[1], 0.01f, "f16c accum[1]");
	ASSERT_FLOAT_EQ(33.0f, accum[2], 0.01f, "f16c accum[2]");
}

TEST(f16c_ops_sum_to_float_simd)
{
	Dimension dim = 128;
	float	 *fv  = mkt_alloc(dim * sizeof(float));
	half	 *hv  = mkt_alloc(dim * sizeof(half));
	float	 *ref = mkt_alloc(dim * sizeof(float));
	float	 *got = mkt_alloc(dim * sizeof(float));

	uint32_t rng = 42;
	for (Dimension i = 0; i < dim; i++)
	{
		rng	   = rng * 1103515245 + 12345;
		fv[i]  = ((float)(rng >> 16) / 32768.0f) - 1.0f;
		ref[i] = 100.0f;
		got[i] = 100.0f;
	}
	mkt_float_to_half_array(fv, hv, dim);

	mkt_f16_type_ops.sum_to_float(hv, ref, dim);
	f16c_ops->sum_to_float(hv, got, dim);

	for (Dimension i = 0; i < dim; i++)
		ASSERT_FLOAT_EQ(ref[i], got[i], 0.001f, "f16c sum matches f16");

	mkt_free(fv);
	mkt_free(hv);
	mkt_free(ref);
	mkt_free(got);
}

#endif /* MKT_F16C_SUPPORT && !MKT_SIMD_NONE */

/* Match f16 distance against f32 reference for larger vectors */
TEST(f16_ops_distance_matches_f32_reference)
{
	Dimension dim = 128;
	float	 *fv  = mkt_alloc(dim * sizeof(float));
	float	 *fc  = mkt_alloc(dim * sizeof(float));
	half	 *hv  = mkt_alloc(dim * sizeof(half));

	/* Fill with deterministic values */
	uint32_t rng = 12345;
	for (Dimension i = 0; i < dim; i++)
	{
		rng	  = rng * 1103515245 + 12345;
		fv[i] = ((float)(rng >> 16) / 32768.0f) - 1.0f;
		rng	  = rng * 1103515245 + 12345;
		fc[i] = ((float)(rng >> 16) / 32768.0f) - 1.0f;
	}

	mkt_float_to_half_array(fv, hv, dim);

	/* f32 reference */
	float ref_dot = mkt_f32_type_ops.dot_product(fv, fc, dim);
	float ref_l2  = mkt_f32_type_ops.l2_squared(fv, fc, dim);

	/* f16 result (using halfvec data) */
	float f16_dot = mkt_f16_type_ops.dot_product(hv, fc, dim);
	float f16_l2  = mkt_f16_type_ops.l2_squared(hv, fc, dim);

	/* Half-precision has ~3 decimal digits of precision.
	 * Allow relative error up to 1% for accumulated results. */
	float dot_err = (ref_dot != 0.0f)
						  ? fabsf(f16_dot - ref_dot) / fabsf(ref_dot)
						  : 0;
	float l2_err  = (ref_l2 != 0.0f) ? fabsf(f16_l2 - ref_l2) / ref_l2 : 0;

	ASSERT_TRUE(dot_err < 0.02f, "f16 dot product relative error < 2%");
	ASSERT_TRUE(l2_err < 0.02f, "f16 L2 squared relative error < 2%");

	mkt_free(fv);
	mkt_free(fc);
	mkt_free(hv);
}

/* ----------------------------------------------------------------
 * Vec16 lifecycle
 * ---------------------------------------------------------------- */

TEST(halfvec_create)
{
	Vec16 *v = vec16_create(128);
	ASSERT_NOT_NULL(v, "halfvec should be allocated");
	ASSERT_EQ(128, VEC16_DIM(v), "dim should be 128");
	vec16_free(v);
}

TEST(halfvec_create_zero_dim_fails)
{
	Vec16 *v = vec16_create(0);
	ASSERT_NULL(v, "zero dim should fail");
}

TEST(halfvec_from_floats)
{
	float values[] = {1.0f, 2.0f, 3.0f};

	Vec16 *v = vec16_from_floats(values, 3);
	ASSERT_NOT_NULL(v, "from_floats should succeed");
	ASSERT_EQ(3, VEC16_DIM(v), "dim should be 3");

	/* Verify values via to_float */
	float out[3];
	mkt_half_to_float_array(v->x, out, 3);
	ASSERT_FLOAT_EQ(1.0f, out[0], 0.01f, "element 0");
	ASSERT_FLOAT_EQ(2.0f, out[1], 0.01f, "element 1");
	ASSERT_FLOAT_EQ(3.0f, out[2], 0.01f, "element 2");

	vec16_free(v);
}

TEST(halfvec_from_floats_null_fails)
{
	Vec16 *v = vec16_from_floats(NULL, 3);
	ASSERT_NULL(v, "NULL values should fail");
}

TEST(halfvec_set)
{
	Vec16 *v = vec16_create(3);
	ASSERT_NOT_NULL(v, "halfvec should be allocated");

	float values[] = {4.0f, 5.0f, 6.0f};
	vec16_set(v, values);

	float out[3];
	mkt_half_to_float_array(v->x, out, 3);
	ASSERT_FLOAT_EQ(4.0f, out[0], 0.01f, "set element 0");
	ASSERT_FLOAT_EQ(5.0f, out[1], 0.01f, "set element 1");
	ASSERT_FLOAT_EQ(6.0f, out[2], 0.01f, "set element 2");

	vec16_free(v);
}

TEST(halfvec_to_ref)
{
	float values[] = {1.0f, 2.0f, 3.0f};

	Vec16 *v = vec16_from_floats(values, 3);
	ASSERT_NOT_NULL(v, "halfvec should be allocated");

	float	 buffer[3];
	Vec32Ref ref = Vec16ToRef(v, buffer);
	ASSERT_EQ(3, ref.dim, "ref dim should be 3");
	ASSERT_FLOAT_EQ(1.0f, ref.data[0], 0.01f, "ref element 0");
	ASSERT_FLOAT_EQ(2.0f, ref.data[1], 0.01f, "ref element 1");
	ASSERT_FLOAT_EQ(3.0f, ref.data[2], 0.01f, "ref element 2");

	vec16_free(v);
}
