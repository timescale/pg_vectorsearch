/*
 * test_mkt_vector.c - Vector type and operations tests
 */

#include <math.h>

#include "mkt_test.h"
#include "mkt_vector.h"

TEST_GROUP(Vector);

TEST(vector_create_and_dim)
{
	MktVector *v = mkt_vector_create(128);
	ASSERT_NOT_NULL(v, "vector should be allocated");
	ASSERT_EQ(128, MKT_VECTOR_DIM(v), "dimension should be 128");
	mkt_vector_free(v);
}

TEST(vector_create_zero_dim_fails)
{
	MktVector *v = mkt_vector_create(0);
	ASSERT_NULL(v, "zero dimension should fail");
}

TEST(vector_create_max_dim)
{
	MktVector *v = mkt_vector_create(MKT_VECTOR_MAX_DIM);
	ASSERT_NOT_NULL(v, "max dimension should succeed");
	ASSERT_EQ(
			MKT_VECTOR_MAX_DIM, MKT_VECTOR_DIM(v), "dimension should be max");
	mkt_vector_free(v);
}

TEST(vector_create_over_max_dim_fails)
{
	MktVector *v = mkt_vector_create(MKT_VECTOR_MAX_DIM + 1);
	ASSERT_NULL(v, "over max dimension should fail");
}

TEST(vector_set_get_elements)
{
	MktVector *v = mkt_vector_create(3);
	ASSERT_NOT_NULL(v, "vector should be allocated");

	float values[] = {1.0f, 2.0f, 3.0f};
	mkt_vector_set(v, values);

	ASSERT_FLOAT_EQ(1.0f, MKT_VECTOR_DATA(v)[0], 1e-6f, "element 0");
	ASSERT_FLOAT_EQ(2.0f, MKT_VECTOR_DATA(v)[1], 1e-6f, "element 1");
	ASSERT_FLOAT_EQ(3.0f, MKT_VECTOR_DATA(v)[2], 1e-6f, "element 2");

	mkt_vector_free(v);
}

TEST(vector_zero)
{
	MktVector *v = mkt_vector_create(3);
	ASSERT_NOT_NULL(v, "vector should be allocated");

	float values[] = {1.0f, 2.0f, 3.0f};
	mkt_vector_set(v, values);
	mkt_vector_zero(v);

	ASSERT_FLOAT_EQ(
			0.0f, MKT_VECTOR_DATA(v)[0], 1e-6f, "element 0 should be 0");
	ASSERT_FLOAT_EQ(
			0.0f, MKT_VECTOR_DATA(v)[1], 1e-6f, "element 1 should be 0");
	ASSERT_FLOAT_EQ(
			0.0f, MKT_VECTOR_DATA(v)[2], 1e-6f, "element 2 should be 0");

	mkt_vector_free(v);
}

TEST(vector_fill)
{
	MktVector *v = mkt_vector_create(3);
	ASSERT_NOT_NULL(v, "vector should be allocated");

	mkt_vector_fill(v, 5.0f);

	ASSERT_FLOAT_EQ(5.0f, MKT_VECTOR_DATA(v)[0], 1e-6f, "element 0");
	ASSERT_FLOAT_EQ(5.0f, MKT_VECTOR_DATA(v)[1], 1e-6f, "element 1");
	ASSERT_FLOAT_EQ(5.0f, MKT_VECTOR_DATA(v)[2], 1e-6f, "element 2");

	mkt_vector_free(v);
}

TEST(vector_copy)
{
	MktVector *a = mkt_vector_create(3);
	ASSERT_NOT_NULL(a, "vector should be allocated");

	float values[] = {1.0f, 2.0f, 3.0f};
	mkt_vector_set(a, values);

	MktVector *b = mkt_vector_copy(a);
	ASSERT_NOT_NULL(b, "copy should be allocated");
	ASSERT_EQ(MKT_VECTOR_DIM(a), MKT_VECTOR_DIM(b), "dimensions should match");

	ASSERT_FLOAT_EQ(1.0f, MKT_VECTOR_DATA(b)[0], 1e-6f, "element 0");
	ASSERT_FLOAT_EQ(2.0f, MKT_VECTOR_DATA(b)[1], 1e-6f, "element 1");
	ASSERT_FLOAT_EQ(3.0f, MKT_VECTOR_DATA(b)[2], 1e-6f, "element 2");

	/* Verify independence */
	MKT_VECTOR_DATA(a)[0] = 99.0f;
	ASSERT_FLOAT_EQ(
			1.0f, MKT_VECTOR_DATA(b)[0], 1e-6f, "copy should be independent");

	mkt_vector_free(a);
	mkt_vector_free(b);
}

TEST(vector_dot_product)
{
	MktVector *a = mkt_vector_create(3);
	MktVector *b = mkt_vector_create(3);
	ASSERT_NOT_NULL(a, "vector a should be allocated");
	ASSERT_NOT_NULL(b, "vector b should be allocated");

	float va[] = {1.0f, 2.0f, 3.0f};
	float vb[] = {4.0f, 5.0f, 6.0f};
	mkt_vector_set(a, va);
	mkt_vector_set(b, vb);

	/* dot(a, b) = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32 */
	float dot = mkt_vector_dot(a, b);
	ASSERT_FLOAT_EQ(32.0f, dot, 1e-6f, "dot product should be 32");

	mkt_vector_free(a);
	mkt_vector_free(b);
}

TEST(vector_norm)
{
	MktVector *v = mkt_vector_create(3);
	ASSERT_NOT_NULL(v, "vector should be allocated");

	float values[] = {3.0f, 4.0f, 0.0f};
	mkt_vector_set(v, values);

	/* norm = sqrt(9 + 16 + 0) = sqrt(25) = 5 */
	float norm = mkt_vector_norm(v);
	ASSERT_FLOAT_EQ(5.0f, norm, 1e-6f, "norm should be 5");

	mkt_vector_free(v);
}

TEST(vector_normalize)
{
	MktVector *v = mkt_vector_create(3);
	ASSERT_NOT_NULL(v, "vector should be allocated");

	float values[] = {3.0f, 4.0f, 0.0f};
	mkt_vector_set(v, values);
	mkt_vector_normalize(v);

	/* normalized: [3/5, 4/5, 0] = [0.6, 0.8, 0] */
	ASSERT_FLOAT_EQ(0.6f, MKT_VECTOR_DATA(v)[0], 1e-6f, "element 0");
	ASSERT_FLOAT_EQ(0.8f, MKT_VECTOR_DATA(v)[1], 1e-6f, "element 1");
	ASSERT_FLOAT_EQ(0.0f, MKT_VECTOR_DATA(v)[2], 1e-6f, "element 2");

	/* norm should be 1 */
	float norm = mkt_vector_norm(v);
	ASSERT_FLOAT_EQ(1.0f, norm, 1e-6f, "normalized vector should have norm 1");

	mkt_vector_free(v);
}

TEST(vector_normalize_zero)
{
	MktVector *v = mkt_vector_create(3);
	ASSERT_NOT_NULL(v, "vector should be allocated");

	mkt_vector_zero(v);
	mkt_vector_normalize(v); /* Should not crash */

	/* Should remain zero */
	ASSERT_FLOAT_EQ(0.0f, MKT_VECTOR_DATA(v)[0], 1e-6f, "element 0");
	ASSERT_FLOAT_EQ(0.0f, MKT_VECTOR_DATA(v)[1], 1e-6f, "element 1");
	ASSERT_FLOAT_EQ(0.0f, MKT_VECTOR_DATA(v)[2], 1e-6f, "element 2");

	mkt_vector_free(v);
}

TEST(vector_to_ref)
{
	MktVector *v = mkt_vector_create(3);
	ASSERT_NOT_NULL(v, "vector should be allocated");

	float values[] = {1.0f, 2.0f, 3.0f};
	mkt_vector_set(v, values);

	VectorRef ref = MktVectorToRef(v);
	ASSERT_EQ(3, ref.dim, "ref dimension should be 3");
	ASSERT_TRUE(
			ref.data == MKT_VECTOR_DATA(v), "ref data should point to vector");

	mkt_vector_free(v);
}

TEST(vector_copy_null)
{
	MktVector *v = mkt_vector_copy(NULL);
	ASSERT_NULL(v, "copying NULL should return NULL");
}
