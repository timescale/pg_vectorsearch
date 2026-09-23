/*
 * test_vec32.c - Vector type and operations tests
 */

#include "core/memory.h"
#include "types/vec32.h"
#include "vs_test.h"

TEST_GROUP(Vector);

/* Create and switch to a memory context for each test */
TEST_MEMCTX_FIXTURE();

TEST(vector_create_and_dim)
{
	Vec32 *v = vec32_create(128);
	ASSERT_NOT_NULL(v, "vector should be allocated");
	ASSERT_EQ(128, VEC32_DIM(v), "dimension should be 128");
	vec32_free(v);
}

TEST(vector_create_zero_dim_fails)
{
	Vec32 *v = vec32_create(0);
	ASSERT_NULL(v, "zero dimension should fail");
}

TEST(vector_create_max_dim)
{
	Vec32 *v = vec32_create(VEC32_MAX_DIM);
	ASSERT_NOT_NULL(v, "max dimension should succeed");
	ASSERT_EQ(VEC32_MAX_DIM, VEC32_DIM(v), "dimension should be max");
	vec32_free(v);
}

TEST(vector_create_over_max_dim_fails)
{
	Vec32 *v = vec32_create(VEC32_MAX_DIM + 1);
	ASSERT_NULL(v, "over max dimension should fail");
}

TEST(vector_set_get_elements)
{
	Vec32 *v = vec32_create(3);
	ASSERT_NOT_NULL(v, "vector should be allocated");

	float values[] = {1.0f, 2.0f, 3.0f};
	vec32_set(v, values);

	ASSERT_FLOAT_EQ(1.0f, VEC32_DATA(v)[0], 1e-6f, "element 0");
	ASSERT_FLOAT_EQ(2.0f, VEC32_DATA(v)[1], 1e-6f, "element 1");
	ASSERT_FLOAT_EQ(3.0f, VEC32_DATA(v)[2], 1e-6f, "element 2");

	vec32_free(v);
}

TEST(vector_zero)
{
	Vec32 *v = vec32_create(3);
	ASSERT_NOT_NULL(v, "vector should be allocated");

	float values[] = {1.0f, 2.0f, 3.0f};
	vec32_set(v, values);
	vec32_zero(v);

	ASSERT_FLOAT_EQ(0.0f, VEC32_DATA(v)[0], 1e-6f, "element 0 should be 0");
	ASSERT_FLOAT_EQ(0.0f, VEC32_DATA(v)[1], 1e-6f, "element 1 should be 0");
	ASSERT_FLOAT_EQ(0.0f, VEC32_DATA(v)[2], 1e-6f, "element 2 should be 0");

	vec32_free(v);
}

TEST(vector_fill)
{
	Vec32 *v = vec32_create(3);
	ASSERT_NOT_NULL(v, "vector should be allocated");

	vec32_fill(v, 5.0f);

	ASSERT_FLOAT_EQ(5.0f, VEC32_DATA(v)[0], 1e-6f, "element 0");
	ASSERT_FLOAT_EQ(5.0f, VEC32_DATA(v)[1], 1e-6f, "element 1");
	ASSERT_FLOAT_EQ(5.0f, VEC32_DATA(v)[2], 1e-6f, "element 2");

	vec32_free(v);
}

TEST(vector_copy)
{
	Vec32 *a = vec32_create(3);
	ASSERT_NOT_NULL(a, "vector should be allocated");

	float values[] = {1.0f, 2.0f, 3.0f};
	vec32_set(a, values);

	Vec32 *b = vec32_copy(a);
	ASSERT_NOT_NULL(b, "copy should be allocated");
	ASSERT_EQ(VEC32_DIM(a), VEC32_DIM(b), "dimensions should match");

	ASSERT_FLOAT_EQ(1.0f, VEC32_DATA(b)[0], 1e-6f, "element 0");
	ASSERT_FLOAT_EQ(2.0f, VEC32_DATA(b)[1], 1e-6f, "element 1");
	ASSERT_FLOAT_EQ(3.0f, VEC32_DATA(b)[2], 1e-6f, "element 2");

	/* Verify independence */
	VEC32_DATA(a)[0] = 99.0f;
	ASSERT_FLOAT_EQ(
			1.0f, VEC32_DATA(b)[0], 1e-6f, "copy should be independent");

	vec32_free(a);
	vec32_free(b);
}

TEST(vector_dot_product)
{
	Vec32 *a = vec32_create(3);
	Vec32 *b = vec32_create(3);
	ASSERT_NOT_NULL(a, "vector a should be allocated");
	ASSERT_NOT_NULL(b, "vector b should be allocated");

	float va[] = {1.0f, 2.0f, 3.0f};
	float vb[] = {4.0f, 5.0f, 6.0f};
	vec32_set(a, va);
	vec32_set(b, vb);

	/* dot(a, b) = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32 */
	float dot = vec32_dot(a, b);
	ASSERT_FLOAT_EQ(32.0f, dot, 1e-6f, "dot product should be 32");

	vec32_free(a);
	vec32_free(b);
}

TEST(vector_norm)
{
	Vec32 *v = vec32_create(3);
	ASSERT_NOT_NULL(v, "vector should be allocated");

	float values[] = {3.0f, 4.0f, 0.0f};
	vec32_set(v, values);

	/* norm = sqrt(9 + 16 + 0) = sqrt(25) = 5 */
	float norm = vec32_norm(v);
	ASSERT_FLOAT_EQ(5.0f, norm, 1e-6f, "norm should be 5");

	vec32_free(v);
}

TEST(vector_normalize)
{
	Vec32 *v = vec32_create(3);
	ASSERT_NOT_NULL(v, "vector should be allocated");

	float values[] = {3.0f, 4.0f, 0.0f};
	vec32_set(v, values);
	vec32_normalize(v);

	/* normalized: [3/5, 4/5, 0] = [0.6, 0.8, 0] */
	ASSERT_FLOAT_EQ(0.6f, VEC32_DATA(v)[0], 1e-6f, "element 0");
	ASSERT_FLOAT_EQ(0.8f, VEC32_DATA(v)[1], 1e-6f, "element 1");
	ASSERT_FLOAT_EQ(0.0f, VEC32_DATA(v)[2], 1e-6f, "element 2");

	/* norm should be 1 */
	float norm = vec32_norm(v);
	ASSERT_FLOAT_EQ(1.0f, norm, 1e-6f, "normalized vector should have norm 1");

	vec32_free(v);
}

TEST(vector_normalize_zero)
{
	Vec32 *v = vec32_create(3);
	ASSERT_NOT_NULL(v, "vector should be allocated");

	vec32_zero(v);
	vec32_normalize(v); /* Should not crash */

	/* Should remain zero */
	ASSERT_FLOAT_EQ(0.0f, VEC32_DATA(v)[0], 1e-6f, "element 0");
	ASSERT_FLOAT_EQ(0.0f, VEC32_DATA(v)[1], 1e-6f, "element 1");
	ASSERT_FLOAT_EQ(0.0f, VEC32_DATA(v)[2], 1e-6f, "element 2");

	vec32_free(v);
}

TEST(vector_to_ref)
{
	Vec32 *v = vec32_create(3);
	ASSERT_NOT_NULL(v, "vector should be allocated");

	float values[] = {1.0f, 2.0f, 3.0f};
	vec32_set(v, values);

	Vec32Ref ref = Vec32ToRef(v);
	ASSERT_EQ(3, ref.dim, "ref dimension should be 3");
	ASSERT_TRUE(ref.data == VEC32_DATA(v), "ref data should point to vector");

	vec32_free(v);
}

TEST(vector_copy_null)
{
	Vec32 *v = vec32_copy(NULL);
	ASSERT_NULL(v, "copying NULL should return NULL");
}
