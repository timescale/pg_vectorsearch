/*
 * test_vecops.c - Tests for general vector operations
 *
 * Tests cover:
 * - Basic arithmetic: add, sub, scale, dot product
 * - Norms: L2 norm, L2 norm squared
 * - Distance: L2 distance squared
 * - Sum
 * - SIMD dispatch: init, reinit, impl_name
 */

#include <math.h>
#include <string.h>

#include "algo/vecops.h"
#include "core/memory.h"
#include "vs_test.h"

TEST_GROUP(VecOps);
TEST_MEMCTX_FIXTURE();

/*
 * Dispatch / Init Tests
 */

TEST(impl_name_is_valid)
{
	const char *name = vs_vecops_impl_name();
	ASSERT_NOT_NULL(name, "implementation name should not be null");

	int valid =
			(strcmp(name, "avx512") == 0 || strcmp(name, "avx2") == 0 ||
			 strcmp(name, "neon") == 0 || strcmp(name, "compiler") == 0);

	char msg[128];
	snprintf(msg, sizeof(msg), "unknown implementation: %s", name);
	ASSERT_TRUE(valid, msg);
}

TEST(init_idempotent)
{
	/* Calling init multiple times should be harmless */
	int ret1 = vs_vecops_init();
	int ret2 = vs_vecops_init();
	ASSERT_EQ(0, ret1, "first init should succeed");
	ASSERT_EQ(0, ret2, "second init should succeed");
}

TEST(force_reinit)
{
	/* Reinit should reset state, next call auto-inits */
	vs_vecops_force_reinit();

	/* After reinit, impl_name triggers auto-init */
	const char *name = vs_vecops_impl_name();
	ASSERT_NOT_NULL(name, "impl_name should work after reinit");

	/* Reinit again, then call a vecops function */
	vs_vecops_force_reinit();

	float a[] = {1.0f, 2.0f, 3.0f};
	float b[] = {4.0f, 5.0f, 6.0f};
	float dot = vs_dot_product(a, b, 3);
	ASSERT_FLOAT_EQ(32.0f, dot, 1e-5f, "dot product after reinit");
}

/*
 * Dot Product Tests
 */

TEST(dot_product_basic)
{
	float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
	float b[] = {5.0f, 6.0f, 7.0f, 8.0f};

	/* 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70 */
	float dot = vs_dot_product(a, b, 4);
	ASSERT_FLOAT_EQ(70.0f, dot, 1e-5f, "dot product should be 70");
}

TEST(dot_product_orthogonal)
{
	float a[] = {1.0f, 0.0f};
	float b[] = {0.0f, 1.0f};

	float dot = vs_dot_product(a, b, 2);
	ASSERT_FLOAT_EQ(0.0f, dot, 1e-5f, "orthogonal dot product");
}

/*
 * L2 Norm Tests
 */

TEST(l2_norm_squared_basic)
{
	float v[] = {3.0f, 4.0f};
	/* 3^2 + 4^2 = 25 */
	float nsq = vs_l2_norm_squared(v, 2);
	ASSERT_FLOAT_EQ(25.0f, nsq, 1e-5f, "norm squared of [3,4]");
}

TEST(l2_norm_basic)
{
	float v[] = {3.0f, 4.0f};
	/* sqrt(9 + 16) = 5 */
	float norm = vs_l2_norm(v, 2);
	ASSERT_FLOAT_EQ(5.0f, norm, 1e-5f, "norm of [3,4]");
}

TEST(l2_norm_unit_vector)
{
	float v[]  = {1.0f, 0.0f, 0.0f};
	float norm = vs_l2_norm(v, 3);
	ASSERT_FLOAT_EQ(1.0f, norm, 1e-5f, "norm of unit vector");
}

TEST(l2_norm_zero_vector)
{
	float v[]  = {0.0f, 0.0f, 0.0f};
	float norm = vs_l2_norm(v, 3);
	ASSERT_FLOAT_EQ(0.0f, norm, 1e-5f, "norm of zero vector");
}

/*
 * Vector Sum Tests
 */

TEST(vector_sum_basic)
{
	float v[] = {1.0f, 2.0f, 3.0f, 4.0f};
	float sum = vec32_sum(v, 4);
	ASSERT_FLOAT_EQ(10.0f, sum, 1e-5f, "sum of [1,2,3,4]");
}

TEST(vector_sum_negative)
{
	float v[] = {-1.0f, 2.0f, -3.0f, 4.0f};
	float sum = vec32_sum(v, 4);
	ASSERT_FLOAT_EQ(2.0f, sum, 1e-5f, "sum with negatives");
}

/*
 * Vector Sub Tests
 */

TEST(vector_sub_basic)
{
	float a[]	= {5.0f, 10.0f, 15.0f};
	float b[]	= {1.0f, 2.0f, 3.0f};
	float out[] = {0.0f, 0.0f, 0.0f};

	vec32_sub(a, b, out, 3);
	ASSERT_FLOAT_EQ(4.0f, out[0], 1e-5f, "sub[0]");
	ASSERT_FLOAT_EQ(8.0f, out[1], 1e-5f, "sub[1]");
	ASSERT_FLOAT_EQ(12.0f, out[2], 1e-5f, "sub[2]");
}

TEST(vector_sub_in_place)
{
	float a[] = {5.0f, 10.0f, 15.0f};
	float b[] = {1.0f, 2.0f, 3.0f};

	/* Output aliases input */
	vec32_sub(a, b, a, 3);
	ASSERT_FLOAT_EQ(4.0f, a[0], 1e-5f, "in-place sub[0]");
	ASSERT_FLOAT_EQ(8.0f, a[1], 1e-5f, "in-place sub[1]");
	ASSERT_FLOAT_EQ(12.0f, a[2], 1e-5f, "in-place sub[2]");
}

/*
 * Vector Add Tests
 */

TEST(vector_add_basic)
{
	float a[]	= {1.0f, 2.0f, 3.0f};
	float b[]	= {4.0f, 5.0f, 6.0f};
	float out[] = {0.0f, 0.0f, 0.0f};

	vec32_add(a, b, out, 3);
	ASSERT_FLOAT_EQ(5.0f, out[0], 1e-5f, "add[0]");
	ASSERT_FLOAT_EQ(7.0f, out[1], 1e-5f, "add[1]");
	ASSERT_FLOAT_EQ(9.0f, out[2], 1e-5f, "add[2]");
}

TEST(vector_add_in_place)
{
	float a[] = {1.0f, 2.0f, 3.0f};
	float b[] = {4.0f, 5.0f, 6.0f};

	vec32_add(a, b, a, 3);
	ASSERT_FLOAT_EQ(5.0f, a[0], 1e-5f, "in-place add[0]");
	ASSERT_FLOAT_EQ(7.0f, a[1], 1e-5f, "in-place add[1]");
	ASSERT_FLOAT_EQ(9.0f, a[2], 1e-5f, "in-place add[2]");
}

TEST(vector_add_negative)
{
	float a[]	= {1.0f, -2.0f, 3.0f};
	float b[]	= {-1.0f, 2.0f, -3.0f};
	float out[] = {0.0f, 0.0f, 0.0f};

	vec32_add(a, b, out, 3);
	ASSERT_FLOAT_EQ(0.0f, out[0], 1e-5f, "add negatives[0]");
	ASSERT_FLOAT_EQ(0.0f, out[1], 1e-5f, "add negatives[1]");
	ASSERT_FLOAT_EQ(0.0f, out[2], 1e-5f, "add negatives[2]");
}

/*
 * Vector Scale Tests
 */

TEST(vector_scale_basic)
{
	float v[]	= {1.0f, 2.0f, 3.0f};
	float out[] = {0.0f, 0.0f, 0.0f};

	vec32_scale(v, 2.0f, out, 3);
	ASSERT_FLOAT_EQ(2.0f, out[0], 1e-5f, "scale[0]");
	ASSERT_FLOAT_EQ(4.0f, out[1], 1e-5f, "scale[1]");
	ASSERT_FLOAT_EQ(6.0f, out[2], 1e-5f, "scale[2]");
}

TEST(vector_scale_zero)
{
	float v[]	= {1.0f, 2.0f, 3.0f};
	float out[] = {0.0f, 0.0f, 0.0f};

	vec32_scale(v, 0.0f, out, 3);
	ASSERT_FLOAT_EQ(0.0f, out[0], 1e-5f, "scale zero[0]");
	ASSERT_FLOAT_EQ(0.0f, out[1], 1e-5f, "scale zero[1]");
	ASSERT_FLOAT_EQ(0.0f, out[2], 1e-5f, "scale zero[2]");
}

TEST(vector_scale_negative)
{
	float v[]	= {1.0f, 2.0f, 3.0f};
	float out[] = {0.0f, 0.0f, 0.0f};

	vec32_scale(v, -1.0f, out, 3);
	ASSERT_FLOAT_EQ(-1.0f, out[0], 1e-5f, "scale neg[0]");
	ASSERT_FLOAT_EQ(-2.0f, out[1], 1e-5f, "scale neg[1]");
	ASSERT_FLOAT_EQ(-3.0f, out[2], 1e-5f, "scale neg[2]");
}

TEST(vector_scale_in_place)
{
	float v[] = {1.0f, 2.0f, 3.0f};

	vec32_scale(v, 3.0f, v, 3);
	ASSERT_FLOAT_EQ(3.0f, v[0], 1e-5f, "in-place scale[0]");
	ASSERT_FLOAT_EQ(6.0f, v[1], 1e-5f, "in-place scale[1]");
	ASSERT_FLOAT_EQ(9.0f, v[2], 1e-5f, "in-place scale[2]");
}

/*
 * L2 Distance Squared Tests
 */

TEST(l2_distance_squared_basic)
{
	float a[] = {1.0f, 0.0f};
	float b[] = {0.0f, 1.0f};

	/* (1-0)^2 + (0-1)^2 = 2 */
	float dist = vs_l2_distance_squared(a, b, 2);
	ASSERT_FLOAT_EQ(2.0f, dist, 1e-5f, "distance squared");
}

TEST(l2_distance_squared_same)
{
	float a[] = {3.0f, 4.0f, 5.0f};

	float dist = vs_l2_distance_squared(a, a, 3);
	ASSERT_FLOAT_EQ(0.0f, dist, 1e-5f, "self-distance should be 0");
}

TEST(l2_distance_squared_known)
{
	float a[] = {1.0f, 2.0f, 3.0f};
	float b[] = {4.0f, 6.0f, 8.0f};

	/* (1-4)^2 + (2-6)^2 + (3-8)^2 = 9 + 16 + 25 = 50 */
	float dist = vs_l2_distance_squared(a, b, 3);
	ASSERT_FLOAT_EQ(50.0f, dist, 1e-5f, "known distance squared");
}

/*
 * Auto-init via lazy initialization
 */

TEST(lazy_init_all_functions)
{
	/* Force reinit to clear state, then call each function
	 * to verify lazy initialization works for all of them.
	 */
	float a[]	= {1.0f, 2.0f};
	float b[]	= {3.0f, 4.0f};
	float out[] = {0.0f, 0.0f};

	vs_vecops_force_reinit();
	float nsq = vs_l2_norm_squared(a, 2);
	ASSERT_TRUE(nsq > 0, "l2_norm_squared lazy init");

	vs_vecops_force_reinit();
	float sum = vec32_sum(a, 2);
	ASSERT_FLOAT_EQ(3.0f, sum, 1e-5f, "vector_sum lazy init");

	vs_vecops_force_reinit();
	vec32_sub(a, b, out, 2);
	ASSERT_FLOAT_EQ(-2.0f, out[0], 1e-5f, "vector_sub lazy init");

	vs_vecops_force_reinit();
	vec32_add(a, b, out, 2);
	ASSERT_FLOAT_EQ(4.0f, out[0], 1e-5f, "vector_add lazy init");

	vs_vecops_force_reinit();
	vec32_scale(a, 2.0f, out, 2);
	ASSERT_FLOAT_EQ(2.0f, out[0], 1e-5f, "vector_scale lazy init");

	vs_vecops_force_reinit();
	float dist = vs_l2_distance_squared(a, b, 2);
	ASSERT_TRUE(dist > 0, "l2_distance_squared lazy init");

	/* Restore */
	vs_vecops_init();
}
