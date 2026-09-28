/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * test_vs_fixtures.c - Demonstration of per-group and per-test fixtures
 */

#include "vs_test.h"

TEST_GROUP(Fixtures);

/* Track setup/teardown calls to verify execution order */
static int group_setup_count	  = 0;
static int group_teardown_count	  = 0;
static int test_setup_count		  = 0;
static int test_teardown_count	  = 0;
static int special_setup_count	  = 0;
static int special_teardown_count = 0;

static void
group_setup(void)
{
	group_setup_count++;
}

static void
group_teardown(void)
{
	group_teardown_count++;
}

static void
test_setup(void)
{
	test_setup_count++;
}

static void
test_teardown(void)
{
	test_teardown_count++;
}

static void
special_setup(void)
{
	special_setup_count++;
}

static void
special_teardown(void)
{
	special_teardown_count++;
}

/* Register per-group fixtures (run once for all tests) */
GROUP_FIXTURE(group_setup, group_teardown);

/* Register per-test fixtures (run before/after each test) */
TEST_FIXTURE(test_setup, test_teardown);

TEST(first_test)
{
	/* Group setup should have run once */
	ASSERT_EQ(1, group_setup_count, "group setup called once");
	ASSERT_EQ(0, group_teardown_count, "group teardown not yet called");

	/* Test setup should have run once */
	ASSERT_EQ(1, test_setup_count, "test setup called once");
	ASSERT_EQ(0, test_teardown_count, "test teardown not yet called");
}

TEST(second_test)
{
	/* Group setup still only called once */
	ASSERT_EQ(1, group_setup_count, "group setup called once");
	ASSERT_EQ(0, group_teardown_count, "group teardown not yet called");

	/* Test setup called again (5th time - after first + 3 param tests) */
	ASSERT_EQ(5, test_setup_count, "test setup called 5 times");
	ASSERT_EQ(
			4,
			test_teardown_count,
			"test teardown called 4 times (after first + 3 param tests)");
}

/* Test with its own specific fixture in addition to group fixtures */
TEST_WITH_FIXTURE(special_test, special_setup, special_teardown)
{
	/* Group fixtures still run */
	ASSERT_EQ(1, group_setup_count, "group setup called once");
	ASSERT_EQ(6, test_setup_count, "test setup called 6 times");

	/* Test-specific fixture also ran */
	ASSERT_EQ(1, special_setup_count, "special setup called");
	ASSERT_EQ(0, special_teardown_count, "special teardown not yet called");
}

TEST(third_test)
{
	/* Group setup still only called once */
	ASSERT_EQ(1, group_setup_count, "group setup called once");
	ASSERT_EQ(0, group_teardown_count, "group teardown not yet called");

	/* Test setup called for 7th time (after first + 3 param + second +
	 * special) */
	ASSERT_EQ(7, test_setup_count, "test setup called 7 times");
	ASSERT_EQ(6, test_teardown_count, "test teardown called 6 times");
}

/*
 * Parameterized test example - demonstrates TEST_PARAMETERIZED macro
 *
 * Each parameter string registers as a separate test, and fixtures run
 * before/after each iteration. Note that parameterized tests affect the
 * fixture call counts in other tests due to alphabetical ordering.
 *
 * These run alphabetically as: param_example_bar, param_example_baz,
 * param_example_foo (NOT in iteration order 0,1,2).
 */
TEST_PARAMETERIZED(param_example, "foo", "bar", "baz")
{
	/* Both 'iteration' (0,1,2) and 'param' ("foo","bar","baz") available */
	ASSERT_TRUE(param != NULL, "param should not be NULL");
	ASSERT_TRUE(iteration >= 0 && iteration < 3, "iteration in range");

	/* Verify fixtures ran (but don't check exact count since tests run
	   alphabetically, not in iteration order) */
	ASSERT_TRUE(
			test_setup_count >= 2 && test_setup_count <= 4,
			"test setup called for parameterized test");

	TEST_PRINT(
			"Parameterized test with param: %s (iteration=%d)\n",
			param,
			iteration);
}
