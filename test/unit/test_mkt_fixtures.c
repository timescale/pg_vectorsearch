/*
 * test_mkt_fixtures.c - Demonstration of per-group and per-test fixtures
 */

#include "mkt_test.h"

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

	/* Test setup called again (second time) */
	ASSERT_EQ(2, test_setup_count, "test setup called twice");
	ASSERT_EQ(
			1,
			test_teardown_count,
			"test teardown called once (after first test)");
}

/* Test with its own specific fixture in addition to group fixtures */
TEST_WITH_FIXTURE(special_test, special_setup, special_teardown)
{
	/* Group fixtures still run */
	ASSERT_EQ(1, group_setup_count, "group setup called once");
	ASSERT_EQ(3, test_setup_count, "test setup called three times");

	/* Test-specific fixture also ran */
	ASSERT_EQ(1, special_setup_count, "special setup called");
	ASSERT_EQ(0, special_teardown_count, "special teardown not yet called");
}

TEST(third_test)
{
	/* Group setup still only called once */
	ASSERT_EQ(1, group_setup_count, "group setup called once");
	ASSERT_EQ(0, group_teardown_count, "group teardown not yet called");

	/* Test setup called for fourth time (after special_test) */
	ASSERT_EQ(4, test_setup_count, "test setup called four times");
	ASSERT_EQ(
			3,
			test_teardown_count,
			"test teardown called three times (after first, second, special)");
}
