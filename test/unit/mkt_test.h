/*
 * mkt_test.h - Simple C test framework with automatic test registration
 *
 * Basic Usage:
 *   TEST_GROUP(MyTests)
 *
 *   TEST(my_test_name) {
 *       ASSERT_EQ(5, 2 + 3, "addition works");
 *       ASSERT_TRUE(ptr != NULL, "pointer is not null");
 *   }
 *
 * Fixture Usage:
 *   TEST_GROUP(DatabaseTests)
 *
 *   // Per-group fixtures (run once for the entire group)
 *   static void group_setup(void) {
 *       // Initialize database connection pool
 *   }
 *   static void group_teardown(void) {
 *       // Close database connection pool
 *   }
 *   GROUP_FIXTURE(group_setup, group_teardown);
 *
 *   // Per-test fixtures (run before/after each test)
 *   static void test_setup(void) {
 *       // Begin transaction
 *   }
 *   static void test_teardown(void) {
 *       // Rollback transaction
 *   }
 *   TEST_FIXTURE(test_setup, test_teardown);
 *
 *   TEST(insert_record) { ... }
 *   TEST(delete_record) { ... }
 *   // Execution order:
 *   //   group_setup()
 *   //   test_setup() -> insert_record() -> test_teardown()
 *   //   test_setup() -> delete_record() -> test_teardown()
 *   //   group_teardown()
 */

#ifndef MKT_TEST_H
#define MKT_TEST_H

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ANSI color codes */
#define MKT_COLOR_RESET	 "\033[0m"
#define MKT_COLOR_RED	 "\033[31m"
#define MKT_COLOR_GREEN	 "\033[32m"
#define MKT_COLOR_YELLOW "\033[33m"
#define MKT_COLOR_CYAN	 "\033[36m"

/* Test result structure */
typedef struct
{
	const char *test_name;
	bool		passed;
	char		failure_msg[512];
	const char *file;
	int			line;
} MktTestResult;

/* Test function signature */
typedef void (*MktTestFunc)(MktTestResult *result);

/* Fixture function signature */
typedef void (*MktFixtureFunc)(void);

/* Test registry entry */
typedef struct
{
	const char *name;
	const char *group;
	MktTestFunc func;
} MktTestEntry;

/* Register a test (called automatically by TEST macro) */
void mkt_test_register(const char *name, const char *group, MktTestFunc func);

/* Register group fixtures (called automatically by GROUP_FIXTURE macro) */
void mkt_test_register_group_fixture(
		const char *group, MktFixtureFunc setup, MktFixtureFunc teardown);

/* Run all registered tests, returns 0 on success, 1 on failure */
int mkt_test_run_all(
		bool		 tap_output,
		const char **test_filters,
		int			 test_filter_count,
		const char **group_filters,
		int			 group_filter_count);

/* Internal: mark current test as failed */
void mkt_test_fail(const char *file, int line, const char *msg);

/* Internal: check if running in TAP mode */
bool mkt_test_is_tap_mode(void);

/* Internal: printf wrapper for tests */
void mkt_test_printf(const char *fmt, ...)
		__attribute__((format(printf, 1, 2)));

/* TEST_GROUP macro - sets the group for subsequent tests in this file */
#define TEST_GROUP(group_name)                             \
	static const char *_MKT_TEST_GROUP		= #group_name; \
	static void (*_mkt_test_setup)(void)	= NULL;        \
	static void (*_mkt_test_teardown)(void) = NULL

/*
 * GROUP_FIXTURE - register per-group setup/teardown functions
 *
 * These run once when entering/leaving the test group (before first test,
 * after last test).
 *
 * Usage:
 *   TEST_GROUP(MyTests);
 *   GROUP_FIXTURE(group_setup, group_teardown);
 *
 *   TEST(test1) { ... }
 *   TEST(test2) { ... }
 *   // group_setup() runs before test1
 *   // group_teardown() runs after test2
 */
#define GROUP_FIXTURE(setup_fn, teardown_fn)                              \
	__attribute__((constructor)) static void _mkt_register_group_fixture( \
			void)                                                         \
	{                                                                     \
		mkt_test_register_group_fixture(                                  \
				_MKT_TEST_GROUP, setup_fn, teardown_fn);                  \
	}

/*
 * TEST_FIXTURE - register per-test setup/teardown functions
 *
 * These run before/after each individual test in the group.
 *
 * Usage:
 *   TEST_GROUP(MyTests);
 *   TEST_FIXTURE(my_setup, my_teardown);
 *
 *   TEST(my_test) {
 *       // my_setup() called before, my_teardown() called after
 *   }
 */
#define TEST_FIXTURE(setup_fn, teardown_fn)                              \
	__attribute__((constructor)) static void _mkt_register_fixture(void) \
	{                                                                    \
		_mkt_test_setup	   = setup_fn;                                   \
		_mkt_test_teardown = teardown_fn;                                \
	}

/* Deprecated alias for TEST_FIXTURE (for backward compatibility) */
#define TEST_GROUP_FIXTURE(setup_fn, teardown_fn) \
	TEST_FIXTURE(setup_fn, teardown_fn)

/*
 * TEST_MEMCTX_FIXTURE - convenience macro for memory context per-test fixture
 *
 * Creates a memory context before each test and destroys it after. This is
 * a common pattern for tests that need isolated memory allocation.
 *
 * Usage:
 *   TEST_GROUP(MyTests);
 *   TEST_MEMCTX_FIXTURE();
 *
 *   TEST(my_test) {
 *       // Memory context active, allocations are isolated
 *   }
 */
#define TEST_MEMCTX_FIXTURE()                                        \
	static MktMemCtx _mkt_test_memctx = NULL;                        \
	static void		 _mkt_memctx_setup(void)                         \
	{                                                                \
		_mkt_test_memctx = mkt_memctx_create(NULL, _MKT_TEST_GROUP); \
		mkt_memctx_switch(_mkt_test_memctx);                         \
	}                                                                \
	static void _mkt_memctx_teardown(void)                           \
	{                                                                \
		mkt_memctx_switch(NULL);                                     \
		mkt_memctx_delete(_mkt_test_memctx);                         \
		_mkt_test_memctx = NULL;                                     \
	}                                                                \
	TEST_FIXTURE(_mkt_memctx_setup, _mkt_memctx_teardown)

/*
 * TEST_PRINT - printf for tests that respects TAP format
 *
 * Use this instead of printf() in tests to output diagnostic information.
 * In TAP mode, output is prefixed with '# ' to mark it as a comment.
 * In human-readable mode, output is indented to align with test output.
 *
 * Usage:
 *   TEST_PRINT("Detected value: %d\n", value);
 */
#define TEST_PRINT(...) mkt_test_printf(__VA_ARGS__)

/* TEST macro - defines and auto-registers a test */
#define TEST(name)                                                 \
	static void test_##name##_impl(MktTestResult *result);         \
	static void test_##name(MktTestResult *result)                 \
	{                                                              \
		if (_mkt_test_setup)                                       \
			_mkt_test_setup();                                     \
		test_##name##_impl(result);                                \
		if (_mkt_test_teardown)                                    \
			_mkt_test_teardown();                                  \
	}                                                              \
	__attribute__((constructor)) static void register_##name(void) \
	{                                                              \
		mkt_test_register(#name, _MKT_TEST_GROUP, test_##name);    \
	}                                                              \
	static void test_##name##_impl(MktTestResult *result)

/*
 * TEST_WITH_FIXTURE - define a test with its own specific setup/teardown
 *
 * The test's own fixture runs in addition to the group's TEST_FIXTURE
 * (if one is registered).
 *
 * Execution order:
 *   1. Group TEST_FIXTURE setup (if exists)
 *   2. Test-specific setup
 *   3. Test body
 *   4. Test-specific teardown
 *   5. Group TEST_FIXTURE teardown (if exists)
 *
 * Usage:
 *   TEST_WITH_FIXTURE(my_special_test, my_setup, my_teardown) {
 *       // Test body with custom setup/teardown
 *   }
 */
#define TEST_WITH_FIXTURE(name, setup_fn, teardown_fn)             \
	static void test_##name##_impl(MktTestResult *result);         \
	static void test_##name(MktTestResult *result)                 \
	{                                                              \
		if (_mkt_test_setup)                                       \
			_mkt_test_setup();                                     \
		setup_fn();                                                \
		test_##name##_impl(result);                                \
		teardown_fn();                                             \
		if (_mkt_test_teardown)                                    \
			_mkt_test_teardown();                                  \
	}                                                              \
	__attribute__((constructor)) static void register_##name(void) \
	{                                                              \
		mkt_test_register(#name, _MKT_TEST_GROUP, test_##name);    \
	}                                                              \
	static void test_##name##_impl(MktTestResult *result)

/*
 * TEST_PARAMETERIZED - run a test multiple times with different parameters
 *
 * Each iteration is registered as a separate test with a descriptive name
 * (e.g., "test_AVX2", "test_NEON") for better test reporting and filtering.
 *
 * Test receives two arguments (both marked as __attribute__((unused))):
 *   - int iteration: index from 0 to count-1
 *   - const char *param: the parameter string for this iteration
 *
 * Both parameters are marked unused, so you only need to use what you need.
 * Fixtures run before/after each iteration. Supports 2-10 parameters.
 *
 * Usage:
 *   TEST_PARAMETERIZED(l2_correctness, "scalar", "AVX2", "AVX512") {
 *       const uint32_t masks[] = {SIMD_NONE, SIMD_AVX2, SIMD_AVX512F};
 *       reinit_distance_with_simd(masks[iteration]);
 *       TEST_PRINT("Testing with: %s\n", param);
 *       // ...
 *   }
 *
 * This registers 3 separate tests: l2_correctness_scalar,
 * l2_correctness_AVX2, l2_correctness_AVX512.
 */

/* Count variadic arguments (supports 1-10 args) */
#define _COUNT_ARGS(...) \
	_COUNT_ARGS_IMPL(__VA_ARGS__, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
#define _COUNT_ARGS_IMPL(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, N, ...) N

/* Concatenation helper for macro expansion */
#define _CONCAT(a, b)	   _CONCAT_IMPL(a, b)
#define _CONCAT_IMPL(a, b) a##b

/* Helper macros to generate test wrappers for each iteration */
#define _TEST_PARAM_WRAPPER(name, iter)                                       \
	static void test_##name##_##iter(MktTestResult *result)                   \
	{                                                                         \
		if (_mkt_test_setup)                                                  \
			_mkt_test_setup();                                                \
		test_##name##_impl(result, iter, _##name##_param_names[iter]);        \
		if (_mkt_test_teardown)                                               \
			_mkt_test_teardown();                                             \
	}                                                                         \
	__attribute__((constructor)) static void register_##name##_##iter(void)   \
	{                                                                         \
		static char test_name[256]; /* STATIC - persists after constructor */ \
		snprintf(                                                             \
				test_name,                                                    \
				sizeof(test_name),                                            \
				"%s_%s",                                                      \
				#name,                                                        \
				_##name##_param_names[iter]);                                 \
		mkt_test_register(test_name, _MKT_TEST_GROUP, test_##name##_##iter);  \
	}

/* Generate wrappers for supported iteration counts (2-10) */
#define _TEST_PARAM_WRAPPERS_2(name) \
	_TEST_PARAM_WRAPPER(name, 0)     \
	_TEST_PARAM_WRAPPER(name, 1)

#define _TEST_PARAM_WRAPPERS_3(name) \
	_TEST_PARAM_WRAPPERS_2(name)     \
	_TEST_PARAM_WRAPPER(name, 2)

#define _TEST_PARAM_WRAPPERS_4(name) \
	_TEST_PARAM_WRAPPERS_3(name)     \
	_TEST_PARAM_WRAPPER(name, 3)

#define _TEST_PARAM_WRAPPERS_5(name) \
	_TEST_PARAM_WRAPPERS_4(name)     \
	_TEST_PARAM_WRAPPER(name, 4)

#define _TEST_PARAM_WRAPPERS_6(name) \
	_TEST_PARAM_WRAPPERS_5(name)     \
	_TEST_PARAM_WRAPPER(name, 5)

#define _TEST_PARAM_WRAPPERS_7(name) \
	_TEST_PARAM_WRAPPERS_6(name)     \
	_TEST_PARAM_WRAPPER(name, 6)

#define _TEST_PARAM_WRAPPERS_8(name) \
	_TEST_PARAM_WRAPPERS_7(name)     \
	_TEST_PARAM_WRAPPER(name, 7)

#define _TEST_PARAM_WRAPPERS_9(name) \
	_TEST_PARAM_WRAPPERS_8(name)     \
	_TEST_PARAM_WRAPPER(name, 8)

#define _TEST_PARAM_WRAPPERS_10(name) \
	_TEST_PARAM_WRAPPERS_9(name)      \
	_TEST_PARAM_WRAPPER(name, 9)

/* Dispatcher macro to select the right wrapper generator based on count */
#define _TEST_PARAM_DISPATCH(name, count) \
	_CONCAT(_TEST_PARAM_WRAPPERS_, count)(name)

/* Main TEST_PARAMETERIZED macro */
#define TEST_PARAMETERIZED(name, ...)                                        \
	static const char *_##name##_param_names[] = {__VA_ARGS__};              \
	static void		   test_##name##_impl(                                   \
			   MktTestResult *result, int iteration, const char *param); \
	_TEST_PARAM_DISPATCH(name, _COUNT_ARGS(__VA_ARGS__))                     \
	static void test_##name##_impl(                                          \
			MktTestResult			   *result,                              \
			int __attribute__((unused)) iteration,                           \
			const char __attribute__((unused)) * param)

/* Assertion macros */
#define ASSERT_TRUE(cond, msg)                                            \
	do                                                                    \
	{                                                                     \
		(void)result;                                                     \
		if (!(cond))                                                      \
		{                                                                 \
			char buf[256];                                                \
			snprintf(buf, sizeof(buf), "%s (condition: %s)", msg, #cond); \
			mkt_test_fail(__FILE__, __LINE__, buf);                       \
			return;                                                       \
		}                                                                 \
	} while (0)

#define ASSERT_FALSE(cond, msg)                                           \
	do                                                                    \
	{                                                                     \
		(void)result;                                                     \
		if (cond)                                                         \
		{                                                                 \
			char buf[256];                                                \
			snprintf(buf, sizeof(buf), "%s (condition: %s)", msg, #cond); \
			mkt_test_fail(__FILE__, __LINE__, buf);                       \
			return;                                                       \
		}                                                                 \
	} while (0)

#define ASSERT_EQ(expected, actual, msg)                 \
	do                                                   \
	{                                                    \
		(void)result;                                    \
		if ((expected) != (actual))                      \
		{                                                \
			char buf[256];                               \
			snprintf(                                    \
					buf,                                 \
					sizeof(buf),                         \
					"%s (expected: %lld, actual: %lld)", \
					msg,                                 \
					(long long)(expected),               \
					(long long)(actual));                \
			mkt_test_fail(__FILE__, __LINE__, buf);      \
			return;                                      \
		}                                                \
	} while (0)

#define ASSERT_NEQ(val1, val2, msg)                 \
	do                                              \
	{                                               \
		(void)result;                               \
		if ((val1) == (val2))                       \
		{                                           \
			char buf[256];                          \
			snprintf(                               \
					buf,                            \
					sizeof(buf),                    \
					"%s (both values: %lld)",       \
					msg,                            \
					(long long)(val1));             \
			mkt_test_fail(__FILE__, __LINE__, buf); \
			return;                                 \
		}                                           \
	} while (0)

#define ASSERT_FLOAT_EQ(expected, actual, epsilon, msg)        \
	do                                                         \
	{                                                          \
		(void)result;                                          \
		double _diff = (double)(expected) - (double)(actual);  \
		if (_diff < 0)                                         \
			_diff = -_diff;                                    \
		if (_diff > (epsilon))                                 \
		{                                                      \
			char buf[256];                                     \
			snprintf(                                          \
					buf,                                       \
					sizeof(buf),                               \
					"%s (expected: %f, actual: %f, diff: %f)", \
					msg,                                       \
					(double)(expected),                        \
					(double)(actual),                          \
					_diff);                                    \
			mkt_test_fail(__FILE__, __LINE__, buf);            \
			return;                                            \
		}                                                      \
	} while (0)

#define ASSERT_NULL(ptr, msg)                       \
	do                                              \
	{                                               \
		(void)result;                               \
		if ((ptr) != NULL)                          \
		{                                           \
			char buf[256];                          \
			snprintf(                               \
					buf,                            \
					sizeof(buf),                    \
					"%s (pointer: %p)",             \
					msg,                            \
					(void *)(ptr));                 \
			mkt_test_fail(__FILE__, __LINE__, buf); \
			return;                                 \
		}                                           \
	} while (0)

#define ASSERT_NOT_NULL(ptr, msg)                   \
	do                                              \
	{                                               \
		(void)result;                               \
		if ((ptr) == NULL)                          \
		{                                           \
			mkt_test_fail(__FILE__, __LINE__, msg); \
			return;                                 \
		}                                           \
	} while (0)

#define ASSERT_STR_EQ(expected, actual, msg)                 \
	do                                                       \
	{                                                        \
		(void)result;                                        \
		if (strcmp((expected), (actual)) != 0)               \
		{                                                    \
			char buf[512];                                   \
			snprintf(                                        \
					buf,                                     \
					sizeof(buf),                             \
					"%s (expected: \"%s\", actual: \"%s\")", \
					msg,                                     \
					(expected),                              \
					(actual));                               \
			mkt_test_fail(__FILE__, __LINE__, buf);          \
			return;                                          \
		}                                                    \
	} while (0)

#define ASSERT_MEM_EQ(expected, actual, len, msg)     \
	do                                                \
	{                                                 \
		(void)result;                                 \
		if (memcmp((expected), (actual), (len)) != 0) \
		{                                             \
			mkt_test_fail(__FILE__, __LINE__, msg);   \
			return;                                   \
		}                                             \
	} while (0)

#endif /* MKT_TEST_H */
