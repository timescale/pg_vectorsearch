/*
 * mkt_test.h - Simple C test framework with automatic test registration
 *
 * Usage:
 *   TEST_GROUP(MyTests)
 *
 *   TEST(my_test_name) {
 *       ASSERT_EQ(5, 2 + 3, "addition works");
 *       ASSERT_TRUE(ptr != NULL, "pointer is not null");
 *   }
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

/* Test registry entry */
typedef struct
{
	const char *name;
	const char *group;
	MktTestFunc func;
} MktTestEntry;

/* Register a test (called automatically by TEST macro) */
void mkt_test_register(const char *name, const char *group, MktTestFunc func);

/* Run all registered tests, returns 0 on success, 1 on failure */
int mkt_test_run_all(void);

/* Internal: mark current test as failed */
void mkt_test_fail(const char *file, int line, const char *msg);

/* TEST_GROUP macro - sets the group for subsequent tests in this file */
#define TEST_GROUP(group_name) static const char *_MKT_TEST_GROUP = #group_name

/* TEST macro - defines and auto-registers a test */
#define TEST(name)                                                 \
	static void test_##name(MktTestResult *result);                \
	__attribute__((constructor)) static void register_##name(void) \
	{                                                              \
		mkt_test_register(#name, _MKT_TEST_GROUP, test_##name);    \
	}                                                              \
	static void test_##name(MktTestResult *result)

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
