/*
 * mkt_test.c - Test framework implementation
 */

#include <stdarg.h>

#include "mkt_test.h"

/* Test registry */
static MktTestEntry *test_registry = NULL;
static int			 test_count	   = 0;
static int			 test_capacity = 0;

/* TAP mode flag (set by test runner) */
static bool running_in_tap_mode = false;

/* Group fixture registry */
typedef struct
{
	const char	  *group;
	MktFixtureFunc setup;
	MktFixtureFunc teardown;
} GroupFixtureEntry;

static GroupFixtureEntry *group_fixture_registry = NULL;
static int				  group_fixture_count	 = 0;
static int				  group_fixture_capacity = 0;

/* Current test result (set by runner, used by mkt_test_fail) */
static MktTestResult *current_result = NULL;

void
mkt_test_register(const char *name, const char *group, MktTestFunc func)
{
	/* Grow registry if needed */
	if (test_count >= test_capacity)
	{
		int new_capacity = test_capacity == 0 ? 64 : test_capacity * 2;
		MktTestEntry *new_registry =
				realloc(test_registry,
						(size_t)new_capacity * sizeof(MktTestEntry));
		if (new_registry == NULL)
		{
			fprintf(stderr, "Failed to allocate test registry\n");
			exit(1);
		}
		test_registry = new_registry;
		test_capacity = new_capacity;
	}

	test_registry[test_count].name	= name;
	test_registry[test_count].group = group;
	test_registry[test_count].func	= func;
	test_count++;
}

void
mkt_test_register_group_fixture(
		const char *group, MktFixtureFunc setup, MktFixtureFunc teardown)
{
	/* Grow registry if needed */
	if (group_fixture_count >= group_fixture_capacity)
	{
		int				   new_capacity = group_fixture_capacity == 0
												? 16
												: group_fixture_capacity * 2;
		GroupFixtureEntry *new_registry =
				realloc(group_fixture_registry,
						(size_t)new_capacity * sizeof(GroupFixtureEntry));
		if (new_registry == NULL)
		{
			fprintf(stderr, "Failed to allocate group fixture registry\n");
			exit(1);
		}
		group_fixture_registry = new_registry;
		group_fixture_capacity = new_capacity;
	}

	group_fixture_registry[group_fixture_count].group	 = group;
	group_fixture_registry[group_fixture_count].setup	 = setup;
	group_fixture_registry[group_fixture_count].teardown = teardown;
	group_fixture_count++;
}

/* Look up group fixture by group name */
static GroupFixtureEntry *
find_group_fixture(const char *group)
{
	for (int i = 0; i < group_fixture_count; i++)
	{
		if (strcmp(group_fixture_registry[i].group, group) == 0)
			return &group_fixture_registry[i];
	}
	return NULL;
}

/* Comparison function for sorting tests by group, then by name */
static int
compare_tests(const void *a, const void *b)
{
	const MktTestEntry *ta = a;
	const MktTestEntry *tb = b;

	int group_cmp = strcmp(ta->group, tb->group);
	if (group_cmp != 0)
		return group_cmp;

	return strcmp(ta->name, tb->name);
}

/* Check if a test should run based on filters */
static bool
should_run_test(
		const char	*test_name,
		const char	*group_name,
		const char **test_filters,
		int			 test_filter_count,
		const char **group_filters,
		int			 group_filter_count)
{
	/* No filters means run all tests */
	if (test_filter_count == 0 && group_filter_count == 0)
		return true;

	/* Check test name filters */
	for (int i = 0; i < test_filter_count; i++)
	{
		if (strcmp(test_name, test_filters[i]) == 0)
			return true;
	}

	/* Check group name filters */
	for (int i = 0; i < group_filter_count; i++)
	{
		if (strcmp(group_name, group_filters[i]) == 0)
			return true;
	}

	return false;
}

void
mkt_test_fail(const char *file, int line, const char *msg)
{
	if (current_result != NULL)
	{
		current_result->passed = false;
		current_result->file   = file;
		current_result->line   = line;
		snprintf(
				current_result->failure_msg,
				sizeof(current_result->failure_msg),
				"%s",
				msg);
	}
}

bool
mkt_test_is_tap_mode(void)
{
	return running_in_tap_mode;
}

void
mkt_test_printf(const char *fmt, ...)
{
	va_list args;
	va_start(args, fmt);

	if (running_in_tap_mode)
	{
		/* In TAP mode, prefix all output with '# ' */
		printf("# ");
		vprintf(fmt, args);
	}
	else
	{
		/* In human-readable mode, indent to align with test output */
		printf("    ");
		vprintf(fmt, args);
	}

	va_end(args);
}

int
mkt_test_run_all(
		bool		 tap_output,
		const char **test_filters,
		int			 test_filter_count,
		const char **group_filters,
		int			 group_filter_count)
{
	int				   total_passed	   = 0;
	int				   total_failed	   = 0;
	int				   tests_run	   = 0;
	int				   group_passed	   = 0;
	int				   group_failed	   = 0;
	const char		  *current_group   = NULL;
	GroupFixtureEntry *current_fixture = NULL;
	MktTestResult	   result;

	/* Set TAP mode flag for TEST_PRINT macro */
	running_in_tap_mode = tap_output;

	/* Sort tests by group, then by name for consistent output */
	qsort(test_registry,
		  (size_t)test_count,
		  sizeof(MktTestEntry),
		  compare_tests);

	/* Count tests that will run (for TAP plan) */
	int planned_tests = 0;
	if (test_filter_count > 0 || group_filter_count > 0)
	{
		for (int i = 0; i < test_count; i++)
		{
			if (should_run_test(
						test_registry[i].name,
						test_registry[i].group,
						test_filters,
						test_filter_count,
						group_filters,
						group_filter_count))
			{
				planned_tests++;
			}
		}
	}
	else
	{
		planned_tests = test_count;
	}

	/* Check if any tests will run */
	if (planned_tests == 0)
	{
		fprintf(stderr, "Error: No tests match the specified filters\n");
		return 1;
	}

	/* TAP format: print version and plan */
	if (tap_output)
	{
		printf("TAP version 13\n");
		printf("1..%d\n", planned_tests);
	}
	else
	{
		printf("\n");
	}

	for (int i = 0; i < test_count; i++)
	{
		MktTestEntry *entry = &test_registry[i];

		/* Skip tests that don't match filters */
		if (!should_run_test(
					entry->name,
					entry->group,
					test_filters,
					test_filter_count,
					group_filters,
					group_filter_count))
		{
			continue;
		}

		tests_run++;

		/* Handle group changes */
		if (current_group == NULL || strcmp(current_group, entry->group) != 0)
		{
			/* Call previous group's teardown */
			if (current_group != NULL && current_fixture != NULL &&
				current_fixture->teardown != NULL)
			{
				current_fixture->teardown();
			}

			/* Print previous group summary (human-readable only) */
			if (!tap_output && current_group != NULL)
			{
				if (group_failed == 0)
					printf("  %s%s: %d passed%s\n\n",
						   MKT_COLOR_GREEN,
						   current_group,
						   group_passed,
						   MKT_COLOR_RESET);
				else
					printf("  %s%s: %d passed, %d failed%s\n\n",
						   MKT_COLOR_RED,
						   current_group,
						   group_passed,
						   group_failed,
						   MKT_COLOR_RESET);
			}

			/* Print group header (human-readable only) */
			if (!tap_output)
			{
				printf("%s=== %s ===%s\n",
					   MKT_COLOR_CYAN,
					   entry->group,
					   MKT_COLOR_RESET);
			}

			current_group	= entry->group;
			current_fixture = find_group_fixture(current_group);
			group_passed	= 0;
			group_failed	= 0;

			/* Call new group's setup */
			if (current_fixture != NULL && current_fixture->setup != NULL)
			{
				current_fixture->setup();
			}
		}

		/* Initialize result */
		result.test_name	  = entry->name;
		result.passed		  = true;
		result.failure_msg[0] = '\0';
		result.file			  = NULL;
		result.line			  = 0;

		/* Set current result for mkt_test_fail */
		current_result = &result;

		/* Run test */
		entry->func(&result);

		/* Clear current result */
		current_result = NULL;

		/* Print result */
		if (result.passed)
		{
			if (tap_output)
			{
				printf("ok %d - %s::%s\n",
					   tests_run,
					   entry->group,
					   entry->name);
			}
			else
			{
				printf("  %s[PASS]%s %s\n",
					   MKT_COLOR_GREEN,
					   MKT_COLOR_RESET,
					   entry->name);
			}
			total_passed++;
			group_passed++;
		}
		else
		{
			if (tap_output)
			{
				printf("not ok %d - %s::%s\n",
					   tests_run,
					   entry->group,
					   entry->name);
				printf("# Failed at %s:%d\n", result.file, result.line);
				printf("# %s\n", result.failure_msg);
			}
			else
			{
				printf("  %s[FAIL]%s %s\n",
					   MKT_COLOR_RED,
					   MKT_COLOR_RESET,
					   entry->name);
				printf("         %s:%d: %s\n",
					   result.file,
					   result.line,
					   result.failure_msg);
			}
			total_failed++;
			group_failed++;
		}
	}

	/* Call final group's teardown */
	if (current_group != NULL && current_fixture != NULL &&
		current_fixture->teardown != NULL)
	{
		current_fixture->teardown();
	}

	/* Print summaries (human-readable only) */
	if (!tap_output)
	{
		/* Print final group summary */
		if (current_group != NULL)
		{
			if (group_failed == 0)
				printf("  %s%s: %d passed%s\n",
					   MKT_COLOR_GREEN,
					   current_group,
					   group_passed,
					   MKT_COLOR_RESET);
			else
				printf("  %s%s: %d passed, %d failed%s\n",
					   MKT_COLOR_RED,
					   current_group,
					   group_passed,
					   group_failed,
					   MKT_COLOR_RESET);
		}

		/* Print total summary */
		printf("\n%s========================================%s\n",
			   MKT_COLOR_CYAN,
			   MKT_COLOR_RESET);
		if (total_failed == 0)
		{
			printf("%sTotal: %d passed, 0 failed%s\n",
				   MKT_COLOR_GREEN,
				   total_passed,
				   MKT_COLOR_RESET);
		}
		else
		{
			printf("%sTotal: %d passed, %d failed%s\n",
				   MKT_COLOR_RED,
				   total_passed,
				   total_failed,
				   MKT_COLOR_RESET);
		}
		printf("%s========================================%s\n\n",
			   MKT_COLOR_CYAN,
			   MKT_COLOR_RESET);
	}

	return total_failed > 0 ? 1 : 0;
}
