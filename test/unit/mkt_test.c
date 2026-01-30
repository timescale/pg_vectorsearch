/*
 * mkt_test.c - Test framework implementation
 */

#include "mkt_test.h"

/* Test registry */
static MktTestEntry *test_registry = NULL;
static int			 test_count	   = 0;
static int			 test_capacity = 0;

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

int
mkt_test_run_all(void)
{
	int			  passed		= 0;
	int			  failed		= 0;
	const char	 *current_group = NULL;
	MktTestResult result;

	printf("\n");

	for (int i = 0; i < test_count; i++)
	{
		MktTestEntry *entry = &test_registry[i];

		/* Print group header if group changed */
		if (current_group == NULL || strcmp(current_group, entry->group) != 0)
		{
			if (current_group != NULL)
				printf("\n");
			printf("%s=== %s ===%s\n",
				   MKT_COLOR_CYAN,
				   entry->group,
				   MKT_COLOR_RESET);
			current_group = entry->group;
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
			printf("  %s[PASS]%s %s\n",
				   MKT_COLOR_GREEN,
				   MKT_COLOR_RESET,
				   entry->name);
			passed++;
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
			failed++;
		}
	}

	/* Print summary */
	printf("\n%s========================================%s\n",
		   MKT_COLOR_CYAN,
		   MKT_COLOR_RESET);
	if (failed == 0)
	{
		printf("%sTotal: %d passed, 0 failed%s\n",
			   MKT_COLOR_GREEN,
			   passed,
			   MKT_COLOR_RESET);
	}
	else
	{
		printf("%sTotal: %d passed, %d failed%s\n",
			   MKT_COLOR_RED,
			   passed,
			   failed,
			   MKT_COLOR_RESET);
	}
	printf("%s========================================%s\n\n",
		   MKT_COLOR_CYAN,
		   MKT_COLOR_RESET);

	return failed > 0 ? 1 : 0;
}
