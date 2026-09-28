/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * run_tests.c - Main test runner entry point
 */

#include <stdlib.h>
#include <string.h>

#include "vs_test.h"

#define MAX_FILTERS 64

/* Split a comma-separated string into multiple filters */
static char *
parse_comma_separated(
		const char	*input,
		const char **filters,
		int			*filter_count,
		int			 max_filters,
		const char	*filter_type)
{
	char *input_copy = strdup(input);
	if (input_copy == NULL)
	{
		fprintf(stderr, "Error: Out of memory\n");
		exit(1);
	}

	char *token = strtok(input_copy, ",");
	while (token != NULL)
	{
		/* Skip leading/trailing whitespace */
		while (*token == ' ' || *token == '\t')
			token++;

		if (*filter_count >= max_filters)
		{
			fprintf(stderr,
					"Error: Too many %s filters (max %d)\n",
					filter_type,
					max_filters);
			free(input_copy);
			exit(1);
		}

		/* Point directly to token in input_copy (no strdup needed) */
		filters[*filter_count] = token;
		(*filter_count)++;
		token = strtok(NULL, ",");
	}

	/* Return input_copy so caller can free it */
	return input_copy;
}

int
main(int argc, char *argv[])
{
	bool		tap_output = false;
	const char *test_filters[MAX_FILTERS];
	int			test_filter_count = 0;
	const char *group_filters[MAX_FILTERS];
	int			group_filter_count = 0;
	char	   *test_filter_buf	   = NULL;
	char	   *group_filter_buf   = NULL;

	/* Parse command line arguments */
	for (int i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "--tap") == 0)
		{
			tap_output = true;
		}
		else if (strcmp(argv[i], "--test") == 0 || strcmp(argv[i], "-t") == 0)
		{
			if (i + 1 >= argc)
			{
				fprintf(stderr, "Error: %s requires an argument\n", argv[i]);
				return 1;
			}
			test_filter_buf = parse_comma_separated(
					argv[++i],
					test_filters,
					&test_filter_count,
					MAX_FILTERS,
					"test");
		}
		else if (strcmp(argv[i], "--group") == 0 || strcmp(argv[i], "-g") == 0)
		{
			if (i + 1 >= argc)
			{
				fprintf(stderr, "Error: %s requires an argument\n", argv[i]);
				return 1;
			}
			group_filter_buf = parse_comma_separated(
					argv[++i],
					group_filters,
					&group_filter_count,
					MAX_FILTERS,
					"group");
		}
		else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
		{
			printf("Usage: %s [OPTIONS]\n", argv[0]);
			printf("\nOptions:\n");
			printf("  --tap          Output in TAP (Test Anything Protocol) "
				   "format\n");
			printf("  --test, -t     Run only tests matching the given "
				   "name\n");
			printf("                 (comma-separated or repeated multiple "
				   "times)\n");
			printf("  --group, -g    Run only tests in the given group\n");
			printf("                 (comma-separated or repeated multiple "
				   "times)\n");
			printf("  --help, -h     Show this help message\n");
			printf("\nExamples:\n");
			printf("  %s --test vector_norm\n", argv[0]);
			printf("  %s --group Vector\n", argv[0]);
			printf("  %s -t test1 -t test2 --group MyGroup\n", argv[0]);
			printf("  %s --test test1,test2 --group Group1,Group2\n", argv[0]);
			printf("  %s --tap --group Vector\n", argv[0]);
			return 0;
		}
		else
		{
			fprintf(stderr, "Unknown option: %s\n", argv[i]);
			fprintf(stderr, "Use --help for usage information\n");
			return 1;
		}
	}

	int result = vs_test_run_all(
			tap_output,
			test_filters,
			test_filter_count,
			group_filters,
			group_filter_count);

	/* Free filter buffers */
	if (test_filter_buf != NULL)
		free(test_filter_buf);
	if (group_filter_buf != NULL)
		free(group_filter_buf);

	return result;
}
