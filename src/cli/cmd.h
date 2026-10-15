/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * cmd.h - Command context and shared CLI infrastructure
 *
 * Provides a structured context for CLI commands instead of raw argc/argv.
 * Allows shared argument parsing in main and per-command specialization.
 */

#ifndef VS_CLI_CMD_H
#define VS_CLI_CMD_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "core/memory.h"
#include "kv.h"

/*
 * CmdContext - Shared context for all CLI commands
 *
 * Passed to command handlers instead of raw argc/argv. Allows:
 * - Shared state (memory context, verbosity, etc.)
 * - Command-specific arguments parsing
 * - Extensible via inheritance (embed in larger structs)
 */
/*
 * x / y for reported figures, or 0 when y is 0. A benchmark can measure
 * an interval the clock cannot resolve, or be asked for no runs at all,
 * and a rate of infinity reads as a result rather than as the absence of
 * one.
 */
static inline double
cmd_ratio(double x, double y)
{
	return y > 0.0 ? x / y : 0.0;
}

/*
 * Benchmark input data. rand()'s sequence belongs to the C library, so
 * the same seed gives different data on different platforms; this is the
 * same everywhere. The multiply is widened to 64 bits and masked back,
 * so nothing overflows on the way to the low 32 bits.
 */
static inline uint32_t
cmd_rand(uint32_t *state)
{
	*state = (uint32_t)(((uint64_t)*state * 1103515245u + 12345u) &
						0xFFFFFFFFu);
	return *state;
}

typedef struct CmdContext
{
	/* Command-line arguments (after command name) */
	int	   argc;
	char **argv;

	/* Program name (for usage messages) */
	const char *prog_name;

	/* Memory context for command allocations */
	VsMemCtx memctx;

	/* Global flags (parsed by main before dispatch) */
	bool verbose;
	bool quiet;

	/* --json [path] for bench commands; NULL if unset. "-" is stdout. */
	const char *json_path;

	/* Subcommand name (e.g., "distance" for "vectorsearch bench distance") */
	const char *subcmd_name;
} CmdContext;

/*
 * Command handler signature
 *
 * All command handlers take a CmdContext* and return:
 * - 0 on success
 * - Non-zero error code on failure
 */
typedef int (*CmdHandler)(CmdContext *ctx);

/* Print the tree as text; also JSON if json_path is set. */
static inline int
cmd_kv_print(const Kv *root, const char *json_path)
{
	int rc = 0;

	if (root == NULL)
		return 0;
	char *text = kv_to_text(root);
	if (text != NULL)
	{
		fputs(text, stdout);
		vs_free(text);
	}
	if (json_path != NULL)
		rc = kv_write_json(root, json_path);
	return rc;
}

/*
 * Helper macros for usage messages
 */
#define CMD_USAGE_HEADER(ctx, subcmd) \
	printf("Usage: %s %s [OPTIONS]\n\n", (ctx)->prog_name, subcmd)

#define CMD_USAGE_EXAMPLE(ctx, subcmd, args) \
	printf("  %s %s %s\n", (ctx)->prog_name, subcmd, args)

#define CMD_USAGE_JSON()                                     \
	printf("  --json [path]     Write JSON results to path " \
		   "(omit or '-' = stdout)\n");                      \
	printf("  --quiet           Suppress human bench output\n")

#endif /* VS_CLI_CMD_H */
