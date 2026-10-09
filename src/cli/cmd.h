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

#include "core/memory.h"

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

/*
 * Helper macros for usage messages
 */
#define CMD_USAGE_HEADER(ctx, subcmd) \
	printf("Usage: %s %s [OPTIONS]\n\n", (ctx)->prog_name, subcmd)

#define CMD_USAGE_EXAMPLE(ctx, subcmd, args) \
	printf("  %s %s %s\n", (ctx)->prog_name, subcmd, args)

#endif /* VS_CLI_CMD_H */
