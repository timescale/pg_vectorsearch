/*
 * meerkat CLI tool
 *
 * Command-line tool for testing, benchmarking, and debugging meerkat
 * components in standalone mode (without PostgreSQL).
 */

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "algo/kmeans.h"
#include "cmd.h"
#include "core/memory.h"

/* Forward declarations of command handlers */
int cmd_bench_distance(CmdContext *ctx);
int cmd_bench_quantize(CmdContext *ctx);
int cmd_bench_cluster(CmdContext *ctx);
int cmd_bench_search(CmdContext *ctx);
int cmd_bench_page_score(CmdContext *ctx);

/* Subcommand structure */
typedef struct
{
	const char *name;
	CmdHandler	handler;
	const char *description;
} Command;

/*
 * Available subcommands
 *
 * Organized into categories for help display.
 */
static const Command bench_commands[] = {
		{"distance", cmd_bench_distance, "Benchmark distance computations"},
		{"quantize", cmd_bench_quantize, "Benchmark quantization methods"},
		{"cluster", cmd_bench_cluster, "Benchmark clustering algorithms"},
		{"search", cmd_bench_search, "Benchmark end-to-end search"},
		{"page-score", cmd_bench_page_score, "Benchmark page-level scoring"},
		{NULL, NULL, NULL},
};

/*
 * print_usage - Display usage information
 */
static void
print_usage(const char *prog)
{
	printf("Usage: %s <command> [options]\n\n", prog);
	printf("Commands:\n");
	printf("  Benchmarks:\n");
	for (const Command *cmd = bench_commands; cmd->name; cmd++)
	{
		printf("    bench %-12s %s\n", cmd->name, cmd->description);
	}
	printf("\n");
	printf("Options:\n");
	printf("  -h, --help       Show this help message\n");
	printf("  -v, --version    Show version information\n");
	printf("\n");
	printf("Examples:\n");
	printf("  %s bench distance --dim 768 --count 10000 --metric l2\n", prog);
	printf("  %s bench quantize --dataset sift1m --k 10\n", prog);
	printf("  %s bench cluster --dim 128 --nvecs 100000 --nlist 1000\n", prog);
	printf("\n");
	printf("For help on a specific command:\n");
	printf("  %s <command> --help\n", prog);
}

/*
 * print_version - Display version information
 */
static void
print_version(void)
{
	printf("meerkat 0.1.0\n");
	printf("PostgreSQL index access method for ANN vector search\n");
}

/*
 * dispatch_bench_command - Dispatch to benchmark subcommand
 */
static int
dispatch_bench_command(CmdContext *parent_ctx, int argc, char **argv)
{
	if (argc < 1)
	{
		fprintf(stderr, "Error: 'bench' requires a subcommand\n\n");
		printf("Available subcommands:\n");
		for (const Command *cmd = bench_commands; cmd->name; cmd++)
		{
			printf("  %-12s %s\n", cmd->name, cmd->description);
		}
		return 1;
	}

	const char *subcmd = argv[0];

	/* Find handler */
	const Command *cmd = NULL;
	for (const Command *c = bench_commands; c->name; c++)
	{
		if (strcmp(subcmd, c->name) == 0)
		{
			cmd = c;
			break;
		}
	}

	if (cmd == NULL)
	{
		fprintf(stderr, "Error: unknown benchmark subcommand '%s'\n", subcmd);
		return 1;
	}

	/* Create command context */
	CmdContext cmd_ctx = {
			.argc		 = argc,
			.argv		 = argv,
			.prog_name	 = parent_ctx->prog_name,
			.memctx		 = parent_ctx->memctx,
			.verbose	 = parent_ctx->verbose,
			.quiet		 = parent_ctx->quiet,
			.subcmd_name = subcmd,
	};

	/* Dispatch to handler */
	return cmd->handler(&cmd_ctx);
}

/*
 * main - Entry point
 */
int
main(int argc, char **argv)
{
	/*
	 * BLAS libraries use OpenMP/pthreads internally. In PostgreSQL,
	 * multi-threaded BLAS is unsafe (backends fork). Warn if not
	 * configured for single-threaded to keep benchmarks PG-representative.
	 */
	if (!mkt_cblas_is_single_threaded())
	{
		fprintf(stderr,
				"Note: OMP_NUM_THREADS not set to 1. "
				"BLAS may use multiple threads.\n"
				"  For PG-representative benchmarks: "
				"OMP_NUM_THREADS=1 %s ...\n\n",
				argv[0]);
	}

	/* Initialize memory context for CLI */
	MktMemCtx cli_memctx = mkt_memctx_create(NULL, "cli");
	mkt_memctx_switch(cli_memctx);

	/* Need at least one argument (command) */
	if (argc < 2)
	{
		print_usage(argv[0]);
		mkt_memctx_switch(NULL);
		mkt_memctx_delete(cli_memctx);
		return 1;
	}

	/* Create main command context */
	CmdContext main_ctx = {
			.argc		 = argc - 1,
			.argv		 = argv + 1,
			.prog_name	 = argv[0],
			.memctx		 = cli_memctx,
			.verbose	 = false,
			.quiet		 = false,
			.subcmd_name = NULL,
	};

	const char *cmd = argv[1];

	/* Handle help and version flags */
	if (strcmp(cmd, "-h") == 0 || strcmp(cmd, "--help") == 0)
	{
		print_usage(argv[0]);
		mkt_memctx_switch(NULL);
		mkt_memctx_delete(cli_memctx);
		return 0;
	}

	if (strcmp(cmd, "-v") == 0 || strcmp(cmd, "--version") == 0)
	{
		print_version();
		mkt_memctx_switch(NULL);
		mkt_memctx_delete(cli_memctx);
		return 0;
	}

	/* Dispatch to command handler */
	int ret;
	if (strcmp(cmd, "bench") == 0)
	{
		ret = dispatch_bench_command(&main_ctx, argc - 2, argv + 2);
	}
	else
	{
		fprintf(stderr, "Error: unknown command '%s'\n\n", cmd);
		print_usage(argv[0]);
		ret = 1;
	}

	/* Clean up memory context */
	mkt_memctx_switch(NULL);
	mkt_memctx_delete(cli_memctx);
	return ret;
}
