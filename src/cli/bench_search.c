/*
 * mkt bench search
 *
 * Benchmark end-to-end index search performance.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cmd.h"

int
cmd_bench_search(CmdContext *ctx)
{
	printf("Search benchmark not yet implemented\n");
	CMD_USAGE_HEADER(ctx, "bench search");
	printf("Options:\n");
	printf("  --index <path>     Path to index file\n");
	printf("  --queries <path>   Path to query vectors (.fvecs)\n");
	printf("  --k <int>          Number of neighbors (default: 10)\n");
	printf("  --nprobe <int>     Number of clusters to search (default: "
		   "10)\n");
	printf("  --help             Show this help\n");
	return 1;
}
