/*
 * mkt bench quantize
 *
 * Benchmark quantization methods on standard ANN datasets.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cmd.h"

int
cmd_bench_quantize(CmdContext *ctx)
{
	printf("Quantization benchmark not yet implemented\n");
	CMD_USAGE_HEADER(ctx, "bench quantize");
	printf("Options:\n");
	printf("  --dataset <name>   Dataset name: sift1m, gist1m, deep1m, "
		   "glove\n");
	printf("  --base <path>      Path to base vectors (.fvecs)\n");
	printf("  --query <path>     Path to query vectors (.fvecs)\n");
	printf("  --gt <path>        Path to ground truth (.ivecs)\n");
	printf("  --k <int>          Number of neighbors (default: 10)\n");
	printf("  --methods <list>   Comma-separated methods (default: all)\n");
	printf("  --output <path>    Write results to CSV\n");
	printf("  --help             Show this help\n");
	return 1;
}
