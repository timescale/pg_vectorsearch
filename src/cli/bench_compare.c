/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * vectorsearch bench compare
 *
 * Diff the metrics arrays of two JSON result files. Pair by name.
 * Unit decides whether higher or lower is better.
 */

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "cmd.h"
#include "kv.h"
#include "kv_cmp.h"

typedef struct Cmp
{
	int fails;
	int warns;
} Cmp;

static const Kv *
find_named(const Kv *arr, const char *name)
{
	size_t n = kv_count(arr);

	for (size_t i = 0; i < n; i++)
	{
		const Kv   *e  = kv_at(arr, i);
		const char *en = kv_as_str(kv_get(e, "name"));

		if (en != NULL && name != NULL && strcmp(en, name) == 0)
			return e;
	}
	return NULL;
}

/* Older results may omit sample statistics. */
static bool
valid_stat(const Kv *stat)
{
	return stat == NULL ||
		   (kv_is_num(stat) && isfinite(kv_num(stat)) && kv_num(stat) >= 0);
}

/* Validate even metrics that will not have a partner. */
static bool
validate_metrics(const Kv *arr, const char *path)
{
	for (size_t i = 0; i < kv_count(arr); i++)
	{
		const Kv *entry = kv_at(arr, i);
		KvMetric  m;

		kv_metric_load(entry, &m);
		if (m.name == NULL || m.name[0] == '\0')
		{
			fprintf(stderr, "Error: unnamed metric in %s\n", path);
			return false;
		}
		if (find_named(arr, m.name) != entry)
		{
			fprintf(stderr,
					"Error: duplicate metric '%s' in %s\n",
					m.name,
					path);
			return false;
		}
		if (!m.has_value || !isfinite(m.value) ||
			!valid_stat(kv_get(entry, "stddev")) ||
			!valid_stat(kv_get(entry, "n")))
		{
			fprintf(stderr,
					"Error: invalid measurement for '%s' in %s\n",
					m.name,
					path);
			return false;
		}
		if (kv_get(entry, "unit") != NULL &&
			(m.unit == NULL || m.unit[0] == '\0'))
		{
			fprintf(stderr,
					"Error: invalid unit for '%s' in %s\n",
					m.name,
					path);
			return false;
		}
	}
	return true;
}

static const char *
tag_of(Cmp *c, const KvCmp *r)
{
	if (r->kind == KV_CMP_FAIL)
	{
		c->fails++;
		return " FAIL";
	}
	if (r->kind == KV_CMP_WARN)
	{
		c->warns++;
		return " WARN";
	}
	if (r->inside_noise && r->worse)
		return " noise";
	return "";
}

static void
print_usage(CmdContext *ctx)
{
	CMD_USAGE_HEADER(ctx, "bench compare");
	printf("Compare two JSON result files.\n\n");
	printf("  %s bench compare BASE.json HEAD.json\n", ctx->prog_name);
	printf("\nExit status: 0 pass or warn, 1 regression, 2 error.\n");
}

static int
compare_files(const char *base_path, const char *head_path)
{
	Kv		   *base;
	Kv		   *head;
	const Kv   *ba;
	const Kv   *ha;
	const char *bname;
	const char *hname;
	Cmp			c = {0};
	size_t		n;

	base = kv_from_json_file(base_path);
	head = kv_from_json_file(head_path);
	if (base == NULL || head == NULL)
		return CMD_BENCH_ERROR;

	ba = kv_get(base, "metrics");
	ha = kv_get(head, "metrics");
	if (!kv_is_arr(ba) || !kv_is_arr(ha))
	{
		fprintf(stderr, "Error: JSON has no metrics array\n");
		return CMD_BENCH_ERROR;
	}

	bname = kv_as_str(kv_get(base, "bench"));
	hname = kv_as_str(kv_get(head, "bench"));
	if (kv_get(base, "bench") != NULL || kv_get(head, "bench") != NULL)
	{
		if (bname == NULL || hname == NULL || bname[0] == '\0' ||
			strcmp(bname, hname) != 0)
		{
			fprintf(stderr, "Error: bench names must match\n");
			return CMD_BENCH_ERROR;
		}
	}
	n = kv_count(ba);
	if (n == 0)
	{
		fprintf(stderr, "Error: base has no metrics\n");
		return CMD_BENCH_ERROR;
	}
	if (!valid_stat(kv_get(base, "runs")) || !valid_stat(kv_get(head, "runs")))
	{
		fprintf(stderr, "Error: invalid run count\n");
		return CMD_BENCH_ERROR;
	}
	if (!validate_metrics(ba, base_path) || !validate_metrics(ha, head_path))
		return CMD_BENCH_ERROR;

	printf("compare  %s -> %s\n", base_path, head_path);
	if (bname != NULL)
		printf("bench    %s\n", bname);
	/* The Δ is two bytes, so give its one glyph an extra column. */
	printf("  %-24s %12s %12s %10s\n", "metric", "base", "head", "Δ");

	/* Every base metric must still exist in head. */
	for (size_t i = 0; i < n; i++)
	{
		const char *name = kv_as_str(kv_get(kv_at(ba, i), "name"));

		if (find_named(ha, name) == NULL)
		{
			c.fails++;
			printf("  %-24s  missing from head FAIL\n", name);
		}
	}

	/* Older writers put the run count at the root only. */
	double bruns = kv_num(kv_get(base, "runs"));
	double hruns = kv_num(kv_get(head, "runs"));

	n = kv_count(ha);
	for (size_t i = 0; i < n; i++)
	{
		const Kv *he = kv_at(ha, i);
		KvMetric  hf;
		KvMetric  bf;
		const Kv *be;
		KvCmp	  r;
		char	  delta[16];

		kv_metric_load(he, &hf);
		be = find_named(ba, hf.name);
		if (be == NULL)
		{
			printf("  %-24s  (new)\n", hf.name);
			continue;
		}
		kv_metric_load(be, &bf);
		/* With no unit on either side the better direction is
		 * unknown, so a regression could pass as an improvement. */
		if (bf.unit == NULL && hf.unit == NULL)
		{
			fprintf(stderr, "Error: no unit for '%s'\n", hf.name);
			return CMD_BENCH_ERROR;
		}
		/* A changed unit makes the values incomparable. */
		if (bf.unit != NULL && hf.unit != NULL &&
			strcmp(bf.unit, hf.unit) != 0)
		{
			fprintf(stderr,
					"Error: unit changed for '%s' (%s -> %s)\n",
					hf.name,
					bf.unit,
					hf.unit);
			return CMD_BENCH_ERROR;
		}
		kv_cmp_values(
				hf.unit != NULL ? hf.unit : bf.unit,
				(KvSample){bf.value, bf.stddev, bf.n > 0 ? bf.n : bruns},
				(KvSample){hf.value, hf.stddev, hf.n > 0 ? hf.n : hruns},
				&r);
		/* A change against a zero baseline has no percentage. */
		if (bf.value == 0.0 && hf.value != 0.0)
			snprintf(delta, sizeof(delta), "%9s", "n/a");
		else
			snprintf(delta, sizeof(delta), "%+8.2f%%", r.rel * 100.0);
		printf("  %-24s %12.6g %12.6g %s%s\n",
			   hf.name,
			   bf.value,
			   hf.value,
			   delta,
			   tag_of(&c, &r));
	}
	printf("  %d fail, %d warn\n", c.fails, c.warns);
	return c.fails > 0 ? CMD_BENCH_REGRESSION : 0;
}

int
cmd_bench_compare(CmdContext *ctx)
{
	KvBuild *kb;
	int		 ret;

	for (int i = 1; i < ctx->argc; i++)
	{
		if (strcmp(ctx->argv[i], "-h") == 0 ||
			strcmp(ctx->argv[i], "--help") == 0)
		{
			print_usage(ctx);
			return 0;
		}
	}
	/* Exactly two result files; a stray third would go unexamined. */
	if (ctx->argc != 3)
	{
		print_usage(ctx);
		return CMD_BENCH_ERROR;
	}
	/* Compare reads result files and writes none of its own. */
	if (ctx->json_path != NULL || ctx->json_root != NULL)
	{
		fprintf(stderr, "Error: compare does not take --json\n");
		return CMD_BENCH_ERROR;
	}

	/* Both parsed trees live in one arena, dropped as a whole. */
	kb = kv_build_open();
	if (kb == NULL)
		return CMD_BENCH_ERROR;
	ret = compare_files(ctx->argv[1], ctx->argv[2]);
	kv_build_close(kb);
	if (fflush(stdout) == EOF || ferror(stdout))
	{
		fprintf(stderr, "Error: cannot write comparison report\n");
		return CMD_BENCH_ERROR;
	}
	return ret;
}
