/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * vectorsearch bench compare
 *
 * Diff the metrics arrays of two JSON result files. Pair by name.
 * Unit decides whether higher or lower is better.
 */

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
	size_t i, n;

	if (!kv_is_arr(arr) || name == NULL)
		return NULL;
	n = kv_count(arr);
	for (i = 0; i < n; i++)
	{
		const Kv   *e  = kv_at(arr, i);
		const char *en = kv_as_str(kv_get(e, "name"));

		if (en != NULL && strcmp(en, name) == 0)
			return e;
	}
	return NULL;
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
}

int
cmd_bench_compare(CmdContext *ctx)
{
	const char *base_path;
	const char *head_path;
	Kv		   *base;
	Kv		   *head;
	const Kv   *ba;
	const Kv   *ha;
	const char *bname;
	const char *hname;
	Cmp			c = {0};
	size_t		i, n;

	if (ctx->argc < 3)
	{
		print_usage(ctx);
		return 1;
	}
	base_path = ctx->argv[1];
	head_path = ctx->argv[2];
	base	  = kv_from_json_file(base_path);
	head	  = kv_from_json_file(head_path);
	if (base == NULL || head == NULL)
		return 1;

	ba = kv_get(base, "metrics");
	ha = kv_get(head, "metrics");
	if (!kv_is_arr(ba) || !kv_is_arr(ha))
	{
		fprintf(stderr, "Error: JSON has no metrics array\n");
		return 1;
	}

	bname = kv_as_str(kv_get(base, "bench"));
	hname = kv_as_str(kv_get(head, "bench"));
	printf("compare  %s -> %s\n", base_path, head_path);
	if (bname != NULL)
		printf("bench    %s\n", bname);
	if (hname != NULL && bname != NULL && strcmp(bname, hname) != 0)
		printf("warning: bench names differ (%s vs %s)\n", bname, hname);
	printf("  %-24s %12s %12s %8s\n", "metric", "base", "head", "Δ");

	n = kv_count(ha);
	for (i = 0; i < n; i++)
	{
		const Kv *he = kv_at(ha, i);
		KvMetric  hf;
		KvMetric  bf;
		const Kv *be;
		KvCmp	  r;

		kv_metric_load(he, &hf);
		if (hf.name == NULL || !hf.has_value)
			continue;
		be = find_named(ba, hf.name);
		if (be == NULL)
		{
			printf("  %-24s  (new)\n", hf.name);
			continue;
		}
		kv_metric_load(be, &bf);
		kv_cmp_values(
				hf.unit != NULL ? hf.unit : bf.unit,
				(KvSample){bf.value, bf.stddev, bf.n},
				(KvSample){hf.value, hf.stddev, hf.n},
				&r);
		printf("  %-24s %12.6g %12.6g %+8.2f%%%s\n",
			   hf.name,
			   bf.value,
			   hf.value,
			   r.rel * 100.0,
			   tag_of(&c, &r));
	}
	printf("  %d fail, %d warn\n", c.fails, c.warns);
	return c.fails > 0 ? 1 : 0;
}
