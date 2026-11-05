/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * test_bench_compare.c - The bench compare command on result files
 */

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "cli/cmd.h"
#include "vs_test.h"

int cmd_bench_compare(CmdContext *ctx);

TEST_GROUP(BenchCompare);
TEST_MEMCTX_FIXTURE();

/* Write json to a fresh temp file; path is empty on failure. */
static void
write_tmp(char *path, size_t sz, const char *json)
{
	const char *dir = getenv("TMPDIR");
	FILE	   *f;
	int			fd;

	snprintf(path, sz, "%s/vs-cmp-XXXXXX", dir != NULL ? dir : "/tmp");
	fd = mkstemp(path);
	if (fd < 0)
	{
		path[0] = '\0';
		return;
	}
	f = fdopen(fd, "w");
	if (f == NULL)
	{
		close(fd);
		path[0] = '\0';
		return;
	}
	/* A short write must not leave a file that parses differently. */
	if (fputs(json, f) < 0 || fclose(f) != 0)
		path[0] = '\0';
}

/* Run the handler with stdout muted: the command's report would
 * otherwise corrupt the runner's TAP stream. -1 when muting fails. */
static int
run_muted(CmdContext *ctx)
{
	int saved;
	int devnull;
	int ret;

	fflush(stdout);
	saved	= dup(STDOUT_FILENO);
	devnull = open("/dev/null", O_WRONLY);
	if (saved < 0 || devnull < 0)
	{
		if (saved >= 0)
			close(saved);
		if (devnull >= 0)
			close(devnull);
		return -1;
	}
	dup2(devnull, STDOUT_FILENO);
	close(devnull);
	ret = cmd_bench_compare(ctx);
	fflush(stdout);
	dup2(saved, STDOUT_FILENO);
	close(saved);
	return ret;
}

/* Run compare over the two documents; -1 when the files fail. */
static int
run_compare(const char *base_json, const char *head_json)
{
	char	   base_path[128];
	char	   head_path[128];
	char	   cmd_name[] = "compare";
	char	  *argv[]	  = {cmd_name, base_path, head_path};
	CmdContext ctx		  = {
				   .argc	  = 3,
				   .argv	  = argv,
				   .prog_name = "vectorsearch",
	   };
	int ret;

	write_tmp(base_path, sizeof(base_path), base_json);
	write_tmp(head_path, sizeof(head_path), head_json);
	if (base_path[0] == '\0' || head_path[0] == '\0')
	{
		if (base_path[0] != '\0')
			unlink(base_path);
		if (head_path[0] != '\0')
			unlink(head_path);
		return -1;
	}
	ret = run_muted(&ctx);
	unlink(base_path);
	unlink(head_path);
	return ret;
}

TEST(root_runs_feed_noise_gate)
{
	/* Quantize's layout: stddev on the metric, runs at the root.
	 * A 35% drop sits inside 2·SE of five such noisy runs. */
	int ret = run_compare(
			"{\"bench\":\"quantize\",\"runs\":5,\"metrics\":["
			"{\"name\":\"encode/single\",\"unit\":\"vec/s\","
			"\"value\":10000000,\"stddev\":4000000}]}",
			"{\"bench\":\"quantize\",\"runs\":5,\"metrics\":["
			"{\"name\":\"encode/single\",\"unit\":\"vec/s\","
			"\"value\":6500000,\"stddev\":4000000}]}");

	ASSERT_EQ(0, ret, "noise judged from the root run count");
}

TEST(duplicate_base_metric_rejected)
{
	int ret = run_compare(
			"{\"metrics\":["
			"{\"name\":\"x\",\"unit\":\"vec/s\",\"value\":100},"
			"{\"name\":\"x\",\"unit\":\"vec/s\",\"value\":200}]}",
			"{\"metrics\":["
			"{\"name\":\"x\",\"unit\":\"vec/s\",\"value\":100}]}");

	ASSERT_EQ(1, ret, "ambiguous baseline pairing");
}

TEST(duplicate_head_metric_rejected)
{
	int ret = run_compare(
			"{\"metrics\":["
			"{\"name\":\"x\",\"unit\":\"vec/s\",\"value\":100}]}",
			"{\"metrics\":["
			"{\"name\":\"x\",\"unit\":\"vec/s\",\"value\":100},"
			"{\"name\":\"x\",\"unit\":\"vec/s\",\"value\":200}]}");

	ASSERT_EQ(1, ret, "ambiguous head pairing");
}

TEST(identical_results_pass)
{
	int ret = run_compare(
			"{\"metrics\":["
			"{\"name\":\"x\",\"unit\":\"vec/s\",\"value\":100}]}",
			"{\"metrics\":["
			"{\"name\":\"x\",\"unit\":\"vec/s\",\"value\":100}]}");

	ASSERT_EQ(0, ret, "no change passes");
}

TEST(warn_only_exits_zero)
{
	/* A 20% drop with no stats warns; a warning must not flip the
	 * exit code. */
	int ret = run_compare(
			"{\"metrics\":["
			"{\"name\":\"x\",\"unit\":\"vec/s\",\"value\":100}]}",
			"{\"metrics\":["
			"{\"name\":\"x\",\"unit\":\"vec/s\",\"value\":80}]}");

	ASSERT_EQ(0, ret, "warning keeps exit 0");
}

TEST(unit_change_fails)
{
	int ret = run_compare(
			"{\"metrics\":["
			"{\"name\":\"t\",\"unit\":\"us\",\"value\":100}]}",
			"{\"metrics\":["
			"{\"name\":\"t\",\"unit\":\"ms\",\"value\":1}]}");

	ASSERT_EQ(1, ret, "incomparable units fail");
}

TEST(missing_units_fail)
{
	/* With no unit on either side the better direction is unknown;
	 * a doubled latency must not pass as an improvement. */
	int ret = run_compare(
			"{\"metrics\":[{\"name\":\"t\",\"value\":100}]}",
			"{\"metrics\":[{\"name\":\"t\",\"value\":200}]}");

	ASSERT_EQ(1, ret, "unitless pair fails");
}

TEST(one_sided_unit_tolerated)
{
	/* A single annotated side still gives the direction. */
	int ret = run_compare(
			"{\"metrics\":["
			"{\"name\":\"t\",\"unit\":\"us\",\"value\":100}]}",
			"{\"metrics\":[{\"name\":\"t\",\"value\":100}]}");

	ASSERT_EQ(0, ret, "base unit alone suffices");
}

TEST(paired_metric_without_value_fails)
{
	int ret = run_compare(
			"{\"metrics\":["
			"{\"name\":\"t\",\"unit\":\"us\",\"value\":null}]}",
			"{\"metrics\":["
			"{\"name\":\"t\",\"unit\":\"us\",\"value\":100}]}");

	ASSERT_EQ(1, ret, "failed baseline measurement fails");
}

TEST(missing_metrics_array_fails)
{
	int ret = run_compare(
			"{\"bench\":\"quantize\"}",
			"{\"metrics\":["
			"{\"name\":\"x\",\"unit\":\"vec/s\",\"value\":100}]}");

	ASSERT_EQ(1, ret, "no metrics array errors");
}

TEST(unnamed_base_metric_rejected)
{
	int ret = run_compare(
			"{\"metrics\":["
			"{\"name\":\"x\",\"unit\":\"vec/s\",\"value\":100},"
			"{\"unit\":\"vec/s\",\"value\":100}]}",
			"{\"metrics\":["
			"{\"name\":\"x\",\"unit\":\"vec/s\",\"value\":100}]}");

	ASSERT_EQ(1, ret, "a nameless base entry errors");
}

TEST(unnamed_head_metric_rejected)
{
	/* Nameless entries can never be paired, so a failed measurement
	 * must not hide behind having no name. */
	int ret = run_compare(
			"{\"metrics\":["
			"{\"name\":\"x\",\"unit\":\"vec/s\",\"value\":100}]}",
			"{\"metrics\":["
			"{\"name\":\"x\",\"unit\":\"vec/s\",\"value\":100},"
			"{\"unit\":\"vec/s\",\"value\":null}]}");

	ASSERT_EQ(1, ret, "a nameless head entry errors");
}

TEST(empty_base_metrics_fail)
{
	int ret = run_compare(
			"{\"metrics\":[]}",
			"{\"metrics\":["
			"{\"name\":\"x\",\"unit\":\"vec/s\",\"value\":100}]}");

	ASSERT_EQ(1, ret, "an empty base errors");
}

TEST(json_flag_rejected)
{
	char	   cmd_name[] = "compare";
	char	   a[]		  = "a.json";
	char	   b[]		  = "b.json";
	char	  *argv[]	  = {cmd_name, a, b};
	CmdContext ctx		  = {
				   .argc	  = 3,
				   .argv	  = argv,
				   .prog_name = "vectorsearch",
				   .json_root = kv_obj(),
	   };

	/* Compare writes no result file; a --json request is a mistake. */
	ASSERT_EQ(1, run_muted(&ctx), "--json rejected");
}

TEST(regression_outside_noise_fails)
{
	/* The same 35% drop with tight per-metric stats must fail. */
	int ret = run_compare(
			"{\"metrics\":["
			"{\"name\":\"q\",\"unit\":\"vec/s\",\"value\":10000000,"
			"\"stddev\":100000,\"n\":9}]}",
			"{\"metrics\":["
			"{\"name\":\"q\",\"unit\":\"vec/s\",\"value\":6500000,"
			"\"stddev\":100000,\"n\":9}]}");

	ASSERT_EQ(1, ret, "stable drop fails");
}

TEST(new_metric_without_value_fails)
{
	/* A failed measurement must not hide behind having no baseline. */
	int ret = run_compare(
			"{\"metrics\":["
			"{\"name\":\"x\",\"unit\":\"vec/s\",\"value\":100}]}",
			"{\"metrics\":["
			"{\"name\":\"x\",\"unit\":\"vec/s\",\"value\":100},"
			"{\"name\":\"y\",\"unit\":\"vec/s\",\"value\":null}]}");

	ASSERT_EQ(1, ret, "new metric with no value fails");
}

TEST(missing_head_metric_fails)
{
	int ret = run_compare(
			"{\"metrics\":["
			"{\"name\":\"x\",\"unit\":\"vec/s\",\"value\":100},"
			"{\"name\":\"y\",\"unit\":\"vec/s\",\"value\":100}]}",
			"{\"metrics\":["
			"{\"name\":\"x\",\"unit\":\"vec/s\",\"value\":100}]}");

	ASSERT_EQ(1, ret, "vanished metric fails");
}

TEST(surplus_argument_rejected)
{
	/* A third file (say from a CI glob) must not pass unexamined. */
	char	   cmd_name[] = "compare";
	char	   a[]		  = "a.json";
	char	   b[]		  = "b.json";
	char	   c[]		  = "c.json";
	char	  *argv[]	  = {cmd_name, a, b, c};
	CmdContext ctx		  = {
				   .argc	  = 4,
				   .argv	  = argv,
				   .prog_name = "vectorsearch",
	   };

	ASSERT_EQ(1, run_muted(&ctx), "extra file rejected");
}
