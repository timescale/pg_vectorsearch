/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * test_kv_json.c - JSON encoder
 */

#include <math.h>
#include <unistd.h>

#include "cli/kv.h"
#include "core/memory.h"
#include "vs_test.h"

TEST_GROUP(KvJson);
TEST_MEMCTX_FIXTURE();

TEST(empty_object)
{
	Kv	 *o = kv_obj();
	char *s = kv_to_json(o);
	ASSERT_STR_EQ("{}", s, "empty object");
	vs_free(s);
}

TEST(empty_array)
{
	Kv	 *a = kv_arr();
	char *s = kv_to_json(a);
	ASSERT_STR_EQ("[]", s, "empty array");
	vs_free(s);
}

TEST(scalars)
{
	Kv *o = kv_obj();
	kv_obj_put(o, "n", kv_null());
	kv_obj_put(o, "t", kv_bool(true));
	kv_obj_put(o, "f", kv_bool(false));
	kv_obj_put(o, "i", kv_int(-42));
	kv_obj_put(o, "s", kv_str("hi"));
	char *s = kv_to_json(o);
	ASSERT_STR_EQ(
			"{\"n\":null,\"t\":true,\"f\":false,\"i\":-42,\"s\":\"hi\"}",
			s,
			"scalars");
	vs_free(s);
}

TEST(escape)
{
	Kv *o = kv_obj();
	kv_obj_put(o, "q", kv_str("a\"b\\c"));
	kv_obj_put(o, "n", kv_str("x\ny"));
	char *s = kv_to_json(o);
	ASSERT_STR_EQ("{\"q\":\"a\\\"b\\\\c\",\"n\":\"x\\ny\"}", s, "escape");
	vs_free(s);
}

TEST(control_hex)
{
	char  raw[2] = {1, 0};
	Kv	 *o		 = kv_str(raw);
	char *s		 = kv_to_json(o);
	ASSERT_STR_EQ("\"\\u0001\"", s, "control as hex");
	vs_free(s);
}

TEST(nested)
{
	Kv *arr = kv_arr();
	kv_arr_add(arr, kv_int(1));
	kv_arr_add(arr, kv_int(2));
	Kv *inner = kv_obj();
	kv_obj_put(inner, "k", kv_str("v"));
	kv_arr_add(arr, inner);
	Kv *root = kv_obj();
	kv_obj_put(root, "a", arr);
	char *s = kv_to_json(root);
	ASSERT_STR_EQ("{\"a\":[1,2,{\"k\":\"v\"}]}", s, "nested");
	vs_free(s);
}

TEST(metric)
{
	Kv	 *m = kv_metric("l2/avx2", "vec/s", 1000.0);
	char *s = kv_to_json(m);
	ASSERT_STR_EQ(
			"{\"name\":\"l2/avx2\",\"unit\":\"vec/s\",\"value\":1000}",
			s,
			"metric helper");
	vs_free(s);
}

TEST(arr_add_metric_fmt)
{
	Kv *arr = kv_arr();
	Kv *m	= kv_arr_add_metric(arr, "ns", 1.5, "%s/batch_seq", "avx2");
	ASSERT_NOT_NULL(m, "returns the metric");
	char *s = kv_to_json(arr);
	ASSERT_STR_EQ(
			"[{\"name\":\"avx2/batch_seq\",\"unit\":\"ns\",\"value\":1.5}]",
			s,
			"formatted name");
	vs_free(s);
	ASSERT_NULL(kv_arr_add_metric(NULL, "ns", 1.0, "x"), "NULL arr is no-op");
}

TEST(nonfinite_double_is_null)
{
	Kv	 *o = kv_dbl(INFINITY);
	char *s = kv_to_json(o);
	ASSERT_STR_EQ("null", s, "Inf becomes null");
	vs_free(s);
}

TEST(take_path_space)
{
	char *argv[]	 = {"distance", "--dim", "8", "--json", "out.json", NULL};
	int	  argc		 = 5;
	const char *path = NULL;
	ASSERT_EQ(0, json_take_path(&argc, argv, &path), "take --json");
	ASSERT_STR_EQ("out.json", path, "path");
	ASSERT_EQ(3, argc, "argc");
	ASSERT_STR_EQ("distance", argv[0], "argv0");
	ASSERT_STR_EQ("--dim", argv[1], "argv1");
	ASSERT_STR_EQ("8", argv[2], "argv2");
}

TEST(take_path_equals)
{
	char	   *argv[] = {"distance", "--json=foo.json", "--runs", "3", NULL};
	int			argc   = 4;
	const char *path   = NULL;
	ASSERT_EQ(0, json_take_path(&argc, argv, &path), "take --json=");
	ASSERT_STR_EQ("foo.json", path, "path");
	ASSERT_EQ(3, argc, "argc");
	ASSERT_STR_EQ("--runs", argv[1], "shifted");
}

TEST(take_path_stdout_bare)
{
	char	   *argv[] = {"distance", "--json", NULL};
	int			argc   = 2;
	const char *path   = NULL;
	ASSERT_EQ(0, json_take_path(&argc, argv, &path), "bare --json");
	ASSERT_STR_EQ("-", path, "stdout");
	ASSERT_EQ(1, argc, "argc");
}

TEST(take_path_stdout_before_flag)
{
	char	   *argv[] = {"distance", "--json", "--dim", "8", NULL};
	int			argc   = 4;
	const char *path   = NULL;
	ASSERT_EQ(0, json_take_path(&argc, argv, &path), "--json then flag");
	ASSERT_STR_EQ("-", path, "stdout");
	ASSERT_EQ(3, argc, "argc");
	ASSERT_STR_EQ("--dim", argv[1], "flag kept");
}

TEST(take_quiet)
{
	char *argv[] = {"distance", "--quiet", "--dim", "8", NULL};
	int	  argc	 = 4;
	bool  quiet	 = false;
	json_take_quiet(&argc, argv, &quiet);
	ASSERT_TRUE(quiet, "quiet");
	ASSERT_EQ(3, argc, "argc");
	ASSERT_STR_EQ("--dim", argv[1], "flag kept");
}

TEST(take_quiet_ignores_dash_q)
{
	char *argv[] = {"search", "-q", "10", NULL};
	int	  argc	 = 3;
	bool  quiet	 = false;
	json_take_quiet(&argc, argv, &quiet);
	ASSERT_FALSE(quiet, "-q is search queries, not quiet");
	ASSERT_EQ(3, argc, "argc");
	ASSERT_STR_EQ("-q", argv[1], "left in place");
}

TEST(take_path_dash)
{
	char	   *argv[] = {"distance", "--json", "-", NULL};
	int			argc   = 3;
	const char *path   = NULL;
	ASSERT_EQ(0, json_take_path(&argc, argv, &path), "--json -");
	ASSERT_STR_EQ("-", path, "stdout");
	ASSERT_EQ(1, argc, "argc");
}

TEST(write_path_dash)
{
	/* Smoke: stringify path "-" would hit stdout; use a temp file. */
	Kv *o = kv_obj();
	kv_obj_put(o, "ok", kv_bool(true));
	char tmp[] = "/tmp/vs-json-test-XXXXXX";
	int	 fd	   = mkstemp(tmp);
	ASSERT_TRUE(fd >= 0, "mkstemp");
	close(fd);
	ASSERT_EQ(0, kv_write_json(o, tmp), "write");
	FILE *f = fopen(tmp, "r");
	ASSERT_NOT_NULL(f, "reopen");
	char got[64] = {0};
	ASSERT_NOT_NULL(fgets(got, sizeof(got), f), "read");
	fclose(f);
	unlink(tmp);
	ASSERT_STR_EQ("{\"ok\":true}\n", got, "file contents plus newline");
}
