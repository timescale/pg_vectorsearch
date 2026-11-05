/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * test_kv_json.c - JSON encoder and decoder
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "cli/kv.h"
#include "core/memory.h"
#include "vs_test.h"

/* Write len bytes to a fresh temp file; path is empty on failure. */
static void
write_tmp(char *path, size_t sz, const char *data, size_t len)
{
	const char *dir = getenv("TMPDIR");
	FILE	   *f;
	int			fd;

	snprintf(path, sz, "%s/vs-json-XXXXXX", dir != NULL ? dir : "/tmp");
	fd = mkstemp(path);
	if (fd < 0)
	{
		path[0] = '\0';
		return;
	}
	f = fdopen(fd, "wb");
	fwrite(data, 1, len, f);
	fclose(f);
}

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

TEST(from_json_roundtrip)
{
	Kv *o = kv_obj();
	kv_obj_put(o, "bench", kv_str("distance"));
	kv_obj_put(o, "dim", kv_int(8));
	Kv *m = kv_arr();
	kv_arr_add_metric(m, "vec/s", 1e7, "l2/compiler");
	kv_obj_put(o, "metrics", m);
	char *s	 = kv_to_json(o);
	Kv	 *p	 = kv_from_json(s);
	char *s2 = kv_to_json(p);
	ASSERT_NOT_NULL(p, "parse");
	ASSERT_STR_EQ(s, s2, "roundtrip");
	vs_free(s);
	vs_free(s2);
}

TEST(from_json_ws)
{
	Kv *p = kv_from_json(" { \"a\" : 1 } ");
	ASSERT_TRUE(kv_is_obj(p), "obj");
	ASSERT_TRUE(kv_is_num(kv_get(p, "a")), "num");
	ASSERT_EQ(1.0, kv_num(kv_get(p, "a")), "a");
}

TEST(from_json_bad)
{
	static const char *const bad[] = {
			"",
			"{",
			"}",
			"[",
			"]",
			"{]",
			"[}",
			"{\"a\"}",
			"{\"a\":}",
			"{\"a\":1,}",
			"[1,]",
			"{\"a\":1",
			"[1",
			"\"unterminated",
			"tru",
			"nul",
			"fals",
			"1 2",
			"{a:1}",
			"{\"a\":1} junk",
			",",
			":",
			"nan",
			"inf",
			"-inf",
			"1e999",
			"+1",
			"01",
			".5",
			"1.",
			"0x10",
			"1e",
			"1e+",
			"-",
			"\"a\nb\"",
			"\"a\tb\"",
			/* Ambiguous: readers disagree on which key wins. */
			"{\"a\":1,\"a\":2}",
			"{\"a\":null,\"a\":1}",
	};

	ASSERT_NULL(kv_from_json(NULL), "NULL");
	for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++)
		ASSERT_NULL(kv_from_json(bad[i]), bad[i]);
	ASSERT_NULL(kv_from_json_file("/no/such/vs-json"), "missing file");
}

TEST(from_json_escapes)
{
	/* Exactly the encoder's escape set round-trips: the shorthand
	 * pairs plus \u00XX for other control characters. */
	const char *raw = "a\"b\\c\bd\fe\nf\rg\th\x01i";
	Kv		   *s	= kv_str(raw);
	Kv		   *p	= kv_from_json(kv_to_json(s));

	ASSERT_STR_EQ(raw, kv_as_str(p), "string round trip");
	ASSERT_STR_EQ("\x1f", kv_as_str(kv_from_json("\"\\u001f\"")), "hex");
}

TEST(from_json_bad_escapes)
{
	/* Everything outside the encoder's escape set stays an error. */
	ASSERT_NULL(kv_from_json("\"\\q\""), "unknown escape");
	ASSERT_NULL(kv_from_json("\"\\u0041\""), "printable hex escape");
	ASSERT_NULL(kv_from_json("\"\\u0000\""), "NUL escape");
	ASSERT_NULL(kv_from_json("\"\\u001\""), "short hex escape");
	ASSERT_NULL(kv_from_json("\"\\u1001\""), "wide hex escape");
	ASSERT_NULL(kv_from_json("\"\\\""), "trailing backslash");
}

TEST(from_json_named)
{
	Kv *p = kv_from_json(
			"{\"metrics\":[{\"name\":\"x\",\"unit\":\"vec/s\",\"value\":2}]}");
	const Kv   *m	 = kv_get(p, "metrics");
	const Kv   *e	 = kv_at(m, 0);
	const char *name = kv_as_str(kv_get(e, "name"));
	ASSERT_STR_EQ("x", name, "name");
	ASSERT_EQ(2.0, kv_num(kv_get(e, "value")), "value");
}

TEST(metric_load_reads_stats)
{
	Kv *m = kv_from_json(
			"{\"name\":\"qps\",\"unit\":\"1/s\",\"value\":100,"
			"\"stddev\":2,\"n\":8}");
	KvMetric f;

	kv_metric_load(m, &f);
	ASSERT_STR_EQ("qps", f.name, "name");
	ASSERT_STR_EQ("1/s", f.unit, "unit");
	ASSERT_TRUE(f.has_value, "has value");
	ASSERT_EQ(100.0, f.value, "value");
	ASSERT_EQ(2.0, f.stddev, "stddev");
	ASSERT_EQ(8.0, f.n, "n");
}

TEST(from_json_int_precision)
{
	/* Integral tokens stay integers: a double would round these. */
	Kv *p = kv_from_json("9007199254740993");
	ASSERT_STR_EQ("9007199254740993", kv_to_json(p), "2^53 + 1");
	p = kv_from_json("-9007199254740993");
	ASSERT_STR_EQ("-9007199254740993", kv_to_json(p), "negative");
	p = kv_from_json("9223372036854775807");
	ASSERT_STR_EQ("9223372036854775807", kv_to_json(p), "INT64_MAX");
	/* Wider than int64 spills to double rather than failing. */
	ASSERT_NOT_NULL(kv_from_json("9223372036854775808"), "overflow");
}

TEST(from_json_number_grammar)
{
	/* Spellings the encoder produces parse... */
	ASSERT_NOT_NULL(kv_from_json("-0.5"), "fraction");
	ASSERT_NOT_NULL(kv_from_json("1e+300"), "signed exponent");
	ASSERT_NOT_NULL(kv_from_json("0"), "zero");
	/* ...and 17 significant digits survive the round trip. */
	Kv *p = kv_from_json("2.9802322387695312e-08");
	ASSERT_EQ(2.9802322387695312e-08, kv_num(p), "value");
}

TEST(from_json_depth_capped)
{
	/* 80 nested arrays exceed the parser's depth bound; 8 do not. */
	char deep[163];
	for (int i = 0; i < 80; i++)
	{
		deep[i]		 = '[';
		deep[80 + i] = ']';
	}
	deep[160] = '\0';
	ASSERT_NULL(kv_from_json(deep), "too deep");
	ASSERT_NOT_NULL(kv_from_json("[[[[[[[[1]]]]]]]]"), "sane depth");
}

TEST(from_json_shallow_siblings)
{
	/* Siblings only two levels deep: 80 empty arrays and objects. */
	char wide[1 + 80 * 6 + 2];
	int	 w = 0;

	wide[w++] = '[';
	for (int i = 0; i < 80; i++)
	{
		const char *elem = (i % 2 == 0) ? "[]," : "{},";
		memcpy(wide + w, elem, 3);
		w += 3;
	}
	wide[w - 1] = ']';
	wide[w]		= '\0';

	Kv *p = kv_from_json(wide);
	ASSERT_NOT_NULL(p, "width is not depth");
	ASSERT_EQ(80, kv_count(p), "all elements");
}

TEST(from_json_file_roundtrip)
{
	const char *json = "{\"a\":1}";
	char		path[128];

	write_tmp(path, sizeof(path), json, strlen(json));
	ASSERT_TRUE(path[0] != '\0', "temp file");
	Kv *p = kv_from_json_file(path);
	unlink(path);
	ASSERT_NOT_NULL(p, "parsed");
	ASSERT_EQ(1.0, kv_num(kv_get(p, "a")), "value");
}

TEST(from_json_file_embedded_nul)
{
	/* A NUL would end the parse early and hide what follows. */
	const char data[] = "{\"a\":1}\0garbage";
	char	   path[128];

	write_tmp(path, sizeof(path), data, sizeof(data) - 1);
	ASSERT_TRUE(path[0] != '\0', "temp file");
	Kv *p = kv_from_json_file(path);
	unlink(path);
	ASSERT_NULL(p, "NUL rejected");
}

TEST(from_json_file_too_large)
{
	/* One byte past the loader's 1 MiB cap. The capped prefix is a
	 * number padded with spaces, so it would parse if the loader
	 * silently truncated instead of rejecting the file. */
	size_t cap = 1 << 20;
	char  *big = vs_alloc(cap);
	char   path[128];

	memset(big, ' ', cap);
	big[0]		 = '1';
	big[cap - 1] = 'x';
	write_tmp(path, sizeof(path), big, cap);
	ASSERT_TRUE(path[0] != '\0', "temp file");
	Kv *p = kv_from_json_file(path);
	unlink(path);
	vs_free(big);
	ASSERT_NULL(p, "oversized file rejected");
}
