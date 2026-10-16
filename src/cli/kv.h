/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * kv.h - Write-only key-value tree
 *
 * kv_obj_put / kv_arr_add take ownership of the child.
 */

#ifndef VS_CLI_KV_H
#define VS_CLI_KV_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct Kv	   Kv;
typedef struct KvBuild KvBuild;

/*
 * Scratch arena for a tree. kv_* constructors allocate from it while
 * the build is open. kv_build_close drops the whole tree.
 */
KvBuild *kv_build_open(void);
void	 kv_build_close(KvBuild *b);

Kv *kv_null(void);
Kv *kv_bool(bool v);
Kv *kv_int(int64_t v);
Kv *kv_dbl(double v);
Kv *kv_str(const char *v);
Kv *kv_arr(void);
Kv *kv_obj(void);

void kv_arr_add(Kv *arr, Kv *v);
void kv_obj_put(Kv *obj, const char *key, Kv *v);

Kv *kv_metric(const char *name, const char *unit, double value);

Kv *kv_arr_add_metric(
		Kv *arr, const char *unit, double value, const char *name_fmt, ...)
		__attribute__((format(printf, 4, 5)));

/* Attach run count and stddev (stddev only when n > 1). */
void kv_metric_stats(Kv *m, double stddev, int64_t n);

/* Walkers. */
bool		kv_is_arr(const Kv *v);
bool		kv_is_obj(const Kv *v);
bool		kv_is_num(const Kv *v);
size_t		kv_count(const Kv *v);
const Kv   *kv_at(const Kv *arr, size_t i);
const char *kv_key(const Kv *obj, size_t i);
const Kv   *kv_val(const Kv *obj, size_t i);
const Kv   *kv_get(const Kv *obj, const char *key);
double		kv_num(const Kv *v);
const char *kv_as_str(const Kv *v);

/* One pass over a metric object. Missing keys stay 0 / NULL. */
typedef struct KvMetric
{
	const char *name;
	const char *unit;
	double		value;
	double		stddev;
	double		n;
	bool		has_value;
} KvMetric;

void kv_metric_load(const Kv *m, KvMetric *out);

/* Encoders. Caller vs_free's the string. */
char *kv_to_json(const Kv *v);
char *kv_to_text(const Kv *v);

/* JSON encoder to a file. path "-" is stdout. */
int kv_write_json(const Kv *v, const char *path);

/* JSON decoder. */
Kv *kv_from_json(const char *s);
Kv *kv_from_json_file(const char *path);

/*
 * --json [path] / --json=path. argv[0] stays (getopt program name).
 * Missing path, "-", and empty --json= mean stdout.
 */
int json_take_path(int *argc, char **argv, const char **out);

void json_take_quiet(int *argc, char **argv, bool *quiet);

int	 json_begin_human(const char *json_path, bool quiet);
void json_end_human(void);

#endif /* VS_CLI_KV_H */
