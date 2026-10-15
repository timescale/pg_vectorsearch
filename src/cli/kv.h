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

/* Encoders. Caller vs_free's the string. */
char *kv_to_json(const Kv *v);
char *kv_to_text(const Kv *v);

/* JSON encoder to a file. path "-" is stdout. */
int kv_write_json(const Kv *v, const char *path);

/*
 * --json [path] / --json=path. argv[0] stays (getopt program name).
 * Missing path, "-", and empty --json= mean stdout.
 */
int json_take_path(int *argc, char **argv, const char **out);

void json_take_quiet(int *argc, char **argv, bool *quiet);

int	 json_begin_human(const char *json_path, bool quiet);
void json_end_human(void);

#endif /* VS_CLI_KV_H */
