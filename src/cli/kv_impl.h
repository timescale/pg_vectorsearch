/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * kv_impl.h - Kv internals shared with encoders
 */

#ifndef VS_CLI_KV_IMPL_H
#define VS_CLI_KV_IMPL_H

#include <stddef.h>

#include "kv.h"

typedef struct Buf
{
	char  *p;
	size_t n;
	size_t cap;
} Buf;

typedef struct KvClass KvClass;

struct Kv
{
	const KvClass *cls;

	union
	{
		bool	b;
		int64_t i;
		double	d;
		char   *s;

		struct
		{
			Kv	 **v;
			size_t n;
			size_t cap;
		} arr;

		struct
		{
			char **k;
			Kv	 **v;
			size_t n;
			size_t cap;
		} obj;
	};
};

struct KvClass
{
	bool (*to_string)(Buf *b, const Kv *v, int indent);
};

extern const KvClass kv_class_null;
extern const KvClass kv_class_bool;
extern const KvClass kv_class_int;
extern const KvClass kv_class_dbl;
extern const KvClass kv_class_str;
extern const KvClass kv_class_arr;
extern const KvClass kv_class_obj;

bool kv_buf_reserve(Buf *b, size_t extra);
bool kv_buf_putc(Buf *b, char c);
bool kv_buf_puts(Buf *b, const char *s);
bool kv_to_string(Buf *b, const Kv *v, int indent);

/* Saved stdout while --json '-' redirects human chatter. */
extern int kv_saved_stdout;

#endif /* VS_CLI_KV_IMPL_H */
