/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * kv_text.c - Text encoder
 */

#include <string.h>

#include "kv_impl.h"

char *
kv_to_text(const Kv *v)
{
	Buf b = {0};
	if (!kv_to_string(&b, v, 0) || !kv_buf_reserve(&b, 0))
		return NULL;
	b.p[b.n] = '\0';
	if (b.n > 0 && b.p[0] == '\n')
	{
		memmove(b.p, b.p + 1, b.n);
		b.n--;
	}
	return b.p;
}
