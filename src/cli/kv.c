/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 */

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "core/memory.h"
#include "kv_impl.h"

struct KvBuild
{
	VsMemCtx ctx;
	VsMemCtx prev;
};

/*
 * Switch to a child arena for Kv allocations.
 */
KvBuild *
kv_build_open(void)
{
	VsMemCtx prev = vs_memctx_current();
	VsMemCtx ctx  = vs_memctx_create(prev, "kv");
	KvBuild *b;

	if (ctx == NULL)
		return NULL;
	b = vs_memctx_alloc(ctx, sizeof(*b));
	if (b == NULL)
	{
		vs_memctx_delete(ctx);
		return NULL;
	}
	b->ctx	= ctx;
	b->prev = prev;
	vs_memctx_switch(ctx);
	return b;
}

/*
 * Delete the build arena and restore the previous context.
 */
void
kv_build_close(KvBuild *b)
{
	VsMemCtx ctx;

	if (b == NULL)
		return;
	ctx = b->ctx;
	vs_memctx_switch(b->prev);
	vs_memctx_delete(ctx);
}

/* Allocate a node and attach its class. */
static Kv *
kv_new(const KvClass *cls)
{
	Kv *v = vs_alloc0(sizeof(*v));
	if (v != NULL)
		v->cls = cls;
	return v;
}

/* Grow a pointer array to at least n elements. */
static bool
grow_ptr(void **p, size_t *cap, size_t n, size_t elem)
{
	if (n <= *cap)
		return true;
	size_t ncap = *cap == 0 ? 8 : *cap;
	while (ncap < n)
		ncap *= 2;
	void *np = vs_realloc(*p, ncap * elem);
	if (np == NULL)
		return false;
	*p	 = np;
	*cap = ncap;
	return true;
}

/* ---- Buf ------------------------------------------------------ */

/* Ensure the buffer can hold extra more bytes. */
bool
kv_buf_reserve(Buf *b, size_t extra)
{
	if (b->n + extra + 1 <= b->cap)
		return true;
	size_t ncap = b->cap == 0 ? 64 : b->cap;
	while (b->n + extra + 1 > ncap)
		ncap *= 2;
	char *np = vs_realloc(b->p, ncap);
	if (np == NULL)
		return false;
	b->p   = np;
	b->cap = ncap;
	return true;
}

/* Append one character. */
bool
kv_buf_putc(Buf *b, char c)
{
	if (!kv_buf_reserve(b, 1))
		return false;
	b->p[b->n++] = c;
	return true;
}

/* Append a C string. */
bool
kv_buf_puts(Buf *b, const char *s)
{
	size_t n = strlen(s);
	if (!kv_buf_reserve(b, n))
		return false;
	memcpy(b->p + b->n, s, n);
	b->n += n;
	return true;
}

/* Write indent levels of two spaces. */
static bool
kv_buf_indent(Buf *b, int indent)
{
	for (int i = 0; i < indent; i++)
	{
		if (!kv_buf_puts(b, "  "))
			return false;
	}
	return true;
}

/* Dispatch to the node's to_string method. */
bool
kv_to_string(Buf *b, const Kv *v, int indent)
{
	if (v == NULL || v->cls == NULL || v->cls->to_string == NULL)
		return kv_buf_puts(b, "null\n");
	return v->cls->to_string(b, v, indent);
}

/* ---- scalars -------------------------------------------------- */

/* Format a null node. */
static bool
to_string_null(Buf *b, const Kv *v, int indent)
{
	(void)v;
	(void)indent;
	return kv_buf_puts(b, "null\n");
}

/* Format a boolean node. */
static bool
to_string_bool(Buf *b, const Kv *v, int indent)
{
	(void)indent;
	return kv_buf_puts(b, v->b ? "true\n" : "false\n");
}

/* Format an integer node. */
static bool
to_string_int(Buf *b, const Kv *v, int indent)
{
	char tmp[64];

	(void)indent;
	snprintf(tmp, sizeof(tmp), "%lld\n", (long long)v->i);
	return kv_buf_puts(b, tmp);
}

/* Format a floating-point node. */
static bool
to_string_dbl(Buf *b, const Kv *v, int indent)
{
	char tmp[64];

	(void)indent;
	if (!isfinite(v->d))
		return kv_buf_puts(b, "null\n");
	snprintf(tmp, sizeof(tmp), "%.17g\n", v->d);
	return kv_buf_puts(b, tmp);
}

/* Format a string node. */
static bool
to_string_str(Buf *b, const Kv *v, int indent)
{
	(void)indent;
	return kv_buf_puts(b, v->s ? v->s : "") && kv_buf_putc(b, '\n');
}

const KvClass kv_class_null = {.to_string = to_string_null};
const KvClass kv_class_bool = {.to_string = to_string_bool};
const KvClass kv_class_int	= {.to_string = to_string_int};
const KvClass kv_class_dbl	= {.to_string = to_string_dbl};
const KvClass kv_class_str	= {.to_string = to_string_str};

/* Return a null node. */
Kv *
kv_null(void)
{
	return kv_new(&kv_class_null);
}

/* Return a boolean node. */
Kv *
kv_bool(bool v)
{
	Kv *n = kv_new(&kv_class_bool);
	if (n != NULL)
		n->b = v;
	return n;
}

/* Return an integer node. */
Kv *
kv_int(int64_t v)
{
	Kv *n = kv_new(&kv_class_int);
	if (n != NULL)
		n->i = v;
	return n;
}

/* Return a floating-point node. */
Kv *
kv_dbl(double v)
{
	Kv *n = kv_new(&kv_class_dbl);
	if (n != NULL)
		n->d = v;
	return n;
}

/* Return a string node; NULL v becomes empty. */
Kv *
kv_str(const char *v)
{
	Kv *n = kv_new(&kv_class_str);
	if (n == NULL)
		return NULL;
	if (v == NULL)
		v = "";
	size_t len = strlen(v);
	n->s	   = vs_alloc(len + 1);
	if (n->s == NULL)
		return NULL;
	memcpy(n->s, v, len + 1);
	return n;
}

/* ---- array / object ------------------------------------------- */

/* Format an array as an indented list. */
static bool
to_string_arr(Buf *b, const Kv *v, int indent)
{
	if (v->arr.n == 0)
		return kv_buf_puts(b, "[]\n");
	if (!kv_buf_putc(b, '\n'))
		return false;
	for (size_t i = 0; i < v->arr.n; i++)
	{
		if (!kv_buf_indent(b, indent) || !kv_buf_puts(b, "- "))
			return false;
		if (!kv_to_string(b, v->arr.v[i], indent + 1))
			return false;
	}
	return true;
}

/* Format an object as indented key: value lines. */
static bool
to_string_obj(Buf *b, const Kv *v, int indent)
{
	if (v->obj.n == 0)
		return kv_buf_puts(b, "{}\n");
	if (!kv_buf_putc(b, '\n'))
		return false;
	for (size_t i = 0; i < v->obj.n; i++)
	{
		if (!kv_buf_indent(b, indent) || !kv_buf_puts(b, v->obj.k[i]) ||
			!kv_buf_puts(b, ": "))
			return false;
		if (!kv_to_string(b, v->obj.v[i], indent + 1))
			return false;
	}
	return true;
}

const KvClass kv_class_arr = {.to_string = to_string_arr};
const KvClass kv_class_obj = {.to_string = to_string_obj};

/* Return an empty array. */
Kv *
kv_arr(void)
{
	return kv_new(&kv_class_arr);
}

/* Return an empty object. */
Kv *
kv_obj(void)
{
	return kv_new(&kv_class_obj);
}

/* Append v to arr; NULL v becomes a null node. */
void
kv_arr_add(Kv *arr, Kv *v)
{
	if (arr == NULL || arr->cls != &kv_class_arr)
		return;
	if (v == NULL)
		v = kv_null();
	if (!grow_ptr(
				(void **)&arr->arr.v,
				&arr->arr.cap,
				arr->arr.n + 1,
				sizeof(Kv *)))
		return;
	arr->arr.v[arr->arr.n++] = v;
}

/* Store key/value in obj, copying the key. */
void
kv_obj_put(Kv *obj, const char *key, Kv *v)
{
	if (obj == NULL || obj->cls != &kv_class_obj || key == NULL)
		return;
	if (v == NULL)
		v = kv_null();

	size_t need = obj->obj.n + 1;
	if (need > obj->obj.cap)
	{
		size_t ncap = obj->obj.cap == 0 ? 8 : obj->obj.cap * 2;
		while (ncap < need)
			ncap *= 2;
		char **nk = vs_realloc(obj->obj.k, ncap * sizeof(char *));
		Kv	 **nv = vs_realloc(obj->obj.v, ncap * sizeof(Kv *));
		if (nk == NULL || nv == NULL)
			return;
		obj->obj.k	 = nk;
		obj->obj.v	 = nv;
		obj->obj.cap = ncap;
	}

	size_t n	= strlen(key);
	char  *kdup = vs_alloc(n + 1);
	if (kdup == NULL)
		return;
	memcpy(kdup, key, n + 1);
	obj->obj.k[obj->obj.n] = kdup;
	obj->obj.v[obj->obj.n] = v;
	obj->obj.n++;
}

/* Object with name, unit, and value fields. */
Kv *
kv_metric(const char *name, const char *unit, double value)
{
	Kv *o = kv_obj();
	kv_obj_put(o, "name", kv_str(name));
	kv_obj_put(o, "unit", kv_str(unit));
	kv_obj_put(o, "value", kv_dbl(value));
	return o;
}

/* Format a metric name and append it to arr. */
Kv *
kv_arr_add_metric(
		Kv *arr, const char *unit, double value, const char *name_fmt, ...)
{
	char	name[128];
	va_list ap;

	if (arr == NULL || name_fmt == NULL)
		return NULL;
	va_start(ap, name_fmt);
	/* NOLINTNEXTLINE(clang-analyzer-valist.Uninitialized) */
	vsnprintf(name, sizeof(name), name_fmt, ap);
	va_end(ap);
	Kv *m = kv_metric(name, unit, value);
	kv_arr_add(arr, m);
	return m;
}

/* Strip --quiet from argv. Not -q: bench search uses -q for query count. */
void
json_take_quiet(int *argc, char **argv, bool *quiet)
{
	int i = 1;
	while (i < *argc)
	{
		const char *a = argv[i];
		if (a == NULL)
			break;
		if (strcmp(a, "--quiet") == 0)
		{
			*quiet	 = true;
			int rest = *argc - i - 1;
			memmove(&argv[i], &argv[i + 1], (size_t)rest * sizeof(char *));
			*argc -= 1;
			argv[*argc] = NULL;
			continue;
		}
		i++;
	}
}
