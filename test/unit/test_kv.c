/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * test_kv.c - Kv tree (constructors, vtable, to_string)
 */

#include <string.h>

#include "cli/kv_impl.h"
#include "core/memory.h"
#include "vs_test.h"

TEST_GROUP(Kv);
TEST_MEMCTX_FIXTURE();

TEST(build_open_close)
{
	VsMemCtx parent = vs_memctx_current();
	KvBuild *b		= kv_build_open();
	ASSERT_NOT_NULL(b, "open");
	ASSERT_TRUE(vs_memctx_current() != parent, "switched into build ctx");
	Kv *o = kv_obj();
	kv_obj_put(o, "k", kv_int(1));
	ASSERT_EQ(1, o->obj.n, "tree lives in the build ctx");
	kv_build_close(b);
	ASSERT_PTR_EQ(parent, vs_memctx_current(), "restored");
}

TEST(build_close_null)
{
	kv_build_close(NULL);
	ASSERT_TRUE(true, "close NULL is a no-op");
}

TEST(constructors_set_class)
{
	Kv *n = kv_null();
	Kv *b = kv_bool(true);
	Kv *i = kv_int(3);
	Kv *d = kv_dbl(1.5);
	Kv *s = kv_str("x");
	Kv *a = kv_arr();
	Kv *o = kv_obj();

	ASSERT_PTR_EQ(&kv_class_null, n->cls, "null");
	ASSERT_PTR_EQ(&kv_class_bool, b->cls, "bool");
	ASSERT_PTR_EQ(&kv_class_int, i->cls, "int");
	ASSERT_PTR_EQ(&kv_class_dbl, d->cls, "dbl");
	ASSERT_PTR_EQ(&kv_class_str, s->cls, "str");
	ASSERT_PTR_EQ(&kv_class_arr, a->cls, "arr");
	ASSERT_PTR_EQ(&kv_class_obj, o->cls, "obj");
	ASSERT_TRUE(b->b, "bool value");
	ASSERT_EQ(3, i->i, "int value");
	ASSERT_STR_EQ("x", s->s, "str value");
	ASSERT_EQ(0, a->arr.n, "empty arr");
	ASSERT_EQ(0, o->obj.n, "empty obj");
}

TEST(str_copies)
{
	char buf[] = "ab";
	Kv	*s	   = kv_str(buf);
	buf[0]	   = 'z';
	ASSERT_STR_EQ("ab", s->s, "node has its own copy");
}

TEST(str_null_is_empty)
{
	Kv *s = kv_str(NULL);
	ASSERT_STR_EQ("", s->s, "NULL string");
}

TEST(arr_add)
{
	Kv *a = kv_arr();
	kv_arr_add(a, kv_int(1));
	kv_arr_add(a, kv_int(2));
	kv_arr_add(a, NULL);
	ASSERT_EQ(3, a->arr.n, "len");
	ASSERT_EQ(1, a->arr.v[0]->i, "first");
	ASSERT_EQ(2, a->arr.v[1]->i, "second");
	ASSERT_PTR_EQ(&kv_class_null, a->arr.v[2]->cls, "NULL child is kv_null");
}

TEST(arr_add_ignored_on_obj)
{
	Kv *o = kv_obj();
	kv_arr_add(o, kv_int(1));
	ASSERT_EQ(0, o->obj.n, "obj unchanged");
}

TEST(obj_put)
{
	Kv *o = kv_obj();
	kv_obj_put(o, "k", kv_int(9));
	kv_obj_put(o, "s", kv_str("v"));
	ASSERT_EQ(2, o->obj.n, "len");
	ASSERT_STR_EQ("k", o->obj.k[0], "key0");
	ASSERT_EQ(9, o->obj.v[0]->i, "val0");
	ASSERT_STR_EQ("s", o->obj.k[1], "key1");
	ASSERT_STR_EQ("v", o->obj.v[1]->s, "val1");
}

TEST(obj_put_copies_key)
{
	char key[] = "ab";
	Kv	*o	   = kv_obj();
	kv_obj_put(o, key, kv_int(1));
	key[0] = 'z';
	ASSERT_STR_EQ("ab", o->obj.k[0], "key copied");
}

TEST(obj_put_ignored_on_arr)
{
	Kv *a = kv_arr();
	kv_obj_put(a, "k", kv_int(1));
	ASSERT_EQ(0, a->arr.n, "arr unchanged");
}

TEST(nested_free)
{
	Kv *inner = kv_arr();
	kv_arr_add(inner, kv_str("x"));
	Kv *o = kv_obj();
	kv_obj_put(o, "a", inner);
	ASSERT_EQ(1, o->obj.n, "nested before free");
}

static char *
dump(const Kv *v)
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

TEST(to_string_scalars)
{
	Kv	 *v;
	char *s;

	v = kv_null();
	s = dump(v);
	ASSERT_STR_EQ("null\n", s, "null");
	vs_free(s);

	v = kv_bool(false);
	s = dump(v);
	ASSERT_STR_EQ("false\n", s, "bool");
	vs_free(s);

	v = kv_int(-7);
	s = dump(v);
	ASSERT_STR_EQ("-7\n", s, "int");
	vs_free(s);

	v = kv_str("hi");
	s = dump(v);
	ASSERT_STR_EQ("hi\n", s, "str");
	vs_free(s);
}

TEST(to_string_object)
{
	Kv *o = kv_obj();
	kv_obj_put(o, "bench", kv_str("distance"));
	kv_obj_put(o, "dim", kv_int(8));
	char *s = dump(o);
	ASSERT_STR_EQ("bench: distance\ndim: 8\n", s, "object");
	vs_free(s);
}

TEST(to_string_array)
{
	Kv *a = kv_arr();
	kv_arr_add(a, kv_int(1));
	kv_arr_add(a, kv_int(2));
	char *s = dump(a);
	ASSERT_STR_EQ("- 1\n- 2\n", s, "array");
	vs_free(s);
}

TEST(buf_puts)
{
	Buf b = {0};
	ASSERT_TRUE(kv_buf_puts(&b, "ab"), "puts");
	ASSERT_TRUE(kv_buf_putc(&b, 'c'), "putc");
	ASSERT_TRUE(kv_buf_reserve(&b, 0), "nul");
	b.p[b.n] = '\0';
	ASSERT_STR_EQ("abc", b.p, "contents");
	vs_free(b.p);
}

TEST(arr_add_metric)
{
	Kv *arr = kv_arr();
	Kv *m	= kv_arr_add_metric(arr, "ns", 1.5, "%s/batch_seq", "avx2");
	ASSERT_NOT_NULL(m, "returns metric");
	ASSERT_EQ(1, arr->arr.n, "appended");
	ASSERT_PTR_EQ(m, arr->arr.v[0], "same object");
	ASSERT_STR_EQ("name", m->obj.k[0], "name key");
	ASSERT_STR_EQ("avx2/batch_seq", m->obj.v[0]->s, "formatted name");
	ASSERT_NULL(kv_arr_add_metric(NULL, "ns", 1.0, "x"), "NULL arr");
}
