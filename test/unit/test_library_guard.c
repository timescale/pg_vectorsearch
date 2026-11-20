/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * test_library_guard.c - finding the extension library among loaded ones
 */

#include "pg/library_guard.h"
#include "vs_test.h"

#define LIB VS_MODULE_NAME

/* A fake module list. A handle points at an entry of a NULL-ended array. */
static const char **fake_paths;

static void *
fake_first(void)
{
	return *fake_paths ? fake_paths : NULL;
}

static void *
fake_next(void *module)
{
	const char **next = (const char **)module + 1;

	return *next ? next : NULL;
}

static const char *
fake_path(void *module)
{
	return *(const char **)module;
}

static const VsModuleList fake_modules = {
		.first = fake_first,
		.next  = fake_next,
		.path  = fake_path,
};

static const char *
find_in(const char **paths)
{
	const char *file;

	fake_paths = paths;
	file	   = vs_find_extension_library(&fake_modules);
	fake_paths = NULL;

	return file;
}

TEST_GROUP(LibraryGuard);

TEST(matches_unversioned_library)
{
	ASSERT_STR_EQ(
			LIB ".so",
			vs_extension_library_file("/usr/lib/" LIB ".so"),
			"unversioned library");
	ASSERT_STR_EQ(
			LIB ".dylib",
			vs_extension_library_file(LIB ".dylib"),
			"no directory, other suffix");
}

TEST(matches_versioned_library)
{
	ASSERT_STR_EQ(
			LIB "-0.1.0.so",
			vs_extension_library_file("/usr/lib/" LIB "-0.1.0.so"),
			"version-named library");
}

TEST(ignores_other_libraries)
{
	ASSERT_NULL(
			vs_extension_library_file("/usr/lib/" LIB "_helpers.so"),
			"sibling with the same prefix");
	ASSERT_NULL(
			vs_extension_library_file("/usr/lib/" LIB),
			"name without a suffix");
	ASSERT_NULL(
			vs_extension_library_file("/usr/lib/" LIB "-helper.so"),
			"sibling named like a version");
	ASSERT_NULL(
			vs_extension_library_file("/usr/lib/" LIB "-0"),
			"version without a suffix");
	ASSERT_NULL(
			vs_extension_library_file("/usr/lib/" LIB "."), "empty suffix");
	ASSERT_NULL(
			vs_extension_library_file("/usr/lib/plpgsql.so"),
			"unrelated library");
	ASSERT_NULL(
			vs_extension_library_file("/opt/" LIB "-0.1.0/vector.so"),
			"name in a directory only");
	ASSERT_NULL(vs_extension_library_file(""), "empty path");
}

TEST(finds_nothing_in_empty_list)
{
	const char *paths[] = {NULL};

	ASSERT_NULL(find_in(paths), "empty list");
}

TEST(finds_nothing_among_other_libraries)
{
	const char *paths[] =
			{"/usr/lib/plpgsql.so",
			 "/usr/lib/" LIB "_helpers.so",
			 "/usr/lib/vector.so",
			 NULL};

	ASSERT_NULL(find_in(paths), "no extension library loaded");
}

TEST(finds_library_at_any_position)
{
	const char *first[] =
			{"/usr/lib/" LIB "-0.1.0.so", "/usr/lib/plpgsql.so", NULL};
	const char *middle[] =
			{"/usr/lib/plpgsql.so",
			 "/usr/lib/" LIB ".so",
			 "/usr/lib/vector.so",
			 NULL};
	const char *last[] =
			{"/usr/lib/plpgsql.so",
			 "/usr/lib/vector.so",
			 "/usr/lib/" LIB "-0.1.0.so",
			 NULL};

	ASSERT_STR_EQ(LIB "-0.1.0.so", find_in(first), "first entry");
	ASSERT_STR_EQ(LIB ".so", find_in(middle), "middle entry");
	ASSERT_STR_EQ(LIB "-0.1.0.so", find_in(last), "last entry");
}

TEST(reports_first_of_two_copies)
{
	const char *paths[] =
			{"/usr/lib/" LIB "-0.1.0.so", "/usr/lib/" LIB ".so", NULL};

	ASSERT_STR_EQ(LIB "-0.1.0.so", find_in(paths), "earliest loaded copy");
}
