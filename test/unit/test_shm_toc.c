/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * test_shm_toc.c - Keyed shared-region table (standalone shim) tests
 *
 * Covers the shm_toc API the parallel build relies on: estimator sizing,
 * 8-byte-aligned arena allocation within bounds, key insert/lookup round-trip,
 * and the noError path for a missing key.
 */

#include <stdint.h>

#include "standalone/shm_toc.h"
#include "vs_test.h"

TEST_GROUP(ShmToc);

TEST(estimator_sums_chunks)
{
	shm_toc_estimator e;

	shm_toc_initialize_estimator(&e);
	shm_toc_estimate_chunk(&e, 1);	 /* aligns up to 8 */
	shm_toc_estimate_chunk(&e, 100); /* aligns up to 104 */
	shm_toc_estimate_keys(&e, 3);

	ASSERT_EQ(
			112, (int)shm_toc_estimate(&e), "chunk sizes are 8-byte aligned");
}

TEST(allocate_is_aligned_and_distinct)
{
	char	 arena[512];
	shm_toc *toc = shm_toc_create(0xABCD, arena, sizeof(arena));

	char *a = (char *)shm_toc_allocate(toc, 5);
	char *b = (char *)shm_toc_allocate(toc, 16);
	char *c = (char *)shm_toc_allocate(toc, 1);

	ASSERT_TRUE(((uintptr_t)a & 7) == 0, "first allocation is 8-aligned");
	ASSERT_TRUE(((uintptr_t)b & 7) == 0, "second allocation is 8-aligned");
	ASSERT_TRUE(((uintptr_t)c & 7) == 0, "third allocation is 8-aligned");

	/* 5 rounds up to 8, so b is 8 bytes past a; 16 stays 16, so c is 24 past.
	 */
	ASSERT_EQ(8, (int)(b - a), "5-byte chunk consumes an aligned 8 bytes");
	ASSERT_EQ(24, (int)(c - a), "16-byte chunk consumes 16 bytes");

	vs_shm_toc_free(toc);
}

TEST(insert_lookup_roundtrip)
{
	char	 arena[256];
	shm_toc *toc = shm_toc_create(0x1, arena, sizeof(arena));

	int *x = (int *)shm_toc_allocate(toc, sizeof(int));
	int *y = (int *)shm_toc_allocate(toc, sizeof(int));
	*x	   = 42;
	*y	   = 99;

	shm_toc_insert(toc, 1000, x);
	shm_toc_insert(toc, 2000, y);

	ASSERT_EQ(x, shm_toc_lookup(toc, 1000, false), "key 1000 resolves to x");
	ASSERT_EQ(y, shm_toc_lookup(toc, 2000, false), "key 2000 resolves to y");
	ASSERT_EQ(42, *(int *)shm_toc_lookup(toc, 1000, false), "x survives");
	ASSERT_EQ(99, *(int *)shm_toc_lookup(toc, 2000, false), "y survives");

	vs_shm_toc_free(toc);
}

TEST(lookup_missing_noerror_returns_null)
{
	char	 arena[64];
	shm_toc *toc = shm_toc_create(0x1, arena, sizeof(arena));

	shm_toc_insert(toc, 7, shm_toc_allocate(toc, 8));

	ASSERT_NULL(
			shm_toc_lookup(toc, 12345, true),
			"missing key is NULL w/ noError");

	vs_shm_toc_free(toc);
}

/* Force several growths of the key map to exercise the realloc path. */
TEST(many_keys_grow_map)
{
	char	 arena[4096];
	shm_toc *toc = shm_toc_create(0x1, arena, sizeof(arena));

	for (uint64_t k = 0; k < 50; k++)
	{
		uint64_t *slot = (uint64_t *)shm_toc_allocate(toc, sizeof(uint64_t));
		*slot		   = k * 7;
		shm_toc_insert(toc, k, slot);
	}

	for (uint64_t k = 0; k < 50; k++)
	{
		uint64_t *slot = (uint64_t *)shm_toc_lookup(toc, k, false);
		if (*slot != k * 7)
		{
			ASSERT_EQ((int)(k * 7), (int)*slot, "every key round-trips");
			break;
		}
	}

	vs_shm_toc_free(toc);
}
