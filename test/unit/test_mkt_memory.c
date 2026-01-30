/*
 * test_mkt_memory.c - Memory allocation tests
 */

#include <stdint.h>
#include <string.h>

#include "mkt_memory.h"
#include "mkt_test.h"

TEST_GROUP(Memory);

TEST(alloc_and_free)
{
	void *ptr = mkt_alloc(64);
	ASSERT_NOT_NULL(ptr, "allocation should succeed");
	mkt_free(ptr);
}

TEST(alloc0_zeroes_memory)
{
	size_t		   size = 64;
	unsigned char *ptr	= mkt_alloc0(size);
	ASSERT_NOT_NULL(ptr, "allocation should succeed");

	/* Verify memory is zeroed */
	for (size_t i = 0; i < size; i++)
		ASSERT_EQ(0, ptr[i], "memory should be zeroed");

	mkt_free(ptr);
}

TEST(realloc_grows_memory)
{
	void *ptr = mkt_alloc(32);
	ASSERT_NOT_NULL(ptr, "initial allocation should succeed");

	/* Write pattern to original memory */
	memset(ptr, 0xAB, 32);

	/* Grow the allocation */
	void *new_ptr = mkt_realloc(ptr, 64);
	ASSERT_NOT_NULL(new_ptr, "realloc should succeed");

	/* Original data should be preserved */
	unsigned char *bytes = new_ptr;
	for (size_t i = 0; i < 32; i++)
		ASSERT_EQ(0xAB, bytes[i], "original data should be preserved");

	mkt_free(new_ptr);
}

TEST(alloc_aligned_basic)
{
	/* 64-byte alignment (typical cache line) */
	size_t alignment = 64;
	void  *ptr		 = mkt_alloc_aligned(128, alignment);
	ASSERT_NOT_NULL(ptr, "aligned allocation should succeed");

	/* Verify alignment */
	uintptr_t addr = (uintptr_t)ptr;
	ASSERT_EQ(0, addr % alignment, "address should be aligned");

	mkt_free_aligned(ptr);
}

TEST(alloc_aligned_simd)
{
	/* 32-byte alignment (AVX) */
	size_t alignment = 32;
	void  *ptr		 = mkt_alloc_aligned(256, alignment);
	ASSERT_NOT_NULL(ptr, "aligned allocation should succeed");

	uintptr_t addr = (uintptr_t)ptr;
	ASSERT_EQ(0, addr % alignment, "address should be 32-byte aligned");

	mkt_free_aligned(ptr);
}
