/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * test_vs_memory.c - Arena allocator tests
 */

#include <stdint.h>
#include <string.h>

#include "core/memory.h"
#include "vs_test.h"

TEST_GROUP(Memory);

/* Shared test context for tests that need one */
static VsMemCtx test_ctx = NULL;

static void
memory_test_setup(void)
{
	test_ctx = vs_memctx_create(NULL, "test");
	vs_memctx_switch(test_ctx);
}

static void
memory_test_teardown(void)
{
	vs_memctx_switch(NULL);
	vs_memctx_delete(test_ctx);
	test_ctx = NULL;
}

TEST_GROUP_FIXTURE(memory_test_setup, memory_test_teardown);

TEST(context_create_delete)
{
	VsMemCtx ctx = vs_memctx_create(NULL, "test_context");
	ASSERT_NOT_NULL(ctx, "context creation should succeed");
	ASSERT_EQ(
			0,
			vs_memctx_total_allocated(ctx),
			"new context should have zero allocations");
	vs_memctx_delete(ctx);
}

TEST(alloc_and_free)
{
	void *ptr = vs_alloc(64);
	ASSERT_NOT_NULL(ptr, "allocation should succeed");

	/* In arena mode, free is a no-op but should not crash */
	vs_free(ptr);
}

TEST(alloc0_zeroes_memory)
{
	size_t		   size = 64;
	unsigned char *ptr	= vs_alloc0(size);
	ASSERT_NOT_NULL(ptr, "allocation should succeed");

	/* Verify memory is zeroed */
	for (size_t i = 0; i < size; i++)
		ASSERT_EQ(0, ptr[i], "memory should be zeroed");
}

TEST(memctx_alloc)
{
	VsMemCtx ctx = vs_memctx_create(NULL, "test");
	ASSERT_NOT_NULL(ctx, "context creation should succeed");

	void *ptr = vs_memctx_alloc(ctx, 128);
	ASSERT_NOT_NULL(ptr, "allocation should succeed");
	ASSERT_TRUE(
			vs_memctx_total_allocated(ctx) >= 128,
			"total allocated should include our allocation");

	void *ptr2 = vs_memctx_alloc0(ctx, 64);
	ASSERT_NOT_NULL(ptr2, "zero allocation should succeed");

	/* Verify zeroed */
	unsigned char *bytes = ptr2;
	for (size_t i = 0; i < 64; i++)
		ASSERT_EQ(0, bytes[i], "memory should be zeroed");

	vs_memctx_delete(ctx);
}

TEST(context_reset)
{
	VsMemCtx ctx = vs_memctx_create(NULL, "test");

	/* Allocate some memory */
	vs_memctx_alloc(ctx, 1024);
	vs_memctx_alloc(ctx, 2048);
	ASSERT_TRUE(vs_memctx_total_allocated(ctx) > 0, "should have allocations");

	/* Reset should clear allocations but keep context */
	vs_memctx_reset(ctx);
	ASSERT_EQ(
			0,
			vs_memctx_total_allocated(ctx),
			"reset should clear allocations");

	/* Should be able to allocate again */
	void *ptr = vs_memctx_alloc(ctx, 512);
	ASSERT_NOT_NULL(ptr, "allocation after reset should succeed");

	vs_memctx_delete(ctx);
}

/* Callback counter for hierarchy test */
static int delete_callback_count = 0;

static void
delete_callback(void *arg)
{
	int *counter = arg;
	(*counter)++;
}

TEST(context_hierarchy)
{
	delete_callback_count = 0;

	VsMemCtx parent = vs_memctx_create(NULL, "parent");
	ASSERT_NOT_NULL(parent, "parent creation should succeed");

	VsMemCtx child1 = vs_memctx_create(parent, "child1");
	VsMemCtx child2 = vs_memctx_create(parent, "child2");
	ASSERT_NOT_NULL(child1, "child1 creation should succeed");
	ASSERT_NOT_NULL(child2, "child2 creation should succeed");

	/* Register callbacks on each context */
	VsMemCtxCallback *cb_parent =
			vs_memctx_alloc(parent, sizeof(VsMemCtxCallback));
	VsMemCtxCallback *cb_child1 =
			vs_memctx_alloc(child1, sizeof(VsMemCtxCallback));
	VsMemCtxCallback *cb_child2 =
			vs_memctx_alloc(child2, sizeof(VsMemCtxCallback));

	vs_memctx_register_reset_callback(
			parent, cb_parent, delete_callback, &delete_callback_count);
	vs_memctx_register_reset_callback(
			child1, cb_child1, delete_callback, &delete_callback_count);
	vs_memctx_register_reset_callback(
			child2, cb_child2, delete_callback, &delete_callback_count);

	ASSERT_EQ(0, delete_callback_count, "no callbacks should have fired yet");

	/* Delete parent should delete children too */
	vs_memctx_delete(parent);

	/* All three callbacks should have been invoked */
	ASSERT_EQ(3, delete_callback_count, "all contexts should be deleted");
}

TEST(context_switch)
{
	/* Start fresh - save fixture context and switch to NULL */
	VsMemCtx saved = vs_memctx_switch(NULL);

	VsMemCtx ctx1 = vs_memctx_create(NULL, "ctx1");
	VsMemCtx ctx2 = vs_memctx_create(NULL, "ctx2");

	VsMemCtx old = vs_memctx_switch(ctx1);
	ASSERT_NULL(old, "context should be NULL after explicit switch");

	old = vs_memctx_switch(ctx2);
	ASSERT_PTR_EQ(ctx1, old, "previous context should be ctx1");

	old = vs_memctx_switch(NULL);
	ASSERT_PTR_EQ(ctx2, old, "previous context should be ctx2");

	vs_memctx_delete(ctx1);
	vs_memctx_delete(ctx2);

	/* Restore fixture context for teardown */
	vs_memctx_switch(saved);
}

TEST(alloc_aligned_basic)
{
	/* 64-byte alignment (typical cache line) */
	size_t alignment = 64;
	void  *ptr		 = vs_alloc_aligned(128, alignment);
	ASSERT_NOT_NULL(ptr, "aligned allocation should succeed");

	/* Verify alignment */
	uintptr_t addr = (uintptr_t)ptr;
	ASSERT_EQ(0, addr % alignment, "address should be aligned");

	/* free_aligned is no-op in arena mode */
	vs_free_aligned(ptr);
}

TEST(alloc_aligned_simd)
{
	/* 32-byte alignment (AVX) */
	size_t alignment = 32;
	void  *ptr		 = vs_alloc_aligned(256, alignment);
	ASSERT_NOT_NULL(ptr, "aligned allocation should succeed");

	uintptr_t addr = (uintptr_t)ptr;
	ASSERT_EQ(0, addr % alignment, "address should be 32-byte aligned");

	vs_free_aligned(ptr);
}

TEST(large_allocation)
{
	/* Allocate larger than default block size */
	size_t large_size = VS_ARENA_BLOCK_SIZE * 2;
	void  *ptr		  = vs_alloc(large_size);
	ASSERT_NOT_NULL(ptr, "large allocation should succeed");

	/* Write to it to ensure it's usable */
	memset(ptr, 0xAB, large_size);
}

TEST(many_small_allocations)
{
	/* Many small allocations should work efficiently */
	for (int i = 0; i < 10000; i++)
	{
		void *ptr = vs_alloc(64);
		ASSERT_NOT_NULL(ptr, "small allocation should succeed");
	}
}

TEST(scoped_context)
{
	/* Test VS_MEMCTX_SCOPE macro */
	VsMemCtx outer = vs_memctx_create(NULL, "outer");
	vs_memctx_switch(outer);

	{
		VS_MEMCTX_SCOPE(inner);
		VsMemCtx old = vs_memctx_switch(inner);

		void *ptr = vs_alloc(128);
		ASSERT_NOT_NULL(ptr, "allocation in scoped context should succeed");

		vs_memctx_switch(old);
		/* inner automatically deleted at end of scope */
	}

	vs_memctx_switch(NULL);
	vs_memctx_delete(outer);
}

TEST(realloc_null_ptr)
{
	/* realloc with NULL ptr should act like alloc */
	void *ptr = vs_realloc(NULL, 64);
	ASSERT_NOT_NULL(ptr, "realloc(NULL, size) should allocate");
}

TEST(realloc_zero_size)
{
	void *ptr = vs_alloc(64);
	ASSERT_NOT_NULL(ptr, "initial allocation should succeed");

	/* realloc with size 0 returns NULL */
	void *new_ptr = vs_realloc(ptr, 0);
	ASSERT_NULL(new_ptr, "realloc(ptr, 0) should return NULL");
}

TEST(realloc_normal)
{
	void *ptr = vs_alloc(64);
	ASSERT_NOT_NULL(ptr, "initial allocation should succeed");
	memset(ptr, 0xAB, 64);

	void *new_ptr = vs_realloc(ptr, 128);
	ASSERT_NOT_NULL(new_ptr, "realloc should succeed");

	/* Verify old data is preserved */
	uint8_t *bytes	   = new_ptr;
	bool	 preserved = true;
	for (int i = 0; i < 64; i++)
	{
		if (bytes[i] != 0xAB)
		{
			preserved = false;
			break;
		}
	}
	ASSERT_TRUE(preserved, "realloc should preserve old data");
}

TEST(alloc_zero_size)
{
	/* Allocating zero bytes returns NULL */
	void *ptr = vs_memctx_alloc(test_ctx, 0);
	ASSERT_NULL(ptr, "allocating 0 bytes should return NULL");
}

TEST(reset_null_context)
{
	/* Reset on NULL should not crash */
	vs_memctx_reset(NULL);
	ASSERT_TRUE(true, "reset NULL should not crash");
}

TEST(reset_with_children)
{
	VsMemCtx parent = vs_memctx_create(NULL, "parent");

	/* Create children */
	VsMemCtx child1 = vs_memctx_create(parent, "child1");
	VsMemCtx child2 = vs_memctx_create(parent, "child2");
	(void)child1;
	(void)child2;

	/* Allocate in children */
	vs_memctx_alloc(child1, 1024);
	vs_memctx_alloc(child2, 2048);

	/* Reset parent should delete children */
	vs_memctx_reset(parent);

	/* Parent should still be usable */
	void *ptr = vs_memctx_alloc(parent, 512);
	ASSERT_NOT_NULL(ptr, "allocation after reset should succeed");

	vs_memctx_delete(parent);
}

TEST(reset_with_multiple_blocks)
{
	VsMemCtx ctx = vs_memctx_create(NULL, "test");

	/* Allocate enough to trigger multiple blocks */
	for (int i = 0; i < 100; i++)
		vs_memctx_alloc(ctx, 1024);

	ASSERT_TRUE(
			vs_memctx_total_allocated(ctx) > VS_ARENA_BLOCK_SIZE,
			"should have allocated multiple blocks worth");

	/* Reset should free extra blocks but keep first */
	vs_memctx_reset(ctx);
	ASSERT_EQ(
			0,
			vs_memctx_total_allocated(ctx),
			"reset should clear allocations");

	/* Should still work */
	void *ptr = vs_memctx_alloc(ctx, 64);
	ASSERT_NOT_NULL(ptr, "allocation after reset should succeed");

	vs_memctx_delete(ctx);
}

TEST(delete_null_context)
{
	/* Delete on NULL should not crash */
	vs_memctx_delete(NULL);
	ASSERT_TRUE(true, "delete NULL should not crash");
}
