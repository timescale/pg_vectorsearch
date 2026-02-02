/*
 * memory_standalone.c - Arena allocator implementation
 *
 * Implements a memory context system compatible with PostgreSQL's semantics
 * but without PostgreSQL dependencies. Used for standalone testing.
 */

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "memory_standalone.h"

/* Thread-local current context */
_Thread_local MktMemCtx mkt_current_memctx = NULL;

/* Align size up to alignment boundary */
static inline size_t
align_up(size_t size, size_t alignment)
{
	return (size + alignment - 1) & ~(alignment - 1);
}

/* Allocate a new block */
static MktArenaBlock *
arena_block_create(size_t min_size)
{
	size_t block_size = min_size > MKT_ARENA_BLOCK_SIZE ? min_size
														: MKT_ARENA_BLOCK_SIZE;
	/* Include space for header, aligned */
	size_t header_size = align_up(sizeof(MktArenaBlock), MKT_ARENA_ALIGNMENT);
	size_t total_size  = header_size + block_size;

	MktArenaBlock *block = aligned_alloc(MKT_ARENA_ALIGNMENT, total_size);
	if (!block)
		return NULL;

	block->next = NULL;
	block->size = block_size;
	block->used = 0;
	return block;
}

/* Free a block */
static void
arena_block_free(MktArenaBlock *block)
{
	free(block);
}

/* Get data pointer for block */
static inline void *
arena_block_data(MktArenaBlock *block)
{
	size_t header_size = align_up(sizeof(MktArenaBlock), MKT_ARENA_ALIGNMENT);
	return (char *)block + header_size;
}

/* Allocate from arena with specified alignment */
static void *
arena_alloc(MktArena *arena, size_t size, size_t alignment)
{
	if (size == 0)
		return NULL;

	size_t aligned_size = align_up(size, alignment);

	/* Try current block first */
	if (arena->current)
	{
		void	 *data		   = arena_block_data(arena->current);
		uintptr_t base		   = (uintptr_t)data + arena->current->used;
		uintptr_t aligned_base = align_up(base, alignment);
		size_t	  padding	   = aligned_base - (uintptr_t)data;

		if (padding + aligned_size <= arena->current->size)
		{
			void *ptr			 = (void *)aligned_base;
			arena->current->used = padding + aligned_size;
			arena->total_allocated += aligned_size;
			return ptr;
		}
	}

	/*
	 * Need a new block. Request extra space for alignment padding.
	 * Worst case: we need (alignment - 1) extra bytes to align.
	 */
	size_t		   extra_for_alignment = alignment > MKT_ARENA_ALIGNMENT
											   ? alignment - MKT_ARENA_ALIGNMENT
											   : 0;
	MktArenaBlock *block			   = arena_block_create(
			  aligned_size + extra_for_alignment);
	if (!block)
		return NULL;

	/* Link new block */
	block->next	   = arena->blocks;
	arena->blocks  = block;
	arena->current = block;

	/* Allocate from new block with proper alignment */
	void	 *data		   = arena_block_data(block);
	uintptr_t base		   = (uintptr_t)data;
	uintptr_t aligned_base = align_up(base, alignment);
	size_t	  padding	   = aligned_base - base;

	block->used = padding + aligned_size;
	arena->total_allocated += aligned_size;
	return (void *)aligned_base;
}

/* Create a new arena context */
MktMemCtx
mkt_memctx_create(MktMemCtx parent, const char *name)
{
	/* Allocate arena struct from parent or malloc */
	MktArena *arena;
	if (parent)
		arena = arena_alloc(parent, sizeof(MktArena), MKT_ARENA_ALIGNMENT);
	else
		arena = aligned_alloc(
				MKT_ARENA_ALIGNMENT,
				align_up(sizeof(MktArena), MKT_ARENA_ALIGNMENT));

	if (!arena)
		return NULL;

	memset(arena, 0, sizeof(MktArena));
	arena->name		  = name;
	arena->parent	  = parent;
	arena->block_size = MKT_ARENA_BLOCK_SIZE;

	/* Link into parent's child list */
	if (parent)
	{
		arena->next_sibling = parent->first_child;
		parent->first_child = arena;
	}

	return arena;
}

/* Invoke all reset callbacks for an arena */
static void
invoke_reset_callbacks(MktArena *arena)
{
	MktMemCtxCallback *cb = arena->reset_callbacks;
	while (cb)
	{
		cb->func(cb->arg);
		cb = cb->next;
	}
}

/* Delete arena and all children */
void
mkt_memctx_delete(MktMemCtx ctx)
{
	if (!ctx)
		return;

	MktArena *arena = ctx;

	/* Invoke callbacks before destroying anything */
	invoke_reset_callbacks(arena);

	/* Recursively delete children first */
	MktArena *child = arena->first_child;
	while (child)
	{
		MktArena *next = child->next_sibling;
		mkt_memctx_delete(child);
		child = next;
	}

	/* Unlink from parent */
	if (arena->parent)
	{
		MktArena **pp = &arena->parent->first_child;
		while (*pp && *pp != arena)
			pp = &(*pp)->next_sibling;
		if (*pp)
			*pp = arena->next_sibling;
	}

	/* Free all blocks */
	MktArenaBlock *block = arena->blocks;
	while (block)
	{
		MktArenaBlock *next = block->next;
		arena_block_free(block);
		block = next;
	}

	/* Free arena struct if it was top-level (no parent) */
	if (!arena->parent)
		free(arena);
}

/* Reset arena - free all allocations but keep arena */
void
mkt_memctx_reset(MktMemCtx ctx)
{
	if (!ctx)
		return;

	MktArena *arena = ctx;

	/* Invoke callbacks before destroying anything */
	invoke_reset_callbacks(arena);

	/* Recursively delete children */
	MktArena *child = arena->first_child;
	while (child)
	{
		MktArena *next = child->next_sibling;
		mkt_memctx_delete(child);
		child = next;
	}
	arena->first_child	   = NULL;
	arena->reset_callbacks = NULL; /* Callbacks were in arena memory */

	/* Free all blocks except first (if any) */
	if (arena->blocks)
	{
		MktArenaBlock *keep	 = arena->blocks;
		MktArenaBlock *block = keep->next;
		while (block)
		{
			MktArenaBlock *next = block->next;
			arena_block_free(block);
			block = next;
		}
		keep->next	   = NULL;
		keep->used	   = 0;
		arena->blocks  = keep;
		arena->current = keep;
	}
	else
	{
		arena->current = NULL;
	}
	arena->total_allocated = 0;
}

/* Allocate from current context */
void *
mkt_alloc(size_t size)
{
	assert(mkt_current_memctx != NULL);
	return arena_alloc(mkt_current_memctx, size, MKT_ARENA_ALIGNMENT);
}

void *
mkt_alloc0(size_t size)
{
	void *ptr = mkt_alloc(size);
	if (ptr)
		memset(ptr, 0, size);
	return ptr;
}

void *
mkt_memctx_alloc(MktMemCtx ctx, size_t size)
{
	return arena_alloc(ctx, size, MKT_ARENA_ALIGNMENT);
}

void *
mkt_memctx_alloc0(MktMemCtx ctx, size_t size)
{
	void *ptr = mkt_memctx_alloc(ctx, size);
	if (ptr)
		memset(ptr, 0, size);
	return ptr;
}

/*
 * Realloc is limited in arena mode - we can't reclaim the old space.
 * This just allocates new space and copies. Use sparingly.
 *
 * Note: We don't know the old size, so the caller must handle copying
 * if they need to preserve data. This matches the limitation of arena
 * allocators.
 */
void *
mkt_realloc(void *ptr, size_t size)
{
	if (!ptr)
		return mkt_alloc(size);
	if (size == 0)
		return NULL;

	/* Just allocate new space - caller responsible for copying */
	return mkt_alloc(size);
}

/* Free is a no-op in arena mode */
void
mkt_free(void *ptr)
{
	(void)ptr; /* Memory freed on context reset/delete */
}

/* Aligned allocation from current context */
void *
mkt_alloc_aligned(size_t size, size_t alignment)
{
	assert(mkt_current_memctx != NULL);
	return arena_alloc(mkt_current_memctx, size, alignment);
}

void
mkt_free_aligned(void *ptr)
{
	(void)ptr; /* No-op in arena mode */
}

/* Switch context, return old context */
MktMemCtx
mkt_memctx_switch(MktMemCtx ctx)
{
	MktMemCtx old	   = mkt_current_memctx;
	mkt_current_memctx = ctx;
	return old;
}

/* Get total bytes allocated in context */
size_t
mkt_memctx_total_allocated(MktMemCtx ctx)
{
	return ctx ? ctx->total_allocated : 0;
}

/* Register a callback to be called on context reset or delete */
void
mkt_memctx_register_reset_callback(
		MktMemCtx ctx, MktMemCtxCallback *cb, void (*func)(void *), void *arg)
{
	cb->func			 = func;
	cb->arg				 = arg;
	cb->next			 = ctx->reset_callbacks;
	ctx->reset_callbacks = cb;
}
