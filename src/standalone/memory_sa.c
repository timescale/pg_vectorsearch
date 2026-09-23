/*
 * memory_standalone.c - Arena allocator implementation
 *
 * Implements a memory context system compatible with PostgreSQL's semantics
 * but without PostgreSQL dependencies. Used for standalone testing.
 */

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "standalone/memory_sa.h"

/* Thread-local current context */
_Thread_local VsMemCtx vs_current_memctx = NULL;

/* Align size up to alignment boundary */
static inline size_t
align_up(size_t size, size_t alignment)
{
	return (size + alignment - 1) & ~(alignment - 1);
}

/* Allocate a new block */
static VsArenaBlock *
arena_block_create(size_t min_size)
{
	size_t block_size = min_size > VS_ARENA_BLOCK_SIZE ? min_size
													   : VS_ARENA_BLOCK_SIZE;
	/* Include space for header, aligned */
	size_t header_size = align_up(sizeof(VsArenaBlock), VS_ARENA_ALIGNMENT);
	size_t total_size = align_up(header_size + block_size, VS_ARENA_ALIGNMENT);

	VsArenaBlock *block = aligned_alloc(VS_ARENA_ALIGNMENT, total_size);
	if (!block)
		return NULL;

	block->next = NULL;
	block->size = block_size;
	block->used = 0;
	return block;
}

/* Free a block */
static void
arena_block_free(VsArenaBlock *block)
{
	free(block);
}

/* Get data pointer for block */
static inline void *
arena_block_data(VsArenaBlock *block)
{
	size_t header_size = align_up(sizeof(VsArenaBlock), VS_ARENA_ALIGNMENT);
	return (char *)block + header_size;
}

/*
 * Each allocation is prefixed with a size_t header storing the
 * requested size. This lets vs_realloc copy min(old, new) bytes
 * without reading past the old allocation.
 */
#define ALLOC_HDR_SIZE align_up(sizeof(size_t), VS_ARENA_ALIGNMENT)

/* Allocate from arena with specified alignment */
static void *
arena_alloc(VsArena *arena, size_t size, size_t alignment)
{
	if (size == 0)
		return NULL;

	size_t aligned_size = align_up(size, alignment);
	size_t need			= ALLOC_HDR_SIZE + aligned_size;

	/* Try current block first */
	if (arena->current)
	{
		void	 *data = arena_block_data(arena->current);
		uintptr_t base = (uintptr_t)data + arena->current->used;

		/* Align the user pointer; header sits just before it */
		uintptr_t user_ptr = align_up(base + ALLOC_HDR_SIZE, alignment);
		uintptr_t hdr_ptr  = user_ptr - ALLOC_HDR_SIZE;
		size_t	  end	   = (hdr_ptr - (uintptr_t)data) + need;

		if (end <= arena->current->size)
		{
			*(size_t *)hdr_ptr	 = size;
			arena->current->used = end;
			arena->total_allocated += need;
			return (void *)user_ptr;
		}
	}

	/*
	 * Need a new block. Request extra space for alignment padding.
	 * Worst case: we need (alignment - 1) extra bytes to align.
	 */
	size_t		  extra_for_alignment = alignment > VS_ARENA_ALIGNMENT
											  ? alignment - VS_ARENA_ALIGNMENT
											  : 0;
	VsArenaBlock *block = arena_block_create(need + extra_for_alignment);
	if (!block)
		return NULL;

	/* Link new block */
	block->next	   = arena->blocks;
	arena->blocks  = block;
	arena->current = block;

	/* Allocate from new block — align the user pointer */
	void	 *data	   = arena_block_data(block);
	uintptr_t base	   = (uintptr_t)data;
	uintptr_t user_ptr = align_up(base + ALLOC_HDR_SIZE, alignment);
	uintptr_t hdr_ptr  = user_ptr - ALLOC_HDR_SIZE;
	size_t	  end	   = (hdr_ptr - base) + need;

	*(size_t *)hdr_ptr = size;
	block->used		   = end;
	arena->total_allocated += need;
	return (void *)user_ptr;
}

/* Create a new arena context */
VsMemCtx
vs_memctx_create(VsMemCtx parent, const char *name)
{
	/* Allocate arena struct from parent or malloc */
	VsArena *arena;
	if (parent)
		arena = arena_alloc(parent, sizeof(VsArena), VS_ARENA_ALIGNMENT);
	else
		arena = aligned_alloc(
				VS_ARENA_ALIGNMENT,
				align_up(sizeof(VsArena), VS_ARENA_ALIGNMENT));

	if (!arena)
		return NULL;

	memset(arena, 0, sizeof(VsArena));
	arena->name		  = name;
	arena->parent	  = parent;
	arena->block_size = VS_ARENA_BLOCK_SIZE;

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
invoke_reset_callbacks(VsArena *arena)
{
	VsMemCtxCallback *cb = arena->reset_callbacks;
	while (cb)
	{
		cb->func(cb->arg);
		cb = cb->next;
	}
}

/* Delete arena and all children */
void
vs_memctx_delete(VsMemCtx ctx)
{
	if (!ctx)
		return;

	VsArena *arena = ctx;

	/* Invoke callbacks before destroying anything */
	invoke_reset_callbacks(arena);

	/* Recursively delete children first */
	VsArena *child = arena->first_child;
	while (child)
	{
		VsArena *next = child->next_sibling;
		vs_memctx_delete(child);
		child = next;
	}

	/* Unlink from parent */
	if (arena->parent)
	{
		VsArena **pp = &arena->parent->first_child;
		while (*pp && *pp != arena)
			pp = &(*pp)->next_sibling;
		if (*pp)
			*pp = arena->next_sibling;
	}

	/* Free all blocks */
	VsArenaBlock *block = arena->blocks;
	while (block)
	{
		VsArenaBlock *next = block->next;
		arena_block_free(block);
		block = next;
	}

	/* Free arena struct if it was top-level (no parent) */
	if (!arena->parent)
		free(arena);
}

/* Reset arena - free all allocations but keep arena */
void
vs_memctx_reset(VsMemCtx ctx)
{
	if (!ctx)
		return;

	VsArena *arena = ctx;

	/* Invoke callbacks before destroying anything */
	invoke_reset_callbacks(arena);

	/* Recursively delete children */
	VsArena *child = arena->first_child;
	while (child)
	{
		VsArena *next = child->next_sibling;
		vs_memctx_delete(child);
		child = next;
	}
	arena->first_child	   = NULL;
	arena->reset_callbacks = NULL; /* Callbacks were in arena memory */

	/* Free all blocks except first (if any) */
	if (arena->blocks)
	{
		VsArenaBlock *keep	= arena->blocks;
		VsArenaBlock *block = keep->next;
		while (block)
		{
			VsArenaBlock *next = block->next;
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
vs_alloc(size_t size)
{
	assert(vs_current_memctx != NULL);
	return arena_alloc(vs_current_memctx, size, VS_ARENA_ALIGNMENT);
}

void *
vs_alloc0(size_t size)
{
	void *ptr = vs_alloc(size);
	if (ptr)
		memset(ptr, 0, size);
	return ptr;
}

void *
vs_memctx_alloc(VsMemCtx ctx, size_t size)
{
	return arena_alloc(ctx, size, VS_ARENA_ALIGNMENT);
}

void *
vs_memctx_alloc0(VsMemCtx ctx, size_t size)
{
	void *ptr = vs_memctx_alloc(ctx, size);
	if (ptr)
		memset(ptr, 0, size);
	return ptr;
}

void *
vs_realloc(void *ptr, size_t size)
{
	if (!ptr)
		return vs_alloc(size);
	if (size == 0)
		return NULL;

	size_t old_size = *(size_t *)((char *)ptr - ALLOC_HDR_SIZE);
	void  *new_ptr	= vs_alloc(size);
	memcpy(new_ptr, ptr, old_size < size ? old_size : size);
	return new_ptr;
}

/* Free is a no-op in arena mode */
void
vs_free(void *ptr)
{
	(void)ptr; /* Memory freed on context reset/delete */
}

/* Aligned allocation from current context */
void *
vs_alloc_aligned(size_t size, size_t alignment)
{
	assert(vs_current_memctx != NULL);
	return arena_alloc(vs_current_memctx, size, alignment);
}

void
vs_free_aligned(void *ptr)
{
	(void)ptr; /* No-op in arena mode */
}

/* Switch context, return old context */
VsMemCtx
vs_memctx_switch(VsMemCtx ctx)
{
	VsMemCtx old	  = vs_current_memctx;
	vs_current_memctx = ctx;
	return old;
}

/* Get total bytes allocated in context */
size_t
vs_memctx_total_allocated(VsMemCtx ctx)
{
	return ctx ? ctx->total_allocated : 0;
}

/* Register a callback to be called on context reset or delete */
void
vs_memctx_register_reset_callback(
		VsMemCtx ctx, VsMemCtxCallback *cb, void (*func)(void *), void *arg)
{
	cb->func			 = func;
	cb->arg				 = arg;
	cb->next			 = ctx->reset_callbacks;
	ctx->reset_callbacks = cb;
}
