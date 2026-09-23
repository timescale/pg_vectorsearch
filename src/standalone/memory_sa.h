/*
 * memory_standalone.h - Standalone arena allocator types and declarations
 *
 * An arena allocates memory from large blocks using bump-pointer allocation.
 * Individual allocations cannot be freed; memory is released all at once
 * when the arena is reset or destroyed. This matches PostgreSQL memory
 * context semantics and provides excellent cache locality.
 */

#ifndef VS_MEMORY_STANDALONE_H
#define VS_MEMORY_STANDALONE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define VS_ARENA_BLOCK_SIZE (64 * 1024) /* 64 KB default block size */
#define VS_ARENA_ALIGNMENT	16			/* Default alignment */

/* Block header for arena memory blocks */
typedef struct VsArenaBlock
{
	struct VsArenaBlock *next; /* Next block in chain */
	size_t				 size; /* Total size of this block */
	size_t				 used; /* Bytes used in this block */
	/* Data follows immediately after header, aligned */
} VsArenaBlock;

/* Callback for context reset/delete (like PostgreSQL's MemoryContextCallback)
 */
typedef struct VsMemCtxCallback
{
	struct VsMemCtxCallback *next;
	void (*func)(void *arg);
	void *arg;
} VsMemCtxCallback;

/* Arena memory context */
typedef struct VsArena
{
	const char		 *name;			/* Context name for debugging */
	struct VsArena	 *parent;		/* Parent context (for hierarchy) */
	struct VsArena	 *first_child;	/* First child context */
	struct VsArena	 *next_sibling; /* Next sibling in parent's child list */
	VsArenaBlock	 *current;		/* Current block for allocations */
	VsArenaBlock	 *blocks;		/* All blocks (for freeing) */
	VsMemCtxCallback *reset_callbacks; /* Callbacks for reset/delete */
	size_t			  block_size;	   /* Size for new blocks */
	size_t			  total_allocated; /* Stats: total bytes allocated */
} VsArena;

typedef VsArena *VsMemCtx;

/* Current memory context (thread-local) */
extern _Thread_local VsMemCtx vs_current_memctx;

/* Accessor mirroring the PG side's CurrentMemoryContext, so shared code
 * can capture an owning context without mode-specific #ifdefs. */
#define vs_memctx_current() vs_current_memctx

/* Allocation functions */
void *vs_alloc(size_t size);
void *vs_alloc0(size_t size);
void *vs_realloc(void *ptr, size_t size);
void  vs_free(void *ptr);
void *vs_memctx_alloc(VsMemCtx ctx, size_t size);
void *vs_memctx_alloc0(VsMemCtx ctx, size_t size);
void *vs_alloc_aligned(size_t size, size_t alignment);
void  vs_free_aligned(void *ptr);

/* Context management */
VsMemCtx vs_memctx_create(VsMemCtx parent, const char *name);
void	 vs_memctx_delete(VsMemCtx ctx);
void	 vs_memctx_reset(VsMemCtx ctx);
VsMemCtx vs_memctx_switch(VsMemCtx ctx);
size_t	 vs_memctx_total_allocated(VsMemCtx ctx);

/* Register a callback to be called when context is reset or deleted.
 * The callback is allocated from the context itself, so no cleanup needed.
 * Like PostgreSQL's MemoryContextRegisterResetCallback(). */
void vs_memctx_register_reset_callback(
		VsMemCtx ctx, VsMemCtxCallback *cb, void (*func)(void *), void *arg);

/* Varlena compatibility macros (standalone: direct size field) */
#define VS_SET_VARSIZE(ptr, size) ((ptr)->vl_len_ = (int32_t)(size))
#define VS_VARSIZE(ptr)			  ((ptr)->vl_len_)

#endif /* VS_MEMORY_STANDALONE_H */
