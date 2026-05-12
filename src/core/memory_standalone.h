/*
 * memory_standalone.h - Standalone arena allocator types and declarations
 *
 * An arena allocates memory from large blocks using bump-pointer allocation.
 * Individual allocations cannot be freed; memory is released all at once
 * when the arena is reset or destroyed. This matches PostgreSQL memory
 * context semantics and provides excellent cache locality.
 */

#ifndef MKT_MEMORY_STANDALONE_H
#define MKT_MEMORY_STANDALONE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define MKT_ARENA_BLOCK_SIZE (64 * 1024) /* 64 KB default block size */
#define MKT_ARENA_ALIGNMENT	 16			 /* Default alignment */

/* Block header for arena memory blocks */
typedef struct MktArenaBlock
{
	struct MktArenaBlock *next; /* Next block in chain */
	size_t				  size; /* Total size of this block */
	size_t				  used; /* Bytes used in this block */
	/* Data follows immediately after header, aligned */
} MktArenaBlock;

/* Callback for context reset/delete (like PostgreSQL's MemoryContextCallback)
 */
typedef struct MktMemCtxCallback
{
	struct MktMemCtxCallback *next;
	void (*func)(void *arg);
	void *arg;
} MktMemCtxCallback;

/* Arena memory context */
typedef struct MktArena
{
	const char		  *name;		 /* Context name for debugging */
	struct MktArena	  *parent;		 /* Parent context (for hierarchy) */
	struct MktArena	  *first_child;	 /* First child context */
	struct MktArena	  *next_sibling; /* Next sibling in parent's child list */
	MktArenaBlock	  *current;		 /* Current block for allocations */
	MktArenaBlock	  *blocks;		 /* All blocks (for freeing) */
	MktMemCtxCallback *reset_callbacks; /* Callbacks for reset/delete */
	size_t			   block_size;		/* Size for new blocks */
	size_t			   total_allocated; /* Stats: total bytes allocated */
} MktArena;

typedef MktArena *MktMemCtx;

/* Current memory context (thread-local) */
extern _Thread_local MktMemCtx mkt_current_memctx;

/* Allocation functions */
void *mkt_alloc(size_t size);
void *mkt_alloc0(size_t size);
void *mkt_realloc(void *ptr, size_t old_size, size_t new_size);
void  mkt_free(void *ptr);
void *mkt_memctx_alloc(MktMemCtx ctx, size_t size);
void *mkt_memctx_alloc0(MktMemCtx ctx, size_t size);
void *mkt_alloc_aligned(size_t size, size_t alignment);
void  mkt_free_aligned(void *ptr);

/* Context management */
MktMemCtx mkt_memctx_create(MktMemCtx parent, const char *name);
void	  mkt_memctx_delete(MktMemCtx ctx);
void	  mkt_memctx_reset(MktMemCtx ctx);
MktMemCtx mkt_memctx_switch(MktMemCtx ctx);
size_t	  mkt_memctx_total_allocated(MktMemCtx ctx);

/* Register a callback to be called when context is reset or deleted.
 * The callback is allocated from the context itself, so no cleanup needed.
 * Like PostgreSQL's MemoryContextRegisterResetCallback(). */
void mkt_memctx_register_reset_callback(
		MktMemCtx ctx, MktMemCtxCallback *cb, void (*func)(void *), void *arg);

/* Varlena compatibility macros (standalone: direct size field) */
#define MKT_SET_VARSIZE(ptr, size) ((ptr)->vl_len_ = (int32_t)(size))
#define MKT_VARSIZE(ptr)		   ((ptr)->vl_len_)

#endif /* MKT_MEMORY_STANDALONE_H */
