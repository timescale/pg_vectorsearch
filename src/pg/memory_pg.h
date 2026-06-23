/*
 * memory_pg.h - PostgreSQL memory wrappers (header-only)
 *
 * Maps Meerkat memory functions to PostgreSQL's palloc/MemoryContext API.
 * This file is only included when building as a PostgreSQL extension.
 */

#ifndef MKT_MEMORY_PG_H
#define MKT_MEMORY_PG_H

#include <postgres.h>

#include <utils/memutils.h>
#include <varatt.h>

typedef MemoryContext		  MktMemCtx;
typedef MemoryContextCallback MktMemCtxCallback;

/* Direct mappings to palloc family.
 * Use the HUGE variants so legitimately-large build buffers (k-means
 * sample shards, per-cluster arrays at high nlist) are not rejected by
 * the 1GB MaxAllocSize cap. For sub-1GB allocations these behave
 * identically to plain palloc (same context, same cost). */
#define mkt_alloc(size)				 palloc_extended((size), MCXT_ALLOC_HUGE)
#define mkt_alloc0(size) \
	palloc_extended((size), MCXT_ALLOC_HUGE | MCXT_ALLOC_ZERO)
#define mkt_realloc(ptr, size)		 repalloc_huge((ptr), (size))
#define mkt_free(ptr)				 pfree(ptr)
#define mkt_memctx_alloc(ctx, size)	 \
	MemoryContextAllocExtended((ctx), (size), MCXT_ALLOC_HUGE)
#define mkt_memctx_alloc0(ctx, size) \
	MemoryContextAllocExtended((ctx), (size), MCXT_ALLOC_HUGE | MCXT_ALLOC_ZERO)
#define mkt_alloc_aligned(sz, al)	 palloc_aligned(sz, al, 0)
#define mkt_free_aligned(ptr)		 pfree(ptr)

/* Context management */
#define mkt_memctx_create(parent, name)                 \
	AllocSetContextCreate(                              \
			(parent) ? (parent) : CurrentMemoryContext, \
			(name),                                     \
			ALLOCSET_DEFAULT_SIZES)
#define mkt_memctx_delete(ctx)			MemoryContextDelete(ctx)
#define mkt_memctx_reset(ctx)			MemoryContextReset(ctx)
#define mkt_memctx_switch(ctx)			MemoryContextSwitchTo(ctx)
#define mkt_memctx_total_allocated(ctx) ((size_t)0)

/* Callback registration - wraps PostgreSQL's
 * MemoryContextRegisterResetCallback. Note: In PG, 'func' signature is
 * void(*)(void*) and uses cb->arg directly. */
#define mkt_memctx_register_reset_callback(ctx, cb, func_ptr, arg_ptr) \
	do                                                                 \
	{                                                                  \
		(cb)->func = (func_ptr);                                       \
		(cb)->arg  = (arg_ptr);                                        \
		MemoryContextRegisterResetCallback((ctx), (cb));               \
	} while (0)

/* Varlena macros (PG: SET_VARSIZE handles TOAST flag bits) */
#define MKT_SET_VARSIZE(ptr, size) SET_VARSIZE(ptr, size)
#define MKT_VARSIZE(ptr)		   VARSIZE(ptr)

#endif /* MKT_MEMORY_PG_H */
