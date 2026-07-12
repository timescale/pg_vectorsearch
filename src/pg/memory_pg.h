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

/* Direct mappings to palloc family */
#define mkt_alloc(size)				 palloc(size)
#define mkt_alloc0(size)			 palloc0(size)
#define mkt_realloc(ptr, size)		 repalloc(ptr, size)
#define mkt_free(ptr)				 pfree(ptr)
#define mkt_memctx_alloc(ctx, size)	 MemoryContextAlloc(ctx, size)
#define mkt_memctx_alloc0(ctx, size) MemoryContextAllocZero(ctx, size)
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
#define mkt_memctx_current()			CurrentMemoryContext
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
