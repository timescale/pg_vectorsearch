/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * memory_pg.h - PostgreSQL memory wrappers (header-only)
 *
 * Maps pg_vectorsearch memory functions to PostgreSQL's
 * palloc/MemoryContext API.
 * This file is only included when building as a PostgreSQL extension.
 */

#ifndef VS_MEMORY_PG_H
#define VS_MEMORY_PG_H

#include <postgres.h>

#include <utils/memutils.h>
#include <varatt.h>

typedef MemoryContext		  VsMemCtx;
typedef MemoryContextCallback VsMemCtxCallback;

/* Direct mappings to palloc family */
#define vs_alloc(size)				palloc(size)
#define vs_alloc0(size)				palloc0(size)
#define vs_realloc(ptr, size)		repalloc(ptr, size)
#define vs_free(ptr)				pfree(ptr)
#define vs_memctx_alloc(ctx, size)	MemoryContextAlloc(ctx, size)
#define vs_memctx_alloc0(ctx, size) MemoryContextAllocZero(ctx, size)
#define vs_alloc_aligned(sz, al)	palloc_aligned(sz, al, 0)
#define vs_free_aligned(ptr)		pfree(ptr)

/* Context management */
#define vs_memctx_create(parent, name)                  \
	AllocSetContextCreate(                              \
			(parent) ? (parent) : CurrentMemoryContext, \
			(name),                                     \
			ALLOCSET_DEFAULT_SIZES)
#define vs_memctx_delete(ctx)		   MemoryContextDelete(ctx)
#define vs_memctx_reset(ctx)		   MemoryContextReset(ctx)
#define vs_memctx_switch(ctx)		   MemoryContextSwitchTo(ctx)
#define vs_memctx_current()			   CurrentMemoryContext
#define vs_memctx_total_allocated(ctx) ((size_t)0)

/* Callback registration - wraps PostgreSQL's
 * MemoryContextRegisterResetCallback. Note: In PG, 'func' signature is
 * void(*)(void*) and uses cb->arg directly. */
#define vs_memctx_register_reset_callback(ctx, cb, func_ptr, arg_ptr) \
	do                                                                \
	{                                                                 \
		(cb)->func = (func_ptr);                                      \
		(cb)->arg  = (arg_ptr);                                       \
		MemoryContextRegisterResetCallback((ctx), (cb));              \
	} while (0)

/* Varlena macros (PG: SET_VARSIZE handles TOAST flag bits) */
#define VS_SET_VARSIZE(ptr, size) SET_VARSIZE(ptr, size)
#define VS_VARSIZE(ptr)			  VARSIZE(ptr)

#endif /* VS_MEMORY_PG_H */
