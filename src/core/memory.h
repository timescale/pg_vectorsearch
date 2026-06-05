/*
 * memory.h - Memory abstraction interface
 *
 * Provides a unified interface for memory allocation that works in both
 * standalone mode (arena allocator) and PostgreSQL mode (palloc/pfree).
 *
 * Build configuration:
 * - Standalone: define MKT_STANDALONE, compile memory_standalone.c
 * - PostgreSQL: no define needed, header-only (macros map to palloc)
 */

#ifndef MKT_MEMORY_H
#define MKT_MEMORY_H

#ifdef MKT_STANDALONE
#include "standalone/memory_sa.h"
#else
#include "pg/memory_pg.h"
#endif

/* Helper for cleanup attribute (works in both modes) */
static inline void
mkt_memctx_delete_ptr(MktMemCtx *ctx)
{
	if (*ctx)
		mkt_memctx_delete(*ctx);
}

/*
 * Scoped context (RAII-style via cleanup attribute)
 *
 * Usage:
 *   void some_function(void) {
 *       MKT_MEMCTX_SCOPE(work_ctx);
 *       MktMemCtx old = mkt_memctx_switch(work_ctx);
 *       // allocations happen in work_ctx
 *       // ...
 *       mkt_memctx_switch(old);
 *   }  // work_ctx automatically deleted here
 */
#ifdef MKT_STANDALONE
#define MKT_MEMCTX_SCOPE(name)                                       \
	MktMemCtx name __attribute__((cleanup(mkt_memctx_delete_ptr))) = \
			mkt_memctx_create(mkt_current_memctx, #name)
#else
#define MKT_MEMCTX_SCOPE(name)                                       \
	MktMemCtx name __attribute__((cleanup(mkt_memctx_delete_ptr))) = \
			mkt_memctx_create(CurrentMemoryContext, #name)
#endif

#endif /* MKT_MEMORY_H */
