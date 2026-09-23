/*
 * memory.h - Memory abstraction interface
 *
 * Provides a unified interface for memory allocation that works in both
 * standalone mode (arena allocator) and PostgreSQL mode (palloc/pfree).
 *
 * Build configuration:
 * - Standalone: define VS_STANDALONE, compile memory_standalone.c
 * - PostgreSQL: no define needed, header-only (macros map to palloc)
 */

#ifndef VS_MEMORY_H
#define VS_MEMORY_H

#ifdef VS_STANDALONE
#include "standalone/memory_sa.h"
#else
#include "pg/memory_pg.h"
#endif

/* Helper for cleanup attribute (works in both modes) */
static inline void
vs_memctx_delete_ptr(VsMemCtx *ctx)
{
	if (*ctx)
		vs_memctx_delete(*ctx);
}

/*
 * Scoped context (RAII-style via cleanup attribute)
 *
 * Usage:
 *   void some_function(void) {
 *       VS_MEMCTX_SCOPE(work_ctx);
 *       VsMemCtx old = vs_memctx_switch(work_ctx);
 *       // allocations happen in work_ctx
 *       // ...
 *       vs_memctx_switch(old);
 *   }  // work_ctx automatically deleted here
 */
#ifdef VS_STANDALONE
#define VS_MEMCTX_SCOPE(name)                                      \
	VsMemCtx name __attribute__((cleanup(vs_memctx_delete_ptr))) = \
			vs_memctx_create(vs_current_memctx, #name)
#else
#define VS_MEMCTX_SCOPE(name)                                      \
	VsMemCtx name __attribute__((cleanup(vs_memctx_delete_ptr))) = \
			vs_memctx_create(CurrentMemoryContext, #name)
#endif

#endif /* VS_MEMORY_H */
