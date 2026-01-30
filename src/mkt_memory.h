/*
 * mkt_memory.h - Memory allocation abstraction
 *
 * Provides a unified interface for memory allocation that works in both
 * standalone mode (using malloc/free) and PostgreSQL mode (using
 * palloc/pfree).
 */

#ifndef MKT_MEMORY_H
#define MKT_MEMORY_H

#include <stddef.h>

/*
 * Basic allocation functions.
 *
 * In standalone mode, these map to malloc/free.
 * In PostgreSQL mode, these map to palloc/pfree.
 */
void *mkt_alloc(size_t size);
void *mkt_alloc0(size_t size); /* Zero-initialized */
void *mkt_realloc(void *ptr, size_t size);
void  mkt_free(void *ptr);

/*
 * Aligned allocation for SIMD operations.
 *
 * Alignment must be a power of 2 and at least sizeof(void*).
 * Memory allocated with mkt_alloc_aligned must be freed with mkt_free_aligned.
 */
void *mkt_alloc_aligned(size_t size, size_t alignment);
void  mkt_free_aligned(void *ptr);

#endif /* MKT_MEMORY_H */
