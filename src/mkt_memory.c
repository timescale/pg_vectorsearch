/*
 * mkt_memory.c - Memory allocation implementation (standalone mode)
 */

#include <stdlib.h>
#include <string.h>

#include "mkt_memory.h"

void *
mkt_alloc(size_t size)
{
	return malloc(size);
}

void *
mkt_alloc0(size_t size)
{
	return calloc(1, size);
}

void *
mkt_realloc(void *ptr, size_t size)
{
	return realloc(ptr, size);
}

void
mkt_free(void *ptr)
{
	free(ptr);
}

void *
mkt_alloc_aligned(size_t size, size_t alignment)
{
	return aligned_alloc(alignment, size);
}

void
mkt_free_aligned(void *ptr)
{
	free(ptr);
}
