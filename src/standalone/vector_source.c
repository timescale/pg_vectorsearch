/*
 * vector_source.c - Array-backed vector source implementation
 */

#include <stddef.h>

#include "standalone/vector_source.h"

static bool
array_source_next(
		MktVectorSource *src,
		uint32_t		 stride,
		const float	   **vec_out,
		uint32_t		*id_out)
{
	MktArraySource *as = (MktArraySource *)src;
	if (as->pos >= src->nvecs)
		return false;
	*id_out	 = as->pos;
	*vec_out = as->data + (size_t)as->pos * src->dim;
	as->pos += stride;
	return true;
}

static void
array_source_reset(MktVectorSource *src)
{
	MktArraySource *as = (MktArraySource *)src;
	as->pos			   = 0;
}

void
mkt_array_source_init(
		MktArraySource *src, const float *data, uint32_t nvecs, uint32_t dim)
{
	src->base.next	   = array_source_next;
	src->base.reset	   = array_source_reset;
	src->base.read_all = NULL;
	src->base.nvecs	   = nvecs;
	src->base.dim	   = dim;
	src->data		   = data;
	src->pos		   = 0;
}
