/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * vec32_source.c - Array-backed vector source implementation
 */

#include <stddef.h>

#include "standalone/vec32_source.h"

static bool
array_source_next(
		Vec32Source	 *src,
		uint32_t	  stride,
		const float **vec_out,
		uint32_t	 *id_out)
{
	VsArraySource *as = (VsArraySource *)src;
	if (as->pos >= src->nvecs)
		return false;
	*id_out	 = as->pos;
	*vec_out = as->data + (size_t)as->pos * src->dim;
	as->pos += stride;
	return true;
}

static void
array_source_reset(Vec32Source *src)
{
	VsArraySource *as = (VsArraySource *)src;
	as->pos			  = 0;
}

void
vs_array_source_init(
		VsArraySource *src, const float *data, uint32_t nvecs, uint32_t dim)
{
	src->base.next	   = array_source_next;
	src->base.reset	   = array_source_reset;
	src->base.read_all = NULL;
	src->base.nvecs	   = nvecs;
	src->base.dim	   = dim;
	src->data		   = data;
	src->pos		   = 0;
}
