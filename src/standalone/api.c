/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * api.c - Public C API for standalone pg_vectorsearch library
 *
 * Thin wrapper around PrismIndex (standalone/index.h) and PrismQueryCtx
 * (standalone/query.h). The handle bundles both so callers get a
 * single opaque pointer.
 *
 * A top-level memory context owns the handle. The index and query
 * context create their own child contexts internally.
 */

#include <string.h>

#include "core/memory.h"
#include "standalone/api.h"
#include "standalone/index.h"
#include "standalone/query.h"

struct VsHandle
{
	VsMemCtx	   memctx; /* top-level context, owns everything */
	PrismIndex	  *idx;
	PrismQueryCtx *qctx;
};

/* ----------------------------------------------------------------
 * Parse helpers
 * ---------------------------------------------------------------- */

static DistanceMetric
parse_metric(const char *s)
{
	if (s == NULL)
		return DISTANCE_L2;
	if (strcmp(s, "angular") == 0 || strcmp(s, "cosine") == 0)
		return DISTANCE_COSINE;
	return DISTANCE_L2;
}

static PrismCentroidFormat
parse_centroid_fmt(const char *s)
{
	if (s == NULL)
		return PRISM_CENTROID_FMT_RABITQ;
	if (strcmp(s, "float32") == 0)
		return PRISM_CENTROID_FMT_FLOAT;
	if (strcmp(s, "float16") == 0)
		return PRISM_CENTROID_FMT_HALF;
	if (strcmp(s, "fastscan") == 0)
		return PRISM_CENTROID_FMT_FASTSCAN;
	return PRISM_CENTROID_FMT_RABITQ;
}

static PrismPostingFormat
parse_posting_fmt(const char *s)
{
	if (s != NULL && strcmp(s, "flat") == 0)
		return PRISM_POSTING_FMT_FLAT;
	return PRISM_POSTING_FMT_PAGES; /* default: pages (matches PG on-disk) */
}

static VsDistanceMode
parse_distance_mode(const char *s)
{
	if (s != NULL && strcmp(s, "symmetric") == 0)
		return VS_DISTANCE_MODE_SYMMETRIC;
	return VS_DISTANCE_MODE_ASYMMETRIC;
}

/* ----------------------------------------------------------------
 * API
 * ---------------------------------------------------------------- */

VsHandle *
vs_handle_create(
		Vec32Source	   *src,
		uint32_t		nlist,
		uint32_t		fan_out,
		const char	   *metric,
		const char	   *centroid_fmt,
		const char	   *posting_fmt,
		uint32_t		km_nredo,
		uint32_t		km_max_iter,
		double			soar_lambda,
		double			boundary_epsilon,
		int				fastscan,
		int32_t			nworkers,
		PrismBuildInfo *info)
{
	/* Create context as child of the current context if one exists,
	 * otherwise as a top-level context. This works both when called
	 * from the CLI (which sets up a cli_memctx) and from Python
	 * ctypes (where no context is set). */
	VsMemCtx parent	 = vs_current_memctx;
	VsMemCtx memctx	 = vs_memctx_create(parent, "vs_handle");
	VsMemCtx old_ctx = vs_memctx_switch(memctx);

	PrismCentroidFormat fmt = parse_centroid_fmt(centroid_fmt);

	PrismIndexConfig config = {
			.nlist			  = nlist,
			.fan_out		  = fan_out,
			.centroid_fmt	  = fmt,
			.metric			  = parse_metric(metric),
			.km_nredo		  = km_nredo,
			.km_max_iter	  = km_max_iter,
			.soar_lambda	  = soar_lambda,
			.boundary_epsilon = boundary_epsilon,
			.fastscan		  = fastscan,
			.nworkers		  = nworkers,
			.encode_rabitq	  = true,
			.posting_fmt	  = parse_posting_fmt(posting_fmt),
	};

	PrismBuildStats build_stats;
	PrismIndex	   *idx = prism_index_build(src, &config, &build_stats);
	if (idx == NULL)
	{
		vs_memctx_switch(old_ctx);
		vs_memctx_delete(memctx);
		return NULL;
	}

	/* Fill build stats if requested */
	if (info != NULL)
	{
		info->stats		  = build_stats;
		info->nlist		  = idx->nlist;
		info->nlevels	  = idx->base.nlevels;
		info->nvecs		  = idx->nvecs;
		info->max_cluster = 0;
		info->min_cluster = UINT32_MAX;
		for (uint32_t c = 0; c < idx->nlist; c++)
		{
			uint32_t sz = idx->clusters[c].count;
			if (sz > info->max_cluster)
				info->max_cluster = sz;
			if (sz < info->min_cluster)
				info->min_cluster = sz;
		}
		if (idx->nlist == 0)
		{
			/* No centroids, so there is nothing to estimate from. */
			info->min_cluster = 0;
		}
		else if (info->max_cluster == 0)
		{
			/* Pages mode skips cluster list construction —
			 * fall back to estimates */
			info->max_cluster = (idx->nvecs + idx->nlist - 1) / idx->nlist;
			info->min_cluster = idx->nvecs / idx->nlist;
		}
	}

	/* Pre-allocate query context: k up to 100, nprobe up to 400 */
	PrismQueryCtx *qctx = prism_query_ctx_create(idx, 100, 400);
	if (qctx == NULL)
	{
		prism_index_destroy(idx);
		vs_memctx_switch(old_ctx);
		vs_memctx_delete(memctx);
		return NULL;
	}

	VsHandle *handle = vs_alloc(sizeof(VsHandle));
	handle->memctx	 = memctx;
	handle->idx		 = idx;
	handle->qctx	 = qctx;

	vs_memctx_switch(old_ctx);

	return handle;
}

VsHandle *
vs_handle_create_from_array(
		const float	   *vectors,
		uint32_t		nvecs,
		uint32_t		dim,
		uint32_t		nlist,
		uint32_t		fan_out,
		const char	   *metric,
		const char	   *centroid_fmt,
		const char	   *posting_fmt,
		uint32_t		km_nredo,
		uint32_t		km_max_iter,
		double			soar_lambda,
		double			boundary_epsilon,
		int				fastscan,
		int32_t			nworkers,
		PrismBuildInfo *info)
{
	VsArraySource array_src;
	vs_array_source_init(&array_src, vectors, nvecs, dim);
	return vs_handle_create(
			&array_src.base,
			nlist,
			fan_out,
			metric,
			centroid_fmt,
			posting_fmt,
			km_nredo,
			km_max_iter,
			soar_lambda,
			boundary_epsilon,
			fastscan,
			nworkers,
			info);
}

uint32_t
vs_handle_query(
		VsHandle	*handle,
		const float *query,
		uint32_t	 k,
		uint32_t	 nprobe,
		const char	*distance_mode,
		bool		 rerank,
		uint32_t	*result_ids)
{
	if (handle == NULL)
		return 0;

	VsDistanceMode mode = parse_distance_mode(distance_mode);

	return prism_query_exec(
			handle->qctx, query, k, nprobe, mode, rerank, result_ids);
}

void
vs_handle_destroy(VsHandle *handle)
{
	if (handle == NULL)
		return;

	/* Destroy query ctx and index first (they manage their own
	 * child contexts), then delete the top-level context which
	 * frees the handle itself. */
	prism_query_ctx_destroy(handle->qctx);
	prism_index_destroy(handle->idx);

	VsMemCtx memctx = handle->memctx;
	vs_memctx_delete(memctx);
}
