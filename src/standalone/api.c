/*
 * api.c - Public C API for standalone meerkat library
 *
 * Thin wrapper around MktIndex (standalone/index.h) and MktQueryCtx
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

struct MktHandle
{
	MktMemCtx	 memctx; /* top-level context, owns everything */
	MktIndex	*idx;
	MktQueryCtx *qctx;
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

static MktCentroidFormat
parse_centroid_fmt(const char *s)
{
	if (s == NULL)
		return MKT_CENTROID_FMT_RABITQ;
	if (strcmp(s, "float32") == 0)
		return MKT_CENTROID_FMT_FLOAT;
	if (strcmp(s, "float16") == 0)
		return MKT_CENTROID_FMT_HALF;
	return MKT_CENTROID_FMT_RABITQ;
}

static MktPostingFormat
parse_posting_fmt(const char *s)
{
	if (s != NULL && strcmp(s, "flat") == 0)
		return MKT_POSTING_FMT_FLAT;
	return MKT_POSTING_FMT_PAGES; /* default: pages (matches PG on-disk) */
}

static MktDistanceMode
parse_distance_mode(const char *s)
{
	if (s != NULL && strcmp(s, "symmetric") == 0)
		return MKT_DISTANCE_MODE_SYMMETRIC;
	return MKT_DISTANCE_MODE_ASYMMETRIC;
}

/* ----------------------------------------------------------------
 * API
 * ---------------------------------------------------------------- */

MktHandle *
mkt_handle_create(
		MktVectorSource *src,
		uint32_t		 nlist,
		uint32_t		 fan_out,
		const char		*metric,
		const char		*centroid_fmt,
		const char		*posting_fmt,
		uint32_t		 km_nredo,
		uint32_t		 km_max_iter,
		MktBuildInfo	*info)
{
	/* Create context as child of the current context if one exists,
	 * otherwise as a top-level context. This works both when called
	 * from the CLI (which sets up a cli_memctx) and from Python
	 * ctypes (where no context is set). */
	MktMemCtx parent  = mkt_current_memctx;
	MktMemCtx memctx  = mkt_memctx_create(parent, "mkt_handle");
	MktMemCtx old_ctx = mkt_memctx_switch(memctx);

	MktCentroidFormat fmt = parse_centroid_fmt(centroid_fmt);

	MktIndexConfig config = {
			.nlist		   = nlist,
			.fan_out	   = fan_out,
			.centroid_fmt  = fmt,
			.metric		   = parse_metric(metric),
			.km_nredo	   = km_nredo,
			.km_max_iter   = km_max_iter,
			.encode_rabitq = true,
			.posting_fmt   = parse_posting_fmt(posting_fmt),
	};

	MktIndex *idx = mkt_index_build(src, &config);
	if (idx == NULL)
	{
		mkt_memctx_switch(old_ctx);
		mkt_memctx_delete(memctx);
		return NULL;
	}

	/* Fill build stats if requested */
	if (info != NULL)
	{
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
	}

	/* Pre-allocate query context: k up to 100, nprobe up to 400 */
	MktQueryCtx *qctx = mkt_query_ctx_create(idx, 100, 400);
	if (qctx == NULL)
	{
		mkt_index_destroy(idx);
		mkt_memctx_switch(old_ctx);
		mkt_memctx_delete(memctx);
		return NULL;
	}

	MktHandle *handle = mkt_alloc(sizeof(MktHandle));
	handle->memctx	  = memctx;
	handle->idx		  = idx;
	handle->qctx	  = qctx;

	mkt_memctx_switch(old_ctx);

	return handle;
}

MktHandle *
mkt_handle_create_from_array(
		const float	 *vectors,
		uint32_t	  nvecs,
		uint32_t	  dim,
		uint32_t	  nlist,
		uint32_t	  fan_out,
		const char	 *metric,
		const char	 *centroid_fmt,
		const char	 *posting_fmt,
		uint32_t	  km_nredo,
		uint32_t	  km_max_iter,
		MktBuildInfo *info)
{
	MktArraySource array_src;
	mkt_array_source_init(&array_src, vectors, nvecs, dim);
	return mkt_handle_create(
			&array_src.base,
			nlist,
			fan_out,
			metric,
			centroid_fmt,
			posting_fmt,
			km_nredo,
			km_max_iter,
			info);
}

uint32_t
mkt_handle_query(
		MktHandle	*handle,
		const float *query,
		uint32_t	 k,
		uint32_t	 nprobe,
		const char	*distance_mode,
		bool		 rerank,
		uint32_t	*result_ids)
{
	if (handle == NULL)
		return 0;

	MktDistanceMode mode = parse_distance_mode(distance_mode);

	return mkt_query_exec(
			handle->qctx, query, k, nprobe, mode, rerank, result_ids);
}

void
mkt_handle_destroy(MktHandle *handle)
{
	if (handle == NULL)
		return;

	/* Destroy query ctx and index first (they manage their own
	 * child contexts), then delete the top-level context which
	 * frees the handle itself. */
	mkt_query_ctx_destroy(handle->qctx);
	mkt_index_destroy(handle->idx);

	MktMemCtx memctx = handle->memctx;
	mkt_memctx_delete(memctx);
}
