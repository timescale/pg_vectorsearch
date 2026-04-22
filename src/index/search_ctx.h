/*
 * search_ctx.h - Pre-allocated search context for ANN queries
 *
 * Shared between standalone and PG. All buffers are allocated once
 * at init time; the query hot path does zero allocations (modulo
 * rare topk growth).
 *
 * Usage:
 *   MktSearchCtx ctx;
 *   mkt_search_ctx_init(&ctx, &cfg);
 *
 *   // Per-query (zero-alloc hot path):
 *   uint32_t n = mkt_search_execute(&ctx, query, k, nprobe, mode, &stats);
 *   // Results in ctx.candidates[0..n)
 *
 *   mkt_search_ctx_cleanup(&ctx);
 */

#ifndef MKT_SEARCH_CTX_H
#define MKT_SEARCH_CTX_H

#include "algo/topk.h"
#include "index/centroid_search.h"
#include "index/posting_scan.h"
#include "index/storage.h"
#include "quant/rabitq.h"

/* ----------------------------------------------------------------
 * Search statistics (portable — no PG-specific types)
 * ---------------------------------------------------------------- */

typedef struct MktSearchStats
{
	uint32_t centroid_pages_read;
	uint32_t posting_pages_read;
	uint32_t posting_entries_scanned;
	uint32_t clusters_scanned;
} MktSearchStats;

/* ----------------------------------------------------------------
 * Configuration for search context initialization
 * ---------------------------------------------------------------- */

typedef struct MktSearchCtxConfig
{
	/* Borrowed pointers — must outlive the context */
	const RaBitQParams *params;
	const float		   *pt_global_mean;

	/* PG: both point to the same MktannStorage (one index relation).
	 * Standalone: separate ArrayPageStorage for centroids vs postings. */
	MktStorage *centroid_storage;
	MktStorage *posting_storage;

	/* Non-NULL for inline page access (standalone postings) */
	char *page_base;

	/* Index metadata */
	Dimension		  dim;
	uint8_t			  nlevels;
	BlockNumber		  first_centroid;
	uint32_t		  max_k;
	uint32_t		  max_nprobe;
	DistanceMetric	  metric;
	MktCentroidFormat centroid_format;
} MktSearchCtxConfig;

/* ----------------------------------------------------------------
 * Search context
 * ---------------------------------------------------------------- */

typedef struct MktSearchCtx
{
	/* Index metadata (immutable after init) */
	const RaBitQParams *params;
	const float		   *pt_global_mean;
	Dimension			dim;
	uint8_t				nlevels;
	BlockNumber			first_centroid;
	uint32_t			max_k;
	uint32_t			max_nprobe;
	DistanceMetric		metric;
	MktCentroidFormat	centroid_format;

	/* Storage for page I/O */
	MktStorage *centroid_storage;
	MktStorage *posting_storage;
	char	   *page_base;

	/* Pre-allocated query buffers */
	float	*query_buf;
	float	*pt_query;
	float	*beam_transformed;
	float	*cluster_transformed;
	uint8_t *beam_query_bits;
	uint8_t *cluster_query_bits;

	RaBitQQueryState beam_qs;
	RaBitQQueryState cluster_qs;

	/* Pre-allocated search state (reset per query) */
	MktCentroidResult *beam_results;
	MktTopK			   topk;
	MktPostingScan	   pscan;
	MktTopKEntry	  *candidates;
	uint32_t		   cand_cap;
	uint32_t		   ncandidates;
} MktSearchCtx;

/* ----------------------------------------------------------------
 * API
 * ---------------------------------------------------------------- */

void mkt_search_ctx_init(MktSearchCtx *ctx, const MktSearchCtxConfig *cfg);

void mkt_search_ctx_cleanup(MktSearchCtx *ctx);

/*
 * Execute one ANN search query.
 *
 * Normalizes the query (cosine), runs beam search over centroids,
 * scans posting lists, and collects approximate candidates.
 *
 * Returns: number of candidates.
 * Results are in ctx->candidates[0..return_count), sorted by
 * distance ascending. The caller owns reranking (if any).
 */
uint32_t mkt_search_execute(
		MktSearchCtx   *ctx,
		const float	   *query,
		uint32_t		k,
		uint32_t		nprobe,
		MktDistanceMode mode,
		MktSearchStats *stats);

#endif /* MKT_SEARCH_CTX_H */
