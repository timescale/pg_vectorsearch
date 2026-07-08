/*
 * query_scan.h - Shared query execution for ANN search
 *
 * MktQueryState owns all pre-allocated buffers for query execution.
 * Shared between standalone and PG. The per-query execute path
 * does zero allocations (except rare topk candidate buffer growth).
 *
 * Usage:
 *   MktQueryState qs;
 *   mkt_query_state_init(&qs, &index_base, max_k, max_nprobe);
 *
 *   uint32_t n = mkt_query_execute(&qs, query, k, nprobe, mode, &stats);
 *   // Results in qs.candidates[0..n)
 *
 *   mkt_query_state_cleanup(&qs);
 */

#ifndef MKT_QUERY_SCAN_H
#define MKT_QUERY_SCAN_H

#include "algo/topk.h"
#include "index/centroid_search.h"
#include "index/index_base.h"
#include "index/posting_scan.h"

/* ----------------------------------------------------------------
 * Query statistics
 * ---------------------------------------------------------------- */

typedef struct MktQueryStats
{
	uint32_t centroid_pages_read;
	uint32_t posting_pages_read;
	uint32_t posting_pages_skipped; /* tombstoned all-dead pages skipped */
	uint32_t posting_entries_scanned;
	uint32_t clusters_scanned;
	/* Per-phase wall time in nanoseconds (0 when not measured). */
	uint64_t centroid_ns; /* query rotation + centroid beam search */
	uint64_t posting_ns;  /* cluster scan + candidate extraction */
	uint64_t rerank_ns;	  /* exact-distance rerank (heap fetches) */
	/* Fine-grained centroid sub-phases (subsets of centroid_ns). */
	uint64_t rotation_ns;		 /* query rotation P^T*query */
	uint64_t centroid_lut_ns;	 /* fastscan LUT build (once/query) */
	uint64_t centroid_pageread_ns; /* centroid page read/release */
	uint64_t centroid_score_ns;	 /* centroid scoring (incl. lut) */
} MktQueryStats;

/* ----------------------------------------------------------------
 * Query state — pre-allocated, reused across queries
 * ---------------------------------------------------------------- */

typedef struct MktQueryState
{
	MktIndexBase *index;
	uint32_t	  max_k;
	uint32_t	  max_nprobe;

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
	MktCentroidResult  *beam_results;
	MktCentroidScratch *centroid_scratch;
	MktTopK				topk;
	MktPostingScan		pscan;
	MktTopKEntry	   *candidates;
	uint32_t			cand_cap;
	uint32_t			ncandidates;

	/* TID dedup hash for replicated vectors (NULL = disabled).
	 * Uses generation counter — no memset per query. */
	uint64_t *dedup_set;
	uint32_t *dedup_gens;
	uint32_t  dedup_cap; /* power of 2 */
	uint32_t  dedup_gen; /* bumped per query */

	/* Result ordering (indices into candidates + final distances) */
	uint32_t *result_order;
	Distance *result_dists;
	uint32_t  nresults;
} MktQueryState;

/* ----------------------------------------------------------------
 * API
 * ---------------------------------------------------------------- */

void mkt_query_state_init(
		MktQueryState *qs,
		MktIndexBase  *index,
		uint32_t	   max_k,
		uint32_t	   max_nprobe);

void mkt_query_state_cleanup(MktQueryState *qs);

/*
 * Execute one ANN search query.
 *
 * Normalizes the query (cosine), runs beam search over centroids,
 * scans posting lists, and collects approximate candidates.
 *
 * Returns: number of candidates.
 * Results are in qs->candidates[0..return_count), sorted by
 * distance ascending. The caller owns reranking (if any).
 */
uint32_t mkt_query_execute(
		MktQueryState  *qs,
		const float	   *query,
		uint32_t		k,
		uint32_t		nprobe,
		MktDistanceMode mode,
		bool			rerank,
		MktQueryStats  *stats);

/*
 * Route a vector to its nearest leaf posting list(s) — the centroid-search
 * half of mkt_query_execute, without scanning postings. Normalizes the vector
 * (cosine), rotates it into qs->pt_query, and runs the beam search. Returns
 * the number of leaves found; qs->beam_results[0..return) hold them
 * (posting_head), and qs->pt_query holds P^T * (normalized vector) for the
 * caller to reuse (the insert path encodes from it). beam_stats may be NULL.
 *
 * Shared so an inserted vector routes exactly the way a query does.
 */
uint32_t mkt_query_route(
		MktQueryState		   *qs,
		const float			   *query,
		uint32_t				nprobe,
		MktDistanceMode			mode,
		MktCentroidSearchStats *beam_stats);

#endif /* MKT_QUERY_SCAN_H */
