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
	uint64_t rotation_ns;		   /* query rotation P^T*query */
	uint64_t centroid_lut_ns;	   /* fastscan LUT build (once/query) */
	uint64_t centroid_pageread_ns; /* centroid page read/release */
	uint64_t centroid_score_ns;	   /* centroid page scoring (incl. lut) */
	/* Routing-quality diagnostic: deepest probe rank (0-based) among the
	 * final top-k results, i.e. how many probed clusters were needed. */
	uint32_t max_contrib_rank;
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

	/* Probe-order scratch: exact centroid distance + index per routed
	 * cluster, used to re-rank the expanded probe set (mkt.probe_expand).
	 * Sized to max_nprobe at init. */
	float	 *probe_dists;
	uint32_t *probe_order;

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

/* Cap the exact-rerank candidate pool: 0 = automatic (16 * k),
 * -1 = unlimited, positive = absolute cap (never effective below k). */
void mkt_query_set_rerank_pool(int32_t n);

/*
 * Probe-order refinement (mkt.probe_expand).
 *
 * Routes ceil(nprobe * expand) leaf candidates through the centroid
 * beam, re-ranks them by EXACT query-centroid distance (the
 * full-precision rotated centroid on each cluster's first posting
 * page), and scans only the best nprobe in that order — fixing the
 * probe-order noise of compressed (RaBitQ) centroid routing.
 *
 * 1.0 means no expansion (identity). Enabled by default (2.0): gains
 * saturate around a factor of 2. The extra routed candidates are
 * capped (MKT_PROBE_EXPAND_MAX_EXTRA) so overhead stays bounded at
 * large nprobe, and the phase is skipped for indexes whose centroid
 * pages are exact (float/half) — there is no ordering noise to fix.
 */
void mkt_query_set_probe_expand(double expand);

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

/*
 * Default nprobe for an index with `nlist` clusters, used when the
 * caller does not set one. Recall at a fixed nprobe/nlist ratio rises
 * with cluster count, so iso-recall nprobe grows roughly with
 * sqrt(nlist): 0.5 * sqrt(nlist) measured ~0.93-0.96 recall@10 across
 * the 10M-100M benchmark sweeps. Floored at 10 so tiny indexes probe
 * a meaningful set, capped at 2048 (past the measured range; explicit
 * settings go higher), and never above nlist itself.
 */
uint32_t mkt_auto_nprobe(uint32_t nlist);

/*
 * Floor for the centroid-search beam width, in tree candidates. The
 * benchmark-tuned query shapes keep the beam equal to nprobe at small
 * nprobe (wide routing matters most when few lists are probed) and at
 * half of nprobe beyond it; with centroid_beam_scale at its 0.5
 * default, flooring the beam at this many candidates — never more
 * than nprobe — reproduces exactly that two-regime shape.
 */
#define MKT_CENTROID_BEAM_FLOOR 80

#endif /* MKT_QUERY_SCAN_H */
