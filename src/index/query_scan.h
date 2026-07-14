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

	/* The query's probe list: posting-head blocks in final scan-rank
	 * order (the output of mkt_query_order_probes). Entries can be
	 * InvalidBlockNumber (empty clusters); they consume a rank but are
	 * skipped by the scan. Sized to max_nprobe at init. */
	BlockNumber *probe_heads;

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
/*
 * Phase A: turn the routed beam results (qs->beam_results[0..n_results))
 * into the final probe list qs->probe_heads[0..n_scan), re-ranked by
 * exact query-centroid distance when the probe set was expanded past
 * scan_limit. Returns n_scan. Requires qs->pscan.storage to be set.
 */
uint32_t mkt_query_order_probes(
		MktQueryState *qs, uint32_t n_results, uint32_t scan_limit);

/*
 * Scan one probed cluster (one posting-list chain) into topk. rank is
 * the cluster's position in the probe list; it stamps inserted
 * candidates for the routing-quality diagnostics. Page/entry counts
 * are accumulated into *stats when non-NULL. This is the parallel
 * work unit: participants call it for the clusters they claim.
 */
void mkt_query_scan_cluster(
		MktQueryState  *qs,
		BlockNumber		posting_head,
		uint32_t		rank,
		MktDistanceMode mode,
		MktTopK		   *topk,
		MktQueryStats  *stats);

uint32_t mkt_query_execute(
		MktQueryState  *qs,
		const float	   *query,
		uint32_t		k,
		uint32_t		nprobe,
		MktDistanceMode mode,
		bool			rerank,
		MktQueryStats  *stats);

/*
 * Produce the query's probe list end to end: probe expansion, centroid
 * descent, and phase-A exact re-rank into qs->probe_heads. Used by the
 * parallel rendezvous winner; the serial path composes the same pieces
 * inside mkt_query_execute. Returns the probe count.
 */
uint32_t mkt_query_route_probes(
		MktQueryState		   *qs,
		const float			   *query,
		uint32_t				nprobe,
		MktDistanceMode			mode,
		MktCentroidSearchStats *beam_stats);

/*
 * Parallel-participant execution: like mkt_query_execute, but the probe
 * list is read from the published shared state instead of being routed;
 * clusters are claimed through the shared cursor; candidates are capped
 * to pool_limit and claimed in the shared TID table before the exact
 * rerank (per-participant emission topology). The caller runs the
 * rendezvous first. Results land in qs exactly as in the serial path.
 */
struct MktQueryShared;
uint32_t mkt_query_execute_parallel(
		MktQueryState		  *qs,
		const float			  *query,
		uint32_t			   k,
		MktDistanceMode		   mode,
		bool				   rerank,
		MktQueryStats		  *stats,
		struct MktQueryShared *shared,
		uint32_t			   pool_limit);

/* Resolve the effective rerank-pool cap for k (mkt.rerank_pool
 * semantics: 0 = auto 16*k, -1 = unlimited -> UINT32_MAX, else the
 * value, never below k). */
uint32_t mkt_query_rerank_pool(uint32_t k);

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

#endif /* MKT_QUERY_SCAN_H */
