/*
 * query_scan.h - Shared query execution for ANN search
 *
 * PrismQueryState owns all pre-allocated buffers for query execution.
 * Shared between standalone and PG. The per-query execute path
 * does zero allocations (except rare topk candidate buffer growth).
 *
 * Usage:
 *   PrismQueryState qs;
 *   prism_query_state_init(&qs, &index_base, max_k, max_nprobe);
 *
 *   uint32_t n = prism_query_execute(&qs, query, k, nprobe, mode, &stats);
 *   // Results in qs.candidates[0..n)
 *
 *   prism_query_state_cleanup(&qs);
 */

#ifndef PRISM_QUERY_SCAN_H
#define PRISM_QUERY_SCAN_H

#include "algo/topk.h"
#include "index/centroid_search.h"
#include "index/index_base.h"
#include "index/posting_scan.h"

/* ----------------------------------------------------------------
 * Query statistics
 * ---------------------------------------------------------------- */

typedef struct PrismQueryStats
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
} PrismQueryStats;

/*
 * Candidate slots the extraction buffer holds per row of the top-k.
 *
 * The error-bound gate admits more candidates than the top-k returns, so
 * the extraction buffer and its two ordering arrays are sized to a
 * multiple of max_k. extract_candidates grows them further if a query
 * turns out to admit more.
 *
 * The scan's work_mem budget prices a row from this (see
 * MKT_TOP_K_BYTES_PER_ROW in src/pg/scan.c), so the two must agree.
 */
#define PRISM_QUERY_CAND_PER_K 16

/* ----------------------------------------------------------------
 * Query state — pre-allocated, reused across queries
 * ---------------------------------------------------------------- */

typedef struct PrismQueryState
{
	/*
	 * Owns every buffer below, so that discarding the state is one delete
	 * and cannot miss an allocation. The top-k's and the centroid
	 * scratch's own contexts are created under it and go with it.
	 *
	 * Buffers here are sized to max_k and max_nprobe: resizing means
	 * building a new state, and the old one has to go somewhere. The
	 * posting scan's pinned page is the one thing a context teardown
	 * cannot release, which is why prism_query_state_cleanup exists rather
	 * than callers deleting this directly.
	 */
	MktMemCtx memctx;

	PrismIndexBase *index;
	uint32_t		max_k;
	uint32_t		max_nprobe;

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
	PrismCentroidResult	 *beam_results;
	PrismCentroidScratch *centroid_scratch;
	MktTopK				  topk;
	PrismPostingScan	  pscan;
	MktTopKEntry		 *candidates;
	uint32_t			  cand_cap;
	uint32_t			  ncandidates;

	/* Probe-order scratch: exact centroid distance + index per routed
	 * cluster, used to re-rank the expanded probe set (prism.probe_expand).
	 * Sized to max_nprobe at init. */
	float	 *probe_dists;
	uint32_t *probe_order;

	/* Result ordering (indices into candidates + final distances) */
	uint32_t *result_order;
	Distance *result_dists;
	uint32_t  nresults;
} PrismQueryState;

/* ----------------------------------------------------------------
 * API
 * ---------------------------------------------------------------- */

void prism_query_state_init(
		PrismQueryState *qs,
		PrismIndexBase	*index,
		uint32_t		 max_k,
		uint32_t		 max_nprobe);

void prism_query_state_cleanup(PrismQueryState *qs);

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
uint32_t prism_query_execute(
		PrismQueryState *qs,
		const float		*query,
		uint32_t		 k,
		uint32_t		 nprobe,
		MktDistanceMode	 mode,
		bool			 rerank,
		PrismQueryStats *stats);

/* Cap the exact-rerank candidate pool: 0 = automatic (3 * k *
 * nprobe^0.15, growing further under noisy estimates), -1 = unlimited,
 * positive = absolute cap (never effective below k). */
void prism_query_set_rerank_pool(int32_t n);

/*
 * Size of the rerank pool -- how many candidates the scan will score
 * exactly -- for a given k and nprobe, without the scan's own noise-driven
 * floor.
 *
 * Returns 0 when the pool is uncapped, which means every threshold survivor
 * is reranked rather than none of them. Shared with the cost model.
 */
uint32_t prism_query_rerank_pool_estimate(uint32_t k, uint32_t nprobe);

/*
 * Leaf clusters the centroid beam routes to in order to read nprobe of
 * them, capped at cap. More than nprobe for compressed centroid formats,
 * which route wide and let phase A re-rank on exact distances; exactly
 * nprobe for the exact formats. Called by prism_query_execute and the cost
 * model.
 */
uint32_t prism_query_routed_clusters(
		uint32_t nprobe, uint32_t cap, PrismCentroidFormat centroid_format);

/*
 * Centroid slots the beam keeps per intermediate level for a given nprobe.
 * A fraction of nprobe raised by three floors below which leaves become
 * unreachable outright. Called by prism_query_execute and the cost model.
 */
uint32_t prism_query_beam_width(
		uint32_t nprobe, uint32_t nlist, uint32_t fan_out, double beam_scale);

/*
 * Probe-order refinement (prism.probe_expand).
 *
 * Routes ceil(nprobe * expand) leaf candidates through the centroid
 * beam, re-ranks them by EXACT query-centroid distance (the
 * full-precision rotated centroid on each cluster's first posting
 * page), and scans only the best nprobe in that order — fixing the
 * probe-order noise of compressed (RaBitQ) centroid routing.
 *
 * 1.0 means no expansion (identity). Enabled by default (2.0): gains
 * saturate around a factor of 2. The extra routed candidates are
 * capped (PRISM_PROBE_EXPAND_MAX_EXTRA) so overhead stays bounded at
 * large nprobe, and the phase is skipped for indexes whose centroid
 * pages are exact (float/half) — there is no ordering noise to fix.
 */
void prism_query_set_probe_expand(double expand);

/*
 * Route a vector to its nearest leaf posting list(s) — the centroid-search
 * half of prism_query_execute, without scanning postings. Normalizes the
 * vector (cosine), rotates it into qs->pt_query, and runs the beam search.
 * Returns the number of leaves found; qs->beam_results[0..return) hold them
 * (posting_head), and qs->pt_query holds P^T * (normalized vector) for the
 * caller to reuse (the insert path encodes from it). beam_stats may be NULL.
 *
 * Shared so an inserted vector routes exactly the way a query does.
 */
uint32_t prism_query_route(
		PrismQueryState			 *qs,
		const float				 *query,
		uint32_t				  nprobe,
		MktDistanceMode			  mode,
		PrismCentroidSearchStats *beam_stats);

/*
 * Default nprobe for an index with `nlist` clusters, used when the
 * caller does not set one. Recall at a fixed nprobe/nlist ratio rises
 * with cluster count, so iso-recall nprobe grows roughly with
 * sqrt(nlist): 0.5 * sqrt(nlist) measured ~0.93-0.96 recall@10 across
 * the 10M-100M benchmark sweeps. Floored at 10 so tiny indexes probe
 * a meaningful set, capped at 2048 (past the measured range; explicit
 * settings go higher), and never above nlist itself.
 */
uint32_t prism_auto_nprobe(uint32_t nlist);

/*
 * Floor for the centroid-search beam width, in tree candidates. The
 * benchmark-tuned query shapes keep the beam equal to nprobe at small
 * nprobe (wide routing matters most when few lists are probed) and at
 * half of nprobe beyond it; with centroid_beam_scale at its 0.5
 * default, flooring the beam at this many candidates — never more
 * than nprobe — reproduces exactly that two-regime shape.
 */
#define PRISM_CENTROID_BEAM_FLOOR 80

#endif /* PRISM_QUERY_SCAN_H */
