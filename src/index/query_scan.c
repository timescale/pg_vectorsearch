/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * query_scan.c - Shared query execution for ANN search
 *
 * Beam search over centroids, posting list scan with RaBitQ
 * scoring, and candidate extraction. All buffers are pre-allocated
 * in init; the per-query path does zero allocations (except rare
 * topk candidate buffer growth).
 */

#include "vs_config.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "algo/topk.h"
#include "algo/vecops.h"
#include "core/injection.h"
#include "core/log.h"
#include "core/memory.h"
#include "core/platform.h"
#include "index/centroid_search.h"
#include "index/posting_page.h"
#include "index/posting_scan.h"
#include "index/query_scan.h"
#include "quant/rabitq.h"

/* ----------------------------------------------------------------
 * Init / cleanup
 * ---------------------------------------------------------------- */

void
prism_query_state_init(
		PrismQueryState *qs,
		PrismIndexBase	*index,
		uint32_t		 max_k,
		uint32_t		 max_nprobe)
{
	memset(qs, 0, sizeof(*qs));
	qs->index	   = index;
	qs->max_k	   = max_k;
	qs->max_nprobe = max_nprobe;

	prism_index_ensure_rabitq(index);

	/*
	 * Everything below is allocated here, including the contexts the top-k
	 * and the centroid scratch create for themselves, so that the whole
	 * state can be released at once. Parented to the caller's current
	 * context, so a caller that never calls cleanup still loses it with
	 * whatever scope it allocated the state in.
	 */
	qs->memctx			= vs_memctx_create(NULL, "vs query state");
	VsMemCtx caller_ctx = vs_memctx_switch(qs->memctx);

	Dimension dim		   = index->dim;
	uint32_t  packed_bytes = VS_RABITQ_BYTES(dim);

	/* Query buffers */
	qs->query_buf			= vs_alloc(dim * sizeof(float));
	qs->pt_query			= vs_alloc_aligned(dim * sizeof(float), 64);
	qs->beam_transformed	= vs_alloc_aligned(dim * sizeof(float), 64);
	qs->cluster_transformed = vs_alloc_aligned(dim * sizeof(float), 64);
	qs->beam_query_bits		= vs_alloc_aligned(packed_bytes, 64);
	qs->cluster_query_bits	= vs_alloc_aligned(packed_bytes, 64);

	/* Wire up query state buffers */
	qs->beam_qs.transformed	   = qs->beam_transformed;
	qs->beam_qs.query_bits	   = qs->beam_query_bits;
	qs->cluster_qs.transformed = qs->cluster_transformed;
	qs->cluster_qs.query_bits  = qs->cluster_query_bits;

	vs_rabitq_init_query_constants(&qs->beam_qs, dim);
	vs_rabitq_init_query_constants(&qs->cluster_qs, dim);

	/* Beam search results + per-scan scratch.
	 * Pre-allocating the scratch here means prism_centroid_beam_search
	 * skips 8 vs_alloc calls and 2 memory-context creations on
	 * every query (the largest remaining source of per-query
	 * allocator traffic after the dedup-gens fix). Sized to the
	 * worst case beam_width == max_nprobe. */
	qs->beam_results	 = vs_alloc(max_nprobe * sizeof(PrismCentroidResult));
	qs->centroid_scratch = prism_centroid_scratch_create(dim, max_nprobe);

	/* Probe-order scratch (exact centroid re-rank of the expanded
	 * probe set; see prism_query_set_probe_expand). */
	qs->probe_dists = vs_alloc(max_nprobe * sizeof(float));
	qs->probe_order = vs_alloc(max_nprobe * sizeof(uint32_t));

	/* Top-K */
	vs_topk_init(&qs->topk, max_k);

	/* Candidate extraction buffer */
	qs->cand_cap   = max_k * PRISM_QUERY_CAND_PER_K;
	qs->candidates = vs_alloc(qs->cand_cap * sizeof(VsTopKEntry));

	/* Result ordering */
	qs->result_order = vs_alloc(qs->cand_cap * sizeof(uint32_t));
	qs->result_dists = vs_alloc(qs->cand_cap * sizeof(Distance));

	/* Posting scan iterator */
	uint32_t max_entries = prism_posting_max_entries(dim);
	prism_posting_scan_init(
			&qs->pscan,
			index->posting_storage,
			index->page_base,
			index->params,
			dim,
			max_entries);

	vs_memctx_switch(caller_ctx);
}

void
prism_query_state_cleanup(PrismQueryState *qs)
{
	if (qs == NULL)
		return;

	/*
	 * The pinned page first: it is the one resource the state holds that is
	 * not memory, and deleting the context below would strand it.
	 */
	prism_posting_scan_cleanup(&qs->pscan);

	/* Sub-contexts before the context that parents them. */
	vs_topk_cleanup(&qs->topk);
	prism_centroid_scratch_free(qs->centroid_scratch);
	qs->centroid_scratch = NULL;

	if (qs->memctx != NULL)
	{
		vs_memctx_delete(qs->memctx);
		qs->memctx = NULL;
	}
}

/* ----------------------------------------------------------------
 * Per-query execution
 * ---------------------------------------------------------------- */

static const float *
prepare_query(PrismQueryState *qs, const float *query)
{
	if (qs->index->metric != DISTANCE_COSINE)
		return query;

	Dimension dim = qs->index->dim;
	memcpy(qs->query_buf, query, dim * sizeof(float));
	float norm = vs_l2_norm(qs->query_buf, dim);
	if (norm > 0.0f)
		vec32_scale(qs->query_buf, 1.0f / norm, qs->query_buf, dim);
	return qs->query_buf;
}

static uint32_t
search_centroids(
		PrismQueryState			 *qs,
		const float				 *qvec,
		uint32_t				  nprobe,
		VsDistanceMode			  mode,
		PrismCentroidSearchStats *beam_stats)
{
	const PrismIndexBase *idx = qs->index;
	Dimension			  dim = idx->dim;

	RaBitQQueryState *rqs = NULL;
	if (idx->centroid_format == PRISM_CENTROID_FMT_RABITQ ||
		idx->centroid_format == PRISM_CENTROID_FMT_FASTSCAN)
	{
		vs_rabitq_init_query_state(
				&qs->beam_qs, qs->pt_query, idx->pt_global_mean, dim, mode);
		rqs = &qs->beam_qs;
	}

	uint32_t beam_w = prism_query_beam_width(
			nprobe, idx->nlist, idx->fan_out, idx->centroid_beam_scale);

	PrismCentroidSearchState search = {
			.qstate		 = rqs,
			.query		 = qvec,
			.storage	 = idx->centroid_storage,
			.beam_width	 = beam_w,
			.nprobe		 = nprobe,
			.dim		 = dim,
			.metric		 = idx->metric,
			.error_scale = idx->centroid_error_scale,
			.scratch	 = qs->centroid_scratch,
			/* NULL on every query path; only the build route sets it. */
			.exact_internal = idx->exact_internal,
	};

	return prism_centroid_beam_search(
			&search,
			idx->first_centroid,
			idx->nlevels,
			qs->beam_results,
			NULL,
			beam_stats);
}

/* Probe-order refinement factor (prism.probe_expand); see query_scan.h.
 * Enabled by default: expansion gains saturate around a factor of 2,
 * so 2.0 captures ~all the recall benefit of exact probe ordering.
 * 1.0 means no expansion (identity). */
static double g_probe_expand = 2.0;

/* Cap on extra routed candidates. At large nprobe a deep scan already
 * covers cluster membership, so ordering refinement adds little while
 * the phase-A cost keeps growing linearly; capping the expansion keeps
 * the overhead bounded (measured to retain nearly all of the recall
 * gain at high nprobe). */
#define PRISM_PROBE_EXPAND_MAX_EXTRA 256

void
prism_query_set_probe_expand(double expand)
{
	g_probe_expand = expand;
}

/*
 * Centroid slots the beam keeps at each intermediate level.
 *
 * Starts at a fraction of nprobe and is then raised by three floors, each
 * of which exists because dropping below it loses leaves outright rather
 * than merely ranking them lower:
 *
 *   - Routing floor. Below PRISM_CENTROID_BEAM_FLOOR a scaled-down beam saves
 *     almost nothing and mis-routes, so the beam covers every probed list.
 *
 *   - Coverage floor. A kept set of beam_w parents exposes at most
 *     beam_w * fan_out leaves, so returning nprobe of them needs
 *     ceil(nprobe / fan_out) parents.
 *
 *   - Probe-everything floor. The coverage floor assumes every child
 *     carries fan_out leaves; an unbalanced tree holds fewer, so once
 *     nprobe covers every leaf the beam keeps whole levels.
 *
 * Called by prism_query_execute and by the cost model, which prices a page
 * read per kept slot per level.
 */
uint32_t
prism_query_beam_width(
		uint32_t nprobe, uint32_t nlist, uint32_t fan_out, double beam_scale)
{
	uint32_t beam_w = (uint32_t)((double)nprobe * beam_scale);

	if (beam_w < 1)
		beam_w = 1;

	uint32_t floor_w = nprobe < PRISM_CENTROID_BEAM_FLOOR
							 ? nprobe
							 : PRISM_CENTROID_BEAM_FLOOR;

	if (beam_w < floor_w)
		beam_w = floor_w;

	if (fan_out > 0)
	{
		floor_w = (nprobe + fan_out - 1) / fan_out;
		if (beam_w < floor_w)
			beam_w = floor_w;
	}

	if (nlist > 0 && nprobe >= nlist && beam_w < nprobe)
		beam_w = nprobe;

	return beam_w;
}

/*
 * Leaf clusters the centroid beam routes to when the scan will read nprobe
 * of them, capped at cap (the most the caller can route to: the scan's beam
 * capacity, or the number of posting lists).
 *
 * Exact centroid formats need no expansion -- their probe order is already
 * correct -- so those route exactly nprobe. Compressed formats route more
 * and let phase A re-rank the wider set on exact distances, keeping the
 * best nprobe of them.
 *
 * Called by prism_query_execute and by the cost model, which prices one head
 * page read per routed cluster.
 */
uint32_t
prism_query_routed_clusters(
		uint32_t nprobe, uint32_t cap, PrismCentroidFormat centroid_format)
{
	if (g_probe_expand <= 1.0 || centroid_format == PRISM_CENTROID_FMT_FLOAT ||
		centroid_format == PRISM_CENTROID_FMT_HALF)
		return nprobe;

	double	 expanded = (double)nprobe * g_probe_expand;
	uint32_t n_route  = (uint32_t)(expanded + 0.5);

	if (n_route > nprobe + PRISM_PROBE_EXPAND_MAX_EXTRA)
		n_route = nprobe + PRISM_PROBE_EXPAND_MAX_EXTRA;
	if (n_route > cap)
		n_route = cap;
	if (n_route < nprobe)
		n_route = nprobe;

	return n_route;
}

/* qsort comparator for probe_order indices by probe_dists (context via
 * a file-static base pointer; the scan path is single-threaded per
 * backend). */
static const float *g_probe_sort_dists;

static int
cmp_probe_order(const void *a, const void *b)
{
	float da = g_probe_sort_dists[*(const uint32_t *)a];
	float db = g_probe_sort_dists[*(const uint32_t *)b];
	if (da < db)
		return -1;
	if (da > db)
		return 1;
	return 0;
}

static void
scan_clusters(
		PrismQueryState			  *qs,
		const PrismCentroidResult *beam_results,
		uint32_t				   n_results,
		uint32_t				   scan_limit,
		VsDistanceMode			   mode,
		VsTopK					  *topk,
		PrismQueryStats			  *stats)
{
	const PrismIndexBase *idx = qs->index;
	Dimension			  dim = idx->dim;

	qs->pscan.storage = idx->posting_storage;

	/*
	 * Warm the cache before the serial per-cluster reads below. Both the
	 * Phase-A re-rank and the scan itself fetch each probed cluster's head
	 * page one at a time; on a cold buffer cache that is a string of
	 * synchronous random reads. Issue async prefetches for every head up
	 * front so the reads overlap. Best-effort and a no-op where the storage
	 * has no prefetch (standalone) or the page is already resident.
	 */
	for (uint32_t j = 0; j < n_results; j++)
		vs_storage_prefetch(qs->pscan.storage, beam_results[j].posting_head);

	/*
	 * Phase A (only when the probe set was expanded): re-rank the routed
	 * clusters by EXACT query-centroid distance. The beam's RaBitQ
	 * distances are 1-bit estimates whose noise scrambles the probe
	 * order; each cluster's first posting page stores the full-precision
	 * rotated centroid, so one page read + one O(dim) distance per
	 * candidate recovers the true order. Only the best `scan_limit`
	 * clusters are then scanned.
	 */
	const uint32_t *order  = NULL;
	uint32_t		n_scan = n_results;

	if (n_results > scan_limit)
	{
		for (uint32_t j = 0; j < n_results; j++)
		{
			qs->probe_order[j] = j;
			qs->probe_dists[j] = FLT_MAX;

			BlockNumber ph = beam_results[j].posting_head;
			if (ph == InvalidBlockNumber)
				continue;

			prism_posting_scan_begin_cluster(&qs->pscan, &qs->cluster_qs, ph);
			const float *pt_cent = prism_posting_scan_pt_centroid(&qs->pscan);
			if (pt_cent != NULL)
				qs->probe_dists[j] =
						vs_l2_distance_squared(qs->pt_query, pt_cent, dim);
			prism_posting_scan_end_cluster(&qs->pscan);
		}

		g_probe_sort_dists = qs->probe_dists;
		qsort(qs->probe_order, n_results, sizeof(uint32_t), cmp_probe_order);

		order  = qs->probe_order;
		n_scan = scan_limit;
	}

	uint32_t total_pages   = 0;
	uint32_t total_skipped = 0;
	uint32_t total_entries = 0;
	uint32_t scanned	   = 0;

	for (uint32_t r = 0; r < n_scan; r++)
	{
		uint32_t	j  = order ? order[r] : r;
		BlockNumber ph = beam_results[j].posting_head;
		if (ph == InvalidBlockNumber)
			continue;

		/* Diagnostic: stamp candidates inserted while scanning this cluster
		 * with its scan rank r (the position in the exact-re-ranked probe
		 * order, not the beam index j), so the deepest contributing rank
		 * measures how many probed clusters the query actually needed. */
		topk->cur_src = r;

		prism_posting_scan_begin_cluster(&qs->pscan, &qs->cluster_qs, ph);

		const float *pt_cent = prism_posting_scan_pt_centroid(&qs->pscan);
		if (pt_cent == NULL)
		{
			prism_posting_scan_end_cluster(&qs->pscan);
			continue;
		}

		vs_rabitq_init_query_state(
				&qs->cluster_qs, qs->pt_query, pt_cent, dim, mode);

		if (idx->fastscan && qs->pscan.fs_lut != NULL)
			prism_posting_scan_cluster_fastscan(&qs->pscan, topk);
		else
			prism_posting_scan_cluster(&qs->pscan, topk);
		total_pages += qs->pscan.pages_read;
		total_skipped += qs->pscan.pages_skipped;
		total_entries += qs->pscan.entries_scanned;
		prism_posting_scan_end_cluster(&qs->pscan);
		scanned++;
	}

	qs->pscan.storage = NULL;

	if (stats != NULL)
	{
		stats->clusters_scanned		   = scanned;
		stats->posting_pages_read	   = total_pages;
		stats->posting_pages_skipped   = total_skipped;
		stats->posting_entries_scanned = total_entries;
	}
}

static uint32_t
extract_candidates(PrismQueryState *qs, uint32_t cap)
{
	if (qs->topk.cand_count > qs->cand_cap)
	{
		/* Grow geometrically so a run of gradually larger queries does
		 * not realloc+copy on every step to the high-water mark. */
		uint32_t want = qs->cand_cap * 2;
		if (want < qs->topk.cand_count)
			want = qs->topk.cand_count;
		qs->cand_cap = want;
		qs->candidates =
				vs_realloc(qs->candidates, qs->cand_cap * sizeof(VsTopKEntry));
		qs->result_order =
				vs_realloc(qs->result_order, qs->cand_cap * sizeof(uint32_t));
		qs->result_dists =
				vs_realloc(qs->result_dists, qs->cand_cap * sizeof(Distance));
	}

	uint32_t ncands;
	vs_topk_extract_sorted_capped(&qs->topk, qs->candidates, &ncands, cap);
	qs->ncandidates = ncands;
	return ncands;
}

/* Monotonic nanosecond clock for per-phase query instrumentation. */
static inline uint64_t
prism_query_now_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * VS_NS_PER_SEC + (uint64_t)ts.tv_nsec;
}

uint32_t
prism_query_route(
		PrismQueryState			 *qs,
		const float				 *query,
		uint32_t				  nprobe,
		VsDistanceMode			  mode,
		PrismCentroidSearchStats *beam_stats)
{
	if (nprobe > qs->max_nprobe)
		nprobe = qs->max_nprobe;

	PrismCentroidSearchStats  local = {0};
	PrismCentroidSearchStats *bs	= beam_stats ? beam_stats : &local;

	const float *qvec  = prepare_query(qs, query);
	uint64_t	 t_rot = prism_query_now_ns();
	vs_rabitq_rotate(qs->index->params, qvec, qs->pt_query);
	bs->rotation_ns = prism_query_now_ns() - t_rot;

	return search_centroids(qs, qvec, nprobe, mode, bs);
}

/* Cap on the rerank candidate pool (prism.rerank_pool). Candidates are
 * sorted by approximate distance, so capping keeps the most promising
 * ones and bounds the exact-distance heap fetches. 0 (default) resolves
 * to an automatic cap of max(3 * k * nprobe^0.15, candidate-buffer count
 * / 8) / rerank_cost_scale: the buffer population directly measures
 * estimate noise, so the nprobe-scaled floor (fit against rekall's
 * cohere-1m recall-vs-pool sweep: the pool needed to keep the
 * rerank-induced recall deficit under 0.1% relative to an unbounded
 * pool, at nprobe in {10,20,40,80,160}, power-law-fits to
 * 30 * nprobe^0.15 for k=10) grows further when noisy estimates flood
 * the buffer and ranking into just that floor would silently cap
 * recall far below what the probed clusters contain. A flat 16 * k
 * floor (this formula's predecessor) was measured recall-neutral, but
 * oversized at every nprobe on that sweep -- e.g. 2-4x more pool than
 * needed below nprobe=80, wasting rerank work without buying recall.
 * -1 disables the cap entirely; positive values are absolute. The
 * effective cap is never below k, so a cap can never truncate the
 * result set.
 *
 * The fit above was measured on a host where the working set (heap +
 * index) fits comfortably in cache, so every rerank candidate is a
 * cheap in-memory fetch -- the formula has no notion of a candidate
 * ever costing more than that. On a host where the working set exceeds
 * available cache, each rerank is a real disk read instead, and the
 * same pool that was "free" on the reference host becomes the
 * dominant query cost for a shrinking marginal recall gain (measured
 * on a 1.8GB/1-vCPU host against a 4.1GB cohere-1m table+index: the
 * auto pool bought 0.0025 recall over a fixed pool of 40 for ~30% more
 * heap I/O and 18% less QPS). rerank_cost_scale (mkt.rerank_cost_scale,
 * see prism_query_set_rerank_cost_scale) lets the caller fold in a
 * relative rerank-cost signal -- 1.0 (default) reproduces the original
 * fit exactly; values above 1.0 shrink the auto pool for hosts where a
 * candidate costs more than the reference assumed. */
#define PRISM_RERANK_POOL_AUTO_COEFF 3.0
#define PRISM_RERANK_POOL_AUTO_EXP	 0.15

/* Recall floor for the cost-scaled auto pool: cost scaling may shrink
 * the pool, but not below 1.2 * k * nprobe^0.3, and never past the
 * unscaled fit. A uniform divisor of the nprobe^0.15 fit cannot work:
 * the pool the recall ceiling needs grows with nprobe (measured on
 * cohere-1m 768d: ~42 at nprobe=10 to ~79 at nprobe=640, i.e.
 * nprobe^0.15), so any scale large enough to help at low nprobe closes
 * the >= 0.995 recall region at high nprobe -- and probes cannot
 * substitute, since the true neighbours a small pool drops rank too
 * poorly by approximate distance to survive truncation no matter how
 * many lists are scanned. The 0.3 exponent grows faster than the need,
 * so the floor stays out of the way at low nprobe (where wider probing
 * is the cheaper recall currency and scaling wins) and only binds in
 * the high-recall regime, where the scan already dominates query cost
 * and a truncated pool throws away recall the scan paid for. The 1.2
 * coefficient puts the floor above the fit from nprobe ~450 up, i.e.
 * deep probes always rerank the full fit. At scale=1 the floor never
 * exceeds the base fit, reproducing the original formula exactly. */
#define PRISM_RERANK_POOL_RECALL_COEFF 1.2
#define PRISM_RERANK_POOL_RECALL_EXP   0.3

/* Floor for prism_query_set_rerank_cost_scale: guards against a
 * misconfigured near-zero scale blowing the auto pool up toward
 * "unbounded" through the division below. */
#define PRISM_RERANK_COST_SCALE_MIN 0.01

static int32_t g_rerank_pool	   = 0;
static double  g_rerank_cost_scale = 1.0;

void
prism_query_set_rerank_pool(int32_t n)
{
	g_rerank_pool = n;
}

/*
 * Calculates the size of the rerank pool from k, the number of neighbours
 * the query will return, and nprobe, the number of posting lists it will
 * scan -- as far as that can be known without running the scan.
 *
 * A return of 0 means the pool is uncapped: it is the value
 * vs_topk_extract_sorted_capped reads as "keep every survivor", so 0 is
 * the widest possible pool and not the narrowest. Callers that price the
 * pool have to special-case it.
 *
 * Shared with the cost model, which has to price the fetches the scan will
 * actually make. The scan adds one term this cannot: a floor at an eighth of
 * the candidate buffer, which measures estimate noise and so does not exist
 * until the clusters have been scanned.
 */
uint32_t
prism_query_rerank_pool_estimate(uint32_t k, uint32_t nprobe)
{
	if (g_rerank_pool < 0)
		return 0; /* uncapped: every survivor is reranked */

	if (g_rerank_pool > 0)
	{
		uint32_t pool = (uint32_t)g_rerank_pool;

		return pool < k ? k : pool;
	}

	double base = PRISM_RERANK_POOL_AUTO_COEFF * (double)k *
				  pow((double)nprobe, PRISM_RERANK_POOL_AUTO_EXP);

	/* Cost scaling may shrink the base, but not below the recall floor;
	 * the floor in turn never exceeds the unscaled fit (see
	 * PRISM_RERANK_POOL_RECALL_COEFF), so scale=1 is exactly the
	 * original formula. */
	double pool			= base / g_rerank_cost_scale;
	double recall_floor = PRISM_RERANK_POOL_RECALL_COEFF * (double)k *
						  pow((double)nprobe, PRISM_RERANK_POOL_RECALL_EXP);

	if (recall_floor > base)
		recall_floor = base;
	if (pool < recall_floor)
		pool = recall_floor;

	uint32_t capped = (uint32_t)(pool + 0.5);

	return capped < k ? k : capped;
}

void
prism_query_set_rerank_cost_scale(double scale)
{
	g_rerank_cost_scale = scale < PRISM_RERANK_COST_SCALE_MIN
								? PRISM_RERANK_COST_SCALE_MIN
								: scale;
}

uint32_t
prism_query_execute(
		PrismQueryState *qs,
		const float		*query,
		uint32_t		 k,
		uint32_t		 nprobe,
		VsDistanceMode	 mode,
		bool			 rerank,
		PrismQueryStats *stats)
{
	if (k > qs->max_k)
		k = qs->max_k;
	if (nprobe > qs->max_nprobe)
		nprobe = qs->max_nprobe;

	/* reset_to_k also (re)sizes the heap when k grows between queries —
	 * resetting first and assigning k afterwards left the heap sized for
	 * the previous k. */
	vs_topk_reset_to_k(&qs->topk, k);

	/* Probe expansion: route extra leaf candidates so phase A of
	 * scan_clusters can pick the best `nprobe` by exact centroid
	 * distance. n_route == nprobe (expand == 1, no expansion) keeps
	 * the classic single-phase behavior. Skipped entirely when the
	 * centroid pages are exact (float/half): the beam distances are
	 * already exact, so there is no ordering noise to correct. */
	uint32_t n_route = prism_query_routed_clusters(
			nprobe, qs->max_nprobe, qs->index->centroid_format);

	uint64_t t0 = prism_query_now_ns();

	PrismCentroidSearchStats beam_stats = {0};
	uint32_t				 ncentroids =
			prism_query_route(qs, query, n_route, mode, &beam_stats);

	/* prism_query_route already normalized the query into qs->query_buf (for
	 * cosine) via prepare_query; reuse it for the rerank below instead of
	 * re-normalizing (a redundant O(dim) memcpy+norm+scale per query). For
	 * non-cosine metrics prepare_query is a no-op and returns the raw query.
	 */
	const float *qvec = (qs->index->metric == DISTANCE_COSINE) ? qs->query_buf
															   : query;

	uint64_t t1 = prism_query_now_ns();

	/*
	 * The probed heads are chosen and their centroid pages released, but none
	 * of the lists has been opened yet. A concurrent split can replace and
	 * retire a head in this window, which is why the old chain stays readable
	 * until no snapshot can reach it; an isolation test pauses here to hold a
	 * head across exactly that.
	 */
	VS_INJECTION_POINT("prism-scan-routed");

	scan_clusters(
			qs, qs->beam_results, ncentroids, nprobe, mode, &qs->topk, stats);

	/* Rerank-pool cap: the first `pool` candidates by approximate
	 * distance are the most promising; see prism_query_set_rerank_pool.
	 * Resolved before extraction so the extract can select the capped
	 * prefix instead of fully sorting an unbounded survivor set.
	 *
	 * The automatic cap also scales with the candidate-buffer
	 * population, which directly measures estimate noise: accurate
	 * estimates keep the threshold at the nprobe-scaled floor above,
	 * while noisy estimates (low dimension, wide norm spread) flood
	 * the buffer -- and then ranking into just that floor is
	 * meaningless, silently capping recall well below what the probed
	 * clusters contain. 1/8th of the buffer restores the recall
	 * ceiling at a rerank cost proportionate to the observed noise. */
	uint32_t pool = prism_query_rerank_pool_estimate(k, nprobe);

	/*
	 * The noise term needs the candidate population, which exists only now
	 * that the clusters have been scanned -- so it cannot be part of the
	 * shared estimate the planner uses. It is scaled down like the base
	 * fit when candidates cost more than cache-resident (see
	 * prism_query_set_rerank_cost_scale); the estimate's recall floor is
	 * unaffected by the noise term either way, since the max() here can
	 * only raise the pool.
	 */
	if (g_rerank_pool == 0)
	{
		uint32_t noise = (uint32_t)((double)(qs->topk.cand_count / 8) /
											g_rerank_cost_scale +
									0.5);

		if (pool < noise)
			pool = noise;
	}
	if (pool > 0 && pool < k)
		pool = k;

	uint32_t ncands = extract_candidates(qs, pool);

	uint64_t t2 = prism_query_now_ns();

	/* Rerank with exact distances if enabled and storage supports it */
	VsStorage *ps = qs->index->posting_storage;
	if (rerank && ncands > 0 && ps != NULL && ps->ops->rerank != NULL)
	{
		qs->nresults = vs_storage_rerank(
				ps,
				qvec,
				qs->index->dim,
				qs->candidates,
				ncands,
				k,
				qs->result_order,
				qs->result_dists);
	}
	else
	{
		qs->nresults = ncands < k ? ncands : k;
		for (uint32_t i = 0; i < qs->nresults; i++)
		{
			qs->result_order[i] = i;
			qs->result_dists[i] = qs->candidates[i].distance;
		}
	}

#ifndef NDEBUG
	for (uint32_t i = 0; i < qs->nresults; i++)
	{
		uint64_t id_i = qs->candidates[qs->result_order[i]].id;
		for (uint32_t j = i + 1; j < qs->nresults; j++)
			if (id_i == qs->candidates[qs->result_order[j]].id)
				vs_warn(VS_EXTENSION_NAME ": duplicate result at positions "
										  "%u and %u",
						i,
						j);
	}
#endif

	uint64_t t3 = prism_query_now_ns();

	/* Routing-quality diagnostic: deepest probe rank contributing a final
	 * top-k result. Low values (relative to nprobe) => over-probing; values
	 * near nprobe => neighbors genuinely routed deep (mis-routing). */
	uint32_t max_rank = 0;
	for (uint32_t i = 0; i < qs->nresults; i++)
	{
		uint32_t r = qs->candidates[qs->result_order[i]].src;
		if (r > max_rank)
			max_rank = r;
	}

	if (stats != NULL)
	{
		/* clusters_scanned is set by scan_clusters (actual count). */
		stats->max_contrib_rank		= max_rank;
		stats->centroid_pages_read	= beam_stats.pages_read;
		stats->centroid_ns			= t1 - t0;
		stats->posting_ns			= t2 - t1;
		stats->rerank_ns			= t3 - t2;
		stats->rotation_ns			= beam_stats.rotation_ns;
		stats->centroid_lut_ns		= beam_stats.lut_ns;
		stats->centroid_pageread_ns = beam_stats.pageread_ns;
		stats->centroid_score_ns	= beam_stats.score_ns;
	}

	return qs->nresults;
}

uint32_t
prism_auto_nprobe(uint32_t nlist)
{
	/* See the header: ~0.5 * sqrt(nlist), floored at 10, capped at
	 * 2048, never above nlist. */
	uint32_t nprobe = (uint32_t)ceil(0.5 * sqrt((double)nlist));
	if (nprobe < 10)
		nprobe = 10;
	if (nprobe > 2048)
		nprobe = 2048;
	if (nlist > 0 && nprobe > nlist)
		nprobe = nlist;
	return nprobe;
}
