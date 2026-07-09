/*
 * query_scan.c - Shared query execution for ANN search
 *
 * Beam search over centroids, posting list scan with RaBitQ
 * scoring, and candidate extraction. All buffers are pre-allocated
 * in init; the per-query path does zero allocations (except rare
 * topk candidate buffer growth).
 */

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "algo/topk.h"
#include "algo/vecops.h"
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
mkt_query_state_init(
		MktQueryState *qs,
		MktIndexBase  *index,
		uint32_t	   max_k,
		uint32_t	   max_nprobe)
{
	memset(qs, 0, sizeof(*qs));
	qs->index	   = index;
	qs->max_k	   = max_k;
	qs->max_nprobe = max_nprobe;

	mkt_index_ensure_rabitq(index);

	Dimension dim		   = index->dim;
	uint32_t  packed_bytes = MKT_RABITQ_BYTES(dim);

	/* Query buffers */
	qs->query_buf			= mkt_alloc(dim * sizeof(float));
	qs->pt_query			= mkt_alloc_aligned(dim * sizeof(float), 64);
	qs->beam_transformed	= mkt_alloc_aligned(dim * sizeof(float), 64);
	qs->cluster_transformed = mkt_alloc_aligned(dim * sizeof(float), 64);
	qs->beam_query_bits		= mkt_alloc_aligned(packed_bytes, 64);
	qs->cluster_query_bits	= mkt_alloc_aligned(packed_bytes, 64);

	/* Wire up query state buffers */
	qs->beam_qs.transformed	   = qs->beam_transformed;
	qs->beam_qs.query_bits	   = qs->beam_query_bits;
	qs->cluster_qs.transformed = qs->cluster_transformed;
	qs->cluster_qs.query_bits  = qs->cluster_query_bits;

	mkt_rabitq_init_query_constants(&qs->beam_qs, dim);
	mkt_rabitq_init_query_constants(&qs->cluster_qs, dim);

	/* Beam search results + per-scan scratch.
	 * Pre-allocating the scratch here means mkt_centroid_beam_search
	 * skips 8 mkt_alloc calls and 2 memory-context creations on
	 * every query (the largest remaining source of per-query
	 * allocator traffic after the dedup-gens fix). Sized to the
	 * worst case beam_width == max_nprobe. */
	qs->beam_results	 = mkt_alloc(max_nprobe * sizeof(MktCentroidResult));
	qs->centroid_scratch = mkt_centroid_scratch_create(dim, max_nprobe);

	/* Probe-order scratch (exact centroid re-rank of the expanded
	 * probe set; see mkt_query_set_probe_expand). */
	qs->probe_dists = mkt_alloc(max_nprobe * sizeof(float));
	qs->probe_order = mkt_alloc(max_nprobe * sizeof(uint32_t));

	/* Top-K */
	mkt_topk_init(&qs->topk, max_k);

	/* Candidate extraction buffer */
	qs->cand_cap   = max_k * 16;
	qs->candidates = mkt_alloc(qs->cand_cap * sizeof(MktTopKEntry));

	/* Result ordering */
	qs->result_order = mkt_alloc(qs->cand_cap * sizeof(uint32_t));
	qs->result_dists = mkt_alloc(qs->cand_cap * sizeof(Distance));

	/* Posting scan iterator */
	uint32_t max_entries = mkt_posting_max_entries(dim);
	mkt_posting_scan_init(
			&qs->pscan,
			index->posting_storage,
			index->page_base,
			index->params,
			dim,
			max_entries);
}

void
mkt_query_state_cleanup(MktQueryState *qs)
{
	if (qs == NULL)
		return;

	mkt_posting_scan_cleanup(&qs->pscan);
	mkt_topk_cleanup(&qs->topk);
	mkt_centroid_scratch_free(qs->centroid_scratch);
	qs->centroid_scratch = NULL;
}

/* ----------------------------------------------------------------
 * Per-query execution
 * ---------------------------------------------------------------- */

static const float *
prepare_query(MktQueryState *qs, const float *query)
{
	if (qs->index->metric != DISTANCE_COSINE)
		return query;

	Dimension dim = qs->index->dim;
	memcpy(qs->query_buf, query, dim * sizeof(float));
	float norm = mkt_l2_norm(qs->query_buf, dim);
	if (norm > 0.0f)
		mkt_vector_scale(qs->query_buf, 1.0f / norm, qs->query_buf, dim);
	return qs->query_buf;
}

static uint32_t
search_centroids(
		MktQueryState		   *qs,
		const float			   *qvec,
		uint32_t				nprobe,
		MktDistanceMode			mode,
		MktCentroidSearchStats *beam_stats)
{
	const MktIndexBase *idx = qs->index;
	Dimension			dim = idx->dim;

	RaBitQQueryState *rqs = NULL;
	if (idx->centroid_format == MKT_CENTROID_FMT_RABITQ ||
		idx->centroid_format == MKT_CENTROID_FMT_FASTSCAN)
	{
		mkt_rabitq_init_query_state(
				&qs->beam_qs, qs->pt_query, idx->pt_global_mean, dim, mode);
		rqs = &qs->beam_qs;
	}

	uint32_t beam_w = (uint32_t)(nprobe * idx->centroid_beam_scale);
	if (beam_w < 1)
		beam_w = 1;

	/* Coverage floor: an intermediate keep of beam_w exposes at most
	 * beam_w * fan_out leaves, so returning the top nprobe leaves needs
	 * beam_w >= ceil(nprobe / fan_out) -- below that, whole subtrees are
	 * unreachable at ANY nprobe, not just ranked lower. */
	if (idx->fan_out > 0)
	{
		uint32_t floor_w = (nprobe + idx->fan_out - 1) / idx->fan_out;
		if (beam_w < floor_w)
			beam_w = floor_w;
	}

	/* Probe-everything floor: the fan_out floor above assumes each child
	 * carries fan_out leaves, but an unbalanced tree can hold fewer — a
	 * kept set of ceil(nprobe / fan_out) children then exposes fewer than
	 * nprobe leaves and prunes whole subtrees that can never be reached at
	 * ANY nprobe. When nprobe covers every leaf, keep whole levels (a
	 * level's entry count never exceeds nlist, so beam_w = nprobe bounds
	 * it) — probing everything must reach everything. */
	if (idx->nlist > 0 && nprobe >= idx->nlist && beam_w < nprobe)
		beam_w = nprobe;

	MktCentroidSearchState search = {
			.qstate		 = rqs,
			.query		 = qvec,
			.storage	 = idx->centroid_storage,
			.beam_width	 = beam_w,
			.nprobe		 = nprobe,
			.dim		 = dim,
			.metric		 = idx->metric,
			.error_scale = idx->centroid_error_scale,
			.scratch	 = qs->centroid_scratch,
	};

	return mkt_centroid_beam_search(
			&search,
			idx->first_centroid,
			idx->nlevels,
			qs->beam_results,
			NULL,
			beam_stats);
}

/* Probe-order refinement factor (mkt.probe_expand); see query_scan.h.
 * Enabled by default: expansion gains saturate around a factor of 2,
 * so 2.0 captures ~all the recall benefit of exact probe ordering.
 * 1.0 means no expansion (identity). */
static double g_probe_expand = 2.0;

/* Cap on extra routed candidates. At large nprobe a deep scan already
 * covers cluster membership, so ordering refinement adds little while
 * the phase-A cost keeps growing linearly; capping the expansion keeps
 * the overhead bounded (measured to retain nearly all of the recall
 * gain at high nprobe). */
#define MKT_PROBE_EXPAND_MAX_EXTRA 256

void
mkt_query_set_probe_expand(double expand)
{
	g_probe_expand = expand;
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
		MktQueryState			*qs,
		const MktCentroidResult *beam_results,
		uint32_t				 n_results,
		uint32_t				 scan_limit,
		MktDistanceMode			 mode,
		MktTopK					*topk,
		MktQueryStats			*stats)
{
	const MktIndexBase *idx = qs->index;
	Dimension			dim = idx->dim;

	qs->pscan.storage = idx->posting_storage;

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

			mkt_posting_scan_begin_cluster(&qs->pscan, &qs->cluster_qs, ph);
			const float *pt_cent = mkt_posting_scan_pt_centroid(&qs->pscan);
			if (pt_cent != NULL)
				qs->probe_dists[j] =
						mkt_l2_distance_squared(qs->pt_query, pt_cent, dim);
			mkt_posting_scan_end_cluster(&qs->pscan);
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

		mkt_posting_scan_begin_cluster(&qs->pscan, &qs->cluster_qs, ph);

		const float *pt_cent = mkt_posting_scan_pt_centroid(&qs->pscan);
		if (pt_cent == NULL)
		{
			mkt_posting_scan_end_cluster(&qs->pscan);
			continue;
		}

		mkt_rabitq_init_query_state(
				&qs->cluster_qs, qs->pt_query, pt_cent, dim, mode);

		if (idx->fastscan && qs->pscan.fs_lut != NULL)
			mkt_posting_scan_cluster_fastscan(&qs->pscan, topk);
		else
			mkt_posting_scan_cluster(&qs->pscan, topk);
		total_pages += qs->pscan.pages_read;
		total_skipped += qs->pscan.pages_skipped;
		total_entries += qs->pscan.entries_scanned;
		mkt_posting_scan_end_cluster(&qs->pscan);
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
extract_candidates(MktQueryState *qs)
{
	if (qs->topk.cand_count > qs->cand_cap)
	{
		/* Grow geometrically so a run of gradually larger queries does
		 * not realloc+copy on every step to the high-water mark. */
		uint32_t want = qs->cand_cap * 2;
		if (want < qs->topk.cand_count)
			want = qs->topk.cand_count;
		qs->cand_cap   = want;
		qs->candidates = mkt_realloc(
				qs->candidates, qs->cand_cap * sizeof(MktTopKEntry));
		qs->result_order =
				mkt_realloc(qs->result_order, qs->cand_cap * sizeof(uint32_t));
		qs->result_dists =
				mkt_realloc(qs->result_dists, qs->cand_cap * sizeof(Distance));
	}

	uint32_t ncands;
	mkt_topk_extract_sorted(&qs->topk, qs->candidates, &ncands);
	qs->ncandidates = ncands;
	return ncands;
}

/* Monotonic nanosecond clock for per-phase query instrumentation. */
static inline uint64_t
mkt_query_now_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * MKT_NS_PER_SEC + (uint64_t)ts.tv_nsec;
}

uint32_t
mkt_query_route(
		MktQueryState		   *qs,
		const float			   *query,
		uint32_t				nprobe,
		MktDistanceMode			mode,
		MktCentroidSearchStats *beam_stats)
{
	if (nprobe > qs->max_nprobe)
		nprobe = qs->max_nprobe;

	MktCentroidSearchStats	local = {0};
	MktCentroidSearchStats *bs	  = beam_stats ? beam_stats : &local;

	const float *qvec  = prepare_query(qs, query);
	uint64_t	 t_rot = mkt_query_now_ns();
	mkt_rabitq_rotate(qs->index->params, qvec, qs->pt_query);
	bs->rotation_ns = mkt_query_now_ns() - t_rot;

	return search_centroids(qs, qvec, nprobe, mode, bs);
}

/* Cap on the rerank candidate pool (mkt.rerank_pool). Candidates are
 * sorted by approximate distance, so capping keeps the most promising
 * ones and bounds the exact-distance heap fetches. 0 (default) resolves
 * to an automatic cap of 16 * k — measured recall-neutral across the
 * probe range while bounding pathological survivor counts; -1 disables
 * the cap entirely; positive values are absolute. The effective cap is
 * never below k, so a cap can never truncate the result set. */
#define MKT_RERANK_POOL_AUTO_MULT 16

static int32_t g_rerank_pool = 0;

void
mkt_query_set_rerank_pool(int32_t n)
{
	g_rerank_pool = n;
}

uint32_t
mkt_query_execute(
		MktQueryState  *qs,
		const float	   *query,
		uint32_t		k,
		uint32_t		nprobe,
		MktDistanceMode mode,
		bool			rerank,
		MktQueryStats  *stats)
{
	if (k > qs->max_k)
		k = qs->max_k;
	if (nprobe > qs->max_nprobe)
		nprobe = qs->max_nprobe;

	/* reset_to_k also (re)sizes the heap when k grows between queries —
	 * resetting first and assigning k afterwards left the heap sized for
	 * the previous k. */
	mkt_topk_reset_to_k(&qs->topk, k);

	/* Probe expansion: route extra leaf candidates so phase A of
	 * scan_clusters can pick the best `nprobe` by exact centroid
	 * distance. n_route == nprobe (expand == 1, no expansion) keeps
	 * the classic single-phase behavior. Skipped entirely when the
	 * centroid pages are exact (float/half): the beam distances are
	 * already exact, so there is no ordering noise to correct. */
	uint32_t n_route = nprobe;
	if (g_probe_expand > 1.0 &&
		qs->index->centroid_format != MKT_CENTROID_FMT_FLOAT &&
		qs->index->centroid_format != MKT_CENTROID_FMT_HALF)
	{
		double expanded = (double)nprobe * g_probe_expand;
		n_route			= (uint32_t)(expanded + 0.5);
		if (n_route > nprobe + MKT_PROBE_EXPAND_MAX_EXTRA)
			n_route = nprobe + MKT_PROBE_EXPAND_MAX_EXTRA;
		if (n_route > qs->max_nprobe)
			n_route = qs->max_nprobe;
		if (n_route < nprobe)
			n_route = nprobe;
	}

	uint64_t t0 = mkt_query_now_ns();

	MktCentroidSearchStats beam_stats = {0};
	uint32_t			   ncentroids =
			mkt_query_route(qs, query, n_route, mode, &beam_stats);

	/* mkt_query_route already normalized the query into qs->query_buf (for
	 * cosine) via prepare_query; reuse it for the rerank below instead of
	 * re-normalizing (a redundant O(dim) memcpy+norm+scale per query). For
	 * non-cosine metrics prepare_query is a no-op and returns the raw query.
	 */
	const float *qvec = (qs->index->metric == DISTANCE_COSINE) ? qs->query_buf
															   : query;

	uint64_t t1 = mkt_query_now_ns();

	scan_clusters(
			qs, qs->beam_results, ncentroids, nprobe, mode, &qs->topk, stats);

	uint32_t ncands = extract_candidates(qs);

	/* Rerank-pool cap: the first `pool` candidates by approximate
	 * distance are the most promising; see mkt_query_set_rerank_pool. */
	if (g_rerank_pool >= 0)
	{
		uint32_t pool = (g_rerank_pool == 0) ? MKT_RERANK_POOL_AUTO_MULT * k
											 : (uint32_t)g_rerank_pool;
		if (pool < k)
			pool = k;
		if (ncands > pool)
			ncands = pool;
	}

	uint64_t t2 = mkt_query_now_ns();

	/* Rerank with exact distances if enabled and storage supports it */
	MktStorage *ps = qs->index->posting_storage;
	if (rerank && ncands > 0 && ps != NULL && ps->ops->rerank != NULL)
	{
		qs->nresults = mkt_storage_rerank(
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
				mkt_warn(
						"meerkat: duplicate result at positions "
						"%u and %u",
						i,
						j);
	}
#endif

	uint64_t t3 = mkt_query_now_ns();

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
