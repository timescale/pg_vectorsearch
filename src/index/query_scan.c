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

/* Probe-order controls (mkt.probe_expand / mkt.probe_patience /
 * mkt.probe_cutoff); see query_scan.h. Defaults preserve the classic
 * single-phase scan. */
static double	g_probe_expand	 = 1.0;
static uint32_t g_probe_patience = 0;
static double	g_probe_cutoff	 = 0.0;

void
mkt_query_set_probe_expand(double expand)
{
	g_probe_expand = expand;
}

void
mkt_query_set_probe_patience(uint32_t patience)
{
	g_probe_patience = patience;
}

void
mkt_query_set_probe_cutoff(double cutoff)
{
	g_probe_cutoff = cutoff;
}

/* Rank-based adaptive termination (mkt.probe_beta, 0 = off): stop at
 * rank max(floor, beta * last rank that improved the top-k). Easy
 * queries whose results settle early stop early; hard queries keep
 * extending their own deadline. Scale-free — no dependence on absolute
 * distance magnitudes (unlike probe_cutoff). */
static double	g_probe_beta	  = 0.0;
static uint32_t g_probe_min_scan = 32;

void
mkt_query_set_probe_beta(double beta, uint32_t min_scan)
{
	g_probe_beta	 = beta;
	g_probe_min_scan = min_scan;
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

	/* Enable TID dedup if caller set up a hash set.
	 * Bump generation instead of memset — O(1) reset. */
	if (qs->dedup_set != NULL && n_results > 1)
	{
		qs->dedup_gen++;
		qs->pscan.seen_tids		= qs->dedup_set;
		qs->pscan.seen_gens		= qs->dedup_gens;
		qs->pscan.seen_tids_cap = qs->dedup_cap;
		qs->pscan.seen_gen		= qs->dedup_gen;
	}
	else
	{
		qs->pscan.seen_tids		= NULL;
		qs->pscan.seen_tids_cap = 0;
	}

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
				qs->probe_dists[j] = mkt_l2_distance_squared(
						qs->pt_query, pt_cent, dim);
			mkt_posting_scan_end_cluster(&qs->pscan);
		}

		g_probe_sort_dists = qs->probe_dists;
		qsort(qs->probe_order,
			  n_results,
			  sizeof(uint32_t),
			  cmp_probe_order);

		order  = qs->probe_order;
		n_scan = scan_limit;
	}

	uint32_t total_pages   = 0;
	uint32_t total_skipped = 0;
	uint32_t total_entries = 0;
	uint32_t scanned	   = 0;

	/* Phase B: scan clusters in probe order, optionally stopping early
	 * once `patience` consecutive clusters fail to improve the top-k
	 * threshold (checked only while the top-k heap is full). */
	uint32_t stall		  = 0;
	uint32_t last_improve = 0; /* last rank that improved the top-k */

	for (uint32_t r = 0; r < n_scan; r++)
	{
		/* Rank-based adaptive termination: stop once r exceeds both the
		 * floor and beta * (last rank that improved the top-k). */
		if (g_probe_beta > 0.0 && r >= g_probe_min_scan &&
			(double)r > g_probe_beta * (double)(last_improve + 1))
			break;

		uint32_t	j  = order ? order[r] : r;
		BlockNumber ph = beam_results[j].posting_head;
		if (ph == InvalidBlockNumber)
			continue;

		/* Diagnostic: stamp candidates inserted while scanning this cluster
		 * with its probe rank r, so we can measure how deep in the probe
		 * order the final top-k results actually came from. */
		topk->cur_src = r;

		bool	 heap_was_full = (topk->ub_count == topk->k);
		Distance prev_thresh   = heap_was_full ? topk->ub_heap[0] : 0.0f;

		/* Distance-based early termination: once the top-k heap is full,
		 * stop when the next cluster's EXACT centroid distance exceeds
		 * cutoff * (current k-th best distance). Clusters are visited in
		 * ascending exact centroid distance (order != NULL), so every
		 * later cluster is at least this far away. Requires phase A
		 * (probe_expand > 1) for the exact distances. */
		if (g_probe_cutoff > 0.0 && order != NULL && heap_was_full &&
			qs->probe_dists[j] > (float)g_probe_cutoff * prev_thresh)
			break;

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

		bool improved = !heap_was_full || topk->ub_heap[0] != prev_thresh;
		if (improved)
			last_improve = r;

		if (g_probe_patience > 0)
		{
			if (!improved)
			{
				if (++stall >= g_probe_patience)
					break;
			}
			else
				stall = 0;
		}
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
		qs->cand_cap   = qs->topk.cand_count;
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
	return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
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

/* Optional cap on the rerank candidate pool (0 = rerank all survivors).
 * Set via the mkt.rerank_pool GUC. Candidates are sorted by approximate
 * distance, so capping keeps the most promising ones and cuts the
 * exact-distance heap fetches — a large fixed per-query cost at low nprobe. */
static uint32_t g_rerank_pool = 0;

void
mkt_query_set_rerank_pool(uint32_t n)
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

	mkt_topk_reset(&qs->topk);
	qs->topk.k = k;

	uint64_t t0 = mkt_query_now_ns();

	/* Probe expansion: route extra leaf candidates so phase A of
	 * scan_clusters can pick the best `nprobe` by exact centroid
	 * distance. n_route == nprobe (expand <= 1) keeps the classic
	 * single-phase behavior. */
	uint32_t n_route = nprobe;
	if (g_probe_expand > 1.0)
	{
		double expanded = (double)nprobe * g_probe_expand;
		n_route			= (uint32_t)(expanded + 0.5);
		if (n_route > qs->max_nprobe)
			n_route = qs->max_nprobe;
		if (n_route < nprobe)
			n_route = nprobe;
	}

	MktCentroidSearchStats beam_stats = {0};
	uint32_t			   ncentroids =
			mkt_query_route(qs, query, n_route, mode, &beam_stats);

	/* mkt_query_route already normalized the query into qs->query_buf (for
	 * cosine) via prepare_query; reuse it for the rerank below instead of
	 * re-normalizing (a redundant O(dim) memcpy+norm+scale per query). For
	 * non-cosine metrics prepare_query is a no-op and returns the raw query. */
	const float *qvec =
			(qs->index->metric == DISTANCE_COSINE) ? qs->query_buf : query;

	uint64_t t1 = mkt_query_now_ns();

	scan_clusters(
			qs, qs->beam_results, ncentroids, nprobe, mode, &qs->topk, stats);

	uint32_t ncands = extract_candidates(qs);

	/* Optional rerank-pool cap: candidates are sorted by approximate
	 * distance, so the first g_rerank_pool are the most promising. Capping
	 * cuts exact-distance heap fetches at a small recall risk. */
	if (g_rerank_pool > 0 && ncands > g_rerank_pool)
		ncands = g_rerank_pool;

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
		/* clusters_scanned is set by scan_clusters (actual count, which
		 * can be below nprobe under patience early-exit). */
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
