/*
 * query_scan.c - Shared query execution for ANN search
 *
 * Beam search over centroids, posting list scan with RaBitQ
 * scoring, and candidate extraction. All buffers are pre-allocated
 * in init; the per-query path does zero allocations (except rare
 * topk candidate buffer growth).
 */

#include <math.h>
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

/* Per-phase timing is opt-in (mkt.profile). Off by default so the hot
 * query path makes zero clock_gettime calls. */
static bool g_mkt_profile = false;

void
mkt_query_set_profile(bool enabled)
{
	g_mkt_profile = enabled;
}

static inline uint64_t
mkt_now_ns(void)
{
	if (!g_mkt_profile)
		return 0;
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

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

	MktCentroidSearchState search = {
			.qstate		= rqs,
			.query		= qvec,
			.storage	= idx->centroid_storage,
			.beam_width = nprobe,
			.nprobe		= nprobe,
			.dim		= dim,
			.metric		= idx->metric,
			.route_ip	= idx->route_ip,
			.scratch	= qs->centroid_scratch,
	};

	return mkt_centroid_beam_search(
			&search,
			idx->first_centroid,
			idx->nlevels,
			qs->beam_results,
			NULL,
			beam_stats);
}

static uint32_t rerank_beam(
		MktQueryState *qs, uint32_t ncentroids, uint32_t nprobe);

/*
 * Run only the centroid beam search for a query and return the selected
 * clusters' posting-head block numbers (the clusters the full scan would
 * read). Diagnostic entry point for routing introspection — mirrors the
 * front half of mkt_query_execute without scanning/reranking.
 */
uint32_t
mkt_query_scanned_heads(
		MktQueryState  *qs,
		const float	   *query,
		uint32_t		nprobe,
		MktDistanceMode mode,
		BlockNumber	   *out_heads,
		uint32_t		out_cap)
{
	if (nprobe > qs->max_nprobe)
		nprobe = qs->max_nprobe;

	const float *qvec = prepare_query(qs, query);
	mkt_rabitq_rotate(qs->index->params, qvec, qs->pt_query);

	/* Two-stage routing (mkt.centroid_rerank): widen the beam, then re-rank
	 * the shortlist by exact centroid distance. Mirrors mkt_query_execute so
	 * the diagnostic measures what the scan actually selects. */
	uint32_t factor = (uint32_t)qs->index->centroid_rerank;
	uint32_t beam_n = nprobe;
	if (factor > 1)
	{
		beam_n = factor * nprobe;
		if (beam_n > qs->max_nprobe)
			beam_n = qs->max_nprobe;
	}

	MktCentroidSearchStats beam_stats = {0};
	uint32_t			   ncentroids =
			search_centroids(qs, qvec, beam_n, mode, &beam_stats);
	if (factor > 1 && ncentroids > nprobe)
		ncentroids = rerank_beam(qs, ncentroids, nprobe);

	uint32_t m = 0;
	for (uint32_t i = 0; i < ncentroids && m < out_cap; i++)
	{
		BlockNumber ph = qs->beam_results[i].posting_head;
		if (ph != InvalidBlockNumber)
			out_heads[m++] = ph;
	}
	return m;
}

static int
cmp_centroid_result(const void *a, const void *b)
{
	Distance da = ((const MktCentroidResult *)a)->distance;
	Distance db = ((const MktCentroidResult *)b)->distance;
	return (da > db) - (da < db);
}

/*
 * Two-stage centroid routing. The (fast, compressed) beam already filled
 * qs->beam_results[0..ncentroids) with a shortlist. Re-rank that shortlist
 * by the EXACT full-precision query-centroid distance — read each cluster's
 * stored centroid (P^T*c on the posting head) and use the rotated query, so
 * <pt_query, pt_centroid> = <q,c> exactly — then keep the best nprobe. This
 * recovers float-centroid routing accuracy (compressed centroids mis-rank
 * boundary clusters) while the beam itself stays cheap. Returns the new
 * count (min(ncentroids, nprobe)).
 */
static uint32_t
rerank_beam(
		MktQueryState *qs,
		uint32_t	   ncentroids,
		uint32_t	   nprobe)
{
	const MktIndexBase *idx = qs->index;
	Dimension			dim = idx->dim;

	qs->pscan.storage = idx->posting_storage;
	float norm_q = (idx->metric == DISTANCE_COSINE)
						 ? mkt_l2_norm(qs->pt_query, dim)
						 : 0.0f;

	for (uint32_t j = 0; j < ncentroids; j++)
	{
		BlockNumber ph = qs->beam_results[j].posting_head;
		if (ph == InvalidBlockNumber)
		{
			qs->beam_results[j].distance = INFINITY;
			continue;
		}
		mkt_posting_scan_begin_cluster(&qs->pscan, &qs->cluster_qs, ph);
		const float *pt_cent = mkt_posting_scan_pt_centroid(&qs->pscan);
		if (pt_cent == NULL)
		{
			mkt_posting_scan_end_cluster(&qs->pscan);
			qs->beam_results[j].distance = INFINITY;
			continue;
		}
		/* pt_query and pt_cent are both rotated by P^T (orthonormal), so
		 * their dot product equals <q,c> and their norms equal ||q||,||c||. */
		Distance d;
		if (idx->metric == DISTANCE_INNER_PRODUCT)
			d = -mkt_dot_product(qs->pt_query, pt_cent, dim);
		else if (idx->metric == DISTANCE_COSINE)
		{
			float dot	= mkt_dot_product(qs->pt_query, pt_cent, dim);
			float nc	= mkt_l2_norm(pt_cent, dim);
			float denom = norm_q * nc;
			d			= (denom > 0.0f) ? 1.0f - dot / denom : 1.0f;
		}
		else
			d = mkt_l2_distance_squared(qs->pt_query, pt_cent, dim);
		qs->beam_results[j].distance = d;
		mkt_posting_scan_end_cluster(&qs->pscan);
	}
	qs->pscan.storage = NULL;

	qsort(qs->beam_results,
		  ncentroids,
		  sizeof(MktCentroidResult),
		  cmp_centroid_result);

	return ncentroids < nprobe ? ncentroids : nprobe;
}

static void
scan_clusters(
		MktQueryState			*qs,
		const MktCentroidResult *beam_results,
		uint32_t				 n_results,
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

	uint32_t total_pages   = 0;
	uint32_t total_entries = 0;

	for (uint32_t j = 0; j < n_results; j++)
	{
		BlockNumber ph = beam_results[j].posting_head;
		if (ph == InvalidBlockNumber)
			continue;

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
		total_entries += qs->pscan.entries_scanned;
		mkt_posting_scan_end_cluster(&qs->pscan);
	}

	qs->pscan.storage = NULL;

	if (stats != NULL)
	{
		stats->posting_pages_read	   = total_pages;
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

uint32_t
mkt_query_execute(
		MktQueryState  *qs,
		const float	   *query,
		uint32_t		k,
		uint32_t		nprobe,
		MktDistanceMode mode,
		bool			rerank,
		uint32_t		rerank_pool,
		MktQueryStats  *stats)
{
	if (k > qs->max_k)
		k = qs->max_k;
	if (nprobe > qs->max_nprobe)
		nprobe = qs->max_nprobe;

	mkt_topk_reset(&qs->topk);
	qs->topk.k = k;

	const float *qvec = prepare_query(qs, query);

	mkt_rabitq_rotate(qs->index->params, qvec, qs->pt_query);

	/* Two-stage centroid routing: widen the (cheap) beam to factor*nprobe,
	 * then re-rank that shortlist by exact full-precision centroid distance
	 * and keep the best nprobe. */
	uint32_t factor = (uint32_t)qs->index->centroid_rerank;
	uint32_t beam_n = nprobe;
	if (factor > 1)
	{
		beam_n = factor * nprobe;
		if (beam_n > qs->max_nprobe)
			beam_n = qs->max_nprobe;
	}

	uint64_t			   t0		  = mkt_now_ns();
	MktCentroidSearchStats beam_stats = {0};
	uint32_t			   ncentroids =
			search_centroids(qs, qvec, beam_n, mode, &beam_stats);
	if (factor > 1 && ncentroids > nprobe)
		ncentroids = rerank_beam(qs, ncentroids, nprobe);

	uint64_t t1 = mkt_now_ns();
	scan_clusters(qs, qs->beam_results, ncentroids, mode, &qs->topk, stats);

	uint64_t t2		= mkt_now_ns();
	uint32_t ncands = extract_candidates(qs);

	/* Bound the rerank pool. Candidates are sorted by quantized distance
	 * ascending (extract_candidates), so the first rerank_pool entries are
	 * the most promising by the approximate score. Reranking only those
	 * caps the number of full-precision heap fetches — the dominant cost
	 * of the rerank stage — at a small recall cost. rerank_pool == 0 keeps
	 * the legacy behavior of reranking every prune survivor. The pool is
	 * never shrunk below k, so the exact top-k is always achievable. */
	if (rerank_pool > 0 && ncands > rerank_pool)
		ncands = rerank_pool < k ? k : rerank_pool;

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

	uint64_t t3 = mkt_now_ns();

	if (stats != NULL)
	{
		stats->centroid_pages_read = beam_stats.pages_read;
		stats->clusters_scanned	   = ncentroids;
		stats->rerank_candidates   = ncands;
		stats->centroid_ns		   = t1 - t0;
		stats->posting_ns		   = t2 - t1;
		stats->rerank_ns		   = t3 - t2;
	}

	return qs->nresults;
}
