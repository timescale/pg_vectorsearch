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

#include "algo/topk.h"
#include "algo/vecops.h"
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

	/* Beam search results */
	qs->beam_results = mkt_alloc(max_nprobe * sizeof(MktCentroidResult));

	/* Top-K */
	mkt_topk_init(&qs->topk, max_k);

	/* Candidate extraction buffer */
	qs->cand_cap   = max_k * 16;
	qs->candidates = mkt_alloc(qs->cand_cap * sizeof(MktTopKEntry));

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
	if (idx->centroid_format == MKT_CENTROID_FMT_RABITQ)
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
	};

	return mkt_centroid_beam_search(
			&search,
			idx->first_centroid,
			idx->nlevels,
			qs->beam_results,
			NULL,
			beam_stats);
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
	uint32_t ncands;

	if (qs->topk.cand_count <= qs->cand_cap)
	{
		mkt_topk_extract_sorted(&qs->topk, qs->candidates, &ncands);
	}
	else
	{
		MktTopKEntry *tmp = mkt_alloc(
				qs->topk.cand_count * sizeof(MktTopKEntry));
		mkt_topk_extract_sorted(&qs->topk, tmp, &ncands);
		uint32_t copy = ncands < qs->cand_cap ? ncands : qs->cand_cap;
		memcpy(qs->candidates, tmp, copy * sizeof(MktTopKEntry));
		ncands = copy;
		mkt_free(tmp);
	}

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

	MktCentroidSearchStats beam_stats = {0};
	uint32_t			   ncentroids =
			search_centroids(qs, qvec, nprobe, mode, &beam_stats);

	scan_clusters(qs, qs->beam_results, ncentroids, mode, &qs->topk, stats);

	uint32_t ncands = extract_candidates(qs);

	if (stats != NULL)
	{
		stats->centroid_pages_read = beam_stats.pages_read;
		stats->clusters_scanned	   = ncentroids;
	}

	return ncands;
}
