/*
 * mktann_scan.c - Index scan for mktann
 *
 * Two-phase search:
 *   1. Beam search over centroid pages to find nprobe leaf clusters
 *   2. Scan posting lists in those clusters with RaBitQ filtering
 *
 * Posting lists are always RaBitQ-encoded regardless of centroid format.
 * For RaBitQ centroids, beam search uses approximate distances. For
 * float/half centroids, beam search uses exact distances. All results
 * use internal reranking so the executor receives exact distances.
 *
 * Memory layout:
 *   scan_ctx    — scan lifetime (params, global_mean, ss)
 *     search_ctx — per-search temporaries; reset on rescan
 */

#include <postgres.h>

#include <access/relscan.h>
#include <access/tableam.h>
#include <executor/tuptable.h>
#include <math.h>
#include <storage/bufmgr.h>
#include <utils/memutils.h>
#include <utils/rel.h>
#include <utils/snapmgr.h>

#include "algo/topk.h"
#include "algo/vecops.h"
#include "index/centroid_search.h"
#include "index/posting_scan.h"
#include "mkt_pg.h"
#include "mkt_vector.h"
#include "mktann_meta.h"
#include "mktann_scan.h"
#include "mktann_storage.h"
#include "quant/rabitq.h"

/* ----------------------------------------------------------------
 * Process-local cache for RaBitQParams
 *
 * The random rotation matrix P is deterministic (depends only on
 * dim and seed from index metadata) and costs O(dim^3) to compute
 * via Gram-Schmidt QR. Cache it to avoid recomputation per query.
 * ---------------------------------------------------------------- */
static RaBitQParams *cached_params		= NULL;
static Dimension	 cached_params_dim	= 0;
static uint64_t		 cached_params_seed = 0;

static RaBitQParams *
get_rabitq_params(Dimension dim, uint64_t seed)
{
	if (cached_params != NULL && cached_params_dim == dim &&
		cached_params_seed == seed)
		return cached_params;

	if (cached_params != NULL)
		mkt_rabitq_destroy(cached_params);

	/* Allocate in TopMemoryContext so it survives across queries */
	MemoryContext old = MemoryContextSwitchTo(TopMemoryContext);
	cached_params	  = mkt_rabitq_create(dim, seed);
	MemoryContextSwitchTo(old);

	cached_params_dim  = dim;
	cached_params_seed = seed;
	return cached_params;
}

/* ----------------------------------------------------------------
 * Scan result entry — stored per search
 * ---------------------------------------------------------------- */

typedef struct MktannScanResult
{
	ItemPointerData tid;
	Distance		distance;
	Distance		error;
} MktannScanResult;

/* ----------------------------------------------------------------
 * Scan state
 * ---------------------------------------------------------------- */

typedef struct MktannScanState
{
	MktannScanResult *sorted_results; /* posting scan output */
	uint32_t		  nresults;		  /* count returned */
	uint32_t		  curr;			  /* next to return */
	bool			  first;		  /* first gettuple call? */
	RaBitQParams	 *params;		  /* for query prep */
	float			 *global_mean;	  /* for RaBitQ query prep */
	Dimension		  dim;
	uint8_t			  nlevels;
	BlockNumber		  first_centroid;
	uint32_t		  nprobe;
	uint32_t		  nlist; /* total leaf centroids */
	DistanceMetric	  metric;
	MktCentroidFormat centroid_format;
	MemoryContext	  scan_ctx;	  /* scan lifetime */
	MemoryContext	  search_ctx; /* per-search temps */
} MktannScanState;

/* ----------------------------------------------------------------
 * Encode/decode TID as uint64_t for top-K
 * ---------------------------------------------------------------- */

static inline uint64_t
encode_tid(const ItemPointerData *tid)
{
	return ((uint64_t)ItemPointerGetBlockNumber(tid) << 16) |
		   ItemPointerGetOffsetNumber(tid);
}

static inline ItemPointerData
decode_tid(uint64_t id)
{
	ItemPointerData tid;
	ItemPointerSet(&tid, (BlockNumber)(id >> 16), (OffsetNumber)(id & 0xFFFF));
	return tid;
}

/* ----------------------------------------------------------------
 * beginscan
 * ---------------------------------------------------------------- */

IndexScanDesc
mktann_beginscan(Relation index, int nkeys, int norderbys)
{
	IndexScanDesc scan = RelationGetIndexScan(index, nkeys, norderbys);

	MemoryContext scan_ctx = AllocSetContextCreate(
			CurrentMemoryContext, "mktann scan", ALLOCSET_DEFAULT_SIZES);
	MemoryContext old_ctx = MemoryContextSwitchTo(scan_ctx);

	MktannScanState *ss = palloc0(sizeof(MktannScanState));
	ss->scan_ctx		= scan_ctx;
	ss->search_ctx		= AllocSetContextCreate(
			 scan_ctx, "mktann search", ALLOCSET_DEFAULT_SIZES);
	ss->first		   = true;
	ss->curr		   = 0;
	ss->nresults	   = 0;
	ss->sorted_results = NULL;
	ss->nprobe		   = (uint32_t)mkt_nprobe;

	/* Read metadata page */
	Buffer meta_buf = ReadBuffer(index, 0);
	LockBuffer(meta_buf, BUFFER_LOCK_SHARE);
	Page meta_page = BufferGetPage(meta_buf);

	const MktannMetaPage *meta = (const MktannMetaPage *)PageGetSpecialPointer(
			meta_page);
	Assert(meta->magic == MKT_META_MAGIC);

	ss->dim				= meta->dim;
	ss->nlevels			= meta->nlevels;
	ss->first_centroid	= meta->first_centroid;
	ss->metric			= (DistanceMetric)meta->metric;
	ss->centroid_format = (MktCentroidFormat)meta->centroid_format;
	ss->nlist			= meta->nlist;

	/* Cap nprobe to nlist */
	if (ss->nprobe > meta->nlist)
		ss->nprobe = meta->nlist;

	/* RaBitQ params and global mean are always needed because posting
	 * lists are always RaBitQ-encoded, regardless of centroid format */
	const float *src_mean = mktann_meta_global_mean_const(meta);
	ss->global_mean		  = palloc(meta->dim * sizeof(float));
	memcpy(ss->global_mean, src_mean, meta->dim * sizeof(float));
	ss->params = get_rabitq_params(meta->dim, meta->rabitq_seed);

	UnlockReleaseBuffer(meta_buf);

	/* Allocate order-by value/null arrays (AM is responsible) */
	if (norderbys > 0)
	{
		scan->xs_orderbyvals  = palloc0(norderbys * sizeof(Datum));
		scan->xs_orderbynulls = palloc(norderbys * sizeof(bool));
		memset(scan->xs_orderbynulls, true, norderbys * sizeof(bool));
	}

	MemoryContextSwitchTo(old_ctx);
	scan->opaque = ss;
	return scan;
}

/* ----------------------------------------------------------------
 * rescan
 * ---------------------------------------------------------------- */

void
mktann_rescan(
		IndexScanDesc scan,
		ScanKey		  keys,
		int			  nkeys,
		ScanKey		  orderbys,
		int			  norderbys)
{
	MktannScanState *ss = (MktannScanState *)scan->opaque;

	if (keys && scan->numberOfKeys > 0)
		memcpy(scan->keyData, keys, scan->numberOfKeys * sizeof(ScanKeyData));
	if (orderbys && scan->numberOfOrderBys > 0)
		memcpy(scan->orderByData,
			   orderbys,
			   scan->numberOfOrderBys * sizeof(ScanKeyData));

	ss->first		   = true;
	ss->curr		   = 0;
	ss->sorted_results = NULL;
	ss->nresults	   = 0;

	/* Reset search context — frees previous results and any
	 * leftover temporaries from beam search in one shot */
	MemoryContextReset(ss->search_ctx);
}

/* ----------------------------------------------------------------
 * gettuple
 * ---------------------------------------------------------------- */

bool
mktann_gettuple(IndexScanDesc scan, ScanDirection direction)
{
	MktannScanState *ss = (MktannScanState *)scan->opaque;

	(void)direction;

	if (ss->first)
	{
		ss->first = false;

		/* No ORDER BY → no results */
		if (scan->numberOfOrderBys == 0)
			return false;

		/* Extract query vector from orderby */
		Datum	   query_datum = scan->orderByData[0].sk_argument;
		MktVector *query_vec   = DatumGetMktVector(query_datum);
		VectorRef  qref		   = MktVectorToRef(query_vec);

		if (qref.dim != ss->dim)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("query dimension %u does not match "
							"index dimension %u",
							qref.dim,
							ss->dim)));

		/* Switch to search context for temporaries */
		MemoryContext old_ctx = MemoryContextSwitchTo(ss->search_ctx);

		/* Normalize query for cosine so all distances are computed
		 * in L2 space on unit vectors */
		if (ss->metric == DISTANCE_COSINE)
		{
			float *norm_q = palloc(ss->dim * sizeof(float));
			memcpy(norm_q, qref.data, ss->dim * sizeof(float));
			mkt_normalize(norm_q, ss->dim);
			qref = (VectorRef){.data = norm_q, .dim = ss->dim};
		}

		/* Prepare RaBitQ query state for centroid beam search.
		 * Only needed for RaBitQ centroids; float/half centroids
		 * use exact distances and pass qstate=NULL. */
		RaBitQQueryState *qstate = NULL;
		if (ss->centroid_format == MKT_CENTROID_FMT_RABITQ)
		{
			VectorRef mean_ref = {.data = ss->global_mean, .dim = ss->dim};
			qstate			   = mkt_rabitq_prepare_query_ex(
					ss->params,
					qref,
					mean_ref,
					(MktDistanceMode)mkt_distance_mode);
		}

		/* Set up storage with heap relation */
		MktannStorage storage;
		mktann_storage_init(
				&storage, scan->indexRelation, scan->heapRelation, ss->metric);

		/* 1. Beam search — get nprobe leaf centroid winners.
		 * Also retrieve centroid vectors for use as posting list
		 * encoding references (float/half formats). */
		MktCentroidResult *centroid_results = palloc(
				ss->nprobe * sizeof(MktCentroidResult));

		float *centroid_vecs = palloc(ss->nprobe * ss->dim * sizeof(float));

		MktCentroidSearchState search = {
				.qstate		= qstate,
				.query		= qref.data,
				.storage	= &storage.base,
				.beam_width = ss->nprobe,
				.nprobe		= ss->nprobe,
				.dim		= ss->dim,
				.metric		= ss->metric,
		};

		uint32_t ncentroids = mkt_centroid_beam_search(
				&search,
				ss->first_centroid,
				ss->nlevels,
				centroid_results,
				centroid_vecs,
				NULL);

		/* 2. Initialize posting scan + top-K
		 *
		 * The top-K threshold heap tracks the K-th smallest upper
		 * bound. Candidates whose lower bound exceeds this threshold
		 * are pruned. Candidates with overlapping error bounds are
		 * kept in the candidate buffer for reranking.
		 *
		 * K = LIMIT from the planner hook, or a fallback default. */
		uint32_t topk_limit = MKT_DEFAULT_TOPK;
		if (mkt_query_limit > 0)
			topk_limit = (uint32_t)mkt_query_limit;
		MktTopK topk;
		mkt_topk_init(&topk, topk_limit);

		MktPostingScan pscan;
		mkt_posting_scan_init(&pscan, &storage.base, ss->params, ss->dim);

		Distance threshold = mkt_topk_threshold(&topk);
		mkt_posting_scan_set_threshold(&pscan, &threshold);

		/* 3. For each winning cluster, scan its posting list. */
		BlockNumber nblocks = RelationGetNumberOfBlocks(scan->indexRelation);

		for (uint32_t j = 0; j < ncentroids; j++)
		{
			BlockNumber ph = centroid_results[j].posting_head;
			if (ph == InvalidBlockNumber)
				continue;
			if (ph >= nblocks)
				ereport(ERROR,
						(errmsg("mktann scan: posting_head %u >= nblocks %u "
								"for centroid %u",
								ph,
								nblocks,
								j)));

			float	 *cdata = centroid_vecs + (size_t)j * ss->dim;
			VectorRef cvref = {.data = cdata, .dim = ss->dim};

			mkt_posting_scan_begin_cluster(&pscan, qref, cvref, ph);

			MktPostingScanResult pr;
			while (mkt_posting_scan_next(&pscan, &pr))
			{
				mkt_topk_insert(
						&topk, pr.distance, pr.error, encode_tid(&pr.tid));
				/* Update threshold for push-down pruning */
				threshold = mkt_topk_threshold(&topk);
			}
			mkt_posting_scan_end_cluster(&pscan);
		}

		pfree(centroid_vecs);
		mkt_posting_scan_cleanup(&pscan);

		/* 4. Extract candidates from top-K */
		MktTopKEntry *entries = palloc(topk.cand_count * sizeof(MktTopKEntry));
		uint32_t	  ncands;
		mkt_topk_extract_sorted(&topk, entries, &ncands);

		/* 5. Rerank: fetch exact distances from heap tuples.
		 * Unpack candidates into parallel arrays for pg_rerank. */
		ItemPointerData *cand_tids = palloc(ncands * sizeof(ItemPointerData));
		Distance		*cand_dist = palloc(ncands * sizeof(Distance));
		Distance		*cand_err  = palloc(ncands * sizeof(Distance));
		for (uint32_t i = 0; i < ncands; i++)
		{
			cand_tids[i] = decode_tid(entries[i].id);
			cand_dist[i] = entries[i].distance;
			cand_err[i]	 = entries[i].error;
		}

		uint32_t *rr_indices   = palloc(topk_limit * sizeof(uint32_t));
		Distance *rr_distances = palloc(topk_limit * sizeof(Distance));
		uint32_t  nresults	   = mkt_storage_rerank(
				 &storage.base,
				 query_datum,
				 ss->dim,
				 cand_tids,
				 cand_dist,
				 cand_err,
				 ncands,
				 topk_limit,
				 rr_indices,
				 rr_distances);

		/* Convert reranked results to scan format.
		 * pg_rerank returns distances in metric space. For cosine,
		 * convert back to operator space: cosine_dist = l2sq / 2. */
		ss->sorted_results = palloc(nresults * sizeof(MktannScanResult));
		for (uint32_t i = 0; i < nresults; i++)
		{
			uint32_t idx				= rr_indices[i];
			ss->sorted_results[i].tid	= cand_tids[idx];
			ss->sorted_results[i].error = 0.0f;
			if (ss->metric == DISTANCE_COSINE)
				ss->sorted_results[i].distance = rr_distances[i] / 2.0f;
			else
				ss->sorted_results[i].distance = rr_distances[i];
		}
		ss->nresults = nresults;

		pfree(rr_indices);
		pfree(rr_distances);
		pfree(cand_tids);
		pfree(cand_dist);
		pfree(cand_err);
		pfree(entries);
		mkt_topk_cleanup(&topk);
		pfree(centroid_results);

		MemoryContextSwitchTo(old_ctx);
		ss->curr = 0;
	}

	/* Return next result */
	if (ss->curr >= ss->nresults)
		return false;

	MktannScanResult *entry = &ss->sorted_results[ss->curr];

	scan->xs_heaptid = entry->tid;

	/* Return exact distance from internal rerank. No executor
	 * recheck needed — distances are already in operator space. */
	scan->xs_recheckorderby	 = false;
	scan->xs_orderbyvals[0]	 = Float8GetDatum((double)entry->distance);
	scan->xs_orderbynulls[0] = false;

	ss->curr++;
	return true;
}

/* ----------------------------------------------------------------
 * endscan
 * ---------------------------------------------------------------- */

void
mktann_endscan(IndexScanDesc scan)
{
	MktannScanState *ss = (MktannScanState *)scan->opaque;

	if (ss != NULL)
	{
		/* scan_ctx owns everything: ss, params, global_mean,
		 * search_ctx (and its children: results, query state,
		 * beam search buffers). One delete frees all. */
		MemoryContextDelete(ss->scan_ctx);
		scan->opaque = NULL;
	}
}
