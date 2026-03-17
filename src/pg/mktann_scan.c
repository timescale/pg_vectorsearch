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
 * use xs_recheckorderby so the executor recomputes exact distances
 * from heap tuples.
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

/* Fallback top-K when the planner cannot determine LIMIT. */
#define MKT_DEFAULT_TOPK 100

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
	RaBitQParams	 *params;		  /* for query prep (NULL if !RaBitQ) */
	float			 *global_mean;	  /* NULL if not RaBitQ */
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
		 * For float/half centroids, also retrieve centroid vectors
		 * for use as posting list encoding references. */
		MktCentroidResult *centroid_results = palloc(
				ss->nprobe * sizeof(MktCentroidResult));

		bool   need_cvecs = (ss->centroid_format != MKT_CENTROID_FMT_RABITQ);
		float *centroid_vecs = need_cvecs ? palloc(ss->nprobe * ss->dim *
												   sizeof(float))
										  : NULL;

		MktCentroidSearchState search = {
				.qstate		 = qstate,
				.query		 = qref.data,
				.query_datum = query_datum,
				.storage	 = &storage.base,
				.beam_width	 = ss->nprobe,
				.nprobe		 = ss->nprobe,
				.dim		 = ss->dim,
				.metric		 = ss->metric,
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
		 * The top-K threshold heap tracks the K-th smallest upper bound.
		 * Candidates whose lower bound exceeds this threshold are pruned
		 * (K better candidates are guaranteed). Candidates with
		 * overlapping error bounds are kept in the candidate buffer for
		 * reranking — so the buffer may exceed K.
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

		/* 3. For each winning cluster, get centroid reference vector
		 * and scan posting list.
		 *
		 * RaBitQ centroids: fetch medoid vector from heap by TID.
		 * Float/half centroids: use vector from beam search output. */
		AttrNumber		vec_attnum = 0;
		TupleTableSlot *slot	   = NULL;
		float		   *ref_buf	   = palloc(ss->dim * sizeof(float));

		if (!need_cvecs)
		{
			vec_attnum = scan->indexRelation->rd_index->indkey.values[0];
			slot	   = table_slot_create(scan->heapRelation, NULL);
		}

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

			float *cdata;

			if (need_cvecs)
			{
				/* Float/half: centroid vector from beam search */
				cdata = centroid_vecs + (size_t)j * ss->dim;
			}
			else
			{
				/* RaBitQ: fetch medoid from heap by TID */
				ItemPointerData med_tid = centroid_results[j].medoid_tid;
				if (!ItemPointerIsValid(&med_tid))
					continue;

				ItemPointerData tid_copy = med_tid;
				if (!table_tuple_fetch_row_version(
							scan->heapRelation, &tid_copy, SnapshotAny, slot))
					continue;

				bool  isnull;
				Datum val = slot_getattr(slot, vec_attnum, &isnull);
				if (isnull)
				{
					ExecClearTuple(slot);
					continue;
				}

				MktVector *mvec = DatumGetMktVector(val);
				memcpy(ref_buf, mvec->x, ss->dim * sizeof(float));
				ExecClearTuple(slot);
				if (ss->metric == DISTANCE_COSINE)
					mkt_normalize(ref_buf, ss->dim);
				cdata = ref_buf;
			}

			VectorRef cref = {
					.data = cdata,
					.dim  = ss->dim,
			};

			/* Scan posting list entries */
			mkt_posting_scan_begin_cluster(&pscan, qref, cref, ph);

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

		if (slot != NULL)
			ExecDropSingleTupleTableSlot(slot);
		pfree(ref_buf);
		if (centroid_vecs != NULL)
			pfree(centroid_vecs);
		mkt_posting_scan_cleanup(&pscan);

		/* 4. Extract results from top-K */
		MktTopKEntry *entries = palloc(topk.cand_count * sizeof(MktTopKEntry));
		uint32_t	  nresults;
		mkt_topk_extract_sorted(&topk, entries, &nresults);

		/* Convert to scan result format */
		ss->sorted_results = palloc(nresults * sizeof(MktannScanResult));
		for (uint32_t i = 0; i < nresults; i++)
		{
			ss->sorted_results[i].tid	   = decode_tid(entries[i].id);
			ss->sorted_results[i].distance = entries[i].distance;
			ss->sorted_results[i].error	   = entries[i].error;
		}
		ss->nresults = nresults;

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

	/* All RaBitQ distances are approximate — always recheck so the
	 * executor computes exact distances from heap tuples.
	 *
	 * Return 0.0 as the lower bound. The topk results are sorted by
	 * approximate distance, but (distance - error) can be non-monotonic
	 * across entries since error varies per vector. The executor
	 * requires xs_orderbyvals to be non-decreasing, so a trivial
	 * lower bound is the safe choice. The top-K limit bounds the
	 * number of heap refetches. */
	scan->xs_recheckorderby	 = true;
	scan->xs_orderbyvals[0]	 = Float8GetDatum(-1.0);
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
