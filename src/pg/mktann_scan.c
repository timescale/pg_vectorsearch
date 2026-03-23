/*
 * mktann_scan.c - Index scan for mktann
 *
 * Beam search over centroid pages to find nearest leaf centroids.
 * Currently only performs centroid routing; returning heap tuples
 * requires posting lists (not yet implemented).
 *
 * Memory layout:
 *   scan_ctx    — scan lifetime (params, global_mean, results, ss)
 *     search_ctx — per-search temporaries; reset on rescan
 */

#include <postgres.h>

#include <access/relscan.h>
#include <utils/memutils.h>
#include <utils/rel.h>

#include "index/centroid_search.h"
#include "mkt_pg.h"
#include "mkt_vector.h"
#include "mktann_meta.h"
#include "mktann_scan.h"
#include "mktann_storage.h"
#include "quant/rabitq.h"

/* Default nprobe — will become a GUC later */
#define MKT_DEFAULT_NPROBE 10

/* ----------------------------------------------------------------
 * Scan state
 * ---------------------------------------------------------------- */

typedef struct MktannScanState
{
	MktCentroidResult *results;		/* beam search output */
	uint32_t		   nresults;	/* count returned */
	uint32_t		   curr;		/* next to return */
	bool			   first;		/* first gettuple call? */
	RaBitQParams	  *params;		/* for query prep (NULL if !RaBitQ) */
	float			  *global_mean; /* NULL if not RaBitQ */
	Dimension		   dim;
	uint8_t			   nlevels;
	BlockNumber		   first_centroid;
	uint32_t		   nprobe;
	DistanceMetric	   metric;			/* distance metric from opclass */
	MktCentroidFormat  centroid_format; /* centroid page format */
	MemoryContext	   scan_ctx;		/* scan lifetime */
	MemoryContext	   search_ctx;		/* per-search temps */
} MktannScanState;

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
	ss->first	 = true;
	ss->curr	 = 0;
	ss->nresults = 0;
	ss->results	 = NULL;
	ss->nprobe	 = MKT_DEFAULT_NPROBE;

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

	/* Cap nprobe to nlist */
	if (ss->nprobe > meta->nlist)
		ss->nprobe = meta->nlist;

	/* RaBitQ params only needed for compressed centroids */
	if (ss->centroid_format == MKT_CENTROID_FMT_RABITQ)
	{
		const float *src_mean = mktann_meta_global_mean_const(meta);
		ss->global_mean		  = palloc(meta->dim * sizeof(float));
		memcpy(ss->global_mean, src_mean, meta->dim * sizeof(float));
		ss->params = mkt_rabitq_create(meta->dim, meta->rabitq_seed);
	}
	else
	{
		ss->global_mean = NULL;
		ss->params		= NULL;
	}

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

	ss->first	 = true;
	ss->curr	 = 0;
	ss->results	 = NULL;
	ss->nresults = 0;

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

		/* Centroid routing doesn't store heap TIDs — scans need
		 * posting lists (not yet implemented) */
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("scan requires posting lists "
						"(not yet implemented)")));

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

		/* Switch to search context for temporaries.
		 * beam search and query state allocate via mkt_alloc
		 * (= palloc), so they land here and get freed on
		 * rescan/endscan via context reset/delete. */
		MemoryContext old_ctx = MemoryContextSwitchTo(ss->search_ctx);

		/* Prepare RaBitQ query state */
		VectorRef		  mean_ref = {.data = ss->global_mean, .dim = ss->dim};
		RaBitQQueryState *qstate   = mkt_rabitq_prepare_query_ex(
				  ss->params,
				  qref,
				  mean_ref,
				  (MktDistanceMode)mkt_distance_mode);

		/* Set up storage with heap relation for reranking */
		MktannStorage storage;
		mktann_storage_init(
				&storage, scan->indexRelation, scan->heapRelation, ss->metric);

		/* Allocate results */
		ss->results = palloc(ss->nprobe * sizeof(MktCentroidResult));

		/* Run beam search */
		MktCentroidSearchState search = {
				.qstate		= qstate,
				.query		= qref.data,
				.storage	= &storage.base,
				.beam_width = ss->nprobe,
				.nprobe		= ss->nprobe,
				.dim		= ss->dim,
				.metric		= ss->metric,
		};

		ss->nresults = mkt_centroid_beam_search(
				&search,
				ss->first_centroid,
				ss->nlevels,
				ss->results,
				NULL,
				NULL);

		MemoryContextSwitchTo(old_ctx);
		ss->curr = 0;
	}

	/* Return next result */
	if (ss->curr >= ss->nresults)
		return false;

	/* TODO: return TID from posting list scan */
	scan->xs_recheckorderby = true;

	/* Provide estimated distance for executor reorder */
	scan->xs_orderbyvals[0] = Float8GetDatum(
			(double)ss->results[ss->curr].distance);
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
