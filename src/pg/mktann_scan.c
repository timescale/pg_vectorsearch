/*
 * mktann_scan.c - Index scan for mktann
 *
 * Uses MktQueryState (shared with standalone) for the search hot
 * path. PG-specific concerns: scan iterator protocol, memory
 * contexts, query vector extraction.
 *
 * Memory layout:
 *   scan_ctx  — scan lifetime (index base, query state, storage)
 */

#include <postgres.h>

#include <access/relscan.h>
#include <portability/instr_time.h>
#include <utils/memutils.h>
#include <utils/rel.h>

#include "algo/vecops.h"
#include "index/posting_page.h"
#include "index/query_scan.h"
#include "mkt_pg.h"
#include "mkt_vector.h"
#include "mktann_cache.h"
#include "mktann_meta.h"
#include "mktann_scan.h"
#include "mktann_storage.h"
#include "quant/rabitq.h"

/* Default nprobe — will become a GUC later */
#define MKT_DEFAULT_NPROBE 10
#define MKT_DEFAULT_K	   10

/* ----------------------------------------------------------------
 * Scan result entry
 * ---------------------------------------------------------------- */

typedef struct MktannScanResult
{
	ItemPointerData tid;
	Distance		distance;
} MktannScanResult;

/* ----------------------------------------------------------------
 * Scan state
 * ---------------------------------------------------------------- */

typedef struct MktannScanState
{
	/* Common index descriptor (first for cast compatibility) */
	MktIndexBase index_base;

	/* Result iterator */
	MktannScanResult *results;
	uint32_t		  nresults;
	uint32_t		  curr;
	bool			  first;

	/* Shared query state (pre-allocated, zero-alloc hot path) */
	MktQueryState qstate;

	/* PG storage (index page I/O) */
	MktannStorage storage;

	/* EXPLAIN ANALYZE stats (accumulated across rescans) */
	MktannScanStats stats;

	MemoryContext scan_ctx;
} MktannScanState;

const MktannScanStats *
mktann_scan_get_stats(IndexScanDesc scan)
{
	MktannScanState *ss = (MktannScanState *)scan->opaque;
	return ss ? &ss->stats : NULL;
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
	ss->first			= true;

	/* Read metadata page */
	Buffer meta_buf = ReadBuffer(index, 0);
	LockBuffer(meta_buf, BUFFER_LOCK_SHARE);
	Page meta_page = BufferGetPage(meta_buf);

	const MktannMetaPage *meta = (const MktannMetaPage *)PageGetSpecialPointer(
			meta_page);
	Assert(meta->magic == MKT_META_MAGIC);

	Dimension dim		 = meta->dim;
	uint32_t  max_k		 = MKT_DEFAULT_K;
	uint32_t  max_nprobe = meta->nlist < 512 ? meta->nlist : 512;

	/* Populate MktIndexBase from meta page */
	ss->index_base.dim			   = dim;
	ss->index_base.metric		   = (DistanceMetric)meta->metric;
	ss->index_base.centroid_format = (MktCentroidFormat)meta->centroid_format;
	ss->index_base.nlevels		   = meta->nlevels;
	ss->index_base.first_centroid  = meta->first_centroid;

	ss->index_base.rabitq_seed = meta->rabitq_seed;

	UnlockReleaseBuffer(meta_buf);

	/* Cached RaBitQ params + rotated global mean */
	MktannIndexCache cache		  = mktann_cache_get(index);
	ss->index_base.params		  = cache.params;
	ss->index_base.pt_global_mean = (float *)cache.pt_global_mean;

	/* Initialize PG storage */
	mktann_storage_init(&ss->storage, index, NULL, ss->index_base.metric);
	ss->index_base.centroid_storage = &ss->storage.base;
	ss->index_base.posting_storage	= &ss->storage.base;
	ss->index_base.page_base		= NULL;

	/* Initialize shared query state */
	mkt_query_state_init(&ss->qstate, &ss->index_base, max_k, max_nprobe);

	/* Enable TID dedup if index uses vector replication */
	MktannOptions *opts			   = (MktannOptions *)index->rd_options;
	bool		   has_replication = opts != NULL && opts->soar_lambda > 0.0;
	if (has_replication)
	{
		uint32_t avg_per_cluster = meta->ntuples / Max(meta->nlist, 1);
		uint32_t est_entries	 = max_nprobe * avg_per_cluster * 2;
		uint32_t cap			 = 1024;
		while (cap < est_entries * 2)
			cap *= 2;
		ss->qstate.dedup_set  = palloc(cap * sizeof(uint64_t));
		ss->qstate.dedup_gens = palloc0(cap * sizeof(uint32_t));
		ss->qstate.dedup_cap  = cap;
		ss->qstate.dedup_gen  = 0;
	}

	/* Pre-allocate result buffer */
	ss->results = palloc(max_k * sizeof(MktannScanResult));

	/* Order-by arrays */
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
	ss->nresults = 0;
}

/* ----------------------------------------------------------------
 * Search execution (called on first gettuple)
 * ---------------------------------------------------------------- */

static void
execute_search(IndexScanDesc scan)
{
	MktannScanState *ss = (MktannScanState *)scan->opaque;

	/* Lazily set heap relation for reranking */
	if (scan->heapRelation != NULL && ss->storage.rel == NULL)
		ss->storage.rel = scan->heapRelation;

	/* Extract query vector */
	Datum	   query_datum = scan->orderByData[0].sk_argument;
	MktVector *query_vec   = DatumGetMktVector(query_datum);
	VectorRef  qref		   = MktVectorToRef(query_vec);

	if (qref.dim != ss->index_base.dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("query dimension %u does not match "
						"index dimension %u",
						qref.dim,
						ss->index_base.dim)));

	/* Execute shared search */
	uint32_t k		= mkt_query_limit > 0 ? (uint32_t)mkt_query_limit
										  : ss->qstate.max_k;
	uint32_t nprobe = (uint32_t)mkt_nprobe;

	MktQueryStats qstats   = {0};
	ss->storage.read_count = 0;

	mkt_query_execute(
			&ss->qstate,
			qref.data,
			k,
			nprobe,
			(MktDistanceMode)mkt_distance_mode,
			mkt_rerank,
			&qstats);

	ss->stats.clusters_scanned		  = qstats.clusters_scanned;
	ss->stats.centroid_pages_read	  = qstats.centroid_pages_read;
	ss->stats.posting_pages_read	  = qstats.posting_pages_read;
	ss->stats.posting_entries_scanned = qstats.posting_entries_scanned;
	ss->stats.rerank_candidates		  = ss->qstate.ncandidates;
	ss->stats.rerank_results		  = ss->qstate.nresults;
	ss->stats.storage_reads			  = ss->storage.read_count;

	/* Copy results from result ordering */
	uint32_t nresults = ss->qstate.nresults;
	for (uint32_t i = 0; i < nresults; i++)
	{
		uint32_t ci		   = ss->qstate.result_order[i];
		ss->results[i].tid = mkt_posting_decode_tid(
				ss->qstate.candidates[ci].id);
		ss->results[i].distance = ss->qstate.result_dists[i];
	}
	ss->nresults = nresults;
	ss->curr	 = 0;
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

		if (scan->numberOfOrderBys == 0)
			return false;

		execute_search(scan);
	}

	if (ss->curr >= ss->nresults)
		return false;

	MktannScanResult *entry = &ss->results[ss->curr];

	scan->xs_heaptid = entry->tid;

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
		mkt_query_state_cleanup(&ss->qstate);
		MemoryContextDelete(ss->scan_ctx);
		scan->opaque = NULL;
	}
}
