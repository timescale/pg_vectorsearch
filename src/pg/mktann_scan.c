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

#include <access/genam.h>
#include <access/relscan.h>
#include <fmgr.h>
#include <pgstat.h>
#include <utils/builtins.h>
#include <utils/memutils.h>
#include <utils/rel.h>

#include "algo/vecops.h"
#include "core/platform.h"
#include "index/posting_page.h"
#include "index/query_scan.h"
#include "mktann_cache.h"
#include "mktann_scan.h"
#include "mktann_storage.h"
#include "quant/rabitq.h"
#include "support_pg.h"
#include "typeinfo.h"
#include "types/vector.h"

/* Default nprobe — will become a GUC later */
#define MKT_DEFAULT_NPROBE 10
#define MKT_DEFAULT_K	   10

/* ----------------------------------------------------------------
 * Process-global per-phase accumulators (diagnostic).
 *
 * Summed across every mktann index scan in this backend so phase timing
 * can be measured over a large query set (e.g. a full 10k-query
 * benchmark run in one session) instead of eyeballing EXPLAIN on a
 * single query. Exposed via mkt_phase_stats() / mkt_phase_stats_reset():
 *
 *   CREATE FUNCTION mkt_phase_stats_reset() RETURNS void
 *     AS '$libdir/meerkat','mkt_phase_stats_reset' LANGUAGE C;
 *   CREATE FUNCTION mkt_phase_stats() RETURNS text
 *     AS '$libdir/meerkat','mkt_phase_stats' LANGUAGE C;
 * ---------------------------------------------------------------- */
static uint64_t g_phase_nqueries	 = 0;
static uint64_t g_phase_centroid_ns	 = 0;
static uint64_t g_phase_posting_ns	 = 0;
static uint64_t g_phase_rerank_ns	 = 0;
static uint64_t g_phase_entries		 = 0;
static uint64_t g_phase_rotation_ns	 = 0;
static uint64_t g_phase_clut_ns		 = 0;
static uint64_t g_phase_cpageread_ns = 0;
static uint64_t g_phase_cscore_ns	 = 0;
/* Routing-depth histogram: deepest probe rank the final top-k came from. */
static uint64_t g_route_sum	  = 0;
static uint64_t g_route_le[7] = {0}; /* <=8,16,32,64,128,256,>256 */

PG_FUNCTION_INFO_V1(mkt_phase_stats_reset);

Datum
mkt_phase_stats_reset(PG_FUNCTION_ARGS)
{
	g_phase_nqueries	 = 0;
	g_phase_centroid_ns	 = 0;
	g_phase_posting_ns	 = 0;
	g_phase_rerank_ns	 = 0;
	g_phase_entries		 = 0;
	g_phase_rotation_ns	 = 0;
	g_phase_clut_ns		 = 0;
	g_phase_cpageread_ns = 0;
	g_phase_cscore_ns	 = 0;
	g_route_sum			 = 0;
	mkt_bufcache_hits	 = 0;
	mkt_bufcache_cold	 = 0;
	mkt_bufcache_stale	 = 0;
	for (int i = 0; i < 7; i++)
		g_route_le[i] = 0;
	PG_RETURN_VOID();
}

PG_FUNCTION_INFO_V1(mkt_routing_stats);

Datum
mkt_routing_stats(PG_FUNCTION_ARGS)
{
	char	 buf[448];
	uint64_t n = g_phase_nqueries ? g_phase_nqueries : 1;
	snprintf(
			buf,
			sizeof(buf),
			"bufcache hits=" UINT64_FORMAT " cold=" UINT64_FORMAT
			" stale=" UINT64_FORMAT " | "
			"queries=" UINT64_FORMAT " avg_deepest_contrib_rank=%.1f | "
			"deepest-rank histogram: <=8:%.1f%% <=16:%.1f%% <=32:%.1f%% "
			"<=64:%.1f%% <=128:%.1f%% <=256:%.1f%% >256:%.1f%%",
			mkt_bufcache_hits,
			mkt_bufcache_cold,
			mkt_bufcache_stale,
			g_phase_nqueries,
			(double)g_route_sum / n,
			100.0 * g_route_le[0] / n,
			100.0 * g_route_le[1] / n,
			100.0 * g_route_le[2] / n,
			100.0 * g_route_le[3] / n,
			100.0 * g_route_le[4] / n,
			100.0 * g_route_le[5] / n,
			100.0 * g_route_le[6] / n);
	PG_RETURN_TEXT_P(cstring_to_text(buf));
}

PG_FUNCTION_INFO_V1(mkt_phase_stats);

Datum
mkt_phase_stats(PG_FUNCTION_ARGS)
{
	char	 buf[256];
	uint64_t n = g_phase_nqueries ? g_phase_nqueries : 1;
	snprintf(
			buf,
			sizeof(buf),
			"queries=" UINT64_FORMAT " | per-query us: centroid=%.1f "
			"[rot=%.1f lut=%.1f pageread=%.1f score=%.1f] "
			"posting=%.1f rerank=%.1f | entries/q=" UINT64_FORMAT,
			g_phase_nqueries,
			(double)g_phase_centroid_ns / n / MKT_NS_PER_US,
			(double)g_phase_rotation_ns / n / MKT_NS_PER_US,
			(double)g_phase_clut_ns / n / MKT_NS_PER_US,
			(double)g_phase_cpageread_ns / n / MKT_NS_PER_US,
			(double)g_phase_cscore_ns / n / MKT_NS_PER_US,
			(double)g_phase_posting_ns / n / MKT_NS_PER_US,
			(double)g_phase_rerank_ns / n / MKT_NS_PER_US,
			g_phase_entries / n);
	PG_RETURN_TEXT_P(cstring_to_text(buf));
}

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
	uint32_t		  results_cap;
	uint32_t		  nresults;
	uint32_t		  curr;
	bool			  first;

	/* Shared query state (pre-allocated, zero-alloc hot path) */
	MktQueryState qstate;

	/* PG storage (index page I/O) */
	MktannStorage storage;

	/* EXPLAIN ANALYZE stats (accumulated across rescans) */
	MktannScanStats stats;

	/* Resource owner the params checkout was registered with (the
	 * CurrentResourceOwner at the mktann_index_base_init call below);
	 * the endscan release must name the same owner. */
	ResourceOwner params_owner;

	MemoryContext scan_ctx;

	/*
	 * Reads the query vector out of the ORDER BY argument: the opclass input
	 * type bound to this index's dimension, plus the conversion buffer a
	 * float32 view needs. It is the same binding build and insert use for
	 * column values, because the ORDER BY argument has the opclass's input
	 * type -- an access, not the vector itself.
	 */
	MktVectorAccess query_vector_access;
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

	/* Immutable index parameters from the per-backend cache (metapage read at
	 * most once per backend). */
	mktann_index_base_init(index, &ss->index_base);
	ss->params_owner	= CurrentResourceOwner;
	MktannScanInfo info = mktann_cache_scan_info(index);

	/* From the per-backend cache; the buffer must outlive a rescan. */
	ss->query_vector_access = mkt_vector_access(
			mktann_cache_type_info(index), ss->index_base.dim, scan_ctx);

	/* Size the top-K for the requested result count: mkt.query_limit is
	 * set before the query runs (same contract as the nprobe sizing
	 * below). Without this, mkt_query_execute clamps k to the allocated
	 * max_k and a query_limit above the default silently returned only
	 * MKT_DEFAULT_K results. */
	uint32_t max_k = MKT_DEFAULT_K;
	if (mkt_query_limit > 0 && (uint32_t)mkt_query_limit > max_k)
		max_k = (uint32_t)mkt_query_limit;

	/* Size the per-scan query buffers to the nprobe actually requested
	 * (the GUC is set before the query runs) rather than the worst-case
	 * ceiling: the centroid-search scratch alone is
	 * O(max_nprobe * entries_per_page) candidates, several MB per query
	 * at the ceiling but a few hundred KB at typical nprobe. Headroom
	 * covers routing more leaf candidates than are scanned (bounded
	 * probe expansion); requests beyond the sizing are clamped by
	 * mkt_query_execute exactly as they were against the old ceiling. */
	uint32_t req_nprobe = mkt_nprobe > 0 ? (uint32_t)mkt_nprobe
										 : mkt_auto_nprobe(info.nlist);
	uint32_t max_nprobe = req_nprobe + Min(req_nprobe, 256) + 16;
	if (max_nprobe > 4096)
		max_nprobe = 4096;
	if (max_nprobe > info.nlist)
		max_nprobe = info.nlist;

	bool has_fastscan = ss->index_base.fastscan != 0;

	/* Initialize PG storage */
	mktann_storage_init(&ss->storage, index, NULL, ss->index_base.metric);
	ss->index_base.centroid_storage = &ss->storage.base;
	ss->index_base.posting_storage	= &ss->storage.base;
	ss->index_base.page_base		= NULL;

	/* Initialize shared query state */
	mkt_query_state_init(&ss->qstate, &ss->index_base, max_k, max_nprobe);

	if (has_fastscan)
		mkt_posting_scan_enable_fastscan(&ss->qstate.pscan, mkt_fastscan_bits);

	/* Pre-allocate result buffer. The error-bound rerank can return more than
	 * max_k results, so this is a starting size; rescan grows it as needed. */
	ss->results		= palloc(max_k * sizeof(MktannScanResult));
	ss->results_cap = max_k;

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

	/*
	 * Count the search where it runs, as every core AM does at the start of
	 * its own search: pg_stat_*_indexes.idx_scan, and the per-scan counter
	 * PostgreSQL 18 prints as EXPLAIN's "Index Searches" (also the one a
	 * parallel scan aggregates across workers). Core's indexam.c maintains
	 * neither; it counts only the tuples the scan returns.
	 */
	pgstat_count_index_scan(scan->indexRelation);
	if (scan->instrument != NULL)
		scan->instrument->nsearches++;

	/* Lazily set heap relation for reranking (rel is NULL at
	 * beginscan time; heapRelation becomes available later) */
	if (scan->heapRelation != NULL && ss->storage.rel == NULL)
		mktann_storage_set_rel(&ss->storage, scan->heapRelation);

	/*
	 * Extract query vector. The ORDER BY operator belongs to the opclass, so
	 * its argument has the opclass's input type and the same descriptor
	 * converts it -- into the scan-lifetime buffer, so a rescan does not leak
	 * one per execution. mkt_vector_read rejects a dimension mismatch.
	 */
	Datum	  query_datum = scan->orderByData[0].sk_argument;
	VectorRef qref = mkt_vector_read(&ss->query_vector_access, query_datum);

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
	uint32_t nprobe = mkt_nprobe > 0 ? (uint32_t)mkt_nprobe
									 : mkt_auto_nprobe(ss->index_base.nlist);

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
	ss->stats.posting_pages_skipped	  = qstats.posting_pages_skipped;
	ss->stats.posting_entries_scanned = qstats.posting_entries_scanned;
	ss->stats.rerank_candidates		  = ss->qstate.ncandidates;
	ss->stats.rerank_results		  = ss->qstate.nresults;
	ss->stats.storage_reads			  = ss->storage.read_count;
	ss->stats.centroid_ns			  = qstats.centroid_ns;
	ss->stats.posting_ns			  = qstats.posting_ns;
	ss->stats.rerank_ns				  = qstats.rerank_ns;

	/* Accumulate into the process-global diagnostic counters. */
	g_phase_nqueries++;
	g_phase_centroid_ns += qstats.centroid_ns;
	g_phase_posting_ns += qstats.posting_ns;
	g_phase_rerank_ns += qstats.rerank_ns;
	g_phase_entries += qstats.posting_entries_scanned;
	g_phase_rotation_ns += qstats.rotation_ns;
	g_phase_clut_ns += qstats.centroid_lut_ns;
	g_phase_cpageread_ns += qstats.centroid_pageread_ns;
	g_phase_cscore_ns += qstats.centroid_score_ns;

	{
		uint32_t r = qstats.max_contrib_rank;
		g_route_sum += r;
		if (r <= 8)
			g_route_le[0]++;
		else if (r <= 16)
			g_route_le[1]++;
		else if (r <= 32)
			g_route_le[2]++;
		else if (r <= 64)
			g_route_le[3]++;
		else if (r <= 128)
			g_route_le[4]++;
		else if (r <= 256)
			g_route_le[5]++;
		else
			g_route_le[6]++;
	}

	/* Copy results from result ordering. The error-bound rerank can return
	 * more than the beginscan max_k (the rerank set is inflated beyond k to
	 * guarantee correctness), so grow the result buffer to fit. */
	uint32_t nresults = ss->qstate.nresults;
	if (nresults > ss->results_cap)
	{
		ss->results =
				repalloc(ss->results, nresults * sizeof(MktannScanResult));
		ss->results_cap = nresults;
	}
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
		/* Check the RaBitQParams checkout back in before the scan's own
		 * memory goes away — see mktann_index_base_init / the beginscan
		 * call above. */
		mktann_release_params(
				ss->index_base.dim,
				ss->index_base.rabitq_seed,
				ss->params_owner);
		MemoryContextDelete(ss->scan_ctx);
		scan->opaque = NULL;
	}
}
