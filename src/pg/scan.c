/*
 * scan.c - Index scan for prism
 *
 * Uses PrismQueryState (shared with standalone) for the search hot
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
#include <miscadmin.h>
#include <pgstat.h>
#include <utils/builtins.h>
#include <utils/memutils.h>
#include <utils/rel.h>

#include "algo/vecops.h"
#include "amcache.h"
#include "build.h"
#include "core/platform.h"
#include "index/posting_page.h"
#include "index/query_scan.h"
#include "pg/bufstorage.h"
#include "quant/rabitq.h"
#include "scan.h"
#include "scan_bound.h"
#include "support_pg.h"
#include "typeinfo.h"
#include "types/vec32.h"

/* Default nprobe — will become a GUC later */
#define PRISM_DEFAULT_NPROBE 10
/*
 * Smallest top-k any scan is sized for. Not a default in the sense of "what
 * you get when you ask for nothing" -- a query with no LIMIT is sized from
 * work_mem (see resolve_top_k) -- but a floor under every sizing, so that a
 * LIMIT 1 has slack for a candidate that turns out to be a dead tuple the
 * heap fetch discards, and a work_mem too small to hold more still answers.
 */

/* ----------------------------------------------------------------
 * Process-global per-phase accumulators (diagnostic).
 *
 * Summed across every prism index scan in this backend so phase timing
 * can be measured over a large query set (e.g. a full 10k-query
 * benchmark run in one session) instead of eyeballing EXPLAIN on a
 * single query. Exposed via vs_phase_stats() / vs_phase_stats_reset():
 *
 *   CREATE FUNCTION vs_phase_stats_reset() RETURNS void
 *     AS '$libdir/pg_vectorsearch','vs_phase_stats_reset' LANGUAGE C;
 *   CREATE FUNCTION vs_phase_stats() RETURNS text
 *     AS '$libdir/pg_vectorsearch','vs_phase_stats' LANGUAGE C;
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

PG_FUNCTION_INFO_V1(vs_phase_stats_reset);

Datum
vs_phase_stats_reset(PG_FUNCTION_ARGS)
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
	vs_bufcache_hits	 = 0;
	vs_bufcache_cold	 = 0;
	vs_bufcache_stale	 = 0;
	for (int i = 0; i < 7; i++)
		g_route_le[i] = 0;
	PG_RETURN_VOID();
}

PG_FUNCTION_INFO_V1(vs_routing_stats);

Datum
vs_routing_stats(PG_FUNCTION_ARGS)
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
			vs_bufcache_hits,
			vs_bufcache_cold,
			vs_bufcache_stale,
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

PG_FUNCTION_INFO_V1(vs_phase_stats);

Datum
vs_phase_stats(PG_FUNCTION_ARGS)
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
			(double)g_phase_centroid_ns / n / VS_NS_PER_US,
			(double)g_phase_rotation_ns / n / VS_NS_PER_US,
			(double)g_phase_clut_ns / n / VS_NS_PER_US,
			(double)g_phase_cpageread_ns / n / VS_NS_PER_US,
			(double)g_phase_cscore_ns / n / VS_NS_PER_US,
			(double)g_phase_posting_ns / n / VS_NS_PER_US,
			(double)g_phase_rerank_ns / n / VS_NS_PER_US,
			g_phase_entries / n);
	PG_RETURN_TEXT_P(cstring_to_text(buf));
}

/* ----------------------------------------------------------------
 * Scan result entry
 * ---------------------------------------------------------------- */

typedef struct PrismScanResult
{
	ItemPointerData tid;
	Distance		distance;
} PrismScanResult;

/* ----------------------------------------------------------------
 * Scan state
 * ---------------------------------------------------------------- */

typedef struct PrismScanState
{
	/* Common index descriptor (first for cast compatibility) */
	PrismIndexBase index_base;

	/* Result iterator */
	PrismScanResult *results;
	uint32_t		 results_cap;
	uint32_t		 nresults;
	uint32_t		 curr;
	bool			 first;

	/* Shared query state. Allocated on the first search rather than at
	 * beginscan, so it can be sized for the top-k the query actually
	 * asks for -- the LIMIT hint arrives between the two (see
	 * scan_bound.c). Rebuilt only if a later search needs a larger k. */
	PrismQueryState qstate;
	bool			qstate_ready;
	uint32_t		max_nprobe;
	bool			has_fastscan;

	/* Rows the enclosing LIMIT will pull, 0 when unknown */
	uint32_t scan_bound;

	/* PG storage (index page I/O) */
	VsPgStorage storage;

	/* EXPLAIN ANALYZE stats (accumulated across rescans) */
	PrismScanStats stats;

	/* Resource owner the params checkout was registered with (the
	 * CurrentResourceOwner at the prism_index_base_init call below);
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
	Vec32Access query_vector_access;
} PrismScanState;

const PrismScanStats *
prism_scan_get_stats(IndexScanDesc scan)
{
	PrismScanState *ss = (PrismScanState *)scan->opaque;
	return ss ? &ss->stats : NULL;
}

/* ----------------------------------------------------------------
 * beginscan
 * ---------------------------------------------------------------- */

IndexScanDesc
prism_beginscan(Relation index, int nkeys, int norderbys)
{
	IndexScanDesc scan = RelationGetIndexScan(index, nkeys, norderbys);

	MemoryContext scan_ctx = AllocSetContextCreate(
			CurrentMemoryContext, "prism scan", ALLOCSET_DEFAULT_SIZES);
	MemoryContext old_ctx = MemoryContextSwitchTo(scan_ctx);

	PrismScanState *ss = palloc0(sizeof(PrismScanState));
	ss->scan_ctx	   = scan_ctx;
	ss->first		   = true;

	/* Immutable index parameters from the per-backend cache (metapage read at
	 * most once per backend). */
	prism_index_base_init(index, &ss->index_base);
	ss->params_owner   = CurrentResourceOwner;
	PrismScanInfo info = prism_cache_scan_info(index);

	/* From the per-backend cache; the buffer must outlive a rescan. */
	ss->query_vector_access = vec32_access(
			prism_cache_type_info(index), ss->index_base.dim, scan_ctx);

	/* Size the per-scan query buffers to the nprobe actually requested
	 * (the GUC is set before the query runs) rather than the worst-case
	 * ceiling: the centroid-search scratch alone is
	 * O(max_nprobe * entries_per_page) candidates, several MB per query
	 * at the ceiling but a few hundred KB at typical nprobe. Headroom
	 * covers routing more leaf candidates than are scanned (bounded
	 * probe expansion); requests beyond the sizing are clamped by
	 * prism_query_execute exactly as they were against the old ceiling. */
	uint32_t req_nprobe = prism_nprobe > 0 ? (uint32_t)prism_nprobe
										   : prism_auto_nprobe(info.nlist);
	uint32_t max_nprobe = req_nprobe + Min(req_nprobe, 256) + 16;
	if (max_nprobe > 4096)
		max_nprobe = 4096;
	if (max_nprobe > info.nlist)
		max_nprobe = info.nlist;
	ss->max_nprobe	 = max_nprobe;
	ss->has_fastscan = ss->index_base.fastscan != 0;

	/* Initialize PG storage */
	vs_pg_storage_init(&ss->storage, index, NULL, ss->index_base.metric);
	ss->index_base.centroid_storage = &ss->storage.base;
	ss->index_base.posting_storage	= &ss->storage.base;
	ss->index_base.page_base		= NULL;

	/* The query state and result buffer are sized on the first search
	 * (ensure_query_state), once the top-k is known. */

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
prism_rescan(
		IndexScanDesc scan,
		ScanKey		  keys,
		int			  nkeys,
		ScanKey		  orderbys,
		int			  norderbys)
{
	PrismScanState *ss = (PrismScanState *)scan->opaque;

	if (keys && scan->numberOfKeys > 0)
		memcpy(scan->keyData, keys, scan->numberOfKeys * sizeof(ScanKeyData));
	if (orderbys && scan->numberOfOrderBys > 0)
		memcpy(scan->orderByData,
			   orderbys,
			   scan->numberOfOrderBys * sizeof(ScanKeyData));

	ss->first	 = true;
	ss->curr	 = 0;
	ss->nresults = 0;

	/*
	 * Size the top-k from the query's LIMIT, if this scan runs under one.
	 * Here rather than pushed in from the executor: by rescan the executor
	 * node already points at this scan, so scan_bound.c can find the right
	 * node by identity, and nothing has to open the scan descriptor before
	 * the executor would.
	 *
	 * Re-resolved on every rescan, not cached: a correlated LIMIT takes a
	 * new value for each outer row, and a stale one would return too few
	 * rows -- the very failure this exists to prevent. nodeLimit.c
	 * re-derives its own bound per rescan for the same reason ("in case
	 * this is a rescan and the previous time we got a different result").
	 */
	ss->scan_bound = prism_scan_bound(scan);
}

/*
 * Bytes the scan commits per row its top-k can return.
 *
 * Every allocation that scales with k, and none scale with the vector
 * dimension. Sizing this from anything less makes the work_mem ceiling
 * below a fiction: the extraction buffer alone is PRISM_QUERY_CAND_PER_K
 * entries per row, which dominates the rest by an order of magnitude.
 *
 *   vs_topk_init         one upper bound and one id per row, plus a
 *                         candidate array of two entries per row
 *   prism_query_state_init  PRISM_QUERY_CAND_PER_K candidates per row and an
 *                         index and a distance for each of them
 *   execute_search        one result slot per row
 *
 * Keep in step with those three. PRISM_QUERY_CAND_PER_K is the same constant
 * the allocator uses.
 *
 * This prices the state as initialized. extract_candidates doubles the
 * extraction arrays if a query admits more candidates than the sizing
 * allowed for, so a scan can exceed this budget; work_mem bounds what the
 * scan asks for, not the high-water mark of a pathological query.
 */
#define PRISM_TOP_K_BYTES_PER_ROW                                          \
	(sizeof(Distance) + sizeof(uint64_t) + 2 * sizeof(VsTopKEntry) +       \
	 PRISM_QUERY_CAND_PER_K *                                              \
			 (sizeof(VsTopKEntry) + sizeof(uint32_t) + sizeof(Distance)) + \
	 sizeof(PrismScanResult))

/*
 * Rows the top-k may be sized to, from work_mem.
 *
 * The ceiling belongs to the memory the administrator granted, not to a row
 * constant: a session with work_mem raised for a large query should be able
 * to ask for a large LIMIT, and one with it lowered should not be able to
 * commit the backend to more. The LIMIT and the relation's row count are
 * inputs to the sizing rather than limits on it; prism.query_limit lowers it
 * when set, and this bounds whatever the rest of the sizing arrives at.
 *
 * Floored at the built-in default so a query always answers something. A
 * scan, unlike a split, can always return a few rows -- so a work_mem too
 * small to hold more is a reason to return fewer, not to raise an error.
 */
static uint32_t
max_top_k_for_work_mem(void)
{
	uint64 rows = ((uint64)work_mem * 1024) / PRISM_TOP_K_BYTES_PER_ROW;

	if (rows < PRISM_DEFAULT_K)
		return PRISM_DEFAULT_K;
	if (rows > PG_UINT32_MAX)
		return PG_UINT32_MAX;
	return (uint32_t)rows;
}

/*
 * Resolve the top-k for a search.
 *
 * The rows the query's LIMIT asks for (resolved by scan_bound.c) are the
 * primary source. With no usable LIMIT the query has asked for every row in
 * distance order, so the scan is sized for as many as it could possibly
 * return: what work_mem affords, or the relation's estimated row count if
 * that is smaller -- an ordered scan cannot return more rows than exist.
 * Sizing for a fixed handful instead would silently answer a complete
 * ordered scan with a fraction of it.
 *
 * prism.query_limit then lowers the result if it is set below it, and
 * work_mem bounds it in every case, so a query asking for more rows than
 * the backend may hold returns as many as it can.
 *
 * The built-in default is the floor. It keeps a little slack under a small
 * LIMIT for rows the executor's heap fetch discards (deleted but not yet
 * vacuumed), which would otherwise leave a LIMIT 1 empty when its single
 * candidate is dead, and it leaves a query something to answer with under a
 * work_mem too small to hold more.
 */
static uint32_t
resolve_top_k(const PrismScanState *ss, Relation heap)
{
	double rows = heap != NULL ? prism_estimate_heap_tuples(heap) : -1.0;

	return prism_scan_resolve_top_k(ss->scan_bound, rows);
}

uint32_t
prism_scan_resolve_top_k(uint32_t scan_bound, double heap_rows)
{
	uint32_t cap = max_top_k_for_work_mem();
	uint32_t k	 = scan_bound;

	if (k == 0)
	{
		/* No LIMIT to size from: the query has asked for every row in
		 * order, so size for as many as it could return. */
		k = cap;
		if (heap_rows >= 1.0 && heap_rows < (double)cap)
			k = (uint32_t)heap_rows;
	}

	/*
	 * prism.query_limit only ever lowers the sizing. Raising it above what
	 * the query asked for would have the scan rank rows the LIMIT then
	 * throws away; the lever exists to cap a query that asks for too much
	 * -- one with no LIMIT, or with one set far higher than the rows the
	 * caller will read.
	 */
	if (prism_query_limit > 0 && (uint32_t)prism_query_limit < k)
		k = (uint32_t)prism_query_limit;

	if (k < PRISM_DEFAULT_K)
		k = PRISM_DEFAULT_K;

	return k > cap ? cap : k;
}

/*
 * Allocate (or resize) the shared query state and result buffer for a
 * top-k of at least k. prism_query_execute clamps k to the allocated
 * max_k, so undersizing here is what silently truncates results.
 *
 * A resize discards the previous state rather than adding to it. Every
 * buffer in it is sized to max_k, and a rescan can resize -- a correlated
 * LIMIT resolves afresh for each outer row -- so keeping the old ones
 * would accumulate a full set per resize for the life of the scan.
 * prism_query_state_cleanup owns that: the state holds its buffers in a
 * context of its own.
 *
 * The result array is deliberately not part of that state. It grows with
 * what the rerank returns rather than with the sizing, so it lives in the
 * scan context where repalloc preserves it across a resize.
 */
static void
ensure_query_state(PrismScanState *ss, uint32_t k)
{
	uint32_t max_k = Max(k, (uint32_t)PRISM_DEFAULT_K);

	if (ss->qstate_ready && max_k <= ss->qstate.max_k)
		return;

	MemoryContext old_ctx = MemoryContextSwitchTo(ss->scan_ctx);

	if (ss->qstate_ready)
	{
		prism_query_state_cleanup(&ss->qstate);
		ss->qstate_ready = false;
	}

	prism_query_state_init(
			&ss->qstate, &ss->index_base, max_k, ss->max_nprobe);
	if (ss->has_fastscan)
		prism_posting_scan_enable_fastscan(
				&ss->qstate.pscan, prism_fastscan_bits);
	ss->qstate_ready = true;

	/* The error-bound rerank can return more than max_k results, so this
	 * is a starting size; execute_search grows it as needed. */
	if (ss->results_cap < max_k)
	{
		size_t bytes	= max_k * sizeof(PrismScanResult);
		ss->results		= ss->results ? repalloc(ss->results, bytes)
									  : palloc(bytes);
		ss->results_cap = max_k;
	}

	MemoryContextSwitchTo(old_ctx);
}

/* ----------------------------------------------------------------
 * Search execution (called on first gettuple)
 * ---------------------------------------------------------------- */

static void
execute_search(IndexScanDesc scan)
{
	PrismScanState *ss = (PrismScanState *)scan->opaque;

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

	uint32_t k = resolve_top_k(ss, scan->heapRelation);
	ensure_query_state(ss, k);

	/* Lazily set heap relation for reranking (rel is NULL at
	 * beginscan time; heapRelation becomes available later) */
	if (scan->heapRelation != NULL && ss->storage.rel == NULL)
		vs_pg_storage_set_rel(&ss->storage, scan->heapRelation);

	/*
	 * Extract query vector. The ORDER BY operator belongs to the opclass, so
	 * its argument has the opclass's input type and the same descriptor
	 * converts it -- into the scan-lifetime buffer, so a rescan does not leak
	 * one per execution. vec32_read rejects a dimension mismatch.
	 */
	Datum	 query_datum = scan->orderByData[0].sk_argument;
	Vec32Ref qref		 = vec32_read(&ss->query_vector_access, query_datum);

	if (qref.dim != ss->index_base.dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("query dimension %u does not match "
						"index dimension %u",
						qref.dim,
						ss->index_base.dim)));

	/* Execute shared search */
	uint32_t nprobe = prism_nprobe > 0
							? (uint32_t)prism_nprobe
							: prism_auto_nprobe(ss->index_base.nlist);

	PrismQueryStats qstats = {0};
	ss->storage.read_count = 0;

	prism_query_execute(
			&ss->qstate,
			qref.data,
			k,
			nprobe,
			(VsDistanceMode)prism_distance_mode,
			prism_rerank,
			&qstats);

	ss->stats.clusters_scanned		  = qstats.clusters_scanned;
	ss->stats.centroid_pages_read	  = qstats.centroid_pages_read;
	ss->stats.posting_pages_read	  = qstats.posting_pages_read;
	ss->stats.posting_pages_skipped	  = qstats.posting_pages_skipped;
	ss->stats.posting_entries_scanned = qstats.posting_entries_scanned;
	ss->stats.rerank_candidates		  = ss->qstate.ncandidates;
	ss->stats.rerank_results		  = ss->qstate.nresults;
	ss->stats.storage_reads			  = ss->storage.read_count;
	ss->stats.top_k					  = k;
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
				repalloc(ss->results, nresults * sizeof(PrismScanResult));
		ss->results_cap = nresults;
	}
	for (uint32_t i = 0; i < nresults; i++)
	{
		uint32_t ci		   = ss->qstate.result_order[i];
		ss->results[i].tid = prism_posting_decode_tid(
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
prism_gettuple(IndexScanDesc scan, ScanDirection direction)
{
	PrismScanState *ss = (PrismScanState *)scan->opaque;

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

	PrismScanResult *entry = &ss->results[ss->curr];

	scan->xs_heaptid = entry->tid;

	/* prism never returns a lossy match: the heap tuple always satisfies
	 * the original qual, so no recheck is ever needed. */
	scan->xs_recheck		 = false;
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
prism_endscan(IndexScanDesc scan)
{
	PrismScanState *ss = (PrismScanState *)scan->opaque;

	if (ss != NULL)
	{
		if (ss->qstate_ready)
			prism_query_state_cleanup(&ss->qstate);
		/* Check the RaBitQParams checkout back in before the scan's own
		 * memory goes away — see prism_index_base_init / the beginscan
		 * call above. */
		prism_release_params(
				ss->index_base.dim,
				ss->index_base.rabitq_seed,
				ss->params_owner);
		MemoryContextDelete(ss->scan_ctx);
		scan->opaque = NULL;
	}
}
