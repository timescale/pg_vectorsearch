/*
 * mktann_parallel.c - Parallel index scan DSM setup (PostgreSQL)
 *
 * amestimateparallelscan / aminitparallelscan / amparallelrescan, plus
 * the layout of the shared area (see mktann_parallel.h). The scan-side
 * rendezvous and participant execution live in mktann_scan.c.
 *
 * The rotation params must reach fresh worker processes without each
 * one paying the O(dim^3) matrix construction. aminitparallelscan gets
 * only a bare pointer (no Relation), so the estimate call -- which does
 * get the Relation, and always runs immediately before init in the same
 * executor startup -- stashes what init needs. Params depend only on
 * (dim, seed), and both are embedded in the flat RaBitQParams copy, so
 * workers validate the published params against their own index cache
 * and fall back to building their own on any mismatch (a WARNING-worthy
 * anomaly, not an error).
 */

#include <postgres.h>

#include <miscadmin.h>
#include <optimizer/cost.h>

#include "mkt_pg.h"
#include "pg/mktann_cache.h"
#include "pg/mktann_parallel.h"
#include "pg/mktann_scan.h"

/*
 * Estimate-to-init stash. ExecParallelEstimate walks the plan calling
 * every node's estimate, then ExecParallelInitializeDSM walks the SAME
 * tree order calling init -- a FIFO pairs them even when one query
 * contains several parallel mktann scans. Entries carry everything init
 * needs to lay out and fill the area. The queue is small and per
 * backend; overflow falls back to params-less init (workers then build
 * their own params -- slow but correct).
 */
typedef struct MktannParallelSizing
{
	uint32		  max_nprobe;
	uint32		  pool;
	uint32		  nparticipants;
	uint32		  nclaim_slots;
	Size		  params_size;
	RaBitQParams *params; /* backend-lifetime cache pointer */
} MktannParallelSizing;

#define MKTANN_SIZING_QUEUE_LEN 8

static MktannParallelSizing sizing_queue[MKTANN_SIZING_QUEUE_LEN];
static uint32				sizing_head; /* next to consume */
static uint32				sizing_tail; /* next to fill */

/* Layout: header, heads[], claims[], params -- offsets from area start. */
typedef struct MktannParallelLayout
{
	Size heads_off;
	Size claims_off;
	Size params_off;
	Size total;
} MktannParallelLayout;

static MktannParallelLayout
parallel_layout(const MktannParallelSizing *sz)
{
	MktannParallelLayout lo;

	lo.heads_off  = MAXALIGN(sizeof(MktannParallelScan));
	lo.claims_off = lo.heads_off +
					MAXALIGN(mkt_pquery_heads_size(sz->max_nprobe));
	lo.params_off = lo.claims_off +
					MAXALIGN(mkt_pquery_claims_size(sz->nclaim_slots));
	lo.total = lo.params_off + MAXALIGN(sz->params_size);
	return lo;
}

/*
 * Per-scan sizing from the leader's executor-startup GUCs and the index
 * metapage cache. The same formulas the serial scan uses (see
 * mktann_scan_sizing); the per-participant pool additionally clamps the
 * unlimited rerank-pool setting to a sized bound so the claim table
 * stays finite.
 */
static MktannParallelSizing
parallel_sizing(Relation index)
{
	MktannParallelSizing sz;
	uint32_t			 max_k;
	uint32_t			 max_nprobe;

	mktann_scan_sizing(index, &max_k, &max_nprobe);

	uint32_t pool = mkt_query_rerank_pool(max_k);
	if (pool > 64 * max_k)
		pool = 64 * max_k;

	sz.max_nprobe	 = max_nprobe;
	sz.pool			 = pool;
	sz.nparticipants = (uint32)max_parallel_workers_per_gather + 1;
	sz.nclaim_slots	 = mkt_pquery_claim_slots(sz.nparticipants, pool);
	sz.params		 = mktann_cache_params(index);
	sz.params_size	 = MKT_RABITQ_PARAMS_SIZE(sz.params->dim);
	return sz;
}

Size
mktann_estimateparallelscan(Relation index, int nkeys, int norderbys)
{
	(void)nkeys;
	(void)norderbys;

	MktannParallelSizing sz = parallel_sizing(index);

	if (sizing_tail - sizing_head < MKTANN_SIZING_QUEUE_LEN)
		sizing_queue[sizing_tail++ % MKTANN_SIZING_QUEUE_LEN] = sz;
	/* else: overflow; init falls back to params-less setup */

	return parallel_layout(&sz).total;
}

void
mktann_initparallelscan(void *target)
{
	MktannParallelScan *pscan = (MktannParallelScan *)target;

	if (sizing_head == sizing_tail)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("parallel mktann scan initialized without a "
						"matching size estimate")));

	MktannParallelSizing sz =
			sizing_queue[sizing_head++ % MKTANN_SIZING_QUEUE_LEN];
	if (sizing_head == sizing_tail)
		sizing_head = sizing_tail = 0;

	MktannParallelLayout lo = parallel_layout(&sz);

	mkt_pquery_shared_init(
			&pscan->core,
			sz.max_nprobe,
			sz.nparticipants,
			sz.nclaim_slots,
			lo.heads_off,
			lo.claims_off);

	SpinLockInit(&pscan->mutex);
	ConditionVariableInit(&pscan->cv);
	pscan->state	  = MKTANN_PSCAN_PENDING;
	pscan->sized_pool = sz.pool;
	pscan->params_off = lo.params_off;

	memcpy((char *)pscan + lo.params_off, sz.params, sz.params_size);
	pscan->params_ready = true;
}

void
mktann_parallelrescan(IndexScanDesc scan)
{
	MktannParallelScan *pscan = mktann_parallel_area(scan);

	/* The executor guarantees no participant is running (Gather Merge
	 * rescan reinitializes before relaunching workers). The published
	 * rotation params are index-immutable and survive the rescan. */
	mkt_pquery_begin(&pscan->core);
	pscan->state = MKTANN_PSCAN_PENDING;
}
