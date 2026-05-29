/*
 * mktann_parallel.c - PG parallel worker for mktann index build
 *
 * Worker entry point for parallel index builds. Each worker:
 *   1. Attaches to shared state in DSM (tree, reserve, params)
 *   2. Opens heap and index relations
 *   3. Runs a cooperative parallel heap scan
 *   4. For each tuple: tree descent + RaBitQ encode + stream to
 *      posting pages via atomic block reservation
 *   5. Reports partial page info and stats back to DSM
 *
 * The leader merges partial pages after all workers complete.
 */

#include <postgres.h>

#include <access/parallel.h>
#include <access/table.h>
#include <access/tableam.h>
#include <catalog/index.h>
#include <miscadmin.h>
#include <pgstat.h>
#include <storage/shm_toc.h>
#include <tcop/tcopprot.h>
#include <utils/memutils.h>
#include <utils/rel.h>

#include "algo/hkmeans.h"
#include "index/posting_build.h"
#include "index/posting_build_parallel.h"
#include "mkt_pg.h"
#include "mkt_vector.h"
#include "mktann_parallel.h"
#include "mktann_storage.h"
#include "quant/rabitq.h"

/* ----------------------------------------------------------------
 * Per-tuple callback state
 * ---------------------------------------------------------------- */

typedef struct MktParallelCbState
{
	const HKMeansResult	  *tree;
	MktBuildParams		   bp;
	MktBuildWorkerBufs	   bufs;
	MktPostingWorkerState *ws;
	double				   indtuples;
	double				   soar_dupes;
	MemoryContext		   tmp_ctx;
	MemoryContext		   worker_ctx;
} MktParallelCbState;

static void
parallel_build_callback(
		Relation	index,
		ItemPointer tid,
		Datum	   *values,
		bool	   *isnull,
		bool		tuple_is_alive,
		void	   *state)
{
	MktParallelCbState *cbs = (MktParallelCbState *)state;

	(void)index;
	(void)tuple_is_alive;

	if (isnull[0])
		return;

	MemoryContext old_ctx = MemoryContextSwitchTo(cbs->tmp_ctx);

	MktVector *vec	= DatumGetMktVector(values[0]);
	VectorRef  vref = MktVectorToRef(vec);

	MktBuildAssignment asgn = mkt_build_assign_vector(
			cbs->tree, vref.data, &cbs->bp, &cbs->bufs);

	MemoryContextSwitchTo(cbs->worker_ctx);

	mkt_posting_worker_add_heap(
			cbs->ws, *tid, asgn.enc_vector, asgn.primary, asgn.secondary);

	MemoryContextSwitchTo(old_ctx);

	cbs->indtuples++;
	if (asgn.secondary != MKT_INVALID_CLUSTER)
		cbs->soar_dupes++;

	MemoryContextReset(cbs->tmp_ctx);
}

/* ----------------------------------------------------------------
 * DSM reserve → local MktPostingReserve adapter
 * ---------------------------------------------------------------- */

static void
dsm_reserve_to_local(MktDsmReserve *dsm, MktPostingReserve *local)
{
	local->starts = mktann_dsm_reserve_starts(dsm);
	local->counts = mktann_dsm_reserve_counts(dsm);
	local->nexts  = (mkt_atomic_uint32 *)mktann_dsm_reserve_nexts(dsm);
	local->nlist  = dsm->nlist;
	local->total  = dsm->total_reserved;
}

/* ----------------------------------------------------------------
 * Worker entry point
 * ---------------------------------------------------------------- */

void
mktann_parallel_build_main(dsm_segment *seg, shm_toc *toc)
{
	(void)seg;

	MktBuildShared *shared = shm_toc_lookup(toc, MKTANN_KEY_SHARED, false);

	char *sharedquery  = shm_toc_lookup(toc, MKTANN_KEY_QUERY_TEXT, true);
	debug_query_string = sharedquery;
	pgstat_report_activity(STATE_RUNNING, debug_query_string);
	pgstat_report_query_id(shared->queryid, false);

	Relation heapRel  = table_open(shared->heaprelid, ShareLock);
	Relation indexRel = index_open(shared->indexrelid, AccessExclusiveLock);

	HKMeansResult *tree = shm_toc_lookup(toc, MKTANN_KEY_TREE, false);
	MktDsmReserve *dsm_reserve =
			shm_toc_lookup(toc, MKTANN_KEY_RESERVE, false);
	char *worker_output_base =
			shm_toc_lookup(toc, MKTANN_KEY_WORKER_OUTPUT, false);
	char *partials_base = shm_toc_lookup(toc, MKTANN_KEY_PARTIALS, true);

	int worker_id = ParallelWorkerNumber + 1;

	RaBitQParams *rq_params =
			mkt_rabitq_create(shared->dim, shared->rabitq_seed);

	MktannStorage storage;
	mktann_storage_init(&storage, indexRel, NULL, shared->metric);
	storage.build_mode = true;

	MktPostingReserve reserve;
	dsm_reserve_to_local(dsm_reserve, &reserve);

	uint32_t nlist = shared->nlist;
	char	*my_partials =
			   (partials_base != NULL)
					   ? mktann_worker_partials(partials_base, nlist, worker_id)
					   : NULL;

	const float *leaf_cents = hk_leaf_centroids(tree);

	float *pt_centroids = palloc((size_t)nlist * shared->dim * sizeof(float));
	for (uint32_t c = 0; c < nlist; c++)
		mkt_rabitq_rotate(
				rq_params,
				leaf_cents + (size_t)c * shared->dim,
				pt_centroids + (size_t)c * shared->dim);

	MemoryContext worker_ctx = AllocSetContextCreate(
			CurrentMemoryContext,
			"mktann worker posting",
			ALLOCSET_DEFAULT_SIZES);
	MemoryContext prev = MemoryContextSwitchTo(worker_ctx);

	MktPostingWorkerState ws;
	mkt_posting_worker_init(
			&ws,
			worker_id,
			nlist,
			shared->dim,
			shared->fastscan,
			&storage.base,
			rq_params,
			leaf_cents,
			pt_centroids,
			&reserve,
			my_partials);

	MktBuildWorkerBufs bufs = mkt_build_worker_bufs_create(shared->dim);
	MemoryContextSwitchTo(prev);

	MktParallelCbState cbs = {
			.tree		= tree,
			.bp			= {.dim				 = shared->dim,
						   .metric			 = shared->metric,
						   .soar_lambda		 = shared->soar_lambda,
						   .boundary_epsilon = shared->boundary_epsilon},
			.bufs		= bufs,
			.ws			= &ws,
			.indtuples	= 0,
			.soar_dupes = 0,
			.tmp_ctx	= AllocSetContextCreate(
					   CurrentMemoryContext,
					   "mktann parallel tuple",
					   ALLOCSET_DEFAULT_SIZES),
			.worker_ctx = worker_ctx,
	};

	InstrStartParallelQuery();

	IndexInfo	 *indexInfo = BuildIndexInfo(indexRel);
	TableScanDesc scan		= table_beginscan_parallel(
			 heapRel, ParallelTableScanFromMktShared(shared));

	table_index_build_scan(
			heapRel,
			indexRel,
			indexInfo,
			true,
			false,
			parallel_build_callback,
			&cbs,
			scan);

	mkt_posting_worker_finish(&ws);

	BlockNumber *my_heads =
			mktann_worker_heads(worker_output_base, nlist, worker_id);
	BlockNumber *my_tails =
			mktann_worker_tails(worker_output_base, nlist, worker_id);
	bool *my_active =
			mktann_worker_active(worker_output_base, nlist, worker_id);

	memcpy(my_heads, ws.heads, nlist * sizeof(BlockNumber));
	memcpy(my_tails, ws.tails, nlist * sizeof(BlockNumber));
	memcpy(my_active, ws.active, nlist * sizeof(bool));

	SpinLockAcquire(&shared->mutex);
	shared->nparticipantsdone++;
	shared->indtuples += cbs.indtuples;
	shared->soar_dupes += cbs.soar_dupes;
	SpinLockRelease(&shared->mutex);
	ConditionVariableSignal(&shared->workersdonecv);

	BufferUsage *bufferusage =
			shm_toc_lookup(toc, MKTANN_KEY_BUFFER_USAGE, false);
	WalUsage *walusage = shm_toc_lookup(toc, MKTANN_KEY_WAL_USAGE, false);
	InstrEndParallelQuery(
			&bufferusage[ParallelWorkerNumber],
			&walusage[ParallelWorkerNumber]);

	MemoryContextDelete(worker_ctx);
	pfree(pt_centroids);

	index_close(indexRel, AccessExclusiveLock);
	table_close(heapRel, ShareLock);
}
