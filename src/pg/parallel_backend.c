/*
 * parallel_backend.c - PostgreSQL implementations of the build's back-end
 * seams.
 *
 * The build driver is shared between the PG extension and the standalone
 * engine; the parts that genuinely differ by back-end are expressed as
 * same-named seam functions, with the PG versions here and the thread-based
 * versions under src/standalone. This file currently holds the heap-scan seam;
 * the DSM/launch/teardown seams join it as the driver moves to shared code.
 */

#include <postgres.h>

#include <access/parallel.h>
#include <access/table.h>
#include <access/tableam.h>
#include <catalog/index.h>
#include <pgstat.h>
#include <tcop/tcopprot.h>
#include <utils/rel.h>

#include "mkt_pg.h"
#include "mkt_vector.h"
#include "parallel_build.h"

/*
 * Bridges PostgreSQL's heap-tuple callback to the back-end-neutral scan
 * callback: skip nulls and hand the vector's data pointer (into the varlena,
 * no copy) plus its tid to the shared logic.
 */
typedef struct MktPgScanAdapter
{
	MktBuildScanCb cb;
	void		  *state;
} MktPgScanAdapter;

static void
mkt_pg_scan_adapter(
		Relation	index,
		ItemPointer tid,
		Datum	   *values,
		bool	   *isnull,
		bool		tuple_is_alive,
		void	   *adapter_state)
{
	MktPgScanAdapter *a = (MktPgScanAdapter *)adapter_state;

	(void)index;
	(void)tuple_is_alive;

	if (isnull[0])
		return;

	a->cb(a->state, *tid, MKT_VECTOR_DATA(DatumGetMktVector(values[0])));
}

/*
 * Scan every vector via the shared parallel table scan, invoking cb per live
 * tuple. The standalone back-end provides a same-named function that iterates
 * its in-memory vector array (work-stealing) instead. allow_sync/progress
 * are PostgreSQL table_index_build_scan flags, ignored in standalone; progress
 * gates pg_stat_progress_create_index reporting, so only the leader sets it
 * (otherwise every participant would inflate the tuples-scanned counter).
 */
void
mkt_build_scan(
		Relation		  heap,
		Relation		  index,
		struct IndexInfo *indexInfo,
		MktBuildShared	 *shared,
		bool			  allow_sync,
		bool			  progress,
		MktBuildScanCb	  cb,
		void			 *state)
{
	MktPgScanAdapter actx = {.cb = cb, .state = state};
	TableScanDesc	 scan = table_beginscan_parallel(
			   heap, ParallelTableScanFromMktShared(shared));

	table_index_build_scan(
			heap,
			index,
			indexInfo,
			allow_sync,
			progress,
			mkt_pg_scan_adapter,
			&actx,
			scan);
}

/*
 * Join the parallel build: look up the shared state, open the heap and index,
 * start per-worker instrumentation, and attach to the phase barrier. The
 * standalone back-end provides a same-named function that takes the shared
 * state and vectors directly and joins a thread barrier.
 */
void
mkt_pbuild_worker_attach(shm_toc *toc, MktPBuildWorker *w)
{
	MktBuildShared *shared	= shm_toc_lookup(toc, MKTANN_KEY_SHARED, false);
	Barrier		   *barrier = shm_toc_lookup(toc, MKTANN_KEY_BARRIER, false);

	char *sharedquery  = shm_toc_lookup(toc, MKTANN_KEY_QUERY_TEXT, true);
	debug_query_string = sharedquery;
	pgstat_report_activity(STATE_RUNNING, debug_query_string);
	pgstat_report_query_id(shared->queryid, false);

	w->shared	 = shared;
	w->barrier	 = barrier;
	w->heapRel	 = table_open(shared->heaprelid, ShareLock);
	w->indexRel	 = index_open(shared->indexrelid, AccessExclusiveLock);
	w->worker_id = ParallelWorkerNumber + 1;
	w->dim		 = shared->dim;

	InstrStartParallelQuery();

	/*
	 * Attach to the phase barrier. The leader initializes the barrier with
	 * itself as the sole party and waits for every launched worker to attach
	 * before advancing, so attaching here (before the first phase) keeps all
	 * participants in lockstep regardless of how many workers were launched.
	 */
	BarrierAttach(barrier);
}

/*
 * Leave the parallel build: report this worker's buffer/WAL usage back to the
 * leader and close the relations. The standalone back-end's same-named
 * function joins the thread and is otherwise a no-op.
 */
void
mkt_pbuild_worker_detach(shm_toc *toc, MktPBuildWorker *w)
{
	BufferUsage *bufferusage =
			shm_toc_lookup(toc, MKTANN_KEY_BUFFER_USAGE, false);
	WalUsage *walusage = shm_toc_lookup(toc, MKTANN_KEY_WAL_USAGE, false);
	InstrEndParallelQuery(
			&bufferusage[ParallelWorkerNumber],
			&walusage[ParallelWorkerNumber]);

	index_close(w->indexRel, AccessExclusiveLock);
	table_close(w->heapRel, ShareLock);
}
