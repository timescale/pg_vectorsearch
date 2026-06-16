/*
 * parallel_backend.c - PostgreSQL implementations of the build's back-end
 * seams.
 *
 * The build driver is shared between the PG extension and the standalone
 * engine; the parts that genuinely differ by back-end are expressed as
 * same-named seam functions, with the PG versions here and the thread-based
 * versions under src/standalone. It holds the PG implementations of all the
 * build seams: DSM setup over shm_toc, ParallelContext launch/teardown, worker
 * attach/detach, the heap parallel scan, and per-worker count reduction.
 */

#include <postgres.h>

#include <access/parallel.h>
#include <access/table.h>
#include <access/tableam.h>
#include <catalog/index.h>
#include <miscadmin.h>
#include <pgstat.h>
#include <storage/latch.h>
#include <storage/proc.h>
#include <storage/spin.h>
#include <tcop/tcopprot.h>
#include <utils/rel.h>
#include <utils/snapmgr.h>
#include <utils/wait_event.h>

#include "index/index_build.h"
#include "index/parallel_build.h"
#include "mkt_pg.h"
#include "mkt_vector.h"
#include "mktann_build.h"
#include "quant/matrix.h"

/*
 * PG-specific shared build state: the neutral MktBuildShared plus the relation
 * identity, query id, and the spinlock guarding its counters. A
 * ParallelTableScanDesc is appended after it in the DSM segment. The base is
 * the first member, so the MktBuildShared * the workers look up out of the toc
 * is recovered here as a MktBuildSharedPg *.
 */
typedef struct MktBuildSharedPg
{
	MktBuildShared base;
	Oid			   heaprelid;
	Oid			   indexrelid;
	int64		   queryid;
	slock_t		   mutex;
} MktBuildSharedPg;

#define ParallelTableScanFromMktShared(shared)  \
	((ParallelTableScanDesc)((char *)(shared) + \
							 BUFFERALIGN(sizeof(MktBuildSharedPg))))

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
	MktBuildShared	 *shared = shm_toc_lookup(toc, MKT_DSM_KEY_SHARED, false);
	MktBuildSharedPg *pg	 = (MktBuildSharedPg *)shared;
	Barrier *barrier		 = shm_toc_lookup(toc, MKT_DSM_KEY_BARRIER, false);

	char *sharedquery  = shm_toc_lookup(toc, MKT_DSM_KEY_QUERY_TEXT, true);
	debug_query_string = sharedquery;
	pgstat_report_activity(STATE_RUNNING, debug_query_string);
	pgstat_report_query_id(pg->queryid, false);

	w->shared	 = shared;
	w->barrier	 = barrier;
	w->heapRel	 = table_open(pg->heaprelid, ShareLock);
	w->indexRel	 = index_open(pg->indexrelid, AccessExclusiveLock);
	w->worker_id = ParallelWorkerNumber + 1;
	w->dim		 = shared->dim;

	InstrStartParallelQuery();

	/*
	 * Attach to the dynamic phase barrier (before the first phase). The leader
	 * attaches before launching workers, so the party tracks all participants
	 * that actually start and stays in lockstep regardless of how many workers
	 * PostgreSQL launched.
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
			shm_toc_lookup(toc, MKT_DSM_KEY_BUFFER_USAGE, false);
	WalUsage *walusage = shm_toc_lookup(toc, MKT_DSM_KEY_WAL_USAGE, false);
	InstrEndParallelQuery(
			&bufferusage[ParallelWorkerNumber],
			&walusage[ParallelWorkerNumber]);

	index_close(w->indexRel, AccessExclusiveLock);
	table_close(w->heapRel, ShareLock);
}

/*
 * Tear the parallel context down and leave parallel mode. The standalone
 * back-end provides a same-named function that joins its worker threads and
 * frees the shared arena instead.
 */
void
mkt_pbuild_teardown(ParallelContext *pcxt)
{
	DestroyParallelContext(pcxt);
	ExitParallelMode();
}

/*
 * Launch the worker participants and wait until they have all attached to the
 * barrier (so the dynamic party reaches launched+1 before the leader advances
 * the first phase). Returns false — after tearing the context down — if no
 * workers started, so the caller falls back to a serial build. The standalone
 * back-end provides a same-named function that spawns threads and joins them
 * at the barrier instead.
 */
bool
mkt_pbuild_launch(ParallelContext *pcxt, Barrier *barrier)
{
	LaunchParallelWorkers(pcxt);

	if (pcxt->nworkers_launched == 0)
	{
		WaitForParallelWorkersToFinish(pcxt);
		mkt_pbuild_teardown(pcxt);
		return false;
	}

	/*
	 * Workers attach to the barrier dynamically, so the party (1 leader + N
	 * launched) is not final until they all have; if the leader arrived first
	 * it could advance the phase alone and strand late workers.
	 * WaitForParallelWorkersToAttach surfaces a startup failure as an error
	 * rather than a hang; then poll until the live participant count is whole.
	 */
	WaitForParallelWorkersToAttach(pcxt);
	while (BarrierParticipants(barrier) < pcxt->nworkers_launched + 1)
	{
		CHECK_FOR_INTERRUPTS();
		(void)WaitLatch(
				MyLatch,
				WL_LATCH_SET | WL_TIMEOUT | WL_EXIT_ON_PM_DEATH,
				1L,
				WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);
		ResetLatch(MyLatch);
	}
	return true;
}

/*
 * Allocate and populate the parallel build's shared state: the DSM segment and
 * its regions (shared header, barrier, sample/centroid/assignment slots, the
 * tree blob, the per-worker page queues, usage counters). Does not launch
 * workers; the caller does. Coarse PG block — the standalone back-end provides
 * a same-named function over a heap arena. Returns false (after tearing the
 * parallel context down) if the DSM segment could not be created.
 */
bool
mkt_pbuild_setup_shared(
		MktPBuildLeader		 *lead,
		Relation			  heap,
		Relation			  index,
		const MktBuildConfig *config,
		int					  nworkers)
{
	Dimension dim			= config->dim;
	uint32_t  nlist			= config->nlist;
	int		  nparticipants = nworkers + 1;
	uint64_t  rabitq_seed	= 42;
	uint32_t  fan_out		= config->fan_out > 0 ? config->fan_out
												  : mkt_auto_fan_out(0, nlist, 0);
	uint32_t  km_k			= fan_out < nlist ? fan_out : nlist;

	/* Per-child subtree blob slot: each root child's subtree targets
	 * ~nlist/fan_out leaves; size the slot for that tree's worst case. */
	uint32_t nlist_c   = (nlist + fan_out - 1) / fan_out;
	uint64_t slot_size = mkt_hkmeans_max_blob_size(nlist_c, fan_out, dim);

	/* Compute sample budget per worker */
	uint32_t total_samples	= Max(10000, (int)(nlist * 256));
	uint32_t max_per_worker = (total_samples + nparticipants - 1) /
							  nparticipants;

	EnterParallelMode();

	ParallelContext *pcxt = CreateParallelContext(
			"meerkat", "mkt_parallel_build_main", nworkers);

	/* Estimate DSM size for ALL phases */
	Snapshot snapshot	= SnapshotAny;
	Size	 est_shared = add_size(
			BUFFERALIGN(sizeof(MktBuildSharedPg)),
			table_parallelscan_estimate(heap, snapshot));

	shm_toc_estimate_chunk(&pcxt->estimator, est_shared);
	shm_toc_estimate_chunk(&pcxt->estimator, sizeof(Barrier));
	/* Sampling */
	shm_toc_estimate_chunk(
			&pcxt->estimator,
			mkt_dsm_samples_size(nparticipants, max_per_worker, dim));
	/* K-means shared centroids + norms (root level, k=km_k) */
	shm_toc_estimate_chunk(
			&pcxt->estimator, mkt_dsm_centroids_size(km_k, dim));
	/* Per-child subtree blobs (work-partitioned phase 2c) */
	shm_toc_estimate_chunk(
			&pcxt->estimator, mkt_dsm_child_subtrees_size(fan_out, slot_size));
	/* K-means per-worker accumulators */
	shm_toc_estimate_chunk(
			&pcxt->estimator,
			mkt_dsm_km_workers_size(nparticipants, km_k, dim));
	/* Root assignments: per-worker uint32_t[max_per_worker] */
	shm_toc_estimate_chunk(
			&pcxt->estimator,
			mkt_dsm_root_assign_size(nparticipants, max_per_worker));
	/* Tree blob (placeholder — allocated later by leader, but
	 * we need the max possible size. Use a generous estimate.) */
	Size max_tree_sz = sizeof(HKMeansResult) +
					   (Size)nlist * 2 * sizeof(HKMeansNode) +
					   (Size)nlist * dim * sizeof(float) * 2;
	shm_toc_estimate_chunk(&pcxt->estimator, max_tree_sz);
	/* Bounded streaming posting phase: per-worker shm_mq queues carry full
	 * pages from the workers to the leader, which writes them. */
	shm_toc_estimate_chunk(
			&pcxt->estimator, mkt_dsm_posting_queues_size(nparticipants));
	/* Worker output (active flags) */
	shm_toc_estimate_chunk(
			&pcxt->estimator,
			mkt_dsm_worker_output_size(nlist, nparticipants));
	/* Per-worker trailing partial pages (both formats): the leader folds
	 * them into each list's head during finalize. */
	shm_toc_estimate_chunk(
			&pcxt->estimator, mkt_dsm_partials_size(nlist, nparticipants));

	shm_toc_estimate_chunk(
			&pcxt->estimator, mul_size(sizeof(WalUsage), pcxt->nworkers));
	shm_toc_estimate_chunk(
			&pcxt->estimator, mul_size(sizeof(BufferUsage), pcxt->nworkers));

	int querylen = 0;
	if (debug_query_string)
	{
		querylen = strlen(debug_query_string);
		shm_toc_estimate_chunk(&pcxt->estimator, querylen + 1);
	}

	/* nkeys: shared, barrier, samples, centroids, km_workers, root_assign,
	 * tree, posting_queues, worker_output, partials, child_subtrees, wal,
	 * buffer + optionally query_text */
	int nkeys = 13;
	if (debug_query_string)
		nkeys++;
	shm_toc_estimate_keys(&pcxt->estimator, nkeys);

	InitializeParallelDSM(pcxt);

	if (pcxt->seg == NULL)
	{
		mkt_pbuild_teardown(pcxt);
		return false;
	}

	/* ---- Populate shared state ---- */
	MktBuildSharedPg *pg		   = shm_toc_allocate(pcxt->toc, est_shared);
	MktBuildShared	 *shared	   = &pg->base;
	pg->heaprelid				   = RelationGetRelid(heap);
	pg->indexrelid				   = RelationGetRelid(index);
	pg->queryid					   = pgstat_get_my_query_id();
	shared->dim					   = dim;
	shared->metric				   = config->metric;
	shared->nlist				   = nlist;
	shared->fan_out				   = fan_out; /* resolved (auto if config 0) */
	shared->subtree_slot_size	   = slot_size;
	shared->soar_lambda			   = config->soar_lambda;
	shared->boundary_epsilon	   = config->boundary_epsilon;
	shared->fastscan			   = config->fastscan;
	shared->centroid_format		   = config->centroid_format;
	shared->rabitq_seed			   = rabitq_seed;
	shared->nparticipants		   = nparticipants;
	shared->max_samples_per_worker = max_per_worker;
	shared->km_max_iterations	   = 20;
	shared->km_nredo = config->kmeans_nredo > 0 ? config->kmeans_nredo : 1;
	shared->avq_eta				   = (float)config->avq_eta;
	shared->km_tolerance		   = 1e-4f;
	shared->km_k				   = km_k;
	shared->km_converged		   = false;
	SpinLockInit(&pg->mutex);
	shared->reltuples  = 0.0;
	shared->indtuples  = 0.0;
	shared->soar_dupes = 0.0;
	table_parallelscan_initialize(
			heap, ParallelTableScanFromMktShared(shared), snapshot);
	shm_toc_insert(pcxt->toc, MKT_DSM_KEY_SHARED, shared);

	/*
	 * Barrier for phase synchronization. It is a *dynamic* barrier (init with
	 * 0 parties): the leader and every launched worker join via BarrierAttach,
	 * so the party tracks the participants that actually start. A non-zero
	 * (static) party is wrong here — BarrierAttach forbids it
	 * (Assert(!static_party)), and a fixed planned count would deadlock when
	 * PostgreSQL launches fewer workers than planned. The leader attaches in
	 * do_parallel_build before launching workers.
	 */
	Barrier *barrier = shm_toc_allocate(pcxt->toc, sizeof(Barrier));
	BarrierInit(barrier, 0);
	shm_toc_insert(pcxt->toc, MKT_DSM_KEY_BARRIER, barrier);

	/* Sample slots */
	Size samp_sz = mkt_dsm_samples_size(nparticipants, max_per_worker, dim);
	MktDsmSamples *dsm_samples = shm_toc_allocate(pcxt->toc, samp_sz);
	/* Only the header + per-participant counts are read before being written;
	 * the sample data is filled by the sampling pass and read back bounded by
	 * those counts, so zeroing the (multi-GB) data region is wasted work. */
	memset(dsm_samples,
		   0,
		   MAXALIGN(sizeof(MktDsmSamples)) +
				   (size_t)nparticipants * sizeof(uint32_t));
	dsm_samples->nparticipants	= nparticipants;
	dsm_samples->max_per_worker = max_per_worker;
	dsm_samples->dim			= dim;
	shm_toc_insert(pcxt->toc, MKT_DSM_KEY_SAMPLES, dsm_samples);

	/* Shared centroids + norms (root k-means, k=km_k) */
	Size  cent_sz		 = mkt_dsm_centroids_size(km_k, dim);
	char *centroids_base = shm_toc_allocate(pcxt->toc, cent_sz);
	memset(centroids_base, 0, cent_sz);
	shm_toc_insert(pcxt->toc, MKT_DSM_KEY_CENTROIDS, centroids_base);
	float *cents = mkt_dsm_centroids(centroids_base);

	/* Per-child subtree blobs (phase 2c, work-partitioned): each participant
	 * builds the full subtree for the root children it owns into its slot, and
	 * the leader grafts them into the final tree. */
	char *child_subtrees_base = shm_toc_allocate(
			pcxt->toc, mkt_dsm_child_subtrees_size(fan_out, slot_size));
	shm_toc_insert(pcxt->toc, MKT_DSM_KEY_CHILD_SUBTREES, child_subtrees_base);

	/* Per-worker k-means accumulators */
	Size  km_sz			  = mkt_dsm_km_workers_size(nparticipants, km_k, dim);
	char *km_workers_base = shm_toc_allocate(pcxt->toc, km_sz);
	memset(km_workers_base, 0, km_sz);
	shm_toc_insert(pcxt->toc, MKT_DSM_KEY_KM_WORKERS, km_workers_base);

	/* Root assignment slots */
	Size ra_sz = mkt_dsm_root_assign_size(nparticipants, max_per_worker);
	MktDsmRootAssign *dsm_ra = shm_toc_allocate(pcxt->toc, ra_sz);
	memset(dsm_ra, 0, ra_sz);
	dsm_ra->nparticipants  = nparticipants;
	dsm_ra->max_per_worker = max_per_worker;
	shm_toc_insert(pcxt->toc, MKT_DSM_KEY_ROOT_ASSIGN, dsm_ra);

	/* Tree blob — allocated now, populated after k-means */
	void *dsm_tree = shm_toc_allocate(pcxt->toc, max_tree_sz);
	memset(dsm_tree, 0, max_tree_sz);
	shm_toc_insert(pcxt->toc, MKT_DSM_KEY_TREE, dsm_tree);

	/* Per-worker shm_mq posting-page queues. The leader is the receiver of
	 * every queue; the launched workers attach as senders in phase 3 and
	 * stream their full pages. Create and register the receiver here,
	 * before launch. */
	Size  queues_sz	  = mkt_dsm_posting_queues_size(nparticipants);
	char *queues_base = shm_toc_allocate(pcxt->toc, queues_sz);
	memset(queues_base, 0, queues_sz);
	for (int i = 0; i < nparticipants; i++)
	{
		shm_mq *mq = shm_mq_create(
				mkt_dsm_posting_queue(queues_base, i),
				mkt_dsm_posting_queue_bytes());
		shm_mq_set_receiver(mq, MyProc);
	}
	shm_toc_insert(pcxt->toc, MKT_DSM_KEY_POSTING_QUEUES, queues_base);

	/* Worker output (active flags) */
	Size  out_sz		= mkt_dsm_worker_output_size(nlist, nparticipants);
	char *worker_output = shm_toc_allocate(pcxt->toc, out_sz);
	memset(worker_output, 0, out_sz);
	shm_toc_insert(pcxt->toc, MKT_DSM_KEY_WORKER_OUTPUT, worker_output);

	/* Per-worker trailing partial pages (both AoS and fastscan). Each worker
	 * holds at most one partial page per cluster here; the leader folds them
	 * into the list's head during finalize. */
	Size  part_sz	   = mkt_dsm_partials_size(nlist, nparticipants);
	char *dsm_partials = shm_toc_allocate(pcxt->toc, part_sz);
	memset(dsm_partials, 0, part_sz);
	shm_toc_insert(pcxt->toc, MKT_DSM_KEY_PARTIALS, dsm_partials);

	WalUsage *walusage = shm_toc_allocate(
			pcxt->toc, mul_size(sizeof(WalUsage), pcxt->nworkers));
	memset(walusage, 0, mul_size(sizeof(WalUsage), pcxt->nworkers));
	shm_toc_insert(pcxt->toc, MKT_DSM_KEY_WAL_USAGE, walusage);

	BufferUsage *bufferusage = shm_toc_allocate(
			pcxt->toc, mul_size(sizeof(BufferUsage), pcxt->nworkers));
	memset(bufferusage, 0, mul_size(sizeof(BufferUsage), pcxt->nworkers));
	shm_toc_insert(pcxt->toc, MKT_DSM_KEY_BUFFER_USAGE, bufferusage);

	if (debug_query_string)
	{
		char *sq = shm_toc_allocate(pcxt->toc, querylen + 1);
		memcpy(sq, debug_query_string, querylen + 1);
		shm_toc_insert(pcxt->toc, MKT_DSM_KEY_QUERY_TEXT, sq);
	}

	lead->pcxt				  = pcxt;
	lead->shared			  = shared;
	lead->barrier			  = barrier;
	lead->dsm_samples		  = dsm_samples;
	lead->centroids_base	  = centroids_base;
	lead->cents				  = cents;
	lead->child_subtrees_base = child_subtrees_base;
	lead->km_workers_base	  = km_workers_base;
	lead->dsm_ra			  = dsm_ra;
	lead->dsm_tree			  = dsm_tree;
	lead->queues_base		  = queues_base;
	lead->dsm_partials		  = dsm_partials;
	lead->walusage			  = walusage;
	lead->bufferusage		  = bufferusage;
	lead->nparticipants		  = nparticipants;
	lead->km_k				  = km_k;
	lead->max_per_worker	  = max_per_worker;
	lead->dim				  = dim;
	lead->nlist				  = nlist;
	lead->rabitq_seed		  = rabitq_seed;
	lead->fan_out			  = fan_out;
	lead->max_tree_sz		  = max_tree_sz;
	return true;
}

/*
 * Accumulate one worker's tuple counts into the shared state under the lock.
 * Back-end seam: only the spinlock is PG-specific (it lives in the derived
 * struct); the standalone version locks a pthread mutex around the same adds.
 */
void
mkt_pbuild_worker_add_counts(
		MktBuildShared *shared, double indtuples, double soar_dupes)
{
	MktBuildSharedPg *pg = (MktBuildSharedPg *)shared;

	SpinLockAcquire(&pg->mutex);
	shared->indtuples += indtuples;
	shared->soar_dupes += soar_dupes;
	SpinLockRelease(&pg->mutex);
}

/*
 * Re-initialize the parallel scan for the posting pass; the sampling pass
 * consumed the first one. Back-end seam: the standalone version resets its
 * work-stealing cursor.
 */
void
mkt_pbuild_rescan(Relation heap, MktBuildShared *shared)
{
	table_parallelscan_reinitialize(
			heap, ParallelTableScanFromMktShared(shared));
}
