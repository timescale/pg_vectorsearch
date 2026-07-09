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
#include <catalog/pg_operator_d.h>
#include <catalog/pg_type_d.h>
#include <executor/tuptable.h>
#include <miscadmin.h>
#include <optimizer/plancat.h>
#include <pgstat.h>
#include <storage/latch.h>
#include <storage/proc.h>
#include <storage/spin.h>
#include <tcop/tcopprot.h>
#include <utils/rel.h>
#include <utils/snapmgr.h>
#include <utils/tuplesort.h>
#include <utils/wait_event.h>

#include "index/index_build.h"
#include "index/parallel_build.h"
#include "mkt_pg.h"
#include "mkt_vector.h"
#include "mktann_build.h"
#include "mktann_storage.h"
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
	/* Striped locks guarding the shared leaf-refinement accumulator. */
	slock_t accum_locks[MKT_REFINE_LOCK_STRIPES];
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
 * Relation lock modes a parallel worker uses for the heap and index. A
 * concurrent build (CREATE INDEX CONCURRENTLY) must take weak locks: the
 * leader holds only ShareUpdateExclusive on the heap, and acquiring a strong
 * index lock in a worker logs it for hot-standby and assigns an XID, which is
 * illegal in a parallel worker. Matches PostgreSQL's btree parallel build. The
 * same pair is used to open the relations on attach and to close them on
 * detach.
 */
static void
worker_lockmodes(bool concurrent, LOCKMODE *heapmode, LOCKMODE *indexmode)
{
	*heapmode  = concurrent ? ShareUpdateExclusiveLock : ShareLock;
	*indexmode = concurrent ? RowExclusiveLock : AccessExclusiveLock;
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

	LOCKMODE heapmode, indexmode;
	worker_lockmodes(shared->concurrent, &heapmode, &indexmode);

	w->shared	 = shared;
	w->barrier	 = barrier;
	w->heapRel	 = table_open(pg->heaprelid, heapmode);
	w->indexRel	 = index_open(pg->indexrelid, indexmode);
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

	LOCKMODE heapmode, indexmode;
	worker_lockmodes(w->shared->concurrent, &heapmode, &indexmode);
	index_close(w->indexRel, indexmode);
	table_close(w->heapRel, heapmode);
}

/*
 * Page-backed routing storage seam (see parallel_build.h). PG workers are
 * separate processes, so each opens its own MktStorage on the worker's index
 * relation; the leader's storage pointer cannot cross the process boundary, so
 * publish is a no-op here.
 */
void
mkt_pbuild_publish_storage(MktBuildShared *shared, MktStorage *s)
{
	(void)shared;
	(void)s;
}

MktStorage *
mkt_pbuild_worker_storage(MktPBuildWorker *w)
{
	/* No table relation needed (routing reads index pages only, no rerank). */
	MktannStorage *s = palloc(sizeof(MktannStorage));
	mktann_storage_init(s, w->indexRel, NULL, w->shared->metric);
	s->build_mode = true; /* reads only; matches the leader's build storage */
	return &s->base;	  /* base is the first member */
}

void
mkt_pbuild_worker_storage_release(MktStorage *s)
{
	/* The route helper releases every page it reads, so no buffer stays
	 * pinned; just free the wrapper (allocated in the worker's memory
	 * context). */
	pfree(s);
}

/*
 * Snapshot used to initialize the parallel heap scan for CREATE INDEX
 * CONCURRENTLY (an MVCC snapshot; SnapshotAny/NULL for a normal build). It
 * must stay registered for the whole parallel operation and is released at
 * teardown. A file-static is safe: an index build is single-threaded and
 * non-reentrant in the leader backend, and every parallel-build exit path runs
 * mkt_pbuild_teardown.
 */
static Snapshot mkt_pbuild_snapshot = NULL;

/*
 * Tear the parallel context down and leave parallel mode. The standalone
 * back-end provides a same-named function that joins its worker threads and
 * frees the shared arena instead.
 */
void
mkt_pbuild_teardown(ParallelContext *pcxt)
{
	if (mkt_pbuild_snapshot != NULL)
	{
		UnregisterSnapshot(mkt_pbuild_snapshot);
		mkt_pbuild_snapshot = NULL;
	}
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
mkt_pbuild_launch(
		ParallelContext *pcxt, Barrier *barrier, MktBuildShared *shared)
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

	/* The launch can fall short of the plan (the parallel-worker pool under
	 * max_parallel_workers is shared with concurrent queries). Narrow the
	 * participant count to the party that attached so the phases partition
	 * their work over participants that exist; the per-participant DSM
	 * regions keep their planned size and leave the tail slots unused. */
	shared->nparticipants = pcxt->nworkers_launched + 1;
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

	/*
	 * Size the k-means sample set. The samples live in one shared-memory
	 * region (total_samples * dim floats) that must stay resident for the
	 * whole tree build -- root k-means and every subtree -- so its size is the
	 * build's dominant memory cost. The ideal is ~256 samples per list, but at
	 * fine nlist that can dwarf available RAM (nlist=480k -> 123M samples ->
	 * ~360 GB), which previously overflowed the DSM.
	 *
	 * Bound it by maintenance_work_mem: that is the build's memory budget and
	 * the knob operators already raise for large index builds. When the budget
	 * is smaller than the ideal the stride sampler simply draws a coarser (but
	 * still uniform) subsample to fit. estimate_rel_size() caps it to the rows
	 * that actually exist so small tables don't over-allocate; it is only a
	 * hint now -- the budget is the hard bound, so an inaccurate estimate can
	 * no longer over-commit shared memory.
	 */
	uint64_t want_samples = (uint64_t)nlist * 256;

	/* maintenance_work_mem is in kB; reserve it for the sample region. */
	uint64_t mem_bytes = (uint64_t)maintenance_work_mem * UINT64CONST(1024);
	uint64_t per_vec   = (uint64_t)dim * sizeof(float);
	uint64_t budget	   = per_vec > 0 ? mem_bytes / per_vec : want_samples;
	if (budget < 10000)
		budget = 10000; /* k-means needs a workable minimum */

	uint64_t total64 = Min(want_samples, budget);

	BlockNumber est_pages;
	double		est_tuples;
	double		allvisfrac;
	estimate_rel_size(heap, NULL, &est_pages, &est_tuples, &allvisfrac);
	if (est_tuples > 0.0 && (double)total64 > est_tuples)
		total64 = (uint64_t)est_tuples;

	/* total64 <= want_samples = nlist*256 <= 512M (nlist reloption max 2M), so
	 * it always fits a uint32. */
	uint32_t total_samples = (uint32_t)Max(total64, UINT64CONST(1));

	if (want_samples > budget && est_tuples > (double)budget)
		elog(LOG,
			 "meerkat: k-means sample set limited to %u of the ideal %lu "
			 "vectors by maintenance_work_mem (%d kB); raise "
			 "maintenance_work_mem for finer centroid training on large "
			 "tables",
			 total_samples,
			 (unsigned long)want_samples,
			 maintenance_work_mem);

	uint32_t max_per_worker = (total_samples + nparticipants - 1) /
							  nparticipants;

	/*
	 * Refine leaf centroids on the full table afterward only when the
	 * structure was built from a strict subset (i.e. the sample was
	 * budget-bounded below the table). Both leader and workers gate the refine
	 * phase on shared->refine_iters so they run the identical barrier
	 * sequence.
	 */
	uint32_t refine_iters = ((double)total_samples < est_tuples &&
							 mkt_leaf_refine_iters > 0)
								  ? (uint32_t)mkt_leaf_refine_iters
								  : 0;

	EnterParallelMode();

	ParallelContext *pcxt = CreateParallelContext(
			"meerkat", "mkt_parallel_build_main", nworkers);

	/*
	 * The heap scan's snapshot. A normal build sees all tuples (SnapshotAny);
	 * CREATE INDEX CONCURRENTLY must use an MVCC snapshot so it indexes only
	 * tuples visible to it (heapam asserts SnapshotAny <-> a valid OldestXmin,
	 * so the concurrent path must not pass SnapshotAny). Register it for the
	 * duration — its serialized size also affects the DSM estimate below — and
	 * release it in mkt_pbuild_teardown. Mirrors PostgreSQL's nbtsort.c.
	 */
	Snapshot snapshot	= config->concurrent
								? RegisterSnapshot(GetTransactionSnapshot())
								: SnapshotAny;
	mkt_pbuild_snapshot = (snapshot != SnapshotAny) ? snapshot : NULL;
	Size est_shared		= add_size(
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
	/* Shared coordinator for the cluster-keyed posting sort (sort seam). Sized
	 * for the planned participant count (upper bound on launched workers). */
	shm_toc_estimate_chunk(
			&pcxt->estimator, mkt_pbuild_sort_shared_size(nparticipants));

	/* Page-backed routing: leaf posting-head blocks (head->leaf map) and the
	 * global mean, published by the leader before the tree-ready barrier.
	 * Sized for the worst-case leaf count (nlist here is the max bound). */
	shm_toc_estimate_chunk(
			&pcxt->estimator, (Size)nlist * sizeof(BlockNumber));
	shm_toc_estimate_chunk(&pcxt->estimator, (Size)dim * sizeof(float));

	/* Shared leaf-refinement accumulator (one copy; only when refining). Sized
	 * to a bounded tile (cap_bytes = min(maintenance_work_mem, MaxAllocSize)),
	 * not O(nlist): refine processes leaves in tiles of this capacity. */
	uint64_t refine_cap_bytes =
			Min((uint64_t)maintenance_work_mem * 1024, (uint64_t)MaxAllocSize);
	uint32_t refine_tile_cap =
			mkt_refine_tile_leaves(nlist, dim, refine_cap_bytes);
	if (refine_iters > 0)
		shm_toc_estimate_chunk(
				&pcxt->estimator,
				mkt_dsm_refine_accum_size(refine_tile_cap, dim));

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
	 * tree, sortshared, child_subtrees, wal, buffer, posting_heads,
	 * global_mean + optionally refine_accum / query_text */
	int nkeys = 13;
	if (debug_query_string)
		nkeys++;
	if (refine_iters > 0)
		nkeys++;
	shm_toc_estimate_keys(&pcxt->estimator, nkeys);

	/* Total bytes the leader is about to commit to the DSM segment (sum of all
	 * estimated chunks), for the planned-allocation introspection line. */
	Size dsm_total = pcxt->estimator.space_for_chunks;

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
	shared->concurrent			   = config->concurrent;
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
	shared->work_mem_kb			   = maintenance_work_mem;
	shared->max_samples_per_worker = max_per_worker;
	shared->km_max_iterations	   = 20;
	shared->km_tolerance		   = 1e-4f;
	shared->km_k				   = km_k;
	shared->km_converged		   = false;
	shared->refine_iters		   = refine_iters;
	shared->centroid_error_scale   = (float)mkt_centroid_error_scale;
	shared->centroid_beam_scale	   = (float)mkt_centroid_beam_scale;
	shared->fastscan_bits		   = mkt_fastscan_bits;
	SpinLockInit(&pg->mutex);
	for (int i = 0; i < MKT_REFINE_LOCK_STRIPES; i++)
		SpinLockInit(&pg->accum_locks[i]);
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

	/* Shared coordinator for the cluster-keyed posting sort (sort seam). Sized
	 * for the planned participant count; the leader calls
	 * mkt_pbuild_sort_shared_init with the actual launched count after launch.
	 */
	Size  sort_sz	 = mkt_pbuild_sort_shared_size(nparticipants);
	void *sortshared = shm_toc_allocate(pcxt->toc, sort_sz);
	memset(sortshared, 0, sort_sz);
	shm_toc_insert(pcxt->toc, MKT_DSM_KEY_SORTSHARED, sortshared);

	/* Page-backed routing regions (filled by the leader before the tree-ready
	 * barrier): the leaf posting-head blocks and the global mean. */
	BlockNumber *dsm_heads =
			shm_toc_allocate(pcxt->toc, (Size)nlist * sizeof(BlockNumber));
	memset(dsm_heads, 0, (Size)nlist * sizeof(BlockNumber));
	shm_toc_insert(pcxt->toc, MKT_DSM_KEY_POSTING_HEADS, dsm_heads);

	float *dsm_gmean = shm_toc_allocate(pcxt->toc, (Size)dim * sizeof(float));
	memset(dsm_gmean, 0, (Size)dim * sizeof(float));
	shm_toc_insert(pcxt->toc, MKT_DSM_KEY_GLOBAL_MEAN, dsm_gmean);

	/* Shared leaf-refinement accumulator, sized to the bounded tile capacity
	 * (refine_tile_cap); the refine exec processes leaves in tiles of this
	 * size, re-scanning the heap per tile. accum->nleaves carries the capacity
	 * so the leader and workers derive the tile count identically. Only when
	 * refining. */
	if (refine_iters > 0)
	{
		Size acc_sz = mkt_dsm_refine_accum_size(refine_tile_cap, dim);
		MktDsmRefineAccum *accum = shm_toc_allocate(pcxt->toc, acc_sz);
		accum->nleaves			 = refine_tile_cap;
		accum->dim				 = dim;
		shm_toc_insert(pcxt->toc, MKT_DSM_KEY_REFINE_ACCUM, accum);
	}

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
	lead->queues_base		  = NULL; /* sort-seam path: no shm_mq queues */
	lead->dsm_partials		  = NULL; /* sort-seam path: no partials region */
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
	lead->dsm_total			  = dsm_total;
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

/*
 * Striped-lock seam for the shared leaf-refinement accumulator. Contention is
 * low because rows spread across nleaves leaves into MKT_REFINE_LOCK_STRIPES
 * stripes.
 */
void
mkt_pbuild_accum_lock(MktBuildShared *shared, uint32_t stripe)
{
	MktBuildSharedPg *pg = (MktBuildSharedPg *)shared;
	SpinLockAcquire(&pg->accum_locks[stripe]);
}

void
mkt_pbuild_accum_unlock(MktBuildShared *shared, uint32_t stripe)
{
	MktBuildSharedPg *pg = (MktBuildSharedPg *)shared;
	SpinLockRelease(&pg->accum_locks[stripe]);
}

/* ----------------------------------------------------------------
 * Posting sort seam (PG back-end) — parallel tuplesort
 *
 * Each entry is sorted by a uint32 cluster id; the payload is a fixed-size
 * opaque blob carried in a bytea column. Workers feed partial runs into the
 * shared Sharedsort (in the build DSM); the leader merges and reads them back
 * grouped by cluster. Memory is bounded by maintenance_work_mem (tuplesort
 * spills past it). The standalone back-end provides the same-named seam over
 * in-memory arrays.
 * ---------------------------------------------------------------- */
struct MktSorter
{
	Tuplesortstate *ts;
	TupleDesc		tupdesc;
	TupleTableSlot *slot;
	SortCoordinate	coord;
	uint32_t		entry_size;
	bytea		   *payload; /* reusable scratch: VARHDRSZ + entry_size */
	bool			is_leader;
};

Size
mkt_pbuild_sort_shared_size(int nparticipants)
{
	return tuplesort_estimate_shared(nparticipants);
}

void
mkt_pbuild_sort_shared_init(void *region, int nparticipants, void *seg)
{
	tuplesort_initialize_shared(
			(Sharedsort *)region, nparticipants, (dsm_segment *)seg);
}

MktSorter *
mkt_pbuild_sort_begin(
		void	*region,
		void	*seg,
		int		 participant,
		int		 nparticipants,
		bool	 is_leader,
		uint32_t entry_size,
		int		 work_mem_kb)
{
	(void)participant;
	MktSorter *s  = palloc0(sizeof(MktSorter));
	s->entry_size = entry_size;
	s->is_leader  = is_leader;

	s->tupdesc = CreateTemplateTupleDesc(2);
	TupleDescInitEntry(s->tupdesc, 1, "cluster", INT4OID, -1, 0);
	TupleDescInitEntry(s->tupdesc, 2, "payload", BYTEAOID, -1, 0);

	/* region == NULL: a plain, non-parallel sort (the serial build) — no
	 * coordinate, no attach. Otherwise a parallel participant: a leader
	 * merging runs, or a worker producing one (which attaches to the shared
	 * fileset). */
	if (region != NULL)
	{
		s->coord			 = palloc0(sizeof(SortCoordinateData));
		s->coord->sharedsort = (Sharedsort *)region;
		if (is_leader)
		{
			s->coord->isWorker		= false;
			s->coord->nParticipants = nparticipants;
		}
		else
		{
			s->coord->isWorker		= true;
			s->coord->nParticipants = -1;
		}
	}

	AttrNumber attNums[1]	= {1};
	Oid		   sortOps[1]	= {Int4LessOperator};
	Oid		   sortColls[1] = {InvalidOid};
	bool	   nullsF[1]	= {false};
	s->ts					= tuplesort_begin_heap(
			  s->tupdesc,
			  1,
			  attNums,
			  sortOps,
			  sortColls,
			  nullsF,
			  work_mem_kb,
			  s->coord,
			  TUPLESORT_NONE);

	/* Workers attach to the shared fileset; the leader holds it via the
	 * backend's dsm reference and must not attach (per tuplesort.h). */
	if (region != NULL && !is_leader)
		tuplesort_attach_shared((Sharedsort *)region, (dsm_segment *)seg);

	s->slot	   = MakeSingleTupleTableSlot(s->tupdesc, &TTSOpsMinimalTuple);
	s->payload = (bytea *)palloc(VARHDRSZ + entry_size);
	SET_VARSIZE(s->payload, VARHDRSZ + entry_size);
	return s;
}

void
mkt_pbuild_sort_put(MktSorter *s, uint32_t cluster, const void *entry)
{
	memcpy(VARDATA(s->payload), entry, s->entry_size);
	ExecClearTuple(s->slot);
	s->slot->tts_values[0] = Int32GetDatum((int32)cluster);
	s->slot->tts_isnull[0] = false;
	s->slot->tts_values[1] = PointerGetDatum(s->payload);
	s->slot->tts_isnull[1] = false;
	ExecStoreVirtualTuple(s->slot);
	tuplesort_puttupleslot(s->ts, s->slot);
}

void
mkt_pbuild_sort_performsort(MktSorter *s)
{
	tuplesort_performsort(s->ts);
}

bool
mkt_pbuild_sort_getnext(MktSorter *s, uint32_t *cluster, const void **entry)
{
	bool isnull;
	if (!tuplesort_gettupleslot(s->ts, true, false, s->slot, NULL))
		return false;
	*cluster  = (uint32_t)DatumGetInt32(slot_getattr(s->slot, 1, &isnull));
	bytea *pl = DatumGetByteaPP(slot_getattr(s->slot, 2, &isnull));
	*entry	  = VARDATA_ANY(pl);
	return true;
}

void
mkt_pbuild_sort_end(MktSorter *s)
{
	tuplesort_end(s->ts);
	ExecDropSingleTupleTableSlot(s->slot);
	FreeTupleDesc(s->tupdesc);
	if (s->coord != NULL) /* NULL for a non-parallel (serial) sort */
		pfree(s->coord);
	pfree(s->payload);
	pfree(s);
}
