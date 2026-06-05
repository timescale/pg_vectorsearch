/*
 * test_parallel_ctx.c - Parallel context lifecycle (standalone) tests
 *
 * Drives the full lifecycle the way the build's leader does — create,
 * estimate, InitializeParallelDSM, publish a barrier + accumulator into the
 * toc, launch, participate as worker_id 0, wait, destroy — and checks the
 * workers ran SPMD: each adds its worker_id through a barrier-synchronized
 * reduction.
 */

#include <stdatomic.h>
#include <stdint.h>

#include "mkt_test.h"
#include "standalone/barrier.h"
#include "standalone/parallel_ctx.h"

TEST_GROUP(ParallelCtx);

#define PC_KEY_BARRIER 1
#define PC_KEY_ACCUM   2
#define PC_WORKERS	   4

/* SPMD worker: attach, add worker_id to the shared accumulator, detach. */
static void
pc_sum_worker(dsm_segment *seg, shm_toc *toc)
{
	Barrier		 *b		= shm_toc_lookup(toc, PC_KEY_BARRIER, false);
	_Atomic(int) *accum = shm_toc_lookup(toc, PC_KEY_ACCUM, false);
	int			  wid	= ParallelWorkerNumber + 1; /* 1..N */

	(void)seg;

	BarrierAttach(b);
	BarrierArriveAndWait(b, 0); /* all attached (leader + workers) */
	atomic_fetch_add(accum, wid);
	BarrierArriveAndWait(b, 0); /* all added */
	BarrierDetach(b);
}

TEST(spmd_lifecycle_barrier_toc)
{
	mkt_parallel_register_worker("pc_sum_worker", pc_sum_worker);

	EnterParallelMode();

	ParallelContext *pcxt =
			CreateParallelContext("meerkat", "pc_sum_worker", PC_WORKERS);

	shm_toc_estimate_chunk(&pcxt->estimator, sizeof(Barrier));
	shm_toc_estimate_chunk(&pcxt->estimator, sizeof(_Atomic(int)));
	shm_toc_estimate_keys(&pcxt->estimator, 2);

	InitializeParallelDSM(pcxt);
	ASSERT_NOT_NULL(pcxt->seg, "DSM is ready (non-NULL sentinel)");

	Barrier *b = shm_toc_allocate(pcxt->toc, sizeof(Barrier));
	BarrierInit(b, 1); /* the leader is the initial party */
	shm_toc_insert(pcxt->toc, PC_KEY_BARRIER, b);

	_Atomic(int) *accum = shm_toc_allocate(pcxt->toc, sizeof(_Atomic(int)));
	atomic_store(accum, 0);
	shm_toc_insert(pcxt->toc, PC_KEY_ACCUM, accum);

	LaunchParallelWorkers(pcxt);
	ASSERT_EQ(PC_WORKERS, pcxt->nworkers_launched, "all workers launched");

	WaitForParallelWorkersToAttach(pcxt);
	while (BarrierParticipants(b) < pcxt->nworkers_launched + 1)
		sched_yield();

	/* Leader participates as worker_id 0 (ParallelWorkerNumber == -1). */
	BarrierArriveAndWait(b, 0);
	atomic_fetch_add(accum, ParallelWorkerNumber + 1);
	BarrierArriveAndWait(b, 0);
	BarrierDetach(b);

	WaitForParallelWorkersToFinish(pcxt);

	int sum = atomic_load(accum);
	DestroyParallelContext(pcxt);
	ExitParallelMode();

	/* leader 0 + workers 1+2+3+4 = 10. */
	ASSERT_EQ(
			10, sum, "SPMD reduction across leader + workers via barrier+toc");
}
