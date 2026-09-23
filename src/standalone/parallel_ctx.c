/*
 * parallel_ctx.c - Parallel context lifecycle over pthreads
 *
 * Standalone implementation of the ParallelContext lifecycle the build uses
 * (see vs_parallel_ctx.h). PG builds use PostgreSQL's ParallelContext, so
 * this file is compiled only for standalone.
 */

#ifdef VS_STANDALONE

#include <stdlib.h>
#include <string.h>

#include "core/log.h"
#include "core/memory.h"
#include "standalone/parallel_ctx.h"

#define VS_PARALLEL_TOC_MAGIC		UINT64_C(0x56535f504152) /* "VS_PAR" */
#define VS_PARALLEL_MAX_WORKERS_REG 8

__thread int ParallelWorkerNumber = -1;

/* Name -> worker-entry registry (populated once at module init). */
static struct
{
	const char		  *name;
	VsParallelWorkerFn fn;
} vs_worker_registry[VS_PARALLEL_MAX_WORKERS_REG];

static int vs_worker_registry_count = 0;

void
vs_parallel_register_worker(const char *name, VsParallelWorkerFn fn)
{
	if (vs_worker_registry_count == VS_PARALLEL_MAX_WORKERS_REG)
	{
		vs_error("vs_parallel_register_worker: registry full");
	}
	vs_worker_registry[vs_worker_registry_count].name = name;
	vs_worker_registry[vs_worker_registry_count].fn	  = fn;
	vs_worker_registry_count++;
}

static VsParallelWorkerFn
vs_worker_lookup(const char *name)
{
	for (int i = 0; i < vs_worker_registry_count; i++)
		if (strcmp(vs_worker_registry[i].name, name) == 0)
			return vs_worker_registry[i].fn;

	vs_error("vs_worker_lookup: '%s' not registered", name);
}

void
EnterParallelMode(void)
{
}

void
ExitParallelMode(void)
{
}

ParallelContext *
CreateParallelContext(const char *library, const char *function, int nworkers)
{
	ParallelContext *pcxt = calloc(1, sizeof(ParallelContext));

	(void)library;

	if (pcxt == NULL)
	{
		vs_error("CreateParallelContext: out of memory");
	}

	pcxt->nworkers			= nworkers;
	pcxt->nworkers_launched = 0;
	pcxt->function_name		= function;
	pcxt->pool				= vs_thread_pool_create((uint32_t)nworkers);
	shm_toc_initialize_estimator(&pcxt->estimator);

	/*
	 * Bind the leader's latch now and keep it in the context: workers set it
	 * (via the page queues' receiver) while the leader drains, so it must
	 * outlive them — the context is freed only after they are all joined.
	 */
	InitLatch(&pcxt->leader_latch);
	vs_latch_attach_self(&pcxt->leader_latch);
	return pcxt;
}

void
InitializeParallelDSM(ParallelContext *pcxt)
{
	pcxt->arena_size = shm_toc_estimate(&pcxt->estimator);
	pcxt->arena		 = malloc(pcxt->arena_size);
	if (pcxt->arena == NULL)
	{
		vs_error("InitializeParallelDSM: out of memory");
	}
	pcxt->toc = shm_toc_create(
			VS_PARALLEL_TOC_MAGIC, pcxt->arena, pcxt->arena_size);

	/*
	 * A non-NULL sentinel: PG sets seg to the DSM segment and the driver
	 * treats NULL as "could not start". Standalone always starts, so seg just
	 * has to be non-NULL; nothing dereferences it.
	 */
	pcxt->seg = (dsm_segment *)pcxt;
}

/*
 * Pool trampoline: runs on a pool worker thread as participant 1..nworkers.
 * Binds this thread's worker number (ParallelWorkerNumber = participant - 1)
 * and its stable latch, then runs the registered entry. The pool runs this
 * once per worker per launch; afterwards the worker waits at the pool barrier
 * for the leader's join (WaitForParallelWorkersToFinish).
 */
static void
vs_pool_worker_trampoline(uint32_t participant_id, void *arg)
{
	ParallelContext *pcxt		   = (ParallelContext *)arg;
	int				 worker_number = (int)participant_id - 1;

	ParallelWorkerNumber = worker_number;
	vs_latch_attach_self(&pcxt->worker_latches[worker_number]);

	/*
	 * Each worker thread needs its own current memory context — the analog of
	 * a PG worker process's CurrentMemoryContext — for the entry's allocations
	 * (the worker's per-phase contexts are created under it). Freed when the
	 * worker returns.
	 */
	VsMemCtx wctx = vs_memctx_create(NULL, "vs parallel worker");
	VsMemCtx prev = vs_memctx_switch(wctx);
	pcxt->worker_fn(pcxt->seg, pcxt->toc);
	vs_memctx_switch(prev);
	vs_memctx_delete(wctx);
}

void
LaunchParallelWorkers(ParallelContext *pcxt)
{
	if (pcxt->nworkers == 0)
	{
		pcxt->nworkers_launched = 0;
		return;
	}

	pcxt->worker_fn		 = vs_worker_lookup(pcxt->function_name);
	pcxt->worker_latches = calloc(pcxt->nworkers, sizeof(Latch));
	if (pcxt->worker_latches == NULL)
	{
		vs_error("LaunchParallelWorkers: out of memory");
	}
	for (int i = 0; i < pcxt->nworkers; i++)
		InitLatch(&pcxt->worker_latches[i]);

	/*
	 * Dispatch the entry onto the pool's worker threads and return: the leader
	 * then participates in k-means and drains the workers' streamed pages
	 * concurrently, joining them in WaitForParallelWorkersToFinish.
	 */
	vs_thread_pool_launch(pcxt->pool, vs_pool_worker_trampoline, pcxt);
	pcxt->nworkers_launched = pcxt->nworkers;
}

void
WaitForParallelWorkersToAttach(ParallelContext *pcxt)
{
	/*
	 * No-op: the pool threads are already running the entry and will attach to
	 * the phase barrier. The leader's launch path polls BarrierParticipants
	 * until the party is whole, which is the real attach wait.
	 */
	(void)pcxt;
}

void
WaitForParallelWorkersToFinish(ParallelContext *pcxt)
{
	vs_thread_pool_join(pcxt->pool);
}

void
DestroyParallelContext(ParallelContext *pcxt)
{
	/* Workers have been joined by now, so their latches are safe to free. */
	vs_thread_pool_destroy(pcxt->pool);
	free(pcxt->worker_latches);
	free(pcxt->arena);
	vs_shm_toc_free(pcxt->toc);
	free(pcxt);
}

#endif /* VS_STANDALONE */
