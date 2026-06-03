/*
 * parallel_ctx.c - Parallel context lifecycle over pthreads
 *
 * Standalone implementation of the ParallelContext lifecycle the build uses
 * (see mkt_parallel_ctx.h). PG builds use PostgreSQL's ParallelContext, so
 * this file is compiled only for standalone.
 */

#ifdef MKT_STANDALONE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/mkt_parallel_ctx.h"

#define MKT_PARALLEL_TOC_MAGIC		 UINT64_C(0x4d4b54504152) /* "MKTPAR" */
#define MKT_PARALLEL_MAX_WORKERS_REG 8

__thread int ParallelWorkerNumber = -1;

/* Name -> worker-entry registry (populated once at module init). */
static struct
{
	const char		   *name;
	MktParallelWorkerFn fn;
} mkt_worker_registry[MKT_PARALLEL_MAX_WORKERS_REG];

static int mkt_worker_registry_count = 0;

void
mkt_parallel_register_worker(const char *name, MktParallelWorkerFn fn)
{
	if (mkt_worker_registry_count == MKT_PARALLEL_MAX_WORKERS_REG)
	{
		fprintf(stderr, "mkt_parallel_register_worker: registry full\n");
		abort();
	}
	mkt_worker_registry[mkt_worker_registry_count].name = name;
	mkt_worker_registry[mkt_worker_registry_count].fn	= fn;
	mkt_worker_registry_count++;
}

static MktParallelWorkerFn
mkt_worker_lookup(const char *name)
{
	for (int i = 0; i < mkt_worker_registry_count; i++)
		if (strcmp(mkt_worker_registry[i].name, name) == 0)
			return mkt_worker_registry[i].fn;

	fprintf(stderr, "mkt_worker_lookup: '%s' not registered\n", name);
	abort();
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
		fprintf(stderr, "CreateParallelContext: out of memory\n");
		abort();
	}

	pcxt->nworkers			= nworkers;
	pcxt->nworkers_launched = 0;
	pcxt->function_name		= function;
	shm_toc_initialize_estimator(&pcxt->estimator);

	/*
	 * Bind the leader's latch now and keep it in the context: workers set it
	 * (via the page queues' receiver) while the leader drains, so it must
	 * outlive them — the context is freed only after they are all joined.
	 */
	InitLatch(&pcxt->leader_latch);
	mkt_latch_attach_self(&pcxt->leader_latch);
	return pcxt;
}

void
InitializeParallelDSM(ParallelContext *pcxt)
{
	pcxt->arena_size = shm_toc_estimate(&pcxt->estimator);
	pcxt->arena		 = malloc(pcxt->arena_size);
	if (pcxt->arena == NULL)
	{
		fprintf(stderr, "InitializeParallelDSM: out of memory\n");
		abort();
	}
	pcxt->toc = shm_toc_create(
			MKT_PARALLEL_TOC_MAGIC, pcxt->arena, pcxt->arena_size);

	/*
	 * A non-NULL sentinel: PG sets seg to the DSM segment and the driver
	 * treats NULL as "could not start". Standalone always starts, so seg just
	 * has to be non-NULL (the shm_mq shim ignores it).
	 */
	pcxt->seg = (dsm_segment *)pcxt;
}

/* Trampoline: bind this thread's worker number + latch, then run the entry. */
typedef struct
{
	MktParallelWorkerFn fn;
	int					worker_number;
	Latch			   *latch;
	dsm_segment		   *seg;
	shm_toc			   *toc;
} MktWorkerThreadArg;

static void *
mkt_worker_trampoline(void *arg)
{
	MktWorkerThreadArg *w = (MktWorkerThreadArg *)arg;

	ParallelWorkerNumber = w->worker_number;
	mkt_latch_attach_self(w->latch);
	w->fn(w->seg, w->toc);
	free(w);
	return NULL;
}

void
LaunchParallelWorkers(ParallelContext *pcxt)
{
	MktParallelWorkerFn fn = mkt_worker_lookup(pcxt->function_name);

	if (pcxt->nworkers == 0)
	{
		pcxt->nworkers_launched = 0;
		return;
	}

	pcxt->worker_latches = calloc(pcxt->nworkers, sizeof(Latch));
	pcxt->threads		 = calloc(pcxt->nworkers, sizeof(pthread_t));
	if (pcxt->worker_latches == NULL || pcxt->threads == NULL)
	{
		fprintf(stderr, "LaunchParallelWorkers: out of memory\n");
		abort();
	}

	for (int i = 0; i < pcxt->nworkers; i++)
	{
		MktWorkerThreadArg *w = malloc(sizeof(MktWorkerThreadArg));

		InitLatch(&pcxt->worker_latches[i]);
		w->fn			 = fn;
		w->worker_number = i;
		w->latch		 = &pcxt->worker_latches[i];
		w->seg			 = pcxt->seg;
		w->toc			 = pcxt->toc;

		if (pthread_create(
					&pcxt->threads[i], NULL, mkt_worker_trampoline, w) != 0)
		{
			fprintf(stderr, "LaunchParallelWorkers: pthread_create failed\n");
			abort();
		}
	}
	pcxt->nworkers_launched = pcxt->nworkers;
}

void
WaitForParallelWorkersToAttach(ParallelContext *pcxt)
{
	/*
	 * No-op: the threads are already created and will attach to the phase
	 * barrier. The leader's launch path polls BarrierParticipants until the
	 * party is whole, which is the real attach wait.
	 */
	(void)pcxt;
}

void
WaitForParallelWorkersToFinish(ParallelContext *pcxt)
{
	for (int i = 0; i < pcxt->nworkers_launched; i++)
		pthread_join(pcxt->threads[i], NULL);
}

void
DestroyParallelContext(ParallelContext *pcxt)
{
	/* Workers have been joined by now, so their latches are safe to free. */
	free(pcxt->threads);
	free(pcxt->worker_latches);
	free(pcxt->arena);
	mkt_shm_toc_free(pcxt->toc);
	free(pcxt);
}

#endif /* MKT_STANDALONE */
