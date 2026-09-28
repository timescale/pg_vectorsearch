/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * vs_parallel_ctx.h - Parallel context lifecycle
 *
 * The build's leader sets up a parallel context, launches workers, waits for
 * them, and tears it down. In a PostgreSQL build this is PostgreSQL's
 * ParallelContext (access/parallel.h), backed by background workers and a DSM
 * segment. In a standalone (thread-based) build we provide the same lifecycle
 * over a persistent thread pool: InitializeParallelDSM lays a shm_toc over a
 * heap arena, and LaunchParallelWorkers dispatches the registered worker entry
 * onto the pool's worker threads (asynchronously, so the leader can drain),
 * mirroring how PostgreSQL launches background workers on a registered entry.
 *
 * Workers are resolved by name the way PostgreSQL resolves a background-worker
 * entry point: the worker module registers its function with
 * vs_parallel_register_worker, and CreateParallelContext records the name.
 *
 * Each launched thread gets a stable latch (owned by the context, freed only
 * at DestroyParallelContext after every worker has been joined) so the leader
 * can safely set it while a worker may still be blocked on it.
 */

#ifndef VS_PARALLEL_CTX_H
#define VS_PARALLEL_CTX_H

#include <stddef.h>

#include "standalone/latch.h"
#include "standalone/pg_compat.h" /* dsm_segment */
#include "standalone/shm_toc.h"
#include "standalone/thread_pool.h"

/*
 * The parallel worker index of the running thread: -1 in the leader, 0..N-1 in
 * the workers (PostgreSQL's ParallelWorkerNumber). The build derives worker_id
 * = ParallelWorkerNumber + 1 from it, so the leader is worker_id 0.
 */
extern __thread int ParallelWorkerNumber;

typedef void (*VsParallelWorkerFn)(dsm_segment *seg, shm_toc *toc);

typedef struct ParallelContext
{
	int				  nworkers;			 /* requested */
	int				  nworkers_launched; /* actually started */
	shm_toc			 *toc;
	dsm_segment		 *seg; /* non-NULL sentinel in standalone */
	shm_toc_estimator estimator;

	/* standalone internals */
	const char		  *function_name;
	VsParallelWorkerFn worker_fn; /* resolved at launch */
	void			  *arena;
	size_t			   arena_size;
	VsThreadPool	  *pool; /* persistent worker threads */
	Latch *worker_latches; /* stable [nworkers]; bound as each thread's MyLatch
							*/
	Latch leader_latch;	   /* stable; the leader's MyLatch */
} ParallelContext;

/* Register a worker entry under a name, resolved later by
 * CreateParallelContext. */
extern void
vs_parallel_register_worker(const char *name, VsParallelWorkerFn fn);

extern void EnterParallelMode(void);
extern void ExitParallelMode(void);

extern ParallelContext *
CreateParallelContext(const char *library, const char *function, int nworkers);
extern void InitializeParallelDSM(ParallelContext *pcxt);
extern void LaunchParallelWorkers(ParallelContext *pcxt);
extern void WaitForParallelWorkersToAttach(ParallelContext *pcxt);
extern void WaitForParallelWorkersToFinish(ParallelContext *pcxt);
extern void DestroyParallelContext(ParallelContext *pcxt);

#endif /* VS_PARALLEL_CTX_H */
