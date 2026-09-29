/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * thread_pool.h - Reusable thread pool for parallel build phases
 *
 * Creates N background worker threads that persist for the pool's
 * lifetime. All parallelism uses barrier-based iteration: workers
 * synchronize via pthread_barrier between iterations, with the
 * leader calling a reduce function to merge results and check
 * convergence.
 *
 * parallel_for is a convenience wrapper for single-iteration use.
 *
 * Standalone only (pthreads). PG parallel builds use PG's own
 * parallel worker infrastructure.
 */

#ifndef VS_THREAD_POOL_H
#define VS_THREAD_POOL_H

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>

/* macOS lacks pthread_barrier_t (optional POSIX extension). */
#ifdef __APPLE__

#define PTHREAD_BARRIER_SERIAL_THREAD (-1)

typedef struct
{
	pthread_mutex_t mutex;
	pthread_cond_t	cv;
	uint32_t		count;
	uint32_t		waiting;
	uint32_t		generation;
} pthread_barrier_t;

typedef void pthread_barrierattr_t;

static inline int
pthread_barrier_init(
		pthread_barrier_t			*b,
		const pthread_barrierattr_t *attr,
		unsigned					 count)
{
	(void)attr;
	pthread_mutex_init(&b->mutex, NULL);
	pthread_cond_init(&b->cv, NULL);
	b->count	  = count;
	b->waiting	  = 0;
	b->generation = 0;
	return 0;
}

static inline int
pthread_barrier_wait(pthread_barrier_t *b)
{
	pthread_mutex_lock(&b->mutex);
	uint32_t gen = b->generation;
	b->waiting++;

	if (b->waiting == b->count)
	{
		b->waiting = 0;
		b->generation++;
		pthread_cond_broadcast(&b->cv);
		pthread_mutex_unlock(&b->mutex);
		return PTHREAD_BARRIER_SERIAL_THREAD;
	}

	while (gen == b->generation)
		pthread_cond_wait(&b->cv, &b->mutex);
	pthread_mutex_unlock(&b->mutex);
	return 0;
}

static inline int
pthread_barrier_destroy(pthread_barrier_t *b)
{
	pthread_mutex_destroy(&b->mutex);
	pthread_cond_destroy(&b->cv);
	return 0;
}

#endif /* __APPLE__ */

typedef void (*VsParallelForFn)(
		uint32_t thread_id, uint32_t start, uint32_t end, void *arg);

typedef bool (*VsReduceFn)(void *arg, uint32_t iteration);

typedef struct VsThreadPool VsThreadPool;

/*
 * Create a pool with nthreads background worker threads.
 * Thread IDs are 0..nthreads-1; the leader gets ID nthreads.
 */
VsThreadPool *vs_thread_pool_create(uint32_t nthreads);

/*
 * Iterative parallel-for with barrier synchronization.
 *
 * Each iteration: all threads (N workers + leader) execute work_fn
 * on their chunk, then synchronize at a barrier. The leader calls
 * reduce_fn to merge results; if it returns true, the next iteration
 * starts, otherwise the loop ends.
 *
 * Workers stay alive across iterations — synchronization is a
 * lightweight barrier, not dispatch/wake per round.
 *
 * If reduce_fn is NULL, runs a single iteration regardless of
 * max_iterations (used by parallel_for).
 *
 * The leader (calling thread) participates as thread nthreads.
 * With nthreads=0, runs single-threaded.
 *
 * Maps to PG's Barrier (BarrierArriveAndWait) for PG builds.
 */
void vs_thread_pool_iterate(
		VsThreadPool   *pool,
		uint32_t		total,
		VsParallelForFn work_fn,
		VsReduceFn		reduce_fn,
		void		   *arg,
		uint32_t		max_iterations);

/*
 * Single-iteration parallel-for. Equivalent to iterate with
 * reduce_fn=NULL and max_iterations=1.
 */
void vs_thread_pool_parallel_for(
		VsThreadPool *pool, uint32_t total, VsParallelForFn fn, void *arg);

uint32_t vs_thread_pool_nthreads(const VsThreadPool *pool);

void vs_thread_pool_destroy(VsThreadPool *pool);

/*
 * SPMD dispatch over the persistent pool.
 *
 * fn runs once per worker as fn(participant_id, arg) with participant_id in
 * 1..nthreads (the leader is participant 0). Unlike parallel_for there is no
 * chunking and no per-iteration barrier: each worker runs the whole function
 * and is expected to self-synchronize internally (e.g. an SPMD index-build
 * worker coordinating its phases through a dynamic Barrier). This is how the
 * standalone back-end drives the shared parallel build on the pool, mirroring
 * how PostgreSQL launches parallel workers on a single registered entry.
 *
 * launch/join are the asynchronous pair the index build needs: launch wakes
 * the workers and returns immediately, so the leader can run its own code
 * (participate in k-means, drain the workers' streamed output) concurrently;
 * join then rendezvous with the workers once the leader is done. Between them
 * the leader must NOT touch the pool. With nthreads=0 they are no-ops (there
 * are no workers; the caller does all the work itself).
 *
 * run_spmd is the synchronous convenience: launch, run fn as participant 0 on
 * the calling thread, then join — i.e. every participant runs fn. With
 * nthreads=0 it runs fn(0, arg) inline.
 */
typedef void (*VsSpmdFn)(uint32_t participant_id, void *arg);
void vs_thread_pool_launch(VsThreadPool *pool, VsSpmdFn fn, void *arg);
void vs_thread_pool_join(VsThreadPool *pool);
void vs_thread_pool_run_spmd(VsThreadPool *pool, VsSpmdFn fn, void *arg);

#endif /* VS_THREAD_POOL_H */
