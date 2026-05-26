/*
 * thread_pool.c - Reusable thread pool with barrier-based iteration
 *
 * All parallelism funnels through iterate: workers wake from a
 * condvar, enter a barrier-synchronized loop (work → barrier →
 * leader reduce → barrier → repeat), then return to idle.
 * parallel_for is iterate with one iteration and no reduce.
 */

#include <stdlib.h>

#include "standalone/thread_pool.h"

typedef struct MktWorker
{
	pthread_t thread;
	uint32_t  id;
	uint32_t  start;
	uint32_t  end;
} MktWorker;

struct MktThreadPool
{
	MktWorker *workers;
	uint32_t   nthreads;

	/* Current iterate parameters (set by leader before waking) */
	MktParallelForFn work_fn;
	void			*arg;
	uint32_t		 max_iterations;
	volatile bool	 keep_going;

	/* Inter-iteration barrier (nthreads + 1 participants) */
	pthread_barrier_t barrier;

	/* Idle synchronization */
	pthread_mutex_t mutex;
	pthread_cond_t	wake_cv;
	uint32_t		generation;
	int				shutdown;
};

typedef struct WorkerArg
{
	MktThreadPool *pool;
	uint32_t	   id;
} WorkerArg;

static void *
pool_worker_fn(void *raw)
{
	WorkerArg	  *wa	= (WorkerArg *)raw;
	MktThreadPool *pool = wa->pool;
	uint32_t	   id	= wa->id;
	free(wa);

	uint32_t my_gen = 0;

	for (;;)
	{
		pthread_mutex_lock(&pool->mutex);
		while (pool->generation == my_gen && !pool->shutdown)
			pthread_cond_wait(&pool->wake_cv, &pool->mutex);

		if (pool->shutdown)
		{
			pthread_mutex_unlock(&pool->mutex);
			break;
		}

		my_gen = pool->generation;
		pthread_mutex_unlock(&pool->mutex);

		/* Iterate loop with barrier synchronization */
		for (uint32_t iter = 0; iter < pool->max_iterations; iter++)
		{
			if (pool->workers[id].start < pool->workers[id].end)
				pool->work_fn(
						id,
						pool->workers[id].start,
						pool->workers[id].end,
						pool->arg);

			pthread_barrier_wait(&pool->barrier);

			/* Leader does reduce between these two barriers */
			pthread_barrier_wait(&pool->barrier);

			if (!pool->keep_going)
				break;
		}
	}

	return NULL;
}

MktThreadPool *
mkt_thread_pool_create(uint32_t nthreads)
{
	MktThreadPool *pool = calloc(1, sizeof(MktThreadPool));
	pool->nthreads		= nthreads;
	pool->workers		= calloc(nthreads, sizeof(MktWorker));

	pthread_mutex_init(&pool->mutex, NULL);
	pthread_cond_init(&pool->wake_cv, NULL);

	if (nthreads > 0)
		pthread_barrier_init(&pool->barrier, NULL, nthreads + 1);

	for (uint32_t i = 0; i < nthreads; i++)
	{
		pool->workers[i].id = i;
		WorkerArg *wa		= malloc(sizeof(WorkerArg));
		wa->pool			= pool;
		wa->id				= i;
		pthread_create(&pool->workers[i].thread, NULL, pool_worker_fn, wa);
	}

	return pool;
}

void
mkt_thread_pool_iterate(
		MktThreadPool	*pool,
		uint32_t		 total,
		MktParallelForFn work_fn,
		MktReduceFn		 reduce_fn,
		void			*arg,
		uint32_t		 max_iterations)
{
	if (total == 0 || max_iterations == 0)
		return;

	uint32_t nt = pool->nthreads;

	if (nt == 0)
	{
		for (uint32_t iter = 0; iter < max_iterations; iter++)
		{
			work_fn(0, 0, total, arg);
			if (!reduce_fn || !reduce_fn(arg, iter))
				break;
		}
		return;
	}

	/* Partition across N+1 (workers + leader) */
	uint32_t all	   = nt + 1;
	uint32_t per	   = total / all;
	uint32_t remainder = total % all;
	uint32_t off	   = 0;

	for (uint32_t i = 0; i < nt; i++)
	{
		uint32_t chunk		   = per + (i < remainder ? 1 : 0);
		pool->workers[i].start = off;
		pool->workers[i].end   = off + chunk;
		off += chunk;
	}

	uint32_t leader_start = off;
	uint32_t leader_end	  = total;
	uint32_t leader_id	  = nt;

	pool->work_fn		 = work_fn;
	pool->arg			 = arg;
	pool->max_iterations = max_iterations;
	pool->keep_going	 = true;

	/* Wake workers */
	pthread_mutex_lock(&pool->mutex);
	pool->generation++;
	pthread_cond_broadcast(&pool->wake_cv);
	pthread_mutex_unlock(&pool->mutex);

	/* Leader participates in iterate loop */
	for (uint32_t iter = 0; iter < max_iterations; iter++)
	{
		if (leader_start < leader_end)
			work_fn(leader_id, leader_start, leader_end, arg);

		/* Wait for all workers to finish this iteration */
		pthread_barrier_wait(&pool->barrier);

		/* Leader does reduction */
		pool->keep_going = reduce_fn ? reduce_fn(arg, iter) : false;

		/* Release workers for next iteration (or exit) */
		pthread_barrier_wait(&pool->barrier);

		if (!pool->keep_going)
			break;
	}
}

void
mkt_thread_pool_parallel_for(
		MktThreadPool *pool, uint32_t total, MktParallelForFn fn, void *arg)
{
	mkt_thread_pool_iterate(pool, total, fn, NULL, arg, 1);
}

uint32_t
mkt_thread_pool_nthreads(const MktThreadPool *pool)
{
	return pool->nthreads;
}

void
mkt_thread_pool_destroy(MktThreadPool *pool)
{
	if (pool == NULL)
		return;

	pthread_mutex_lock(&pool->mutex);
	pool->shutdown = 1;
	pthread_cond_broadcast(&pool->wake_cv);
	pthread_mutex_unlock(&pool->mutex);

	for (uint32_t i = 0; i < pool->nthreads; i++)
		pthread_join(pool->workers[i].thread, NULL);

	if (pool->nthreads > 0)
		pthread_barrier_destroy(&pool->barrier);
	pthread_mutex_destroy(&pool->mutex);
	pthread_cond_destroy(&pool->wake_cv);
	free(pool->workers);
	free(pool);
}
