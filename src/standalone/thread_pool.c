/*
 * thread_pool.c - Reusable thread pool for parallel build phases
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

	/* Current round's work */
	MktParallelForFn fn;
	void			*arg;

	/* Synchronization: workers wait on work_cv for a new round,
	 * main thread waits on done_cv for all workers to finish. */
	pthread_mutex_t mutex;
	pthread_cond_t	work_cv;
	pthread_cond_t	done_cv;
	uint32_t		generation;
	uint32_t		workers_done;
	uint32_t		dispatched; /* workers woken this round */
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
			pthread_cond_wait(&pool->work_cv, &pool->mutex);

		if (pool->shutdown)
		{
			pthread_mutex_unlock(&pool->mutex);
			break;
		}

		my_gen				   = pool->generation;
		uint32_t		 start = pool->workers[id].start;
		uint32_t		 end   = pool->workers[id].end;
		MktParallelForFn fn	   = pool->fn;
		void			*arg   = pool->arg;
		pthread_mutex_unlock(&pool->mutex);

		if (start < end)
			fn(id, start, end, arg);

		pthread_mutex_lock(&pool->mutex);
		pool->workers_done++;
		if (pool->workers_done == pool->dispatched)
			pthread_cond_signal(&pool->done_cv);
		pthread_mutex_unlock(&pool->mutex);
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
	pthread_cond_init(&pool->work_cv, NULL);
	pthread_cond_init(&pool->done_cv, NULL);

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

static void
partition(MktWorker *workers, uint32_t nworkers, uint32_t total)
{
	uint32_t per	   = total / nworkers;
	uint32_t remainder = total % nworkers;
	uint32_t off	   = 0;

	for (uint32_t i = 0; i < nworkers; i++)
	{
		uint32_t chunk	 = per + (i < remainder ? 1 : 0);
		workers[i].start = off;
		workers[i].end	 = off + chunk;
		off += chunk;
	}
}

static void
wake_workers(
		MktThreadPool *pool, uint32_t nworkers, MktParallelForFn fn, void *arg)
{
	pthread_mutex_lock(&pool->mutex);
	pool->fn		   = fn;
	pool->arg		   = arg;
	pool->workers_done = 0;
	pool->dispatched   = nworkers;
	pool->generation++;
	pthread_cond_broadcast(&pool->work_cv);
	pthread_mutex_unlock(&pool->mutex);
}

void
mkt_thread_pool_dispatch(
		MktThreadPool *pool, uint32_t total, MktParallelForFn fn, void *arg)
{
	if (pool->nthreads == 0 || total == 0)
	{
		pool->dispatched = 0;
		return;
	}

	partition(pool->workers, pool->nthreads, total);
	wake_workers(pool, pool->nthreads, fn, arg);
}

void
mkt_thread_pool_wait(MktThreadPool *pool)
{
	if (pool->nthreads == 0)
		return;

	pthread_mutex_lock(&pool->mutex);
	while (pool->workers_done < pool->dispatched)
		pthread_cond_wait(&pool->done_cv, &pool->mutex);
	pthread_mutex_unlock(&pool->mutex);
}

void
mkt_thread_pool_parallel_for(
		MktThreadPool *pool, uint32_t total, MktParallelForFn fn, void *arg)
{
	if (total == 0)
		return;

	uint32_t nt = pool->nthreads;

	if (nt == 0)
	{
		fn(0, 0, total, arg);
		return;
	}

	/* Partition across N+1: workers get chunks 0..N-1,
	 * caller gets chunk N (the last one). */
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

	uint32_t caller_start = off;
	uint32_t caller_end	  = total;

	wake_workers(pool, nt, fn, arg);

	/* Caller runs the last chunk */
	if (caller_start < caller_end)
		fn(nt, caller_start, caller_end, arg);

	mkt_thread_pool_wait(pool);
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
	pthread_cond_broadcast(&pool->work_cv);
	pthread_mutex_unlock(&pool->mutex);

	for (uint32_t i = 0; i < pool->nthreads; i++)
		pthread_join(pool->workers[i].thread, NULL);

	pthread_mutex_destroy(&pool->mutex);
	pthread_cond_destroy(&pool->work_cv);
	pthread_cond_destroy(&pool->done_cv);
	free(pool->workers);
	free(pool);
}
