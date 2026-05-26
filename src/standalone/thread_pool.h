/*
 * thread_pool.h - Reusable thread pool for parallel build phases
 *
 * Creates N background worker threads that persist across multiple
 * dispatch/wait cycles, eliminating per-phase thread creation
 * overhead. Workers block on a condition variable between rounds.
 *
 * Standalone only (pthreads). PG parallel builds use PG's own
 * parallel worker infrastructure.
 */

#ifndef MKT_THREAD_POOL_H
#define MKT_THREAD_POOL_H

#include <pthread.h>
#include <stdint.h>

typedef void (*MktParallelForFn)(
		uint32_t thread_id, uint32_t start, uint32_t end, void *arg);

typedef struct MktThreadPool MktThreadPool;

/*
 * Create a pool with nthreads background worker threads.
 * Thread IDs are 0..nthreads-1.
 */
MktThreadPool *mkt_thread_pool_create(uint32_t nthreads);

/*
 * Dispatch work to the pool's background threads (non-blocking).
 *
 * Divides [0, total) into nthreads contiguous chunks and wakes
 * all workers. Returns immediately — the caller is free to do
 * other work while the pool executes.
 *
 * The same arg pointer is passed to every worker. Use it to pass
 * shared read-only state or an array of per-thread contexts
 * indexed by thread_id. Thread IDs are stable across rounds:
 * thread i always gets the i-th chunk.
 *
 * Must call mkt_thread_pool_wait() before the next dispatch.
 */
void mkt_thread_pool_dispatch(
		MktThreadPool *pool, uint32_t total, MktParallelForFn fn, void *arg);

/*
 * Block until all dispatched workers finish.
 */
void mkt_thread_pool_wait(MktThreadPool *pool);

/*
 * Convenience: dispatch + caller participates + wait.
 *
 * Splits [0, total) across nthreads + 1 chunks: the N pool workers
 * each get a chunk, and the calling thread runs one chunk itself.
 * Blocks until all work is complete. This gives N+1 way parallelism.
 *
 * With nthreads=0 (no pool workers), runs fn(0, 0, total, arg)
 * directly on the calling thread.
 */
void mkt_thread_pool_parallel_for(
		MktThreadPool *pool, uint32_t total, MktParallelForFn fn, void *arg);

uint32_t mkt_thread_pool_nthreads(const MktThreadPool *pool);

void mkt_thread_pool_destroy(MktThreadPool *pool);

#endif /* MKT_THREAD_POOL_H */
