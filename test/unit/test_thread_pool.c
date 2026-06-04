/*
 * test_thread_pool.c - Thread pool tests
 */

#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

#include "mkt_test.h"
#include "standalone/thread_pool.h"

TEST_GROUP(ThreadPool);

/* ----------------------------------------------------------------
 * Helpers
 * ---------------------------------------------------------------- */

static void
sum_fn(uint32_t thread_id, uint32_t start, uint32_t end, void *arg)
{
	(void)thread_id;
	_Atomic(uint64_t) *total = (_Atomic(uint64_t) *)arg;
	uint64_t		   local = 0;
	for (uint32_t i = start; i < end; i++)
		local += i;
	atomic_fetch_add(total, local);
}

typedef struct
{
	uint32_t *counts;
	uint32_t  nslots;
} CoverageArg;

static void
coverage_fn(uint32_t thread_id, uint32_t start, uint32_t end, void *arg)
{
	(void)thread_id;
	CoverageArg *ca = (CoverageArg *)arg;
	for (uint32_t i = start; i < end; i++)
		ca->counts[i]++;
}

typedef struct
{
	_Atomic(uint32_t) max_thread_id;
	_Atomic(uint32_t) call_count;
} ThreadIdArg;

static void
thread_id_fn(uint32_t thread_id, uint32_t start, uint32_t end, void *arg)
{
	(void)start;
	(void)end;
	ThreadIdArg *ta = (ThreadIdArg *)arg;
	atomic_fetch_add(&ta->call_count, 1);
	uint32_t cur = atomic_load(&ta->max_thread_id);
	while (thread_id > cur)
	{
		if (atomic_compare_exchange_weak(&ta->max_thread_id, &cur, thread_id))
			break;
	}
}

/* ----------------------------------------------------------------
 * Create / destroy
 * ---------------------------------------------------------------- */

TEST(create_destroy)
{
	MktThreadPool *pool = mkt_thread_pool_create(4);
	ASSERT_NOT_NULL(pool, "pool creation should succeed");
	ASSERT_EQ(4, mkt_thread_pool_nthreads(pool), "should have 4 threads");
	mkt_thread_pool_destroy(pool);
}

TEST(create_zero_workers)
{
	MktThreadPool *pool = mkt_thread_pool_create(0);
	ASSERT_NOT_NULL(pool, "pool with 0 workers should succeed");
	ASSERT_EQ(0, mkt_thread_pool_nthreads(pool), "should have 0 threads");
	mkt_thread_pool_destroy(pool);
}

TEST(destroy_null)
{
	mkt_thread_pool_destroy(NULL);
	ASSERT_TRUE(true, "destroy(NULL) should not crash");
}

/* ----------------------------------------------------------------
 * parallel_for
 * ---------------------------------------------------------------- */

TEST(parallel_for_zero_workers)
{
	MktThreadPool *pool = mkt_thread_pool_create(0);

	_Atomic(uint64_t) total = 0;
	mkt_thread_pool_parallel_for(pool, 100, sum_fn, &total);

	uint64_t expected = 0;
	for (uint32_t i = 0; i < 100; i++)
		expected += i;
	ASSERT_EQ(expected, atomic_load(&total), "zero-worker parallel_for");

	mkt_thread_pool_destroy(pool);
}

TEST(parallel_for_one_worker)
{
	MktThreadPool *pool = mkt_thread_pool_create(1);

	_Atomic(uint64_t) total = 0;
	mkt_thread_pool_parallel_for(pool, 100, sum_fn, &total);

	uint64_t expected = 0;
	for (uint32_t i = 0; i < 100; i++)
		expected += i;
	ASSERT_EQ(expected, atomic_load(&total), "one-worker parallel_for");

	mkt_thread_pool_destroy(pool);
}

TEST(parallel_for_multi)
{
	MktThreadPool *pool = mkt_thread_pool_create(7);

	_Atomic(uint64_t) total = 0;
	mkt_thread_pool_parallel_for(pool, 10000, sum_fn, &total);

	uint64_t expected = 0;
	for (uint32_t i = 0; i < 10000; i++)
		expected += i;
	ASSERT_EQ(expected, atomic_load(&total), "multi-worker parallel_for");

	mkt_thread_pool_destroy(pool);
}

TEST(parallel_for_full_coverage)
{
	MktThreadPool *pool = mkt_thread_pool_create(7);

	uint32_t	n	   = 1000;
	uint32_t   *counts = calloc(n, sizeof(uint32_t));
	CoverageArg ca	   = {.counts = counts, .nslots = n};

	mkt_thread_pool_parallel_for(pool, n, coverage_fn, &ca);

	for (uint32_t i = 0; i < n; i++)
		ASSERT_EQ(1, counts[i], "each element visited exactly once");

	free(counts);
	mkt_thread_pool_destroy(pool);
}

TEST(parallel_for_zero_total)
{
	MktThreadPool	 *pool	= mkt_thread_pool_create(4);
	_Atomic(uint64_t) total = 0;
	mkt_thread_pool_parallel_for(pool, 0, sum_fn, &total);
	ASSERT_EQ(0, atomic_load(&total), "zero work should produce zero");
	mkt_thread_pool_destroy(pool);
}

TEST(parallel_for_fewer_items_than_threads)
{
	MktThreadPool *pool = mkt_thread_pool_create(16);

	uint32_t	n	   = 3;
	uint32_t   *counts = calloc(n, sizeof(uint32_t));
	CoverageArg ca	   = {.counts = counts, .nslots = n};

	mkt_thread_pool_parallel_for(pool, n, coverage_fn, &ca);

	for (uint32_t i = 0; i < n; i++)
		ASSERT_EQ(1, counts[i], "each element visited exactly once");

	free(counts);
	mkt_thread_pool_destroy(pool);
}

TEST(parallel_for_one_item)
{
	MktThreadPool *pool = mkt_thread_pool_create(4);

	uint32_t	count = 0;
	CoverageArg ca	  = {.counts = &count, .nslots = 1};

	mkt_thread_pool_parallel_for(pool, 1, coverage_fn, &ca);
	ASSERT_EQ(1, count, "single item should be visited once");

	mkt_thread_pool_destroy(pool);
}

TEST(parallel_for_multiple_rounds)
{
	MktThreadPool *pool = mkt_thread_pool_create(4);

	for (int round = 0; round < 10; round++)
	{
		_Atomic(uint64_t) total = 0;
		mkt_thread_pool_parallel_for(pool, 1000, sum_fn, &total);

		uint64_t expected = 0;
		for (uint32_t i = 0; i < 1000; i++)
			expected += i;
		ASSERT_EQ(expected, atomic_load(&total), "round correctness");
	}

	mkt_thread_pool_destroy(pool);
}

TEST(parallel_for_thread_ids)
{
	MktThreadPool *pool = mkt_thread_pool_create(7);

	ThreadIdArg ta = {0};
	mkt_thread_pool_parallel_for(pool, 1000, thread_id_fn, &ta);

	/* 7 workers + 1 caller = 8 threads */
	ASSERT_EQ(8, atomic_load(&ta.call_count), "N+1 threads called");
	ASSERT_TRUE(
			atomic_load(&ta.max_thread_id) <= 7,
			"thread IDs should be in [0, nthreads]");

	mkt_thread_pool_destroy(pool);
}

TEST(parallel_for_large)
{
	MktThreadPool *pool = mkt_thread_pool_create(4);

	_Atomic(uint64_t) total = 0;
	uint32_t		  n		= 1000000;
	mkt_thread_pool_parallel_for(pool, n, sum_fn, &total);

	uint64_t expected = (uint64_t)(n - 1) * n / 2;
	ASSERT_EQ(expected, atomic_load(&total), "large workload sum");

	mkt_thread_pool_destroy(pool);
}

/* ----------------------------------------------------------------
 * iterate tests
 * ---------------------------------------------------------------- */

typedef struct
{
	_Atomic(uint64_t) total;
	uint32_t		  target_iters;
} IterSumArg;

static void
iter_sum_fn(uint32_t thread_id, uint32_t start, uint32_t end, void *arg)
{
	(void)thread_id;
	IterSumArg *ia	  = (IterSumArg *)arg;
	uint64_t	local = 0;
	for (uint32_t i = start; i < end; i++)
		local += i;
	atomic_fetch_add(&ia->total, local);
}

static bool
iter_reduce_fn(void *arg, uint32_t iteration)
{
	IterSumArg *ia = (IterSumArg *)arg;
	return iteration + 1 < ia->target_iters;
}

TEST(iterate_basic)
{
	MktThreadPool *pool = mkt_thread_pool_create(4);

	IterSumArg ia = {.total = 0, .target_iters = 5};
	mkt_thread_pool_iterate(pool, 1000, iter_sum_fn, iter_reduce_fn, &ia, 100);

	uint64_t one_iter = 0;
	for (uint32_t i = 0; i < 1000; i++)
		one_iter += i;
	ASSERT_EQ(one_iter * 5, atomic_load(&ia.total), "5 iterations sum");

	mkt_thread_pool_destroy(pool);
}

TEST(iterate_single_thread)
{
	MktThreadPool *pool = mkt_thread_pool_create(0);

	IterSumArg ia = {.total = 0, .target_iters = 3};
	mkt_thread_pool_iterate(pool, 100, iter_sum_fn, iter_reduce_fn, &ia, 100);

	uint64_t one_iter = 0;
	for (uint32_t i = 0; i < 100; i++)
		one_iter += i;
	ASSERT_EQ(one_iter * 3, atomic_load(&ia.total), "3 iterations serial");

	mkt_thread_pool_destroy(pool);
}

TEST(iterate_early_stop)
{
	MktThreadPool *pool = mkt_thread_pool_create(4);

	IterSumArg ia = {.total = 0, .target_iters = 2};
	mkt_thread_pool_iterate(pool, 1000, iter_sum_fn, iter_reduce_fn, &ia, 100);

	uint64_t one_iter = 0;
	for (uint32_t i = 0; i < 1000; i++)
		one_iter += i;
	ASSERT_EQ(one_iter * 2, atomic_load(&ia.total), "early stop at 2");

	mkt_thread_pool_destroy(pool);
}

TEST(iterate_max_iterations)
{
	MktThreadPool *pool = mkt_thread_pool_create(4);

	IterSumArg ia = {.total = 0, .target_iters = 999};
	mkt_thread_pool_iterate(pool, 100, iter_sum_fn, iter_reduce_fn, &ia, 5);

	uint64_t one_iter = 0;
	for (uint32_t i = 0; i < 100; i++)
		one_iter += i;
	ASSERT_EQ(one_iter * 5, atomic_load(&ia.total), "capped at max 5");

	mkt_thread_pool_destroy(pool);
}

TEST(iterate_zero_total)
{
	MktThreadPool *pool = mkt_thread_pool_create(4);
	IterSumArg	   ia	= {.total = 0, .target_iters = 5};
	mkt_thread_pool_iterate(pool, 0, iter_sum_fn, iter_reduce_fn, &ia, 10);
	ASSERT_EQ(0, atomic_load(&ia.total), "zero total no-op");
	mkt_thread_pool_destroy(pool);
}

/* ----------------------------------------------------------------
 * SPMD dispatch tests
 * ---------------------------------------------------------------- */

typedef struct
{
	_Atomic(uint32_t) call_count;
	_Atomic(uint32_t) id_seen[64]; /* id_seen[i] = times participant i ran */
} SpmdArg;

static void
spmd_fn(uint32_t participant_id, void *arg)
{
	SpmdArg *sa = (SpmdArg *)arg;
	atomic_fetch_add(&sa->call_count, 1);
	if (participant_id < 64)
		atomic_fetch_add(&sa->id_seen[participant_id], 1);
}

TEST(spmd_multi)
{
	MktThreadPool *pool = mkt_thread_pool_create(7);

	SpmdArg sa = {0};
	mkt_thread_pool_run_spmd(pool, spmd_fn, &sa);

	/* 7 workers + 1 leader = 8 participants, ids 0..7, once each. */
	ASSERT_EQ(8, atomic_load(&sa.call_count), "N+1 participants ran");
	for (uint32_t i = 0; i < 8; i++)
		ASSERT_EQ(
				1,
				atomic_load(&sa.id_seen[i]),
				"participant ran exactly once");
	for (uint32_t i = 8; i < 64; i++)
		ASSERT_EQ(
				0, atomic_load(&sa.id_seen[i]), "no out-of-range participant");

	mkt_thread_pool_destroy(pool);
}

TEST(spmd_zero_workers)
{
	MktThreadPool *pool = mkt_thread_pool_create(0);

	SpmdArg sa = {0};
	mkt_thread_pool_run_spmd(pool, spmd_fn, &sa);

	/* Serial: only the calling thread runs, as participant 0. */
	ASSERT_EQ(1, atomic_load(&sa.call_count), "sole participant ran");
	ASSERT_EQ(1, atomic_load(&sa.id_seen[0]), "participant 0 ran");

	mkt_thread_pool_destroy(pool);
}

TEST(spmd_one_worker)
{
	MktThreadPool *pool = mkt_thread_pool_create(1);

	SpmdArg sa = {0};
	mkt_thread_pool_run_spmd(pool, spmd_fn, &sa);

	ASSERT_EQ(2, atomic_load(&sa.call_count), "leader + 1 worker ran");
	ASSERT_EQ(1, atomic_load(&sa.id_seen[0]), "leader ran");
	ASSERT_EQ(1, atomic_load(&sa.id_seen[1]), "worker ran");

	mkt_thread_pool_destroy(pool);
}

TEST(spmd_multiple_rounds)
{
	MktThreadPool *pool = mkt_thread_pool_create(4);

	for (int round = 0; round < 10; round++)
	{
		SpmdArg sa = {0};
		mkt_thread_pool_run_spmd(pool, spmd_fn, &sa);
		ASSERT_EQ(5, atomic_load(&sa.call_count), "5 participants each round");
	}

	mkt_thread_pool_destroy(pool);
}

/* The pool must switch cleanly between SPMD and chunked dispatch. */
TEST(spmd_interleaved_with_parallel_for)
{
	MktThreadPool *pool = mkt_thread_pool_create(4);

	SpmdArg sa = {0};
	mkt_thread_pool_run_spmd(pool, spmd_fn, &sa);
	ASSERT_EQ(5, atomic_load(&sa.call_count), "spmd round ran");

	_Atomic(uint64_t) total = 0;
	mkt_thread_pool_parallel_for(pool, 1000, sum_fn, &total);
	uint64_t expected = 0;
	for (uint32_t i = 0; i < 1000; i++)
		expected += i;
	ASSERT_EQ(expected, atomic_load(&total), "parallel_for after spmd");

	SpmdArg sa2 = {0};
	mkt_thread_pool_run_spmd(pool, spmd_fn, &sa2);
	ASSERT_EQ(
			5, atomic_load(&sa2.call_count), "spmd round after parallel_for");

	mkt_thread_pool_destroy(pool);
}
