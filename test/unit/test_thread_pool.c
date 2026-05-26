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
 * dispatch + wait
 * ---------------------------------------------------------------- */

TEST(dispatch_wait_basic)
{
	MktThreadPool *pool = mkt_thread_pool_create(4);

	_Atomic(uint64_t) total = 0;
	mkt_thread_pool_dispatch(pool, 10000, sum_fn, &total);
	mkt_thread_pool_wait(pool);

	uint64_t expected = 0;
	for (uint32_t i = 0; i < 10000; i++)
		expected += i;
	ASSERT_EQ(expected, atomic_load(&total), "dispatch+wait sum");

	mkt_thread_pool_destroy(pool);
}

TEST(dispatch_wait_coverage)
{
	MktThreadPool *pool = mkt_thread_pool_create(8);

	uint32_t	n	   = 1000;
	uint32_t   *counts = calloc(n, sizeof(uint32_t));
	CoverageArg ca	   = {.counts = counts, .nslots = n};

	mkt_thread_pool_dispatch(pool, n, coverage_fn, &ca);
	mkt_thread_pool_wait(pool);

	for (uint32_t i = 0; i < n; i++)
		ASSERT_EQ(1, counts[i], "each element visited exactly once");

	free(counts);
	mkt_thread_pool_destroy(pool);
}

TEST(dispatch_zero_workers)
{
	MktThreadPool	 *pool	= mkt_thread_pool_create(0);
	_Atomic(uint64_t) total = 0;
	mkt_thread_pool_dispatch(pool, 100, sum_fn, &total);
	mkt_thread_pool_wait(pool);
	ASSERT_EQ(0, atomic_load(&total), "dispatch with 0 workers is a no-op");
	mkt_thread_pool_destroy(pool);
}

TEST(dispatch_zero_total)
{
	MktThreadPool	 *pool	= mkt_thread_pool_create(4);
	_Atomic(uint64_t) total = 0;
	mkt_thread_pool_dispatch(pool, 0, sum_fn, &total);
	mkt_thread_pool_wait(pool);
	ASSERT_EQ(0, atomic_load(&total), "zero total is a no-op");
	mkt_thread_pool_destroy(pool);
}

TEST(dispatch_one_worker)
{
	MktThreadPool *pool = mkt_thread_pool_create(1);

	_Atomic(uint64_t) total = 0;
	mkt_thread_pool_dispatch(pool, 100, sum_fn, &total);
	/* Caller is free here — work runs on background thread */
	mkt_thread_pool_wait(pool);

	uint64_t expected = 0;
	for (uint32_t i = 0; i < 100; i++)
		expected += i;
	ASSERT_EQ(expected, atomic_load(&total), "single worker dispatch");

	mkt_thread_pool_destroy(pool);
}

TEST(dispatch_multiple_rounds)
{
	MktThreadPool *pool = mkt_thread_pool_create(4);

	for (int round = 0; round < 10; round++)
	{
		_Atomic(uint64_t) total = 0;
		mkt_thread_pool_dispatch(pool, 1000, sum_fn, &total);
		mkt_thread_pool_wait(pool);

		uint64_t expected = 0;
		for (uint32_t i = 0; i < 1000; i++)
			expected += i;
		ASSERT_EQ(expected, atomic_load(&total), "round correctness");
	}

	mkt_thread_pool_destroy(pool);
}

TEST(dispatch_thread_ids)
{
	MktThreadPool *pool = mkt_thread_pool_create(8);

	ThreadIdArg ta = {0};
	mkt_thread_pool_dispatch(pool, 1000, thread_id_fn, &ta);
	mkt_thread_pool_wait(pool);

	/* 8 workers only, caller does not participate */
	ASSERT_EQ(8, atomic_load(&ta.call_count), "8 workers called");
	ASSERT_TRUE(
			atomic_load(&ta.max_thread_id) < 8,
			"dispatch thread IDs should be in [0, nthreads)");

	mkt_thread_pool_destroy(pool);
}

TEST(dispatch_caller_does_work)
{
	MktThreadPool *pool = mkt_thread_pool_create(4);

	_Atomic(uint64_t) pool_total = 0;
	mkt_thread_pool_dispatch(pool, 1000, sum_fn, &pool_total);

	/* Caller does independent work while pool runs */
	uint64_t caller_sum = 0;
	for (uint32_t i = 1000; i < 2000; i++)
		caller_sum += i;

	mkt_thread_pool_wait(pool);

	uint64_t pool_expected = 0;
	for (uint32_t i = 0; i < 1000; i++)
		pool_expected += i;

	ASSERT_EQ(
			pool_expected,
			atomic_load(&pool_total),
			"pool computed correctly");

	uint64_t caller_expected = 0;
	for (uint32_t i = 1000; i < 2000; i++)
		caller_expected += i;
	ASSERT_EQ(caller_expected, caller_sum, "caller computed correctly");

	mkt_thread_pool_destroy(pool);
}
