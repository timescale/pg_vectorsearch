/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * test_latch.c - Wait/wake latch (standalone shim) tests
 *
 * Covers the latch behavior the posting drain loop relies on: a set latch
 * satisfies a wait immediately, a cross-thread SetLatch wakes a blocked
 * waiter, WaitLatch honors a timeout on an unset latch, and ResetLatch clears
 * the set state.
 */

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>

#include "standalone/latch.h"
#include "vs_test.h"

TEST_GROUP(Latch);

TEST(set_then_wait_returns_immediately)
{
	Latch latch;

	InitLatch(&latch);
	SetLatch(&latch);

	int ev = WaitLatch(&latch, WL_LATCH_SET, -1, 0);
	ASSERT_TRUE((ev & WL_LATCH_SET) != 0, "already-set latch wakes at once");
}

TEST(timeout_on_unset_latch)
{
	Latch latch;

	InitLatch(&latch);

	int ev = WaitLatch(&latch, WL_LATCH_SET | WL_TIMEOUT, 20, 0);
	ASSERT_TRUE((ev & WL_TIMEOUT) != 0, "unset latch times out");
	ASSERT_TRUE((ev & WL_LATCH_SET) == 0, "no spurious set reported");
}

TEST(reset_clears_set_state)
{
	Latch latch;

	InitLatch(&latch);
	SetLatch(&latch);
	ResetLatch(&latch);

	int ev = WaitLatch(&latch, WL_LATCH_SET | WL_TIMEOUT, 20, 0);
	ASSERT_TRUE((ev & WL_TIMEOUT) != 0, "reset latch is no longer set");
}

typedef struct
{
	Latch		 *latch;
	_Atomic(int) *woke;
} WakeArg;

static void *
waiter_thread(void *arg)
{
	WakeArg *a = (WakeArg *)arg;

	/* Bind MyLatch the way a participating thread would, then block on it. */
	vs_latch_attach_self(a->latch);
	WaitLatch(MyLatch, WL_LATCH_SET, -1, 0);
	atomic_store(a->woke, 1);
	return NULL;
}

TEST(cross_thread_setlatch_wakes_waiter)
{
	Latch		 latch;
	_Atomic(int) woke = 0;
	pthread_t	 th;

	InitLatch(&latch);

	WakeArg arg = {.latch = &latch, .woke = &woke};
	pthread_create(&th, NULL, waiter_thread, &arg);

	/* The waiter blocks until this set; if SetLatch failed to wake it, the
	 * join would hang and the suite would time out (a loud failure). */
	SetLatch(&latch);
	pthread_join(th, NULL);

	ASSERT_EQ(1, atomic_load(&woke), "SetLatch woke the blocked waiter");
}
