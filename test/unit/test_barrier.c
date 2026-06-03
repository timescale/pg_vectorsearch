/*
 * test_barrier.c - Dynamic phase barrier (standalone shim) tests
 *
 * Exercises the pthread-backed Barrier that mirrors PostgreSQL's
 * storage/barrier.h API: dynamic attach/detach, per-phase election, and the
 * detach-completes-the-phase path the parallel build's launch wait relies on.
 */

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <unistd.h>

#include "core/mkt_barrier.h"
#include "mkt_test.h"

TEST_GROUP(Barrier);

/* ----------------------------------------------------------------
 * Single-participant semantics
 * ---------------------------------------------------------------- */

TEST(init_attach_phase)
{
	Barrier b;

	BarrierInit(&b, 0);
	ASSERT_EQ(0, BarrierParticipants(&b), "fresh barrier has no participants");
	ASSERT_EQ(0, BarrierPhase(&b), "fresh barrier starts at phase 0");

	ASSERT_EQ(0, BarrierAttach(&b), "attach returns the current phase");
	ASSERT_EQ(1, BarrierParticipants(&b), "attach grows the party");

	/* A lone participant completes the phase on its own arrival. */
	ASSERT_TRUE(BarrierArriveAndWait(&b, 0), "sole arriver is elected");
	ASSERT_EQ(1, BarrierPhase(&b), "completing a phase advances the counter");
}

/* ----------------------------------------------------------------
 * Dynamic attach + many synchronized rounds
 * ---------------------------------------------------------------- */

#define BR_WORKERS 4
#define BR_ROUNDS  100

typedef struct
{
	Barrier		 *b;
	int			  rounds;
	int			  party;
	_Atomic(int) *counter;
	_Atomic(int) *errors;
	_Atomic(int) *elected;
} BRoundArg;

/*
 * One round: bump a shared counter, sync, verify every participant has bumped
 * it (so the barrier really blocked until all arrived), then a second sync
 * gates the next round's bump so the read window above is race-free.
 */
static void
br_round(BRoundArg *a, int r)
{
	atomic_fetch_add(a->counter, 1);
	if (BarrierArriveAndWait(a->b, 0))
		atomic_fetch_add(a->elected, 1);

	if (atomic_load(a->counter) != (r + 1) * a->party)
		atomic_fetch_add(a->errors, 1);

	if (BarrierArriveAndWait(a->b, 0))
		atomic_fetch_add(a->elected, 1);
}

static void *
br_worker(void *arg)
{
	BRoundArg *a = (BRoundArg *)arg;

	BarrierAttach(a->b);
	for (int r = 0; r < a->rounds; r++)
		br_round(a, r);
	BarrierDetach(a->b);
	return NULL;
}

TEST(dynamic_attach_multiround)
{
	Barrier		 b;
	_Atomic(int) counter = 0;
	_Atomic(int) errors	 = 0;
	_Atomic(int) elected = 0;
	int			 party	 = BR_WORKERS + 1; /* workers + leader */
	pthread_t	 th[BR_WORKERS];

	BarrierInit(&b, 1); /* leader is the initial party */

	BRoundArg arg = {
			.b		 = &b,
			.rounds	 = BR_ROUNDS,
			.party	 = party,
			.counter = &counter,
			.errors	 = &errors,
			.elected = &elected,
	};

	for (int i = 0; i < BR_WORKERS; i++)
		pthread_create(&th[i], NULL, br_worker, &arg);

	/*
	 * Wait for every worker to attach before the leader arrives, so a partial
	 * party can never complete a phase early (this is what the real launch
	 * wait does via WaitForParallelWorkersToAttach + BarrierParticipants).
	 */
	while (BarrierParticipants(&b) < party)
		sched_yield();

	for (int r = 0; r < BR_ROUNDS; r++)
		br_round(&arg, r);

	for (int i = 0; i < BR_WORKERS; i++)
		pthread_join(th[i], NULL);

	ASSERT_EQ(0, atomic_load(&errors), "all participants synced every round");
	ASSERT_EQ(
			BR_ROUNDS * 2,
			atomic_load(&elected),
			"exactly one election per phase completion");
	ASSERT_EQ(
			BR_ROUNDS * 2,
			BarrierPhase(&b),
			"phase advanced once per barrier");
	ASSERT_EQ(1, BarrierParticipants(&b), "workers detached, leader remains");
}

/* ----------------------------------------------------------------
 * Detach completes the phase for the remaining (already-arrived) party
 * ---------------------------------------------------------------- */

typedef struct
{
	Barrier		 *b;
	_Atomic(int) *released;
} BDetachArg;

static void *
br_detach_worker(void *arg)
{
	BDetachArg *a = (BDetachArg *)arg;

	BarrierAttach(a->b);
	/* Blocks until the leader's detach completes the phase. */
	BarrierArriveAndWait(a->b, 0);
	atomic_store(a->released, 1);
	BarrierDetach(a->b);
	return NULL;
}

TEST(detach_releases_waiters)
{
	Barrier		 b;
	_Atomic(int) released = 0;
	pthread_t	 th;

	BarrierInit(&b, 1); /* leader */

	BDetachArg arg = {.b = &b, .released = &released};
	pthread_create(&th, NULL, br_detach_worker, &arg);

	/* Worker has attached: party is now 2. */
	while (BarrierParticipants(&b) < 2)
		sched_yield();

	/* Let the worker reach and block in BarrierArriveAndWait. */
	usleep(20000);

	/*
	 * Leader detaches instead of arriving: party 2 -> 1, and the single
	 * already-arrived worker now equals the party, so the phase completes and
	 * the worker is released (rather than waiting forever for the departed
	 * leader).
	 */
	bool released_by_detach = BarrierDetach(&b);

	pthread_join(th, NULL);

	ASSERT_TRUE(
			released_by_detach, "detach that empties the party releases it");
	ASSERT_EQ(1, atomic_load(&released), "the waiting worker was woken");
}
