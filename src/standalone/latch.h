/*
 * mkt_latch.h - Wait/wake latch
 *
 * The leader's posting drain loop sleeps on its latch when no worker queue has
 * a page ready, and a worker wakes it by setting that latch when it sends or
 * detaches. In a PostgreSQL build this is PostgreSQL's Latch
 * (storage/latch.h), driven by the process's MyLatch. In a standalone
 * (thread-based) build we provide the same API over a pthread condition
 * variable, so the drain loop runs unchanged — and is therefore actually
 * exercised by standalone tests.
 */

#ifndef MKT_LATCH_H
#define MKT_LATCH_H

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>

typedef struct Latch
{
	pthread_mutex_t mutex;
	pthread_cond_t	cv;
	bool			is_set;
} Latch;

/* WaitLatch wake-event flags (values are local; only the bits matter). */
#define WL_LATCH_SET		(1 << 0)
#define WL_TIMEOUT			(1 << 1)
#define WL_EXIT_ON_PM_DEATH (1 << 2)

/*
 * The latch the calling thread waits on. Each participating thread owns a
 * latch and binds it with mkt_latch_attach_self before waiting; senders reach
 * a thread's latch through the queue's stored receiver, mirroring how PG
 * reaches a process's MyLatch.
 */
extern __thread Latch *MyLatch;

extern void InitLatch(Latch *latch);
extern void SetLatch(Latch *latch);
extern void ResetLatch(Latch *latch);

/*
 * Wait until the latch is set, or (when WL_TIMEOUT is requested and timeout_ms
 * >= 0) until the timeout elapses. Returns the wake events that fired
 * (WL_LATCH_SET and/or WL_TIMEOUT). The wait-event argument is ignored.
 */
extern int WaitLatch(
		Latch	*latch,
		int		 wakeEvents,
		long	 timeout_ms,
		uint32_t wait_event_info);

/* Standalone-only: bind this thread's MyLatch to a latch it owns. */
extern void mkt_latch_attach_self(Latch *latch);

#endif /* MKT_LATCH_H */
