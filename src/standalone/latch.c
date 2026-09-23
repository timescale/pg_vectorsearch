/*
 * latch_standalone.c - Wait/wake latch over a condition variable
 *
 * Standalone implementation of the Latch API the posting drain loop uses (see
 * vs_latch.h). PG builds use PostgreSQL's Latch instead, so this file is
 * compiled only for standalone.
 */

#ifdef VS_STANDALONE

#include <errno.h>
#include <time.h>

#include "standalone/latch.h"

__thread Latch *MyLatch = NULL;

void
InitLatch(Latch *latch)
{
	pthread_mutex_init(&latch->mutex, NULL);
	pthread_cond_init(&latch->cv, NULL);
	latch->is_set = false;
}

void
vs_latch_attach_self(Latch *latch)
{
	MyLatch = latch;
}

void
SetLatch(Latch *latch)
{
	pthread_mutex_lock(&latch->mutex);
	latch->is_set = true;
	pthread_cond_broadcast(&latch->cv);
	pthread_mutex_unlock(&latch->mutex);
}

void
ResetLatch(Latch *latch)
{
	pthread_mutex_lock(&latch->mutex);
	latch->is_set = false;
	pthread_mutex_unlock(&latch->mutex);
}

int
WaitLatch(
		Latch	*latch,
		int		 wakeEvents,
		long	 timeout_ms,
		uint32_t wait_event_info)
{
	int result = 0;

	(void)wait_event_info;

	pthread_mutex_lock(&latch->mutex);

	if (!latch->is_set)
	{
		if ((wakeEvents & WL_TIMEOUT) && timeout_ms >= 0)
		{
			struct timespec ts;
			int				rc = 0;

			clock_gettime(CLOCK_REALTIME, &ts);
			ts.tv_sec += timeout_ms / 1000;
			ts.tv_nsec += (timeout_ms % 1000) * 1000000L;
			if (ts.tv_nsec >= 1000000000L)
			{
				ts.tv_sec++;
				ts.tv_nsec -= 1000000000L;
			}

			while (!latch->is_set && rc != ETIMEDOUT)
				rc = pthread_cond_timedwait(&latch->cv, &latch->mutex, &ts);

			if (rc == ETIMEDOUT)
				result |= WL_TIMEOUT;
		}
		else
		{
			while (!latch->is_set)
				pthread_cond_wait(&latch->cv, &latch->mutex);
		}
	}

	if (latch->is_set)
		result |= WL_LATCH_SET;

	pthread_mutex_unlock(&latch->mutex);
	return result;
}

#endif /* VS_STANDALONE */
