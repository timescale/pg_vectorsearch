/*
 * barrier_standalone.c - Dynamic phase barrier over pthreads
 *
 * Standalone implementation of the PostgreSQL Barrier API used by the
 * parallel build. See mkt_barrier.h. PG builds use PostgreSQL's Barrier
 * instead, so this file is compiled only for standalone.
 */

#ifdef MKT_STANDALONE

#include "standalone/barrier.h"

void
BarrierInit(Barrier *barrier, int participants)
{
	pthread_mutex_init(&barrier->mutex, NULL);
	pthread_cond_init(&barrier->cv, NULL);
	barrier->participants = participants;
	barrier->arrived	  = 0;
	barrier->phase		  = 0;
}

bool
BarrierArriveAndWait(Barrier *barrier, uint32_t wait_event_info)
{
	bool elected;

	(void)wait_event_info; /* no wait-event reporting in standalone */

	pthread_mutex_lock(&barrier->mutex);

	int my_phase = barrier->phase;

	if (++barrier->arrived == barrier->participants)
	{
		/* Last to arrive: complete the phase and release everyone. */
		barrier->arrived = 0;
		barrier->phase++;
		pthread_cond_broadcast(&barrier->cv);
		elected = true;
	}
	else
	{
		/* Wait until some other participant completes the phase. */
		while (barrier->phase == my_phase)
			pthread_cond_wait(&barrier->cv, &barrier->mutex);
		elected = false;
	}

	pthread_mutex_unlock(&barrier->mutex);
	return elected;
}

int
BarrierAttach(Barrier *barrier)
{
	int phase;

	pthread_mutex_lock(&barrier->mutex);
	barrier->participants++;
	phase = barrier->phase;
	pthread_mutex_unlock(&barrier->mutex);
	return phase;
}

bool
BarrierDetach(Barrier *barrier)
{
	bool released = false;

	pthread_mutex_lock(&barrier->mutex);
	barrier->participants--;

	/*
	 * If the participants that remain have all already arrived, this detach
	 * completes the current phase — otherwise they would wait forever for a
	 * party member that has left.
	 */
	if (barrier->arrived > 0 && barrier->arrived == barrier->participants)
	{
		barrier->arrived = 0;
		barrier->phase++;
		pthread_cond_broadcast(&barrier->cv);
		released = true;
	}

	pthread_mutex_unlock(&barrier->mutex);
	return released;
}

int
BarrierPhase(Barrier *barrier)
{
	int phase;

	pthread_mutex_lock(&barrier->mutex);
	phase = barrier->phase;
	pthread_mutex_unlock(&barrier->mutex);
	return phase;
}

int
BarrierParticipants(Barrier *barrier)
{
	int n;

	pthread_mutex_lock(&barrier->mutex);
	n = barrier->participants;
	pthread_mutex_unlock(&barrier->mutex);
	return n;
}

#endif /* MKT_STANDALONE */
