/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * vs_barrier.h - Dynamic phase barrier
 *
 * The parallel build synchronizes its phases (sampling, k-means iterations,
 * posting) with a barrier whose party size changes at run time: participants
 * attach before the first phase and detach when they finish, because the
 * number that actually shows up is not known up front.
 *
 * In a PostgreSQL build this is exactly PostgreSQL's own Barrier
 * (storage/barrier.h), so we just use that. In a standalone (thread-based)
 * build we provide the same API over pthreads. A plain pthread_barrier_t
 * cannot serve here: its party size is fixed at init, with no attach/detach.
 */

#ifndef VS_BARRIER_H
#define VS_BARRIER_H

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>

/*
 * Same semantics as PostgreSQL's Barrier: a monotonically increasing phase
 * counter, a dynamic party size, and an arrived count for the current phase.
 * The last participant to arrive (or the detach that leaves every remaining
 * participant already arrived) advances the phase and releases the waiters.
 */
typedef struct Barrier
{
	pthread_mutex_t mutex;
	pthread_cond_t	cv;
	int				participants; /* current party size */
	int				arrived;	  /* arrived in the current phase */
	int				phase;		  /* advances each time the party completes */
} Barrier;

/* Initialize with a starting party of `participants` (may be 0). */
extern void BarrierInit(Barrier *barrier, int participants);

/*
 * Arrive at the current phase and block until every participant has. Returns
 * true to exactly one caller per phase (the one that completed it), false to
 * the rest — mirroring PostgreSQL so "elected" work can run once. The wait
 * event argument is ignored in standalone (no wait-event reporting).
 */
extern bool BarrierArriveAndWait(Barrier *barrier, uint32_t wait_event_info);

/* Add one participant to the party; returns the current phase. */
extern int BarrierAttach(Barrier *barrier);

/*
 * Remove one participant from the party. Returns true if this detach completed
 * the current phase (i.e. it left every remaining participant already
 * arrived), releasing the waiters.
 */
extern bool BarrierDetach(Barrier *barrier);

/* Current phase / party size (taken under the lock). */
extern int BarrierPhase(Barrier *barrier);
extern int BarrierParticipants(Barrier *barrier);

#endif /* VS_BARRIER_H */
