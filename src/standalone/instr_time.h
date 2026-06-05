/*
 * mkt_instr_time.h - Monotonic timing for build phase measurements
 *
 * The build logs phase durations using PostgreSQL's instr_time. In a PG build
 * that's portability/instr_time.h; in standalone we provide the small subset
 * the build uses (set/subtract/get-millis) over clock_gettime, so the shared
 * timing code is unchanged.
 */

#ifndef MKT_INSTR_TIME_H
#define MKT_INSTR_TIME_H

#include <stdint.h>
#include <time.h>

typedef struct instr_time
{
	int64_t ticks; /* nanoseconds */
} instr_time;

static inline void
mkt_instr_now(instr_time *t)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	t->ticks = (int64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
}

#define INSTR_TIME_SET_CURRENT(t)  mkt_instr_now(&(t))
#define INSTR_TIME_SUBTRACT(x, y)  ((x).ticks -= (y).ticks)
#define INSTR_TIME_GET_MILLISEC(t) ((double)(t).ticks / 1e6)

#endif /* MKT_INSTR_TIME_H */
