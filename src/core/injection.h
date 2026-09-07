/*
 * injection.h - Test injection points, compiled out of the standalone build
 *
 * Wraps PostgreSQL's INJECTION_POINT so backend-neutral code can carry a
 * pause point for isolation tests without a version-specific path at the call
 * site. Standalone has no concurrent writers to race against, so there it
 * compiles to nothing.
 *
 * Each point is a no-op unless PostgreSQL was built with injection points
 * (USE_INJECTION_POINTS) and a test attached an action to that name.
 */

#ifndef MKT_INJECTION_H
#define MKT_INJECTION_H

#ifdef MKT_STANDALONE

#define MKT_INJECTION_POINT(name) ((void)0)

#else

#include <postgres.h>

#include <utils/injection_point.h>

#define MKT_INJECTION_POINT(name) INJECTION_POINT(name, NULL)

#endif

#endif /* MKT_INJECTION_H */
