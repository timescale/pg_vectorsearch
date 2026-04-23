/*
 * log.h - Platform-independent logging
 *
 * Maps to elog() in PostgreSQL, fprintf(stderr) in standalone.
 */

#ifndef MKT_LOG_H
#define MKT_LOG_H

#ifdef MKT_STANDALONE

#include <stdio.h>

#define mkt_warn(...) fprintf(stderr, __VA_ARGS__)

#else /* PostgreSQL */

#include <postgres.h>

#define mkt_warn(...) elog(WARNING, __VA_ARGS__)

#endif

#endif /* MKT_LOG_H */
