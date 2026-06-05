/*
 * log.h - Platform-independent logging
 *
 * Maps to elog() in PostgreSQL, fprintf(stderr) in standalone.
 */

#ifndef MKT_LOG_H
#define MKT_LOG_H

#ifdef MKT_STANDALONE

#include <stdio.h>
#include <stdlib.h>

#define mkt_log(...)  fprintf(stderr, __VA_ARGS__)
#define mkt_warn(...) fprintf(stderr, __VA_ARGS__)

/* Verbose diagnostics (e.g. per-phase build timings). Standalone keeps writing
 * to stderr so the CLI still shows them; under PG (below) this is DEBUG1,
 * hidden unless log_min_messages is lowered. */
#define mkt_debug(...) fprintf(stderr, __VA_ARGS__)

/* Unrecoverable error: report and abort (PG's elog(ERROR) longjmps instead).
 */
#define mkt_error(...)                \
	do                                \
	{                                 \
		fprintf(stderr, __VA_ARGS__); \
		fputc('\n', stderr);          \
		abort();                      \
	} while (0)

#else /* PostgreSQL */

#include <postgres.h>

#define mkt_log(...)   elog(LOG, __VA_ARGS__)
#define mkt_warn(...)  elog(WARNING, __VA_ARGS__)
#define mkt_debug(...) elog(DEBUG1, __VA_ARGS__)
#define mkt_error(...) elog(ERROR, __VA_ARGS__)

#endif

#endif /* MKT_LOG_H */
