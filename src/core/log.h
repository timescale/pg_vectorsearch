/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * log.h - Platform-independent logging
 *
 * Maps to elog() in PostgreSQL, fprintf(stderr) in standalone.
 */

#ifndef VS_LOG_H
#define VS_LOG_H

#ifdef VS_STANDALONE

#include <stdio.h>
#include <stdlib.h>

#define vs_log(...)	 fprintf(stderr, __VA_ARGS__)
#define vs_warn(...) fprintf(stderr, __VA_ARGS__)

/* Verbose diagnostics (e.g. per-phase build timings). Standalone keeps writing
 * to stderr so the CLI still shows them; under PG (below) this is DEBUG1,
 * hidden unless log_min_messages is lowered. */
#define vs_debug(...) fprintf(stderr, __VA_ARGS__)

/* Unrecoverable error: report and abort (PG's elog(ERROR) longjmps instead).
 */
#define vs_error(...)                 \
	do                                \
	{                                 \
		fprintf(stderr, __VA_ARGS__); \
		fputc('\n', stderr);          \
		abort();                      \
	} while (0)

#else /* PostgreSQL */

#include <postgres.h>

#define vs_log(...)	  elog(LOG, __VA_ARGS__)
#define vs_warn(...)  elog(WARNING, __VA_ARGS__)
#define vs_debug(...) elog(DEBUG1, __VA_ARGS__)
#define vs_error(...) elog(ERROR, __VA_ARGS__)

#endif

#endif /* VS_LOG_H */
