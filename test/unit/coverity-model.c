/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * coverity-model.c - Coverity Scan analysis model
 *
 * Not in unit_test_sources, so nothing compiles it. Scan reads a model
 * from the project's analysis settings rather than from the uploaded
 * capture, so a change here takes effect only once a project admin
 * re-uploads the file:
 *
 *   https://scan.coverity.com/projects/timescale-pg_vectorsearch
 *   -> Analysis Settings -> upload this file
 *
 * A model describes what the analysis cannot see for itself. It may not
 * include headers, and needs only the declarations it uses -- a
 * modelled function's body replaces the real one, so it stands in for
 * behaviour rather than reproducing it.
 */

/*
 * The assertion macros in test/unit/vs_test.h report through this and
 * then return from the test body, which skips every destructor below
 * them. The analysis follows that path and reports whatever the test had
 * allocated as leaked -- once per allocation, in every test that cleans
 * up after an assertion. Treating a reported failure as terminal keeps
 * the path out of the analysis, which is what the test binary does with
 * it in practice.
 */
void
vs_test_fail(const char *file, int line, const char *msg)
{
	(void)file;
	(void)line;
	(void)msg;
	__coverity_panic__();
}
