-- Silence this file's own definitions and documentation so they stay
-- out of the expected output of any test that includes it. Capture the
-- caller's current ECHO into a variable and restore it at the end
-- (below) rather than assuming a value, so whatever the including test
-- was using keeps working. Everything past \set ECHO none -- these
-- comments included -- is read with echo off.
\set saved_echo :ECHO
\set ECHO none

-- SQL entry points for the test-only helper module
-- (test/pg/src/test_helpers.c), which exposes introspection the
-- extension itself must not ship. Include from a test with:
--
--     \getenv abs_srcdir PG_ABS_SRCDIR
--     \set helper_sql :abs_srcdir '/sql/test_helpers.sql'
--     \i :helper_sql
--
-- (absolute path because pg_regress feeds test scripts to psql on
-- stdin, so a relative \ir has no directory to resolve against), and
-- DROP the functions when the test is done.

-- Force-load the extension library first: the helper module's symbols
-- resolve against it (RTLD_NOW), and calling any extension C function
-- loads it. A DO block returns nothing, so it stays silent.
DO $$ BEGIN PERFORM extension_version(); END $$;

CREATE OR REPLACE FUNCTION rabitq_params_cache(
    OUT dim integer,
    OUT refcount integer,
    OUT usage float8)
RETURNS SETOF record
AS '$libdir/meerkat_test_helpers', 'mkt_test_rabitq_params_cache'
LANGUAGE C PARALLEL RESTRICTED;

CREATE OR REPLACE FUNCTION rabitq_cache_clear()
RETURNS integer
AS '$libdir/meerkat_test_helpers', 'mkt_test_rabitq_cache_clear'
LANGUAGE C PARALLEL RESTRICTED;

-- Restore the caller's echo setting captured above.
\set ECHO :saved_echo
