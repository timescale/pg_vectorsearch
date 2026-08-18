-- SQL entry points for the test-only helper module
-- (test/pg/src/test_helpers.c).
--
-- Include from a test with:
--     \getenv abs_srcdir PG_ABS_SRCDIR
--     \set helper_sql :abs_srcdir '/sql/test_helpers.sql'
--     \i :helper_sql
-- (absolute path because pg_regress feeds test scripts to psql on
-- stdin), and DROP the functions when the test is done.
--
-- The probe below loads the extension library, which the helper
-- module's symbols require (RTLD_NOW).
SELECT mkt.extension_version() IS NOT NULL AS extension_loaded;
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
