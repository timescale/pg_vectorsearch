-- SQL entry points for the test-only helper module
-- (test/pg/src/test_helpers.c). Not a test: \ir this file from tests
-- that need the extra introspection, and DROP the functions when done.
-- The extension library must be loaded before the helper module
-- resolves its exported symbols (RTLD_NOW), hence the probe call.
SELECT mkt.extension_version() IS NOT NULL AS extension_loaded;
CREATE OR REPLACE FUNCTION rabitq_params_cache(
    OUT dim integer,
    OUT refcount integer,
    OUT usage float8)
RETURNS SETOF record
AS '$libdir/meerkat_test_helpers', 'mkt_test_rabitq_params_cache'
LANGUAGE C PARALLEL RESTRICTED;
