-- search_path / privilege-escalation hardening tests
--
-- Simulates the attacks that motivated the install-script hardening and
-- asserts they no longer work. The behaviour is made visible: each check
-- prints PASS/FAIL, and the script exits non-zero if any check fails.
--
-- Background. setup_pgvector_compat() and on_extension_create() run at
-- runtime -- from the install DO block and from an event trigger that
-- fires on any later CREATE EXTENSION -- under the DDL-runner's own
-- search_path. Before hardening, an unqualified format() call there could
-- be hijacked by an attacker-planted overload
-- (format(text,text,text,text,text) beats pg_catalog.format(text,
-- VARIADIC "any") from any path position), and the CREATE CAST steps
-- silently adopted any pre-existing cast. pgvector is a "trusted"
-- extension, so a non-superuser can own public.vector and reach both.
--
-- Prerequisites:
--   pgvector and meerkat must be installed in PostgreSQL, and the
--   connected role must be a superuser (to install extensions and
--   exercise the event-trigger escalation path).
--
-- Usage:
--   psql -f test/pg/compat/security.sql

\set ON_ERROR_STOP on
\pset tuples_only on
\pset format unaligned

-- Known, deterministic starting state: meerkat first (so its event
-- trigger is active), then pgvector (fires the trigger -> creates casts).
DROP EXTENSION IF EXISTS meerkat CASCADE;
DROP EXTENSION IF EXISTS vector CASCADE;
CREATE EXTENSION meerkat;
CREATE EXTENSION vector;

CREATE TEMP TABLE test_results (name text, passed bool);

CREATE OR REPLACE FUNCTION assert_test(test_name text, condition bool)
RETURNS void AS $$
BEGIN
    INSERT INTO test_results VALUES (test_name, condition);
    IF condition THEN
        RAISE NOTICE 'PASS: %', test_name;
    ELSE
        RAISE WARNING 'FAIL: %', test_name;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- =====================================================================
-- 1. The interop functions pin their search_path
-- =====================================================================
-- A pinned path is what makes every unqualified name in these bodies
-- resolve in pg_catalog rather than an attacker-controlled schema.

SELECT assert_test('setup_pgvector_compat pins search_path',
    (SELECT proconfig @> ARRAY['search_path=pg_catalog, pg_temp']
       FROM pg_proc
      WHERE proname = 'setup_pgvector_compat'
        AND pronamespace = 'mkt'::regnamespace));

SELECT assert_test('on_extension_create pins search_path',
    (SELECT proconfig @> ARRAY['search_path=pg_catalog, pg_temp']
       FROM pg_proc
      WHERE proname = 'on_extension_create'
        AND pronamespace = 'mkt'::regnamespace));

-- The dynamic-SQL sink is schema-qualified as pg_catalog.format.
SELECT assert_test('setup body calls pg_catalog.format, not bare format',
    pg_get_functiondef('mkt.setup_pgvector_compat'::regproc)
        LIKE '%pg_catalog.format(%');

-- =====================================================================
-- 2. A planted format() overload is NOT invoked
-- =====================================================================
-- Put an attacker-shaped format(text,text,text,text,text) on the caller's
-- search_path, then run the compat setup. The pinned path + qualified call
-- must resolve to pg_catalog.format, so the planted body never fires.

CREATE SCHEMA hijack_probe;
CREATE TABLE hijack_probe.hit (seen bool);

CREATE FUNCTION hijack_probe.format(text, text, text, text, text)
RETURNS text LANGUAGE plpgsql AS $fmt$
BEGIN
    INSERT INTO hijack_probe.hit VALUES (true);   -- observable side effect
    RETURN 'SELECT 1';
END;
$fmt$;

SET search_path = hijack_probe, public;
SELECT mkt.setup_pgvector_compat();   -- idempotent; casts already present
RESET search_path;

SELECT assert_test('planted format() overload is NOT invoked',
    NOT EXISTS (SELECT 1 FROM hijack_probe.hit));

DROP SCHEMA hijack_probe CASCADE;

-- =====================================================================
-- 3. A tampered (WITH FUNCTION) cast is rejected, not silently adopted
-- =====================================================================
-- Replace the expected binary cast with a WITH FUNCTION cast -- the shape
-- a public.vector owner could plant, whose function would then run as the
-- querying role. setup_pgvector_compat() must refuse it (RAISE), where the
-- old blind "EXCEPTION WHEN duplicate_object THEN NULL" would have kept it.

DROP CAST IF EXISTS (public.vector AS mkt.vector);
CREATE FUNCTION public.evil_cast(public.vector) RETURNS mkt.vector
    LANGUAGE sql IMMUTABLE AS $ec$ SELECT '[0]'::mkt.vector $ec$;
CREATE CAST (public.vector AS mkt.vector)
    WITH FUNCTION public.evil_cast(public.vector) AS IMPLICIT;

DO $chk$
DECLARE
    raised bool := false;
BEGIN
    BEGIN
        PERFORM mkt.setup_pgvector_compat();
    EXCEPTION WHEN OTHERS THEN
        raised := true;
    END;
    PERFORM assert_test(
        'tampered WITH FUNCTION cast is rejected (fail loud)', raised);
END;
$chk$;

-- Restore the expected binary cast.
DROP CAST (public.vector AS mkt.vector);
DROP FUNCTION public.evil_cast(public.vector);
CREATE CAST (public.vector AS mkt.vector) WITHOUT FUNCTION AS IMPLICIT;

-- =====================================================================
-- 4. End-to-end: a non-superuser cannot escalate via the event trigger
-- =====================================================================
-- The headline attack. A NOSUPERUSER plants an escalating format()
-- overload; a superuser then runs CREATE EXTENSION vector, which fires
-- meerkat's event trigger -> setup_pgvector_compat(). If the sink were
-- hijackable, the planted body would run as the superuser and grant the
-- attacker SUPERUSER. It must not.

DROP ROLE IF EXISTS mkt_attacker;
CREATE ROLE mkt_attacker NOSUPERUSER LOGIN;
-- Simulate a deployment where the role can create in a schema on the
-- admin's search_path (public); the attack does not depend on which
-- schema, only that the overload is visible.
GRANT CREATE ON SCHEMA public TO mkt_attacker;

SET ROLE mkt_attacker;
CREATE FUNCTION public.format(text, text, text, text, text)
RETURNS text LANGUAGE plpgsql AS $atk$
BEGIN
    IF NOT (SELECT rolsuper FROM pg_roles WHERE rolname = 'mkt_attacker')
    THEN
        EXECUTE 'ALTER ROLE mkt_attacker SUPERUSER';
    END IF;
    RETURN 'SELECT 1';
END;
$atk$;
RESET ROLE;

-- Superuser re-installs pgvector, firing the event trigger.
DROP EXTENSION vector CASCADE;
CREATE EXTENSION vector;

SELECT assert_test('non-superuser did NOT escalate via event trigger',
    NOT (SELECT rolsuper FROM pg_roles WHERE rolname = 'mkt_attacker'));

-- Cleanup. DROP OWNED clears the schema grant (and anything else the role
-- holds) so DROP ROLE does not fail on dependent privileges.
DROP FUNCTION public.format(text, text, text, text, text);
DROP OWNED BY mkt_attacker;
DROP ROLE mkt_attacker;

-- =====================================================================
-- Summary
-- =====================================================================

\echo
\echo ====================================
\echo search_path hardening test results
\echo ====================================

SELECT format('%s: %s',
    CASE WHEN passed THEN 'PASS' ELSE 'FAIL' END, name)
FROM test_results
ORDER BY passed, name;

\echo

SELECT format('Total: %s passed, %s failed out of %s tests',
    count(*) FILTER (WHERE passed),
    count(*) FILTER (WHERE NOT passed),
    count(*))
FROM test_results;

-- Exit with error if any check failed.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM test_results WHERE NOT passed) THEN
        RAISE EXCEPTION 'Some search_path hardening tests failed';
    END IF;
END;
$$;
