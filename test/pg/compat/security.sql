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
-- The extension schema is never hard-coded below. @extschema@ is
-- substituted only by CREATE EXTENSION while it reads the extension
-- script; this is a plain psql script, so the schema is looked up from
-- the catalog once and interpolated as a psql variable (:extschema).
-- Dollar-quoted bodies, which psql does not interpolate, look it up again
-- at runtime.
--
-- Prerequisites:
--   pgvector and meerkat must be installed in PostgreSQL, and the
--   connected role must be a superuser (to install extensions and
--   exercise the event-trigger escalation path). This suite drops and
--   recreates both extensions and the `mkt` schema (CASCADE) as it runs, so
--   run it against a throwaway/clean database, not one holding data you care
--   about. The CI script uses a fresh instance.
--
-- Usage:
--   psql -f test/pg/compat/security.sql

\set ON_ERROR_STOP on
\pset tuples_only on
\pset format unaligned

CREATE TEMP TABLE test_results (name text, passed bool);

CREATE OR REPLACE FUNCTION assert_test(test_name text, condition bool)
RETURNS void AS $$
BEGIN
    -- Normalize NULL to false: a check that evaluates to NULL (e.g. a missing
    -- proconfig pin) is a failure, not a pass, and must not slip past the
    -- final "WHERE NOT passed" exit guard as neither-true-nor-false.
    condition := COALESCE(condition, false);
    INSERT INTO test_results VALUES (test_name, condition);
    IF condition THEN
        RAISE NOTICE 'PASS: %', test_name;
    ELSE
        RAISE WARNING 'FAIL: %', test_name;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Plant attacker format() overloads across EVERY arity the install script
-- calls format() with. The sinks currently use 2 args (ALTER EXTENSION DROP
-- CAST), 4 (the cast loop) and 8 (the operator-family loop); planting a whole
-- range (2..10) means a hijack is caught regardless of arity, so a future
-- change to a format() call's arity cannot silently disarm these checks -- an
-- earlier version planted only a 5-arg overload and quietly matched none of
-- the real sinks. Each overload runs `body` (records the call, or escalates)
-- and then delegates to pg_catalog.format, so if it is ever reached the DDL
-- still runs correctly and the check reports a clean FAIL instead of aborting.
CREATE OR REPLACE FUNCTION plant_format_overloads(tgt_schema text, body text)
RETURNS void LANGUAGE plpgsql AS $pf$
DECLARE
    n int;
    params text;
    passthru text;
BEGIN
    FOR n IN 2..10 LOOP
        params   := (SELECT string_agg('text', ', ')
                       FROM generate_series(1, n));
        passthru := (SELECT string_agg('$' || i, ', ')
                       FROM generate_series(2, n) AS i);
        EXECUTE pg_catalog.format(
            'CREATE FUNCTION %I.format(%s) RETURNS text LANGUAGE plpgsql AS '
            '$body$ BEGIN %s; RETURN pg_catalog.format($1, VARIADIC '
            'ARRAY[%s]::text[]); END $body$',
            tgt_schema, params, body, passthru);
    END LOOP;
END;
$pf$;

CREATE OR REPLACE FUNCTION drop_format_overloads(tgt_schema text)
RETURNS void LANGUAGE plpgsql AS $pf$
DECLARE
    n int;
    params text;
BEGIN
    FOR n IN 2..10 LOOP
        params := (SELECT string_agg('text', ', ') FROM generate_series(1, n));
        EXECUTE pg_catalog.format('DROP FUNCTION IF EXISTS %I.format(%s)',
                                  tgt_schema, params);
    END LOOP;
END;
$pf$;

-- =====================================================================
-- 0. Install-time: a format() planted in a pre-created extension schema
--    is not invoked by CREATE EXTENSION itself
-- =====================================================================
-- The real install-time attack. An attacker pre-creates the extension
-- schema (relocatable=false pins it to a known name) and plants format()
-- overloads before CREATE EXTENSION meerkat runs. During install the
-- schema is on the forced search_path, so a bare format() in
-- setup_pgvector_compat() (reached because pgvector is already present)
-- would resolve to the plant and run as the installing superuser.
--
-- This is the one place that must name the schema literally: the extension
-- does not exist yet, so its schema cannot be looked up from the catalog.
DROP EXTENSION IF EXISTS meerkat CASCADE;
DROP EXTENSION IF EXISTS vector CASCADE;
DROP SCHEMA IF EXISTS install_probe CASCADE;
DROP SCHEMA IF EXISTS mkt CASCADE;

CREATE SCHEMA install_probe;
CREATE TABLE install_probe.hit (seen bool);
CREATE SCHEMA mkt;   -- literal: the attacker targets the known schema name
SELECT plant_format_overloads('mkt',
    'INSERT INTO install_probe.hit VALUES (true)');

-- pgvector first so the compat path runs inside CREATE EXTENSION meerkat.
CREATE EXTENSION vector;
CREATE EXTENSION meerkat;

-- Resolve the extension's schema now that it exists. Used as :"extschema"
-- (identifier) and :'extschema' (literal) throughout the rest of the file.
SELECT n.nspname AS extschema
  FROM pg_extension e
  JOIN pg_namespace n ON n.oid = e.extnamespace
 WHERE e.extname = 'meerkat' \gset

SELECT assert_test('CREATE EXTENSION does not call a planted mkt.format()',
    NOT EXISTS (SELECT 1 FROM install_probe.hit));

-- Prove the compat path actually ran (else "not called" would be vacuous):
-- setup_pgvector_compat() creates the pgvector->meerkat cast.
SELECT assert_test('CREATE EXTENSION still created the pgvector compat cast',
    EXISTS (SELECT 1 FROM pg_cast
             WHERE castsource = 'public.vector'::regtype
               AND casttarget = (:'extschema' || '.vector')::regtype));

SELECT drop_format_overloads('mkt');
DROP SCHEMA install_probe CASCADE;

-- =====================================================================
-- 0b. Install is refused when the extension schema is pre-owned by an
--     untrusted role
-- =====================================================================
-- A role with CREATE on the database can pre-create mkt and keep owning it
-- after install, then add objects that shadow pgvector's operators for any
-- role with mkt ahead of public (meerkat's documented usage). The schema
-- ownership guard at the top of the install script must refuse that install.
DROP EXTENSION IF EXISTS meerkat CASCADE;
DROP EXTENSION IF EXISTS vector CASCADE;
DROP SCHEMA IF EXISTS mkt CASCADE;
DROP ROLE IF EXISTS mkt_squatter;
CREATE ROLE mkt_squatter NOSUPERUSER;
CREATE SCHEMA mkt AUTHORIZATION mkt_squatter;   -- untrusted role owns mkt

\set ON_ERROR_STOP off
CREATE EXTENSION meerkat;   -- must be refused by the ownership guard
\set ON_ERROR_STOP on

SELECT assert_test(
    'install refused when the extension schema is pre-owned by an untrusted role',
    NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'meerkat'));

DROP SCHEMA mkt CASCADE;
DROP OWNED BY mkt_squatter;
DROP ROLE mkt_squatter;

-- Deterministic starting state for the remaining checks: drop and reinstall
-- cleanly (meerkat first so its event trigger is active, then pgvector).
DROP EXTENSION IF EXISTS meerkat CASCADE;
DROP EXTENSION IF EXISTS vector CASCADE;
CREATE EXTENSION meerkat;
CREATE EXTENSION vector;

-- =====================================================================
-- 1. The interop functions pin their search_path
-- =====================================================================
-- A pinned path is what makes every unqualified name in these bodies
-- resolve in pg_catalog rather than an attacker-controlled schema.

-- COALESCE so a missing pin (proconfig NULL) is a hard false, not NULL --
-- otherwise the row escapes the final "WHERE NOT passed" exit check.
SELECT assert_test('setup_pgvector_compat pins search_path',
    COALESCE((SELECT proconfig @> ARRAY['search_path=pg_catalog, pg_temp']
       FROM pg_proc
      WHERE proname = 'setup_pgvector_compat'
        AND pronamespace = :'extschema'::regnamespace), false));

SELECT assert_test('on_extension_create pins search_path',
    COALESCE((SELECT proconfig @> ARRAY['search_path=pg_catalog, pg_temp']
       FROM pg_proc
      WHERE proname = 'on_extension_create'
        AND pronamespace = :'extschema'::regnamespace), false));

-- The dynamic-SQL sink is schema-qualified as pg_catalog.format.
SELECT assert_test('setup body calls pg_catalog.format, not bare format',
    pg_get_functiondef((:'extschema' || '.setup_pgvector_compat')::regproc)
        LIKE '%pg_catalog.format(%');

-- =====================================================================
-- 2. A planted format() overload is NOT invoked
-- =====================================================================
-- Put attacker-shaped format() overloads (every arity the compat setup calls)
-- on the caller's search_path, then run the compat setup. The overloads RAISE
-- rather than record a row: post-install the only format() setup reaches is in
-- the operator loop, wrapped in "EXCEPTION WHEN duplicate_object" -- which
-- would roll back a recorded INSERT. A raise of a different error propagates
-- past that handler and out of setup, where the wrapper below catches it. The
-- pinned path + qualified call must resolve to pg_catalog.format, so no
-- planted body fires and setup completes without raising.

CREATE SCHEMA hijack_probe;
SELECT plant_format_overloads('hijack_probe',
    'RAISE EXCEPTION ''planted format() overload was invoked''');

SET search_path = hijack_probe, public;
DO $chk$
DECLARE
    invoked bool := false;
    eschema text := (SELECT n.nspname FROM pg_catalog.pg_extension e
                       JOIN pg_catalog.pg_namespace n ON n.oid = e.extnamespace
                      WHERE e.extname = 'meerkat');
BEGIN
    BEGIN
        -- pg_catalog.format so the wrapper itself cannot be hijacked.
        EXECUTE pg_catalog.format('SELECT %I.setup_pgvector_compat()', eschema);
    EXCEPTION WHEN OTHERS THEN
        invoked := true;   -- a planted overload raised -> it was called
    END;
    PERFORM assert_test('planted format() overload is NOT invoked',
        NOT invoked);
END;
$chk$;
RESET search_path;

SELECT drop_format_overloads('hijack_probe');
DROP SCHEMA hijack_probe CASCADE;

-- =====================================================================
-- 2b. A format() planted in the EXTENSION SCHEMA is NOT invoked
-- =====================================================================
-- The extension schema is on the forced search_path while CREATE EXTENSION
-- runs, so a bare format() there would resolve to a pre-planted
-- <extschema>.format (exact arity beats pg_catalog.format's VARIADIC "any").
-- The helpers pin their search_path to pg_catalog and call pg_catalog.format,
-- so format() overloads sitting in the extension schema must never run.

-- RAISE (not record) for the same reason as check 2: the operator loop's
-- duplicate_object handler would roll back a recorded row post-install.
SELECT plant_format_overloads(:'extschema',
    'RAISE EXCEPTION ''format() planted in the extension schema was invoked''');

-- Put the extension schema first on the path, as the install-time forced
-- search_path does, then run the compat setup.
SET search_path = :"extschema", public;
DO $chk$
DECLARE
    invoked bool := false;
    eschema text := (SELECT n.nspname FROM pg_catalog.pg_extension e
                       JOIN pg_catalog.pg_namespace n ON n.oid = e.extnamespace
                      WHERE e.extname = 'meerkat');
BEGIN
    BEGIN
        EXECUTE pg_catalog.format('SELECT %I.setup_pgvector_compat()', eschema);
    EXCEPTION WHEN OTHERS THEN
        invoked := true;
    END;
    PERFORM assert_test(
        'format() planted in the extension schema is NOT invoked', NOT invoked);
END;
$chk$;
RESET search_path;

SELECT drop_format_overloads(:'extschema');

-- =====================================================================
-- 3. A tampered (WITH FUNCTION) cast is rejected, not silently adopted
-- =====================================================================
-- Replace the expected binary cast with a WITH FUNCTION cast -- the shape
-- a public.vector owner could plant, whose function would then run as the
-- querying role. setup_pgvector_compat() must refuse it (RAISE), where the
-- old blind "EXCEPTION WHEN duplicate_object THEN NULL" would have kept it.
-- Built with dynamic SQL: the cast function's body must name the return
-- type, so %I keeps the fixture schema-agnostic.

DO $fix$
DECLARE
    eschema text := (SELECT n.nspname FROM pg_extension e
                       JOIN pg_namespace n ON n.oid = e.extnamespace
                      WHERE e.extname = 'meerkat');
BEGIN
    EXECUTE pg_catalog.format(
        'DROP CAST IF EXISTS (public.vector AS %I.vector)', eschema);
    EXECUTE pg_catalog.format(
        'CREATE FUNCTION public.evil_cast(public.vector) '
        'RETURNS %I.vector LANGUAGE sql IMMUTABLE AS %L',
        eschema,
        pg_catalog.format('SELECT ''[0]''::%I.vector', eschema));
    EXECUTE pg_catalog.format(
        'CREATE CAST (public.vector AS %I.vector) '
        'WITH FUNCTION public.evil_cast(public.vector) AS IMPLICIT',
        eschema);
END;
$fix$;

DO $chk$
DECLARE
    raised bool := false;
    eschema text := (SELECT n.nspname FROM pg_extension e
                       JOIN pg_namespace n ON n.oid = e.extnamespace
                      WHERE e.extname = 'meerkat');
BEGIN
    BEGIN
        EXECUTE pg_catalog.format('SELECT %I.setup_pgvector_compat()',
                                  eschema);
    EXCEPTION WHEN OTHERS THEN
        raised := true;
    END;
    PERFORM assert_test(
        'tampered WITH FUNCTION cast is rejected (fail loud)', raised);
END;
$chk$;

-- Restore the expected binary cast.
DROP CAST (public.vector AS :"extschema".vector);
DROP FUNCTION public.evil_cast(public.vector);
CREATE CAST (public.vector AS :"extschema".vector) WITHOUT FUNCTION AS IMPLICIT;

-- =====================================================================
-- 3b. A binary cast with the WRONG context is rejected too
-- =====================================================================
-- The pgv->mkt direction must be IMPLICIT and mkt->pgv ASSIGNMENT. A binary
-- (WITHOUT FUNCTION) cast planted with the wrong context still has method
-- 'b', so a method-only check would adopt it -- changing coercion/operator
-- resolution. setup must reject it on the castcontext mismatch.
DROP CAST (public.vector AS :"extschema".vector);
CREATE CAST (public.vector AS :"extschema".vector)
    WITHOUT FUNCTION AS ASSIGNMENT;   -- wrong: this direction must be IMPLICIT

DO $chk$
DECLARE
    raised bool := false;
    eschema text := (SELECT n.nspname FROM pg_catalog.pg_extension e
                       JOIN pg_catalog.pg_namespace n ON n.oid = e.extnamespace
                      WHERE e.extname = 'meerkat');
BEGIN
    BEGIN
        EXECUTE pg_catalog.format('SELECT %I.setup_pgvector_compat()', eschema);
    EXCEPTION WHEN OTHERS THEN
        raised := true;
    END;
    PERFORM assert_test(
        'binary cast with the wrong context is rejected (fail loud)', raised);
END;
$chk$;

-- Restore the expected binary cast.
DROP CAST (public.vector AS :"extschema".vector);
CREATE CAST (public.vector AS :"extschema".vector) WITHOUT FUNCTION AS IMPLICIT;

-- =====================================================================
-- 4. End-to-end: a non-superuser cannot escalate via the event trigger
-- =====================================================================
-- The headline attack. A NOSUPERUSER plants an escalating format()
-- overload; a superuser then runs CREATE EXTENSION vector, which fires
-- meerkat's event trigger -> setup_pgvector_compat(). If the sink were
-- hijackable, the planted body would run as the superuser and grant the
-- attacker SUPERUSER. It must not.

DROP ROLE IF EXISTS mkt_attacker;
CREATE ROLE mkt_attacker NOSUPERUSER NOLOGIN;   -- SET ROLE needs no LOGIN
-- Simulate a deployment where the role can create in a schema on the
-- admin's search_path (public); the attack does not depend on which
-- schema, only that the overload is visible.
GRANT CREATE ON SCHEMA public TO mkt_attacker;

SET ROLE mkt_attacker;
SELECT plant_format_overloads('public',
    'IF NOT (SELECT rolsuper FROM pg_roles WHERE rolname = ''mkt_attacker'')'
    ' THEN EXECUTE ''ALTER ROLE mkt_attacker SUPERUSER''; END IF');
RESET ROLE;

-- Superuser re-installs pgvector, firing the event trigger.
DROP EXTENSION vector CASCADE;
CREATE EXTENSION vector;

SELECT assert_test('non-superuser did NOT escalate via event trigger',
    NOT (SELECT rolsuper FROM pg_roles WHERE rolname = 'mkt_attacker'));

-- Cleanup. DROP OWNED clears the schema grant (and anything else the role
-- holds) so DROP ROLE does not fail on dependent privileges.
SELECT drop_format_overloads('public');
DROP OWNED BY mkt_attacker;
DROP ROLE mkt_attacker;

-- =====================================================================
-- Summary
-- =====================================================================

\echo
\echo ====================================
\echo search_path hardening test results
\echo ====================================

-- pg_catalog.format here too: this suite plants format() overloads, and a
-- pre-existing one on the session path could otherwise run as the superuser
-- during the summary.
SELECT pg_catalog.format('%s: %s',
    CASE WHEN passed THEN 'PASS' ELSE 'FAIL' END, name)
FROM test_results
ORDER BY passed, name;

\echo

SELECT pg_catalog.format('Total: %s passed, %s failed out of %s tests',
    count(*) FILTER (WHERE passed),
    count(*) FILTER (WHERE NOT passed),
    count(*))
FROM test_results;

-- Exit with error if any check failed. `passed IS NOT TRUE` also catches a
-- NULL that slipped through (belt-and-suspenders with assert_test's COALESCE).
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM test_results WHERE passed IS NOT TRUE) THEN
        RAISE EXCEPTION 'Some search_path hardening tests failed';
    END IF;
END;
$$;
