-- pgvector compatibility tests
--
-- Verifies that meerkat's vector/halfvec types produce the same results as
-- pgvector's, and that data can be exchanged between the two extensions via
-- binary casts (WITHOUT FUNCTION). Also tests cast lifecycle across all
-- install/drop orderings. This is NOT a hard dependency — meerkat works
-- standalone — but users with existing pgvector workflows should get
-- identical behavior.
--
-- Prerequisites:
--   pgvector and meerkat must be installed in PostgreSQL.
--
-- Usage:
--   psql -f test/pg/compat/pgvector.sql

\set ON_ERROR_STOP on
\pset tuples_only on
\pset format unaligned

-- Ensure both extensions are loaded
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS meerkat;

-- Track pass/fail counts
CREATE TEMP TABLE test_results (
    name text,
    passed bool
);

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

-- Helper: count binary casts between pgvector and meerkat
CREATE OR REPLACE FUNCTION count_pgvector_casts() RETURNS int AS $$
    SELECT count(*)::int FROM pg_cast
    WHERE castmethod = 'b'
      AND ((castsource::regtype::text IN ('vector', 'halfvec')
            AND casttarget::regtype::text
                IN ('mkt.vector', 'mkt.halfvec'))
        OR (castsource::regtype::text
                IN ('mkt.vector', 'mkt.halfvec')
            AND casttarget::regtype::text
                IN ('vector', 'halfvec')));
$$ LANGUAGE sql;

-- Helper: check that casts are NOT owned by any extension (standalone)
CREATE OR REPLACE FUNCTION casts_are_standalone() RETURNS bool AS $$
    SELECT NOT EXISTS (
        SELECT 1 FROM pg_depend d
        JOIN pg_cast c ON c.oid = d.objid
        WHERE d.deptype = 'e'
          AND c.castmethod = 'b'
          AND ((c.castsource::regtype::text IN ('vector', 'halfvec')
                AND c.casttarget::regtype::text
                    IN ('mkt.vector', 'mkt.halfvec'))
            OR (c.castsource::regtype::text
                    IN ('mkt.vector', 'mkt.halfvec')
                AND c.casttarget::regtype::text
                    IN ('vector', 'halfvec')))
    );
$$ LANGUAGE sql;

-- Helper: count pgvector ordering operators registered in meerkat's mktann
-- opfamilies (the operators that let the planner match a native pgvector query
-- to a mktann index built on a public.vector column).
CREATE OR REPLACE FUNCTION count_mktann_pgvector_ops() RETURNS int AS $$
    SELECT count(*)::int
    FROM pg_amop amop
    JOIN pg_opfamily f ON f.oid = amop.amopfamily
    JOIN pg_am am ON am.oid = f.opfmethod AND am.amname = 'mktann'
    WHERE amop.amoppurpose = 'o'
      AND amop.amoplefttype = 'public.vector'::regtype;
$$ LANGUAGE sql;

-- Helper: total ORDER BY operators in mktann families (does not reference
-- public.vector, so it is safe to call after pgvector has been dropped).
CREATE OR REPLACE FUNCTION count_mktann_all_orderops() RETURNS int AS $$
    SELECT count(*)::int
    FROM pg_amop amop
    JOIN pg_opfamily f ON f.oid = amop.amopfamily
    JOIN pg_am am ON am.oid = f.opfmethod AND am.amname = 'mktann'
    WHERE amop.amoppurpose = 'o';
$$ LANGUAGE sql;

-- Helper: does the plan for `qry` use an index scan on `idx`? Forces
-- enable_seqscan off so this tests that the ordering path is GENERATED
-- (the fix), independent of cost tuning.
CREATE OR REPLACE FUNCTION plan_uses_index(qry text, idx text)
RETURNS bool AS $$
DECLARE
    line text;
    hit  bool := false;
BEGIN
    SET LOCAL enable_seqscan = off;
    FOR line IN EXECUTE 'EXPLAIN (COSTS OFF) ' || qry LOOP
        IF line ILIKE '%Index Scan using ' || idx || '%' THEN
            hit := true;
        END IF;
    END LOOP;
    RETURN hit;
END;
$$ LANGUAGE plpgsql;

-- =====================================================================
-- 1. Binary casts exist
-- =====================================================================

SELECT assert_test('cast public.vector -> mkt.vector exists',
    EXISTS (SELECT 1 FROM pg_cast
            WHERE castsource = 'public.vector'::regtype
              AND casttarget = 'mkt.vector'::regtype
              AND castmethod = 'b'));

SELECT assert_test('cast public.halfvec -> mkt.halfvec exists',
    EXISTS (SELECT 1 FROM pg_cast
            WHERE castsource = 'public.halfvec'::regtype
              AND casttarget = 'mkt.halfvec'::regtype
              AND castmethod = 'b'));

SELECT assert_test('cast mkt.vector -> public.vector exists',
    EXISTS (SELECT 1 FROM pg_cast
            WHERE castsource = 'mkt.vector'::regtype
              AND casttarget = 'public.vector'::regtype
              AND castmethod = 'b'));

SELECT assert_test('cast mkt.halfvec -> public.halfvec exists',
    EXISTS (SELECT 1 FROM pg_cast
            WHERE castsource = 'mkt.halfvec'::regtype
              AND casttarget = 'public.halfvec'::regtype
              AND castmethod = 'b'));

SELECT assert_test('4 binary casts total',
    count_pgvector_casts() = 4);

SELECT assert_test('casts are standalone (not owned by extension)',
    casts_are_standalone());

-- =====================================================================
-- 2. Binary compatibility: same text representation
-- =====================================================================

SELECT assert_test('vector text I/O matches',
    '[1,2,3]'::mkt.vector::text = '[1,2,3]'::public.vector::text);

SELECT assert_test('halfvec text I/O matches',
    '[1,2,3]'::mkt.halfvec::text = '[1,2,3]'::public.halfvec::text);

SELECT assert_test('vector float precision matches',
    '[1.5,-2.3,0]'::mkt.vector::text
    = '[1.5,-2.3,0]'::public.vector::text);

-- =====================================================================
-- 3. Direct binary casts work
-- =====================================================================

SELECT assert_test('vector binary cast round-trip',
    '[1,2,3]'::public.vector::mkt.vector::text = '[1,2,3]');

SELECT assert_test('halfvec binary cast round-trip',
    '[1,2,3]'::public.halfvec::mkt.halfvec::text = '[1,2,3]');

SELECT assert_test('mkt vector -> pgvector cast',
    '[4,5,6]'::mkt.vector::public.vector::text = '[4,5,6]');

SELECT assert_test('mkt halfvec -> pgvector cast',
    '[4,5,6]'::mkt.halfvec::public.halfvec::text = '[4,5,6]');

-- =====================================================================
-- 4. Distance functions produce same results
-- =====================================================================

-- L2 distance
SELECT assert_test('L2 distance matches',
    abs(mkt.l2_distance('[1,2,3]'::mkt.vector,
                        '[4,5,6]'::mkt.vector)
      - public.l2_distance('[1,2,3]'::public.vector,
                           '[4,5,6]'::public.vector))
    < 1e-6);

-- Inner product
SELECT assert_test('inner product matches',
    abs(mkt.inner_product('[1,2,3]'::mkt.vector,
                          '[4,5,6]'::mkt.vector)
      - public.inner_product('[1,2,3]'::public.vector,
                             '[4,5,6]'::public.vector))
    < 1e-6);

-- Cosine distance
SELECT assert_test('cosine distance matches',
    abs(mkt.cosine_distance('[1,0]'::mkt.vector,
                            '[0,1]'::mkt.vector)
      - public.cosine_distance('[1,0]'::public.vector,
                               '[0,1]'::public.vector))
    < 1e-6);

-- Cosine distance: identical vectors = 0
SELECT assert_test('cosine identical = 0',
    abs(mkt.cosine_distance('[1,1,1]'::mkt.vector,
                            '[1,1,1]'::mkt.vector)
      - public.cosine_distance('[1,1,1]'::public.vector,
                               '[1,1,1]'::public.vector))
    < 1e-6);

-- =====================================================================
-- 5. Operators produce same results
-- =====================================================================

-- <-> (L2 distance)
SELECT assert_test('<-> operator matches',
    abs(('[0,0]'::mkt.vector
         OPERATOR(mkt.<->) '[3,4]'::mkt.vector)
      - ('[0,0]'::public.vector
         OPERATOR(public.<->) '[3,4]'::public.vector))
    < 1e-6);

-- <#> (negative inner product)
SELECT assert_test('<#> operator matches',
    abs(('[1,2,3]'::mkt.vector
         OPERATOR(mkt.<#>) '[4,5,6]'::mkt.vector)
      - ('[1,2,3]'::public.vector
         OPERATOR(public.<#>) '[4,5,6]'::public.vector))
    < 1e-6);

-- <=> (cosine distance)
SELECT assert_test('<=> operator matches',
    abs(('[1,0]'::mkt.vector
         OPERATOR(mkt.<=>) '[0,1]'::mkt.vector)
      - ('[1,0]'::public.vector
         OPERATOR(public.<=>) '[0,1]'::public.vector))
    < 1e-6);

-- =====================================================================
-- 6. Halfvec distances match
-- =====================================================================

SELECT assert_test('halfvec L2 distance matches',
    abs(mkt.l2_distance('[1,2,3]'::mkt.halfvec,
                        '[4,5,6]'::mkt.halfvec)
      - public.l2_distance('[1,2,3]'::public.halfvec,
                           '[4,5,6]'::public.halfvec))
    < 1e-2);  -- halfvec has lower precision

SELECT assert_test('halfvec inner product matches',
    abs(mkt.inner_product('[1,2,3]'::mkt.halfvec,
                          '[4,5,6]'::mkt.halfvec)
      - public.inner_product('[1,2,3]'::public.halfvec,
                             '[4,5,6]'::public.halfvec))
    < 1e-2);

SELECT assert_test('halfvec cosine distance matches',
    abs(mkt.cosine_distance('[1,0]'::mkt.halfvec,
                            '[0,1]'::mkt.halfvec)
      - public.cosine_distance('[1,0]'::public.halfvec,
                               '[0,1]'::public.halfvec))
    < 1e-2);

-- =====================================================================
-- 7. Cross-type: meerkat ops on pgvector data (via implicit cast)
-- =====================================================================

-- Create a table with pgvector's vector type
CREATE TEMP TABLE pgv_items (
    id serial PRIMARY KEY,
    v public.vector(3)
);
INSERT INTO pgv_items (v) VALUES
    ('[1,0,0]'), ('[0,1,0]'), ('[0,0,1]'),
    ('[1,1,0]'), ('[0,1,1]'), ('[1,0,1]');

-- Meerkat distance on pgvector column via binary cast
SELECT assert_test('mkt distance on pgvector column via cast',
    (SELECT mkt.l2_distance(v::mkt.vector,
                            '[1,1,1]'::mkt.vector)
     FROM pgv_items WHERE id = 1) IS NOT NULL);

-- Implicit cast: pgvector column used directly with meerkat operator
SELECT assert_test('implicit cast with mkt operator',
    (SELECT v OPERATOR(mkt.<->) '[1,1,1]'::mkt.vector
     FROM pgv_items WHERE id = 1) IS NOT NULL);

-- ORDER BY with meerkat operators on pgvector data (implicit cast)
SELECT assert_test('mkt ORDER BY on pgvector data via cast',
    (SELECT array_agg(id ORDER BY
        v OPERATOR(mkt.<->) '[1,1,1]'::mkt.vector)
     FROM pgv_items) IS NOT NULL);

-- =====================================================================
-- 8. Cross-type: pgvector ops on meerkat data (via explicit cast)
-- =====================================================================

CREATE TEMP TABLE mkt_items (
    id serial PRIMARY KEY,
    v mkt.vector(3)
);
INSERT INTO mkt_items (v) VALUES
    ('[1,0,0]'), ('[0,1,0]'), ('[0,0,1]'),
    ('[1,1,0]'), ('[0,1,1]'), ('[1,0,1]');

-- pgvector distance on meerkat column via explicit cast
SELECT assert_test('pgvector distance on mkt column via cast',
    (SELECT public.l2_distance(v::public.vector,
                               '[1,1,1]'::public.vector)
     FROM mkt_items WHERE id = 1) IS NOT NULL);

-- =====================================================================
-- 9. NN ordering matches between implementations
-- =====================================================================

-- Same query on both types should return same ordering
SELECT assert_test('NN ordering matches (vector)',
    (SELECT array_agg(id ORDER BY
        v OPERATOR(mkt.<->) '[1,1,1]'::mkt.vector)
     FROM pgv_items)
    =
    (SELECT array_agg(id ORDER BY
        v OPERATOR(public.<->) '[1,1,1]'::public.vector)
     FROM pgv_items));

-- =====================================================================
-- 10. Array round-trip compatibility
-- =====================================================================

SELECT assert_test('vector -> array -> vector round-trip',
    ('[1,2,3]'::mkt.vector)::real[]::mkt.vector::text = '[1,2,3]');

SELECT assert_test('pgvector array cast matches mkt array cast',
    ('[1,2,3]'::mkt.vector)::real[]::text
    = ('[1,2,3]'::public.vector)::real[]::text);

-- =====================================================================
-- 11. Dimension and norm utilities match
-- =====================================================================

SELECT assert_test('vector_dims matches',
    mkt.vector_dims('[1,2,3]'::mkt.vector)
    = public.vector_dims('[1,2,3]'::public.vector));

SELECT assert_test('vector_norm matches',
    abs(mkt.vector_norm('[3,4]'::mkt.vector)
      - public.vector_norm('[3,4]'::public.vector))
    < 1e-6);

-- =====================================================================
-- 12. Mixed-type distance via implicit cast
-- =====================================================================

-- pgvector value as LHS with meerkat function (implicit cast)
SELECT assert_test('mkt.l2_distance with pgvector arg (implicit)',
    abs(mkt.l2_distance('[1,2,3]'::public.vector,
                        '[4,5,6]'::mkt.vector)
      - public.l2_distance('[1,2,3]'::public.vector,
                           '[4,5,6]'::public.vector))
    < 1e-6);

SELECT assert_test('mkt.cosine_distance with pgvector arg (implicit)',
    abs(mkt.cosine_distance('[1,0]'::public.vector,
                            '[0,1]'::mkt.vector)
      - public.cosine_distance('[1,0]'::public.vector,
                               '[0,1]'::public.vector))
    < 1e-6);

-- =====================================================================
-- 12b. Planner picks the mktann index for NATIVE pgvector queries
-- =====================================================================
--
-- The point of registering pgvector's ordering operators in meerkat's
-- opfamilies: an mktann index built on a pgvector (public.vector) column must
-- be usable by an idiomatic `col <-> $q` query that resolves to pgvector's own
-- operator. Binary casts alone do NOT achieve this -- without the operators
-- the planner never generates the ordering path and silently seq-scans.

SELECT assert_test('3 pgvector ops registered in mktann families',
    count_mktann_pgvector_ops() = 3);

CREATE TABLE compat_items (id int, v public.vector(16));
INSERT INTO compat_items (id, v)
SELECT g, (array_agg(random() ORDER BY d))::real[]::public.vector
FROM generate_series(1, 2000) g CROSS JOIN generate_series(1, 16) d
GROUP BY g;
CREATE INDEX compat_items_mkt ON compat_items USING mktann (v);
ANALYZE compat_items;

SELECT assert_test('native <-> query uses mktann index',
    plan_uses_index(
        'SELECT id FROM compat_items ORDER BY v <-> ''['
        || (SELECT string_agg('0.5', ',') FROM generate_series(1, 16))
        || ']''::public.vector LIMIT 5',
        'compat_items_mkt'));

-- Results are correct (the self-match is returned as the nearest neighbor).
SELECT assert_test('native query returns correct nearest neighbor',
    (SELECT id FROM compat_items
     ORDER BY v <-> (SELECT v FROM compat_items WHERE id = 1)
     LIMIT 1) = 1);

DROP TABLE compat_items;

-- =====================================================================
-- Cast lifecycle tests
-- =====================================================================
--
-- Tests install/drop orderings to verify casts are auto-created and
-- auto-cleaned via type dependencies.

-- =====================================================================
-- 13. Install path: pgvector first, then meerkat (DO block)
-- =====================================================================

DROP EXTENSION IF EXISTS meerkat CASCADE;
DROP EXTENSION IF EXISTS vector CASCADE;

CREATE EXTENSION vector;
CREATE EXTENSION meerkat;

SELECT assert_test('pgv-first: 4 binary casts',
    count_pgvector_casts() = 4);

SELECT assert_test('pgv-first: 3 mktann pgvector ops',
    count_mktann_pgvector_ops() = 3);

SELECT assert_test('pgv-first: casts are standalone',
    casts_are_standalone());

SELECT assert_test('pgv-first: pgvector->mkt cast works',
    '[1,2,3]'::public.vector::mkt.vector::text = '[1,2,3]');

SELECT assert_test('pgv-first: mkt->pgvector cast works',
    '[1,2,3]'::mkt.vector::public.vector::text = '[1,2,3]');

-- =====================================================================
-- 14. Install path: meerkat first, then pgvector (event trigger)
-- =====================================================================

DROP EXTENSION meerkat CASCADE;
DROP EXTENSION vector;

CREATE EXTENSION meerkat;
CREATE EXTENSION vector;

SELECT assert_test('mkt-first: 4 binary casts',
    count_pgvector_casts() = 4);

SELECT assert_test('mkt-first: 3 mktann pgvector ops',
    count_mktann_pgvector_ops() = 3);

SELECT assert_test('mkt-first: casts are standalone',
    casts_are_standalone());

SELECT assert_test('mkt-first: pgvector->mkt cast works',
    '[1,2,3]'::public.vector::mkt.vector::text = '[1,2,3]');

SELECT assert_test('mkt-first: mkt->pgvector cast works',
    '[1,2,3]'::mkt.vector::public.vector::text = '[1,2,3]');

-- =====================================================================
-- 15. DROP meerkat CASCADE: casts dropped, pgvector survives
-- =====================================================================

DROP EXTENSION meerkat CASCADE;

SELECT assert_test('drop-mkt: no casts remain',
    count_pgvector_casts() = 0);

SELECT assert_test('drop-mkt: pgvector still loaded',
    EXISTS (SELECT 1 FROM pg_extension
            WHERE extname = 'vector'));

SELECT assert_test('drop-mkt: pgvector still works',
    '[1,2,3]'::public.vector::text = '[1,2,3]');

-- =====================================================================
-- 16. DROP pgvector CASCADE: casts dropped, meerkat survives
-- =====================================================================

-- Restore both (pgvector-first path)
CREATE EXTENSION meerkat;

DROP EXTENSION vector CASCADE;

SELECT assert_test('drop-pgv: meerkat still loaded',
    EXISTS (SELECT 1 FROM pg_extension
            WHERE extname = 'meerkat'));

-- The pgvector ordering operators depend on pgvector's operators, so dropping
-- pgvector removes them from meerkat's families, leaving only the 3 native
-- (mkt.vector) ordering operators. meerkat itself is unaffected.
SELECT assert_test('drop-pgv: pgvector ops removed from mktann families',
    count_mktann_all_orderops() = 3);

SELECT assert_test('drop-pgv: mkt.vector still works',
    '[1,2,3]'::mkt.vector::text = '[1,2,3]');

SELECT assert_test('drop-pgv: event trigger still exists',
    EXISTS (SELECT 1 FROM pg_event_trigger
            WHERE evtname = 'mkt_pgvector_cast_trigger'));

-- =====================================================================
-- 17. Re-create pgvector: event trigger re-creates casts
-- =====================================================================

CREATE EXTENSION vector;

SELECT assert_test('recreate-pgv: 4 casts re-created',
    count_pgvector_casts() = 4);

SELECT assert_test('recreate-pgv: 3 mktann pgvector ops re-created',
    count_mktann_pgvector_ops() = 3);

SELECT assert_test('recreate-pgv: casts are standalone',
    casts_are_standalone());

SELECT assert_test('recreate-pgv: cast works',
    '[1,2,3]'::public.vector::mkt.vector::text = '[1,2,3]');

-- =====================================================================
-- 18. meerkat standalone (no pgvector)
-- =====================================================================

DROP EXTENSION meerkat CASCADE;
DROP EXTENSION vector;

CREATE EXTENSION meerkat;

SELECT assert_test('standalone: meerkat loads without pgvector',
    EXISTS (SELECT 1 FROM pg_extension
            WHERE extname = 'meerkat'));

SELECT assert_test('standalone: mkt.vector works',
    '[1,2,3]'::mkt.vector::text = '[1,2,3]');

SELECT assert_test('standalone: event trigger exists',
    EXISTS (SELECT 1 FROM pg_event_trigger
            WHERE evtname = 'mkt_pgvector_cast_trigger'));

DROP EXTENSION meerkat;

SELECT assert_test('standalone: clean drop',
    NOT EXISTS (SELECT 1 FROM pg_extension
                WHERE extname = 'meerkat'));

-- =====================================================================
-- Summary
-- =====================================================================

\echo
\echo ====================================
\echo pgvector compatibility test results
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

-- Exit with error if any test failed
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM test_results WHERE NOT passed) THEN
        RAISE EXCEPTION 'Some compatibility tests failed';
    END IF;
END;
$$;
