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
-- Indexing pgvector-typed columns with mktann
-- =====================================================================
-- Checks that an mktann index over a pgvector-typed column builds, is chosen
-- by the planner, and answers queries correctly.
--
-- The binary casts tested above are what make it reachable: PostgreSQL matches
-- an opclass to a column by binary coercibility, so an opclass declared FOR
-- TYPE mkt.halfvec accepts a public.halfvec column. The access method then has
-- to read that column as f16, and it gets that from the opclass's own type
-- descriptor rather than by identifying the column's type itself -- so a
-- pgvector column and a meerkat column are handled identically, with nothing
-- in the access method comparing type OIDs. These tests pin that down: a
-- descriptor keyed on anything narrower would read the f16 pairs as float32
-- and return garbage from an index the planner considers valid.
--
-- Note the query has to resolve to *meerkat's* distance operator for the
-- index to be considered at all; pgvector's <-> belongs to pgvector's
-- opfamily. That is true of vector columns too and is not specific to
-- halfvec.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS meerkat;
SET search_path = mkt, public;

CREATE TEMP TABLE idx_src (id int, txt text);
INSERT INTO idx_src
    SELECT g, '[' || (SELECT string_agg(
                          round(sin(g * 0.7 + j * 1.3)::numeric, 4)::text, ',')
                      FROM generate_series(1, 8) j) || ']'
    FROM generate_series(1, 2000) g;

CREATE TABLE idx_pgv_h (id int, v public.halfvec(8));
CREATE TABLE idx_mkt_h (id int, v mkt.halfvec(8));
INSERT INTO idx_pgv_h SELECT id, txt::public.halfvec(8) FROM idx_src;
INSERT INTO idx_mkt_h SELECT id, txt::mkt.halfvec(8) FROM idx_src;
ANALYZE idx_pgv_h;
ANALYZE idx_mkt_h;

-- Brute-force truth, established before any index exists.
CREATE TEMP TABLE idx_truth AS
    SELECT array_agg(id ORDER BY id) AS ids
      FROM (SELECT id FROM idx_pgv_h
             ORDER BY v OPERATOR(mkt.<->) '[0,0,0,0,0,0,0,0]'::mkt.halfvec(8)
             LIMIT 10) t;

CREATE INDEX idx_pgv_h_i ON idx_pgv_h USING mktann (v mkt.halfvec_l2_ops);
CREATE INDEX idx_mkt_h_i ON idx_mkt_h USING mktann (v mkt.halfvec_l2_ops);

-- The planner must actually choose the index, or the rest proves nothing.
-- EXPLAIN cannot appear in a subquery, hence the helper.
CREATE OR REPLACE FUNCTION plan_uses_index_scan(q text) RETURNS bool
    LANGUAGE plpgsql AS $fn$
DECLARE
    line text;
BEGIN
    FOR line IN EXECUTE 'EXPLAIN (COSTS OFF) ' || q LOOP
        IF line LIKE '%Index Scan%' THEN
            RETURN true;
        END IF;
    END LOOP;
    RETURN false;
END $fn$;

SET enable_seqscan = off;
SELECT assert_test('mktann index is used on a pgvector halfvec column',
    plan_uses_index_scan($q$SELECT id FROM idx_pgv_h
        ORDER BY v OPERATOR(mkt.<->) '[0,0,0,0,0,0,0,0]'::mkt.halfvec(8)
        LIMIT 10$q$));

-- Recall against brute force. A misdecoded column scores near zero here, so
-- 8 of 10 is a generous floor that still fails hard on a decode bug while
-- tolerating ordinary ANN approximation at the default nprobe.
SELECT assert_test(
    'pgvector halfvec column: mktann recall >= 8/10',
    (SELECT count(*) FROM (
        SELECT id FROM idx_pgv_h
         ORDER BY v OPERATOR(mkt.<->) '[0,0,0,0,0,0,0,0]'::mkt.halfvec(8)
         LIMIT 10) g
      WHERE g.id = ANY (SELECT unnest(ids) FROM idx_truth)) >= 8);

-- Control: meerkat's own halfvec column, same data, same expectation.
SELECT assert_test(
    'mkt halfvec column: mktann recall >= 8/10',
    (SELECT count(*) FROM (
        SELECT id FROM idx_mkt_h
         ORDER BY v OPERATOR(mkt.<->) '[0,0,0,0,0,0,0,0]'::mkt.halfvec(8)
         LIMIT 10) g
      WHERE g.id = ANY (SELECT unnest(ids) FROM idx_truth)) >= 8);

-- Centroids follow the column for a pgvector column too, by the same
-- binary-coercibility test. Compression off, so the format is decided by the
-- column type rather than by RaBitQ.
CREATE INDEX idx_fmt_i ON idx_pgv_h USING mktann (v mkt.halfvec_l2_ops)
    WITH (centroid_compression = off, centroid_fastscan = off,
          fastscan = off, soar_lambda = 0, boundary_epsilon = 0);
SELECT assert_test('pgvector halfvec column gets half-precision centroids',
    (SELECT bool_and(format = 'half')
       FROM mkt.centroid_pages('idx_fmt_i'::regclass)));

RESET enable_seqscan;
DROP FUNCTION plan_uses_index_scan(text);
DROP TABLE idx_pgv_h;
DROP TABLE idx_mkt_h;
RESET search_path;

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
