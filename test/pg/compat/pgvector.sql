-- pgvector compatibility tests
--
-- Verifies that meerkat's vector/halfvec types produce the same results as
-- pgvector's, and that data can be exchanged between the two extensions via
-- text casts. This is NOT a hard dependency — meerkat works standalone — but
-- users with existing pgvector workflows should get identical behavior.
--
-- Prerequisites:
--   CREATE EXTENSION vector;   -- pgvector (installed to public schema)
--   CREATE EXTENSION meerkat;  -- meerkat  (installed to mkt schema)
--
-- Usage:
--   psql -f test/pg/compat/pgvector.sql

\set ON_ERROR_STOP on
\pset tuples_only on
\pset format unaligned

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

-- =====================================================================
-- 1. Binary compatibility: same text representation
-- =====================================================================

SELECT assert_test('vector text I/O matches',
    '[1,2,3]'::mkt.vector::text = '[1,2,3]'::public.vector::text);

SELECT assert_test('halfvec text I/O matches',
    '[1,2,3]'::mkt.halfvec::text = '[1,2,3]'::public.halfvec::text);

SELECT assert_test('vector float precision matches',
    '[1.5,-2.3,0]'::mkt.vector::text = '[1.5,-2.3,0]'::public.vector::text);

-- =====================================================================
-- 2. Distance functions produce same results
-- =====================================================================

-- L2 distance
SELECT assert_test('L2 distance matches',
    abs(mkt.l2_distance('[1,2,3]'::mkt.vector, '[4,5,6]'::mkt.vector)
      - public.l2_distance('[1,2,3]'::public.vector, '[4,5,6]'::public.vector))
    < 1e-6);

-- Inner product
SELECT assert_test('inner product matches',
    abs(mkt.inner_product('[1,2,3]'::mkt.vector, '[4,5,6]'::mkt.vector)
      - public.inner_product('[1,2,3]'::public.vector, '[4,5,6]'::public.vector))
    < 1e-6);

-- Cosine distance
SELECT assert_test('cosine distance matches',
    abs(mkt.cosine_distance('[1,0]'::mkt.vector, '[0,1]'::mkt.vector)
      - public.cosine_distance('[1,0]'::public.vector, '[0,1]'::public.vector))
    < 1e-6);

-- Cosine distance: identical vectors = 0
SELECT assert_test('cosine identical = 0',
    abs(mkt.cosine_distance('[1,1,1]'::mkt.vector, '[1,1,1]'::mkt.vector)
      - public.cosine_distance('[1,1,1]'::public.vector, '[1,1,1]'::public.vector))
    < 1e-6);

-- =====================================================================
-- 3. Operators produce same results
-- =====================================================================

-- <-> (L2 distance)
SELECT assert_test('<-> operator matches',
    abs(('[0,0]'::mkt.vector OPERATOR(mkt.<->) '[3,4]'::mkt.vector)
      - ('[0,0]'::public.vector OPERATOR(public.<->) '[3,4]'::public.vector))
    < 1e-6);

-- <#> (negative inner product)
SELECT assert_test('<#> operator matches',
    abs(('[1,2,3]'::mkt.vector OPERATOR(mkt.<#>) '[4,5,6]'::mkt.vector)
      - ('[1,2,3]'::public.vector OPERATOR(public.<#>) '[4,5,6]'::public.vector))
    < 1e-6);

-- <=> (cosine distance)
SELECT assert_test('<=> operator matches',
    abs(('[1,0]'::mkt.vector OPERATOR(mkt.<=>) '[0,1]'::mkt.vector)
      - ('[1,0]'::public.vector OPERATOR(public.<=>) '[0,1]'::public.vector))
    < 1e-6);

-- =====================================================================
-- 4. Halfvec distances match
-- =====================================================================

SELECT assert_test('halfvec L2 distance matches',
    abs(mkt.l2_distance('[1,2,3]'::mkt.halfvec, '[4,5,6]'::mkt.halfvec)
      - public.l2_distance('[1,2,3]'::public.halfvec, '[4,5,6]'::public.halfvec))
    < 1e-2);  -- halfvec has lower precision

SELECT assert_test('halfvec inner product matches',
    abs(mkt.inner_product('[1,2,3]'::mkt.halfvec, '[4,5,6]'::mkt.halfvec)
      - public.inner_product('[1,2,3]'::public.halfvec, '[4,5,6]'::public.halfvec))
    < 1e-2);

SELECT assert_test('halfvec cosine distance matches',
    abs(mkt.cosine_distance('[1,0]'::mkt.halfvec, '[0,1]'::mkt.halfvec)
      - public.cosine_distance('[1,0]'::public.halfvec, '[0,1]'::public.halfvec))
    < 1e-2);

-- =====================================================================
-- 5. Cross-type queries: meerkat functions on pgvector data
-- =====================================================================

-- Create a table with pgvector's vector type
CREATE TEMP TABLE pgv_items (
    id serial PRIMARY KEY,
    v public.vector(3)
);
INSERT INTO pgv_items (v) VALUES
    ('[1,0,0]'), ('[0,1,0]'), ('[0,0,1]'),
    ('[1,1,0]'), ('[0,1,1]'), ('[1,0,1]');

-- Query pgvector table using meerkat's operators via explicit cast
-- This tests that meerkat can work with data stored as pgvector types
SELECT assert_test('mkt distance on pgvector column via cast',
    (SELECT mkt.l2_distance(v::text::mkt.vector, '[1,1,1]'::mkt.vector)
     FROM pgv_items WHERE id = 1) IS NOT NULL);

-- ORDER BY with meerkat operators on casted pgvector data
SELECT assert_test('mkt ORDER BY on pgvector data via cast',
    (SELECT array_agg(id ORDER BY
        v::text::mkt.vector OPERATOR(mkt.<->) '[1,1,1]'::mkt.vector)
     FROM pgv_items) IS NOT NULL);

-- =====================================================================
-- 6. Cross-type queries: pgvector functions on meerkat data
-- =====================================================================

CREATE TEMP TABLE mkt_items (
    id serial PRIMARY KEY,
    v mkt.vector(3)
);
INSERT INTO mkt_items (v) VALUES
    ('[1,0,0]'), ('[0,1,0]'), ('[0,0,1]'),
    ('[1,1,0]'), ('[0,1,1]'), ('[1,0,1]');

-- pgvector distance on meerkat column via text cast
SELECT assert_test('pgvector distance on mkt column via cast',
    (SELECT public.l2_distance(v::text::public.vector,
        '[1,1,1]'::public.vector)
     FROM mkt_items WHERE id = 1) IS NOT NULL);

-- =====================================================================
-- 7. Nearest-neighbor results match between implementations
-- =====================================================================

-- Same query on both types should return same ordering
SELECT assert_test('NN ordering matches (vector)',
    (SELECT array_agg(id ORDER BY
        v::text::mkt.vector OPERATOR(mkt.<->) '[1,1,1]'::mkt.vector)
     FROM pgv_items)
    =
    (SELECT array_agg(id ORDER BY
        v OPERATOR(public.<->) '[1,1,1]'::public.vector)
     FROM pgv_items));

-- =====================================================================
-- 8. Array round-trip compatibility
-- =====================================================================

SELECT assert_test('vector -> array -> vector round-trip',
    ('[1,2,3]'::mkt.vector)::real[]::mkt.vector::text = '[1,2,3]');

SELECT assert_test('pgvector array cast matches mkt array cast',
    ('[1,2,3]'::mkt.vector)::real[]::text
    = ('[1,2,3]'::public.vector)::real[]::text);

-- =====================================================================
-- 9. Dimension and norm utilities match
-- =====================================================================

SELECT assert_test('vector_dims matches',
    mkt.vector_dims('[1,2,3]'::mkt.vector)
    = public.vector_dims('[1,2,3]'::public.vector));

SELECT assert_test('vector_norm matches',
    abs(mkt.vector_norm('[3,4]'::mkt.vector)
      - public.vector_norm('[3,4]'::public.vector))
    < 1e-6);

-- =====================================================================
-- Summary
-- =====================================================================

\echo
\echo ====================================
\echo pgvector compatibility test results
\echo ====================================

SELECT format('%s: %s', CASE WHEN passed THEN 'PASS' ELSE 'FAIL' END, name)
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
