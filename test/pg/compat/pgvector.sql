-- pgvector compatibility tests
--
-- Verifies that pg_vectorsearch's vector/halfvec types produce the same
-- results as pgvector's, and that data can be exchanged between the two
-- extensions via binary casts (WITHOUT FUNCTION). Also tests cast
-- lifecycle across all install/drop orderings. This is NOT a hard
-- dependency — pg_vectorsearch works standalone — but users with existing
-- pgvector workflows should get identical behavior.
--
-- Prerequisites:
--   pgvector and pg_vectorsearch must be installed in PostgreSQL. This
--   suite installs and drops both extensions (and a scratch schema) as it
--   exercises the install/drop orderings, so run it against a
--   throwaway/clean database, not one holding data you care about. The CI
--   script uses a fresh instance.
--
-- Usage:
--   psql -f test/pg/compat/pgvector.sql

\set ON_ERROR_STOP on
\pset tuples_only on
\pset format unaligned

-- Ensure both extensions are loaded
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_vectorsearch;

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

-- Helper: count binary casts between pgvector and pg_vectorsearch. Compares
-- regtype::text output, which format_type renders unqualified whenever the
-- type's home schema is on the caller's search_path -- true here for both
-- sides, since pg_vectorsearch's default (no SCHEMA clause) install and
-- pgvector's both land in public, which is on the default search_path.
CREATE OR REPLACE FUNCTION count_pgvector_casts() RETURNS int AS $$
    SELECT count(*)::int FROM pg_cast
    WHERE castmethod = 'b'
      AND ((castsource::regtype::text IN ('vector', 'halfvec')
            AND casttarget::regtype::text IN ('vec32', 'vec16'))
        OR (castsource::regtype::text IN ('vec32', 'vec16')
            AND casttarget::regtype::text IN ('vector', 'halfvec')));
$$ LANGUAGE sql;

-- Helper: check that casts are NOT owned by any extension (standalone)
CREATE OR REPLACE FUNCTION casts_are_standalone() RETURNS bool AS $$
    SELECT NOT EXISTS (
        SELECT 1 FROM pg_depend d
        JOIN pg_cast c ON c.oid = d.objid
        WHERE d.deptype = 'e'
          AND c.castmethod = 'b'
          AND ((c.castsource::regtype::text IN ('vector', 'halfvec')
                AND c.casttarget::regtype::text IN ('vec32', 'vec16'))
            OR (c.castsource::regtype::text IN ('vec32', 'vec16')
                AND c.casttarget::regtype::text IN ('vector', 'halfvec')))
    );
$$ LANGUAGE sql;

-- Helper: pgvector's distance operators registered as ordering members of
-- pg_vectorsearch's prism families. pg_vectorsearch's types now install
-- alongside pgvector's own (both default to public), so namespace alone
-- no longer distinguishes the two -- identify pgvector's operators by
-- their left argument type (vector/halfvec) instead. to_regtype (not
-- ::regtype) so a literal cast never errors when pgvector has been
-- dropped -- it returns NULL instead, and the joins simply match nothing,
-- returning 0.
CREATE OR REPLACE FUNCTION count_pgvector_prism_ops() RETURNS int AS $$
    SELECT count(*)::int
    FROM pg_amop ao
    JOIN pg_opfamily f ON f.oid = ao.amopfamily
    JOIN pg_am am ON am.oid = f.opfmethod AND am.amname = 'prism'
    JOIN pg_operator op ON op.oid = ao.amopopr
    WHERE ao.amoppurpose = 'o'
      AND op.oprleft = ANY (ARRAY[to_regtype('vector'), to_regtype('halfvec')]);
$$ LANGUAGE sql;

-- Helper: prism's own ordering operators (on vec32/vec16), so a test can
-- assert pg_vectorsearch's opclasses survive a pgvector drop untouched.
CREATE OR REPLACE FUNCTION count_pgvs_prism_ops() RETURNS int AS $$
    SELECT count(*)::int
    FROM pg_amop ao
    JOIN pg_opfamily f ON f.oid = ao.amopfamily
    JOIN pg_am am ON am.oid = f.opfmethod AND am.amname = 'prism'
    JOIN pg_operator op ON op.oid = ao.amopopr
    WHERE ao.amoppurpose = 'o'
      AND op.oprleft = ANY (ARRAY[to_regtype('vec32'), to_regtype('vec16')]);
$$ LANGUAGE sql;

-- Helper: true if `cmd` raises (is refused), rolling back its own subxact.
-- Used to assert that a bare DROP EXTENSION is blocked by the binary casts, so
-- that removing either extension needs CASCADE and cannot silently succeed.
CREATE OR REPLACE FUNCTION stmt_is_refused(cmd text) RETURNS bool AS $$
BEGIN
    EXECUTE cmd;
    RETURN false;
EXCEPTION WHEN OTHERS THEN
    RETURN true;
END;
$$ LANGUAGE plpgsql;

-- =====================================================================
-- 1. Binary casts exist
-- =====================================================================

SELECT assert_test('cast public.vector -> public.vec32 exists',
    EXISTS (SELECT 1 FROM pg_cast
            WHERE castsource = 'public.vector'::regtype
              AND casttarget = 'public.vec32'::regtype
              AND castmethod = 'b'));

SELECT assert_test('cast public.halfvec -> public.vec16 exists',
    EXISTS (SELECT 1 FROM pg_cast
            WHERE castsource = 'public.halfvec'::regtype
              AND casttarget = 'public.vec16'::regtype
              AND castmethod = 'b'));

SELECT assert_test('cast public.vec32 -> public.vector exists',
    EXISTS (SELECT 1 FROM pg_cast
            WHERE castsource = 'public.vec32'::regtype
              AND casttarget = 'public.vector'::regtype
              AND castmethod = 'b'));

SELECT assert_test('cast public.vec16 -> public.halfvec exists',
    EXISTS (SELECT 1 FROM pg_cast
            WHERE castsource = 'public.vec16'::regtype
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
    '[1,2,3]'::public.vec32::text = '[1,2,3]'::public.vector::text);

SELECT assert_test('halfvec text I/O matches',
    '[1,2,3]'::public.vec16::text = '[1,2,3]'::public.halfvec::text);

SELECT assert_test('vector float precision matches',
    '[1.5,-2.3,0]'::public.vec32::text
    = '[1.5,-2.3,0]'::public.vector::text);

-- =====================================================================
-- 3. Direct binary casts work
-- =====================================================================

SELECT assert_test('vector binary cast round-trip',
    '[1,2,3]'::public.vector::public.vec32::text = '[1,2,3]');

SELECT assert_test('halfvec binary cast round-trip',
    '[1,2,3]'::public.halfvec::public.vec16::text = '[1,2,3]');

SELECT assert_test('pgvs vector -> pgvector cast',
    '[4,5,6]'::public.vec32::public.vector::text = '[4,5,6]');

SELECT assert_test('pgvs halfvec -> pgvector cast',
    '[4,5,6]'::public.vec16::public.halfvec::text = '[4,5,6]');

-- =====================================================================
-- 4. Distance functions produce same results
-- =====================================================================

-- L2 distance
SELECT assert_test('L2 distance matches',
    abs(public.l2_distance('[1,2,3]'::public.vec32,
                        '[4,5,6]'::public.vec32)
      - public.l2_distance('[1,2,3]'::public.vector,
                           '[4,5,6]'::public.vector))
    < 1e-6);

-- Inner product
SELECT assert_test('inner product matches',
    abs(public.inner_product('[1,2,3]'::public.vec32,
                          '[4,5,6]'::public.vec32)
      - public.inner_product('[1,2,3]'::public.vector,
                             '[4,5,6]'::public.vector))
    < 1e-6);

-- Cosine distance
SELECT assert_test('cosine distance matches',
    abs(public.cosine_distance('[1,0]'::public.vec32,
                            '[0,1]'::public.vec32)
      - public.cosine_distance('[1,0]'::public.vector,
                               '[0,1]'::public.vector))
    < 1e-6);

-- Cosine distance: identical vectors = 0
SELECT assert_test('cosine identical = 0',
    abs(public.cosine_distance('[1,1,1]'::public.vec32,
                            '[1,1,1]'::public.vec32)
      - public.cosine_distance('[1,1,1]'::public.vector,
                               '[1,1,1]'::public.vector))
    < 1e-6);

-- =====================================================================
-- 5. Operators produce same results
-- =====================================================================

-- <-> (L2 distance)
SELECT assert_test('<-> operator matches',
    abs(('[0,0]'::public.vec32
         OPERATOR(public.<->) '[3,4]'::public.vec32)
      - ('[0,0]'::public.vector
         OPERATOR(public.<->) '[3,4]'::public.vector))
    < 1e-6);

-- <#> (negative inner product)
SELECT assert_test('<#> operator matches',
    abs(('[1,2,3]'::public.vec32
         OPERATOR(public.<#>) '[4,5,6]'::public.vec32)
      - ('[1,2,3]'::public.vector
         OPERATOR(public.<#>) '[4,5,6]'::public.vector))
    < 1e-6);

-- <=> (cosine distance)
SELECT assert_test('<=> operator matches',
    abs(('[1,0]'::public.vec32
         OPERATOR(public.<=>) '[0,1]'::public.vec32)
      - ('[1,0]'::public.vector
         OPERATOR(public.<=>) '[0,1]'::public.vector))
    < 1e-6);

-- =====================================================================
-- 6. Halfvec distances match
-- =====================================================================

SELECT assert_test('halfvec L2 distance matches',
    abs(public.l2_distance('[1,2,3]'::public.vec16,
                        '[4,5,6]'::public.vec16)
      - public.l2_distance('[1,2,3]'::public.halfvec,
                           '[4,5,6]'::public.halfvec))
    < 1e-2);  -- halfvec has lower precision

SELECT assert_test('halfvec inner product matches',
    abs(public.inner_product('[1,2,3]'::public.vec16,
                          '[4,5,6]'::public.vec16)
      - public.inner_product('[1,2,3]'::public.halfvec,
                             '[4,5,6]'::public.halfvec))
    < 1e-2);

SELECT assert_test('halfvec cosine distance matches',
    abs(public.cosine_distance('[1,0]'::public.vec16,
                            '[0,1]'::public.vec16)
      - public.cosine_distance('[1,0]'::public.halfvec,
                               '[0,1]'::public.halfvec))
    < 1e-2);

-- =====================================================================
-- 7. Cross-type: pg_vectorsearch ops on pgvector data (via implicit cast)
-- =====================================================================

-- Create a table with pgvector's vector type
CREATE TEMP TABLE pgv_items (
    id serial PRIMARY KEY,
    v public.vector(3)
);
INSERT INTO pgv_items (v) VALUES
    ('[1,0,0]'), ('[0,1,0]'), ('[0,0,1]'),
    ('[1,1,0]'), ('[0,1,1]'), ('[1,0,1]');

-- pg_vectorsearch distance on pgvector column via binary cast
SELECT assert_test('pgvs distance on pgvector column via cast',
    (SELECT public.l2_distance(v::public.vec32,
                            '[1,1,1]'::public.vec32)
     FROM pgv_items WHERE id = 1) IS NOT NULL);

-- Implicit cast: pgvector column used directly with pg_vectorsearch operator
SELECT assert_test('implicit cast with pgvs operator',
    (SELECT v OPERATOR(public.<->) '[1,1,1]'::public.vec32
     FROM pgv_items WHERE id = 1) IS NOT NULL);

-- ORDER BY with pg_vectorsearch operators on pgvector data (implicit cast)
SELECT assert_test('pgvs ORDER BY on pgvector data via cast',
    (SELECT array_agg(id ORDER BY
        v OPERATOR(public.<->) '[1,1,1]'::public.vec32)
     FROM pgv_items) IS NOT NULL);

-- =====================================================================
-- 8. Cross-type: pgvector ops on pg_vectorsearch data (via explicit cast)
-- =====================================================================

CREATE TEMP TABLE pgvs_items (
    id serial PRIMARY KEY,
    v public.vec32(3)
);
INSERT INTO pgvs_items (v) VALUES
    ('[1,0,0]'), ('[0,1,0]'), ('[0,0,1]'),
    ('[1,1,0]'), ('[0,1,1]'), ('[1,0,1]');

-- pgvector distance on pg_vectorsearch column via explicit cast
SELECT assert_test('pgvector distance on pgvs column via cast',
    (SELECT public.l2_distance(v::public.vector,
                               '[1,1,1]'::public.vector)
     FROM pgvs_items WHERE id = 1) IS NOT NULL);

-- =====================================================================
-- 9. NN ordering matches between implementations
-- =====================================================================

-- Same query on both types should return same ordering
SELECT assert_test('NN ordering matches (vector)',
    (SELECT array_agg(id ORDER BY
        v OPERATOR(public.<->) '[1,1,1]'::public.vec32)
     FROM pgv_items)
    =
    (SELECT array_agg(id ORDER BY
        v OPERATOR(public.<->) '[1,1,1]'::public.vector)
     FROM pgv_items));

-- =====================================================================
-- 10. Array round-trip compatibility
-- =====================================================================

SELECT assert_test('vector -> array -> vector round-trip',
    ('[1,2,3]'::public.vec32)::real[]::public.vec32::text = '[1,2,3]');

SELECT assert_test('pgvector array cast matches pgvs array cast',
    ('[1,2,3]'::public.vec32)::real[]::text
    = ('[1,2,3]'::public.vector)::real[]::text);

-- =====================================================================
-- 11. Dimension and norm utilities match
-- =====================================================================

SELECT assert_test('vector_dims matches',
    public.vec32_dims('[1,2,3]'::public.vec32)
    = public.vector_dims('[1,2,3]'::public.vector));

SELECT assert_test('vector_norm matches',
    abs(public.vec32_norm('[3,4]'::public.vec32)
      - public.vector_norm('[3,4]'::public.vector))
    < 1e-6);

-- =====================================================================
-- 12. Mixed-type distance via implicit cast
-- =====================================================================

-- pgvector value as LHS with pg_vectorsearch function (implicit cast)
SELECT assert_test('public.l2_distance with pgvector arg (implicit)',
    abs(public.l2_distance('[1,2,3]'::public.vector,
                        '[4,5,6]'::public.vec32)
      - public.l2_distance('[1,2,3]'::public.vector,
                           '[4,5,6]'::public.vector))
    < 1e-6);

SELECT assert_test('public.cosine_distance with pgvector arg (implicit)',
    abs(public.cosine_distance('[1,0]'::public.vector,
                            '[0,1]'::public.vec32)
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
-- 13. Install path: pgvector first, then pg_vectorsearch (DO block)
-- =====================================================================

DROP EXTENSION IF EXISTS pg_vectorsearch CASCADE;
DROP EXTENSION IF EXISTS vector CASCADE;

CREATE EXTENSION vector;
CREATE EXTENSION pg_vectorsearch;

SELECT assert_test('pgv-first: 4 binary casts',
    count_pgvector_casts() = 4);

SELECT assert_test('pgv-first: casts are standalone',
    casts_are_standalone());

SELECT assert_test('pgv-first: pgvector->pgvs cast works',
    '[1,2,3]'::public.vector::public.vec32::text = '[1,2,3]');

SELECT assert_test('pgv-first: pgvs->pgvector cast works',
    '[1,2,3]'::public.vec32::public.vector::text = '[1,2,3]');

-- =====================================================================
-- 14. Install path: pg_vectorsearch first, then pgvector (event trigger)
-- =====================================================================

DROP EXTENSION pg_vectorsearch CASCADE;
DROP EXTENSION vector;

CREATE EXTENSION pg_vectorsearch;
CREATE EXTENSION vector;

SELECT assert_test('pgvs-first: 4 binary casts',
    count_pgvector_casts() = 4);

SELECT assert_test('pgvs-first: casts are standalone',
    casts_are_standalone());

SELECT assert_test('pgvs-first: pgvector->pgvs cast works',
    '[1,2,3]'::public.vector::public.vec32::text = '[1,2,3]');

SELECT assert_test('pgvs-first: pgvs->pgvector cast works',
    '[1,2,3]'::public.vec32::public.vector::text = '[1,2,3]');

-- =====================================================================
-- 15. DROP pg_vectorsearch CASCADE: casts dropped, pgvector survives
-- =====================================================================

-- A bare drop is refused: the binary casts depend on pg_vectorsearch's
-- types, so removing pg_vectorsearch needs CASCADE and cannot silently
-- succeed. The subxact rolls back, so pg_vectorsearch is still installed
-- afterwards.
SELECT assert_test('drop-pgvs: bare DROP is refused (needs CASCADE)',
    stmt_is_refused('DROP EXTENSION pg_vectorsearch'));
SELECT assert_test(
    'drop-pgvs: pg_vectorsearch still installed after refused bare DROP',
    EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_vectorsearch'));

DROP EXTENSION pg_vectorsearch CASCADE;

SELECT assert_test('drop-pgvs: no casts remain',
    count_pgvector_casts() = 0);

SELECT assert_test('drop-pgvs: pgvector still loaded',
    EXISTS (SELECT 1 FROM pg_extension
            WHERE extname = 'vector'));

SELECT assert_test('drop-pgvs: pgvector still works',
    '[1,2,3]'::public.vector::text = '[1,2,3]');

-- pgvector's own operators are untouched -- CASCADE reached only the compat
-- objects, never through to pgvector.
SELECT assert_test('drop-pgvs: pgvector operator still works',
    ('[0,0]'::public.vector OPERATOR(public.<->) '[3,4]'::public.vector) = 5);

-- With pg_vectorsearch (and its casts) gone, pgvector is no longer
-- encumbered: a bare DROP now succeeds without CASCADE -- pg_vectorsearch
-- only forces CASCADE while it is installed. (This drops pgvector;
-- recreate it for the next section.)
SELECT assert_test('drop-pgvs: pgvector then drops without CASCADE',
    NOT stmt_is_refused('DROP EXTENSION vector'));
CREATE EXTENSION vector;

-- =====================================================================
-- 16. DROP pgvector CASCADE: casts dropped, pg_vectorsearch survives
-- =====================================================================

-- Restore both (pgvector-first path)
CREATE EXTENSION pg_vectorsearch;

-- Both present again: the six pgvector ordering operators are members, and a
-- bare drop is refused (the casts block it, same as the pg_vectorsearch side).
SELECT assert_test('drop-pgv: 6 pgvector ops are members before drop',
    count_pgvector_prism_ops() = 6);
SELECT assert_test('drop-pgv: bare DROP is refused (needs CASCADE)',
    stmt_is_refused('DROP EXTENSION vector'));

DROP EXTENSION vector CASCADE;

SELECT assert_test('drop-pgv: pg_vectorsearch still loaded',
    EXISTS (SELECT 1 FROM pg_extension
            WHERE extname = 'pg_vectorsearch'));

SELECT assert_test('drop-pgv: public.vec32 still works',
    '[1,2,3]'::public.vec32::text = '[1,2,3]');

-- The pgvector operator members are removed with pgvector's operators,
-- while pg_vectorsearch's own six ordering operators and its opclasses
-- are untouched.
SELECT assert_test('drop-pgv: pgvector ops removed from prism families',
    count_pgvector_prism_ops() = 0);
SELECT assert_test('drop-pgv: pg_vectorsearch native ops intact',
    count_pgvs_prism_ops() = 6);
SELECT assert_test('drop-pgv: prism opclasses intact',
    (SELECT count(*) FROM pg_opclass oc
        JOIN pg_am am ON am.oid = oc.opcmethod AND am.amname = 'prism') = 6);

SELECT assert_test('drop-pgv: event trigger still exists',
    EXISTS (SELECT 1 FROM pg_event_trigger
            WHERE evtname = 'prism_pgvector_cast_trigger'));

-- =====================================================================
-- 17. Re-create pgvector: event trigger re-creates casts
-- =====================================================================

CREATE EXTENSION vector;

SELECT assert_test('recreate-pgv: 4 casts re-created',
    count_pgvector_casts() = 4);

SELECT assert_test('recreate-pgv: casts are standalone',
    casts_are_standalone());

SELECT assert_test('recreate-pgv: cast works',
    '[1,2,3]'::public.vector::public.vec32::text = '[1,2,3]');

-- =====================================================================
-- 18. pg_vectorsearch standalone (no pgvector)
-- =====================================================================

DROP EXTENSION pg_vectorsearch CASCADE;
DROP EXTENSION vector;

CREATE EXTENSION pg_vectorsearch;

SELECT assert_test('standalone: pg_vectorsearch loads without pgvector',
    EXISTS (SELECT 1 FROM pg_extension
            WHERE extname = 'pg_vectorsearch'));

SELECT assert_test('standalone: public.vec32 works',
    '[1,2,3]'::public.vec32::text = '[1,2,3]');

SELECT assert_test('standalone: event trigger exists',
    EXISTS (SELECT 1 FROM pg_event_trigger
            WHERE evtname = 'prism_pgvector_cast_trigger'));

DROP EXTENSION pg_vectorsearch;

SELECT assert_test('standalone: clean drop',
    NOT EXISTS (SELECT 1 FROM pg_extension
                WHERE extname = 'pg_vectorsearch'));

-- =====================================================================
-- Indexing pgvector-typed columns with prism
-- =====================================================================
-- Checks that a prism index over a pgvector-typed column builds, is chosen
-- by the planner, and answers queries correctly.
--
-- The binary casts tested above are what make it reachable: PostgreSQL matches
-- an opclass to a column by binary coercibility, so an opclass declared FOR
-- TYPE public.vec16 accepts a public.halfvec column. The access method then has
-- to read that column as f16, and it gets that from the opclass's own type
-- descriptor rather than by identifying the column's type itself -- so a
-- pgvector column and a pg_vectorsearch column are handled identically,
-- with nothing in the access method comparing type OIDs. These tests pin
-- that down: a descriptor keyed on anything narrower would read the f16
-- pairs as float32 and return garbage from an index the planner
-- considers valid.
--
-- Each recall check is paired with a plan check. An index is only considered
-- for an ORDER BY when the ordering operator belongs to the index's operator
-- family, so a plan check is what distinguishes a working index scan from a
-- sequential scan that happens to return the same rows; without it the recall
-- assertions would pass on brute force and say nothing about the index.
--
-- Both spellings of the operator have to reach the index. pg_vectorsearch adds
-- pgvector's <->, <#> and <=> to its own prism families as ordering members
-- precisely so that a query written against pgvector -- or an unqualified
-- <-> resolving to pgvector under a pgvector-first search_path -- is not
-- silently downgraded to a sequential scan.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_vectorsearch;
SET search_path = public;

CREATE TEMP TABLE idx_src (id int, txt text);
INSERT INTO idx_src
    SELECT g, '[' || (SELECT string_agg(
                          round(sin(g * 0.7 + j * 1.3)::numeric, 4)::text, ',')
                      FROM generate_series(1, 8) j) || ']'
    FROM generate_series(1, 2000) g;

CREATE TABLE idx_pgv_h (id int, v public.halfvec(8));
CREATE TABLE idx_pgvs_h (id int, v public.vec16(8));
INSERT INTO idx_pgv_h SELECT id, txt::public.halfvec(8) FROM idx_src;
INSERT INTO idx_pgvs_h SELECT id, txt::public.vec16(8) FROM idx_src;
ANALYZE idx_pgv_h;
ANALYZE idx_pgvs_h;

-- Brute-force truth, established before any index exists.
CREATE TEMP TABLE idx_truth AS
    SELECT array_agg(id ORDER BY id) AS ids
      FROM (SELECT id FROM idx_pgv_h
             ORDER BY v OPERATOR(public.<->) '[0,0,0,0,0,0,0,0]'::public.vec16(8)
             LIMIT 10) t;

CREATE INDEX idx_pgv_h_i ON idx_pgv_h USING prism (v public.vec16_l2_ops);
CREATE INDEX idx_pgvs_h_i ON idx_pgvs_h USING prism (v public.vec16_l2_ops);

-- The planner must actually choose the index, or the rest proves nothing.
-- EXPLAIN cannot appear in a subquery, hence the helper.
CREATE OR REPLACE FUNCTION public.plan_uses_index_scan(q text) RETURNS bool
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

-- Rerank every candidate RaBitQ cannot exclude, instead of the automatic
-- pool. What follows tests decoding and operator wiring, not recall tuning,
-- and the query makes those two indistinguishable otherwise: the origin,
-- against vectors of near-constant norm, sits at almost the same distance
-- from every row -- the whole dataset spans 1.89 to 2.10 and the true top
-- ten fall within 0.0002 of each other. No approximate ordering survives
-- that, so with a capped pool recall measures the size of the pool rather
-- than whether the column decoded. Uncapped, a misdecoded column still
-- fails hard, because exact distances over wrong values rank wrongly.
SET prism.rerank_pool = -1;

SELECT assert_test('prism index is used on a pgvector halfvec column',
    public.plan_uses_index_scan($q$SELECT id FROM idx_pgv_h
        ORDER BY v OPERATOR(public.<->) '[0,0,0,0,0,0,0,0]'::public.vec16(8)
        LIMIT 10$q$));

-- Recall against brute force. A misdecoded column scores near zero here, so
-- 8 of 10 is a generous floor that still fails hard on a decode bug while
-- tolerating the clusters the scan does not probe.
SELECT assert_test(
    'pgvector halfvec column: prism recall >= 8/10',
    (SELECT count(*) FROM (
        SELECT id FROM idx_pgv_h
         ORDER BY v OPERATOR(public.<->) '[0,0,0,0,0,0,0,0]'::public.vec16(8)
         LIMIT 10) g
      WHERE g.id = ANY (SELECT unnest(ids) FROM idx_truth)) >= 8);

-- Control: pg_vectorsearch's own halfvec column, same data, same expectation.
SELECT assert_test(
    'pgvs halfvec column: prism recall >= 8/10',
    (SELECT count(*) FROM (
        SELECT id FROM idx_pgvs_h
         ORDER BY v OPERATOR(public.<->) '[0,0,0,0,0,0,0,0]'::public.vec16(8)
         LIMIT 10) g
      WHERE g.id = ANY (SELECT unnest(ids) FROM idx_truth)) >= 8);

-- Centroids follow the column for a pgvector column too, by the same
-- binary-coercibility test. Compression off, so the format is decided by the
-- column type rather than by RaBitQ.
CREATE INDEX idx_fmt_i ON idx_pgv_h USING prism (v public.vec16_l2_ops)
    WITH (centroid_compression = off, centroid_fastscan = off,
          fastscan = off, soar_lambda = 0, boundary_epsilon = 0);
SELECT assert_test('pgvector halfvec column gets half-precision centroids',
    (SELECT bool_and(format = 'half')
       FROM prism.centroid_pages('idx_fmt_i'::regclass)));

-- pgvector's three distance operators are ordering members of each of
-- pg_vectorsearch's six prism families.
SELECT assert_test('pgvector distance operators joined the prism families',
    count_pgvector_prism_ops() = 6);

-- A prism index is reachable from pgvector's operator, including as a bare
-- <-> under a search_path that resolves to pgvector.
SELECT assert_test(
    'prism index is used via pgvector''s operator on a pgvector column',
    public.plan_uses_index_scan($q$SELECT id FROM idx_pgv_h
        ORDER BY v OPERATOR(public.<->) '[0,0,0,0,0,0,0,0]'::halfvec(8)
        LIMIT 10$q$));
SELECT assert_test(
    'prism index is used via an unqualified <-> resolving to pgvector',
    public.plan_uses_index_scan($q$SELECT id FROM idx_pgv_h
        ORDER BY v <-> '[0,0,0,0,0,0,0,0]'::halfvec(8) LIMIT 10$q$));

-- And returns the same neighbours through that operator as brute force does.
SELECT assert_test(
    'pgvector operator on prism index: recall >= 8/10',
    (SELECT count(*) FROM (
        SELECT id FROM idx_pgv_h
         ORDER BY v OPERATOR(public.<->) '[0,0,0,0,0,0,0,0]'::halfvec(8)
         LIMIT 10) g
      WHERE g.id = ANY (SELECT unnest(ids) FROM idx_truth)) >= 8);

-- The reverse pairing -- a pg_vectorsearch-typed column with pgvector's
-- operator -- does not resolve, and is not expected to: the
-- pg_vectorsearch -> pgvector cast is ASSIGNMENT rather than IMPLICIT,
-- deliberately, so that having both extensions installed does not make
-- every operator call ambiguous. Queries over pg_vectorsearch's own types
-- use pg_vectorsearch's own operators. pg_vectorsearch's types now
-- default to the same schema as pgvector's own (public), so namespace can
-- no longer tell the two apart -- identify "pgvector's own <->" by
-- extension membership instead: no <-> operator owned by the vector
-- extension takes a pg_vectorsearch vec16 as its left argument.
SELECT assert_test(
    'pg_vectorsearch column with pgvector operator does not resolve',
    NOT EXISTS (
        SELECT 1 FROM pg_operator op
        JOIN pg_depend d ON d.objid = op.oid AND d.deptype = 'e'
        JOIN pg_extension e ON e.oid = d.refobjid AND e.extname = 'vector'
         WHERE op.oprname = '<->'
           AND op.oprleft = 'public.vec16'::regtype));

RESET enable_seqscan;
DROP FUNCTION public.plan_uses_index_scan(text);
DROP TABLE idx_pgv_h;
DROP TABLE idx_pgvs_h;
RESET search_path;

-- =====================================================================
-- 19. Both extensions in NON-default schemas (dynamic discovery)
-- =====================================================================
-- pgvector is relocatable, and pg_vectorsearch's types/operators/AM are
-- install-time relocatable too (prism.rebalance and friends are the one
-- exception -- see sql/pg_vectorsearch.sql). setup_pgvector_compat() and the
-- install DO block/event trigger discover BOTH schemas from
-- pg_extension.extnamespace rather than assuming either is public. Install
-- pgvector into pgv_alt and pg_vectorsearch into pgvs_alt, and assert the
-- compat
-- casts, the operator family memberships, and a real index scan all still
-- come out right -- for both install orderings.

DROP EXTENSION IF EXISTS pg_vectorsearch CASCADE;
DROP EXTENSION IF EXISTS vector CASCADE;
-- No pre-emptive DROP SCHEMA: these are our names, but cascade-dropping
-- whatever a user might have under them would be too aggressive. A clean DB
-- is a documented prerequisite, so a bare CREATE fails loudly if a name is
-- already taken.
CREATE SCHEMA pgv_alt;
CREATE SCHEMA pgvs_alt;

-- Ordering A: pgvector (in pgv_alt) first, pg_vectorsearch (in
-- pgvs_alt) second
-- (install DO block).
CREATE EXTENSION vector SCHEMA pgv_alt;
CREATE EXTENSION pg_vectorsearch SCHEMA pgvs_alt;

SELECT assert_test('custom-schema (pgv-first): 4 binary compat casts',
    (SELECT count(*) FROM pg_cast c
       JOIN pg_type s ON s.oid = c.castsource
       JOIN pg_type t ON t.oid = c.casttarget
      WHERE c.castmethod = 'b'
        AND ((s.typnamespace = 'pgv_alt'::regnamespace
              AND s.typname IN ('vector', 'halfvec')
              AND t.typnamespace = 'pgvs_alt'::regnamespace
              AND t.typname IN ('vec32', 'vec16'))
          OR (t.typnamespace = 'pgv_alt'::regnamespace
              AND t.typname IN ('vector', 'halfvec')
              AND s.typnamespace = 'pgvs_alt'::regnamespace
              AND s.typname IN ('vec32', 'vec16')))) = 4);

SELECT assert_test(
    'custom-schema (pgv-first): 6 pgvector ops are prism members',
    (SELECT count(*) FROM pg_amop ao
       JOIN pg_opfamily f ON f.oid = ao.amopfamily
       JOIN pg_am am ON am.oid = f.opfmethod AND am.amname = 'prism'
       JOIN pg_operator op ON op.oid = ao.amopopr
      WHERE ao.amoppurpose = 'o'
        AND op.oprnamespace = 'pgv_alt'::regnamespace) = 6);

SELECT assert_test('custom-schema: pgvector->pgvs cast round-trips',
    '[1,2,3]'::pgv_alt.vector::pgvs_alt.vec32::text = '[1,2,3]');
SELECT assert_test('custom-schema: pgvs->pgvector cast round-trips',
    '[1,2,3]'::pgvs_alt.vec32::pgv_alt.vector::text = '[1,2,3]');

-- The build-identity functions travel with @extschema@, unlike prism's
-- own procedures -- confirm they land inside pgvs_alt alongside the types.
SELECT assert_test(
    'custom-schema: build identity lands in pgvs_alt',
    to_regprocedure('pgvs_alt.pg_vectorsearch_git_commit()') IS NOT NULL
    AND to_regprocedure('pgvs_alt.pg_vectorsearch_version()') IS NOT NULL);

-- A real index scan over a pgv_alt.vector column via pgvector's operator: the
-- operator only reaches the index because discovery added it to the family.
CREATE OR REPLACE FUNCTION public.plan_uses_index_scan(q text) RETURNS bool
    LANGUAGE plpgsql AS $fn$
DECLARE
    line text;
BEGIN
    FOR line IN EXECUTE 'EXPLAIN (COSTS OFF) ' || q LOOP
        IF line LIKE '%Index Scan%' THEN RETURN true; END IF;
    END LOOP;
    RETURN false;
END $fn$;

CREATE TABLE idx_alt (id int, v pgv_alt.vector(8));
INSERT INTO idx_alt
    SELECT g, ('[' || (SELECT string_agg(
                          round(sin(g * 0.7 + j * 1.3)::numeric, 4)::text, ',')
                       FROM generate_series(1, 8) j) || ']')::pgv_alt.vector(8)
    FROM generate_series(1, 2000) g;
ANALYZE idx_alt;
CREATE INDEX idx_alt_i ON idx_alt USING prism (v pgvs_alt.vec32_l2_ops);
SET enable_seqscan = off;
SET prism.rerank_pool = -1;
SELECT assert_test(
    'custom-schema: prism index used via pgvector operator on pgv_alt column',
    public.plan_uses_index_scan($q$SELECT id FROM idx_alt
        ORDER BY v OPERATOR(pgv_alt.<->) '[0,0,0,0,0,0,0,0]'::pgv_alt.vector(8)
        LIMIT 10$q$));
RESET enable_seqscan;
RESET prism.rerank_pool;
DROP TABLE idx_alt;
DROP FUNCTION public.plan_uses_index_scan(text);

-- Ordering B: pg_vectorsearch (in pgvs_alt) first, pgvector (in
-- pgv_alt) second
-- (event trigger).
DROP EXTENSION pg_vectorsearch CASCADE;
DROP EXTENSION vector CASCADE;
CREATE EXTENSION pg_vectorsearch SCHEMA pgvs_alt;
CREATE EXTENSION vector SCHEMA pgv_alt;   -- fires prism_pgvector_cast_trigger

SELECT assert_test('custom-schema (pgvs-first): event trigger created 4 casts',
    (SELECT count(*) FROM pg_cast c
       JOIN pg_type s ON s.oid = c.castsource
       JOIN pg_type t ON t.oid = c.casttarget
      WHERE c.castmethod = 'b'
        AND ((s.typnamespace = 'pgv_alt'::regnamespace
              AND s.typname IN ('vector', 'halfvec')
              AND t.typnamespace = 'pgvs_alt'::regnamespace
              AND t.typname IN ('vec32', 'vec16'))
          OR (t.typnamespace = 'pgv_alt'::regnamespace
              AND t.typname IN ('vector', 'halfvec')
              AND s.typnamespace = 'pgvs_alt'::regnamespace
              AND s.typname IN ('vec32', 'vec16')))) = 4);

-- Mirror the pgv-first checks: the event-trigger path must also add the six
-- operators to the prism families and yield a real index scan, or a
-- regression there could pass on casts alone while queries silently seq-scan.
SELECT assert_test(
    'custom-schema (pgvs-first): 6 pgvector ops are prism members',
    (SELECT count(*) FROM pg_amop ao
       JOIN pg_opfamily f ON f.oid = ao.amopfamily
       JOIN pg_am am ON am.oid = f.opfmethod AND am.amname = 'prism'
       JOIN pg_operator op ON op.oid = ao.amopopr
      WHERE ao.amoppurpose = 'o'
        AND op.oprnamespace = 'pgv_alt'::regnamespace) = 6);

CREATE OR REPLACE FUNCTION public.plan_uses_index_scan(q text) RETURNS bool
    LANGUAGE plpgsql AS $fn$
DECLARE
    line text;
BEGIN
    FOR line IN EXECUTE 'EXPLAIN (COSTS OFF) ' || q LOOP
        IF line LIKE '%Index Scan%' THEN RETURN true; END IF;
    END LOOP;
    RETURN false;
END $fn$;

CREATE TABLE idx_altb (id int, v pgv_alt.vector(8));
INSERT INTO idx_altb
    SELECT g, ('[' || (SELECT string_agg(
                          round(sin(g * 0.7 + j * 1.3)::numeric, 4)::text, ',')
                       FROM generate_series(1, 8) j) || ']')::pgv_alt.vector(8)
    FROM generate_series(1, 2000) g;
ANALYZE idx_altb;
CREATE INDEX idx_altb_i ON idx_altb USING prism (v pgvs_alt.vec32_l2_ops);
SET enable_seqscan = off;
SET prism.rerank_pool = -1;
SELECT assert_test(
    'custom-schema (pgvs-first): prism index used via pgvector operator',
    public.plan_uses_index_scan($q$SELECT id FROM idx_altb
        ORDER BY v OPERATOR(pgv_alt.<->) '[0,0,0,0,0,0,0,0]'::pgv_alt.vector(8)
        LIMIT 10$q$));
RESET enable_seqscan;
RESET prism.rerank_pool;
DROP TABLE idx_altb;
DROP FUNCTION public.plan_uses_index_scan(text);

-- Restore the default public install for the summary / any later re-run.
-- Both alt schemas are empty once their extension is gone, so a plain DROP
-- (no CASCADE) suffices and would fail loudly if anything unexpected were
-- left behind.
DROP EXTENSION pg_vectorsearch CASCADE;
DROP EXTENSION vector CASCADE;
DROP SCHEMA pgv_alt;
DROP SCHEMA pgvs_alt;
CREATE EXTENSION vector;
CREATE EXTENSION pg_vectorsearch;

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
