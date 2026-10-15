-- Hybrid search tests: pg_textsearch (BM25) + pg_vectorsearch (PRISM vector ANN)
--
-- Verifies the hybrid-search query patterns documented in
-- docs/architecture.md and https://www.tigerdata.com/docs/build/examples/hybrid-search
-- actually work when both extensions are installed together:
--
--   1. Reciprocal Rank Fusion (RRF) -- two independently ranked lists (BM25
--      keyword rank, vector distance rank) combined via dual CTEs, a
--      FULL JOIN, and the 1/(k+rank) formula.
--   2. BM25 recall -> vector rerank -- narrow to a keyword-relevant
--      candidate set first, then rank that set by vector distance.
--   3. Vector recall -> BM25 rerank -- the symmetric direction.
--
-- Each case checks both that the *recall* step actually uses its extension's
-- index (not a seqscan that happens to return the same rows -- see the
-- testing skill's "PG suites load the installed .so" gotcha) and that the
-- result is correct against an independently-known answer (which topic a
-- document belongs to), not merely "didn't crash".
--
-- Prerequisites:
--   pg_textsearch and pg_vectorsearch must be installed in PostgreSQL.
--
-- Usage:
--   psql -f test/pg/compat/hybridsearch.sql

\set ON_ERROR_STOP on
\pset tuples_only on
\pset format unaligned

CREATE EXTENSION IF NOT EXISTS pg_textsearch;
CREATE EXTENSION IF NOT EXISTS pg_vectorsearch;

-- =====================================================================
-- Test helpers
-- =====================================================================

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

-- Prints the plan for a combined (BM25 + vector) query so CI logs show
-- exactly how the planner executes today's two-index-scans-plus-glue
-- approach -- useful for spotting integration opportunities even though
-- this test only asserts on correctness, not on plan shape here.
CREATE OR REPLACE FUNCTION show_plan(label text, q text) RETURNS void
    LANGUAGE plpgsql AS $fn$
DECLARE
    line text;
BEGIN
    RAISE NOTICE '--- plan: % ---', label;
    FOR line IN EXECUTE 'EXPLAIN (COSTS OFF) ' || q LOOP
        RAISE NOTICE '%', line;
    END LOOP;
END $fn$;

-- =====================================================================
-- Test data: five topics, six documents each. Content vocabulary is
-- disjoint per topic so BM25 cleanly separates them; embeddings are
-- one-hot per topic so vector distance does too. Both signals therefore
-- have an independently-known correct answer -- "topic" -- rather than
-- relying on whatever the code under test happens to output.
-- =====================================================================

CREATE TABLE hybrid_docs (
    id serial PRIMARY KEY,
    topic text NOT NULL,
    content text NOT NULL,
    embedding vec32(5) NOT NULL
);

CREATE TEMP TABLE topic_defs (
    topic text,
    embedding vec32(5),
    texts text[]
);

INSERT INTO topic_defs VALUES
('astronomy', '[1,0,0,0,0]', ARRAY[
    'The astronomer pointed the telescope toward a distant galaxy.',
    'Nebula clouds glow near the newly discovered exoplanet orbit.',
    'Scientists measured the orbit of the comet around the star.',
    'The observatory captured images of a spiral galaxy at night.',
    'A telescope revealed faint starlight from a distant nebula.',
    'Astronomers tracked the planet as it crossed the bright star.'
]),
('cooking', '[0,1,0,0,0]', ARRAY[
    'The chef baked bread using flour, yeast, and warm water.',
    'A pinch of sugar balanced the tart flavor of the sauce.',
    'The recipe called for butter, eggs, and a hot oven.',
    'She simmered the soup with onions, garlic, and fresh herbs.',
    'The bakery sold pastries dusted with powdered sugar.',
    'He seasoned the roast with salt, pepper, and rosemary.'
]),
('finance', '[0,0,1,0,0]', ARRAY[
    'The investor reviewed quarterly earnings before buying more stock.',
    'Interest rates rose, pushing bond yields higher this quarter.',
    'The company reported strong revenue growth and rising profit.',
    'Analysts revised their forecast after the earnings report.',
    'The stock market rallied on news of falling interest rates.',
    'Shareholders approved the dividend increase at the annual meeting.'
]),
('sports', '[0,0,0,1,0]', ARRAY[
    'The striker scored a goal in the final minute of the match.',
    'The coach praised the team after their championship victory.',
    'Fans cheered as the runner crossed the finish line first.',
    'The pitcher threw a fastball to strike out the batter.',
    'The team celebrated their playoff win with the trophy.',
    'The goalkeeper made a diving save to win the match.'
]),
('medicine', '[0,0,0,0,1]', ARRAY[
    'The doctor prescribed medication to treat the infection.',
    'The surgeon performed a delicate operation on the patient.',
    'Nurses monitored the patient vital signs throughout the night.',
    'The vaccine trial showed strong immunity against the virus.',
    'The clinic treated patients with a new therapy for the disease.',
    'The physician diagnosed the illness after reviewing the symptoms.'
]);

INSERT INTO hybrid_docs (topic, content, embedding)
SELECT t.topic, doc, t.embedding
FROM topic_defs t, unnest(t.texts) AS doc;

SELECT assert_test('corpus has 30 documents', (SELECT count(*) FROM hybrid_docs) = 30);

CREATE INDEX hybrid_bm25_idx ON hybrid_docs
    USING bm25 (content) WITH (text_config = 'english');
CREATE INDEX hybrid_prism_idx ON hybrid_docs USING prism (embedding);
ANALYZE hybrid_docs;

SET enable_seqscan = off;

-- Shared query target: 'astronomy'. Both the keyword query and the vector
-- query point at the same topic, so a correct hybrid query should surface
-- astronomy documents no matter which signal (or combination) drives it.
-- text: uses words unique to the astronomy documents above.
-- vector: the astronomy centroid itself.

-- =====================================================================
-- Case 1: Reciprocal Rank Fusion
-- =====================================================================

SELECT assert_test('RRF: BM25 recall list uses the bm25 index',
    plan_uses_index_scan($q$
        SELECT id FROM hybrid_docs
            ORDER BY content <@> 'telescope galaxy nebula orbit comet star observatory starlight astronomer planet exoplanet' LIMIT 10$q$));

SELECT assert_test('RRF: vector recall list uses the prism index',
    plan_uses_index_scan($q$
        SELECT id FROM hybrid_docs
            ORDER BY embedding <-> '[1,0,0,0,0]' LIMIT 10$q$));

SELECT show_plan('RRF combined query', $q$
    WITH bm25_results AS (
        SELECT id, ROW_NUMBER() OVER (
            ORDER BY content <@> 'telescope galaxy nebula orbit comet star observatory starlight astronomer planet exoplanet'
        ) AS rank
        FROM hybrid_docs
        ORDER BY content <@> 'telescope galaxy nebula orbit comet star observatory starlight astronomer planet exoplanet'
        LIMIT 10
    ),
    vector_results AS (
        SELECT id, ROW_NUMBER() OVER (
            ORDER BY embedding <-> '[1,0,0,0,0]'
        ) AS rank
        FROM hybrid_docs
        ORDER BY embedding <-> '[1,0,0,0,0]'
        LIMIT 10
    ),
    fused AS (
        SELECT COALESCE(b.id, v.id) AS id,
            COALESCE(1.0 / (60 + b.rank), 0)
                + COALESCE(1.0 / (60 + v.rank), 0) AS rrf_score
        FROM bm25_results b
        FULL JOIN vector_results v ON b.id = v.id
        ORDER BY rrf_score DESC
        LIMIT 10
    )
    SELECT f.id, d.topic, f.rrf_score
    FROM fused f
    JOIN hybrid_docs d ON d.id = f.id$q$);

CREATE TEMP TABLE rrf_result AS
WITH bm25_results AS (
    SELECT id, ROW_NUMBER() OVER (
        ORDER BY content <@> 'telescope galaxy nebula orbit comet star observatory starlight astronomer planet exoplanet'
    ) AS rank
    FROM hybrid_docs
    ORDER BY content <@> 'telescope galaxy nebula orbit comet star observatory starlight astronomer planet exoplanet'
    LIMIT 10
),
vector_results AS (
    SELECT id, ROW_NUMBER() OVER (
        ORDER BY embedding <-> '[1,0,0,0,0]'
    ) AS rank
    FROM hybrid_docs
    ORDER BY embedding <-> '[1,0,0,0,0]'
    LIMIT 10
),
fused AS (
    SELECT COALESCE(b.id, v.id) AS id,
        COALESCE(1.0 / (60 + b.rank), 0)
            + COALESCE(1.0 / (60 + v.rank), 0) AS rrf_score
    FROM bm25_results b
    FULL JOIN vector_results v ON b.id = v.id
    ORDER BY rrf_score DESC
    LIMIT 10
)
SELECT f.id, d.topic, f.rrf_score
FROM fused f
JOIN hybrid_docs d ON d.id = f.id;

SELECT assert_test('RRF: top fused result is on-topic',
    (SELECT topic FROM rrf_result ORDER BY rrf_score DESC LIMIT 1) = 'astronomy');

SELECT assert_test('RRF: every astronomy document (found by both signals) outranks every off-topic document',
    (SELECT min(rrf_score) FROM rrf_result WHERE topic = 'astronomy')
        > (SELECT COALESCE(max(rrf_score), -1) FROM rrf_result WHERE topic <> 'astronomy'));

DROP TABLE rrf_result;

-- =====================================================================
-- Case 2: BM25 recall, vector rerank
-- =====================================================================

SELECT assert_test('BM25->vector: recall step uses the bm25 index',
    plan_uses_index_scan($q$
        SELECT id FROM hybrid_docs
            ORDER BY content <@> 'telescope galaxy nebula orbit comet star observatory starlight astronomer planet exoplanet' LIMIT 10$q$));

SELECT show_plan('BM25 recall -> vector rerank combined query', $q$
    WITH candidates AS (
        SELECT id FROM hybrid_docs
        ORDER BY content <@> 'telescope galaxy nebula orbit comet star observatory starlight astronomer planet exoplanet'
        LIMIT 10
    )
    SELECT d.id, d.topic
    FROM hybrid_docs d
    JOIN candidates c ON c.id = d.id
    ORDER BY d.embedding <-> '[1,0,0,0,0]'
    LIMIT 5$q$);

-- Computed independently of the query below, so the next two assertions
-- actually exercise the JOIN rather than merely re-deriving the right
-- answer from the vector rerank signal alone (which this corpus's
-- one-hot embeddings would do even with e.g. a `JOIN candidates c ON
-- true` typo -- verified by hand that such a typo still passes an
-- on-topic-only check, just with the top row duplicated 5 times).
CREATE TEMP TABLE bm25_recall_ids AS
SELECT id FROM hybrid_docs
ORDER BY content <@> 'telescope galaxy nebula orbit comet star observatory starlight astronomer planet exoplanet'
LIMIT 10;

CREATE TEMP TABLE bm25_then_vector AS
WITH candidates AS (
    SELECT id FROM hybrid_docs
    ORDER BY content <@> 'telescope galaxy nebula orbit comet star observatory starlight astronomer planet exoplanet'
    LIMIT 10
)
SELECT d.id, d.topic
FROM hybrid_docs d
JOIN candidates c ON c.id = d.id
ORDER BY d.embedding <-> '[1,0,0,0,0]'
LIMIT 5;

SELECT assert_test('BM25->vector: reranked top result is on-topic',
    (SELECT topic FROM bm25_then_vector LIMIT 1) = 'astronomy');

SELECT assert_test('BM25->vector: all 5 reranked results are on-topic',
    (SELECT count(*) FROM bm25_then_vector WHERE topic = 'astronomy') = 5);

SELECT assert_test('BM25->vector: reranked results are 5 distinct documents',
    (SELECT count(DISTINCT id) FROM bm25_then_vector) = 5);

SELECT assert_test('BM25->vector: reranked results are all within the BM25 recall set',
    NOT EXISTS (SELECT id FROM bm25_then_vector EXCEPT SELECT id FROM bm25_recall_ids));

DROP TABLE bm25_then_vector;
DROP TABLE bm25_recall_ids;

-- =====================================================================
-- Case 3: Vector recall, BM25 rerank (symmetric direction)
-- =====================================================================

SELECT assert_test('vector->BM25: recall step uses the prism index',
    plan_uses_index_scan($q$
        SELECT id FROM hybrid_docs
            ORDER BY embedding <-> '[1,0,0,0,0]' LIMIT 10$q$));

SELECT show_plan('vector recall -> BM25 rerank combined query', $q$
    WITH candidates AS (
        SELECT id FROM hybrid_docs
        ORDER BY embedding <-> '[1,0,0,0,0]'
        LIMIT 10
    )
    SELECT d.id, d.topic
    FROM hybrid_docs d
    JOIN candidates c ON c.id = d.id
    ORDER BY d.content <@> 'telescope galaxy nebula orbit comet star observatory starlight astronomer planet exoplanet'
    LIMIT 5$q$);

-- Independently computed for the same reason as bm25_recall_ids above.
CREATE TEMP TABLE vector_recall_ids AS
SELECT id FROM hybrid_docs
ORDER BY embedding <-> '[1,0,0,0,0]'
LIMIT 10;

CREATE TEMP TABLE vector_then_bm25 AS
WITH candidates AS (
    SELECT id FROM hybrid_docs
    ORDER BY embedding <-> '[1,0,0,0,0]'
    LIMIT 10
)
SELECT d.id, d.topic
FROM hybrid_docs d
JOIN candidates c ON c.id = d.id
ORDER BY d.content <@> 'telescope galaxy nebula orbit comet star observatory starlight astronomer planet exoplanet'
LIMIT 5;

SELECT assert_test('vector->BM25: reranked top result is on-topic',
    (SELECT topic FROM vector_then_bm25 LIMIT 1) = 'astronomy');

SELECT assert_test('vector->BM25: all 5 reranked results are on-topic',
    (SELECT count(*) FROM vector_then_bm25 WHERE topic = 'astronomy') = 5);

SELECT assert_test('vector->BM25: reranked results are 5 distinct documents',
    (SELECT count(DISTINCT id) FROM vector_then_bm25) = 5);

SELECT assert_test('vector->BM25: reranked results are all within the vector recall set',
    NOT EXISTS (SELECT id FROM vector_then_bm25 EXCEPT SELECT id FROM vector_recall_ids));

DROP TABLE vector_then_bm25;
DROP TABLE vector_recall_ids;

RESET enable_seqscan;
RESET search_path;

-- =====================================================================
-- Cleanup
-- =====================================================================

DROP FUNCTION show_plan(text, text);
DROP FUNCTION plan_uses_index_scan(text);
DROP TABLE hybrid_docs;

-- =====================================================================
-- Summary
-- =====================================================================

\echo
\echo ================================
\echo hybrid search test results
\echo ================================

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
        RAISE EXCEPTION 'Some hybrid search tests failed';
    END IF;
END;
$$;
