-- Rerank correctness: index scan results must match sequential scan

-- Use higher dimensionality to exercise the full rerank path
-- (low-dim vectors can sometimes bypass rerank entirely)
CREATE TABLE rerank_test (id serial, v vector(32));

INSERT INTO rerank_test (v)
    SELECT (
        SELECT array_agg(sin(i * 0.1 + j * 0.7)::real)
        FROM generate_series(0, 31) j
    )::vector(32)
    FROM generate_series(1, 500) i;

-- L2 distance
CREATE INDEX idx_rerank_l2 ON rerank_test USING mktann (v)
    WITH (centroid_compression = true);

-- Compute ground truth via sequential scan
SET enable_indexscan = off;
CREATE TEMP TABLE truth_l2 AS
    SELECT id, v <-> (SELECT v FROM rerank_test WHERE id = 42) AS dist
    FROM rerank_test
    ORDER BY dist LIMIT 10;
RESET enable_indexscan;

-- Compute index scan results (exercises rerank)
SET enable_seqscan = off;
CREATE TEMP TABLE result_l2 AS
    SELECT id, v <-> (SELECT v FROM rerank_test WHERE id = 42) AS dist
    FROM rerank_test
    ORDER BY v <-> (SELECT v FROM rerank_test WHERE id = 42) LIMIT 10;
RESET enable_seqscan;

-- All ground truth top-10 must appear in index results
SELECT count(*) AS matching_ids
    FROM truth_l2 t JOIN result_l2 r USING (id);

-- Distances must match exactly (rerank uses full-precision vectors)
SELECT bool_and(abs(t.dist - r.dist) < 1e-5) AS distances_match
    FROM truth_l2 t JOIN result_l2 r USING (id);

-- Cosine distance
CREATE INDEX idx_rerank_cos ON rerank_test
    USING mktann (v vector_cosine_ops)
    WITH (centroid_compression = true);

SET enable_indexscan = off;
CREATE TEMP TABLE truth_cos AS
    SELECT id, v <=> (SELECT v FROM rerank_test WHERE id = 42) AS dist
    FROM rerank_test
    ORDER BY dist LIMIT 10;
RESET enable_indexscan;

SET enable_seqscan = off;
CREATE TEMP TABLE result_cos AS
    SELECT id, v <=> (SELECT v FROM rerank_test WHERE id = 42) AS dist
    FROM rerank_test
    ORDER BY v <=> (SELECT v FROM rerank_test WHERE id = 42) LIMIT 10;
RESET enable_seqscan;

SELECT count(*) AS matching_ids
    FROM truth_cos t JOIN result_cos r USING (id);

SELECT bool_and(abs(t.dist - r.dist) < 1e-5) AS distances_match
    FROM truth_cos t JOIN result_cos r USING (id);

-- Automatic rerank_pool: the pool must respect the MULT * k floor.
--
-- Asserted as bounds rather than an exact count, because the survivor
-- population depends on floating-point distance estimates and so is not
-- portable, whereas the cap that bounds it is. The upper bound is what
-- cap that bounds it is. At 500 rows the survivor population is ~16, well
-- under either the current 4 * k or the former 16 * k, so this cannot pin
-- the multiplier -- test/unit/test_rerank_pool.c does that. What this
-- covers is the end-to-end plumbing: that the GUC reaches the scan, that
-- an explicit cap is honoured, and that neither ever truncates below k.
SET enable_seqscan = off;
SET mkt.rerank_pool = 0;   -- automatic
DO $$
DECLARE
    ej json;
    stats json;
    cands int;
BEGIN
    EXECUTE 'EXPLAIN (ANALYZE, VERBOSE, COSTS OFF, FORMAT JSON)
        SELECT id FROM rerank_test
        ORDER BY v <-> (SELECT v FROM rerank_test WHERE id = 42) LIMIT 10'
    INTO ej;
    stats := ej->0->'Plan'->'Plans'->1->'Mktann';
    cands := (stats->>'Rerank Candidates')::int;
    -- A wrong JSON path yields NULL, and every comparison below would then
    -- be NULL rather than true -- i.e. the whole test would pass without
    -- asserting anything. Fail loudly instead.
    IF cands IS NULL THEN
        RAISE EXCEPTION 'Rerank Candidates not found in EXPLAIN output';
    END IF;
    -- Never below k: a cap must not truncate the result set.
    IF cands < 10 THEN
        RAISE EXCEPTION 'auto rerank pool % is below k=10', cands;
    END IF;
    -- Never above MULT * k = 40 for k = 10.
    IF cands > 40 THEN
        RAISE EXCEPTION 'auto rerank pool % exceeds 4 * k = 40', cands;
    END IF;
END $$;

-- An explicit cap is honoured, and still floored at k.
SET mkt.rerank_pool = 12;
DO $$
DECLARE
    ej json;
    cands int;
BEGIN
    EXECUTE 'EXPLAIN (ANALYZE, VERBOSE, COSTS OFF, FORMAT JSON)
        SELECT id FROM rerank_test
        ORDER BY v <-> (SELECT v FROM rerank_test WHERE id = 42) LIMIT 10'
    INTO ej;
    cands := ((ej->0->'Plan'->'Plans'->1->'Mktann')->>'Rerank Candidates')::int;
    IF cands IS NULL THEN
        RAISE EXCEPTION 'Rerank Candidates not found in EXPLAIN output';
    END IF;
    IF cands > 12 OR cands < 10 THEN
        RAISE EXCEPTION 'explicit rerank pool 12 gave % candidates', cands;
    END IF;
END $$;
RESET mkt.rerank_pool;
RESET enable_seqscan;
SELECT 'auto and explicit rerank_pool bounds hold' AS rerank_pool_bounds;

-- Cleanup
DROP TABLE rerank_test;
