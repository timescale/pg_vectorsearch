-- Rerank correctness: index scan results must match sequential scan

-- Use higher dimensionality to exercise the full rerank path
-- (low-dim vectors can sometimes bypass rerank entirely)
CREATE TABLE rerank_test (id serial, v vec32(32));

INSERT INTO rerank_test (v)
    SELECT (
        SELECT array_agg(sin(i * 0.1 + j * 0.7)::real)
        FROM generate_series(0, 31) j
    )::vec32(32)
    FROM generate_series(1, 500) i;

-- The query vector, hoisted into an immutable function so the planner folds
-- it to a literal. Read with a subquery it becomes an InitPlan whose own
-- sequential scan lands in every plan and cost below.
SELECT v::text AS rqv FROM rerank_test WHERE id = 42 \gset
SELECT format($f$
    CREATE FUNCTION rqv() RETURNS vec32(32)
        LANGUAGE sql IMMUTABLE PARALLEL SAFE
        AS $b$ SELECT %L::vec32(32) $b$
$f$, :'rqv') \gexec

-- L2 distance
CREATE INDEX idx_rerank_l2 ON rerank_test USING prism (v)
    WITH (centroid_compression = true);

-- Compute ground truth via sequential scan
SET enable_indexscan = off;
CREATE TEMP TABLE truth_l2 AS
    SELECT id, v <-> rqv() AS dist
    FROM rerank_test
    ORDER BY dist LIMIT 10;
RESET enable_indexscan;

-- Compute index scan results (exercises rerank)
-- Small tables, so a sequential scan is the plan the cost model correctly
-- prefers and the scan below is forced through the index. The rows come
-- back either way, so this pins the plan.
SET enable_seqscan = off;
EXPLAIN (COSTS OFF)
    SELECT id FROM rerank_test
    ORDER BY v <-> rqv() LIMIT 10;
CREATE TEMP TABLE result_l2 AS
    SELECT id, v <-> rqv() AS dist
    FROM rerank_test
    ORDER BY v <-> rqv() LIMIT 10;
RESET enable_seqscan;

-- All ground truth top-10 must appear in index results
SELECT count(*) AS matching_ids
    FROM truth_l2 t JOIN result_l2 r USING (id);

-- Distances must match exactly (rerank uses full-precision vectors)
SELECT bool_and(abs(t.dist - r.dist) < 1e-5) AS distances_match
    FROM truth_l2 t JOIN result_l2 r USING (id);

-- Cosine distance
CREATE INDEX idx_rerank_cos ON rerank_test
    USING prism (v vec32_cosine_ops)
    WITH (centroid_compression = true);

SET enable_indexscan = off;
CREATE TEMP TABLE truth_cos AS
    SELECT id, v <=> rqv() AS dist
    FROM rerank_test
    ORDER BY dist LIMIT 10;
RESET enable_indexscan;

SET enable_seqscan = off;
CREATE TEMP TABLE result_cos AS
    SELECT id, v <=> rqv() AS dist
    FROM rerank_test
    ORDER BY v <=> rqv() LIMIT 10;
RESET enable_seqscan;

SELECT count(*) AS matching_ids
    FROM truth_cos t JOIN result_cos r USING (id);

SELECT bool_and(abs(t.dist - r.dist) < 1e-5) AS distances_match
    FROM truth_cos t JOIN result_cos r USING (id);

-- prism.rerank_pool: the plumbing, not the formula.
--
-- At this table size the survivor population is far below any pool the
-- formula produces, so the automatic cap never binds and this cannot pin it
-- (test/unit/test_rerank_pool.c does that). What it does cover is that the
-- GUC reaches the scan at all, that an explicit cap is honoured, and that
-- neither ever truncates below the query's k.

-- Candidates the index reported reranking, read back from EXPLAIN.
--
-- Raises when the counter is absent rather than returning NULL: every
-- comparison against NULL is NULL rather than false, so a probe that stopped
-- finding the field would pass while asserting nothing.
CREATE FUNCTION rerank_candidates(lim int) RETURNS int LANGUAGE plpgsql AS $$
DECLARE
    ej    json;
    cands int;
BEGIN
    EXECUTE format(
        'EXPLAIN (ANALYZE, VERBOSE, COSTS OFF, TIMING OFF, FORMAT JSON)
         SELECT id FROM rerank_test
         ORDER BY v <-> (SELECT v FROM rerank_test WHERE id = 42)
         LIMIT %s', lim)
    INTO ej;
    cands := ((ej->0->'Plan'->'Plans'->1->'Prism')->>'Rerank Candidates')::int;
    IF cands IS NULL THEN
        RAISE EXCEPTION 'Rerank Candidates not found in EXPLAIN output';
    END IF;
    RETURN cands;
END $$;

SET enable_seqscan = off;

-- The automatic pool never reranks fewer than k, or the result set would be
-- truncated.
SET prism.rerank_pool = 0;
SELECT rerank_candidates(10) >= 10 AS auto_pool_at_least_k;

-- An explicit cap is honoured, and still floored at k.
SET prism.rerank_pool = 12;
SELECT rerank_candidates(10) BETWEEN 10 AND 12 AS explicit_cap_honoured;

-- A cap below k must not truncate the result set.
SET prism.rerank_pool = 2;
SELECT count(*) AS rows_at_cap_below_k
    FROM (SELECT id FROM rerank_test
          ORDER BY v <-> (SELECT v FROM rerank_test WHERE id = 42)
          LIMIT 10) s;
RESET prism.rerank_pool;
RESET enable_seqscan;
DROP FUNCTION rerank_candidates(int);

-- Cleanup
DROP TABLE rerank_test;

DROP FUNCTION rqv();
