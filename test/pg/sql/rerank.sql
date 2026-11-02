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

-- A rerank reads only rows the scan can see. A vector too wide for its heap
-- page lives in TOAST, and a dead row's chunks can be gone before the row
-- itself is: the delete, or the abort of the insert, frees them, a VACUUM of
-- the toast table reclaims them, and the heap tuple stays until its page is
-- pruned. Fetching that tuple's vector would fail on a missing chunk.
CREATE TABLE rerank_toast (id serial, v vec32(1024));
INSERT INTO rerank_toast (v)
    SELECT (
        SELECT array_agg(sin(i * 0.1 + j * 0.7)::real)
        FROM generate_series(0, 1023) j
    )::vec32(1024)
    FROM generate_series(1, 100) i;
CREATE INDEX idx_rerank_toast ON rerank_toast USING prism (v)
    WITH (nlist = 1, centroid_compression = true);

-- Built inside a DO block and in plpgsql, so neither the echo nor the
-- planner's inlining puts a thousand floats into the output; the scan still
-- evaluates it once.
DO $$
BEGIN
    EXECUTE format($f$
        CREATE FUNCTION tqv() RETURNS vec32(1024)
            LANGUAGE plpgsql STABLE PARALLEL SAFE
            AS $b$ BEGIN RETURN %L::vec32(1024); END $b$
    $f$, (SELECT v::text FROM rerank_toast WHERE id = 42));
END $$;

-- The nearest row is deleted, an identical one is inserted and rolled back,
-- and only the toast table is vacuumed: both heap tuples are still there for
-- the index entries to point at, their chunks are not.
DELETE FROM rerank_toast WHERE id = 42;
BEGIN;
INSERT INTO rerank_toast (v) VALUES (tqv());
ROLLBACK;
SELECT reltoastrelid::regclass AS toastrel FROM pg_class
    WHERE oid = 'rerank_toast'::regclass \gset
VACUUM :toastrel;

SET enable_seqscan = off;
EXPLAIN (COSTS OFF)
    SELECT id FROM rerank_toast ORDER BY v <-> tqv() LIMIT 3;
SELECT id FROM rerank_toast ORDER BY v <-> tqv() LIMIT 3;
RESET enable_seqscan;

DROP TABLE rerank_toast;
DROP FUNCTION tqv();

-- Cleanup
DROP TABLE rerank_test;

DROP FUNCTION rqv();
