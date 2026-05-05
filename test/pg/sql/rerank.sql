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

-- Cleanup
DROP TABLE rerank_test;
