-- Recall tests using GloVe-100 sample data
--
-- Loads 1000 vectors from glove-100-angular, builds mktann indexes,
-- and compares index scan results against precomputed ground truth.

-- ================================================================
-- Load data
-- ================================================================

CREATE TABLE glove (id serial PRIMARY KEY, v vector(100));
CREATE TABLE glove_queries (id serial PRIMARY KEY, v vector(100));
CREATE TABLE glove_truth_cosine (qid int, ids int[]);
CREATE TABLE glove_truth_l2 (qid int, ids int[]);

\copy glove (v) FROM PROGRAM 'gunzip -c ../test/pg/data/glove_vectors.csv.gz'
\copy glove_queries (v) FROM '../test/pg/data/glove_queries.csv'
\copy glove_truth_cosine FROM '../test/pg/data/glove_truth_cosine.csv'
\copy glove_truth_l2 FROM '../test/pg/data/glove_truth_l2.csv'

ANALYZE glove;
ANALYZE glove_queries;

SELECT count(*) AS vectors FROM glove;
SELECT count(*) AS queries FROM glove_queries;

-- ================================================================
-- Cosine recall
-- ================================================================

CREATE INDEX idx_glove_cos ON glove
    USING mktann (v vector_cosine_ops);

-- Verify index scan is used for cosine ORDER BY
SET enable_seqscan = off;
EXPLAIN (COSTS OFF)
SELECT id FROM glove ORDER BY v <=> (SELECT v FROM glove_queries WHERE id = 1) LIMIT 10;

-- Compute index scan results
CREATE TEMP TABLE idx_cos AS
    SELECT q.id AS qid, array_agg(r.id) AS ids
    FROM glove_queries q
    CROSS JOIN LATERAL (
        SELECT id FROM glove ORDER BY v <=> q.v LIMIT 10
    ) r
    GROUP BY q.id;
RESET enable_seqscan;

-- Compare ground truth vs index top-10 for each query
SELECT gt.qid,
       gt.ids AS truth,
       i.ids AS index,
       (SELECT count(*) FROM unnest(gt.ids) gid
        WHERE gid = ANY(i.ids))::int AS hits
FROM glove_truth_cosine gt
JOIN idx_cos i ON gt.qid = i.qid
ORDER BY gt.qid;

-- Results should be ordered by distance
SET enable_seqscan = off;
SELECT bool_and(dist <= next_dist) AS cosine_ordered FROM (
    SELECT dist, lead(dist) OVER () AS next_dist FROM (
        SELECT v <=> (SELECT v FROM glove WHERE id = 1) AS dist
        FROM glove
        ORDER BY v <=> (SELECT v FROM glove WHERE id = 1)
        LIMIT 20
    ) t
) t2 WHERE next_dist IS NOT NULL;
RESET enable_seqscan;

DROP INDEX idx_glove_cos;

-- ================================================================
-- L2 recall
-- ================================================================

CREATE INDEX idx_glove_l2 ON glove
    USING mktann (v);

-- Verify index scan is used for L2 ORDER BY
SET enable_seqscan = off;
EXPLAIN (COSTS OFF)
SELECT id FROM glove ORDER BY v <-> (SELECT v FROM glove_queries WHERE id = 1) LIMIT 10;

-- Compute L2 index scan results
CREATE TEMP TABLE idx_l2 AS
    SELECT q.id AS qid, array_agg(r.id) AS ids
    FROM glove_queries q
    CROSS JOIN LATERAL (
        SELECT id FROM glove ORDER BY v <-> q.v LIMIT 10
    ) r
    GROUP BY q.id;

-- Compare L2 ground truth vs index top-10 for each query
SELECT gt.qid,
       gt.ids AS truth,
       i.ids AS index,
       (SELECT count(*) FROM unnest(gt.ids) gid
        WHERE gid = ANY(i.ids))::int AS hits
FROM glove_truth_l2 gt
JOIN idx_l2 i ON gt.qid = i.qid
ORDER BY gt.qid;

-- Self-query should return itself as top-1
SELECT id = 1 AS l2_found_self FROM (
    SELECT id FROM glove
    ORDER BY v <-> (SELECT v FROM glove WHERE id = 1)
    LIMIT 1
) t;

RESET enable_seqscan;

-- ================================================================
-- Cleanup
-- ================================================================

DROP TABLE glove_truth_cosine;
DROP TABLE glove_truth_l2;
DROP TABLE glove_queries;
DROP TABLE glove;
