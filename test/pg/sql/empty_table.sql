-- CREATE INDEX on tables with no indexable rows.
--
-- The build substitutes one synthetic sample when the heap yields none,
-- emitting a valid single-cluster index: scans on it return nothing,
-- later inserts route into it like any single-cluster index, and
-- REINDEX after loading re-clusters for real. Covers the truly empty
-- table, a non-empty heap whose rows are all dead, and an all-NULL
-- column.

SET enable_seqscan = off;

-- Truly empty table (cosine).
CREATE TABLE empty_cos (id int, v vector(8));
CREATE INDEX empty_cos_idx ON empty_cos USING mktann (v vector_cosine_ops);
SELECT id FROM empty_cos ORDER BY v OPERATOR(mkt.<=>) '[1,0,0,0,0,0,0,0]' LIMIT 5;

-- Inserts route into the degenerate index and are found.
INSERT INTO empty_cos
SELECT g, ('[' || g || ',1,0,0,0,0,0,0]')::vector
FROM generate_series(1, 50) g;
SELECT id FROM empty_cos ORDER BY v OPERATOR(mkt.<=>) '[10,1,0,0,0,0,0,0]' LIMIT 3;

-- REINDEX with rows present re-clusters and preserves results.
-- squawk-ignore require-concurrent-reindex
REINDEX INDEX empty_cos_idx;
SELECT id FROM empty_cos ORDER BY v OPERATOR(mkt.<=>) '[10,1,0,0,0,0,0,0]' LIMIT 3;

-- Non-empty heap, zero live rows (l2).
CREATE TABLE all_dead (id int, v vector(8));
INSERT INTO all_dead
SELECT g, ('[' || g || ',0,0,0,0,0,0,0]')::vector
FROM generate_series(1, 50) g;
DELETE FROM all_dead;
CREATE INDEX all_dead_idx ON all_dead USING mktann (v vector_l2_ops);
SELECT id FROM all_dead ORDER BY v OPERATOR(mkt.<->) '[1,0,0,0,0,0,0,0]' LIMIT 5;
INSERT INTO all_dead VALUES (1, '[2,0,0,0,0,0,0,0]');
SELECT id FROM all_dead ORDER BY v OPERATOR(mkt.<->) '[1,0,0,0,0,0,0,0]' LIMIT 5;

-- All rows NULL in the indexed column.
CREATE TABLE all_null (id int, v vector(8));
INSERT INTO all_null SELECT g, NULL FROM generate_series(1, 20) g;
CREATE INDEX all_null_idx ON all_null USING mktann (v vector_cosine_ops);
SELECT id FROM all_null WHERE v IS NOT NULL
ORDER BY v OPERATOR(mkt.<=>) '[1,0,0,0,0,0,0,0]' LIMIT 5;

-- Maintenance on a degenerate index.
VACUUM empty_cos;
VACUUM all_dead;

-- An index grown from empty to many tuples keeps answering queries, and
-- REINDEX re-optimizes it: the single degenerate cluster becomes a real
-- k-means partitioning and the insert-appended AoS pages become fastscan.
CREATE TABLE grow (id int, v vector(8));
CREATE INDEX grow_idx ON grow USING mktann (v vector_l2_ops);
INSERT INTO grow
SELECT g, ('[' || g || ',1,0,0,0,0,0,0]')::vector
FROM generate_series(1, 2000) g;
SELECT count(DISTINCT cluster_id) AS nclusters,
       sum(entry_count) AS entries,
       bool_or(format = 'aos') AS has_aos_pages
FROM mkt.posting_pages('grow_idx');
SELECT id FROM grow
ORDER BY v OPERATOR(mkt.<->) '[1000.4,1,0,0,0,0,0,0]' LIMIT 3;
-- squawk-ignore require-concurrent-reindex
REINDEX INDEX grow_idx;
SELECT count(DISTINCT cluster_id) BETWEEN 20 AND 100 AS reclustered,
       bool_and(format = 'fastscan') AS all_fastscan,
       sum(entry_count) >= 2000 AS entries_ok
FROM mkt.posting_pages('grow_idx');
SELECT id FROM grow
ORDER BY v OPERATOR(mkt.<->) '[1000.4,1,0,0,0,0,0,0]' LIMIT 3;

-- VACUUM FULL rebuilds the index and re-optimizes the same way.
CREATE TABLE grow2 (id int, v vector(8));
CREATE INDEX grow2_idx ON grow2 USING mktann (v vector_l2_ops);
INSERT INTO grow2
SELECT g, ('[' || g || ',1,0,0,0,0,0,0]')::vector
FROM generate_series(1, 2000) g;
VACUUM FULL grow2;
SELECT count(DISTINCT cluster_id) BETWEEN 20 AND 100 AS reclustered,
       bool_and(format = 'fastscan') AS all_fastscan,
       sum(entry_count) >= 2000 AS entries_ok
FROM mkt.posting_pages('grow2_idx');
SELECT id FROM grow2
ORDER BY v OPERATOR(mkt.<->) '[1000.4,1,0,0,0,0,0,0]' LIMIT 3;

RESET enable_seqscan;
DROP TABLE empty_cos;
DROP TABLE all_dead;
DROP TABLE all_null;
DROP TABLE grow;
DROP TABLE grow2;
