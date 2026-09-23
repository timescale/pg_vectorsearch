-- Auto-nlist row estimation on never-analyzed tables.
--
-- When a table has no reltuples yet, the build estimates the row count
-- by sampling heap pages, so the estimate must track the true count for
-- any column storage: PLAIN keeps full vectors in the main fork (a few
-- rows per page) while the default EXTENDED moves wide vectors to TOAST
-- (many small stubs per page). Each case builds once from the estimate,
-- then again after ANALYZE has recorded reltuples, and compares
-- the automatic cluster counts: a density estimate that is wrong for a
-- storage layout inflates or starves the partitioning by an order of
-- magnitude, while a correct one lands within k-means jitter.

CREATE TEMP TABLE cluster_counts (which text, n int);

-- Vectors are wide enough to pass the TOAST threshold and filled with
-- varied values so they stay incompressible (EXTENDED must actually
-- move them out of line).
-- autovacuum_enabled = off: the test exercises the estimator fallback
-- for tables with no reltuples, so an automatic ANALYZE sneaking in
-- between the INSERT and the first CREATE INDEX would silently bypass
-- the very path under test.
CREATE TABLE est_plain (id int, v vec32(512))
    WITH (autovacuum_enabled = off);
ALTER TABLE est_plain ALTER COLUMN v SET STORAGE PLAIN;
INSERT INTO est_plain
SELECT g,
       (SELECT '[' ||
               string_agg((((g * 7919 + i * 104729) % 997) / 10.0)::text,
                          ',' ORDER BY i) ||
               ']'
        FROM generate_series(1, 512) i)::vec32
FROM generate_series(1, 1000) g;

CREATE TABLE est_toast (id int, v vec32(512))
    WITH (autovacuum_enabled = off);
INSERT INTO est_toast SELECT id, v FROM est_plain;

-- The two layouts must genuinely differ in main-fork density.
SELECT pg_relation_size('est_plain') > 8 * pg_relation_size('est_toast')
       AS plain_is_larger;

-- Estimate-driven builds (no stats yet).
CREATE INDEX est_plain_idx ON est_plain USING prism (v vec32_l2_ops);
CREATE INDEX est_toast_idx ON est_toast USING prism (v vec32_l2_ops);
INSERT INTO cluster_counts
SELECT 'plain_est', count(DISTINCT cluster_id)
FROM prism_posting_pages('est_plain_idx');
INSERT INTO cluster_counts
SELECT 'toast_est', count(DISTINCT cluster_id)
FROM prism_posting_pages('est_toast_idx');

-- Stats-driven rebuilds.
ANALYZE est_plain;
ANALYZE est_toast;
-- squawk-ignore require-concurrent-reindex
REINDEX INDEX est_plain_idx;
-- squawk-ignore require-concurrent-reindex
REINDEX INDEX est_toast_idx;
INSERT INTO cluster_counts
SELECT 'plain_true', count(DISTINCT cluster_id)
FROM prism_posting_pages('est_plain_idx');
INSERT INTO cluster_counts
SELECT 'toast_true', count(DISTINCT cluster_id)
FROM prism_posting_pages('est_toast_idx');

SELECT (SELECT n FROM cluster_counts WHERE which = 'plain_est')::float
       / (SELECT n FROM cluster_counts WHERE which = 'plain_true')
       BETWEEN 0.5 AND 2.0 AS plain_estimate_tracks;
SELECT (SELECT n FROM cluster_counts WHERE which = 'toast_est')::float
       / (SELECT n FROM cluster_counts WHERE which = 'toast_true')
       BETWEEN 0.5 AND 2.0 AS toast_estimate_tracks;

DROP TABLE est_plain;
DROP TABLE est_toast;
