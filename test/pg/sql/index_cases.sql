-- Shared build, query, and maintenance cases. The caller sets dim,
-- reference_pages, mean_pages, spilled, and expected_magic.
CREATE TABLE dimtest (id int, v vec32(:dim));
INSERT INTO dimtest
    SELECT g, (SELECT array_agg(sin(g * 0.37 + j * 0.11)::real)
               FROM generate_series(0, :dim - 1) j)::vec32
    FROM generate_series(1, 600) g;

SET max_parallel_maintenance_workers = 0;
CREATE INDEX dimtest_idx ON dimtest USING prism (v)
    WITH (nlist = 4, centroid_fastscan = off);
SELECT name, setting FROM prism_index_settings('dimtest_idx')
    WHERE name IN ('dim', 'nlist', 'fastscan', 'centroid_format')
    ORDER BY name;
SELECT to_hex(meta_magic('dimtest_idx')) = :'expected_magic' AS correct_magic;
SELECT (SELECT setting::int FROM prism_index_settings('dimtest_idx')
         WHERE name = 'first_posting') -
       (SELECT setting::int FROM prism_index_settings('dimtest_idx')
         WHERE name = 'centroid_pages') = 1 + :mean_pages AS centroid_start;

SELECT count(*) = :reference_pages *
           (SELECT setting::int FROM prism_index_settings('dimtest_idx')
            WHERE name = 'nlist') AS reference_count,
       coalesce(bool_and(entry_count = 0 AND max_entries = 0), true) AS empty,
       coalesce(bool_and(chain_pos < :reference_pages), true)
           AS lead_the_chain,
       coalesce(bool_and(is_first = (chain_pos = 0)), true) AS head_first
    FROM prism_posting_pages('dimtest_idx') WHERE format = 'reference';
SELECT bool_and(chain_pos >= :reference_pages) AS entries_follow
    FROM prism_posting_pages('dimtest_idx') WHERE format <> 'reference';

-- A table-reading SQL function keeps InitPlan numbering out of EXPLAIN.
CREATE FUNCTION dimtest_query(int) RETURNS vec32 LANGUAGE sql STABLE AS $$
    SELECT v FROM dimtest WHERE id = $1
$$;
SET enable_seqscan = off;
EXPLAIN (COSTS OFF)
SELECT id FROM dimtest ORDER BY v <-> dimtest_query(7) LIMIT 5;
SELECT id FROM dimtest ORDER BY v <-> dimtest_query(7) LIMIT 1;
SELECT id FROM dimtest ORDER BY v <-> dimtest_query(599) LIMIT 1;
SET prism.nprobe = 2;
SELECT id FROM dimtest ORDER BY v <-> dimtest_query(7) LIMIT 1;
RESET prism.nprobe;

INSERT INTO dimtest
    SELECT g, (SELECT array_agg(sin(g * 0.37 + j * 0.11)::real)
               FROM generate_series(0, :dim - 1) j)::vec32
    FROM generate_series(601, 640) g;
SELECT id FROM dimtest ORDER BY v <-> dimtest_query(640) LIMIT 1;
DELETE FROM dimtest WHERE id <= 500;
VACUUM dimtest;
SELECT count(*) AS tombstoned_references
    FROM prism_posting_pages('dimtest_idx')
    WHERE format = 'reference' AND tombstoned;
SELECT id FROM dimtest ORDER BY v <-> dimtest_query(520) LIMIT 1;

DROP INDEX dimtest_idx;
TRUNCATE dimtest;
INSERT INTO dimtest
    SELECT g, (SELECT array_agg(sin(g * 0.37 + j * 0.11)::real)
               FROM generate_series(0, :dim - 1) j)::vec32
    FROM generate_series(1, 600) g;
CREATE INDEX dimtest_idx ON dimtest USING prism (v)
    WITH (nlist = 1, centroid_fastscan = off);
SET client_min_messages = warning;
CALL prism_rebalance('dimtest_idx', 100);
RESET client_min_messages;
SELECT setting::int > 1 AS split_into_several_lists
    FROM prism_index_settings('dimtest_idx') WHERE name = 'nlist';
SELECT bool_and((format = 'reference') = :spilled) AS head_format_matches
    FROM prism_posting_pages('dimtest_idx') WHERE is_first AND NOT tombstoned;
SELECT id FROM dimtest ORDER BY v <-> dimtest_query(300) LIMIT 1;

-- squawk-ignore require-concurrent-reindex
REINDEX INDEX dimtest_idx;
SELECT to_hex(meta_magic('dimtest_idx')) = :'expected_magic' AS correct_magic;
SELECT id FROM dimtest ORDER BY v <-> dimtest_query(42) LIMIT 1;
DROP INDEX dimtest_idx;
ALTER TABLE dimtest SET (parallel_workers = 2);
SET max_parallel_maintenance_workers = 2;
SET min_parallel_table_scan_size = 0;
CREATE INDEX dimtest_idx ON dimtest USING prism (v) WITH (nlist = 4);
SELECT id FROM dimtest ORDER BY v <-> dimtest_query(42) LIMIT 1;
RESET max_parallel_maintenance_workers;
RESET min_parallel_table_scan_size;
ALTER TABLE dimtest RESET (parallel_workers);

CREATE INDEX dimtest_cos ON dimtest USING prism (v vec32_cosine_ops)
    WITH (nlist = 4);
EXPLAIN (COSTS OFF)
SELECT id FROM dimtest ORDER BY v <=> dimtest_query(7) LIMIT 1;
SELECT id FROM dimtest ORDER BY v <=> dimtest_query(7) LIMIT 1;
DROP INDEX dimtest_cos;

CREATE TABLE dimhalf (id int, v vec16(:dim));
INSERT INTO dimhalf SELECT id, v::vec16 FROM dimtest WHERE id <= 200;
CREATE INDEX dimhalf_idx ON dimhalf USING prism (v) WITH (nlist = 2);
CREATE FUNCTION dimhalf_query(int) RETURNS vec16 LANGUAGE sql STABLE AS $$
    SELECT v FROM dimhalf WHERE id = $1
$$;
EXPLAIN (COSTS OFF)
SELECT id FROM dimhalf ORDER BY v <-> dimhalf_query(150) LIMIT 1;
SELECT id FROM dimhalf ORDER BY v <-> dimhalf_query(150) LIMIT 1;
RESET enable_seqscan;

DROP FUNCTION dimhalf_query(int);
DROP FUNCTION dimtest_query(int);
DROP TABLE dimhalf;
DROP TABLE dimtest;
