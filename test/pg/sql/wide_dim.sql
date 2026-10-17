-- ============================================================
-- Wide dimensions: the spilled reference layout
-- ============================================================
-- Past 1968 dimensions a posting list's float encode reference no longer
-- fits beside an entry on its head page, so the head carries only the first
-- chunk of it, further REFERENCE pages carry the rest (three in all at 4096),
-- and the entries start after them. The metapage's global mean spills the
-- same way into the blocks right after block 0. Everything below is the same
-- index behaviour as at a narrow dimension, exercised at 4096 so the gather
-- paths, not the inline ones, answer.
SET maintenance_work_mem = '512MB';

CREATE TABLE wd (id int, v vec32(4096));
INSERT INTO wd
    SELECT g, (SELECT array_agg(sin(g * 0.37 + j * 0.11)::real)
               FROM generate_series(0, 4095) j)::vec32(4096)
    FROM generate_series(1, 600) g;

-- Serial build; fastscan resolves to off on its own (no group fits a page).
CREATE INDEX wd_idx ON wd USING prism (v) WITH (nlist = 4);
SELECT name, setting FROM prism_index_settings('wd_idx')
    WHERE name IN ('dim', 'nlist', 'fastscan', 'centroid_format',
                   'first_posting', 'posting_pages')
    ORDER BY name;

-- Block 0 is the metapage, blocks 1..3 the mean's reference pages, so the
-- centroid region starts at block 4 and runs to first_posting.
SELECT (SELECT setting::int FROM prism_index_settings('wd_idx')
         WHERE name = 'first_posting') -
       (SELECT setting::int FROM prism_index_settings('wd_idx')
         WHERE name = 'centroid_pages') = 4 AS centroids_start_at_block_4;

-- Every list opens with three reference pages: no entries, no capacity,
-- chain positions 0..2, and nothing else is a reference page.
SELECT count(*) = 3 * (SELECT setting::int FROM prism_index_settings('wd_idx')
                        WHERE name = 'nlist')       AS three_per_list,
       bool_and(entry_count = 0 AND max_entries = 0) AS empty,
       bool_and(chain_pos < 3)                       AS lead_the_chain,
       bool_and(is_first = (chain_pos = 0))          AS head_first
    FROM prism_posting_pages('wd_idx') WHERE format = 'reference';
SELECT bool_and(chain_pos >= 3) AS entries_follow
    FROM prism_posting_pages('wd_idx') WHERE format <> 'reference';

-- The planner uses the index, and a row's own vector finds the row.
SET enable_seqscan = off;
EXPLAIN (COSTS OFF)
SELECT id FROM wd ORDER BY v <-> (SELECT v FROM wd WHERE id = 7) LIMIT 5;
SELECT id FROM wd ORDER BY v <-> (SELECT v FROM wd WHERE id = 7) LIMIT 1;
SELECT id FROM wd ORDER BY v <-> (SELECT v FROM wd WHERE id = 599) LIMIT 1;
-- With nprobe below the routed set the probe re-rank runs: each routed
-- list's reference is gathered once and reused by the scan pass.
SET prism.nprobe = 2;
SELECT id FROM wd ORDER BY v <-> (SELECT v FROM wd WHERE id = 7) LIMIT 1;
RESET prism.nprobe;

-- Insert: the reference is gathered from the chain and the entry appended
-- after it.
INSERT INTO wd
    SELECT g, (SELECT array_agg(sin(g * 0.37 + j * 0.11)::real)
               FROM generate_series(0, 4095) j)::vec32(4096)
    FROM generate_series(601, 640) g;
SELECT id FROM wd ORDER BY v <-> (SELECT v FROM wd WHERE id = 640) LIMIT 1;

-- Delete most of the rows and vacuum: tombstoning never touches a reference
-- page, and the survivors still answer.
DELETE FROM wd WHERE id <= 500;
VACUUM wd;
SELECT count(*) AS tombstoned_reference_pages
    FROM prism_posting_pages('wd_idx') WHERE format = 'reference' AND tombstoned;
SELECT id FROM wd ORDER BY v <-> (SELECT v FROM wd WHERE id = 520) LIMIT 1;

-- Split and rebalance: new lists get spilled heads of their own. The split's
-- k-means scratch is sized from maintenance_work_mem, which at this width
-- needs well over the 64 MB default.
DROP INDEX wd_idx;
DELETE FROM wd;
INSERT INTO wd
    SELECT g, (SELECT array_agg(sin(g * 0.37 + j * 0.11)::real)
               FROM generate_series(0, 4095) j)::vec32(4096)
    FROM generate_series(1, 600) g;
CREATE INDEX wd_idx ON wd USING prism (v) WITH (nlist = 1);
CALL prism_rebalance('wd_idx', 100);
SELECT setting::int > 1 AS split_into_several_lists
    FROM prism_index_settings('wd_idx') WHERE name = 'nlist';
SELECT bool_and(entry_count = 0 AND max_entries = 0) AS new_heads_are_references
    FROM prism_posting_pages('wd_idx') WHERE is_first AND NOT tombstoned;
SELECT id FROM wd ORDER BY v <-> (SELECT v FROM wd WHERE id = 300) LIMIT 1;

-- REINDEX and a parallel build both lay the strided head region out again.
-- squawk-ignore require-concurrent-reindex
REINDEX INDEX wd_idx;
SELECT id FROM wd ORDER BY v <-> (SELECT v FROM wd WHERE id = 42) LIMIT 1;
DROP INDEX wd_idx;
ALTER TABLE wd SET (parallel_workers = 2);
SET max_parallel_maintenance_workers = 2;
SET min_parallel_table_scan_size = 0;
CREATE INDEX wd_idx ON wd USING prism (v) WITH (nlist = 4);
SELECT id FROM wd ORDER BY v <-> (SELECT v FROM wd WHERE id = 42) LIMIT 1;
RESET max_parallel_maintenance_workers;
RESET min_parallel_table_scan_size;
ALTER TABLE wd RESET (parallel_workers);

-- Cosine routes on compressed centroids as well, and a vec16 column halves
-- the heap the rerank reads.
CREATE INDEX wd_cos ON wd USING prism (v vec32_cosine_ops) WITH (nlist = 4);
EXPLAIN (COSTS OFF)
SELECT id FROM wd ORDER BY v <=> (SELECT v FROM wd WHERE id = 7) LIMIT 1;
SELECT id FROM wd ORDER BY v <=> (SELECT v FROM wd WHERE id = 7) LIMIT 1;
DROP INDEX wd_cos;
CREATE TABLE wh (id int, v vec16(4096));
INSERT INTO wh SELECT id, v::vec16(4096) FROM wd WHERE id <= 200;
CREATE INDEX wh_idx ON wh USING prism (v) WITH (nlist = 2);
EXPLAIN (COSTS OFF)
SELECT id FROM wh ORDER BY v <-> (SELECT v FROM wh WHERE id = 150) LIMIT 1;
SELECT id FROM wh ORDER BY v <-> (SELECT v FROM wh WHERE id = 150) LIMIT 1;
RESET enable_seqscan;

-- What does not work at this width, each refused up front with the reason.
-- Uncompressed centroids are whole vectors on a centroid page and no longer
-- fit one, which is what inner product routes on (float16 fits up to 4072).
CREATE INDEX wd_ip ON wd USING prism (v vec32_ip_ops);
CREATE INDEX wd_off ON wd USING prism (v) WITH (centroid_compression = off);
CREATE INDEX wh_ip ON wh USING prism (v vec16_ip_ops);
-- No fastscan group fits a page either, at build or by conversion later.
CREATE INDEX wd_fs ON wd USING prism (v) WITH (fastscan = on);
CALL prism_convert_posting_to_fastscan('wd_idx', 0);
-- Past the inline threshold the dimension must suit the fast rotation, and
-- 8192 is the ceiling.
CREATE TABLE wbad (v vec32(3000));
CREATE INDEX wbad_idx ON wbad USING prism (v);
DROP TABLE wbad;
CREATE TABLE wbad (v vec32(8193));
CREATE INDEX wbad_idx ON wbad USING prism (v);
DROP TABLE wbad;
-- The dense params generator keeps its own, narrower cap.
SELECT rabitq_params_generate(4096, 42);

DROP TABLE wh;
DROP TABLE wd;
RESET maintenance_work_mem;
