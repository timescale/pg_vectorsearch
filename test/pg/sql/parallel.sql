-- Parallel vs serial index build paths
--
-- Exercises the mktann parallel build (do_parallel_build) and serial build
-- (do_serial_build) and asserts both produce a usable index. A small table is
-- enough: setting the table's parallel_workers storage parameter forces a
-- parallel build regardless of heap size (it bypasses the size/memory
-- heuristics in plan_create_index_workers), so we get parallel coverage
-- without a slow, large dataset.
--
-- The parallel path runs the full option matrix (it is otherwise untested);
-- the serial path runs the options no other regression test covers serially
-- (fastscan / SOAR / boundary), since mktann.sql already covers serial
-- L2/cosine/float/multi-level builds.

CREATE TABLE embeddings (id serial, v vector(3));

INSERT INTO embeddings (v)
    SELECT format('[%s,%s,%s]', x * 0.1, y * 0.1, z * 0.1)::vector
    FROM generate_series(0, 9) x,
         generate_series(0, 9) y,
         generate_series(0, 9) z;

SELECT count(*) FROM embeddings;

-- Build the index from a passed definition tail, verify it has pages and that
-- an index-only ORDER BY scan returns the requested LIMIT, then drop it. The
-- order-by operator is a parameter so cosine (<=>) and L2 (<->) opclasses can
-- share the helper.
CREATE FUNCTION pbuild_check(idxdef text, op text DEFAULT '<->')
    RETURNS TABLE (has_pages bool, nres bigint)
    LANGUAGE plpgsql AS $$
BEGIN
    EXECUTE 'CREATE INDEX tmp_idx ON embeddings USING mktann ' || idxdef;
    RETURN QUERY EXECUTE format(
        'SELECT (SELECT c.relpages > 0 FROM pg_class c'
        '          WHERE c.relname = ''tmp_idx''),'
        '       (SELECT count(*) FROM ('
        '          SELECT id FROM embeddings'
        '          ORDER BY v %s ''[0.5,0.5,0.5]'' LIMIT 5) t)',
        op);
    EXECUTE 'DROP INDEX tmp_idx';
END $$;

SET enable_seqscan = off;

-- ============================================================
-- Parallel build path (full option matrix)
-- ============================================================
-- The parallel_workers storage parameter is honored directly by
-- plan_create_index_workers (capped at max_parallel_maintenance_workers),
-- which is how we drive the parallel path on a tiny table.
SET max_parallel_maintenance_workers = 2;
ALTER TABLE embeddings SET (parallel_workers = 2);
SELECT reloptions FROM pg_class WHERE relname = 'embeddings';

-- L2, RaBitQ centroids
SELECT * FROM pbuild_check('(v) WITH (centroid_compression = true)');
-- L2, float centroids
SELECT * FROM pbuild_check('(v) WITH (centroid_compression = off)');
-- Cosine
SELECT * FROM pbuild_check(
    '(v vector_cosine_ops) WITH (centroid_compression = off)', '<=>');
-- Fastscan posting lists
SELECT * FROM pbuild_check('(v) WITH (fastscan = true)');
-- Multi-level tree
SELECT * FROM pbuild_check('(v) WITH (fan_out = 4, centroid_compression = true)');
-- SOAR replication
SELECT * FROM pbuild_check('(v) WITH (soar_lambda = 1.0)');
-- Boundary replication
SELECT * FROM pbuild_check('(v) WITH (boundary_epsilon = 0.1)');

-- ============================================================
-- Serial build path (options not covered serially elsewhere)
-- ============================================================
-- max_parallel_maintenance_workers = 0 forces the serial fallback even though
-- the table still carries parallel_workers = 2.
SET max_parallel_maintenance_workers = 0;

SELECT * FROM pbuild_check('(v) WITH (fastscan = true)');
SELECT * FROM pbuild_check('(v) WITH (soar_lambda = 1.0)');
SELECT * FROM pbuild_check('(v) WITH (boundary_epsilon = 0.1)');

RESET enable_seqscan;

-- ============================================================
-- Sample budget bound by maintenance_work_mem
-- ============================================================
-- The k-means sample set is held in one shared-memory region and bounded by
-- maintenance_work_mem. A low setting on a 768-dim table forces the stride
-- sampler to draw a coarser (uniform) subsample to fit (~341 samples at 1MB,
-- below the row count); the build must still succeed and produce a usable
-- index.
CREATE TABLE wide_emb (id serial, v vector(768));
INSERT INTO wide_emb (v)
    SELECT (
        SELECT array_agg((sin(i * 0.1 + j))::real)
        FROM generate_series(0, 767) j
    )::vector(768)
    FROM generate_series(1, 600) i;
ALTER TABLE wide_emb SET (parallel_workers = 2);
SET max_parallel_maintenance_workers = 2;
SET maintenance_work_mem = '1MB';

CREATE INDEX idx_wide_budget ON wide_emb USING mktann (v)
    WITH (centroid_compression = true);

SET enable_seqscan = off;
SELECT (SELECT relpages > 0 FROM pg_class WHERE relname = 'idx_wide_budget')
           AS has_pages,
       (SELECT count(*) FROM (
            SELECT id FROM wide_emb
            ORDER BY v <-> (SELECT v FROM wide_emb WHERE id = 1) LIMIT 5) t)
           AS nres;
RESET enable_seqscan;

RESET maintenance_work_mem;
DROP TABLE wide_emb;

-- ============================================================
-- Edge case: parallel build of an empty table must not crash
-- ============================================================
CREATE TABLE empty_emb (id serial, v vector(3));
ALTER TABLE empty_emb SET (parallel_workers = 2);
SET max_parallel_maintenance_workers = 2;
CREATE INDEX idx_empty ON empty_emb USING mktann (v);
SELECT count(*) FROM empty_emb;
DROP TABLE empty_emb;

RESET max_parallel_maintenance_workers;

-- ============================================================
-- Parallel build refreshes the table's row-count statistics
-- ============================================================
-- An index build scans every heap tuple, so index_update_stats writes the
-- count the access method returns into pg_class.reltuples: a build on a
-- never-analyzed table must SET the count, and a rebuild on an analyzed
-- table must PRESERVE it. A parallel build that reports zero instead
-- clobbers the planner's row estimate (and everything derived from
-- reltuples) until the next ANALYZE.
CREATE TABLE relstats (id serial, v vector(3));
INSERT INTO relstats (v)
    SELECT format('[%s,%s,%s]', x * 0.1, x * 0.2, x * 0.3)::vector
    FROM generate_series(1, 1000) x;
ALTER TABLE relstats SET (parallel_workers = 2);
SET max_parallel_maintenance_workers = 2;

-- Never analyzed (reltuples = -1): the build supplies the count.
CREATE INDEX relstats_idx ON relstats USING mktann (v);
SELECT reltuples::bigint AS reltuples_never_analyzed
    FROM pg_class WHERE relname = 'relstats';
DROP INDEX relstats_idx;

-- Analyzed: a rebuild must not destroy the existing estimate.
ANALYZE relstats;
CREATE INDEX relstats_idx ON relstats USING mktann (v);
SELECT reltuples::bigint AS reltuples_after_rebuild
    FROM pg_class WHERE relname = 'relstats';

RESET max_parallel_maintenance_workers;
DROP TABLE relstats;

-- ============================================================
-- Exactness: every build path returns the exact top-k
-- ============================================================
-- A smoke check (has_pages + returns LIMIT) cannot catch a build that produces
-- a *wrong* index. Here the data is well-separated so the 10 nearest neighbors
-- of the probe are unambiguous, and the query uses nprobe >= nlist (probe every
-- cluster) so the result depends only on the index contents, not on which lists
-- happened to be probed nor on how clustering placed the centroids. Each build
-- path must therefore return exactly the seqscan top-10.
--
-- exact_check builds the index from a definition tail, collects the index
-- top-10 (forcing an index scan, probing all lists) and the seqscan top-10 for
-- the same probe, and returns whether the two id sets match.
CREATE FUNCTION exact_check(tbl text, idxdef text, q text,
                            op text DEFAULT '<->')
    RETURNS bool LANGUAGE plpgsql AS $$
DECLARE
    idx_ids int[];
    seq_ids int[];
BEGIN
    EXECUTE format('CREATE INDEX ex_idx ON %I USING mktann %s', tbl, idxdef);
    SET LOCAL enable_seqscan = off;
    SET LOCAL mkt.nprobe = 10000; -- >= nlist for these tables: probe all
    EXECUTE format(
        'SELECT array_agg(id ORDER BY id) FROM '
        '(SELECT id FROM %I ORDER BY v %s %L LIMIT 10) t', tbl, op, q)
        INTO idx_ids;
    SET LOCAL enable_seqscan = on;
    SET LOCAL enable_indexscan = off;
    EXECUTE format(
        'SELECT array_agg(id ORDER BY id) FROM '
        '(SELECT id FROM %I ORDER BY v %s %L LIMIT 10) t', tbl, op, q)
        INTO seq_ids;
    SET LOCAL enable_indexscan = on;
    EXECUTE 'DROP INDEX ex_idx';
    RETURN idx_ids = seq_ids;
END $$;

-- 50 well-separated points on a line; the 10 nearest to [0.5,0,0] are ids 1..10
-- (strictly increasing distance, gaps of 1.0), so quantization cannot reorder
-- the top-10.
CREATE TABLE line3 (id int, v vector(3));
INSERT INTO line3 SELECT g, format('[%s,0,0]', g)::vector
    FROM generate_series(1, 50) g;
ALTER TABLE line3 SET (parallel_workers = 2);

SET max_parallel_maintenance_workers = 2;
SELECT exact_check('line3', '(v) WITH (centroid_compression = true)',
                   '[0.5,0,0]') AS parallel_exact;
SELECT exact_check('line3', '(v) WITH (fan_out = 4, soar_lambda = 1.0)',
                   '[0.5,0,0]') AS parallel_soar_exact;
SELECT exact_check('line3', '(v) WITH (fastscan = true)',
                   '[0.5,0,0]') AS parallel_fastscan_exact;
SET max_parallel_maintenance_workers = 0;
SELECT exact_check('line3', '(v) WITH (centroid_compression = true)',
                   '[0.5,0,0]') AS serial_exact;
SELECT exact_check('line3', '(v) WITH (fastscan = true, soar_lambda = 1.0)',
                   '[0.5,0,0]') AS serial_soar_exact;

-- Worker shortfall: the number of workers that actually launch is capped by
-- the free slots under max_parallel_workers at CREATE INDEX time, which can
-- be below the planned count. The build must produce the same tree shape
-- with however many workers attach: every root child subtree clustered and
-- written exactly once. max_parallel_workers is session-settable, so setting
-- it below the planned worker count reproduces the shortfall
-- deterministically (two workers planned, one launches). nlist > fan_out
-- forces the multi-level tree whose subtree batches are keyed on the
-- participant count.
SET max_parallel_maintenance_workers = 2;
SET max_parallel_workers = 1;
SELECT exact_check('line3', '(v) WITH (nlist = 12, fan_out = 4)',
                   '[0.5,0,0]') AS shortfall_exact;
CREATE INDEX line3_short ON line3 USING mktann (v)
    WITH (nlist = 12, fan_out = 4);
-- Structural check: no two internal entries may point at the same child
-- page. A participant that never launches must not leave its root children
-- unbuilt (their block ranges collapse onto the next child's pages, leaving
-- the root with aliased duplicate child pointers while the planned
-- partitions silently never exist).
SELECT count(*) = count(DISTINCT child_blkno) AS no_duplicate_children
  FROM centroid_pages('line3_short') WHERE NOT is_leaf;
-- Reachability: probing every list with the result cap lifted must return
-- every row.
SET enable_seqscan = off;
SET mkt.nprobe = 10000;
SET mkt.query_limit = 100;
SELECT count(*) AS shortfall_reachable FROM (
    SELECT id FROM line3 ORDER BY v <-> '[0.5,0,0]' LIMIT 100) t;
RESET mkt.query_limit;
RESET mkt.nprobe;
RESET enable_seqscan;
RESET max_parallel_workers;
DROP INDEX line3_short;

-- FASTSCAN centroids through the streaming writers, at the fastscan group
-- boundary and one past it (32 and 33 leaves requested). The group math is
-- computed independently by the plan pass and the page writers; a one-page
-- disagreement would overrun the reserved range into the head region, so
-- exactness at nprobe >= nlist is the gate. Serial and parallel.
SET max_parallel_maintenance_workers = 2;
SELECT exact_check('line3',
                   '(v) WITH (centroid_compression = true, '
                   'centroid_fastscan = true, nlist = 32, fan_out = 8)',
                   '[0.5,0,0]') AS parallel_cfs_group_exact;
SELECT exact_check('line3',
                   '(v) WITH (centroid_compression = true, '
                   'centroid_fastscan = true, nlist = 33, fan_out = 8)',
                   '[0.5,0,0]') AS parallel_cfs_group1_exact;
SET max_parallel_maintenance_workers = 0;
SELECT exact_check('line3',
                   '(v) WITH (centroid_compression = true, '
                   'centroid_fastscan = true, nlist = 33, fan_out = 8)',
                   '[0.5,0,0]') AS serial_cfs_group1_exact;

-- Depth-3 tree through the parallel streaming build (subtrees with internal
-- pages; the earlier parallel exactness gates build 1-2 levels). Includes
-- the head bijection: every centroid leaf entry must point at exactly one
-- posting head whose cluster id matches the formula (head = first_posting +
-- cluster), and no head may be shared -- a wrong-head mapping degrades
-- recall silently, passing the exactness gate.
SET max_parallel_maintenance_workers = 2;
SELECT exact_check('line3', '(v) WITH (nlist = 40, fan_out = 4)',
                   '[0.5,0,0]') AS parallel_depth3_exact;
CREATE INDEX line3_d3 ON line3 USING mktann (v)
    WITH (nlist = 40, fan_out = 4);
SELECT count(*) = count(DISTINCT child_blkno) AS d3_no_duplicate_children
  FROM centroid_pages('line3_d3') WHERE NOT is_leaf;
WITH leaves AS (
    SELECT child_blkno FROM centroid_pages('line3_d3') WHERE is_leaf
), heads AS (
    SELECT blkno, cluster_id FROM mkt.posting_pages('line3_d3')
    WHERE is_first
)
SELECT (SELECT count(*) FROM leaves) = (SELECT count(*) FROM heads)
       AND NOT EXISTS (
           SELECT 1 FROM leaves l LEFT JOIN heads h ON h.blkno = l.child_blkno
           WHERE h.blkno IS NULL)
       AND (SELECT count(DISTINCT blkno - cluster_id) FROM heads) = 1
       AS d3_head_bijection;
DROP INDEX line3_d3;

-- Shortfall at depth 3: the largest-first batch schedule and the blob
-- replay must agree under a narrowed participant count too.
SET max_parallel_workers = 1;
SELECT exact_check('line3', '(v) WITH (nlist = 40, fan_out = 4)',
                   '[0.5,0,0]') AS shortfall_depth3_exact;
RESET max_parallel_workers;
DROP TABLE line3;

-- ============================================================
-- Serial fallback after a failed parallel launch
-- ============================================================
-- Planning asks for parallel workers but the launch returns none
-- (max_parallel_workers = 0), so the build lands on the serial path after
-- the parallel attempt already sized its shared memory. The fallback index
-- must return the exact top-10.
CREATE TABLE line1k (id int, v vector(3));
INSERT INTO line1k SELECT g, format('[%s,0,0]', g)::vector
    FROM generate_series(1, 1000) g;
ALTER TABLE line1k SET (parallel_workers = 2);
SET max_parallel_maintenance_workers = 2;
SET max_parallel_workers = 0;
SELECT exact_check('line1k', '(v) WITH (nlist = 20, fan_out = 4)',
                   '[500,0,0]') AS fallback_exact;
RESET max_parallel_workers;
RESET max_parallel_maintenance_workers;
DROP TABLE line1k;

-- 768-dim well-separated points (only the first coordinate varies). At
-- maintenance_work_mem = 1MB the k-means sample (~341 vectors) is bounded below
-- the 600 rows, exercising the bounded subsample + full-table leaf refinement;
-- the bounded/refined index must still return the exact top-10.
CREATE TABLE line768 (id int, v vector(768));
INSERT INTO line768
    SELECT g, ('[' || g || repeat(',0', 767) || ']')::vector(768)
    FROM generate_series(1, 600) g;
ALTER TABLE line768 SET (parallel_workers = 2);

SET max_parallel_maintenance_workers = 2;
SET maintenance_work_mem = '1MB';
SET mkt.leaf_refine_threshold = 0;
SELECT exact_check('line768', '(v) WITH (centroid_compression = true)',
                   '[0.5' || repeat(',0', 767) || ']') AS bounded_norefine_exact;
SET mkt.leaf_refine_threshold = 100000;
SELECT exact_check('line768', '(v) WITH (centroid_compression = true)',
                   '[0.5' || repeat(',0', 767) || ']') AS bounded_refine_exact;
RESET mkt.leaf_refine_threshold;
RESET maintenance_work_mem;
RESET max_parallel_maintenance_workers;
DROP TABLE line768;

-- ============================================================
-- Serial refine + big subtree blobs (12k rows: the serial sample cap
-- floors at 10000, so the build is genuinely subsampled)
-- ============================================================
CREATE TABLE line12k (id int, v vector(64));
INSERT INTO line12k
    SELECT g, (SELECT ('[' || string_agg((sin(g * 0.01 + j))::text, ',') ||
                       ']')
               FROM generate_series(1, 64) j)::vector(64)
    FROM generate_series(1, 12000) g;
ALTER TABLE line12k SET (parallel_workers = 2);

-- Serial page-backed refine over a multi-level tree (the bounded exactness
-- gates above are parallel and flat); with fastscan posting heads the
-- refine rewrites fastscan head pages.
SET max_parallel_maintenance_workers = 0;
SET mkt.leaf_refine_threshold = 100000;
-- The 1MB budget forces the sample cap to its 10000-vector floor, below the
-- 12000 rows, so the build is genuinely subsampled and refine runs.
SET maintenance_work_mem = '1MB';
-- The probe mirrors mid-table row 6000's construction (sin(60 + j)) without
-- reading line12k, so no scan is open on it while exact_check indexes it.
SELECT exact_check('line12k', '(v) WITH (nlist = 48, fan_out = 4)',
                   (SELECT '[' || string_agg((sin(60 + j))::text, ',') || ']'
                    FROM generate_series(1, 64) j))
    AS serial_refine_multilevel_exact;
SELECT exact_check('line12k',
                   '(v) WITH (nlist = 48, fan_out = 4, fastscan = true)',
                   (SELECT '[' || string_agg((sin(60 + j))::text, ',') || ']'
                    FROM generate_series(1, 64) j))
    AS serial_refine_fastscan_exact;
RESET maintenance_work_mem;
RESET mkt.leaf_refine_threshold;

-- Subtree blobs several times the BufFile buffer (nlist 64 / fan_out 8 at
-- dim 64 gives ~multi-page blobs), so the leader's replay crosses buffer
-- boundaries; a framing bug would stream a garbage subtree with no error.
SET max_parallel_maintenance_workers = 2;
SELECT exact_check('line12k', '(v) WITH (nlist = 64, fan_out = 8)',
                   (SELECT '[' || string_agg((sin(60 + j))::text, ',') || ']'
                    FROM generate_series(1, 64) j))
    AS parallel_bigblob_exact;
DROP TABLE line12k;

-- ============================================================
-- Degenerate data through the streaming build
-- ============================================================
-- One row, and all-identical rows: k-means collapses to mostly-empty
-- clusters, exercising the empty-cluster compaction and a head region far
-- smaller than the requested nlist. Serial and parallel.
CREATE TABLE degen (id int, v vector(3));
INSERT INTO degen VALUES (1, '[1,2,3]');
CREATE INDEX degen_one ON degen USING mktann (v) WITH (nlist = 8);
SET enable_seqscan = off;
SELECT count(*) AS one_row FROM (
    SELECT id FROM degen ORDER BY v <-> '[0,0,0]' LIMIT 10) t;
RESET enable_seqscan;
DROP INDEX degen_one;
INSERT INTO degen SELECT g, '[1,2,3]' FROM generate_series(2, 100) g;
ALTER TABLE degen SET (parallel_workers = 2);
SET max_parallel_maintenance_workers = 2;
CREATE INDEX degen_same ON degen USING mktann (v)
    WITH (nlist = 12, fan_out = 4);
SET enable_seqscan = off;
SET mkt.nprobe = 10000;
SET mkt.query_limit = 150;
SELECT count(*) AS identical_rows FROM (
    SELECT id FROM degen ORDER BY v <-> '[1,2,3]' LIMIT 150) t;
RESET mkt.query_limit;
RESET mkt.nprobe;
RESET enable_seqscan;
SET max_parallel_maintenance_workers = 0;
DROP INDEX degen_same;
CREATE INDEX degen_serial ON degen USING mktann (v)
    WITH (nlist = 12, fan_out = 4);
SET enable_seqscan = off;
SET mkt.nprobe = 10000;
SET mkt.query_limit = 150;
SELECT count(*) AS identical_rows_serial FROM (
    SELECT id FROM degen ORDER BY v <-> '[1,2,3]' LIMIT 150) t;
RESET mkt.query_limit;
RESET mkt.nprobe;
RESET enable_seqscan;
RESET max_parallel_maintenance_workers;
DROP TABLE degen;


-- ============================================================
-- CREATE INDEX CONCURRENTLY builds a correct index
-- ============================================================
-- CIC drives ambuild through index_concurrently_build with an MVCC snapshot in
-- its own transactions; the parallel workers must take weak locks and mark their
-- IndexInfo concurrent, or the build errors (XID assigned in a worker / snapshot
-- mismatch). CIC cannot run inside a function/txn block, so build it inline.
-- Well-separated points on a line: with nprobe >= nlist the top-10 of [0.5,0,0]
-- must be exactly ids 1..10.
CREATE TABLE cic_pts (id int, v vector(3));
INSERT INTO cic_pts SELECT g, format('[%s,0,0]', g)::vector
    FROM generate_series(1, 50) g;
ALTER TABLE cic_pts SET (parallel_workers = 2);
SET max_parallel_maintenance_workers = 2;
CREATE INDEX CONCURRENTLY cic_idx ON cic_pts USING mktann (v)
    WITH (centroid_compression = true);
-- CIC must have actually produced a valid index (not errored out).
SELECT relpages > 0 AS cic_has_pages FROM pg_class WHERE relname = 'cic_idx';
SET enable_seqscan = off;
SET mkt.nprobe = 10000;
SELECT array_agg(id ORDER BY id) = ARRAY[1,2,3,4,5,6,7,8,9,10] AS cic_exact
    FROM (SELECT id FROM cic_pts ORDER BY v <-> '[0.5,0,0]' LIMIT 10) t;
RESET mkt.nprobe;
RESET enable_seqscan;
RESET max_parallel_maintenance_workers;
DROP TABLE cic_pts;

-- ============================================================
-- Cosine metric through the streamed tree
-- ============================================================
-- Distinct directions per row (two varying coordinates). The probe sits
-- outside the angle range (all candidates on one side), because a probe
-- between grid points ties symmetric neighbors EXACTLY in cosine distance
-- (cos is even), making top-10 membership rounding-dependent — with a
-- mid-range probe the ranks 10/11 candidates carry bit-identical
-- distances and the winner varies by platform. One-sided, distances are
-- strictly increasing with gaps of thousands of ulps. Covers the
-- cosine-specific code in the streamed build: sample normalization,
-- root/global-mean normalization, multi-level batched and flat parallel
-- shapes, and the serial shape.
CREATE TABLE cosdirs (id int, v vector(8));
INSERT INTO cosdirs
    SELECT g, ('[' || cos(g * 0.001) || ',' || sin(g * 0.001) ||
               repeat(',0.01', 6) || ']')::vector(8)
    FROM generate_series(1, 2000) g;
ALTER TABLE cosdirs SET (parallel_workers = 2);
SET max_parallel_maintenance_workers = 2;
-- multi-level batched parallel tree
SELECT exact_check('cosdirs', '(v vector_cosine_ops) WITH (nlist = 48, fan_out = 4)',
                   '[' || cos(-0.5) || ',' || sin(-0.5) || repeat(',0.01', 6) || ']',
                   '<=>') AS cos_parallel_multilevel_exact;
-- flat parallel tree
SELECT exact_check('cosdirs', '(v vector_cosine_ops) WITH (nlist = 8, fan_out = 8)',
                   '[' || cos(-0.5) || ',' || sin(-0.5) || repeat(',0.01', 6) || ']',
                   '<=>') AS cos_parallel_flat_exact;
SET max_parallel_maintenance_workers = 0;
-- serial streamed tree
SELECT exact_check('cosdirs', '(v vector_cosine_ops) WITH (nlist = 48, fan_out = 4)',
                   '[' || cos(-0.5) || ',' || sin(-0.5) || repeat(',0.01', 6) || ']',
                   '<=>') AS cos_serial_multilevel_exact;
RESET max_parallel_maintenance_workers;
DROP TABLE cosdirs;

-- ============================================================
-- Flat fastscan centroid page
-- ============================================================
-- nlist <= fan_out builds a one-level tree whose single centroid page is
-- root and leaf-parent at once, with leaf-ness inferred from the descent
-- level; fastscan group layout on that page was previously only exercised
-- via multi-level builds.
CREATE TABLE line3f (id int, v vector(3));
INSERT INTO line3f SELECT g, format('[%s,0,0]', g)::vector
    FROM generate_series(1, 50) g;
ALTER TABLE line3f SET (parallel_workers = 2);
SET max_parallel_maintenance_workers = 2;
SELECT exact_check('line3f',
                   '(v) WITH (nlist = 8, fan_out = 8, centroid_compression = true, centroid_fastscan = true)',
                   '[0.5,0,0]') AS flat_fastscan_exact;
RESET max_parallel_maintenance_workers;
DROP TABLE line3f;

-- ============================================================
-- Multi-tile leaf refinement
-- ============================================================
-- A partition count above the 1MB tile capacity (~2000 leaves at dim 64)
-- forces the refine accumulator through several tiles, each re-scanning the
-- table -- the tile boundary handling and per-tile rescan reset are
-- otherwise never exercised. The build is genuinely subsampled (12000 rows
-- against the 10000-sample floor) and the forced threshold refines every
-- leaf; the refined index must still return the exact top-10.
CREATE TABLE line12kb (id int, v vector(64));
INSERT INTO line12kb
    SELECT g, (SELECT ('[' || string_agg((sin(g * 0.01 + j))::text, ',') ||
                       ']')
               FROM generate_series(1, 64) j)::vector(64)
    FROM generate_series(1, 12000) g;
SET max_parallel_maintenance_workers = 0;
SET maintenance_work_mem = '1MB';
SET mkt.leaf_refine_threshold = 100000;
SELECT exact_check('line12kb', '(v) WITH (nlist = 4096, fan_out = 8)',
                   (SELECT '[' || string_agg((sin(60 + j))::text, ',') || ']'
                    FROM generate_series(1, 64) j))
    AS multitile_refine_exact;
RESET mkt.leaf_refine_threshold;
RESET maintenance_work_mem;
RESET max_parallel_maintenance_workers;
DROP TABLE line12kb;

-- Cleanup
DROP FUNCTION exact_check(text, text, text, text);
DROP FUNCTION pbuild_check(text, text);
DROP TABLE embeddings;
