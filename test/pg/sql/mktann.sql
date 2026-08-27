-- mktann index access method

-- Create table with vector column
CREATE TABLE embeddings (id serial, v vector(3));

-- Insert deterministic data (grid of vectors)
INSERT INTO embeddings (v)
    SELECT format('[%s,%s,%s]', x * 0.1, y * 0.1, z * 0.1)::vector
    FROM generate_series(0, 9) x,
         generate_series(0, 9) y,
         generate_series(0, 9) z;

-- Verify data inserted
SELECT count(*) FROM embeddings;

-- L2 with centroid_compression (RaBitQ centroids, supports scan)
CREATE INDEX idx_l2c ON embeddings USING mktann (v)
    WITH (centroid_compression = true);

-- Verify index was built (should have pages)
SELECT relpages > 0 AS has_pages FROM pg_class
    WHERE relname = 'idx_l2c';

-- Verify index scan is used for ORDER BY <-> LIMIT
SET enable_seqscan = off;
EXPLAIN (COSTS OFF)
SELECT id, v <-> '[0.5,0.5,0.5]' AS dist
    FROM embeddings ORDER BY v <-> '[0.5,0.5,0.5]' LIMIT 5;

-- ORDER BY distance with LIMIT — should return results via index scan
SELECT count(*) FROM (
    SELECT id, v <-> '[0.5,0.5,0.5]' AS dist
    FROM embeddings ORDER BY v <-> '[0.5,0.5,0.5]' LIMIT 5
) t;
RESET enable_seqscan;

-- Create index with distance_mode relopt
CREATE INDEX idx_sym ON embeddings USING mktann (v)
    WITH (distance_mode = 'symmetric', centroid_compression = true);
CREATE INDEX idx_asym ON embeddings USING mktann (v)
    WITH (distance_mode = 'asymmetric', centroid_compression = true);

-- GUC: check default, set, and reset
SHOW mkt.distance_mode;
SET mkt.distance_mode = 'symmetric';
SHOW mkt.distance_mode;
SET mkt.distance_mode = 'default';
SHOW mkt.distance_mode;

-- L2 with uncompressed float centroids — build should work
CREATE INDEX idx_l2_float ON embeddings USING mktann (v)
    WITH (centroid_compression = off);
SELECT relpages > 0 AS has_pages FROM pg_class
    WHERE relname = 'idx_l2_float';

-- IP opclass (centroid_compression=auto falls back to float) — build works
CREATE INDEX idx_ip ON embeddings USING mktann (v vector_ip_ops);
SELECT relpages > 0 AS has_pages FROM pg_class
    WHERE relname = 'idx_ip';

-- Cosine opclass with uncompressed float centroids — build should work
CREATE INDEX idx_cos ON embeddings USING mktann (v vector_cosine_ops)
    WITH (centroid_compression = off);
SELECT relpages > 0 AS has_pages FROM pg_class
    WHERE relname = 'idx_cos';

-- Cosine with compression — build should work
CREATE INDEX idx_cosc ON embeddings
    USING mktann (v vector_cosine_ops)
    WITH (centroid_compression = true);
SELECT relpages > 0 AS has_pages FROM pg_class
    WHERE relname = 'idx_cosc';

-- Validation: compression + IP should ERROR
CREATE INDEX idx_bad ON embeddings
    USING mktann (v vector_ip_ops)
    WITH (centroid_compression = true);

-- ============================================================
-- halfvec columns
-- ============================================================
-- The index is the same either way: postings hold RaBitQ codes, and the AM
-- widens a halfvec tuple to float32 on read. What halfvec changes is the
-- heap an exact rerank reads -- at 768d a vector row is 3080 bytes and fits
-- 2 to an 8 kB page against halfvec's 1544 and 5.
--
-- Every coordinate below is a multiple of 0.1, none of which is exactly
-- representable in f16, so the rows genuinely round on the way in and an
-- exact rerank has to compare against the rounded values rather than the
-- literals.
CREATE TABLE h_embeddings (id serial, v halfvec(3));
INSERT INTO h_embeddings (v)
    SELECT format('[%s,%s,%s]', x * 0.1, y * 0.1, z * 0.1)::halfvec
    FROM generate_series(0, 9) x,
         generate_series(0, 9) y,
         generate_series(0, 9) z;
ANALYZE h_embeddings;

-- All three halfvec opclasses build.
CREATE INDEX h_idx_l2 ON h_embeddings USING mktann (v)
    WITH (centroid_compression = true);
SELECT relpages > 0 AS has_pages FROM pg_class WHERE relname = 'h_idx_l2';
CREATE INDEX h_idx_ip ON h_embeddings USING mktann (v halfvec_ip_ops);
SELECT relpages > 0 AS has_pages FROM pg_class WHERE relname = 'h_idx_ip';
CREATE INDEX h_idx_cos ON h_embeddings USING mktann (v halfvec_cosine_ops);
SELECT relpages > 0 AS has_pages FROM pg_class WHERE relname = 'h_idx_cos';

-- The planner picks the index for an ORDER BY over a halfvec query argument.
SET enable_seqscan = off;
EXPLAIN (COSTS OFF)
SELECT id FROM h_embeddings ORDER BY v <-> '[0.5,0.5,0.5]'::halfvec LIMIT 5;
RESET enable_seqscan;

-- Correctness against brute force. Compare the sorted distance sequence
-- rather than ids: the grid is full of exact ties, so which of several
-- equidistant rows comes back is arbitrary while the distances are not.
-- Probing every cluster makes the comparison exact, so a mismatch is a
-- decode or rerank bug and not a recall miss.
SET enable_indexscan = off;
SET enable_bitmapscan = off;
CREATE TEMP TABLE h_exact AS
    SELECT round((v <-> '[0.5,0.5,0.5]'::halfvec)::numeric, 6) AS d
        FROM h_embeddings ORDER BY 1 LIMIT 5;
RESET enable_indexscan;
RESET enable_bitmapscan;

SET enable_seqscan = off;
SET mkt.nprobe = 1000;
CREATE TEMP TABLE h_idx AS
    SELECT round((v <-> '[0.5,0.5,0.5]'::halfvec)::numeric, 6) AS d
        FROM h_embeddings ORDER BY v <-> '[0.5,0.5,0.5]'::halfvec LIMIT 5;
RESET mkt.nprobe;
RESET enable_seqscan;

SELECT (SELECT array_agg(d ORDER BY d) FROM h_idx)
     = (SELECT array_agg(d ORDER BY d) FROM h_exact) AS matches_exact,
       (SELECT count(*) FROM h_idx) AS idx_rows;

-- Insert into an indexed halfvec table, then find the new row: its own value
-- must come back at distance 0, which only holds if the insert encoded the
-- widened f16 value and the rerank compared against the same.
INSERT INTO h_embeddings (v) VALUES ('[0.55,0.55,0.55]');
SET enable_seqscan = off;
SET mkt.nprobe = 1000;
SELECT round((v <-> '[0.55,0.55,0.55]'::halfvec)::numeric, 6) AS inserted_dist
    FROM h_embeddings ORDER BY v <-> '[0.55,0.55,0.55]'::halfvec LIMIT 1;
RESET mkt.nprobe;
RESET enable_seqscan;

DROP TABLE h_embeddings;

-- Insert after index creation (should not crash)
INSERT INTO embeddings (v) VALUES ('[1,1,1]');

-- VACUUM on the index (should not crash)
VACUUM embeddings;

-- Multi-level tree with fan_out (nlist=31, fan_out=4 → 3 levels)
CREATE INDEX idx_ml ON embeddings USING mktann (v)
    WITH (fan_out = 4, centroid_compression = true);
SELECT relpages > 0 AS has_pages FROM pg_class
    WHERE relname = 'idx_ml';

-- Multi-level tree scan should return results
SET enable_seqscan = off;
SELECT count(*) FROM (
    SELECT id, v <-> '[0.5,0.5,0.5]' AS dist
    FROM embeddings ORDER BY v <-> '[0.5,0.5,0.5]' LIMIT 5
) t;
RESET enable_seqscan;

-- Multi-level tree with uncompressed float centroids
CREATE INDEX idx_ml_float ON embeddings USING mktann (v)
    WITH (fan_out = 4, centroid_compression = off);
SELECT relpages > 0 AS has_pages FROM pg_class
    WHERE relname = 'idx_ml_float';

-- fan_out = 2 (deepest tree)
CREATE INDEX idx_fo2 ON embeddings USING mktann (v)
    WITH (fan_out = 2, centroid_compression = true);
SELECT relpages > 0 AS has_pages FROM pg_class
    WHERE relname = 'idx_fo2';

-- Validation: fan_out must be >= 2
CREATE INDEX idx_bad_fo ON embeddings USING mktann (v)
    WITH (fan_out = 1);

-- EXPLAIN ANALYZE: verify mktann stats are present with sensible values
SET enable_seqscan = off;
CREATE FUNCTION test_explain_stats() RETURNS TABLE (
    has_clusters bool,
    has_centroid_pages bool,
    has_posting_pages bool,
    has_entries bool,
    has_rerank_cands bool,
    has_results bool,
    has_storage_reads bool
) LANGUAGE plpgsql AS $$
DECLARE
    explain_json json;
    stats json;
BEGIN
    EXECUTE 'EXPLAIN (ANALYZE, VERBOSE, COSTS OFF, FORMAT JSON)
        SELECT id, v <-> ''[0.5,0.5,0.5]'' AS dist
        FROM embeddings ORDER BY v <-> ''[0.5,0.5,0.5]'' LIMIT 5'
    INTO explain_json;
    stats := explain_json->0->'Plan'->'Plans'->0->'Mktann';
    RETURN QUERY SELECT
        (stats->>'Posting Lists Scanned')::int > 0,
        (stats->>'Centroid Pages Read')::int > 0,
        (stats->>'Posting Pages Read')::int > 0,
        (stats->>'Posting Entries Scanned')::int > 0,
        (stats->>'Rerank Candidates')::int >= 0,
        (stats->>'Rerank Results')::int > 0,
        (stats->>'Storage Reads')::int > 0;
END $$;
SELECT * FROM test_explain_stats();
DROP FUNCTION test_explain_stats();
RESET enable_seqscan;

-- Cleanup
DROP TABLE embeddings;

-- ============================================================
-- Auto nlist on an unanalyzed TOASTed table
-- ============================================================
-- With no reltuples yet, the partition count is derived from the heap size.
-- Incompressible rows this wide land in TOAST, so the main fork holds only
-- narrow stubs and a width-from-dimension guess undercounts the rows by
-- almost two orders of magnitude, starving the automatic partition count.
-- autovacuum is off so no analyze sneaks in a real reltuples before the
-- build. 2000 rows target ~44 partitions; the band tolerates estimator
-- slack in both directions but not the undercount.
CREATE TABLE hd (id int, v vector(768)) WITH (autovacuum_enabled = off);
-- external (uncompressed) storage makes the TOASTing deterministic
ALTER TABLE hd ALTER COLUMN v SET STORAGE EXTERNAL;
INSERT INTO hd
    SELECT g, (SELECT ('[' || string_agg((sin(g * 0.7 + j))::text, ',') ||
                       ']')
               FROM generate_series(1, 768) j)::vector(768)
    FROM generate_series(1, 2000) g;
CREATE INDEX hd_i ON hd USING mktann (v);
SELECT count(*) BETWEEN 20 AND 120 AS toast_auto_nlist
  FROM centroid_pages('hd_i') WHERE is_leaf;
DROP TABLE hd;

-- ============================================================
-- Index layout dimension ceiling
-- ============================================================
-- A posting list's first page carries the float encode reference and must
-- still fit one entry, which caps the indexable dimension. One past the
-- ceiling is rejected up front (it would corrupt the first-page capacity
-- arithmetic); the ceiling itself builds and answers exactly.
CREATE TABLE dimcap (id int, v vector(1969));
CREATE INDEX dimcap_i ON dimcap USING mktann (v);
DROP TABLE dimcap;
CREATE TABLE dimcap (id int, v vector(1968));
INSERT INTO dimcap
    SELECT g, (SELECT ('[' || string_agg((sin(g + j))::text, ',') || ']')
               FROM generate_series(1, 1968) j)::vector(1968)
    FROM generate_series(1, 20) g;
CREATE INDEX dimcap_i ON dimcap USING mktann (v) WITH (nlist = 4);
SET enable_seqscan = off;
SELECT id FROM dimcap ORDER BY v <-> (SELECT v FROM dimcap WHERE id = 7)
    LIMIT 1;
RESET enable_seqscan;
DROP TABLE dimcap;

-- ============================================================
-- RaBitQ rotation-matrix cache liveness across differently-dimensioned scans
-- ============================================================
-- The per-backend cache that holds the (expensive to build) RaBitQ
-- rotation matrix is keyed by (dim, seed). A scan checks the matrix out
-- at beginscan and holds a direct reference to it (mktann_rescan never
-- re-derives it), so the cache must never free an entry that a
-- still-open scan is holding, no matter how many other dimensions get
-- checked out and evicted around it.
--
-- idx_a is dimension 8. idx_c is dimension 256 -- a different (dim, seed)
-- cache key. A single query's target list runs a correlated subquery
-- against each, once per outer row of "probes": PostgreSQL reuses
-- (rescans) the same inner scan node across outer rows rather than
-- starting a fresh scan each time, so row 2's evaluation of idx_a's
-- subquery rescans the exact same scan idx_a's row-1 evaluation opened,
-- with whatever matrix pointer that scan cached back then -- after row
-- 1's idx_c subquery evaluation has already checked out (and, if the
-- cache does not keep it alive, evicted) a dimension-256 entry in
-- between. If the dimension-8 entry was freed instead of kept alive,
-- idx_a's row-2 result reads freed memory and returns a wrong nearest
-- neighbor.
SET enable_seqscan = off;

CREATE TABLE cache_a (id int, v vector(8));
INSERT INTO cache_a
    SELECT g, format('[%s,0,0,0,0,0,0,0]', g)::vector
    FROM generate_series(1, 2000) g;
CREATE INDEX idx_a ON cache_a USING mktann (v) WITH (nlist = 64);

CREATE TABLE cache_c (id int, v vector(256));
INSERT INTO cache_c
    SELECT g, ('[' || g::text || repeat(',0', 255) || ']')::vector
    FROM generate_series(1, 2000) g;
CREATE INDEX idx_c ON cache_c USING mktann (v) WITH (nlist = 64);

-- Non-integer query points avoid distance ties, so each nearest-1 answer
-- is unambiguous: row 1 probes near 1500, row 2 probes near 1600.
CREATE TABLE probes (rown int, qa vector(8), qc vector(256));
INSERT INTO probes VALUES
    (1, '[1500.3,0,0,0,0,0,0,0]',
        ('[' || '1500.6' || repeat(',0', 255) || ']')::vector),
    (2, '[1600.3,0,0,0,0,0,0,0]',
        ('[' || '1600.6' || repeat(',0', 255) || ']')::vector);

-- Both subqueries must return their query point's exact nearest integer
-- on every row, including row 2 -- the row whose idx_a evaluation
-- rescans a scan that idx_c's row-1 evaluation may have invalidated in
-- between.
SELECT
    rown,
    (SELECT id FROM cache_a ORDER BY v <-> qa LIMIT 1) AS nearest_a,
    (SELECT id FROM cache_c ORDER BY v <-> qc LIMIT 1) AS nearest_c
FROM probes ORDER BY rown;

RESET enable_seqscan;
DROP TABLE probes;
DROP TABLE cache_a;
DROP TABLE cache_c;

-- ============================================================
-- Per-backend RaBitQ params cache: refcounting, decay, eviction
-- ============================================================
-- Include test helpers for introspecting the cache. The helper file
-- silences its own definitions and docs (see sql/test_helpers.sql), so
-- they don't land in this test's expected output.
\getenv abs_srcdir PG_ABS_SRCDIR
\set helper_sql :abs_srcdir '/sql/test_helpers.sql'
\i :helper_sql

-- The cache is per-backend state and the exact usage arithmetic below
-- counts every checkout from zero, so start from an empty cache,
-- independent of whatever the sections above checked out. (The count
-- dropped here depends on those sections, so only assert the reset.)
SELECT rabitq_cache_clear() >= 0 AS cleared;

-- Checkouts happen at query time (a scan's first fetch) and at index
-- tuple insertion; index BUILDS generate their rotation matrix
-- privately and never touch this cache -- the setup below creates all
-- ten indexes first and the stats stay empty until the first query.
SET enable_seqscan = off;

-- Ten tiny tables+indexes, dims 8..80. Rows are inserted before each
-- index exists, so nothing here checks a matrix out.
DO $$
DECLARE d int;
BEGIN
    FOREACH d IN ARRAY ARRAY[8, 16, 24, 32, 40, 48, 56, 64, 72, 80] LOOP
        EXECUTE format('CREATE TABLE cache_t%s (id int, v vector(%s))',
                       d, d);
        EXECUTE format($i$INSERT INTO cache_t%s
            SELECT g, ('[' || g::text || repeat(',0', %s - 1) || ']')::vector
            FROM generate_series(1, 256) g$i$, d, d);
        EXECUTE format('CREATE INDEX cache_i%s ON cache_t%s '
                       'USING mktann (v) WITH (nlist = 4)', d, d);
    END LOOP;
END $$;

SELECT dim, refcount, round(usage::numeric, 4) AS usage
FROM rabitq_params_cache() ORDER BY dim;

-- Fill the cache to its soft target (8 entries) with a distinct usage
-- score per entry — the first checkout of a dimension enters at 1.0,
-- every further checkout adds 1.0 — so each eviction below has exactly
-- one possible victim. dim 8 is queried once: the unique minimum.
DO $$
DECLARE d int; i int; r int;
BEGIN
    FOREACH d IN ARRAY ARRAY[8, 16, 24, 32, 40, 48, 56, 64] LOOP
        FOR i IN 1 .. d / 8 LOOP
            EXECUTE format(
                'SELECT id FROM cache_t%s ORDER BY v <-> %L LIMIT 1',
                d, '[1' || repeat(',0', d - 1) || ']') INTO r;
        END LOOP;
    END LOOP;
END $$;

SELECT dim, refcount, round(usage::numeric, 4) AS usage
FROM rabitq_params_cache() ORDER BY dim;

-- A miss with the cache at its target — the first dim-72 query — decays
-- every usage score by 0.99 and evicts the least-used idle entry: dim 8.
SELECT id FROM cache_t72
ORDER BY v <-> ('[1' || repeat(',0', 71) || ']')::vector LIMIT 1;

SELECT dim, refcount, round(usage::numeric, 4) AS usage
FROM rabitq_params_cache() ORDER BY dim;

-- Refcounting: a scan checks its entry out at its first fetch (scan
-- setup is lazy — DECLARE alone holds nothing) and checks it back in
-- when the scan closes. Two concurrent scans of the same-dimension
-- index hold two checkouts.
BEGIN;
DECLARE c16a CURSOR FOR
    SELECT id FROM cache_t16 ORDER BY v <-> ('[1' || repeat(',0', 15) || ']')::vector LIMIT 5;
DECLARE c16b CURSOR FOR
    SELECT id FROM cache_t16 ORDER BY v <-> ('[2' || repeat(',0', 15) || ']')::vector LIMIT 5;
SELECT refcount FROM rabitq_params_cache() WHERE dim = 16;
FETCH 1 FROM c16a;
SELECT refcount, round(usage::numeric, 4) AS usage
FROM rabitq_params_cache() WHERE dim = 16;
FETCH 1 FROM c16b;
SELECT refcount, round(usage::numeric, 4) AS usage
FROM rabitq_params_cache() WHERE dim = 16;
CLOSE c16a;
SELECT refcount FROM rabitq_params_cache() WHERE dim = 16;
CLOSE c16b;
SELECT refcount FROM rabitq_params_cache() WHERE dim = 16;
COMMIT;

-- Held-entry immunity: check dim 72 out (usage 2.0 — the strict global
-- minimum: every idle entry is >= 2.9403 by now) and, while holding it,
-- force another miss with the first dim-80 query. The sweep must skip
-- the held global minimum and evict the least-used idle entry instead:
-- dim 24.
BEGIN;
DECLARE c72 CURSOR FOR
    SELECT id FROM cache_t72 ORDER BY v <-> ('[1' || repeat(',0', 71) || ']')::vector LIMIT 5;
FETCH 1 FROM c72;
SELECT refcount, round(usage::numeric, 4) AS usage
FROM rabitq_params_cache() WHERE dim = 72;

SELECT id FROM cache_t80
ORDER BY v <-> ('[1' || repeat(',0', 79) || ']')::vector LIMIT 1;

-- dim 24 gone; dim 72 survived its own global-minimum usage because it
-- was held. Everything else decayed by another factor of 0.99.
SELECT dim, refcount, round(usage::numeric, 4) AS usage
FROM rabitq_params_cache() ORDER BY dim;
COMMIT;

-- The commit closed the held scan: checkout released.
SELECT refcount FROM rabitq_params_cache() WHERE dim = 72;

-- ============================================================
-- Error safety: checkouts must be returned on every error path
-- ============================================================
-- Checkouts are registered with the checkout-time resource owner, so a
-- scan or insert that never reaches its normal release still returns
-- its refcount when the owner is released. Without that, one aborted
-- query would pin its entry unevictable for the backend's lifetime.

-- Abort with an open cursor (the portal is dropped without endscan).
-- Expected: refcount 1 while the cursor is open, back to 0 after
-- ROLLBACK; usage keeps its checkout bump.
BEGIN;
DECLARE cabort CURSOR FOR
    SELECT id FROM cache_t80 ORDER BY v <-> ('[1' || repeat(',0', 79) || ']')::vector LIMIT 5;
FETCH 1 FROM cabort;
SELECT refcount, round(usage::numeric, 4) AS usage
FROM rabitq_params_cache() WHERE dim = 80;
SELECT 1 / 0;
ROLLBACK;
SELECT refcount, round(usage::numeric, 4) AS usage
FROM rabitq_params_cache() WHERE dim = 80;

-- Savepoint scoping: two cursors, the inner one opened inside a
-- savepoint that is rolled back. Expected: refcount 1 (outer) -> 2
-- (inner) -> 1 after ROLLBACK TO with the outer cursor still
-- fetchable -> 0 after COMMIT.
BEGIN;
DECLARE couter CURSOR FOR
    SELECT id FROM cache_t72 ORDER BY v <-> ('[1' || repeat(',0', 71) || ']')::vector LIMIT 5;
FETCH 1 FROM couter;
SELECT refcount FROM rabitq_params_cache() WHERE dim = 72;
SAVEPOINT s1;
DECLARE cinner CURSOR FOR
    SELECT id FROM cache_t72 ORDER BY v <-> ('[2' || repeat(',0', 71) || ']')::vector LIMIT 5;
FETCH 1 FROM cinner;
SELECT refcount FROM rabitq_params_cache() WHERE dim = 72;
ROLLBACK TO s1;
SELECT refcount FROM rabitq_params_cache() WHERE dim = 72;
FETCH 1 FROM couter;
COMMIT;
SELECT refcount FROM rabitq_params_cache() WHERE dim = 72;

-- Insert path: each inserted tuple checks the matrix out and releases
-- it. Expected: usage up by exactly 1.0, refcount back at 0.
INSERT INTO cache_t80
    SELECT 999, ('[9' || repeat(',0', 79) || ']')::vector;
SELECT refcount, round(usage::numeric, 4) AS usage
FROM rabitq_params_cache() WHERE dim = 80;

-- Cleanup.
DO $$
DECLARE d int;
BEGIN
    FOREACH d IN ARRAY ARRAY[8, 16, 24, 32, 40, 48, 56, 64, 72, 80] LOOP
        EXECUTE format('DROP TABLE IF EXISTS cache_t%s', d);
    END LOOP;
END $$;

DROP FUNCTION rabitq_params_cache();
DROP FUNCTION rabitq_cache_clear();
RESET enable_seqscan;
