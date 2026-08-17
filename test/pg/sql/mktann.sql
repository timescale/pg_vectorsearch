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
-- Reconnect first: the cache is per-backend state, and the exact usage
-- arithmetic below relies on a fresh backend whose cache is empty and
-- on counting every checkout from zero, independent of everything the
-- sections above did.
\c -
-- Test-only introspection (the extension ships no SQL surface for the
-- cache): rabitq_params_cache() from the always-built helper module.
-- pg_regress feeds this script to psql on stdin, so relative \ir would
-- resolve against the harness working directory; include by absolute
-- path via the PG_ABS_SRCDIR pg_regress exports.
\getenv abs_srcdir PG_ABS_SRCDIR
\set helper_sql :abs_srcdir '/sql/test_helpers.sql'
\i :helper_sql

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

-- Cleanup.
DO $$
DECLARE d int;
BEGIN
    FOREACH d IN ARRAY ARRAY[8, 16, 24, 32, 40, 48, 56, 64, 72, 80] LOOP
        EXECUTE format('DROP TABLE IF EXISTS cache_t%s', d);
    END LOOP;
END $$;

DROP FUNCTION rabitq_params_cache();
RESET enable_seqscan;
