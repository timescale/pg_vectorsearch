-- Shared centroid cache (meerkat preloaded): availability, build-on-first-
-- query, FASTSCAN-only gating, result parity, and the zero-budget edge.
--
-- Slots are identified by joining relfilenode to pg_class. We never print raw
-- slot indexes, last_used, byte counts, or result-id lists (none are
-- guaranteed reproducible) — only counts, states, and in-session booleans.

SELECT mkt.centroid_cache_available() AS available;

CREATE TABLE emb (id serial, v vector(8));
INSERT INTO emb (v)
    SELECT format('[%s,%s,%s,%s,%s,%s,%s,%s]',
                  (i % 7) * 0.1, (i % 5) * 0.1, (i % 3) * 0.1, (i % 11) * 0.1,
                  (i % 2) * 0.1, (i % 13) * 0.1, (i % 4) * 0.1, (i % 9) * 0.1
           )::vector
    FROM generate_series(1, 3000) i;

SET enable_seqscan = off;
SET mkt.nprobe = 16;

-- ------------------------------------------------------------------
-- FASTSCAN index: cacheable
-- ------------------------------------------------------------------
CREATE INDEX idx_fs ON emb USING mktann (v vector_cosine_ops)
    WITH (nlist = 64, fastscan = true, centroid_fastscan = true,
          centroid_compression = true);

-- Baseline result with the cache OFF (page reads), captured for parity.
SET mkt.enable_centroid_cache = off;
CREATE TEMP TABLE parity_off AS
    SELECT id FROM emb
    ORDER BY v <=> '[0.3,0.2,0,0.4,0,0.1,0.1,0.2]' LIMIT 10;

-- No slot is built while the cache is off.
SELECT count(*) AS slots_idx_fs_off
    FROM mkt.centroid_cache_stats() s
    JOIN pg_class c ON c.relfilenode = s.relfilenode AND c.relname = 'idx_fs';

-- Cache ON: the first query builds exactly one READY slot for idx_fs.
SET mkt.enable_centroid_cache = on;
CREATE TEMP TABLE parity_on AS
    SELECT id FROM emb
    ORDER BY v <=> '[0.3,0.2,0,0.4,0,0.1,0.1,0.2]' LIMIT 10;

SELECT s.state, s.bytes > 0 AS has_bytes, s.index_len > 0 AS has_index,
       s.refcount
    FROM mkt.centroid_cache_stats() s
    JOIN pg_class c ON c.relfilenode = s.relfilenode AND c.relname = 'idx_fs';

-- Result parity: the cache returns exactly the page-reads result, i.e. the
-- symmetric difference of the two id sets is empty.
SELECT count(*) AS parity_diff FROM (
    (SELECT id FROM parity_off EXCEPT SELECT id FROM parity_on)
    UNION ALL
    (SELECT id FROM parity_on EXCEPT SELECT id FROM parity_off)) d;

-- ------------------------------------------------------------------
-- non-FASTSCAN index: not cacheable (cache is FASTSCAN-only)
-- ------------------------------------------------------------------
DROP INDEX idx_fs;
CREATE INDEX idx_nofs ON emb USING mktann (v vector_cosine_ops)
    WITH (nlist = 64, centroid_fastscan = false, centroid_compression = true);

-- Query it (cache still on) — must not build a slot for a non-fastscan index.
SELECT count(*) AS hits FROM (
    SELECT id FROM emb
    ORDER BY v <=> '[0.3,0.2,0,0.4,0,0.1,0.1,0.2]' LIMIT 10) t;

SELECT count(*) AS slots_idx_nofs
    FROM mkt.centroid_cache_stats() s
    JOIN pg_class c ON c.relfilenode = s.relfilenode AND c.relname = 'idx_nofs';

-- ------------------------------------------------------------------
-- Zero budget: an index whose compact form exceeds the budget is not cached.
-- ------------------------------------------------------------------
DROP INDEX idx_nofs;
CREATE INDEX idx_fs2 ON emb USING mktann (v vector_cosine_ops)
    WITH (nlist = 64, fastscan = true, centroid_fastscan = true,
          centroid_compression = true);

SET mkt.centroid_cache_max_mb = 0;
SELECT count(*) AS hits FROM (
    SELECT id FROM emb
    ORDER BY v <=> '[0.3,0.2,0,0.4,0,0.1,0.1,0.2]' LIMIT 10) t;

SELECT count(*) AS slots_idx_fs2_zero_budget
    FROM mkt.centroid_cache_stats() s
    JOIN pg_class c ON c.relfilenode = s.relfilenode AND c.relname = 'idx_fs2';
RESET mkt.centroid_cache_max_mb;
