-- DROP INDEX behavior with the shared cache (lazy reclamation).
--
-- The cache frees slots lazily: DROP INDEX does not proactively remove the
-- index's slot. It lingers as an orphan keyed by the now-defunct relfilenode
-- and is reclaimed only LRU-first under later pressure (see centroid_cache_lru).
-- Assertions are scoped to the index's captured relfilenode, so they are
-- robust to the instance-global cache holding other slots.

SET enable_seqscan = off;
SET mkt.enable_centroid_cache = on;
SET mkt.nprobe = 8;

CREATE TABLE d (id serial, v vector(4));
INSERT INTO d (v)
    SELECT format('[%s,%s,%s,%s]',
                  (i % 7) * 0.1, (i % 3) * 0.1, (i % 5) * 0.1, (i % 2) * 0.1
           )::vector
    FROM generate_series(1, 300) i;
CREATE INDEX d_idx ON d USING mktann (v vector_cosine_ops)
    WITH (nlist = 16, fastscan = true, centroid_fastscan = true,
          centroid_compression = true);

-- Build the slot and capture the index's relfilenode.
SELECT count(*) AS hits FROM (
    SELECT id FROM d ORDER BY v <=> '[0.1,0.1,0.1,0.1]' LIMIT 5) t;
SELECT relfilenode AS rfn FROM pg_class WHERE relname = 'd_idx' \gset

-- A slot exists for d_idx.
SELECT count(*) AS slot_before_drop
    FROM mkt.centroid_cache_stats() WHERE relfilenode = :rfn;

DROP INDEX d_idx;

-- Lazy: the slot lingers (keyed by the dropped relfilenode)...
SELECT count(*) AS orphan_slot_after_drop
    FROM mkt.centroid_cache_stats() WHERE relfilenode = :rfn;
-- ...and that relfilenode no longer resolves to any relation (a true orphan).
SELECT count(*) AS rfn_in_pg_class
    FROM pg_class WHERE relfilenode = :rfn;
