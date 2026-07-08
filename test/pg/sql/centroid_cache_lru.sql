-- LRU eviction at the cache's slot capacity (CC_MAX_SLOTS = 64).
--
-- Build 65 tiny FASTSCAN indexes on distinct tables and query them in order.
-- The cache holds at most 64 slots, so a query that needs a new slot evicts
-- the least-recently-used unpinned one. The cache is instance-global and may
-- already hold orphan slots from earlier work, so every assertion is scoped to
-- this test's own indexes (relname LIKE 'li%'); because we query 65 freshly,
-- ours are always the most-recently-used and any older orphans are evicted
-- first.

SET enable_seqscan = off;
SET mkt.enable_centroid_cache = on;
SET mkt.nprobe = 4;

-- 65 tiny tables + fastscan indexes (nlist=8 over 200 rows: ~25 rows/cluster,
-- well clear of degenerate clustering).
DO $$
BEGIN
  FOR k IN 1..65 LOOP
    EXECUTE format('CREATE TABLE lt%s (id serial, v vector(4))', k);
    EXECUTE format($i$INSERT INTO lt%s (v)
        SELECT format('[%%s,%%s,%%s,%%s]',
                      (i%%7)*0.1, (i%%3)*0.1, (i%%5)*0.1, (i%%2)*0.1)::vector
        FROM generate_series(1, 200) i$i$, k);
    EXECUTE format($x$CREATE INDEX li%s ON lt%s USING mktann (v vector_cosine_ops)
        WITH (nlist = 8, fastscan = true, centroid_fastscan = true,
              centroid_compression = true)$x$, k, k);
  END LOOP;
END $$;

-- Query li1..li64 in order: li1 ends up least-recently-used, li64 most.
DO $$
BEGIN
  FOR k IN 1..64 LOOP
    EXECUTE format($q$SELECT id FROM lt%s
        ORDER BY v <=> '[0.1,0.1,0.1,0.1]' LIMIT 3$q$, k);
  END LOOP;
END $$;

-- All 64 of ours are cached (capacity is 64 and ours are the newest, so any
-- pre-existing orphans were evicted to make room).
SELECT count(*) AS mine_cached
    FROM mkt.centroid_cache_stats() s
    JOIN pg_class c ON c.relfilenode = s.relfilenode
    WHERE c.relname LIKE 'li%';

-- Query li65: cache is full -> evict the LRU slot, which is li1.
SELECT count(*) AS hits FROM (
    SELECT id FROM lt65 ORDER BY v <=> '[0.1,0.1,0.1,0.1]' LIMIT 3) t;

SELECT
    (SELECT count(*) FROM mkt.centroid_cache_stats() s
       JOIN pg_class c ON c.relfilenode = s.relfilenode
       WHERE c.relname = 'li1')  AS li1_evicted,
    (SELECT count(*) FROM mkt.centroid_cache_stats() s
       JOIN pg_class c ON c.relfilenode = s.relfilenode
       WHERE c.relname = 'li65') AS li65_present,
    (SELECT count(*) FROM mkt.centroid_cache_stats() s
       JOIN pg_class c ON c.relfilenode = s.relfilenode
       WHERE c.relname = 'li2')  AS li2_still;

-- Re-query li1: rebuilt into a fresh slot, evicting the next LRU victim (li2).
SELECT count(*) AS hits FROM (
    SELECT id FROM lt1 ORDER BY v <=> '[0.1,0.1,0.1,0.1]' LIMIT 3) t;

SELECT
    (SELECT count(*) FROM mkt.centroid_cache_stats() s
       JOIN pg_class c ON c.relfilenode = s.relfilenode
       WHERE c.relname = 'li1') AS li1_rebuilt,
    (SELECT count(*) FROM mkt.centroid_cache_stats() s
       JOIN pg_class c ON c.relfilenode = s.relfilenode
       WHERE c.relname = 'li2') AS li2_evicted;

-- The cache never exceeds its capacity.
SELECT count(*) <= 64 AS within_capacity FROM mkt.centroid_cache_stats();
