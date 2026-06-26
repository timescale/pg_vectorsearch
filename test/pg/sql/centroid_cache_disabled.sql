-- Shared centroid cache when meerkat is NOT preloaded.
--
-- This regress instance does not set shared_preload_libraries, so the cache's
-- shared control struct never exists: the cache must report unavailable,
-- enabling it must error (GUC check hook), and queries must still work by
-- reading centroid pages.

-- Unavailable, and the stats view is empty.
SELECT mkt.centroid_cache_available() AS available;
SELECT count(*) AS stat_rows FROM mkt.centroid_cache_stats();

-- Enabling errors via the check hook (the shared segment only exists when
-- preloaded). The DETAIL names the requirement.
SET mkt.enable_centroid_cache = on;

-- It stayed off.
SHOW mkt.enable_centroid_cache;

-- Queries still work (page reads) with a FASTSCAN index.
CREATE TABLE emb (id serial, v vector(8));
INSERT INTO emb (v)
    SELECT format('[%s,%s,%s,%s,%s,%s,%s,%s]',
                  (i % 7) * 0.1, (i % 5) * 0.1, (i % 3) * 0.1, (i % 11) * 0.1,
                  (i % 2) * 0.1, (i % 13) * 0.1, (i % 4) * 0.1, (i % 9) * 0.1
           )::vector
    FROM generate_series(1, 1000) i;

CREATE INDEX ON emb USING mktann (v vector_cosine_ops)
    WITH (nlist = 32, fastscan = true, centroid_fastscan = true,
          centroid_compression = true);

SET enable_seqscan = off;
SET mkt.nprobe = 8;
SELECT count(*) AS hits FROM (
    SELECT id FROM emb
    ORDER BY v <=> '[0.3,0.2,0,0.4,0,0.1,0.1,0.2]' LIMIT 5) t;

-- Still no slots: the cache never activates when not preloaded.
SELECT count(*) AS stat_rows_after FROM mkt.centroid_cache_stats();
