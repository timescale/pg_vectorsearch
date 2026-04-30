-- mkt_centroid_pages inspection function

-- Create table with vector column
CREATE TABLE embeddings (id serial, v vector(3));

-- Insert deterministic data (grid of vectors)
INSERT INTO embeddings (v)
    SELECT format('[%s,%s,%s]', x * 0.1, y * 0.1, z * 0.1)::vector
    FROM generate_series(0, 9) x,
         generate_series(0, 9) y,
         generate_series(0, 9) z;

-- Single-level RaBitQ index — summary per level
-- (Leaf count varies across platforms due to k-means convergence,
-- so check structure and format without asserting exact counts.)
CREATE INDEX idx_l2c ON embeddings USING mktann (v)
    WITH (centroid_compression = true);

SELECT level,
       count(child_blkno) = count(*) AS all_have_children,
       count(*) FILTER (WHERE is_leaf) = count(*) AS all_are_leaves,
       count(*) > 0 AS has_entries,
       min(format) AS format
    FROM mkt_centroid_pages('idx_l2c'::regclass)
    GROUP BY level ORDER BY level;

-- Multi-level tree (fan_out = 4) — internal nodes show tree topology
CREATE INDEX idx_ml ON embeddings USING mktann (v)
    WITH (fan_out = 4, centroid_compression = true);

SELECT * FROM mkt_centroid_pages('idx_ml'::regclass)
    WHERE NOT is_leaf
    ORDER BY blkno, entry;

-- Leaf summary for multi-level tree
SELECT level, count(*) AS leaf_entries
    FROM mkt_centroid_pages('idx_ml'::regclass)
    WHERE is_leaf
    GROUP BY level
    ORDER BY level;

-- Uncompressed float centroids — verify format string
CREATE INDEX idx_float ON embeddings USING mktann (v);

SELECT level,
       count(child_blkno) = count(*) AS all_have_children,
       count(*) FILTER (WHERE is_leaf) = count(*) AS all_are_leaves,
       count(*) > 0 AS has_entries,
       min(format) AS format
    FROM mkt_centroid_pages('idx_float'::regclass)
    GROUP BY level ORDER BY level;

-- Higher-dim vectors to force page overflow (next_blkno chains).
-- Float format with dim=256: max 7 entries/page, nlist=10 overflows.
CREATE TABLE wide (id serial, v vector(256));

INSERT INTO wide (v)
    SELECT (
        SELECT array_agg(sin(i + j * 0.1)::real)
        FROM generate_series(0, 255) j
    )::vector(256)
    FROM generate_series(1, 100) i;

CREATE INDEX idx_wide ON wide USING mktann (v);

-- Entries from chained pages appear naturally in output
SELECT * FROM mkt_centroid_pages('idx_wide'::regclass)
    ORDER BY blkno, entry;

-- =====================================================================
-- mkt.posting_pages inspection function
-- =====================================================================

-- Posting pages summary for single-level index
-- (Use idx_l2c which has embeddings with 1000 rows, ~32 clusters)
SELECT count(*) > 0 AS has_pages,
       count(DISTINCT cluster_id) > 0 AS has_clusters,
       bool_and(entry_count > 0) AS all_have_entries,
       bool_and(max_entries > 0) AS all_have_capacity,
       bool_and(chain_pos >= 0) AS valid_chain_pos,
       count(*) FILTER (WHERE is_first) > 0 AS has_first_pages
    FROM mkt.posting_pages('idx_l2c'::regclass);

-- First page of each cluster has chain_pos=0
SELECT bool_and(chain_pos = 0) AS first_at_pos_zero
    FROM mkt.posting_pages('idx_l2c'::regclass)
    WHERE is_first;

-- Total entries across all posting pages should equal table row count
SELECT sum(entry_count) AS total_entries
    FROM mkt.posting_pages('idx_l2c'::regclass);

-- Multi-level tree posting pages
SELECT count(*) > 0 AS has_pages,
       count(DISTINCT cluster_id) AS nclusters
    FROM mkt.posting_pages('idx_ml'::regclass);

-- Posting chains: verify next_blkno links are consistent
-- (non-first pages should have chain_pos > 0)
SELECT bool_and(chain_pos > 0) AS continuation_pages_ok
    FROM mkt.posting_pages('idx_l2c'::regclass)
    WHERE NOT is_first;

-- Error case: not an mktann index
CREATE INDEX IF NOT EXISTS idx_btree ON embeddings (id);
SELECT * FROM mkt_centroid_pages('idx_btree'::regclass);
SELECT * FROM mkt.posting_pages('idx_btree'::regclass);

-- Cleanup
DROP TABLE embeddings;
DROP TABLE wide;
