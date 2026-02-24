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

-- Error case: not an mktann index
CREATE INDEX idx_btree ON embeddings (id);
SELECT * FROM mkt_centroid_pages('idx_btree'::regclass);

-- Cleanup
DROP TABLE embeddings;
DROP TABLE wide;
