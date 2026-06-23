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
CREATE INDEX idx_float ON embeddings USING mktann (v)
    WITH (centroid_compression = off);

SELECT level,
       count(child_blkno) = count(*) AS all_have_children,
       count(*) FILTER (WHERE is_leaf) = count(*) AS all_are_leaves,
       count(*) > 0 AS has_entries,
       min(format) AS format
    FROM mkt_centroid_pages('idx_float'::regclass)
    GROUP BY level ORDER BY level;

-- centroid_compression tri-state on L2 (default opclass): the default and
-- 'auto' compress, 'on' compresses, 'off' is float (idx_float above).
CREATE INDEX idx_cc_default ON embeddings USING mktann (v);
CREATE INDEX idx_cc_auto ON embeddings USING mktann (v)
    WITH (centroid_compression = auto);
CREATE INDEX idx_cc_on ON embeddings USING mktann (v)
    WITH (centroid_compression = on);
SELECT
    (SELECT min(format) FROM mkt_centroid_pages('idx_cc_default'::regclass))
        AS default_fmt,
    (SELECT min(format) FROM mkt_centroid_pages('idx_cc_auto'::regclass))
        AS auto_fmt,
    (SELECT min(format) FROM mkt_centroid_pages('idx_cc_on'::regclass))
        AS on_fmt;

-- Higher-dim vectors to force page overflow (next_blkno chains).
-- Float format with dim=256: max 7 entries/page, nlist=10 overflows.
-- Pin float (off) so the page-chain layout this test asserts is stable.
CREATE TABLE wide (id serial, v vector(256));

INSERT INTO wide (v)
    SELECT (
        SELECT array_agg(sin(i + j * 0.1)::real)
        FROM generate_series(0, 255) j
    )::vector(256)
    FROM generate_series(1, 100) i;

CREATE INDEX idx_wide ON wide USING mktann (v)
    WITH (centroid_compression = off);

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

-- =====================================================================
-- mkt.convert_posting_to_fastscan
-- =====================================================================

-- Convert cluster 0 from AoS to fastscan
SELECT mkt.convert_posting_to_fastscan('idx_l2c'::regclass, 0) IS NOT NULL
    AS converted;

-- Verify the converted cluster has fastscan format
SELECT format AS cluster0_format
    FROM mkt.posting_pages('idx_l2c'::regclass)
    WHERE cluster_id = 0 AND is_first;

-- Non-converted clusters still show 'aos'
SELECT bool_and(format = 'aos') AS others_aos
    FROM mkt.posting_pages('idx_l2c'::regclass)
    WHERE cluster_id != 0 AND is_first;

-- Converting again should be a no-op (returns same head)
SELECT mkt.convert_posting_to_fastscan('idx_l2c'::regclass, 0) IS NOT NULL
    AS idempotent;

-- Query still works after partial conversion (mixed AoS + fastscan)
SET enable_seqscan = off;
SELECT count(*) FROM (
    SELECT id, v <-> '[0.5,0.5,0.5]' AS dist
    FROM embeddings ORDER BY v <-> '[0.5,0.5,0.5]' LIMIT 5
) t;
RESET enable_seqscan;

-- Convert all remaining clusters
SELECT count(*) AS converted_count FROM (
    SELECT mkt.convert_posting_to_fastscan('idx_l2c'::regclass, cluster_id)
    FROM mkt.posting_pages('idx_l2c'::regclass)
    WHERE is_first AND cluster_id != 0
) t;

-- Query still works after full conversion
SET enable_seqscan = off;
SELECT count(*) FROM (
    SELECT id, v <-> '[0.5,0.5,0.5]' AS dist
    FROM embeddings ORDER BY v <-> '[0.5,0.5,0.5]' LIMIT 5
) t;
RESET enable_seqscan;

-- Error: non-existent cluster_id
SELECT mkt.convert_posting_to_fastscan('idx_l2c'::regclass, 99999);

-- Error: not an mktann index
SELECT mkt.convert_posting_to_fastscan('idx_btree'::regclass, 0);

-- =====================================================================
-- mkt.tids_clusters
-- =====================================================================

-- Fresh index with AoS posting lists for a clean format round-trip.
-- (Cluster ids are k-means dependent, so assert structure, not the
-- specific tid -> cluster mapping.)
CREATE INDEX idx_tc ON embeddings USING mktann (v)
    WITH (centroid_compression = true);

-- AoS path: every heap TID maps to exactly one cluster (no SOAR/boundary
-- replication configured), and every reported cluster_id is a real cluster.
SELECT count(*) = (SELECT count(*) FROM embeddings) AS all_rows_mapped,
       count(DISTINCT tid) = count(*) AS one_cluster_each,
       bool_and(cluster_id IN (
           SELECT cluster_id FROM mkt.posting_pages('idx_tc'::regclass)
       )) AS clusters_valid
    FROM mkt.tids_clusters('idx_tc'::regclass,
                           (SELECT array_agg(ctid) FROM embeddings));

-- The mapping must be identical whether posting lists are AoS or fastscan:
-- capture it, convert every cluster, and diff both directions (0 == equal).
CREATE TEMP TABLE tc_aos AS
    SELECT tid, cluster_id
        FROM mkt.tids_clusters('idx_tc'::regclass,
                               (SELECT array_agg(ctid) FROM embeddings));

SELECT count(*) > 0 AS converted_all FROM (
    SELECT mkt.convert_posting_to_fastscan('idx_tc'::regclass, cluster_id)
        FROM mkt.posting_pages('idx_tc'::regclass)
        WHERE is_first
) t;

CREATE TEMP TABLE tc_fastscan AS
    SELECT tid, cluster_id
        FROM mkt.tids_clusters('idx_tc'::regclass,
                               (SELECT array_agg(ctid) FROM embeddings));

SELECT
    (SELECT count(*) FROM
        (SELECT * FROM tc_aos EXCEPT SELECT * FROM tc_fastscan) a)
        AS aos_only,
    (SELECT count(*) FROM
        (SELECT * FROM tc_fastscan EXCEPT SELECT * FROM tc_aos) b)
        AS fastscan_only;

-- A TID that isn't in the index is simply not reported (no error).
SELECT count(*) AS absent_hits
    FROM mkt.tids_clusters('idx_tc'::regclass, ARRAY['(99999,1)']::tid[]);

DROP TABLE tc_aos;
DROP TABLE tc_fastscan;

-- Cleanup
DROP TABLE embeddings;
DROP TABLE wide;

-- ---------------------------------------------------------------------
-- mkt.git_commit() — exposes the git commit the extension was built
-- from. Verify the contract without printing the actual hash so the
-- expected output stays deterministic across builds.
-- ---------------------------------------------------------------------

-- Format: 40-char lowercase hex (git's full SHA-1) or the "unknown"
-- fallback used when vcs_tag had no git checkout available.
SELECT mkt.git_commit() ~ '^([0-9a-f]{40}|unknown)$' AS valid_format;

-- Callers (e.g. rekall) use this as a string, so the return type
-- must be text.
SELECT pg_typeof(mkt.git_commit())::text = 'text' AS returns_text;

-- The function must be IMMUTABLE STRICT PARALLEL SAFE — the commit
-- doesn't change within a single backend's lifetime, the input is
-- void so STRICT is trivially satisfied, and there's no per-backend
-- state that would break parallel workers.
SELECT
    p.provolatile      = 'i' AS immutable,
    p.proisstrict            AS strict,
    p.proparallel      = 's' AS parallel_safe
FROM pg_proc p
JOIN pg_namespace n ON n.oid = p.pronamespace
WHERE n.nspname = 'mkt' AND p.proname = 'git_commit';
