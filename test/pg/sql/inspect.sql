-- prism_centroid_pages inspection function

-- Create table with vec32 column
CREATE TABLE embeddings (id serial, v vec32(3));

-- Insert deterministic data (grid of vectors)
INSERT INTO embeddings (v)
    SELECT format('[%s,%s,%s]', x * 0.1, y * 0.1, z * 0.1)::vec32
    FROM generate_series(0, 9) x,
         generate_series(0, 9) y,
         generate_series(0, 9) z;

-- Pin reltuples via ANALYZE so automatic index sizing works from the
-- recorded row count: this test asserts tree structure, which must not
-- wobble with the page-sampling estimator used for never-analyzed
-- tables.
ANALYZE embeddings;

-- Single-level RaBitQ index — summary per level
-- (Leaf count varies across platforms due to k-means convergence,
-- so check structure and format without asserting exact counts.)
CREATE INDEX idx_l2c ON embeddings USING prism (v)
    WITH (centroid_compression = true, centroid_fastscan = off,
          fastscan = off, soar_lambda = 0, boundary_epsilon = 0);

SELECT level,
       count(child_blkno) = count(*) AS all_have_children,
       count(*) FILTER (WHERE is_leaf) = count(*) AS all_are_leaves,
       count(*) > 0 AS has_entries,
       min(format) AS format
    FROM prism_centroid_pages('idx_l2c'::regclass)
    GROUP BY level ORDER BY level;

-- Centroid format follows the column type: a vec16 column stores
-- half-precision centroids, a vec32 column float, with every option
-- otherwise identical. This is a regression test as much as a feature test --
-- the AM resolves vec16's OID during CREATE INDEX, where PostgreSQL
-- has narrowed the search path (RestrictSearchPath in DefineIndex), so an
-- unqualified type lookup silently reports "not vec16" and the whole
-- vec16 input path decodes as float32.
CREATE TABLE h_embeddings (id serial, v vec16(3));
INSERT INTO h_embeddings (v)
    SELECT format('[%s,%s,%s]', x * 0.1, y * 0.1, z * 0.1)::vec16
    FROM generate_series(0, 9) x,
         generate_series(0, 9) y,
         generate_series(0, 9) z;
ANALYZE h_embeddings;
CREATE INDEX h_idx_fmt ON h_embeddings USING prism (v)
    WITH (centroid_compression = off, centroid_fastscan = off,
          fastscan = off, soar_lambda = 0, boundary_epsilon = 0);
SELECT DISTINCT format AS halfvec_centroid_format
    FROM prism_centroid_pages('h_idx_fmt'::regclass);
CREATE INDEX v_idx_fmt ON embeddings USING prism (v)
    WITH (centroid_compression = off, centroid_fastscan = off,
          fastscan = off, soar_lambda = 0, boundary_epsilon = 0);
SELECT DISTINCT format AS vector_centroid_format
    FROM prism_centroid_pages('v_idx_fmt'::regclass);
DROP INDEX v_idx_fmt;
DROP TABLE h_embeddings;

-- Multi-level tree (fan_out = 4) — internal nodes show tree topology
CREATE INDEX idx_ml ON embeddings USING prism (v)
    WITH (fan_out = 4, centroid_compression = true,
          centroid_fastscan = off);

SELECT * FROM prism_centroid_pages('idx_ml'::regclass)
    WHERE NOT is_leaf
    ORDER BY blkno, entry;

-- Leaf summary for multi-level tree
SELECT level, count(*) AS leaf_entries
    FROM prism_centroid_pages('idx_ml'::regclass)
    WHERE is_leaf
    GROUP BY level
    ORDER BY level;

-- Uncompressed float centroids — verify format string
CREATE INDEX idx_float ON embeddings USING prism (v)
    WITH (centroid_compression = off);

SELECT level,
       count(child_blkno) = count(*) AS all_have_children,
       count(*) FILTER (WHERE is_leaf) = count(*) AS all_are_leaves,
       count(*) > 0 AS has_entries,
       min(format) AS format
    FROM prism_centroid_pages('idx_float'::regclass)
    GROUP BY level ORDER BY level;

-- centroid_compression tri-state on L2 (default opclass): the default and
-- 'auto' compress, 'on' compresses, 'off' is float (idx_float above).
CREATE INDEX idx_cc_default ON embeddings USING prism (v)
    WITH (centroid_fastscan = off);
CREATE INDEX idx_cc_auto ON embeddings USING prism (v)
    WITH (centroid_compression = auto, centroid_fastscan = off);
CREATE INDEX idx_cc_on ON embeddings USING prism (v)
    WITH (centroid_compression = on, centroid_fastscan = off);
SELECT
    (SELECT min(format) FROM prism_centroid_pages('idx_cc_default'::regclass))
        AS default_fmt,
    (SELECT min(format) FROM prism_centroid_pages('idx_cc_auto'::regclass))
        AS auto_fmt,
    (SELECT min(format) FROM prism_centroid_pages('idx_cc_on'::regclass))
        AS on_fmt;

-- Higher-dim vectors to force page overflow (next_blkno chains).
-- Float format with dim=256: max 7 entries/page, nlist=10 overflows.
-- Pin float (off) so the page-chain layout this test asserts is stable.
CREATE TABLE wide (id serial, v vec32(256));

INSERT INTO wide (v)
    SELECT (
        SELECT array_agg(sin(i + j * 0.1)::real)
        FROM generate_series(0, 255) j
    )::vec32(256)
    FROM generate_series(1, 100) i;

ANALYZE wide;
CREATE INDEX idx_wide ON wide USING prism (v)
    WITH (centroid_compression = off);

-- Entries from chained pages appear naturally in output
SELECT * FROM prism_centroid_pages('idx_wide'::regclass)
    ORDER BY blkno, entry;

-- =====================================================================
-- prism_posting_pages inspection function
-- =====================================================================

-- Posting pages summary for single-level index
-- (Use idx_l2c which has embeddings with 1000 rows, ~32 clusters)
-- Note: an empty head page (is_first) is valid — page-backed build routing can
-- leave a cluster empty — but an empty continuation page would be a bug, so
-- all_have_entries requires entries on every non-head page.
SELECT count(*) > 0 AS has_pages,
       count(DISTINCT cluster_id) > 0 AS has_clusters,
       bool_and(entry_count > 0 OR is_first) AS all_have_entries,
       bool_and(max_entries > 0) AS all_have_capacity,
       bool_and(chain_pos >= 0) AS valid_chain_pos,
       count(*) FILTER (WHERE is_first) > 0 AS has_first_pages,
       bool_and(NOT tombstoned) AS none_tombstoned,
       bool_and(dead_count = 0) AS none_dead
    FROM prism_posting_pages('idx_l2c'::regclass);

-- First page of each cluster has chain_pos=0
SELECT bool_and(chain_pos = 0) AS first_at_pos_zero
    FROM prism_posting_pages('idx_l2c'::regclass)
    WHERE is_first;

-- Total entries across all posting pages should equal table row count
SELECT sum(entry_count) AS total_entries
    FROM prism_posting_pages('idx_l2c'::regclass);

-- Multi-level tree posting pages
SELECT count(*) > 0 AS has_pages,
       count(DISTINCT cluster_id) AS nclusters
    FROM prism_posting_pages('idx_ml'::regclass);

-- Posting chains: verify next_blkno links are consistent
-- (non-first pages should have chain_pos > 0)
SELECT bool_and(chain_pos > 0) AS continuation_pages_ok
    FROM prism_posting_pages('idx_l2c'::regclass)
    WHERE NOT is_first;

-- Error case: not a prism index
CREATE INDEX IF NOT EXISTS idx_btree ON embeddings (id);
SELECT * FROM prism_centroid_pages('idx_btree'::regclass);
SELECT * FROM prism_posting_pages('idx_btree'::regclass);

-- =====================================================================
-- prism_convert_posting_to_fastscan
-- =====================================================================

-- Convert cluster 0 from AoS to fastscan
SELECT prism_convert_posting_to_fastscan('idx_l2c'::regclass, 0) IS NOT NULL
    AS converted;

-- Verify the converted cluster has fastscan format
SELECT format AS cluster0_format
    FROM prism_posting_pages('idx_l2c'::regclass)
    WHERE cluster_id = 0 AND is_first;

-- Non-converted clusters still show 'aos'
SELECT bool_and(format = 'aos') AS others_aos
    FROM prism_posting_pages('idx_l2c'::regclass)
    WHERE cluster_id != 0 AND is_first;

-- Converting again should be a no-op (returns same head)
SELECT prism_convert_posting_to_fastscan('idx_l2c'::regclass, 0) IS NOT NULL
    AS idempotent;

-- Query still works after partial conversion (mixed AoS + fastscan).
--
-- A small table, so a sequential scan is the plan the cost model correctly
-- prefers and the scans here are forced through the index. The rows come
-- back either way, so the plan is pinned once here.
SET enable_seqscan = off;
EXPLAIN (COSTS OFF)
    SELECT id FROM embeddings ORDER BY v <-> '[0.5,0.5,0.5]' LIMIT 5;
SELECT count(*) FROM (
    SELECT id, v <-> '[0.5,0.5,0.5]' AS dist
    FROM embeddings ORDER BY v <-> '[0.5,0.5,0.5]' LIMIT 5
) t;
RESET enable_seqscan;

-- Convert all remaining clusters
SELECT count(*) AS converted_count FROM (
    SELECT prism_convert_posting_to_fastscan('idx_l2c'::regclass, cluster_id)
    FROM prism_posting_pages('idx_l2c'::regclass)
    WHERE is_first AND cluster_id != 0
) t;

-- Query still works after full conversion
SET enable_seqscan = off;
SELECT count(*) FROM (
    SELECT id, v <-> '[0.5,0.5,0.5]' AS dist
    FROM embeddings ORDER BY v <-> '[0.5,0.5,0.5]' LIMIT 5
) t;
RESET enable_seqscan;

-- Entries VACUUM has marked dead must not survive the conversion. A fastscan
-- page deletes at page granularity -- there is no per-entry flag -- so a dead
-- entry copied into one comes back as live and can never be marked again: a
-- later VACUUM can only tombstone a page once every entry on it is dead, which
-- a page holding live entries never is. The count after converting is the
-- assertion: it has to be the live rows, not the rows the list was built with.
CREATE TABLE conv_dead (id int, v vec32(4));
INSERT INTO conv_dead
    SELECT g, format('[%s,0,0,0]', g)::vec32(4) FROM generate_series(1, 40) g;
CREATE INDEX conv_dead_idx ON conv_dead USING prism (v)
    WITH (nlist = 1, fastscan = off, centroid_fastscan = off);

DELETE FROM conv_dead WHERE id % 3 = 0;
VACUUM conv_dead;

-- 40 entries, 13 of them now dead
SELECT format, entry_count, dead_count
    FROM prism_posting_pages('conv_dead_idx') WHERE is_first;

SELECT prism_convert_posting_to_fastscan('conv_dead_idx', cluster_id) IS NOT NULL
        AS converted
    FROM prism_posting_pages('conv_dead_idx') WHERE is_first;

-- 27 rows remain, so the fastscan list must hold 27 entries
SELECT format, entry_count FROM prism_posting_pages('conv_dead_idx')
    WHERE is_first;
SELECT count(*) AS live_rows FROM conv_dead;

-- The count matters because a resurrected entry is not merely stale: it takes
-- a slot in the candidate set a scan collects, so a query asking for k rows
-- gets fewer than k, and fewer than exist. Ten is well inside the 27 rows that
-- remain, so all ten must come back. (Checking the ids are not the deleted
-- ones would prove nothing -- their heap rows are gone, so the executor
-- filters them out whether or not the index still points at them.)
SET enable_seqscan = off;
SET prism.nprobe = 4;
SELECT count(*) AS rows_for_limit_10 FROM (
    SELECT id FROM conv_dead
    ORDER BY v <-> '[1,0,0,0]'::vec32(4) LIMIT 10) t;
RESET prism.nprobe;
RESET enable_seqscan;
DROP TABLE conv_dead;

-- Error: non-existent cluster_id
SELECT prism_convert_posting_to_fastscan('idx_l2c'::regclass, 99999);

-- Error: not a prism index
SELECT prism_convert_posting_to_fastscan('idx_btree'::regclass, 0);

-- =====================================================================
-- prism_tids_clusters
-- =====================================================================

-- Fresh index with AoS posting lists for a clean format round-trip.
-- (Cluster ids are k-means dependent, so assert structure, not the
-- specific tid -> cluster mapping.)
CREATE INDEX idx_tc ON embeddings USING prism (v)
    WITH (centroid_compression = true, fastscan = off,
          soar_lambda = 0, boundary_epsilon = 0);

-- AoS path: every heap TID maps to exactly one cluster (no SOAR/boundary
-- replication configured), and every reported cluster_id is a real cluster.
SELECT count(*) = (SELECT count(*) FROM embeddings) AS all_rows_mapped,
       count(DISTINCT tid) = count(*) AS one_cluster_each,
       bool_and(cluster_id IN (
           SELECT cluster_id FROM prism_posting_pages('idx_tc'::regclass)
       )) AS clusters_valid
    FROM prism_tids_clusters('idx_tc'::regclass,
                           (SELECT array_agg(ctid) FROM embeddings));

-- The mapping must be identical whether posting lists are AoS or fastscan:
-- capture it, convert every cluster, and diff both directions (0 == equal).
CREATE TEMP TABLE tc_aos AS
    SELECT tid, cluster_id
        FROM prism_tids_clusters('idx_tc'::regclass,
                               (SELECT array_agg(ctid) FROM embeddings));

SELECT count(*) > 0 AS converted_all FROM (
    SELECT prism_convert_posting_to_fastscan('idx_tc'::regclass, cluster_id)
        FROM prism_posting_pages('idx_tc'::regclass)
        WHERE is_first
) t;

CREATE TEMP TABLE tc_fastscan AS
    SELECT tid, cluster_id
        FROM prism_tids_clusters('idx_tc'::regclass,
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
    FROM prism_tids_clusters('idx_tc'::regclass, ARRAY['(99999,1)']::tid[]);

DROP TABLE tc_aos;
DROP TABLE tc_fastscan;

-- =====================================================================
-- Privilege checks
-- =====================================================================
-- The inspection functions expose an index's internal layout, so they
-- are gated on the caller's privileges on the underlying table (the same
-- model pgrowlocks uses). EXECUTE stays public; the checks are runtime.
-- Read-only inspectors need SELECT on the table; convert_posting_to_
-- fastscan mutates the index and needs table ownership.

CREATE ROLE regress_inspect_unpriv NOLOGIN;
-- No schema GRANT is needed: the functions and embeddings/idx_l2c above
-- all live in public (created unqualified), whose USAGE is granted to
-- PUBLIC by default.

SET ROLE regress_inspect_unpriv;

-- Without SELECT on embeddings, every read-only inspector is denied.
SELECT * FROM prism_centroid_pages('idx_l2c'::regclass);
SELECT * FROM prism_posting_pages('idx_l2c'::regclass);
SELECT * FROM prism_tids_clusters('idx_l2c'::regclass, ARRAY['(0,1)']::tid[]);
SELECT * FROM prism_index_settings('idx_l2c'::regclass);

-- The mutating function requires ownership, not merely SELECT.
SELECT prism_convert_posting_to_fastscan('idx_l2c'::regclass, 0);

RESET ROLE;

-- Granting SELECT on the table lets the read-only inspectors run...
GRANT SELECT ON embeddings TO regress_inspect_unpriv;

SET ROLE regress_inspect_unpriv;

SELECT count(*) > 0 AS centroid_ok
    FROM prism_centroid_pages('idx_l2c'::regclass);
SELECT count(*) > 0 AS posting_ok
    FROM prism_posting_pages('idx_l2c'::regclass);
SELECT count(*) >= 0 AS tids_ok
    FROM prism_tids_clusters('idx_l2c'::regclass, ARRAY['(0,1)']::tid[]);
SELECT count(*) > 0 AS settings_ok
    FROM prism_index_settings('idx_l2c'::regclass);

-- ...but SELECT is still not enough to mutate the index.
SELECT prism_convert_posting_to_fastscan('idx_l2c'::regclass, 0);

RESET ROLE;

REVOKE SELECT ON embeddings FROM regress_inspect_unpriv;
DROP ROLE regress_inspect_unpriv;

-- Cleanup
-- =====================================================================
-- Default index shape
-- =====================================================================
-- With no options, an L2/cosine index resolves to the tuned defaults:
-- FASTSCAN centroid and posting layouts, and SOAR + boundary
-- replication (so posting entries exceed the row count while
-- prism_tids_clusters still reports every row, deduplicated by TID).
CREATE INDEX idx_default ON embeddings USING prism (v);

SELECT (SELECT min(format)
            FROM prism_centroid_pages('idx_default'::regclass))
            AS centroid_fmt,
       (SELECT min(format) FROM prism_posting_pages('idx_default'::regclass)
            WHERE is_first) AS posting_fmt;

SELECT sum(entry_count) >= (SELECT count(*) FROM embeddings) AS replicated
    FROM prism_posting_pages('idx_default'::regclass);

SELECT count(DISTINCT tid) = (SELECT count(*) FROM embeddings)
        AS all_rows_mapped
    FROM prism_tids_clusters('idx_default'::regclass,
                           (SELECT array_agg(ctid) FROM embeddings));

-- Queries are served by the index and return exact top-1 on a
-- distinct-distance probe (nprobe >= nlist: scan every list, so the
-- result does not depend on routing).
SET enable_seqscan = off;
SET prism.nprobe = 64;
SELECT id FROM embeddings ORDER BY v <-> '[0.31,0.32,0.33]' LIMIT 1;
RESET prism.nprobe;
RESET enable_seqscan;

-- =====================================================================
-- prism_index_settings
-- =====================================================================

-- Explicit options are reported back as set, with source 'option'.
-- (nlist and nlevels are asserted by predicate, not value: the built
-- cluster count can vary with k-means convergence across platforms.)
CREATE INDEX idx_settings ON embeddings USING prism (v)
    WITH (nlist = 20, fan_out = 8, centroid_compression = on,
          centroid_fastscan = off, fastscan = off, soar_lambda = 0.5,
          boundary_epsilon = 0.1, kmeans_nredo = 2,
          distance_mode = symmetric);

SELECT name, setting, source
    FROM prism_index_settings('idx_settings'::regclass)
    WHERE name NOT IN ('nlist', 'nlevels');

SELECT name, setting::int > 0 AS positive, source
    FROM prism_index_settings('idx_settings'::regclass)
    WHERE name IN ('nlist', 'nlevels');

-- The default index resolves its automatic settings: every value is
-- concrete (no 'auto' sentinels like nlist=0) and the sources tell
-- that defaults were in effect. The small table resolves to the
-- auto-nprobe floor of 10.
SELECT name,
       setting <> '0' AND setting <> '' AS resolved,
       source
    FROM prism_index_settings('idx_default'::regclass);

SELECT name, setting, source
    FROM prism_index_settings('idx_default'::regclass)
    WHERE name = 'nprobe';

-- Session GUCs override the index setting and report source 'session'.
SET prism.nprobe = 33;
SET prism.distance_mode = 'symmetric';
SELECT name, setting, source
    FROM prism_index_settings('idx_default'::regclass)
    WHERE name IN ('distance_mode', 'nprobe');
RESET prism.nprobe;
RESET prism.distance_mode;

-- Error: not a prism index
SELECT * FROM prism_index_settings('idx_btree'::regclass);

DROP INDEX idx_settings;
DROP INDEX idx_default;

DROP TABLE embeddings;
DROP TABLE wide;

-- ---------------------------------------------------------------------
-- pg_vectorsearch_git_commit() — exposes the git commit the extension
-- was built from. Verify the contract without printing the actual hash
-- so the expected output stays deterministic across builds. Extension-
-- wide, not index-specific, so it follows @extschema@ like the
-- vec32/vec16 types rather than living in the fixed prism schema.
-- ---------------------------------------------------------------------

-- Format: 40-char lowercase hex (git's full SHA-1) or the "unknown"
-- fallback used when vcs_tag had no git checkout available.
SELECT pg_vectorsearch_git_commit() ~ '^([0-9a-f]{40}|unknown)$'
    AS valid_format;

-- Callers (e.g. rekall) use this as a string, so the return type
-- must be text.
SELECT pg_typeof(pg_vectorsearch_git_commit())::text = 'text'
    AS returns_text;

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
WHERE n.nspname = 'public' AND p.proname = 'pg_vectorsearch_git_commit';

-- ---------------------------------------------------------------------
-- pg_vectorsearch_version() — the extension version the loaded library
-- was built as.
-- ---------------------------------------------------------------------

-- The binary's version must match the installed extension's version;
-- a mismatch means the library and the SQL scripts come from
-- different builds (e.g. a stale install).
SELECT pg_vectorsearch_version() =
    (SELECT extversion FROM pg_extension WHERE extname = 'pg_vectorsearch')
    AS version_matches_extension;

-- Same contract as pg_vectorsearch_git_commit() above: consumed as a
-- string (e.g. by rekall), so it must return text and be IMMUTABLE
-- STRICT PARALLEL SAFE.
SELECT pg_typeof(pg_vectorsearch_version())::text = 'text'
    AS returns_text;

SELECT
    p.proname,
    p.provolatile      = 'i' AS immutable,
    p.proisstrict            AS strict,
    p.proparallel      = 's' AS parallel_safe
FROM pg_proc p
JOIN pg_namespace n ON n.oid = p.pronamespace
WHERE n.nspname = 'public' AND p.proname = 'pg_vectorsearch_version';
