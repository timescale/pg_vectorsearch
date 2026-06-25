-- Phase 0 incremental inserts across index variants.
--
-- aminsert indexes new rows into an existing index. Covered for AoS-only,
-- FASTSCAN (where inserts form a mixed AoS-overflow chain on the packed base),
-- and empty indexes (build is refused), plus nearest-neighbour membership of a
-- freshly inserted vector. Probe vectors are widely separated so ANN ranking is
-- deterministic; region membership is asserted by id range where gaps are
-- small.

-- ===== AoS-only index (centroid_compression on, no fastscan) ================
CREATE TABLE mut (id int, v vector(3));
INSERT INTO mut SELECT g, format('[%s,0,0]', g)::vector FROM generate_series(1, 50) g;
CREATE INDEX idx_mut ON mut USING mktann (v)
    WITH (nlist = 4, centroid_compression = true);
SET enable_seqscan = off;
SET mkt.nprobe = 4;             -- = nlist: scan every list, deterministic

INSERT INTO mut VALUES (1001, '[100,0,0]');
SELECT count(*) AS aos_ins FROM (
    SELECT id FROM mut ORDER BY v <-> '[100,0,0]' LIMIT 1) t WHERE id = 1001;
-- Bulk insert into a new region grows AoS overflow pages. Verify that EVERY
-- row was indexed, not just one. A nearest-neighbour query is the wrong probe
-- for this: these axis-aligned 1-D vectors all quantize to near-identical
-- RaBitQ codes, so approximate ranking does not surface all 40 even when all
-- are present. Instead count physical index entries directly via
-- mkt.posting_pages, which is independent of search recall: the index must
-- hold all 91 rows (50 initial + the 1001 outlier + 40 bulk).
INSERT INTO mut SELECT 2000 + g, format('[%s,0,0]', 300 + g)::vector
    FROM generate_series(1, 40) g;
SELECT sum(entry_count) AS aos_total
    FROM mkt.posting_pages('idx_mut'::regclass);
-- Separately, ANN ranking must return the true nearest neighbour: the probe
-- matches row 2020 exactly (distance 0), so it is the unambiguous top-1.
SELECT id AS aos_bulk_nn FROM mut ORDER BY v <-> '[320,0,0]' LIMIT 1;
RESET enable_seqscan;
RESET mkt.nprobe;
DROP TABLE mut;

-- ===== FASTSCAN index: inserts append an AoS overflow page (mixed chain) =====
CREATE TABLE mutfs (id int, v vector(3));
INSERT INTO mutfs SELECT g, format('[%s,0,0]', g)::vector
    FROM generate_series(1, 60) g;
CREATE INDEX idx_mutfs ON mutfs USING mktann (v)
    WITH (nlist = 4, centroid_compression = true, fastscan = true);
SET enable_seqscan = off;
SET mkt.nprobe = 4;

INSERT INTO mutfs VALUES (1001, '[500,0,0]');
SELECT count(*) AS fs_ins FROM (
    SELECT id FROM mutfs ORDER BY v <-> '[500,0,0]' LIMIT 1) t WHERE id = 1001;
-- mixed chain: a query whose top-2 spans the FASTSCAN base (id 60) and the
-- AoS-inserted neighbour (id 61) must return both.
INSERT INTO mutfs VALUES (61, '[61,0,0]');
SELECT count(*) AS fs_mixed FROM (
    SELECT id FROM mutfs ORDER BY v <-> '[61,0,0]' LIMIT 2) t
    WHERE id IN (60, 61);
RESET enable_seqscan;
RESET mkt.nprobe;
DROP TABLE mutfs;

-- ===== Empty index: building on a 0-row table is refused =====================
-- Phase 0 limitation: an empty heap has no centroids to route inserts to, so
-- the build fails loudly instead of producing an unusable index. A later phase
-- can build a degenerate single-cluster index; when it does, this expected
-- output must be updated.
CREATE TABLE mutempty (id int, v vector(3));
CREATE INDEX idx_mutempty ON mutempty USING mktann (v)
    WITH (nlist = 4, centroid_compression = true);  -- expect ERROR
DROP TABLE mutempty;

-- ===== A freshly inserted vector is a genuine nearest neighbour =============
CREATE TABLE mutnn (id int, v vector(3));
INSERT INTO mutnn SELECT g, format('[%s,0,0]', g)::vector
    FROM generate_series(1, 50) g;
CREATE INDEX idx_mutnn ON mutnn USING mktann (v)
    WITH (nlist = 4, centroid_compression = true);
SET enable_seqscan = off;
SET mkt.nprobe = 4;
INSERT INTO mutnn VALUES (1001, '[100,0,0]');
-- query NEAR (not equal to) the inserted vector; it must be the top neighbour
SELECT count(*) AS nn_member FROM (
    SELECT id FROM mutnn ORDER BY v <-> '[99,0,0]' LIMIT 1) t WHERE id = 1001;
RESET enable_seqscan;
RESET mkt.nprobe;
DROP TABLE mutnn;

-- ===== DELETE: dead rows leave the result; VACUUM tombstones the entries =====
-- A deleted row must never be returned by an index scan (MVCC filters it even
-- before VACUUM), and VACUUM's ambulkdelete must run cleanly and tombstone the
-- dead entries while leaving the rest of the index correct and usable. Probes
-- are outliers so the assertions are exact-NN, not recall-dependent.
CREATE TABLE mutdel (id int, v vector(3));
INSERT INTO mutdel SELECT g, format('[%s,0,0]', g)::vector
    FROM generate_series(1, 50) g;
INSERT INTO mutdel VALUES (1001, '[100,0,0]'), (1002, '[200,0,0]');
CREATE INDEX idx_mutdel ON mutdel USING mktann (v)
    WITH (nlist = 4, centroid_compression = true);
SET enable_seqscan = off;
SET mkt.nprobe = 4;

-- Guard: index serves the probe, not a seq scan (see the AoS section).
EXPLAIN (COSTS OFF) SELECT id FROM mutdel ORDER BY v <-> '[100,0,0]' LIMIT 1;
-- Present before delete: the [100] outlier is its own nearest neighbour.
SELECT id AS del_before FROM mutdel ORDER BY v <-> '[100,0,0]' LIMIT 1;
DELETE FROM mutdel WHERE id = 1001;
-- Gone immediately (MVCC recheck), even though VACUUM has not run: the nearest
-- live row to [100] is now id 50, not the deleted 1001.
SELECT id AS del_after FROM mutdel ORDER BY v <-> '[100,0,0]' LIMIT 1;
-- Physical entries are unchanged by DELETE (no reclaim yet): 52 rows indexed.
SELECT sum(entry_count) AS del_entries_pre_vacuum
    FROM mkt.posting_pages('idx_mutdel'::regclass);
VACUUM mutdel;   -- runs ambulkdelete: tombstones the dead 1001 entry
-- Still correct after VACUUM, and the other outlier is unaffected.
SELECT id AS del_after_vacuum FROM mutdel ORDER BY v <-> '[100,0,0]' LIMIT 1;
SELECT id AS del_other FROM mutdel ORDER BY v <-> '[200,0,0]' LIMIT 1;
-- Tombstones are skipped, not physically removed in Phase 0, so the page entry
-- count is unchanged; reclaim happens at a later compaction/rebuild.
SELECT sum(entry_count) AS del_entries_post_vacuum
    FROM mkt.posting_pages('idx_mutdel'::regclass);
RESET enable_seqscan;
RESET mkt.nprobe;
DROP TABLE mutdel;

-- ===== UPDATE: insert-new + clean-old, and HOT for non-indexed columns =======
-- Postgres turns a vector-column UPDATE into a new tuple (aminsert) plus a dead
-- old version (cleaned by VACUUM, like DELETE). An UPDATE that leaves the
-- vector unchanged is HOT (index_unchanged), so no index work happens and the
-- row stays findable. The payload column is non-indexed, to drive the HOT case.
CREATE TABLE mutupd (id int, payload int, v vector(3));
INSERT INTO mutupd SELECT g, 0, format('[%s,0,0]', g)::vector
    FROM generate_series(1, 50) g;
INSERT INTO mutupd VALUES (1001, 0, '[100,0,0]');
CREATE INDEX idx_mutupd ON mutupd USING mktann (v)
    WITH (nlist = 4, centroid_compression = true);
SET enable_seqscan = off;
SET mkt.nprobe = 4;

-- Guard: index serves the probe, not a seq scan (see the AoS section).
EXPLAIN (COSTS OFF) SELECT id FROM mutupd ORDER BY v <-> '[100,0,0]' LIMIT 1;
-- Before: the outlier sits at [100].
SELECT id AS upd_before FROM mutupd ORDER BY v <-> '[100,0,0]' LIMIT 1;
-- Move the vector to [200]: non-HOT, so aminsert indexes the new version.
UPDATE mutupd SET v = '[200,0,0]' WHERE id = 1001;
-- New location is found ...
SELECT id AS upd_new_loc FROM mutupd ORDER BY v <-> '[200,0,0]' LIMIT 1;
-- ... and the old location no longer returns it (dead old version, MVCC): the
-- nearest live row to [100] is id 50.
SELECT id AS upd_old_loc FROM mutupd ORDER BY v <-> '[100,0,0]' LIMIT 1;
-- HOT update of a non-indexed column does no index work; the row is unchanged
-- in the index and still found at [200].
UPDATE mutupd SET payload = payload + 1 WHERE id = 1001;
SELECT id AS upd_hot FROM mutupd ORDER BY v <-> '[200,0,0]' LIMIT 1;
-- VACUUM cleans the dead old version; the current version stays findable.
VACUUM mutupd;
SELECT id AS upd_after_vacuum FROM mutupd ORDER BY v <-> '[200,0,0]' LIMIT 1;
RESET enable_seqscan;
RESET mkt.nprobe;
DROP TABLE mutupd;
