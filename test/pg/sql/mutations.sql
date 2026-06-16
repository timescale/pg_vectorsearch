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
