-- Query-time GUC behavior
--
-- Home for tests of the mkt.* GUCs that shape an index scan (query
-- limit, probe count, rerank pool, ...): defaults, bounds, and their
-- interaction with the query.

-- Tie-free deterministic data (grid data would produce distance ties)
CREATE TABLE query_settings_test (id serial, v vector(32));

INSERT INTO query_settings_test (v)
    SELECT (
        SELECT array_agg(sin(i * 0.1 + j * 0.7)::real)
        FROM generate_series(0, 31) j
    )::vector(32)
    FROM generate_series(1, 500) i;

CREATE INDEX idx_query_settings ON query_settings_test USING mktann (v)
    WITH (centroid_compression = true);

SET enable_seqscan = off;

-- Probe everything so results are deterministic.
SET mkt.nprobe = 1000;

-- mkt.query_limit sizes the scan's top-K, so a LIMIT above the
-- built-in default k returns the full result count (a regression here
-- silently truncates results to the default).
SET mkt.query_limit = 50;
SELECT count(*) AS rows_k50 FROM (
    SELECT id FROM query_settings_test
    ORDER BY v <-> (SELECT v FROM query_settings_test WHERE id = 42)
    LIMIT 50) t;

SET mkt.query_limit = 200;
SELECT count(*) AS rows_k200 FROM (
    SELECT id FROM query_settings_test
    ORDER BY v <-> (SELECT v FROM query_settings_test WHERE id = 42)
    LIMIT 200) t;

-- A LIMIT below query_limit returns the LIMIT, not the GUC value.
SET mkt.query_limit = 100;
SELECT count(*) AS rows_limit_below_guc FROM (
    SELECT id FROM query_settings_test
    ORDER BY v <-> (SELECT v FROM query_settings_test WHERE id = 42)
    LIMIT 7) t;

-- The enlarged top-K must hold the right rows, not merely enough rows:
-- the index top-50 must match the exact top-50.
SET mkt.query_limit = 50;
CREATE TEMP TABLE result_k50 AS
    SELECT id FROM query_settings_test
    ORDER BY v <-> (SELECT v FROM query_settings_test WHERE id = 42)
    LIMIT 50;
RESET enable_seqscan;

SET enable_indexscan = off;
CREATE TEMP TABLE truth_k50 AS
    SELECT id FROM query_settings_test
    ORDER BY v <-> (SELECT v FROM query_settings_test WHERE id = 42)
    LIMIT 50;
RESET enable_indexscan;

SELECT count(*) AS matching_top50
    FROM truth_k50 t JOIN result_k50 r USING (id);

-- Default (query_limit = 0): the scan returns at most the built-in
-- default k regardless of a larger LIMIT. This pins the current
-- contract; update deliberately if the default ever changes.
RESET mkt.query_limit;
SET enable_seqscan = off;
SELECT count(*) AS rows_default_guc FROM (
    SELECT id FROM query_settings_test
    ORDER BY v <-> (SELECT v FROM query_settings_test WHERE id = 42)
    LIMIT 50) t;

RESET enable_seqscan;
RESET mkt.nprobe;

-- ============================================================
-- mkt.probe_expand: the routing-depth diagnostic reports scan ranks
-- ============================================================
-- With probe expansion the beam routes extra cluster candidates and
-- scan_clusters re-ranks them by exact centroid distance before scanning
-- the best nprobe. The deepest-contributing-rank diagnostic
-- (mkt_routing_stats) must report positions in that final scan order, so
-- with nprobe = 8 every reported rank is below 8 -- the histogram's <=8
-- bucket holds 100% of queries. Ranks above nprobe are impossible unless
-- the diagnostic leaks pre-re-rank beam indexes.
CREATE FUNCTION mkt_routing_stats() RETURNS text
    AS '$libdir/meerkat', 'mkt_routing_stats' LANGUAGE C;
CREATE FUNCTION mkt_phase_stats_reset() RETURNS void
    AS '$libdir/meerkat', 'mkt_phase_stats_reset' LANGUAGE C;

-- 40 clusters of 50 points: each cluster sits on a pseudo-random direction
-- at radius ~1 from the origin, members add small deterministic noise. A
-- probe near the origin is nearly equidistant from every cluster, so the
-- exact probe order is decided by near-ties that the 1-bit beam estimates
-- scramble -- exactly the regime probe expansion exists to repair, and the
-- one where beam indexes differ visibly from scan ranks.
CREATE TABLE probe_depth_test (id serial, v vector(32));
INSERT INTO probe_depth_test (v)
    SELECT (SELECT array_agg((sin(c * 12.9898 + j * 78.233) +
                              0.05 * sin((c * 50 + m) * 3.7 + j * 1.3))::real)
            FROM generate_series(0, 31) j)::vector(32)
    FROM generate_series(0, 39) c, generate_series(1, 50) m;
CREATE INDEX probe_depth_idx ON probe_depth_test USING mktann (v)
    WITH (centroid_compression = true, nlist = 40);

SET enable_seqscan = off;
SET mkt.nprobe = 8;
SET mkt.probe_expand = 4.0;
SELECT mkt_phase_stats_reset();
DO $$
DECLARE q vector(32);
BEGIN
    FOR g IN 1..20 LOOP
        SELECT (SELECT array_agg((0.03 * sin(g * 7.7 + j * 2.9))::real)
                FROM generate_series(0, 31) j)::vector(32) INTO q;
        PERFORM id FROM probe_depth_test ORDER BY v <-> q LIMIT 10;
    END LOOP;
END $$;
SELECT mkt_routing_stats() LIKE '%<=8:100.0%%' AS ranks_within_nprobe;

RESET mkt.probe_expand;
RESET mkt.nprobe;
RESET enable_seqscan;
DROP TABLE probe_depth_test;
DROP FUNCTION mkt_routing_stats();
DROP FUNCTION mkt_phase_stats_reset();

-- ============================================================
-- mkt.centroid_beam_scale: a narrow beam must keep every leaf reachable
-- ============================================================
-- The beam scale narrows the intermediate-level beam to nprobe * scale.
-- An intermediate keep of W can expose at most W * fan_out leaves, so
-- covering the top nprobe leaves requires W >= nprobe / fan_out. At
-- fan_out = 3 the default scale (0.25) computes W below that bound for
-- every possible leaf count, so entire clusters drop out of reach at
-- ANY nprobe: probing every list must return every row, at the default
-- beam scale, for an index whose fan_out is smaller than 1 / scale.
CREATE TABLE beam_floor_test (id int, v vector(3));
INSERT INTO beam_floor_test SELECT g, format('[%s,0,0]', g)::vector
    FROM generate_series(1, 60) g;
CREATE INDEX beam_floor_idx ON beam_floor_test USING mktann (v)
    WITH (nlist = 12, fan_out = 3);
SET enable_seqscan = off;
SET mkt.nprobe = 10000;
SET mkt.query_limit = 100;
SELECT count(*) = 60 AS all_rows_reachable FROM (
    SELECT id FROM beam_floor_test
    ORDER BY v <-> '[0.5,0,0]' LIMIT 100) t;
RESET mkt.query_limit;
RESET mkt.nprobe;
RESET enable_seqscan;
DROP TABLE beam_floor_test;

-- Cleanup
DROP TABLE query_settings_test;
