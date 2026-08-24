-- Incremental posting-list split: mkt.compact / mkt.split_postinglist
--
-- A single-partition index is split into two lists; index-scan results must
-- match the sequential-scan ground truth both before and after the split.

CREATE TABLE split_test (id serial, v vector(4));

-- Two well-separated clusters of 12 points each (deterministic values).
INSERT INTO split_test (v)
    SELECT ARRAY[(i % 3)::real, (i % 2)::real, 0, 0]::vector(4)
    FROM generate_series(1, 12) i;
INSERT INTO split_test (v)
    SELECT ARRAY[10 + (i % 3)::real, 10 + (i % 2)::real, 10, 10]::vector(4)
    FROM generate_series(1, 12) i;

-- Single flat partition with RaBitQ centroid pages (no fastscan packing),
-- which is the shape the phase-1 split supports.
CREATE INDEX split_idx ON split_test USING mktann (v)
    WITH (nlist = 1, centroid_fastscan = off);

SET mkt.nprobe = 10;

-- Ground truth (sequential scan) for a query in the first cluster.
SET enable_indexscan = off;
CREATE TEMP TABLE truth AS
    SELECT id FROM split_test
    ORDER BY v <-> '[0,0,0,0]'::vector(4) LIMIT 10;
RESET enable_indexscan;

-- Index results before the split.
SET enable_seqscan = off;
CREATE TEMP TABLE before AS
    SELECT id FROM split_test
    ORDER BY v <-> '[0,0,0,0]'::vector(4) LIMIT 10;
RESET enable_seqscan;

SELECT count(*) AS before_matches FROM truth t JOIN before b USING (id);

-- Split the single 24-entry list. Threshold 20 triggers one split into two
-- ~12-entry lists, both below the threshold.
SET mkt.max_postinglist_size = 20;
SELECT mkt.compact('split_idx') AS lists_split;

-- Index results after the split must still match the ground truth.
SET enable_seqscan = off;
CREATE TEMP TABLE after AS
    SELECT id FROM split_test
    ORDER BY v <-> '[0,0,0,0]'::vector(4) LIMIT 10;
RESET enable_seqscan;

SELECT count(*) AS after_matches FROM truth t JOIN after a USING (id);

-- A second pass is a no-op: both lists are now under the threshold.
SELECT mkt.compact('split_idx') AS second_pass;

-- split_postinglist rejects a non-index argument.
SELECT mkt.split_postinglist('split_test', 1);

-- Cleanup
DROP TABLE split_test;
