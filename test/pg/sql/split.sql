-- Incremental posting-list split: mkt.rebalance / mkt.split_posting_list
--
-- A single-partition index is split into two lists; index-scan results must
-- match the sequential-scan ground truth both before and after the split.

CREATE TABLE split_test (id serial, v vec32(4));

-- Two well-separated clusters of 12 points each (deterministic values).
INSERT INTO split_test (v)
    SELECT ARRAY[(i % 3)::real, (i % 2)::real, 0, 0]::vec32(4)
    FROM generate_series(1, 12) i;
INSERT INTO split_test (v)
    SELECT ARRAY[10 + (i % 3)::real, 10 + (i % 2)::real, 10, 10]::vec32(4)
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
    ORDER BY v <-> '[0,0,0,0]'::vec32(4) LIMIT 10;
RESET enable_indexscan;

-- Index results before the split.
-- Small tables, so a sequential scan is the plan the cost model correctly
-- prefers and the scan below is forced through the index. The rows come
-- back either way, so this pins the plan.
SET enable_seqscan = off;
EXPLAIN (COSTS OFF)
    SELECT id FROM split_test ORDER BY v <-> '[0,0,0,0]'::vec32(4) LIMIT 10;
CREATE TEMP TABLE before AS
    SELECT id FROM split_test
    ORDER BY v <-> '[0,0,0,0]'::vec32(4) LIMIT 10;
RESET enable_seqscan;

SELECT count(*) AS before_matches FROM truth t JOIN before b USING (id);

-- Split the single 24-entry list. A target of 10 puts the split trigger at 20,
-- which 24 entries clear, and the width at round(24/10) = 2 -- so one bisection
-- into two ~12-entry lists, each resting near the target with room to grow back
-- to the trigger. rebalance is a procedure and reports the count via NOTICE.
CALL mkt.rebalance('split_idx', 10);

-- The metapage's centroid page count is maintained, not derived from the
-- block layout: a split with no room on a level-0 centroid page chains the
-- new page past the posting region, where no block range finds it. Assert
-- the recorded count against what walking the tree actually finds -- that
-- catches a wrong value from the build, a split whose addition was never
-- persisted, and one counted twice. It is what the planner's descent term
-- reads, so a drift here silently mis-costs every scan.
--
-- These lists are far too small to overflow a centroid page, so the count
-- does not move here; the point is that it still agrees.
SELECT (SELECT setting::int FROM mkt.index_settings('split_idx')
          WHERE name = 'centroid_pages')
     = (SELECT count(DISTINCT blkno) FROM mkt.centroid_pages('split_idx'))
    AS centroid_pages_matches_the_tree;

-- The new lists are written in the index's posting format (fastscan here), so a
-- split re-optimizes the data rather than leaving AoS lists behind.
SELECT bool_and(format = 'fastscan') AS new_lists_fastscan
    FROM mkt.posting_pages('split_idx') WHERE is_first;

-- Index results after the split must still match the ground truth.
SET enable_seqscan = off;
CREATE TEMP TABLE after AS
    SELECT id FROM split_test
    ORDER BY v <-> '[0,0,0,0]'::vec32(4) LIMIT 10;
RESET enable_seqscan;

SELECT count(*) AS after_matches FROM truth t JOIN after a USING (id);

-- The split retires the old list lazily: it stays readable (not tombstoned)
-- until its deletion XID clears the global visibility horizon, so a concurrent
-- scan holding a stale pointer never misses it. Right after the split nothing
-- is tombstoned yet. (Physical reclaim happens on a later maintenance pass once
-- the horizon advances; that timing is snapshot-dependent, so not asserted.)
SELECT count(*) AS tombstoned_before_reclaim
    FROM mkt.posting_pages('split_idx') WHERE tombstoned;

-- A second pass splits nothing: both lists sit inside the operating band, well
-- under the trigger. Pinning the trigger at the target instead would re-split
-- them here, which is the thrash the band exists to prevent.
CALL mkt.rebalance('split_idx', 10);

-- split_posting_list rejects a non-index argument.
CALL mkt.split_posting_list('split_test', 1);

-- The width is derived from the target, so a list far above it is right-sized
-- in one pass instead of bisected repeatedly. 60 entries against a target of 10
-- is round(60/10) = 6 parts, all of which then sit inside the band -- so the
-- pass after it splits nothing.
CREATE TABLE split_wide (id serial, v vec32(4));
INSERT INTO split_wide (v)
    SELECT ARRAY[i::real, 0, 0, 0]::vec32(4) FROM generate_series(1, 60) i;
CREATE INDEX split_wide_idx ON split_wide USING mktann (v)
    WITH (nlist = 1, centroid_fastscan = off);

SET enable_indexscan = off;
CREATE TEMP TABLE wide_truth AS
    SELECT id FROM split_wide ORDER BY v <-> '[1,0,0,0]'::vec32(4) LIMIT 10;
RESET enable_indexscan;

CALL mkt.rebalance('split_wide_idx', 10);
SELECT setting AS nlist_after_split
    FROM mkt.index_settings('split_wide_idx') WHERE name = 'nlist';
CALL mkt.rebalance('split_wide_idx', 10);

-- Results still match the sequential-scan ground truth after a 6-way split.
SET enable_seqscan = off;
CREATE TEMP TABLE wide_after AS
    SELECT id FROM split_wide ORDER BY v <-> '[1,0,0,0]'::vec32(4) LIMIT 10;
RESET enable_seqscan;

SELECT count(*) AS wide_after_matches
    FROM wide_truth t JOIN wide_after a USING (id);

-- Same invariant after a 6-way split on the wider index.
SELECT (SELECT setting::int FROM mkt.index_settings('split_wide_idx')
          WHERE name = 'centroid_pages')
     = (SELECT count(DISTINCT blkno)
          FROM mkt.centroid_pages('split_wide_idx'))
    AS wide_centroid_pages_matches_the_tree;

-- With no target the index supplies one: nlist was set explicitly at build
-- time, so maintenance honours it rather than overriding it with its own
-- automatic value. 60 rows over nlist = 1 is a target of 60, whose trigger of
-- 120 no list reaches -- so a bare CALL splits nothing here.
CALL mkt.rebalance('split_wide_idx');

-- A target must be positive when given at all.
CALL mkt.rebalance('split_wide_idx', 0);

-- A procedure cannot be STRICT, so null arguments reach the implementation and
-- have to be rejected by name -- otherwise a null index reads as OID 0 and
-- reports "could not open relation with OID 0", describing the consequence
-- rather than the mistake. A null target is the exception: it means "derive it".
CALL mkt.rebalance(NULL, 10);
CALL mkt.split_posting_list(NULL, 1);
CALL mkt.split_posting_list('split_wide_idx', NULL);

-- A split sizes its clustering sample from maintenance_work_mem, so the range
-- of that setting is part of the interface. A large one must not turn into a
-- failed allocation: a backend refuses a single allocation near a gigabyte, and
-- an operator who has raised the setting for index builds should not find that
-- splitting stops working. Setting it here costs nothing -- the sample is sized
-- to the list, which is tiny.
SET maintenance_work_mem = '2GB';
CALL mkt.rebalance('split_wide_idx');

-- The manual entry point names a head block directly: CALL takes no subquery,
-- so the block number is a literal -- and the row above it is what proves the
-- literal still points at a head.
SELECT blkno AS head_to_split FROM mkt.posting_pages('split_wide_idx')
    WHERE is_first ORDER BY blkno LIMIT 1;
CALL mkt.split_posting_list('split_wide_idx', 3);
RESET maintenance_work_mem;

-- A split with no room left on the level-0 centroid page extends the
-- relation and chains the new page there, which puts a centroid page past
-- the posting region and leaves the centroid pages discontiguous. That is
-- why the count lives on the metapage instead of being derived from
-- first_posting: after this, no block range measures it.
--
-- 1024 dimensions puts about 56 leaf entries on a centroid page, and one
-- pass takes the leaf count from 2 to 64, so a single rebalance crosses it.
CREATE TABLE split_append (id serial, v vec32(1024));
INSERT INTO split_append (v)
    SELECT ARRAY(SELECT ((i * 7 + g) % 97)::real
                 FROM generate_series(1, 1024) g)::vec32(1024)
    FROM generate_series(1, 600) i;
CREATE INDEX split_append_idx ON split_append USING mktann (v)
    WITH (nlist = 2, centroid_fastscan = off, centroid_compression = on);

SELECT setting AS centroid_pages_before
    FROM mkt.index_settings('split_append_idx') WHERE name = 'centroid_pages';

CALL mkt.rebalance('split_append_idx', 8);

SELECT setting AS centroid_pages_after
    FROM mkt.index_settings('split_append_idx') WHERE name = 'centroid_pages';

-- The recorded count still matches what walking the tree finds, and the
-- pages are no longer contiguous -- so the count was maintained, not
-- inferred.
SELECT (SELECT setting::int FROM mkt.index_settings('split_append_idx')
          WHERE name = 'centroid_pages')
     = (SELECT count(DISTINCT blkno)
          FROM mkt.centroid_pages('split_append_idx'))
    AS appended_count_matches_the_tree;

SELECT max(blkno) - min(blkno) + 1 > count(DISTINCT blkno)
    AS centroid_pages_are_discontiguous
    FROM mkt.centroid_pages('split_append_idx');

-- The other end: a budget too small to pay for even two partitions' worth of
-- sample is refused, rather than quietly allocating the minimum anyway. Wide
-- vectors reach that end at a setting PostgreSQL still accepts. The shortfall
-- is named, so the expected output pins it -- changing the memory model is
-- meant to show up here.
CREATE TABLE split_wide_dim (id serial, v vec32(1024));
INSERT INTO split_wide_dim (v)
    SELECT ARRAY(SELECT (i % 7)::real FROM generate_series(1, 1024))::vec32(1024)
    FROM generate_series(1, 4) i;
CREATE INDEX split_wide_dim_idx ON split_wide_dim USING mktann (v)
    WITH (nlist = 1, centroid_fastscan = off);
SET maintenance_work_mem = '1MB';
CALL mkt.rebalance('split_wide_dim_idx');
RESET maintenance_work_mem;
DROP TABLE split_wide_dim;

-- The column type is the index's, not an assumed one. A vec16 index passes
-- the shape check this maintenance requires, and reading its 16-bit payload as
-- float32 runs off the end of each value -- which clusters garbage and, with
-- other data, would write it. Splitting one and still matching the
-- sequential-scan ground truth is what says the values were read as vec16.
CREATE TABLE split_half (id serial, v vec16(8));
INSERT INTO split_half (v)
    SELECT ARRAY[(i % 2)::real,0,0,0,0,0,0,0]::vec16(8)
    FROM generate_series(1, 30) i;
INSERT INTO split_half (v)
    SELECT ARRAY[10 + (i % 2)::real,10,10,10,10,10,10,10]::vec16(8)
    FROM generate_series(1, 30) i;
CREATE INDEX split_half_idx ON split_half USING mktann (v vec16_l2_ops)
    WITH (nlist = 1, centroid_fastscan = off);

SET enable_indexscan = off;
CREATE TEMP TABLE half_truth AS
    SELECT id FROM split_half
    ORDER BY v <-> '[0,0,0,0,0,0,0,0]'::vec16(8) LIMIT 5;
RESET enable_indexscan;

CALL mkt.rebalance('split_half_idx', 10);

SET enable_seqscan = off;
SET mkt.nprobe = 16;
SELECT count(*) AS half_matches_after_split
    FROM half_truth t JOIN (
        SELECT id FROM split_half
        ORDER BY v <-> '[0,0,0,0,0,0,0,0]'::vec16(8) LIMIT 5) a USING (id);
RESET enable_seqscan;
RESET mkt.nprobe;
DROP TABLE split_half;

-- Neither procedure runs inside a caller's transaction. They reorganize the
-- index outside transaction control -- a rollback would leave the lists split,
-- the old chains retired and the leaf count raised -- and a pass that commits
-- per list, which is where this is going, cannot commit at all from inside
-- someone else's transaction.
BEGIN;
CALL mkt.rebalance('split_wide_idx');
ROLLBACK;

-- A block that opens a subtransaction is a caller's transaction by the same
-- reasoning, and is refused for the same reason.
DO $$
BEGIN
    CALL mkt.rebalance('split_wide_idx');
    RAISE NOTICE 'ran inside a subtransaction';
EXCEPTION WHEN active_sql_transaction THEN
    RAISE NOTICE 'refused inside a subtransaction';
END $$;

-- A plain top-level DO block is not: it can commit on its own behalf, so a
-- procedure that manages transactions may run inside one.
DO $$ BEGIN CALL mkt.rebalance('split_wide_idx'); END $$;

-- Ownership gate: both mutating maintenance procedures require ownership of the
-- underlying table, not merely SELECT, and the owner check runs before any
-- other validation. EXECUTE is granted to PUBLIC; USAGE on the mkt schema is
-- only needed to reach the procedures.
CREATE ROLE regress_split_unpriv NOLOGIN;
GRANT USAGE ON SCHEMA mkt TO regress_split_unpriv;
GRANT SELECT ON split_test TO regress_split_unpriv;
SET ROLE regress_split_unpriv;
CALL mkt.split_posting_list('split_idx', 1);
CALL mkt.rebalance('split_idx', 10);
RESET ROLE;
REVOKE SELECT ON split_test FROM regress_split_unpriv;
REVOKE USAGE ON SCHEMA mkt FROM regress_split_unpriv;
DROP ROLE regress_split_unpriv;

-- The two shapes incremental split declines, which are also the two shapes
-- CREATE INDEX hands you by default: centroid pages are packed for fastscan
-- unless asked otherwise, and a list count past the fan-out grows a second
-- level. Until phase 2 covers them, an index has to be built with RaBitQ
-- centroid pages and few enough lists to stay flat for rebalance to run at
-- all, so it is worth having the boundary in a test rather than in a commit
-- message.
CREATE TABLE split_shape (id serial, v vec32(4));
INSERT INTO split_shape (v)
    SELECT ARRAY[(i % 97)::real, (i % 89)::real, (i % 83)::real,
                 (i % 79)::real]::vec32(4)
    FROM generate_series(1, 400) i;

CREATE INDEX split_shape_fs ON split_shape USING mktann (v);
SELECT setting AS centroid_format
    FROM mkt.index_settings('split_shape_fs') WHERE name = 'centroid_format';
CALL mkt.rebalance('split_shape_fs', 30);

CREATE INDEX split_shape_deep ON split_shape USING mktann (v)
    WITH (centroid_fastscan = off, nlist = 64);
SELECT setting AS nlevels
    FROM mkt.index_settings('split_shape_deep') WHERE name = 'nlevels';
CALL mkt.rebalance('split_shape_deep', 3);

DROP TABLE split_shape;

-- A list whose chain spans many posting pages, split repeatedly. The write
-- pass walks the source chain while the builder writes new pages through the
-- same storage, which holds one page at a time; every other fixture here has
-- a single-page source chain, so the walk never had to survive the builder
-- taking that slot.
--
-- Repeating the split also covers the leaf page's capacity: a page written
-- outside index build comes back with pd_lower covering the content area, so
-- the first split used to leave the leaf page looking full and every later
-- split put its leaves on a page of their own. It takes a second and third
-- round on the same page to show.
CREATE TABLE split_many (id serial, v vec32(4));
INSERT INTO split_many (v)
    SELECT ARRAY[(i % 97)::real, (i % 89)::real, (i % 83)::real,
                 (i % 79)::real]::vec32(4)
    FROM generate_series(1, 3000) i;
CREATE INDEX split_many_idx ON split_many USING mktann (v)
    WITH (nlist = 1, centroid_fastscan = off);

SET mkt.nprobe = 200;
SET enable_indexscan = off;
CREATE TEMP TABLE truth_many AS
    SELECT id FROM split_many
    ORDER BY v <-> '[0,0,0,0]'::vec32(4) LIMIT 10;
RESET enable_indexscan;

CALL mkt.rebalance('split_many_idx', 500);
CALL mkt.rebalance('split_many_idx', 100);
CALL mkt.rebalance('split_many_idx', 30);

-- Every leaf still fits on the one level-0 page it was built on: entries are
-- small at this dimension, so a page holds hundreds of them.
SELECT count(*) > 40 AS many_leaves, count(DISTINCT blkno) = 1 AS one_page
    FROM mkt.centroid_pages('split_many_idx');

-- ... and results still match the sequential-scan ground truth.
SET enable_seqscan = off;
SELECT count(*) AS matches
    FROM (SELECT id FROM split_many
          ORDER BY v <-> '[0,0,0,0]'::vec32(4) LIMIT 10) r
    JOIN truth_many USING (id);
RESET enable_seqscan;
RESET mkt.nprobe;
DROP TABLE split_many;

-- A list whose pages are not all in the same format, and what a split does
-- with it. Inserts into a fastscan index append AoS pages -- fastscan packs
-- vectors in fixed groups, which one insert cannot fill -- so any list that
-- has taken inserts is part fastscan, part AoS. Two things follow: the read
-- side of the split has to walk a mixed chain, and the write side rewrites
-- everything in the index's own format, which makes a split the point where
-- such a list is re-optimised back to fastscan.
CREATE TABLE split_drift (id int, v vec32(3));
INSERT INTO split_drift SELECT g, format('[%s,0,0]', g)::vec32
    FROM generate_series(1, 300) g;
CREATE INDEX split_drift_idx ON split_drift USING mktann (v)
    WITH (nlist = 1, fastscan = on, centroid_fastscan = off);
INSERT INTO split_drift SELECT g, format('[%s,0,0]', g)::vec32
    FROM generate_series(301, 600) g;

SELECT string_agg(DISTINCT format, '+' ORDER BY format) AS formats_before
    FROM mkt.posting_pages('split_drift_idx');

SET enable_indexscan = off;
CREATE TEMP TABLE truth_drift AS
    SELECT id FROM split_drift ORDER BY v <-> '[1,0,0]'::vec32 LIMIT 10;
RESET enable_indexscan;

CALL mkt.rebalance('split_drift_idx', 60);

-- Every page is fastscan again, and no entry was dropped on the way.
SELECT string_agg(DISTINCT format, '+' ORDER BY format) AS formats_after,
       sum(entry_count) AS entries_after
    FROM mkt.posting_pages('split_drift_idx');

SET enable_seqscan = off;
SET mkt.nprobe = 10000;
SELECT count(*) AS matches
    FROM (SELECT id FROM split_drift
          ORDER BY v <-> '[1,0,0]'::vec32 LIMIT 10) r
    JOIN truth_drift USING (id);
RESET enable_seqscan;
RESET mkt.nprobe;

-- Reclaim of the chains a split retires. The split that retires a chain does
-- not free it: a scan that read the old head before the flip may still be
-- walking it, so the pages stay readable until no snapshot can reach them,
-- and a later pass frees them. The count in the NOTICE is the only handle on
-- that, so pin the sequence -- the pass that splits reclaims nothing, the
-- next one reclaims what it retired, and a third finds nothing left.
CALL mkt.rebalance('split_drift_idx', 60);
CALL mkt.rebalance('split_drift_idx', 60);

DROP TABLE split_drift;

-- Cleanup
DROP TABLE split_test;
DROP TABLE split_wide;
