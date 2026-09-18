# What a split races against: an insert, a scan, and VACUUM.
#
# All three share one fixture and one shape. Nothing stops any of them
# overlapping a split: the maintenance lock the split holds on the index
# (ShareUpdateExclusiveLock) conflicts with neither an insert's nor VACUUM's
# RowExclusiveLock, by design, so maintenance does not block writes wholesale.
# They meet on the per-cluster head page lock instead, and an injection point
# pauses one side of each pair inside the window that matters.
#
# Requires a PostgreSQL built with injection points (same gate as
# insert_serialize).
#
# Fixture: one list of 60 points along the x-axis, which a target of 10 splits
# six ways. The three nearest [1,0,0] are ids 1, 2 and 3 whichever way the list
# ends up partitioned, so those assertions are fixed rather than
# partition-order artefacts. The probe row for the insert case sits at
# [61,0,0], beyond every existing point.

setup
{
    CREATE EXTENSION injection_points;
    CREATE TABLE race (id int, v vec32(3));
    INSERT INTO race SELECT g, format('[%s,0,0]', g)::vec32
        FROM generate_series(1, 60) g;
    CREATE INDEX race_idx ON race USING mktann (v)
        WITH (nlist = 1, centroid_fastscan = off);
}

teardown
{
    DROP TABLE race;
    SELECT injection_points_detach('mktann-split-locked');
    SELECT injection_points_detach('mktann-scan-routed');
    DROP EXTENSION injection_points;
}

# The paused split. It stops holding the old head's page lock with the centroid
# leaf still pointing there, which is the window an insert or a VACUUM has to
# survive. set_local confines the wait to this session, so the split run by
# session w below is unaffected and the other sessions never pause.
session m
setup
{
    SELECT injection_points_set_local();
    SELECT injection_points_attach('mktann-split-locked', 'wait');
}
step m_split { CALL mkt.rebalance('race_idx', 10); }

# A split that does not pause, for the scan case: there the reader is what
# pauses, and the writer has to run to completion underneath it.
#
# Each call is its own transaction -- the procedures refuse to run inside a
# caller's one -- which is also what lets a second pass reclaim what the first
# retired: a pass can never reclaim its own retirements, because its own
# transaction id is still running and so can never be below the horizon.
session w
step w_split   { CALL mkt.rebalance('race_idx', 10); }
step w_reclaim { CALL mkt.rebalance('race_idx', 10); }

# The reader, paused after routing: it holds a posting head and a snapshot,
# with the centroid page already released. A split can then repoint the leaf
# and retire the chain the reader is holding -- the case the whole
# retire-rather-than-overwrite design exists for.
session r
setup
{
    SET enable_seqscan = off;
    SET mkt.nprobe = 16;
    SELECT injection_points_set_local();
    SELECT injection_points_attach('mktann-scan-routed', 'wait');
}
step r_query { SELECT id FROM race ORDER BY v <-> '[1,0,0]' LIMIT 3; }

# The insert. It routes to the old head, blocks on its page lock, and on wake
# must notice the head is retired and route again -- otherwise the row is
# appended to a list nothing can reach.
session i
step i_ins { INSERT INTO race VALUES (1001, '[61,0,0]'); }

# VACUUM, which reaches the index through ambulkdelete and stops at the locked
# head. It matters that it stops: bulkdelete decrements the head's live_count,
# and a retired head holds delete_xid in that same field, so decrementing it
# would leave the reclaim gate reading a smaller transaction id than the split
# wrote -- older than it really is -- bringing the chain's physical reclaim
# forward past the scans it was kept alive for. bulkdelete decides a page is a
# live head under a buffer lock it then releases, so the flag has to be
# re-checked under the page lock.
session v
step v_del    { DELETE FROM race WHERE id % 10 = 0; }
step v_vacuum { VACUUM race; }

session obs
setup { SET enable_seqscan = off; SET mkt.nprobe = 16; }
step o_wake_split { SELECT injection_points_wakeup('mktann-split-locked'); }
step o_wake_scan  { SELECT injection_points_wakeup('mktann-scan-routed'); }
step o_near       { SELECT id FROM race ORDER BY v <-> '[1,0,0]' LIMIT 3; }
step o_probe
{
    SELECT count(*) AS found FROM (
        SELECT id FROM race ORDER BY v <-> '[61,0,0]' LIMIT 1) t
        WHERE id = 1001;
}

# Insert against a split: m_split pauses holding the head page lock, i_ins
# routes there and blocks (<waiting ...>), wake lets the split finish, and the
# insert then finds the head retired, routes again, and the row is findable.
permutation m_split i_ins o_wake_split o_probe

# Scan against a split: r_query pauses holding a head; the split replaces and
# retires that chain; the reclaim pass reports reclaiming nothing, because the
# reader can still reach it; on wake the query returns the three nearest rows
# regardless.
permutation r_query w_split w_reclaim o_wake_scan

# Control for the above: the same two passes with no reader paused mid-scan.
# The split's transaction id has cleared the horizon by the time the second
# pass runs, so that pass does reclaim the chain -- which is what makes the
# zero above evidence of the gate rather than of reclaim never happening.
permutation w_split w_reclaim

# VACUUM against a split, with rows deleted first so VACUUM has index entries
# to remove and actually calls into the index: m_split pauses holding the head
# page lock, v_vacuum blocks on it, wake releases the split, VACUUM then finds
# the head retired and skips it, and the index still answers correctly.
permutation v_del m_split v_vacuum o_wake_split o_near
