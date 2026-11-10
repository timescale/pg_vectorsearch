# A split keeps VACUUM out of the TOAST relation it is reading from.
#
# A split re-reads every entry's vector from the heap, and a vector wide
# enough to be stored out of line is reassembled from the TOAST relation on
# each fetch. The versions a split must keep include ones written by
# transactions that have not finished yet, and if such a transaction aborts
# its TOAST chunks are dead to everyone the instant it does -- no snapshot
# holds them back the way the horizon holds back a committed delete's. A
# VACUUM of the TOAST relation in that window removes chunks the split is part
# way through reading, and the read fails with "missing chunk number".
#
# So a pass holds ShareUpdateExclusiveLock -- VACUUM's own lock -- on the
# TOAST relation for its whole duration. That is all it holds there: the heap
# stays open to reads, to writes, and to its own VACUUM.
#
# This pins it. An injection point pauses the split mid-call, and a VACUUM
# aimed at the TOAST relation alone (PROCESS_MAIN false, so it never touches
# the heap or its indexes and cannot be blocked by the maintenance lock
# instead) has to wait for the wake-up. Without the lock the step runs
# straight through.
#
# Requires a PostgreSQL built with injection points (same gate as
# insert_serialize).
#
# Fixture: 40 vectors of 1024 dimensions, which at 4 KB apiece are stored out
# of line, in a single list that a target of 10 splits several ways. Half the
# rows sit around 10 on every axis and half around 0, so the list has two
# clusters to split into.

setup
{
    CREATE EXTENSION injection_points;
    CREATE TABLE toasty (id serial, v vec32(1024));
    INSERT INTO toasty (v)
        SELECT (SELECT array_agg(((i % 2) * 10 + sin(i * 0.1 + j * 0.7))::real)
                FROM generate_series(0, 1023) j)::vec32(1024)
        FROM generate_series(1, 40) i;
    CREATE INDEX toasty_idx ON toasty USING prism (v)
        WITH (nlist = 1, centroid_fastscan = off);
}

teardown
{
    DROP TABLE toasty;
    SELECT injection_points_detach('prism-split-locked');
    DROP EXTENSION injection_points;
}

# The paused split. It holds the TOAST relation's lock from the start of the
# pass, well before the pause point. set_local confines the wait to this
# session so the VACUUM is not paused too.
session m
setup
{
    SELECT injection_points_set_local();
    SELECT injection_points_attach('prism-split-locked', 'wait');
}
step m_split { CALL prism_rebalance('toasty_idx', 10); }

session v
step v_toast { VACUUM (PROCESS_MAIN FALSE) toasty; }

session obs
setup
{
    SET enable_seqscan = off;
    SET prism.nprobe = 10;
}
step o_wake { SELECT injection_points_wakeup('prism-split-locked'); }
step o_rows
{
    SELECT count(*) AS found FROM (
        SELECT id FROM toasty
        ORDER BY v <-> (SELECT v FROM toasty WHERE id = 8) LIMIT 100) t;
}

# The VACUUM blocks (<waiting ...>) until the split is woken, and every row is
# still reachable through the new lists afterwards.
permutation m_split v_toast o_wake o_rows
