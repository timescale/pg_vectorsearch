# Two inserts into the same posting list serialize on the per-cluster page lock.
#
# aminsert takes a heavyweight page lock on the cluster's head block for the
# brief chain-append critical section (released per insert, not at xact end). An
# injection point ("mktann-insert-locked") pauses an insert while it holds that
# lock, so a second insert into the same cluster blocks on the lock until the
# first is woken. This proves the serialization actually blocks AND does not
# deadlock/hang: both inserts complete and both rows end up in the index.
#
# Requires a PostgreSQL built with injection points; the meson harness only
# schedules this spec when USE_INJECTION_POINTS holds and the injection_points
# extension is installed (same gate as build_progress).
#
# Fixture mirrors the other concurrency specs: 50 points along [1,0,0]..[50,0,0];
# the two probe rows sit far out at [100,0,0] (1001) and [99,0,0] (1002), so they
# route to the same (highest-x) cluster and are the top-2 nearest neighbors of
# the query [100,0,0]. The checker probes every list (nprobe = nlist).

setup
{
    CREATE EXTENSION injection_points;
    CREATE TABLE ins (id int, v vector(3));
    INSERT INTO ins SELECT g, format('[%s,0,0]', g)::vector
        FROM generate_series(1, 50) g;
    CREATE INDEX ins_idx ON ins USING mktann (v)
        WITH (nlist = 4, centroid_compression = true);
}

teardown
{
    DROP TABLE ins;
    SELECT injection_points_detach('mktann-insert-locked');
    DROP EXTENSION injection_points;
}

# First inserter: pauses while holding the cluster head's page lock. set_local
# confines the wait to this session, so the second inserter is not paused at the
# injection point — it blocks on the page lock instead, then proceeds once this
# session is woken.
session s1
setup
{
    SELECT injection_points_set_local();
    SELECT injection_points_attach('mktann-insert-locked', 'wait');
}
step s1_ins { INSERT INTO ins VALUES (1001, '[100,0,0]'); }

# Second inserter into the same cluster.
session s2
step s2_ins { INSERT INTO ins VALUES (1002, '[99,0,0]'); }

# Wakes the paused first insert, then checks the result. Returns 2 only if both
# inserts landed (the top-2 nearest of [100,0,0] are exactly 1001 and 1002).
session obs
setup { SET enable_seqscan = off; SET mkt.nprobe = 4; }
step wake { SELECT injection_points_wakeup('mktann-insert-locked'); }
step chk
{
    SELECT count(*) AS found FROM (
        SELECT id FROM ins ORDER BY v <-> '[100,0,0]' LIMIT 2) t
        WHERE id IN (1001, 1002);
}

# s1_ins pauses holding the page lock; s2_ins blocks on it (<waiting ...>); wake
# releases s1, then s2 completes. Both rows indexed -> 2.
permutation s1_ins s2_ins wake chk
