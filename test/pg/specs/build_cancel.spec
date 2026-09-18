# A build canceled mid-phase must fail cleanly and leave the table ready for
# an immediate rebuild.
#
# The batched subtree phase runs the leader and every worker in barrier
# lockstep (two barriers per batch), and the leader-only streaming write
# follows it -- fresh surface for a cancellation to strand the party. The
# injection point pauses the leader at the subtree phase; the watcher
# cancels the build session's backend, and the follow-up build (same table,
# same shape, no pause) must succeed.
#
# Requires a PostgreSQL built with injection points; the meson harness only
# schedules this spec when -Dpg_srcdir is set and USE_INJECTION_POINTS holds.

setup
{
    CREATE EXTENSION injection_points;
    CREATE TABLE emb (v vec32(3)) WITH (parallel_workers = 2);
    INSERT INTO emb (v)
        SELECT format('[%s,%s,%s]', x * 0.1, y * 0.1, z * 0.1)::vec32
        FROM generate_series(0, 9) x,
             generate_series(0, 9) y,
             generate_series(0, 9) z;
}

teardown
{
    DROP TABLE emb;
    DROP EXTENSION injection_points;
}

# The build session pauses at the subtree phase (multi-level tree via the
# explicit nlist/fan_out); set_local confines the wait to this session.
session s_build
setup
{
    SELECT injection_points_set_local();
    SELECT injection_points_attach('mktann-build-subtrees', 'wait');
}
step b_start
{
    SET max_parallel_maintenance_workers = 2;
    CREATE INDEX i_cancel ON emb USING mktann (v)
        WITH (nlist = 12, fan_out = 4);
}
step b_detach
{
    SELECT injection_points_detach('mktann-build-subtrees');
}
step b_rebuild
{
    SET max_parallel_maintenance_workers = 2;
    CREATE INDEX i_again ON emb USING mktann (v)
        WITH (nlist = 12, fan_out = 4);
}
step b_check
{
    SELECT count(*) FROM pg_class WHERE relname IN ('i_cancel', 'i_again');
}

session s_watch
step w_cancel
{
    SELECT count(pg_cancel_backend(pid)) > 0 AS canceled
        FROM pg_stat_activity
        WHERE query LIKE '%CREATE INDEX i_cancel%'
          AND backend_type = 'client backend'
          AND pid <> pg_backend_pid();
}

# Cancel mid subtree batch: the build errors out cleanly (no deadlock, no
# stranded workers), and the immediate rebuild on the same table succeeds.
permutation b_start w_cancel b_detach b_rebuild b_check
