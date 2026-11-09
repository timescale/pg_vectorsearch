# A parallel build advances pg_stat_progress_create_index.tuples_done while
# its heap scans run, not only once they finish, so an operator watching a
# long build can see it moving.
#
# The build stops just before the leader replaces that counter with the exact
# count its finished workers hand it. So what the watcher reads there is what
# live reporting alone produced, and it should already account for every row
# scanned. Reporting only at the end of each participant's scan would leave
# it at the remainders, far short of the row count.
#
# The leader always reaches that point, whatever share of the scan
# work-stealing gave it -- it does not scan that phase at all, and the counter
# it stops with is the workers', relayed over the parallel message queue and
# necessarily already applied. Being the backend running b_build, it is also
# the one isolationtester can see stop; a paused worker would leave the step
# looking merely slow, until the cancel.
#
# The build is frozen while the watcher looks, so it reads the view once.
# There is nothing to poll for.
#
# Requires a PostgreSQL built with injection points; the meson harness only
# schedules this spec when -Dpg_srcdir is set and USE_INJECTION_POINTS holds.

setup
{
    CREATE EXTENSION injection_points;
    CREATE TABLE emb (id int, v vec32(8)) WITH (parallel_workers = 2);
    INSERT INTO emb
        SELECT g, ('[' || g || ',' || g % 97 || repeat(',0.1', 6) ||
                   ']')::vec32(8)
        FROM generate_series(1, 30000) g;
}

teardown
{
    DROP TABLE emb;
    DROP EXTENSION injection_points;
}

session s_build
setup
{
    -- set_local only so a failed permutation cannot leave the point
    -- attached for the rest of the run. The workers never reach it.
    SELECT injection_points_set_local();
    SELECT injection_points_attach('prism-scan-progress', 'wait');
    SET max_parallel_maintenance_workers = 2;
}
step b_build
{
    CREATE INDEX i_prog ON emb USING prism (v) WITH (nlist = 32);
}
step b_done
{
    SELECT injection_points_detach('prism-scan-progress');
}

session s_watch
step w_read
{
    -- Every row the scan saw, not just the per-participant remainders it
    -- flushes at the end: those cannot add up to the row count.
    SELECT tuples_done >= (SELECT count(*) FROM emb) AS scan_reported_live
        FROM pg_stat_progress_create_index;
}
step w_wake { SELECT injection_points_wakeup('prism-scan-progress'); }

permutation b_build w_read w_wake b_done
