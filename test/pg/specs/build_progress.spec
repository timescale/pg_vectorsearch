# pg_stat_progress_create_index reflects an in-progress mktann build and
# distinguishes the parallel build path from the serial one.
#
# An injection point ("mktann-build-load") pauses the build at the loading
# phase so a second session can read the progress view while the build is
# blocked, then wake it. The build session forces the path (parallel via the
# table's parallel_workers reloption + max_parallel_maintenance_workers,
# serial by setting the latter to 0).
#
# Requires a PostgreSQL built with injection points; the meson harness only
# schedules this spec when -Dpg_srcdir is set and USE_INJECTION_POINTS holds.

setup
{
    CREATE EXTENSION injection_points;
    CREATE TABLE emb (v vector(3)) WITH (parallel_workers = 2);
    INSERT INTO emb (v)
        SELECT format('[%s,%s,%s]', x * 0.1, y * 0.1, z * 0.1)::vector
        FROM generate_series(0, 9) x,
             generate_series(0, 9) y,
             generate_series(0, 9) z;
}

teardown
{
    DROP TABLE emb;
    SELECT injection_points_detach('mktann-build-load');
    DROP EXTENSION injection_points;
}

# The build session pauses at the loading phase. set_local keeps the wait
# confined to this session, so the parallel workers and the observer are
# unaffected; the observer still wakes it cross-session.
session s_build
setup
{
    SELECT injection_points_set_local();
    SELECT injection_points_attach('mktann-build-load', 'wait');
}
step b_parallel
{
    SET max_parallel_maintenance_workers = 2;
    CREATE INDEX i_par ON emb USING mktann (v);
}
step b_serial
{
    SET max_parallel_maintenance_workers = 0;
    CREATE INDEX i_ser ON emb USING mktann (v);
}
step b_done { }

session s_watch
step w_phase
{
    SELECT phase FROM pg_stat_progress_create_index
        WHERE relid = 'emb'::regclass;
}
step w_wakeup { SELECT injection_points_wakeup('mktann-build-load'); }

# Parallel build: the view shows the parallel loading phase while paused.
permutation b_parallel w_phase w_wakeup b_done
# Serial build: same injection point, serial loading phase.
permutation b_serial w_phase w_wakeup b_done
