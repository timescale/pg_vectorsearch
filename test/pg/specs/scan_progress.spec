# Parallel builds advance pg_stat_progress_create_index.tuples_done while
# the heap scans run, not only after they finish.
#
# The build session attaches a 'wait' on the progress-flush injection point
# (session-local, so only the leader pauses -- at its first flush, which is
# guaranteed no later than its end-of-scan remainder even if work-stealing
# starved it of rows). The watcher then polls the progress view: with live
# reporting the leader has already applied its first increment before it
# pauses, so tuples_done is nonzero mid-scan; the watcher wakes the build
# afterwards. Without live reporting the injection point never fires, the
# build runs to completion, and the poll times out at zero.
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
    SELECT injection_points_set_local();
    SELECT injection_points_attach('mktann-scan-progress', 'wait');
    SET max_parallel_maintenance_workers = 2;
}
step b_build
{
    CREATE INDEX i_prog ON emb USING mktann (v) WITH (nlist = 32);
}
step b_done
{
    SELECT injection_points_detach('mktann-scan-progress');
}

session s_watch
step w_poll
{
    DO $$
    DECLARE
        observed bool := false;
    BEGIN
        FOR i IN 1..50 LOOP
            PERFORM 1 FROM pg_stat_progress_create_index
                WHERE tuples_done > 0;
            IF FOUND THEN
                observed := true;
                EXIT;
            END IF;
            PERFORM pg_sleep(0.1);
        END LOOP;
        RAISE NOTICE 'tuples_done advanced mid-scan: %', observed;
    END $$;
}
step w_wake	{ SELECT injection_points_wakeup('mktann-scan-progress'); }

permutation b_build w_poll w_wake b_done
