# Granular build-phase introspection: pg_stat_progress_create_index advances
# through the per-phase model for BOTH the parallel and serial builds. The
# parallel build previously reported only "scanning table (parallel)" for the
# entire build (it set the subphase once and never updated it); this asserts the
# intermediate clustering phases are now visible, and that the row total is
# published (percent_complete used to be broken because tuples_total was 0).
#
# The build-progress seam fires a distinct injection point at each phase
# boundary; attaching 'wait' to several pauses the build at each so a second
# session can read the live phase. fan_out = 4 with nlist = 16 forces a
# two-level tree, so the parallel path runs (and surfaces) the subtree phase.
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
    SELECT injection_points_detach('prism-build-kmeans');
    SELECT injection_points_detach('prism-build-subtrees');
    SELECT injection_points_detach('prism-build-posting');
    DROP EXTENSION injection_points;
}

# The build pauses at each attached phase boundary. set_local keeps the waits
# confined to this backend (the leader), so the parallel workers and the
# observer are unaffected; the observer still wakes it cross-session.
session s_build
setup
{
    SELECT injection_points_set_local();
    SELECT injection_points_attach('prism-build-kmeans', 'wait');
    SELECT injection_points_attach('prism-build-subtrees', 'wait');
    SELECT injection_points_attach('prism-build-posting', 'wait');
}
step b_parallel
{
    SET max_parallel_maintenance_workers = 2;
    CREATE INDEX i_par ON emb USING prism (v) WITH (fan_out = 4, nlist = 16);
}
step b_serial
{
    SET max_parallel_maintenance_workers = 0;
    CREATE INDEX i_ser ON emb USING prism (v) WITH (fan_out = 4, nlist = 16);
}
step b_done { }

session s_watch
# Reports the live phase and that the row total has been published (the % is
# computable), the two things that were broken for the parallel build.
step w_phase
{
    SELECT phase, (tuples_total > 0) AS has_total
        FROM pg_stat_progress_create_index
        WHERE relid = 'emb'::regclass;
}
step w_kmeans	{ SELECT injection_points_wakeup('prism-build-kmeans'); }
step w_subtrees { SELECT injection_points_wakeup('prism-build-subtrees'); }
step w_posting	{ SELECT injection_points_wakeup('prism-build-posting'); }

# Parallel build: clustering (k-means) -> clustering (subtrees) ->
# finalizing posting lists, each observed while paused.
permutation b_parallel w_phase w_kmeans w_phase w_subtrees w_phase w_posting b_done
# Serial build: no subtree phase (k-means builds the whole tree in one phase);
# clustering (k-means) -> finalizing posting lists.
permutation b_serial w_phase w_kmeans w_phase w_posting b_done
