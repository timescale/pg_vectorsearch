# The refine gate fires on sample-thin leaves and stays quiet on well-fed
# ones.
#
# A subsampled build refines its leaf encode references on the full table
# only when the leaves hold fewer than mkt.leaf_refine_threshold samples
# each; above that the sample means are already exact for quantization
# purposes and the extra scan is skipped. Both directions are proven by an
# injection point on the refine phase: the forced-refine build pauses there
# (the watcher observes and wakes it); the default-threshold build on the
# same well-fed table completes without ever reaching the point.
#
# The table holds 12000 64-dim rows against the 10000-slot sample floor at
# a 1MB budget, so both builds keep fewer rows than their scans see
# (genuinely subsampled). How many rows are kept varies with how the
# work-stealing scan lands on the three participants (each is capped at a
# third of the slots), so the well-fed step uses a threshold of 16 -- the
# gate would need fewer than 1024 kept samples across the 64 leaves, and
# the busiest participant alone always keeps more than that -- rather than
# a value near the observed samples-per-leaf.
#
# Requires a PostgreSQL built with injection points; the meson harness only
# schedules this spec when -Dpg_srcdir is set and USE_INJECTION_POINTS holds.

setup
{
    CREATE EXTENSION injection_points;
    CREATE TABLE emb (id int, v vec32(64)) WITH (parallel_workers = 2);
    INSERT INTO emb
        SELECT g, ('[' || g || repeat(',0', 63) || ']')::vec32(64)
        FROM generate_series(1, 12000) g;
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
    SELECT injection_points_attach('mktann-build-refine', 'wait');
    SET max_parallel_maintenance_workers = 2;
    SET maintenance_work_mem = '1MB';
}
step b_thin
{
    SET mkt.leaf_refine_threshold = 100000;
    CREATE INDEX i_thin ON emb USING mktann (v) WITH (nlist = 64);
}
step b_wellfed
{
    SET mkt.leaf_refine_threshold = 16;
    CREATE INDEX i_wellfed ON emb USING mktann (v) WITH (nlist = 64);
}
step b_done
{
    SELECT injection_points_detach('mktann-build-refine');
}

session s_watch
step w_wakeup { SELECT injection_points_wakeup('mktann-build-refine'); }

# Forced below the threshold: the build pauses at the refine phase (proof it
# runs); the watcher wakes it.
permutation b_thin w_wakeup b_done
# Well-fed leaves: the build never reaches the refine point and completes
# on its own.
permutation b_wellfed b_done
