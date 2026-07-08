# BUILDING-state race in the shared centroid cache (meerkat preloaded).
#
# When one backend is building a slot, it sets the slot to BUILDING and
# releases the registry lock before doing the (page-reading) build. A second
# backend that queries the same index during that window sees BUILDING and
# must fall back to reading centroid pages -- returning correct results without
# waiting for or corrupting the in-progress build.
#
# The "mkt-centroid-cache-build" injection point pauses the builder exactly in
# that window so the race is deterministic. Requires a PostgreSQL built with
# injection points; the meson harness only schedules this spec when
# USE_INJECTION_POINTS holds and the injection_points extension is installed.
#
# Fixture: probe row 1001 at [100,0,0,0] is the unambiguous L2 nearest neighbor
# of the query [100,0,0,0]; with nprobe = nlist every list is probed, so the
# checker returns 1 iff the result (from cache or page reads) is correct. The
# index is created but not queried in setup, so the first query triggers a
# build.

setup
{
    CREATE EXTENSION injection_points;
    CREATE TABLE emb (id int, v vector(4));
    INSERT INTO emb SELECT g, format('[%s,0,0,0]', g * 0.01)::vector
        FROM generate_series(1, 200) g;
    INSERT INTO emb VALUES (1001, '[100,0,0,0]');
    CREATE INDEX emb_idx ON emb USING mktann (v)
        WITH (nlist = 8, fastscan = true, centroid_fastscan = true,
              centroid_compression = true);
}
teardown
{
    DROP TABLE emb;
    SELECT injection_points_detach('mkt-centroid-cache-build');
    DROP EXTENSION injection_points;
}

# The builder pauses mid-build. set_local confines the wait to this backend so
# the observer can still wake it cross-session.
session builder
setup
{
    SET enable_seqscan = off; SET mkt.nprobe = 8;
    SET mkt.enable_centroid_cache = on;
    SELECT injection_points_set_local();
    SELECT injection_points_attach('mkt-centroid-cache-build', 'wait');
}
step bld_query
{
    SELECT count(*) AS found FROM (
        SELECT id FROM emb ORDER BY v <-> '[100,0,0,0]' LIMIT 1) t WHERE id = 1001;
}

# A reader that queries while the builder is paused (slot BUILDING). It must
# fall back to page reads and return the correct result. A final query after
# the builder finishes is served from the now-READY cache.
session reader
setup { SET enable_seqscan = off; SET mkt.nprobe = 8;
        SET mkt.enable_centroid_cache = on; }
step rd_during
{
    SELECT count(*) AS found FROM (
        SELECT id FROM emb ORDER BY v <-> '[100,0,0,0]' LIMIT 1) t WHERE id = 1001;
}
step wakeup { SELECT injection_points_wakeup('mkt-centroid-cache-build'); }
step rd_after
{
    SELECT count(*) AS found FROM (
        SELECT id FROM emb ORDER BY v <-> '[100,0,0,0]' LIMIT 1) t WHERE id = 1001;
}

# bld_query starts a build and pauses (slot BUILDING). rd_during sees BUILDING
# and reads pages -> correct (1). wakeup lets the builder finish (slot READY)
# -> bld_query returns 1. rd_after hits the cache -> still correct (1).
permutation bld_query rd_during wakeup rd_after
