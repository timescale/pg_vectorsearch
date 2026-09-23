# The insert route-retry budget, and what happens at the end of it.
#
# An insert routes to a cluster, then takes a heavyweight lock on that
# cluster's head. Between those two steps a split can retire the head, so the
# insert re-routes and tries again. The loop is bounded
# (PRISM_INSERT_ROUTE_ATTEMPTS): running out has to fail the insert, because
# returning normally would leave a committed row that no scan of this index
# can find.
#
# Reaching the end of the budget for real needs the routed head retired
# between the route and the lock, eight times running, which the per-cluster
# lock makes vanishingly unlikely -- the budget is a safety valve, not a path
# under load. An injection point stands in for the retired head instead, so
# the last iteration of the valve is exercised rather than argued.
#
# The notice on each attempt makes the budget itself visible: one line per
# attempt, then the error. The insert must be rejected, not silently accepted,
# and the row must not be in the table afterwards.
#
# Requires a PostgreSQL built with injection points.

setup
{
    CREATE EXTENSION injection_points;
    CREATE TABLE ir (id int, v vec32(3));
    INSERT INTO ir SELECT g, format('[%s,0,0]', g)::vec32
        FROM generate_series(1, 60) g;
    CREATE INDEX ir_idx ON ir USING prism (v)
        WITH (nlist = 1, centroid_fastscan = off);
}

teardown
{
    DROP TABLE ir;
    SELECT injection_points_detach('prism-insert-force-reroute');
    DROP EXTENSION injection_points;
}

# set_local keeps the forced re-route in this session, so the fixture and the
# checks around it insert normally.
session s
setup
{
    SELECT injection_points_set_local();
    SELECT injection_points_attach('prism-insert-force-reroute', 'notice');
}
step s_insert { INSERT INTO ir VALUES (1000, '[1,0,0]'); }
step s_rows   { SELECT count(*) AS rows FROM ir WHERE id = 1000; }

permutation s_insert s_rows
