# What a backend's cached index metadata has to survive.
#
# Each backend caches the index's metapage-derived state in its relcache
# entry. Another session can change that state, and the change only reaches
# this backend as a relcache invalidation, which it processes the next time
# it takes a lock. Each permutation below is one way the cached copy can go
# stale or be freed under a backend that is still using it.
#
# Fixture: dim 72 and one list (nlist = 1), so every query probes cluster 0.
# The index is built in AoS format (fastscan = off). cinv_other is only there
# to be locked.

setup
{
    CREATE TABLE cinv (id int, v vec32(72));
    INSERT INTO cinv
        SELECT g, ('[' || string_agg(sin(g * 0.37 + j)::text, ',') || ']')
                  ::vec32
        FROM generate_series(1, 300) g, generate_series(1, 72) j
        GROUP BY g;
    CREATE INDEX cinv_idx ON cinv USING prism (v)
        WITH (nlist = 1, fastscan = off, centroid_fastscan = off);
    CREATE TABLE cinv_other (x int);
}

teardown
{
    DROP TABLE cinv, cinv_other;
}

session reader
setup { SET enable_seqscan = off; }
step r_begin { BEGIN; }
# A cursor over a parameterized scan: each outer row rescans the index. The
# outer rows are indexed vectors, so each one's nearest neighbor is itself.
step r_declare
{
    DECLARE c CURSOR FOR
        SELECT q.id AS qid, n.id
        FROM (SELECT id, v FROM cinv WHERE id IN (10, 20, 30, 40, 50)
              ORDER BY id) q
        CROSS JOIN LATERAL
            (SELECT id FROM cinv c ORDER BY c.v <-> q.v LIMIT 1) n;
}
step r_fetch1 { FETCH 1 FROM c; }
# Locking a relation not yet locked in this transaction processes pending
# invalidations.
step r_lock   { SELECT count(*) AS other_rows FROM cinv_other; }
step r_fetch  { FETCH ALL FROM c; }
step r_commit { COMMIT; }
step r_query
{
    SELECT id FROM cinv
    ORDER BY v <-> (SELECT v FROM cinv WHERE id = 42)
    LIMIT 5;
}

session maint
# Invalidates the relcache entry without changing the index's pages.
step m_alter    { ALTER INDEX cinv_idx SET (target_pages = 2); }
step m_begin    { BEGIN; }
step m_convert  { CALL prism_convert_posting_to_fastscan('cinv_idx', 0); }
step m_rollback { ROLLBACK; }

# Freed metadata: the invalidation frees the cached metadata while the
# cursor's scan is still open, and the rescans after it must not read it.
# The remaining rows are 20, 30, 40 and 50, each its own neighbor.
permutation r_begin r_declare r_fetch1 m_alter r_lock r_fetch r_commit

# Stale fastscan flag: the reader fills its cache, m_convert rewrites cluster
# 0 as fastscan, and the reader queries again. A reader still using the
# cached "no fastscan" flag would score fastscan pages with the AoS kernel:
# wrong distances, garbage TIDs, and at dim 65-80 an out-of-bounds write into
# the scan buffers. Both queries must return the same rows.
permutation r_query m_convert r_query

# Stale flag that is never refreshed: a rollback keeps the converted pages
# but discards the invalidation, so the reader keeps its cached "no fastscan"
# flag. It must still score the converted cluster from the pages themselves.
permutation r_query m_begin m_convert m_rollback r_query
