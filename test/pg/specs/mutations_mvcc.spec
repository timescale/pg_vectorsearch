# MVCC visibility for incremental writes.
#
# aminsert places an entry in the posting list immediately (shared buffer), so
# a concurrent reader's index scan physically finds it — but the executor's
# heap visibility recheck must hide it until the writer commits, and forever if
# the writer aborts. This verifies meerkat relies on MVCC for correctness (the
# index is not snapshot-aware; it returns candidates and MVCC filters them).

setup
{
    CREATE TABLE iso (id int, v vector(3));
    INSERT INTO iso SELECT g, format('[%s,0,0]', g)::vector
        FROM generate_series(1, 50) g;
    CREATE INDEX iso_idx ON iso USING mktann (v)
        WITH (nlist = 4, centroid_compression = true);
}
teardown { DROP TABLE iso; }

session writer
setup { BEGIN; }
step w_insert { INSERT INTO iso VALUES (1001, '[100,0,0]'); }
step w_commit { COMMIT; }
step w_abort  { ROLLBACK; }

session reader
setup { SET enable_seqscan = off; SET mkt.nprobe = 4; }
# count is 1 only if the committed-and-visible row 1001 is the nearest result
step r_query
{
    SELECT count(*) AS sees_1001 FROM (
        SELECT id FROM iso ORDER BY v <-> '[100,0,0]' LIMIT 1) t WHERE id = 1001;
}

# Uncommitted insert is invisible (0); after commit it becomes visible (1).
permutation w_insert r_query w_commit r_query
# Aborted insert never appears (0 before and after).
permutation w_insert r_query w_abort r_query
