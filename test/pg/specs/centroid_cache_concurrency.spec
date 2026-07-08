# Concurrency of the shared centroid cache (meerkat preloaded) around REINDEX.
#
# REINDEX assigns the index a new relfilenode, and the cache is keyed by
# relfilenode, so a query after REINDEX must build a fresh slot and must never
# serve the stale one. These permutations assert correctness (the unambiguous
# top-1 neighbor is still found) across a reindex and when REINDEX is
# serialized behind a cache-using reader.
#
# Fixture: 200 rows packed near the origin along x, plus a probe row 1001 far
# out at [100,0,0,0] that is the unambiguous L2 nearest neighbor of the query
# [100,0,0,0]. With nprobe = nlist every list is probed, so the checker
# returns 1 iff the index (served from cache or pages) is correct and complete.

setup
{
    CREATE TABLE iso (id int, v vector(4));
    INSERT INTO iso SELECT g, format('[%s,0,0,0]', g * 0.01)::vector
        FROM generate_series(1, 200) g;
    INSERT INTO iso VALUES (1001, '[100,0,0,0]');
    CREATE INDEX iso_idx ON iso USING mktann (v)
        WITH (nlist = 8, fastscan = true, centroid_fastscan = true,
              centroid_compression = true);
}
teardown { DROP TABLE iso; }

# Reader that uses the cache. Returns 1 iff the top-1 neighbor is row 1001.
session reader
setup { SET enable_seqscan = off; SET mkt.nprobe = 8;
        SET mkt.enable_centroid_cache = on; }
step r_chk
{
    SELECT count(*) AS found FROM (
        SELECT id FROM iso ORDER BY v <-> '[100,0,0,0]' LIMIT 1) t
    WHERE id = 1001;
}

# A cache-using reader inside an explicit transaction. Its index scan holds
# AccessShare on iso_idx until commit, so a concurrent REINDEX must block.
# (Session GUCs are set in setup; the transaction is opened by tr_begin so the
# lock is held across the following steps.)
session txnreader
setup { SET enable_seqscan = off; SET mkt.nprobe = 8;
        SET mkt.enable_centroid_cache = on; }
step tr_begin { BEGIN; }
step tr_chk
{
    SELECT count(*) AS found FROM (
        SELECT id FROM iso ORDER BY v <-> '[100,0,0,0]' LIMIT 1) t
    WHERE id = 1001;
}
step tr_commit { COMMIT; }

session maint
step m_reindex { REINDEX INDEX iso_idx; }

# --- Cache stays correct across REINDEX --------------------------------------
# r_chk builds a slot for the original relfilenode; REINDEX swaps in a new one;
# the next r_chk must rebuild for the new relfilenode and stay correct, never
# serving the stale slot. Expect 1, then 1.
permutation r_chk m_reindex r_chk

# --- REINDEX serializes behind a cache-using reader, then the cache rebuilds --
# tr_begin opens the transaction; tr_chk builds a slot and holds AccessShare on
# the index, so REINDEX blocks until tr_commit (observe <waiting>/<completed>).
# After the rebuild, a fresh reader rebuilds the cache for the new relfilenode
# and is correct. Expect 1, REINDEX waits, then 1.
permutation tr_begin tr_chk m_reindex tr_commit r_chk
