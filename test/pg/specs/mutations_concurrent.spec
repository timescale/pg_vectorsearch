# Concurrency / mutability behavior of the mktann index under DML.
#
# This spec consolidates the incremental-write isolation scenarios into one
# shared fixture. Every permutation exercises the same question from a
# different angle: when a row is inserted into an indexed table, under what
# conditions does an index-driven nearest-neighbor query return it?
#
# Shared fixture
# --------------
# The table holds 50 rows laid out along the x-axis at [1,0,0] .. [50,0,0].
# The probe row (id 1001) sits far out at [100,0,0], so once it is present
# AND visible it is the unambiguous top-1 nearest neighbor of the query
# vector [100,0,0]. The checker query returns 1 exactly when row 1001 is that
# nearest result and 0 otherwise, which makes "is 1001 in the index and
# MVCC-visible?" a single clean integer assertion.
#
# pg_isolation_regress runs `setup` before every permutation and `teardown`
# after every permutation, so each scenario starts from an identical, isolated
# table+index — no cross-permutation state leaks.

setup
{
    CREATE TABLE iso (id int, v vector(3));
    INSERT INTO iso SELECT g, format('[%s,0,0]', g)::vector
        FROM generate_series(1, 50) g;
    CREATE INDEX iso_idx ON iso USING mktann (v)
        WITH (nlist = 4, centroid_compression = true);
}
teardown { DROP TABLE iso; }

# The writer owns an explicit transaction so other sessions can interleave
# while its insert is still uncommitted. w_commit / w_abort decide the fate of
# the inserted row 1001.
session writer
setup { BEGIN; }
step w_ins    { INSERT INTO iso VALUES (1001, '[100,0,0]'); }
step w_commit { COMMIT; }
step w_abort  { ROLLBACK; }

# Maintenance operations that rebuild the index. Both take an ACCESS EXCLUSIVE
# lock, so when issued against a table with an open writer they must block
# until that writer's transaction ends.
session maint
step m_reindex { REINDEX INDEX iso_idx; }
step m_vacfull { VACUUM FULL iso; }

# The checker forces an index scan (no seqscan) and probes every list
# (nprobe = 4 == nlist) so the result depends only on index contents +
# visibility, never on which lists happened to be probed. The query returns 1
# iff the top-1 nearest neighbor of [100,0,0] is row 1001.
session checker
setup { SET enable_seqscan = off; SET mkt.nprobe = 4; }
step c_chk
{
    SELECT count(*) AS found FROM (
        SELECT id FROM iso ORDER BY v <-> '[100,0,0]' LIMIT 1) t WHERE id = 1001;
}

# --- MVCC visibility (the core aminsert contract) ---------------------------
# aminsert places the entry into the posting list in shared buffers the moment
# the row is inserted, so a concurrent reader's index scan *physically* finds
# it even before commit. Correctness therefore depends on the executor's heap
# visibility recheck, not on the index: meerkat returns candidates and MVCC
# filters them. Expect 0 while the insert is uncommitted, then 1 after commit.
permutation w_ins c_chk w_commit c_chk

# --- MVCC abort (the entry must never become visible) -----------------------
# Same as above, but the writer aborts. The posting-list entry physically
# remains (it is reclaimed later by vacuum), yet the MVCC recheck hides the
# dead tuple forever. Expect 0 both before and after the rollback — proving the
# index never returns an aborted row.
permutation w_ins c_chk w_abort c_chk

# --- REINDEX serialized behind an open writer -------------------------------
# REINDEX needs ACCESS EXCLUSIVE on the index, so it blocks behind the open
# insert (observe the <waiting ...> / <... completed> markers). Once the writer
# commits, REINDEX rebuilds from the now-committed heap and the fresh index
# contains row 1001. Expect 1 — the rebuilt index is correct and complete.
permutation w_ins m_reindex w_commit c_chk

# --- VACUUM FULL serialized behind an open writer ---------------------------
# VACUUM FULL rewrites the heap and rebuilds every index under ACCESS
# EXCLUSIVE, so it likewise waits for the open writer. After the writer
# commits it proceeds, and the committed row 1001 is present in the rebuilt
# index. Expect 1.
permutation w_ins m_vacfull w_commit c_chk
