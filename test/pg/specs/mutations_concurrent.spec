# Concurrency / mutability behavior of the prism index under DML.
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
# vec32 [100,0,0]. The checker query returns 1 exactly when row 1001 is that
# nearest result and 0 otherwise, which makes "is 1001 in the index and
# MVCC-visible?" a single clean integer assertion.
#
# pg_isolation_regress runs `setup` before every permutation and `teardown`
# after every permutation, so each scenario starts from an identical, isolated
# table+index — no cross-permutation state leaks.

setup
{
    CREATE TABLE iso (id int, v vec32(3));
    INSERT INTO iso SELECT g, format('[%s,0,0]', g)::vec32
        FROM generate_series(1, 50) g;
    CREATE INDEX iso_idx ON iso USING prism (v)
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

# A second writer that inserts another far-out point (id 1002 at [99,0,0]) which
# routes to the same cluster as 1001. BEGIN is an explicit step (not setup) so it
# runs only in the permutation that uses it — a setup BEGIN would leak an open
# transaction into the permutations that don't.
session writer2
step w2_begin  { BEGIN; }
step w2_ins    { INSERT INTO iso VALUES (1002, '[99,0,0]'); }
step w2_commit { COMMIT; }

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
setup { SET enable_seqscan = off; SET prism.nprobe = 4; }
step c_chk
{
    SELECT count(*) AS found FROM (
        SELECT id FROM iso ORDER BY v <-> '[100,0,0]' LIMIT 1) t WHERE id = 1001;
}
# Both far-out rows present: the top-2 nearest neighbors of [100,0,0] must be
# exactly 1001 and 1002. Returns 2 only if both concurrent inserts landed.
step c_chk2
{
    SELECT count(*) AS found FROM (
        SELECT id FROM iso ORDER BY v <-> '[100,0,0]' LIMIT 2) t
        WHERE id IN (1001, 1002);
}

# --- MVCC visibility (the core aminsert contract) ---------------------------
# aminsert places the entry into the posting list in shared buffers the moment
# the row is inserted, so a concurrent reader's index scan *physically* finds
# it even before commit. Correctness therefore depends on the executor's heap
# visibility recheck, not on the index: prism returns candidates and MVCC
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

# --- Two concurrent inserts into the same posting list ----------------------
# Both writers append to the same cluster's chain from separate, overlapping
# transactions: the per-insert page lock is released at statement end, so they
# do not block each other, and the second insert builds on the chain metadata
# (tail/live) the first left in shared buffers. This guards the insert path's
# chain maintenance under concurrency — neither row may be lost or corrupt the
# list — and, by completing, that the per-cluster locking never deadlocks/hangs.
# After both commit, the top-2 nearest of [100,0,0] are exactly 1001 and 1002.
# Expect 2.
permutation w_ins w2_begin w2_ins w_commit w2_commit c_chk2
