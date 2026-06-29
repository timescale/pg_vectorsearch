# Parallel mktann build serialized behind a concurrent writer.
#
# A from-scratch CREATE INDEX takes a SHARE lock on the table, which conflicts
# with an open writer's ROW EXCLUSIVE lock, so the build must block until the
# writer's transaction ends and then build from the now-settled heap. This
# exercises the parallel build path (forced via the table's parallel_workers
# reloption + max_parallel_maintenance_workers) under real session concurrency,
# and checks the resulting index is correct: a row committed before the build
# starts is indexed; a row that was rolled back is not.
#
# Fixture: 50 well-separated points on a line at [1,0,0]..[50,0,0]; the probe
# row 1001 sits far out at [100,0,0], so once it is committed it is the
# unambiguous top-1 nearest neighbor of the query [100,0,0]. The checker probes
# every list (nprobe = nlist), so the answer depends only on index contents +
# MVCC visibility.

setup
{
    CREATE TABLE bld (id int, v vector(3));
    INSERT INTO bld SELECT g, format('[%s,0,0]', g)::vector
        FROM generate_series(1, 50) g;
    ALTER TABLE bld SET (parallel_workers = 2);
}
teardown { DROP TABLE bld; }

# The writer owns an explicit transaction so the build can collide with it
# while the insert is still uncommitted.
session writer
setup { BEGIN; }
step w_ins    { INSERT INTO bld VALUES (1001, '[100,0,0]'); }
step w_commit { COMMIT; }
step w_abort  { ROLLBACK; }

# The builder forces the parallel build and creates the index from scratch.
# Against the open writer it blocks on the SHARE lock (observe the
# <waiting>/<completed> markers).
session builder
setup { SET max_parallel_maintenance_workers = 2; }
step b_create
{
    CREATE INDEX bld_idx ON bld USING mktann (v) WITH (nlist = 4);
}
# A non-concurrent CREATE INDEX may run inside a transaction block and holds the
# table SHARE lock until commit. These let the build acquire the lock first and
# keep holding it while a concurrent insert blocks behind it.
step b_begin  { BEGIN; }
step b_commit { COMMIT; }

# The checker forces an index scan over every list; returns 1 iff the top-1
# nearest neighbor of [100,0,0] is row 1001.
session checker
setup { SET enable_seqscan = off; SET mkt.nprobe = 4; }
step c_chk
{
    SELECT count(*) AS found FROM (
        SELECT id FROM bld ORDER BY v <-> '[100,0,0]' LIMIT 1) t WHERE id = 1001;
}

# Writer commits before the blocked build proceeds: the parallel build sees the
# committed row 1001 and indexes it. Expect 1.
permutation w_ins b_create w_commit c_chk

# Writer aborts: the build proceeds over a heap without row 1001, so it is never
# indexed. Expect 0.
permutation w_ins b_create w_abort c_chk

# Reverse order: the build acquires the table SHARE lock first (inside an open
# transaction) and keeps it; the writer's insert then blocks behind the build
# instead of the other way around. The row is inserted only after the build
# commits, so it is NOT in the from-scratch build — it must reach the index
# through the insert path (aminsert) once the writer unblocks. This checks that
# a write serialized behind a build is still indexed (not lost). Expect 1.
permutation b_begin b_create w_ins b_commit w_commit c_chk
