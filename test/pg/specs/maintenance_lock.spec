# What the maintenance lock excludes: other maintenance, and DDL.
#
# Both entry points take ShareUpdateExclusiveLock on the index
# (PRISM_MAINT_LOCK) and hold it for the call. RowExclusiveLock, which
# they used to take, does not
# conflict with itself, and a split mints ids for its new clusters from the leaf
# count it read and writes the updated count back -- so two overlapping passes
# would hand the same ids to different lists and persist a count short by one
# pass's worth, which then undercounts leaves for the automatic probe count and
# the cost model.
#
# DDL is the other side of the same lock. Every statement here needs
# AccessExclusiveLock, on the index or on its table, so the two must exclude
# each other in both directions: neither may proceed while the other rewrites
# the same pages. A REINDEX that swapped the relation under a running split
# would leave the split writing into a relfilenode nobody is going to read, and
# a DROP would leave it writing into files being unlinked.
#
# The procedures refuse to run inside a caller's transaction, so a session
# cannot be made to hold the lock by wrapping the call in BEGIN. An injection
# point ("prism-split-locked") pauses a split mid-call instead, while it holds
# the lock, and the other session blocks until it is woken. The last pairing
# needs the wake-up too: the split blocks on lock acquisition first, then still
# has to pass the pause point once the DDL session lets go.
#
# Requires a PostgreSQL built with injection points (same gate as
# insert_serialize).
#
# Fixture: one list of 60 points along the x-axis, which a target of 10 splits
# six ways (trigger 20, width round(60/10) = 6). The manual split names block 2,
# which is where a freshly built single-list index puts its only posting head --
# a literal because CALL takes no subquery and a DO block would be a caller's
# transaction. The step that reads the head back proves the literal is right.

setup
{
    CREATE EXTENSION injection_points;
    CREATE TABLE maint (id int, v vec32(3));
    INSERT INTO maint SELECT g, format('[%s,0,0]', g)::vec32
        FROM generate_series(1, 60) g;
    CREATE INDEX maint_idx ON maint USING prism (v)
        WITH (nlist = 1, centroid_fastscan = off);
}

teardown
{
    DROP TABLE maint;
    SELECT injection_points_detach('prism-split-locked');
    DROP EXTENSION injection_points;
}

# Pauses mid-split, holding the maintenance lock. set_local confines the wait to
# this session, so the second call is not paused too -- it blocks on the lock.
session m1
setup
{
    SELECT injection_points_set_local();
    SELECT injection_points_attach('prism-split-locked', 'wait');
}
step m1_head  { SELECT blkno AS head_blkno FROM prism.posting_pages('maint_idx')
                    WHERE is_first ORDER BY blkno LIMIT 1; }
step m1_reb   { CALL prism.rebalance('maint_idx', 10); }
step m1_split { CALL prism.split_posting_list('maint_idx', 2); }

# The second session, through either entry point.
session m2
step m2_reb   { CALL prism.rebalance('maint_idx', 10); }
step m2_split { CALL prism.split_posting_list('maint_idx', 2); }

session d
step d_drop     { DROP INDEX maint_idx; }
step d_reindex  { REINDEX INDEX maint_idx; }
step d_begin    { BEGIN; }
step d_alter    { ALTER TABLE maint SET (fillfactor = 90); }
step d_commit   { COMMIT; }

session obs
step o_wake  { SELECT injection_points_wakeup('prism-split-locked'); }
step o_nlist { SELECT setting::int AS nlist FROM prism.index_settings('maint_idx')
                   WHERE name = 'nlist'; }
step o_heads { SELECT count(*) AS heads FROM prism.posting_pages('maint_idx')
                   WHERE is_first; }
step o_gone  { SELECT count(*) AS still_there FROM pg_class
                   WHERE relname = 'maint_idx'; }

# All four pairings of the two entry points block. In each, m1 pauses holding
# the lock and m2 waits for the wake-up, after which both complete.
permutation m1_head m1_reb   m2_reb   o_wake o_nlist
permutation m1_head m1_reb   m2_split o_wake o_nlist
permutation m1_head m1_split m2_reb   o_wake o_nlist
permutation m1_head m1_split m2_split o_wake o_nlist

# A split in progress delays the DDL. DROP goes through once the split has
# committed, so the index is gone afterwards rather than half-rewritten.
permutation m1_reb d_drop o_wake o_gone

# REINDEX likewise waits, and what it rebuilds to is the point. The split
# cleared the nlist reloption -- the declaration no longer described an index
# it had just re-partitioned -- so the rebuild sizes from the row count
# instead: prism_auto_nlist(60) is max(60/256, sqrt(60)) = 7. Were the
# reloption still there, REINDEX would come back to the single list the
# fixture declared and undo the split entirely, which is the regression this
# pins. (Ordering is the other half: the two never interleave.)
permutation m1_reb d_reindex o_wake o_heads

# The other direction: DDL holding AccessExclusiveLock delays the split. The
# split resumes when the DDL transaction ends, and still splits the list.
permutation d_begin d_alter m1_reb d_commit o_wake o_heads
