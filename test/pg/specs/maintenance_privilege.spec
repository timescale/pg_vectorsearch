# A caller without privileges on the table is refused before the maintenance
# lock is requested.
#
# The maintenance procedures take ShareUpdateExclusiveLock on the index for the
# length of the call. That lock conflicts with VACUUM, autovacuum and DDL, so a
# request for it must not be queued on behalf of a role that will be refused
# anyway: a role with nothing but CONNECT could otherwise stall maintenance on,
# or get autovacuum cancelled for, any relation it can name by OID. The
# privilege checks therefore run under AccessShareLock, and the maintenance
# lock is taken only once they pass.
#
# The holder keeps a lock that conflicts with ShareUpdateExclusiveLock but not
# with AccessShareLock: SHARE on the table, and for the index itself the
# ShareUpdateExclusiveLock that ALTER INDEX SET takes (LOCK TABLE refuses an
# index). A refused call returns at once; one that requested the maintenance
# lock first would wait for the holder to commit.

setup
{
    CREATE ROLE regress_maint_unpriv NOLOGIN;
    CREATE ROLE regress_maint_owner NOLOGIN;
    CREATE TABLE maint (id int, v vec32(3));
    INSERT INTO maint SELECT g, format('[%s,0,0]', g)::vec32
        FROM generate_series(1, 60) g;
    CREATE INDEX maint_idx ON maint USING prism (v)
        WITH (nlist = 1, centroid_fastscan = off);
    ALTER TABLE maint OWNER TO regress_maint_owner;
}

teardown
{
    DROP TABLE maint;
    DROP ROLE regress_maint_unpriv;
    DROP ROLE regress_maint_owner;
}

session holder
step h_begin  { BEGIN; }
step h_lock   { LOCK TABLE maint IN SHARE MODE; }
step h_alter  { ALTER INDEX maint_idx SET (nlist = 1); }
step h_reindex { REINDEX INDEX maint_idx; }
step h_drop    { DROP INDEX maint_idx; }
step h_chown   { ALTER TABLE maint OWNER TO regress_maint_unpriv; }
step h_commit { COMMIT; }

# The table's owner, a plain role so the ownership check is what admits it.
# Its call is allowed and so has to wait for the holder.
session owner
setup { SET ROLE regress_maint_owner; }
step o_reb { CALL prism_rebalance('maint_idx', 10); }

session unpriv
setup { SET ROLE regress_maint_unpriv; }
step u_convert  { CALL prism_convert_posting_to_fastscan('maint', 0); }
step u_split    { CALL prism_split_posting_list('maint', 1); }
step u_reb      { CALL prism_rebalance('maint'); }
step u_iconvert { CALL prism_convert_posting_to_fastscan('maint_idx', 0); }
step u_isplit   { CALL prism_split_posting_list('maint_idx', 1); }
step u_ireb     { CALL prism_rebalance('maint_idx'); }

# The unprivileged role names the table while the holder has SHARE on it.
# Each call fails with "is not an index" at once: it is refused before any
# lock that SHARE conflicts with is requested, so it never waits for the
# holder to commit.
permutation h_begin h_lock u_convert u_split u_reb h_commit

# The unprivileged role names the index while the holder has
# ShareUpdateExclusiveLock on it. Each call fails with "must be owner" at
# once: ownership is checked under AccessShareLock, which the holder's lock
# does not conflict with, and the maintenance lock is never requested. This is
# the permutation that pins the ordering. The table calls above are refused by
# index_open() before the ownership check runs, so they would pass even with
# the maintenance lock taken first.
permutation h_begin h_alter u_iconvert u_isplit u_ireb h_commit

# An allowed call that waits for the maintenance lock must hold no other lock
# on the index while it waits. The holder has ShareUpdateExclusiveLock from
# ALTER INDEX SET and upgrades to AccessExclusiveLock with REINDEX; a waiter
# still holding the AccessShareLock its checks ran under would deadlock with
# that upgrade. With nothing held, the REINDEX goes ahead of the waiter and
# both complete.
permutation h_begin h_alter o_reb h_reindex h_commit

# Nothing is held while the call waits, so the index it was checked against
# can be gone by the time the maintenance lock is granted. The second open
# reports that instead of failing on a stale OID.
permutation h_begin h_alter o_reb h_drop h_commit

# Likewise the table can change hands while the call waits. Ownership is
# checked again under the maintenance lock, so the call is refused rather than
# run on behalf of a role that no longer owns the table.
permutation h_begin h_alter o_reb h_chown h_commit
