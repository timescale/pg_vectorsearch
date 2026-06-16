# Concurrent insert / update / delete from multiple sessions.
#
# Interleaved DML from two writers (inserts route to the same cluster) must all
# land correctly — both rows present and findable, an update relocates a row,
# and a delete of the other session's row removes it. Per-cluster append
# serialization (the heavyweight page lock in aminsert) keeps the chain
# consistent under interleaving.

setup
{
    CREATE TABLE iso (id int, v vector(3));
    INSERT INTO iso SELECT g, format('[%s,0,0]', g)::vector
        FROM generate_series(1, 50) g;
    CREATE INDEX iso_idx ON iso USING mktann (v)
        WITH (nlist = 4, centroid_compression = true);
}
teardown { DROP TABLE iso; }

session s1
step s1_ins { INSERT INTO iso VALUES (1001, '[100,0,0]'); }
step s1_upd { UPDATE iso SET v = '[150,0,0]' WHERE id = 1001; }
step s1_del { DELETE FROM iso WHERE id = 1002; }

session s2
step s2_ins { INSERT INTO iso VALUES (1002, '[101,0,0]'); }

session checker
setup { SET enable_seqscan = off; SET mkt.nprobe = 4; }
step chk_both { SELECT count(*) AS present FROM iso WHERE id IN (1001, 1002); }
step chk_find
{
    SELECT count(*) AS found_updated FROM (
        SELECT id FROM iso ORDER BY v <-> '[150,0,0]' LIMIT 1) t WHERE id = 1001;
}

# Both inserts present (2); after deleting 1002 only 1001 remains (1); the
# updated 1001 is found at its new location.
permutation s1_ins s2_ins chk_both s1_upd s1_del chk_both chk_find
