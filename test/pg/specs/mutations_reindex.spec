# REINDEX concurrent with DML.
#
# REINDEX takes an ACCESS EXCLUSIVE lock on the index, so it waits behind an
# open writer; once the writer commits, REINDEX rebuilds and the committed row
# is present in the fresh index.

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
step w_ins    { INSERT INTO iso VALUES (1001, '[100,0,0]'); }
step w_commit { COMMIT; }

session maint
step m_reindex { REINDEX INDEX iso_idx; }

session checker
setup { SET enable_seqscan = off; SET mkt.nprobe = 4; }
step c_chk
{
    SELECT count(*) AS found FROM (
        SELECT id FROM iso ORDER BY v <-> '[100,0,0]' LIMIT 1) t WHERE id = 1001;
}

# REINDEX blocks behind the open insert, rebuilds after commit, and the row is
# present in the rebuilt index.
permutation w_ins m_reindex w_commit c_chk
