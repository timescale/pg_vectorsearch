# The mutations_concurrent scenarios at a spilled dimension.
#
# Past 1968 dimensions a posting list's head page holds only a chunk of its
# encode reference and the entries start after the reference pages, so an
# insert's reference gather, its append past a reference tail, and the scan's
# gather are different code than at a narrow dimension. The fixture is the
# same x-axis layout, embedded in 4096 dimensions: row g sits at g on axis 0
# and zero elsewhere, the probe row 1001 at 100, so the checker query returns
# 1 exactly when row 1001 is the visible nearest neighbour of [100,0,...].

setup
{
    CREATE TABLE iso (id int, v vec32(4096));
    INSERT INTO iso
        SELECT g, (SELECT array_agg(CASE WHEN j = 0 THEN g::real ELSE 0::real END)
                   FROM generate_series(0, 4095) j)::vec32(4096)
        FROM generate_series(1, 50) g;
    CREATE INDEX iso_idx ON iso USING prism (v)
        WITH (nlist = 4, centroid_compression = true);
    CREATE FUNCTION axis_point(x real) RETURNS vec32 AS $$
        SELECT (SELECT array_agg(CASE WHEN j = 0 THEN x ELSE 0::real END)
                FROM generate_series(0, 4095) j)::vec32(4096)
    $$ LANGUAGE sql IMMUTABLE;
}
teardown
{
    DROP TABLE iso;
    DROP FUNCTION axis_point(real);
}

session writer
setup { BEGIN; }
step w_ins    { INSERT INTO iso VALUES (1001, axis_point(100)); }
step w_commit { COMMIT; }
step w_abort  { ROLLBACK; }

session writer2
step w2_begin  { BEGIN; }
step w2_ins    { INSERT INTO iso VALUES (1002, axis_point(99)); }
step w2_commit { COMMIT; }

session maint
step m_reindex { REINDEX INDEX iso_idx; }

session checker
setup { SET enable_seqscan = off; SET prism.nprobe = 4; }
step c_chk
{
    SELECT count(*) AS found FROM (
        SELECT id FROM iso ORDER BY v <-> axis_point(100) LIMIT 1) t
        WHERE id = 1001;
}
step c_chk2
{
    SELECT count(*) AS found FROM (
        SELECT id FROM iso ORDER BY v <-> axis_point(100) LIMIT 2) t
        WHERE id IN (1001, 1002);
}

# Visible after commit, never after abort: the appended entry lives past the
# reference pages and the MVCC recheck decides.
permutation w_ins c_chk w_commit c_chk
permutation w_ins c_chk w_abort c_chk

# REINDEX waits for the open writer, then rebuilds a spilled layout that
# holds the committed row.
permutation w_ins m_reindex w_commit c_chk

# Two inserts into the same list from overlapping transactions: the second
# appends behind the chain metadata the first left, past the reference tail.
permutation w_ins w2_begin w2_ins w_commit w2_commit c_chk2
