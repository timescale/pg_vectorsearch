-- Parallel vs serial index build paths
--
-- Exercises the mktann parallel build (do_parallel_build) and serial build
-- (do_serial_build) and asserts both produce a usable index. A small table is
-- enough: setting the table's parallel_workers storage parameter forces a
-- parallel build regardless of heap size (it bypasses the size/memory
-- heuristics in plan_create_index_workers), so we get parallel coverage
-- without a slow, large dataset.
--
-- The parallel path runs the full option matrix (it is otherwise untested);
-- the serial path runs the options no other regression test covers serially
-- (fastscan / SOAR / boundary), since mktann.sql already covers serial
-- L2/cosine/float/multi-level builds.

CREATE TABLE embeddings (id serial, v vector(3));

INSERT INTO embeddings (v)
    SELECT format('[%s,%s,%s]', x * 0.1, y * 0.1, z * 0.1)::vector
    FROM generate_series(0, 9) x,
         generate_series(0, 9) y,
         generate_series(0, 9) z;

SELECT count(*) FROM embeddings;

-- Build the index from a passed definition tail, verify it has pages and that
-- an index-only ORDER BY scan returns the requested LIMIT, then drop it. The
-- order-by operator is a parameter so cosine (<=>) and L2 (<->) opclasses can
-- share the helper.
CREATE FUNCTION pbuild_check(idxdef text, op text DEFAULT '<->')
    RETURNS TABLE (has_pages bool, nres bigint)
    LANGUAGE plpgsql AS $$
BEGIN
    EXECUTE 'CREATE INDEX tmp_idx ON embeddings USING mktann ' || idxdef;
    RETURN QUERY EXECUTE format(
        'SELECT (SELECT c.relpages > 0 FROM pg_class c'
        '          WHERE c.relname = ''tmp_idx''),'
        '       (SELECT count(*) FROM ('
        '          SELECT id FROM embeddings'
        '          ORDER BY v %s ''[0.5,0.5,0.5]'' LIMIT 5) t)',
        op);
    EXECUTE 'DROP INDEX tmp_idx';
END $$;

SET enable_seqscan = off;

-- ============================================================
-- Parallel build path (full option matrix)
-- ============================================================
-- The parallel_workers storage parameter is honored directly by
-- plan_create_index_workers (capped at max_parallel_maintenance_workers),
-- which is how we drive the parallel path on a tiny table.
SET max_parallel_maintenance_workers = 2;
ALTER TABLE embeddings SET (parallel_workers = 2);
SELECT reloptions FROM pg_class WHERE relname = 'embeddings';

-- L2, RaBitQ centroids
SELECT * FROM pbuild_check('(v) WITH (centroid_compression = true)');
-- L2, float centroids
SELECT * FROM pbuild_check('(v) WITH (centroid_compression = off)');
-- Cosine
SELECT * FROM pbuild_check(
    '(v vector_cosine_ops) WITH (centroid_compression = off)', '<=>');
-- Fastscan posting lists
SELECT * FROM pbuild_check('(v) WITH (fastscan = true)');
-- Multi-level tree
SELECT * FROM pbuild_check('(v) WITH (fan_out = 4, centroid_compression = true)');
-- SOAR replication
SELECT * FROM pbuild_check('(v) WITH (soar_lambda = 1.0)');
-- Boundary replication
SELECT * FROM pbuild_check('(v) WITH (boundary_epsilon = 0.1)');

-- ============================================================
-- Serial build path (options not covered serially elsewhere)
-- ============================================================
-- max_parallel_maintenance_workers = 0 forces the serial fallback even though
-- the table still carries parallel_workers = 2.
SET max_parallel_maintenance_workers = 0;

SELECT * FROM pbuild_check('(v) WITH (fastscan = true)');
SELECT * FROM pbuild_check('(v) WITH (soar_lambda = 1.0)');
SELECT * FROM pbuild_check('(v) WITH (boundary_epsilon = 0.1)');

RESET enable_seqscan;

-- ============================================================
-- Edge case: parallel build of an empty table must not crash
-- ============================================================
CREATE TABLE empty_emb (id serial, v vector(3));
ALTER TABLE empty_emb SET (parallel_workers = 2);
SET max_parallel_maintenance_workers = 2;
CREATE INDEX idx_empty ON empty_emb USING mktann (v);
SELECT count(*) FROM empty_emb;
DROP TABLE empty_emb;

RESET max_parallel_maintenance_workers;

-- Cleanup
DROP FUNCTION pbuild_check(text, text);
DROP TABLE embeddings;
