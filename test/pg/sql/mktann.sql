-- mktann index access method

-- Create table with vector column
CREATE TABLE embeddings (id serial, v vector(3));

-- Insert deterministic data (grid of vectors)
INSERT INTO embeddings (v)
    SELECT format('[%s,%s,%s]', x * 0.1, y * 0.1, z * 0.1)::vector
    FROM generate_series(0, 9) x,
         generate_series(0, 9) y,
         generate_series(0, 9) z;

-- Verify data inserted
SELECT count(*) FROM embeddings;

-- L2 with centroid_compression (RaBitQ centroids, supports scan)
CREATE INDEX idx_l2c ON embeddings USING mktann (v)
    WITH (centroid_compression = true);

-- Verify index was built (should have pages)
SELECT relpages > 0 AS has_pages FROM pg_class
    WHERE relname = 'idx_l2c';

-- Verify index scan is used for ORDER BY <-> LIMIT
SET enable_seqscan = off;
EXPLAIN (COSTS OFF)
SELECT id, v <-> '[0.5,0.5,0.5]' AS dist
    FROM embeddings ORDER BY v <-> '[0.5,0.5,0.5]' LIMIT 5;

-- ORDER BY distance with LIMIT — should return results via index scan
SELECT count(*) FROM (
    SELECT id, v <-> '[0.5,0.5,0.5]' AS dist
    FROM embeddings ORDER BY v <-> '[0.5,0.5,0.5]' LIMIT 5
) t;
RESET enable_seqscan;

-- Create index with distance_mode relopt
CREATE INDEX idx_sym ON embeddings USING mktann (v)
    WITH (distance_mode = 'symmetric', centroid_compression = true);
CREATE INDEX idx_asym ON embeddings USING mktann (v)
    WITH (distance_mode = 'asymmetric', centroid_compression = true);

-- GUC: check default, set, and reset
SHOW mkt.distance_mode;
SET mkt.distance_mode = 'symmetric';
SHOW mkt.distance_mode;
SET mkt.distance_mode = 'default';
SHOW mkt.distance_mode;

-- L2 default (uncompressed float centroids) — build should work
CREATE INDEX idx_l2_float ON embeddings USING mktann (v);
SELECT relpages > 0 AS has_pages FROM pg_class
    WHERE relname = 'idx_l2_float';

-- IP opclass (uncompressed float centroids) — build should work
CREATE INDEX idx_ip ON embeddings USING mktann (v vector_ip_ops);
SELECT relpages > 0 AS has_pages FROM pg_class
    WHERE relname = 'idx_ip';

-- Cosine opclass (uncompressed float centroids) — build should work
CREATE INDEX idx_cos ON embeddings USING mktann (v vector_cosine_ops);
SELECT relpages > 0 AS has_pages FROM pg_class
    WHERE relname = 'idx_cos';

-- Cosine with compression — build should work
CREATE INDEX idx_cosc ON embeddings
    USING mktann (v vector_cosine_ops)
    WITH (centroid_compression = true);
SELECT relpages > 0 AS has_pages FROM pg_class
    WHERE relname = 'idx_cosc';

-- Validation: compression + IP should ERROR
CREATE INDEX idx_bad ON embeddings
    USING mktann (v vector_ip_ops)
    WITH (centroid_compression = true);

-- Insert after index creation (should not crash)
INSERT INTO embeddings (v) VALUES ('[1,1,1]');

-- VACUUM on the index (should not crash)
VACUUM embeddings;

-- Multi-level tree with fan_out (nlist=31, fan_out=4 → 3 levels)
CREATE INDEX idx_ml ON embeddings USING mktann (v)
    WITH (fan_out = 4, centroid_compression = true);
SELECT relpages > 0 AS has_pages FROM pg_class
    WHERE relname = 'idx_ml';

-- Multi-level tree scan should return results
SET enable_seqscan = off;
SELECT count(*) FROM (
    SELECT id, v <-> '[0.5,0.5,0.5]' AS dist
    FROM embeddings ORDER BY v <-> '[0.5,0.5,0.5]' LIMIT 5
) t;
RESET enable_seqscan;

-- Multi-level tree with uncompressed float centroids
CREATE INDEX idx_ml_float ON embeddings USING mktann (v)
    WITH (fan_out = 4);
SELECT relpages > 0 AS has_pages FROM pg_class
    WHERE relname = 'idx_ml_float';

-- fan_out = 2 (deepest tree)
CREATE INDEX idx_fo2 ON embeddings USING mktann (v)
    WITH (fan_out = 2, centroid_compression = true);
SELECT relpages > 0 AS has_pages FROM pg_class
    WHERE relname = 'idx_fo2';

-- Validation: fan_out must be >= 2
CREATE INDEX idx_bad_fo ON embeddings USING mktann (v)
    WITH (fan_out = 1);

-- EXPLAIN ANALYZE: verify mktann stats are present with sensible values
SET enable_seqscan = off;
CREATE FUNCTION test_explain_stats() RETURNS TABLE (
    has_clusters bool,
    has_centroid_pages bool,
    has_posting_pages bool,
    has_entries bool,
    has_rerank_cands bool,
    has_results bool,
    has_storage_reads bool
) LANGUAGE plpgsql AS $$
DECLARE
    explain_json json;
    stats json;
BEGIN
    EXECUTE 'EXPLAIN (ANALYZE, VERBOSE, COSTS OFF, FORMAT JSON)
        SELECT id, v <-> ''[0.5,0.5,0.5]'' AS dist
        FROM embeddings ORDER BY v <-> ''[0.5,0.5,0.5]'' LIMIT 5'
    INTO explain_json;
    stats := explain_json->0->'Plan'->'Plans'->0->'Mktann';
    RETURN QUERY SELECT
        (stats->>'Posting Lists Scanned')::int > 0,
        (stats->>'Centroid Pages Read')::int > 0,
        (stats->>'Posting Pages Read')::int > 0,
        (stats->>'Posting Entries Scanned')::int > 0,
        (stats->>'Rerank Candidates')::int >= 0,
        (stats->>'Rerank Results')::int > 0,
        (stats->>'Storage Reads')::int > 0;
END $$;
SELECT * FROM test_explain_stats();
DROP FUNCTION test_explain_stats();
RESET enable_seqscan;

-- Cleanup
DROP TABLE embeddings;
