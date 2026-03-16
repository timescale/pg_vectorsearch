-- Verify posting list pages are laid out contiguously on disk.
--
-- The build pre-reserves contiguous block ranges per cluster so that
-- sequential scans and prefetching work efficiently. This test
-- validates that pages within each posting chain are back-to-back.

-- dim=3, ~407 entries/page, nlist=3, 10000 vectors → ~8-11 pages/cluster.
CREATE TABLE posting_test (id serial, v vector(3));
INSERT INTO posting_test (v)
    SELECT format('[%s,%s,%s]',
        sin(i * 0.1)::real,
        cos(i * 0.1)::real,
        sin(i * 0.2)::real)::vector
    FROM generate_series(1, 10000) i;

CREATE INDEX idx_posting ON posting_test USING mktann (v)
    WITH (nlist = 3);

-- Show posting page layout per cluster
SELECT cluster_id, page_seq, blkno, entry_count
    FROM mkt_posting_pages('idx_posting'::regclass)
    ORDER BY cluster_id, page_seq;

-- Verify pages within each cluster are contiguous (blkno increments by 1)
SELECT bool_and(is_contiguous) AS all_contiguous
    FROM (
        SELECT blkno - lag(blkno) OVER (
                PARTITION BY cluster_id ORDER BY page_seq
            ) = 1 AS is_contiguous
        FROM mkt_posting_pages('idx_posting'::regclass)
    ) t
    WHERE is_contiguous IS NOT NULL;

DROP TABLE posting_test;
