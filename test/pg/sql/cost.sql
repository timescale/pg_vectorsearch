-- Cost model for index scans
--
-- The model prices the work a scan does rather than delegating to
-- genericcostestimate. What these assert is not
-- absolute numbers -- those depend on the machine's page costs -- but that
-- the estimate moves the right way with every input, and that the planner
-- reaches the right conclusion at both ends of the size range.

CREATE TABLE cost_test (id serial, grp int, v vec32(32));

INSERT INTO cost_test (grp, v)
    SELECT i % 10, (
        SELECT array_agg(sin(i * 0.1 + j * 0.7)::real)
        FROM generate_series(0, 31) j
    )::vec32(32)
    FROM generate_series(1, 20000) i;

CREATE INDEX idx_cost ON cost_test USING prism (v);
ANALYZE cost_test;

-- The query vector, fetched once into a psql variable and wrapped in an
-- immutable function so the planner folds it to a literal. Reading it with
-- a subquery instead puts an InitPlan in every plan, and that InitPlan's
-- own cost grows with the table, landing in the totals being compared.
--
-- psql substitutes a variable only outside quotes, so it cannot appear
-- inside the $q$...$q$ query texts below; building the function body with
-- format() and \gexec is what gets it in.
SELECT v::text AS qv FROM cost_test WHERE id = 42 \gset
SELECT format($f$
    CREATE FUNCTION qv() RETURNS vec32(32)
        LANGUAGE sql IMMUTABLE PARALLEL SAFE
        AS $b$ SELECT %L::vec32(32) $b$
$f$, :'qv') \gexec

-- The index scan's startup cost. This model puts the work there: a scan
-- elects its whole top-k before it can return a row, so a LIMIT must not
-- discount it.
CREATE FUNCTION cost_startup(q text) RETURNS float8
    LANGUAGE plpgsql AS $$
DECLARE
    j json;
    c float8;
BEGIN
    EXECUTE 'EXPLAIN (FORMAT JSON) ' || q INTO j;
    c := (jsonb_path_query_first(j::jsonb,
        '$.** ? (@."Node Type" == "Index Scan")."Startup Cost"'))::float8;
    IF c IS NULL THEN
        RAISE EXCEPTION 'no index scan in plan for: %', q;
    END IF;
    RETURN c;
END $$;

CREATE FUNCTION plan_uses(q text, idx text) RETURNS boolean
    LANGUAGE plpgsql AS $$
DECLARE
    j json;
BEGIN
    EXECUTE 'EXPLAIN (FORMAT JSON) ' || q INTO j;
    -- The index name as a name, not as a substring of the whole plan:
    -- 'idx_cost' otherwise matches idx_cost_small and idx_cost_aos too.
    RETURN j::text LIKE '%"Index Name": "' || idx || '"%';
END $$;

-- At 20,000 rows the index earns its cost and the planner takes it.
SELECT plan_uses($q$
    SELECT id FROM cost_test
    ORDER BY v <-> qv()
    LIMIT 10$q$, 'idx_cost') AS index_chosen_at_20k;

-- A 200-row table: the one size where the model prefers a sequential scan
-- while the index is in fact faster. No assertion on the plan here -- the
-- case and its cause are pinned further down, under "Two hundred rows".
CREATE TABLE cost_small (id serial, v vec32(32));
INSERT INTO cost_small (v)
    SELECT (SELECT array_agg(sin(i * 0.1 + j * 0.7)::real)
            FROM generate_series(0, 31) j)::vec32(32)
    FROM generate_series(1, 200) i;
CREATE INDEX idx_cost_small ON cost_small USING prism (v);
ANALYZE cost_small;

-- Every input moves the estimate the way the work moves. Captured rather
-- than compared inline because each needs its own GUC setting.
CREATE TEMP TABLE costs (label text primary key, c float8);

CREATE FUNCTION record_cost(label text, lim int DEFAULT 10) RETURNS void
    LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO costs
    SELECT label, cost_startup(format($q$
        SELECT id FROM cost_test
        ORDER BY v <-> qv()
        LIMIT %s$q$, lim));
END $$;

SET prism.nprobe = 1;    SELECT record_cost('nprobe1');
SET prism.nprobe = 10;   SELECT record_cost('nprobe10');
SET prism.nprobe = 100;  SELECT record_cost('nprobe100');
RESET prism.nprobe;

-- More probed clusters is more scanning, more I/O and a bigger rerank pool.
SELECT (SELECT c FROM costs WHERE label = 'nprobe1')
         < (SELECT c FROM costs WHERE label = 'nprobe10')
   AND (SELECT c FROM costs WHERE label = 'nprobe10')
         < (SELECT c FROM costs WHERE label = 'nprobe100')
    AS cost_rises_with_nprobe;

SELECT record_cost('k10', 10);
SELECT record_cost('k1000', 1000);

-- A larger LIMIT costs more: the rerank pool grows with the top-k, and the
-- heap maintenance costs more per entry scanned.
SELECT (SELECT c FROM costs WHERE label = 'k10')
         < (SELECT c FROM costs WHERE label = 'k1000')
    AS cost_rises_with_limit;

-- prism.query_limit only ever lowers the sizing (it caps a query that asks
-- for more than the caller will read), so it can only lower the estimate --
-- and the planner sees the same cap the scan will apply, because both call
-- one function to resolve it.
SET prism.query_limit = 20;
SELECT record_cost('k1000_capped', 1000);
RESET prism.query_limit;

SELECT (SELECT c FROM costs WHERE label = 'k1000_capped')
         < (SELECT c FROM costs WHERE label = 'k1000')
    AS query_limit_lowers_cost;

SET prism.rerank_pool = 20;   SELECT record_cost('pool20');
SET prism.rerank_pool = 2000; SELECT record_cost('pool2000');
RESET prism.rerank_pool;

-- Every candidate in the pool is a heap fetch and an exact distance.
SELECT (SELECT c FROM costs WHERE label = 'pool20')
         < (SELECT c FROM costs WHERE label = 'pool2000')
    AS cost_rises_with_rerank_pool;

-- A filter the executor applies above the scan does not restrict the scan:
-- it throws rows away afterwards, so the scan must produce more of them to
-- yield the LIMIT. That raises the cost, which is what lets the planner
-- prefer a filter-first plan when the filter is selective.
SELECT cost_startup($q$
        SELECT id FROM cost_test
        ORDER BY v <-> qv()
        LIMIT 10$q$)
     < cost_startup($q$
        SELECT id FROM cost_test WHERE grp = 3
        ORDER BY v <-> qv()
        LIMIT 10$q$)
    AS filter_raises_cost;

-- The inflation is priced, but not so heavily that it gives the plan away.
-- A tenth-selective filter means the scan has to produce about ten times
-- the LIMIT for ten to survive, and the index is still the faster way to do
-- that at this size.
SELECT plan_uses($q$
    SELECT id FROM cost_test WHERE grp = 3
    ORDER BY v <-> qv()
    LIMIT 10$q$, 'idx_cost') AS filter_keeps_the_index;

-- The rerank pool is priced in all three of its modes, and the order between
-- them is the point. prism.rerank_pool = -1 uncaps the pool, which makes the
-- pool-size estimator return 0 -- the value the extract step reads as "keep
-- every survivor". Read naively that zero prices the widest pool in the
-- system as no work at all, putting it level with reranking switched off and
-- making the index look cheapest exactly where it is most expensive.
SET prism.rerank = off;
SELECT record_cost('rr_off');
SET prism.rerank = on;
SET prism.rerank_pool = 0;
SELECT record_cost('rr_auto');
SET prism.rerank_pool = -1;
SELECT record_cost('rr_uncapped');
RESET prism.rerank_pool;
RESET prism.rerank;

SELECT (SELECT c FROM costs WHERE label = 'rr_off')
         < (SELECT c FROM costs WHERE label = 'rr_auto')
   AND (SELECT c FROM costs WHERE label = 'rr_auto')
         < (SELECT c FROM costs WHERE label = 'rr_uncapped')
    AS cost_rises_with_rerank_pool_width;

-- The pool's heap reads cover only the pages cost_index has not already
-- paid for, and that subtraction happens in pages, not candidates:
-- Mackert-Lohman saturates at the heap's size, so on a small heap removing
-- candidates removes no pages and the survivors get billed twice.
--
-- Widening the pool from a few candidates to every survivor must therefore
-- add less than a whole heap's worth of page cost.
SET enable_seqscan = off;

SET prism.rerank_pool = 5;
SELECT cost_startup($q$
    SELECT id FROM cost_small
    ORDER BY v <-> (SELECT v FROM cost_small WHERE id = 1)
    LIMIT 10$q$) AS c INTO TEMP pool_narrow;
SET prism.rerank_pool = -1;
SELECT cost_startup($q$
    SELECT id FROM cost_small
    ORDER BY v <-> (SELECT v FROM cost_small WHERE id = 1)
    LIMIT 10$q$) AS c INTO TEMP pool_wide;
RESET prism.rerank_pool;
RESET enable_seqscan;

SELECT (SELECT c FROM pool_wide) - (SELECT c FROM pool_narrow)
         < pg_relation_size('cost_small')
           / current_setting('block_size')::float8
           * current_setting('seq_page_cost')::float8
    AS rerank_pool_pages_not_billed_twice;

DROP TABLE pool_narrow, pool_wide;

-- Two hundred rows: the model calls for a sequential scan here, and the
-- reason is pages. mkt_auto_nlist floors the list count at sqrt(rows) and
-- a posting list takes at least a page, so this index is sixteen pages
-- against the heap's five, priced at the cost of reads that miss the
-- cache. The two assertions below pin that cause rather than the outcome.
--
-- First: the estimate is denominated in the page costs, so tuning them
-- down for a cached database selects the index. The sequential scan is
-- charged for its pages at the same rate, so this is not a small-table
-- special case.
SET seq_page_cost = 0.1;
SET random_page_cost = 0.1;

SELECT plan_uses($q$
    SELECT id FROM cost_small
    ORDER BY v <-> (SELECT v FROM cost_small WHERE id = 1)
    LIMIT 10$q$, 'idx_cost_small') AS index_chosen_at_200_rows_when_cached;

RESET seq_page_cost;
RESET random_page_cost;

-- Second: pack the same rows into one list and the index stops being
-- bigger than the table it indexes, so it wins at default page costs. The
-- pages were the whole of the difference.
CREATE INDEX idx_cost_small_packed ON cost_small
    USING prism (v) WITH (nlist = 1);

SELECT plan_uses($q$
    SELECT id FROM cost_small
    ORDER BY v <-> (SELECT v FROM cost_small WHERE id = 1)
    LIMIT 10$q$, 'idx_cost_small_packed') AS index_chosen_at_200_rows_packed;

DROP INDEX idx_cost_small_packed;

-- Scan setup is CPU work -- allocations and context setup -- so it is
-- denominated in cpu_operator_cost and not in a page cost. The difference
-- is invisible at default settings, where the two happen to coincide, and
-- shows up for an operator who tunes them apart: raising the CPU price
-- while dropping the page price has to raise the estimate, because the
-- setup the scan pays does not get cheaper when the data is cached.
SET seq_page_cost = 0.01;
SET random_page_cost = 0.01;
SET cpu_operator_cost = 0.0025;
SELECT cost_startup($q$
    SELECT id FROM cost_small
    ORDER BY v <-> (SELECT v FROM cost_small WHERE id = 1)
    LIMIT 10$q$) AS c INTO TEMP cpu_cheap;
SET cpu_operator_cost = 0.25;
SELECT cost_startup($q$
    SELECT id FROM cost_small
    ORDER BY v <-> (SELECT v FROM cost_small WHERE id = 1)
    LIMIT 10$q$) AS c INTO TEMP cpu_dear;
RESET cpu_operator_cost;
RESET seq_page_cost;
RESET random_page_cost;

-- A hundredfold rise in the CPU price has to move the estimate by at least
-- the setup term's share of it, which a page-denominated setup would not.
SELECT (SELECT c FROM cpu_dear) - (SELECT c FROM cpu_cheap) > 100
    AS setup_follows_cpu_cost_not_page_cost;

DROP TABLE cpu_cheap, cpu_dear;

-- An AoS index is costed and remains selectable. The two formats carry
-- different per-entry constants -- an AoS entry costs a small multiple of
-- a fastscan one, the layouts being the difference -- but this test does
-- not assert which index is dearer. At regress scale the per-entry term is
-- a few percent of the estimate, so the constants move the total by around
-- a percent and the ordering between the two is an accident of page counts
-- rather than of format. Asserting it would be asserting noise.
CREATE INDEX idx_cost_aos ON cost_test USING prism (v) WITH (fastscan = off);
DROP INDEX idx_cost;

SELECT (SELECT format FROM prism.posting_pages('idx_cost_aos') LIMIT 1)
    AS aos_index_is_aos;

SELECT plan_uses($q$
    SELECT id FROM cost_test
    ORDER BY v <-> qv()
    LIMIT 10$q$, 'idx_cost_aos') AS aos_index_is_chosen;

-- Centroid pages carry the same grouped-versus-per-vector distinction the
-- posting pages do: a FASTSCAN centroid page is scored with the group
-- kernel, a RABITQ one a vector at a time over the same codes, and
-- centroid_score_cost prices them apart.
--
-- As with the posting formats, this does not assert which index is dearer.
-- RABITQ packs more centroids into a page than FASTSCAN does, so it has
-- fewer centroid pages, and that difference in the page term can outweigh
-- the per-centroid rate either way depending on the tree's shape. What is
-- asserted is that the reloption reaches the format, so the branch is
-- exercised and a costed plan still comes out.
CREATE INDEX idx_cost_cfs ON cost_test
    USING prism (v) WITH (centroid_fastscan = on);
CREATE INDEX idx_cost_cnofs ON cost_test
    USING prism (v) WITH (centroid_fastscan = off);
CREATE INDEX idx_cost_cflt ON cost_test
    USING prism (v) WITH (centroid_compression = off);

SELECT (SELECT setting FROM prism.index_settings('idx_cost_cfs')
          WHERE name = 'centroid_format') = 'fastscan'
   AND (SELECT setting FROM prism.index_settings('idx_cost_cnofs')
          WHERE name = 'centroid_format') = 'rabitq'
   AND (SELECT setting FROM prism.index_settings('idx_cost_cflt')
          WHERE name = 'centroid_format') = 'float'
    AS centroid_formats_are_what_the_reloptions_asked_for;

SET enable_seqscan = off;

SELECT cost_startup($q$
        SELECT id FROM cost_test
        ORDER BY v <-> qv()
        LIMIT 10$q$) > 0 AS every_centroid_format_is_costed;

RESET enable_seqscan;
DROP INDEX idx_cost_cfs, idx_cost_cnofs, idx_cost_cflt;

-- A chain built as fastscan and then given inserts holds both formats at
-- once: the build writes packed fastscan pages, later inserts append AoS
-- ones. The estimate cannot see that mix -- it prices every entry at the
-- build format's rate -- so this asserts only that such an index is still
-- costed and still chosen, not that the number accounts for the drift.
-- Measured, a third of the pages being AoS makes the scan about 28% slower
-- than the estimate reflects.
DROP INDEX idx_cost_aos;
CREATE INDEX idx_cost ON cost_test USING prism (v);

INSERT INTO cost_test (grp, v)
    SELECT i % 10, (
        SELECT array_agg(sin(i * 0.37 + j * 0.7)::real)
        FROM generate_series(0, 31) j
    )::vec32(32)
    FROM generate_series(20001, 30000) i;
ANALYZE cost_test;

SELECT count(DISTINCT format) > 1 AS index_holds_both_formats
    FROM prism.posting_pages('idx_cost');

SELECT plan_uses($q$
    SELECT id FROM cost_test
    ORDER BY v <-> qv()
    LIMIT 10$q$, 'idx_cost') AS mixed_format_index_still_chosen;

-- Posting heads are priced as a prefetched batch only where prefetching
-- happens. The storage layer skips it when effective_io_concurrency is
-- zero, so in that configuration the head reads are independent seeks and
-- cost more; the estimate applies the same test the storage layer does,
-- rather than assuming a discount execution will not receive.
SET effective_io_concurrency = 16;
SELECT record_cost('eic_on');
SET effective_io_concurrency = 0;
SELECT record_cost('eic_off');
RESET effective_io_concurrency;

SELECT (SELECT c FROM costs WHERE label = 'eic_on')
         < (SELECT c FROM costs WHERE label = 'eic_off')
    AS seeks_cost_more_than_a_prefetched_batch;

-- Where the indexed vectors live decides what reranking one candidate
-- costs, and the model reads that from the statistics rather than inferring
-- it from the declared width. ANALYZE records the width of what is stored
-- in the tuple, so a column whose values were pushed out of line reports
-- the pointer left behind, and one holding them inline reports the vector.
--
-- A 600-dimension vector occupies 2408 bytes, past TOAST_TUPLE_THRESHOLD,
-- so default storage puts these out of line.
CREATE TABLE wide_vec (id serial, v vec32(600));
INSERT INTO wide_vec (v)
    SELECT (SELECT array_agg(sin(i * 0.11 + j * 0.3)::real)
            FROM generate_series(0, 599) j)::vec32(600)
    FROM generate_series(1, 300) i;
CREATE INDEX wide_vec_mkt ON wide_vec USING prism (v);
ANALYZE wide_vec;

-- 18 bytes stored for a 2408-byte vector: the signal the model reads.
SELECT avg_width AS stored_width_is_a_toast_pointer
    FROM pg_stats WHERE tablename = 'wide_vec' AND attname = 'v';

-- Toggling SET STORAGE here would prove nothing: it does not rewrite
-- existing rows, so the vectors stay out of line and the statistics keep
-- reporting the pointer's width. The estimate follows the statistics, so it
-- correctly does not move. The comparison that does test this is the one
-- between two physically distinct tables further down.


-- A parameterized scan inside a nested loop. The estimate is for one
-- scan: cost_nestloop applies the outer row count itself, so multiplying
-- here too would price a lateral kNN at the square of it.
--
-- Both assertions below are guards rather than discriminating tests. A
-- lateral whose subquery carries its own LIMIT is planned as a subquery,
-- and loop_count above 1 only reaches a parameterized path of an outer
-- baserel, so this shape arrives with loop_count = 1 and costs the same
-- either way. They pin the shape and the plan, not the multiplication.
SELECT cost_startup($q$
    SELECT o.id, s.id
    FROM (SELECT id, v FROM cost_test LIMIT 50) o,
         LATERAL (SELECT id FROM cost_test
                  ORDER BY v <-> o.v LIMIT 10) s$q$)
         <= cost_startup($q$
    SELECT id FROM cost_test
    ORDER BY v <-> qv()
    LIMIT 10$q$)
    AS repeated_scan_not_multiplied_by_loop_count;


SELECT plan_uses($q$
    SELECT o.id, s.id
    FROM (SELECT id, v FROM cost_test LIMIT 50) o,
         LATERAL (SELECT id FROM cost_test
                  ORDER BY v <-> o.v LIMIT 10) s$q$,
    'idx_cost') AS lateral_knn_uses_the_index;

-- At 5000 rows the index is measurably the faster plan, and the estimate has
-- to agree. This is the assertion the descent term moves: pricing the beam's
-- slots as though each were a page charged a fan-out's worth of random reads
-- against a centroid region that is only a few pages, which is the largest
-- single term at this scale and pushed the crossover well past where the
-- index starts winning.
CREATE TABLE cost_mid (id serial, v vec32(32));
INSERT INTO cost_mid (v)
    SELECT (SELECT array_agg(sin(i * 0.1 + j * 0.7)::real)
            FROM generate_series(0, 31) j)::vec32(32)
    FROM generate_series(1, 5000) i;
CREATE INDEX idx_cost_mid ON cost_mid USING prism (v);
ANALYZE cost_mid;

SELECT plan_uses($q$
    SELECT id FROM cost_mid
    ORDER BY v <-> qv()
    LIMIT 10$q$, 'idx_cost_mid') AS index_chosen_at_5k;

-- The descent cannot read more centroid pages than the index has.
SELECT (SELECT count(*) FROM prism.centroid_pages('idx_cost_mid')) > 0
    AS centroid_region_is_measurable;

-- A LIMIT the planner cannot fold. Under a generic plan limit_tuples is -1,
-- so without a fallback the scan would be sized for a full ranking of the
-- table and priced accordingly. root->tuple_fraction covers it: 0.10 for an
-- unfoldable LIMIT, which is the planner's own guess at how much of the
-- output will be read.
--
-- What this asserts is that the fallback lands between the two extremes --
-- more expensive than a LIMIT the planner can see, far cheaper than no
-- LIMIT at all -- not that the index wins. At a tenth of the table the
-- planner may well be right to sort instead, and that is its call to make
-- on the estimate.

-- Both readings below need the sequential scan disabled, and only because
-- it is the right plan for both: sized for a tenth of the table the planner
-- is correct to sort instead. The estimate is still what is under test, so
-- it has to be readable -- and cost_startup raises if no index scan appears,
-- so neither reading can come from a plan nobody looked at.
SET enable_seqscan = off;

PREPARE knn_limit(int) AS
    SELECT id FROM cost_test
    ORDER BY v <-> qv()
    LIMIT $1;
SET plan_cache_mode = force_generic_plan;

SELECT cost_startup($q$EXECUTE knn_limit(10)$q$) AS c INTO TEMP generic_limit;

RESET plan_cache_mode;
DEALLOCATE knn_limit;

-- preprocess_limit leaves tuple_fraction at 0.10 for a LIMIT it cannot
-- fold, so the fallback must size the scan for a tenth of the rows the
-- index holds -- which makes the estimate identical to that of the same
-- query with that tenth written out as a LIMIT. Equality rather than an
-- inequality because both go through the same arithmetic: anything else
-- means the fallback did not fire, or fired on the wrong quantity.
SELECT cost_startup(format($q$
    SELECT id FROM cost_test
    ORDER BY v <-> qv()
    LIMIT %s$q$,
    (SELECT ceil(reltuples * 0.10)::bigint
       FROM pg_class WHERE relname = 'idx_cost'))) AS c INTO TEMP tenth_limit;


SELECT (SELECT c FROM generic_limit) = (SELECT c FROM tenth_limit)
    AS unfoldable_limit_sized_from_tuple_fraction;

RESET enable_seqscan;

-- The same comparison across two physically distinct tables, which is what
-- exercises the statistics path: the vectors really are out of line in one
-- and inline in the other, and the stored widths differ accordingly. The
-- rerank pool is pinned small so the heap-page terms of both tables saturate
-- at the same value and the detoast term is what separates them -- left on
-- automatic the pool approaches the heap page count of the small
-- out-of-line heap, and the page term dominates instead (see the note on
-- rerank_cost about that inversion).
CREATE TABLE wide_plain (id serial, v vec32(600));
ALTER TABLE wide_plain ALTER COLUMN v SET STORAGE PLAIN;
INSERT INTO wide_plain (id, v) SELECT id, v FROM wide_vec;
CREATE INDEX wide_plain_mkt ON wide_plain USING prism (v);
ANALYZE wide_plain;

SELECT (SELECT avg_width FROM pg_stats
          WHERE tablename = 'wide_vec' AND attname = 'v') AS out_of_line_width,
       (SELECT avg_width FROM pg_stats
          WHERE tablename = 'wide_plain' AND attname = 'v') AS inline_width;

-- Three hundred rows: the planner is right to prefer a sequential scan
-- here, so it has to be disabled for the estimate to be readable at all.
-- What is under test is the cost of reranking an out-of-line vector, not
-- which plan wins -- and cost_startup still raises if no index scan
-- appears, so the reading cannot silently come from the wrong plan.
SET enable_seqscan = off;
SET prism.rerank_pool = 5;

SELECT cost_startup($q$
           SELECT id FROM wide_plain
           ORDER BY v <-> (SELECT v FROM wide_plain WHERE id = 1)
           LIMIT 10$q$)
         < cost_startup($q$
           SELECT id FROM wide_vec
           ORDER BY v <-> (SELECT v FROM wide_vec WHERE id = 1)
           LIMIT 10$q$)
    AS separate_tables_out_of_line_costs_more;

RESET prism.rerank_pool;
RESET enable_seqscan;

-- A LIMIT in an outer query above the subquery that holds the ORDER BY. The
-- subquery's own PlannerInfo never sees that LIMIT, so limit_tuples is -1
-- there; the executor walks out through the SubqueryScan and finds it, and
-- root->tuple_fraction is what carries it down to the planner. Without that
-- the scan is sized as if every row were wanted.

-- The no-LIMIT reading on the right needs the sequential scan disabled: with
-- every row wanted the planner is right to sort the table, and the estimate
-- still has to be readable to compare against. cost_startup raises if no
-- index scan appears, so neither reading can come from the wrong plan.
SET enable_seqscan = off;

SELECT cost_startup($q$
    SELECT * FROM (
        SELECT id FROM cost_test
        ORDER BY v <-> qv()
    ) sub LIMIT 10$q$)
         < cost_startup($q$
    SELECT id FROM cost_test
    ORDER BY v <-> qv()$q$)
    AS outer_limit_over_subquery_sizes_the_scan;

RESET enable_seqscan;


-- Which centroid-tree shape each index in this file has, since the descent
-- term treats them differently: a single-level tree is read in full, a tree
-- with a root above the leaves is read one page per kept slot per level,
-- bounded by the region. Both branches want exercising, and both are --
-- pinned here so that stays true if the clustering changes.
SELECT (SELECT count(DISTINCT level) FROM prism.centroid_pages('idx_cost_small'))
    AS small_index_levels,
       (SELECT count(DISTINCT level) FROM prism.centroid_pages('idx_cost'))
    AS default_index_levels;

-- A deliberately wide tree, to cover the nlist reloption and a region large
-- enough that the bound on the descent is what decides the term rather than
-- the slot count.
CREATE INDEX idx_cost_wide ON cost_mid USING prism (v) WITH (nlist = 500);
ANALYZE cost_mid;

SELECT (SELECT count(DISTINCT blkno)
          FROM prism.centroid_pages('idx_cost_wide'))
         > (SELECT count(DISTINCT blkno)
          FROM prism.centroid_pages('idx_cost_mid'))
    AS wide_tree_has_a_larger_centroid_region;

-- With both indexes present the planner prefers the narrower one, and it is
-- right to: fewer lists is a cheaper descent for the same data. So the wide
-- one is costed on its own.
BEGIN;
DROP INDEX idx_cost_mid;
SELECT plan_uses($q$
    SELECT id FROM cost_mid
    ORDER BY v <-> qv()
    LIMIT 10$q$, 'idx_cost_wide') AS wide_tree_index_is_chosen;
ROLLBACK;

-- Which centroid-tree branch of the descent term each index exercises is
-- pinned above: idx_cost_small is a single level, idx_cost two, so both the
-- whole-region case and the min(region, nlevels * beam) case run here. What
-- no index in this file reaches is a third level -- the clustering would
-- need far more rows than a regression test should build -- so the deeper
-- recursion of that branch is unexercised.
DROP FUNCTION record_cost(text, int);
DROP FUNCTION qv();
DROP TABLE wide_vec, wide_plain, cost_mid;
DROP FUNCTION cost_startup(text);
DROP FUNCTION plan_uses(text, text);
DROP TABLE cost_small;
DROP TABLE cost_test;
