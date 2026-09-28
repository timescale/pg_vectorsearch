# Index Maintenance

**Experimental**, and still missing functionality.

An insert appends to whichever posting list its vector routes to, so lists
grow as rows arrive and never split on their own. A `REINDEX` can rebuild
the index so the partitions match the new size, but that rewrites the whole
index. A list can instead be split in place. Two procedures do that on
demand.

## Rebalancing and splitting

The examples use the table from the README's Usage section. Splitting needs
a flat index with RaBitQ centroid pages; the default build is neither.

```sql
CREATE TABLE items (
    id serial PRIMARY KEY,
    embedding vec32(3) STORAGE PLAIN
);

INSERT INTO items (embedding)
SELECT ARRAY[i::real, (i % 7)::real, (i % 5)::real]::vec32
FROM generate_series(1, 500) i;

CREATE INDEX items_idx ON items USING prism (embedding)
    WITH (nlist = 1, centroid_fastscan = off);

CALL prism_rebalance('items_idx');
-- NOTICE:  rebalance: split 1 posting list(s), reclaimed 0 retired chain(s)

-- Resting-size override. Lists already under twice this are left alone.
CALL prism_rebalance('items_idx', 256);

-- Head block from prism_posting_pages. CALL cannot take a subquery.
DO $$
DECLARE
    head bigint;
BEGIN
    SELECT min(blkno) INTO head
      FROM prism_posting_pages('items_idx')
     WHERE is_first;
    CALL prism_split_posting_list('items_idx', head);
END $$;
```

`target_entries` is the size a list rests at, not a ceiling: a list is left
alone until it holds twice that, so it has room to absorb inserts instead of
re-splitting on the next row. Passing `NULL` (the default) derives it from the
table's row count. A list over the trigger is divided into
`round(entries / target)` parts, which leaves each new list at the target.

## nlist and splits

A rebalance typically happens because the indexed dataset has grown, so the
index needs restructuring to keep performing well: the configuration chosen at
build time may be sub-optimal at the new size.

A rebalance therefore clears `nlist` from the index's reloptions, since the
value is no longer accurate. Later rebuilds, `REINDEX` included, size the
index from the current row count instead.

The rebalance above does that. `items_idx` is created with `nlist=1`; after
the call that splits, the reloption is gone and only `centroid_fastscan=off`
remains.

Set `nlist` again at any time to pin a width; the next rebalance that splits
will clear it again.

## Disk space

A rebalance leaves the lists it replaced in place rather than deleting them
immediately, so an index may temporarily use more disk space after a split or
rebalance. A `REINDEX` reclaims it.

## Supported index shapes

Splitting currently supports flat (single-level) indexes with RaBitQ centroid
pages; on any other shape the procedures raise an error naming the index. Note
that both are off the default path: `CREATE INDEX` packs centroid pages for
fastscan unless told otherwise, and a list count past the fan-out grows a
second level. To use these, build with `centroid_fastscan = off` and an
`nlist` that stays within one level.

## Inspecting the result

```sql
-- Leaf count, tree depth, centroid format, and the rest
SELECT * FROM prism_index_settings('items_idx');

-- One row per posting page; is_first marks a list's head
SELECT count(*) AS lists FROM prism_posting_pages('items_idx')
 WHERE is_first;
```
