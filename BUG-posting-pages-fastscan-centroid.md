# Bug: `mkt.posting_pages` / `collect_leaf_entries` crash on fastscan-centroid indexes

## Summary
`collect_leaf_entries()` in `src/pg/mktann_inspect.c` is not centroid-format
aware. It unconditionally reads the array-of-structs per-entry layout
(`mkt_centroid_meta(page, i)` → `entry->flags` / `entry->child_blkno`). On an
index whose centroid tree is stored in **FASTSCAN** format, that layout does not
hold — the per-entry metadata is replaced by a packed group section. The
function therefore reads garbage as `child_blkno`, then `ReadBuffer()`s those
bogus block numbers, leading to a crash or error.

This breaks every consumer of `collect_leaf_entries`:
- `mkt.posting_pages(regclass)` (`mkt_posting_pages`)
- `mkt.convert_posting_to_fastscan(regclass, int)`
  (`mkt_convert_posting_to_fastscan`)

## Affected versions / configs
Any index built with `centroid_fastscan=true` (which requires
`centroid_compression=true`). This is the default fast configuration used in the
cohere-50M benchmarks, e.g.:

```sql
CREATE INDEX ... USING mktann (v mkt.vector_cosine_ops)
  WITH (nlist=60000, centroid_compression=true, centroid_fastscan=true,
        fastscan=true, soar_lambda=1, boundary_epsilon=0.05);
```

## Reproduction
```sql
SELECT * FROM mkt.posting_pages('<fastscan_centroid_index>'::regclass) LIMIT 1;
-- crashes / errors instead of returning posting page rows
```

## Root cause
`mkt_centroid_pages()` (same file) already handles both layouts correctly: it
branches on `fmt == MKT_CENTROID_FMT_FASTSCAN`, derives leaf status from page
level (`opaque->level == nlevels - 1`) rather than a per-entry flag, and reads
children via `mkt_centroid_fastscan_group_child()`. `collect_leaf_entries()`
predates / was never updated for that layout and has none of this branching.

## Fix
Make `collect_leaf_entries()` format-aware, mirroring `mkt_centroid_pages()`:
read `nlevels` and `dim` from the meta page (pass them in from the callers,
which already hold the meta page), and on FASTSCAN pages determine leaf status
from page level and read posting heads from the group-child array.

## Note
The new `mkt.tids_clusters()` diagnostic deliberately avoids
`collect_leaf_entries` by scanning posting pages directly, which is why it works
on fastscan-centroid indexes today.
