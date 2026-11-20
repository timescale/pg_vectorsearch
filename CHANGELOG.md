# Changelog

## [0.2.0] - 2026-11-20

### Highlights

This release follows 0.1.0 with new features, faster queries, a more
compact index and bug fixes. It is still alpha quality. The on-disk
format changed, so every `prism` index has to be rebuilt with `REINDEX`
after the upgrade.

- Wider vectors. PRISM indexes vectors of up to 8192 dimensions, up
  from 1968. Up to 1968 any dimension works. Above that the dimension
  must be a power of two times an odd factor of at most 8, such as
  2048, 3072, 4096, 6144 or 8192. Inner-product indexes stop lower
  (see Known issues). The README's "Dimensions" section has the
  details.
- A more compact index. A posting list's codes now run on from one
  page to the next. Before, each group of codes had to fit whole on a
  page, which could leave up to half of the page empty. Each list also
  stores its reference vector as float16, half the size. Fewer pages
  per list means fewer pages read per query.
- Faster queries. Fastscan scoring now applies at every dimension. It
  used to stop at 944. Queries with a large `LIMIT` no longer slow
  down out of proportion to the `LIMIT`
  ([#369](https://github.com/timescale/pg_vectorsearch/pull/369), [#373](https://github.com/timescale/pg_vectorsearch/pull/373)).
- `l2_norm` and `vector_dims` take pgvector's names, so the same query
  works on pgvector and pg_vectorsearch columns.

New index option:

- `posting_reference_format` (`float16` or `float32`, default
  `float16`) sets how each posting list stores its reference vector.
  The build or a split fails if an element of a reference is beyond
  65504, float16's largest value. Use `float32` then. Vectors with an
  L2 norm up to 65504, and every cosine index, are always safe.

### Breaking changes & upgrade notes

- **Indexes must be rebuilt.** The on-disk index format changed (0x06 -> 0x07),
  so indexes built by earlier versions are rejected at open and have to be
  recreated.
- `vec32_norm` is renamed to `l2_norm`, for both `vec32` and `vec16`.
  `vec32_dims(vec16)` is renamed to `vec16_dims`. `vec32_dims(vec32)`
  keeps its name. Queries that call the old names must change. Views
  and indexes that use them keep working, because a rename keeps the
  function's OID.
- `vec16` refuses a value float16 cannot hold (beyond about ±65504) with
  an "out of range" error, as pgvector's `halfvec` does.

Upgrade from 0.1.0 with `ALTER EXTENSION pg_vectorsearch UPDATE`. Run it
in a new session that has not used the extension yet. The 0.2.0 library
refuses to load next to the 0.1.0 one. Then `REINDEX` each `prism` index.

### Known issues

- A `WHERE` filter can return fewer rows than `LIMIT` when the rows
  that pass are not spread through the candidates the scan ranked. The
  scan does not go back for more
  ([#188](https://github.com/timescale/pg_vectorsearch/issues/188)).
- The number of rows an ordered scan can return is capped by a buffer
  sized from `work_mem`. A larger `LIMIT`, or no `LIMIT` at all, is
  silently truncated to what fits (or to the table's estimated row
  count, if smaller). Raise `work_mem` to return more rows.
- `prism_rebalance` leaves the posting lists it replaced on disk. Only
  `REINDEX` reclaims that space
  ([#224](https://github.com/timescale/pg_vectorsearch/issues/224)).
- Setting `prism.leaf_refine_threshold` above 0 can build an index with
  much lower recall. Leave it at 0
  ([#183](https://github.com/timescale/pg_vectorsearch/issues/183)).
- An inner-product index is limited to 1968 dimensions on a `vec32`
  column and 3584 on a `vec16` column. `CREATE INDEX` fails above
  that. Use L2 or cosine for wider vectors. Normalized embeddings rank
  the same under cosine.
- `prism` refuses unlogged tables and expression indexes.

### Thanks

- @pavanmanishd for reporting and fixing two slowdowns in queries with
  a large `LIMIT`
  ([#371](https://github.com/timescale/pg_vectorsearch/issues/371), [#372](https://github.com/timescale/pg_vectorsearch/issues/372))
- @andreespirela for adding support for vectors of up to 8192
  dimensions ([#370](https://github.com/timescale/pg_vectorsearch/pull/370))
- @rebasedming for fixing memory growth in parallel index builds on
  toasted vectors ([#417](https://github.com/timescale/pg_vectorsearch/pull/417))

### Changes

- Indexes on vectors of up to 8192 dimensions
  ([25e4dea](https://github.com/timescale/pg_vectorsearch/commit/25e4deaebb114b9bf8f9ac077be0f85f7dd3b899))
- Posting-list codes continue across pages, so pages are filled and
  fastscan works at every dimension
  ([526c1bd](https://github.com/timescale/pg_vectorsearch/commit/526c1bdcbb5fc3ea01276ecfeef8fc34d5a30034))
- Posting lists store their reference vector as float16, with the
  `posting_reference_format` index option to keep float32
  ([ae2a75f](https://github.com/timescale/pg_vectorsearch/commit/ae2a75f23c6f8477807425cddbccd80c76cf106c))
- Faster queries with a large `LIMIT`. Collecting and deduplicating
  the candidates no longer grows faster than the `LIMIT` does
  ([5876aaf](https://github.com/timescale/pg_vectorsearch/commit/5876aaf91b812e23b6970e8178d8a64acf5b1dcd),
  [e7595f6](https://github.com/timescale/pg_vectorsearch/commit/e7595f61ee0a003afe92070aeef52203a5b18db9))
- Faster searches from less per-page overhead in the centroid search
  and the fastscan posting scan
  ([9f90dda](https://github.com/timescale/pg_vectorsearch/commit/9f90dda453a7aecee77a094230fcb467f7b1648f),
  [136c4a2](https://github.com/timescale/pg_vectorsearch/commit/136c4a27a0cca9eaec3482a9ebab1f4731300968))
- Faster `vec32` and `vec16` distance operators on ARM (NEON)
  ([29870c9](https://github.com/timescale/pg_vectorsearch/commit/29870c975bbbd4f5778e072a9f563990e0510527))
- Fixed wrong results from a scan after another session converted a
  posting list to the fastscan layout
  ([d9281db](https://github.com/timescale/pg_vectorsearch/commit/d9281db5102d980e49a7a380d56be1f0a370afdc))
- Fixed a scan missing a row that had been HOT-updated, and a split
  dropping rows that an older snapshot still needed
  ([1ffa59f](https://github.com/timescale/pg_vectorsearch/commit/1ffa59f367c8da525e1cac56f4bc7d8a05869dbf))
- Fixed a scan left open across a `prism_rebalance`, such as a cursor,
  reading freed memory
  ([32f0091](https://github.com/timescale/pg_vectorsearch/commit/32f009120a401e9e1bae15c256a29aafd4b0a0a6))
- Fixed parallel index builds on toasted vectors growing memory
  without bound
  ([93567fd](https://github.com/timescale/pg_vectorsearch/commit/93567fd1a10096a28f11a247151fe5acae8ee2d9))
- The `prism_*` inspection and maintenance functions error on an index
  that is not a `prism` index instead of misreading it
  ([ca2ea61](https://github.com/timescale/pg_vectorsearch/commit/ca2ea618aecccb4069e0db6bb58c5199acad28cf))
- Fixed a role with no privileges on a table being able to block
  `VACUUM` and DDL on it by calling a maintenance procedure
  ([4bd921d](https://github.com/timescale/pg_vectorsearch/commit/4bd921d68d5328731fe941f50f2efe6bac14e748))

[Full changelog](https://github.com/timescale/pg_vectorsearch/compare/v0.1.0...v0.2.0)

## [0.1.0] - 2026-10-13

### Highlights

This is the first release of pg_vectorsearch, currently at alpha level. The
core works and you can test it on real data today, but there are gaps you
should know about before you do (see Known Issues below). This is what works
today:

- Creating a PRISM index. Index builds on a populated table, including
  parallel builds, and k-nearest-neighbor search with L2, inner
  product and cosine distance.
- No dependencies. pg_vectorsearch ships its own `vec32` and `vec16`
  types so that it can operate independently. But PRISM also works
  with pgvector's `vector` and `halfvec` data types. You can build a
  PRISM index directly on existing pgvector columns with no rewrites
  and no copies, and queries written with pgvector's operators work
  unchanged.
- Simplified tuning. For queries, there is a single parameter to
  tune: `prism.nprobe`, the number of clusters a query scans. The
  default of 0 picks a value automatically to target roughly 95%
  recall.
- Inserts and updates work, but heavy churn can cause recall drift
  without manual intervention. Right now, partitions are not
  re-split automatically under churn, so the partitioning can become
  suboptimal. You can split oversized partitions manually by calling
  `prism_rebalance`, or rebuild the full index from scratch with
  `REINDEX`. We plan to remove the need for these manual steps and
  handle drift automatically. For this version, we don't recommend
  creating the index on an empty table and filling it afterwards.
- Delete operations work but don't yet reuse space. Deletes are
  turned into tombstones by VACUUM so that scans can skip them, but
  the space inside index pages isn't yet reused. Run `REINDEX` to
  reclaim it.

### Breaking changes & upgrade notes

None.

There is no upgrade path from 0.1.0-rc1 or 0.1.0-rc2. To switch, drop
and recreate the extension and its indexes. From this release
on, upgrades between final releases ship an upgrade script.

### Known issues

- A `WHERE` filter can return fewer rows than `LIMIT` when the rows
  that pass are not spread through the candidates the scan ranked. The
  scan does not go back for more
  ([#188](https://github.com/timescale/pg_vectorsearch/issues/188)).
- The number of rows an ordered scan can return is capped by a
  buffer sized from `work_mem`. A `LIMIT` larger than that buffer
  holds -- or no `LIMIT` at all -- is truncated to however many
  rows fit (or the table's estimated row count, if smaller),
  silently, with no error. Raise `work_mem` to return more rows.
- `prism_rebalance` leaves the posting lists it replaced on disk. Only
  `REINDEX` reclaims that space
  ([#224](https://github.com/timescale/pg_vectorsearch/issues/224)).
- Setting `prism.leaf_refine_threshold` above 0 can build an index with
  much lower recall. Leave it at 0
  ([#183](https://github.com/timescale/pg_vectorsearch/issues/183)).
- `prism` refuses unlogged tables and expression indexes, and indexes
  at most 1968 dimensions.

### Changes

- The extension's description in `\dx` now reads "ANN vector search
  with PRISM index"
  ([eedfabe](https://github.com/timescale/pg_vectorsearch/commit/eedfabe376857dab69365dc42329095edf4381d6))
- The README's usage example inserts enough rows for the planner to
  choose the index
  ([da022db](https://github.com/timescale/pg_vectorsearch/commit/da022dbd99310cb64c5aa4e14e0ec9d50a68805f))

Every change is in the commit history:
<https://github.com/timescale/pg_vectorsearch/commits/v0.1.0>

[Full changelog](https://github.com/timescale/pg_vectorsearch/compare/v0.1.0-rc2...v0.1.0)

## [0.1.0-rc2] - 2026-10-11

### Highlights

This is the second release candidate for 0.1.0. The extension is the
same as in 0.1.0-rc1. It is still alpha quality and meant for testing,
not production.

0.1.0-rc1 was published without a source tarball and checksum. This
candidate is the first to ship them. If you already run 0.1.0-rc1,
there is no reason to upgrade.

### Breaking changes & upgrade notes

None.

There is no upgrade path from 0.1.0-rc1. `ALTER EXTENSION ... UPDATE`
has no script to run. To switch, drop and recreate the extension and
its indexes.

### Known issues

The known issues listed for 0.1.0-rc1 below all still apply.

### Changes

#### Bug fixes

- *ci*: Release assets are attached before the release is published.
  An immutable release refused them afterwards, which left 0.1.0-rc1
  without a tarball ([d4d3167](https://github.com/timescale/pg_vectorsearch/commit/d4d3167e4b214326869b6cac94af6ceb56a99ace))

[Full changelog](https://github.com/timescale/pg_vectorsearch/compare/v0.1.0-rc1...v0.1.0-rc2)

## [0.1.0-rc1] - 2026-10-08

### Highlights

This is the first release candidate for 0.1.0, and the first release of
pg_vectorsearch. It is alpha quality and meant for testing, not
production.

### Breaking changes & upgrade notes

None.

An upgrade from this release candidate to a later release may not be
possible. `CREATE EXTENSION` warns about this on any prerelease. Plan to
drop and recreate the extension and its indexes.

### Known issues

- A `WHERE` filter can return fewer rows than `LIMIT` when the rows
  that pass are not spread through the candidates the scan ranked. The
  scan does not go back for more
  ([#188](https://github.com/timescale/pg_vectorsearch/issues/188)).
- An ordered scan with no `LIMIT`, or a `LIMIT` larger than `work_mem`
  allows it to rank, returns only as many rows as that budget affords,
  or the table's estimated row count if that is smaller. It does not
  error. Raising `work_mem` covers the budget case.
- `prism_rebalance` leaves the posting lists it replaced on disk. Only
  `REINDEX` reclaims that space
  ([#224](https://github.com/timescale/pg_vectorsearch/issues/224)).
- Setting `prism.leaf_refine_threshold` above 0 can build an index with
  much lower recall. Leave it at 0
  ([#183](https://github.com/timescale/pg_vectorsearch/issues/183)).
- `prism` refuses unlogged tables and expression indexes, and indexes
  at most 1968 dimensions.
