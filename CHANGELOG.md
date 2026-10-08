# Changelog

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
