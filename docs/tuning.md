# Tuning Guide

An index created without options, queried without settings, targets
roughly 0.95 recall@10 out of the box. Most deployments should only
ever touch the settings in the first section — usually just
`prism.nprobe`.

Index parameters go in `CREATE INDEX ... WITH (...)` and are fixed at
build time (changing them means rebuilding the index). GUCs are set
per session or per query:

```sql
CREATE INDEX ON items USING prism (embedding vec32_cosine_ops)
    WITH (nlist = 40000);

SET prism.nprobe = 40;          -- session
BEGIN;
SET LOCAL prism.nprobe = 400;   -- one transaction
...
```

## Settings users should know

### prism.nprobe (GUC, query time)

**Affects: raising it gives higher recall at the cost of slower
queries.**

The speed/recall dial, and the one setting every deployment should
understand. The index partitions vectors into `nlist` clusters; a
query scans the `nprobe` clusters nearest the query vector.

- **Default:** `0` = automatic: `~0.5 * sqrt(nlist)` (at least 10, at
  most 2048), targeting roughly 0.95 recall@10 regardless of index
  size.
- **Lower it** when latency/throughput matters more than the recall
  tail (interactive search, candidate generation feeding a reranker).
  As a rule of thumb, roughly a third of the auto value gives ~0.90
  recall at several times the throughput.
- **Raise it** for recall-critical queries; returns diminish past a
  few times the auto value. `SET LOCAL` lets a single transaction pay
  for recall only where it matters.

### prism.query_limit (GUC, query time)

**Affects: how many rows an index scan is sized to return; normally
derived from the query, set it only to cap it.**

An index scan computes its whole top-k on the first fetch, so it has to
know k up front in order to use RaBitQ bounds to prune the candidate set
that would otherwise require expensive reranking on the full vectors.
The scan takes it from the query: a `LIMIT` (plus any `OFFSET`) directly
above the scan, or above nodes that keep one output row per input row (a
projection, a `row_number()` window), sizes the scan for that many rows
-- including inside a CTE, on the inner side of a join or `LATERAL`, and
in a cached prepared statement.

When the scan also filters via a `WHERE` clause, the executor applies
the filtering after the index has emitted its top-k, so only a fraction
of the emitted rows survive. The top-k is therefore inflated based on the
`WHERE` clause's selectivity in order to produce enough tuples to achieve
the required k after filtering. A filter anti-correlated with vector
proximity, whose survivors are not spread through the top-k the way the
estimate assumes, can still come up short; a partial index on the filter
avoids the estimate entirely.

The top-k heap is bounded by `work_mem`. A query asking for more -- a
large `LIMIT`, or a selective filter inflating the sizing -- returns as
many rows as that budget affords, and **raising `work_mem` is how to ask
for more**.

If none of the above mechanisms for managing the size of the top-k heap
are sufficient, the `prism.query_limit` GUC is a manual lever to cap the
number of tuples returned when the query lacks a `LIMIT` or it is set
very high.

- **Default:** `0` = size from the query's `LIMIT`, never below 10; with
  no usable `LIMIT` the query has asked for every row in order, so the
  sizing is what `work_mem` affords or the table's row count, whichever
  is smaller. Note an unlimited ordered scan therefore reranks every
  candidate it scans, which is inherent to returning them all in exact
  order. The floor keeps a little slack under a small `LIMIT` for rows
  the heap fetch discards (deleted but not yet vacuumed).

The top-k a scan actually used is reported as `Top-K` under
`EXPLAIN (ANALYZE, VERBOSE)`.

### nlist (index parameter, set at build)

**Affects: the recall/speed balance every later query starts from;
more lists also mean longer builds.**

The number of clusters the index partitions vectors into. Recall/QPS
at a given recall is surprisingly flat around the optimum, but far-off
values hurt: too few lists mean each probe scans too many vectors, too
many mean routing overhead and starved k-means training.

- **Default:** `0` = automatic: `rows / 256`, floored at
  `sqrt(rows)` — a couple of hundred vectors per list is the sweet
  spot between per-probe scan cost and routing overhead.
- **Change it** only when the table will grow far beyond its size at
  `CREATE INDEX` time (the estimate is taken then): either build with
  the target size's value in mind, or rebuild after loading.

### Build resources (PostgreSQL settings)

**Affects: build time and clustering quality; no effect on queries
beyond that.**

Index builds cluster a sample of the table and then scan it once.
Two standard PostgreSQL settings dominate build time:

- `maintenance_work_mem` — bounds the k-means sample and the build's
  sort memory. Bigger is better until the sample fits.
- `max_parallel_maintenance_workers` (plus the table's
  `parallel_workers` storage parameter) — parallel builds scale to
  tens of workers.

## Settings users should rarely touch

The defaults below are chosen for the best recall/throughput
tradeoff; they are documented for completeness and for controlled
experiments. Changing them without a measurement to justify it
usually makes things worse.

### Index parameters (set at build time)

All of these are fixed when the index is built; what they *affect* is
noted per row — most shape query behavior, some only the build.

| Parameter | Default | Affects | What it does | Why you might touch it |
|---|---|---|---|---|
| `fastscan` | `auto` (on) | Query speed. | Packed SIMD posting-page layout scanned via VPSHUFB lookup tables. `auto` uses it wherever a 32-vector group fits a page (it always does up to ~1900 dims at 8 KB pages); `on` errors past that, `off` forces the array-of-structs layout. | Only as an experiment control. |
| `centroid_fastscan` | `auto` (on) | Query speed. | The same packed layout for the centroid routing tree. `auto` additionally requires centroid compression to be in effect. | Only as an experiment control. |
| `centroid_compression` | `auto` | Query speed and routing memory. | RaBitQ-compress the routing tree's centroids (with exact re-rank of probe order via `prism.probe_expand`). `auto` compresses for L2/cosine and keeps exact centroids for inner product, whose ordering RaBitQ's estimate cannot recover. | Only as an experiment control. |
| `soar_lambda` | `1.0` | Higher recall at a given nprobe; slower build. | SOAR replication: assigns each vector to a second, spill-orthogonal cluster so near-boundary neighbors are found without extra probes. `0` disables. | Disable to trade recall for a faster build. |
| `boundary_epsilon` | `0.35` | Higher recall at a given nprobe; larger index. | Boundary replication band: additionally replicates vectors whose two nearest centroids are within this relative gap. Wider bands buy recall for a few percent of index size; `0` disables. | Shrink if index size is critical. |
| `fan_out` | `32` | Routing-tree shape (depth vs width). | Children per routing-tree node. | Never; the auto shape adapts. |
| `kmeans_nredo` | `1` | Build time vs marginal cluster quality. | K-means restarts during build. | Never. |
| `distance_mode` | `asymmetric` | Accuracy vs speed of distance estimates. | RaBitQ distance estimator. `symmetric` is faster with a larger estimation error. | Never; the rerank stage depends on asymmetric accuracy. |

### Query-time GUCs

Take effect per session or per query.

| GUC | Default | Affects | What it does | Why you might touch it |
|---|---|---|---|---|
| `prism.rerank` | `on` | Recall (off = quantized-only accuracy). | Re-rank candidates by exact distance from the heap before returning. | Off only to measure raw quantized accuracy. |
| `prism.rerank_pool` | `0` (auto) | Lower caps = faster queries, lower recall ceiling. | Caps the exact-rerank candidate pool. | Controlled experiments. |
| `prism.fastscan_bits` | `16` | 8 = faster scans, more estimation error. | Fastscan lookup-table precision. | Controlled experiments. |
| `prism.centroid_beam_scale` | `0.5` | Higher = slightly higher recall, slower queries. | Routing beam width as a fraction of nprobe (with a floor of 80 candidates, never more than nprobe). | Raise toward 1.0 to chase the last recall fraction at high nprobe. |
| `prism.centroid_error_scale` | `0` | Above 0: much slower routing for negligible recall. | Widens the routing beam by the RaBitQ error margin — keep at 0. | Never. |
| `prism.probe_expand` | `2.0` | Slightly higher recall for a little routing work. | Routes extra leaf candidates and re-ranks probe order by exact centroid distance, fixing compressed-routing noise. Gains saturate at 2.0; capped internally. | Never. |
| `prism.recent_buffers` | `on` | Warm-query speed; small per-backend memory. | Backend-local buffer-id cache that skips the shared buffer-mapping lookup on warm re-pins. | Never. |
| `prism.distance_mode` | `default` | Accuracy vs speed of distance estimates. | Per-session override of the index's `distance_mode`. | Never. |

### Build-time GUCs

Take effect during `CREATE INDEX` / `REINDEX`.

| GUC | Default | Affects | What it does | Why you might touch it |
|---|---|---|---|---|
| `prism.leaf_refine_threshold` | `0` (off) | Recall of sub-sampled builds; one extra table scan. | Re-center leaf encode references from the full table when the build sample was thin. | Large sub-sampled builds with recall shortfalls. |
| `prism.log_build_stats` | `off` | Log volume only. | Per-phase build resource logging (superuser). | Debugging build performance. |

## Reading query behavior

`EXPLAIN (ANALYZE)` on an index scan reports clusters scanned, pages
read, and rerank counts, which is usually enough to see whether a
recall problem is routing (raise `prism.nprobe`) or sizing
(`prism.query_limit`).

## Inspecting effective settings

Most settings default to automatic values (`nlist = 0`,
`prism.nprobe = 0`, `fastscan = auto`, ...), so neither
`pg_class.reloptions` nor `pg_settings` shows what an index actually
uses. `prism.index_settings(regclass)` reports the resolved values:

```sql
SELECT * FROM prism.index_settings('my_index');
       name       | setting | source
------------------+---------+---------
 dim              | 768     | column
 metric           | cosine  | opclass
 nlist            | 3906    | auto
 fan_out          | 16      | auto
 nlevels          | 3       | derived
 centroid_format  | fastscan| auto
 fastscan         | on      | auto
 soar_lambda      | 1       | default
 boundary_epsilon | 0.35    | default
 kmeans_nredo     | 1       | default
 distance_mode    | asymmetric | default
 nprobe           | 32      | auto
```

The `source` column tells where each value came from: `option`
(explicit reloption), `auto` (resolved automatic default), `default`
(reloption default), `column`/`opclass` (index definition), `derived`
(computed from other settings), or `session` (a GUC overriding the
index setting, as with `SET prism.nprobe`). Values the build persists
(`nlist`, `fan_out`, `nlevels`, the page formats) are read from the
index metadata and are authoritative for the index as built; options
the build consumes without persisting (`soar_lambda`,
`boundary_epsilon`, `kmeans_nredo`) are read from the catalog and
reflect the build only if they have not been changed with
`ALTER INDEX ... SET` since.
