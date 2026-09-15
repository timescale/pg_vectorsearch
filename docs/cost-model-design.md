# Design: a cost model for mktann index scans

Status: proposal for review. Nothing here is implemented; measurements are
from this box (PostgreSQL 18.4, warm shared buffers, 8 vCPU) on 2026-09-04.

## 1. Goals

1. An `ORDER BY v <-> q LIMIT k` query must choose the mktann scan over a
   sequential scan plus top-N sort whenever the scan is cheaper, which is
   every table beyond a few hundred rows, and may choose the sequential scan
   on a tiny table where routing plus reranking costs more than reading it.
2. When another ANN index exists on the same column (pgvector HNSW during a
   migration, say), the planner should pick mktann where it is faster, which
   is nearly always, without a GUC.
3. The estimate must follow the index's actual structure: how many posting
   lists there are and how long they are (an index that started empty and only
   ever saw inserts can be one giant list), which page format each list uses
   (fastscan pages from the build, AoS pages from inserts, converted lists),
   and how much of it is dead (tombstoned pages after deletes).
4. It must follow the session's query knobs: `mkt.nprobe` (or the automatic
   value), `mkt.rerank_pool`, `mkt.query_limit`, and the query's own `LIMIT`
   and `WHERE` clause.
5. It must be right from 200 rows to 100M rows with one formula. Nothing may
   be tuned to a size.
6. Plans must be stable: the estimate depends on catalog statistics and
   settings, not on the momentary content of shared buffers.

Non-goals for the first version: parallel scan costing (the unmerged branch
has a design for it; see §4 and §9), and choosing `nprobe` for the user.

## 2. What the planner does with what we return

`amcostestimate` returns five numbers and the core does the rest:

- `indexStartupCost` / `indexTotalCost`: the cost to produce the first and
  the last index tuple. `cost_index` adds the heap fetches for the rows the
  scan returns, priced with the Mackert-Lohman cache model over
  `effective_cache_size`, and the qual evaluation.
- `indexSelectivity`: the fraction of the table the scan returns. It sets the
  path's row estimate and therefore how many heap fetches the core adds.
- `indexCorrelation`: how well index order matches heap order; 0 for us.
- `indexPages`: index pages touched, used by the parallel-worker gate.

Two facts shape everything below. First, the planner does not tell the AM the
`LIMIT`, but `root->limit_tuples` is set while planning a query with a
constant `LIMIT`, and `root->tuple_fraction` otherwise; pg_textsearch already
reads the former in its cost estimator, and PR #227 hands the executor the
same value. Second, with a `LIMIT`, the planner charges a path
`startup + (total − startup) × k / rows`. An ordered index scan that does all
of its work before the first tuple must therefore put that work in
`startup`, or the `LIMIT` fraction hides it. The current estimator returns
`startup = 0` with `rows = N`, which is why it "wins" today: a 300k-row scan
is charged 8 cost units. That is an accident, not a model, and it would also
make a 1%-selective filtered scan look free.

## 3. Measurements this design is calibrated against

Clustered synthetic corpora: 300k rows at 128 dimensions (`nlist = 1225`,
`nprobe` auto = 17) and 60k rows at 768 dimensions (`nlist = 234`); k = 10
unless noted; every page in shared buffers. Per-phase times are the ones
`EXPLAIN (ANALYZE, VERBOSE)` already reports.

| Query | Pages read | Entries scanned | Rerank cands | Centroid | Posting scan | Rerank | Total |
|---|---|---|---|---|---|---|---|
| 128d fastscan, nprobe 10 | 59 | 12,018 | 270 | 0.05 ms | 0.65 ms | 0.25 ms | 0.9 ms |
| 128d fastscan, nprobe 30 | 153 | 30,590 | 270 | 0.20 | 1.26 | 0.25 | 1.8 |
| 128d fastscan, nprobe 100 | 321 | 60,918 | 270 | 0.09 | 2.55 | 0.22 | 3.0 |
| 128d fastscan, nprobe 300 | 796 | 142,832 | 270 | 0.16 | 6.65 | 0.21 | 7.1 |
| 128d fastscan, nprobe 30, k 100 | 153 | 30,590 | 1,600 | 0.06 | 1.31 | 4.93 | 6.5 |
| 128d fastscan, nprobe 30, k 1000 | 153 | 30,590 | 2,879 | 0.08 | 4.01 | 3.63 | 8.6 |
| 128d AoS, nprobe 300 | 985 | 188,087 | 165 | 0.11 | 2.74 | 0.08 | 3.0 |
| 768d fastscan, nprobe 60 | 587 | 33,684 | 160 | 0.04 | 0.78 | 0.39 | 1.2 |
| 768d AoS, nprobe 60 | 533 | 33,684 | 160 | 0.03 | 1.01 | 0.31 | 1.4 |
| 128d seq scan + top-N sort | 21,429 | 300,000 | — | — | — | — | 200 ms |
| 128d pgvector HNSW, ef 40 | 1,073 buffer hits | ~750 nodes | — | — | — | — | 0.49 ms |

Derived unit costs, warm:

| Quantity | 128d | 768d | Note |
|---|---|---|---|
| Fastscan scoring, per entry | ~45 ns (incl. per-cluster LUT setup) | ~23 ns | 16-bit LUT; setup is ~20 µs per cluster at 128d |
| AoS scoring, per entry | ~15 ns | ~30 ns | scalar/AVX kernel, no setup |
| Rerank, per candidate | 0.3–3 µs | 2.4 µs | heap fetch (cached) + exact distance; grows with dim and with k |
| Top-k maintenance | posting scan ×3 at k = 1000 vs k = 10 | | heap inserts scale with k |
| Fixed per-query overhead | ~10 µs | ~50 µs | rotation, state setup; negligible |
| Centroid descent | 0.05–0.2 ms | ~0.04 ms | small at these sizes; grows with beam × levels |

Two conclusions that shape the model. Format matters less than expected:
fastscan is ~25% faster per entry at 768d and slower per entry at 128d because
its per-cluster lookup-table setup is not amortized over a ~250-entry list.
And per-candidate rerank cost is the term most sensitive to `k` and to
dimension, so `k` and `mkt.rerank_pool` have to be first-class inputs.

## 4. The unmerged model on `feature/parallel-query`

`src/pg/mktann_cost.{c,h}` on that branch is the right skeleton and this
design keeps its shape: one `MktannCosts` struct with per-phase terms
(`descent`, `scan`, `rerank`, `io`, `emitted`, `index_pages`,
`selectivity`), computed from the metapage and the session GUCs, memoized per
statement, with `amcostestimate` reporting it as all-startup and a
`set_rel_pathlist_hook` re-costing partial paths for parallel query. It
already prices `mkt.nprobe`, probe expansion, and the rerank pool through the
same rule the executor uses (`mkt_query_rerank_pool`).

What it gets wrong, and what this design changes:

- **Constants are unmeasured and mis-scaled.** `MKT_COST_FETCH = 450`
  cpu_operator_cost units per rerank fetch is ~1.1 cost units, four times a
  cached heap page; measured cost is 0.3–2.4 µs. `MKT_COST_ENTRY = 0.2` is
  too low relative to that. The structure is right; the numbers need the
  table in §3.
- **I/O is priced in a different currency from everything it competes with.**
  It charges every non-resident posting page `4 × random_page_cost` (16 cost
  units). A sequential scan charges `seq_page_cost` per page and the HNSW
  estimator charges `random_page_cost` diluted through `genericcostestimate`;
  neither pays 16 per page. See §5.
- **Residency is sampled from the buffer mapping table at plan time.** That
  makes plans depend on cache state and differ between backends and moments;
  it also costs 64 hash probes per estimate. PostgreSQL's convention is the
  Mackert-Lohman estimate over `effective_cache_size`, which is deterministic
  and is what the sequential scan and the heap fetches are priced with.
  Residency sampling can return later as an opt-in refinement.
- **It reads `ntuples` from the metapage**, a field that no longer exists:
  it was the build-time count, never moved by inserts, and was dropped
  rather than kept current (see §7). The branch therefore does not build
  against `main` as it stands, and the model's row source is
  `indexinfo->tuples` alone.
- **No `k`.** It sizes the pool from `mkt.query_limit` or the default 10.
  `root->limit_tuples` is available to the estimator, and #227 has since
  merged, so the executor already derives the same number at run time --
  which is what phase 3 shares rather than duplicates.
- **No filter selectivity, no format mix, no dead pages, no per-dimension
  scaling**, and a tree-depth formula with a magic base of 74.

## 5. The currency problem, and why it decides the I/O model

The planner compares our estimate against two other estimators, and they
disagree about what a cost unit is worth.

The sequential scan on the 300k table is charged 38,382 units and takes
200 ms warm: about 5 µs per unit. pgvector's HNSW estimator charges 309 units
of startup for a query that takes 0.49 ms: about 1.6 µs per unit. Its formula
is `ratio × generic_total` where `ratio` is an estimate of nodes visited over
N (grows with log N, ~750 here) and `generic_total ≈ random_page_cost ×
pages + cpu × N`. Since pages grow linearly with N, the product is roughly
`visited × 0.4` and nearly independent of table size. Computed from the
formula:

| N | seq scan + sort | HNSW visited | HNSW startup |
|---|---|---|---|
| 1,000 | 85 | 406 | 165 |
| 100,000 | 8,665 | 687 | 280 |
| 1,000,000 | 87,483 | 812 | 331 |
| 10,000,000 | 883,134 | 953 | 388 |
| 100,000,000 | 8,914,386 | 1,094 | 446 |

In practice an HNSW query touches one to two random pages per visited node,
so the estimator prices HNSW as if its index were fully cached and cheap. Any
mktann model that charges raw `random_page_cost` per posting page will lose to
it on paper while winning in practice: at `random_page_cost = 4` a warm
300k-row mktann scan reading 57 pages would be charged more than HNSW's 309
before scoring a single entry.

The resolution is not to under-cost ourselves to match; it is to price pages
the way PostgreSQL prices every other cached read: `index_pages_fetched(pages,
total_index_pages, ...)` against `effective_cache_size`, times
`random_page_cost`. For an index smaller than `effective_cache_size` that
prices a page at a small fraction of a unit, which is what the sequential scan
and the heap-fetch terms assume too, and it degrades gracefully when the index
outgrows the cache. With that, and CPU anchored at one `cpu_operator_cost` per
~50 ns of measured work (PostgreSQL's own defaults imply roughly that:
`cpu_tuple_cost` 0.01 ≈ 4 operators per tuple), the proposed model lands
here (128d, auto nprobe, k = 10, `effective_cache_size` ≥ index):

| N | nlist | nprobe | entries | pages | mktann estimate | HNSW estimate | seq scan |
|---|---|---|---|---|---|---|---|
| 200 | 14 | 10 | 143 | 10 | ~40 | 82 | 17 |
| 1,000 | 31 | 10 | 323 | 10 | ~45 | 165 | 85 |
| 100,000 | 391 | 10 | 2,558 | 20 | ~55 | 280 | 8,665 |
| 1,000,000 | 3,906 | 31 | 7,937 | 62 | ~120 | 331 | 87,483 |
| 10,000,000 | 39,063 | 98 | 25,088 | 196 | ~330 | 388 | 883,134 |
| 100,000,000 | 390,625 | 312 | 79,872 | 624 | ~1,000 | 446 | 8,914,386 |

Reading it against the goals: the index wins over the sequential scan from
about a thousand rows and the sequential scan wins at 200 rows, both correct.
Against HNSW, mktann wins through 10M rows and loses at 100M with the
automatic `nprobe` of 312, where it is in reality two to four times faster
(rekall, cohere-100M). That last row is the one place honesty is not enough:
pgvector's estimate stops growing while real work does not. Three options,
in order of preference:

1. Accept it and document it. Two ANN indexes on one column is a migration
   state, and the operator can drop the old index.
2. A `mkt.cost_scale` GUC (default 1.0) that multiplies the estimate, so an
   operator can bias the choice without touching pgvector.
3. Anchor the CPU unit at ~100 ns instead of 50 ns. It halves every mktann
   estimate, keeps the ordering against the sequential scan (which is off by
   three to four orders of magnitude, not two), and puts the 100M row at
   ~500, a coin toss against HNSW. Cheap, but it is calibrating to a
   competitor's error rather than to the machine.

The review question is which of these to take. The proposal is 1 plus 2.

## 6. The model

All inputs are available at plan time without reading index pages beyond the
metapage, which `rd_amcache` already holds.

**Inputs.**

| Symbol | Source |
|---|---|
| `N` | `indexinfo->tuples` (pg_class.reltuples of the index; see §7), falling back to the heap's reltuples |
| `P` | `indexinfo->pages` |
| `nlist`, `dim`, `nlevels`, `fan_out`, formats | metapage via `rd_amcache` |
| health block | metapage, refreshed by VACUUM only (§7): `entries_live`, `entries_walked`, `sum_sizes`, `sum_sizes_sq`, `pages_live_fs`, `pages_live_aos`, `pages_tombstoned`, `rows_at_refresh` |
| `k` | `root->limit_tuples` if set, else `mkt.query_limit`, else 10; clamped to `MKT_QUERY_LIMIT_MAX`; `max(k, mkt.query_limit)` |
| `sel` | `clauselist_selectivity(root, baserestrictinfo)` when the relation has quals, else 1 |
| `k_eff` | `k` when `sel = 1`, else `ceil(margin × k / sel)` with the same rule and margin #227's executor uses (one shared function, so planner and executor agree) |
| `nprobe` | `mkt.nprobe`, else `mkt_auto_nprobe(nlist)`; clamped to `nlist` |
| `n_route` | `nprobe × mkt.probe_expand`, capped as the executor caps it |
| `pool` | `mkt_query_rerank_pool(k_eff)`, clamped to `nprobe × E` |
| `posting_pages` | `posting_pages_reachable` from the metapage (§7) plus growth since the last refresh; `P − centroid_pages` when the counter is absent |
| `pages_per_list` | `posting_pages / nlist` |
| `epp` | entries per page for the format, from `dim` (`mkt_posting_max_entries`) |
| `E` | expected entries per *probed* list: `pages_per_list × epp × (1 + CV²)`, where `CV² = nlist × sum_sizes_sq / sum_sizes² − 1` from the health block, else 0 |
| `d` | dead fraction `1 − entries_live / entries_walked` from the health block, else 0 |

Work is derived from pages, not from the row count. Pages already hold every
effect that makes a probe more expensive than the row count suggests:
SOAR and boundary replicas (about 1.35 entries per row on a healthy index),
dead entries that fastscan pages cannot flag individually and so keep scoring
until the whole page is dead, AoS pages appended by inserts, and, until
VACUUM reclaims them, the chains a split retired. Rows are used only where
rows are the unit: the selectivity.

**Terms** (in cost units; `cop = cpu_operator_cost`, `C_x` are the
calibrated constants of §8):

```
descent   = levels × beam × C_centroid_page × cop
          + n_route × C_probe_first_page × cop
scan_cpu  = nprobe × C_cluster(dim, fmt) × cop            -- per-list setup (LUT for fastscan)
          + entries × (f_fs × C_fs(dim) + (1 − f_fs) × C_aos(dim)) × cop
          + topk_factor(k_eff) × entries × cop            -- heap maintenance grows with k
index_io  = index_pages_fetched(nprobe × pages_per_list × (1 + CV²) + n_route
                                + descent_pages, P, ...) × random_page_cost
rerank    = pool / (1 − d) × (C_fetch(dim) × cop
          + index_pages_fetched(pool, heap_pages, ...) / pool × random_page_cost)
          -- dead candidates are fetched and yield nothing
emitted   = k_eff

indexStartupCost = (descent + scan_cpu + index_io + rerank) × loop_count
indexTotalCost   = indexStartupCost + emitted × cpu_index_tuple_cost
indexSelectivity = min(1, k_eff / N)
indexCorrelation = 0
indexPages       = descent_pages + nprobe × pages_per_list
```

where `entries = nprobe × E`, `f_fs` is the fastscan share of live posting
pages (the build's format when the health block is absent), and
`levels`/`beam` come from `nlevels` and the beam rule the executor uses
(`mkt.centroid_beam_scale`), not a fitted logarithm.

**Why each piece is there.**

- *Startup carries the work.* The scan materializes its result before the
  first tuple, so the LIMIT fraction must not discount it. `loop_count`
  multiplies it because a rescan inside a nested loop reruns the search.
- *Selectivity is `k_eff / N`.* The scan returns `k_eff` rows; the core then
  applies the qual and estimates `k_eff × sel ≈ k` output rows, and charges
  heap fetches for `k_eff`. That is exactly what happens.
- *Filters raise cost by `1 / sel`.* A 1%-selective filter makes `k_eff`
  100× larger, the pool larger, and the estimate correspondingly higher, so
  the planner can prefer a btree on the filter column plus a top-N sort. That
  was measured in the #227 work as 2 ms correct versus 52 ms and zero rows.
- *The one-giant-list index.* `nlist = 1` gives `nprobe = 1` and
  `pages_per_list` = every posting page: the estimate is a scan of all the
  codes. At 1M rows and 128d that is ~1,500 units against ~120 for the same
  rows in 3,900 balanced lists and 87,000 for the sequential scan — twelve
  times worse than balanced, still far ahead of the heap scan, which is also
  the truth (a million 25 ns code scores is 25 ms; the heap scan is 200 ms).
- *Skew.* The average list size hides an index where maintenance has not
  run and a few lists hold most of the rows. If a probe lands on a list with
  probability proportional to its size, the expected entries per probe are
  `Σ s² / Σ s = mean × (1 + CV²)`; a balanced index pays the mean, one with
  half its rows in a single list pays several times that. VACUUM reads every
  head's `live_count` anyway, so `Σ s` and `Σ s²` are free.
- *Dead entries.* Pages price the scoring of entries fastscan cannot flag.
  What pages do not price is the wasted rerank: a pool of `p` candidates with
  dead fraction `d` yields `p × (1 − d)` rows, so the scan either comes up
  short of `k` or, once resumable scans (#188) land, fetches `p / (1 − d)`.
  The `1 / (1 − d)` factor on the rerank term is that cost.
- *Formats.* Measured at most ~25% apart per entry, so the blend by live-page
  share is within calibration noise; the per-list setup term is what makes
  fastscan lose on short lists at low dimension.
- *Replication and inserts.* Replicas are real work and pages price them;
  `posting_pages × epp / reltuples` is ~1.35 on a healthy index with
  replication on. Inserts add single entries with no replica, so that ratio
  drifts down and the AoS share drifts up; growth since the last refresh is
  attributed to AoS pages at one entry per row. Lower replication lowers
  recall at a given `nprobe`, which is the auto-nprobe formula's concern
  (it was calibrated with replication on), not the cost model's.
- *Tiny tables.* Nothing special: at 200 rows the automatic `nprobe` floor of
  10 lists and a pool of 160 candidates cost more than reading 13 heap pages
  and sorting 200 rows, so the sequential scan wins by a little, as it should.

**Bloat: what it costs, measured.** Whether a bloated index can be
recognised from `relpages / ideal_pages` alone, with `ideal_pages ≈ nlist ×
ceil(r × N / nlist / epp)`, was tested two ways on 150,000 live rows at
128d, `nprobe` 30, the same query vector throughout.

| Index state | nlist | relpages | pages read | entries scored | rerank cands | time |
|---|---|---|---|---|---|---|
| fresh fastscan build | 625 | 1,716 | 51 | 7,759 | 229 | 0.28 ms |
| fastscan, 50% deleted, then VACUUM | 1,225 | 3,246 (1,188 of 3,209 posting pages tombstoned) | 168 | 31,554 | 631 | 1.08 |
| AoS, 50% deleted, then VACUUM | 1,225 | 3,194 | 69 | 6,000 | 304 | 0.54 |
| built at 10k rows, grown to 150k by inserts | 100 | 802 | 394 | 82,602 | 160 | 0.90 |
| fresh build on those same 150k rows | 625 | 1,601 | 86 | 15,213 | 160 | 0.22 |

Delete bloat shows in the page ratio (1.9×) and costs more than the ratio
says (3–4×): tombstoned pages are still read to follow the chain, dead
entries on fastscan pages are still scored because the format cannot flag
them, and dead candidates inflate the rerank pool almost threefold. VACUUM
fixes none of that on fastscan pages; a split or rebuild does. AoS pages can
skip flagged entries, so the same deletes cost about 2×.

Insert growth does not show in the page ratio at all: the grown index has
*fewer* pages than a fresh build (inserts add single entries with no
replicas, and AoS pages pack well at this dimension), yet a probe costs four
to five times more because 150,000 entries sit in 100 lists instead of 625.
That degradation lives in `pages_per_list`, which the model prices directly.

Retired chains after a split are never reached, so they cost nothing at
query time; they only inflate `relpages`, so a page-based estimate would
over-charge a split-heavy index by the unreachable fraction, which can
approach half the file until page recycling exists.

Hence the row-to-page ratio is exposed as an operator health number but is
not the model's input. The model needs the *reachable* posting page count,
kept exact by a `posting_pages_reachable` metapage counter maintained by the
operations that already write the metapage and know their page counts (build,
split: add the new lists and subtract the retired chain, convert), with no
VACUUM dependency and nothing on the insert path; and the VACUUM health block
for what pages cannot tell: the dead fraction that inflates the rerank pool
and the tombstoned share that is read but not scored.

## 7. Statistics the model needs, and two bugs to fix first

**Bug 1: VACUUM corrupts the index's `pg_class.reltuples`.** The build
sets it correctly on both the serial and parallel paths, but
`mktann_vacuumcleanup` returned a zeroed `IndexBulkDeleteResult` when no
bulk delete ran, so every such VACUUM (autovacuum after a bulk load included)
wrote `reltuples = 0`; and after a bulk delete it reported the sum of posting
live counts, which includes SOAR and boundary replicas and overstated 15,000
live rows as 20,428. Measured: 20,000 after build, 0 after an empty VACUUM,
300,000 again only after ANALYZE. Fixed in PR #237 (merged), following
nbtree and
pgvector: return NULL when nothing was deleted so the statistics stand, and
report `info->num_heap_tuples` with `estimated_count` otherwise.

**Bug 2, since resolved by deletion: the metapage `ntuples` was the
build-time count.** Nothing updated it after the build. PR #238 proposed
refreshing it from VACUUM's exact live count and exposing it through
`mkt.index_settings`; it was closed and the field removed instead, on the
grounds that a second row count which only VACUUM moves is a copy to keep
honest for no reader that `reltuples` does not already serve.

So the model has one row source, `indexinfo->tuples`, and no metapage
fallback -- which is simpler than this document originally assumed, and
removes a prerequisite rather than adding one. What it costs is a floor:
between vacuums the planner's count is as stale as `reltuples` is, and
there is no exact figure to fall back on.

**Two page counters in the metapage, maintained by the writers.**
`posting_pages_reachable` is set by the build, adjusted by a split (plus the
new lists' pages, minus the retired chain's) and by convert (new chain for
old), all of which already write the metapage. It is exact for the pages a
scan can reach, needs no VACUUM, and costs nothing on the insert path; growth
since the last VACUUM refresh is attributed to AoS pages as below.

`posting_pages_free` counts the pages that are neither reachable nor still
awaiting their reclaim horizon: pages the reclaim pass has recorded in the
free-space map (`RecordFreeIndexPage`, as nbtree does for deleted pages) and
nothing has taken back yet. It is incremented where a chain is recorded free
and decremented where a split or insert takes a page from the map instead of
extending the relation (`GetFreeIndexPage`); both sites already write index
pages, and the insert-side decrement fires once per new page, not per row.
The free-space map itself is the allocation structure and stays out of the
estimator: it can say whether any page is free from its root, but counting
free pages means walking its leaf level, which for a 4M-block index is about
a thousand pages of I/O — far beyond what any estimator may read, and no
PostgreSQL estimator reads the FSM. The counter is the number; the map is the
bitmap; VACUUM cross-checks the two the way it re-derives the health block.

Between them the counters partition `relpages − centroid_pages` into
reachable, free, and retired-awaiting-horizon pages, so the estimator can use
`posting_pages_reachable` for work and the health view can show both the
reusable and the not-yet-reusable bloat. Once new pages are taken from the
map before the relation is extended, `relpages` stops growing under churn and
the gap the counters explain shrinks; until a compaction pass exists to
truncate the tail, they are what keeps split and delete history out of the
estimate.

**Health block in the metapage, refreshed by VACUUM only.** The page-based
model needs no statistics to price a healthy index. What it cannot see
without help is *how* degraded a degraded index is: how skewed the list
sizes are, what share of entries are dead, and what share of pages a split
has retired. `bulkdelete` already walks every posting page and reads every
head's `live_count`, so `vacuumcleanup` can write, in one WAL record on
block 0: `entries_live`, `entries_walked`, `sum_sizes`, `sum_sizes_sq`,
`pages_live_fs`, `pages_live_aos`, `pages_tombstoned`, and `rows_at_refresh`
(the heap count at that moment). Inserts never touch it; the model
attributes rows added since `rows_at_refresh` to single AoS entries, which is
exactly what inserts produce.

This block would be the first VACUUM-time value written into the metapage
and read back through `rd_amcache`; #238 would have established that
pattern, and its closure means the health block has to carry the whole
argument for the mechanism itself rather than following one already in
place. That is a fair question to put to review before writing it.

The same numbers are what an operator needs to decide when to run
`mkt.rebalance`, so they should also be exposed, as rows of
`mkt.index_settings` or a small `mkt.index_health(regclass)` function:
replication ratio, dead fraction, list-size CV, free pages, and pages
awaiting reclaim. `pg_freespacemap` can confirm the free-page figure on
demand, which is fine for a view that runs occasionally and not for a
planner that runs on every statement.

## 8. Calibration

The constants are per-dimension functions, fitted to the measurements in §3
and re-measurable with the existing `EXPLAIN (ANALYZE, VERBOSE)` phase output
and the `mkt bench` standalone kernels:

| Constant | Form | 128d | 768d |
|---|---|---|---|
| `C_fs(dim)` | `a + b × dim`, ns → cop at 50 ns | ~0.5 cop | ~0.5 cop |
| `C_aos(dim)` | `a + b × dim` | ~0.3 cop | ~0.6 cop |
| `C_cluster(dim, fastscan)` | LUT build ∝ dim | ~400 cop | ~2,000 cop |
| `C_fetch(dim)` | tuple deform + exact distance, `a + b × dim` | ~6 cop | ~50 cop |
| `topk_factor(k)` | `1 + c × log2(k / 10)` from the k = 10 / 1000 pair | | |

They live in one header as named constants with the measurement they came
from, replaceable by GUCs (`mkt.cost_*`) if a deployment needs to override
them. A later phase can make them self-calibrating: every scan already
measures its own per-phase nanoseconds and counts, so a slow exponential
average of observed ns-per-entry and ns-per-fetch per backend could replace
the static constants. That is deliberately not in the first version because
it makes plans drift with load; it is listed under open questions.

## 9. Phases

1. **Prerequisites, now largely settled.** `reltuples` across VACUUM is
   fixed and merged (#237). The metapage row count (#238) is closed and the
   field deleted, so there is nothing left to build there. What remains of
   this phase is the VACUUM-refreshed health block of §7 and its exposure
   through `mkt.index_settings`, so the degradation tests in phase 2 have
   something to assert on -- and per §7 that block now needs its own
   justification rather than #238's.
2. **The serial model (main PR).** New `src/pg/cost.c` implementing §6 with
   §8's constants, replacing `genericcostestimate` in `mktann_costestimate`.
   Statement-scoped memo as on the branch. Regression tests in
   `test/pg/sql/cost.sql`: the index is chosen at 300k; the sequential scan
   or a near tie at 200 rows; a btree-plus-sort plan for a 1% anti-correlated
   filter; estimates monotone in `nprobe`, `k` and `mkt.query_limit`; the
   estimate rises when pages are AoS or tombstoned; and, in the pgvector
   compat suite, mktann chosen over HNSW on the same column at 300k rows.
3. **Shared `k` seeding with #227, which has merged.** The executor side
   now lives in `src/pg/scan_bound.c` (`mkt_scan_bound`) and
   `resolve_top_k` in `scan.c`, including the filter-selectivity inflation
   and the `work_mem` ceiling. One function that both that path and the
   estimator call, so the pool the planner prices is the pool the scan
   uses.
4. **Parallel re-costing.** Port the `set_rel_pathlist_hook` from
   `feature/parallel-query` on top of the new terms once parallel scans
   merge; its Amdahl split (descent serial, scan and rerank divided) is
   right and needs no change.
5. **Validation on rekall.** cohere-10M and cohere-100M with both indexes
   present: assert `EXPLAIN` picks mktann, and compare estimated against
   measured per-phase time to refit §8.

## 10. Open questions for review

1. Currency anchor: 50 ns per `cpu_operator_cost` (§5, honest against the
   sequential scan, loses to HNSW's estimate at 100M with auto nprobe) or
   100 ns (wins there, is really a bias)? Proposal: 50 ns plus a
   `mkt.cost_scale` GUC.
2. Residency sampling from the branch: drop, or keep behind a GUC default
   off? Proposal: drop for now; Mackert-Lohman is the convention.
3. The health block is refreshed by VACUUM only, with growth since the
   refresh attributed to single AoS entries. Is that enough between vacuums
   on an insert-heavy index, or does anything need an incremental counter?
   Proposal: VACUUM only; measure the drift on rekall before adding
   anything to the insert path.
4. Self-calibrating constants from observed scan timings: worth the plan
   instability? Proposal: not in the first version.
5. Page counters: `posting_pages_free` puts a metapage decrement on the
   insert path once per page taken from the free-space map. Acceptable, or
   should free-page accounting be VACUUM-only like the health block and the
   insert side left to growth attribution? Proposal: maintain it at the take
   site; one write per new page is already the cost of extending the
   relation.
6. Should the estimator read `root->tuple_fraction` when there is no constant
   `LIMIT` (cursors, `FETCH FIRST n PERCENT`)? Proposal: yes, as a second
   source for `k`, since the executor hint cannot see those either.
