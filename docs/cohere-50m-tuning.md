# Beating pgvector HNSW on cohere-50M — tuning guide & session lessons

Dataset: **cohere-wikipedia-22-12-50M**, 50M × 768-d, angular/cosine, k=10.
Hardware used: AWS r8i.16xlarge (64 vCPU, 495 GB RAM), PostgreSQL 18,
`shared_buffers = 128 GB`. Baseline competitor: **pgvector HNSW m24,
ef_construction=200** (191 GB index, ~5.5 h build).

## TL;DR — the winning recipe

1. **Build** the index (one-time):

   ```sql
   CREATE INDEX ON cohere_50m USING mktann (v mkt.vector_cosine_ops)
   WITH (nlist = 240000, centroid_compression = true,
         centroid_fastscan = true, fastscan = true,
         soar_lambda = 1.0, boundary_epsilon = 0.05);
   ```
   **Use `nlist = 240000`, not 480000.** n240k matches n480k's recall/QPS at the
   mid-band while being cheaper to build *and* faster to route (half the centroids
   to score). See "Granularity vs sampling" below — finer cells help only up to a
   point, and 480k is past it. Result: **13 GB index** (~15× smaller than HNSW's
   191 GB).

   Requires `ANALYZE cohere_50m;` first (otherwise `reltuples=0` makes the
   parallel-build sample DSM try to allocate ~437 GB and fail). Set
   `max_parallel_maintenance_workers=32`, `maintenance_work_mem='48GB'`.

2. **Query GUCs** (all recall-neutral except nprobe):

   ```sql
   SET mkt.centroid_error_scale = 0;      -- recall-neutral, +5..12% QPS
   SET mkt.centroid_beam_scale  = 0.25;   -- recall-neutral, +7..9% QPS (try 0.15–0.5)
   SET mkt.nprobe = 320;                  -- ~0.93; 480 → ~0.95; 640 → ~0.965
   ```

3. **System (critical, helps any large-index pgvector too):** back
   `shared_buffers` with huge pages. Transparent Huge Pages defaulting to
   `madvise`/shmem `never` leaves shared_buffers on 4 KB pages and TLB-thrashes
   large random-access indexes.

   ```bash
   echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
   echo always | sudo tee /sys/kernel/mm/transparent_hugepage/shmem_enabled
   sudo systemctl restart postgresql@18-main   # shared_buffers -> ShmemHugePages
   ```
   (Explicit hugepages via `nr_hugepages`+`huge_pages=on` is fragile — if the
   pool is short of total shared memory, **PG won't start**. THP `always` is the
   safe path.)

## Results (rekall, 3000 queries, both indexes prewarmed + THP-backed)

Matched-recall QPS on the real dataset in Postgres. The HNSW head-to-head below
was measured with **n480k**; n240k ties n480k at matched recall (see "Granularity
vs sampling"), so these numbers are a conservative stand-in for the recommended
n240k — n240k posts the same HNSW-relative margins at lower build/routing cost.

| Recall | meerkat n480k | HNSW m24 | winner |
|-------:|--------------:|---------:|:-------|
| 0.917  | 313 | 352 | HNSW +12% |
| 0.928  | **300** | 282 | **mkt +6%** |
| 0.935  | **292** | ~257 | **mkt +14%** |
| 0.94   | **246** | 241 | **mkt +2%** |
| 0.95   | **210** | 203 | **mkt +3%** |
| ≥0.965 | meerkat | — | **mkt** (widens with recall) |

Plus, in regimes the all-cached microbenchmark hides:
- **Cold / memory-pressured** (index can't stay fully resident — any normal
  32–128 GB instance): meerkat **2.5–3× faster** at matched mid-band recall,
  because its 15 GB index causes far less disk I/O than HNSW's 191 GB.
- meerkat QPS is **stable** across cache states; HNSW's swings ~30% with how
  much of its 191 GB is hot (it never fits 128 GB `shared_buffers`).

HNSW still wins **low recall (<0.92)** — its compute-light traversal shines when
few nodes need visiting.

## What actually moved the score (and by how much)

| Change | Effect | Where |
|---|---|---|
| **`mkt.centroid_error_scale=0`** | +5–12% QPS, recall-neutral | `centroid_search.c` (FASTSCAN `score_page`): the centroid routing error bound used for top-k *pruning* expands extra subtrees with zero recall benefit. Scaling that error by `error_scale` (default 0) prunes tighter while keeping the accurate `hacc` estimate. |
| **`mkt.centroid_beam_scale`** | +7–9% QPS, recall-neutral | `query_scan.c` `search_centroids`: intermediate tree-level beam width only needs `beam_width × fan_out ≥ nprobe` to cover the top-nprobe leaves, so it can be < nprobe. Scores far fewer centroids at high nprobe. |
| **Finer cells (nlist 80k→240k)** | ~50% higher QPS at matched recall vs n80k; sets the competitive operating point. Plateaus by 240k (n480k ties n240k) — see Granularity vs sampling | build params + the build-infra below |
| **Build infra** | *enabled* fine builds at all | `memory_pg.h` (palloc → `*_huge`, lifts 1 GB cap), `kmeans.c` (Elkan→Hamerly/CBLAS fallback above the 1 GB bound — note Elkan/Hamerly/Lloyd give identical clustering, this is a memory/speed choice not a quality one), `parallel_backend.c` (cap sample count to `reltuples` — **a stopgap, see Known limitations**), `mkt_pg.h` (`MKTANN_MAX_NLIST` 100k→2M), `mktann_scan.c` (`max_nprobe` 512→4096) |
| **Tree-aware SOAR build** (prior commit `0961bb9`) | fine builds ~4.5 h instead of ~15 h | top-64 candidate approximation for SOAR/boundary secondary — verified recall-neutral |
| **THP-backed shared_buffers** | ~+7–10% for *both*; restores fair comparison | system tuning, not code — but decisive for measurement honesty |

Net: from **19–35% behind HNSW everywhere** at session start → **winning the
mid-band, winning high recall, winning cold, at 12–16× smaller index and ~6×
faster build.**

## Granularity vs sampling — what drives the win (measured)

Going from coarse to fine changed *two* things at once — number of centroids
**and** k-means training-set size (the sample budget is `nlist × 256` capped to
row count, so at high `nlist` it hits 100% of the data). We isolated them.

Controlled four-way, identical query GUCs (`error_scale=0, beam_scale=0.25`), one
session, same build options except `nlist`:

| index | pts/centroid | sample | QPS @ ~0.94 (matched recall) |
|---|---|---|---|
| n80k-default | 256 | 41% (20.5M) | ~187 |
| **n80k-full** | **625** | **100% (50M)** | **~192** |
| n240k | 208 | 100% (capped) | ~245 |
| n480k | 104 | 100% (capped) | ~245 (ties n240k) |

**Finding 1 — sampling is a no-op.** Training the *same* n80k on the full 50M
(625 pts/centroid) instead of a 41% random sample (256) changed recall by ±0.008
(noise) and QPS not at all, at every nprobe. 256 samples/centroid is already past
k-means convergence; the extra centroid shift from more data is ~0.02σ and a
random sample is representative. So the full-dataset training that n240k/n480k get
is **incidental** (`nlist×256 > nrows` forces it), not causal — and you can safely
*reduce* sampling for faster builds (knob: `mkt.kmeans_sample_per_centroid`).

**Finding 2 — granularity is the driver, and it plateaus at ~240k.** n80k-full is
still ~25% behind n240k despite full training, so the gap is the finer *partition*
(tighter cells → vectors in a scanned cluster are closer to the query → more
relevant entries per probe), not training quality. But n480k only *ties* n240k:
finer cells help until per-centroid training data runs thin. n480k is *forced* to
104 pts/centroid (only ~104 vectors exist per cluster), now *below* the converged
~256, so its centroids are mildly under-trained — that offsets its finer
granularity. **n240k (208 pts/centroid, near-converged, finer than n80k) is the
sweet spot:** finest cells that still have enough data to train good centroids.

## What did NOT work (so don't repeat it)

- **Reducing SOAR replication** — uniform (`boundary_epsilon=0`, full SOAR ≈ same
  recall, *more* entries at fine cells) and **adaptive `soar_ortho_cutoff`**
  (prune replicas whose secondary residual is too parallel). The cutoff cut ~19%
  of entries at matched nprobe but was **Pareto-dominated**: a replica's *routing
  coverage* (vector findable via a 2nd cluster) is **independent of
  orthogonality**, and at fine granularity routing coverage — not estimation
  quality — is the binding constraint. Dropping any replica costs recall →
  needs more nprobe → net slower. (Code preserved on `experiment/soar-ortho-cutoff`.)
- **8-bit posting kernel** (`mkt.fastscan_bits=8`): memory/latency-bound, loses
  recall, no speedup.
- **Posting-scan prefetch tuning**: the existing 1-group-ahead prefetch (issued
  during the prune phase) is already optimal; moving it before the kernel /
  widening it / NTA locality all **regressed ~10%**.
- **Finer than n240k**: granularity plateaus — n480k *ties* n240k (and n80k-full
  is ~25% behind both), so going past ~240k buys nothing and costs a slower build
  + more centroid routing. Per-centroid training data runs thin (n480k = 104
  pts/centroid, under the converged ~256), so finer cells stop paying off. The
  deeper floor: at matched recall the entries scanned are set by *routing
  precision*, which graph traversal (HNSW) does better in high-D — its structural
  edge and the IVF floor.

## Measurement lessons (hard-won)

- **Compare within ONE benchmark invocation.** Run-to-run variance is ±10–30%
  from cache warmth and (especially) **huge-page/TLB state**. The same HNSW
  index measured 232 vs 306 QPS @0.928 across sessions purely from THP state.
- **Measure both systems with THP on**, or large-index HNSW is unfairly
  penalised (its random 191 GB walk is acutely TLB-sensitive; meerkat's compact
  15 GB barely cares).
- **Warm-vs-cold matters enormously.** A 200-query fully-cached microbenchmark is
  HNSW's *best* case; cold/pressured is meerkat's. Report the regime.
- **rekall** reuses an index if its generated name matches an existing one
  (skips build; `--rebuild` forces). It prewarms before measuring but **caps
  prewarm at ~96 GB**, so a 191 GB index is never fully prewarmed by rekall
  alone — prewarm large indexes manually (`pg_prewarm(..., 'read')`) for a fair
  warm comparison.
- Phase timing: `SET mkt.profile=on` + `EXPLAIN (ANALYZE, VERBOSE)` (custom
  lines need VERBOSE). At the mid-band the query is ~71% posting scan, ~11%
  centroid (after error_scale/beam_scale), ~7% rerank.

## Known limitations & follow-ups

- **Build-time sampling is mis-architected (the `reltuples` cap is a stopgap).**
  The k-means sample budget is `nlist*256`, and the parallel build preallocates a
  shared-memory segment sized to it — so at high `nlist` it overshoots (480k×256 =
  123M vs 50M rows → ~377 GB DSM → crash). The current fix caps to `reltuples`,
  but that's planner stats: it can be stale/wrong, and it requires a fresh
  `ANALYZE`. It also still preallocates proportional to the (capped) sample size —
  ~153 GB to materialize 50M sampled vectors — which doesn't scale. Proper fix:
  don't preallocate from a guessed count. Options: (a) **reservoir-sample** into a
  buffer whose size is the *target* bounded by an absolute/memory cap — no row
  count needed, and "fewer rows than target" falls out for free; (b) size from the
  heap's actual block count (`RelationGetNumberOfBlocks`, reliable, no ANALYZE)
  rather than `reltuples`; (c) best — **stream k-means over fixed-size batches**
  from the (cached) heap instead of materializing the whole sample set, removing
  the giant DSM entirely. NB: since >256 pts/centroid is a measured no-op, the
  target can also just be bounded low for a faster build.
- **Centroid beam is a flat `beam_scale`, not adaptive.** A tapered (wide-at-root,
  narrowing-down) beam + per-level error cutoff should match `beam_scale=1.0`
  recall at better-than-`0.25` speed. Query-side only (no rebuild). Deferred.
- **n480k / n80k-full are spent experiment indexes** (dead-ends); droppable.

## How to reproduce

- Build the meerkat extension (release) and install the `.so`:
  `meson compile -C builddir && sudo cp builddir/src/pg/meerkat.so "$(pg_config --pkglibdir)/"`
  (a new backend/session picks up the new `.so`; no PG restart needed for `.so`-only changes).
- Benchmark config: `benchmarks/cohere-50m-defn.toml` (sweeps nprobe × beam_scale
  for meerkat and ef_search for HNSW, runs=3, 3000 queries). Run with
  `rekall run benchmarks/cohere-50m-defn.toml` (no `--rebuild`).
