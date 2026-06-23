# Beating pgvector HNSW on cohere-50M — tuning guide & session lessons

Dataset: **cohere-wikipedia-22-12-50M**, 50M × 768-d, angular/cosine, k=10.
Hardware used: AWS r8i.16xlarge (64 vCPU, 495 GB RAM), PostgreSQL 18,
`shared_buffers = 128 GB`. Baseline competitor: **pgvector HNSW m24,
ef_construction=200** (191 GB index, ~5.5 h build).

## TL;DR — the winning recipe

1. **Build** the index (one-time, ~4.5 h):

   ```sql
   CREATE INDEX ON cohere_50m USING mktann (v mkt.vector_cosine_ops)
   WITH (nlist = 480000, centroid_compression = true,
         centroid_fastscan = true, fastscan = true,
         soar_lambda = 1.0, boundary_epsilon = 0.05);
   ```
   Requires `ANALYZE cohere_50m;` first (otherwise `reltuples=0` makes the
   parallel-build sample DSM try to allocate ~437 GB and fail). Set
   `max_parallel_maintenance_workers=32`, `maintenance_work_mem='48GB'`.
   Result: **15 GB index** (12–16× smaller than HNSW's 191 GB).

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

Matched-recall QPS on the real dataset in Postgres:

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
| **Finer cells (nlist 60k→480k)** | halves posting entries scanned per recall vs coarse cells; sets the competitive operating point | build params + the build-infra below |
| **Build infra** | *enabled* fine builds at all | `memory_pg.h` (palloc → `*_huge`, lifts 1 GB cap), `kmeans.c` (Elkan→Hamerly/CBLAS fallback above the 1 GB bound), `parallel_backend.c` (cap sample count to `reltuples`), `mkt_pg.h` (`MKTANN_MAX_NLIST` 100k→2M), `mktann_scan.c` (`max_nprobe` 512→4096) |
| **Tree-aware SOAR build** (prior commit `0961bb9`) | fine builds ~4.5 h instead of ~15 h | top-64 candidate approximation for SOAR/boundary secondary — verified recall-neutral |
| **THP-backed shared_buffers** | ~+7–10% for *both*; restores fair comparison | system tuning, not code — but decisive for measurement honesty |

Net: from **19–35% behind HNSW everywhere** at session start → **winning the
mid-band, winning high recall, winning cold, at 12–16× smaller index and ~6×
faster build.**

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
- **Finer than n480k**: at matched recall the entries-scanned floor is set by
  *routing precision*, not granularity — finer cells just need proportionally
  more nprobe for the same coverage. Graph traversal (HNSW) is fundamentally
  more precise in high-D; that's its structural edge and the IVF floor.

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

## How to reproduce

- Build the meerkat extension (release) and install the `.so`:
  `meson compile -C builddir && sudo cp builddir/src/pg/meerkat.so "$(pg_config --pkglibdir)/"`
  (a new backend/session picks up the new `.so`; no PG restart needed for `.so`-only changes).
- Benchmark config: `benchmarks/cohere-50m-defn.toml` (sweeps nprobe × beam_scale
  for meerkat and ef_search for HNSW, runs=3, 3000 queries). Run with
  `rekall run benchmarks/cohere-50m-defn.toml` (no `--rebuild`).
