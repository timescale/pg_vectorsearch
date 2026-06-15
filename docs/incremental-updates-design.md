# Incremental Inserts, Updates, and Deletes — Research & Design

## 1. Goal and scope

Meerkat is currently **build-only**: the index is created by `ambuild` from a
static table snapshot, and the posting lists and centroid tree are immutable
afterwards. Table mutations are not reflected in the index.

The goal of this work is to let the index track table `INSERT` / `UPDATE` /
`DELETE` so it stays correct and reasonably accurate **without a full
rebuild**. We will start simple (parity with pgvector `ivfflat` / VectorChord
`vchordrq`, plus correct delete handling) and build toward the state of the
art for IVF mutability — **SPFresh-style incremental rebalancing** — which
keeps recall stable under heavy mutation.

This document reports the research findings (current state, PostgreSQL
mechanisms, prior art) and proposes a phased design.

## 2. Current state (build-only)

Findings from the code:

- **`aminsert` is a stub that silently drops the entry.** `mktann_insert`
  (`src/pg/mktann.c`) just `return false;`. The `bool` result is only
  meaningful for unique indexes, so this is not an error — but it adds **no
  index entry**. Net effect today: rows inserted/updated after build are
  **silently missing from the index** (index scans won't return them). This is
  the first thing to fix.
- **`ambulkdelete` / `amvacuumcleanup` are no-op stubs.** Dead tuples are never
  removed from the index. The index accumulates dead TIDs over time; PostgreSQL
  filters them by heap visibility, but they still consume top-k slots.
- **Posting pages are immutable.** AoS entries (`MktPostingEntryHeader`:
  TID + `f_add`/`f_rescale`/`f_error` + RaBitQ bits) or FASTSCAN SoA in
  32-vector groups, chained by `next_blkno`. `MKT_POSTING_FLAG_DELETED` is
  defined but **never set or checked**.
- **Centroid tree is immutable** (hierarchical k-means, in-memory cache). But
  `mkt_centroid_beam_search()` already routes a vector to its nearest leaf
  posting(s) — directly reusable for insert routing.
- **Storage already supports incremental writes.** `new_page()` / `extend()` /
  `commit_page()` exist; runtime (non-build) writes go through `GenericXLog`
  with full-page images, while build batches WAL via `log_newpage_range()`.

So the routing, encoding, page-write, and TID machinery all exist; what is
missing is the write path (`aminsert`), delete handling, and any notion of
mutable storage or rebalancing.

## 3. PostgreSQL mechanisms to leverage

A correct, efficient mutable index should ride PostgreSQL's existing
machinery rather than reinvent it:

- **`aminsert`** is called per inserted/updated heap tuple. It must route +
  encode + append the vector and record the heap TID. (Replace the stub.)
- **HOT (Heap-Only Tuples).** An `UPDATE` that changes only *non-indexed*
  columns produces a HOT chain and **no new index entry** — the index keeps
  pointing at the chain head. So updates that don't touch the **vector column
  are free** (zero index work). PostgreSQL passes the `index_unchanged` hint
  (PG14+) to `aminsert` so the AM can also short-circuit when the indexed
  value is unchanged.
- **Vector-column `UPDATE` is non-HOT**: old tuple dies, new tuple is inserted
  → one `aminsert` for the new version, and the old index entry must be cleaned
  by vacuum. So "update" largely reduces to "insert new + delete old."
- **MVCC visibility / over-fetch.** The index returns *candidate* TIDs; the
  executor rechecks visibility against the heap. Dead/invisible tuples are
  filtered *after* the index returns them, so for top-k ANN the index must
  **over-fetch** (return more than k candidates) to still yield k live results.
  pgvector does exactly this.
- **`kill_prior_tuple` / LP_DEAD-style hints.** When the executor finds a
  returned TID is dead it sets `scan->kill_prior_tuple`; the AM can then mark
  that entry dead (reuse `MKT_POSTING_FLAG_DELETED`) so future scans skip it.
  Cheap, lazy dead-entry cleanup with no vacuum.
- **`ambulkdelete` / `amvacuumcleanup`.** `VACUUM` calls `ambulkdelete` with a
  "is this TID dead?" callback; the AM tombstones/removes matching entries.
  `amvacuumcleanup` finalizes (GC, page compaction, stats). This is the bulk
  dead-entry removal path.
- **`REINDEX CONCURRENTLY`** is the escape hatch for centroid drift in the
  simple phases — rebuild without blocking writers.

## 4. State of the art

### 4.1 Baseline: append-to-nearest + periodic rebuild (ivfflat, vchordrq)

pgvector `ivfflat` and **VectorChord `vchordrq`** both support inserts by
routing the new vector to its nearest list(s) and appending. The IVF
**centroids are fixed at build time**. As data grows or drifts, the fixed
partitioning becomes suboptimal and **recall degrades**, so they rely on a
periodic `REINDEX` (ideally `CONCURRENTLY`). VectorChord's docs describe insert
as placing "vectors in lists corresponding to their appropriate leaf nodes"
and document **no rebalancing** — i.e. the same drift-then-rebuild model.

This is simple and robust but: recall decays between rebuilds, and rebuild is
`O(N)`. It is the right **starting point** for Meerkat (and matches the
extension everyone is comparing against).

### 4.2 HNSW incremental (pgvector hnsw) — for contrast

HNSW links each new vector into the graph incrementally, so there is no
centroid drift and no rebuild requirement — but the graph must fit in RAM and
inserts do random traversal/writes. Not Meerkat's architecture, but it sets the
bar for "incremental without rebuild."

### 4.3 SPFresh / LIRE — the target

**SPFresh** (SOSP '23) builds on **SPANN** (an in-memory graph index over
centroids + on-disk posting lists, with boundary replication — structurally
very close to Meerkat's hierarchical k-means tree + posting lists + SOAR). It
adds **in-place incremental updates** via **LIRE (Lightweight Incremental
REbalancing)**:

- **Insert**: append the vector to its nearest posting list(s) in place.
- **Delete**: record in a deletion map (tombstone); garbage-collect later.
- **Split**: when a posting list exceeds a max size, split it via local k-means
  (`k = 2`), producing two new centroids that replace the old one in the
  centroid index; the list's vectors are reassigned to the two new lists.
- **Merge**: when a posting list drops below a min size, reassign its vectors
  to the nearest neighboring lists and remove the centroid.
- **Reassign (the core idea)**: after a split/merge the partition boundaries
  move, so some vectors now have a different nearest centroid. LIRE re-checks
  and moves **only vectors in the affected lists and their neighbors** — a
  bounded local region, not the whole dataset — maintaining the *Nearest
  Partition Assignment* (NPA) invariant approximately. This locality is what
  makes rebalancing cheap.
- **Async background jobs**: foreground inserts just append; split/merge/
  reassign run on a background job queue, with crash-consistent in-place block
  updates.

Reported result: **stable recall under continuous high-rate updates at
billion scale**, where no-rebalance baselines degrade and global rebuild is far
too expensive.

**Why this fits Meerkat**: the hierarchical k-means tree + posting lists +
SOAR/boundary replication is essentially "SPANN with extras," so LIRE maps
naturally onto it. SPFresh-style rebalancing is also exactly the lever that
would differentiate Meerkat from VectorChord, which has no rebalancing.

**Active follow-up work** (further reading): Quake (adaptive indexing),
"Updatable Balanced Index for stable streaming search," "Incremental IVF Index
Maintenance for Streaming Vector Search," LSM-VEC, DGAI. See references.

## 5. Meerkat-specific challenges

- **FASTSCAN immutability.** FASTSCAN packs codes in 32-vector SIMD groups, so
  appending one vector means a partial group / repack. Inserts cannot cheaply
  append to packed pages. → Need an **append-friendly write region** (unpacked
  AoS, the existing `MktPostingEntryHeader`) separate from the packed base;
  background compaction repacks it into FASTSCAN groups.
- **Mutable hierarchical centroid tree.** Routing (beam search) works unchanged
  for inserts, but split/merge must add/remove **leaf centroids** and keep the
  multi-level tree and the in-memory centroid cache consistent — a leaf split
  can cascade to a parent. This is the hardest part of the SPFresh phase.
- **Centroid-relative RaBitQ.** A vector encoded relative to centroid A can't
  be reused if reassigned to centroid B — re-encode (recompute `f_add` /
  `f_rescale`) on reassign.
- **SOAR / boundary replication on insert.** To preserve the recall benefit, an
  inserted vector should also get secondary (boundary) assignment, i.e. land in
  more than one posting; the scan-time top-k dedup must cover base + buffer.
- **In-memory centroid cache coherence.** The cache is built at index open;
  split/merge must invalidate/update it across all backends.
- **Dead-tuple over-fetch.** Top-k ANN must over-fetch and filter
  tombstoned/invisible entries to still return k live results.
- **WAL / crash safety.** `GenericXLog` full-image writes work but are heavy
  per page; background rebalancing jobs must be crash-consistent (idempotent /
  restartable).
- **Concurrency.** Concurrent inserts to a posting need buffer locks;
  background rebalancing vs foreground scans/inserts needs coordination
  (per-cluster locks, versioning).
- **Standalone parity.** Routing / encode / append / rebalance **primitives**
  should live in shared `src/index/` so the standalone build can test them;
  `aminsert` and vacuum are the PostgreSQL-specific glue.

## 6. Phased design

### Phase 0 — Correctness: index inserts + handle deletes (no rebalancing)

Stop silently dropping inserts; handle deletes via MVCC. Parity with
ivfflat/vchordrq plus correct delete behavior.

- **`aminsert`**:
  1. If `index_unchanged` (vector not changed by an `UPDATE`) → rely on HOT,
     return without work.
  2. Route via `mkt_centroid_beam_search` to the nearest leaf posting(s)
     (a small insert-time nprobe; optionally + boundary lists for SOAR).
  3. RaBitQ-encode relative to the chosen centroid(s).
  4. Append to the cluster's **write buffer** — an append-friendly AoS overflow
     page chain (`MktPostingEntryHeader`), new pages via `new_page` +
     `GenericXLog`, linked by `next_blkno`.
- **Scan**: merge the immutable FASTSCAN/RaBitQ base + the AoS write buffer for
  each probed cluster (score both into the same top-k).
- **Deletes**:
  - `kill_prior_tuple` → set `MKT_POSTING_FLAG_DELETED` lazily on entries the
    executor reports dead.
  - `ambulkdelete` → mark dead TIDs deleted across base + buffer during
    `VACUUM`.
  - Scan skips `DELETED` entries and over-fetches to refill k.
- **Accuracy**: centroids fixed → drift; document `REINDEX CONCURRENTLY`
  guidance, as ivfflat does.

This alone moves Meerkat from "build-only" to "mutable with rebuild."

### Phase 1 — Compaction: memtable → FASTSCAN segments (LSM-ish)

Keep the fast path fast and bound the write buffer.

- Treat the per-cluster AoS write buffer as a **memtable**. Background (or
  vacuum-time) compaction repacks accumulated entries into FASTSCAN-packed
  **segment** pages and garbage-collects tombstones.
- This mirrors `pg_textsearch`'s memtable/segment design — reuse the
  segment/merge machinery and lessons.
- Search merges base segments + memtable; fewer packed segments keep scan fast.
- Still fixed centroids (drift), but no unbounded buffer growth.

### Phase 2 — LIRE: incremental rebalancing (the SPFresh target)

Stable recall under heavy mutation **without** rebuild.

- Track posting sizes; **split** when `> max` (local k-means `k = 2`, two new
  leaf centroids, update tree + cache, re-encode + reassign), **merge** when
  `< min` (reassign to neighbors, remove centroid).
- **Bounded local reassign** to maintain NPA — re-check only affected +
  neighboring postings.
- **Background job queue** decoupled from foreground inserts; crash-consistent,
  idempotent jobs.
- Hard parts: mutable hierarchical centroid tree (split a leaf, possible parent
  cascade), in-memory cache coherence across backends, RaBitQ re-encode, SOAR
  re-replication, WAL.
- Outcome: recall stays stable; periodic `REINDEX` no longer required.

## 7. Key decisions and open questions

- **Write-buffer location/format**: per-cluster on-disk AoS overflow chain
  (durable, MVCC-natural, reuses the storage API — recommended) vs a shared
  in-memory/shmem memtable (faster, needs flush + crash handling).
- **Insert routing fan-out**: insert-time nprobe, and whether to SOAR-replicate
  inserts in Phase 0 (recall vs write amplification).
- **Over-fetch factor** for dead-tuple filtering in top-k.
- **Tombstone GC cadence**: lazy `kill_prior_tuple` + `VACUUM` + compaction.
- **Centroid-tree mutability representation** (Phase 2): how to add/remove
  leaves in the on-page tree and the cache without a rebuild; cascade handling.
- **Concurrency model**: per-cluster locks; background-job vs foreground
  coordination; standalone (threads) vs PostgreSQL (processes + shmem) parity.
- **Crash consistency** of rebalancing jobs.

## 8. Recommended path

1. **Ship Phase 0 first** — correctness. Inserts get indexed, deletes are
   handled, drift is documented with `REINDEX` guidance. This alone closes the
   "build-only" gap and reaches parity with ivfflat/vchordrq.
2. **Phase 1 (compaction)** to stay performant under sustained inserts.
3. **Phase 2 (LIRE)** is the differentiator: stable recall without rebuild, the
   SPFresh end goal, and the capability VectorChord lacks.

## 9. References

- SPFresh: Incremental In-Place Update for Billion-Scale Vector Search,
  SOSP '23 — <https://dl.acm.org/doi/10.1145/3600006.3613166>
- SPANN: Highly-efficient Billion-scale Approximate Nearest Neighbor Search,
  NeurIPS '21 (the index SPFresh updates).
- Incremental IVF Index Maintenance for Streaming Vector Search —
  <https://arxiv.org/abs/2411.00970>
- Quake: Adaptive Indexing for Vector Search —
  <https://arxiv.org/abs/2506.03437>
- Updatable Balanced Index for Stable Streaming Similarity Search —
  <https://arxiv.org/abs/2602.00563>
- LSM-VEC: A Large-Scale Disk-Based System for Dynamic Vector Search —
  <https://arxiv.org/abs/2505.17152>
- DGAI: Decoupled On-Disk Graph-Based ANN Index for Efficient Updates —
  <https://arxiv.org/abs/2510.25401>
- PostgreSQL: HOT updates, Index Access Method interface (`aminsert`,
  `ambulkdelete`, `amvacuumcleanup`), `kill_prior_tuple` / LP_DEAD hints,
  `REINDEX CONCURRENTLY`.
- pgvector `ivfflat` / `hnsw` insert behavior; VectorChord `vchordrq` insert
  (append-to-nearest-leaf, fixed centroids).
