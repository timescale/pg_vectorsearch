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
  (`src/pg/iam_handler.c`) just `return false;`. The `bool` result is only
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

### Phase 1 — Compaction: write-buffer pages → FASTSCAN segments (LSM-ish)

Keep the fast path fast and bound the write buffer.

- The per-cluster AoS write buffer (Phase 0) is the mutable "write tier." It is
  **on-disk overflow pages in the index relation**, *not* a separate in-memory
  or shared-memory structure — PostgreSQL's `shared_buffers` is the in-memory
  cache for them, and `GenericXLog` provides durability + crash recovery for
  free.
- Compaction repacks accumulated write-tier entries into FASTSCAN-packed
  **segment** pages and garbage-collects tombstones. It must be **online and
  crash-safe**: build the new segment fully, flip the posting-chain pointer in
  one WAL-logged step, then free the old pages — never expose a half-converted
  chain. **No dedicated background worker is needed at this phase**; three
  complementary triggers (mirroring GIN's pending list) drive it:
  - **Vacuum hooks** (`ambulkdelete` / `amvacuumcleanup`) — the steady-state +
    GC path, auto-scheduled by autovacuum on table churn. Idiomatic: GIN flushes
    its pending list into the main index during vacuum, and these hooks can read
    AoS write-tier pages and write FASTSCAN segments directly.
  - **Inline overflow** — when a cluster's write buffer crosses a size
    threshold, the inserting backend (or next scan) repacks it then and there.
    Vacuum fires on table *dead-tuple* thresholds, the wrong signal for an
    insert-grown buffer, so this keeps insert-heavy workloads from accumulating
    unpacked data between vacuums (GIN's inline pending-list flush).
  - **Manual procedure** — a user-callable `mkt.compact(index)` to force
    compaction on demand (GIN's `gin_clean_pending_list()`).
- Search merges base + segments + write tier; fewer packed segments keep scan
  fast.
- Still fixed centroids (drift), but no unbounded write-tier growth.

The write-tier + immutable-segment + background-compaction structure is just
standard **LSM / Lucene-segment tiering** — nothing novel. The one real choice
is the write tier's substrate:

- **On-disk write-buffer pages (recommended)**: durability, MVCC, and
  cross-backend visibility come for free from the buffer manager + WAL; no
  shared-memory sizing or memtable crash-recovery to build.
- **Shared-memory (DSA) memtable**: faster in-memory appends, but you must
  build a WAL/crash-recovery story for the in-memory entries plus shared-memory
  sizing.

Start with on-disk pages; a DSA memtable is a later optimization only if insert
throughput demands it.

### Phase 2 — LIRE: incremental rebalancing (the SPFresh target)

Stable recall under heavy mutation **without** rebuild.

- Track posting sizes; **split** when a list outgrows the trigger (local
  k-means, k new leaf centroids, update tree + cache, re-encode + reassign),
  **merge** when it falls below the merge threshold (reassign to neighbors,
  remove centroid).

#### Sizing: one target, a band around it

Three sizes, derived from one:

| name | value | role |
| --- | --- | --- |
| target `T` | `rows / nlist` — see `mkt_target_entries_per_list` | the size a list rests at |
| split trigger | `T * MKT_SPLIT_TRIGGER_FACTOR` (2) | grow past this and the list splits |
| merge threshold | `T / MKT_SPLIT_TRIGGER_FACTOR` | shrink below this and the list merges |

The target is **not a free parameter**: it is the size the build chose, since
`mkt_auto_nlist` targets `MKT_TARGET_ENTRIES_PER_LIST` vectors per list (above
its sqrt-floor crossover; below it, `sqrt(rows)`). Both the automatic probe
count and the cost model are derived from `nlist`, and the per-list size is its
inverse — so aiming maintenance at a different size silently decouples the index
from both. `mkt.rebalance()` therefore derives its target by default, honouring
an explicit `nlist` reloption when the index was built with one.

The **trigger is deliberately above the target**, which is the part that is easy
to get wrong. Pin the trigger at the target and every freshly split list starts
exactly on it — one insert from splitting again. The gap is what amortizes the
split over the inserts that fill it, and with the merge threshold below the
target by the same factor the target sits at the geometric centre of the band:
room to absorb inserts and deletes, and room for the unevenness of k-means
splits and of merges and reassignment.

**Width is `round(count / T)`, not `ceil`.** At the trigger the ratio is a shade
over the factor, so rounding up would ask for one partition more than the
entries justify and land every one of them *below* the target. Rounding gives
exactly `factor` parts there — a bisection at factor 2, as the paper does — and
keeps parts within `[0.75, 1.25]` of the target elsewhere. A wider split is
therefore only what a batch pass does when it meets a list that has been
neglected; `MKT_SPLIT_MAX_PARTS` bounds that catch-up case.

The size is re-checked **after** the entries are collected, not just from the
head's live count: collection drops entries whose vector can no longer be
fetched, and those are not reflected in the live count, so a list can look
oversized and turn out not to be. LIRE does the same (garbage-collect,
re-verify against the split limit, complete without splitting if it now fits).

#### Transactions: maintenance owns its own

Neither entry point runs inside a caller's transaction — the same restriction
VACUUM has, in the form a procedure has available (a top-level `CALL` gets a
non-atomic call context; a transaction block, a function, or a block that opens
a subtransaction gets an atomic one).

Two reasons, and the second is the load-bearing one:

- These procedures reorganize the index, not the data it points at, and their
  page writes are not transactional. A `ROLLBACK` would leave the lists split,
  the old chains retired and the leaf count raised, while discarding the
  relcache invalidation that tells other backends the leaf count moved — so
  other sessions would keep clamping their probes to the pre-split count until
  something else invalidated them. Offering a rollback that rolls nothing back
  is worse than refusing.
- A pass that commits as it goes cannot commit at all from inside someone
  else's transaction. Splitting each list in its own transaction — bounding the
  transaction of a long pass, and letting it reclaim what its own earlier
  splits retired — is the reason a procedure was the right shape here.

#### Memory: a split streams its list

A split cannot hold the list it is splitting. At `entries * dim * 4` bytes, the
tool for splitting an oversized list would fail in proportion to how oversized
the list is — and a list that has outgrown the trigger by a long way, because
maintenance has not run, is exactly the one that has to be splittable. The same
reasoning the bulk build applies to its sample region.

So the list is streamed twice:

1. **Count and sample.** One pass counts the fetchable entries — which is also
   the count the post-collection size re-check needs, since the head's
   `live_count` does not know about entries whose vector can no longer be
   fetched — and reservoir-samples them into a buffer whose size comes from
   `MktSplitConfig.sample_budget_bytes` (the PostgreSQL layer passes
   `maintenance_work_mem`). A reservoir rather than every n'th entry, because a
   stride needs an estimate of the list's length and an estimate that came out
   low would fill the sample before the chain ended, leaving the tail — the most
   recently inserted entries — unsampled. The head's `live_count` only sizes the
   initial buffer, where being wrong costs a realloc.
2. **Cluster the sample**, and drop the clusters holding too few entries to be
   worth a posting list of their own — which means dropping their centroids,
   before anything is assigned to them.
3. **Assign and write.** A second pass sends each entry to the list whose
   surviving centroid is nearest, appending to `k` page builders. Each builder
   assembles pages in its own memory and touches storage only to flush a full
   one, so `k` of them cost `k` page images rather than `k` pinned buffers.

Private memory is then the sample plus `O(k * dim)` and `O(k * BLCKSZ)`,
whatever the list holds. What remains proportional to the list is the cost of
*reading* it: touching a list's worth of heap and TOAST pages maps them into
the backend, which no arrangement of the split can avoid.

The sample size comes from a peak model rather than from dividing the budget by
the vector size (`mkt_split_sample_cap`), because clustering costs more than
the sample it clusters: per point, an assignment, an L2 norm, an initialisation
distance, the result's own copy of the assignments, and Elkan's bounds — one per
point plus **one per point per centroid**. At a wide dimension that is a few
percent of the point; at a narrow one it is several times the point. The model
reserves the per-centroid part for the widest split, since the width is not
known until the list has been counted, along with the centroid sets and
k-means' blocked scratch.

Two ceilings sit above the budget. The sample's own allocation is capped
(`MKT_SPLIT_MAX_SAMPLE_BYTES`) below the largest single allocation a backend
permits, so a generous `maintenance_work_mem` cannot turn into a failed
allocation — and the buffer is sized to `min(list, budget)` and grown if
needed, so a generous budget does not cost generous memory on a list that does
not fill it. Below the budget, a split that cannot afford even two partitions'
worth of sample is refused with the shortfall named, rather than quietly
exceeding what it was given.

A list that fits the budget is sampled whole, so the sizes a split normally
meets behave exactly as clustering the list directly would. Above the budget,
two things change: cluster sizes come from scaled sample tallies rather than
exact counts, and the width is capped at what the sample can speak for
(`MKT_SPLIT_MIN_SAMPLE_PER_PART`) so a list too big for its budget is split
less far per pass rather than split badly.

Note that entries are assigned to the nearest of the centroids actually stored,
not to k-means' own final assignments — which are one iteration stale, since
the algorithm returns after updating centroids. That is the assignment queries
will reproduce at scan time.

#### Sizing: the floor, and why a split can decline

A partition of one is worse than no partition: a page holding one entry, a
centroid that routes a single vector, and an `nlist` inflated by lists holding
almost no data. k-means minimizes distortion, not partition size, so an outlier
comes out as its own cluster — and unlike LIRE, which splits with SPANN's
multi-constraint balanced clustering, we have no size constraint to prevent it
(see "Known divergences" below).

So any cluster below the merge threshold is **folded** into the nearest
surviving one, each entry to its own nearest surviving centroid. Surviving
centroids are deliberately not recomputed: dragging one toward a few absorbed
outliers would encode the many worse to encode the few better, and the resulting
NPA drift is what reassign is for.

If folding cannot leave two partitions standing, the split **widens** by one
partition and retries (`MKT_SPLIT_WIDEN_ATTEMPTS`) before declining. At the
bisection width a dense region plus a straggler has no second partition clearing
the floor; the k-means seed is fixed, so declining there would decline
identically on every later pass and leave a list above the trigger forever. One
more partition usually resolves it. Data that resists it — identical vectors,
say — is declined, which is the right answer.

#### Known divergences from LIRE

- **No balanced clustering.** LIRE splits with SPANN's multi-constraint balanced
  clustering, so its parts land *on* the target; ours scatter around it. This is
  index-wide, not split-local: the split calls the same k-means the build's
  hierarchical clustering calls at every node, and changing it only in the split
  would break the invariant that a split matches a from-scratch rebuild. The
  property it protects is tail latency, which our throughput-at-recall
  benchmarks do not measure — so measure before changing.
- **Split and merge can feed each other.** An unbalanced split emits a part
  under the merge threshold, merge folds it into a neighbour and pushes that over
  the split trigger. The fold above closes the split half; the merge half needs
  merge to refuse a result above the trigger, and lands with merge. Note that
  SPFresh's convergence proof does not cover this: it rests on splits
  monotonically growing the centroid set, which merge breaks.
- **Bounded local reassign** to maintain NPA — re-check only affected +
  neighboring postings.
- **Async execution** — a **dedicated background worker** (a PG dynamic
  bgworker consuming a shared job queue) runs the size-triggered **split**:
  vacuum's dead-tuple trigger is the wrong signal for an insert-grown posting,
  and split is too heavy for foreground `aminsert`. **Merge and tombstone-GC can
  stay on the vacuum hooks** (delete-driven shrink fits the dead-tuple model).
  Jobs are crash-consistent and idempotent.
- Hard parts: mutable hierarchical centroid tree (split a leaf, possible parent
  cascade), in-memory cache coherence across backends, RaBitQ re-encode, SOAR
  re-replication, WAL.
- Outcome: recall stays stable; periodic `REINDEX` no longer required.

### Deletes: tombstone lifecycle and the over-fetch problem

Deletes are first-class and load-bearing: since updates reduce to insert +
delete (§3), the delete path serves both explicit `DELETE`s and the
old-version cleanup of every vector-column `UPDATE`.

**Correctness vs optimization.** PostgreSQL's MVCC is the correctness backstop —
the index may return a dead/invisible TID and the executor's visibility recheck
against the heap filters it. So index-level deletion is an *optimization*
(skip known-dead entries to avoid wasting top-k slots and heap fetches), not a
correctness requirement. The index must never physically remove an entry just
because a `DELETE` ran; removal is safe only once VACUUM finds the tuple dead to
all snapshots.

**Tombstone lifecycle (three stages):**

1. **Lazy mark** (`kill_prior_tuple` / LP_DEAD-style): when a scan returns a TID
   the executor finds dead-to-everyone, it sets `scan->kill_prior_tuple`; the AM
   marks that entry (`MKT_POSTING_FLAG_DELETED` for AoS, a per-group deletion
   bit for FASTSCAN) so later scans skip it. No vacuum needed; near-free.
2. **Bulk mark** (`ambulkdelete`): VACUUM calls it with a "is this TID dead?"
   callback; the AM tombstones all matching entries across base + segments +
   write tier.
3. **Physical reclaim**: tombstoned entries are dropped only when the
   page/segment is rewritten — during Phase 1 compaction or Phase 2 split/merge.

**Tombstoning inside immutable FASTSCAN groups.** A FASTSCAN posting packs 32
vectors per SIMD group, so an entry can't be removed in place. Mark deletes in a
**per-group (or per-segment) deletion bitmap** (one bit per entry), consulted at
scan time to mask dead lanes; physical removal happens at compaction. The AoS
write tier can use the entry flag directly (the currently-unused
`MKT_POSTING_FLAG_DELETED` is the starting point).

**Page-level tombstones (implemented).** When VACUUM leaves an entire page dead
— common for bulk/range deletes (`DELETE FROM t`, `DELETE ... WHERE id BETWEEN
...`) that wipe whole pages or clusters — the page is flagged
`MKT_POSTING_PAGE_TOMBSTONED` and the scan skips its scoring kernel entirely
(both AoS and FASTSCAN), only following the chain past it. This is also the
granularity at which FASTSCAN deletes get recorded at all (packed groups can't
be flagged per entry), detected by testing every group TID with an early exit
on the first live one. The page stays linked in the chain — this is a scan
optimization, not reclamation.

**Page reclamation (future).** The page tombstone is the prerequisite for
reusing the space; two options, in increasing cost/power:

- *In-chain reuse on insert (cluster-local).* A new insert reuses a tombstoned
  page in place (clear the flag, overwrite). This needs **none** of btree's
  page-recycle machinery, precisely because the page never leaves its cluster's
  chain: VACUUM only tombstones entries dead to *every* snapshot, so
  overwriting loses nothing a reader needs, and a concurrent scanner that lands
  on the page mid-reuse just sees valid same-cluster candidates (MVCC rechecks
  them) — there's no "repurposed into a different key range" hazard. The only
  cost is *finding* a reusable page; a per-cluster write cursor / first-free
  hint in the head opaque keeps the common path O(1) (the bounded hunt runs
  only when the current write page fills, the same cadence at which append
  already allocates). Limitation: reuse is cluster-local, so under skew (some
  clusters shrink while others grow) dead pages stay stranded in the shrunk
  clusters.
- *Unlink + FSM (global).* Unlink an all-dead page (`prev->next_blkno =
  page->next_blkno`) and record it in the Free Space Map for *any* cluster to
  reuse via `GetFreeIndexPage`. This is the standard index reclaim path and
  bounds bloat globally, but it inherits btree's page-deletion complexity: a
  freed page can't be recycled until no snapshot could still be mid-scan
  through it (a deletion-XID / `GlobalVisCheckRemovableXid` gate), head pages
  can't be unlinked (the centroid tree references them by block number), and
  the FSM is only a hint so reuse must re-verify under lock. Best done within
  the compaction phase, which already has the VACUUM/visibility context.

Neither returns space to the OS without a trailing-page truncation pass; both
bound growth by reuse. The cluster-local scheme is the simpler first step; FSM
is the heavier follow-on for cross-cluster balance.

**Returning k live results.** IVF scans the *full* posting lists of the nprobe
nearest centroids, so the candidate pool is normally far larger than k; dead and
MVCC-invisible entries are filtered during top-k selection at no extra scan
cost, and k live results still come back unless a large fraction of the scanned
lists is dead. A pre-sized "over-fetch factor" is the wrong tool.

- **Phase 0 stance (simplest):** scan, filter, return top-k live. Accept that an
  extreme local dead ratio can transiently return fewer/worse results, and rely
  on GC — compaction (Phase 1), LIRE merge (Phase 2), or a rebuild — to keep the
  dead ratio low and the degradation self-healing. This is approximate search,
  so a slightly-short top-k between GC passes is acceptable (and matches how
  ivfflat behaves with stale data). The trade-off is a *silent* quality dip, not
  slower-but-correct, so document it; GC cadence (autovacuum + compaction) is the
  lever that bounds it. The pathological case is a burst of deletes on a hot
  cluster between vacuums, which self-heals.
- **Optional refinement:** for workloads that can't tolerate the transient dip, a
  **resumable scan** with a live-result counter that **expands nprobe** (descends
  to the next-nearest centroids) when the current lists are exhausted before k
  live results — strictly better than pgvector ivfflat/hnsw, which don't
  auto-expand. (Meerkat's current scan materializes a fixed top-k up front in
  `execute_search`; this refinement makes it resumable/expandable.)

Either way, index-level tombstones are an *optimization* — skip known-dead
entries before the exact-distance rerank and heap fetch — not a correctness
mechanism; MVCC's visibility recheck is the backstop.

**GC cadence and recall.** Accumulated tombstones inflate scan work (more
candidates scanned per live result) and degrade effective QPS, so reclamation
cadence matters: lazy marking keeps scans correct, but VACUUM + compaction must
run often enough to bound the dead ratio. In Phase 2, heavy deletes also shrink
postings below the min size and trigger LIRE **merge**, reclaiming space and
keeping the partitioning balanced.

### Phase dependencies: are 0/1 stepping stones to LIRE?

Mostly stepping stones, not detours. LIRE reuses the bulk of Phase 0/1 and adds
one genuinely new piece.

**Carried forward to LIRE (built once, in Phase 0/1):**

- Mutable posting storage + the `aminsert` route → encode → append path; LIRE
  still appends new vectors to their nearest postings.
- Delete / tombstone / GC (`kill_prior_tuple`, `ambulkdelete`).
- Multi-tier scan + over-fetch + dead-tuple filtering.
- The background-job + compaction machinery — a split is "recluster a cluster's
  vectors into k new postings" and a compaction is "rewrite a cluster's
  postings into packed segments": the same rewrite-postings primitive.
- The background-rewrite-vs-foreground-scan concurrency model and
  crash-consistent incremental WAL writes — the hardest concurrency hazard
  (compaction vs scan, half-written segments) is solved here, *before* centroids
  also start moving.

**Net-new in Phase 2 (LIRE's hard ~20-30%):**

- A mutable centroid tree (add/remove leaf centroids on split/merge, keep the
  multi-level tree + in-memory cache coherent, handle parent cascade) and the
  bounded-reassign / NPA logic that rides on it (plus RaBitQ re-encode). Phases
  0/1 deliberately assume fixed centroids and do not advance this — it is where
  the real risk lives.

**Is there a shortcut straight to LIRE?** No. LIRE requires essentially all of
Phase 0 and most of Phase 1's machinery regardless, so jumping ahead does not
skip building them — it only skips *shipping* them as milestones. The only
throwaway is the fixed-centroid + `REINDEX` accuracy posture, which is
documentation, not code. Shipping 0/1 first delivers immediate value (parity
with ivfflat/vchordrq, closing the build-only gap) and de-risks the concurrency
foundation before the centroid-mutability work.

**Build Phase 0/1 "LIRE-aware"** so they are stepping stones, not side quests:

1. Track posting sizes from Phase 0 (LIRE's split/merge triggers).
2. Make compaction a generalizable "rewrite a cluster's vectors into N
   postings" primitive (plain compaction = N=1 repack; split = N=2 with new
   centroids).
3. Settle the background-rebalance-vs-scan concurrency model once, in Phase 1.
4. Keep route / encode / append / rewrite as shared `src/index/` primitives so
   the standalone build can test them.

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
