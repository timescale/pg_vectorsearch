# Parallel Index Build for Meerkat

## Context

Meerkat's index build is single-threaded. For 10M vectors with nlist=16000 +
SOAR, the build takes ~4 hours. The bottleneck is the heap scan phase where
each vector is: tree-descended to find its cluster, RaBitQ-encoded, optionally
SOAR-replicated, and streamed to a posting builder. This is embarrassingly
parallel per-vector but currently runs on a single core.

The goal: parallelize the build so it scales with available cores, working in
both standalone (threadpool) and PostgreSQL (PG parallel workers) contexts.
The core parallel logic lives in shared code (`src/index/`) with no PG
dependencies — only the dispatch layer differs between standalone and PG.

## Target: meerkat repo at ~/ClaudeWorkspace/meerkat/

## What to Parallelize

The build has 10 phases. Only some benefit from parallelism:

| Phase | Current | Parallel? | Why |
|-------|---------|-----------|-----|
| 1. Determine dimension | instant | no | trivial |
| 2. Resolve params | instant | no | trivial |
| 3. Sample vectors | fast | no | small sample |
| 4. K-means clustering | moderate | yes (CBLAS/threads) | matrix ops parallelize well |
| 5. Compute global mean | fast | no | trivial |
| 6. Reserve pages | fast | no | metadata only |
| **7. Heap scan + assign + encode + write** | **bottleneck** | **yes** | **per-vector, embarrassingly parallel** |
| 8. Finish builders (merge partial pages) | moderate | no | coordinator merges tail pages |
| 9. Update metadata | instant | no | single write |
| 10. WAL-log | moderate | no | sequential by nature |

Phase 7 is 90%+ of the build time. Parallelizing it gives near-linear speedup.

## Architecture: Two-Layer Parallelism

### Layer 1: Core parallel work functions (shared code, `src/index/`)

Thread-safe functions that process vectors and write posting pages. No PG
dependencies — works identically in standalone and PG contexts.

```c
// Per-worker state: tree descent + encode + write to claimed pages
typedef struct {
    uint32 worker_id;
    uint32 nworkers;
    const HKMeansResult *tree;       // read-only, shared
    const float *global_mean;        // read-only, shared
    const MktIndexConfig *config;    // read-only, shared
    MktPostingBuilder *builders;     // per-cluster builders with interleaved pages
} MktBuildWorkerState;

void mkt_build_worker_init(MktBuildWorkerState *state, ...);
void mkt_build_worker_process_vector(MktBuildWorkerState *state,
                                     const float *vec, int64 tid);
void mkt_build_worker_finish(MktBuildWorkerState *state);
```

### Layer 2: Dispatch (platform-specific)

**Standalone (pthreads)** — `src/standalone/`:
```c
typedef struct {
    pthread_t *threads;
    uint32 nthreads;
} MktThreadPool;
```

**PostgreSQL (parallel workers)** — `src/pg/`:
Each PG parallel worker runs a heap scan portion via `ParallelTableScanDesc`,
calling the same shared `mkt_build_worker_process_vector()` function.

## Key Design: Interleaved Page Claiming

Instead of buffering encoded entries and merging later, workers write
posting pages directly during the scan with zero contention via
**interleaved page reservation**.

### How It Works

For each cluster's posting list, blocks are reserved contiguously at build
start (existing behavior). Workers claim pages in round-robin order:

```
Cluster C, 12 reserved pages, 4 workers:

Worker 0: pages 0, 4, 8
Worker 1: pages 1, 5, 9
Worker 2: pages 2, 6, 10
Worker 3: pages 3, 7, 11
```

Each worker writes ONLY to its claimed pages. When a worker fills a page
for cluster C, it moves to its next claimed page for that cluster. No
locking needed — each page has exactly one writer.

### Page Chain Order

The `next_blkno` chain links pages sequentially by block number:
page 0 → 1 → 2 → 3 → 4 → ... This is correct regardless of which
worker wrote which page, since the read order follows block number.

### Handling Under-Reservation

If the initial page reservation is too small (more vectors than estimated),
workers that exhaust their reserved pages must allocate on-demand:

1. Worker tries to claim its next interleaved page
2. If no more reserved pages → allocate a new page via storage extend
3. The on-demand page won't be contiguous with the reserved range
4. Update the page chain: last reserved page's `next_blkno` → new page
5. The posting list remains correct (chain is followed), just not fully
   sequential on disk for the tail pages

This requires a lightweight lock on the per-cluster chain tail pointer,
but it's rare (only when reservation is exhausted) and cheap.

### Partial Page Merge (Tail Handling)

After the scan completes, each worker holds at most one partial (non-full)
page per cluster. With N workers and K clusters, there are up to N×K
partial pages. The coordinator:

1. Collects partial pages from all workers for each cluster
2. Packs entries from multiple partial pages into minimum full pages
3. Writes the consolidated pages to remaining reserved blocks (or
   allocates new blocks if needed)
4. Updates the chain terminator (`next_blkno = InvalidBlockNumber`)

This avoids wasting space from N partially-filled last pages per cluster.
The merge is fast since entries are already encoded — just memcpy into
new page layouts.

### Per-Worker Posting Builder

Each worker has its own set of per-cluster posting builders:

```c
typedef struct {
    uint32 nlist;
    uint32 worker_id;
    uint32 nworkers;
    struct {
        BlockNumber *reserved_pages;  // this worker's pages for this cluster
        uint32 n_reserved;
        uint32 next_page_idx;         // next page to write
        uint8 *mem_page;              // single in-memory page buffer
        uint32 entries_in_page;       // current page fill level
    } *clusters;
} MktParallelPostingBuilder;
```

The interleaving is computed at init time:
```c
for (uint32 c = 0; c < nlist; c++) {
    uint32 total_pages = cluster_reserved_pages[c];
    // This worker gets pages: worker_id, worker_id + nworkers, ...
    for (uint32 p = worker_id; p < total_pages; p += nworkers) {
        builder->clusters[c].reserved_pages[n++] = reserve_start[c] + p;
    }
}
```

## Implementation Plan

### Phase 1: Refactor build into worker-processable functions

**Files**: `src/pg/mktann_build.c`, new `src/index/build_parallel.c`,
new `src/index/build_parallel.h`

1. Extract per-vector work from `build_callback()` into standalone
   `mkt_build_encode_and_assign()` — takes vector + tree + config,
   returns cluster ID + encoded RaBitQData + optional SOAR secondary.

2. Create `MktParallelPostingBuilder` with interleaved page claiming.
   Init function takes (worker_id, nworkers, reserved_pages_per_cluster).

3. Create `mkt_parallel_builder_add()` — encodes vector and writes to
   the worker's claimed page for that cluster. When page is full, flush
   to storage and advance to next claimed page.

4. Create `mkt_parallel_builder_finish()` — returns partial pages for
   coordinator merge.

5. Create `mkt_parallel_merge_tails()` — coordinator consolidates
   partial pages across all workers.

### Phase 2: Standalone threadpool

**Files**: new `src/util/threadpool.c`, new `src/util/threadpool.h`

1. Simple pthreads threadpool with static partitioning (each thread gets
   a vector range).

2. In standalone build path: after k-means, create N worker states,
   partition the vector source into N ranges, each thread processes its
   range using `mkt_build_encode_and_assign()` + parallel builder.

3. After all threads join, coordinator merges tail pages.

4. Configurable thread count via CLI (`--threads N`), default to
   hardware concurrency.

### Phase 3: PG parallel workers

**Files**: `src/pg/mktann_build.c`, `src/pg/mktann.c`

1. Implement `amestimateparallelscan` and `aminitparallelscan` callbacks.

2. Each PG worker:
   - Attaches to shared memory (tree, config, global mean — read-only)
   - Gets assigned worker_id from shared counter
   - Creates local MktParallelPostingBuilder with interleaved pages
   - Runs parallel heap scan, calling `mkt_build_encode_and_assign()`
     for each tuple, writing to its claimed pages
   - On completion, returns partial page info to shared memory

3. Leader:
   - Launches workers via `CreateParallelContext()`
   - Also participates in scanning (worker_id = 0)
   - After all workers complete, runs `mkt_parallel_merge_tails()`
   - Writes centroid pages and updates metadata

4. Shared memory layout:
   ```
   MktBuildShared:
     - heaprelid, indexrelid
     - HKMeansResult (centroid tree, read-only after k-means)
     - global_mean vector (read-only)
     - MktIndexConfig (read-only)
     - ParallelTableScanDesc (for coordinated heap scan)
     - per-cluster reserved page ranges (read-only)
     - ConditionVariable workersdonecv
     - slock_t mutex
     - atomic worker_id_counter (for worker_id assignment)
     - nparticipantsdone
     - reltuples (accumulated)
     - per-worker partial page info (written by workers, read by leader)
   ```

### Phase 4: K-means parallelism (optional, lower priority)

1. CBLAS `sgemm` for distance matrix (already supported, uses OpenMP).
2. Parallelize Lloyd's assignment step with the same threadpool.

## Memory Considerations

- Each worker needs: 1 page buffer (8KB) per cluster for the current page
  being filled. With 16K clusters: ~128MB per worker.
- Reserved page arrays: small (just block numbers).
- The centroid tree and global mean are read-only shared — no duplication.
- For PG: tree/config are in shared memory (DSM), posting builders are in
  per-worker local memory.
- Scale `maintenance_work_mem` with worker count for PG.

## Expected Speedup

- Phase 7 (heap scan + encode + write) is ~90% of build time
- With N workers, expect ~0.8-0.9 × N speedup on that phase
  (slightly less than linear due to tail merge + page reservation overhead)
- With 8 cores: ~6-7x speedup on bottleneck → ~4-5x total
- 4-hour build → ~50-60 min

## Determinism

The parallel build is **not** bit-reproducible, and a parallel-built index
does not match a serial-built one cluster-for-cluster. This is expected — see
the sources below — and is not a correctness problem: recall is statistically
equivalent, just not identical.

What stays deterministic (given the tree):
- Tree descent — a vector's cluster assignment is fixed once the tree exists.
- RaBitQ encoding — deterministic given the seed.

What varies — all rooted in the **work-stealing heap scan**, where which worker
reads which blocks depends on runtime timing:
- **k-means seeding.** Sampling collects per-worker sample slots via the
  work-stealing scan, so the slot contents (and their concatenation order, from
  which the leader picks the initial centroids) differ run to run. Different
  seeds → k-means converges to a different local optimum → a different tree.
- **Floating-point reduction.** Each k-means iteration sums per-worker partial
  centroid sums; float addition is non-associative, so a different
  vectors-to-workers partition yields slightly different centroids, compounding
  over iterations.

A serial build (`max_parallel_maintenance_workers = 0`) is fully deterministic:
fixed scan order → fixed samples → fixed seeds → identical tree and recall on
every rebuild. So run-to-run recall variance appears only with workers > 0.

Making parallel builds deterministic would require seeding k-means from a
canonically-ordered sample set (removing the slot-order dependence) and
accumulating the reduction in a fixed order. Not done — the variance is small
and unbiased, and the work-stealing scan is kept for load balancing.

## Verification

1. **Correctness**: Compare index built with 1 worker vs N workers:
   - Tree descent and RaBitQ encoding are deterministic *given the tree*, but
     the tree itself differs (see Determinism above), so do not expect
     identical cluster assignments or page contents.
   - **Equivalent** recall/QPS at the same nprobe is the primary correctness
     test (within the small run-to-run variance), not bit-identical results.

2. **Performance**: Benchmark build time on 10M dataset:
   - 1 thread baseline
   - 2, 4, 8, 16 threads
   - Verify near-linear scaling up to core count
   - Measure tail merge overhead

3. **Edge cases**:
   - Under-reservation: verify on-demand page allocation works
   - Empty clusters: some clusters may get zero vectors from some workers
   - Very small datasets: verify 1-thread fallback works correctly
   - SOAR: secondary assignments may go to any cluster — verify
     interleaved page claiming handles cross-worker cluster writes

4. **Tests**: Unit tests for:
   - `mkt_build_encode_and_assign()` (shared code)
   - `MktParallelPostingBuilder` init/add/finish/merge
   - Page chain correctness after merge
   - On-demand allocation chain correctness
