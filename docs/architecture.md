# TigerANN Architecture

High-level architecture for TigerANN, a PostgreSQL index access method for
approximate nearest neighbor (ANN) vector search.

## Overview

TigerANN uses an inverted index approach inspired by SPANN and ScaNN, where the
vector space is partitioned into clusters. Each cluster has a centroid and a
posting list containing the vectors assigned to that cluster. Search proceeds
in two phases: first find the closest centroids, then scan their posting lists.

```
┌─────────────────────────────────────────────────────────────────┐
│                         Query Vector                            │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Centroid Search (in memory)                  │
│         Find top-k centroids closest to query vector            │
│                    Uses quantized centroids                     │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Posting List Scan (disk/buffer)                │
│     Read posting lists for selected centroids from disk         │
│       Compute distances using quantized vectors first           │
│         Re-rank top candidates with full precision              │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Result Set                               │
│                   Top-k nearest neighbors                       │
└─────────────────────────────────────────────────────────────────┘
```

## Design Principles

1. **Tiered memory hierarchy**: Centroids in fast memory (shared memory cache),
   posting lists in PostgreSQL buffer cache, vectors on disk.

2. **Quantization for speed**: Use scalar/binary quantization for fast
   approximate distance computation, full precision for final re-ranking.

3. **Sequential I/O**: Vectors close in embedding space stored sequentially
   on disk for efficient bulk reads.

4. **SIMD everywhere**: All distance computations vectorized with AVX512/NEON.

5. **PostgreSQL native**: Use buffer cache, WAL, MVCC, and standard IAM APIs.

## Components

### 1. Centroid Cache

Centroids are representative vectors that partition the vector space. Each
centroid defines a cluster, and every indexed vector is assigned to one or more
clusters based on proximity.

**Centroid representation - two approaches:**

| Approach | Description | Used by |
|----------|-------------|---------|
| True centroid | Mean of all vectors in cluster (synthetic) | ScaNN |
| Medoid | Actual vector closest to the mean | SPANN |

*True centroid (mean):*
- Mathematically optimal cluster center, minimizes within-cluster variance
- Synthetic vector—may not exist in the dataset
- Requires dedicated storage for centroid vectors
- Standard k-means approach

*Medoid (closest actual vector):*
- Real data point from the dataset
- Can be stored as a TID pointing to the heap (no separate vector storage)
- More robust to outliers
- If deleted, must find new representative
- Used by SPANN for "more meaningful navigation" with graph indexes

For TigerANN, the **medoid approach** is preferred: the centroid is chosen as
the actual vector closest to the cluster mean. The medoid is computed during
index build.

**Centroid deletion handling**: If the centroid vector is deleted from the
heap, the centroid entry remains in the posting list for navigation purposes.
This means the centroid entry must store the actual vector data (or quantized
version), not just a TID—otherwise the vector would be lost when the heap tuple
is vacuumed. The centroid effectively becomes a "navigation-only" synthetic
vector, similar to the true centroid approach. This is acceptable because
centroids are used for cluster navigation, not returned as query results.

**On-disk storage**: Centroids are stored as the first entry in each posting
list—no separate centroid pages. The metapage contains a directory of posting
list head block numbers; the centroid for cluster i is the first vector on page
`posting_list_head[i]`. This simplifies the storage layout and eliminates
centroid growth issues.

**In-memory cache**: At startup, the index reads the first page of each posting
list, extracts the centroid vector, and builds an optimized shared memory
structure. This cache differs from the buffer cache in important ways:

| Aspect | Buffer Cache | Centroid Cache |
|--------|--------------|----------------|
| Format | Raw PostgreSQL pages | Optimized for SIMD search |
| Access | Pin/unpin, shared locks | Direct memory access |
| Eviction | LRU, can be evicted | Pinned for index lifetime |
| Layout | Page headers, tuple format | Contiguous, aligned vectors |

The cache uses a flat array layout optimized for SIMD distance computation.
For typical centroid counts (<100k), linear scan with SIMD is faster than
tree-based structures due to cache efficiency and lack of branch mispredictions.
For very large centroid sets, a hierarchical or graph-based approach may be
added.

- **Size**: Typically sqrt(N) to N/100 centroids for N vectors
- **Quantization**: Optional 4-bit or 8-bit scalar quantization reduces memory
  footprint and improves cache utilization

**Future optimization**: For very large centroid counts (>100k), a flat SIMD
scan may become a bottleneck. Future versions could build a true in-memory ANN
index over centroids using approaches like:
- [pgvectorscale's streaming disk ANN](https://github.com/timescale/pgvectorscale)
- [SPTAG library](https://github.com/microsoft/SPTAG) (used by SPANN for centroid navigation)

### 2. Posting Lists

A posting list is the set of vectors assigned to a cluster (borrowing
terminology from inverted indexes in text search). Each centroid points to its
posting list, which contains:

- **Vector TIDs**: Pointers to the full-precision vectors in the heap
- **Quantized vectors**: Compressed vector representations for fast approximate
  distance computation during search

**Why buffer cache (not dedicated cache)?** Unlike centroids which are small
and accessed on every query, posting lists are large (the bulk of index data)
and only a subset is accessed per query (based on nprobe). The buffer cache
provides:
- Automatic caching of hot posting lists (frequently accessed clusters)
- Memory sharing across backends
- Standard PostgreSQL page management and WAL logging

**Cluster assignment**: During index build, each vector is assigned to its
nearest centroid. Vectors near cluster boundaries (within some distance
threshold of multiple centroids) are assigned to multiple posting lists to
improve recall. This replication factor is tunable—higher replication improves
recall but increases storage and scan cost.

**Balancing**: Ideally, posting lists should be roughly equal in size for
predictable query latency. The clustering algorithm aims for balanced clusters,
but natural data distribution may cause imbalance. Very large clusters can be
split; very small clusters may be merged or eliminated.

### 3. Vector Storage

Full-precision vectors are stored in the heap of the indexed table—no separate
vector storage. For typical embedding dimensions (768-1536), vectors exceed
PostgreSQL's inline storage threshold and are TOASTed automatically.

**TOAST access overhead**: Reading a TOASTed vector requires multiple I/O steps:
1. TID lookup to find the heap tuple
2. Read heap page, discover vector is TOASTed
3. Index scan on TOAST chunk_id index
4. Read TOAST heap page(s) to retrieve chunks

This multi-step access is expensive. Therefore, the index should minimize heap
access by:
- Using quantized vectors in posting lists for approximate distance computation
- Only accessing the heap for final re-ranking of top candidates (~1% of scanned vectors)
- Tuning `rerank_k` to balance recall vs. heap access cost

**Future exploration**: If full-precision vector access becomes a bottleneck,
alternatives to TOAST storage could be explored (e.g., storing vectors in index
pages, a dedicated vector heap, columnar storage, or a custom TOAST table
access method optimized for vector retrieval).

### 4. Metadata

Index metadata is stored in multiple locations depending on its nature:

**Reloptions** (pg_class.reloptions, specified at CREATE INDEX):
- Build parameters: `nlist` (number of clusters), `fillfactor`
- Search defaults: `nprobe` (clusters to search), `rerank_k`
- Over-allocation: `reserved_pages`

**GUCs** (session/server-level, can override reloptions):
- `tigerann.nprobe` - clusters to search per query
- `tigerann.rerank_k` - candidates to re-rank with full precision

**Catalog tables** (managed by PostgreSQL):
- Structural info in pg_class, pg_index, pg_am, pg_opclass

**Metapage** (stored in index file, persists with the index):
- Posting list directory (head block per cluster)
- Quantization codebook (if using scalar quantization)
- Index version, dimension, distance metric
- Statistics: cluster sizes, total vectors indexed

## Index Operations

### Build

```
1. Sample vectors for clustering (e.g., 10% or fixed sample)
2. Run k-means or hierarchical clustering to find centroids
3. Assign each vector to nearest centroid(s)
4. Build posting lists with quantized vectors
5. Store centroids in metapage, posting lists in data pages
```

**Optimizations**:
- Hierarchical balanced clustering for uniform posting list sizes
- Parallel clustering using multiple workers
- Streaming build to limit memory usage

### Search

```
1. Load centroids into shared memory cache (if not cached)
2. Compute distances from query to all centroids (SIMD)
3. Select top-k centroids (k = nprobe parameter)
4. Read posting lists for selected centroids
5. Compute approximate distances using quantized vectors
6. Re-rank top candidates with full-precision vectors
7. Return top-k results
```

**Parameters**:
- `nprobe`: Number of posting lists to scan (recall/speed tradeoff)
- `rerank_k`: Number of candidates to re-rank with full precision

### Insert

Two strategies depending on insert rate:

**Low-rate inserts**: Assign to nearest centroid(s), append to posting list.
The index degrades gracefully as cluster sizes become unbalanced.

**High-rate inserts**: Buffer inserts in a separate structure, periodically
merge or trigger partial rebuild. Consider SPFresh-style approaches for
maintaining quality under updates.

### Delete

Mark vectors as deleted in posting lists. Periodically compact to reclaim
space. MVCC handled through standard PostgreSQL visibility checks.

## Storage Layout

### Page Organization

Pages are organized in two regions:

```
┌─────────────────────────────────────────────────────────────────┐
│ Blocks 0..M: Metapages (linked if >2k clusters)                 │
│   - Index metadata, parameters, reloptions                      │
│   - Posting list directory: head block for each cluster         │
├─────────────────────────────────────────────────────────────────┤
│ Blocks M+1..N: Posting List Pages                               │
│   [Cluster 0 pages][Cluster 1 pages]...[Cluster K pages]        │
│    ↑ first entry = centroid                                     │
└─────────────────────────────────────────────────────────────────┘
```

**Metapage (block 0+)**: Index metadata, parameters, reloptions, and posting
list directory (`posting_list_head[cluster_id] → block number`).

The directory stores one BlockNumber (4 bytes) per cluster. A single 8KB page
holds ~2,000 entries. For larger cluster counts, the metapage overflows to
additional pages using linked list structure (same as posting lists). At build
time, metapages are written contiguously starting at block 0.

**Posting list pages**: Store vector entries (TIDs + quantized data). The first
entry in each posting list is the cluster's centroid (medoid). At build time,
pages for each cluster are written consecutively for sequential I/O.

There are no separate centroid pages. Centroids are embedded as the first entry
of each posting list. At startup, the centroid cache is built by reading the
first page of each posting list and extracting the first vector.

### Page Linking

Posting list pages use a linked list structure. Each page header contains a
`next_blkno` field pointing to the next page (or `InvalidBlockNumber` if last).

Contiguity is determined at runtime by comparing block numbers:

```c
bool is_contiguous = (next_blkno == current_blkno + 1);
```

The scan code uses this to optimize I/O: contiguous pages can be read
sequentially or prefetched; non-contiguous pages require a seek.

### Initial State: Clustered

At index build time, pages are laid out optimally (clustered):

```
Build-time layout (clustered):

[Meta][PL0-a][PL0-b][PL0-c][PL1-a][PL1-b][PL2-a]...
blk 0  blk 1  blk 2  blk 3  blk 4  blk 5  blk 6
         ↑                    ↑            ↑
      centroid 0          centroid 1    centroid 2
      (first entry)

       └──────┴──────┘       └──────┘
       contiguous            contiguous
```

All `next_blkno` values point to `current + 1`. Scanning a posting list reads
sequential disk blocks. Centroids are the first entry on the first page of each
posting list.

### Growth and Declustering

As vectors are inserted, pages fill up and new pages are allocated at the end
of the file, breaking physical contiguity:

```
After inserts (declustered):

Original:  [Meta][PL0-a][PL0-b][PL1-a][PL1-b]...
           blk 0  blk 1  blk 2  blk 3  blk 4
                           │
                           │ next_blkno = 500 (jump!)
                           ▼
New pages at EOF:              [PL0-c][PL1-c]...
                               blk 500  501
```

The linked list structure remains intact, but links now span non-adjacent
blocks (`next_blkno != current + 1`). This causes:
- Random I/O when scanning fragmented posting lists
- Gradual performance degradation proportional to fragmentation

### Over-Allocation for Growth

To reduce fragmentation, the index supports pre-allocating extra space at build
time, controlled by reloptions:

**`fillfactor`** (default: 90): Percentage of each page to fill during build.
Leaving 10% free allows inserts to append to existing pages before needing new
ones.

```sql
CREATE INDEX ON vectors USING tigerann (embedding)
  WITH (fillfactor = 70);  -- 30% room for growth per page
```

**`reserved_pages`** (default: 0): Number of empty pages to reserve after each
posting list during build. These pages are pre-linked but empty, allowing
growth without allocation at EOF.

```sql
CREATE INDEX ON vectors USING tigerann (embedding)
  WITH (reserved_pages = 2);  -- 2 empty pages per posting list
```

Trade-off: Higher over-allocation wastes space but delays fragmentation.
Setting to 0 disables pre-allocation (suitable for static datasets).

### Restoring Clustered Layout

When fragmentation becomes significant:

- **REINDEX**: Full rebuild restores optimal clustered layout
- **VACUUM (future)**: Could reorganize pages to restore locality without full
  rebuild

### Cluster Splitting

When a cluster grows too large (exceeds a size threshold), it can be split into
two sub-clusters without affecting other clusters:

**Split procedure:**

1. Identify oversized cluster C (e.g., posting list exceeds threshold)
2. Scan C's posting list and compute two sub-centroids (k-means with k=2)
3. Assign each vector to C1 or C2 based on nearest sub-centroid
4. Rewrite existing pages in-place for C1 (centroid as first entry)
5. Write new pages at EOF for C2 (centroid as first entry)
6. Update metapage posting list directory (add C2's head block)
7. Update centroid cache

```
Before: Cluster C spans pages P1, P2, P3, P4

Split into C1 (~60%), C2 (~40%):

Pages P1, P2, P3: rewritten in-place for C1
Pages at EOF:     new pages for C2
P4:               marked as free (reclaimed by vacuum)
```

**Benefits:**
- Only one sub-cluster needs new page allocation
- Approximately half the pages are reused in-place
- Other clusters are unaffected (no global reorganization)

**Edge cases:**
- Single-page cluster: Must allocate at least one new page
- Uneven split: Larger sub-cluster gets the in-place pages

**Triggering splits:**
- Manual: Explicit maintenance command
- Automatic (future): Background worker monitors cluster sizes

For the initial implementation, cluster splitting is a manual maintenance
operation. Automatic splitting can be added later.

### Future: Fork-Based Separation

PostgreSQL relations support multiple forks (MAIN, FSM, VM, INIT). Indexes
typically only use MAIN and FSM. A future optimization could store centroids in
a separate fork (e.g., repurposing the unused VM fork for indexes), allowing:

- Independent growth of centroid and posting list regions
- Atomic replacement of centroids during reorganization
- Different caching strategies per fork

This requires careful consideration of PostgreSQL compatibility and tooling
expectations, so it's deferred for later exploration.

## PostgreSQL Integration

### Index Access Method API

Implement required callbacks:
- `ambuild`: Build index from heap scan
- `aminsert`: Insert new vector
- `ambulkdelete` / `amvacuumcleanup`: Handle deletions
- `amgettuple` / `amgetbitmap`: Return search results
- `amcostestimate`: Query planner cost estimation

### Buffer Cache Usage

Posting list pages go through the standard buffer cache:
- Benefits from PostgreSQL's LRU caching
- Supports concurrent access with proper locking
- WAL-logged for crash recovery

### Shared Memory Cache

Separate cache for centroids (not in buffer cache):
- Allocated at server startup via `shmem_request_hook`
- Faster access than buffer cache (no page locking overhead)
- Read-only after index build (no consistency concerns)

### MVCC Support

- Store XID/CID with each posting list entry
- Check visibility during scan using standard HeapTupleSatisfiesVisibility
- Vacuum removes dead entries from posting lists

## Performance Considerations

### Memory Budget

| Component | Size Estimate | Location |
|-----------|--------------|----------|
| Centroids (quantized) | ~1MB per 10k centroids | Shared memory |
| Posting list pages | Variable | Buffer cache |
| Working memory | O(nprobe * avg_list_size) | Backend memory |

### I/O Patterns

- **Centroid search**: Memory-only, no I/O
- **Posting list read**: Sequential within list, random across lists
- **Re-ranking**: Random access to heap (minimize with good quantization)

### SIMD Requirements

Critical paths requiring SIMD optimization:
1. Distance computation (L2, inner product, cosine)
2. Quantized distance computation
3. Top-k selection

## Future Extensions

1. **Product quantization**: Better compression than scalar quantization
2. **Graph-based refinement**: HNSW layer over centroids for faster search
3. **Tiered storage**: Extend to cloud storage for cold data
4. **Learned quantization**: Data-adaptive quantization schemes
5. **Filtered search**: Efficient pre-filtering with predicates

## References

- [SPANN paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2021/11/SPANN_finalversion1.pdf)
- [ScaNN for AlloyDB whitepaper](https://services.google.com/fh/files/misc/scann_for_alloydb_whitepaper.pdf)
- [turbopuffer ANN v3](https://turbopuffer.com/blog/ann-v3)
- [SPFresh paper](https://dl.acm.org/doi/epdf/10.1145/3600006.3613166)
