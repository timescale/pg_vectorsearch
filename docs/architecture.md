# Meerkat Architecture

High-level architecture for Meerkat, a PostgreSQL index access method for
approximate nearest neighbor (ANN) vector search.

## Overview

Meerkat is designed for **billion-scale vector search** within PostgreSQL,
using an inverted index approach inspired by SPANN, ScaNN, and SPFresh. The
vector space is partitioned into clusters via hierarchical clustering, enabling
efficient search by narrowing the search space through multiple levels before
scanning posting lists.

Key design elements:
- **Hierarchical clustering** to handle billion-scale datasets efficiently
- **RaBitQ quantization** with theoretical error bounds for two-stage search
- **Multi-tenant support** via composite keys or separate indexes
- **Native PostgreSQL integration** using shared buffers and standard APIs

### Hybrid Search with pg_textsearch

Meerkat is designed as part of a unified search stack alongside
[pg_textsearch](https://github.com/timescale/pg_textsearch), enabling hybrid
search that combines semantic (vector) and keyword (BM25) retrieval.

**Filtered semantic search** (keyword filter → vector ranking):

```sql
SELECT * FROM documents
WHERE textsearch @@ plainto_tsquery('quarterly revenue')
ORDER BY embedding <-> query_embedding
LIMIT 10;
```

**Reciprocal Rank Fusion (RRF)** for combining independent rankings:

```sql
WITH semantic AS (
    SELECT id, row_number() OVER (ORDER BY embedding <-> query_embedding) AS rank
    FROM documents LIMIT 100
),
keyword AS (
    SELECT id, row_number() OVER (ORDER BY ts_rank(textsearch, query) DESC) AS rank
    FROM documents WHERE textsearch @@ query LIMIT 100
)
SELECT id, 1.0/(60+s.rank) + 1.0/(60+k.rank) AS rrf_score
FROM semantic s JOIN keyword k USING (id)
ORDER BY rrf_score DESC LIMIT 10;
```

Both extensions share design principles: PostgreSQL-native, billion-scale,
multi-tenant, and optimized for modern hardware (SIMD, NVMe).

### Search Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         Query Vector                            │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│               Hierarchical Centroid Routing                     │
│     Navigate tree of centroids to find candidate clusters       │
│     Pages in shared buffers, frequently accessed = cached       │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Posting List Scan (disk/buffer)                │
│     Read posting lists for selected clusters from disk          │
│     Two-stage search with RaBitQ error bounds:                  │
│       1. Fast 1-bit estimates filter 80-95% of candidates       │
│       2. Re-rank remaining with full precision                  │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Result Set                               │
│                   Top-k nearest neighbors                       │
└─────────────────────────────────────────────────────────────────┘
```

## Performance Targets

| Metric | Target |
|--------|--------|
| Recall@10 | ≥95% (90-95% acceptable) |
| Query latency (p99) | <100ms at 1B scale |
| Index build | Hours, not days |
| Insert latency (p99) | <10ms (foreground only) |
| Update throughput | 1% daily churn without rebuild |

## Design Principles

1. **Billion-scale architecture**: Hierarchical clustering narrows the search
   space efficiently, avoiding linear scans even at massive scale.

2. **Tiered memory hierarchy**: Centroid pages in shared buffers (hot due to
   frequent access), posting lists in buffer cache, full vectors on disk.

3. **RaBitQ quantization with error bounds**: Theoretical guarantees enable
   two-stage search—fast 1-bit filtering followed by selective re-ranking.

4. **Sequential I/O**: Vectors close in embedding space stored sequentially
   on disk for efficient bulk reads.

5. **SIMD everywhere**: All distance computations vectorized with AVX512/NEON.

6. **PostgreSQL native**: Use shared buffers, WAL, MVCC, and standard IAM APIs.

7. **Multi-tenant ready**: Support both index-per-tenant and composite key
   `(tenant_id, vector)` approaches with tenant-segmented routing.

## Components

### 1. Hierarchical Centroid Structure

To handle billion-scale datasets, Meerkat uses hierarchical clustering inspired
by SPFresh (and similar to approaches in ScaNN and SPANN). Instead of a flat
list of centroids requiring linear scan, centroids are organized in a tree
structure that narrows the search space at each level.

```
                    Root Level (few hundred centroids)
                           /    |    \
                          /     |     \
                Level 1 (thousands of centroids per subtree)
                       /   |   \        /   |   \
                      /    |    \      /    |    \
            Level 2 (leaf clusters pointing to posting lists)
```

**Hierarchical routing**: A query vector traverses the tree top-down, selecting
the best branch(es) at each level. This reduces centroid comparisons from O(N)
to O(log N) while maintaining high recall through multi-path exploration.

**Centroid representation**: Centroids are **medoids** — actual data vectors
from the dataset, referenced by heap TID (`ItemPointerData`). Each centroid
entry stores the TID of the medoid vector rather than a full-precision copy.
This eliminates dedicated centroid vector storage in the index. Medoids are
selected during hierarchical k-means as the cluster member closest to the
mean.

### 2. Centroid Pages in Shared Buffers

Centroid data is stored in **dedicated centroid pages** within the index,
separate from posting list pages. These pages live in PostgreSQL's standard
shared buffer cache—there is no separate dedicated cache structure.

**Why shared buffers work well for centroids**:
- Centroid pages are accessed on every query (hot data)
- Frequently accessed pages naturally stay in the buffer cache
- Standard PostgreSQL infrastructure: locking, WAL, visibility
- No custom shared memory allocation or startup coordination

**Page layout**: Centroid pages store RaBitQ-encoded medoid vectors in a
bidirectional layout for SIMD-friendly distance computation. Each page
contains:
- Per-entry metadata (child pointer, medoid TID, flags) growing forward
- Contiguous `RaBitQData` entries (f_add, f_rescale, bits) growing backward
- Page opaque area with entry count, tree level, and sibling link

All centroids at all levels are RaBitQ-encoded relative to the global data
mean, sharing a single orthogonal matrix. The query is transformed once and
reused at every tree level.

**Sizing**: At 768 dimensions, each entry takes 120 bytes (16B metadata +
4B f_add + 4B f_rescale + 96B bits), giving 67 entries per 8KB page. For
billion-scale indexes:
- Root level: ~256-1024 centroids (4-16 pages)
- Intermediate levels: Branch factor of 32-64 (1 page per subtree)
- Leaf level: Millions of clusters, each with a posting list

### 3. Multi-Tenant Centroid Routing

For multi-tenant deployments using composite keys `(tenant_id, vector)`, the
centroid hierarchy is **segmented by tenant**. Each tenant has its own subtree
within the hierarchy:

```
             Root (tenant directory)
            /     |     \      \
       Tenant A  Tenant B  Tenant C  ...
          |         |         |
    (per-tenant hierarchical clusters)
```

This ensures:
- Queries only traverse centroids for the specified tenant
- No cross-tenant interference in search
- Tenants can have different cluster counts based on data volume
- Efficient tenant isolation without separate indexes

See the Multi-Tenant Support section for deployment options.

### 4. Posting Lists

A posting list is the set of vectors assigned to a leaf cluster (borrowing
terminology from inverted indexes in text search). Each leaf centroid points to
its posting list, which contains:

- **Vector TIDs**: Pointers to the full-precision vectors in the heap
- **RaBitQ quantized vectors**: Binary quantized vectors with error factors for
  two-stage search (see Quantization section)

**Tenant segmentation**: For composite key indexes `(tenant_id, vector)`,
posting lists are segmented by tenant. A leaf centroid's posting list only
contains vectors from the tenant whose subtree it belongs to. This is natural
since the hierarchical routing already separates tenants at the root level.

**Why buffer cache?** Unlike centroid pages which are accessed on every query,
posting lists are large (the bulk of index data) and only a subset is accessed
per query (based on nprobe). The buffer cache provides:
- Automatic caching of hot posting lists (frequently accessed clusters)
- Memory sharing across backends
- Standard PostgreSQL page management and WAL logging

**Cluster assignment**: During index build, each vector is assigned to its
nearest leaf centroid. However, single-cluster assignment causes a "boundary
problem": if a query's true nearest neighbor lies just across a cluster
boundary, it may be missed when only searching the query's nearest cluster.

### Improving Recall: Multi-Cluster Assignment

Two main approaches exist for assigning vectors to multiple clusters:

**SPANN approach: Boundary-only replication (closure augmentation)**

Only vectors near cluster boundaries are duplicated to neighboring clusters.
Vectors close to their centroid remain in a single cluster.

- Uses distance threshold to identify boundary vectors
- Applies [RNG (Relative Neighborhood Graph) rule](https://arxiv.org/abs/2111.08566)
  to select which clusters receive duplicates, reducing redundancy between
  similar clusters
- Limits replication factor (SPANN uses max 8 replicas)
- Trade-off: Lower storage overhead, but requires tuning the boundary threshold

**ScaNN/SOAR approach: Universal replication with orthogonal residuals**

All vectors are assigned to multiple clusters, with secondary assignments
chosen to provide independent "backup" coverage.

- Primary assignment: nearest centroid (standard k-means)
- Secondary assignments: chosen so residual errors are orthogonal to primary
  residual ([SOAR algorithm](https://research.google/blog/soar-new-algorithms-for-even-faster-vector-search-with-scann/))
- When query is parallel to primary residual (high error), it's orthogonal to
  secondary residual (low error)—providing effective redundancy
- Trade-off: Higher storage overhead, but more systematic recall improvement

**Meerkat approach:**

For initial implementation, use SPANN-style boundary-only replication:
- Lower storage overhead (important for disk-based index)
- Simpler to implement and tune
- Replication factor controlled by reloption `max_replicas` (default: 1, meaning
  no replication; set higher for better recall)

Future versions may explore SOAR-style orthogonal secondary assignments for
workloads where recall is critical.

**Balancing**: Ideally, posting lists should be roughly equal in size for
predictable query latency. The clustering algorithm aims for balanced clusters,
but natural data distribution may cause imbalance. Very large clusters can be
split; very small clusters may be merged or eliminated.

### 5. Vector Quantization (RaBitQ)

Meerkat uses **RaBitQ** (Random Bit Quantization) for vector compression, which
provides theoretical error bounds enabling efficient two-stage search.

**Why RaBitQ?**
- State-of-the-art binary quantization with provable error bounds
- 32x compression (float32 → 1-bit per dimension)
- Error bounds enable filtering without false negatives
- Fast SIMD-friendly binary operations

**Two-stage search with error bounds**:

RaBitQ computes both an estimated distance and an error bound for each vector.
The key insight is that the lower bound (estimate - error) is guaranteed to be
≤ the true distance. This enables:

1. **Stage 1 - Fast filtering**: Compute 1-bit distance estimates for all
   vectors in selected posting lists. Use error bounds to identify candidates
   that *could* be in the top-k (lower bound ≤ current k-th best).

2. **Stage 2 - Precise re-ranking**: Only fetch full-precision vectors for
   candidates that passed filtering (~1-5% of vectors). Compute exact distances
   and return true top-k.

**Error bound components**:
- `f_error`: Per-vector factor computed at index time (stored with each vector)
- `g_error`: Per-query factor computed at search time
- Combined bound: `|true_dist - estimate| ≤ f_error × g_error`

This approach filters 80-95% of candidates using fast binary operations while
guaranteeing no false negatives—every vector in the true top-k will pass to
stage 2.

### 6. Vector Storage

Full-precision vectors are stored in the heap of the indexed table—no separate
vector storage. For typical embedding dimensions (768-1536), vectors exceed
PostgreSQL's inline storage threshold and are TOASTed automatically.

**TOAST access overhead**: Reading a TOASTed vector requires multiple I/O steps:
1. TID lookup to find the heap tuple
2. Read heap page, discover vector is TOASTed
3. Index scan on TOAST chunk_id index
4. Read TOAST heap page(s) to retrieve chunks

This multi-step access is expensive. The two-stage RaBitQ search minimizes heap
access by only fetching full-precision vectors for the ~1-5% of candidates that
pass error-bounded filtering.

**Future exploration**: If full-precision vector access becomes a bottleneck,
alternatives to TOAST storage could be explored (e.g., storing vectors in index
pages, a dedicated vector heap, columnar storage, or a custom TOAST table
access method optimized for vector retrieval).

### 7. Metadata

Index metadata is stored in multiple locations depending on its nature:

**Reloptions** (pg_class.reloptions, specified at CREATE INDEX):
- Build parameters: `nlist` (number of leaf clusters), `fillfactor`
- Search defaults: `nprobe` (clusters to search), `rerank_k`
- Over-allocation: `reserved_pages`
- Multi-tenant: `tenant_column` (for composite key indexes)

**GUCs** (session/server-level, can override reloptions):
- `meerkat.nprobe` - clusters to search per query
- `meerkat.rerank_k` - candidates to re-rank with full precision

**Catalog tables** (managed by PostgreSQL):
- Structural info in pg_class, pg_index, pg_am, pg_opclass

**Metapage** (stored in index file, persists with the index):
- Hierarchical centroid tree structure (root page, level info)
- Posting list directory (head block per leaf cluster)
- RaBitQ normalization factors
- Index version, dimension, distance metric
- Statistics: cluster sizes, total vectors indexed, per-tenant stats

## Multi-Tenant Support

Meerkat supports multi-tenant deployments through two approaches:

### Option 1: Index-per-Tenant

Create separate tables and indexes for each tenant:

```sql
CREATE TABLE tenant_123_vectors (id bigint, embedding vector(768));
CREATE INDEX ON tenant_123_vectors USING meerkat (embedding);
```

**Pros**:
- Complete isolation (no cross-tenant data leakage possible)
- Independent scaling, tuning, and maintenance per tenant
- Can drop tenant data instantly

**Cons**:
- Connection/catalog overhead with many tenants
- Cannot easily query across tenants
- More complex application logic

**Best for**: Strict isolation requirements, large tenants with distinct
workloads, regulatory compliance scenarios.

### Option 2: Composite Key Index

Store all tenants in one table with a composite index key:

```sql
CREATE TABLE vectors (
    tenant_id int,
    id bigint,
    embedding vector(768)
);
CREATE INDEX ON vectors USING meerkat ((tenant_id, embedding));
```

**How it works**:
- The hierarchical centroid tree is segmented by `tenant_id` at the root level
- Each tenant has its own subtree of centroids and posting lists
- Queries specify `tenant_id` and route directly to that tenant's subtree
- No cross-tenant centroid comparisons or posting list scans

```sql
-- Query automatically routes to tenant 42's subtree
SELECT * FROM vectors
WHERE tenant_id = 42
ORDER BY embedding <-> '[...]'::vector
LIMIT 10;
```

**Pros**:
- Single table and index to manage
- Efficient storage (shared infrastructure)
- Simpler application logic
- Can query across tenants if needed (with appropriate permissions)

**Cons**:
- Shared buffer cache (hot tenant may evict cold tenant's pages)
- Maintenance operations affect all tenants
- Tenant deletion requires DELETE + VACUUM

**Best for**: Many small-to-medium tenants, shared infrastructure, simpler
operations.

### Tenant-Segmented Storage

For composite key indexes, both centroid routing and posting lists are
segmented by tenant:

1. **Root-level tenant directory**: The root of the centroid tree contains a
   tenant directory mapping `tenant_id` → subtree root page.

2. **Per-tenant centroid hierarchy**: Each tenant has independent centroid
   pages forming their own tree. Tenant data volume determines cluster count.

3. **Per-tenant posting lists**: Leaf clusters only contain vectors from their
   tenant. No mixing of tenant data in posting lists.

This segmentation ensures queries only access pages for the target tenant,
providing both performance isolation and predictable access patterns.

## Index Operations

### Build

```
1. For composite key index: group vectors by tenant_id
2. For each tenant (or all vectors if single-tenant):
   a. Sample vectors for clustering
   b. Run hierarchical k-means to build centroid tree
   c. Assign each vector to nearest leaf centroid(s)
   d. Compute RaBitQ quantization and f_error factors
3. Write centroid pages (hierarchical structure)
4. Write posting list pages with quantized vectors
5. Write metapage with tree root and directory
```

**Optimizations**:
- Hierarchical balanced clustering for uniform posting list sizes
- Parallel clustering using multiple workers
- Streaming build to limit memory usage
- Per-tenant parallel builds for composite key indexes

### Search

```
1. For composite key: lookup tenant subtree root from tenant directory
2. Navigate centroid tree top-down:
   a. At each level, compute distances to child centroids
   b. Select best branch(es) to explore (beam search)
   c. Repeat until reaching leaf level
3. For selected leaf clusters (nprobe clusters):
   a. Read posting list pages
   b. Compute RaBitQ estimates and error bounds
   c. Filter candidates: keep if lower_bound ≤ k-th best
4. Re-rank surviving candidates (~1-5%) with full precision
5. Return top-k results
```

**Beam search**: Centroid routing uses level-by-level beam search
(`mkt_centroid_beam_search`) rather than best-first search. This maps well
to PostgreSQL's page-based buffer cache: each level is processed as a batch,
enabling SIMD distance computation on entries within each page.
Upper-level pages stay hot in `shared_buffers` since they are accessed on
every query.

**Parameters**:
- `nprobe`: Number of leaf posting lists to scan (recall/speed tradeoff)
- `beam_width`: Candidates to keep at each tree level (default: 1)
- `rerank_k`: Number of candidates for full-precision re-ranking

### Dynamic Updates (LIRE Protocol)

Meerkat adopts the **LIRE (Lightweight Incremental RE-balancing)** protocol
from SPFresh for maintaining index quality under continuous updates without
full rebuilds.

#### Foreground/Background Architecture

| Stage | Operations | Characteristics |
|-------|------------|-----------------|
| **Foreground** | Insert, Delete (tombstone) | Fast, predictable latency |
| **Background** | Split, Merge, Reassign, GC | Heavy lifting, off critical path |

This separation keeps insert latency low while moving expensive rebalancing to
background workers (PostgreSQL's parallel worker framework).

#### Insert

```
1. Find nearest leaf centroid(s) for the new vector
2. Append vector + RaBitQ encoding to posting list tail
3. Update in-memory metadata (version, count)
4. If posting exceeds size threshold → queue split job
```

Inserts are append-only in the foreground. Only ~0.4% of insertions trigger
rebalancing operations (based on SPFresh measurements).

#### Delete

```
1. Set tombstone bit in posting list entry
2. Vector immediately excluded from search results
3. Physical removal deferred to background GC during rebalancing
```

Tombstones use PostgreSQL's standard MVCC visibility—deleted vectors are
invisible to new transactions immediately.

#### Split

Triggered when a posting list exceeds the size threshold:

```
1. Garbage collect deleted vectors first
2. If still oversized: apply balanced 2-means clustering
3. Generate two new centroids, redistribute vectors
4. Update centroid page (replace old entry with two new)
5. Queue reassignment jobs for affected vectors
```

#### Merge

Triggered when a posting list falls below minimum threshold:

```
1. Find nearest posting as merge candidate
2. Append vectors from smaller posting to larger
3. Remove redundant centroid from tree
4. Queue reassignment jobs for merged vectors
```

#### Reassign

Corrects **NPA (Nearest Partition Assignment) violations** after split/merge.
A vector should be in the posting list of its nearest centroid.

**After split**: Check if vectors in the split posting should move to the
*other* new posting or to a neighboring posting.

**After merge**: Check if vectors in neighbor postings should move to the
newly merged posting (which may now be closer).

**Key insight**: Only boundary vectors require reassignment. In a well-balanced
index, ~79 vectors are reassigned out of ~5000 evaluated per rebalancing
operation (SPFresh measurements).

#### Version Tracking

Each vector entry includes a version byte:
- 7 bits: reassign version (incremented on each reassignment)
- 1 bit: deletion flag

During search, vectors with stale versions (lower than current metadata) are
skipped. This enables lock-free reads—searches never block on updates.

### Recall Measurement

Meerkat supports measuring recall directly within PostgreSQL via an EXPLAIN
option:

```sql
EXPLAIN (ANALYZE, RECALL)
SELECT * FROM documents
ORDER BY embedding <-> query_embedding
LIMIT 10;
```

**How it works**:

1. Execute the query using the ANN index (normal path)
2. Execute an exact brute-force scan over the same data
3. Compare result sets to compute recall = |ANN ∩ exact| / k

**Output includes**:
- Standard EXPLAIN ANALYZE output (timing, rows, etc.)
- Recall percentage (e.g., "Recall: 95.0% (19/20 exact matches)")
- Number of vectors scanned vs. total
- Quantization filtering statistics (candidates before/after RaBitQ filtering)

**Use cases**:
- Tuning `nprobe` and `beam_width` for recall/speed tradeoff
- Validating index quality after build or maintenance
- Comparing different index configurations
- Debugging unexpected search results

**Performance note**: The exact scan can be expensive on large tables. For
billion-scale tables, consider using `RECALL` with a `WHERE` clause to limit
the scope, or sample-based recall estimation:

```sql
-- Recall on a subset (faster)
EXPLAIN (ANALYZE, RECALL)
SELECT * FROM documents
WHERE tenant_id = 42
ORDER BY embedding <-> query_embedding
LIMIT 10;

-- Sample-based recall (future)
EXPLAIN (ANALYZE, RECALL, RECALL_SAMPLE 10000)
SELECT * FROM documents
ORDER BY embedding <-> query_embedding
LIMIT 10;
```

## Storage Layout

### Page Organization

Pages are organized in three regions:

```
┌─────────────────────────────────────────────────────────────────┐
│ Block 0: Metapage                                               │
│   - Index metadata, parameters, dimension, distance metric      │
│   - Root centroid page pointer                                  │
│   - Tenant directory (for composite key indexes)                │
├─────────────────────────────────────────────────────────────────┤
│ Blocks 1..C: Centroid Pages (hierarchical tree)                 │
│   - Level 0 (root): few hundred centroids                       │
│   - Level 1..N-1: intermediate levels                           │
│   - Level N (leaves): pointers to posting lists                 │
├─────────────────────────────────────────────────────────────────┤
│ Blocks C+1..P: Posting List Pages                               │
│   - Quantized vectors with RaBitQ data (TID, bits, f_error)     │
│   - Linked list structure within each cluster                   │
└─────────────────────────────────────────────────────────────────┘
```

### Centroid Pages

Centroid pages store the hierarchical routing structure. Each page contains:
- Array of quantized centroid vectors (SIMD-aligned)
- Child pointers (page numbers for next level, or posting list heads for leaves)
- Level indicator and parent pointer for navigation

Centroid pages are stored in PostgreSQL's standard shared buffer cache. Because
they are accessed on every query, they naturally remain cached (hot pages).
There is no separate dedicated cache structure—just standard buffer management.

**Multi-tenant layout**: For composite key indexes, the root level contains a
tenant directory. Each tenant's subtree is stored contiguously:

```
[Meta][TenantDir][Tenant1-L0][Tenant1-L1...][Tenant2-L0][Tenant2-L1...]...
```

### Posting List Pages

Posting list pages store vector entries with RaBitQ quantization:
- TID: pointer to full-precision vector in heap
- Quantized bits: 1-bit per dimension (d/8 bytes)
- f_error: per-vector error factor for distance bounds

Pages within a posting list are linked. Each page header contains `next_blkno`.

### Initial State: Clustered

At build time, pages are laid out for sequential I/O:

```
Build-time layout:

[Meta][Centroid pages...][PL0-a][PL0-b][PL1-a][PL1-b][PL2-a]...

Posting lists are contiguous per cluster.
```

Contiguity is checked at runtime:

```c
bool is_contiguous = (next_blkno == current_blkno + 1);
```

Contiguous pages can be prefetched; fragmented pages require seeks.

### Growth and Declustering

As vectors are inserted, posting list pages fill up and new pages are allocated
at the end of the file, breaking physical contiguity:

```
After inserts (declustered):

Original:  [Meta][Centroids...][PL0-a][PL0-b][PL1-a][PL1-b]...
                                        │
                                        │ next_blkno = 500 (jump!)
                                        ▼
New pages at EOF:                           [PL0-c][PL1-c]...
                                            blk 500  501
```

The linked list structure remains intact, but links now span non-adjacent
blocks (`next_blkno != current + 1`). This causes:
- Random I/O when scanning fragmented posting lists
- Gradual performance degradation proportional to fragmentation

Centroid pages are rarely modified after build (only during cluster splits or
rebalancing), so they remain contiguous.

### Over-Allocation for Growth

To reduce fragmentation, the index supports pre-allocating extra space at build
time, controlled by reloptions:

**`fillfactor`** (default: 90): Percentage of each page to fill during build.
Leaving 10% free allows inserts to append to existing pages before needing new
ones.

```sql
CREATE INDEX ON vectors USING meerkat (embedding)
  WITH (fillfactor = 70);  -- 30% room for growth per page
```

**`reserved_pages`** (default: 0): Number of empty pages to reserve after each
posting list during build. These pages are pre-linked but empty, allowing
growth without allocation at EOF.

```sql
CREATE INDEX ON vectors USING meerkat (embedding)
  WITH (reserved_pages = 2);  -- 2 empty pages per posting list
```

Trade-off: Higher over-allocation wastes space but delays fragmentation.
Setting to 0 disables pre-allocation (suitable for static datasets).

### Restoring Clustered Layout

When fragmentation becomes significant:

- **REINDEX**: Full rebuild restores optimal clustered layout
- **VACUUM (future)**: Could reorganize pages to restore locality without full
  rebuild

### Rebalancing Storage Layout

Split and merge operations (see LIRE Protocol above) affect storage layout:

**Split storage flow:**

```
Before: Cluster C spans pages P1, P2, P3, P4

After split into C1 (~60%), C2 (~40%):

Pages P1, P2, P3: rewritten in-place for C1
New pages at EOF: new posting list for C2
P4:               marked as free (reclaimed by vacuum)
Centroid page:    C entry replaced with C1, C2
```

**Merge storage flow:**

```
Before: Small cluster C1 (pages P1) merges into neighbor C2 (pages P2, P3)

After merge:

Pages P2, P3, + new page: combined posting list for C2
P1:                       marked as free
Centroid page:            C1 entry removed
```

**Cascade bounds** (from SPFresh measurements):
- Only ~0.4% of insertions trigger rebalancing
- Average cascade length: 3 operations
- Maximum observed: 160 splits in a cascade
- Convergence guaranteed (each split adds exactly one centroid)

**Thresholds:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `split_threshold` | 2× target size | Trigger split when posting exceeds |
| `merge_threshold` | 0.25× target size | Trigger merge when posting falls below |
| `reassign_range` | 64 postings | Neighbor postings to check for reassignment |

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

Both centroid pages and posting list pages use the standard shared buffer cache:

**Centroid pages**:
- Accessed on every query (hot data)
- Naturally stay cached due to frequent access
- No separate dedicated cache structure needed
- Standard PostgreSQL locking and WAL apply

**Posting list pages**:
- Accessed based on query routing (subset per query)
- LRU caching benefits frequently accessed clusters
- Sequential prefetching for contiguous pages

### MVCC Support

- Store XID/CID with each posting list entry
- Check visibility during scan using standard HeapTupleSatisfiesVisibility
- Vacuum removes dead entries from posting lists

## Performance Considerations

### Memory Budget

| Component | Size Estimate | Location |
|-----------|--------------|----------|
| Centroid pages | ~1MB per 10k centroids | Shared buffers (hot) |
| Posting list pages | Bulk of index | Shared buffers (LRU) |
| Working memory | O(nprobe × avg_list_size) | Backend memory |
| RaBitQ query state | O(dimension) | Backend memory |

### I/O Patterns

- **Centroid routing**: Typically cached (hot pages in shared buffers)
- **Posting list read**: Sequential within list, random across lists
- **Re-ranking**: Random access to heap (~1-5% of candidates)

### SIMD Requirements

Critical paths requiring SIMD optimization:
1. Distance computation (L2, inner product, cosine)
2. RaBitQ binary distance and error bound computation
3. Top-k selection during centroid routing

## Billion-Scale Feasibility Study

This section analyzes Meerkat's feasibility at 1 billion vectors, using SPFresh
measurements as a baseline and calculating PostgreSQL-specific estimates.

### SPFresh Reference Numbers (1B vectors, 96 dimensions)

From the SPFresh paper (SPACEV1B dataset):

| Metric | SPFresh Result |
|--------|----------------|
| Recall@10 | 90-97% (tunable) |
| Query latency (p50) | 2-4ms |
| Query latency (p99) | 8-12ms |
| Update throughput | 1,000-5,000 vectors/sec |
| Memory usage | 10GB (vs 1000GB for DiskANN) |
| Index size | ~100GB |
| Rebalancing overhead | 0.4% of inserts trigger split |

Note: SPACEV1B uses 96-dimensional vectors. Modern embeddings (768-1536 dims)
require proportionally more storage and compute.

### Meerkat Estimates (1B vectors, 768 dimensions)

**Base latency assumptions** (from [napkin-math]):

| Operation | Latency | Throughput |
|-----------|---------|------------|
| Sequential memory (SIMD) | 0.5 ns | 20 GB/s |
| Random memory (64 bytes) | 50 ns | 1 GB/s |
| Sequential SSD read (8KB) | 1 μs | 4 GB/s |
| Random SSD read (8KB) | 100 μs | 70 MB/s |
| Same-zone network | 100 μs | 10 GB/s |

**AWS storage options** (from [AWS i4i]/[AWS i3en] specs):

| Storage | Random IOPS (4KB) | Seq throughput | Latency | Cost |
|---------|-------------------|----------------|---------|------|
| i4i.4xlarge NVMe | 400K read | ~3 GB/s | ~100 μs | ~$1/hr |
| i4i.16xlarge NVMe | 1.6M read | ~7 GB/s | ~100 μs | ~$4/hr |
| EBS gp3 (baseline) | 3K | 125 MB/s | ~200 μs | $0.08/GB |
| EBS gp3 (max) | 16K | 1 GB/s | ~200 μs | +$0.005/IOPS |
| EBS io2 Block Express | 256K | 4 GB/s | ~200 μs | $0.065/GB |

**Platform assumptions:**
- 1 billion vectors, 768 dimensions (OpenAI ada-002 scale)
- **float16 (2 bytes)** per dimension for full-precision vectors
- **Index on local NVMe** (i4i): 400K IOPS, ~3 GB/s, ~100 μs latency
- **Heap on EBS gp3**: 16K IOPS provisioned, 1 GB/s, ~200 μs latency
- 8KB PostgreSQL pages
- 1 million leaf clusters (1,000 vectors per cluster average)
- Boundary replication factor: 2× average

[AWS i4i]: https://aws.amazon.com/ec2/instance-types/i4i/
[AWS i3en]: https://aws.amazon.com/ec2/instance-types/i3en/

**Vector element size comparison:**

| Type | Bytes | 768d vector | 100d vector (SPACEV1B) |
|------|-------|-------------|------------------------|
| int8 | 1 | 768 B | 100 B (SPFresh) |
| float16 | 2 | 1,536 B | 200 B |
| float32 | 4 | 3,072 B | 400 B |

We use **float16** as a practical middle ground: sufficient precision for most
embedding models, avoids TOAST overhead for 768d, and halves storage vs float32.

[napkin-math]: https://github.com/sirupsen/napkin-math

#### Storage Requirements

**Full-precision vectors (heap on EBS):**

PostgreSQL TOASTs values exceeding ~2KB. With float16:

| Dimensions | Vector size (float16) | TOAST? | Notes |
|------------|----------------------|--------|-------|
| 768 | 1,536 bytes | No | Inline in heap tuple |
| 1024 | 2,048 bytes | Borderline | May be compressed inline |
| 1536 | 3,072 bytes | Yes | Stored in TOAST table |
| 3072 | 6,144 bytes | Yes | Stored in TOAST table |

**Scenario A: Inline vectors (768d, float16) — recommended**

| Component | Calculation | Size |
|-----------|-------------|------|
| Heap tuples | 1B × (1,536 + 24 header) bytes | 1.46 TB |
| Page overhead | ~10% | 146 GB |
| **Total heap** | | **~1.6 TB** |

**Scenario B: TOASTed vectors (1536d float16 or 768d float32)**

| Component | Calculation | Size |
|-----------|-------------|------|
| TOAST chunks | 1B × 3,072 bytes | 2.9 TB |
| TOAST overhead | chunk headers + index | ~50 GB |
| Main heap tuples | 1B × ~40 bytes (TOAST pointer) | 40 GB |
| **Total heap + TOAST** | | **~3.0 TB** |

**Scenario C: Vectors stored in index (no heap access)**

| Component | Calculation | Size |
|-----------|-------------|------|
| Full vectors in index | 1B × 1,536 bytes × 2 (replication) | 2.9 TB |
| RaBitQ + metadata | (as below) | 260 GB |
| **Total index** | | **~3.2 TB** |

Trade-off: 13× larger index but eliminates heap access entirely.

**Index storage (NVMe) - quantized only:**

| Component | Calculation | Size |
|-----------|-------------|------|
| RaBitQ bits | 768 bits / 8 = 96 bytes/vector | 96 GB |
| f_add | 4 bytes/entry | 4 GB |
| f_rescale | 4 bytes/entry | 4 GB |
| TID | 6 bytes/entry | 6 GB |
| Flags + reserved | 2 bytes/entry | 2 GB |
| **Per-entry total** | **112 bytes** | **112 GB** |
| Boundary replication (2×) | 112 GB × 2 | 224 GB |
| Centroids (quantized) | 1M × 96 bytes | 96 MB |
| Centroid tree overhead | ~3 levels | 10 MB |
| Page headers/fragmentation | ~15% | 34 GB |
| **Total index** | | **~260 GB** |

**Storage summary:**

| Configuration | Vector size | Heap | Index | Total |
|---------------|-------------|------|-------|-------|
| Inline (768d, float16) | 1,536 B | 1.6 TB | 260 GB | **1.9 TB** |
| TOASTed (1536d, float16) | 3,072 B | 3.0 TB | 260 GB | **3.3 TB** |
| Vectors in index | 1,536 B | 0 | 3.2 TB | **3.2 TB** |
| SPFresh (100d, int8) | 100 B | ~100 GB | ~100 GB | **~200 GB** |

#### Query Performance Analysis

All calculations use [napkin-math] reference latencies.

**Centroid routing (hierarchical tree traversal):**

| Level | Centroids | Data | Location | Latency |
|-------|-----------|------|----------|---------|
| Root | 256 | 24 KB | L3 cache (hot) | 256 × 50ns = **13 μs** |
| Level 1 | 4,000 | 384 KB | Shared buffers | 4K × 50ns = **200 μs** |
| Level 2 (leaves) | 1M | 96 MB | Shared buffers/NVMe | **0.1-1 ms** |
| **Total routing** | | | | **0.3-1.2 ms** |

Note: Hot paths stay in shared_buffers. Cold tenant routing may hit NVMe
(100 μs per random 8KB page).

**Posting list scan (nprobe=20) on i4i NVMe:**

| Step | Calculation | Latency |
|------|-------------|---------|
| Vectors to scan | 20 clusters × 1,000 vectors | 20,000 vectors |
| Index data to read | 20,000 × 112 bytes | 2.2 MB |
| Pages to read | 2.2 MB / 8 KB | 275 pages |
| RaBitQ compute (SIMD) | 20K × ~10 cycles / 3 GHz | **0.07 ms** |

I/O latency depends critically on access pattern and async I/O:

| Scenario | Access pattern | Calculation | Latency |
|----------|----------------|-------------|---------|
| Fresh index (contiguous) | Sequential read | 2.2 MB / 3 GB/s | **0.7 ms** |
| Fragmented, serial I/O | Random QD=1 | 275 × 100 μs | **28 ms** |
| Fragmented, async I/O | Random QD=275 | 275 / 400K + overhead | **1-2 ms** |

*QD (queue depth) = I/O requests in flight simultaneously. NVMe achieves 400K
IOPS only at QD≥32. Serial reads (QD=1) pay full 100μs latency per page.*

**Key insight**: The 400K IOPS figure requires **queue depth saturation**. Serial
random reads are 100 μs each. To achieve low latency on fragmented posting lists:

1. **Prefetching**: PostgreSQL's `effective_io_concurrency` enables async prefetch
2. **io_uring**: Submit all page reads in parallel, wait for completion
3. **Keep lists contiguous**: Fresh builds are sequential; REINDEX restores this

For fresh/defragmented indexes, posting scan is ~1ms. For fragmented indexes
without async I/O, it degrades to ~27ms. Async I/O (io_uring) recovers to ~2ms.

**Re-ranking (top candidates survive RaBitQ filtering):**

| Step | Calculation | Latency |
|------|-------------|---------|
| Candidates after filter | 20,000 × 5% | 1,000 vectors |
| Distance compute (SIMD) | 1K × 768 × 2B / 20 GB/s | **0.08 ms** |

Heap access latency depends on storage configuration:

**Scenario A: Inline vectors (768d, float16) — heap on EBS gp3**

| Step | I/O ops | Calculation | Latency |
|------|---------|-------------|---------|
| Heap page reads | 1,000 | 1,000 / 16K IOPS | **62 ms** |
| With prefetch batching | | 4× improvement | **15-20 ms** |

**Scenario A': Inline vectors — heap on local NVMe (i4i)**

| Step | I/O ops | Calculation | Latency |
|------|---------|-------------|---------|
| Heap page reads | 1,000 | 1,000 / 400K IOPS | **2.5 ms** |

**Scenario B: TOASTed vectors (1536d) — heap on EBS gp3**

| Step | I/O ops | Calculation | Latency |
|------|---------|-------------|---------|
| Heap + TOAST reads | 4,000 | 4,000 / 16K IOPS | **250 ms** |
| With prefetch batching | | 4× improvement | **60-80 ms** |

**Scenario C: Vectors stored in index (NVMe only)**

| Step | I/O ops | Calculation | Latency |
|------|---------|-------------|---------|
| Already in posting list | 0 | 0 | **0 ms** |
| (Larger posting scan) | +190 pages | +190 / 400K IOPS | **+0.5 ms** |

**Query latency summary by configuration:**

Assumes contiguous posting lists (fresh index) or async I/O for fragmented lists.

| Configuration | Heap storage | Routing | Posting | Re-rank | **p50** | **p99** |
|---------------|--------------|---------|---------|---------|---------|---------|
| Inline (768d) | EBS gp3 16K | <1ms | 1-2ms | 15ms | **~17ms** | **~40ms** |
| Inline (768d) | i4i NVMe | <1ms | 1-2ms | 2.5ms | **~5ms** | **~12ms** |
| TOASTed (1536d) | EBS gp3 16K | <1ms | 1-2ms | 60ms | **~62ms** | **~110ms** |
| Vectors in index | N/A | <1ms | 2-3ms | <1ms | **~4ms** | **~10ms** |
| SPFresh (100d) | local NVMe | <1ms | 2ms | 2ms | **~4ms** | **~10ms** |

**Fragmentation impact**: Without async I/O, fragmented posting lists degrade
posting scan from ~1ms to ~27ms. Mitigation: use io_uring, maintain contiguity
via REINDEX, or set `effective_io_concurrency` appropriately.

**Key insights**:

1. **Posting scan is I/O-bound**: RaBitQ binary distance with AVX-512 is ~0.1ms
   for 20K vectors. NVMe read latency dominates, consistent with SPFresh.

2. **Async I/O is critical**: Serial random reads are 100μs each. Without async
   I/O (io_uring or prefetch), fragmented posting lists degrade to ~27ms.
   PostgreSQL 16+ supports io_uring; earlier versions use `effective_io_concurrency`.

3. **Heap access dominates total latency**: With EBS, re-ranking is 15-60ms.
   With local NVMe (i4i), re-ranking drops to 2.5ms.

4. **i4i NVMe matches SPFresh latency**: 5ms p50 vs SPFresh's 4ms despite
   15× larger vectors. The architecture is equally efficient per-byte.

5. **Vectors-in-index achieves 4ms p50** at ~2× storage cost. Consider for
   latency-critical workloads.

**Storage cost vs latency tradeoff:**

| Configuration | Storage | p50 latency | Monthly cost (1B vectors) |
|---------------|---------|-------------|---------------------------|
| Inline + EBS gp3 | 1.9 TB | 17ms | ~$150 storage + $80 IOPS |
| Inline + i4i NVMe | 1.9 TB | 5ms | ~$1,000 (i4i.4xlarge) |
| Vectors in index | 3.2 TB | 4ms | ~$1,500 (larger i4i) |

*Requires async I/O or contiguous posting lists. See fragmentation discussion above.*

#### Index Build Performance

Using [napkin-math] reference throughputs. Assumes 768d float16 vectors.

**Phase 1: Sampling and clustering (leader only)**

| Step | Calculation | Time |
|------|-------------|------|
| Sample 1% of vectors | 10M vectors × 1.5 KB = 15 GB | |
| Sequential heap read | 15 GB / 1 GB/s (EBS) | 15 sec |
| Load to memory | 15 GB / 20 GB/s (SIMD) | 0.75 sec |
| Hierarchical k-means | 10M × 768d × 20 iterations | ~30 min |
| **Phase 1 total** | | **~32 min** |

Clustering runs on the leader process only (shared centroid state). Could be
parallelized with parallel k-means, but 30 min is acceptable for 1B vectors.

**Phase 2: Full scan, assignment, quantization (parallel workers)**

PostgreSQL's parallel index build infrastructure partitions the heap scan across
workers. Each worker independently:
1. Scans assigned heap pages
2. Finds nearest centroid for each vector
3. Computes RaBitQ encoding
4. Writes to worker-local buffer

| Step | Serial | Parallelizable? |
|------|--------|-----------------|
| Heap scan | 27 min | Yes (I/O bandwidth limited) |
| Centroid search | 25 min | Yes (CPU, scales linearly) |
| RaBitQ encoding | 5 min | Yes (CPU, scales linearly) |
| **Serial total** | **57 min** | |

**Parallel scaling analysis:**

| Workers | Heap scan | Centroid | RaBitQ | Merge | **Total** | Speedup |
|---------|-----------|----------|--------|-------|-----------|---------|
| 1 | 27 min | 25 min | 5 min | 0 | **57 min** | 1.0× |
| 2 | 27 min | 12.5 min | 2.5 min | 1 min | **43 min** | 1.3× |
| 4 | 27 min | 6.3 min | 1.3 min | 2 min | **37 min** | 1.5× |
| 8 | 27 min | 3.1 min | 0.6 min | 3 min | **34 min** | 1.7× |
| 16 | 27 min | 1.6 min | 0.3 min | 4 min | **33 min** | 1.7× |

*Heap scan is I/O-bound at 1 GB/s (EBS). CPU work scales but I/O doesn't.*

**With local NVMe (i4i: 3 GB/s read):**

| Workers | Heap scan | Centroid | RaBitQ | Merge | **Total** | Speedup |
|---------|-----------|----------|--------|-------|-----------|---------|
| 1 | 9 min | 25 min | 5 min | 0 | **39 min** | 1.0× |
| 4 | 9 min | 6.3 min | 1.3 min | 2 min | **19 min** | 2.1× |
| 8 | 9 min | 3.1 min | 0.6 min | 3 min | **16 min** | 2.4× |
| 16 | 9 min | 1.6 min | 0.3 min | 4 min | **15 min** | 2.6× |

*Local NVMe removes I/O bottleneck; build becomes CPU-bound and scales better.*

**Amdahl's Law analysis:**

```
Serial fraction (EBS):  ~50% (heap scan I/O)
Serial fraction (NVMe): ~25% (heap scan I/O)

Max speedup (EBS):  1 / 0.50 = 2×    → diminishing returns after 4 workers
Max speedup (NVMe): 1 / 0.25 = 4×    → scales to 8-16 workers
```

**Phase 3: Merge and write (leader + I/O)**

| Step | Calculation | Time |
|------|-------------|------|
| Merge worker buffers | Combine posting lists | 2-4 min |
| Write posting lists | 260 GB / 4 GB/s (NVMe seq) | 65 sec |
| Fsync overhead | ~10% | 6 sec |
| Write centroid pages | 100 MB / 4 GB/s | <1 sec |
| **Phase 3 total** | | **~3-5 min** |

**Build time summary:**

| Configuration | Phase 1 | Phase 2 | Phase 3 | **Total** |
|---------------|---------|---------|---------|-----------|
| EBS, 1 worker | 32 min | 57 min | 2 min | **91 min** |
| EBS, 8 workers | 32 min | 34 min | 5 min | **71 min** |
| i4i NVMe, 8 workers | 32 min | 16 min | 5 min | **53 min** |
| i4i NVMe, 16 workers | 32 min | 15 min | 5 min | **52 min** |

**PostgreSQL parallel build configuration:**

```sql
-- Enable parallel index build
SET max_parallel_maintenance_workers = 8;  -- Workers for CREATE INDEX
SET maintenance_work_mem = '8GB';          -- Memory per worker

-- Create index with parallel workers
CREATE INDEX CONCURRENTLY ON documents
USING meerkat (embedding vector_cosine_ops)
WITH (workers = 8);
```

**Recommendation:** Use 8 workers on i4i NVMe for ~50 min builds. Beyond 8
workers, I/O becomes the bottleneck and additional CPU provides diminishing
returns. For faster builds, the clustering phase (32 min) becomes dominant—
consider pre-computed centroids or incremental builds for frequent rebuilds.

#### Multi-Tenant Scaling

For composite key indexes with N tenants:

| Tenants | Vectors/tenant | Cluster overhead | Query isolation |
|---------|----------------|------------------|-----------------|
| 10 | 100M | Minimal | Full |
| 100 | 10M | ~10% | Full |
| 1,000 | 1M | ~15% | Full |
| 10,000 | 100K | ~25% | Full |

Per-tenant query cost is independent of total dataset size—a query for a
tenant with 1M vectors has the same cost whether the total index has 1B or
10B vectors.

#### Comparison with SPFresh

| Metric | SPFresh | Meerkat (EBS) | Meerkat (i4i) | Meerkat In-Index |
|--------|---------|---------------|---------------|------------------|
| Dimensions | 100 | 768 | 768 | 768 |
| Element type | int8 | float16 | float16 | float16 |
| Vector size | 100 B | 1,536 B | 1,536 B | 1,536 B |
| Heap storage | NVMe | EBS gp3 | i4i NVMe | N/A |
| Query p50 | 2-4ms | ~17ms | **~5ms** | **~4ms** |
| Query p99 | 8-12ms | ~40ms | **~12ms** | **~10ms** |
| Index size | ~100 GB | 260 GB | 260 GB | 3.2 TB |
| Heap size | ~100 GB | 1.6 TB | 1.6 TB | 0 |
| Total storage | ~200 GB | 1.9 TB | 1.9 TB | 3.2 TB |
| Build time | N/A | ~1 hour | ~30 min | ~1 hour |
| Memory | 10 GB | 32-64 GB | 32-64 GB | 64-128 GB |
| Instance cost | N/A | ~$230/mo | ~$730/mo | ~$1,100/mo |

**Analysis:**

- **SPFresh baseline**: 100d int8 vectors are 15× smaller than 768d float16.
  Both systems are I/O-bound on posting scan; RaBitQ compute is negligible.

- **Meerkat on EBS**: Cost-effective at ~$230/month, but EBS IOPS limits
  re-ranking latency to ~15ms. Good for throughput-oriented workloads.

- **Meerkat on i4i NVMe**: Achieves **5ms p50**—matching SPFresh despite 15×
  larger vectors. Proves the architecture scales efficiently with vector size.

- **Vectors in index**: Best latency at **3ms p50** by eliminating heap access.
  ~2× storage cost. Competitive with SPFresh at any vector size.

**Recommendations by workload:**

| Workload | Configuration | Instance | p50 | Cost/month |
|----------|---------------|----------|-----|------------|
| Cost-sensitive | Inline + EBS | r6i.4xlarge | 17ms | ~$500 |
| Balanced | Inline + i4i | i4i.4xlarge | 5ms | ~$1,000 |
| Latency-critical | In-index | i4i.8xlarge | 4ms | ~$2,000 |

*Latencies assume async I/O (io_uring) or contiguous posting lists. Fragmented
indexes without async I/O add ~25ms to posting scan.*
| Latency-critical (<30ms p99) | Vectors in index | No heap access |
| Cost-sensitive | Inline (768d, float16) | Smallest total storage |

## Future Extensions

1. **Product quantization**: Even better compression for very high dimensions
2. **Graph-based leaf refinement**: HNSW within large clusters
3. **Tiered storage**: Extend to cloud storage for cold data
4. **Filtered search**: Efficient pre-filtering with predicates
5. **Cross-tenant queries**: Aggregate search across multiple tenants

## References

**Research papers:**

- [SPANN paper][spann] - Disk-based IVF with boundary replication
- [SPFresh paper][spfresh] - LIRE protocol for incremental updates
- [ScaNN for AlloyDB whitepaper][scann-alloydb] - Hierarchical clustering
- [RaBitQ paper][rabitq] - Binary quantization with error bounds

**Implementation references:**

- [turbopuffer ANN v3][turbopuffer] - Production IVF lessons
- [napkin-math][napkin] - Systems performance reference numbers
- [AWS i4i instances][aws-i4i] - Local NVMe storage specs
- [AWS i3en instances][aws-i3en] - High-density NVMe storage

[spann]: https://www.microsoft.com/en-us/research/wp-content/uploads/2021/11/SPANN_finalversion1.pdf
[spfresh]: https://arxiv.org/pdf/2410.14452
[scann-alloydb]: https://services.google.com/fh/files/misc/scann_for_alloydb_whitepaper.pdf
[rabitq]: https://arxiv.org/abs/2405.12497
[turbopuffer]: https://turbopuffer.com/blog/ann-v3
[napkin]: https://github.com/sirupsen/napkin-math
[aws-i4i]: https://aws.amazon.com/ec2/instance-types/i4i/
[aws-i3en]: https://aws.amazon.com/ec2/instance-types/i3en/
