# PRISM Architecture

High-level architecture for PRISM, the pg_vectorsearch extension's
PostgreSQL index access method for approximate nearest neighbor (ANN)
vector search.

## Overview

PRISM is designed for **billion-scale vector search** within PostgreSQL,
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

pg_vectorsearch is designed as part of a unified search stack alongside
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
│     Format-aware distance: RaBitQ approximate with error        │
│     bounds, float32/float16 exact L2 — no reranking needed      │
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

To handle billion-scale datasets, PRISM uses hierarchical clustering inspired
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

**Centroid representation**: Centroids are stored in the index, not as heap
TIDs. Each entry is an 8-byte header (`child_blkno`, `child_count`, `flags`)
plus a format-dependent payload. An internal node points at a child centroid
page; a leaf points at a posting-list head. There is no medoid TID.

### 2. Centroid Pages in Shared Buffers

Centroid data is stored in **dedicated centroid pages** within the index,
separate from posting list pages. These pages live in PostgreSQL's standard
shared buffer cache—there is no separate dedicated cache structure.

**Why shared buffers work well for centroids**:
- Centroid pages are accessed on every query (hot data)
- Frequently accessed pages naturally stay in the buffer cache
- Standard PostgreSQL infrastructure: locking, WAL, visibility
- No custom shared memory allocation or startup coordination

**Page layout**: Centroid pages use a bidirectional layout with per-entry
metadata growing forward and vector data growing backward. Each page stores
centroids in one of three data formats, selected at page initialization and
recorded in the opaque flags:

| Format | Data encoding | Use case |
|--------|---------------|----------|
| **RaBitQ** | Quantized bits + f_add/f_rescale | Compact, approximate, with error bounds |
| **Fastscan** | The same RaBitQ codes, packed for SIMD lookup | Default when a group fits on a page |
| **Float32** | Full-precision float vectors | Exact routing |
| **Float16** | Half-precision float vectors | Near-exact routing, more entries per page |

Metadata is 8 bytes in every format. RaBitQ pages encode centroids relative
to the global data mean. The query is transformed once and reused at every
tree level. Float and half pages store the centroid itself and compute exact
distances, so routing does not rerank them. Fastscan is a layout of the
RaBitQ codes, not a different quantizer.

**Sizing** (768 dimensions, 8KB pages, 8-byte metadata):

| Format | Entry size | Entries per page |
|--------|-----------|-----------------|
| RaBitQ | 112B (8B meta + 104B data) | ~72 |
| Float16 | 1544B (8B meta + 1536B data) | 5 |
| Float32 | 3080B (8B meta + 3072B data) | 2 |

Fastscan stores centroids in 32-wide groups, so its capacity is a group
count rather than an entry size.

For billion-scale indexes (RaBitQ format):
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

**PRISM approach:**

Both are build options, not a single `max_replicas` knob (that reloption
does not exist):

- `boundary_epsilon` (default 0.35) is the SPANN-style band. `0` disables it.
- `soar_lambda` (default 1.0) is the SOAR secondary assignment. `0` disables it.

Foreground inserts assign a new row to one list. Replication is applied at
build time. A later compaction or `REINDEX` is what would repay that for rows
inserted since the build; that compaction is not implemented.

#### Assignment methods: tree descent vs. brute-force

Independent of *which* clusters a vector is replicated to, there is the
question of *how* the assignment is computed. PRISM deliberately uses two
different methods for the two kinds of assignment, because they are different
problems:

**Primary assignment → tree descent.** The primary (home) cluster is the
nearest centroid, found by a greedy descent of the k-means tree
(`O(fan_out · nlevels)` distance evaluations along one root-to-leaf path).
This is chosen for two reasons:

1. *Cost.* For the primary we only need the (approximate) nearest centroid,
   and a greedy descent finds it in roughly `log(nlist)` work instead of
   scanning all leaves.
2. *Consistency with the query path.* Queries route to clusters by descending
   the same tree. Assigning vectors the same way makes a vector and a nearby
   query land in the same cluster. Assigning by *exact* nearest instead would
   create a build/query routing mismatch — a vector could sit in its
   exact-nearest leaf that a nearby query's greedy descent never reaches —
   which measurably lowers recall. Benchmarks confirm exact (brute-force)
   primary assignment is both slower and slightly lower recall than tree
   descent, and the speed gap widens with `nlist`.

**Secondary assignment (boundary + SOAR) → batched brute-force SIMD.** The
replica search is inherently a full-scan problem: boundary replication needs
the *2nd-nearest* centroid, and SOAR needs the centroid minimizing an
orthogonality-amplified distance whose optimum need *not* be among the
nearest-by-distance leaves — so a tree cannot prune to it, and the search must
run over the flat centroid set. The implementation batches B vectors and
computes their distances to all centroids with a single `sgemm`, streamed in
cache-sized centroid tiles, so the centroid block is read once per batch and
the kernel is compute-bound. This was measured faster than a per-vector tree
beam at every `nlist` tested (1k–64k), with the gap *widening* at scale: the
beam's per-vector, random-access scan degrades as centroids spill from cache,
while the tiled GEMM streams sequentially. For the boundary case, recall is
identical to the beam (primary stays tree descent in both, and the beam finds
the same close 2nd-nearest), so brute-force is strictly better here — faster
with no recall cost. A hierarchy would only help in a fundamentally different
regime (billion-scale tiered storage, where the centroid set itself exceeds
memory).

When no BLAS library is available, a fallback per-vector path is used instead:
boundary via a wider tree beam search, SOAR via a per-vector SIMD full scan.
Correct but slower; only used when the GEMM kernel is unavailable.

**Balancing**: Ideally, posting lists should be roughly equal in size for
predictable query latency. The clustering algorithm aims for balanced clusters,
but natural data distribution may cause imbalance. Very large clusters can be
split; very small clusters may be merged or eliminated.

### 5. Vector Quantization (RaBitQ)

PRISM uses **RaBitQ** (Random Bit Quantization) for vector compression, which
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

Full-precision vectors are stored in the heap of the indexed table. There is
no separate vector heap in the index. `vec32` and `vec16` default to
`STORAGE EXTERNAL`: a value over the ~2 kB toast threshold is stored out of
line and not compressed. `STORAGE PLAIN` keeps it inline when the row fits on
a page, which is what a rerank wants. Compression does not help dense
embeddings, so `MAIN` is not a substitute.

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
- `nlist`, `fan_out`, `kmeans_nredo`
- `distance_mode` (`asymmetric` or `symmetric`, default `asymmetric`)
- `soar_lambda`, `boundary_epsilon`
- `centroid_compression`, `fastscan`, `centroid_fastscan`

`nprobe` is not a reloption. There is no `fillfactor`, `reserved_pages`,
`rerank_k`, or `tenant_column`.

**GUCs** (session-level):
- `prism.nprobe` — clusters to search. `0` derives it from the index.
- `prism.query_limit` — cap on the top-k the scan sizes from `LIMIT`.
- `prism.rerank`, `prism.rerank_pool` — exact heap rerank, and its pool cap.
- `prism.distance_mode` — `default` uses the index setting; `asymmetric` or
  `symmetric` overrides it for the session.
- `prism.fastscan_bits`, `prism.centroid_beam_scale`,
  `prism.centroid_error_scale`, `prism.probe_expand`, `prism.recent_buffers`
- `prism.leaf_refine_threshold`, `prism.log_build_stats` — build time only.

**Catalog tables** (managed by PostgreSQL):
- Structural info in pg_class, pg_index, pg_am, pg_opclass

**Metapage** (stored in index file, persists with the index):
- Root centroid page, first posting page, `nlist`, `nlevels`, `fan_out`
- Centroid format, distance metric, RaBitQ seed, global mean
- No per-leaf posting directory. A leaf centroid entry holds its list head.
- Statistics: per-cluster live counts live on each posting list's head page,
  not here -- inserts already hold that page, so they can maintain them, while
  a metapage counter would serialize every insert on block 0

## Multi-Tenant Support

The access method is single-column (`amcanmulticol` is false), so a composite
key index is not implemented. Two deployment shapes are still the design:

### Option 1: Index-per-Tenant

Create separate tables and indexes for each tenant:

```sql
CREATE TABLE tenant_123_vectors (id bigint, embedding vec32(768));
CREATE INDEX ON tenant_123_vectors USING prism (embedding);
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
    embedding vec32(768)
);
CREATE INDEX ON vectors USING prism ((tenant_id, embedding));
```

**Not implemented.** The intended shape is:
- The hierarchical centroid tree is segmented by `tenant_id` at the root level
- Each tenant has its own subtree of centroids and posting lists
- Queries specify `tenant_id` and route directly to that tenant's subtree
- No cross-tenant centroid comparisons or posting list scans

```sql
-- Query automatically routes to tenant 42's subtree
SELECT * FROM vectors
WHERE tenant_id = 42
ORDER BY embedding <-> '[...]'::vec32
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
(`prism_centroid_beam_search`) rather than best-first search. This maps well
to PostgreSQL's page-based buffer cache: each level is processed as a batch,
enabling SIMD distance computation on entries within each page.
Upper-level pages stay hot in `shared_buffers` since they are accessed on
every query.

**Parameters**:
- `prism.nprobe`: leaf posting lists to scan. `0` derives it from the index.
- `prism.centroid_beam_scale`: routing beam as a fraction of nprobe, not a
  fixed `beam_width`.
- `prism.rerank` / `prism.rerank_pool`: exact heap rerank and its pool cap.
  There is no `rerank_k` reloption.

### Dynamic Updates (LIRE Protocol)

PRISM adopts the **LIRE (Lightweight Incremental RE-balancing)** protocol
from SPFresh for maintaining index quality under continuous updates without
full rebuilds.

> **What is implemented.** Incremental `INSERT`, `DELETE`, and `UPDATE` work.
> `aminsert` routes a new vector to one leaf and appends it (an AoS overflow
> page, including on fastscan indexes, whose packed base is immutable). The
> scan reads each page in that page's own format. `DELETE` is correct because
> the executor rechecks heap visibility; `VACUUM` tombstones dead entries
> (`PRISM_POSTING_FLAG_DELETED` for AoS, a per-group `tombstone_mask` bit for
> FASTSCAN lanes) before the heap can recycle their line pointers. An update
> of the vector column is insert-new
> plus delete-old; an update that leaves the vector unchanged is HOT.
> Centroids do not move with inserts, so churn drifts the partitioning.
> `REINDEX` rebuilds it. `prism_rebalance` and `prism_split_posting_list` split
> oversized lists on demand, and only for a flat index with RaBitQ centroid
> pages. Merge, reassign, a version byte, and a background worker are not
> implemented. The subsections below describe that protocol; only split exists,
> and only in the limited form just stated.

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

Not implemented. The intended interface is an EXPLAIN option:

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
│   - No posting-list directory and no tenant directory           │
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
- Format-dependent vector encoding (RaBitQ quantized, float32 exact, or
  float16 near-exact) — selected per page at initialization
- Per-entry metadata with child pointers (page numbers for next level, or
  posting list heads for leaves)
- Level indicator and sibling link for navigation

Centroid pages are stored in PostgreSQL's standard shared buffer cache. Because
they are accessed on every query, they naturally remain cached (hot pages).
There is no separate dedicated cache structure—just standard buffer management.

**Multi-tenant layout** is not implemented. The intended root would be a
tenant directory, with each tenant's subtree stored contiguously:

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

Not implemented. `fillfactor` and `reserved_pages` are not reloptions. Inserts
that do not fit the current posting page allocate a new page at the end of the
index. A future build could leave free space or pre-link empty pages per list
to delay that. The tradeoff is wasted space against later random I/O.

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

Registered callbacks:
- `ambuild` / `ambuildparallel`: build from a heap scan, serially or in parallel
- `aminsert`: append a vector to one posting list
- `ambulkdelete` / `amvacuumcleanup`: tombstone dead entries
- `amgettuple`: return search results. `amgetbitmap` is not implemented.
- `amcostestimate`: planner cost

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

- The index does not store an XID per posting entry. The executor rechecks
  heap visibility, which is what makes a delete correct immediately.
- `VACUUM` sets `PRISM_POSTING_FLAG_DELETED` on dead entries. Scans skip that
  flag. Physical reclaim of the bytes is not done yet.

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


[spann]: https://www.microsoft.com/en-us/research/wp-content/uploads/2021/11/SPANN_finalversion1.pdf
[spfresh]: https://arxiv.org/pdf/2410.14452
[scann-alloydb]: https://services.google.com/fh/files/misc/scann_for_alloydb_whitepaper.pdf
[rabitq]: https://arxiv.org/abs/2405.12497
