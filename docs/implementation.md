# Meerkat Implementation Guide

Detailed implementation specification for Meerkat, organized for iterative
development. Each section builds on previous work, with standalone components
developed and tested before PostgreSQL integration.

## Guiding Principles

### Performance

1. **SIMD first**: All vector operations must have SIMD implementations from
   day one. Non-SIMD fallbacks exist only for correctness testing.

2. **Memory hierarchy awareness**: Design data structures for cache efficiency.
   Sequential access patterns. Minimize pointer chasing.

3. **Batch operations**: Process vectors in batches to amortize function call
   overhead and enable SIMD across multiple vectors.

4. **Zero-copy where possible**: Avoid copying vector data. Use pointers/slices
   into existing buffers.

### Scale

1. **Streaming builds**: Index construction must not require all vectors in
   memory. Process in chunks, write incrementally.

2. **Bounded memory**: All operations have predictable memory bounds controlled
   by configuration parameters.

3. **Horizontal partitioning**: Large indexes naturally partition into
   independent clusters that can be processed in parallel.

### PostgreSQL Integration

1. **Buffer cache native**: Posting list pages are standard PostgreSQL pages
   managed by the buffer cache. No custom page management.

2. **WAL logged**: All modifications are WAL logged for crash recovery and
   replication support.

3. **MVCC compliant**: Visibility checks use standard PostgreSQL snapshot
   mechanisms.

4. **Planner integration**: Cost estimates enable the query planner to choose
   between index scan and sequential scan appropriately.

---

## Component Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PostgreSQL Integration                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────────┐  │
│  │ IAM Handler │  │ Page Layout │  │ Shmem Cache │  │ Cost Estimate │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └───────────────┘  │
├─────────────────────────────────────────────────────────────────────────┤
│                          Index Operations                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────────┐  │
│  │ Index Build │  │   Search    │  │   Insert    │  │    Vacuum     │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └───────────────┘  │
├─────────────────────────────────────────────────────────────────────────┤
│                         Core Algorithms                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────────┐  │
│  │  Distance   │  │ Quantization│  │  Clustering │  │    Top-K      │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └───────────────┘  │
├─────────────────────────────────────────────────────────────────────────┤
│                         Foundation                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────────┐  │
│  │    SIMD     │  │   Memory    │  │  Platform   │  │    Types      │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

Components are developed bottom-up. Foundation and Core Algorithms are
standalone C libraries testable without PostgreSQL.

---

## Part 1: Foundation Layer

### 1.1 Type Definitions

**File**: `src/mkt_types.h`

```c
#include <stdint.h>
#include <stdbool.h>

// Vector element type (matches pgvector)
typedef float Vector32;

// Quantized representations
typedef uint8_t  ScalarQ8;   // 8-bit scalar quantized
typedef uint8_t  BinaryQ;    // Binary quantized (1 bit per dim, packed)

// Dimension type (max 65535 dimensions)
typedef uint16_t Dimension;

// Cluster/centroid identifier
typedef uint32_t ClusterId;

// Distance type (always float for intermediate computations)
typedef float Distance;

// Vector reference (pointer + dimension, no ownership)
typedef struct {
    const Vector32 *data;
    Dimension       dim;
} VectorRef;

// Mutable vector (for building/modifying)
typedef struct {
    Vector32  *data;
    Dimension  dim;
} VectorMut;

// Distance metric enum
typedef enum {
    DISTANCE_L2,           // Euclidean (L2 squared)
    DISTANCE_INNER_PRODUCT,// Negative inner product (for max similarity)
    DISTANCE_COSINE        // 1 - cosine similarity
} DistanceMetric;
```

**Testability**: Header-only, no tests needed. Used by all other components.

### 1.2 Memory Abstraction

**File**: `src/mkt_memory.h`, `src/mkt_memory.c`

Abstraction layer allowing the same code to run with PostgreSQL's memory
contexts or standard malloc/free.

```c
// Memory context handle (opaque)
typedef struct MktMemoryContext *MktMemCtx;

// Global context for current allocation scope
extern MktMemCtx mkt_current_memctx;

// Allocation functions
void *mkt_alloc(size_t size);
void *mkt_alloc0(size_t size);  // Zero-initialized
void *mkt_realloc(void *ptr, size_t size);
void  mkt_free(void *ptr);

// Aligned allocation (for SIMD)
void *mkt_alloc_aligned(size_t size, size_t alignment);
void  mkt_free_aligned(void *ptr);

// Context management
MktMemCtx mkt_memctx_create(MktMemCtx parent, const char *name);
void        mkt_memctx_delete(MktMemCtx ctx);
void        mkt_memctx_reset(MktMemCtx ctx);  // Free all allocations

// Scoped context (RAII-style via cleanup attribute)
#define MKT_MEMCTX_SCOPE(name) \
    MktMemCtx name __attribute__((cleanup(mkt_memctx_delete_ptr))) = \
        mkt_memctx_create(mkt_current_memctx, #name)
```

**Standalone implementation** (`mkt_memory_standalone.c`):
- Uses standard `malloc`/`free`
- Memory contexts track allocations in a linked list for bulk free
- Aligned allocation via `aligned_alloc` or `posix_memalign`

**PostgreSQL implementation** (`mkt_memory_pg.c`):
- Maps to `palloc`/`pfree`
- Memory contexts map to PostgreSQL MemoryContexts
- Aligned allocation uses `MemoryContextAllocAligned` (PG16+) or manual alignment

**Tests**: Unit tests verify allocation, context creation/deletion, bulk reset.

### 1.3 Platform Abstraction

**File**: `src/mkt_platform.h`

```c
// SIMD capability detection
typedef enum {
    SIMD_NONE    = 0,
    SIMD_SSE2    = 1 << 0,
    SIMD_SSE4_1  = 1 << 1,
    SIMD_AVX2    = 1 << 2,
    SIMD_AVX512F = 1 << 3,
    SIMD_NEON    = 1 << 4,
} SimdCapability;

// Detect CPU capabilities at runtime
SimdCapability mkt_detect_simd(void);

// Cache line size (typically 64 bytes)
#define MKT_CACHE_LINE 64

// Prefetch hints
#define mkt_prefetch_read(addr)  __builtin_prefetch((addr), 0, 3)
#define mkt_prefetch_write(addr) __builtin_prefetch((addr), 1, 3)

// Likely/unlikely branch hints
#define mkt_likely(x)   __builtin_expect(!!(x), 1)
#define mkt_unlikely(x) __builtin_expect(!!(x), 0)

// Compiler barriers
#define mkt_compiler_barrier() __asm__ __volatile__("" ::: "memory")
```

**Tests**: Verify SIMD detection matches actual CPU capabilities.

---

## Part 2: Core Algorithms

### 2.1 Distance Computation

**Files**: `src/distance.h`, `src/distance.c`, `src/distance_avx512.c`,
`src/distance_neon.c`

Distance computation is the most performance-critical operation. Multiple
implementations selected at runtime based on CPU capabilities.

#### Interface

```c
// Single vector pair distance
Distance mkt_distance_l2(VectorRef a, VectorRef b);
Distance mkt_distance_ip(VectorRef a, VectorRef b);  // Inner product
Distance mkt_distance_cosine(VectorRef a, VectorRef b);

// Generic dispatch
Distance mkt_distance(VectorRef a, VectorRef b, DistanceMetric metric);

// Batch: distances from one query to multiple vectors
// Results written to `distances` array (must be pre-allocated)
void mkt_distance_batch(
    VectorRef query,
    const Vector32 *vectors,  // Contiguous array of vectors
    uint32_t count,
    Dimension dim,
    DistanceMetric metric,
    Distance *distances       // Output: count distances
);

// Batch with early termination: stop when found k vectors below threshold
// Returns number of vectors actually processed
uint32_t mkt_distance_batch_threshold(
    VectorRef query,
    const Vector32 *vectors,
    uint32_t count,
    Dimension dim,
    DistanceMetric metric,
    Distance threshold,
    uint32_t k,
    Distance *distances,
    uint32_t *indices         // Output: indices of qualifying vectors
);
```

#### L2 Distance Implementation (AVX-512)

```c
// Process 16 floats per iteration
Distance mkt_distance_l2_avx512(VectorRef a, VectorRef b) {
    const Vector32 *pa = a.data;
    const Vector32 *pb = b.data;
    Dimension dim = a.dim;

    __m512 sum = _mm512_setzero_ps();

    // Main loop: 16 elements per iteration
    Dimension i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m512 va = _mm512_loadu_ps(pa + i);
        __m512 vb = _mm512_loadu_ps(pb + i);
        __m512 diff = _mm512_sub_ps(va, vb);
        sum = _mm512_fmadd_ps(diff, diff, sum);
    }

    // Horizontal sum
    Distance result = _mm512_reduce_add_ps(sum);

    // Scalar tail
    for (; i < dim; i++) {
        Distance diff = pa[i] - pb[i];
        result += diff * diff;
    }

    return result;
}
```

#### Batch Distance with Prefetching

```c
void mkt_distance_batch_l2_avx512(
    VectorRef query,
    const Vector32 *vectors,
    uint32_t count,
    Dimension dim,
    Distance *distances
) {
    const size_t vector_bytes = dim * sizeof(Vector32);
    const size_t prefetch_ahead = 4;  // Prefetch 4 vectors ahead

    for (uint32_t i = 0; i < count; i++) {
        // Prefetch future vectors
        if (i + prefetch_ahead < count) {
            mkt_prefetch_read(vectors + (i + prefetch_ahead) * dim);
        }

        VectorRef v = { .data = vectors + i * dim, .dim = dim };
        distances[i] = mkt_distance_l2_avx512(query, v);
    }
}
```

#### Function Dispatch

At initialization, select optimal implementation based on CPU:

```c
typedef Distance (*DistanceFn)(VectorRef, VectorRef);
typedef void (*DistanceBatchFn)(VectorRef, const Vector32*, uint32_t,
                                 Dimension, Distance*);

static DistanceFn      g_distance_l2_fn;
static DistanceBatchFn g_distance_batch_l2_fn;

void mkt_distance_init(void) {
    SimdCapability caps = mkt_detect_simd();

    if (caps & SIMD_AVX512F) {
        g_distance_l2_fn = mkt_distance_l2_avx512;
        g_distance_batch_l2_fn = mkt_distance_batch_l2_avx512;
    } else if (caps & SIMD_AVX2) {
        g_distance_l2_fn = mkt_distance_l2_avx2;
        g_distance_batch_l2_fn = mkt_distance_batch_l2_avx2;
    } else if (caps & SIMD_NEON) {
        g_distance_l2_fn = mkt_distance_l2_neon;
        g_distance_batch_l2_fn = mkt_distance_batch_l2_neon;
    } else {
        g_distance_l2_fn = mkt_distance_l2_scalar;
        g_distance_batch_l2_fn = mkt_distance_batch_l2_scalar;
    }
}
```

**Tests**:
- Correctness: Compare SIMD results against scalar reference implementation
- Accuracy: Verify numerical precision within acceptable bounds
- Performance: Benchmark each implementation, verify SIMD speedup
- Edge cases: Zero-length vectors, single element, non-aligned pointers

**CLI Tool**: `mkt_distance_bench` - benchmark distance computations

```
$ mkt_distance_bench --dim 768 --count 10000 --metric l2
L2 distance (dim=768, count=10000):
  scalar:  45.2 ms (221k vec/s)
  avx2:    8.1 ms (1.23M vec/s)
  avx512:  4.3 ms (2.33M vec/s)
```

### 2.2 Quantization

**Files**: `src/quantize.h`, `src/quantize.c`, `src/quantize_avx512.c`

Quantization compresses vectors for faster approximate distance computation.
Meerkat uses scalar quantization (SQ8) as the primary method, with binary
quantization (BQ) as an optional faster but less accurate alternative.

#### Scalar Quantization (SQ8)

Each dimension is linearly mapped from [min, max] to [0, 255].

```c
// Quantization parameters (learned from data)
typedef struct {
    Vector32 *mins;    // Per-dimension minimums
    Vector32 *maxs;    // Per-dimension maximums
    Vector32 *scales;  // (max - min) / 255 per dimension
    Dimension dim;
} SQ8Params;

// Learn quantization parameters from a sample of vectors
SQ8Params *mkt_sq8_learn(
    const Vector32 *vectors,
    uint32_t count,
    Dimension dim
);

// Quantize a single vector
void mkt_sq8_encode(
    const SQ8Params *params,
    VectorRef input,
    ScalarQ8 *output          // dim bytes
);

// Quantize multiple vectors (batch)
void mkt_sq8_encode_batch(
    const SQ8Params *params,
    const Vector32 *inputs,
    uint32_t count,
    Dimension dim,
    ScalarQ8 *outputs         // count * dim bytes
);

// Approximate L2 distance between quantized vectors
// Uses lookup table for speed
Distance mkt_sq8_distance_l2(
    const ScalarQ8 *a,
    const ScalarQ8 *b,
    const SQ8Params *params
);

// Asymmetric distance: float query vs quantized vector
// More accurate than symmetric (both quantized)
Distance mkt_sq8_distance_asymmetric_l2(
    VectorRef query,          // Full precision
    const ScalarQ8 *quantized,
    const SQ8Params *params
);
```

#### SQ8 Encoding (AVX-512)

```c
void mkt_sq8_encode_avx512(
    const SQ8Params *params,
    VectorRef input,
    ScalarQ8 *output
) {
    const Vector32 *mins = params->mins;
    const Vector32 *scales = params->scales;
    Dimension dim = params->dim;

    Dimension i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m512 v = _mm512_loadu_ps(input.data + i);
        __m512 vmin = _mm512_loadu_ps(mins + i);
        __m512 vscale = _mm512_loadu_ps(scales + i);

        // Normalize to [0, 255]
        __m512 normalized = _mm512_div_ps(
            _mm512_sub_ps(v, vmin),
            vscale
        );

        // Clamp and convert to int
        normalized = _mm512_max_ps(normalized, _mm512_setzero_ps());
        normalized = _mm512_min_ps(normalized, _mm512_set1_ps(255.0f));
        __m512i vi = _mm512_cvtps_epi32(normalized);

        // Pack to bytes (16 floats -> 16 bytes)
        // AVX-512 doesn't have direct 32->8 pack, use permute
        __m128i packed = _mm512_cvtepi32_epi8(vi);
        _mm_storeu_si128((__m128i*)(output + i), packed);
    }

    // Scalar tail
    for (; i < dim; i++) {
        float normalized = (input.data[i] - mins[i]) / scales[i];
        normalized = fmaxf(0.0f, fminf(255.0f, normalized));
        output[i] = (ScalarQ8)normalized;
    }
}
```

#### Asymmetric Distance (Query: float, DB: quantized)

```c
// Precompute lookup table for query
typedef struct {
    float tables[256];  // For each possible quantized value
} SQ8LUT;

// Build lookup table for one dimension
void mkt_sq8_build_lut(
    float query_val,
    float min,
    float scale,
    float *lut  // 256 entries
) {
    for (int i = 0; i < 256; i++) {
        float reconstructed = min + i * scale;
        float diff = query_val - reconstructed;
        lut[i] = diff * diff;  // For L2
    }
}

// Distance using precomputed LUTs
Distance mkt_sq8_distance_lut(
    const SQ8LUT *luts,       // dim LUTs
    const ScalarQ8 *quantized,
    Dimension dim
) {
    Distance sum = 0;
    for (Dimension i = 0; i < dim; i++) {
        sum += luts[i].tables[quantized[i]];
    }
    return sum;
}
```

#### Binary Quantization (Optional)

For very fast approximate filtering before SQ8 re-ranking:

```c
// Each dimension: 1 if >= centroid, 0 otherwise
// Packed: 8 dimensions per byte
typedef struct {
    Vector32 *thresholds;  // Per-dimension thresholds (typically mean)
    Dimension dim;
    uint32_t packed_bytes; // ceil(dim / 8)
} BQParams;

void mkt_bq_encode(const BQParams *params, VectorRef input, BinaryQ *output);

// Hamming distance between binary vectors
uint32_t mkt_bq_distance_hamming(
    const BinaryQ *a,
    const BinaryQ *b,
    uint32_t packed_bytes
);
```

**Tests**:
- Round-trip accuracy: encode, decode, measure reconstruction error
- Distance preservation: verify quantized distances correlate with true distances
- Parameter learning: verify min/max capture data distribution
- Performance: benchmark encode/decode and distance operations

### 2.3 Top-K Selection

**Files**: `src/topk.h`, `src/topk.c`

Efficiently find the K smallest distances from a large set.

#### Interface

```c
// Result entry: distance + identifier
typedef struct {
    Distance distance;
    uint32_t id;  // Vector index, TID, or cluster ID
} TopKEntry;

// Top-K heap (max-heap of K smallest elements)
typedef struct {
    TopKEntry *entries;
    uint32_t   capacity;  // K
    uint32_t   count;     // Current size (<= K)
} TopKHeap;

// Create heap with capacity K
TopKHeap *mkt_topk_create(uint32_t k);
void      mkt_topk_destroy(TopKHeap *heap);
void      mkt_topk_reset(TopKHeap *heap);

// Insert candidate (O(log K) if heap full, O(1) if distance > max)
void mkt_topk_insert(TopKHeap *heap, Distance distance, uint32_t id);

// Get current threshold (max distance in heap, or INFINITY if not full)
Distance mkt_topk_threshold(const TopKHeap *heap);

// Extract results sorted by distance (ascending)
void mkt_topk_extract_sorted(TopKHeap *heap, TopKEntry *results);

// Batch insert with threshold check
// Only inserts candidates with distance < current threshold
void mkt_topk_insert_batch(
    TopKHeap *heap,
    const Distance *distances,
    uint32_t count,
    uint32_t id_offset  // Added to index to get ID
);
```

#### Implementation

```c
struct TopKHeap {
    TopKEntry *entries;
    uint32_t   capacity;
    uint32_t   count;
};

// Max-heap: parent >= children
static inline uint32_t parent(uint32_t i) { return (i - 1) / 2; }
static inline uint32_t left(uint32_t i)   { return 2 * i + 1; }
static inline uint32_t right(uint32_t i)  { return 2 * i + 2; }

static void sift_down(TopKHeap *heap, uint32_t i) {
    while (true) {
        uint32_t largest = i;
        uint32_t l = left(i);
        uint32_t r = right(i);

        if (l < heap->count &&
            heap->entries[l].distance > heap->entries[largest].distance)
            largest = l;
        if (r < heap->count &&
            heap->entries[r].distance > heap->entries[largest].distance)
            largest = r;

        if (largest == i) break;

        // Swap
        TopKEntry tmp = heap->entries[i];
        heap->entries[i] = heap->entries[largest];
        heap->entries[largest] = tmp;
        i = largest;
    }
}

void mkt_topk_insert(TopKHeap *heap, Distance distance, uint32_t id) {
    if (heap->count < heap->capacity) {
        // Heap not full: insert and sift up
        uint32_t i = heap->count++;
        heap->entries[i] = (TopKEntry){ .distance = distance, .id = id };

        while (i > 0 && heap->entries[parent(i)].distance < distance) {
            heap->entries[i] = heap->entries[parent(i)];
            i = parent(i);
        }
        heap->entries[i] = (TopKEntry){ .distance = distance, .id = id };
    } else if (distance < heap->entries[0].distance) {
        // Better than worst in heap: replace root and sift down
        heap->entries[0] = (TopKEntry){ .distance = distance, .id = id };
        sift_down(heap, 0);
    }
    // Otherwise: distance >= threshold, ignore
}

Distance mkt_topk_threshold(const TopKHeap *heap) {
    if (heap->count < heap->capacity) return INFINITY;
    return heap->entries[0].distance;
}
```

#### Batch Insert Optimization

```c
void mkt_topk_insert_batch(
    TopKHeap *heap,
    const Distance *distances,
    uint32_t count,
    uint32_t id_offset
) {
    Distance threshold = mkt_topk_threshold(heap);

    for (uint32_t i = 0; i < count; i++) {
        if (distances[i] < threshold) {
            mkt_topk_insert(heap, distances[i], id_offset + i);
            threshold = mkt_topk_threshold(heap);
        }
    }
}
```

**Tests**:
- Correctness: verify top-K matches brute-force sort
- Edge cases: K=1, K > count, all equal distances
- Performance: benchmark with various K and count values

### 2.4 K-Means Clustering

**Files**: `src/kmeans.h`, `src/kmeans.c`

Clustering is used during index build to partition vectors into clusters.

#### Interface

```c
// Clustering result
typedef struct {
    Vector32   *centroids;     // nlist * dim floats
    ClusterId  *assignments;   // nvecs cluster assignments
    uint32_t   *cluster_sizes; // nlist sizes
    uint32_t    nlist;         // Number of clusters
    Dimension   dim;
} KMeansResult;

// Clustering options
typedef struct {
    uint32_t max_iterations;   // Max Lloyd iterations (default: 20)
    float    tolerance;        // Convergence threshold (default: 1e-4)
    uint64_t seed;             // Random seed for initialization
    bool     verbose;          // Print progress
} KMeansOptions;

// Standard k-means
KMeansResult *mkt_kmeans(
    const Vector32 *vectors,
    uint32_t nvecs,
    Dimension dim,
    uint32_t nlist,
    DistanceMetric metric,
    const KMeansOptions *options
);

// Streaming k-means for large datasets
// Processes vectors in chunks, maintains running centroids
typedef struct MktKMeansStream MktKMeansStream;

MktKMeansStream *mkt_kmeans_stream_create(
    uint32_t nlist,
    Dimension dim,
    DistanceMetric metric,
    const KMeansOptions *options
);

void mkt_kmeans_stream_add(
    MktKMeansStream *stream,
    const Vector32 *vectors,
    uint32_t count
);

KMeansResult *mkt_kmeans_stream_finish(MktKMeansStream *stream);
void mkt_kmeans_stream_destroy(MktKMeansStream *stream);

// Find medoid for each cluster (actual vector closest to centroid)
void mkt_kmeans_compute_medoids(
    KMeansResult *result,
    const Vector32 *vectors,
    uint32_t nvecs
);

// Free result
void mkt_kmeans_result_destroy(KMeansResult *result);
```

#### K-Means++ Initialization

Better initialization for faster convergence:

```c
static void kmeans_plusplus_init(
    const Vector32 *vectors,
    uint32_t nvecs,
    Dimension dim,
    uint32_t nlist,
    Vector32 *centroids,
    uint64_t seed
) {
    // Random state
    uint64_t rng = seed;

    // Pick first centroid uniformly at random
    uint32_t first = xorshift64(&rng) % nvecs;
    memcpy(centroids, vectors + first * dim, dim * sizeof(Vector32));

    // Distance to nearest centroid for each vector
    Distance *min_distances = mkt_alloc(nvecs * sizeof(Distance));
    for (uint32_t i = 0; i < nvecs; i++) {
        min_distances[i] = INFINITY;
    }

    for (uint32_t c = 1; c < nlist; c++) {
        // Update distances to include new centroid
        VectorRef centroid = { .data = centroids + (c - 1) * dim, .dim = dim };
        for (uint32_t i = 0; i < nvecs; i++) {
            VectorRef v = { .data = vectors + i * dim, .dim = dim };
            Distance d = mkt_distance_l2(centroid, v);
            if (d < min_distances[i]) {
                min_distances[i] = d;
            }
        }

        // Sample next centroid proportional to squared distance
        double total = 0;
        for (uint32_t i = 0; i < nvecs; i++) {
            total += min_distances[i];
        }

        double target = (xorshift64(&rng) / (double)UINT64_MAX) * total;
        double cumsum = 0;
        uint32_t next = nvecs - 1;
        for (uint32_t i = 0; i < nvecs; i++) {
            cumsum += min_distances[i];
            if (cumsum >= target) {
                next = i;
                break;
            }
        }

        memcpy(centroids + c * dim, vectors + next * dim, dim * sizeof(Vector32));
    }

    mkt_free(min_distances);
}
```

#### Lloyd's Algorithm

```c
KMeansResult *mkt_kmeans(
    const Vector32 *vectors,
    uint32_t nvecs,
    Dimension dim,
    uint32_t nlist,
    DistanceMetric metric,
    const KMeansOptions *opts
) {
    KMeansResult *result = mkt_alloc0(sizeof(KMeansResult));
    result->nlist = nlist;
    result->dim = dim;
    result->centroids = mkt_alloc_aligned(nlist * dim * sizeof(Vector32), 64);
    result->assignments = mkt_alloc(nvecs * sizeof(ClusterId));
    result->cluster_sizes = mkt_alloc0(nlist * sizeof(uint32_t));

    // Initialize centroids
    kmeans_plusplus_init(vectors, nvecs, dim, nlist, result->centroids, opts->seed);

    // Temporary storage for centroid updates
    Vector32 *new_centroids = mkt_alloc0(nlist * dim * sizeof(Vector32));
    uint32_t *counts = mkt_alloc0(nlist * sizeof(uint32_t));

    for (uint32_t iter = 0; iter < opts->max_iterations; iter++) {
        // Reset accumulators
        memset(new_centroids, 0, nlist * dim * sizeof(Vector32));
        memset(counts, 0, nlist * sizeof(uint32_t));

        // Assignment step: assign each vector to nearest centroid
        for (uint32_t i = 0; i < nvecs; i++) {
            VectorRef v = { .data = vectors + i * dim, .dim = dim };

            Distance best_dist = INFINITY;
            ClusterId best_cluster = 0;

            for (uint32_t c = 0; c < nlist; c++) {
                VectorRef centroid = {
                    .data = result->centroids + c * dim,
                    .dim = dim
                };
                Distance d = mkt_distance(v, centroid, metric);
                if (d < best_dist) {
                    best_dist = d;
                    best_cluster = c;
                }
            }

            result->assignments[i] = best_cluster;

            // Accumulate for centroid update
            for (Dimension d = 0; d < dim; d++) {
                new_centroids[best_cluster * dim + d] += v.data[d];
            }
            counts[best_cluster]++;
        }

        // Update step: compute new centroids
        float max_shift = 0;
        for (uint32_t c = 0; c < nlist; c++) {
            if (counts[c] == 0) continue;

            float shift = 0;
            for (Dimension d = 0; d < dim; d++) {
                float old_val = result->centroids[c * dim + d];
                float new_val = new_centroids[c * dim + d] / counts[c];
                result->centroids[c * dim + d] = new_val;
                float diff = new_val - old_val;
                shift += diff * diff;
            }
            if (shift > max_shift) max_shift = shift;
        }

        result->cluster_sizes = counts;  // Will copy at end

        if (opts->verbose) {
            fprintf(stderr, "K-means iter %u: max shift = %f\n", iter, sqrtf(max_shift));
        }

        // Check convergence
        if (sqrtf(max_shift) < opts->tolerance) {
            break;
        }
    }

    // Copy final cluster sizes
    memcpy(result->cluster_sizes, counts, nlist * sizeof(uint32_t));

    mkt_free(new_centroids);
    mkt_free(counts);

    return result;
}
```

#### Medoid Computation

```c
void mkt_kmeans_compute_medoids(
    KMeansResult *result,
    const Vector32 *vectors,
    uint32_t nvecs
) {
    Dimension dim = result->dim;

    for (uint32_t c = 0; c < result->nlist; c++) {
        VectorRef centroid = {
            .data = result->centroids + c * dim,
            .dim = dim
        };

        Distance best_dist = INFINITY;
        uint32_t best_idx = 0;

        // Find vector in cluster closest to centroid
        for (uint32_t i = 0; i < nvecs; i++) {
            if (result->assignments[i] != c) continue;

            VectorRef v = { .data = vectors + i * dim, .dim = dim };
            Distance d = mkt_distance_l2(centroid, v);
            if (d < best_dist) {
                best_dist = d;
                best_idx = i;
            }
        }

        // Replace centroid with medoid
        memcpy(result->centroids + c * dim,
               vectors + best_idx * dim,
               dim * sizeof(Vector32));
    }
}
```

**Tests**:
- Convergence: verify algorithm converges (centroids stop moving)
- Cluster quality: measure within-cluster variance
- Edge cases: empty clusters, single vector, nlist > nvecs
- Determinism: same seed produces same result
- Performance: benchmark on various sizes

**CLI Tool**: `mkt_cluster_bench`

```
$ mkt_cluster_bench --dim 128 --nvecs 100000 --nlist 1000
K-means clustering (dim=128, nvecs=100000, nlist=1000):
  Initialization: 1.2s
  Lloyd iterations: 8
  Total time: 4.5s
  Avg cluster size: 100 (std: 23)
```

---

## Part 3: Posting List Data Structures

### 3.1 Posting List Entry

**Files**: `src/posting.h`, `src/posting.c`

The posting list entry is the fundamental unit stored in the index.

```c
// Posting list entry (variable size based on quantization)
// Layout: [flags: 1 byte][tid: 6 bytes][quantized: dim bytes]
//
// For centroid entries, TID may be InvalidItemPointer if the original
// vector was deleted (centroid becomes navigation-only)

typedef struct {
    uint8_t  flags;         // Entry flags
    // TID stored as 6 bytes: 4-byte block + 2-byte offset
    uint8_t  tid_bytes[6];
    // Followed by: ScalarQ8 quantized[dim] (not in struct, variable length)
} PostingEntry;

// Entry flags
#define POSTING_FLAG_CENTROID  0x01  // This entry is the cluster centroid
#define POSTING_FLAG_DELETED   0x02  // Soft-deleted, pending vacuum

// Entry size calculation
static inline size_t posting_entry_size(Dimension dim) {
    return sizeof(PostingEntry) + dim * sizeof(ScalarQ8);
}

// Access quantized data
static inline ScalarQ8 *posting_entry_quantized(PostingEntry *entry) {
    return (ScalarQ8 *)(entry + 1);
}

// TID encoding/decoding (6 bytes for block + offset)
static inline void posting_entry_set_tid(PostingEntry *entry,
                                          uint32_t block, uint16_t offset) {
    memcpy(entry->tid_bytes, &block, 4);
    memcpy(entry->tid_bytes + 4, &offset, 2);
}

static inline void posting_entry_get_tid(const PostingEntry *entry,
                                          uint32_t *block, uint16_t *offset) {
    memcpy(block, entry->tid_bytes, 4);
    memcpy(offset, entry->tid_bytes + 4, 2);
}
```

### 3.2 In-Memory Posting List

For building and manipulation before writing to pages:

```c
// In-memory posting list (growable)
typedef struct {
    PostingEntry **entries;  // Array of entry pointers
    uint32_t       count;
    uint32_t       capacity;
    Dimension      dim;
    ClusterId      cluster_id;
} MemPostingList;

MemPostingList *mkt_posting_list_create(ClusterId cluster_id, Dimension dim);
void            mkt_posting_list_destroy(MemPostingList *list);

// Add entry to list
void mkt_posting_list_add(
    MemPostingList *list,
    uint32_t block,
    uint16_t offset,
    const ScalarQ8 *quantized,
    uint8_t flags
);

// Set centroid (must be first entry)
void mkt_posting_list_set_centroid(
    MemPostingList *list,
    uint32_t block,
    uint16_t offset,
    const ScalarQ8 *quantized
);

// Sort by TID for locality (optional optimization)
void mkt_posting_list_sort_by_tid(MemPostingList *list);

// Iteration
typedef struct {
    MemPostingList *list;
    uint32_t        index;
} PostingListIter;

PostingListIter mkt_posting_list_iter(MemPostingList *list);
PostingEntry   *mkt_posting_list_next(PostingListIter *iter);
```

### 3.3 Page Layout

**Files**: `src/page_layout.h`, `src/page_layout.c`

Page layout for PostgreSQL integration, but designed to be testable standalone.

```c
// Page size (matches PostgreSQL default)
#define MKT_PAGE_SIZE 8192

// Page header (at start of each page)
typedef struct {
    uint32_t next_blkno;      // Next page in posting list (or InvalidBlockNumber)
    uint16_t entry_count;     // Number of entries on this page
    uint16_t free_offset;     // Offset to first free byte
    ClusterId cluster_id;     // Which cluster this page belongs to
    uint8_t  flags;           // Page flags
    uint8_t  reserved[3];     // Alignment padding
} MktPageHeader;

#define MKT_PAGE_FLAG_FIRST    0x01  // First page of posting list (has centroid)
#define MKT_PAGE_FLAG_OVERFLOW 0x02  // Overflow page (added after initial build)

// Usable space per page
#define MKT_PAGE_USABLE (MKT_PAGE_SIZE - sizeof(MktPageHeader))

// Calculate entries per page
static inline uint32_t mkt_entries_per_page(Dimension dim) {
    return MKT_PAGE_USABLE / posting_entry_size(dim);
}

// Page operations (work on raw byte buffer)
void mkt_page_init(void *page, ClusterId cluster_id, uint8_t flags);
void mkt_page_set_next(void *page, uint32_t next_blkno);
uint32_t mkt_page_get_next(const void *page);

// Add entry to page (returns false if page full)
bool mkt_page_add_entry(
    void *page,
    Dimension dim,
    uint32_t block,
    uint16_t offset,
    const ScalarQ8 *quantized,
    uint8_t flags
);

// Get entry by index
PostingEntry *mkt_page_get_entry(void *page, uint32_t index, Dimension dim);

// Iterate entries on page
uint16_t mkt_page_entry_count(const void *page);
```

### 3.4 Metapage

```c
// Metapage layout (block 0)
typedef struct {
    uint32_t magic;           // Magic number for validation
    uint32_t version;         // Format version
    Dimension dim;            // Vector dimension
    DistanceMetric metric;    // Distance metric
    uint32_t nlist;           // Number of clusters
    uint64_t nvecs;           // Total vectors indexed
    uint32_t next_meta_blkno; // Next metapage (if directory overflows)
    // SQ8 parameters follow
    // Then: posting_list_heads[nlist] (uint32_t per cluster)
} MktMetapage;

#define MKT_MAGIC 0x54494752  // "TIGR"
#define MKT_VERSION 1

// Directory entries per metapage (after fixed header + SQ8 params)
size_t mkt_meta_directory_capacity(Dimension dim);

// Initialize metapage
void mkt_meta_init(
    void *page,
    Dimension dim,
    DistanceMetric metric,
    uint32_t nlist,
    const SQ8Params *sq8_params
);

// Get/set posting list head for cluster
uint32_t mkt_meta_get_head(const void *page, ClusterId cluster);
void mkt_meta_set_head(void *page, ClusterId cluster, uint32_t block);
```

**Tests**:
- Page layout: verify entry packing, no overlap
- Capacity: verify calculated entries match actual
- Round-trip: write entries, read back, compare
- Overflow: verify full page returns false on add

---

## Part 4: Index Build

### 4.1 Build Pipeline

**Files**: `src/build.h`, `src/build.c`

```c
// Build state machine
typedef enum {
    BUILD_STATE_INIT,
    BUILD_STATE_SAMPLING,
    BUILD_STATE_CLUSTERING,
    BUILD_STATE_ASSIGNING,
    BUILD_STATE_WRITING,
    BUILD_STATE_DONE
} BuildState;

// Build context
typedef struct {
    BuildState      state;
    Dimension       dim;
    DistanceMetric  metric;
    uint32_t        nlist;

    // Sampling
    Vector32       *sample_vectors;
    uint32_t        sample_count;
    uint32_t        sample_capacity;
    float           sample_rate;

    // Clustering result
    KMeansResult   *kmeans;

    // Quantization
    SQ8Params      *sq8_params;

    // Posting lists (in memory during build)
    MemPostingList **posting_lists;

    // Statistics
    uint64_t        total_vectors;
    uint64_t        total_bytes;
} BuildContext;

// Build options
typedef struct {
    uint32_t nlist;           // Number of clusters (0 = auto)
    float    sample_rate;     // Fraction of vectors to sample (default: 0.1)
    uint32_t max_sample;      // Max sample size (default: 100000)
    uint32_t kmeans_iters;    // K-means iterations (default: 20)
    float    fillfactor;      // Page fill factor (default: 0.9)
    uint64_t seed;            // Random seed
} BuildOptions;

// Create build context
BuildContext *mkt_build_create(
    Dimension dim,
    DistanceMetric metric,
    const BuildOptions *options
);

void mkt_build_destroy(BuildContext *ctx);

// Phase 1: Add vectors for sampling
// Call repeatedly with batches of vectors
void mkt_build_add_sample(
    BuildContext *ctx,
    const Vector32 *vectors,
    const ItemPointer *tids,  // Can be NULL for standalone testing
    uint32_t count
);

// Phase 2: Perform clustering on sample
void mkt_build_cluster(BuildContext *ctx);

// Phase 3: Assign vectors to clusters and build posting lists
// Call repeatedly with batches (can be same vectors as sampling, or full scan)
void mkt_build_assign(
    BuildContext *ctx,
    const Vector32 *vectors,
    const ItemPointer *tids,
    uint32_t count
);

// Phase 4: Write to pages
// Callback invoked for each page to write
typedef void (*PageWriteCallback)(
    void *callback_data,
    uint32_t block_number,
    const void *page_data
);

void mkt_build_write_pages(
    BuildContext *ctx,
    PageWriteCallback callback,
    void *callback_data
);

// Get build statistics
typedef struct {
    uint64_t total_vectors;
    uint64_t total_pages;
    uint32_t metapages;
    uint32_t posting_pages;
    uint32_t avg_cluster_size;
    uint32_t max_cluster_size;
    uint32_t min_cluster_size;
} BuildStats;

BuildStats mkt_build_stats(const BuildContext *ctx);
```

### 4.2 Streaming Build

For datasets larger than memory:

```c
// Streaming build: two-pass algorithm
// Pass 1: Sample vectors, cluster, learn quantization
// Pass 2: Assign all vectors, write pages incrementally

typedef struct MktStreamBuild MktStreamBuild;

// Create streaming builder
MktStreamBuild *mkt_stream_build_create(
    Dimension dim,
    DistanceMetric metric,
    const BuildOptions *options
);

// Pass 1: Sampling (call multiple times)
void mkt_stream_build_sample(
    MktStreamBuild *builder,
    const Vector32 *vectors,
    uint32_t count
);

// Finish pass 1: perform clustering
void mkt_stream_build_finish_sampling(MktStreamBuild *builder);

// Pass 2: Assignment (call multiple times)
// Returns pages to write via callback
void mkt_stream_build_assign(
    MktStreamBuild *builder,
    const Vector32 *vectors,
    const ItemPointer *tids,
    uint32_t count,
    PageWriteCallback callback,
    void *callback_data
);

// Finish pass 2: write final pages, metapage
void mkt_stream_build_finish(
    MktStreamBuild *builder,
    PageWriteCallback callback,
    void *callback_data
);

void mkt_stream_build_destroy(MktStreamBuild *builder);
```

**Tests**:
- Small dataset: verify correct pages produced
- Streaming: verify streaming produces same result as batch
- Empty clusters: verify handling of empty clusters
- Page boundaries: verify entries split correctly across pages

**CLI Tool**: `mkt_build`

```
$ mkt_build --input vectors.bin --dim 768 --nlist 1000 --output index.mkt
Sampling: 10000 / 100000 vectors
Clustering: 1000 clusters, 15 iterations
Assigning: 100000 vectors
Writing: 5234 pages
Done. Index size: 42.8 MB
```

---

## Part 5: Search

### 5.1 Search Interface

**Files**: `src/search.h`, `src/search.c`

```c
// Search parameters
typedef struct {
    uint32_t nprobe;      // Number of clusters to search
    uint32_t k;           // Number of results
    uint32_t rerank_k;    // Candidates to re-rank (0 = no reranking)
} SearchParams;

// Search result
typedef struct {
    ItemPointer tid;
    Distance    distance;
} SearchResult;

// Centroid cache (in-memory, optimized for SIMD search)
typedef struct {
    Vector32   *centroids;   // Flat array: nlist * dim
    ScalarQ8   *quantized;   // Quantized centroids (optional)
    uint32_t    nlist;
    Dimension   dim;
    SQ8Params  *sq8_params;
} CentroidCache;

// Load centroids from pages (reads first page of each posting list)
CentroidCache *mkt_centroid_cache_create(
    uint32_t nlist,
    Dimension dim,
    const SQ8Params *sq8_params
);

void mkt_centroid_cache_set(
    CentroidCache *cache,
    ClusterId cluster,
    const Vector32 *centroid
);

void mkt_centroid_cache_destroy(CentroidCache *cache);

// Search: find top-k clusters
void mkt_search_centroids(
    const CentroidCache *cache,
    VectorRef query,
    DistanceMetric metric,
    uint32_t nprobe,
    ClusterId *clusters,      // Output: top nprobe cluster IDs
    Distance *distances       // Output: distances to clusters (optional)
);

// Scan posting list pages
// Callback for page reads (abstraction over buffer cache)
typedef const void *(*PageReadCallback)(
    void *callback_data,
    uint32_t block_number
);

typedef void (*PageReleaseCallback)(
    void *callback_data,
    uint32_t block_number
);

// Scan posting lists and find candidates
void mkt_search_posting_lists(
    const ClusterId *clusters,
    uint32_t nprobe,
    VectorRef query,
    const SQ8Params *sq8_params,
    DistanceMetric metric,
    const SearchParams *params,
    PageReadCallback read_page,
    PageReleaseCallback release_page,
    void *callback_data,
    TopKHeap *results
);

// Full search (centroids + posting lists)
uint32_t mkt_search(
    const CentroidCache *cache,
    VectorRef query,
    DistanceMetric metric,
    const SearchParams *params,
    const uint32_t *posting_list_heads,  // From metapage
    PageReadCallback read_page,
    PageReleaseCallback release_page,
    void *callback_data,
    SearchResult *results  // Output: up to k results
);
```

### 5.2 Centroid Search Implementation

```c
void mkt_search_centroids(
    const CentroidCache *cache,
    VectorRef query,
    DistanceMetric metric,
    uint32_t nprobe,
    ClusterId *clusters,
    Distance *distances
) {
    TopKHeap *heap = mkt_topk_create(nprobe);

    // Batch distance computation with SIMD
    Distance *all_distances = mkt_alloc(cache->nlist * sizeof(Distance));

    mkt_distance_batch(
        query,
        cache->centroids,
        cache->nlist,
        cache->dim,
        metric,
        all_distances
    );

    // Find top nprobe
    mkt_topk_insert_batch(heap, all_distances, cache->nlist, 0);

    // Extract results
    TopKEntry *entries = mkt_alloc(nprobe * sizeof(TopKEntry));
    mkt_topk_extract_sorted(heap, entries);

    for (uint32_t i = 0; i < nprobe; i++) {
        clusters[i] = entries[i].id;
        if (distances) distances[i] = entries[i].distance;
    }

    mkt_free(entries);
    mkt_free(all_distances);
    mkt_topk_destroy(heap);
}
```

### 5.3 Posting List Scan

```c
void mkt_search_posting_lists(
    const ClusterId *clusters,
    uint32_t nprobe,
    VectorRef query,
    const SQ8Params *sq8_params,
    DistanceMetric metric,
    const SearchParams *params,
    PageReadCallback read_page,
    PageReleaseCallback release_page,
    void *callback_data,
    TopKHeap *results
) {
    // Precompute lookup tables for asymmetric distance
    SQ8LUT *luts = mkt_alloc(query.dim * sizeof(SQ8LUT));
    for (Dimension d = 0; d < query.dim; d++) {
        mkt_sq8_build_lut(
            query.data[d],
            sq8_params->mins[d],
            sq8_params->scales[d],
            luts[d].tables
        );
    }

    // Scan each cluster's posting list
    for (uint32_t p = 0; p < nprobe; p++) {
        ClusterId cluster = clusters[p];
        uint32_t block = /* get from metapage */;

        while (block != InvalidBlockNumber) {
            const void *page = read_page(callback_data, block);
            uint16_t entry_count = mkt_page_entry_count(page);

            for (uint16_t i = 0; i < entry_count; i++) {
                PostingEntry *entry = mkt_page_get_entry(
                    (void *)page, i, query.dim
                );

                // Skip deleted entries
                if (entry->flags & POSTING_FLAG_DELETED) continue;

                // Skip centroid in results (used for navigation only)
                if (entry->flags & POSTING_FLAG_CENTROID) continue;

                // Compute approximate distance using LUTs
                ScalarQ8 *quantized = posting_entry_quantized(entry);
                Distance approx_dist = mkt_sq8_distance_lut(
                    luts, quantized, query.dim
                );

                // Add to heap if promising
                if (approx_dist < mkt_topk_threshold(results)) {
                    uint32_t tid_block;
                    uint16_t tid_offset;
                    posting_entry_get_tid(entry, &tid_block, &tid_offset);

                    // Encode TID as single uint64 for heap storage
                    uint64_t tid_encoded = ((uint64_t)tid_block << 16) | tid_offset;
                    mkt_topk_insert(results, approx_dist, tid_encoded);
                }
            }

            uint32_t next = mkt_page_get_next(page);
            release_page(callback_data, block);
            block = next;
        }
    }

    mkt_free(luts);
}
```

### 5.4 Re-ranking

For final precision, re-rank top candidates using full-precision vectors:

```c
// Re-rank candidates with full precision vectors
// Requires callback to fetch full vectors from heap
typedef void (*VectorFetchCallback)(
    void *callback_data,
    ItemPointer tid,
    Vector32 *output  // Pre-allocated buffer
);

void mkt_search_rerank(
    TopKHeap *candidates,      // Input: approximate results
    uint32_t rerank_k,         // How many to re-rank
    VectorRef query,
    DistanceMetric metric,
    Dimension dim,
    VectorFetchCallback fetch_vector,
    void *callback_data,
    TopKHeap *final_results    // Output: precise results
) {
    // Extract top rerank_k candidates
    TopKEntry *entries = mkt_alloc(rerank_k * sizeof(TopKEntry));
    // ... extract from candidates heap ...

    Vector32 *vec_buffer = mkt_alloc_aligned(dim * sizeof(Vector32), 64);

    for (uint32_t i = 0; i < rerank_k; i++) {
        // Decode TID
        uint64_t tid_encoded = entries[i].id;
        ItemPointer tid = /* decode */;

        // Fetch full vector
        fetch_vector(callback_data, tid, vec_buffer);

        // Compute precise distance
        VectorRef v = { .data = vec_buffer, .dim = dim };
        Distance precise_dist = mkt_distance(query, v, metric);

        mkt_topk_insert(final_results, precise_dist, tid_encoded);
    }

    mkt_free(vec_buffer);
    mkt_free(entries);
}
```

**Tests**:
- Recall: measure recall@k against brute-force search
- Performance: benchmark QPS at various nprobe values
- Correctness: verify re-ranking improves result quality

**CLI Tool**: `mkt_search`

```
$ mkt_search --index index.mkt --query query.bin --k 10 --nprobe 20
Results (10 of 100000 vectors):
  1. tid=(42,15)  distance=0.0234
  2. tid=(108,3)  distance=0.0456
  ...
Search time: 2.3ms
```

---

## Part 6: PostgreSQL Integration

### 6.1 Extension Setup

**Files**: `src/meerkat.c`, `src/meerkat.h`

```c
// Extension initialization
void _PG_init(void);

// Shared memory request hook
static void mkt_shmem_request(void);
static void mkt_shmem_startup(void);

// GUC variables
int mkt_default_nprobe;
int mkt_default_rerank_k;
```

### 6.2 Index Access Method Handler

```c
// Index handler function (registered with CREATE ACCESS METHOD)
Datum mkt_handler(PG_FUNCTION_ARGS);

// Required IAM callbacks
static IndexBuildResult *mkt_ambuild(Relation heap, Relation index,
                                        IndexInfo *indexInfo);
static void mkt_ambuildempty(Relation index);
static bool mkt_aminsert(Relation index, Datum *values, bool *isnull,
                           ItemPointer heap_tid, Relation heap,
                           IndexUniqueCheck checkUnique,
                           bool indexUnchanged, IndexInfo *indexInfo);
static IndexScanDesc mkt_ambeginscan(Relation index, int nkeys, int norderbys);
static void mkt_amrescan(IndexScanDesc scan, ScanKey keys, int nkeys,
                           ScanKey orderbys, int norderbys);
static bool mkt_amgettuple(IndexScanDesc scan, ScanDirection direction);
static int64 mkt_amgetbitmap(IndexScanDesc scan, TIDBitmap *tbm);
static void mkt_amendscan(IndexScanDesc scan);
static IndexBulkDeleteResult *mkt_ambulkdelete(IndexVacuumInfo *info,
                                                  IndexBulkDeleteResult *stats,
                                                  IndexBulkDeleteCallback callback,
                                                  void *callback_state);
static IndexBulkDeleteResult *mkt_amvacuumcleanup(IndexVacuumInfo *info,
                                                     IndexBulkDeleteResult *stats);
static void mkt_amcostestimate(PlannerInfo *root, IndexPath *path,
                                  double loop_count, Cost *indexStartupCost,
                                  Cost *indexTotalCost, Selectivity *indexSelectivity,
                                  double *indexCorrelation, double *indexPages);
```

### 6.3 Buffer Cache Integration

```c
// Page read via buffer cache
typedef struct {
    Relation index;
    Buffer   current_buffer;
} BufferReadState;

static const void *buffer_read_page(void *state, uint32_t block) {
    BufferReadState *brs = (BufferReadState *)state;

    if (BufferIsValid(brs->current_buffer)) {
        ReleaseBuffer(brs->current_buffer);
    }

    brs->current_buffer = ReadBuffer(brs->index, block);
    LockBuffer(brs->current_buffer, BUFFER_LOCK_SHARE);

    return BufferGetPage(brs->current_buffer);
}

static void buffer_release_page(void *state, uint32_t block) {
    BufferReadState *brs = (BufferReadState *)state;

    if (BufferIsValid(brs->current_buffer)) {
        UnlockReleaseBuffer(brs->current_buffer);
        brs->current_buffer = InvalidBuffer;
    }
}
```

### 6.4 Shared Memory Centroid Cache

```c
// Shared memory layout for centroid cache
typedef struct {
    LWLock     lock;           // For cache updates
    uint32_t   nlist;
    Dimension  dim;
    bool       initialized;
    // Followed by: centroids[nlist * dim]
    // Followed by: sq8_params
} MktShmemHeader;

// Get or create cache for an index
CentroidCache *mkt_shmem_get_cache(Oid indexoid);
void mkt_shmem_invalidate_cache(Oid indexoid);

// Called during ambuild to populate cache
void mkt_shmem_populate_cache(
    Oid indexoid,
    const KMeansResult *kmeans,
    const SQ8Params *sq8_params
);
```

### 6.5 Scan State

```c
// Scan state (stored in IndexScanDesc->opaque)
typedef struct MktScanOpaque {
    CentroidCache  *cache;
    TopKHeap       *results;
    SearchParams    params;
    VectorRef       query;
    bool            first_call;

    // Buffer management
    BufferReadState buffer_state;

    // Current position in results
    uint32_t        result_index;
    TopKEntry      *sorted_results;
} MktScanOpaque;
```

### 6.6 Cost Estimation

```c
static void mkt_amcostestimate(
    PlannerInfo *root,
    IndexPath *path,
    double loop_count,
    Cost *indexStartupCost,
    Cost *indexTotalCost,
    Selectivity *indexSelectivity,
    double *indexCorrelation,
    double *indexPages
) {
    // Get index statistics from metapage
    Relation index = index_open(path->indexinfo->indexoid, AccessShareLock);
    MktMetapage *meta = /* read metapage */;

    uint32_t nlist = meta->nlist;
    uint64_t nvecs = meta->nvecs;
    uint32_t nprobe = /* from GUC or reloption */;
    uint32_t k = /* from query */;

    // Startup cost: centroid search (memory only)
    *indexStartupCost = nlist * cpu_operator_cost;

    // Per-tuple cost: posting list scan
    double avg_list_size = (double)nvecs / nlist;
    double pages_per_list = avg_list_size / mkt_entries_per_page(meta->dim);
    double pages_scanned = nprobe * pages_per_list;

    *indexTotalCost = *indexStartupCost +
                      pages_scanned * (seq_page_cost + cpu_tuple_cost * avg_list_size);

    // Add re-ranking cost if enabled
    uint32_t rerank_k = /* from GUC or reloption */;
    if (rerank_k > 0) {
        *indexTotalCost += rerank_k * random_page_cost;  // Heap access
    }

    *indexSelectivity = (double)k / nvecs;
    *indexCorrelation = 0.0;  // No physical correlation
    *indexPages = pages_scanned;

    index_close(index, AccessShareLock);
}
```

### 6.7 MVCC Support

```c
// Check visibility during posting list scan
static bool entry_is_visible(
    PostingEntry *entry,
    Snapshot snapshot,
    Relation heap
) {
    uint32_t block;
    uint16_t offset;
    posting_entry_get_tid(entry, &block, &offset);

    ItemPointerData tid;
    ItemPointerSet(&tid, block, offset);

    // Use index-only visibility check if possible
    // Otherwise fetch heap tuple and check
    Buffer buffer;
    bool visible = heap_hot_search_buffer(
        &tid, heap, buffer, snapshot, NULL, NULL, NULL
    );

    return visible;
}
```

---

## Part 7: Maintenance Operations

### 7.1 Insert

```c
static bool mkt_aminsert(
    Relation index,
    Datum *values,
    bool *isnull,
    ItemPointer heap_tid,
    Relation heap,
    IndexUniqueCheck checkUnique,
    bool indexUnchanged,
    IndexInfo *indexInfo
) {
    if (isnull[0]) return false;  // Can't index NULL vectors

    // Get vector from datum
    Vector *vec = DatumGetVector(values[0]);

    // Get centroid cache
    CentroidCache *cache = mkt_shmem_get_cache(RelationGetRelid(index));

    // Find nearest cluster
    ClusterId cluster;
    mkt_search_centroids(cache, VectorToRef(vec), DISTANCE_L2, 1, &cluster, NULL);

    // Quantize vector
    ScalarQ8 *quantized = palloc(vec->dim);
    mkt_sq8_encode(cache->sq8_params, VectorToRef(vec), quantized);

    // Get posting list head from metapage
    Buffer meta_buf = ReadBuffer(index, 0);
    LockBuffer(meta_buf, BUFFER_LOCK_SHARE);
    MktMetapage *meta = (MktMetapage *)BufferGetPage(meta_buf);
    BlockNumber head = mkt_meta_get_head(meta, cluster);
    UnlockReleaseBuffer(meta_buf);

    // Find page with space (scan from head, or use FSM)
    // ... insert entry ...

    pfree(quantized);
    return true;
}
```

### 7.2 Vacuum

```c
static IndexBulkDeleteResult *mkt_ambulkdelete(
    IndexVacuumInfo *info,
    IndexBulkDeleteResult *stats,
    IndexBulkDeleteCallback callback,
    void *callback_state
) {
    Relation index = info->index;

    // Scan all posting list pages
    // For each entry, check if TID should be deleted
    // Mark entry as deleted (soft delete)

    // ... implementation ...

    return stats;
}

static IndexBulkDeleteResult *mkt_amvacuumcleanup(
    IndexVacuumInfo *info,
    IndexBulkDeleteResult *stats
) {
    // Compact pages: remove soft-deleted entries
    // Update FSM with free space
    // Optionally merge underfull pages

    // ... implementation ...

    return stats;
}
```

---

## Development Phases

### Phase 1: Foundation (No PostgreSQL dependency)

**Deliverables**:
1. Type definitions and memory abstraction
2. Platform detection and SIMD wrappers
3. Distance computation (scalar + AVX512 + NEON)
4. Unit tests and benchmark CLI

**Test artifacts**:
- `mkt_distance_bench`: Distance computation benchmark
- Unit test suite for distance functions

### Phase 2: Core Algorithms

**Deliverables**:
1. Scalar quantization (SQ8)
2. Top-K heap
3. K-means clustering
4. Unit tests and CLI tools

**Test artifacts**:
- `mkt_quantize_test`: Quantization accuracy tests
- `mkt_cluster_bench`: Clustering benchmark
- Unit test suite

### Phase 3: Data Structures

**Deliverables**:
1. Posting list entry format
2. Page layout (standalone, not PostgreSQL pages)
3. Metapage format
4. Serialization/deserialization

**Test artifacts**:
- Unit tests for page operations
- Round-trip serialization tests

### Phase 4: Build Pipeline

**Deliverables**:
1. Batch build (all vectors in memory)
2. Streaming build (two-pass)
3. Write to file (standalone index format)

**Test artifacts**:
- `mkt_build`: Build index from vector file
- Build correctness tests

### Phase 5: Search (Standalone)

**Deliverables**:
1. Centroid search
2. Posting list scan
3. Re-ranking (optional in standalone mode)
4. Full search pipeline

**Test artifacts**:
- `mkt_search`: Search standalone index
- Recall benchmarks against brute force

### Phase 6: PostgreSQL Integration

**Deliverables**:
1. Extension scaffolding (SQL, control file)
2. IAM handler registration
3. ambuild / aminsert
4. Scan operations (amgettuple)
5. Cost estimation

**Test artifacts**:
- PostgreSQL regression tests
- SQL-based correctness tests

### Phase 7: Production Features

**Deliverables**:
1. Shared memory centroid cache
2. MVCC visibility checks
3. Vacuum support
4. WAL logging

**Test artifacts**:
- Isolation tests (concurrent operations)
- Recovery tests (crash and restore)

### Phase 8: Performance Optimization

**Deliverables**:
1. Prefetching optimization
2. Batch I/O for posting lists
3. Query planning integration
4. Performance benchmarks

**Test artifacts**:
- Performance regression suite
- Comparison benchmarks (vs pgvector, pgvectorscale)

---

## File Organization

```
meerkat/
├── src/
│   ├── mkt_types.h          # Type definitions
│   ├── mkt_memory.h         # Memory abstraction
│   ├── mkt_memory.c
│   ├── mkt_memory_standalone.c
│   ├── mkt_memory_pg.c
│   ├── mkt_platform.h       # Platform detection
│   ├── mkt_platform.c
│   ├── distance.h             # Distance interface
│   ├── distance.c             # Dispatch + scalar
│   ├── distance_avx512.c
│   ├── distance_avx2.c
│   ├── distance_neon.c
│   ├── quantize.h
│   ├── quantize.c
│   ├── quantize_avx512.c
│   ├── topk.h
│   ├── topk.c
│   ├── kmeans.h
│   ├── kmeans.c
│   ├── posting.h
│   ├── posting.c
│   ├── page_layout.h
│   ├── page_layout.c
│   ├── build.h
│   ├── build.c
│   ├── search.h
│   ├── search.c
│   ├── meerkat.h              # PostgreSQL extension header
│   ├── meerkat.c              # Extension entry point
│   ├── mkt_handler.c          # IAM callbacks
│   ├── mkt_build.c            # PG build integration
│   ├── mkt_scan.c             # PG scan integration
│   ├── mkt_shmem.c            # Shared memory cache
│   └── mkt_vacuum.c           # Vacuum support
├── sql/
│   ├── meerkat--1.0.sql       # Extension SQL
│   └── meerkat.control        # Extension control file
├── test/
│   ├── unit/                  # Unit tests (standalone)
│   ├── regress/               # PostgreSQL regression tests
│   └── bench/                 # Benchmarks
├── tools/
│   ├── mkt_distance_bench.c
│   ├── mkt_cluster_bench.c
│   ├── mkt_build.c
│   └── mkt_search.c
├── docs/
│   ├── architecture.md
│   └── implementation.md
├── meson.build
└── CLAUDE.md
```

---

## References

- [PostgreSQL Index Access Method Documentation](https://www.postgresql.org/docs/current/indexam.html)
- [pgvector source](https://github.com/pgvector/pgvector)
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)
- [ARM NEON Intrinsics Reference](https://developer.arm.com/architectures/instruction-sets/intrinsics/)
