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

## Part 0: Unit Test Framework

A simple unit test framework enables testing core algorithms independently of
PostgreSQL. Tests use constructor-based auto-registration for minimal
boilerplate.

**Files**: `test/unit/mkt_test.h`, `test/unit/mkt_test.c`, `test/unit/run_tests.c`

### Test Framework Interface

```c
/* mkt_test.h - Simple C test framework with automatic test registration */

#ifndef MKT_TEST_H
#define MKT_TEST_H

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ANSI color codes */
#define COLOR_RESET  "\033[0m"
#define COLOR_RED    "\033[31m"
#define COLOR_GREEN  "\033[32m"
#define COLOR_YELLOW "\033[33m"
#define COLOR_CYAN   "\033[36m"

/* Test result structure */
typedef struct {
    const char *test_name;
    bool        passed;
    char        failure_msg[512];
    const char *file;
    int         line;
} MktTestResult;

/* Test function signature */
typedef void (*MktTestFunc)(MktTestResult *result);

/* Test registry entry */
typedef struct {
    const char *name;
    const char *group;
    MktTestFunc func;
} MktTestEntry;

/* Register a test (called automatically by TEST macro) */
void mkt_test_register(const char *name, const char *group, MktTestFunc func);

/* Run all registered tests, returns 0 on success, 1 on failure */
int mkt_test_run_all(void);

/* Internal: mark current test as failed */
void mkt_test_fail(const char *file, int line, const char *msg);

/* TEST_GROUP macro - sets the group for subsequent tests in this file */
#define TEST_GROUP(group_name) static const char *_MKT_TEST_GROUP = #group_name;

/* TEST macro - defines and auto-registers a test */
#define TEST(name)                                                  \
    static void test_##name(MktTestResult *result);                 \
    __attribute__((constructor)) static void register_##name(void)  \
    {                                                               \
        mkt_test_register(#name, _MKT_TEST_GROUP, test_##name);     \
    }                                                               \
    static void test_##name(MktTestResult *result)

/* Assertion macros */
#define ASSERT_TRUE(cond, msg)                                      \
    do {                                                            \
        if (!(cond)) {                                              \
            char buf[256];                                          \
            snprintf(buf, sizeof(buf), "%s (condition: %s)",        \
                     msg, #cond);                                   \
            mkt_test_fail(__FILE__, __LINE__, buf);                 \
            return;                                                 \
        }                                                           \
    } while (0)

#define ASSERT_EQ(expected, actual, msg)                            \
    do {                                                            \
        if ((expected) != (actual)) {                               \
            char buf[256];                                          \
            snprintf(buf, sizeof(buf),                              \
                     "%s (expected: %lld, actual: %lld)",           \
                     msg, (long long)(expected),                    \
                     (long long)(actual));                          \
            mkt_test_fail(__FILE__, __LINE__, buf);                 \
            return;                                                 \
        }                                                           \
    } while (0)

#define ASSERT_FLOAT_EQ(expected, actual, epsilon, msg)             \
    do {                                                            \
        double _diff = (expected) - (actual);                       \
        if (_diff < 0) _diff = -_diff;                              \
        if (_diff > (epsilon)) {                                    \
            char buf[256];                                          \
            snprintf(buf, sizeof(buf),                              \
                     "%s (expected: %f, actual: %f, diff: %f)",     \
                     msg, (double)(expected), (double)(actual),     \
                     _diff);                                        \
            mkt_test_fail(__FILE__, __LINE__, buf);                 \
            return;                                                 \
        }                                                           \
    } while (0)

#define ASSERT_NOT_NULL(ptr, msg)                                   \
    do {                                                            \
        if ((ptr) == NULL) {                                        \
            mkt_test_fail(__FILE__, __LINE__, msg);                 \
            return;                                                 \
        }                                                           \
    } while (0)

#define ASSERT_MEM_EQ(expected, actual, len, msg)                   \
    do {                                                            \
        if (memcmp((expected), (actual), (len)) != 0) {             \
            mkt_test_fail(__FILE__, __LINE__, msg);                 \
            return;                                                 \
        }                                                           \
    } while (0)

#endif /* MKT_TEST_H */
```

### Test Runner

```c
/* run_tests.c - Main test runner entry point */

#include "mkt_test.h"

int
main(void)
{
    return mkt_test_run_all();
}
```

### Example Test File

```c
/* test_mkt_vector.c - Vector type and operations tests */

#include "mkt_test.h"
#include "mkt_vector.h"

TEST_GROUP(Vector)

TEST(vector_create_and_dim)
{
    MktVector *v = mkt_vector_create(128);
    ASSERT_NOT_NULL(v, "vector should be allocated");
    ASSERT_EQ(128, MKT_VECTOR_DIM(v), "dimension should be 128");
    mkt_vector_free(v);
}

TEST(vector_set_get_elements)
{
    MktVector *v = mkt_vector_create(3);
    MKT_VECTOR_DATA(v)[0] = 1.0f;
    MKT_VECTOR_DATA(v)[1] = 2.0f;
    MKT_VECTOR_DATA(v)[2] = 3.0f;

    ASSERT_FLOAT_EQ(1.0f, MKT_VECTOR_DATA(v)[0], 1e-6, "element 0");
    ASSERT_FLOAT_EQ(2.0f, MKT_VECTOR_DATA(v)[1], 1e-6, "element 1");
    ASSERT_FLOAT_EQ(3.0f, MKT_VECTOR_DATA(v)[2], 1e-6, "element 2");

    mkt_vector_free(v);
}

TEST(vector_l2_distance)
{
    MktVector *a = mkt_vector_create(3);
    MktVector *b = mkt_vector_create(3);

    MKT_VECTOR_DATA(a)[0] = 0.0f;
    MKT_VECTOR_DATA(a)[1] = 0.0f;
    MKT_VECTOR_DATA(a)[2] = 0.0f;

    MKT_VECTOR_DATA(b)[0] = 1.0f;
    MKT_VECTOR_DATA(b)[1] = 0.0f;
    MKT_VECTOR_DATA(b)[2] = 0.0f;

    VectorRef va = MktVectorToRef(a);
    VectorRef vb = MktVectorToRef(b);

    Distance d = mkt_distance_l2(va, vb);
    ASSERT_FLOAT_EQ(1.0f, d, 1e-6, "L2 squared distance should be 1.0");

    mkt_vector_free(a);
    mkt_vector_free(b);
}
```

### Build Configuration (Meson)

```meson
# test/unit/meson.build

unit_test_sources = files(
  'mkt_test.c',
  'run_tests.c',
  'test_mkt_vector.c',
  'test_mkt_distance.c',
  'test_mkt_quantize.c',
  'test_mkt_topk.c',
)

test_runner = executable('run_tests',
  unit_test_sources,
  dependencies: [mkt_core_dep],
  include_directories: [
    include_directories('.'),
    include_directories('../../src'),
  ],
  install: false,
)

test('unit_tests', test_runner,
  workdir: meson.project_source_root(),
  timeout: 60,
)
```

### Running Tests

```
$ meson test -C builddir unit_tests

=== Vector ===
[PASS] vector_create_and_dim
[PASS] vector_set_get_elements
[PASS] vector_l2_distance

=== Distance ===
[PASS] distance_l2_scalar
[PASS] distance_l2_avx512
[PASS] distance_batch

========================================
Total: 6 passed, 0 failed
========================================
```

---

## Part 1: Foundation Layer

### 1.1 Standalone Vector Type

**Files**: `src/mkt_vector.h`, `src/mkt_vector.c`

The vector type is binary-compatible with pgvector's `vector` type in both
standalone and PostgreSQL builds. The struct layout is identical; only the
varlena header initialization differs.

pgvector's vector type (from `pgvector/src/vector.h`):

```c
typedef struct Vector
{
    int32       vl_len_;        /* varlena header (do not touch directly!) */
    int16       dim;            /* number of dimensions */
    int16       unused;         /* reserved for future use, always zero */
    float       x[FLEXIBLE_ARRAY_MEMBER];
}           Vector;
```

Meerkat's binary-compatible definition:

```c
/* mkt_vector.h - Vector type compatible with pgvector */

#ifndef MKT_VECTOR_H
#define MKT_VECTOR_H

#include <stdint.h>
#include <stdlib.h>
#include "mkt_types.h"

#define MKT_VECTOR_MAX_DIM 16000

/*
 * MktVector: Binary-compatible with pgvector's Vector type.
 *
 * The struct layout is identical in both standalone and PostgreSQL modes.
 * In standalone mode, vl_len_ stores the total size (not used as varlena).
 * In PostgreSQL mode, vl_len_ is managed by SET_VARSIZE/VARSIZE macros.
 */
typedef struct MktVector
{
    int32       vl_len_;        /* varlena header / size in standalone mode */
    int16       dim;            /* number of dimensions */
    int16       unused;         /* reserved for future use, always zero */
    float       x[];            /* FLEXIBLE_ARRAY_MEMBER */
}           MktVector;

#define MKT_VECTOR_SIZE(_dim) (offsetof(MktVector, x) + sizeof(float)*(_dim))
#define MKT_VECTOR_DIM(v)     ((v)->dim)
#define MKT_VECTOR_DATA(v)    ((v)->x)

/* Convert to VectorRef for internal operations */
static inline VectorRef
MktVectorToRef(const MktVector *v)
{
    return (VectorRef){ .data = v->x, .dim = v->dim };
}

/* Allocation and lifecycle */
MktVector  *mkt_vector_create(Dimension dim);
MktVector  *mkt_vector_copy(const MktVector *src);
void        mkt_vector_free(MktVector *v);

/* Initialization */
void        mkt_vector_set(MktVector *v, const float *values);
void        mkt_vector_zero(MktVector *v);
void        mkt_vector_fill(MktVector *v, float value);

/* Operations */
float       mkt_vector_dot(const MktVector *a, const MktVector *b);
float       mkt_vector_norm(const MktVector *v);
void        mkt_vector_normalize(MktVector *v);

#endif /* MKT_VECTOR_H */
```

### Implementation

```c
/* mkt_vector.c - Vector operations */

#include "mkt_vector.h"
#include "mkt_memory.h"
#include <string.h>
#include <math.h>

MktVector *
mkt_vector_create(Dimension dim)
{
    if (dim > MKT_VECTOR_MAX_DIM)
        return NULL;

    size_t size = MKT_VECTOR_SIZE(dim);
    MktVector *v = mkt_alloc(size);
    if (v == NULL)
        return NULL;

#ifdef MKT_STANDALONE
    v->vl_len_ = (int32) size;  /* Store size directly */
#else
    SET_VARSIZE(v, size);       /* PostgreSQL varlena header */
#endif
    v->dim = dim;
    v->unused = 0;

    return v;
}

void
mkt_vector_free(MktVector *v)
{
    mkt_free(v);
}

MktVector *
mkt_vector_copy(const MktVector *src)
{
    MktVector *dst = mkt_vector_create(src->dim);
    if (dst == NULL)
        return NULL;

    memcpy(dst->x, src->x, src->dim * sizeof(float));
    return dst;
}

void
mkt_vector_zero(MktVector *v)
{
    memset(v->x, 0, v->dim * sizeof(float));
}

float
mkt_vector_dot(const MktVector *a, const MktVector *b)
{
    float sum = 0.0f;
    for (Dimension i = 0; i < a->dim; i++)
        sum += a->x[i] * b->x[i];
    return sum;
}

float
mkt_vector_norm(const MktVector *v)
{
    return sqrtf(mkt_vector_dot(v, v));
}

void
mkt_vector_normalize(MktVector *v)
{
    float norm = mkt_vector_norm(v);
    if (norm > 0.0f) {
        for (Dimension i = 0; i < v->dim; i++)
            v->x[i] /= norm;
    }
}
```

The key design points:

1. **Identical binary layout**: The struct has the same fields in the same order
   regardless of build mode. A pointer to `MktVector` and pgvector's `Vector`
   can be safely cast between each other.

2. **vl_len_ handling**: In standalone mode, `vl_len_` stores the struct size
   directly. In PostgreSQL mode, it's managed by `SET_VARSIZE()`. The field
   exists in both modes to maintain binary compatibility.

3. **Core algorithms use VectorRef**: Distance computation, quantization, and
   other algorithms operate on `VectorRef` (pointer + dimension), not the full
   struct. This decouples algorithms from storage format.

### 1.2 Type Definitions

**File**: `src/mkt_types.h`

```c
#include <stdint.h>
#include <stdbool.h>

// Quantized representations
typedef uint8_t  ScalarQ8;   // 8-bit scalar quantized
typedef uint8_t  BinaryQ;    // Binary quantized byte (for packed bit arrays)
                             // Full vectors use MktBitVector (VarBit-compatible)

// Dimension type (max 65535 dimensions)
typedef uint16_t Dimension;

// Cluster/centroid identifier
typedef uint32_t ClusterId;

// Distance type (always float for intermediate computations)
typedef float Distance;

// Vector reference (pointer + dimension, no ownership)
// Points to float array, does not own the memory
typedef struct {
    const float *data;
    Dimension    dim;
} VectorRef;

// Mutable vector (for building/modifying)
typedef struct {
    float  *data;
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

### 1.3 Memory Abstraction

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

### 1.4 Platform Abstraction

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
    const float *vectors,  // Contiguous array of vectors
    uint32_t count,
    Dimension dim,
    DistanceMetric metric,
    Distance *distances       // Output: count distances
);

// Batch with early termination: stop when found k vectors below threshold
// Returns number of vectors actually processed
uint32_t mkt_distance_batch_threshold(
    VectorRef query,
    const float *vectors,
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
    const float *pa = a.data;
    const float *pb = b.data;
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
    const float *vectors,
    uint32_t count,
    Dimension dim,
    Distance *distances
) {
    const size_t vector_bytes = dim * sizeof(float);
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
typedef void (*DistanceBatchFn)(VectorRef, const float*, uint32_t,
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

**Files**: `src/mkt_quantize.h`, `src/mkt_quantize.c`, `src/mkt_rabitq.c`

Quantization compresses vectors for faster approximate distance computation.
Meerkat supports multiple quantization methods:

| Method | Compression | Recall | Used By |
|--------|-------------|--------|---------|
| **RaBitQ** | 32x | 95-99% | turbopuffer, Elastic, LanceDB |
| SQ8 (Scalar 8-bit) | 4x | 99%+ | Faiss, Qdrant, Milvus, OpenSearch |
| SQ4 (Scalar 4-bit) | 8x | 95-98% | Faiss |
| SBQ (Statistical Binary) | 32x | 95%+ | pgvectorscale |
| PQ (Product Quantization) | 16-64x | 90-98% | Faiss, ScaNN, Pinecone |

**Meerkat uses RaBitQ as the primary quantization method** for several reasons:

1. **Theoretical guarantees**: Error bound O(1/√D) that improves with dimension.
   PQ lacks such bounds and can fail on some datasets (e.g., 50%+ error on MSong).
2. **High compression**: 32x (1 bit/dimension) vs SQ8's 4x.
3. **Speed**: 3x faster than PQ at same accuracy due to bitwise operations.
4. **Architecture fit**: Meerkat re-ranks with full precision vectors, so aggressive
   initial compression is acceptable.

SQ8 is available as a fallback for lower-dimensional vectors where RaBitQ's
benefits are smaller.

#### References

- [RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound
  for Approximate Nearest Neighbor Search](https://arxiv.org/abs/2405.12497)
  (SIGMOD 2024)
- [RaBitQ Reference Implementation](https://github.com/gaoj0017/RaBitQ)
- [turbopuffer ANN v3](https://turbopuffer.com/blog/ann-v3)
- [LanceDB RaBitQ Integration](https://lancedb.com/blog/feature-rabitq-quantization/)

#### PostgreSQL VarBit Compatibility

Binary quantized vectors in PostgreSQL should be compatible with the `bit varying`
(VarBit) type used by pgvector. This enables interoperability and reuse of
pgvector's Hamming/Jaccard distance functions.

```c
// PostgreSQL VarBit layout (from utils/varbit.h):
//   - vl_len_: varlena header
//   - bit_len: number of bits (int32)
//   - bit_dat[]: packed bit data, MSB first within each byte
//
// Key macros:
//   VARBITTOTALLEN(dim)  - total size including headers
//   VARBITLEN(v)         - number of bits
//   VARBITBYTES(v)       - number of data bytes
//   VARBITS(v)           - pointer to bit data

// Meerkat's binary type for standalone mode
typedef uint8_t BinaryQ;

// In PostgreSQL mode, use VarBit directly or ensure layout matches
#ifdef MKT_STANDALONE
typedef struct {
    uint32_t bit_len;           // Number of bits
    uint8_t  bit_dat[];         // Packed bits, MSB first
} MktBitVector;
#define MKT_BITVEC_BYTES(dim) (((dim) + 7) / 8)
#define MKT_BITVEC_SIZE(dim)  (sizeof(MktBitVector) + MKT_BITVEC_BYTES(dim))
#else
// Use PostgreSQL's VarBit type directly
#include "utils/varbit.h"
typedef VarBit MktBitVector;
#define MKT_BITVEC_BYTES(dim) VARBITBYTES(dim)
#define MKT_BITVEC_SIZE(dim)  VARBITTOTALLEN(dim)
#endif
```

**Bit ordering**: PostgreSQL VarBit stores bits MSB-first within each byte
(bit 0 is the high bit of byte 0). Ensure encoding functions match this layout.

#### RaBitQ (Primary Method)

RaBitQ quantizes D-dimensional vectors into D-bit binary codes using a randomly
rotated codebook. The key insight is that vertices of a hypercube (±1/√D per
coordinate) are evenly spread on the unit hypersphere, and random rotation
removes bias toward any particular vector direction.

**Algorithm overview:**

1. **Setup**: Generate a random orthogonal matrix P (Johnson-Lindenstrauss style)
2. **Encode**: For vector **o**, compute P⁻¹**o** and store sign pattern as D bits
3. **Distance**: Use ratio estimator with theoretical error bound O(1/√D)

**Per-vector storage**: RaBitQ requires more than just the binary code:

- **D-bit binary code**: Sign pattern after rotation (D/8 bytes)
- **⟨ō, o⟩**: Inner product between quantized and original normalized vector (4 bytes)
- **‖o - c‖**: Vector norm relative to cluster centroid (4 bytes)

```c
/*
 * MktRaBitQVector: Quantized vector for RaBitQ
 *
 * PostgreSQL varlena-compatible type storing all components needed for
 * RaBitQ distance estimation. Bits are stored MSB-first for VarBit
 * compatibility in the bit array portion.
 */
typedef struct MktRaBitQVector
{
    int32       vl_len_;        /* varlena header (do not touch directly!) */
    int16       dim;            /* number of dimensions (= number of bits) */
    int16       unused;         /* reserved, always zero */
    float       inner_oo;       /* ⟨ō, o⟩: quantized-original inner product */
    float       norm;           /* ‖o - centroid‖: norm relative to centroid */
    uint8       bits[];         /* D/8 bytes, MSB-first bit ordering */
} MktRaBitQVector;

#define MKT_RABITQ_BITS_SIZE(dim)  (((dim) + 7) / 8)
#define MKT_RABITQ_SIZE(dim)       (offsetof(MktRaBitQVector, bits) + \
                                    MKT_RABITQ_BITS_SIZE(dim))
#define MKT_RABITQ_DIM(v)          ((v)->dim)
#define MKT_RABITQ_BITS(v)         ((v)->bits)
#define MKT_RABITQ_INNER_OO(v)     ((v)->inner_oo)
#define MKT_RABITQ_NORM(v)         ((v)->norm)

// RaBitQ quantizer state (shared across all vectors in an index)
typedef struct {
    float    *P;            // Random orthogonal matrix (dim x dim), row-major
    float    *P_inv;        // P inverse (P^T for orthogonal matrix)
    Dimension dim;
    uint32_t  packed_bytes; // ceil(dim / 8)
} RaBitQParams;

// Initialize with random orthogonal matrix
RaBitQParams *mkt_rabitq_create(Dimension dim, uint64_t seed);
void          mkt_rabitq_destroy(RaBitQParams *params);

// Encode vector to RaBitQ format (includes computing inner_oo and norm)
MktRaBitQVector *mkt_rabitq_encode(
    const RaBitQParams *params,
    VectorRef input,          // Original vector
    VectorRef centroid        // Cluster centroid for norm computation
);

// Encode into pre-allocated buffer
void mkt_rabitq_encode_into(
    const RaBitQParams *params,
    VectorRef input,
    VectorRef centroid,
    MktRaBitQVector *output   // Must be MKT_RABITQ_SIZE(dim) bytes
);

// Asymmetric distance estimation (float query vs quantized DB vector)
// Returns estimated squared L2 distance
Distance mkt_rabitq_distance_asymmetric(
    const RaBitQParams *params,
    VectorRef query,                  // Full precision query
    const MktRaBitQVector *quantized  // Quantized DB vector
);
```

**Encoding implementation:**

```c
void
mkt_rabitq_encode_into(const RaBitQParams *params, VectorRef input,
                       VectorRef centroid, MktRaBitQVector *output)
{
    Dimension dim = params->dim;
    float *normalized = mkt_alloc(dim * sizeof(float));
    float *transformed = mkt_alloc(dim * sizeof(float));

    // Step 1: Compute o = (input - centroid) / ||input - centroid||
    float norm_sq = 0.0f;
    for (Dimension i = 0; i < dim; i++) {
        float diff = input.data[i] - centroid.data[i];
        normalized[i] = diff;
        norm_sq += diff * diff;
    }
    float norm = sqrtf(norm_sq);
    float inv_norm = (norm > 1e-10f) ? 1.0f / norm : 0.0f;
    for (Dimension i = 0; i < dim; i++) {
        normalized[i] *= inv_norm;
    }

    // Step 2: Apply inverse rotation: x = P^T * o (P is orthogonal)
    for (Dimension i = 0; i < dim; i++) {
        float sum = 0.0f;
        for (Dimension j = 0; j < dim; j++) {
            sum += params->P_inv[i * dim + j] * normalized[j];
        }
        transformed[i] = sum;
    }

    // Step 3: Extract sign pattern ō = sign(x), MSB-first bit ordering
    uint32_t packed_bytes = MKT_RABITQ_BITS_SIZE(dim);
    memset(output->bits, 0, packed_bytes);
    for (Dimension i = 0; i < dim; i++) {
        if (transformed[i] > 0.0f) {
            output->bits[i / 8] |= 1 << (7 - (i % 8));  // MSB first
        }
    }

    // Step 4: Compute ⟨ō, o⟩ where ō is the quantized (binary) vector
    // ō[i] = +1/√D if bit set, -1/√D otherwise
    float scale = 1.0f / sqrtf((float)dim);
    float inner_oo = 0.0f;
    for (Dimension i = 0; i < dim; i++) {
        int bit = (output->bits[i / 8] >> (7 - (i % 8))) & 1;
        float o_bar_i = bit ? scale : -scale;
        inner_oo += o_bar_i * normalized[i];
    }

    // Step 5: Fill output structure
    SET_VARSIZE(output, MKT_RABITQ_SIZE(dim));
    output->dim = dim;
    output->unused = 0;
    output->inner_oo = inner_oo;
    output->norm = norm;

    mkt_free(transformed);
    mkt_free(normalized);
}
```

**Random orthogonal matrix generation:**

```c
// Generate random orthogonal matrix via QR decomposition of random Gaussian
static void
generate_orthogonal_matrix(float *P, Dimension dim, uint64_t seed)
{
    // Fill with random Gaussian values
    uint64_t rng = seed;
    for (Dimension i = 0; i < dim * dim; i++) {
        P[i] = random_gaussian(&rng);
    }

    // QR decomposition (Gram-Schmidt or Householder)
    // Result: P is orthogonal (P * P^T = I)
    qr_decomposition_inplace(P, dim);
}
```

**Distance estimation:**

The key formula estimates inner product ⟨o, q⟩ from quantized vectors. RaBitQ uses
the stored `inner_oo` value to correct the estimation error:

```c
Distance
mkt_rabitq_distance_asymmetric(const RaBitQParams *params, VectorRef query,
                               const MktRaBitQVector *quantized)
{
    Dimension dim = params->dim;
    float *q_transformed = mkt_alloc(dim * sizeof(float));

    // Transform query: P^T * query
    for (Dimension i = 0; i < dim; i++) {
        float sum = 0.0f;
        for (Dimension j = 0; j < dim; j++) {
            sum += params->P_inv[i * dim + j] * query.data[j];
        }
        q_transformed[i] = sum;
    }

    // Compute inner product ⟨q_transformed, ō⟩ with binary code
    // ō[i] = +1/√D if bit set, -1/√D otherwise
    float scale = 1.0f / sqrtf((float)dim);
    float inner_q_obar = 0.0f;
    for (Dimension i = 0; i < dim; i++) {
        int bit = (quantized->bits[i / 8] >> (7 - (i % 8))) & 1;  // MSB first
        float obar_i = bit ? scale : -scale;
        inner_q_obar += q_transformed[i] * obar_i;
    }

    // RaBitQ distance formula (see paper Section 4):
    // Estimated ⟨o, q⟩ ≈ ⟨ō, q⟩ / ⟨ō, o⟩  (when vectors are normalized)
    // For unnormalized: ||o - q||² = ||o||² + ||q||² - 2⟨o,q⟩
    //
    // Here o is the normalized residual, so we scale by stored norm:
    // actual_distance² = ||input - centroid||² + ||query - centroid||²
    //                    - 2 * norm * ⟨o, q_normalized⟩
    float inner_oo = quantized->inner_oo;
    float norm_o = quantized->norm;

    // Correct inner product estimate using stored ⟨ō, o⟩
    float inner_oq_est = (inner_oo > 1e-10f)
                       ? (inner_q_obar / inner_oo) * norm_o
                       : 0.0f;

    float norm_q = mkt_vector_norm_ref(query);
    float dist_sq = norm_o * norm_o + norm_q * norm_q - 2.0f * inner_oq_est;

    mkt_free(q_transformed);
    return fmaxf(0.0f, dist_sq);  // Clamp to non-negative
}
```

**SIMD optimization:**

Several RaBitQ operations benefit significantly from SIMD acceleration:

**1. Matrix-vector multiplication (P^T × vector):**

The most expensive operation at O(D²). SIMD parallelizes the dot products:

```c
// AVX2: Process 8 floats per iteration
void
mkt_matvec_avx2(const float *P_inv, const float *input, float *output,
                Dimension dim)
{
    for (Dimension i = 0; i < dim; i++) {
        __m256 sum = _mm256_setzero_ps();
        const float *row = P_inv + i * dim;

        Dimension j = 0;
        for (; j + 8 <= dim; j += 8) {
            __m256 p = _mm256_loadu_ps(row + j);
            __m256 v = _mm256_loadu_ps(input + j);
            sum = _mm256_fmadd_ps(p, v, sum);  // FMA: sum += p * v
        }

        // Horizontal sum of 8 floats
        __m128 hi = _mm256_extractf128_ps(sum, 1);
        __m128 lo = _mm256_castps256_ps128(sum);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        output[i] = _mm_cvtss_f32(sum128);

        // Scalar cleanup for remainder
        for (; j < dim; j++) {
            output[i] += row[j] * input[j];
        }
    }
}
```

**2. Sign extraction (quantization):**

Extract sign bits from 8 floats simultaneously:

```c
// AVX2: Extract 8 sign bits at once
void
mkt_extract_signs_avx2(const float *transformed, uint8_t *bits, Dimension dim)
{
    __m256 zero = _mm256_setzero_ps();

    Dimension i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 v = _mm256_loadu_ps(transformed + i);
        // Compare > 0: returns 0xFFFFFFFF for positive, 0 for negative
        __m256 cmp = _mm256_cmp_ps(v, zero, _CMP_GT_OQ);
        // Extract sign bits (MSB of each 32-bit lane)
        int mask = _mm256_movemask_ps(cmp);
        // Reverse bit order for MSB-first storage
        bits[i / 8] = (uint8_t)__builtin_bitreverse8((uint8_t)mask);
    }

    // Scalar cleanup
    for (; i < dim; i++) {
        if (transformed[i] > 0.0f) {
            bits[i / 8] |= 1 << (7 - (i % 8));
        }
    }
}
```

**3. Inner product with binary code (asymmetric distance):**

This is the hot path for queries. Use masked blending:

```c
// AVX2: Compute ⟨q_transformed, ō⟩ where ō is binary
float
mkt_rabitq_inner_avx2(const float *q_transformed, const uint8_t *bits,
                      Dimension dim)
{
    float scale = 1.0f / sqrtf((float)dim);
    __m256 pos_scale = _mm256_set1_ps(scale);
    __m256 neg_scale = _mm256_set1_ps(-scale);
    __m256 sum = _mm256_setzero_ps();

    Dimension i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 q = _mm256_loadu_ps(q_transformed + i);

        // Load 8 bits, expand to mask
        uint8_t byte = bits[i / 8];
        // Reverse for MSB-first ordering
        byte = __builtin_bitreverse8(byte);

        // Expand bits to 32-bit mask per lane
        __m256i mask = _mm256_set_epi32(
            (byte >> 7) & 1 ? -1 : 0,
            (byte >> 6) & 1 ? -1 : 0,
            (byte >> 5) & 1 ? -1 : 0,
            (byte >> 4) & 1 ? -1 : 0,
            (byte >> 3) & 1 ? -1 : 0,
            (byte >> 2) & 1 ? -1 : 0,
            (byte >> 1) & 1 ? -1 : 0,
            (byte >> 0) & 1 ? -1 : 0
        );

        // Blend: select pos_scale if bit=1, neg_scale if bit=0
        __m256 obar = _mm256_blendv_ps(neg_scale, pos_scale,
                                        _mm256_castsi256_ps(mask));
        sum = _mm256_fmadd_ps(q, obar, sum);
    }

    // Horizontal sum
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float result = _mm_cvtss_f32(sum128);

    // Scalar cleanup
    for (; i < dim; i++) {
        int bit = (bits[i / 8] >> (7 - (i % 8))) & 1;
        float obar_i = bit ? scale : -scale;
        result += q_transformed[i] * obar_i;
    }
    return result;
}
```

**4. Hamming distance (symmetric comparisons):**

XOR + popcount is extremely fast with SIMD:

```c
// AVX2 with POPCNT
uint32_t
mkt_hamming_avx2(const uint8_t *a, const uint8_t *b, uint32_t bytes)
{
    uint64_t total = 0;
    uint32_t i = 0;

    // Process 32 bytes (256 bits) at a time
    for (; i + 32 <= bytes; i += 32) {
        __m256i va = _mm256_loadu_si256((__m256i *)(a + i));
        __m256i vb = _mm256_loadu_si256((__m256i *)(b + i));
        __m256i xored = _mm256_xor_si256(va, vb);

        // Extract and popcount (AVX2 lacks native popcount)
        uint64_t lo = _mm256_extract_epi64(xored, 0);
        uint64_t hi = _mm256_extract_epi64(xored, 1);
        uint64_t lo2 = _mm256_extract_epi64(xored, 2);
        uint64_t hi2 = _mm256_extract_epi64(xored, 3);
        total += __builtin_popcountll(lo) + __builtin_popcountll(hi)
               + __builtin_popcountll(lo2) + __builtin_popcountll(hi2);
    }

    // Scalar cleanup
    for (; i < bytes; i++) {
        total += __builtin_popcount(a[i] ^ b[i]);
    }
    return (uint32_t)total;
}

// AVX512-VPOPCNTDQ: Native vector popcount (much faster)
uint32_t
mkt_hamming_avx512(const uint8_t *a, const uint8_t *b, uint32_t bytes)
{
    __m512i total = _mm512_setzero_si512();
    uint32_t i = 0;

    for (; i + 64 <= bytes; i += 64) {
        __m512i va = _mm512_loadu_si512(a + i);
        __m512i vb = _mm512_loadu_si512(b + i);
        __m512i xored = _mm512_xor_si512(va, vb);
        // Native 64-bit popcount per lane
        total = _mm512_add_epi64(total, _mm512_popcnt_epi64(xored));
    }

    return (uint32_t)_mm512_reduce_add_epi64(total);
}
```

**5. Vector norm computation:**

```c
float
mkt_norm_sq_avx2(const float *v, Dimension dim)
{
    __m256 sum = _mm256_setzero_ps();
    Dimension i = 0;

    for (; i + 8 <= dim; i += 8) {
        __m256 x = _mm256_loadu_ps(v + i);
        sum = _mm256_fmadd_ps(x, x, sum);  // sum += x * x
    }

    // Horizontal sum
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float result = _mm_cvtss_f32(sum128);

    // Scalar cleanup
    for (; i < dim; i++) {
        result += v[i] * v[i];
    }
    return result;
}
```

**Performance impact summary:**

| Operation          | Scalar | AVX2       | AVX512      | Notes                |
|--------------------|--------|------------|-------------|----------------------|
| Mat-vec multiply   | O(D²)  | ~4× faster | ~8× faster  | FMA critical         |
| Sign extraction    | O(D)   | ~8× faster | ~16× faster | movemask             |
| Asymmetric inner   | O(D)   | ~4-6× faster | ~8-12× faster | Blend + FMA        |
| Hamming distance   | O(D/8) | ~4× faster | ~8-16× faster | VPOPCNTDQ huge win |
| Norm computation   | O(D)   | ~8× faster | ~16× faster | FMA                  |

The matrix-vector multiplication dominates encoding time, while the asymmetric
inner product dominates query time. Both benefit significantly from SIMD.

**Storage requirements:**

- `MktRaBitQVector`: 4 (vl_len) + 2 (dim) + 2 (unused) + 4 (inner_oo) + 4 (norm)
  + D/8 (bits) = 16 + D/8 bytes per vector
- For 768-dim: 16 + 96 = 112 bytes per vector
- Orthogonal matrix P: D² × 4 bytes (shared across all vectors in index)

Binary codes are stored in VarBit-compatible format (MSB-first bit ordering)
for interoperability with pgvector's `bit` type and distance functions.

For 768-dim vectors: 100 bytes/vector vs 3072 bytes for float32 (30x compression).

#### Scalar Quantization (SQ8)

Each dimension is linearly mapped from [min, max] to [0, 255].

```c
// Quantization parameters (learned from data)
typedef struct {
    float *mins;    // Per-dimension minimums
    float *maxs;    // Per-dimension maximums
    float *scales;  // (max - min) / 255 per dimension
    Dimension dim;
} SQ8Params;

// Learn quantization parameters from a sample of vectors
SQ8Params *mkt_sq8_learn(
    const float *vectors,
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
    const float *inputs,
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
    const float *mins = params->mins;
    const float *scales = params->scales;
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

#### Binary Quantization (BQ)

For very fast approximate filtering before SQ8 re-ranking:

```c
// Each dimension: 1 if >= threshold, 0 otherwise
// Packed: 8 dimensions per byte
typedef struct {
    float    *thresholds;   // Per-dimension thresholds (typically 0 or mean)
    Dimension dim;
    uint32_t  packed_bytes; // ceil(dim / 8)
} BQParams;

void mkt_bq_encode(const BQParams *params, VectorRef input, BinaryQ *output);

// Hamming distance between binary vectors
uint32_t mkt_bq_distance_hamming(
    const BinaryQ *a,
    const BinaryQ *b,
    uint32_t packed_bytes
);
```

#### Statistical Binary Quantization (SBQ)

SBQ improves on simple binary quantization by learning per-dimension statistics
(mean and optionally variance) from the data. Used by pgvectorscale.

**1-bit mode**: Compare each dimension to its learned mean.

**Multi-bit mode** (2-4 bits): Use z-scores to map values into bins, encoded as
thermometer codes (unary: 0=000, 1=100, 2=110, 3=111).

```c
// SBQ parameters learned from data using Welford's online algorithm
typedef struct {
    float    *mean;         // Per-dimension mean
    float    *m2;           // Per-dimension sum of squared differences (for variance)
    uint64_t  count;        // Number of training samples
    Dimension dim;
    uint8_t   bits_per_dim; // 1, 2, or 4 bits per dimension
} SBQParams;

// Online training (Welford's algorithm for numerical stability)
void mkt_sbq_start_training(SBQParams *params, Dimension dim, uint8_t bits);
void mkt_sbq_add_sample(SBQParams *params, VectorRef sample);
void mkt_sbq_finish_training(SBQParams *params);

// Quantize vector
void mkt_sbq_encode(const SBQParams *params, VectorRef input, BinaryQ *output);

// For 1-bit: use Hamming distance (same as BQ)
// For multi-bit: use popcount on thermometer codes
```

**1-bit encoding** (MSB-first for VarBit compatibility):
```c
for (Dimension i = 0; i < dim; i++) {
    if (input.data[i] > params->mean[i]) {
        output[i / 8] |= 1 << (7 - (i % 8));  // MSB first
    }
}
```

**Multi-bit encoding** (thermometer code, MSB-first):
```c
for (Dimension i = 0; i < dim; i++) {
    float variance = params->m2[i] / params->count;
    float std_dev = sqrtf(variance);
    float z_score = (input.data[i] - params->mean[i]) / std_dev;

    // Map z-score [-2, 2] to bins [0, num_bits]
    int bins = params->bits_per_dim + 1;
    int index = (int)((z_score + 2.0f) / (4.0f / bins));
    int count_ones = clamp(index, 0, params->bits_per_dim);

    // Write thermometer code (e.g., 2 ones = 110), MSB-first
    int bit_pos = i * params->bits_per_dim;
    for (int j = 0; j < count_ones; j++) {
        int pos = bit_pos + j;
        output[pos / 8] |= 1 << (7 - (pos % 8));  // MSB first
    }
}
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
    float   *centroids;     // nlist * dim floats
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
    const float *vectors,
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
    const float *vectors,
    uint32_t count
);

KMeansResult *mkt_kmeans_stream_finish(MktKMeansStream *stream);
void mkt_kmeans_stream_destroy(MktKMeansStream *stream);

// Find medoid for each cluster (actual vector closest to centroid)
void mkt_kmeans_compute_medoids(
    KMeansResult *result,
    const float *vectors,
    uint32_t nvecs
);

// Free result
void mkt_kmeans_result_destroy(KMeansResult *result);
```

#### K-Means++ Initialization

Better initialization for faster convergence:

```c
static void kmeans_plusplus_init(
    const float *vectors,
    uint32_t nvecs,
    Dimension dim,
    uint32_t nlist,
    float *centroids,
    uint64_t seed
) {
    // Random state
    uint64_t rng = seed;

    // Pick first centroid uniformly at random
    uint32_t first = xorshift64(&rng) % nvecs;
    memcpy(centroids, vectors + first * dim, dim * sizeof(float));

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

        memcpy(centroids + c * dim, vectors + next * dim, dim * sizeof(float));
    }

    mkt_free(min_distances);
}
```

#### Lloyd's Algorithm

```c
KMeansResult *mkt_kmeans(
    const float *vectors,
    uint32_t nvecs,
    Dimension dim,
    uint32_t nlist,
    DistanceMetric metric,
    const KMeansOptions *opts
) {
    KMeansResult *result = mkt_alloc0(sizeof(KMeansResult));
    result->nlist = nlist;
    result->dim = dim;
    result->centroids = mkt_alloc_aligned(nlist * dim * sizeof(float), 64);
    result->assignments = mkt_alloc(nvecs * sizeof(ClusterId));
    result->cluster_sizes = mkt_alloc0(nlist * sizeof(uint32_t));

    // Initialize centroids
    kmeans_plusplus_init(vectors, nvecs, dim, nlist, result->centroids, opts->seed);

    // Temporary storage for centroid updates
    float *new_centroids = mkt_alloc0(nlist * dim * sizeof(float));
    uint32_t *counts = mkt_alloc0(nlist * sizeof(uint32_t));

    for (uint32_t iter = 0; iter < opts->max_iterations; iter++) {
        // Reset accumulators
        memset(new_centroids, 0, nlist * dim * sizeof(float));
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
    const float *vectors,
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
               dim * sizeof(float));
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

### 2.5 Quantization Benchmark

**Files**: `tools/mkt_quant_bench.c`, `src/bench/quant_bench.h`, `src/bench/quant_bench.c`

A standalone benchmark tool compares quantization methods against full-precision
vectors using standard ANN datasets. This validates correctness and measures
the accuracy/speed tradeoff before PostgreSQL integration.

#### Datasets

Standard ANN benchmark datasets used by RaBitQ (SIGMOD 2024), SPANN, and
ANN-benchmarks. These cover diverse dimensionalities and data distributions:

| Dataset    | Vectors    | Dimensions | Description                    |
|------------|------------|------------|--------------------------------|
| SIFT1M     | 1,000,000  | 128        | SIFT image descriptors         |
| SIFT10M    | 10,000,000 | 128        | Large-scale SIFT               |
| GIST1M     | 1,000,000  | 960        | GIST image descriptors         |
| Deep1M     | 1,000,000  | 256        | Deep learning features         |
| GloVe-1.2M | 1,193,514  | 200        | Word embeddings (Wikipedia)    |
| GloVe-2.2M | 2,196,017  | 300        | Word embeddings (Common Crawl) |
| Tiny5M     | 5,000,000  | 384        | Tiny images features           |
| MSong      | 994,185    | 420        | Million Song audio features    |
| Word2Vec   | 1,000,000  | 300        | Word embeddings (Google News)  |

**Dataset sources:**
- TEXMEX corpus: `ftp://ftp.irisa.fr/local/texmex/corpus/`
- GQR datasets: `https://www.cse.cuhk.edu.hk/systems/hash/gqr/datasets.html`
- ANN-benchmarks: `https://github.com/erikbern/ann-benchmarks`

Dataset files use standard formats:
- `.fvecs` / `.ivecs`: Little-endian vectors with 4-byte dimension prefix
- `.bvecs`: Byte vectors with 4-byte dimension prefix
- Ground truth: `.ivecs` files with true k-nearest neighbors

```c
// Dataset loader
typedef struct {
    float    *vectors;      // Base vectors (nvecs × dim)
    float    *queries;      // Query vectors (nqueries × dim)
    uint32_t *groundtruth;  // True k-NN indices (nqueries × k)
    uint32_t  nvecs;
    uint32_t  nqueries;
    uint32_t  dim;
    uint32_t  k;            // Ground truth k
} BenchDataset;

BenchDataset *mkt_bench_load_fvecs(const char *base_path,
                                    const char *query_path,
                                    const char *gt_path);
void          mkt_bench_dataset_free(BenchDataset *ds);
```

#### Metrics

**Accuracy metrics:**

```c
typedef struct {
    float recall_at_1;      // Fraction where true NN is in result
    float recall_at_10;     // Fraction of true 10-NN in top-10 results
    float recall_at_100;    // Fraction of true 100-NN in top-100 results
    float mean_rank_error;  // Average rank difference from true rank
    float distance_ratio;   // Avg(estimated_dist / true_dist)
    float distance_rmse;    // RMS error in distance estimates
} AccuracyMetrics;

// Compute recall: what fraction of true k-NN are in the result set
float
mkt_bench_recall(const uint32_t *true_nn, const uint32_t *result_nn,
                 uint32_t k_true, uint32_t k_result)
{
    uint32_t found = 0;
    for (uint32_t i = 0; i < k_true; i++) {
        for (uint32_t j = 0; j < k_result; j++) {
            if (true_nn[i] == result_nn[j]) {
                found++;
                break;
            }
        }
    }
    return (float)found / (float)k_true;
}
```

**Performance metrics:**

```c
typedef struct {
    double encode_time_ms;      // Time to encode all vectors
    double query_time_us;       // Avg time per query (microseconds)
    double throughput_qps;      // Queries per second
    size_t memory_bytes;        // Memory for quantized vectors
    float  compression_ratio;   // Original size / quantized size
} PerformanceMetrics;
```

#### Benchmark Runner

```c
typedef enum {
    QUANT_METHOD_NONE = 0,  // Full precision (baseline)
    QUANT_METHOD_RABITQ,
    QUANT_METHOD_SQ8,
    QUANT_METHOD_SBQ,
    QUANT_METHOD_BQ,
    QUANT_METHOD_PQ,
} QuantMethod;

typedef struct {
    QuantMethod method;
    union {
        RaBitQParams  *rabitq;
        SQ8Params     *sq8;
        SBQParams     *sbq;
        BQParams      *bq;
        PQParams      *pq;
    } params;
} QuantConfig;

typedef struct {
    QuantMethod         method;
    AccuracyMetrics     accuracy;
    PerformanceMetrics  perf;
} BenchResult;

// Run benchmark for a single quantization method
BenchResult
mkt_bench_run_method(const BenchDataset *ds, const QuantConfig *config,
                     uint32_t k)
{
    BenchResult result = { .method = config->method };
    uint64_t start, end;

    // 1. Encode all base vectors
    start = mkt_time_ns();
    void *quantized = encode_vectors(ds->vectors, ds->nvecs, ds->dim, config);
    end = mkt_time_ns();
    result.perf.encode_time_ms = (double)(end - start) / 1e6;
    result.perf.memory_bytes = get_quantized_size(ds->nvecs, ds->dim, config);
    result.perf.compression_ratio =
        (float)(ds->nvecs * ds->dim * sizeof(float)) /
        (float)result.perf.memory_bytes;

    // 2. Run queries and measure accuracy
    uint32_t *results = mkt_alloc(ds->nqueries * k * sizeof(uint32_t));
    float total_recall = 0.0f;
    float total_dist_ratio = 0.0f;

    start = mkt_time_ns();
    for (uint32_t q = 0; q < ds->nqueries; q++) {
        VectorRef query = {
            .data = ds->queries + q * ds->dim,
            .dim = ds->dim
        };

        // Find k-NN using quantized distances
        find_knn_quantized(quantized, ds->nvecs, query, config,
                           k, results + q * k);
    }
    end = mkt_time_ns();

    result.perf.query_time_us =
        (double)(end - start) / (double)ds->nqueries / 1000.0;
    result.perf.throughput_qps =
        (double)ds->nqueries / ((double)(end - start) / 1e9);

    // 3. Compute accuracy metrics
    for (uint32_t q = 0; q < ds->nqueries; q++) {
        total_recall += mkt_bench_recall(
            ds->groundtruth + q * ds->k,
            results + q * k,
            MIN(ds->k, k), k);

        // Distance ratio for first result
        VectorRef query = { .data = ds->queries + q * ds->dim, .dim = ds->dim };
        VectorRef true_nn = {
            .data = ds->vectors + ds->groundtruth[q * ds->k] * ds->dim,
            .dim = ds->dim
        };
        VectorRef found_nn = {
            .data = ds->vectors + results[q * k] * ds->dim,
            .dim = ds->dim
        };
        float true_dist = mkt_distance_l2(query, true_nn);
        float found_dist = mkt_distance_l2(query, found_nn);
        if (true_dist > 1e-10f) {
            total_dist_ratio += found_dist / true_dist;
        }
    }

    result.accuracy.recall_at_10 = total_recall / (float)ds->nqueries;
    result.accuracy.distance_ratio = total_dist_ratio / (float)ds->nqueries;

    mkt_free(results);
    free_quantized(quantized, config);
    return result;
}

// Run all methods and compare
void
mkt_bench_run_all(const BenchDataset *ds, uint32_t k)
{
    QuantMethod methods[] = {
        QUANT_METHOD_NONE,
        QUANT_METHOD_RABITQ,
        QUANT_METHOD_SQ8,
        QUANT_METHOD_SBQ,
        QUANT_METHOD_BQ,
    };
    size_t nmethods = sizeof(methods) / sizeof(methods[0]);

    printf("Quantization Benchmark: %u vectors, %u dims, %u queries, k=%u\n\n",
           ds->nvecs, ds->dim, ds->nqueries, k);
    printf("%-10s %8s %8s %10s %10s %8s\n",
           "Method", "Recall@10", "DistRatio", "Encode(ms)", "Query(us)",
           "Compress");
    printf("%-10s %8s %8s %10s %10s %8s\n",
           "------", "--------", "---------", "----------", "---------",
           "--------");

    for (size_t i = 0; i < nmethods; i++) {
        QuantConfig config = create_config(methods[i], ds->dim);
        BenchResult r = mkt_bench_run_method(ds, &config, k);

        printf("%-10s %8.4f %8.4f %10.1f %10.1f %7.1fx\n",
               quant_method_name(r.method),
               r.accuracy.recall_at_10,
               r.accuracy.distance_ratio,
               r.perf.encode_time_ms,
               r.perf.query_time_us,
               r.perf.compression_ratio);

        destroy_config(&config);
    }
}
```

#### CLI Tool

**Tool**: `mkt_quant_bench`

```
$ mkt_quant_bench --dataset sift1m --k 10

Quantization Benchmark: 1000000 vectors, 128 dims, 10000 queries, k=10

Method     Recall@10 DistRatio  Encode(ms)  Query(us)  Compress
------     --------- ---------  ----------  ---------  --------
none         1.0000    1.0000         0.0      125.3     1.0x
rabitq       0.9847    1.0023       245.2       12.4    32.0x
sq8          0.9912    1.0008       102.3       18.7     4.0x
sbq          0.9756    1.0045       312.5       10.2    32.0x
bq           0.8234    1.0312        45.1        5.3    32.0x

$ mkt_quant_bench --dataset gist1m --k 100 --methods rabitq,sq8

Quantization Benchmark: 1000000 vectors, 960 dims, 1000 queries, k=100

Method     Recall@100 DistRatio  Encode(ms)  Query(us)  Compress
------     ---------- ---------  ----------  ---------  --------
none          1.0000    1.0000         0.0      892.1     1.0x
rabitq        0.9723    1.0089      1823.4       89.2    32.0x
sq8           0.9801    1.0034       567.2      134.5     4.0x
```

**Options:**

```
mkt_quant_bench - Quantization method benchmark

Usage: mkt_quant_bench [OPTIONS]

Options:
  --dataset <name>     Dataset name (sift1m, gist1m, glove, deep1m, spacev)
  --base <path>        Path to base vectors (.fvecs)
  --query <path>       Path to query vectors (.fvecs)
  --gt <path>          Path to ground truth (.ivecs)
  --k <int>            Number of neighbors (default: 10)
  --methods <list>     Comma-separated methods (default: all)
  --nprobes <int>      For clustered search (default: 10)
  --seed <int>         Random seed (default: 42)
  --output <path>      Write results to CSV
  --verbose            Show per-query statistics
```

#### Experimental Methodology

Inspired by the RaBitQ paper (SIGMOD 2024), the benchmark evaluates quantization
methods across multiple dimensions:

**1. Accuracy vs Compression Trade-off**

Plot recall@k against compression ratio for each method. This shows the Pareto
frontier of accuracy-efficiency trade-offs:

```
Recall@10
1.00 |  * (none)
     |     * (sq8)  * (rabitq)
0.95 |                 * (sbq)
     |
0.90 |                        * (bq)
     |
     +--------------------------------
         1x    4x    16x    32x  Compression
```

**2. Distance Ratio Distribution**

For each query, compute ratio = estimated_distance / true_distance. Plot the
distribution to understand error characteristics:

- Mean ratio close to 1.0 indicates unbiased estimation
- Low variance indicates consistent accuracy
- Outliers indicate potential failure cases

```c
typedef struct {
    float mean;
    float std;
    float p50;      // Median
    float p95;      // 95th percentile
    float p99;      // 99th percentile
    float max;
} DistRatioStats;

DistRatioStats
mkt_bench_distance_ratio_stats(const float *ratios, uint32_t n);
```

**3. Recall vs Query Time (QPS)**

Measure queries-per-second at various recall targets. Methods that achieve
higher recall at the same QPS are superior:

```
$ mkt_quant_bench --dataset sift1m --sweep-recall 0.90,0.95,0.99

Target    Method      Actual     QPS
------    ------      ------     ------
0.90      rabitq      0.912      85,234
0.90      sq8         0.908      62,451
0.95      rabitq      0.953      71,892
0.95      sq8         0.951      48,123
0.99      rabitq      0.991      42,156
0.99      sq8         0.992      28,934
```

**4. Scalability Analysis**

Run benchmarks across dataset sizes to understand scaling behavior:

```c
// Measure how metrics scale with dataset size
void
mkt_bench_scalability(const char *dataset_path, QuantMethod method,
                      uint32_t sizes[], size_t nsizes)
{
    for (size_t i = 0; i < nsizes; i++) {
        BenchDataset *ds = load_subset(dataset_path, sizes[i]);
        BenchResult r = mkt_bench_run_method(ds, method, 10);
        printf("%8u vectors: recall=%.4f, query=%.1fus\n",
               sizes[i], r.accuracy.recall_at_10, r.perf.query_time_us);
        mkt_bench_dataset_free(ds);
    }
}
```

**5. Dimensionality Impact**

RaBitQ's theoretical error bound is O(1/√D), meaning higher dimensions should
yield better accuracy. Validate this empirically:

| Dimensions | RaBitQ Recall@10 | BQ Recall@10 | Theory Predicts |
|------------|------------------|--------------|-----------------|
| 128        | 0.9847           | 0.8234       | RaBitQ better   |
| 256        | 0.9912           | 0.8567       | RaBitQ better   |
| 512        | 0.9945           | 0.8789       | RaBitQ better   |
| 960        | 0.9967           | 0.8912       | RaBitQ better   |

**6. IVF Integration**

Test quantization within an IVF (Inverted File) index structure, which is
the target deployment scenario:

```c
typedef struct {
    uint32_t nlist;         // Number of clusters
    uint32_t nprobe;        // Clusters to search
    QuantMethod method;     // Quantization for posting lists
} IVFConfig;

// Measure recall vs nprobe trade-off
void
mkt_bench_ivf_nprobe(const BenchDataset *ds, const IVFConfig *config,
                     uint32_t nprobes[], size_t n)
{
    // Build IVF index with clustering
    IVFIndex *idx = build_ivf_index(ds, config);

    for (size_t i = 0; i < n; i++) {
        config->nprobe = nprobes[i];
        float recall = measure_recall(idx, ds->queries, ds->groundtruth);
        float qps = measure_qps(idx, ds->queries);
        printf("nprobe=%3u: recall=%.4f, qps=%.0f\n",
               nprobes[i], recall, qps);
    }
}
```

**References:**
- RaBitQ paper: Gao & Long, "RaBitQ: Quantizing High-Dimensional Vectors with
  a Theoretical Error Bound for Approximate Nearest Neighbor Search", SIGMOD 2024
- Extended-RaBitQ: "Practical and Asymptotically Optimal Quantization of
  High-Dimensional Vectors in Euclidean Space", SIGMOD 2025
- RaBitQ implementation: https://github.com/gaoj0017/RaBitQ

#### Test Cases

```c
TEST_GROUP(quant_bench);

TEST(recall_computation)
{
    uint32_t true_nn[] = {1, 5, 3, 8, 2};
    uint32_t result[]  = {1, 3, 7, 2, 9};
    float recall = mkt_bench_recall(true_nn, result, 5, 5);
    // Found: 1, 3, 2 = 3/5
    ASSERT_FLOAT_EQ(0.6f, recall, 0.001f, "recall should be 0.6");
}

TEST(fvecs_loader)
{
    // Test with small embedded dataset
    BenchDataset *ds = mkt_bench_load_fvecs(
        "testdata/small_base.fvecs",
        "testdata/small_query.fvecs",
        "testdata/small_gt.ivecs"
    );
    ASSERT_NOT_NULL(ds, "dataset should load");
    ASSERT_EQ(1000, ds->nvecs, "should have 1000 vectors");
    ASSERT_EQ(128, ds->dim, "should be 128-dim");
    mkt_bench_dataset_free(ds);
}

TEST(baseline_perfect_recall)
{
    // Full precision should achieve perfect recall
    BenchDataset *ds = load_test_dataset();
    QuantConfig config = { .method = QUANT_METHOD_NONE };
    BenchResult r = mkt_bench_run_method(ds, &config, 10);
    ASSERT_FLOAT_EQ(1.0f, r.accuracy.recall_at_10, 0.001f,
                    "full precision should have perfect recall");
    mkt_bench_dataset_free(ds);
}
```

#### Integration with CI

The benchmark can run in CI with smaller datasets to catch regressions:

```yaml
# .github/workflows/benchmark.yml
benchmark:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - name: Build
      run: |
        meson setup builddir
        meson compile -C builddir
    - name: Download test dataset
      run: |
        wget -q https://example.com/sift10k.tar.gz
        tar xzf sift10k.tar.gz -C testdata/
    - name: Run benchmark
      run: |
        ./builddir/tools/mkt_quant_bench \
          --base testdata/sift10k/base.fvecs \
          --query testdata/sift10k/query.fvecs \
          --gt testdata/sift10k/gt.ivecs \
          --output benchmark_results.csv
    - name: Check regression
      run: |
        python scripts/check_benchmark.py benchmark_results.csv \
          --min-recall 0.95 --max-query-time 50
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

// Access quantized data (const for reading, non-const for writing)
static inline const ScalarQ8 *
posting_entry_quantized(const PostingEntry *entry)
{
    return (const ScalarQ8 *)(entry + 1);
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

// Iteration (read-only traversal)
typedef struct {
    const MemPostingList *list;
    uint32_t              index;
} PostingListIter;

PostingListIter       mkt_posting_list_iter(const MemPostingList *list);
const PostingEntry   *mkt_posting_list_next(PostingListIter *iter);
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
const PostingEntry *mkt_page_get_entry(const void *page, uint32_t index,
                                       Dimension dim);

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
    float       *sample_vectors;
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
    const float *vectors,
    const ItemPointer *tids,  // Can be NULL for standalone testing
    uint32_t count
);

// Phase 2: Perform clustering on sample
void mkt_build_cluster(BuildContext *ctx);

// Phase 3: Assign vectors to clusters and build posting lists
// Call repeatedly with batches (can be same vectors as sampling, or full scan)
void mkt_build_assign(
    BuildContext *ctx,
    const float *vectors,
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
    const float *vectors,
    uint32_t count
);

// Finish pass 1: perform clustering
void mkt_stream_build_finish_sampling(MktStreamBuild *builder);

// Pass 2: Assignment (call multiple times)
// Returns pages to write via callback
void mkt_stream_build_assign(
    MktStreamBuild *builder,
    const float *vectors,
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
    float   *centroids;   // Flat array: nlist * dim
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
    const float *centroid
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
                const PostingEntry *entry = mkt_page_get_entry(
                    page, i, query.dim
                );

                // Skip deleted entries
                if (entry->flags & POSTING_FLAG_DELETED) continue;

                // Skip centroid in results (used for navigation only)
                if (entry->flags & POSTING_FLAG_CENTROID) continue;

                // Compute approximate distance using LUTs
                const ScalarQ8 *quantized = posting_entry_quantized(entry);
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
    float *output  // Pre-allocated buffer
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

    float *vec_buffer = mkt_alloc_aligned(dim * sizeof(float), 64);

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

### 6.2 Vector Type and pgvector Compatibility

**Files**: `src/mkt_vector.h`, `src/mkt_vector.c`

Meerkat defines its own vector type (`mkt_vector`) that is binary-compatible with
pgvector's `vector` type. This allows:

- Standalone builds without pgvector dependency
- Direct indexing of pgvector columns without casts
- Zero-overhead type handling (same memory layout)

#### Type Definition

pgvector's vector type (from `pgvector/src/vector.h`):

```c
#define VECTOR_MAX_DIM 16000
#define VECTOR_SIZE(_dim) (offsetof(Vector, x) + sizeof(float)*(_dim))

typedef struct Vector
{
    int32       vl_len_;        /* varlena header (do not touch directly!) */
    int16       dim;            /* number of dimensions */
    int16       unused;         /* reserved for future use, always zero */
    float       x[FLEXIBLE_ARRAY_MEMBER];
}           Vector;
```

Meerkat's binary-compatible definition:

```c
#define MKT_VECTOR_MAX_DIM 16000
#define MKT_VECTOR_SIZE(_dim) (offsetof(MktVector, x) + sizeof(float)*(_dim))

typedef struct MktVector
{
    int32       vl_len_;        /* varlena header (do not touch directly!) */
    int16       dim;            /* number of dimensions */
    int16       unused;         /* reserved for future use, always zero */
    float       x[FLEXIBLE_ARRAY_MEMBER];
}           MktVector;

// Accessor macros
#define MKT_VECTOR_DIM(v)    ((v)->dim)
#define MKT_VECTOR_DATA(v)   ((v)->x)

// Convert to VectorRef for internal operations
static inline VectorRef
MktVectorToRef(const MktVector *v)
{
    return (VectorRef){ .data = v->x, .dim = v->dim };
}
```

#### pgvector Detection and OID Caching

At extension load time, detect if pgvector is installed and cache its type OID:

```c
// Cached OIDs (InvalidOid if not available)
static Oid mkt_vector_oid = InvalidOid;
static Oid pgvector_oid = InvalidOid;

void
mkt_vector_init(void)
{
    // Cache our own type OID
    mkt_vector_oid = GetSysCacheOid2(TYPENAMENSP,
                                      Anum_pg_type_oid,
                                      CStringGetDatum("mkt_vector"),
                                      ObjectIdGetDatum(get_namespace_oid("public",
                                                                          false)));

    // Check if pgvector is installed (type exists in pg_type)
    Oid vector_oid = GetSysCacheOid2(TYPENAMENSP,
                                      Anum_pg_type_oid,
                                      CStringGetDatum("vector"),
                                      ObjectIdGetDatum(get_namespace_oid("public",
                                                                          false)));
    if (OidIsValid(vector_oid)) {
        pgvector_oid = vector_oid;
        elog(DEBUG1, "meerkat: pgvector detected, OID %u", pgvector_oid);
    }
}

// Check if a type OID is a supported vector type
static inline bool
mkt_is_vector_type(Oid typoid)
{
    return typoid == mkt_vector_oid ||
           (OidIsValid(pgvector_oid) && typoid == pgvector_oid);
}
```

#### Runtime Type Handling in Operators

Operators and index functions accept both `mkt_vector` and pgvector's `vector`:

```c
Datum
mkt_l2_distance(PG_FUNCTION_ARGS)
{
    Oid argtypoid = get_fn_expr_argtype(fcinfo->flinfo, 0);

    if (!mkt_is_vector_type(argtypoid))
        ereport(ERROR,
                errcode(ERRCODE_DATATYPE_MISMATCH),
                errmsg("expected mkt_vector or vector type"));

    // Binary compatible - safe to cast regardless of which type
    MktVector *a = (MktVector *) PG_DETOAST_DATUM(PG_GETARG_DATUM(0));
    MktVector *b = (MktVector *) PG_DETOAST_DATUM(PG_GETARG_DATUM(1));

    if (a->dim != b->dim)
        ereport(ERROR,
                errcode(ERRCODE_DATA_EXCEPTION),
                errmsg("different vector dimensions %d and %d", a->dim, b->dim));

    VectorRef va = MktVectorToRef(a);
    VectorRef vb = MktVectorToRef(b);

    PG_RETURN_FLOAT4(mkt_distance_l2(va, vb));
}
```

#### Index Support for Both Types

The index access method validates column types during `ambuild`:

```c
static IndexBuildResult *
mkt_ambuild(Relation heap, Relation index, IndexInfo *indexInfo)
{
    // Get the indexed column's type
    Oid atttype = TupleDescAttr(RelationGetDescr(heap),
                                 indexInfo->ii_IndexAttrNumbers[0] - 1)->atttypid;

    if (!mkt_is_vector_type(atttype))
        ereport(ERROR,
                errcode(ERRCODE_DATATYPE_MISMATCH),
                errmsg("meerkat index requires mkt_vector or vector column"));

    // Proceed with build - both types have identical layout
    // ...
}
```

**Standalone builds**: For unit tests and CLI tools, use `MktVector` without the
varlena header, or define a minimal mock. The core algorithms operate on
`VectorRef`, which is independent of PostgreSQL types.

### 6.3 Index Access Method Handler

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

### 6.4 Buffer Cache Integration

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

### 6.5 Shared Memory Centroid Cache

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

### 6.6 Scan State

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

### 6.7 Cost Estimation

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

### 6.8 MVCC Support

```c
// Check visibility during posting list scan
static bool
entry_is_visible(const PostingEntry *entry, Snapshot snapshot, Relation heap)
{
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

## Part 8: Filtered Vector Search (Future)

This section covers filtered vector search support, similar to pgvectorscale's
streaming DiskANN with pre-filtering. This is a later feature but requires
architectural consideration early on.

### 8.1 Design Goals

**Functional requirements:**
- Support arbitrary SQL predicates alongside vector similarity
- Pre-filtering: apply predicates before/during vector search
- Post-filtering: rerank results after applying predicates
- Efficient handling of highly selective filters

**Architectural requirements:**
- Streaming result iterator (request K results, then K more, etc.)
- Early termination when enough results found
- Filter pushdown into posting list scan
- No degradation for unfiltered queries

### 8.2 Streaming Search Iterator

The search must support iterative result retrieval. This is essential for:
- Filtered search where many candidates may be filtered out
- LIMIT queries where we don't know K upfront
- Cursor-based pagination

```c
/*
 * MktSearchIterator: Streaming search state
 *
 * Allows fetching results incrementally. The iterator maintains state
 * between calls, enabling "get 10 more" semantics.
 */
typedef struct MktSearchIterator
{
    /* Search parameters (immutable after init) */
    VectorRef           query;
    DistanceType        distance_type;
    uint32_t            nprobe;         /* clusters to search */

    /* Filter state */
    MktFilterContext   *filter;         /* NULL if no filter */

    /* Cluster iteration state */
    ClusterId          *probe_clusters; /* clusters ordered by distance */
    uint32_t            n_probe_clusters;
    uint32_t            current_cluster;

    /* Within-cluster state */
    MktPostingIterator *posting_iter;   /* current cluster's posting list */

    /* Result buffer (min-heap by distance) */
    MktResultHeap      *candidates;     /* candidates not yet returned */
    uint32_t            returned_count; /* results already returned */

    /* Reranking buffer */
    MktResultHeap      *rerank_buffer;  /* for full-precision reranking */
    uint32_t            rerank_size;    /* how many to collect before rerank */

    /* Statistics */
    uint64_t            vectors_scanned;
    uint64_t            vectors_filtered;
    uint64_t            clusters_visited;
} MktSearchIterator;

/* Initialize streaming search */
MktSearchIterator *
mkt_search_iterator_create(
    MktIndex *index,
    VectorRef query,
    DistanceType distance_type,
    uint32_t nprobe,
    MktFilterContext *filter    /* NULL for unfiltered */
);

/* Fetch next batch of results */
uint32_t
mkt_search_iterator_next(
    MktSearchIterator *iter,
    uint32_t max_results,       /* how many to fetch */
    MktSearchResult *results    /* output buffer */
);

/* Check if more results available */
bool
mkt_search_iterator_has_more(const MktSearchIterator *iter);

/* Get statistics */
void
mkt_search_iterator_stats(
    const MktSearchIterator *iter,
    uint64_t *vectors_scanned,
    uint64_t *vectors_filtered
);

/* Cleanup */
void
mkt_search_iterator_destroy(MktSearchIterator *iter);
```

**Usage pattern:**

```c
MktSearchIterator *iter = mkt_search_iterator_create(
    index, query, DISTANCE_L2, nprobe, filter
);

MktSearchResult results[10];
uint32_t n;

/* Fetch first 10 */
n = mkt_search_iterator_next(iter, 10, results);
process_results(results, n);

/* Need more? Fetch next 10 */
if (mkt_search_iterator_has_more(iter)) {
    n = mkt_search_iterator_next(iter, 10, results);
    process_results(results, n);
}

mkt_search_iterator_destroy(iter);
```

### 8.3 Filter Predicate Interface

Filters are evaluated during posting list traversal. The filter interface
abstracts the predicate evaluation.

```c
/*
 * MktFilterContext: Predicate evaluation context
 *
 * Abstracts filter evaluation so the core search doesn't depend on
 * PostgreSQL expression evaluation machinery.
 */
typedef struct MktFilterContext
{
    /* PostgreSQL-specific (set by PG integration layer) */
    void               *pg_state;       /* ExprState, ScanKey, etc. */

    /* Callback to evaluate predicate for a TID */
    bool              (*check_tid)(struct MktFilterContext *ctx,
                                   ItemPointer tid);

    /* Optional: callback for label-based filtering (fast path) */
    bool              (*check_labels)(struct MktFilterContext *ctx,
                                      const int16 *labels,
                                      uint32_t nlabels);

    /* Statistics */
    uint64_t            checked_count;
    uint64_t            passed_count;
} MktFilterContext;

/* Check if a posting entry passes the filter */
static inline bool
mkt_filter_check(MktFilterContext *filter, const MktPostingEntry *entry)
{
    if (filter == NULL) {
        return true;  /* no filter, always pass */
    }

    filter->checked_count++;

    /* Fast path: label-based filtering if available */
    if (filter->check_labels != NULL && entry->has_labels) {
        bool pass = filter->check_labels(filter, entry->labels, entry->nlabels);
        if (pass) filter->passed_count++;
        return pass;
    }

    /* Slow path: TID-based filtering (requires heap fetch) */
    bool pass = filter->check_tid(filter, &entry->tid);
    if (pass) filter->passed_count++;
    return pass;
}
```

### 8.4 Filter Strategies

Different filter selectivities require different strategies:

**1. High selectivity (few matches): Pre-filtering**

When the filter is very selective (e.g., < 1% pass), it's better to:
1. Scan posting lists and apply filter
2. Only compute distances for passing entries
3. May need to scan many clusters to find enough results

```c
/* Adaptive strategy based on filter selectivity */
typedef enum {
    FILTER_STRATEGY_AUTO,       /* decide based on statistics */
    FILTER_STRATEGY_PRE,        /* filter before distance computation */
    FILTER_STRATEGY_POST,       /* compute distance, then filter */
    FILTER_STRATEGY_HYBRID,     /* pre-filter + rerank */
} MktFilterStrategy;

/* Estimate selectivity from table statistics */
float
mkt_estimate_filter_selectivity(
    Relation heap,
    List *predicates,
    PlannerInfo *root
);
```

**2. Low selectivity (many matches): Post-filtering**

When most entries pass the filter, compute distances first, then filter:
1. Find top-K' candidates by distance (K' > K)
2. Apply filter to candidates
3. Return top-K that pass filter

**3. Hybrid approach**

Combine both strategies:
1. Use quantized distances with pre-filtering to get initial candidates
2. Rerank candidates with full-precision distances
3. Apply expensive predicates only to reranked results

### 8.5 Label-Based Filtering

For common filtering patterns (category, tenant, tags), store labels directly
in the posting list entry for fast filtering without heap access.

```c
/*
 * Label storage in posting entry (optional)
 *
 * Labels are small integers (int16) stored inline. This enables
 * filtering without fetching the heap tuple.
 */
typedef struct MktPostingEntryWithLabels
{
    MktPostingEntry     base;
    uint8_t             nlabels;        /* number of labels */
    int16               labels[];       /* inline label array */
} MktPostingEntryWithLabels;

/* Index creation option */
typedef struct MktIndexOptions
{
    /* ... existing options ... */

    /* Label column for fast filtering */
    AttrNumber          label_attr;     /* 0 if not specified */
    bool                labels_indexed; /* true if labels stored in posting */
} MktIndexOptions;
```

**SQL interface:**

```sql
-- Create index with label column for fast filtering
CREATE INDEX ON items USING meerkat (embedding)
    WITH (labels = 'category_id');

-- Query with label filter (uses fast path)
SELECT * FROM items
WHERE category_id = ANY(ARRAY[1, 2, 3])
ORDER BY embedding <-> '[...]'::vector
LIMIT 10;
```

### 8.6 PostgreSQL Integration

The streaming iterator integrates with PostgreSQL's IndexScan via the
`amgettuple` callback.

```c
/*
 * amgettuple: PostgreSQL calls this repeatedly to get results
 *
 * The streaming iterator state is stored in scan->opaque.
 * Each call returns one result, enabling proper interaction with
 * PostgreSQL's executor (LIMIT, cursor, etc.).
 */
static bool
mkt_amgettuple(IndexScanDesc scan, ScanDirection direction)
{
    MktScanState *state = (MktScanState *)scan->opaque;

    /* Lazy initialization on first call */
    if (!state->initialized) {
        mkt_scan_initialize(scan, state);
        state->initialized = true;
    }

    /* Get next result from streaming iterator */
    MktSearchResult result;
    uint32_t n = mkt_search_iterator_next(state->iterator, 1, &result);

    if (n == 0) {
        return false;  /* no more results */
    }

    /* Return TID to PostgreSQL */
    scan->xs_heaptid = result.tid;
    scan->xs_recheckorderby = state->needs_recheck;

    return true;
}

/*
 * amgetbitmap: Alternative for bitmap scans
 *
 * Returns all matching TIDs at once. Used when PostgreSQL wants to
 * combine multiple index scans (BitmapAnd, BitmapOr).
 */
static int64
mkt_amgetbitmap(IndexScanDesc scan, TIDBitmap *tbm)
{
    MktScanState *state = (MktScanState *)scan->opaque;
    int64 count = 0;

    if (!state->initialized) {
        mkt_scan_initialize(scan, state);
        state->initialized = true;
    }

    /* Fetch all results into bitmap */
    MktSearchResult results[100];
    uint32_t n;

    while ((n = mkt_search_iterator_next(state->iterator, 100, results)) > 0) {
        for (uint32_t i = 0; i < n; i++) {
            tbm_add_tuples(tbm, &results[i].tid, 1, false);
            count++;
        }
    }

    return count;
}
```

### 8.7 Adaptive Termination

For filtered search, we don't know upfront how many candidates we need to
scan. Adaptive termination uses statistics to decide when to stop.

```c
typedef struct MktAdaptiveState
{
    /* Running statistics */
    uint32_t    scanned;        /* entries scanned */
    uint32_t    passed;         /* entries passing filter */
    uint32_t    returned;       /* results returned */

    /* Estimated selectivity (updated as we scan) */
    float       selectivity;    /* passed / scanned */

    /* Termination criteria */
    uint32_t    target_k;       /* how many results needed */
    float       confidence;     /* confidence we have enough */
} MktAdaptiveState;

/*
 * Should we continue scanning more clusters?
 *
 * Uses statistics to estimate probability of finding better results.
 */
bool
mkt_should_continue_search(
    const MktAdaptiveState *state,
    Distance worst_result_dist,
    Distance next_cluster_dist
)
{
    /* If we don't have enough results, must continue */
    if (state->returned < state->target_k) {
        return true;
    }

    /* Estimate expected matches in next cluster */
    float expected_matches = CLUSTER_SIZE * state->selectivity;

    /* Probability that next cluster has a better result */
    /* (simplified: based on distance ratio) */
    float prob_better = 0.0f;
    if (next_cluster_dist < worst_result_dist) {
        prob_better = 1.0f - (next_cluster_dist / worst_result_dist);
        prob_better *= expected_matches / state->target_k;
    }

    return prob_better > (1.0f - state->confidence);
}
```

### 8.8 Architectural Implications

The streaming/filtered search design affects earlier components:

**1. Posting list iteration**

The posting list must support efficient iteration with filtering:
```c
/* Posting list must support filtered iteration */
typedef struct MktPostingIterator
{
    /* ... */
    MktFilterContext *filter;   /* optional filter */
    /* ... */
} MktPostingIterator;
```

**2. Result heap**

Need a min-heap that supports incremental insertion and extraction:
```c
/* Result heap for streaming results */
MktResultHeap *mkt_result_heap_create(uint32_t capacity);
void mkt_result_heap_push(MktResultHeap *heap, MktSearchResult *result);
bool mkt_result_heap_pop(MktResultHeap *heap, MktSearchResult *result);
```

**3. Centroid search**

Centroid search should return an iterator, not a fixed array:
```c
/* Return clusters lazily, not all at once */
MktClusterIterator *
mkt_centroid_search_iterator(CentroidCache *cache, VectorRef query);
```

**4. Statistics collection**

Track filter statistics for cost estimation:
```c
/* Per-index statistics for cost estimation */
typedef struct MktIndexStats
{
    /* ... */
    float   avg_filter_selectivity; /* historical average */
    float   avg_recheck_cost;       /* cost of heap fetch */
    /* ... */
} MktIndexStats;
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
