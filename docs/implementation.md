# Meerkat Implementation Guide

Detailed implementation specification for Meerkat, organized for iterative
development. Each section builds on previous work, with standalone components
developed and tested before PostgreSQL integration.

## Technical Overview

Meerkat is a PostgreSQL index access method for approximate nearest neighbor
(ANN) vector search. The implementation combines proven techniques from recent
research with practical considerations for database integration.

### Key Technical Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Index structure** | ScaNN/SPANN-style partitioned index | Inverted file with posting lists enables disk-friendly access patterns and PostgreSQL buffer cache integration |
| **Quantization** | RaBitQ (1-bit binary) | State-of-the-art 32× compression with theoretical error bounds; enables two-stage search with guaranteed recall |
| **Recall improvement** | Boundary-only replication (SPANN) | Vectors near cluster boundaries are replicated to adjacent clusters; improves recall without excessive storage overhead |
| **Dynamic updates** | LIRE-protocol style | Supports high insert rates without degrading query performance; periodic background reorganization |
| **Multi-tenancy** | Composite key support | `(tenant_id, vector)` keys enable efficient per-tenant queries with shared index infrastructure |
| **Async I/O** | PostgreSQL 18 read stream API | Adaptive prefetching for posting list scans, heap reranking; supports `io_uring` for lowest latency |

### Architecture Summary

```
Query Flow:
  1. Traverse centroid tree to find top-k clusters (buffer cache)
  2. Scan posting lists for candidate clusters (disk/buffer cache)
  3. RaBitQ two-stage filtering: 1-bit estimate → error bound check → rerank
  4. Full-precision reranking of top candidates (heap access)
  5. Return k nearest neighbors

Index Structure:
  ┌─────────────────────────────────────────────────────────────┐
  │ Centroid Pages (dedicated pages in shared buffers)          │
  │   - Hierarchical tree: root → intermediate → leaf levels    │
  │   - Quantized centroids for fast cluster selection          │
  │   - Hot pages stay cached; no separate in-memory cache      │
  └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ Posting Lists (PostgreSQL pages)                            │
  │   - One list per leaf centroid/cluster                      │
  │   - RaBitQ-encoded vectors (TID, bits, f_error)             │
  │   - Boundary vectors replicated across adjacent clusters    │
  └─────────────────────────────────────────────────────────────┘
```

### References

The design draws from:

- **ScaNN/SPANN**: Partitioned index structure, posting list organization
- **RaBitQ**: Binary quantization with error bounds (SIGMOD 2024)
- **LIRE**: Dynamic update protocol for IVF indexes
- **SPFresh**: Fresh vector search with streaming updates

See [architecture.md](architecture.md) for high-level design and CLAUDE.md for
full reference list.

---

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

4. **Async I/O ready**: Leverage PostgreSQL 18+ read stream API for efficient
   prefetching. Support all `io_method` options including `io_uring`.

4. **Planner integration**: Cost estimates enable the query planner to choose
   between index scan and sequential scan appropriately.

---

## Component Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PostgreSQL Integration                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────────┐  │
│  │ IAM Handler │  │ Page Layout │  │ Buffer Mgmt │  │ Cost Estimate │  │
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

## Command-Line Interface

Meerkat provides a single unified CLI binary (`mkt`) with subcommands for
development, testing, and benchmarking. This replaces multiple standalone tools.

### Usage

```
mkt <command> [options]

Commands:
  distance, d     Compute distance between vectors
  quantize, q     Quantize vectors using RaBitQ
  cluster,  c     Run clustering algorithms
  build,    b     Build an index from vectors
  search,   s     Search an index
  bench           Run benchmarks (distance, quantize, cluster, search)
  info            Show index metadata
  version         Show version information

Run 'mkt <command> --help' for command-specific options.
```

### Subcommand Examples

**Distance computation:**

```
$ mkt distance --metric l2 --file vectors.fvecs --i 0 --j 1
Distance(L2): 42.315

$ mkt d -m ip -a "[1.0, 2.0, 3.0]" -b "[4.0, 5.0, 6.0]"
Distance(IP): 32.000
```

**Quantization:**

```
$ mkt quantize --input vectors.fvecs --output quantized.rq --dim 768
Quantized 100000 vectors (768-dim) using RaBitQ
  Input:  292.97 MB (3072 bytes/vec)
  Output:  11.04 MB (116 bytes/vec)
  Ratio:  26.5x compression

$ mkt q --decode quantized.rq --index 42
Vector 42 (reconstructed): [0.123, -0.456, ...]
```

**Benchmarks:**

```bash
mkt bench distance --dim 768 --count 10000 --metric l2
mkt bench quantize --dataset sift1m --k 10
mkt bench cluster --dim 128 --nvecs 100000 --nlist 1000
mkt bench search --index index.mkt --queries queries.fvecs --k 10
```

**Index operations:**

```bash
mkt build --input vectors.fvecs --dim 768 --nlist 1000 --output index.mkt
mkt search --index index.mkt --query query.fvecs --k 10 --nprobe 20
mkt info --index index.mkt
```

### Implementation

**Files**: `src/cli/main.c`, `src/cli/cmd_*.c`

```c
// src/cli/main.c - Entry point with subcommand dispatch

typedef struct {
    const char *name;
    const char *alias;
    const char *description;
    int (*handler)(int argc, char **argv);
} MktCommand;

static const MktCommand commands[] = {
    {"distance", "d", "Compute distance between vectors", cmd_distance},
    {"quantize", "q", "Quantize vectors using RaBitQ",    cmd_quantize},
    {"cluster",  "c", "Run clustering algorithms",        cmd_cluster},
    {"build",    "b", "Build an index from vectors",      cmd_build},
    {"search",   "s", "Search an index",                  cmd_search},
    {"bench",    NULL, "Run benchmarks",                  cmd_bench},
    {"info",     NULL, "Show index metadata",             cmd_info},
    {"version",  NULL, "Show version information",        cmd_version},
    {NULL, NULL, NULL, NULL}
};

int main(int argc, char **argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    const char *cmd_name = argv[1];
    for (const MktCommand *cmd = commands; cmd->name; cmd++) {
        if (strcmp(cmd_name, cmd->name) == 0 ||
            (cmd->alias && strcmp(cmd_name, cmd->alias) == 0)) {
            return cmd->handler(argc - 1, argv + 1);
        }
    }

    fprintf(stderr, "Unknown command: %s\n", cmd_name);
    return 1;
}
```

Each subcommand is implemented in a separate file (`mkt_cmd_distance.c`,
`mkt_cmd_quantize.c`, etc.) for maintainability. The `bench` subcommand has
its own sub-subcommands for different benchmark types.

### Meson Build

```meson
# src/cli/meson.build
cli_sources = files(
  'main.c',
  'cmd_distance.c',
  'cmd_quantize.c',
  'cmd_cluster.c',
  'cmd_build.c',
  'cmd_search.c',
  'cmd_bench.c',
  'cmd_info.c',
)
```

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
/* test_vec32.c - Vector type and operations tests */

#include "mkt_test.h"
#include "vec32.h"

TEST_GROUP(Vector)

TEST(vector_create_and_dim)
{
    Vec32 *v = vec32_create(128);
    ASSERT_NOT_NULL(v, "vector should be allocated");
    ASSERT_EQ(128, VEC32_DIM(v), "dimension should be 128");
    vec32_free(v);
}

TEST(vector_set_get_elements)
{
    Vec32 *v = vec32_create(3);
    VEC32_DATA(v)[0] = 1.0f;
    VEC32_DATA(v)[1] = 2.0f;
    VEC32_DATA(v)[2] = 3.0f;

    ASSERT_FLOAT_EQ(1.0f, VEC32_DATA(v)[0], 1e-6, "element 0");
    ASSERT_FLOAT_EQ(2.0f, VEC32_DATA(v)[1], 1e-6, "element 1");
    ASSERT_FLOAT_EQ(3.0f, VEC32_DATA(v)[2], 1e-6, "element 2");

    vec32_free(v);
}

TEST(vector_l2_distance)
{
    Vec32 *a = vec32_create(3);
    Vec32 *b = vec32_create(3);

    VEC32_DATA(a)[0] = 0.0f;
    VEC32_DATA(a)[1] = 0.0f;
    VEC32_DATA(a)[2] = 0.0f;

    VEC32_DATA(b)[0] = 1.0f;
    VEC32_DATA(b)[1] = 0.0f;
    VEC32_DATA(b)[2] = 0.0f;

    Vec32Ref va = Vec32ToRef(a);
    Vec32Ref vb = Vec32ToRef(b);

    Distance d = mkt_distance_l2(va, vb);
    ASSERT_FLOAT_EQ(1.0f, d, 1e-6, "L2 squared distance should be 1.0");

    vec32_free(a);
    vec32_free(b);
}
```

### Build Configuration (Meson)

```meson
# test/unit/meson.build

unit_test_sources = files(
  'mkt_test.c',
  'run_tests.c',
  'test_vec32.c',
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

**Files**: `src/types/vec32.h`, `src/types/vec32.c`

The vec32 type is binary-compatible with pgvector's `vector` type in both
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
/* vec32.h - Vector type compatible with pgvector */

#ifndef VEC32_H
#define VEC32_H

#include <stdint.h>
#include <stdlib.h>
#include "mkt_types.h"

#define VEC32_MAX_DIM 16000

/*
 * Vec32: Binary-compatible with pgvector's Vector type.
 *
 * The struct layout is identical in both standalone and PostgreSQL modes.
 * In standalone mode, vl_len_ stores the total size (not used as varlena).
 * In PostgreSQL mode, vl_len_ is managed by SET_VARSIZE/VARSIZE macros.
 */
typedef struct Vec32
{
    int32       vl_len_;        /* varlena header / size in standalone mode */
    int16       dim;            /* number of dimensions */
    int16       unused;         /* reserved for future use, always zero */
    float       x[];            /* FLEXIBLE_ARRAY_MEMBER */
}           Vec32;

#define VEC32_SIZE(_dim) (offsetof(Vec32, x) + sizeof(float)*(_dim))
#define VEC32_DIM(v)     ((v)->dim)
#define VEC32_DATA(v)    ((v)->x)

/* Convert to Vec32Ref for internal operations */
static inline Vec32Ref
Vec32ToRef(const Vec32 *v)
{
    return (Vec32Ref){ .data = v->x, .dim = v->dim };
}

/* Allocation and lifecycle */
Vec32  *vec32_create(Dimension dim);
Vec32  *vec32_copy(const Vec32 *src);
void        vec32_free(Vec32 *v);

/* Initialization */
void        vec32_set(Vec32 *v, const float *values);
void        vec32_zero(Vec32 *v);
void        vec32_fill(Vec32 *v, float value);

/* Operations */
float       vec32_dot(const Vec32 *a, const Vec32 *b);
float       vec32_norm(const Vec32 *v);
void        vec32_normalize(Vec32 *v);

#endif /* VEC32_H */
```

### Implementation

```c
/* vec32.c - Vector operations */

#include "vec32.h"
#include "mkt_memory.h"
#include <string.h>
#include <math.h>

Vec32 *
vec32_create(Dimension dim)
{
    if (dim > VEC32_MAX_DIM)
        return NULL;

    size_t size = VEC32_SIZE(dim);
    Vec32 *v = mkt_alloc(size);
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
vec32_free(Vec32 *v)
{
    mkt_free(v);
}

Vec32 *
vec32_copy(const Vec32 *src)
{
    Vec32 *dst = vec32_create(src->dim);
    if (dst == NULL)
        return NULL;

    memcpy(dst->x, src->x, src->dim * sizeof(float));
    return dst;
}

void
vec32_zero(Vec32 *v)
{
    memset(v->x, 0, v->dim * sizeof(float));
}

float
vec32_dot(const Vec32 *a, const Vec32 *b)
{
    float sum = 0.0f;
    for (Dimension i = 0; i < a->dim; i++)
        sum += a->x[i] * b->x[i];
    return sum;
}

float
vec32_norm(const Vec32 *v)
{
    return sqrtf(vec32_dot(v, v));
}

void
vec32_normalize(Vec32 *v)
{
    float norm = vec32_norm(v);
    if (norm > 0.0f) {
        for (Dimension i = 0; i < v->dim; i++)
            v->x[i] /= norm;
    }
}
```

The key design points:

1. **Identical binary layout**: The struct has the same fields in the same order
   regardless of build mode. A pointer to `Vec32` and pgvector's `Vector`
   can be safely cast between each other.

2. **vl_len_ handling**: In standalone mode, `vl_len_` stores the struct size
   directly. In PostgreSQL mode, it's managed by `SET_VARSIZE()`. The field
   exists in both modes to maintain binary compatibility.

3. **Core algorithms use Vec32Ref**: Distance computation, quantization, and
   other algorithms operate on `Vec32Ref` (pointer + dimension), not the full
   struct. This decouples algorithms from storage format.

### 1.2 Type Definitions

**File**: `src/core/types.h`

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
} Vec32Ref;

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

**Files**:

- `src/core/memory.h` - Common interface, includes mode-specific header
- `src/core/memory_standalone.h` - Standalone type definitions and declarations
- `src/core/memory_standalone.c` - Standalone arena implementation
- `src/core/memory_pg.h` - PostgreSQL macros and inline wrappers (header-only)

Abstraction layer allowing the same code to run with PostgreSQL's memory
contexts or a standalone arena allocator. The arena allocator provides
efficient bump-pointer allocation with bulk deallocation, matching the
semantics of PostgreSQL memory contexts.

**Build configuration**:

- Standalone: define `MKT_STANDALONE`, compile `mkt_memory_standalone.c`
- PostgreSQL: no define needed, header-only (no `.c` file to compile)

#### Common Interface

```c
/* mkt_memory.h - Memory abstraction interface */

#ifndef MKT_MEMORY_H
#define MKT_MEMORY_H

#ifdef MKT_STANDALONE
#include "mkt_memory_standalone.h"
#else
#include "mkt_memory_pg.h"
#endif

/* Helper for cleanup attribute (works in both modes) */
static inline void
mkt_memctx_delete_ptr(MktMemCtx *ctx)
{
    if (*ctx)
        mkt_memctx_delete(*ctx);
}

/* Scoped context (RAII-style via cleanup attribute) */
#ifdef MKT_STANDALONE
#define MKT_MEMCTX_SCOPE(name) \
    MktMemCtx name __attribute__((cleanup(mkt_memctx_delete_ptr))) = \
        mkt_memctx_create(mkt_current_memctx, #name)
#else
#define MKT_MEMCTX_SCOPE(name) \
    MktMemCtx name __attribute__((cleanup(mkt_memctx_delete_ptr))) = \
        mkt_memctx_create(CurrentMemoryContext, #name)
#endif

#endif /* MKT_MEMORY_H */
```

#### Standalone Header

```c
/* mkt_memory_standalone.h - Standalone arena allocator types and declarations */

#ifndef MKT_MEMORY_STANDALONE_H
#define MKT_MEMORY_STANDALONE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

/*
 * Arena allocator for standalone builds.
 *
 * An arena allocates memory from large blocks using bump-pointer allocation.
 * Individual allocations cannot be freed; memory is released all at once
 * when the arena is reset or destroyed. This matches PostgreSQL memory
 * context semantics and provides excellent cache locality.
 */

#define MKT_ARENA_BLOCK_SIZE (64 * 1024)  /* 64 KB default block size */
#define MKT_ARENA_ALIGNMENT  16           /* Default alignment */

/* Block header for arena memory blocks */
typedef struct MktArenaBlock
{
    struct MktArenaBlock *next;   /* Next block in chain */
    size_t                size;   /* Total size of this block */
    size_t                used;   /* Bytes used in this block */
    /* Data follows immediately after header, aligned */
} MktArenaBlock;

/* Arena memory context */
typedef struct MktArena
{
    const char     *name;         /* Context name for debugging */
    struct MktArena *parent;      /* Parent context (for hierarchy) */
    struct MktArena *first_child; /* First child context */
    struct MktArena *next_sibling;/* Next sibling in parent's child list */
    MktArenaBlock  *current;      /* Current block for allocations */
    MktArenaBlock  *blocks;       /* All blocks (for freeing) */
    size_t          block_size;   /* Size for new blocks */
    size_t          total_allocated; /* Stats: total bytes allocated */
} MktArena;

typedef MktArena *MktMemCtx;

/* Current memory context (thread-local) */
extern MktMemCtx mkt_current_memctx;

/* Allocation functions */
void *mkt_alloc(size_t size);
void *mkt_alloc0(size_t size);
void *mkt_realloc(void *ptr, size_t size);
void  mkt_free(void *ptr);
void *mkt_memctx_alloc(MktMemCtx ctx, size_t size);
void *mkt_memctx_alloc0(MktMemCtx ctx, size_t size);
void *mkt_alloc_aligned(size_t size, size_t alignment);
void  mkt_free_aligned(void *ptr);

/* Context management */
MktMemCtx mkt_memctx_create(MktMemCtx parent, const char *name);
void      mkt_memctx_delete(MktMemCtx ctx);
void      mkt_memctx_reset(MktMemCtx ctx);
MktMemCtx mkt_memctx_switch(MktMemCtx ctx);
size_t    mkt_memctx_total_allocated(MktMemCtx ctx);

#endif /* MKT_MEMORY_STANDALONE_H */
```

#### PostgreSQL Header

```c
/* mkt_memory_pg.h - PostgreSQL memory wrappers (header-only) */
#ifndef MKT_MEMORY_PG_H
#define MKT_MEMORY_PG_H

#include "postgres.h"
#include "utils/memutils.h"

typedef MemoryContext MktMemCtx;

/* Direct mappings to palloc family */
#define mkt_alloc(size)              palloc(size)
#define mkt_alloc0(size)             palloc0(size)
#define mkt_realloc(ptr, size)       repalloc(ptr, size)
#define mkt_free(ptr)                pfree(ptr)
#define mkt_memctx_alloc(ctx, size)  MemoryContextAlloc(ctx, size)
#define mkt_memctx_alloc0(ctx, size) MemoryContextAllocZero(ctx, size)
#define mkt_alloc_aligned(sz, al)    palloc_aligned(sz, al, 0)
#define mkt_free_aligned(ptr)        pfree(ptr)

/* Context management */
#define mkt_memctx_create(parent, name) \
    AllocSetContextCreate((parent) ? (parent) : CurrentMemoryContext, \
                          (name), ALLOCSET_DEFAULT_SIZES)
#define mkt_memctx_delete(ctx)       MemoryContextDelete(ctx)
#define mkt_memctx_reset(ctx)        MemoryContextReset(ctx)
#define mkt_memctx_switch(ctx)       MemoryContextSwitchTo(ctx)
#define mkt_memctx_total_allocated(ctx) ((size_t)0)

#endif /* MKT_MEMORY_PG_H */
```

Note: `palloc_aligned()` requires PostgreSQL 16+. For older versions, add a
compatibility wrapper.

#### Standalone Implementation

```c
/* mkt_memory_standalone.c - Arena allocator implementation */

#include "mkt_memory_standalone.h"

#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* Thread-local current context (or global if no thread support) */
#ifdef _Thread_local
_Thread_local MktMemCtx mkt_current_memctx = NULL;
#else
MktMemCtx mkt_current_memctx = NULL;
#endif

/* Align size up to alignment boundary */
static inline size_t
align_up(size_t size, size_t alignment)
{
    return (size + alignment - 1) & ~(alignment - 1);
}

/* Allocate a new block */
static MktArenaBlock *
arena_block_create(size_t min_size)
{
    size_t block_size = min_size > MKT_ARENA_BLOCK_SIZE
                        ? min_size : MKT_ARENA_BLOCK_SIZE;
    /* Include space for header, aligned */
    size_t header_size = align_up(sizeof(MktArenaBlock), MKT_ARENA_ALIGNMENT);
    size_t total_size = header_size + block_size;

    MktArenaBlock *block = aligned_alloc(MKT_ARENA_ALIGNMENT, total_size);
    if (!block)
        return NULL;

    block->next = NULL;
    block->size = block_size;
    block->used = 0;
    return block;
}

/* Free a block */
static void
arena_block_free(MktArenaBlock *block)
{
    free(block);
}

/* Get data pointer for block */
static inline void *
arena_block_data(MktArenaBlock *block)
{
    size_t header_size = align_up(sizeof(MktArenaBlock), MKT_ARENA_ALIGNMENT);
    return (char *)block + header_size;
}

/* Allocate from arena */
static void *
arena_alloc(MktArena *arena, size_t size, size_t alignment)
{
    if (size == 0)
        return NULL;

    size_t aligned_size = align_up(size, alignment);

    /* Try current block first */
    if (arena->current)
    {
        void *data = arena_block_data(arena->current);
        size_t offset = align_up(arena->current->used, alignment);

        if (offset + aligned_size <= arena->current->size)
        {
            void *ptr = (char *)data + offset;
            arena->current->used = offset + aligned_size;
            arena->total_allocated += aligned_size;
            return ptr;
        }
    }

    /* Need a new block */
    MktArenaBlock *block = arena_block_create(aligned_size);
    if (!block)
        return NULL;

    /* Link new block */
    block->next = arena->blocks;
    arena->blocks = block;
    arena->current = block;

    /* Allocate from new block */
    void *data = arena_block_data(block);
    block->used = aligned_size;
    arena->total_allocated += aligned_size;
    return data;
}

/* Create a new arena context */
MktMemCtx
mkt_memctx_create(MktMemCtx parent, const char *name)
{
    /* Allocate arena struct from parent or malloc */
    MktArena *arena;
    if (parent)
        arena = arena_alloc(parent, sizeof(MktArena), MKT_ARENA_ALIGNMENT);
    else
        arena = aligned_alloc(MKT_ARENA_ALIGNMENT, sizeof(MktArena));

    if (!arena)
        return NULL;

    memset(arena, 0, sizeof(MktArena));
    arena->name = name;
    arena->parent = parent;
    arena->block_size = MKT_ARENA_BLOCK_SIZE;

    /* Link into parent's child list */
    if (parent)
    {
        arena->next_sibling = parent->first_child;
        parent->first_child = arena;
    }

    return arena;
}

/* Delete arena and all children */
void
mkt_memctx_delete(MktMemCtx ctx)
{
    if (!ctx)
        return;

    MktArena *arena = ctx;

    /* Recursively delete children first */
    MktArena *child = arena->first_child;
    while (child)
    {
        MktArena *next = child->next_sibling;
        mkt_memctx_delete(child);
        child = next;
    }

    /* Unlink from parent */
    if (arena->parent)
    {
        MktArena **pp = &arena->parent->first_child;
        while (*pp && *pp != arena)
            pp = &(*pp)->next_sibling;
        if (*pp)
            *pp = arena->next_sibling;
    }

    /* Free all blocks */
    MktArenaBlock *block = arena->blocks;
    while (block)
    {
        MktArenaBlock *next = block->next;
        arena_block_free(block);
        block = next;
    }

    /* Free arena struct if it was top-level (no parent) */
    if (!arena->parent)
        free(arena);
}

/* Reset arena - free all allocations but keep arena */
void
mkt_memctx_reset(MktMemCtx ctx)
{
    if (!ctx)
        return;

    MktArena *arena = ctx;

    /* Recursively delete children */
    MktArena *child = arena->first_child;
    while (child)
    {
        MktArena *next = child->next_sibling;
        mkt_memctx_delete(child);
        child = next;
    }
    arena->first_child = NULL;

    /* Free all blocks except first (if any) */
    if (arena->blocks)
    {
        MktArenaBlock *keep = arena->blocks;
        MktArenaBlock *block = keep->next;
        while (block)
        {
            MktArenaBlock *next = block->next;
            arena_block_free(block);
            block = next;
        }
        keep->next = NULL;
        keep->used = 0;
        arena->blocks = keep;
        arena->current = keep;
    }
    else
    {
        arena->current = NULL;
    }
    arena->total_allocated = 0;
}

/* Allocate from current context */
void *
mkt_alloc(size_t size)
{
    assert(mkt_current_memctx != NULL);
    return arena_alloc(mkt_current_memctx, size, MKT_ARENA_ALIGNMENT);
}

void *
mkt_alloc0(size_t size)
{
    void *ptr = mkt_alloc(size);
    if (ptr)
        memset(ptr, 0, size);
    return ptr;
}

void *
mkt_memctx_alloc(MktMemCtx ctx, size_t size)
{
    return arena_alloc(ctx, size, MKT_ARENA_ALIGNMENT);
}

void *
mkt_memctx_alloc0(MktMemCtx ctx, size_t size)
{
    void *ptr = mkt_memctx_alloc(ctx, size);
    if (ptr)
        memset(ptr, 0, size);
    return ptr;
}

/*
 * Realloc is limited in arena mode - we can't reclaim the old space.
 * This just allocates new space and copies. Use sparingly.
 */
void *
mkt_realloc(void *ptr, size_t size)
{
    if (!ptr)
        return mkt_alloc(size);
    if (size == 0)
        return NULL;

    /* We don't know the old size, so caller must handle copying */
    /* This is a limitation of arena allocators */
    return mkt_alloc(size);
}

/* Free is a no-op in arena mode */
void
mkt_free(void *ptr)
{
    (void)ptr;  /* Intentionally empty - memory freed on context reset/delete */
}

/* Aligned allocation */
void *
mkt_alloc_aligned(size_t size, size_t alignment)
{
    assert(mkt_current_memctx != NULL);
    return arena_alloc(mkt_current_memctx, size, alignment);
}

void
mkt_free_aligned(void *ptr)
{
    (void)ptr;  /* No-op in arena mode */
}

/* Switch context */
MktMemCtx
mkt_memctx_switch(MktMemCtx ctx)
{
    MktMemCtx old = mkt_current_memctx;
    mkt_current_memctx = ctx;
    return old;
}

/* Stats */
size_t
mkt_memctx_total_allocated(MktMemCtx ctx)
{
    return ctx ? ctx->total_allocated : 0;
}
```

#### Function Mapping

The following table shows how Meerkat memory functions map to PostgreSQL:

| Meerkat Function       | PostgreSQL Equivalent                   |
|------------------------|----------------------------------------|
| `mkt_alloc()`          | `palloc()`                             |
| `mkt_alloc0()`         | `palloc0()`                            |
| `mkt_realloc()`        | `repalloc()`                           |
| `mkt_free()`           | `pfree()`                              |
| `mkt_memctx_alloc()`   | `MemoryContextAlloc()`                 |
| `mkt_memctx_create()`  | `AllocSetContextCreate()`              |
| `mkt_memctx_delete()`  | `MemoryContextDelete()`                |
| `mkt_memctx_reset()`   | `MemoryContextReset()`                 |
| `mkt_memctx_switch()`  | `MemoryContextSwitchTo()` (inline)     |

For aligned allocation, PostgreSQL 16+ provides `palloc_aligned()`. On older
versions, a manual alignment wrapper is used.

#### Usage Example

```c
/* Example: using arena for temporary computation */
void
compute_distances(const Vec32 *query, const Vec32 *vectors, int n)
{
    /* Create scoped arena for temporary allocations */
    MKT_MEMCTX_SCOPE(work_ctx);
    MktMemCtx old = mkt_memctx_switch(work_ctx);

    /* Allocate temporary buffer - freed automatically when scope exits */
    float *distances = mkt_alloc(n * sizeof(float));

    /* ... compute distances ... */

    /* Restore previous context */
    mkt_memctx_switch(old);
    /* work_ctx automatically deleted via cleanup attribute */
}

/* Example: persistent arena for index build */
MktMemCtx build_ctx = mkt_memctx_create(NULL, "IndexBuild");
MktMemCtx old = mkt_memctx_switch(build_ctx);

/* All allocations go to build_ctx */
Vec32 *centroids = mkt_alloc(k * sizeof(Vec32));

/* Reset to reuse memory for next phase */
mkt_memctx_reset(build_ctx);

/* Clean up when done */
mkt_memctx_switch(old);
mkt_memctx_delete(build_ctx);
```

#### Design Notes

**Why arena allocation?**

1. **Matches PostgreSQL semantics**: PostgreSQL memory contexts use similar
   bulk-deallocation semantics. Code that works with arenas works unchanged
   with PostgreSQL.

2. **Cache efficiency**: Arena allocations are contiguous, improving cache
   locality for sequential access patterns common in vector operations.

3. **Zero fragmentation**: No free-list management, no fragmentation. Memory
   is compacted after reset.

4. **Fast allocation**: Bump-pointer allocation is O(1) with minimal overhead
   (just pointer arithmetic).

5. **Simplified error handling**: No need to track and free individual
   allocations on error paths.

**Limitations**:

- `mkt_free()` is a no-op; memory is only released via `mkt_memctx_reset()`
  or `mkt_memctx_delete()`
- `mkt_realloc()` cannot reclaim old space; avoid in hot paths
- Not suitable for long-lived allocations with varying lifetimes

**Tests**: Unit tests verify:

- Basic allocation and alignment
- Context hierarchy (parent/child relationships)
- Reset releases memory but keeps arena
- Delete frees all memory including children
- Large allocations (> block size) work correctly
- Stats tracking accuracy

### 1.4 Platform Abstraction

**File**: `src/core/platform.h`

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

**Files**: `src/algo/distance.h`, `src/algo/distance.c`, `src/algo/distance_avx512.c`,
`src/algo/distance_avx2.c`, `src/algo/distance_neon.c`

Distance computation is the most performance-critical operation. Multiple
implementations selected at runtime based on CPU capabilities.

#### Interface

```c
// Single vector pair distance
Distance mkt_distance_l2(Vec32Ref a, Vec32Ref b);
Distance mkt_distance_ip(Vec32Ref a, Vec32Ref b);  // Inner product
Distance mkt_distance_cosine(Vec32Ref a, Vec32Ref b);

// Generic dispatch
Distance mkt_distance(Vec32Ref a, Vec32Ref b, DistanceMetric metric);

// Batch: distances from one query to multiple vectors
// Results written to `distances` array (must be pre-allocated)
void mkt_distance_batch(
    Vec32Ref query,
    const float *vectors,  // Contiguous array of vectors
    uint32_t count,
    Dimension dim,
    DistanceMetric metric,
    Distance *distances       // Output: count distances
);

// Batch with early termination: stop when found k vectors below threshold
// Returns number of vectors actually processed
uint32_t mkt_distance_batch_threshold(
    Vec32Ref query,
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
Distance mkt_distance_l2_avx512(Vec32Ref a, Vec32Ref b) {
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
    Vec32Ref query,
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

        Vec32Ref v = { .data = vectors + i * dim, .dim = dim };
        distances[i] = mkt_distance_l2_avx512(query, v);
    }
}
```

#### Function Dispatch

At initialization, select optimal implementation based on CPU:

```c
typedef Distance (*DistanceFn)(Vec32Ref, Vec32Ref);
typedef void (*DistanceBatchFn)(Vec32Ref, const float*, uint32_t,
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

**CLI**: `mkt bench distance` - benchmark distance computations

```
$ mkt bench distance --dim 768 --count 10000 --metric l2
L2 distance (dim=768, count=10000):
  scalar:  45.2 ms (221k vec/s)
  avx2:    8.1 ms (1.23M vec/s)
  avx512:  4.3 ms (2.33M vec/s)
```

### 2.2 Quantization

**Files**: `src/quant/rabitq.h`, `src/quant/rabitq.c`

Quantization compresses vectors for faster approximate distance computation.
**Meerkat uses RaBitQ as its quantization method.** RaBitQ (Randomized Binary
Quantization) compresses vectors to 1 bit per dimension (32x compression) while
maintaining 95-99% recall through theoretical error bounds.

**Why RaBitQ:**

1. **Theoretical guarantees**: Error bound O(1/√D) that improves with dimension.
   PQ lacks such bounds and can fail on some datasets (e.g., 50%+ error on MSong).
2. **High compression**: 32x (1 bit/dimension) enables large in-memory indexes.
3. **Speed**: Bitwise operations are faster than PQ codebook lookups at same
   accuracy.
4. **Architecture fit**: Meerkat re-ranks with full precision vectors, so
   aggressive initial compression is acceptable.
5. **Industry adoption**: Used by turbopuffer, Elasticsearch, LanceDB.

**Future (lower priority):** SQ8 (scalar 8-bit) may be added as a fallback for
lower-dimensional vectors where RaBitQ's benefits are smaller, or for use cases
requiring higher recall without reranking.

#### References

**Papers:**

- [RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound
  for Approximate Nearest Neighbor Search](https://arxiv.org/abs/2405.12497)
  (SIGMOD 2024)

**Reference implementations:**

- [RaBitQ-Library](https://github.com/VectorDB-NTU/RaBitQ-Library) — Official
  library from paper authors (C++). Local checkout: `../RaBitQ-Library/`
- [RaBitQ](https://github.com/gaoj0017/RaBitQ) — Original research code

**Production implementations to study:**

| Project | Index + RaBitQ | Language | Local |
|---------|----------------|----------|-------|
| [Milvus](https://github.com/milvus-io/milvus) | IVF + RaBitQ | C++ | — |
| [Faiss](https://github.com/facebookresearch/faiss) | IVF + RaBitQ | C++ | `../faiss/` |
| [VSAG](https://github.com/antgroup/vsag) | HGraph + RaBitQ | C++ | — |
| [VectorChord](https://github.com/tensorchord/VectorChord) | IVF + RaBitQ | Rust | — |
| [CockroachDB](https://github.com/cockroachdb/cockroach) | C-SPANN + RaBitQ | Go | — |
| [Elasticsearch](https://github.com/elastic/elasticsearch) | HNSW + RaBitQ | Java | — |
| [Lucene](https://github.com/apache/lucene) | HNSW + RaBitQ | Java | — |

Note: Elasticsearch/Lucene call their implementation "BBQ" (Better Binary
Quantization).

**Blog posts:**

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
- **⟨ō, o⟩**: Inner product between quantized and original normalized vector
  (4 bytes)
- **‖o - c‖**: Vector norm relative to cluster centroid (4 bytes)

```c
/*
 * RaBitQVector: PostgreSQL varlena-compatible quantized vector (future SQL type)
 *
 * Reserved for the future PostgreSQL SQL type. Encoding functions produce
 * RaBitQData (compact form) instead. The data portion (f_add, f_rescale,
 * bits[]) is layout-compatible with RaBitQData for zero-copy access via
 * MKT_RABITQ_DATA().
 */
typedef struct RaBitQVector
{
    int32_t vl_len_;   /* varlena header (for PG compatibility) */
    int16_t dim;       /* dimensions = number of bits */
    int16_t flags;     /* reserved for future use */
    float   f_add;     /* additive factor for distance estimation */
    float   f_rescale; /* scaling factor for distance estimation */
    uint8_t bits[];    /* D/8 bytes, LSB-first bit packing */
} RaBitQVector;

/*
 * RaBitQData: Compact quantized vector (primary encoding form)
 *
 * Stores f_add and f_rescale factors plus D/8 binary code bytes.
 * f_error is derived at query time from f_add and f_rescale.
 * This is the form produced by all encoding functions and consumed
 * by distance functions.
 *
 * Total size: 8 bytes header + ceil(dim/8) bytes data
 */
typedef struct RaBitQData
{
    float   f_add;     /* ||v-c||² - additive distance factor */
    float   f_rescale; /* dp_multiplier - scaling factor */
    uint8_t bits[];    /* D/8 bytes, LSB-first bit packing */
} RaBitQData;

#define MKT_RABITQ_BYTES(dim)      (((dim) + 7) / 8)
#define MKT_RABITQ_DATA_SIZE(dim)  (offsetof(RaBitQData, bits) + \
                                    MKT_RABITQ_BYTES(dim))

/* Access compact data portion of a presentation vector (zero-copy cast) */
#define MKT_RABITQ_DATA(v)         ((RaBitQData *)&(v)->f_add)

/*
 * RaBitQBatch: Batch of encoded vectors in separate arrays
 *
 * Used for batch encoding where separate arrays enable efficient
 * iteration when writing multiple entries to pages.
 */
typedef struct RaBitQBatch
{
    uint16_t count;        /* number of encoded vectors */
    uint16_t packed_bytes; /* ceil(dim/8) per vector */
    float   *f_add;        /* [count] */
    float   *f_rescale;    /* [count] */
    uint8_t *bits;         /* [count * packed_bytes] */
} RaBitQBatch;

// RaBitQ quantizer state (shared across all vectors in an index)
typedef struct {
    float    *P;            // Random orthogonal matrix (dim x dim), row-major
    Dimension dim;
    uint32_t  packed_bytes; // ceil(dim / 8)
    uint64_t  seed;         // Seed for reproducibility
} RaBitQParams;

// Initialize with random orthogonal matrix
RaBitQParams *mkt_rabitq_create(Dimension dim, uint64_t seed);
int           mkt_rabitq_init(RaBitQParams *params, Dimension dim,
                              uint64_t seed);
void          mkt_rabitq_destroy(RaBitQParams *params);
void          mkt_rabitq_cleanup(RaBitQParams *params);

// Encode vector to compact RaBitQ format
RaBitQData *mkt_rabitq_encode(
    const RaBitQParams *params,
    Vec32Ref input,
    Vec32Ref centroid
);

// Encode into pre-allocated buffer
int mkt_rabitq_encode_into(
    const RaBitQParams *params,
    Vec32Ref input,
    Vec32Ref centroid,
    RaBitQData *output   // Must be MKT_RABITQ_DATA_SIZE(dim) bytes
);

// Batch encode to separate output arrays
int mkt_rabitq_encode_batch(
    const RaBitQParams *params,
    const float *vectors, Vec32Ref centroid,
    float *f_add, float *f_rescale, uint8_t *bits,
    uint16_t count
);

// Batch encode with heap-allocated RaBitQBatch output
RaBitQBatch *mkt_rabitq_encode_batch_alloc(
    const RaBitQParams *params,
    const float *vectors, Vec32Ref centroid,
    uint16_t count
);

// Compute estimated L2² distance from compact data
Distance mkt_rabitq_distance(
    const RaBitQQueryState *query_state,
    const RaBitQData *data,
    Dimension dim
);

// Compute estimated distance with derived error bound
void mkt_rabitq_distance_with_bound(
    const RaBitQQueryState *query_state,
    const RaBitQData *data,
    Dimension dim,
    Distance *est_dist,
    Distance *lower_bound
);
```

**Encoding implementation:**

```c
// Error bound constant from RaBitQ-Library (empirically tuned)
#define MKT_RABITQ_EPSILON 1.9f

int
mkt_rabitq_encode_into(const RaBitQParams *params, Vec32Ref input,
                       Vec32Ref centroid, RaBitQData *output)
{
    Dimension dim = params->dim;
    float *trans = mkt_alloc(dim * sizeof(float));

    // Step 1: Compute residual = input - centroid, track norms
    float l2_sqr = 0.0f, l1_norm = 0.0f;
    for (Dimension i = 0; i < dim; i++) {
        float r = input.data[i] - centroid.data[i];
        // ... matrix-vector multiply P^T * residual ...
        trans[i] = sum;
        l2_sqr += r * r;
    }

    // Step 2: Extract sign bits (LSB-first, FAISS-compatible)
    rabitq_extract_signs(trans, output->bits, dim);

    // Step 3: Compute L1 norm of transformed vector
    for (Dimension i = 0; i < dim; i++)
        l1_norm += fabsf(trans[i]);

    // Step 4: Compute factors
    float sqrt_d = sqrtf((float)dim);
    output->f_add = l2_sqr;
    output->f_rescale = l2_sqr * sqrt_d / l1_norm;
    // f_error derived at query time: C_error * sqrt(f_rescale² - f_add)

    mkt_free(trans);
    return 0;
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

The factor-based approach precomputes values so distance estimation requires
only a few arithmetic operations. Query factors are computed once and reused
for all database vectors.

```c
// Query preparation (done once per query)
typedef struct {
    float *transformed;      // P^T * (query - centroid)
    float  g_add;            // ||query - centroid||²
    float  g_error;          // sqrt(g_add) for error bound
    float  sum_transformed;  // sum(transformed) for distance formula
    float  inv_sqrt_d;       // 1 / sqrt(dim)
    float  c_error;          // 2*ε/√(d-1), for deriving f_error
    Dimension dim;
} RaBitQQueryState;

RaBitQQueryState *
mkt_rabitq_prepare_query(const RaBitQParams *params, Vec32Ref query,
                         Vec32Ref centroid)
{
    // ... transforms query through P^T, precomputes factors ...
}

// Estimated L2² distance from compact data (FAISS-style)
//   est_dist = g_add + f_add - 2 * f_rescale * final_dot
// where final_dot = (2 * binary_ip - sum_transformed) * inv_sqrt_d
Distance
mkt_rabitq_distance(const RaBitQQueryState *query_state,
                    const RaBitQData *data, Dimension dim)
{
    float binary_ip = simd_inner_product(query_state->transformed,
                                         data->bits, dim);
    float final_dot = (2.0f * binary_ip - query_state->sum_transformed)
                      * query_state->inv_sqrt_d;
    float est = query_state->g_add + data->f_add
                - 2.0f * data->f_rescale * final_dot;
    return fmaxf(0.0f, est);
}

// Distance with lower bound for two-stage filtering
// f_error derived: c_error * sqrt(f_rescale² - f_add)
void
mkt_rabitq_distance_with_bound(const RaBitQQueryState *query_state,
                               const RaBitQData *data, Dimension dim,
                               Distance *est_dist,
                               Distance *lower_bound)
{
    *est_dist = mkt_rabitq_distance(query_state, data, dim);
    float f_error = query_state->c_error
                    * sqrtf(data->f_rescale * data->f_rescale - data->f_add);
    *lower_bound = fmaxf(0.0f, *est_dist - f_error * query_state->g_error);
}

void
mkt_rabitq_free_query(RaBitQQueryState *state)
{
    mkt_free(state->transformed);
    mkt_free(state);
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

- `RaBitQData` (compact form): 4 (f_add) + 4 (f_rescale) + D/8 (bits)
  = 8 + D/8 bytes per vector
- For 768-dim: 8 + 96 = 104 bytes per vector
- `RaBitQVector` (PG varlena, future): adds 8-byte header = 16 + D/8 bytes
- Orthogonal matrix P: D² × 4 bytes (shared across all vectors in index)

Binary codes use LSB-first bit packing (FAISS-compatible).

For 768-dim vectors: 104 bytes/vector vs 3072 bytes for float32 (~30× compression).

#### Implementation Insights from Reference Code

The following details are derived from analyzing the faiss (`../faiss/`) and
RaBitQ-Library (`../RaBitQ-Library/`) implementations. The `RaBitQData`
struct and encoding/distance functions above already incorporate the factor-based
approach (f_add, f_rescale) used by production implementations. The f_error
bound is derived at query time from f_add and f_rescale.

**Two-stage search with error bounds:**

RaBitQ's theoretical error bound enables two-stage search: use fast 1-bit
estimates to filter candidates, then rerank survivors with full precision.

```c
// kConstEpsilon = 1.9 (from RaBitQ-Library, empirically tuned)
#define MKT_RABITQ_ERROR_EPSILON 1.9f

typedef struct {
    float g_add;    // ||query - centroid||²
    float g_error;  // Query-specific error term
} RaBitQQueryFactors;

// Two-stage search pseudocode:
void mkt_rabitq_search_two_stage(
    const RaBitQQueryFactors *query_factors,
    const RaBitQFactors *vec_factors,
    const uint8_t *binary_codes,
    uint32_t n_vectors,
    uint32_t k,
    float *heap_distances,
    uint32_t *heap_indices
) {
    float threshold = heap_distances[0];  // Current k-th best distance

    for (uint32_t i = 0; i < n_vectors; i++) {
        // Stage 1: Fast 1-bit estimate with lower bound
        float est_dist = compute_1bit_distance(query_factors, &vec_factors[i],
                                                binary_codes + i * code_bytes);

        // Error adjustment for filtering
        float error_adj = vec_factors[i].f_error * query_factors->g_error;
        float lower_bound = est_dist - error_adj;

        // Skip if lower bound can't beat current threshold
        if (lower_bound >= threshold) {
            continue;  // Filtered out
        }

        // Stage 2: Full precision reranking (expensive, but rare)
        float true_dist = compute_full_precision_distance(...);
        if (true_dist < threshold) {
            heap_update(heap_distances, heap_indices, k, i, true_dist);
            threshold = heap_distances[0];
        }
    }
}
```

The skip rate depends on data distribution but typically filters 80-95% of
candidates, significantly reducing full-precision distance computations.

**Batch processing with FastScan:**

Both faiss and RaBitQ-Library process vectors in batches of 32 using lookup
tables (LUT) for SIMD-friendly accumulation:

```c
#define MKT_RABITQ_BATCH_SIZE 32

// Data layout for batch processing (from RaBitQ-Library):
// [Binary codes: padded_dim * 32 / 8 bytes]
// [f_add values: 32 floats]
// [f_rescale values: 32 floats]
// [f_error values: 32 floats]

typedef struct {
    uint8_t *binary_codes;  // Packed codes for 32 vectors
    float f_add[32];
    float f_rescale[32];
    float f_error[32];
} RaBitQBatch;

// FastScan uses 4-bit lookup tables:
// - Each 4 dimensions map to one 16-entry LUT
// - LUT[i] = sum of query values where bits match pattern i
// - SIMD shuffles perform 16 parallel lookups per instruction
```

**Dimension padding requirements:**

For efficient SIMD processing, dimensions must be padded:

```c
// Pad dimension to multiple of 64 for binary code alignment
#define MKT_RABITQ_PAD_DIM(dim) (((dim) + 63) & ~63)

// Example: 768-dim needs no padding (768 % 64 == 0)
// Example: 384-dim needs no padding (384 % 64 == 0)
// Example: 100-dim pads to 128 (100 -> 128)
```

**Query preprocessing:**

Queries require preprocessing to compute factors used across all distance
computations:

```c
typedef struct {
    float *transformed;   // P^T * (query - centroid), dim floats
    float *lut;          // Lookup table for FastScan, 16 * (dim/4) floats
    float g_add;         // ||query - centroid||²
    float g_error;       // Error term for filtering
    float k1xsumq;       // sum(transformed) * (-0.5)
    float sum_vl_lut;    // LUT offset sum
    float delta;         // LUT quantization step
} RaBitQQuery;

RaBitQQuery *mkt_rabitq_prepare_query(
    const RaBitQParams *params,
    Vec32Ref query,
    Vec32Ref centroid
);

void mkt_rabitq_free_query(RaBitQQuery *q);
```

**Multi-bit support (future consideration):**

Faiss supports 1-9 bits per dimension. Higher bit counts improve accuracy at
the cost of storage and speed:

| Bits | Bytes/dim | Compression | Use case                    |
|------|-----------|-------------|----------------------------|
| 1    | 0.125     | 32×         | Initial filtering          |
| 2    | 0.25      | 16×         | Better accuracy, still fast |
| 4    | 0.5       | 8×          | High accuracy requirements |
| 8    | 1.0       | 4×          | Near-lossless              |

For multi-bit, extra bits are stored separately and combined with 1-bit codes
during distance computation. The two-stage approach first evaluates 1-bit
lower bounds, then refines with additional bits only for promising candidates.

**Key constants from reference implementations:**

```c
// Error bound multiplier (RaBitQ-Library)
#define MKT_RABITQ_EPSILON 1.9f

// Tight start values for multi-bit optimization (RaBitQ-Library)
// Used to accelerate finding optimal quantization scaling
static const float kTightStart[] = {
    0.0f,   // 1-bit (unused, sign-based)
    0.15f,  // 2-bit
    0.20f,  // 3-bit
    0.52f,  // 4-bit
    0.59f,  // 5-bit
    0.71f,  // 6-bit
    0.75f,  // 7-bit
    0.77f,  // 8-bit
    0.81f   // 9-bit
};

// Optimal query quantization radii (faiss)
// For centered query quantization to accelerate distance computation
static const float kQueryQuantRadii[] = {
    0.79688f, 1.49375f, 2.05078f, 2.50938f,
    2.91250f, 3.26406f, 3.59844f, 3.91016f
};
```

**Popcount-based distance (alternative formulation):**

For symmetric comparisons (both vectors quantized), Hamming distance via
XOR + popcount is extremely fast:

```c
// Centered inner product using popcount (faiss):
// int_dot = ((1 << qb) - 1) * dim - 2 * popcount(query XOR code)
//
// This exploits: sum of XOR bits = dim - 2 * (matching bits)
// Much faster than float arithmetic when both sides are quantized.
```

#### Scalar Quantization (SQ8) — Future/Lower Priority

> **Note**: SQ8 is not part of the initial implementation. RaBitQ is Meerkat's
> quantization method. This section documents SQ8 for potential future use with
> lower-dimensional vectors or use cases requiring higher recall without
> reranking.

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
    Vec32Ref input,
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
    Vec32Ref query,          // Full precision
    const ScalarQ8 *quantized,
    const SQ8Params *params
);
```

#### SQ8 Encoding (AVX-512)

```c
void mkt_sq8_encode_avx512(
    const SQ8Params *params,
    Vec32Ref input,
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

void mkt_bq_encode(const BQParams *params, Vec32Ref input, BinaryQ *output);

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
void mkt_sbq_add_sample(SBQParams *params, Vec32Ref sample);
void mkt_sbq_finish_training(SBQParams *params);

// Quantize vector
void mkt_sbq_encode(const SBQParams *params, Vec32Ref input, BinaryQ *output);

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

**Files**: `src/algo/topk.h`, `src/algo/topk.c`

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

**Files**: `src/algo/kmeans.h`, `src/algo/kmeans.c`

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
        Vec32Ref centroid = { .data = centroids + (c - 1) * dim, .dim = dim };
        for (uint32_t i = 0; i < nvecs; i++) {
            Vec32Ref v = { .data = vectors + i * dim, .dim = dim };
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
            Vec32Ref v = { .data = vectors + i * dim, .dim = dim };

            Distance best_dist = INFINITY;
            ClusterId best_cluster = 0;

            for (uint32_t c = 0; c < nlist; c++) {
                Vec32Ref centroid = {
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
        Vec32Ref centroid = {
            .data = result->centroids + c * dim,
            .dim = dim
        };

        Distance best_dist = INFINITY;
        uint32_t best_idx = 0;

        // Find vector in cluster closest to centroid
        for (uint32_t i = 0; i < nvecs; i++) {
            if (result->assignments[i] != c) continue;

            Vec32Ref v = { .data = vectors + i * dim, .dim = dim };
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

**CLI**: `mkt bench cluster`

```
$ mkt bench cluster --dim 128 --nvecs 100000 --nlist 1000
K-means clustering (dim=128, nvecs=100000, nlist=1000):
  Initialization: 1.2s
  Lloyd iterations: 8
  Total time: 4.5s
  Avg cluster size: 100 (std: 23)
```

### 2.5 Quantization Benchmark

**Files**: `src/cli/cmd_bench.c`, `test/bench/quant_bench.h`, `test/bench/quant_bench.c`

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
        Vec32Ref query = {
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
        Vec32Ref query = { .data = ds->queries + q * ds->dim, .dim = ds->dim };
        Vec32Ref true_nn = {
            .data = ds->vectors + ds->groundtruth[q * ds->k] * ds->dim,
            .dim = ds->dim
        };
        Vec32Ref found_nn = {
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

#### CLI

**Command**: `mkt bench quantize`

```
$ mkt bench quantize --dataset sift1m --k 10

Quantization Benchmark: 1000000 vectors, 128 dims, 10000 queries, k=10

Method     Recall@10 DistRatio  Encode(ms)  Query(us)  Compress
------     --------- ---------  ----------  ---------  --------
none         1.0000    1.0000         0.0      125.3     1.0x
rabitq       0.9847    1.0023       245.2       12.4    32.0x
sq8          0.9912    1.0008       102.3       18.7     4.0x
sbq          0.9756    1.0045       312.5       10.2    32.0x
bq           0.8234    1.0312        45.1        5.3    32.0x

$ mkt bench quantize --dataset gist1m --k 100 --methods rabitq,sq8

Quantization Benchmark: 1000000 vectors, 960 dims, 1000 queries, k=100

Method     Recall@100 DistRatio  Encode(ms)  Query(us)  Compress
------     ---------- ---------  ----------  ---------  --------
none          1.0000    1.0000         0.0      892.1     1.0x
rabitq        0.9723    1.0089      1823.4       89.2    32.0x
sq8           0.9801    1.0034       567.2      134.5     4.0x
```

**Options:**

```
mkt bench quantize - Quantization method benchmark

Usage: mkt bench quantize [OPTIONS]

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
$ mkt bench quantize --dataset sift1m --sweep-recall 0.90,0.95,0.99

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
- [RaBitQ implementation](https://github.com/gaoj0017/RaBitQ)

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
        ./builddir/mkt bench quantize \
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

### 3.1 Page Layout

**Files**: `src/index/posting_page.h`, `src/index/posting_page.c`

Posting list pages use a forward-growing AoS (Array-of-Structures) layout:
each entry is a fixed-size block containing its metadata, RaBitQ factors, and
quantized bits contiguously. The SIMD scan kernel strides through entries at
`entry_size` (116 bytes at dim=768), which lets consecutive pages' bit regions
sit only tens of bytes apart in memory — the HW prefetcher absorbs the
per-page perturbation cleanly.

Two storage modes share the same entry format and the same scan code:

- **Paged mode** — a chain of BLCKSZ pages with a PG-compatible `PageHeaderData`
  at the start and an opaque footer at the end, matching PG buffer-cache
  conventions (B-tree, GIN, GiST, pgvector).
- **Flat mode** — a single contiguous buffer per cluster with a minimal
  header, exactly sized to the cluster's entry count. Used as a reference
  point and for bindings where no PG framing is needed.

```
┌─────────────────────────────────────────────────────────┐    Paged mode
│ PostgreSQL PageHeaderData                    (24 bytes) │
├─────────────────────────────────────────────────────────┤
│ (first page only) pt_centroid[dim]                      │    MAXALIGN(4*dim)
├─────────────────────────────────────────────────────────┤
│ entry[0]   MktPostingEntryHeader + bits  ┐              │
│ entry[1]                                 │              │
│   ...                                    ├─ AoS entries │
│ entry[max-1]                             ┘              │
├─────────────────────────────────────────────────────────┤
│ MktPostingPageOpaque                         (16 bytes) │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐    Flat mode
│ MktFlatPostingHeader                         (16 bytes) │
├─────────────────────────────────────────────────────────┤
│ entry[0] ... entry[count-1]                             │
└─────────────────────────────────────────────────────────┘
```

#### Per-entry block

Each entry is a 116-byte (at dim=768) contiguous block:

```c
typedef struct MktPostingEntryMeta
{
    ItemPointerData tid;      // 6B — heap TID (PG) or packed vector_id (standalone)
    uint8_t         flags;    // entry flags (DELETED, BOUNDARY)
    uint8_t         reserved;
} MktPostingEntryMeta;        // 8B total

typedef struct MktPostingEntryHeader
{
    MktPostingEntryMeta meta;       // 8B
    float               f_add;      // 4B — per-entry RaBitQ additive factor
    float               f_rescale;  // 4B — scaling factor
    float               f_error;    // 4B — error-bound factor (pre-stored)
} MktPostingEntryHeader;            // 20B total

// Followed in memory by: uint8_t bits[MKT_RABITQ_BYTES(dim)]

// Flags
#define MKT_POSTING_FLAG_DELETED   0x01  // Soft-deleted, pending vacuum
#define MKT_POSTING_FLAG_BOUNDARY  0x02  // Replicated to adjacent cluster
```

Per-entry size:

```c
#define MKT_POSTING_ENTRY_SIZE(dim) \
    (sizeof(MktPostingEntryHeader) + MKT_RABITQ_BYTES(dim))
```

At dim=768: `20 + 96 = 116` bytes per entry.

`f_error` is stored per entry (not derived at query time) so the scan's
pruning step doesn't need a per-entry `sqrtf`. It is computed once per entry
at build time from the entry's factors.

#### Opaque (paged mode)

```c
typedef struct MktPostingPageOpaque
{
    BlockNumber next_blkno;     // 4B — next page in chain, or InvalidBlockNumber
    uint32_t    cluster_id;     // 4B
    uint16_t    entry_count;    // 2B — live entries on this page
    uint16_t    flags;          // 2B — FIRST, OVERFLOW, FASTSCAN (reserved)
    uint16_t    page_id;        // 2B — MKT_POSTING_PAGE_ID ("MP")
    uint16_t    max_entries;    // 2B — capacity at this page's dim
} MktPostingPageOpaque;         // 16B total

#define MKT_POSTING_PAGE_FIRST     0x0001  // First page of posting list
#define MKT_POSTING_PAGE_OVERFLOW  0x0002  // Overflow page
#define MKT_POSTING_PAGE_FASTSCAN  0x0004  // Reserved for future fastscan layout
```

#### pt_centroid on first pages

The first page of each posting list's chain carries `pt_centroid` — the
rotated centroid `P^T · centroid_of_this_cluster` — right after the
PageHeader. The scan reads it when entering a new cluster and uses it to
initialize the per-cluster RaBitQ query state via an `O(dim)` residual
subtraction instead of an `O(dim²)` matrix multiply. Overflow pages do not
carry pt_centroid.

#### Flat mode header

```c
typedef struct MktFlatPostingHeader
{
    uint32_t max_entries;     // == capacity (buffer is exactly sized)
    uint32_t entry_count;
    uint32_t cluster_id;
    uint32_t _pad;
} MktFlatPostingHeader;       // 16B total
```

Flat mode doesn't need `pd_lsn`, `pd_checksum`, `next_blkno`, or any of PG's
page plumbing — there's no buffer cache, no chain, no WAL. It exists as a
zero-framing reference point.

### 3.2 Why AoS

The SIMD scan kernel (`mkt_rabitq_inner_product_multi_avx512`) reads bits at
some stride and does 4-wide masked accumulation across dim per group. The
stride is programmable. With AoS we pass `stride = MKT_POSTING_ENTRY_SIZE(dim)`
and `bits_base = mkt_posting_first_bits(content)` — no new kernel code
required.

Two effects combine:

1. **Small per-page prefetcher perturbation.** Consecutive pages in the same
   posting list sit adjacent in physical memory. The AoS entries on page N
   end near offset `BLCKSZ - opaque_size`; the AoS entries on page N+1 start
   near `BLCKSZ + PageHeader_size`. That leaves only a few dozen bytes of
   non-bits data between the last bits of one page and the first bits of the
   next. The HW stream prefetcher can absorb this.

   An SoA layout, by contrast, interleaves `metas` + `f_add` + `f_rescale` +
   `f_error` at the front of each page before `bits`, so consecutive pages'
   bits regions sit ~1.5 KB apart. The prefetcher loses its lock at every
   page boundary and has to re-acquire. Paged scans paid this cost at every
   transition.

2. **Free metadata colocation.** Striding at `entry_size` pulls each entry's
   20 B header into L1 alongside its 96 B of bits (they share cache lines).
   The distance-conversion and prune step finds `f_add`, `f_rescale`,
   `f_error`, and the TID already warm, so the per-entry scalar math in
   phase 3 runs without cold reads.

### 3.3 Capacity and access helpers

```c
// Usable space between content start and opaque footer
static inline uint32_t
mkt_posting_page_usable(void)
{
    return BLCKSZ - (uint32_t)MAXALIGN(SizeOfPageHeaderData)
                  - sizeof(MktPostingPageOpaque);
}

// Max entries on an overflow page
static inline uint32_t
mkt_posting_max_entries(Dimension dim)
{
    return mkt_posting_page_usable() / MKT_POSTING_ENTRY_SIZE(dim);
}

// Max entries on a first page (reduced by pt_centroid reservation)
static inline uint32_t
mkt_posting_max_entries_first(Dimension dim)
{
    return (mkt_posting_page_usable() - MAXALIGN(dim * sizeof(float)))
           / MKT_POSTING_ENTRY_SIZE(dim);
}

// Entry i's header (meta + factors)
static inline MktPostingEntryHeader *
mkt_posting_entry_at(char *content, uint32_t i, Dimension dim);

// Entry i's bits (immediately follows entry i's header)
static inline uint8_t *
mkt_posting_entry_bits_at(char *content, uint32_t max_entries,
                          Dimension dim, uint32_t i);

// Pointer to entry 0's bits — the base the SIMD kernel strides from
static inline uint8_t *
mkt_posting_first_bits(char *content);
```

At **dim = 768, BLCKSZ = 8192**:

- overflow page: `8152 / 116 = 70` entries
- first page:   `(8152 - 3072) / 116 = 43` entries (reserves 3072 bytes for pt_centroid)

### 3.4 Page operations

```c
// Paged mode
void mkt_posting_page_init(Page page, uint32_t cluster_id,
                           Dimension dim, uint16_t flags);
bool mkt_posting_page_add(Page page, Dimension dim,
                          ItemPointerData tid,
                          float f_add, float f_rescale, float f_error,
                          const uint8_t *bits,
                          uint8_t entry_flags);

// Flat mode
void mkt_posting_flat_init(char *buf, uint32_t max_entries,
                           uint32_t cluster_id);
bool mkt_posting_flat_add(char *buf, Dimension dim,
                          ItemPointerData tid,
                          float f_add, float f_rescale, float f_error,
                          const uint8_t *bits,
                          uint8_t entry_flags);
```

Entry writes go through these for both modes; the writers place the
`MktPostingEntryHeader` and `bits` contiguously at `content + i*entry_size`.

### 3.5 Why not SoA?

An earlier iteration laid out each page as per-array SoA regions:

```
┌────────────────────────────────────────┐
│ PageHeader                             │
├────────────────────────────────────────┤
│ MktPostingEntryMeta[max]      (8B × N) │
│ f_add[max]                    (4B × N) │
│ f_rescale[max]                (4B × N) │
│ f_error[max]                  (4B × N) │
│ bits[max * packed_bytes]               │
├────────────────────────────────────────┤
│ Opaque                                 │
└────────────────────────────────────────┘
```

Within a single page the SIMD kernel sees a clean sequential stride over
`bits` and runs at full speed. But *across* pages, the bits regions of
consecutive pages sit roughly 1.5 KB apart in memory: page N's `bits` ends
near the opaque footer, and page N+1's `bits` starts only after its
PageHeader, the four metadata arrays of N+1, and the opaque of N. For a
cohere-1M workload with ~15 pages per posting list, the scan kernel paid
~14 such ~1.5 KB gaps per cluster.

The cost didn't show up as L1 misses or TLB misses — both metrics looked
fine. It showed up as backend-bound stalls inside the SIMD kernel: the
HW stream prefetcher loses its stride lock at each page boundary and
takes a few cache lines of re-learning before it's prefetching ahead
again. Multiplied across pages, this was the bulk of a ~28% paged-vs-flat
QPS gap we could otherwise not explain.

Three experiments confirmed the story before AoS landed:

- **Span-fix kernel** (one `_multi` invocation per posting list, per-
  candidate bits pointers): flat, no improvement. Disproved "function-call
  overhead per page" as the cause.
- **Cache-line alignment** (round `bits[]` to a 64 B boundary inside the
  page): no improvement. Disproved "cache-line straddling" as the cause.
- **BLCKSZ experiment** (bump standalone BLCKSZ from 8192 to 65535 so a
  posting list fits in ~2 pages): +33% on paged mode, essentially closing
  the gap. Confirmed "bits region transitions" as the cause.

AoS closes the same gap without changing BLCKSZ: consecutive pages'
entry regions sit only ~72 bytes apart (opaque + next PageHeader), a
perturbation small enough for the prefetcher to absorb. Scan code is
unchanged — it just strides at `entry_size` instead of `packed_bytes`.

### 3.6 Building a posting list

**Files**: `src/index/posting_build.h`, `src/index/posting_build.c`

The `MktPostingBuilder` streams entries into the chosen page format. Build
flow for each cluster:

```c
MktPostingBuilder builder;
mkt_posting_builder_init(&builder, storage, rq_params, dim,
                         cluster_id, centroid, pt_centroid);
for each vector in cluster:
    mkt_posting_builder_add(&builder, tid, vector);  // RaBitQ-encoded internally
BlockNumber head = mkt_posting_builder_finish(&builder);
mkt_posting_builder_cleanup(&builder);
```

The builder encodes each input vector with RaBitQ (producing `f_add`,
`f_rescale`, `f_error`, and bits), then appends a page entry. It manages the
BLCKSZ-page chain internally: opens a new page when the current one fills,
links pages via `next_blkno`, and writes `pt_centroid` to the first page's
header area.

Flat mode has an analogous `MktFlatPostingBuilder` that writes into a
caller-provided pre-sized buffer.

### 3.7 Metapage

```c
// Metapage layout (block 0)
typedef struct {
    uint32_t    magic;           // Magic number for validation
    uint32_t    version;         // Format version
    Dimension   dim;             // Vector dimension
    DistanceMetric metric;       // Distance metric
    uint32_t    nlist;           // Number of clusters
    BlockNumber next_meta_blkno; // Next metapage (if directory overflows)
    // RaBitQ parameters follow (orthogonal matrix seed, etc.)
    // Then: posting_list_heads[nlist] (BlockNumber per cluster)
} MktMetapage;

#define MKT_MAGIC 0x4D4B4154  // "MKAT"
#define MKT_VERSION 1

// Directory entries per metapage (after fixed header + RaBitQ params)
size_t mkt_meta_directory_capacity(Dimension dim);

// Initialize metapage
void mkt_meta_init(
    void *page,
    Dimension dim,
    DistanceMetric metric,
    uint32_t nlist,
    const RaBitQParams *rabitq_params
);

// Get/set posting list head for cluster
BlockNumber mkt_meta_get_head(const void *page, ClusterId cluster);
void mkt_meta_set_head(void *page, ClusterId cluster, BlockNumber block);
```

**Tests**:
- Page layout: verify entry packing, no overlap
- Capacity: verify calculated entries match actual
- Round-trip: write entries, read back, compare
- Overflow: verify full page returns false on add

---

## Part 4: Index Build

### 4.1 Build Pipeline

**Files**: `src/pg/build/build.h`, `src/pg/build/build.c`

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

    // Quantization (RaBitQ)
    RaBitQParams   *rabitq_params;

    // One posting-list builder per cluster — streams RaBitQ-encoded
    // entries into paged or flat posting storage as vectors arrive.
    MktPostingBuilder *builders;

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
    BlockNumber block_number,
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

**CLI**: `mkt build`

```
$ mkt build --input vectors.bin --dim 768 --nlist 1000 --output index.mkt
Sampling: 10000 / 100000 vectors
Clustering: 1000 clusters, 15 iterations
Assigning: 100000 vectors
Writing: 5234 pages
Done. Index size: 42.8 MB
```

### 4.3 Build Optimizations (Roadmap)

See `docs/parallel-build-design.md` for the parallel insertion design
(interleaved page claiming, shared assignment code). Additional
optimizations for billion-scale builds are documented below.

#### 4.3.1 Hierarchical K-Means

Standard Lloyd's k-means with `c` clusters has complexity O(n·c·d·i)
where n=samples, c=clusters, d=dimensions, i=iterations. For large
nlist values (e.g., 160K for 1B vectors), the c factor dominates.

**Hierarchical approach**: partition the sample into sqrt(c) groups using
a coarse clustering pass, then run independent k-means within each
group. This reduces complexity from O(n·c·d·i) to O(n·sqrt(c)·d·i) —
roughly 400× faster for c=160,000.

Steps:
1. Run coarse k-means with sqrt(c) centroids on all samples
2. Assign each sample to its nearest coarse centroid
3. Allocate fine clusters proportionally per coarse partition
   (e.g., Modified Webster/Sainte-Laguë method for balanced allocation)
4. Run independent k-means within each partition (parallelizable)
5. Concatenate the resulting centroids

The coarse pass needs only ~10 iterations since its job is rough
partitioning, not precision.

#### 4.3.2 Dimensionality Reduction During Clustering

K-means only needs distances between vectors, not the vectors at full
precision. Reducing dimensionality before clustering cuts both memory
and computation proportionally.

**Approach**: apply a random orthogonal rotation (the same matrix used
by RaBitQ encoding) then truncate to `k` dimensions (e.g., 768 → 100).
This is a form of random projection that preserves distance structure.

Steps:
1. Generate or reuse the RaBitQ rotation matrix P
2. Rotate all sampled vectors: x' = P·x
3. Truncate to first k dimensions
4. Run k-means in reduced space
5. Recompute centroids in full dimensionality by re-scanning and
   accumulating vectors into their assigned clusters

For 768d → 100d, this reduces clustering memory ~7.7× and computation
proportionally. The rotation matrix is already available from the RaBitQ
encoding pipeline.

#### 4.3.3 Parallel Insertion with Multi-List Posting

The current parallel build uses a mutex on the page allocator. For high
worker counts (16+), this becomes a bottleneck. The solution is to
eliminate contention by giving each worker its own list chain per
cluster.

**Multi-list structure**: instead of one posting list per cluster,
maintain `P` separate list chains (where P = number of parallel
workers). Each worker appends exclusively to its own chain. At read
time, all chains for a cluster are scanned sequentially.

```
Cluster C with 4 workers:
  Chain 0: [page] → [page] → [page]  (worker 0 writes here)
  Chain 1: [page] → [page]           (worker 1 writes here)
  Chain 2: [page] → [page] → [page]  (worker 2 writes here)
  Chain 3: [page]                     (worker 3 writes here)
```

Benefits:
- Zero contention during the insertion phase (no mutexes)
- Each worker does sequential page writes to its own chains
- Read path scans all chains — minor overhead since chains are
  page-aligned and benefit from prefetching

The chain heads are stored in the cluster's metadata page. Skip
pointers in each page's opaque data enable O(1) jump to the end of
a chain for appending.

#### 4.3.4 Bulk Page Allocation

Instead of allocating one page at a time via the storage extend API,
request pages in batches (e.g., 16 at once). This allows the OS to
use `fallocate` for contiguous allocation instead of per-page `pwrite`.

For PostgreSQL, this maps to `ExtendBufferedRelBy` with a multi-page
request. Note: `LimitAdditionalPins` may silently reduce the request
size — the caller must loop until all requested pages are allocated.

#### 4.3.5 External Centroid Import

For very large datasets, k-means clustering can be the bottleneck even
with the optimizations above. Allow users to pre-compute centroids
using external tools (e.g., faiss, scikit-learn) and import them.

```sql
-- User pre-computes centroids and loads into a table
CREATE TABLE my_centroids (
    id       int,
    parent   int,
    centroid vector(768)
);
COPY my_centroids FROM 'centroids.csv' WITH (FORMAT csv);

-- Build index using external centroids
CREATE INDEX ON my_table USING mktann (v vec32_cosine_ops)
    WITH (external_centroids = 'my_centroids');
```

The build pipeline skips the sampling and clustering phases entirely,
reading the centroid tree directly from the specified table. The tree
structure is reconstructed from (id, parent) relationships.

This enables:
- GPU-accelerated k-means via faiss
- Custom clustering strategies (e.g., balanced k-means)
- Reproducible builds with fixed centroids
- Faster iteration when tuning index parameters (cluster once, rebuild
  posting lists multiple times)

---

## Part 5: Search

### 5.1 Search Interface

**Files**: `src/index/centroid_search.h`, `src/index/centroid_search.c`

Search is split into two phases: centroid routing (beam search through the
centroid tree) and posting list scan (RaBitQ distance on data vectors).

The centroid search phase uses `MktStorage` callbacks so the same code
runs in both standalone and PostgreSQL mode. There is **no separate in-memory
centroid cache** — search reads centroid pages directly via the storage
vtable, which in PG mode wraps the standard shared buffer cache.

```c
/* Search parameters (from reloptions / GUCs) */
typedef struct {
    uint32_t nprobe;      /* leaf clusters to scan */
    uint32_t k;           /* number of results */
    uint32_t beam_width;  /* candidates kept per tree level */
    uint32_t rerank_k;    /* candidates for full-precision re-rank */
} SearchParams;

/* Centroid search: beam search through centroid tree */
uint32_t mkt_centroid_beam_search(
    const MktCentroidSearchState *state,
    BlockNumber                   first_centroid_blkno,
    uint8_t                       nlevels,
    MktCentroidResult            *results,
    MktCentroidSearchStats       *stats);
```

The `MktCentroidSearchState` bundles a pre-computed `RaBitQQueryState`
(query transformed against the global mean), the raw query float pointer
(for float/half centroid pages), an opaque `query_datum` (for storage-level
reranking), the storage vtable, beam_width, nprobe, and dimension.

Results contain `posting_head` block numbers, `medoid_tid`, estimated
`distance`, and `error` margin for each selected leaf cluster. An
`MktCentroidSearchStats` struct tracks distance computations and rerank
counts for diagnostics.

### 5.2 Centroid Search Implementation

Centroid search uses level-by-level beam search via `MktStorage`
callbacks, reading centroid pages directly from the buffer cache (or from a
flat array in standalone mode). Centroid pages declare their data format
(RaBitQ, float32, or float16) in the opaque flags, and the search algorithm
dispatches format-specific distance computation per page.

**Search state** (`src/index/centroid_search.h`):

```c
typedef struct MktCentroidSearchState
{
    const RaBitQQueryState *qstate;      /* query for RaBitQ pages */
    const float            *query;       /* raw query for float/half pages */
    Datum                   query_datum; /* opaque query for reranking */
    MktStorage             *storage;     /* page and vector I/O */
    uint32_t                beam_width;
    uint32_t                nprobe;
    Dimension               dim;
} MktCentroidSearchState;

typedef struct MktCentroidResult
{
    BlockNumber     posting_head; /* first posting list page */
    ItemPointerData medoid_tid;   /* heap TID of medoid vector */
    Distance        distance;     /* estimated distance to query */
    Distance        error;        /* symmetric error margin */
} MktCentroidResult;

typedef struct MktCentroidSearchStats
{
    uint64_t dist_calcs; /* approximate distance computations */
    uint64_t reranked;   /* exact distance recomputations */
} MktCentroidSearchStats;
```

**Beam search** (`src/index/centroid_search.c`):

```c
uint32_t mkt_centroid_beam_search(
    const MktCentroidSearchState *state,
    BlockNumber                   first_centroid_blkno,
    uint8_t                       nlevels,
    MktCentroidResult            *results,
    MktCentroidSearchStats       *stats);
```

The algorithm is format-aware — each centroid page declares its format in the
opaque flags, and scoring dispatches accordingly:

1. **Level 0**: Read root centroid page(s), score all centroids via
   `score_page()` (see format dispatch below). Keep top `beam_width`
   candidates (or `nprobe` if single-level tree). Rerank RaBitQ survivors
   with exact distances via the storage `rerank` callback.

2. **Intermediate levels**: For each winner, read its child centroid page(s)
   via `child_blkno`, score all entries, select top `beam_width`. Rerank
   after each level.

3. **Leaf level**: Same as intermediate but select top `nprobe`. Return
   `posting_head` (child_blkno of leaf entries), `medoid_tid`, `distance`,
   and `error` for each.

**Format dispatch** (`score_page()` in `centroid_search.c`):

Distance computation is per-page, dispatched by `mkt_centroid_page_format()`:

- **RaBitQ pages**: Batch scoring using `ScorePageScratch` buffers. Gathers
  `f_add`/`f_rescale` arrays and calls
  `mkt_rabitq_distance_batch_multi_with_bound()` (asymmetric) or
  `mkt_rabitq_distance_batch_symmetric_with_bound()` (symmetric). Returns
  approximate distances with error bounds. Supports both asymmetric
  (full-precision query × 1-bit data) and symmetric (1-bit query × 1-bit
  data) modes.

- **Float32 pages**: Exact L2 via `mkt_l2_distance_squared()` on in-page
  float vectors accessed through `mkt_centroid_float_data()`. Returns exact
  distances with error = 0 (no reranking needed).

- **Float16 pages**: Exact L2 via `mkt_f16_l2_squared()` on in-page half
  vectors accessed through `mkt_centroid_half_data()`. Returns exact
  distances with error = 0 (no reranking needed).

**Reranking** (only for RaBitQ candidates):

After scoring each level, `rerank_candidates()` checks whether any candidate
has `error > 0`. If so, it calls `storage->ops->rerank()` to fetch
full-precision vectors and compute exact L2 distances for all candidates,
replacing approximate estimates. Float/half candidates have error = 0 and
skip reranking entirely.

### 5.3 Posting List Scan

```c
void mkt_search_posting_lists(
    const ClusterId *clusters,
    uint32_t nprobe,
    Vec32Ref query,
    const RaBitQParams *rabitq_params,
    DistanceMetric metric,
    const SearchParams *params,
    PageReadCallback read_page,
    PageReleaseCallback release_page,
    void *callback_data,
    TopKHeap *results
) {
    // Precompute query-dependent values for RaBitQ asymmetric distance
    float query_norm = mkt_vec_norm(query);
    float *query_normalized = mkt_alloc(query.dim * sizeof(float));
    mkt_vec_normalize(query, query_normalized);

    // Scan each cluster's posting list
    for (uint32_t p = 0; p < nprobe; p++) {
        ClusterId cluster = clusters[p];
        BlockNumber block = /* get from metapage */;

        while (block != InvalidBlockNumber) {
            const void *page = read_page(callback_data, block);
            MktPostingPageOpaque *opaque = mkt_posting_opaque(page);
            uint16_t entry_count = opaque->entry_count;
            char *content = (opaque->flags & MKT_POSTING_PAGE_FIRST)
                          ? mkt_posting_content_first(page, dim)
                          : mkt_posting_content(page);

            // Phase 1: batch IP for all entries on the page. Kernel
            // strides at entry_size through the AoS entries.
            Distance *distances = mkt_alloc(entry_count * sizeof(Distance));
            mkt_rabitq_inner_product_multi(
                    query_state->transformed,
                    mkt_posting_first_bits(content),
                    MKT_POSTING_ENTRY_SIZE(dim),
                    dim,
                    entry_count,
                    distances);

            // Phase 2: convert to distances, prune, insert survivors.
            // Each entry's header carries meta + f_add/f_rescale/f_error;
            // it's already hot in L1 from the strided bits read.
            for (uint16_t i = 0; i < entry_count; i++) {
                MktPostingEntryHeader *e = mkt_posting_entry_at(content, i, dim);
                if (e->meta.flags & MKT_POSTING_FLAG_DELETED) continue;

                Distance est = e->f_add /* + query-side constants and
                                           f_rescale * distances[i] */;

                if (est < mkt_topk_threshold(results)) {
                    uint64_t tid_encoded = mkt_posting_encode_tid(&e->meta.tid);
                    mkt_topk_insert(results, est, tid_encoded);
                }
            }

            mkt_free(distances);
            BlockNumber next = opaque->next_blkno;
            release_page(callback_data, block);
            block = next;
        }
    }

    mkt_free(query_normalized);
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
    Vec32Ref query,
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
        Vec32Ref v = { .data = vec_buffer, .dim = dim };
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

**CLI**: `mkt search`

```
$ mkt search --index index.mkt --query query.bin --k 10 --nprobe 20
Results (10 of 100000 vectors):
  1. tid=(42,15)  distance=0.0234
  2. tid=(108,3)  distance=0.0456
  ...
Search time: 2.3ms
```

---

## Part 6: PostgreSQL Integration

### 6.1 Extension Setup

**Files**: `src/pg/mkt_pg.c`, `src/pg/mkt_pg.h`

```c
// Extension initialization
void _PG_init(void);

// GUC variables
int mkt_distance_mode;       // MktDistanceMode: default / asymmetric / symmetric

// Index reloptions
relopt_kind mktann_relopt_kind;
```

**GUC**: `mkt.distance_mode` defaults to `'default'` (sentinel value
`MKT_DISTANCE_MODE_DEFAULT = -1`), meaning "use the index's relopt".
When set to `'asymmetric'` or `'symmetric'`, it overrides the index setting
for the current session.

**Relopt**: `distance_mode` is an enum relopt registered with
`add_enum_reloption()`. It defaults to `asymmetric` and is stored in the
index via `WITH (distance_mode = ...)`:

```sql
CREATE INDEX idx ON items USING mktann (v) WITH (distance_mode = 'symmetric');
```

**Resolution order** (at scan time via `MktannGetDistanceMode()`):
1. If GUC != `default` → use GUC value
2. Otherwise → use the index's relopt value
3. If no reloptions set → fall back to `asymmetric`

The `MktannOptions` struct (varlena header + `distance_mode` field) is
parsed by the `mktann_options()` callback using `build_reloptions()`.

### 6.2 Vector Type and pgvector Compatibility

**Files**: `src/types/vec32.h`, `src/types/vec32.c`

Meerkat defines its own `vec32` and `vec16` types, installed into whichever
schema `CREATE EXTENSION meerkat` targets (the `SCHEMA` clause, or the first
existing schema on `search_path` otherwise — typically `public`). Their names
are distinct from pgvector's `vector` and `halfvec`, so both sets of types can
coexist even when the extensions are installed into the same schema. The
corresponding pairs are binary-compatible
(`vec32`/`vector` and `vec16`/`halfvec`). This allows:

- Standalone builds without a pgvector dependency
- Direct indexing of pgvector columns through zero-copy binary casts
- Zero-overhead type handling through the shared memory layouts

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
#define VEC32_MAX_DIM 16000
#define VEC32_SIZE(_dim) (offsetof(Vec32, x) + sizeof(float)*(_dim))

typedef struct Vec32
{
    int32       vl_len_;        /* varlena header (do not touch directly!) */
    int16       dim;            /* number of dimensions */
    int16       unused;         /* reserved for future use, always zero */
    float       x[FLEXIBLE_ARRAY_MEMBER];
}           Vec32;

// Accessor macros
#define VEC32_DIM(v)    ((v)->dim)
#define VEC32_DATA(v)   ((v)->x)

// Convert to Vec32Ref for internal operations
static inline Vec32Ref
Vec32ToRef(const Vec32 *v)
{
    return (Vec32Ref){ .data = v->x, .dim = v->dim };
}
```

#### pgvector Detection and OID Caching

At extension load time, detect if pgvector is installed and cache its type OID:

```c
// Cached OIDs (InvalidOid if not available)
static Oid vec32_oid = InvalidOid;
static Oid pgvector_oid = InvalidOid;

void
vec32_init(void)
{
    // Cache our own type OID
    vec32_oid = GetSysCacheOid2(TYPENAMENSP,
                                      Anum_pg_type_oid,
                                      CStringGetDatum("vec32"),
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
    return typoid == vec32_oid ||
           (OidIsValid(pgvector_oid) && typoid == pgvector_oid);
}
```

#### Runtime Type Handling in Operators

#### Operator Family Membership

Type interoperability does not by itself extend to operators. PostgreSQL
considers an index for an `ORDER BY` only when the ordering operator belongs to
the index's operator family, and pgvector's `<->`, `<#>` and `<=>` belong to
pgvector's own families. Casting the column is not enough.

`setup_pgvector_compat()` therefore adds pgvector's three distance operators to
each of meerkat's six `mktann` families as ordering members, in the same step
that creates the casts:

```sql
ALTER OPERATOR FAMILY myschema.vec16_l2_ops USING mktann
    ADD OPERATOR 1 public.<-> (public.halfvec, public.halfvec)
        FOR ORDER BY pg_catalog.float_ops;
```

Strategy number and sort family match the opclass declarations. It runs on the
same both-install-orders path as the casts: if pgvector is already installed,
meerkat's install script adds them inline while `CREATE EXTENSION meerkat` runs
(an anonymous `DO $$ ... $$` block — a one-off script that runs during
install, not a lock); if pgvector is installed later, an event trigger adds them
then. It is idempotent through exception handling. `DROP EXTENSION vector
CASCADE` removes the members via their dependency on the operators, and a later
reinstall re-adds them.

The failure mode this removes is quiet: without family membership the planner
answers a pgvector-operator query with a sequential scan, which returns correct
rows. Results-only tests pass while measuring brute force, so the compat suite
asserts the plan as well as the recall.

One pairing does not resolve, by design: a `vec32` or `vec16` column
with pgvector's operator. The meerkat → pgvector cast is ASSIGNMENT rather than
IMPLICIT specifically so that having both extensions installed does not make
operator resolution ambiguous, and queries over meerkat's types use meerkat's
operators.

**Standalone builds**: For unit tests and CLI tools, use `Vec32` without the
varlena header, or define a minimal mock. The core algorithms operate on
`Vec32Ref`, which is independent of PostgreSQL types.

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

Page I/O uses the `MktStorage` vtable (see §6.5). In PG mode, the
implementation wraps `ReadBuffer`/`UnlockReleaseBuffer`/`GenericXLog`:

```c
// PG storage implementation embeds MktStorage as first member
typedef struct MktannStorage {
    MktStorage  base;       /* must be first (upcast via pointer) */
    Relation    index;
    Buffer      buffers[MAX_PINNED];
    int         nbuffers;
} MktannStorage;
```

The `read_page` callback calls `ReadBuffer` + `LockBuffer(SHARE)`,
`commit_page` wraps `GenericXLogFinish` + `UnlockReleaseBuffer`, and
`new_page` extends the relation via `ReadBufferExtended`.

### 6.5 Centroid Page Management

Centroids are stored in dedicated pages within the index file, between the
metapage and posting list pages. These pages are managed by PostgreSQL's
standard shared buffer cache—no separate in-memory cache structure is needed.

**Why no dedicated cache:**
- Centroid pages are accessed on every query (hot data)
- Hot pages naturally stay in shared buffers via LRU
- Standard PostgreSQL locking and WAL apply automatically
- Simplifies code and avoids cache coherency issues

**Page layout:**
```
Block 0:        Metapage (index metadata, RaBitQ params)
Blocks 1..C:    Centroid pages (hierarchical tree)
Blocks C+1..:   Posting list pages
```

**Centroid page structure** (`src/index/centroid_page.h`):

Centroid pages use bidirectional growth, inspired by PostgreSQL's standard
page layout. Entry metadata grows forward from the page header; vector data
grows backward from the opaque area. Both metadata size and data size are
format-dependent — the format is stored in the low 2 bits of
`opaque->flags` and set at page initialization:

| Format | Metadata | Data per entry |
|--------|----------|----------------|
| RaBitQ | 16B (`MktCentroidEntryMetaRaBitQ`) | 8 + ceil(D/8) bytes |
| Float32 | 8B (`MktCentroidEntryMeta`) | D × 4 bytes |
| Float16 | 8B (`MktCentroidEntryMeta`) | D × 2 bytes |

The page is full when the two regions would overlap.

```
┌──────────────────────────────────────────────────────────┐
│ PageHeaderData                                (24 bytes) │
├──────────────────────────────────────────────────────────┤
│ EntryMeta[0]              (format-dependent: 8B or 16B)  │
│ EntryMeta[1]                           ← grows forward   │
│       ...                                                │
├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┤
│       free space                                         │
├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┤
│       ...                                                │
│ data[1]  (RaBitQData / float[] / half[])                 │
│ data[0]                                ← grows backward  │
├──────────────────────────────────────────────────────────┤
│ MktCentroidPageOpaque                        (12 bytes)  │
└──────────────────────────────────────────────────────────┘
```

On **RaBitQ pages**, centroids are **medoids** — actual data vectors
referenced by heap TID (`ItemPointerData` stored in the extended metadata).
Medoids are RaBitQ-encoded relative to the global data mean, sharing a
single orthogonal matrix P. The medoid TID enables reranking with exact
distances fetched from the heap.

On **float/half pages**, centroid vectors are stored directly in
full/half precision. No medoid TID is needed because exact distances are
computed on the in-page vectors without reranking.

```c
/* Centroid data format (stored in low 2 bits of opaque->flags) */
typedef enum MktCentroidFormat
{
    MKT_CENTROID_FMT_RABITQ = 0,   /* RaBitQData (default) */
    MKT_CENTROID_FMT_FLOAT  = 1,   /* float32 vectors */
    MKT_CENTROID_FMT_HALF   = 2,   /* float16 vectors */
} MktCentroidFormat;

/* Base per-centroid metadata (8 bytes, all formats) */
typedef struct MktCentroidEntryMeta
{
    BlockNumber child_blkno;       /* child centroid page (internal)
                                      or posting list head (leaf) */
    uint16_t    child_count;       /* children at next level */
    uint16_t    flags;             /* MKT_CENTROID_FLAG_LEAF etc. */
} MktCentroidEntryMeta;            /* 8 bytes */

/* Extended metadata for RaBitQ format (16 bytes) */
typedef struct MktCentroidEntryMetaRaBitQ
{
    MktCentroidEntryMeta base;     /* 8B - common fields */
    ItemPointerData      medoid_tid; /* 6B - heap TID for rerank */
    uint16_t             reserved; /* 2B - alignment */
} MktCentroidEntryMetaRaBitQ;     /* 16 bytes */

/* Page special area (12 bytes, at page end per PG convention) */
typedef struct MktCentroidPageOpaque
{
    BlockNumber next_blkno;        /* next page at same level */
    uint16_t    entry_count;       /* centroids on this page */
    uint8_t     level;             /* tree level (0 = root) */
    uint8_t     flags;             /* low 2 bits: MktCentroidFormat */
    uint16_t    page_id;           /* MKT_CENTROID_PAGE_ID (0x4D43) */
    uint16_t    padding;           /* alignment */
} MktCentroidPageOpaque;           /* 12 bytes */
```

**Capacity** (768 dimensions, usable = 8152 bytes per page):

| Format | Meta | Data | Entry total | Entries per page |
|--------|------|------|-------------|-----------------|
| RaBitQ | 16B | 104B (8B scalars + 96B bits) | 120B | **67** |
| Float16 | 8B | 1536B | 1544B | **5** |
| Float32 | 8B | 3072B | 3080B | **2** |

RaBitQ is the default format, offering 32× compression over float32 with
error-bounded approximate distances. Float/half formats trade capacity for
exact routing — useful for small trees or when accuracy is critical.

**Access helpers:**

```c
/* Opaque area via PG-standard PageGetSpecialPointer */
#define MKT_CENTROID_OPAQUE(page) \
    ((MktCentroidPageOpaque *)PageGetSpecialPointer(page))

/* Page format from opaque flags */
static inline MktCentroidFormat
mkt_centroid_page_format(Page page) {
    return (MktCentroidFormat)(MKT_CENTROID_OPAQUE(page)->flags &
                               MKT_CENTROID_FMT_MASK);
}

/* Format-dependent metadata size */
static inline uint32_t
mkt_centroid_meta_size(MktCentroidFormat fmt) {
    if (fmt == MKT_CENTROID_FMT_RABITQ)
        return sizeof(MktCentroidEntryMetaRaBitQ);  /* 16B */
    return sizeof(MktCentroidEntryMeta);             /* 8B */
}

/* i-th base metadata entry — byte offset arithmetic to handle
 * variable metadata sizes across formats */
static inline const MktCentroidEntryMeta *
mkt_centroid_meta(const Page page, uint32_t index) {
    MktCentroidFormat fmt = mkt_centroid_page_format(page);
    return (const MktCentroidEntryMeta *)
        ((const char *)page + SizeOfPageHeaderData
         + (size_t)index * mkt_centroid_meta_size(fmt));
}

/* i-th RaBitQ extended metadata (only valid on RaBitQ pages) */
static inline const MktCentroidEntryMetaRaBitQ *
mkt_centroid_meta_rabitq(const Page page, uint32_t index) {
    return (const MktCentroidEntryMetaRaBitQ *)
        mkt_centroid_meta(page, index);
}

/* i-th data entry (backward region, format-dependent size) */
static inline const void *
mkt_centroid_entry_data(const Page page, uint32_t index,
                        Dimension dim) {
    MktCentroidFormat fmt       = mkt_centroid_page_format(page);
    uint32_t          data_size = mkt_centroid_data_size(dim, fmt);
    return (const void *)(PageGetSpecialPointer(page)
                          - (size_t)(index + 1) * data_size);
}

/* Typed data accessors */
static inline const RaBitQData *
mkt_centroid_data(const Page page, uint32_t index, Dimension dim) {
    return (const RaBitQData *)mkt_centroid_entry_data(page, index, dim);
}
static inline const float *
mkt_centroid_float_data(const Page page, uint32_t index,
                        Dimension dim) {
    return (const float *)mkt_centroid_entry_data(page, index, dim);
}
static inline const half *
mkt_centroid_half_data(const Page page, uint32_t index,
                       Dimension dim) {
    return (const half *)mkt_centroid_entry_data(page, index, dim);
}
```

**Page operations:**

```c
/* Format-aware initialization (stores format in opaque flags) */
void mkt_centroid_page_init_fmt(Page page, uint8_t level,
                                MktCentroidFormat fmt);

/* Backward-compatible wrapper (defaults to RaBitQ) */
static inline void
mkt_centroid_page_init(Page page, uint8_t level) {
    mkt_centroid_page_init_fmt(page, level, MKT_CENTROID_FMT_RABITQ);
}

/* Generic add: medoid_tid only used for RaBitQ pages (NULL for
 * float/half); data points to RaBitQData, float[], or half[]
 * depending on page format */
bool mkt_centroid_page_add_entry(Page page, Dimension dim,
                                 BlockNumber child_blkno,
                                 uint16_t child_count,
                                 uint16_t flags,
                                 const ItemPointerData *medoid_tid,
                                 const void *data);
```

**I/O abstraction** (`MktStorage`):

All page and vector I/O is abstracted behind a `MktStorageOps` vtable
(`src/index/storage.h`). `MktStorage` is a base struct containing an `ops`
pointer; implementations embed it as their first member and add
implementation-specific fields. WAL logging and durability are internal
to each implementation — callers just see read/release/write/commit:

```c
typedef struct MktStorageOps
{
    Page (*read_page)(MktStorage *self, BlockNumber blkno);
    void (*release_page)(MktStorage *self, BlockNumber blkno);
    Page (*write_page)(MktStorage *self, BlockNumber blkno);
    Page (*new_page)(MktStorage *self, BlockNumber *blkno_out);
    void (*commit_page)(MktStorage *self, BlockNumber blkno);
    Vec32Ref (*fetch_vec)(MktStorage *self, ItemPointerData tid);

    /* Rerank candidates with exact distances (NULL = not supported).
     * Fetches full-precision vectors for candidates whose error > 0,
     * computes exact L2, returns sorted top-keep results. Used by
     * centroid beam search to replace approximate RaBitQ estimates. */
    uint32_t (*rerank)(MktStorage *self, Datum query, Dimension dim,
                       const ItemPointerData *tids, uint32_t count,
                       const Distance *distances,
                       const Distance *errors, uint32_t keep,
                       uint32_t *out_indices,
                       Distance *out_distances);
} MktStorageOps;

struct MktStorage
{
    const MktStorageOps *ops;
};
```

Implementations embed `MktStorage` as first member (C inheritance via
upcast). In standalone mode, the struct wraps an array of `malloc`'d 8KB
buffers. In PG mode, it wraps a `Relation` with buffer cache calls.

Static inline wrapper functions hide the vtable dispatch:

```c
// Clean API — no s->ops->fn(s, ...) at call sites
Page page = mkt_storage_read_page(storage, centroid_blkno);
MktCentroidPageOpaque *opaque = MKT_CENTROID_OPAQUE(page);
// ... per-entry distance on in-page RaBitQData ...
mkt_storage_release_page(storage, centroid_blkno);
```

Hot centroid pages stay cached in shared_buffers. For billion-scale indexes,
the ~96MB centroid tree fits comfortably in a typical shared_buffers setting.

### 6.6 Scan State

```c
// Scan state (stored in IndexScanDesc->opaque)
typedef struct MktScanOpaque {
    // Centroid search state (backend-local, populated from centroid pages)
    CentroidSearchState *centroid_state;
    RaBitQParams        *rabitq_params;

    // Query parameters
    Vec32Ref       query;
    SearchParams    params;
    bool            first_call;

    // Results
    TopKHeap       *results;
    uint32_t        result_index;
    TopKEntry      *sorted_results;

    // Buffer management for posting list scan
    BufferReadState buffer_state;
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

    // Startup cost: centroid tree traversal (typically cached in shared buffers)
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
    BlockNumber block;
    OffsetNumber offset;
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

### 6.9 Async I/O Integration (PostgreSQL 18+)

**Files**: `src/pg/search/stream.h`, `src/pg/search/stream.c`, `src/pg/build/stream.c`

PostgreSQL 18 introduces a powerful async I/O subsystem with the read stream API.
Meerkat leverages this for efficient prefetching during index scans, posting list
traversal, and reranking operations.

#### I/O Method Support

PostgreSQL 18 supports multiple I/O backends via the `io_method` GUC:

| Method     | Description                                    | Availability       |
|------------|------------------------------------------------|--------------------|
| `sync`     | Traditional synchronous I/O                    | All platforms      |
| `worker`   | Dedicated I/O worker processes (default)       | All platforms      |
| `io_uring` | Linux kernel async I/O via liburing            | Linux 5.1+ w/liburing |

Meerkat works with all three methods. The `io_uring` method provides lowest latency
for high-concurrency workloads. Configuration:

```sql
-- PostgreSQL 18+ settings
SET io_method = 'io_uring';              -- Or 'worker' (default), 'sync'
SET effective_io_concurrency = 200;       -- For user queries
SET maintenance_io_concurrency = 10;      -- For index builds, VACUUM
```

#### Why Queue Depth Matters for NVMe

NVMe drives achieve high IOPS (400K+) only when multiple I/O requests are in
flight simultaneously. The number of concurrent requests is called **queue depth
(QD)**:

| Access Pattern | Queue Depth | Latency per 8KB | Throughput |
|----------------|-------------|-----------------|------------|
| Serial sync I/O | QD=1 | 100 μs | ~80 MB/s |
| Async I/O | QD=32+ | 100 μs total | 3+ GB/s |

For posting list scans reading 275 pages:

- **Sync I/O (QD=1)**: 275 × 100 μs = **28 ms** — each read waits for completion
- **Async I/O (QD=275)**: 275 / 400K IOPS = **0.7 ms** — all reads in parallel

The read stream API and `io_uring` enable high queue depth by submitting many
read requests in a single syscall, then processing completions as they arrive.
This is why `effective_io_concurrency = 200` is recommended for Meerkat workloads.

#### Read Stream API Overview

The read stream API provides adaptive prefetching with automatic I/O combining:

```c
#include <storage/read_stream.h>

// Callback type - returns next block to read, InvalidBlockNumber when done
typedef BlockNumber (*ReadStreamBlockNumberCB)(
    ReadStream *stream,
    void *callback_private_data,
    void *per_buffer_data);

// Create a read stream
ReadStream *read_stream_begin_relation(
    int flags,                          // READ_STREAM_* flags
    BufferAccessStrategy strategy,      // Buffer ring strategy
    Relation rel,
    ForkNumber forknum,
    ReadStreamBlockNumberCB callback,
    void *callback_private_data,
    size_t per_buffer_data_size);       // Per-buffer state passed to consumer

// Get next buffer (blocks until available)
Buffer read_stream_next_buffer(ReadStream *stream, void **per_buffer_data);

// Reset for re-scanning
void read_stream_reset(ReadStream *stream);

// Cleanup
void read_stream_end(ReadStream *stream);
```

**Read Stream Flags:**

| Flag                        | Purpose                                      |
|-----------------------------|----------------------------------------------|
| `READ_STREAM_DEFAULT`       | General-purpose, random access patterns      |
| `READ_STREAM_MAINTENANCE`   | Use `maintenance_io_concurrency` setting     |
| `READ_STREAM_SEQUENTIAL`    | Disable explicit prefetch advice (kernel handles) |
| `READ_STREAM_FULL`          | Scanning entire structure, skip ramp-up      |
| `READ_STREAM_USE_BATCHING`  | Enable AIO batch mode (callback must be lock-free) |

#### Use Case 1: Posting List Scan

Scanning posting list pages benefits from prefetching since pages are linked:

```c
// Callback state for posting list traversal
typedef struct PostingListStreamState
{
    BlockNumber     next_block;      // Next block to read
    ClusterId       cluster_id;      // Which cluster we're scanning
    int             pages_read;      // For statistics
} PostingListStreamState;

// Callback provides next posting list page
static BlockNumber
posting_list_stream_next(ReadStream *stream, void *callback_private,
                        void *per_buffer_data)
{
    PostingListStreamState *state = callback_private;

    if (state->next_block == InvalidBlockNumber)
        return InvalidBlockNumber;

    BlockNumber current = state->next_block;

    // Look ahead: read the next-page pointer from cached page if available
    // This allows the stream to prefetch the next page before we need it
    // Note: actual next_block update happens in scan loop after reading page

    state->pages_read++;
    return current;
}

// Initialize stream for posting list scan
ReadStream *
mkt_posting_list_stream_begin(Relation index, ClusterId cluster,
                              BufferAccessStrategy strategy)
{
    PostingListStreamState *state = palloc(sizeof(PostingListStreamState));

    // Get head block from metapage (already cached in centroid cache)
    state->next_block = mkt_meta_get_head(index, cluster);
    state->cluster_id = cluster;
    state->pages_read = 0;

    return read_stream_begin_relation(
        READ_STREAM_DEFAULT,            // Random access pattern
        strategy,
        index,
        MAIN_FORKNUM,
        posting_list_stream_next,
        state,
        0);                             // No per-buffer data needed
}
```

#### Use Case 2: Multi-Cluster Scan (nprobe > 1)

When scanning multiple clusters, interleave reads for better I/O utilization:

```c
typedef struct MultiClusterStreamState
{
    ClusterId      *clusters;         // Clusters to scan (sorted by distance)
    int             nprobe;           // Number of clusters
    int             current_cluster;  // Index into clusters array
    BlockNumber    *heads;            // Head block for each cluster
    BlockNumber    *current_blocks;   // Current block in each cluster
} MultiClusterStreamState;

static BlockNumber
multi_cluster_stream_next(ReadStream *stream, void *callback_private,
                         void *per_buffer_data)
{
    MultiClusterStreamState *state = callback_private;
    ClusterId *cluster_out = per_buffer_data;  // Tell consumer which cluster

    // Round-robin across clusters to interleave I/O
    for (int attempts = 0; attempts < state->nprobe; attempts++)
    {
        int idx = state->current_cluster;
        state->current_cluster = (idx + 1) % state->nprobe;

        BlockNumber blk = state->current_blocks[idx];
        if (blk != InvalidBlockNumber)
        {
            *cluster_out = state->clusters[idx];
            return blk;
        }
    }

    return InvalidBlockNumber;  // All clusters exhausted
}

// After reading each page, update current_blocks[cluster] from page header
```

#### Use Case 3: Heap Reranking (Bitmap Heap Scan Pattern)

Reranking fetches full-precision vectors from heap pages for the top candidates
from the quantized search phase. This follows PostgreSQL's **bitmap heap scan**
pattern: collect TIDs, sort by block number, then scan pages sequentially.

**Why block-sorted access matters:**

- Random TID order → random I/O → ~100-200 IOPS on SSD
- Block-sorted order → sequential I/O → 100K+ IOPS on SSD
- Multiple TIDs per page are processed together (single page read)
- Read stream prefetches upcoming blocks while processing current

```
Unsorted TIDs:        Block-sorted TIDs:
  (5, 3)                (1, 7)
  (1, 7)    ──sort──►   (1, 12)
  (3, 2)                (3, 2)
  (1, 12)               (5, 3)
  (3, 8)                (3, 8)

Pages read: 5,1,3,1,3   Pages read: 1,3,5  (3 pages vs 5 random accesses)
```

**Implementation:**

```c
// Rerank state: TIDs sorted by (block, offset) for sequential access
typedef struct RerankStreamState
{
    ItemPointer     tids;             // TIDs sorted by block number
    int             ntids;            // Total TIDs to fetch
    int             current;          // Current position in tids array
} RerankStreamState;

// Per-buffer data passed to consumer: which TIDs are on this page
typedef struct RerankPageInfo
{
    int             first_tid_idx;    // Index of first TID on this page
    int             ntids_on_page;    // Count of TIDs to process on this page
} RerankPageInfo;

// Callback: return next unique block, skip duplicates
static BlockNumber
rerank_stream_next(ReadStream *stream, void *callback_private,
                  void *per_buffer_data)
{
    RerankStreamState *state = callback_private;
    RerankPageInfo *info = per_buffer_data;

    if (state->current >= state->ntids)
        return InvalidBlockNumber;

    BlockNumber blk = ItemPointerGetBlockNumber(&state->tids[state->current]);

    // Record which TIDs are on this page (they're contiguous after sorting)
    info->first_tid_idx = state->current;
    info->ntids_on_page = 1;

    // Count consecutive TIDs on same block
    while (state->current + info->ntids_on_page < state->ntids)
    {
        BlockNumber next_blk = ItemPointerGetBlockNumber(
            &state->tids[state->current + info->ntids_on_page]);
        if (next_blk != blk)
            break;
        info->ntids_on_page++;
    }

    state->current += info->ntids_on_page;
    return blk;
}

// Compare function for qsort: order by (block, offset)
static int
tid_block_offset_cmp(const void *a, const void *b)
{
    const ItemPointer ta = (const ItemPointer) a;
    const ItemPointer tb = (const ItemPointer) b;

    BlockNumber ba = ItemPointerGetBlockNumber(ta);
    BlockNumber bb = ItemPointerGetBlockNumber(tb);
    if (ba != bb)
        return (ba < bb) ? -1 : 1;

    OffsetNumber oa = ItemPointerGetOffsetNumber(ta);
    OffsetNumber ob = ItemPointerGetOffsetNumber(tb);
    if (oa != ob)
        return (oa < ob) ? -1 : 1;

    return 0;
}
```

**Stream initialization and usage:**

```c
// Prepare TIDs for sequential heap access
static void
mkt_rerank_sort_tids(ItemPointer tids, int ntids)
{
    qsort(tids, ntids, sizeof(ItemPointerData), tid_block_offset_cmp);
}

// Create read stream for reranking
ReadStream *
mkt_rerank_stream_begin(Relation heap, ItemPointer tids, int ntids,
                       BufferAccessStrategy strategy)
{
    // Sort TIDs by block for sequential I/O
    mkt_rerank_sort_tids(tids, ntids);

    RerankStreamState *state = palloc(sizeof(RerankStreamState));
    state->tids = tids;
    state->ntids = ntids;
    state->current = 0;

    // READ_STREAM_DEFAULT: random access pattern (blocks aren't contiguous)
    // READ_STREAM_USE_BATCHING: callback is lock-free (just array traversal)
    return read_stream_begin_relation(
        READ_STREAM_DEFAULT | READ_STREAM_USE_BATCHING,
        strategy,
        heap,
        MAIN_FORKNUM,
        rerank_stream_next,
        state,
        sizeof(RerankPageInfo));    // Per-buffer data for TID ranges
}

// Process reranking
void
mkt_rerank_execute(ReadStream *stream, RerankStreamState *state,
                  Relation heap, TopKCollector *results)
{
    Buffer buf;
    RerankPageInfo *info;

    while ((buf = read_stream_next_buffer(stream, (void **) &info)) != InvalidBuffer)
    {
        Page page = BufferGetPage(buf);

        // Process all TIDs on this page
        for (int i = 0; i < info->ntids_on_page; i++)
        {
            int tid_idx = info->first_tid_idx + i;
            ItemPointer tid = &state->tids[tid_idx];
            OffsetNumber off = ItemPointerGetOffsetNumber(tid);

            // Get tuple, extract vector, compute exact distance
            ItemId itemid = PageGetItemId(page, off);
            HeapTupleHeader htup = (HeapTupleHeader) PageGetItem(page, itemid);
            // ... extract vector from tuple, compute distance, update results
        }

        ReleaseBuffer(buf);
    }

    read_stream_end(stream);
}
```

#### Use Case 4: Index Build (Streaming Heap Scan)

During index build, scan heap pages with prefetching:

```c
ReadStream *
mkt_build_heap_stream(Relation heap, Snapshot snapshot,
                     BufferAccessStrategy strategy)
{
    BlockRangeReadStreamPrivate *state = palloc(sizeof(*state));
    state->current_blocknum = 0;
    state->last_exclusive = RelationGetNumberOfBlocks(heap);

    return read_stream_begin_relation(
        READ_STREAM_MAINTENANCE |        // Use maintenance_io_concurrency
        READ_STREAM_FULL |               // Scanning entire heap
        READ_STREAM_SEQUENTIAL |         // Sequential access pattern
        READ_STREAM_USE_BATCHING,        // Callback is lock-free
        strategy,
        heap,
        MAIN_FORKNUM,
        block_range_read_stream_cb,      // Built-in callback for ranges
        state,
        0);
}
```

#### Batching Mode Restrictions

When using `READ_STREAM_USE_BATCHING`, the callback must not:

1. Block on I/O without calling `pgaio_submit_staged()` first
2. Hold locks that might be held during I/O waits
3. Start a nested batch

Safe patterns for batching:
- Simple arithmetic (block range iteration)
- Lock-free data structure traversal
- Reading from already-pinned buffers

If the callback needs locks, omit `READ_STREAM_USE_BATCHING`:

```c
// Safe: no locks in callback
read_stream_begin_relation(READ_STREAM_DEFAULT | READ_STREAM_USE_BATCHING, ...);

// Callback takes locks - no batching
read_stream_begin_relation(READ_STREAM_DEFAULT, ...);
```

#### Custom Async I/O Layer (Future)

The PostgreSQL read stream API works well for page-oriented access but may be
limiting for some Meerkat use cases:

**Potential limitations:**
- Callback model requires knowing next block before current completes
- No direct support for non-page I/O (e.g., reading raw vector data from files)
- Buffer pool integration assumes PostgreSQL page semantics
- Limited control over I/O prioritization across multiple streams

**Future consideration:** A Meerkat-specific async I/O layer for cases like:
- Direct file I/O for external vector storage
- Custom prefetch patterns for centroid search
- Tiered storage with different I/O characteristics
- Integration with user-space NVMe drivers

If needed, this layer would:
- Use `io_uring` directly on Linux (bypass PG's abstraction)
- Fall back to `libaio` or worker threads on other platforms
- Integrate with PG's buffer manager for hybrid access patterns

```c
// Hypothetical custom async I/O interface
typedef struct MktAsyncIO MktAsyncIO;

MktAsyncIO *mkt_aio_create(int max_concurrent);
void        mkt_aio_submit_read(MktAsyncIO *aio, int fd, off_t offset,
                                void *buf, size_t len, void *user_data);
int         mkt_aio_poll(MktAsyncIO *aio, MktAIOCompletion *completions,
                         int max_completions, int timeout_ms);
void        mkt_aio_destroy(MktAsyncIO *aio);
```

This would live in `src/core/` (standalone) with platform-specific implementations,
used alongside PG's read stream for buffer-managed pages.

#### SPDK Consideration

For maximum I/O performance, we may consider [SPDK (Storage Performance
Development Kit)](https://spdk.io/) — a userspace NVMe driver that bypasses
the kernel entirely. SPFresh uses SPDK to achieve their benchmark results.

**SPDK advantages:**
- Eliminates kernel overhead (syscalls, context switches, interrupts)
- Polled-mode completion (no interrupt latency)
- Direct NVMe queue access with minimal latency (~10-20 μs vs 100 μs)
- Can sustain millions of IOPS from a single core

**SPDK tradeoffs:**
- Requires dedicated NVMe device (can't share with OS/PostgreSQL)
- Must run as root or with special permissions
- Application manages all memory (no page cache)
- Significant complexity increase
- Not integrated with PostgreSQL buffer manager

**Potential use case:** If Meerkat stores posting lists or full-precision vectors
in a separate file (outside PostgreSQL's heap), SPDK could provide a dedicated
high-performance path:

```
┌─────────────────────────────────────────────────────────────────┐
│                     PostgreSQL Process                          │
├─────────────────────────────────────────────────────────────────┤
│  Centroid pages     │  Posting lists    │  Full vectors         │
│  (buffer cache)     │  (buffer cache)   │  (SPDK direct I/O)    │
│  io_uring           │  io_uring         │  userspace NVMe       │
└─────────────────────┴───────────────────┴───────────────────────┘
```

This hybrid approach would use PostgreSQL's buffer cache for metadata and
quantized data (which benefits from caching), while using SPDK for
high-throughput sequential access to full-precision vectors during reranking.

**Decision:** Start with PostgreSQL's native `io_uring` support. Consider SPDK
only if profiling shows kernel I/O overhead is a significant bottleneck (>10%
of query latency) and the operational complexity is acceptable.

---

## Part 7: Maintenance Operations

Meerkat uses the LIRE protocol (from SPFresh) for dynamic updates. This enables
high insert throughput without degrading query performance.

### 7.1 Insert

**Basic insert flow:**

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

    Vector *vec = DatumGetVector(values[0]);

    // 1. Find nearest centroid by traversing centroid pages
    ClusterId cluster;
    Vec32Ref centroid;
    mkt_find_nearest_centroid(index, VectorToRef(vec), &cluster, &centroid);

    // 2. Quantize vector using RaBitQ (relative to centroid)
    RaBitQData *quantized = palloc(MKT_RABITQ_DATA_SIZE(vec->dim));
    RaBitQParams *params = mkt_get_rabitq_params(index);
    mkt_rabitq_encode_into(params, VectorToRef(vec), centroid, quantized);

    // 3. Insert into posting list (may trigger split)
    BlockNumber head = mkt_meta_get_head(index, cluster);
    InsertResult result = mkt_posting_insert(index, head, heap_tid, quantized);

    // 4. Handle LIRE operations if needed
    if (result.needs_split) {
        mkt_lire_split(index, cluster);
    }

    pfree(quantized);
    return true;
}
```

### 7.2 LIRE Protocol

The LIRE protocol maintains index quality during updates:

**Split**: When a posting list exceeds `2× target_size`:
1. Compute new centroid for subset (~40% of vectors)
2. Reassign vectors to original or new cluster
3. Update centroid pages
4. Create new posting list for split cluster

**Merge**: When a posting list falls below `0.25× target_size`:
1. Find nearest neighbor cluster
2. Move all vectors to neighbor
3. Remove centroid from centroid pages (update tree)
4. Reclaim posting list pages

**Reassign**: Periodically check if vectors should move to adjacent clusters:
1. For each vector, check distance to current vs neighbor centroids
2. If closer to neighbor, move vector
3. Helps maintain quality as centroids drift

**Thresholds** (configurable via reloptions):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `split_threshold` | 2× target | Trigger split when exceeded |
| `merge_threshold` | 0.25× target | Trigger merge when below |
| `reassign_fraction` | 0.1 | Fraction of vectors to check per vacuum |

**Cascade bounds** (from SPFresh measurements):
- Only ~0.4% of insertions trigger rebalancing
- Average cascade: 3 operations
- Maximum observed: 160 splits
- Convergence guaranteed (finite operations per insert)

### 7.3 Vacuum

Vacuum performs deletion cleanup and LIRE maintenance:

```c
static IndexBulkDeleteResult *mkt_ambulkdelete(
    IndexVacuumInfo *info,
    IndexBulkDeleteResult *stats,
    IndexBulkDeleteCallback callback,
    void *callback_state
) {
    // Scan all posting list pages
    // Mark entries as deleted (soft delete) if callback returns true

    return stats;
}

static IndexBulkDeleteResult *mkt_amvacuumcleanup(
    IndexVacuumInfo *info,
    IndexBulkDeleteResult *stats
) {
    // 1. Compact pages: remove soft-deleted entries
    // 2. Check for underfull posting lists → LIRE merge
    // 3. Run reassignment on sample of vectors
    // 4. Update FSM with free space

    return stats;
}
```

**Vacuum phases:**

1. **Bulk delete**: Mark entries matching callback as deleted
2. **Compact**: Remove deleted entries, reclaim page space
3. **LIRE merge**: Merge underfull clusters into neighbors
4. **Reassign**: Move misplaced vectors to correct clusters
5. **Defragment**: Optionally reorganize posting lists for contiguity

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
    Vec32Ref           query;
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
    Vec32Ref query,
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
ORDER BY embedding <-> '[...]'::vec32
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
mkt_centroid_search_iterator(CentroidSearchState *cache, Vec32Ref query);
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
- `mkt bench distance`: Distance computation benchmark
- Unit test suite for distance functions

### Phase 2: Core Algorithms

**Deliverables**:
1. RaBitQ quantization
2. Top-K heap
3. K-means clustering
4. Unit tests and CLI tools

**Test artifacts**:
- `mkt_rabitq_test`: RaBitQ encoding/distance accuracy tests
- `mkt bench cluster`: Clustering benchmark
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
- `mkt build`: Build index from vector file
- Build correctness tests

### Phase 5: Search (Standalone)

**Deliverables**:
1. Centroid search
2. Posting list scan
3. Re-ranking (optional in standalone mode)
4. Full search pipeline

**Test artifacts**:
- `mkt search`: Search standalone index
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
1. LIRE protocol (split, merge, reassign)
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

Code is split into **standalone** modules (no PostgreSQL dependency) and
**PostgreSQL-specific** code. This separation enables unit testing and
benchmarking of core algorithms without PostgreSQL.

```
meerkat/
├── src/
│   │
│   │ # ════════════════════════════════════════════════════════════
│   │ # STANDALONE (no PostgreSQL headers or libraries required)
│   │ # ════════════════════════════════════════════════════════════
│   │
│   ├── core/                     # Foundation: types, memory, platform
│   │   ├── types.h               # Dimension, Distance, Vec32Ref, etc.
│   │   ├── memory.h              # Memory abstraction interface
│   │   ├── memory_standalone.h   # Arena allocator declarations
│   │   ├── memory_standalone.c   # Arena allocator implementation
│   │   ├── pg_compat.h           # Standalone PG type shims (BlockNumber,
│   │   │                         #   ItemPointerData, Page, BLCKSZ)
│   │   ├── platform.h            # SIMD detection, prefetch, alignment
│   │   └── platform.c
│   │
│   ├── algo/                     # Core algorithms (SIMD-optimized)
│   │   ├── distance.h            # Distance computation interface
│   │   ├── distance.c            # Dispatch + scalar fallback
│   │   ├── distance_avx512.c     # AVX-512 implementation
│   │   ├── distance_avx2.c       # AVX2 implementation
│   │   ├── distance_neon.c       # ARM NEON implementation
│   │   ├── topk.h                # Top-K selection interface
│   │   ├── topk.c                # Heap-based implementation
│   │   ├── kmeans.h              # K-means clustering interface
│   │   └── kmeans.c              # Lloyd's algorithm + k-means++
│   │
│   ├── quant/                    # Vector quantization
│   │   ├── rabitq.h              # RaBitQ interface
│   │   ├── rabitq.c              # RaBitQ encoding/distance
│   │   ├── rabitq_avx512.c       # SIMD-optimized RaBitQ
│   │   ├── rabitq_avx2.c
│   │   └── rabitq_neon.c
│   │
│   ├── index/                    # Index structures (standalone, no PG)
│   │   ├── storage.h             # MktStorage I/O vtable + wrappers
│   │   ├── centroid_page.h       # Centroid page layout, bidirectional AoS
│   │   ├── centroid_page.c       # Page init, add entry
│   │   ├── centroid_search.h     # Beam search API, search state
│   │   └── centroid_search.c     # Level-by-level beam search
│   │
│   ├── cli/                      # Command-line tool ('mkt' binary)
│   │   ├── main.c                # Entry point, subcommand dispatch
│   │   ├── cmd_distance.c        # mkt distance - test/benchmark
│   │   ├── cmd_quantize.c        # mkt quantize - test RaBitQ
│   │   ├── cmd_cluster.c         # mkt cluster - test k-means
│   │   └── cmd_bench.c           # mkt bench - full algorithm benchmarks
│   │
│   │ # ════════════════════════════════════════════════════════════
│   │ # POSTGRESQL (requires PostgreSQL headers and libraries)
│   │ # ════════════════════════════════════════════════════════════
│   │
│   └── pg/                       # PostgreSQL extension
│       ├── meerkat.h             # Extension public header
│       ├── meerkat.c             # Extension entry point, GUCs
│       ├── memory_pg.h           # palloc/pfree wrappers
│       │
│       ├── index/                # Index data structures (page-based)
│       │   ├── posting.h         # Posting list entry format
│       │   ├── posting.c
│       │   ├── page.h            # Page layout (8KB PostgreSQL pages)
│       │   ├── page.c
│       │   ├── meta.h            # Metapage structure
│       │   └── meta.c
│       │
│       ├── build/                # Index construction
│       │   ├── build.h           # Build interface
│       │   ├── build.c           # Build pipeline, state machine
│       │   ├── sample.c          # Vector sampling for clustering
│       │   └── write.c           # Page writing, WAL logging
│       │
│       ├── search/               # Search operations
│       │   ├── search.h          # Search interface
│       │   ├── search.c          # Search pipeline coordination
│       │   ├── centroid.c        # Centroid tree traversal
│       │   ├── scan.c            # Posting list scanning
│       │   └── rerank.c          # Full-precision reranking
│       │
│       ├── iam/                  # Index Access Method callbacks
│       │   ├── handler.c         # amhandler registration
│       │   ├── build.c           # ambuild, ambuildempty
│       │   ├── insert.c          # aminsert (with LIRE split)
│       │   ├── scan.c            # ambeginscan, amgettuple, amendscan
│       │   └── vacuum.c          # ambulkdelete, amvacuumcleanup (with LIRE merge)
│       │
│       └── lire/                 # LIRE protocol for dynamic updates
│           ├── lire.h            # LIRE interface
│           ├── split.c           # Cluster splitting
│           ├── merge.c           # Cluster merging
│           └── reassign.c        # Vector reassignment
│
├── sql/
│   ├── meerkat--1.0.sql          # Extension SQL definitions
│   └── meerkat.control           # Extension control file
│
├── test/
│   ├── unit/                     # Unit tests (standalone, no PG)
│   │   ├── test_distance.c
│   │   ├── test_rabitq.c
│   │   ├── test_topk.c
│   │   ├── test_kmeans.c
│   │   ├── test_centroid_page.c  # Centroid page layout tests
│   │   ├── test_centroid_search.c # Beam search tests
│   │   └── meson.build
│   ├── regress/                  # PostgreSQL regression tests
│   │   ├── sql/
│   │   └── expected/
│   └── bench/                    # Algorithm benchmarks
│       ├── bench_distance.c
│       ├── bench_rabitq.c
│       └── datasets/             # Test datasets (gitignored)
│
├── scripts/
│   └── ci/
│       ├── build.sh
│       ├── coverage.sh
│       ├── lint.sh
│       └── sanitizers.sh
│
├── docs/
│   ├── architecture.md
│   └── implementation.md
│
├── .github/workflows/
├── meson.build
├── meson_options.txt
└── CLAUDE.md
```

### Module Dependencies

```
                        ┌───────────────────────────────────┐
                        │              src/pg/              │
                        │  (PostgreSQL extension, all of:   │
                        │   build/, search/, iam/, lire/)   │
                        └─────────────────┬─────────────────┘
                                          │
    ┌──────────────────┬──────────────────┼────────────────┐
    │                  │                  │                │
    ▼                  ▼                  ▼                ▼
┌────────┐      ┌───────────┐     ┌───────────┐    ┌───────────┐
│  cli/  │      │  index/   │     │   quant/  │    │   algo/   │
│ (mkt)  │─────►│ centroid  │────►│  rabitq   │───►│ distance  │
└────────┘      │  page +   │     └───────────┘    │   topk    │
                │  search   │                      │  kmeans   │
                └───────────┘                      └─────┬─────┘
                                                          │
                                                          ▼
                                                    ┌───────────┐
                                                    │   core/   │
                                                    │   types   │
                                                    │  memory   │
                                                    │ platform  │
                                                    └───────────┘
```

**Standalone modules** (no PostgreSQL):

- **core/**: Foundation. No dependencies.
- **algo/**: Algorithms. Depends on core/.
- **quant/**: Quantization. Depends on core/, algo/.
- **cli/**: Command-line tool. Depends on quant/, algo/, core/.

**PostgreSQL modules** (require PG headers/libs):

- **pg/**: Everything under src/pg/ requires PostgreSQL. Contains index
  structures (page layout, posting lists), build pipeline, search operations,
  IAM callbacks, and shared memory cache.

### Include Conventions

```c
// From within src/algo/distance.c:
#include "core/types.h"      // Relative to src/
#include "core/platform.h"

// From within src/pg/build/build.c:
#include <postgres.h>        // PostgreSQL system headers first
#include <access/reloptions.h>

#include "core/types.h"      // Then local headers
#include "algo/distance.h"
#include "algo/kmeans.h"
#include "quant/rabitq.h"
#include "pg/index/posting.h"

// From within src/pg/iam/handler.c:
#include <postgres.h>
#include <fmgr.h>

#include "core/types.h"
#include "pg/search/search.h"
```

### Meson Build Structure

```meson
# src/meson.build
subdir('core')
subdir('algo')
subdir('quant')
subdir('cli')

# Core library (standalone, no PostgreSQL)
mkt_core_lib = static_library(
  'mkt_core',
  core_sources + algo_sources + quant_sources,
  include_directories: src_inc,
)

mkt_core_dep = declare_dependency(
  link_with: mkt_core_lib,
  include_directories: src_inc,
)

# CLI binary (standalone)
mkt_exe = executable(
  'mkt',
  cli_sources,
  dependencies: mkt_core_dep,
  install: true,
)

# PostgreSQL extension (links against core)
if pg_config.found()
  subdir('pg')
endif
```

The `pg/` subdirectory has its own structure for PostgreSQL-dependent code:

```meson
# src/pg/meson.build
subdir('index')   # Page layouts, posting lists
subdir('build')   # Index build pipeline
subdir('search')  # Search operations
subdir('iam')     # Index Access Method callbacks
subdir('lire')    # LIRE protocol (split, merge, reassign)

pg_sources = (
  pg_index_sources +
  pg_build_sources +
  pg_search_sources +
  pg_iam_sources +
  pg_lire_sources
)

shared_module(
  'meerkat',
  pg_sources,
  dependencies: [mkt_core_dep, pg_dep],
  install: true,
  install_dir: pg_pkglibdir,
)
```

Each module subdirectory has its own `meson.build` that defines its sources:

```meson
# src/algo/meson.build (standalone)
algo_sources = files(
  'distance.c',
  'distance_avx512.c',
  'distance_avx2.c',
  'distance_neon.c',
  'topk.c',
  'kmeans.c',
)

# src/pg/search/meson.build (PostgreSQL-dependent)
pg_search_sources = files(
  'search.c',
  'recheck.c',
  'scan.c',
)
```

---

## References

- [PostgreSQL Index Access Method Documentation](https://www.postgresql.org/docs/current/indexam.html)
- [pgvector source](https://github.com/pgvector/pgvector)
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)
- [ARM NEON Intrinsics Reference](https://developer.arm.com/architectures/instruction-sets/intrinsics/)
