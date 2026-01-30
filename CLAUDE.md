# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Project Overview

Meerkat is a PostgreSQL index access method (IAM) for Approximate Nearest
Neighbor (ANN) vector search, inspired by Google's ScaNN for AlloyDB and
Microsoft's SPANN. It uses the vector format from
[pgvector](https://github.com/pgvector/pgvector).

### Architecture

The index uses postings lists to partition the vector space into clusters
based on centroids. A shallow tree structure enables fast searches across
a subset of partitions to build the nearest neighbor result set.

### Design Goals

- Native PostgreSQL integration using the IAM API
- High query performance with SIMD optimization (AVX512/NEON)
- Use PostgreSQL's shared buffer cache and on-disk page system
- Separate in-memory cache for centroid search (not relying on buffer cache,
  which deals with raw pages)
- Support for MVCC and streaming replication
- Reasonable, stable, and tunable memory consumption

### Documentation

- `docs/architecture.md` - High-level architecture
- `docs/implementation.md` - Detailed implementation specifications
- `docs/development.md` - Build instructions, testing, CI scripts

### Performance Goals

- Tiered storage: fast memory (CPU cache, RAM) for upper tree levels, SSD/cloud
  for lower levels
- Vector quantization for in-memory ANN with full-precision fallback (~1% of
  cases)
- Sequential disk layout for vectors close in search space
- Bulk load operations should read sequential data without random disk seeks
- High recall rates using state-of-the-art techniques
- Fast index builds via linear scans, SIMD, and efficient clustering
- Support high ingest rates without sacrificing query performance

## References

Research and inspiration:

- [ScaNN reference implementation](https://github.com/google-research/google-research/tree/master/scann) (in-memory)
- [SPANN paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2021/11/SPANN_finalversion1.pdf)
- [SPTAG](https://github.com/microsoft/SPTAG) (ANN library used by SPANN for in-memory centroid index)
- [ScaNN for AlloyDB blog](https://cloud.google.com/blog/products/databases/understanding-the-scann-index-in-alloydb)
- [ScaNN for AlloyDB whitepaper](https://services.google.com/fh/files/misc/scann_for_alloydb_whitepaper.pdf)
- [turbopuffer ANN v3](https://turbopuffer.com/blog/ann-v3)
- [SPFresh paper](https://dl.acm.org/doi/epdf/10.1145/3600006.3613166)

Related PostgreSQL extensions:

- [pgvector](https://github.com/pgvector/pgvector)
- [pgvectorscale](https://github.com/timescale/pgvectorscale)
- [pg_textsearch](https://github.com/timescale/pg_textsearch)

## Implementation

The extension is implemented in C (standard 23) using meson as the build
system. It may later be integrated into `pgvectorscale` for distribution.

### Development Approach

- Iterative development with frequent functional releases
- Well-tested changes using unit tests, PostgreSQL regression tests, and
  isolation tests
- TAP tests for multi-instance scenarios (e.g., streaming replication)
- **Step-by-step with review**: Work in small increments and pause for review
  after each step. Don't implement multiple components at once unless explicitly
  asked to do larger work unattended.
- **CI scripts**: Put larger CI workflow jobs in standalone bash scripts in
  `scripts/ci/`. Scripts should be runnable locally for testing and debugging.
  GitHub Actions workflows should call these scripts rather than inline commands.

### Testability

To enable unit testing, the code uses an abstraction layer decoupled from
PostgreSQL internals:

- Memory allocation macros supporting both `palloc`/`pfree` and `malloc`/`free`
- Abstraction for memory contexts (dummy contexts in standalone mode)
- Minimal use of PostgreSQL internal data structures, with mocks where needed

Note: Some components (buffer cache, IAM handlers) are too integrated with
PostgreSQL for standalone unit testing.

Unit tests link against a standalone static library built from the testable
subset of the code.

## Local Development

### Related Project Checkouts

- `../pgvector/`
- `../pgvectorscale/`
- `../pg_textsearch/`
- `../google-research/scann/`

### PostgreSQL Management

If available, use pgmanager (`pgm`) at `../pgmanager/` to manage PostgreSQL:

- Source: `../pg/src/`
- Builds: `../pg/usr/`
- Runtime: `../pg/run/`

The extension builds against PostgreSQL source managed by `pgm`.

If `pgm` is not available, build against the system PostgreSQL installation
(meson detects this automatically).

### Debugging

To debug a running PostgreSQL instance attach a debugger. Important: attach to the
**backend process**, not psql (client) or postmaster (supervisor).

For crashes:

- Use a debug build to get symbol names
- On Linux, use `coredumpctl debug` to debug the latest crash
- Always check the stacktrace first before adding debug statements

The `pgm` tool can help with attaching to managed PostgreSQL instances on the
system, if present.

### Profiling

Profile frequently to validate performance decisions and catch regressions.
Maintain a performance log over time to track trends.

- Use `perf` to profile
- Create flame graphs

### Benchmarking

Use automated benchmarks to track performance against targets. The SPANN paper
provides datasets and queries suitable for ANN benchmarking. Consider building
a benchmark runner for repeatable, automated testing.

### Coding Patterns

**Enum-to-string conversion:** For `enum`s starting at 0 with no gaps, use a
lookup table with designated initializers instead of a switch statement:

```c
typedef enum ValueType {
    VALUE_TYPE_NULL = 0,
    VALUE_TYPE_BOOL,
    VALUE_TYPE_INT32,
    // ...
} ValueType;

static const char *value_type_names[] = {
    [VALUE_TYPE_NULL]  = "NULL",
    [VALUE_TYPE_BOOL]  = "BOOL",
    [VALUE_TYPE_INT32] = "INT32",
    // ...
};

const char *
value_type_name(ValueType type)
{
    returnvalue_type_names[type];
}
```

This is more compact, faster (O(1) lookup), and keeps `enum` values visually
aligned with their string representations.

Prefer stack allocated objects over heap allocated objects when possible. This
reduces the risk of memory leaks. Add object `<object>_init()` functions
alongside `<object>_create()` functions so that objects can be both stack and
heap allocated. The corresponding `<object>_cleanup()` and `<object>_free()`
functions are provided for cleaning up resources and freeing memory.

### Use `const` when possible

Use `const` when possible to indicate that objects can't be modified. In
functions taking pointer arguments, declare the objects `const` when possible.
Example:

```C
int 
mkt_vector_dot_product(const MktVector *v1, const MktVector *v2)
{
  ...
}
```

## Important Notes

### Dependencies

- `pgvector` extension for vector type and operators.

### Concurrency Safety

- All shared memory structures protected by appropriate locks
- String interning is thread-safe via hash table locks
- Transaction isolation maintained through proper cleanup

### Testing Requirements

- Always run all test suites before committing changes

### Code Style

- Uses clang-format for consistent C formatting
- PostgreSQL coding conventions followed
- Memory allocation patterns follow PostgreSQL standards
- Wrap all lines at 79 characters

#### Header files

Header files should be ordered in categories from top to bottom. Each category
is separated by a newline.

1. Top-level PostgreSQL include file (if applicable) `#include <postgres.h>`
1. Other PostgreSQL includes (if applicable) `#include <access/rel.h>`
1. System category: system includes in style `#include <stdio.h>`
1. Local includes `#include "mkt_types.h`

Exception: the top-level `postgres.h` needs no newline against the rest of the
PostgreSQL include files.

Note: keep the include list minimal. Remove unused includes. The LSP server can
help identify unused includes.

### Documentation

Keep documentation up-to-date with code changes:

- `README.md`: Update with new features, changed requirements, or usage examples
- `docs/architecture.md`: Update when design or structure changes
- `docs/implementation.md`: Update when implementation details change

### Committing Changes

- No "Co-Authored-By: Claude" or "Generated by Claude" in commit messages
- No `git commit -a` or `git add -A` - stage files explicitly
- Split unrelated changes into separate commits
- Ask before committing - manual commits may be preferred
- Run `ninja format` before committing

#### Pre-commit Checklist

Before committing, verify all of the following:

1. **Format code**: `meson compile -C builddir format`
2. **Run all tests**: `meson test -C builddir`
3. **Check lint** (optional but recommended): `./scripts/ci/lint.sh`
4. **Stage files explicitly**: `git add <specific-files>` (never `-a` or `-A`)
5. **Review staged changes**: `git diff --staged`
6. **Write a clear commit message** describing the change
