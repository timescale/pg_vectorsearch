# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository. If available, local instructions can be found in
CLAUDE.local.md.

## Project Overview

pg_vectorsearch is a PostgreSQL extension providing PRISM, an index access
method (IAM) for Approximate Nearest Neighbor (ANN) vector search, inspired
by Google's ScaNN for AlloyDB and Microsoft's SPANN. It uses the vector
format from [pgvector].

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

### Two versions of PRISM: standalone and PostgreSQL

The main purpose of this project is to build PRISM, an index for PostgreSQL
providing first-class vector search performance. Performance is measured by
QPS at a certain recall.

However, PRISM can build both as a standalone (in-memory) vector search
engine and as the pg_vectorsearch extension's PostgreSQL Index Access
Method (IAM).

#### The role of standalone PRISM

The role of the standalone version is to be able to easily test and benchmark
parts of PRISM while isolating it from adverse effects of bottlenecks in
PostgreSQL that we cannot affect. For example, with the standalone CLI, it is
possible to build micro benchmarks for certain SIMD kernels that encode RabitQ
vectors. These same kernels then run in PostgreSQL.

#### Keep standalone and PostgreSQL versions close

It is critically important that the core PRISM logic and architecture stay
close between the standalone version and the PostgreSQL extension. Minimizing
code duplication and version-specific paths of core logic is a critical goal of
the project. If the versions start to diverge in code and their approach,
standalone benchmarks and tests will no longer be representative for the
PostgreSQL version, which defeats the purpose of having a standalone version.

Examples of things that can differ between PRISM's PostgreSQL and standalone
builds:

- Vector storage: standalone stores vectors in memory. There's a point to this:
it provides an upper-bound on performance which allows identifying and
isolating other bottlenecks. For example, if we see good performance of the
search path in standalone, but poor performance in PostgreSQL, we know there's
nothing wrong with the shared code of the scan path.
- Threads vs processes for parallel mode: Standalone uses threads while
PostgreSQL uses worker processes and shared memory. Apart from these
differences, parallel processing code should be as similar as possible.

## References

Research and inspiration:

- [ScaNN reference implementation][scann-ref] (in-memory)
- [SPANN paper]
- [SPTAG] (ANN library used by SPANN for in-memory centroid index)
- [ScaNN for AlloyDB blog][scann-alloydb-blog]
- [ScaNN for AlloyDB whitepaper][scann-alloydb-paper]
- [turbopuffer ANN v3][turbopuffer-ann]
- [SPFresh paper]

Related PostgreSQL extensions:

- [pgvector]
- [pgvectorscale]
- [pg_textsearch]

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

### Claude Code Environment

Claude Code runs in a sandboxed environment with restricted `/tmp` access. Use
`/tmp/claude/` for temporary files instead of `/tmp/` directly. Some tools
(like `lcov`) have `/tmp` hardcoded - use `lcov --tempdir` to specify an
alternative, or use `gcovr` for coverage instead.

### Related Project Checkouts

**PostgreSQL extensions:**

- `../pgvector/` — Vector type and operators (dependency)
- `../pgvectorscale/` — DiskANN-based vector index
- `../pg_textsearch/` — BM25 full-text search

**ANN/quantization reference implementations:**

- `../faiss/` — Facebook AI Similarity Search (IVF, RaBitQ)
- `../RaBitQ-Library/` — Official RaBitQ implementation from paper authors
- `../google-research/scann/` — Google ScaNN (in-memory ANN)

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

**Prerequisites:**

Profiling requires lowering `perf_event_paranoid` to allow non-root profiling:

```bash
# Temporary (until reboot)
sudo sysctl -w kernel.perf_event_paranoid=1

# Permanent (add to /etc/sysctl.conf)
echo "kernel.perf_event_paranoid = 1" | sudo tee -a /etc/sysctl.conf
```

**Usage:**

```bash
# Profile any command
./scripts/profile.sh ./bin/vectorsearch bench distance --dim 768 --count 10000

# Quick benchmark profiling
./scripts/profile-bench.sh 768 10000 avx512

# Profile cache misses
./scripts/profile.sh --events cache-misses ./bin/vectorsearch bench distance
```

Output: `profiles/flamegraph.svg` (open in browser)

All profiling outputs are saved to the `profiles/` directory (gitignored).

See `docs/development.md` for detailed profiling documentation.

### Benchmarking

#### Testing and benchmarking the PostgreSQL build

The PostgreSQL build of PRISM should be benchmarked against a local PostgreSQL
instance.

Prefer "real" datasets over generated data.

#### Critical benchmark instructions

- Core algorithms can be benchmarked using the pg_vectorsearch client
tool's search command (`vectorsearch bench search`) as long as that
exercises a path that is shared between standalone and PostgreSQL
builds of PRISM.
- Always benchmark and profile a release build with all optimizations turned on.
Benchmarking a debug build will _not_ give the correct understanding of the
current performance.
- When running (Claude) in a sandbox, it is not possible for Claude to see the
real running status of PostgreSQL, so it is best to ask the user to restart
PostgreSQL.
- After building and installing a new .so binary of the pg_vectorsearch
extension, it is not necessary to restart PostgreSQL since a new session
will load the new .so (i.e., creating a new session/backend is enough).
Only changes to the SQL code and/or metadata in the extension requires
recreating the extension in PostgreSQL.

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
vec32_dot_product(const Vec32 *v1, const Vec32 *v2)
{
  ...
}
```

## Important Notes

### Dependencies

None at the SQL level. pg_vectorsearch defines its own `vec32`, `vec16`
and `rabitq` types (see `sql/pg_vectorsearch.sql`), so it installs and
runs without pgvector. They install into whichever schema
`CREATE EXTENSION pg_vectorsearch` targets (a `SCHEMA` clause, or the
first existing schema on `search_path`
— typically `public`); maintenance, administration, and inspection
functions are the one exception, always living in a separate, fixed
`vectorsearch` schema regardless of that choice. The distinct
`vec32`/`vec16` names do not collide with pgvector's `vector`/`halfvec`,
even when both extensions' types end up in the same schema.

**pgvector interoperability.** `vec32` is binary-compatible with pgvector's
`vector`, and `vec16` with pgvector's `halfvec`; the extension creates
binary-coercible casts between the two in both directions:

```sql
CREATE CAST (public.vector  AS myschema.vec32)  WITHOUT FUNCTION AS IMPLICIT;
CREATE CAST (myschema.vec32 AS public.vector)   WITHOUT FUNCTION AS ASSIGNMENT;
-- and the same pair for halfvec
```

Direction matters: pgvector → pg_vectorsearch is IMPLICIT so an existing
pgvector column works transparently, while pg_vectorsearch → pgvector is
ASSIGNMENT to avoid operator ambiguity when both extensions are
installed. Install order does not matter —
`prism.setup_pgvector_compat()` runs at `CREATE EXTENSION pg_vectorsearch`
if pgvector is already there, and an event trigger runs it if pgvector
arrives later.

The consequence worth remembering: because the casts are `WITHOUT FUNCTION`,
they satisfy PostgreSQL's binary-coercibility rule for operator classes. That
is what lets a `prism` opclass declared `FOR TYPE public.vec32` be used
on a column of pgvector's `public.vector` — the two are the same bytes,
so no conversion happens and no copy is made. Anything in the
access method that asks "which type is this column?" must ask it the
same way, via `IsBinaryCoercible` rather than OID equality, or it will
disagree with the planner about a column the index was built on.

Operators need separate handling, because casts do not cover them. An index is
only considered for an `ORDER BY` when the ordering operator belongs to the
index's operator family, and pgvector's `<->`, `<#>` and `<=>` belong to
pgvector's families. So `prism.setup_pgvector_compat()` adds them to
`prism`'s operator families as ordering members (strategy 1, `float_ops`)
in the same step as the casts — they are a unit, since the operators rely
on the casts' binary-coercibility. Either spelling of the operator then
reaches the index.

The failure this avoids is quiet rather than loud: with pgvector's operator and
no family membership, the planner picks a sequential scan, which returns the
right rows. Any test that checks only results will pass while measuring brute
force — so plan checks, not just recall checks, are what pin this down.

The `public.<type>` column with pgvector's operator is the one pairing
that does not resolve, and deliberately so: the pg_vectorsearch →
pgvector cast is ASSIGNMENT, not IMPLICIT, precisely to keep operator
calls unambiguous when both extensions are installed.

### Concurrency Safety

- All shared memory structures protected by appropriate locks
- String interning is thread-safe via hash table locks
- Transaction isolation maintained through proper cleanup

### Testing Requirements

- Always run all test suites before committing changes

### GitHub Workflows

Use [zizmor] to check workflow files for
security best practices:

```bash
zizmor --pedantic .github/workflows/
```

Workflow requirements:

- Pin actions to commit hashes (not tags)
- Add `persist-credentials: false` to checkouts
- Add explicit `permissions` blocks
- Add `concurrency` settings to cancel duplicate runs
- When testing optimizations or new experimental features, do not roll the
changes back if the changes didn't show the performance or promise hoped for.
Instead, save the code to an aptly-named branch and ask the user how proceed.
Do not automatically discard the changes!

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
1. Local includes `#include "vs_types.h`

Exception: the top-level `postgres.h` needs no newline against the rest of the
PostgreSQL include files.

Note: keep the include list minimal. Remove unused includes. The LSP server can
help identify unused includes.

### Documentation Maintenance

Keep documentation up-to-date with code changes:

- `README.md`: Update with new features, changed requirements, or usage examples
- `docs/architecture.md`: Update when design or structure changes
- `docs/implementation.md`: Update when implementation details change

## Development Workflow

Work follows a branch and pull request workflow — never commit directly to
`main`. The full checklist (branch naming, pre-commit checks, staging,
Conventional Commits message format, rebase/push, PR body style, and
post-merge cleanup) lives in two Claude Code skills rather than here:

- `git-workflow` — starting a branch, working in reviewable increments,
  addressing review feedback, cleaning up after merge.
- `create-pr` — pre-commit checks, committing, rebasing, pushing, and
  opening the PR.

<!-- Links -->
[pgvector]: https://github.com/pgvector/pgvector
[pgvectorscale]: https://github.com/timescale/pgvectorscale
[pg_textsearch]: https://github.com/timescale/pg_textsearch
[scann-ref]: https://github.com/google-research/google-research/tree/master/scann
[SPANN paper]: https://www.microsoft.com/en-us/research/wp-content/uploads/2021/11/SPANN_finalversion1.pdf
[SPTAG]: https://github.com/microsoft/SPTAG
[scann-alloydb-blog]: https://cloud.google.com/blog/products/databases/understanding-the-scann-index-in-alloydb
[scann-alloydb-paper]: https://services.google.com/fh/files/misc/scann_for_alloydb_whitepaper.pdf
[turbopuffer-ann]: https://turbopuffer.com/blog/ann-v3
[SPFresh paper]: https://dl.acm.org/doi/epdf/10.1145/3600006.3613166
[zizmor]: https://github.com/zizmorcore/zizmor
