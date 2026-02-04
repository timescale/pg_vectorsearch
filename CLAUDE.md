# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Project Overview

Meerkat is a PostgreSQL index access method (IAM) for Approximate Nearest
Neighbor (ANN) vector search, inspired by Google's ScaNN for AlloyDB and
Microsoft's SPANN. It uses the vector format from [pgvector].

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

### Documentation Maintenance

Keep documentation up-to-date with code changes:

- `README.md`: Update with new features, changed requirements, or usage examples
- `docs/architecture.md`: Update when design or structure changes
- `docs/implementation.md`: Update when implementation details change

## Development Workflow

Work follows a branch and pull request workflow:

1. **Create a feature branch** with a descriptive name:
   - **IMPORTANT**: Always ensure `main` is up-to-date before branching:
     ```bash
     git checkout main
     git pull origin main
     git checkout -b feature/branch-name
     ```
   - Branch naming patterns:
     - Features: `feature/branch-name` (e.g., `feature/rabitq-quantization`)
     - Bug fixes: `bugfix/description` (e.g., `bugfix/memory-leak-clustering`)
     - Tests: `test/description` (e.g., `test/fixtures`)
     - Refactoring: `refactor/description`

2. **Work in iterations** on the branch:
   - Make focused commits with clear messages
   - Run pre-commit checks before each commit
   - Keep changes logical and reviewable

3. **Push to GitHub** when ready for review:
   - **IMPORTANT**: Always rebase on `main` before pushing:
     ```bash
     git fetch origin
     git rebase origin/main
     git push -u origin branch-name
     ```
   - If you've already pushed and need to update:
     ```bash
     git fetch origin
     git rebase origin/main
     git push --force-with-lease origin branch-name
     ```

4. **Create a pull request** on GitHub:
   - Write a clear description of the changes
   - Reference any related issues
   - Ensure CI checks pass

5. **Address review feedback**:
   - User reviews the PR
   - Make requested changes in new commits
   - Push updates to the same branch

6. **After merge**:
   - User merges the PR
   - Pull the updated main branch:
     ```bash
     git checkout main
     git pull origin main
     ```
   - Delete the local feature branch:
     ```bash
     git branch -d branch-name
     ```
   - Start the next feature/bugfix

### Committing Changes

#### Pre-commit Checks

**IMPORTANT**: All checks must pass before committing. The pre-commit hooks will
enforce formatting, but coverage and linting must be verified manually.

Before staging changes, verify:

1. **Format code** - `meson compile -C builddir format`
2. **Run tests** - `meson test -C builddir`
3. **Check coverage** - `./scripts/ci/coverage.sh` (90% minimum required)
4. **Check lint** - `./scripts/ci/lint.sh` (must pass with no errors)

```bash
# All checks in sequence
meson compile -C builddir format && \
meson test -C builddir && \
./scripts/ci/coverage.sh && \
./scripts/ci/lint.sh
```

**If coverage or linting fails, fix the issues before committing.** The CI will
reject commits that don't meet these standards.

The coverage script creates a separate build directory (`builddir-cov`) with
`-Db_coverage=true` and generates HTML/XML reports. Reports are written to
`builddir-cov/meson-logs/coveragereport/`.

Pre-commit hooks run automatically on commit, but can also be run manually:

```bash
pre-commit run --all-files    # Run all hooks on all files
pre-commit run --files src/*  # Run on specific files
```

#### Staging

- Never use `git add -A` or `git commit -a` - stage files explicitly
- Split unrelated changes into separate commits
- Review staged changes with `git diff --staged`
- Ask before committing - manual commits may be preferred

#### Commit Message Guidelines

This project follows [Conventional Commits](https://www.conventionalcommits.org/)
for structured, machine-readable commit messages.

**Format:**
```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style/formatting (no functional change)
- `refactor:` - Code refactoring
- `perf:` - Performance improvement
- `test:` - Adding or updating tests
- `build:` - Build system or dependencies
- `ci:` - CI configuration changes
- `chore:` - Other changes (tooling, etc.)

**Examples:**
```
feat(vector): add RaBitQ quantization support
fix(memory): resolve leak in context cleanup
docs: update branch workflow in CLAUDE.md
test: add fixture system with three levels
refactor(clustering): simplify centroid calculation
```

**Body Guidelines:**
- Write human-readable messages focusing on *what* and *why*
- Include key technical insights or trade-offs
- Note caveats or follow-up work if relevant
- Avoid: file lists, long bullet lists, "Co-Authored-By" lines

**Enforcement:**
Pre-commit hooks validate commit messages automatically. Messages must:
- Start with a valid type
- Use lowercase for type and description
- Keep header under 72 characters
- Have blank line before body (if present)

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
