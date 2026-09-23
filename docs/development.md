# Development Guide

## Building

For the common build-and-install flow there is a thin Makefile wrapper
(release build against the installed PostgreSQL by default):

```bash
make                 # build the extension
sudo make install    # configure + build + install
make install PG_CONFIG=/path/to/pg_config   # pick a PostgreSQL
```

See the Makefile header for the other variables (BUILDTYPE, BUILDDIR,
MESON_ARGS). Everything below uses meson directly.

```bash
# Setup build directory
meson setup builddir

# Compile
meson compile -C builddir

# Run tests
meson test -C builddir

# Format code
meson compile -C builddir format
```

## Code Coverage

Meson has built-in coverage support that auto-detects the compiler and tools.

### Requirements

One of:
- **GCC**: gcovr or lcov/genhtml
- **Clang**: llvm-cov

Install gcovr (recommended):
```bash
pip install gcovr
```

### Usage

```bash
# Setup with coverage enabled
meson setup builddir-cov -Db_coverage=true

# Build and run tests
meson compile -C builddir-cov
meson test -C builddir-cov

# Generate HTML report
ninja -C builddir-cov coverage-html
```

The HTML report is generated in `builddir-cov/meson-logs/coveragereport/`.

### Coverage Targets

Meson provides several coverage targets:

| Target | Description |
|--------|-------------|
| `coverage` | Generate coverage report (text) |
| `coverage-html` | Generate HTML report |
| `coverage-xml` | Generate XML report (Cobertura) |
| `coverage-text` | Generate text summary |

## Sanitizers

Meson supports Clang/GCC sanitizers via the `b_sanitize` option.

### AddressSanitizer (ASan)

Detects memory errors: use-after-free, buffer overflow, double-free, memory
leaks.

```bash
meson setup builddir-asan -Db_sanitize=address
meson test -C builddir-asan
```

### UndefinedBehaviorSanitizer (UBSan)

Detects undefined behavior: null pointer dereference, signed integer overflow,
etc.

```bash
meson setup builddir-ubsan -Db_sanitize=undefined
meson test -C builddir-ubsan
```

### Combined Sanitizers

Multiple sanitizers can be combined:

```bash
meson setup builddir-san -Db_sanitize=address,undefined
meson test -C builddir-san
```

### Available Sanitizers

| Sanitizer | Option | Detects |
|-----------|--------|---------|
| AddressSanitizer | `address` | Use-after-free, buffer overflow, leaks |
| LeakSanitizer | `leak` | Memory leaks (included in ASan) |
| UndefinedBehaviorSanitizer | `undefined` | Undefined behavior |
| ThreadSanitizer | `thread` | Data races |
| MemorySanitizer | `memory` | Uninitialized reads (Clang only) |

Note: ThreadSanitizer and MemorySanitizer cannot be combined with
AddressSanitizer.

## Static Analysis

### clang-tidy

Meson generates `compile_commands.json` which clang-tidy uses for analysis.

```bash
# Run clang-tidy via meson target
meson compile -C builddir tidy

# Or run directly on all source files
clang-tidy -p builddir src/*.c

# Run with fixes applied
clang-tidy -p builddir --fix src/*.c
```

### scan-build (Clang Static Analyzer)

Wrap the build command with scan-build:

```bash
# Basic scan
scan-build meson compile -C builddir

# Generate HTML report in reports/
scan-build -o reports meson compile -C builddir

# View report
scan-view reports/<timestamp>/
```

Note: scan-build requires a clean build to analyze all files:
```bash
meson compile -C builddir --clean
scan-build meson compile -C builddir
```

## Debug Builds

```bash
# Setup debug build
meson setup builddir-debug --buildtype=debug

# Setup release build
meson setup builddir-release --buildtype=release
```

## Profiling

### Quick Profiling with perf

The `scripts/profile.sh` script automates profiling with `perf` and generates
interactive flame graphs.

#### Requirements

- `perf` (linux-tools-common, linux-tools-generic)
- FlameGraph (auto-downloaded to `/tmp/FlameGraph` if not found)

Install perf:
```bash
sudo apt-get install linux-tools-common linux-tools-generic
```

#### Basic Usage

```bash
# Profile any command
./scripts/profile.sh ./bin/vectorsearch bench distance --dim 768 --count 10000

# Convenience script for benchmarks
./scripts/profile-bench.sh 768 10000 avx512

# Profile with custom events
./scripts/profile.sh --events cache-misses ./bin/vectorsearch bench distance

# Keep perf.data for manual analysis
./scripts/profile.sh --keep-perf-data ./bin/vectorsearch bench distance
```

#### Output

The script generates files in the `profiles/` directory:
- `profiles/flamegraph.svg` - Interactive flame graph (CPU time by function)
- `profiles/flamegraph-icicle.svg` - Inverted flame graph (call chains from bottom)
- `profiles/flamegraph.folded` - Collapsed stack traces (for manual analysis)
- `profiles/perf.data` - Raw perf data (if `--keep-perf-data` used)

Open in browser:
```bash
firefox profiles/flamegraph.svg
```

#### Profile Build

The script automatically creates `builddir-profile` with:
- Release optimizations (`-O3`)
- Frame pointers (`-fno-omit-frame-pointer`)
- Debug symbols (`-g`)

This ensures accurate stack traces while maintaining realistic performance.

#### Advanced Usage

```bash
# Profile cache misses instead of CPU cycles
./scripts/profile.sh --events cache-misses --output cache-profile \
  ./bin/vectorsearch bench distance --dim 768 --count 100000

# Profile branch mispredictions
./scripts/profile.sh --events branch-misses --output branch-profile \
  ./bin/vectorsearch bench distance --dim 384 --count 50000

# Profile at higher frequency (more samples, more overhead)
./scripts/profile.sh --freq 4999 ./bin/vectorsearch bench distance

# Profile multiple events
./scripts/profile.sh --events cycles,cache-misses \
  ./bin/vectorsearch bench distance
```

#### Interpreting Flame Graphs

- **Width**: Total time spent in function (including children)
- **Color**: Random (for visual distinction, not meaningful)
- **Interactive**: Click to zoom, search for function names
- **Icicle graph**: Shows call chains from root (main) at bottom

Look for:
- Wide bars = hot functions (optimization targets)
- Tall stacks = deep call chains (potential inlining opportunities)
- Unexpected functions = hidden overhead

#### Manual Analysis

For detailed analysis, use `perf report` directly:
```bash
# Generate profile with --keep-perf-data
./scripts/profile.sh --keep-perf-data ./bin/vectorsearch bench distance

# Interactive report
perf report

# Text report
perf report --stdio

# Show annotated source code
perf annotate vs_distance_l2_avx512
```

## Formatting

Code is formatted with clang-format. The project prefers clang-format-18 for
consistency but falls back to any available version.

```bash
# Format all source files
meson compile -C builddir format
```

## CI Scripts

Standalone scripts in `scripts/ci/` can be run locally for testing and
debugging. These are the same scripts used by GitHub Actions.

### Available Scripts

| Script | Description |
|--------|-------------|
| `build.sh` | Build and run tests |
| `coverage.sh` | Build with coverage, generate report |
| `sanitizers.sh` | Build and test with sanitizers |
| `lint.sh` | Check formatting and run clang-tidy |
| `pgspot.sh` | Static security lint of the install-time SQL (pgspot) |

### Usage

```bash
# Basic build and test
./scripts/ci/build.sh

# Build with coverage
./scripts/ci/coverage.sh

# Run with AddressSanitizer
./scripts/ci/sanitizers.sh address

# Run with UndefinedBehaviorSanitizer
./scripts/ci/sanitizers.sh undefined

# Run with both (default)
./scripts/ci/sanitizers.sh address,undefined

# Check formatting and run clang-tidy
./scripts/ci/lint.sh

# Security-lint the install-time SQL (needs: pip install pgspot)
./scripts/ci/pgspot.sh
```

### Custom Build Directory

All scripts accept an optional build directory argument:

```bash
./scripts/ci/build.sh my-builddir
./scripts/ci/coverage.sh my-coverage-dir
```

## GitHub Actions

The project uses GitHub Actions for CI with the following workflows:

| Workflow | Trigger | Description |
|----------|---------|-------------|
| `build.yml` | Push, PR | Build and test |
| `sanitizers.yml` | Push, PR | ASan and UBSan tests |
| `coverage.yml` | Push, PR | Coverage report, upload to Codecov |
| `lint.yml` | Push, PR | Format check and clang-tidy |
| `pgspot.yml` | Push, PR (SQL) | pgspot security lint of install-time SQL |
| `codeql.yml` | Push, PR, Weekly | GitHub CodeQL static analysis |

## Design Patterns

### Config-Based Polymorphism (Vtable Inlining)

PRISM uses `Vec32TypeOps` vtables for type-generic algorithms (k-means,
RaBitQ) that work over both float32 and float16 input. Naive vtable dispatch
through function pointers prevents inlining and auto-vectorization in hot loops,
causing 10-15% overhead on inner loops like dot product.

We use a pattern inspired by [WebKit's libpas allocator][libpas-docs] to get
zero-cost polymorphism in C. The key insight: **dispatch once at the outer
level, then use direct inline functions in the hot loop**.

Three dispatch modes, from fastest to most flexible:

| Mode | How | Use When |
|------|-----|----------|
| **Inlined** | Convert input to known type, call inline function directly | Hot inner loops (distance, dot product) |
| **Specialized** | Call through vtable once per batch/iteration | Warm paths (norm precompute, centroid update) |
| **Virtual** | Call through vtable per element | Cold paths, or when conversion is too expensive |

#### Example: k-means Assignment

The assignment inner loop computes dot products between every vector and every
centroid. This is the hottest code in k-means.

**Bad** — virtual dispatch per dot product (prevents vectorization):

```c
for (uint32_t i = 0; i < nvecs; i++) {
    const void *v = vec_at(st, i);
    for (uint32_t j = 0; j < nlist; j++) {
        /* Function pointer call — not inlineable */
        float dp = st->ops->dot_product(v, centroid[j], dim);
        ...
    }
}
```

**Good** — convert once, inline the hot loop:

```c
/* Warm path: one vtable call per vector to get float32 view */
const float *vf = st->ops->to_float_block(block, buf, count, dim);

for (uint32_t i = 0; i < count; i++) {
    for (uint32_t j = 0; j < nlist; j++) {
        /* Direct inline call — compiler auto-vectorizes */
        float dp = dot_product_f32(vf + i * dim, centroid[j], dim);
        ...
    }
}
```

For float32 input, `to_float_block` returns the source pointer directly
(zero-copy). For float16 input, it converts into a scratch buffer. Either way,
the inner loop operates on `float *` with a direct function call that the
compiler can inline and auto-vectorize with `VS_TARGET_CLONES`.

#### Why This Works

The C compiler applies two optimizations when an `always_inline` function
receives a compile-time-known function pointer to another `always_inline`
function:

1. **Inline the callee** — the function pointer call becomes a direct call
2. **Copy-propagate** — constant arguments flow into the inlined body

This is equivalent to C++ template monomorphization but without code bloat for
cold paths. The libpas documentation calls this "specialization akin to template
monomorphization."

In our case, we achieve the same effect more simply: by converting to a known
type at the batch boundary, the inner loop doesn't need function pointers at
all — it just operates on `float *` directly.

[libpas-docs]: https://github.com/WebKit/WebKit/blob/main/Source/bmalloc/libpas/Documentation.md#libpas-style
