# Development Guide

## Building

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
./scripts/profile.sh ./bin/mkt bench distance --dim 768 --count 10000

# Convenience script for benchmarks
./scripts/profile-bench.sh 768 10000 avx512

# Profile with custom events
./scripts/profile.sh --events cache-misses ./bin/mkt bench distance

# Keep perf.data for manual analysis
./scripts/profile.sh --keep-perf-data ./bin/mkt bench distance
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
  ./bin/mkt bench distance --dim 768 --count 100000

# Profile branch mispredictions
./scripts/profile.sh --events branch-misses --output branch-profile \
  ./bin/mkt bench distance --dim 384 --count 50000

# Profile at higher frequency (more samples, more overhead)
./scripts/profile.sh --freq 4999 ./bin/mkt bench distance

# Profile multiple events
./scripts/profile.sh --events cycles,cache-misses ./bin/mkt bench distance
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
./scripts/profile.sh --keep-perf-data ./bin/mkt bench distance

# Interactive report
perf report

# Text report
perf report --stdio

# Show annotated source code
perf annotate mkt_distance_l2_avx512
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
| `codeql.yml` | Push, PR, Weekly | GitHub CodeQL static analysis |
