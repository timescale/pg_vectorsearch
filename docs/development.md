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

## Formatting

Code is formatted with clang-format. The project prefers clang-format-18 for
consistency but falls back to any available version.

```bash
# Format all source files
meson compile -C builddir format
```
