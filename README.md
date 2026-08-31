# Meerkat

[![Coverage](https://img.shields.io/endpoint?url=https://timescale.github.io/meerkat/coverage-badge.json)](https://timescale.github.io/meerkat/coverage/)

A PostgreSQL index access method for Approximate Nearest Neighbor (ANN) vector
search.

## Overview

Meerkat is a PostgreSQL extension that provides high-performance vector
similarity search using an IVF (Inverted File) index structure with quantized
vectors. It is inspired by [ScaNN][scann] and [SPANN][spann].

Meerkat provides its own `vector` and `halfvec` types that are binary-compatible
with [pgvector][pgvector]. pgvector is not required — meerkat installs and runs
on its own.

Where pgvector is installed, meerkat creates binary casts between the two
extensions' types, so an existing pgvector column can be indexed directly with
no rewrite and no copy, and it adds pgvector's `<->`, `<#>` and `<=>` to its
own index operator families. A query written against pgvector — including a
bare `<->` that resolves to pgvector — therefore uses a meerkat index without
being rewritten.

## Features

- **Billion-scale vector search** with hierarchical clustering
- **Native PostgreSQL integration** via the Index Access Method (IAM) API
- **SIMD-optimized distance computation** (AVX2, AVX512, NEON)
- **RaBitQ quantization** with theoretical error bounds for two-stage search
- **Dynamic updates** via LIRE protocol (split, merge, reassign)
- **Multi-tenant support** with composite key indexes
- **Buffer cache integration** for both centroid and posting list pages
- **MVCC and replication support** through standard PostgreSQL mechanisms

## Architecture

Meerkat uses a hierarchical centroid tree to partition vectors into clusters.
Centroids are stored in dedicated pages within the PostgreSQL buffer cache.
Vectors are RaBitQ-quantized and stored in posting lists with a
bidirectional page layout optimized for SIMD distance computation.

```
Query Flow:
1. Traverse centroid tree to find nearest clusters (buffer cache)
2. Scan posting lists for candidates (buffer cache, async I/O)
3. RaBitQ two-stage filtering: estimate → error bound → rerank
4. Full-precision reranking from heap
5. Return k nearest neighbors
```

For detailed architecture, see [docs/architecture.md][arch-doc].

## Status

**Early development** - Currently in design and planning phase.

See [docs/implementation.md][impl-doc] for the implementation roadmap and
detailed specifications.

## Requirements

- PostgreSQL 18+
- C23 compiler (GCC 13+ or Clang 16+)
- Meson build system
- Optional: [pgvector][pgvector] for compatibility with existing pgvector
  workflows

## Building

```bash
# Setup build directory
meson setup builddir

# Compile
meson compile -C builddir

# Run tests
meson test -C builddir

# Install to PostgreSQL
meson install -C builddir
```

### Build Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `postgresql` | `auto`, `enabled`, `disabled` | `auto` | Build PostgreSQL extension |
| `pg_config` | path | (auto-detect) | Path to `pg_config` |
| `simd` | `full`, `compiler`, `none` | `full` | SIMD implementation mode |
| `native` | `true`, `false` | `false` | Use `-march=native` for local builds |
| `blas` | `auto`, `enabled`, `disabled` | `auto` | CBLAS for matrix operations |
| `tools` | `auto`, `enabled`, `disabled` | `auto` | Developer tools (CLI, standalone library, unit tests) |

Use `-Dpostgresql=enabled` to require the extension build (fails if PostgreSQL
is not found). Use `-Dpostgresql=disabled` to build only the standalone library
and CLI tools.

The `tools` default (`auto`) builds the developer tools in a git checkout but
skips them when building from a release tarball, so packagers get an
extension-only build unless they pass `-Dtools=enabled`. PostgreSQL regression
tests are unaffected.

### Optional: BLAS Library

Installing a CPU-optimized BLAS library significantly improves RaBitQ encoding
performance (~4x speedup for batch operations). The build system auto-detects
CBLAS if available.

**Recommended libraries by CPU:**

| CPU | Library | Link |
|-----|---------|------|
| Intel | Intel oneMKL | [intel.com/oneapi/onemkl][mkl] |
| AMD | AOCL-BLIS | [amd.com/aocl/blis][aocl] |
| ARM | ARM Performance Libraries | [developer.arm.com][armpl] |
| Any | OpenBLAS | [openblas.net][openblas] |

**Quick install:**

```bash
# Arch Linux (AUR packages require yay or similar)
pacman -S blas-openblas      # Generic (official repo)
yay -S blas-mkl              # Intel (AUR)
yay -S blas-aocl-gcc         # AMD (AUR)

# Debian/Ubuntu
apt install libopenblas-dev

# Fedora
dnf install openblas-devel
```

## Benchmarking

The `mkt` CLI tool includes benchmarks for distance computation:

```bash
# Run distance benchmark (default: dim=768, count=10000)
./bin/mkt bench distance

# Compare implementations
./bin/mkt bench distance --dim 1536 --count 50000
```

See [docs/simd.md][simd-doc] for SIMD build options and implementation details.

## Usage

```sql
-- Enable extension
CREATE EXTENSION meerkat;

-- Create table with vector column
CREATE TABLE items (
    id serial PRIMARY KEY,
    embedding mkt.vector(768)
);

-- Create Meerkat index
CREATE INDEX ON items USING meerkat (embedding mkt.vector_l2_ops);

-- Query nearest neighbors
SELECT * FROM items
ORDER BY embedding <-> '[...]'::mkt.vector
LIMIT 10;

-- Speed/recall dial: probes more clusters for higher recall. The
-- default (0 = auto) derives it from the index size, targeting
-- ~0.95 recall; lower is faster, higher is more accurate.
SET mkt.nprobe = 40;

-- Queries with LIMIT above 10 must size the scan accordingly
SET mkt.query_limit = 100;
```

Defaults are tuned from large-scale benchmarks; most deployments only
ever adjust `mkt.nprobe` (and `mkt.query_limit` for larger LIMITs).
See the [tuning guide][tuning-doc] for every index parameter and GUC,
their tradeoffs, and when changing them makes sense.

## Documentation

- [Tuning][tuning-doc] - Index parameters, GUCs, tradeoffs, defaults
- [Architecture][arch-doc] - High-level design and data structures
- [Implementation][impl-doc] - Detailed specifications and development phases
- [SIMD][simd-doc] - SIMD build options and distance computation

## References

### Research

- [RaBitQ][rabitq] - Binary quantization with theoretical error bounds
  (SIGMOD 2024)
- [SPFresh][spfresh] - LIRE protocol for incremental updates (SOSP 2023)
- [SPANN][spann-paper] - Billion-scale ANN with boundary replication
  (NeurIPS 2021)
- [ScaNN for AlloyDB][scann-alloydb] - Hierarchical clustering (Google)

### Related Projects

- [pgvector][pgvector] - Vector type and operators for PostgreSQL
- [pgvectorscale][pgvectorscale] - DiskANN-based vector index for PostgreSQL

## License

TBD

<!-- Links -->
[mkl]: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html
[aocl]: https://www.amd.com/en/developer/aocl/blis.html
[armpl]: https://developer.arm.com/Tools%20and%20Software/Arm%20Performance%20Libraries
[openblas]: https://www.openblas.net/
[scann]: https://github.com/google-research/google-research/tree/master/scann
[spann]: https://www.microsoft.com/en-us/research/publication/spann-highly-efficient-billion-scale-approximate-nearest-neighbor-search/
[pgvector]: https://github.com/pgvector/pgvector
[pgvectorscale]: https://github.com/timescale/pgvectorscale
[rabitq]: https://github.com/gaoj0017/RaBitQ
[spfresh]: https://dl.acm.org/doi/10.1145/3600006.3613166
[spann-paper]: https://www.microsoft.com/en-us/research/wp-content/uploads/2021/11/SPANN_finalversion1.pdf
[scann-alloydb]: https://services.google.com/fh/files/misc/scann_for_alloydb_whitepaper.pdf
[arch-doc]: docs/architecture.md
[impl-doc]: docs/implementation.md
[tuning-doc]: docs/tuning.md
[simd-doc]: docs/simd.md
