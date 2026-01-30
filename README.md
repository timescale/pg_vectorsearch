# Meerkat

A PostgreSQL index access method for Approximate Nearest Neighbor (ANN) vector
search.

## Overview

Meerkat is a PostgreSQL extension that provides high-performance vector
similarity search using an IVF (Inverted File) index structure with quantized
vectors. It is inspired by [ScaNN][scann] and [SPANN][spann], and uses the vector type
from [pgvector][pgvector].

## Features

- **Native PostgreSQL integration** via the Index Access Method (IAM) API
- **SIMD-optimized distance computation** (AVX2, AVX512, NEON)
- **RaBitQ quantization** for memory-efficient vector storage with theoretical
  error bounds
- **Streaming search** with support for filtered queries
- **Buffer cache integration** for posting lists with separate centroid cache
- **MVCC and replication support** through standard PostgreSQL mechanisms

## Architecture

Meerkat partitions the vector space into clusters using k-means clustering.
Each cluster has a centroid stored in a shared memory cache for fast nearest
cluster lookup. Vectors are quantized and stored in posting lists within
standard PostgreSQL pages.

```
Query Flow:
1. Find nearest centroids (in-memory cache)
2. Scan posting lists for candidate clusters (buffer cache)
3. Compute distances using quantized vectors (SIMD)
4. Rerank top candidates with full precision
5. Return k nearest neighbors
```

For detailed architecture, see [docs/architecture.md][arch-doc].

## Status

**Early development** - Currently in design and planning phase.

See [docs/implementation.md][impl-doc] for the implementation roadmap and
detailed specifications.

## Requirements

- PostgreSQL 18+
- [pgvector][pgvector] extension
- C23 compiler (GCC 13+ or Clang 16+)
- Meson build system

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

## Usage

```sql
-- Enable extensions
CREATE EXTENSION vector;
CREATE EXTENSION meerkat;

-- Create table with vector column
CREATE TABLE items (
    id serial PRIMARY KEY,
    embedding vector(768)
);

-- Create Meerkat index
CREATE INDEX ON items USING meerkat (embedding vector_l2_ops);

-- Query nearest neighbors
SELECT * FROM items
ORDER BY embedding <-> '[...]'::vector
LIMIT 10;
```

## Documentation

- [Architecture][arch-doc] - High-level design and data structures
- [Implementation][impl-doc] - Detailed specifications and development phases

## References

### Research

- [RaBitQ][rabitq] - Quantizing High-Dimensional Vectors with a Theoretical
  Error Bound (SIGMOD 2024)
- [SPANN][spann-paper] - Highly-efficient Billion-scale Approximate Nearest
  Neighbor Search (NeurIPS 2021)
- [ScaNN for AlloyDB][scann-alloydb] - Google whitepaper

### Related Projects

- [pgvector][pgvector] - Vector type and operators for PostgreSQL
- [pgvectorscale][pgvectorscale] - DiskANN-based vector index for PostgreSQL

## License

TBD

<!-- Links -->
[scann]: https://github.com/google-research/google-research/tree/master/scann
[spann]: https://www.microsoft.com/en-us/research/publication/spann-highly-efficient-billion-scale-approximate-nearest-neighbor-search/
[pgvector]: https://github.com/pgvector/pgvector
[pgvectorscale]: https://github.com/timescale/pgvectorscale
[rabitq]: https://github.com/gaoj0017/RaBitQ
[spann-paper]: https://www.microsoft.com/en-us/research/wp-content/uploads/2021/11/SPANN_finalversion1.pdf
[scann-alloydb]: https://services.google.com/fh/files/misc/scann_for_alloydb_whitepaper.pdf
[arch-doc]: docs/architecture.md
[impl-doc]: docs/implementation.md
