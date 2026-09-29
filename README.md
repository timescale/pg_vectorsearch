# pg_vectorsearch

[![Build](https://github.com/timescale/pg_vectorsearch/actions/workflows/build.yml/badge.svg?branch=main&event=push)](https://github.com/timescale/pg_vectorsearch/actions/workflows/build.yml?query=branch%3Amain+event%3Apush)
[![Coverage](https://img.shields.io/endpoint?url=https://timescale.github.io/pg_vectorsearch/coverage-badge.json)](https://timescale.github.io/pg_vectorsearch/coverage/)
[![Coverity](https://scan.coverity.com/projects/timescale-pg_vectorsearch/badge.svg)](https://scan.coverity.com/projects/timescale-pg_vectorsearch)

A PostgreSQL extension for high-performance approximate nearest neighbor (ANN)
search.

- **PRISM index** (100M+ vectors) — IVF-style, with hierarchical routing
- **RaBitQ quantization** with theoretical error bounds for two-stage search
- **SIMD-optimized distance computation** (AVX2, AVX512, NEON)
- **Dynamic updates** — inserts, updates and deletes, with posting-list
  splits on demand (see [PRISM index maintenance](#prism-index-maintenance))
- **Vector types** compatible with [pgvector][pgvector]'s types

## PRISM Index

<!-- markdownlint-disable MD033 -->
<p align="center">
<img src="images/prism-logo.jpg" alt="PRISM">
</p>
<!-- markdownlint-enable MD033 -->

PRISM (Partitioned Routing Index for Similarity Matching) is a PostgreSQL-native
index access method that is conceptually similar to [pgvector][pgvector]'s
IVFFlat, but it is built around state-of-the-art quantization, routing, and
partitioning techniques. It is implemented in C for high performance and
seamless integration with PostgreSQL.

The result: an index that is significantly faster than pgvector's IVFFlat and
HNSW, for both querying and index building. In fact, compared to graph-based
HNSW, PRISM is faster at equivalent recall on a 100-million-vector dataset,
on an instance with a quarter of the memory.

To achieve this performance, PRISM uses a hierarchical centroid tree to
partition vectors into clusters, and relies on the PostgreSQL buffer cache to
keep that tree warm in memory. Vectors are [RaBitQ][rabitq]-quantized and stored
in posting lists with a page layout optimized for SIMD distance computation. The
RaBitQ encoding carries error bounds that let a search rule out vectors that
cannot possibly be nearest neighbors, so far fewer of them need reranking
against the full-precision vectors stored in the table.

```
Query Flow:
1. Traverse centroid tree to find nearest clusters (buffer cache)
2. Scan posting lists for candidates (buffer cache, async I/O)
3. RaBitQ two-stage filtering: estimate → error bound → rerank
4. Full-precision reranking from heap
5. Return k nearest neighbors
```

For detailed architecture, see [docs/architecture.md][arch-doc].

## pgvector compatibility

PRISM indexes work with [pgvector][pgvector]'s `vector` and `halfvec` types and
operators, but do not depend on them. The `pg_vectorsearch` extension ships
its own `vec32` and `vec16` types, binary compatible with their pgvector
counterparts but named differently so that both extensions can coexist even
with their objects in the same schema. Binary casts between each pair mean an
existing pgvector column can be indexed as-is, with no rewrite and no copy,
and queries already written against pgvector's operators work unchanged with
a PRISM index.

## Status

**Alpha-level.**

See [docs/architecture.md][arch-doc] for the design.

## Building

**Requirements:**
- PostgreSQL 18+ (with development headers)
- C23 compiler (GCC 13+ or Clang 16+)
- Meson build system
- Optional: CBLAS, for faster RaBitQ encoding. See
  [Development](docs/development.md#blas).

```bash
# Release build
meson setup builddir --buildtype=release

# Build and install to PostgreSQL
meson install -C builddir
```

See the [development guide](docs/development.md#build-options) for build
options.

## Usage

```sql
CREATE EXTENSION pg_vectorsearch;

-- PLAIN keeps the vector inline. EXTERNAL (the default) toasts a value
-- over ~2 kB, and rerank then pays an extra fetch the planner does not cost.
CREATE TABLE items (
    id serial PRIMARY KEY,
    embedding vec32(3) STORAGE PLAIN
);

-- Load first; the index is sized from the row count at CREATE INDEX.
-- 500 rows is enough for the planner to use the index.
INSERT INTO items (embedding)
SELECT ARRAY[i::real, (i % 7)::real, (i % 5)::real]::vec32
FROM generate_series(1, 500) i;

-- Or vec32_ip_ops / vec32_cosine_ops.
CREATE INDEX ON items USING prism (embedding vec32_l2_ops);

-- 0 = auto (~0.95 recall). Raise for recall, lower for speed.
SET prism.nprobe = 40;

SELECT * FROM items
ORDER BY embedding <-> '[3,1,2]'
LIMIT 5;
```

Defaults are tuned from large-scale benchmarks; most deployments only
ever adjust `prism.nprobe`.
See the [tuning guide][tuning-doc] for every index parameter and GUC,
their tradeoffs, and when changing them makes sense.

### Migrating a pgvector table to PRISM

If you already have a table with pgvector's `vector` or `halfvec`
columns, you can build a PRISM index on it directly. Nothing about the
table or your queries needs to change.

```sql
-- The table you already have, included so the example runs on its own.
CREATE EXTENSION vector;

CREATE TABLE documents (
    id        bigserial PRIMARY KEY,
    body      text,
    embedding vector(384)
);
CREATE INDEX documents_embedding_hnsw
    ON documents USING hnsw (embedding vector_l2_ops);

-- Install pg_vectorsearch next to pgvector and build the index on the
-- existing column. The column is not rewritten or copied.
CREATE EXTENSION pg_vectorsearch;

CREATE INDEX documents_embedding_prism
    ON documents USING prism (embedding vec32_l2_ops);

-- Your queries keep working as before. PRISM is normally faster, so
-- the planner should prefer it over the HNSW index while both exist.
-- Verify with an EXPLAIN:
EXPLAIN (COSTS OFF)
SELECT id, body
FROM documents
ORDER BY embedding <-> (SELECT embedding FROM documents WHERE id = 42)
LIMIT 10;
--  Limit
--    InitPlan 1
--      ->  Index Scan using documents_pkey on documents documents_1
--            Index Cond: (id = 42)
--    ->  Index Scan using documents_embedding_prism on documents
--          Order By: (embedding <-> (InitPlan 1).col1)

-- Once you are happy with it, drop the HNSW index.
DROP INDEX documents_embedding_hnsw;
```

Use `vec32_ip_ops` or `vec32_cosine_ops` for an HNSW index built with
`vector_ip_ops` or `vector_cosine_ops`, and the `vec16_*` classes for a
`halfvec` column.

## PRISM index maintenance

An insert appends to whichever posting list its vector routes to, so lists
grow as rows arrive and never split on their own. Besides `REINDEX`, the
`prism_rebalance` procedure splits oversized lists in place.
See the [maintenance guide][maint-doc] for how it works and its current
limitations.

## Documentation

- [Tuning][tuning-doc] - Index parameters, GUCs, tradeoffs, defaults
- [Maintenance][maint-doc] - Rebalancing and splitting posting lists
- [Architecture][arch-doc] - High-level design and data structures
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

[The PostgreSQL License](LICENSE)

<!-- Links -->
[pgvector]: https://github.com/pgvector/pgvector
[pgvectorscale]: https://github.com/timescale/pgvectorscale
[rabitq]: https://github.com/gaoj0017/RaBitQ
[spfresh]: https://dl.acm.org/doi/10.1145/3600006.3613166
[spann-paper]: https://www.microsoft.com/en-us/research/wp-content/uploads/2021/11/SPANN_finalversion1.pdf
[scann-alloydb]: https://services.google.com/fh/files/misc/scann_for_alloydb_whitepaper.pdf
[arch-doc]: docs/architecture.md
[tuning-doc]: docs/tuning.md
[maint-doc]: docs/maintenance.md
[simd-doc]: docs/simd.md
