# Third-Party Patches

Patches for external dependencies used by pg_vectorsearch.

## faiss-rabitq-c-api.patch

Adds C API bindings for FAISS RaBitQuantizer. Required for the benchmark
comparison in `vectorsearch bench quantize`.

**Apply to FAISS:**

```bash
cd /path/to/faiss
git apply /path/to/pg_vectorsearch/third-party/faiss-rabitq-c-api.patch
```

**Files added:**

- `c_api/RaBitQuantizer_c.h` - C header with RaBitQuantizer bindings
- `c_api/RaBitQuantizer_c.cpp` - C wrapper implementation
- `c_api/CMakeLists.txt` - Modified to include new source

**Rebuild FAISS after applying:**

```bash
cd /path/to/faiss/build
cmake --build . --target faiss_c
```
