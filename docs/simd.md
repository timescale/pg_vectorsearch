# SIMD Architecture

This document describes Meerkat's SIMD (Single Instruction, Multiple Data)
implementation for high-performance vector distance computation.

## Overview

Vector search performance depends heavily on distance computation speed.
Computing millions of distance calculations per second requires leveraging
modern CPU SIMD capabilities:

- **AVX-512**: 16 floats per instruction (512 bits)
- **AVX2**: 8 floats per instruction (256 bits)
- **NEON**: 4 floats per instruction (128 bits, ARM)

Meerkat provides both hand-optimized SIMD implementations and
compiler-vectorized fallbacks, with runtime dispatch to select the best
available implementation.

## Build Options

### SIMD Mode (`-Dsimd=`)

| Mode | Description | Use Case |
|------|-------------|----------|
| `full` (default) | Hand-optimized + compiler baseline | Production |
| `compiler` | Compiler vectorization only | Benchmark compiler vs hand-opt |
| `none` | Truly scalar (no vectorization) | Debugging, baseline |

### Native Build (`-Dnative=`)

| Value | Description | Use Case |
|-------|-------------|----------|
| `false` (default) | Portable binaries | Distribution, CI |
| `true` | `-march=native` optimization | Local benchmarking |

### Build Examples

```bash
# Default: portable build with full SIMD
meson setup builddir

# Compiler vectorization only (benchmark vs hand-optimized)
meson setup builddir-compiler -Dsimd=compiler

# No SIMD (debugging baseline)
meson setup builddir-none -Dsimd=none

# Native build for benchmarking
meson setup builddir-bench -Dnative=true
```

## Architecture

```
                    Public API (distance.h)
              mkt_distance_l2(), mkt_distance_ip()
                              │
                        IFUNC resolver
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│   AVX-512     │    │    AVX2       │    │   Compiler    │
│ (hand-opt)    │    │  (hand-opt)   │    │ target_clones │
│ __attribute__ │    │ __attribute__ │    │ ALWAYS built  │
│ target(avx512)│    │ target(avx2)  │    │               │
└───────────────┘    └───────────────┘    └───────────────┘
     ^                     ^                     │
     └─────────────────────┴─────────────────────┘
      Only when simd=full (default)      Always baseline
```

### Dispatch Priority

In `simd=full` mode:
1. Check CPU capabilities at load time (via IFUNC)
2. Select hand-optimized implementation if available (AVX-512 > AVX2 > NEON)
3. Fall back to compiler-vectorized baseline otherwise

In `simd=compiler` mode:
- Always use compiler-vectorized (target_clones handles ISA selection)

In `simd=none` mode:
- Use compiler baseline without vectorization flags

## Implementation Tiers

### Hand-Optimized SIMD

Located in `src/algo/distance_*.c`:
- `distance_avx512.c`: AVX-512F implementations
- `distance_avx2.c`: AVX2+FMA implementations
- `distance_neon.c`: ARM NEON implementations

These files use per-function target attributes:

```c
/* distance_avx512.c */
__attribute__((target("avx512f,avx512dq")))
Distance mkt_distance_l2_avx512(Vec32Ref a, Vec32Ref b)
{
    /* AVX-512 intrinsics */
}
```

This allows compilation as regular source files without separate compiler
flags, simplifying the build system. Per-function attributes are preferred
over file-level pragmas for better IDE/clangd compatibility.

### Compiler-Vectorized Baseline

Located in `src/algo/distance.c`, uses `target_clones` for multi-versioning:

```c
#define MKT_TARGET_CLONES \
    __attribute__((target_clones("default", "arch=x86-64-v3", "arch=x86-64-v4")))

MKT_TARGET_CLONES static float
compiler_l2_loop(int dim, const float *pa, const float *pb)
{
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = pa[i] - pb[i];
        sum += diff * diff;
    }
    return sum;
}
```

The compiler generates multiple function versions, and the dynamic linker
selects the best one at load time.

**Why `arch=x86-64-v3/v4` instead of `avx2/avx512f`:**

The `arch=` specifiers bundle all features for a microarchitecture level:

| Clone | Equivalent features |
|-------|-------------------|
| `arch=x86-64-v3` | AVX2, FMA, BMI1/2, F16C, ... |
| `arch=x86-64-v4` | AVX-512F, AVX-512BW/DQ/VL, FMA, ... |
| `avx512f` (old) | AVX-512F only — **no FMA implied** |

Using `avx512f` alone generates separate `vmulps` + `vaddps` instead of fused
`vfmadd231ps`, roughly halving throughput for dot-product loops.

## Compiler Support

### target_clones

| Compiler | Version | Support |
|----------|---------|---------|
| GCC | 6.0+ | Full x86 support |
| Clang | 13.0+ | Full x86 support |

Detection uses `__has_attribute(target_clones)`.

### Target Attributes

Per-function `__attribute__((target(...)))` is supported by both GCC and Clang,
providing consistent behavior across compilers without conditional compilation.

### FP Contraction and FMA

GCC's `-ffp-contract` flag controls whether the compiler may fuse `a * b + c`
into a single FMA instruction (`vfmadd231ps`). The default depends on the
language standard mode:

| Flag | Default `-ffp-contract` | FMA generated? |
|------|------------------------|----------------|
| `-std=c23` / `-std=c2x` | `off` | No |
| `-std=gnu23` / `-std=gnu2x` | `fast` | Yes |
| (no `-std`) | `fast` | Yes |

ISO C modes default to `off` because the C standard leaves FP contraction
implementation-defined, and contraction changes rounding behavior (FMA rounds
once instead of twice). This matters for strict numerical reproducibility but
not for approximate algorithms like k-means clustering or ANN search.

The project uses `-std=c2x` for C23 features, so we explicitly pass
`-ffp-contract=fast` in `meson.build` to restore FMA generation. Without it,
GCC generates separate multiply + add instructions, and for nested loops
(like batch dot products), also fails to vectorize the reduction properly —
producing horizontal scalar adds per vector chunk instead of accumulating in
a wide register and reducing once.

See: [GCC Optimize Options](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html)

## ARM Considerations

### target_clones on ARM

`target_clones` does NOT work effectively on ARM:
- Generates only a single clone (no multi-versioning)
- NEON is mandatory on AArch64, so there's nothing to dispatch

### ARM SIMD Strategy

| Mode | Behavior |
|------|----------|
| `full` | Use hand-optimized NEON from `distance_neon.c` |
| `compiler` | Compiler vectorizes using NEON via `-ftree-vectorize` |
| `none` | Truly scalar (no `-ftree-vectorize`) |

### 32-bit ARM

For ARMv7 (32-bit ARM), NEON isn't always available. The NEON file is
compiled as a separate library with `-mfpu=neon` to handle this case.

## Runtime Dispatch

### IFUNC (Zero Overhead)

On GNU systems, IFUNC provides zero-overhead dispatch:

```c
Distance mkt_distance_l2(Vec32Ref a, Vec32Ref b)
    __attribute__((ifunc("resolve_distance_l2")));
```

The resolver runs once at program load, binding the symbol directly to
the selected implementation.

### Function Pointer Fallback

For non-GNU systems or testing, function pointers provide equivalent
functionality with minimal overhead (~1 indirect call):

```c
static DistanceFn g_distance_l2_fn = NULL;

Distance mkt_distance_l2(Vec32Ref a, Vec32Ref b)
{
    if (mkt_unlikely(!g_initialized))
        mkt_distance_init();
    return g_distance_l2_fn(a, b);
}
```

## Benchmarking

### pgvector Comparison

The benchmark includes a `pgvector` implementation that matches upstream
pgvector's actual code. Interestingly, `compiler` is often faster than
`pgvector` despite both using auto-vectorization. The difference is in
`target_clones`:

| Implementation | target_clones | Result |
|----------------|---------------|--------|
| pgvector | `"default", "fma"` | FMA only, may use 128-bit SSE |
| compiler | `"default", "arch=x86-64-v3", "arch=x86-64-v4"` | Full 256/512-bit + FMA |

**Why pgvector uses conservative settings:**

pgvector uses `target_clones("default", "fma")` which only enables FMA
(Fused Multiply-Add) instructions. FMA computes `a * b + c` in one operation,
but doesn't imply full AVX2 vectorization. The compiler may still use 128-bit
SSE registers.

Our `compiler` implementation uses
`target_clones("default", "arch=x86-64-v3", "arch=x86-64-v4")` which generates
versions using 256-bit (AVX2+FMA) and 512-bit (AVX-512+FMA) registers, enabling
both wider vectorization and fused multiply-add.

**Benchmark interpretation:**

- **pgvector**: What pgvector actually does (conservative)
- **compiler**: What auto-vectorization *could* achieve with better settings
- **avx2/avx512**: Hand-optimized SIMD (best performance)

The gap between `pgvector` and `compiler` shows performance left on the table
by pgvector's target_clones choice.

### Direct Implementation Calls

For accurate benchmarking, call implementations directly:

```c
// Compiler-vectorized (always available)
mkt_distance_batch_l2_compiler(query, vectors, count, dim, distances);

// Hand-optimized (simd=full only)
mkt_distance_batch_l2_avx512(query, vectors, count, dim, distances);
mkt_distance_batch_l2_avx2(query, vectors, count, dim, distances);
mkt_distance_batch_l2_neon(query, vectors, count, dim, distances);
```

### CLI Benchmarks

```bash
# Build with native optimizations for fair comparison
meson setup builddir-bench -Dsimd=full -Dnative=true
meson compile -C builddir-bench

# Compare implementations
./builddir-bench/mkt bench distance --dim 768 --count 10000 --impls compiler
./builddir-bench/mkt bench distance --dim 768 --count 10000 --impls avx2
./builddir-bench/mkt bench distance --dim 768 --count 10000 --impls avx512
```

### Comparing Build Modes

```bash
# Build both modes
meson setup builddir-full -Dsimd=full -Dnative=true
meson setup builddir-compiler -Dsimd=compiler -Dnative=true
meson compile -C builddir-full
meson compile -C builddir-compiler

# Compare hand-optimized vs compiler
./builddir-full/mkt bench distance --dim 768 --count 10000
./builddir-compiler/mkt bench distance --dim 768 --count 10000
```

## Verification

Check active implementation:

```bash
./builddir-full/mkt info
# Output: "avx512" or "avx2" (depending on CPU)

./builddir-compiler/mkt info
# Output: "compiler"

./builddir-none/mkt info
# Output: "none"
```

## Future Work

### SVE (Scalable Vector Extension)

ARM SVE provides variable-width vectors (128-2048 bits). Future support
would add:

1. `distance_sve.c` with `#pragma GCC target("sve")`
2. Runtime detection via `getauxval(AT_HWCAP)`
3. IFUNC resolver selecting SVE > NEON > compiler

### Additional Optimizations

- Multiple accumulators to hide FMA latency
- Prefetching for batch operations
- Alignment hints for cache efficiency
