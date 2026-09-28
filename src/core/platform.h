/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * platform.h - Platform abstraction and SIMD capability detection
 *
 * Provides runtime CPU feature detection and compiler intrinsics wrappers
 * for portable SIMD code.
 */

#ifndef VS_PLATFORM_H
#define VS_PLATFORM_H

#include <stdint.h>

/*
 * SIMD capability flags detected at runtime.
 * Multiple flags may be set (e.g., AVX512F implies AVX2 implies SSE4.1).
 */
typedef enum
{
	SIMD_NONE			  = 0,
	SIMD_SSE2			  = 1 << 0,
	SIMD_SSE4_1			  = 1 << 1,
	SIMD_AVX2			  = 1 << 2,
	SIMD_AVX512F		  = 1 << 3,
	SIMD_NEON			  = 1 << 4,
	SIMD_AVX512_VPOPCNTDQ = 1 << 5,
	SIMD_AVX512DQ		  = 1 << 6,
	SIMD_AVX512BW		  = 1 << 7,
} SimdCapability;

/*
 * The AVX-512 kernels are compiled for specific sub-extensions (see the
 * VS_TARGET_AVX512* attributes in simd_utils.h and the target attributes in
 * the *_avx512.c files), not plain AVX512F. Dispatch -- and the test/bench
 * overrides that force a path -- must require every bit the kernels use: a
 * CPU can implement F without BW or DQ (e.g. Knights Landing), and running
 * the compiled kernels there faults with SIGILL. VPOPCNTDQ kernels are gated
 * separately on SIMD_AVX512_VPOPCNTDQ (which already implies F).
 */
#define VS_SIMD_AVX512_DQ                                                    \
	(SIMD_AVX512F | SIMD_AVX512DQ)						 /* distance, rabitq \
														  */
#define VS_SIMD_AVX512_BW (SIMD_AVX512F | SIMD_AVX512BW) /* fastscan */

/*
 * Detect CPU SIMD capabilities at runtime.
 *
 * On x86/x64: Uses CPUID to detect SSE2, SSE4.1, AVX2, AVX512F.
 * On ARM: Returns SIMD_NEON if compiled with NEON support.
 *
 * Result is cached after first call.
 */
SimdCapability vs_detect_simd(void);

/*
 * Check if specific SIMD capability is available.
 */
static inline int
vs_has_simd(SimdCapability cap)
{
	return (vs_detect_simd() & cap) != 0;
}

/*
 * Check that ALL bits in mask are available. Use with the VS_SIMD_AVX512_*
 * masks so a multi-bit requirement (F + BW, F + DQ) is tested as a unit.
 */
static inline int
vs_has_all_simd(uint32_t mask)
{
	return ((uint32_t)vs_detect_simd() & mask) == mask;
}

/*
 * Override SIMD capability detection (for testing and benchmarking).
 *
 * Forces all SIMD-using code to use a specific instruction set by masking
 * detected CPU capabilities. This allows:
 * - Testing all SIMD code paths, including fallbacks
 * - Benchmarking different SIMD implementations
 * - Debugging SIMD-specific issues
 *
 * Parameters:
 *   mask: Bitwise OR of SimdCapability flags
 *         - 0 or SIMD_NONE: Force scalar implementation
 *         - SIMD_AVX2: Force AVX2 (on x86-64)
 *         - SIMD_AVX512F: Force AVX-512 (on x86-64)
 *         - SIMD_NEON: Force NEON (on ARM)
 *         - 0xFFFFFFFF: Auto-detect (default, clears override)
 *
 * Note: Call this BEFORE any SIMD detection occurs. Calling after
 * detection may require re-initialization of SIMD-using modules.
 *
 * Example usage:
 *   // Test AVX2 code path even on AVX-512 CPU
 *   vs_simd_set_override(SIMD_AVX2);
 *   // ... run tests or benchmarks ...
 *   vs_simd_set_override(0xFFFFFFFF);  // Reset to auto-detect
 */
void vs_simd_set_override(uint32_t mask);

/*
 * Clear SIMD detection cache, forcing re-detection.
 *
 * Useful when testing different SIMD overrides. Call this after
 * vs_simd_set_override() to ensure the new mask takes effect.
 */
void vs_simd_reset_cache(void);

/* Cache line size (typical for modern CPUs) */
#define VS_CACHE_LINE 64

/*
 * Prefetch hints for memory access optimization.
 *
 * vs_prefetch_read:  Prefetch for reading (non-temporal, keep in all caches)
 * vs_prefetch_write: Prefetch for writing (exclusive access)
 */
#define vs_prefetch_read(addr)	__builtin_prefetch((addr), 0, 3)
#define vs_prefetch_write(addr) __builtin_prefetch((addr), 1, 3)

/*
 * Branch prediction hints.
 *
 * Use sparingly - modern CPUs have good branch predictors.
 * Most useful for error paths that are rarely taken.
 */
/* Time-unit conversion factors for nanosecond-based instrumentation
 * (cf. PostgreSQL's NS_PER_S family in portability/instr_time.h; defined
 * here so shared, non-PG code can use them too). */
#define VS_NS_PER_SEC 1000000000ULL
#define VS_NS_PER_MS  1000000ULL
#define VS_NS_PER_US  1000ULL

#define vs_likely(x)   __builtin_expect(!!(x), 1)
#define vs_unlikely(x) __builtin_expect(!!(x), 0)

/*
 * Compiler memory barrier.
 *
 * Prevents compiler from reordering memory accesses across this point.
 * Does NOT generate CPU fence instructions (use atomics for that).
 */
#define vs_compiler_barrier() __asm__ __volatile__("" ::: "memory")

/*
 * Alignment helpers.
 */
#define VS_ALIGN(x, a)		(((x) + ((a) - 1)) & ~((a) - 1))
#define VS_IS_ALIGNED(x, a) (((uintptr_t)(x) & ((a) - 1)) == 0)

/*
 * SIMD vector alignment (64 bytes for AVX-512).
 */
#define VS_SIMD_ALIGN 64

#endif /* VS_PLATFORM_H */
