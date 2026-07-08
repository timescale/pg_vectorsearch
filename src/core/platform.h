/*
 * platform.h - Platform abstraction and SIMD capability detection
 *
 * Provides runtime CPU feature detection and compiler intrinsics wrappers
 * for portable SIMD code.
 */

#ifndef MKT_PLATFORM_H
#define MKT_PLATFORM_H

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
} SimdCapability;

/*
 * Detect CPU SIMD capabilities at runtime.
 *
 * On x86/x64: Uses CPUID to detect SSE2, SSE4.1, AVX2, AVX512F.
 * On ARM: Returns SIMD_NEON if compiled with NEON support.
 *
 * Result is cached after first call.
 */
SimdCapability mkt_detect_simd(void);

/*
 * Check if specific SIMD capability is available.
 */
static inline int
mkt_has_simd(SimdCapability cap)
{
	return (mkt_detect_simd() & cap) != 0;
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
 *   mkt_simd_set_override(SIMD_AVX2);
 *   // ... run tests or benchmarks ...
 *   mkt_simd_set_override(0xFFFFFFFF);  // Reset to auto-detect
 */
void mkt_simd_set_override(uint32_t mask);

/*
 * Clear SIMD detection cache, forcing re-detection.
 *
 * Useful when testing different SIMD overrides. Call this after
 * mkt_simd_set_override() to ensure the new mask takes effect.
 */
void mkt_simd_reset_cache(void);

/* Cache line size (typical for modern CPUs) */
#define MKT_CACHE_LINE 64

/*
 * Prefetch hints for memory access optimization.
 *
 * mkt_prefetch_read:  Prefetch for reading (non-temporal, keep in all caches)
 * mkt_prefetch_write: Prefetch for writing (exclusive access)
 */
#define mkt_prefetch_read(addr)	 __builtin_prefetch((addr), 0, 3)
#define mkt_prefetch_write(addr) __builtin_prefetch((addr), 1, 3)

/*
 * Branch prediction hints.
 *
 * Use sparingly - modern CPUs have good branch predictors.
 * Most useful for error paths that are rarely taken.
 */
/* Time-unit conversion factors for nanosecond-based instrumentation
 * (cf. PostgreSQL's NS_PER_S family in portability/instr_time.h; defined
 * here so shared, non-PG code can use them too). */
#define MKT_NS_PER_SEC 1000000000ULL
#define MKT_NS_PER_MS  1000000ULL
#define MKT_NS_PER_US  1000ULL

#define mkt_likely(x)	__builtin_expect(!!(x), 1)
#define mkt_unlikely(x) __builtin_expect(!!(x), 0)

/*
 * Compiler memory barrier.
 *
 * Prevents compiler from reordering memory accesses across this point.
 * Does NOT generate CPU fence instructions (use atomics for that).
 */
#define mkt_compiler_barrier() __asm__ __volatile__("" ::: "memory")

/*
 * Alignment helpers.
 */
#define MKT_ALIGN(x, a)		 (((x) + ((a) - 1)) & ~((a) - 1))
#define MKT_IS_ALIGNED(x, a) (((uintptr_t)(x) & ((a) - 1)) == 0)

/*
 * SIMD vector alignment (64 bytes for AVX-512).
 */
#define MKT_SIMD_ALIGN 64

#endif /* MKT_PLATFORM_H */
