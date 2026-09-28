/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * simd_utils.h - Shared SIMD utilities and patterns
 *
 * Provides reusable SIMD infrastructure for distance computations,
 * quantization, and other vectorized operations. This header contains:
 *
 * - Horizontal reduction functions for all SIMD variants
 * - Dispatch macro templates for runtime CPU detection
 * - Common SIMD patterns (prefetch, alignment)
 * - Generic batch processing helpers
 *
 * Design rationale: By centralizing SIMD utilities, we enable ~50-60% code
 * reuse across full-precision, binary, and product quantization distance
 * implementations. Each distance type needs different intrinsics, but shares
 * the infrastructure.
 */

#ifndef VS_SIMD_UTILS_H
#define VS_SIMD_UTILS_H

/* Must be first - defines VS_SIMD_NONE used by VS_TARGET_CLONES */
#include "vs_config.h"

#include <stdatomic.h>
#include <stdint.h>

#include "core/platform.h"

/*
 * Target Attribute Macros
 *
 * These macros specify the CPU features required for each SIMD implementation.
 * Using macros makes it easy to change target features in one place.
 */
#if defined(__x86_64__) || defined(_M_X64)
#define VS_TARGET_AVX512 __attribute__((target("avx512f,avx512dq")))
#define VS_TARGET_AVX512_VPOPCNTDQ \
	__attribute__((target("avx512f,avx512vpopcntdq")))
#define VS_TARGET_AVX2		__attribute__((target("avx2,fma")))
#define VS_TARGET_F16C_AVX2 __attribute__((target("avx2,fma,f16c")))
#elif defined(__aarch64__) || defined(_M_ARM64)
/* NEON is always available on AArch64, no attribute needed */
#define VS_TARGET_NEON
#endif

/*
 * Target Clones Macro
 *
 * Generates multiple function versions for different ISAs. The dynamic linker
 * selects the best version at load time. Only effective on x86 with GCC 6+ or
 * Clang 13+.
 *
 * Note: target_clones doesn't work effectively on ARM (generates single
 * clone).
 */
#ifndef __has_attribute
#define __has_attribute(x) 0
#endif

/*
 * Disable target_clones when:
 * - simd=none (VS_SIMD_NONE)
 * - coverage build (VS_COVERAGE) - each clone is separate, only one executes
 * - compiler lacks target_clones support
 * - non-x86 architecture (ARM target_clones generates single clone anyway)
 */
#if !defined(VS_SIMD_NONE) && !defined(VS_COVERAGE) && \
		__has_attribute(target_clones) &&              \
		(defined(__x86_64__) || defined(__i386__))
#define VS_TARGET_CLONES \
	__attribute__((      \
			target_clones("default", "arch=x86-64-v3", "arch=x86-64-v4")))
#else
#define VS_TARGET_CLONES
#endif

/*
 * Horizontal Reduction Functions
 *
 * These functions reduce a SIMD vector to a single scalar value by summing
 * all elements. Used at the end of SIMD loops to extract the final result.
 */

#if defined(__x86_64__) || defined(_M_X64)

#include <immintrin.h>

/*
 * AVX-512 horizontal sum (16 floats -> 1 float)
 *
 * Uses _mm512_reduce_add_ps intrinsic (AVX-512F).
 */
VS_TARGET_AVX512 static inline float
vs_horizontal_sum_avx512(__m512 v)
{
	return _mm512_reduce_add_ps(v);
}

/*
 * AVX-512 horizontal sum for 64-bit integers (8 uint64_t -> 1 uint64_t)
 *
 * Used for Hamming distance (popcount results).
 */
VS_TARGET_AVX512 static inline uint64_t
vs_horizontal_sum_epi64_avx512(__m512i v)
{
	return _mm512_reduce_add_epi64(v);
}

/*
 * AVX2 horizontal sum (8 floats -> 1 float)
 *
 * AVX2 lacks a reduce intrinsic, so we manually combine the two 128-bit
 * halves and use horizontal adds within each lane.
 */
VS_TARGET_AVX2 static inline float
vs_horizontal_sum_avx2(__m256 v)
{
	/* Extract high and low 128-bit halves */
	__m128 lo = _mm256_castps256_ps128(v);
	__m128 hi = _mm256_extractf128_ps(v, 1);

	/* Add halves */
	__m128 sum = _mm_add_ps(lo, hi);

	/* Horizontal add within 128 bits (twice to reduce to single element) */
	sum = _mm_hadd_ps(sum, sum);
	sum = _mm_hadd_ps(sum, sum);

	/* Extract scalar */
	return _mm_cvtss_f32(sum);
}

/*
 * AVX2 horizontal sum for 64-bit integers (4 uint64_t -> 1 uint64_t)
 */
VS_TARGET_AVX2 static inline uint64_t
vs_horizontal_sum_epi64_avx2(__m256i v)
{
	__m128i lo	= _mm256_castsi256_si128(v);
	__m128i hi	= _mm256_extracti128_si256(v, 1);
	__m128i sum = _mm_add_epi64(lo, hi);

	/* Extract two 64-bit values and add */
	uint64_t a = (uint64_t)_mm_extract_epi64(sum, 0);
	uint64_t b = (uint64_t)_mm_extract_epi64(sum, 1);
	return a + b;
}

#endif /* x86_64 */

#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>

/*
 * NEON horizontal sum (4 floats -> 1 float)
 *
 * Uses vaddvq_f32 (ARMv8.1+ reduction intrinsic).
 */
static inline float
vs_horizontal_sum_neon(float32x4_t v)
{
	return vaddvq_f32(v);
}

/*
 * NEON horizontal sum for 64-bit integers (2 uint64_t -> 1 uint64_t)
 */
static inline uint64_t
vs_horizontal_sum_u64_neon(uint64x2_t v)
{
	return vaddvq_u64(v);
}

#endif /* aarch64 */

/*
 * Dispatch Macros
 *
 * These macros simplify runtime CPU detection and function pointer dispatch.
 * Use DECLARE_DISPATCH to define the function pointer type and global state,
 * then INIT_DISPATCH to initialize based on detected CPU capabilities.
 *
 * Example usage:
 *
 *   // In header:
 *   Distance my_distance_fn(Vec32Ref a, Vec32Ref b);
 *
 *   // In implementation:
 *   DECLARE_DISPATCH(my_distance, Distance, Vec32Ref, Vec32Ref)
 *
 *   INIT_DISPATCH(my_distance,
 *                 my_distance_scalar,
 *                 my_distance_avx512,
 *                 my_distance_avx2,
 *                 my_distance_neon)
 *
 *   Distance my_distance_fn(Vec32Ref a, Vec32Ref b) {
 *       if (vs_unlikely(!g_my_distance_initialized))
 *           my_distance_init();
 *       return g_my_distance_fn(a, b);
 *   }
 */

/*
 * DECLARE_DISPATCH - Declare function pointer type and global state
 *
 * Creates:
 * - typedef for the function pointer type
 * - static global function pointer (initially NULL)
 * - static atomic initialization flag
 */
#define DECLARE_DISPATCH(name, ret_type, ...)                \
	typedef ret_type (*name##_fn_t)(__VA_ARGS__);            \
	static name##_fn_t	   g_##name##_fn		  = NULL;    \
	static _Atomic(bool)   g_##name##_initialized = false;   \
	static inline int	   name##_init(void);                \
	static inline ret_type name##_dispatch(__VA_ARGS__);     \
                                                             \
	static inline ret_type name##_dispatch(__VA_ARGS__ args) \
	{                                                        \
		if (vs_unlikely(!g_##name##_initialized))            \
			name##_init();                                   \
		return g_##name##_fn(args);                          \
	}

/*
 * INIT_DISPATCH - Initialize function pointer based on CPU capabilities
 *
 * Parameters:
 * - name: base name (matches DECLARE_DISPATCH)
 * - scalar: scalar fallback function
 * - avx512: AVX-512 implementation (or NULL)
 * - avx2: AVX2 implementation (or NULL)
 * - neon: NEON implementation (or NULL)
 *
 * Selects the best available implementation based on runtime CPU detection.
 * Falls back to scalar if no SIMD implementation is available.
 */
#define INIT_DISPATCH(name, scalar, avx512, avx2, neon)                    \
	static inline int name##_init(void)                                    \
	{                                                                      \
		if (g_##name##_initialized)                                        \
			return 0;                                                      \
                                                                           \
		SimdCapability caps = vs_detect_simd();                            \
                                                                           \
		/* Require every AVX-512 sub-extension any kernel                  \
		 * here may use (F+DQ+BW), not just F -- see                       \
		 * VS_SIMD_AVX512_* in platform.h. */                              \
		SimdCapability avx512_req = VS_SIMD_AVX512_DQ | VS_SIMD_AVX512_BW; \
		if (((caps & avx512_req) == avx512_req) && (avx512) != NULL)       \
			g_##name##_fn = avx512;                                        \
		else if ((caps & SIMD_AVX2) && (avx2) != NULL)                     \
			g_##name##_fn = avx2;                                          \
		else if ((caps & SIMD_NEON) && (neon) != NULL)                     \
			g_##name##_fn = neon;                                          \
		else                                                               \
			g_##name##_fn = scalar;                                        \
                                                                           \
		g_##name##_initialized = true;                                     \
		return 0;                                                          \
	}

/*
 * Batch Processing Helpers
 *
 * Common patterns for processing multiple vectors in a loop with prefetching.
 */

/*
 * Prefetch distance (in vectors) for batch operations.
 *
 * Prefetch 2 vectors ahead to hide memory latency. Assumes 64-byte cache
 * lines and typical vector sizes (128-1024 dimensions * 4 bytes = 0.5-4 KB).
 */
#define VS_PREFETCH_DISTANCE 2

/*
 * BATCH_LOOP_WITH_PREFETCH - Generic batch loop template
 *
 * Usage:
 *   BATCH_LOOP_WITH_PREFETCH(v, count, dim, vectors) {
 *       Vec32Ref vec = {.data = vectors + v * dim, .dim = dim};
 *       distances[v] = compute_distance(query, vec);
 *   }
 */
#define BATCH_LOOP_WITH_PREFETCH(idx, count, dim, vectors) \
	for (uint32_t idx = 0; idx < (count); (idx)++)         \
	{                                                      \
		if ((idx) + VS_PREFETCH_DISTANCE < (count))        \
			vs_prefetch_read(                              \
					(vectors) + ((idx) + VS_PREFETCH_DISTANCE) * (dim));

#define BATCH_LOOP_END }

#endif /* VS_SIMD_UTILS_H */
