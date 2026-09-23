/*
 * distance.h - High-performance distance computation for vectors
 *
 * Provides SIMD-accelerated distance metrics (L2, inner product, cosine)
 * with runtime CPU detection and automatic dispatch to optimal
 * implementations.
 *
 * Features:
 * - Single-pair and batch distance operations
 * - Runtime SIMD detection (AVX-512, AVX2, NEON, or compiler fallback)
 * - Zero-overhead function pointer dispatch after initialization
 * - Support for full-precision floating-point vectors
 *
 * Build modes (set via meson -Dsimd=):
 * - full (default): Hand-optimized SIMD + compiler-vectorized baseline
 * - compiler: Compiler-vectorized only (target_clones for ISA selection)
 * - none: Truly scalar (no vectorization, for debugging/baseline)
 *
 * Usage:
 *   Vec32Ref a = {.data = vec1, .dim = 128};
 *   Vec32Ref b = {.data = vec2, .dim = 128};
 *
 *   Distance d = vs_distance_l2(a, b);
 *
 * For batch operations (one query against many database vectors):
 *   Distance *dists = malloc(count * sizeof(Distance));
 *   vs_distance_batch_l2(query, db_vectors, count, dim, dists);
 */

#ifndef VS_DISTANCE_H
#define VS_DISTANCE_H

#include "vs_config.h"

#include "core/types.h"

/*
 * Initialize distance computation system.
 *
 * Detects CPU capabilities and sets up function pointer dispatch.
 * Called automatically on first use (lazy initialization), but can be
 * called explicitly during startup for deterministic performance.
 *
 * Returns 0 on success, non-zero on error (though initialization cannot
 * currently fail).
 *
 * Thread-safe: uses atomic flag to ensure single initialization.
 *
 * Note: To test different SIMD implementations, use vs_simd_set_override()
 * from platform.h before calling this function.
 */
int vs_distance_init(void);

/*
 * Single-Pair Distance Functions
 *
 * Compute distance between two vectors. Returns -1.0f on error (dimension
 * mismatch, null pointers, zero dimension).
 */

/*
 * L2 squared distance (Euclidean distance squared).
 *
 * Returns: sum((a[i] - b[i])^2 for i in 0..dim-1)
 *
 * Note: Returns L2 squared, not L2. For ranking, squared distance is
 * sufficient and avoids a sqrt operation. If you need true L2 distance,
 * apply sqrtf() to the result.
 */
Distance vs_distance_l2(Vec32Ref a, Vec32Ref b);

/*
 * Negative inner product (for maximum similarity search).
 *
 * Returns: -sum(a[i] * b[i] for i in 0..dim-1)
 *
 * Note: Negated so that smaller values indicate higher similarity,
 * consistent with distance metrics. To get the actual inner product,
 * negate the result.
 */
Distance vs_distance_ip(Vec32Ref a, Vec32Ref b);

/*
 * Cosine distance (1 - cosine similarity).
 *
 * Returns: 1.0 - (dot(a, b) / (norm(a) * norm(b)))
 *
 * Returns 1.0 for zero vectors (maximum distance). Range is [0, 2] for
 * arbitrary vectors, [0, 1] for normalized vectors.
 */
Distance vs_distance_cosine(Vec32Ref a, Vec32Ref b);

/*
 * Generic distance function with runtime metric selection.
 *
 * Convenient wrapper when the metric is determined at runtime. Slightly
 * less efficient than calling the specific function directly (extra
 * branch), but useful for generic code.
 */
Distance vs_distance(Vec32Ref a, Vec32Ref b, DistanceMetric metric);

/*
 * Batch Distance Functions
 *
 * Compute distances between one query vector and multiple database vectors.
 * More efficient than repeated single-pair calls due to:
 * - Query data stays in cache
 * - Prefetching of database vectors
 * - Better branch prediction
 *
 * Parameters:
 * - query: Query vector
 * - vectors: Array of database vectors (flattened, row-major order)
 * - count: Number of database vectors
 * - dim: Dimension of all vectors (must match query.dim)
 * - distances: Output array (must have space for count elements)
 *
 * Returns: 0 on success, -1 on error (dimension mismatch, null pointers)
 */
int vs_distance_batch_l2(
		Vec32Ref	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances);

int vs_distance_batch_ip(
		Vec32Ref	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances);

int vs_distance_batch_cosine(
		Vec32Ref	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances);

/*
 * Query current SIMD implementation name.
 *
 * Returns: String like "avx512", "avx2", "neon", "compiler", or "none"
 *
 * Useful for debugging and performance validation. The returned string
 * is valid for the lifetime of the program (static storage).
 */
const char *vs_distance_impl_name(void);

/*
 * Reset distance system to uninitialized state (TEST ONLY).
 *
 * Used in conjunction with vs_simd_set_override() to test different
 * SIMD implementations.
 *
 * Example:
 *   vs_simd_set_override(SIMD_AVX2);
 *   vs_distance_force_reinit();  // Will now use AVX2
 */
void vs_distance_force_reinit(void);

/*
 * Implementation-specific functions for benchmarking and testing.
 *
 * These bypass the dispatch mechanism and call specific implementations
 * directly. Useful for accurate performance comparisons.
 *
 * Compiler-vectorized (always available):
 * Uses target_clones for ISA selection on x86. Falls back to basic
 * vectorization on other architectures.
 */

/* Compiler-vectorized single-pair functions */
Distance vs_distance_l2_compiler(Vec32Ref a, Vec32Ref b);
Distance vs_distance_ip_compiler(Vec32Ref a, Vec32Ref b);
Distance vs_distance_cosine_compiler(Vec32Ref a, Vec32Ref b);

/* Compiler-vectorized batch functions */
int vs_distance_batch_l2_compiler(
		Vec32Ref	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances);
int vs_distance_batch_ip_compiler(
		Vec32Ref	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances);
int vs_distance_batch_cosine_compiler(
		Vec32Ref	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances);

/*
 * Hand-optimized SIMD implementations (simd=full mode only).
 *
 * These are only available when built with -Dsimd=full (the default).
 * They provide the best performance on supported hardware.
 */

#ifdef VS_SIMD_FULL

#if defined(__x86_64__) || defined(_M_X64)
/* AVX2 implementations */
Distance vs_distance_l2_avx2(Vec32Ref a, Vec32Ref b);
Distance vs_distance_ip_avx2(Vec32Ref a, Vec32Ref b);
Distance vs_distance_cosine_avx2(Vec32Ref a, Vec32Ref b);

int vs_distance_batch_l2_avx2(
		Vec32Ref	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances);
int vs_distance_batch_ip_avx2(
		Vec32Ref	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances);
int vs_distance_batch_cosine_avx2(
		Vec32Ref	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances);

/* AVX-512 implementations */
Distance vs_distance_l2_avx512(Vec32Ref a, Vec32Ref b);
Distance vs_distance_ip_avx512(Vec32Ref a, Vec32Ref b);
Distance vs_distance_cosine_avx512(Vec32Ref a, Vec32Ref b);

int vs_distance_batch_l2_avx512(
		Vec32Ref	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances);
int vs_distance_batch_ip_avx512(
		Vec32Ref	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances);
int vs_distance_batch_cosine_avx512(
		Vec32Ref	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances);
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
/* NEON implementations */
Distance vs_distance_l2_neon(Vec32Ref a, Vec32Ref b);
Distance vs_distance_ip_neon(Vec32Ref a, Vec32Ref b);
Distance vs_distance_cosine_neon(Vec32Ref a, Vec32Ref b);

int vs_distance_batch_l2_neon(
		Vec32Ref	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances);
int vs_distance_batch_ip_neon(
		Vec32Ref	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances);
int vs_distance_batch_cosine_neon(
		Vec32Ref	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances);
#endif

#endif /* VS_SIMD_FULL */

#endif /* VS_DISTANCE_H */
