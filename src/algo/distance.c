/*
 * distance.c - Distance computation with SIMD dispatch
 *
 * Implements runtime CPU detection and function pointer dispatch for
 * optimal SIMD implementations, with compiler-vectorized fallback.
 *
 * SIMD modes (set via meson -Dsimd=):
 * - full:     Hand-optimized SIMD with compiler-vectorized fallback (default)
 * - compiler: Compiler-vectorized only (target_clones for ISA selection)
 * - none:     Truly scalar (no vectorization, for debugging/baseline)
 */

/* Include generated config first for SIMD mode defines */
#include "mkt_config.h"

#include <math.h>
#include <stddef.h>

#include "algo/distance.h"
#include "algo/simd_utils.h"
#include "core/platform.h"

/*
 * Forward declarations for hand-optimized SIMD implementations.
 * These are only available when MKT_SIMD_FULL is defined (simd=full mode).
 */

#ifdef MKT_SIMD_FULL

#if defined(__x86_64__) || defined(_M_X64)
/* AVX-512 implementations (x86-64 only) */
Distance mkt_distance_l2_avx512(VectorRef a, VectorRef b);
Distance mkt_distance_ip_avx512(VectorRef a, VectorRef b);
Distance mkt_distance_cosine_avx512(VectorRef a, VectorRef b);

int mkt_distance_batch_l2_avx512(
		VectorRef	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances);
int mkt_distance_batch_ip_avx512(
		VectorRef	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances);
int mkt_distance_batch_cosine_avx512(
		VectorRef	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances);

/* AVX2 implementations (x86-64 only) */
Distance mkt_distance_l2_avx2(VectorRef a, VectorRef b);
Distance mkt_distance_ip_avx2(VectorRef a, VectorRef b);
Distance mkt_distance_cosine_avx2(VectorRef a, VectorRef b);

int mkt_distance_batch_l2_avx2(
		VectorRef	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances);
int mkt_distance_batch_ip_avx2(
		VectorRef	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances);
int mkt_distance_batch_cosine_avx2(
		VectorRef	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances);
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
/* NEON implementations (ARM only) */
Distance mkt_distance_l2_neon(VectorRef a, VectorRef b);
Distance mkt_distance_ip_neon(VectorRef a, VectorRef b);
Distance mkt_distance_cosine_neon(VectorRef a, VectorRef b);

int mkt_distance_batch_l2_neon(
		VectorRef	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances);
int mkt_distance_batch_ip_neon(
		VectorRef	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances);
int mkt_distance_batch_cosine_neon(
		VectorRef	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances);
#endif

#endif /* MKT_SIMD_FULL */

/*
 * Compiler-Vectorized Implementations
 *
 * These use target_clones to generate multiple versions for different ISAs.
 * The dynamic linker selects the best version at load time.
 *
 * These serve as:
 * 1. Fallback when hand-optimized SIMD isn't available for the CPU
 * 2. The only implementation when simd=compiler
 * 3. Reference for comparing hand-optimized vs compiler-generated code
 */

/* Static helper with target_clones (MKT_TARGET_CLONES defined in simd_utils.h)
 */
MKT_TARGET_CLONES static float
compiler_l2_loop(int dim, const float *pa, const float *pb)
{
	float sum = 0.0f;
	for (int i = 0; i < dim; i++)
	{
		float diff = pa[i] - pb[i];
		sum += diff * diff;
	}
	return sum;
}

Distance
mkt_distance_l2_compiler(VectorRef a, VectorRef b)
{
	if (mkt_unlikely(
				a.dim != b.dim || a.dim == 0 || a.data == NULL ||
				b.data == NULL))
		return -1.0f;

	return compiler_l2_loop(a.dim, a.data, b.data);
}

/* Static helper with target_clones */
MKT_TARGET_CLONES static float
compiler_ip_loop(int dim, const float *pa, const float *pb)
{
	float sum = 0.0f;
	for (int i = 0; i < dim; i++)
		sum += pa[i] * pb[i];
	return sum;
}

Distance
mkt_distance_ip_compiler(VectorRef a, VectorRef b)
{
	if (mkt_unlikely(
				a.dim != b.dim || a.dim == 0 || a.data == NULL ||
				b.data == NULL))
		return -1.0f;

	return -compiler_ip_loop(a.dim, a.data, b.data);
}

/* Static helper with target_clones - returns similarity (not distance) */
MKT_TARGET_CLONES static double
compiler_cosine_similarity(int dim, const float *pa, const float *pb)
{
	float dot	 = 0.0f;
	float norm_a = 0.0f;
	float norm_b = 0.0f;

	/* Auto-vectorized - match pgvector style exactly */
	for (int i = 0; i < dim; i++)
	{
		dot += pa[i] * pb[i];
		norm_a += pa[i] * pa[i];
		norm_b += pb[i] * pb[i];
	}

	/* Use sqrt(a * b) over sqrt(a) * sqrt(b), with double precision */
	return (double)dot / sqrt((double)norm_a * (double)norm_b);
}

Distance
mkt_distance_cosine_compiler(VectorRef a, VectorRef b)
{
	if (mkt_unlikely(
				a.dim != b.dim || a.dim == 0 || a.data == NULL ||
				b.data == NULL))
		return -1.0f;

	float similarity = compiler_cosine_similarity(a.dim, a.data, b.data);

	/* Handle edge cases */
	if (isnan(similarity) || similarity < -1.0f)
		return 1.0f; /* Maximum distance for zero/invalid vectors */
	if (similarity > 1.0f)
		similarity = 1.0f;

	return 1.0f - similarity;
}

/*
 * Compiler-Vectorized Batch Implementations
 *
 * Dedicated batch functions that avoid function call overhead in the inner
 * loop.
 */

int
mkt_distance_batch_l2_compiler(
		VectorRef	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances)
{
	if (mkt_unlikely(
				query.dim != dim || query.data == NULL || vectors == NULL ||
				distances == NULL))
		return -1;

	const float *q = query.data;

	for (uint32_t i = 0; i < count; i++)
	{
		distances[i] = compiler_l2_loop(dim, q, vectors + i * dim);
	}

	return 0;
}

int
mkt_distance_batch_ip_compiler(
		VectorRef	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances)
{
	if (mkt_unlikely(
				query.dim != dim || query.data == NULL || vectors == NULL ||
				distances == NULL))
		return -1;

	const float *q = query.data;

	for (uint32_t i = 0; i < count; i++)
	{
		distances[i] = -compiler_ip_loop(dim, q, vectors + i * dim);
	}

	return 0;
}

int
mkt_distance_batch_cosine_compiler(
		VectorRef	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances)
{
	if (mkt_unlikely(
				query.dim != dim || query.data == NULL || vectors == NULL ||
				distances == NULL))
		return -1;

	const float *q = query.data;

	for (uint32_t i = 0; i < count; i++)
	{
		float similarity =
				compiler_cosine_similarity(dim, q, vectors + i * dim);

		/* Handle edge cases */
		if (isnan(similarity) || similarity < -1.0f)
			distances[i] = 1.0f;
		else if (similarity > 1.0f)
			distances[i] = 0.0f;
		else
			distances[i] = 1.0f - similarity;
	}

	return 0;
}

/*
 * Function Pointer Dispatch
 *
 * Global function pointers are initialized once at startup based on detected
 * CPU capabilities. After initialization, dispatch has zero overhead (direct
 * function pointer call).
 */

/* Single-pair function pointers */
typedef Distance (*DistanceFn)(VectorRef, VectorRef);
static DistanceFn g_distance_l2_fn	   = NULL;
static DistanceFn g_distance_ip_fn	   = NULL;
static DistanceFn g_distance_cosine_fn = NULL;

/* Batch function pointers */
typedef int (*BatchDistanceFn)(
		VectorRef, const float *, uint32_t, Dimension, Distance *);
static BatchDistanceFn g_batch_l2_fn	 = NULL;
static BatchDistanceFn g_batch_ip_fn	 = NULL;
static BatchDistanceFn g_batch_cosine_fn = NULL;

/* Implementation name and initialization flag */
static const char	*g_impl_name   = NULL;
static _Atomic(bool) g_initialized = false;

void
mkt_distance_force_reinit(void)
{
	g_initialized		 = false;
	g_distance_l2_fn	 = NULL;
	g_distance_ip_fn	 = NULL;
	g_distance_cosine_fn = NULL;
	g_batch_l2_fn		 = NULL;
	g_batch_ip_fn		 = NULL;
	g_batch_cosine_fn	 = NULL;
	g_impl_name			 = NULL;
}

int
mkt_distance_init(void)
{
	/* Double-checked locking pattern for thread-safe lazy init */
	if (g_initialized)
		return 0;

#ifdef MKT_SIMD_FULL
	/* Full mode: prefer hand-optimized, fall back to compiler-vectorized */
	SimdCapability caps = mkt_detect_simd();

#if defined(__x86_64__) || defined(_M_X64)
	if (caps & SIMD_AVX512F)
	{
		g_distance_l2_fn	 = mkt_distance_l2_avx512;
		g_distance_ip_fn	 = mkt_distance_ip_avx512;
		g_distance_cosine_fn = mkt_distance_cosine_avx512;
		g_batch_l2_fn		 = mkt_distance_batch_l2_avx512;
		g_batch_ip_fn		 = mkt_distance_batch_ip_avx512;
		g_batch_cosine_fn	 = mkt_distance_batch_cosine_avx512;
		g_impl_name			 = "avx512";
	}
	else if (caps & SIMD_AVX2)
	{
		g_distance_l2_fn	 = mkt_distance_l2_avx2;
		g_distance_ip_fn	 = mkt_distance_ip_avx2;
		g_distance_cosine_fn = mkt_distance_cosine_avx2;
		g_batch_l2_fn		 = mkt_distance_batch_l2_avx2;
		g_batch_ip_fn		 = mkt_distance_batch_ip_avx2;
		g_batch_cosine_fn	 = mkt_distance_batch_cosine_avx2;
		g_impl_name			 = "avx2";
	}
	else
	{
		g_distance_l2_fn	 = mkt_distance_l2_compiler;
		g_distance_ip_fn	 = mkt_distance_ip_compiler;
		g_distance_cosine_fn = mkt_distance_cosine_compiler;
		g_batch_l2_fn		 = mkt_distance_batch_l2_compiler;
		g_batch_ip_fn		 = mkt_distance_batch_ip_compiler;
		g_batch_cosine_fn	 = mkt_distance_batch_cosine_compiler;
		g_impl_name			 = "compiler";
	}
#elif defined(__aarch64__) || defined(_M_ARM64)
	if (caps & SIMD_NEON)
	{
		g_distance_l2_fn	 = mkt_distance_l2_neon;
		g_distance_ip_fn	 = mkt_distance_ip_neon;
		g_distance_cosine_fn = mkt_distance_cosine_neon;
		g_batch_l2_fn		 = mkt_distance_batch_l2_neon;
		g_batch_ip_fn		 = mkt_distance_batch_ip_neon;
		g_batch_cosine_fn	 = mkt_distance_batch_cosine_neon;
		g_impl_name			 = "neon";
	}
	else
	{
		g_distance_l2_fn	 = mkt_distance_l2_compiler;
		g_distance_ip_fn	 = mkt_distance_ip_compiler;
		g_distance_cosine_fn = mkt_distance_cosine_compiler;
		g_batch_l2_fn		 = mkt_distance_batch_l2_compiler;
		g_batch_ip_fn		 = mkt_distance_batch_ip_compiler;
		g_batch_cosine_fn	 = mkt_distance_batch_cosine_compiler;
		g_impl_name			 = "compiler";
	}
#else
	/* Unknown architecture: use compiler-vectorized */
	g_distance_l2_fn	 = mkt_distance_l2_compiler;
	g_distance_ip_fn	 = mkt_distance_ip_compiler;
	g_distance_cosine_fn = mkt_distance_cosine_compiler;
	g_batch_l2_fn		 = mkt_distance_batch_l2_compiler;
	g_batch_ip_fn		 = mkt_distance_batch_ip_compiler;
	g_batch_cosine_fn	 = mkt_distance_batch_cosine_compiler;
	g_impl_name			 = "compiler";
#endif

#elif defined(MKT_SIMD_NONE)
	/* None mode: truly scalar (no vectorization) */
	g_distance_l2_fn	 = mkt_distance_l2_compiler;
	g_distance_ip_fn	 = mkt_distance_ip_compiler;
	g_distance_cosine_fn = mkt_distance_cosine_compiler;
	g_batch_l2_fn		 = mkt_distance_batch_l2_compiler;
	g_batch_ip_fn		 = mkt_distance_batch_ip_compiler;
	g_batch_cosine_fn	 = mkt_distance_batch_cosine_compiler;
	g_impl_name			 = "none";

#else
	/* Compiler mode (simd=compiler): use target_clones for ISA selection */
	g_distance_l2_fn	 = mkt_distance_l2_compiler;
	g_distance_ip_fn	 = mkt_distance_ip_compiler;
	g_distance_cosine_fn = mkt_distance_cosine_compiler;
	g_batch_l2_fn		 = mkt_distance_batch_l2_compiler;
	g_batch_ip_fn		 = mkt_distance_batch_ip_compiler;
	g_batch_cosine_fn	 = mkt_distance_batch_cosine_compiler;
	g_impl_name			 = "compiler";
#endif

	g_initialized = true;
	return 0;
}

/*
 * Public API - Single-Pair Distance Functions
 *
 * Use IFUNC (indirect function) for zero-overhead dispatch when available.
 * Falls back to manual dispatch for testing or non-GNU toolchains.
 *
 * IFUNC is only used in simd=full mode where we need runtime dispatch.
 * In simd=compiler mode, target_clones handles ISA selection directly.
 */

/*
 * Use IFUNC only when:
 * - simd=full mode (MKT_SIMD_FULL)
 * - not explicitly disabled (MKT_DISABLE_IFUNC)
 * - not a sanitizer build (MKT_SANITIZER) - resolver runs before sanitizer
 * init
 * - not a coverage build (MKT_COVERAGE) - resolver code skews coverage stats
 * - supported platform (x86-64 or AArch64 with GCC)
 */
#if defined(MKT_SIMD_FULL) && defined(__GNUC__) &&                \
		!defined(MKT_DISABLE_IFUNC) && !defined(MKT_SANITIZER) && \
		!defined(MKT_COVERAGE) &&                                 \
		(defined(__x86_64__) || defined(__aarch64__))
#define USE_IFUNC 1
#else
#define USE_IFUNC 0
#endif

#if USE_IFUNC

/* IFUNC resolvers - called once at program load */

static Distance (*resolve_distance_l2(void))(VectorRef, VectorRef)
{
	SimdCapability caps = mkt_detect_simd();

#if defined(__x86_64__) || defined(_M_X64)
	if (caps & SIMD_AVX512F)
		return mkt_distance_l2_avx512;
	else if (caps & SIMD_AVX2)
		return mkt_distance_l2_avx2;
#elif defined(__aarch64__) || defined(_M_ARM64)
	if (caps & SIMD_NEON)
		return mkt_distance_l2_neon;
#endif

	return mkt_distance_l2_compiler;
}

static Distance (*resolve_distance_ip(void))(VectorRef, VectorRef)
{
	SimdCapability caps = mkt_detect_simd();

#if defined(__x86_64__) || defined(_M_X64)
	if (caps & SIMD_AVX512F)
		return mkt_distance_ip_avx512;
	else if (caps & SIMD_AVX2)
		return mkt_distance_ip_avx2;
#elif defined(__aarch64__) || defined(_M_ARM64)
	if (caps & SIMD_NEON)
		return mkt_distance_ip_neon;
#endif

	return mkt_distance_ip_compiler;
}

static Distance (*resolve_distance_cosine(void))(VectorRef, VectorRef)
{
	SimdCapability caps = mkt_detect_simd();

#if defined(__x86_64__) || defined(_M_X64)
	if (caps & SIMD_AVX512F)
		return mkt_distance_cosine_avx512;
	else if (caps & SIMD_AVX2)
		return mkt_distance_cosine_avx2;
#elif defined(__aarch64__) || defined(_M_ARM64)
	if (caps & SIMD_NEON)
		return mkt_distance_cosine_neon;
#endif

	return mkt_distance_cosine_compiler;
}

/* Public API with IFUNC attribute */

Distance mkt_distance_l2(VectorRef a, VectorRef b)
		__attribute__((ifunc("resolve_distance_l2")));

Distance mkt_distance_ip(VectorRef a, VectorRef b)
		__attribute__((ifunc("resolve_distance_ip")));

Distance mkt_distance_cosine(VectorRef a, VectorRef b)
		__attribute__((ifunc("resolve_distance_cosine")));

#else

/* Fallback: manual dispatch (for testing or non-GNU toolchains) */

Distance
mkt_distance_l2(VectorRef a, VectorRef b)
{
	if (mkt_unlikely(!g_initialized))
		mkt_distance_init();
	return g_distance_l2_fn(a, b);
}

Distance
mkt_distance_ip(VectorRef a, VectorRef b)
{
	if (mkt_unlikely(!g_initialized))
		mkt_distance_init();
	return g_distance_ip_fn(a, b);
}

Distance
mkt_distance_cosine(VectorRef a, VectorRef b)
{
	if (mkt_unlikely(!g_initialized))
		mkt_distance_init();
	return g_distance_cosine_fn(a, b);
}

#endif /* USE_IFUNC */

Distance
mkt_distance(VectorRef a, VectorRef b, DistanceMetric metric)
{
	switch (metric)
	{
	case DISTANCE_L2:
		return mkt_distance_l2(a, b);
	case DISTANCE_INNER_PRODUCT:
		return mkt_distance_ip(a, b);
	case DISTANCE_COSINE:
		return mkt_distance_cosine(a, b);
	default:
		return -1.0f;
	}
}

/*
 * Public API - Batch Distance Functions
 */

int
mkt_distance_batch_l2(
		VectorRef	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances)
{
	if (mkt_unlikely(!g_initialized))
		mkt_distance_init();

	if (mkt_unlikely(
				query.dim != dim || query.data == NULL || vectors == NULL ||
				distances == NULL))
		return -1;

	/* Use SIMD batch implementation if available */
	if (g_batch_l2_fn != NULL)
		return g_batch_l2_fn(query, vectors, count, dim, distances);

	/* Fallback: loop over single-pair function */
	for (uint32_t i = 0; i < count; i++)
	{
		VectorRef vec = {.data = vectors + i * dim, .dim = dim};
		distances[i]  = mkt_distance_l2(query, vec);
	}

	return 0;
}

int
mkt_distance_batch_ip(
		VectorRef	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances)
{
	if (mkt_unlikely(!g_initialized))
		mkt_distance_init();

	if (mkt_unlikely(
				query.dim != dim || query.data == NULL || vectors == NULL ||
				distances == NULL))
		return -1;

	if (g_batch_ip_fn != NULL)
		return g_batch_ip_fn(query, vectors, count, dim, distances);

	for (uint32_t i = 0; i < count; i++)
	{
		VectorRef vec = {.data = vectors + i * dim, .dim = dim};
		distances[i]  = mkt_distance_ip(query, vec);
	}

	return 0;
}

int
mkt_distance_batch_cosine(
		VectorRef	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances)
{
	if (mkt_unlikely(!g_initialized))
		mkt_distance_init();

	if (mkt_unlikely(
				query.dim != dim || query.data == NULL || vectors == NULL ||
				distances == NULL))
		return -1;

	if (g_batch_cosine_fn != NULL)
		return g_batch_cosine_fn(query, vectors, count, dim, distances);

	for (uint32_t i = 0; i < count; i++)
	{
		VectorRef vec = {.data = vectors + i * dim, .dim = dim};
		distances[i]  = mkt_distance_cosine(query, vec);
	}

	return 0;
}

const char *
mkt_distance_impl_name(void)
{
	if (mkt_unlikely(!g_initialized))
		mkt_distance_init();
	return g_impl_name;
}
