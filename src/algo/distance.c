/*
 * distance.c - Distance computation with SIMD dispatch
 *
 * Implements runtime CPU detection and function pointer dispatch for
 * optimal SIMD implementations, with scalar fallback.
 */

#include <math.h>
#include <stddef.h>

#include "algo/distance.h"
#include "algo/simd_utils.h"
#include "core/platform.h"

/*
 * Forward declarations for SIMD implementations (defined in separate files)
 */

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

/*
 * Scalar Reference Implementations
 *
 * These are simple, compiler-friendly implementations that serve as:
 * 1. Fallback for systems without SIMD
 * 2. Reference for testing SIMD implementations
 * 3. Basis for compiler auto-vectorization
 */

Distance
mkt_distance_l2_scalar(VectorRef a, VectorRef b)
{
	if (mkt_unlikely(
				a.dim != b.dim || a.dim == 0 || a.data == NULL ||
				b.data == NULL))
		return -1.0f;

	const float *pa	 = a.data;
	const float *pb	 = b.data;
	Dimension	 dim = a.dim;

	float sum = 0.0f;
	for (Dimension i = 0; i < dim; i++)
	{
		float diff = pa[i] - pb[i];
		sum += diff * diff;
	}

	return sum;
}

Distance
mkt_distance_ip_scalar(VectorRef a, VectorRef b)
{
	if (mkt_unlikely(
				a.dim != b.dim || a.dim == 0 || a.data == NULL ||
				b.data == NULL))
		return -1.0f;

	const float *pa	 = a.data;
	const float *pb	 = b.data;
	Dimension	 dim = a.dim;

	float sum = 0.0f;
	for (Dimension i = 0; i < dim; i++)
		sum += pa[i] * pb[i];

	return -sum; /* Negate for distance metric */
}

Distance
mkt_distance_cosine_scalar(VectorRef a, VectorRef b)
{
	if (mkt_unlikely(
				a.dim != b.dim || a.dim == 0 || a.data == NULL ||
				b.data == NULL))
		return -1.0f;

	const float *pa	 = a.data;
	const float *pb	 = b.data;
	Dimension	 dim = a.dim;

	float dot	 = 0.0f;
	float norm_a = 0.0f;
	float norm_b = 0.0f;

	for (Dimension i = 0; i < dim; i++)
	{
		float va = pa[i];
		float vb = pb[i];
		dot += va * vb;
		norm_a += va * va;
		norm_b += vb * vb;
	}

	float denom = sqrtf(norm_a) * sqrtf(norm_b);
	if (denom < 1e-8f)
		return 1.0f; /* Maximum distance for zero vectors */

	return 1.0f - (dot / denom);
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
		g_impl_name			 = "AVX-512";
	}
	else if (caps & SIMD_AVX2)
	{
		g_distance_l2_fn	 = mkt_distance_l2_avx2;
		g_distance_ip_fn	 = mkt_distance_ip_avx2;
		g_distance_cosine_fn = mkt_distance_cosine_avx2;
		g_batch_l2_fn		 = mkt_distance_batch_l2_avx2;
		g_batch_ip_fn		 = mkt_distance_batch_ip_avx2;
		g_batch_cosine_fn	 = mkt_distance_batch_cosine_avx2;
		g_impl_name			 = "AVX2";
	}
	else
#elif defined(__aarch64__) || defined(_M_ARM64)
	if (caps & SIMD_NEON)
	{
		g_distance_l2_fn	 = mkt_distance_l2_neon;
		g_distance_ip_fn	 = mkt_distance_ip_neon;
		g_distance_cosine_fn = mkt_distance_cosine_neon;
		g_batch_l2_fn		 = mkt_distance_batch_l2_neon;
		g_batch_ip_fn		 = mkt_distance_batch_ip_neon;
		g_batch_cosine_fn	 = mkt_distance_batch_cosine_neon;
		g_impl_name			 = "NEON";
	}
	else
#else
	if (0)
	{
		/* No SIMD support */
	}
	else
#endif
	{
		g_distance_l2_fn	 = mkt_distance_l2_scalar;
		g_distance_ip_fn	 = mkt_distance_ip_scalar;
		g_distance_cosine_fn = mkt_distance_cosine_scalar;
		g_batch_l2_fn		 = NULL; /* Use fallback loop */
		g_batch_ip_fn		 = NULL;
		g_batch_cosine_fn	 = NULL;
		g_impl_name			 = "scalar";
	}

	g_initialized = true;
	return 0;
}

/*
 * Public API - Single-Pair Distance Functions
 */

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
