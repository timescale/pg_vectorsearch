/*
 * mkt_types.h - Core type definitions for pg_vectorsearch
 */

#ifndef MKT_TYPES_H
#define MKT_TYPES_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/*
 * Flexible array member marker. PostgreSQL's c.h defines this; for
 * standalone builds we provide an empty fallback so shared headers
 * can use the same syntax in both contexts.
 */
#ifndef FLEXIBLE_ARRAY_MEMBER
#define FLEXIBLE_ARRAY_MEMBER /* empty, C99+ */
#endif

/*
 * MKT_VTABLE_INLINE — hint for vtable functions.
 *
 * Clang inlines always_inline functions through static-const vtable
 * pointers even across target_clones boundaries. GCC errors on
 * target mismatch, so we omit the attribute there.
 */
#ifdef __clang__
#define MKT_VTABLE_INLINE __attribute__((always_inline)) static inline
#else
#define MKT_VTABLE_INLINE static inline
#endif

/* Quantized representations */
typedef uint8_t ScalarQ8; /* 8-bit scalar quantized */
typedef uint8_t BinaryQ;  /* Binary quantized byte (packed bits) */

/* Dimension type (max 65535 dimensions) */
typedef uint16_t Dimension;

/* Cluster/centroid identifier */
typedef uint32_t ClusterId;

/* Distance type (always float for intermediate computations) */
typedef float Distance;

/*
 * Vec32Ref: Non-owning reference to vector data.
 *
 * Used for passing vectors to functions without copying.
 * The caller is responsible for ensuring the data remains valid.
 */
typedef struct
{
	const float *data;
	Dimension	 dim;
} Vec32Ref;

/*
 * VectorMut: Mutable vector reference.
 *
 * Used when the function needs to modify the vector data.
 */
typedef struct
{
	float	 *data;
	Dimension dim;
} VectorMut;

/* Distance metric enum */
typedef enum
{
	DISTANCE_L2,			/* Euclidean (L2 squared) */
	DISTANCE_INNER_PRODUCT, /* Negative inner product (for max similarity) */
	DISTANCE_COSINE			/* 1 - cosine similarity */
} DistanceMetric;

/*
 * MktDistanceMode - RaBitQ distance computation mode
 *
 * Controls whether search uses asymmetric (full-precision query × 1-bit data)
 * or symmetric (1-bit query × 1-bit data) distance. Symmetric is ~4x faster
 * but has higher estimation error.
 */
typedef enum
{
	MKT_DISTANCE_MODE_DEFAULT	 = -1, /* GUC sentinel: use index relopt */
	MKT_DISTANCE_MODE_ASYMMETRIC = 0,
	MKT_DISTANCE_MODE_SYMMETRIC	 = 1,
} MktDistanceMode;

static inline const char *
mkt_distance_mode_name(MktDistanceMode mode)
{
	static const char *names[] = {
			[MKT_DISTANCE_MODE_ASYMMETRIC] = "asymmetric",
			[MKT_DISTANCE_MODE_SYMMETRIC]  = "symmetric",
	};
	return names[mode];
}

/*
 * VecType - Vector element type for dispatch
 *
 * Used by k-means and quantization to select the correct compile-time
 * specialization. F16C is the hand-written AVX2+FMA+F16C path; F16
 * relies on compiler auto-vectorization via TARGET_CLONES.
 */
typedef enum
{
	MKT_VEC_F32	 = 0, /* float32 */
	MKT_VEC_F16	 = 1, /* float16 (scalar / auto-vectorized) */
	MKT_VEC_F16C = 2, /* float16 (hand-written F16C SIMD) */
} VecType;

static inline size_t
mkt_vec_element_size(VecType type)
{
	static const size_t sizes[] =
			{[MKT_VEC_F32] = 4, [MKT_VEC_F16] = 2, [MKT_VEC_F16C] = 2};
	return sizes[type];
}

static inline const char *
mkt_vec_type_name(VecType type)
{
	static const char *names[] = {
			[MKT_VEC_F32]  = "float32",
			[MKT_VEC_F16]  = "float16",
			[MKT_VEC_F16C] = "float16-f16c",
	};
	return names[type];
}

/*
 * Vec32TypeOps - Inline vtable for compile-time specialization
 *
 * Contains function pointers for mixed-type distance computation
 * (vec_type × float32 centroid). Used internally by k-means and
 * quantization impl functions where always_inline enables the
 * compiler to inline through the function pointers.
 */
typedef struct Vec32TypeOps
{
	const char *name;		  /* "float32" or "float16" */
	size_t		element_size; /* 4 or 2 */

	float (*dot_product)(
			const void *vec, const float *centroid, Dimension dim);
	float (*l2_squared)(const void *vec, const float *centroid, Dimension dim);
	float (*norm_sq)(const void *vec, Dimension dim);

	void (*sum_to_float)(const void *vec, float *accum, Dimension dim);
	void (*to_float_one)(const void *src, float *dst, Dimension dim);

	/* Returns pointer to float32 data (zero-copy for f32, converts for f16) */
	const float *(*to_float_block)(
			const void *src, float *dst, uint32_t count, Dimension dim);
} Vec32TypeOps;

#endif /* MKT_TYPES_H */
