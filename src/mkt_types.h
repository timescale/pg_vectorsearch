/*
 * mkt_types.h - Core type definitions for Meerkat
 */

#ifndef MKT_TYPES_H
#define MKT_TYPES_H

#include <stdbool.h>
#include <stdint.h>

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
 * VectorRef: Non-owning reference to vector data.
 *
 * Used for passing vectors to functions without copying.
 * The caller is responsible for ensuring the data remains valid.
 */
typedef struct
{
	const float *data;
	Dimension	 dim;
} VectorRef;

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

#endif /* MKT_TYPES_H */
