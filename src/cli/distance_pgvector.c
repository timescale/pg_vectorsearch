/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * Portions derived from pgvector's src/vector.c
 * (https://github.com/pgvector/pgvector), which carries the following
 * copyright notice and license:
 *
 * Portions Copyright (c) 1996-2026, PostgreSQL Global Development Group
 * Portions Copyright (c) 1994, The Regents of the University of California
 *
 * Permission to use, copy, modify, and distribute this software and its
 * documentation for any purpose, without fee, and without a written agreement
 * is hereby granted, provided that the above copyright notice and this
 * paragraph and the following two paragraphs appear in all copies.
 *
 * IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY FOR
 * DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING
 * LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS
 * DOCUMENTATION, EVEN IF THE UNIVERSITY OF CALIFORNIA HAS BEEN ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * THE UNIVERSITY OF CALIFORNIA SPECIFICALLY DISCLAIMS ANY WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED HEREUNDER IS
 * ON AN "AS IS" BASIS, AND THE UNIVERSITY OF CALIFORNIA HAS NO OBLIGATIONS TO
 * PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
 *
 * distance_pgvector.c - pgvector-style auto-vectorized distance
 *
 * This file contains distance implementations that match pgvector's coding
 * style: simple loops with GCC's target_clones attribute for automatic
 * multi-versioning and compiler auto-vectorization.
 *
 * This demonstrates the performance difference between:
 * - Compiler auto-vectorization (pgvector's approach)
 * - Hand-optimized explicit SIMD (our AVX-512/AVX2 implementations)
 *
 * Code is copied from pgvector/src/vector.c to ensure accurate comparison.
 */

#include <math.h>
#include <stddef.h>

#include "core/platform.h"
#include "distance_pgvector.h"

/*
 * GCC's target_clones creates multiple versions of the function:
 * - "default": Baseline x86-64 (no SIMD)
 * - "fma": Uses FMA instructions (implies AVX2 or later)
 *
 * At runtime, GCC's IFUNC resolver picks the best version for the CPU.
 */
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#define PGVECTOR_TARGET_CLONES __attribute__((target_clones("default", "fma")))
#else
#define PGVECTOR_TARGET_CLONES
#endif

/*
 * L2 squared distance - pgvector style
 * From: pgvector/src/vector.c VectorL2SquaredDistance()
 */
PGVECTOR_TARGET_CLONES static float
pgvector_l2_squared_distance(int dim, const float *ax, const float *bx)
{
	float distance = 0.0f;

	/* Auto-vectorized */
	for (int i = 0; i < dim; i++)
	{
		float diff = ax[i] - bx[i];
		distance += diff * diff;
	}

	return distance;
}

/*
 * Inner product - pgvector style
 * From: pgvector/src/vector.c VectorInnerProduct()
 */
PGVECTOR_TARGET_CLONES static float
pgvector_inner_product(int dim, const float *ax, const float *bx)
{
	float distance = 0.0f;

	/* Auto-vectorized */
	for (int i = 0; i < dim; i++)
		distance += ax[i] * bx[i];

	return distance;
}

/*
 * Cosine similarity - pgvector style
 * From: pgvector/src/vector.c VectorCosineSimilarity()
 *
 * Note: pgvector returns double for this function
 */
PGVECTOR_TARGET_CLONES static double
pgvector_cosine_similarity(int dim, const float *ax, const float *bx)
{
	float similarity = 0.0f;
	float norma		 = 0.0f;
	float normb		 = 0.0f;

	/* Auto-vectorized */
	for (int i = 0; i < dim; i++)
	{
		similarity += ax[i] * bx[i];
		norma += ax[i] * ax[i];
		normb += bx[i] * bx[i];
	}

	/* Use sqrt(a * b) over sqrt(a) * sqrt(b) */
	return (double)similarity / sqrt((double)norma * (double)normb);
}

/*
 * Public wrappers matching our Distance API
 */

Distance
vs_distance_l2_pgvector(Vec32Ref a, Vec32Ref b)
{
	if (vs_unlikely(
				a.dim != b.dim || a.dim == 0 || a.data == NULL ||
				b.data == NULL))
		return -1.0f;

	return pgvector_l2_squared_distance(a.dim, a.data, b.data);
}

Distance
vs_distance_ip_pgvector(Vec32Ref a, Vec32Ref b)
{
	if (vs_unlikely(
				a.dim != b.dim || a.dim == 0 || a.data == NULL ||
				b.data == NULL))
		return -1.0f;

	return -pgvector_inner_product(a.dim, a.data, b.data);
}

Distance
vs_distance_cosine_pgvector(Vec32Ref a, Vec32Ref b)
{
	if (vs_unlikely(
				a.dim != b.dim || a.dim == 0 || a.data == NULL ||
				b.data == NULL))
		return -1.0f;

	double similarity = pgvector_cosine_similarity(a.dim, a.data, b.data);

	/* Keep in range (pgvector does this) */
	if (similarity > 1)
		similarity = 1.0;
	else if (similarity < -1)
		similarity = -1.0;

	return (float)(1.0 - similarity);
}

/*
 * Batch operations (loop over single-pair calls, pgvector style)
 */

int
vs_distance_batch_l2_pgvector(
		Vec32Ref	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances)
{
	if (vs_unlikely(
				query.dim != dim || query.data == NULL || vectors == NULL ||
				distances == NULL))
		return -1;

	const float *q = query.data;

	for (uint32_t i = 0; i < count; i++)
	{
		distances[i] = pgvector_l2_squared_distance(dim, q, vectors + i * dim);
	}

	return 0;
}

int
vs_distance_batch_ip_pgvector(
		Vec32Ref	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances)
{
	if (vs_unlikely(
				query.dim != dim || query.data == NULL || vectors == NULL ||
				distances == NULL))
		return -1;

	const float *q = query.data;

	for (uint32_t i = 0; i < count; i++)
	{
		distances[i] = -pgvector_inner_product(dim, q, vectors + i * dim);
	}

	return 0;
}

int
vs_distance_batch_cosine_pgvector(
		Vec32Ref	 query,
		const float *vectors,
		uint32_t	 count,
		Dimension	 dim,
		Distance	*distances)
{
	if (vs_unlikely(
				query.dim != dim || query.data == NULL || vectors == NULL ||
				distances == NULL))
		return -1;

	const float *q = query.data;

	for (uint32_t i = 0; i < count; i++)
	{
		double similarity =
				pgvector_cosine_similarity(dim, q, vectors + i * dim);

		/* Keep in range */
		if (similarity > 1)
			similarity = 1.0;
		else if (similarity < -1)
			similarity = -1.0;

		distances[i] = (float)(1.0 - similarity);
	}

	return 0;
}
