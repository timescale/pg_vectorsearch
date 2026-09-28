/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * fast_rotate.h - O(d log d) orthonormal rotation
 *
 * Replaces the dense `dim x dim` random orthonormal matrix multiply
 * used in RaBitQ's `pt_query = P^T * query` step with a Randomized
 * Hadamard Transform (RHT): a sign-flip diagonal followed by the
 * Walsh-Hadamard transform, scaled by 1/sqrt(N).
 *
 * Properties:
 *   - Orthonormal: ||F(x)|| = ||x||, same property RaBitQ relies on.
 *   - O(d log d) compute vs O(d^2) for dense sgemv.
 *   - O(d/8) storage for the random sign vector vs O(d^2) for the
 *     dense matrix; the deterministic Hadamard structure is implicit.
 *
 * Supported dimensions: power-of-2 only in this initial version.
 * For dim=768 (cohere) we'd need a mixed-radix or padded variant
 * — out of scope for the first cut.
 */

#ifndef VS_FAST_ROTATE_H
#define VS_FAST_ROTATE_H

#include <stdbool.h>
#include <stdint.h>

#include "core/types.h"

/* Mixed-radix support: factor dim = N * K with N a power-of-two.
 * MIXING_DIM_MAX caps K so we can keep the K×K mixing matrix on the
 * stack as fixed-size. cohere-1M lives at dim=768 = 256*3; vectors at
 * dim = 384 = 128*3, 1536 = 512*3, etc. also work. */
#define VS_FAST_ROTATE_K_MAX 8

/* Max dim with inline signs storage. 8192 covers every embedding
 * dim we care about (cohere-768, openai-1536, etc.) at 1 KB. */
#define VS_FAST_ROTATE_MAX_DIM	 8192
#define VS_FAST_ROTATE_MAX_SIGNS ((VS_FAST_ROTATE_MAX_DIM + 7) / 8)

typedef struct VsFastRotateParams
{
	Dimension dim;	  /* total vector dimension */
	Dimension fwht_n; /* power-of-two FWHT length (= dim when K==1) */
	uint32_t  k;	  /* outer "mixing" radix; K==1 means pure FWHT */
	uint64_t  seed;	  /* regenerates signs+mixer deterministically */
	/* Pre-FWHT sign-flip vector (D1) and post-FWHT sign-flip vector
	 * (D2) — two independent rounds of randomisation. A single round
	 * (D1 only) is isotropic in expectation but its concentration is
	 * weaker than a dense Haar-random orthonormal, enough to break
	 * the strict RaBitQ lower-bound at small dim. Two rounds match
	 * the FJLT recipe and recover the worst-case bound. */
	uint8_t signs1[VS_FAST_ROTATE_MAX_SIGNS];
	uint8_t signs2[VS_FAST_ROTATE_MAX_SIGNS];
	/* Random K×K orthonormal matrix applied across the K sub-blocks
	 * after the per-block FWHT. Generated from seed; identity slot
	 * unused when K==1. Stored row-major as K*K floats. */
	float mixer[VS_FAST_ROTATE_K_MAX * VS_FAST_ROTATE_K_MAX];
} VsFastRotateParams;

/* True if fast rotation is supported at the given dimension.
 * Requires dim = N * K with N a power-of-two and K ≤ K_MAX. */
bool vs_fast_rotate_supported(Dimension dim);

/* Initialise sign vector + mixer from seed. */
void vs_fast_rotate_init(VsFastRotateParams *p, Dimension dim, uint64_t seed);

/* Apply F = (1/sqrt(dim)) * M * H * D to `in`, writing into `out`.
 * H is the block-diagonal FWHT on K blocks of length N; M is the K×K
 * mixer applied across the blocks; D is the diagonal sign flip.
 * `out` may alias `in`. Both buffers must be `dim` floats. */
void
vs_fast_rotate_apply(const VsFastRotateParams *p, const float *in, float *out);

#endif /* VS_FAST_ROTATE_H */
