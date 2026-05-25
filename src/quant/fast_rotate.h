/*
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

#ifndef MKT_FAST_ROTATE_H
#define MKT_FAST_ROTATE_H

#include <stdbool.h>
#include <stdint.h>

#include "mkt_types.h"

/* Mixed-radix support: factor dim = N * K with N a power-of-two.
 * MIXING_DIM_MAX caps K so we can keep the K×K mixing matrix on the
 * stack as fixed-size. cohere-1M lives at dim=768 = 256*3; vectors at
 * dim = 384 = 128*3, 1536 = 512*3, etc. also work. */
#define MKT_FAST_ROTATE_K_MAX 8

typedef struct MktFastRotateParams
{
	Dimension dim;		/* total vector dimension */
	Dimension fwht_n;	/* power-of-two FWHT length (= dim when K==1) */
	uint32_t  k;		/* outer "mixing" radix; K==1 means pure FWHT */
	uint64_t  seed;		/* regenerates signs+mixer deterministically */
	uint8_t	 *signs;	/* dim bits, packed; sign[i] = (signs[i>>3] >> (i&7))&1 */
	/* Random K×K orthonormal matrix applied across the K sub-blocks
	 * after the per-block FWHT. Generated from seed; identity slot
	 * unused when K==1. Stored row-major as K*K floats. */
	float	  mixer[MKT_FAST_ROTATE_K_MAX * MKT_FAST_ROTATE_K_MAX];
} MktFastRotateParams;

/* True if fast rotation is supported at the given dimension.
 * Requires dim = N * K with N a power-of-two and K ≤ K_MAX. */
bool mkt_fast_rotate_supported(Dimension dim);

/* Initialise sign vector + mixer from seed. `signs_buf` must hold at
 * least (dim + 7) / 8 bytes. */
void mkt_fast_rotate_init(
		MktFastRotateParams *p, Dimension dim, uint64_t seed, uint8_t *signs_buf);

/* Apply F = (1/sqrt(dim)) * M * H * D to `in`, writing into `out`.
 * H is the block-diagonal FWHT on K blocks of length N; M is the K×K
 * mixer applied across the blocks; D is the diagonal sign flip.
 * `out` may alias `in`. Both buffers must be `dim` floats. */
void mkt_fast_rotate_apply(
		const MktFastRotateParams *p, const float *in, float *out);

#endif /* MKT_FAST_ROTATE_H */
