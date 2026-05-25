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

/* Random sign vector for the diagonal D in F = (1/sqrt(N)) * H * D.
 * One bit per dimension, packed LSB-first into bytes. */
typedef struct MktFastRotateParams
{
	Dimension dim;		/* must be a power of 2 */
	uint64_t  seed;		/* used to regenerate signs deterministically */
	uint8_t	 *signs;	/* dim bits, packed; sign[i] = (signs[i>>3] >> (i&7)) & 1 */
} MktFastRotateParams;

/* True if fast rotation is supported at the given dimension. */
bool mkt_fast_rotate_supported(Dimension dim);

/* Initialise sign vector from seed. `signs_buf` must hold at least
 * (dim + 7) / 8 bytes. */
void mkt_fast_rotate_init(
		MktFastRotateParams *p, Dimension dim, uint64_t seed, uint8_t *signs_buf);

/* Apply F = (1/sqrt(N)) * H * D to `in`, writing into `out`.
 * `out` may alias `in`. Both buffers must be `dim` floats. */
void mkt_fast_rotate_apply(
		const MktFastRotateParams *p, const float *in, float *out);

#endif /* MKT_FAST_ROTATE_H */
