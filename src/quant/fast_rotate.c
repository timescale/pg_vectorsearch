/*
 * fast_rotate.c - O(d log d) orthonormal rotation
 *
 * Walsh-Hadamard transform with random sign-flip prefix, scaled to be
 * orthonormal. Used as a drop-in for the dense `P^T * x` rotation in
 * RaBitQ when dim is a power of two.
 *
 * The unnormalised in-place butterfly looks like:
 *
 *     for h = 1, 2, 4, ..., N/2:
 *         for each pair (i, i+h) with i in the same length-2h block:
 *             a, b = x[i], x[i+h]
 *             x[i], x[i+h] = a + b, a - b
 *
 * After log2(N) passes each output is a sum of all inputs with ±1
 * weights — that's the unscaled Walsh-Hadamard transform. We then
 * divide by sqrt(N) for orthonormality. The signs are baked into the
 * sign-flip step that runs before the butterflies, so the per-query
 * cost is one sign-flip pass + log2(N) butterfly passes + one scale
 * pass.
 */

#include <math.h>
#include <string.h>

#include "core/memory.h"
#include "quant/fast_rotate.h"

bool
mkt_fast_rotate_supported(Dimension dim)
{
	/* Power of two, ≥ 4 */
	return dim >= 4 && (dim & (dim - 1)) == 0;
}

/* SplitMix64: deterministic PRNG used only for sign-vector init.
 * Self-contained so init doesn't depend on the rest of the codebase. */
static inline uint64_t
splitmix64(uint64_t *state)
{
	uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
	z		   = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
	z		   = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
	return z ^ (z >> 31);
}

void
mkt_fast_rotate_init(
		MktFastRotateParams *p, Dimension dim, uint64_t seed, uint8_t *signs_buf)
{
	p->dim	 = dim;
	p->seed	 = seed;
	p->signs = signs_buf;

	uint32_t nbytes = (uint32_t)((dim + 7) / 8);
	memset(p->signs, 0, nbytes);

	uint64_t s = seed ? seed : 0xDEADBEEFCAFEBABEULL;
	for (Dimension i = 0; i < dim; i++)
	{
		uint64_t r = splitmix64(&s);
		if (r & 1)
			p->signs[i >> 3] |= (uint8_t)(1u << (i & 7));
	}
}

/* Reference scalar implementation. Vectorised variants (NEON / SVE2)
 * will land alongside the SIMD dispatch in fast_rotate_neon.c / _sve2.c
 * once the kernel API is settled. */
void
mkt_fast_rotate_apply(
		const MktFastRotateParams *p, const float *in, float *out)
{
	Dimension dim	 = p->dim;
	float	  invsq	 = 1.0f / sqrtf((float) dim);

	/* Sign flip + copy. Fold the 1/sqrt(N) scale into the sign-flip
	 * pass so the butterflies can run on plain ±-sum data without an
	 * extra scaling sweep at the end. */
	for (Dimension i = 0; i < dim; i++)
	{
		float sign = ((p->signs[i >> 3] >> (i & 7)) & 1u) ? -1.0f : 1.0f;
		out[i]	   = in[i] * sign * invsq;
	}

	/* In-place Walsh-Hadamard butterflies on `out`. */
	for (Dimension h = 1; h < dim; h <<= 1)
	{
		for (Dimension i = 0; i < dim; i += (h << 1))
		{
			for (Dimension j = i; j < i + h; j++)
			{
				float a		   = out[j];
				float b		   = out[j + h];
				out[j]		   = a + b;
				out[j + h]	   = a - b;
			}
		}
	}
}
