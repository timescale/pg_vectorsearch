/*
 * fast_rotate.c - O(d log d) orthonormal rotation
 *
 * Generalised Walsh-Hadamard transform with random sign-flip prefix
 * and a K×K mixer across K sub-blocks, scaled to be orthonormal. Used
 * as a drop-in for the dense `P^T * x` rotation in RaBitQ.
 *
 * For dim = N (power of two) we set K=1 and apply the classic
 * randomised Hadamard transform: D · H_N · scale.
 *
 * For dim = N · K with N power-of-two and K ≥ 2 we view x as a K×N
 * matrix (K rows of length N), apply H_N to each row in place, then
 * apply a K×K orthonormal matrix M across the rows at each column.
 * The combined map M · (I_K ⊗ H_N) · D · scale is orthonormal because
 * each factor is, and the Kronecker structure makes it O(d log N + d·K)
 * to evaluate — for dim=768 = 256·3 that's ~9 k ops vs ~1.18 M ops
 * for the dense sgemv it replaces.
 */

#include <math.h>
#include <string.h>

#include "core/memory.h"
#include "quant/fast_rotate.h"

bool
mkt_fast_rotate_supported(Dimension dim)
{
	if (dim < 4)
		return false;

	/* Factor dim = N * K with N the largest power-of-two divisor.
	 * K must be ≤ MKT_FAST_ROTATE_K_MAX, and N ≥ 4 so the inner
	 * FWHT has at least two butterfly stages. */
	uint32_t n = dim;
	uint32_t k = 1;
	while ((n & 1) == 0)
		n >>= 1;
	/* Now n is the odd part, k = original/odd_part is the power-of-two
	 * part of dim. The decomposition we want is the *opposite*:
	 * fwht_n = power-of-two part, k = odd cofactor. */
	uint32_t fwht_n = dim / n;
	k				= n;
	if (fwht_n < 4)
		return false;
	if (k > MKT_FAST_ROTATE_K_MAX)
		return false;
	return true;
}

/* SplitMix64: deterministic PRNG used only for sign-vector / mixer
 * init. Self-contained so init doesn't depend on the rest of the
 * codebase. */
static inline uint64_t
splitmix64(uint64_t *state)
{
	uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
	z		   = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
	z		   = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
	return z ^ (z >> 31);
}

/* Build a K×K orthonormal matrix from PRNG state via Gram-Schmidt on
 * K random vectors. Cheap because K is small (≤ K_MAX). */
static void
init_mixer(float *m, uint32_t k, uint64_t *prng_state)
{
	float v[MKT_FAST_ROTATE_K_MAX][MKT_FAST_ROTATE_K_MAX];

	/* Random vectors */
	for (uint32_t i = 0; i < k; i++)
		for (uint32_t j = 0; j < k; j++)
		{
			uint64_t r = splitmix64(prng_state);
			/* Standard normal via Box-Muller would be cleaner but the
			 * mixer just needs to be a non-degenerate basis to
			 * orthogonalise — uniform in [-1, 1] is fine. */
			v[i][j] = (float)((double)r / (double)UINT64_MAX) * 2.0f - 1.0f;
		}

	/* Modified Gram-Schmidt */
	for (uint32_t i = 0; i < k; i++)
	{
		for (uint32_t j = 0; j < i; j++)
		{
			float dot = 0.0f;
			for (uint32_t c = 0; c < k; c++)
				dot += v[i][c] * v[j][c];
			for (uint32_t c = 0; c < k; c++)
				v[i][c] -= dot * v[j][c];
		}
		float norm = 0.0f;
		for (uint32_t c = 0; c < k; c++)
			norm += v[i][c] * v[i][c];
		norm = sqrtf(norm);
		/* Degenerate rows are vanishingly unlikely for random uniform
		 * inputs at K ≤ 8; if it happens, fall back to a canonical
		 * basis vector for that slot. */
		if (norm < 1e-6f)
		{
			for (uint32_t c = 0; c < k; c++)
				v[i][c] = (c == i) ? 1.0f : 0.0f;
		}
		else
		{
			for (uint32_t c = 0; c < k; c++)
				v[i][c] /= norm;
		}
	}

	for (uint32_t i = 0; i < k; i++)
		for (uint32_t j = 0; j < k; j++)
			m[i * k + j] = v[i][j];
}

void
mkt_fast_rotate_init(MktFastRotateParams *p, Dimension dim, uint64_t seed)
{
	p->dim	= dim;
	p->seed = seed;

	/* Factor dim = fwht_n * k with fwht_n the power-of-two part. */
	uint32_t odd = dim;
	while ((odd & 1) == 0)
		odd >>= 1;
	p->fwht_n = dim / odd;
	p->k	  = odd;

	memset(p->signs1, 0, sizeof(p->signs1));
	memset(p->signs2, 0, sizeof(p->signs2));

	uint64_t s = seed ? seed : 0xDEADBEEFCAFEBABEULL;
	for (Dimension i = 0; i < dim; i++)
	{
		uint64_t r = splitmix64(&s);
		if (r & 1)
			p->signs1[i >> 3] |= (uint8_t)(1u << (i & 7));
	}
	for (Dimension i = 0; i < dim; i++)
	{
		uint64_t r = splitmix64(&s);
		if (r & 1)
			p->signs2[i >> 3] |= (uint8_t)(1u << (i & 7));
	}

	memset(p->mixer, 0, sizeof(p->mixer));
	if (p->k == 1)
		p->mixer[0] = 1.0f;
	else
		init_mixer(p->mixer, p->k, &s);
}

/* In-place Walsh-Hadamard on a length-N block (N power of two). */
static inline void
fwht_inplace(float *a, Dimension n)
{
	for (Dimension h = 1; h < n; h <<= 1)
	{
		for (Dimension i = 0; i < n; i += (h << 1))
		{
			for (Dimension j = i; j < i + h; j++)
			{
				float x	 = a[j];
				float y	 = a[j + h];
				a[j]	 = x + y;
				a[j + h] = x - y;
			}
		}
	}
}

/* Apply the K×K mixer matrix M to the K-vector formed by taking
 * `col` from each of K rows. Mixer rows = K, columns = K. */
static inline void
apply_mixer_column(
		float *out, const float *m, uint32_t k, Dimension n, Dimension col)
{
	float tmp[MKT_FAST_ROTATE_K_MAX];
	for (uint32_t i = 0; i < k; i++)
		tmp[i] = out[i * n + col];

	for (uint32_t i = 0; i < k; i++)
	{
		float s = 0.0f;
		for (uint32_t j = 0; j < k; j++)
			s += m[i * k + j] * tmp[j];
		out[i * n + col] = s;
	}
}

/* Reference scalar implementation. SIMD variants will land alongside
 * this once the kernel API is settled. */
void
mkt_fast_rotate_apply(
		const MktFastRotateParams *p, const float *in, float *out)
{
	Dimension dim = p->dim;
	Dimension n	  = p->fwht_n;
	uint32_t  k	  = p->k;
	/* Unscaled FWHT on a length-N block has Parseval factor N
	 * (||y||² = N · ||x||²). The K×K mixer is unit-norm. So pre-scale
	 * by 1/sqrt(N), NOT 1/sqrt(dim) — that gets the overall map to
	 * unit-norm regardless of K. */
	float invsq = 1.0f / sqrtf((float)n);

	/* Pre-FWHT sign flip (D1) + scale. Folding the 1/sqrt(N) scale
	 * into this pass means the butterflies and mixer can run on plain
	 * sums without a final scaling sweep. */
	for (Dimension i = 0; i < dim; i++)
	{
		float sign = ((p->signs1[i >> 3] >> (i & 7)) & 1u) ? -1.0f : 1.0f;
		out[i]	   = in[i] * sign * invsq;
	}

	/* FWHT on each of K sub-blocks of length N. When K==1 this is
	 * just the classic randomised Hadamard rotation. */
	for (uint32_t b = 0; b < k; b++)
		fwht_inplace(out + (size_t)b * n, n);

	/* K×K mixer across the blocks. Skip when K==1: mixer is identity. */
	if (k > 1)
	{
		for (Dimension col = 0; col < n; col++)
			apply_mixer_column(out, p->mixer, k, n, col);
	}

	/* Post-FWHT sign flip (D2). This second round of randomisation
	 * tightens the concentration of the transform: with one round,
	 * worst-case Lipschitz can break RaBitQ's strict lower bound at
	 * small dim; two rounds match the FJLT analysis (Ailon-Chazelle)
	 * and restore the bound. Cost is one extra O(d) pass. */
	for (Dimension i = 0; i < dim; i++)
	{
		float sign = ((p->signs2[i >> 3] >> (i & 7)) & 1u) ? -1.0f : 1.0f;
		out[i] *= sign;
	}
}
