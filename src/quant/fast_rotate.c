/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
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
#include "core/platform.h"
#include "quant/fast_rotate.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

bool
vs_fast_rotate_supported(Dimension dim)
{
	if (dim < 4)
		return false;

	/* Factor dim = N * K with N the largest power-of-two divisor.
	 * K must be ≤ VS_FAST_ROTATE_K_MAX, and N ≥ 4 so the inner
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
	if (k > VS_FAST_ROTATE_K_MAX)
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
	float v[VS_FAST_ROTATE_K_MAX][VS_FAST_ROTATE_K_MAX];

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
vs_fast_rotate_init(VsFastRotateParams *p, Dimension dim, uint64_t seed)
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
	float tmp[VS_FAST_ROTATE_K_MAX];
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

/* Reference scalar implementation. */
static void
fast_rotate_apply_scalar(
		const VsFastRotateParams *p, const float *in, float *out)
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

#if defined(__x86_64__) || defined(_M_X64)
/*
 * AVX2 kernel. Vectorizes the parts that map cleanly to 8-wide SIMD:
 *   - D1/D2 sign flips: expand 8 packed sign bits to a ±1 vector, multiply.
 *   - FWHT stages with stride h >= 8: contiguous wide add/sub (bit-identical
 *     to the scalar butterfly). Small stages (h < 8) stay scalar -- they'd
 *     need in-register shuffle networks for little of the total work.
 *   - K x K mixer: vectorized across 8 columns (contiguous within a block).
 */
__attribute__((target("avx2,fma"))) static void
fast_rotate_apply_avx2(
		const VsFastRotateParams *p, const float *in, float *out)
{
	Dimension	  dim	   = p->dim;
	Dimension	  n		   = p->fwht_n;
	uint32_t	  k		   = p->k;
	float		  invsq	   = 1.0f / sqrtf((float)n);
	const __m256i lanebits = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
	const __m256  vpos	   = _mm256_set1_ps(1.0f);
	const __m256  vneg	   = _mm256_set1_ps(-1.0f);

	/* D1: sign flip + 1/sqrt(N) scale. */
	__m256	  vscale = _mm256_set1_ps(invsq);
	Dimension i		 = 0;
	for (; i + 8 <= dim; i += 8)
	{
		__m256i bb = _mm256_and_si256(
				_mm256_set1_epi32(p->signs1[i >> 3]), lanebits);
		__m256 sign = _mm256_blendv_ps(
				vpos,
				vneg,
				_mm256_castsi256_ps(_mm256_cmpeq_epi32(bb, lanebits)));
		__m256 v = _mm256_mul_ps(
				_mm256_mul_ps(_mm256_loadu_ps(in + i), sign), vscale);
		_mm256_storeu_ps(out + i, v);
	}
	for (; i < dim; i++)
	{
		float s = ((p->signs1[i >> 3] >> (i & 7)) & 1u) ? -1.0f : 1.0f;
		out[i]	= in[i] * s * invsq;
	}

	/* FWHT per block: wide add/sub for h >= 8, scalar for the small tail. */
	for (uint32_t blk = 0; blk < k; blk++)
	{
		float *a = out + (size_t)blk * n;
		for (Dimension h = 1; h < n; h <<= 1)
		{
			if (h >= 8)
			{
				for (Dimension s = 0; s < n; s += (h << 1))
					for (Dimension j = s; j < s + h; j += 8)
					{
						__m256 x = _mm256_loadu_ps(a + j);
						__m256 y = _mm256_loadu_ps(a + j + h);
						_mm256_storeu_ps(a + j, _mm256_add_ps(x, y));
						_mm256_storeu_ps(a + j + h, _mm256_sub_ps(x, y));
					}
			}
			else
			{
				for (Dimension s = 0; s < n; s += (h << 1))
					for (Dimension j = s; j < s + h; j++)
					{
						float x	 = a[j];
						float y	 = a[j + h];
						a[j]	 = x + y;
						a[j + h] = x - y;
					}
			}
		}
	}

	/* K x K mixer across blocks, vectorized across columns. */
	if (k > 1)
	{
		Dimension col = 0;
		for (; col + 8 <= n; col += 8)
		{
			__m256 blkv[VS_FAST_ROTATE_K_MAX];
			for (uint32_t j = 0; j < k; j++)
				blkv[j] = _mm256_loadu_ps(out + (size_t)j * n + col);
			for (uint32_t r = 0; r < k; r++)
			{
				__m256 acc = _mm256_mul_ps(
						_mm256_set1_ps(p->mixer[r * k]), blkv[0]);
				for (uint32_t j = 1; j < k; j++)
					acc = _mm256_fmadd_ps(
							_mm256_set1_ps(p->mixer[r * k + j]), blkv[j], acc);
				_mm256_storeu_ps(out + (size_t)r * n + col, acc);
			}
		}
		for (; col < n; col++)
		{
			float tmp[VS_FAST_ROTATE_K_MAX];
			for (uint32_t j = 0; j < k; j++)
				tmp[j] = out[(size_t)j * n + col];
			for (uint32_t r = 0; r < k; r++)
			{
				float s = 0.0f;
				for (uint32_t j = 0; j < k; j++)
					s += p->mixer[r * k + j] * tmp[j];
				out[(size_t)r * n + col] = s;
			}
		}
	}

	/* D2: post-FWHT sign flip. */
	i = 0;
	for (; i + 8 <= dim; i += 8)
	{
		__m256i bb = _mm256_and_si256(
				_mm256_set1_epi32(p->signs2[i >> 3]), lanebits);
		__m256 sign = _mm256_blendv_ps(
				vpos,
				vneg,
				_mm256_castsi256_ps(_mm256_cmpeq_epi32(bb, lanebits)));
		_mm256_storeu_ps(
				out + i, _mm256_mul_ps(_mm256_loadu_ps(out + i), sign));
	}
	for (; i < dim; i++)
	{
		float s = ((p->signs2[i >> 3] >> (i & 7)) & 1u) ? -1.0f : 1.0f;
		out[i] *= s;
	}
}

/*
 * AVX-512 kernel (16-wide). Same structure as the AVX2 kernel; the packed
 * sign bits map directly onto a __mmask16 so D1/D2 need no compare trick.
 * Uses only avx512f (float arithmetic + masked blend), so it is gated on
 * SIMD_AVX512F alone. FWHT: 512-wide for h >= 16, 256-wide for h == 8,
 * scalar below that.
 */
__attribute__((target("avx512f"))) static void
fast_rotate_apply_avx512(
		const VsFastRotateParams *p, const float *in, float *out)
{
	Dimension	 dim	= p->dim;
	Dimension	 n		= p->fwht_n;
	uint32_t	 k		= p->k;
	float		 invsq	= 1.0f / sqrtf((float)n);
	const __m512 vpos	= _mm512_set1_ps(1.0f);
	const __m512 vneg	= _mm512_set1_ps(-1.0f);
	const __m512 vscale = _mm512_set1_ps(invsq);

	/* D1: sign flip + scale. */
	Dimension i = 0;
	for (; i + 16 <= dim; i += 16)
	{
		__mmask16 m = (__mmask16)(*(const uint16_t *)(p->signs1 + (i >> 3)));
		__m512	  sign = _mm512_mask_blend_ps(m, vpos, vneg);
		__m512	  v	   = _mm512_mul_ps(
				  _mm512_mul_ps(_mm512_loadu_ps(in + i), sign), vscale);
		_mm512_storeu_ps(out + i, v);
	}
	for (; i < dim; i++)
	{
		float s = ((p->signs1[i >> 3] >> (i & 7)) & 1u) ? -1.0f : 1.0f;
		out[i]	= in[i] * s * invsq;
	}

	/* FWHT per block. */
	for (uint32_t blk = 0; blk < k; blk++)
	{
		float *a = out + (size_t)blk * n;
		for (Dimension h = 1; h < n; h <<= 1)
		{
			if (h >= 16)
			{
				for (Dimension s = 0; s < n; s += (h << 1))
					for (Dimension j = s; j < s + h; j += 16)
					{
						__m512 x = _mm512_loadu_ps(a + j);
						__m512 y = _mm512_loadu_ps(a + j + h);
						_mm512_storeu_ps(a + j, _mm512_add_ps(x, y));
						_mm512_storeu_ps(a + j + h, _mm512_sub_ps(x, y));
					}
			}
			else if (h == 8)
			{
				for (Dimension s = 0; s < n; s += 16)
				{
					__m256 x = _mm256_loadu_ps(a + s);
					__m256 y = _mm256_loadu_ps(a + s + 8);
					_mm256_storeu_ps(a + s, _mm256_add_ps(x, y));
					_mm256_storeu_ps(a + s + 8, _mm256_sub_ps(x, y));
				}
			}
			else
			{
				for (Dimension s = 0; s < n; s += (h << 1))
					for (Dimension j = s; j < s + h; j++)
					{
						float x	 = a[j];
						float y	 = a[j + h];
						a[j]	 = x + y;
						a[j + h] = x - y;
					}
			}
		}
	}

	/* K x K mixer across blocks, vectorized across columns. */
	if (k > 1)
	{
		Dimension col = 0;
		for (; col + 16 <= n; col += 16)
		{
			__m512 blkv[VS_FAST_ROTATE_K_MAX];
			for (uint32_t j = 0; j < k; j++)
				blkv[j] = _mm512_loadu_ps(out + (size_t)j * n + col);
			for (uint32_t r = 0; r < k; r++)
			{
				__m512 acc = _mm512_mul_ps(
						_mm512_set1_ps(p->mixer[r * k]), blkv[0]);
				for (uint32_t j = 1; j < k; j++)
					acc = _mm512_fmadd_ps(
							_mm512_set1_ps(p->mixer[r * k + j]), blkv[j], acc);
				_mm512_storeu_ps(out + (size_t)r * n + col, acc);
			}
		}
		for (; col < n; col++)
		{
			float tmp[VS_FAST_ROTATE_K_MAX];
			for (uint32_t j = 0; j < k; j++)
				tmp[j] = out[(size_t)j * n + col];
			for (uint32_t r = 0; r < k; r++)
			{
				float s = 0.0f;
				for (uint32_t j = 0; j < k; j++)
					s += p->mixer[r * k + j] * tmp[j];
				out[(size_t)r * n + col] = s;
			}
		}
	}

	/* D2: post-FWHT sign flip. */
	i = 0;
	for (; i + 16 <= dim; i += 16)
	{
		__mmask16 m = (__mmask16)(*(const uint16_t *)(p->signs2 + (i >> 3)));
		__m512	  sign = _mm512_mask_blend_ps(m, vpos, vneg);
		_mm512_storeu_ps(
				out + i, _mm512_mul_ps(_mm512_loadu_ps(out + i), sign));
	}
	for (; i < dim; i++)
	{
		float s = ((p->signs2[i >> 3] >> (i & 7)) & 1u) ? -1.0f : 1.0f;
		out[i] *= s;
	}
}
#endif /* x86_64 */

#if defined(__aarch64__) || defined(_M_ARM64)
/*
 * NEON kernel (128-bit / 4-wide). Same structure as the AVX2 kernel: D1/D2
 * expand packed sign bits to a +/-1 vector via vtstq/vbslq; FWHT stages with
 * stride h >= 4 use contiguous wide add/sub (smaller stages stay scalar);
 * the K x K mixer is vectorized across 4 columns.
 */
static void
fast_rotate_apply_neon(
		const VsFastRotateParams *p, const float *in, float *out)
{
	Dimension		  dim			  = p->dim;
	Dimension		  n				  = p->fwht_n;
	uint32_t		  k				  = p->k;
	float			  invsq			  = 1.0f / sqrtf((float)n);
	const uint32_t	  lanebits_arr[4] = {1, 2, 4, 8};
	const uint32x4_t  lanebits		  = vld1q_u32(lanebits_arr);
	const float32x4_t vpos			  = vdupq_n_f32(1.0f);
	const float32x4_t vneg			  = vdupq_n_f32(-1.0f);

	/* D1: sign flip + scale. */
	float32x4_t vscale = vdupq_n_f32(invsq);
	Dimension	i	   = 0;
	for (; i + 4 <= dim; i += 4)
	{
		uint32_t	nib	 = (uint32_t)((p->signs1[i >> 3] >> (i & 7)) & 0xF);
		uint32x4_t	iss	 = vtstq_u32(vdupq_n_u32(nib), lanebits);
		float32x4_t sign = vbslq_f32(iss, vneg, vpos);
		float32x4_t v = vmulq_f32(vmulq_f32(vld1q_f32(in + i), sign), vscale);
		vst1q_f32(out + i, v);
	}
	for (; i < dim; i++)
	{
		float s = ((p->signs1[i >> 3] >> (i & 7)) & 1u) ? -1.0f : 1.0f;
		out[i]	= in[i] * s * invsq;
	}

	/* FWHT per block: wide add/sub for h >= 4, scalar below. */
	for (uint32_t blk = 0; blk < k; blk++)
	{
		float *a = out + (size_t)blk * n;
		for (Dimension h = 1; h < n; h <<= 1)
		{
			if (h >= 4)
			{
				for (Dimension s = 0; s < n; s += (h << 1))
					for (Dimension j = s; j < s + h; j += 4)
					{
						float32x4_t x = vld1q_f32(a + j);
						float32x4_t y = vld1q_f32(a + j + h);
						vst1q_f32(a + j, vaddq_f32(x, y));
						vst1q_f32(a + j + h, vsubq_f32(x, y));
					}
			}
			else
			{
				for (Dimension s = 0; s < n; s += (h << 1))
					for (Dimension j = s; j < s + h; j++)
					{
						float x	 = a[j];
						float y	 = a[j + h];
						a[j]	 = x + y;
						a[j + h] = x - y;
					}
			}
		}
	}

	/* K x K mixer across blocks, vectorized across columns. */
	if (k > 1)
	{
		Dimension col = 0;
		for (; col + 4 <= n; col += 4)
		{
			float32x4_t blkv[VS_FAST_ROTATE_K_MAX];
			for (uint32_t j = 0; j < k; j++)
				blkv[j] = vld1q_f32(out + (size_t)j * n + col);
			for (uint32_t r = 0; r < k; r++)
			{
				float32x4_t acc =
						vmulq_f32(vdupq_n_f32(p->mixer[r * k]), blkv[0]);
				for (uint32_t j = 1; j < k; j++)
					acc = vfmaq_f32(
							acc, vdupq_n_f32(p->mixer[r * k + j]), blkv[j]);
				vst1q_f32(out + (size_t)r * n + col, acc);
			}
		}
		for (; col < n; col++)
		{
			float tmp[VS_FAST_ROTATE_K_MAX];
			for (uint32_t j = 0; j < k; j++)
				tmp[j] = out[(size_t)j * n + col];
			for (uint32_t r = 0; r < k; r++)
			{
				float s = 0.0f;
				for (uint32_t j = 0; j < k; j++)
					s += p->mixer[r * k + j] * tmp[j];
				out[(size_t)r * n + col] = s;
			}
		}
	}

	/* D2: post-FWHT sign flip. */
	i = 0;
	for (; i + 4 <= dim; i += 4)
	{
		uint32_t	nib	 = (uint32_t)((p->signs2[i >> 3] >> (i & 7)) & 0xF);
		uint32x4_t	iss	 = vtstq_u32(vdupq_n_u32(nib), lanebits);
		float32x4_t sign = vbslq_f32(iss, vneg, vpos);
		vst1q_f32(out + i, vmulq_f32(vld1q_f32(out + i), sign));
	}
	for (; i < dim; i++)
	{
		float s = ((p->signs2[i >> 3] >> (i & 7)) & 1u) ? -1.0f : 1.0f;
		out[i] *= s;
	}
}
#endif /* aarch64 */

/*
 * Pick the best kernel for the detected CPU. Resolved once and cached in
 * the function pointer below (mirroring the distance/fastscan dispatchers)
 * so the hot path -- which runs per encoded vector at build and once per
 * query -- pays no per-call capability check.
 */
typedef void (*vs_fast_rotate_fn)(
		const VsFastRotateParams *, const float *, float *);

static vs_fast_rotate_fn
resolve_fast_rotate(void)
{
#if defined(__x86_64__) || defined(_M_X64)
	if (vs_has_simd(SIMD_AVX512F))
		return fast_rotate_apply_avx512;
	if (vs_has_simd(SIMD_AVX2))
		return fast_rotate_apply_avx2;
#elif defined(__aarch64__) || defined(_M_ARM64)
	if (vs_has_simd(SIMD_NEON))
		return fast_rotate_apply_neon;
#endif
	return fast_rotate_apply_scalar;
}

void
vs_fast_rotate_apply(const VsFastRotateParams *p, const float *in, float *out)
{
	/* Benign race: concurrent resolvers all compute the same pointer, and a
	 * pointer store is atomic on supported platforms. */
	static vs_fast_rotate_fn fn = NULL;
	if (vs_unlikely(fn == NULL))
		fn = resolve_fast_rotate();
	fn(p, in, out);
}
