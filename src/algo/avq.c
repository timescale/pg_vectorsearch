/*
 * avq.c - Anisotropic Vector Quantization partition center
 *
 * See avq.h. Implements the ScaNN anisotropic center for unit-norm
 * data: c = eta * (N*I + (eta-1) * G)^{-1} * s, where G = X^T X and
 * s = sum_i x_i. The Gram matrix G uses CBLAS (ssyrk) when available;
 * the d x d SPD system is solved with a self-contained Cholesky
 * factorization in double precision (one solve per partition, run
 * once after k-means, parallelized across partitions by the caller).
 */

#include "mkt_config.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "algo/avq.h"
#include "core/log.h"
#include "core/memory.h"

#ifdef MKT_HAVE_CBLAS
#if defined(__APPLE__)
#include <vecLib/cblas_new.h>
#else
#include <cblas.h>
#endif
#endif

/* Plain mean of n unit-norm vectors into out (the eta == 1 case). */
static void
avq_mean(const float *pts, uint32_t n, Dimension dim, float *out)
{
	memset(out, 0, (size_t)dim * sizeof(float));
	if (n == 0)
		return;
	for (uint32_t i = 0; i < n; i++)
	{
		const float *x = pts + (size_t)i * dim;
		for (Dimension d = 0; d < dim; d++)
			out[d] += x[d];
	}
	float inv = 1.0f / (float)n;
	for (Dimension d = 0; d < dim; d++)
		out[d] *= inv;
}

/*
 * Accumulate the lower triangle of the Gram matrix G = X^T X into a[]
 * (row-major d x d, only entries with col <= row written/used).
 */
static void
avq_gram_lower(const float *pts, uint32_t n, Dimension dim, double *a)
{
	memset(a, 0, (size_t)dim * dim * sizeof(double));

#ifdef MKT_HAVE_CBLAS
	/* ssyrk computes the lower triangle of X^T X in float; copy out. */
	float *g = mkt_alloc((size_t)dim * dim * sizeof(float));
	memset(g, 0, (size_t)dim * dim * sizeof(float));
	cblas_ssyrk(
			CblasRowMajor,
			CblasLower,
			CblasTrans, /* op(A) = A^T, so C = A^T A */
			(int)dim,	/* N: order of C */
			(int)n,		/* K: contracted dimension */
			1.0f,
			pts,
			(int)dim, /* lda: row stride of X (n x dim, row-major) */
			0.0f,
			g,
			(int)dim);
	for (Dimension r = 0; r < dim; r++)
		for (Dimension c = 0; c <= r; c++)
			a[(size_t)r * dim + c] = (double)g[(size_t)r * dim + c];
	mkt_free(g);
#else
	for (uint32_t i = 0; i < n; i++)
	{
		const float *x = pts + (size_t)i * dim;
		for (Dimension r = 0; r < dim; r++)
		{
			double xr = (double)x[r];
			double *ar = a + (size_t)r * dim;
			for (Dimension c = 0; c <= r; c++)
				ar[c] += xr * (double)x[c];
		}
	}
#endif
}

/*
 * In-place Cholesky factorization of the SPD matrix a (row-major d x d,
 * lower triangle). On return the lower triangle holds L with A = L L^T.
 * Returns 0 on success, -1 if a non-positive pivot is encountered.
 */
static int
avq_cholesky_lower(double *a, Dimension dim)
{
	for (Dimension j = 0; j < dim; j++)
	{
		double *aj	= a + (size_t)j * dim;
		double	sum = aj[j];
		for (Dimension k = 0; k < j; k++)
			sum -= aj[k] * aj[k];
		if (sum <= 0.0)
			return -1;
		double ljj = sqrt(sum);
		aj[j]	   = ljj;

		double inv_ljj = 1.0 / ljj;
		for (Dimension i = j + 1; i < dim; i++)
		{
			double *ai = a + (size_t)i * dim;
			double	s  = ai[j];
			for (Dimension k = 0; k < j; k++)
				s -= ai[k] * aj[k];
			ai[j] = s * inv_ljj;
		}
	}
	return 0;
}

/* Solve L L^T x = b in place (b -> x), L lower-triangular d x d. */
static void
avq_cholesky_solve(const double *l, double *b, Dimension dim)
{
	/* Forward: L y = b */
	for (Dimension i = 0; i < dim; i++)
	{
		const double *li = l + (size_t)i * dim;
		double		  s	 = b[i];
		for (Dimension k = 0; k < i; k++)
			s -= li[k] * b[k];
		b[i] = s / li[i];
	}
	/* Backward: L^T x = y */
	for (Dimension ii = 0; ii < dim; ii++)
	{
		Dimension i = dim - 1 - ii;
		double	  s = b[i];
		for (Dimension k = i + 1; k < dim; k++)
			s -= l[(size_t)k * dim + i] * b[k];
		b[i] = s / l[(size_t)i * dim + i];
	}
}

int
mkt_avq_center(
		const float *pts,
		uint32_t	 n,
		Dimension	 dim,
		float		 eta,
		float		*out_center)
{
	/* eta <= 1 (or NaN): ordinary mean, matching ScaNN's isotropic case. */
	if (n == 0 || isnan(eta) || eta <= 1.0f)
	{
		avq_mean(pts, n, dim, out_center);
		return 0;
	}

	double *a = mkt_alloc((size_t)dim * dim * sizeof(double));
	double *b = mkt_alloc((size_t)dim * sizeof(double));

	/* A = N*I + (eta-1)*G  (lower triangle); b = sum_i x_i */
	avq_gram_lower(pts, n, dim, a);
	double scale = (double)eta - 1.0;
	for (Dimension r = 0; r < dim; r++)
	{
		double *ar = a + (size_t)r * dim;
		for (Dimension c = 0; c <= r; c++)
			ar[c] *= scale;
		ar[r] += (double)n;
	}
	memset(b, 0, (size_t)dim * sizeof(double));
	for (uint32_t i = 0; i < n; i++)
	{
		const float *x = pts + (size_t)i * dim;
		for (Dimension d = 0; d < dim; d++)
			b[d] += (double)x[d];
	}

	int rc = avq_cholesky_lower(a, dim);
	if (rc != 0)
	{
		/* Numerically non-SPD (degenerate partition): fall back to mean. */
		mkt_free(a);
		mkt_free(b);
		avq_mean(pts, n, dim, out_center);
		return -1;
	}

	avq_cholesky_solve(a, b, dim); /* b <- (N*I+(eta-1)G)^{-1} sum_x */
	for (Dimension d = 0; d < dim; d++)
		out_center[d] = (float)((double)eta * b[d]);

	mkt_free(a);
	mkt_free(b);
	return 0;
}
