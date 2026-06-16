/*
 * avq.h - Anisotropic Vector Quantization partition center (AVQ)
 *
 * Score-aware re-centering of a k-means partition, after ScaNN's
 * anisotropic partitioner (scann/partitioning/anisotropic.cc). For a
 * set of points X assigned to one partition, the AVQ center solves
 *
 *     c = eta * (N*I + (eta-1) * X^T X)^{-1} * sum_i x_i
 *
 * for unit-norm (cosine) data. eta is the "parallel cost multiplier":
 * the per-point loss is ||x-c||^2 + (eta-1)*(<x_hat, x-c>)^2, i.e. the
 * residual component parallel to the data point is penalized eta times
 * more than the orthogonal component. eta = 1 (or NaN) reduces to the
 * ordinary mean. eta > 1 pulls the center to better preserve the
 * query-relevant (parallel) direction, which improves routing for
 * boundary points whose cluster mean would otherwise sit far away.
 *
 * Intended as a one-shot recentering applied after standard k-means
 * (matching ScaNN's avq_after_primary), not inside every iteration.
 */

#ifndef MKT_AVQ_H
#define MKT_AVQ_H

#include <stdint.h>

#include "mkt_types.h"

/*
 * Compute the AVQ center of n unit-norm vectors (pts is [n*dim],
 * row-major) into out_center [dim].
 *
 * eta <= 1 or NaN: writes the plain mean.
 * Returns 0 on success. On a numerical failure (non-SPD system) it
 * falls back to the mean and returns -1.
 */
int mkt_avq_center(
		const float *pts,
		uint32_t	 n,
		Dimension	 dim,
		float		 eta,
		float		*out_center);

#endif /* MKT_AVQ_H */
