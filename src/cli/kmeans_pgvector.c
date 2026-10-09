/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * Portions derived from pgvector's src/ivfkmeans.c
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
 * kmeans_pgvector.c - pgvector's Elkan k-means, ported for benchmarking
 *
 * This is a minimally modified copy of pgvector/src/ivfkmeans.c.
 * Changes from the original:
 * - All PostgreSQL includes replaced with kmeans_pgvector.h shim
 * - NormCenters: uses vs_l2_norm instead of PG function call
 * - CheckNorms: uses vs_l2_norm inline
 * - IvfflatKmeans renamed, memory context replaced with direct alloc
 * - pgvector_kmeans() entry point packs flat floats into VectorArray
 *
 * The Elkan algorithm loop, k-means++ init, bounds math, convergence
 * detection, and 500 iteration limit are all unchanged from pgvector.
 */

#include "kmeans_pgvector.h"

/* Thread-local state for the compatibility shim */
_Thread_local double			 pgv_distance_result;
_Thread_local DistanceMetric	 pgv_active_metric;
_Thread_local PgvXoshiro256State pgv_rng;
_Thread_local uint32_t			 pgv_max_iterations = 500;

/* ================================================================
 * Below is pgvector's ivfkmeans.c with minimal modifications.
 * Lines marked [MODIFIED] indicate changes from the original.
 * ================================================================ */

/*
 * Initialize with kmeans++
 *
 * https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf
 */
static void
InitCenters(
		Relation	index,
		VectorArray samples,
		VectorArray centers,
		float	   *lowerBound)
{
	FmgrInfo *procinfo;
	Oid		  collation;
	int64	  j;
	float	 *weight	 = palloc(samples->length * sizeof(float));
	int		  numCenters = centers->maxlen;
	int		  numSamples = samples->length;

	procinfo  = index_getprocinfo(index, 1, IVFFLAT_KMEANS_DISTANCE_PROC);
	collation = index->rd_indcollation[0];

	/* Choose an initial center uniformly at random */
	/* [MODIFIED: use uniform*N instead of next()%N to match
	 * pg_vectorsearch RNG] */
	{
		uint32_t first = (uint32_t)(RandomDouble() * samples->length);
		if ((int)first >= samples->length)
			first = (uint32_t)(samples->length - 1);
		VectorArraySet(centers, 0, VectorArrayGet(samples, first));
	}
	centers->length++;

	for (j = 0; j < numSamples; j++)
		weight[j] = FLT_MAX;

	for (int i = 0; i < numCenters; i++)
	{
		double sum;
		double choice;

		CHECK_FOR_INTERRUPTS();

		sum = 0.0;

		for (j = 0; j < numSamples; j++)
		{
			Datum  vec = PointerGetDatum(VectorArrayGet(samples, j));
			double distance;

			/* Only need to compute distance for new center */
			distance = DatumGetFloat8(FunctionCall2Coll(
					procinfo,
					collation,
					vec,
					PointerGetDatum(VectorArrayGet(centers, i))));

			/* Set lower bound */
			lowerBound[j * numCenters + i] = (float)distance;

			/*
			 * Use distance squared for weighted probability distribution.
			 * [MODIFIED: use L2 squared directly to match pg_vectorsearch]
			 */
			{
				const Vector *va = (const Vector *)VectorArrayGet(samples, j);
				const Vector *vb = (const Vector *)VectorArrayGet(centers, i);
				float		  d2 = vs_l2_distance_squared(
						va->x, vb->x, (Dimension)va->dim);

				if (d2 < weight[j])
					weight[j] = d2;
			}

			sum += weight[j];
		}

		/* Only compute lower bound on last iteration */
		if (i + 1 == numCenters)
			break;

		/* Choose new center using weighted probability distribution. */
		choice = sum * RandomDouble();
		for (j = 0; j < numSamples - 1; j++)
		{
			choice -= weight[j];
			if (choice <= 0)
				break;
		}

		VectorArraySet(centers, i + 1, VectorArrayGet(samples, j));
		centers->length++;
	}

	pfree(weight);
}

/*
 * Norm centers [MODIFIED: standalone normalization using vs_l2_norm]
 */
static void
NormCenters(
		const IvfflatTypeInfo *typeInfo, Oid collation, VectorArray centers)
{
	(void)typeInfo;
	(void)collation;

	for (int j = 0; j < centers->length; j++)
	{
		Vector *vec	 = (Vector *)VectorArrayGet(centers, j);
		float	norm = vs_l2_norm(vec->x, (Dimension)vec->dim);
		if (norm > 1e-30f)
		{
			float inv = 1.0f / norm;
			for (int i = 0; i < vec->dim; i++)
				vec->x[i] *= inv;
		}
	}
}

/*
 * Quick approach if we have no data
 */
static void
RandomCenters(
		Relation index, VectorArray centers, const IvfflatTypeInfo *typeInfo)
{
	int		  dimensions = centers->dim;
	FmgrInfo *normprocinfo =
			IvfflatOptionalProcInfo(index, IVFFLAT_KMEANS_NORM_PROC);
	Oid	   collation = index->rd_indcollation[0];
	float *x		 = (float *)palloc(sizeof(float) * dimensions);

	/* Fill with random data */
	while (centers->length < centers->maxlen)
	{
		Pointer center = VectorArrayGet(centers, centers->length);

		for (int i = 0; i < dimensions; i++)
			x[i] = (float)RandomDouble();

		typeInfo->updateCenter(center, dimensions, x);

		centers->length++;
	}

	if (normprocinfo != NULL)
		NormCenters(typeInfo, collation, centers);
}

/*
 * Sum centers
 */
static void
SumCenters(
		VectorArray			   samples,
		float				  *agg,
		int					  *closestCenters,
		const IvfflatTypeInfo *typeInfo)
{
	for (int j = 0; j < samples->length; j++)
	{
		float *x = agg + ((int64)closestCenters[j] * samples->dim);

		typeInfo->sumCenter(VectorArrayGet(samples, j), x);
	}
}

/*
 * Update centers
 */
static void
UpdateCenters(float *agg, VectorArray centers, const IvfflatTypeInfo *typeInfo)
{
	for (int j = 0; j < centers->length; j++)
	{
		float *x = agg + ((int64)j * centers->dim);

		typeInfo->updateCenter(VectorArrayGet(centers, j), centers->dim, x);
	}
}

/*
 * Compute new centers
 */
static void
ComputeNewCenters(
		VectorArray			   samples,
		float				  *agg,
		VectorArray			   newCenters,
		int					  *centerCounts,
		int					  *closestCenters,
		FmgrInfo			  *normprocinfo,
		Oid					   collation,
		const IvfflatTypeInfo *typeInfo)
{
	int dimensions = newCenters->dim;
	int numCenters = newCenters->length;
	int numSamples = samples->length;

	/* Reset sum and count */
	for (int j = 0; j < numCenters; j++)
	{
		float *x = agg + ((int64)j * dimensions);

		for (int k = 0; k < dimensions; k++)
			x[k] = 0.0;

		centerCounts[j] = 0;
	}

	/* Increment sum of closest center */
	SumCenters(samples, agg, closestCenters, typeInfo);

	/* Increment count of closest center */
	for (int j = 0; j < numSamples; j++)
		centerCounts[closestCenters[j]] += 1;

	/* Divide sum by count */
	for (int j = 0; j < numCenters; j++)
	{
		float *x = agg + ((int64)j * dimensions);

		if (centerCounts[j] > 0)
		{
			for (int k = 0; k < dimensions; k++)
			{
				if (isinf(x[k]))
					x[k] = x[k] > 0 ? FLT_MAX : -FLT_MAX;
			}

			for (int k = 0; k < dimensions; k++)
				x[k] /= centerCounts[j];
		}
		else
		{
			for (int k = 0; k < dimensions; k++)
				x[k] = (float)RandomDouble();
		}
	}

	/* Set new centers */
	UpdateCenters(agg, newCenters, typeInfo);

	/* Normalize if needed */
	if (normprocinfo != NULL)
		NormCenters(typeInfo, collation, newCenters);
}

/*
 * Use Elkan for performance. This requires distance function to
 * satisfy triangle inequality.
 *
 * We use L2 distance for L2 (not L2 squared like index scan)
 * and angular distance for inner product and cosine distance
 *
 * https://www.aaai.org/Papers/ICML/2003/ICML03-022.pdf
 */
static void
ElkanKmeans(
		Relation			   index,
		VectorArray			   samples,
		VectorArray			   centers,
		const IvfflatTypeInfo *typeInfo)
{
	FmgrInfo   *procinfo;
	FmgrInfo   *normprocinfo;
	Oid			collation;
	int			dimensions = centers->dim;
	int			numCenters = centers->maxlen;
	int			numSamples = samples->length;
	VectorArray newCenters;
	float	   *agg;
	int		   *centerCounts;
	int		   *closestCenters;
	float	   *lowerBound;
	float	   *upperBound;
	float	   *s;
	float	   *halfcdist;
	float	   *newcdist;

	/* Calculate allocation sizes */
	Size aggSize			= sizeof(float) * (int64)numCenters * dimensions;
	Size centerCountsSize	= sizeof(int) * numCenters;
	Size closestCentersSize = sizeof(int) * numSamples;
	Size lowerBoundSize		= sizeof(float) * numSamples * numCenters;
	Size upperBoundSize		= sizeof(float) * numSamples;
	Size sSize				= sizeof(float) * numCenters;
	Size halfcdistSize		= sizeof(float) * numCenters * numCenters;
	Size newcdistSize		= sizeof(float) * numCenters;

	/* [MODIFIED: removed maintenance_work_mem check] */

	/* Ensure indexing does not overflow. Widened first: the product of
	 * two ints cannot exceed INT_MAX in any defined way, so the check
	 * never fires when computed in int. */
	if ((int64)numCenters * numCenters > INT_MAX)
		elog(ERROR, "Indexing overflow detected.");

	/* Set support functions */
	procinfo	 = index_getprocinfo(index, 1, IVFFLAT_KMEANS_DISTANCE_PROC);
	normprocinfo = IvfflatOptionalProcInfo(index, IVFFLAT_KMEANS_NORM_PROC);
	collation	 = index->rd_indcollation[0];

	/* Allocate space */
	agg			   = palloc(aggSize);
	centerCounts   = palloc(centerCountsSize);
	closestCenters = palloc(closestCentersSize);
	lowerBound	   = palloc_extended(lowerBoundSize, MCXT_ALLOC_HUGE);
	upperBound	   = palloc(upperBoundSize);
	s			   = palloc(sSize);
	halfcdist	   = palloc_extended(halfcdistSize, MCXT_ALLOC_HUGE);
	newcdist	   = palloc(newcdistSize);

	/* Initialize new centers */
	newCenters = VectorArrayInit(numCenters, dimensions, centers->itemsize);
	newCenters->length = numCenters;

	/* Pick initial centers */
	InitCenters(index, samples, centers, lowerBound);

	/* Assign each x to its closest initial center */
	for (int64 j = 0; j < numSamples; j++)
	{
		float minDistance	= FLT_MAX;
		int	  closestCenter = 0;

		for (int64 k = 0; k < numCenters; k++)
		{
			float distance = lowerBound[(size_t)j * numCenters + k];

			if (distance < minDistance)
			{
				minDistance	  = distance;
				closestCenter = (int)k;
			}
		}

		upperBound[j]	  = minDistance;
		closestCenters[j] = closestCenter;
	}

	/* Give 500 iterations to converge */
	for (int iteration = 0; (uint32_t)iteration < pgv_max_iterations;
		 iteration++)
	{
		int	 changes = 0;
		bool rjreset;

		CHECK_FOR_INTERRUPTS();

		/* Step 1: For all centers, compute distance */
		for (int64 j = 0; j < numCenters; j++)
		{
			Datum vec = PointerGetDatum(VectorArrayGet(centers, j));

			for (int64 k = j + 1; k < numCenters; k++)
			{
				float distance = 0.5f *
								 (float)DatumGetFloat8(FunctionCall2Coll(
										 procinfo,
										 collation,
										 vec,
										 PointerGetDatum(
												 VectorArrayGet(centers, k))));

				halfcdist[j * numCenters + k] = distance;
				halfcdist[k * numCenters + j] = distance;
			}
		}

		/* For all centers c, compute s(c) */
		for (int64 j = 0; j < numCenters; j++)
		{
			float minDistance = FLT_MAX;

			for (int64 k = 0; k < numCenters; k++)
			{
				float distance;

				if (j == k)
					continue;

				distance = halfcdist[j * numCenters + k];
				if (distance < minDistance)
					minDistance = distance;
			}

			s[j] = minDistance;
		}

		rjreset = iteration != 0;

		for (int64 j = 0; j < numSamples; j++)
		{
			bool rj;

			/* Step 2 */
			if (upperBound[j] <= s[closestCenters[j]])
				continue;

			rj = rjreset;

			for (int64 k = 0; k < numCenters; k++)
			{
				Datum vec;
				float dxcx;

				/* Step 3 */
				if (k == closestCenters[j])
					continue;

				if (upperBound[j] <= lowerBound[(size_t)j * numCenters + k])
					continue;

				if (upperBound[j] <=
					halfcdist[(size_t)closestCenters[j] * numCenters + k])
					continue;

				vec = PointerGetDatum(VectorArrayGet(samples, j));

				/* Step 3a */
				if (rj)
				{
					dxcx = (float)DatumGetFloat8(FunctionCall2Coll(
							procinfo,
							collation,
							vec,
							PointerGetDatum(VectorArrayGet(
									centers, closestCenters[j]))));

					lowerBound[j * numCenters + closestCenters[j]] = dxcx;
					upperBound[j]								   = dxcx;

					rj = false;
				}
				else
					dxcx = upperBound[j];

				/* Step 3b */
				if (dxcx > lowerBound[(size_t)j * numCenters + k] ||
					dxcx > halfcdist
									[(size_t)closestCenters[j] * numCenters +
									 k])
				{
					float dxc = (float)DatumGetFloat8(FunctionCall2Coll(
							procinfo,
							collation,
							vec,
							PointerGetDatum(VectorArrayGet(centers, k))));

					lowerBound[(size_t)j * numCenters + k] = dxc;

					if (dxc < dxcx)
					{
						closestCenters[j] = (int)k;
						upperBound[j]	  = dxc;
						changes++;
					}
				}
			}
		}

		/* Step 4 */
		ComputeNewCenters(
				samples,
				agg,
				newCenters,
				centerCounts,
				closestCenters,
				normprocinfo,
				collation,
				typeInfo);

		/* Step 5 */
		for (int j = 0; j < numCenters; j++)
			newcdist[j] = (float)DatumGetFloat8(FunctionCall2Coll(
					procinfo,
					collation,
					PointerGetDatum(VectorArrayGet(centers, j)),
					PointerGetDatum(VectorArrayGet(newCenters, j))));

		for (int64 j = 0; j < numSamples; j++)
		{
			for (int64 k = 0; k < numCenters; k++)
			{
				float distance = lowerBound[(size_t)j * numCenters + k] -
								 newcdist[k];

				if (distance < 0)
					distance = 0;

				lowerBound[(size_t)j * numCenters + k] = distance;
			}
		}

		/* Step 6 */
		for (int j = 0; j < numSamples; j++)
			upperBound[j] += newcdist[closestCenters[j]];

		/* Step 7 */
		for (int j = 0; j < numCenters; j++)
			VectorArraySet(centers, j, VectorArrayGet(newCenters, j));

		if (changes == 0 && iteration != 0)
			break;
	}

	/* Cleanup */
	VectorArrayFree(newCenters);
	pfree(agg);
	pfree(centerCounts);
	pfree(closestCenters);
	pfree(lowerBound);
	pfree(upperBound);
	pfree(s);
	pfree(halfcdist);
	pfree(newcdist);
}

/*
 * Ensure no NaN or infinite values
 */
static void
CheckElements(VectorArray centers, const IvfflatTypeInfo *typeInfo)
{
	float *scratch = palloc(sizeof(float) * centers->dim);

	for (int i = 0; i < centers->length; i++)
	{
		for (int j = 0; j < centers->dim; j++)
			scratch[j] = 0;

		typeInfo->sumCenter(VectorArrayGet(centers, i), scratch);

		for (int j = 0; j < centers->dim; j++)
		{
			if (isnan(scratch[j]))
				elog(ERROR, "NaN detected.");

			if (isinf(scratch[j]))
				elog(ERROR, "Infinite value detected.");
		}
	}

	pfree(scratch);
}

/*
 * Ensure no zero vectors for cosine distance
 * [MODIFIED: standalone norm check using vs_l2_norm]
 */
static void
CheckNorms(VectorArray centers)
{
	if (pgv_active_metric != DISTANCE_INNER_PRODUCT &&
		pgv_active_metric != DISTANCE_COSINE)
		return;

	for (int i = 0; i < centers->length; i++)
	{
		Vector *vec	 = (Vector *)VectorArrayGet(centers, i);
		float	norm = vs_l2_norm(vec->x, (Dimension)vec->dim);

		if (norm == 0)
			elog(ERROR, "Zero norm detected.");
	}
}

/*
 * Detect issues with centers
 * [MODIFIED: takes no Relation, calls CheckNorms without index]
 */
static void
CheckCenters(VectorArray centers, const IvfflatTypeInfo *typeInfo)
{
	if (centers->length != centers->maxlen)
		elog(ERROR, "Not enough centers.");

	CheckElements(centers, typeInfo);
	CheckNorms(centers);
}

/*
 * Run Elkan k-means (replaces IvfflatKmeans)
 * [MODIFIED: no PG memory context, direct alloc/free]
 */
static void
RunElkanKmeans(
		Relation			   index,
		VectorArray			   samples,
		VectorArray			   centers,
		const IvfflatTypeInfo *typeInfo)
{
	if (samples->length == 0)
		RandomCenters(index, centers, typeInfo);
	else
		ElkanKmeans(index, samples, centers, typeInfo);

	CheckCenters(centers, typeInfo);
}

/* ================================================================
 * Public entry point: pgvector_kmeans()
 *
 * Packs flat float vectors into pgvector's VectorArray format,
 * runs Elkan k-means, and extracts results into KMeansResult.
 * ================================================================ */

/*
 * Compute final assignments and cost for a set of centroids.
 */
static KMeansResult *
pgv_extract_result(
		const float	  *vectors,
		uint32_t	   nvecs,
		Dimension	   dim,
		uint32_t	   nlist,
		DistanceMetric metric,
		const float	  *centroids)
{
	KMeansResult *result  = vs_alloc0(sizeof(KMeansResult));
	result->nlist		  = nlist;
	result->dim			  = dim;
	result->centroids	  = vs_alloc((size_t)nlist * dim * sizeof(float));
	result->assignments	  = vs_alloc(nvecs * sizeof(ClusterId));
	result->cluster_sizes = vs_alloc0(nlist * sizeof(uint32_t));
	result->total_cost	  = 0.0f;

	memcpy(result->centroids, centroids, (size_t)nlist * dim * sizeof(float));

	for (uint32_t i = 0; i < nvecs; i++)
	{
		const float *vec	  = vectors + (size_t)i * dim;
		float		 min_dist = FLT_MAX;
		uint32_t	 min_j	  = 0;

		for (uint32_t j = 0; j < nlist; j++)
		{
			float d;
			switch (metric)
			{
			case DISTANCE_L2:
				d = vs_l2_distance_squared(
						vec, centroids + (size_t)j * dim, dim);
				break;
			case DISTANCE_INNER_PRODUCT:
				d = -vs_dot_product(vec, centroids + (size_t)j * dim, dim);
				break;
			case DISTANCE_COSINE:
				d = 1.0f -
					vs_dot_product(vec, centroids + (size_t)j * dim, dim);
				break;
			default:
				d = FLT_MAX;
			}
			if (d < min_dist)
			{
				min_dist = d;
				min_j	 = j;
			}
		}

		result->assignments[i] = min_j;
		result->cluster_sizes[min_j]++;
		result->total_cost += min_dist;
	}

	return result;
}

KMeansResult *
pgvector_kmeans(
		const float			*vectors,
		uint32_t			 nvecs,
		Dimension			 dim,
		uint32_t			 nlist,
		DistanceMetric		 metric,
		const KMeansOptions *options)
{
	if (vectors == NULL || nvecs == 0 || dim == 0 || nlist == 0)
		return NULL;

	if (nlist > nvecs)
		nlist = nvecs;

	KMeansOptions opts = VS_KMEANS_OPTIONS_DEFAULT;
	if (options != NULL)
		opts = *options;
	if (opts.nredo == 0)
		opts.nredo = 1;

	/* Set up thread-local state */
	pgv_active_metric  = metric;
	pgv_max_iterations = opts.max_iterations;

	/* Type info for float vectors */
	static const IvfflatTypeInfo typeInfo = {
			.maxDimensions = IVFFLAT_MAX_DIM,
			.normalize	   = NULL,
			.itemSize	   = PgVectorItemSize,
			.updateCenter  = PgVectorUpdateCenter,
			.sumCenter	   = PgVectorSumCenter,
	};

	/* Create fake Relation for support function dispatch */
	PgvRelationData rel_data = {
			.metric	   = metric,
			.collation = 0,
	};
	Relation index = &rel_data;

	/* Pack input vectors into VectorArray (shared across redos) */
	Size		itemsize = VECTOR_SIZE(dim);
	VectorArray samples	 = VectorArrayInit((int)nvecs, (int)dim, itemsize);
	samples->length		 = (int)nvecs;

	for (uint32_t i = 0; i < nvecs; i++)
	{
		Vector *vec = (Vector *)VectorArrayGet(samples, (int)i);
		SET_VARSIZE(vec, VECTOR_SIZE(dim));
		vec->dim	= (int16)dim;
		vec->unused = 0;
		memcpy(vec->x, vectors + (size_t)i * dim, dim * sizeof(float));
	}

	KMeansResult *best		= NULL;
	float		  best_cost = FLT_MAX;

	for (uint32_t redo = 0; redo < opts.nredo; redo++)
	{
		/* Seed PRNG for this attempt */
		pgv_xo_seed(&pgv_rng, opts.seed + redo);

		/* Fresh centers for each attempt */
		VectorArray centers = VectorArrayInit((int)nlist, (int)dim, itemsize);

		RunElkanKmeans(index, samples, centers, &typeInfo);

		/* Extract centroids into flat array for cost computation */
		float *centroids = vs_alloc((size_t)nlist * dim * sizeof(float));
		for (uint32_t j = 0; j < nlist; j++)
		{
			Vector *vec = (Vector *)VectorArrayGet(centers, (int)j);
			memcpy(centroids + (size_t)j * dim, vec->x, dim * sizeof(float));
		}

		KMeansResult *result = pgv_extract_result(
				vectors, nvecs, dim, nlist, metric, centroids);

		if (result->total_cost < best_cost)
		{
			if (best != NULL)
				vs_kmeans_result_destroy(best);
			best	  = result;
			best_cost = result->total_cost;
		}
		else
		{
			vs_kmeans_result_destroy(result);
		}

		vs_free(centroids);
		VectorArrayFree(centers);
	}

	VectorArrayFree(samples);
	return best;
}
