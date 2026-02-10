/*
 * kmeans_pgvector.h - PostgreSQL compatibility shim for pgvector ivfkmeans.c
 *
 * Provides the minimum types, macros, and function stubs needed to compile
 * pgvector's Elkan k-means implementation standalone (without PostgreSQL)
 * for benchmarking against meerkat's Lloyd+BLAS k-means.
 *
 * Memory: Uses meerkat's arena allocator (mkt_alloc/mkt_free).
 * Distance: Dispatches through meerkat's vecops functions.
 * Random: Uses xoshiro256** PRNG (same as meerkat's kmeans.c).
 */

#ifndef MKT_KMEANS_PGVECTOR_H
#define MKT_KMEANS_PGVECTOR_H

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "algo/kmeans.h"
#include "algo/vecops.h"
#include "core/memory.h"
#include "mkt_types.h"

/* ----------------------------------------------------------------
 * PostgreSQL type compatibility
 * ---------------------------------------------------------------- */
typedef uintptr_t	 Datum;
typedef unsigned int Oid;
typedef char		*Pointer;
typedef size_t		 Size;

typedef int16_t	 int16;
typedef int32_t	 int32;
typedef int64_t	 int64;
typedef uint16_t uint16;
typedef uint32_t uint32;

#define InvalidOid			  0
#define FLEXIBLE_ARRAY_MEMBER /* empty, C99+ */
#define MAXALIGN(x)			  (((x) + 7) & ~7)

/* ----------------------------------------------------------------
 * Datum conversion macros
 * ---------------------------------------------------------------- */
#define PointerGetDatum(p) ((Datum)(p))
#define DatumGetPointer(d) ((Pointer)(d))
#define DatumGetFloat8(d)  ((void)(d), pgv_distance_result)

/* ----------------------------------------------------------------
 * Varlena macros (simplified for flat float vectors)
 * ---------------------------------------------------------------- */
#define SET_VARSIZE(v, sz) ((v)->vl_len_ = (int32)(sz))
#define VARSIZE_ANY(p)	   (((PgVector *)(p))->vl_len_)
#define VECTOR_SIZE(dim)   (offsetof(PgVector, x) + sizeof(float) * (dim))

/* ----------------------------------------------------------------
 * PgVector - pgvector's Vector struct (renamed to avoid conflicts)
 * ---------------------------------------------------------------- */
typedef struct PgVector
{
	int32 vl_len_;
	int16 dim;
	int16 unused;
	float x[FLEXIBLE_ARRAY_MEMBER];
} PgVector;

/* pgvector uses 'Vector' throughout ivfkmeans.c */
typedef PgVector Vector;

/* ----------------------------------------------------------------
 * VectorArray - from pgvector ivfflat.h
 * ---------------------------------------------------------------- */
typedef struct VectorArrayData
{
	int	  length;
	int	  maxlen;
	int	  dim;
	Size  itemsize;
	char *items;
} VectorArrayData;

typedef VectorArrayData *VectorArray;

static inline Pointer
VectorArrayGet(VectorArray arr, int offset)
{
	return arr->items + ((size_t)offset * arr->itemsize);
}

static inline void
VectorArraySet(VectorArray arr, int offset, Pointer val)
{
	memcpy(VectorArrayGet(arr, offset), val, VARSIZE_ANY(val));
}

#define VECTOR_ARRAY_SIZE(length, size) \
	(sizeof(VectorArrayData) + (size_t)(length) * MAXALIGN(size))

static inline VectorArray
VectorArrayInit(int maxlen, int dimensions, Size itemsize)
{
	VectorArray res = mkt_alloc(sizeof(VectorArrayData));
	itemsize		= MAXALIGN(itemsize);
	res->length		= 0;
	res->maxlen		= maxlen;
	res->dim		= dimensions;
	res->itemsize	= itemsize;
	res->items		= mkt_alloc0((size_t)maxlen * itemsize);
	return res;
}

static inline void
VectorArrayFree(VectorArray arr)
{
	mkt_free(arr->items);
	mkt_free(arr);
}

/* ----------------------------------------------------------------
 * IvfflatTypeInfo - from pgvector ivfflat.h
 * ---------------------------------------------------------------- */
typedef struct IvfflatTypeInfo
{
	int	  maxDimensions;
	void *normalize; /* unused in standalone */
	Size (*itemSize)(int dimensions);
	void (*updateCenter)(Pointer v, int dimensions, float *x);
	void (*sumCenter)(Pointer v, float *x);
} IvfflatTypeInfo;

/* Vector type implementations (from pgvector ivfutils.c) */
static inline Size
PgVectorItemSize(int dimensions)
{
	return VECTOR_SIZE(dimensions);
}

static inline void
PgVectorUpdateCenter(Pointer v, int dimensions, float *x)
{
	Vector *vec = (Vector *)v;
	SET_VARSIZE(vec, VECTOR_SIZE(dimensions));
	vec->dim = (int16)dimensions;
	for (int i = 0; i < dimensions; i++)
		vec->x[i] = x[i];
}

static inline void
PgVectorSumCenter(Pointer v, float *x)
{
	Vector *vec = (Vector *)v;
	int		dim = vec->dim;
	for (int i = 0; i < dim; i++)
		x[i] += vec->x[i];
}

/* ----------------------------------------------------------------
 * FmgrInfo / Relation / support function stubs
 *
 * pgvector uses PostgreSQL's function manager to dispatch distance
 * functions. We replace this with a simple static context holding
 * the metric and a pre-set function pointer.
 * ---------------------------------------------------------------- */
typedef struct FmgrInfo
{
	int dummy;
} FmgrInfo;

/* Relation: pointer to context struct with metric + collation */
typedef struct PgvRelationData
{
	DistanceMetric metric;
	Oid			   collation;
	FmgrInfo	   dist_procinfo;
	FmgrInfo	   norm_procinfo;
	Oid			   rd_indcollation[1]; /* pgvector accesses [0] */
} PgvRelationData;

typedef PgvRelationData *Relation;

/* ----------------------------------------------------------------
 * Distance dispatch
 *
 * pgvector calls: FunctionCall2Coll(procinfo, collation, v1, v2)
 * We compute the distance using meerkat's vecops and store the
 * result in a thread-local so DatumGetFloat8() can retrieve it.
 *
 * For Elkan k-means, pgvector uses:
 * - L2 distance (not squared) for L2 metric (triangle inequality)
 * - Angular distance for IP/cosine
 * ---------------------------------------------------------------- */

/* Thread-local distance result for the FunctionCall2Coll -> DatumGetFloat8
 * pattern */
extern _Thread_local double pgv_distance_result;

/* Stored metric for dispatch */
extern _Thread_local DistanceMetric pgv_active_metric;

static inline double
pgv_compute_distance(const Vector *a, const Vector *b, DistanceMetric metric)
{
	int dim = a->dim;

	switch (metric)
	{
	case DISTANCE_L2:
	{
		/* Elkan needs true L2 (not squared) for triangle inequality */
		float d2 = mkt_l2_distance_squared(a->x, b->x, (Dimension)dim);
		return (double)sqrtf(d2);
	}
	case DISTANCE_INNER_PRODUCT:
	case DISTANCE_COSINE:
	{
		/* Angular distance: acos(dot / (norm_a * norm_b)) / pi */
		float dot	 = mkt_dot_product(a->x, b->x, (Dimension)dim);
		float norm_a = mkt_l2_norm(a->x, (Dimension)dim);
		float norm_b = mkt_l2_norm(b->x, (Dimension)dim);
		float denom	 = norm_a * norm_b;
		if (denom < 1e-30f)
			return 0.0;
		double cos_sim = (double)dot / (double)denom;
		if (cos_sim > 1.0)
			cos_sim = 1.0;
		if (cos_sim < -1.0)
			cos_sim = -1.0;
		return acos(cos_sim) / M_PI;
	}
	}
	return 0.0;
}

/* FunctionCall2Coll: compute distance, store result, return dummy Datum */
#define FunctionCall2Coll(procinfo, collation, d1, d2) \
	((void)(procinfo),                                 \
	 (void)(collation),                                \
	 pgv_distance_result = pgv_compute_distance(       \
			 (const Vector *)DatumGetPointer(d1),      \
			 (const Vector *)DatumGetPointer(d2),      \
			 pgv_active_metric),                       \
	 (Datum)0)

/* FunctionCall1Coll: norm check, store result, return dummy Datum */
#define FunctionCall1Coll(procinfo, collation, d1)                   \
	((void)(procinfo),                                               \
	 (void)(collation),                                              \
	 pgv_distance_result = (double)mkt_l2_norm(                      \
			 ((const Vector *)DatumGetPointer(d1))->x,               \
			 (Dimension)((const Vector *)DatumGetPointer(d1))->dim), \
	 (Datum)0)

/* Support function proc numbers from ivfflat.h */
#define IVFFLAT_DISTANCE_PROC		 1
#define IVFFLAT_NORM_PROC			 2
#define IVFFLAT_KMEANS_DISTANCE_PROC 3
#define IVFFLAT_KMEANS_NORM_PROC	 4

/* index_getprocinfo: return pointer to the dummy FmgrInfo */
#define index_getprocinfo(idx, attno, procnum) (&(idx)->dist_procinfo)

/* IvfflatOptionalProcInfo: return norm proc only for IP/cosine */
static inline FmgrInfo *
IvfflatOptionalProcInfo(Relation index, uint16 procnum)
{
	if (procnum == IVFFLAT_KMEANS_NORM_PROC || procnum == IVFFLAT_NORM_PROC)
	{
		if (pgv_active_metric == DISTANCE_INNER_PRODUCT ||
			pgv_active_metric == DISTANCE_COSINE)
			return &index->norm_procinfo;
	}
	return NULL;
}

/* rd_indcollation is a field in PgvRelationData (initialized to {0}) */

/* ----------------------------------------------------------------
 * Xoshiro256** PRNG (identical to meerkat's kmeans.c)
 * ---------------------------------------------------------------- */
typedef struct
{
	uint64_t s[4];
} PgvXoshiro256State;

extern _Thread_local PgvXoshiro256State pgv_rng;

static inline uint64_t
pgv_xo_rotl(uint64_t x, int k)
{
	return (x << k) | (x >> (64 - k));
}

static inline uint64_t
pgv_xo_next(PgvXoshiro256State *state)
{
	uint64_t *s		 = state->s;
	uint64_t  result = pgv_xo_rotl(s[1] * 5, 7) * 9;
	uint64_t  t		 = s[1] << 17;

	s[2] ^= s[0];
	s[3] ^= s[1];
	s[1] ^= s[2];
	s[0] ^= s[3];
	s[2] ^= t;
	s[3] = pgv_xo_rotl(s[3], 45);

	return result;
}

static inline void
pgv_xo_seed(PgvXoshiro256State *state, uint64_t seed)
{
	for (int i = 0; i < 4; i++)
	{
		seed += 0x9e3779b97f4a7c15ULL;
		uint64_t z	= seed;
		z			= (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
		z			= (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
		state->s[i] = z ^ (z >> 31);
	}
}

static inline double
pgv_xo_uniform(PgvXoshiro256State *state)
{
	uint64_t x = pgv_xo_next(state) >> 11;
	return (double)x / (double)(1ULL << 53);
}

/* pgvector random macros */
#define RandomDouble() pgv_xo_uniform(&pgv_rng)
#define RandomInt()	   ((uint32_t)pgv_xo_next(&pgv_rng))
#define SeedRandom(s)  pgv_xo_seed(&pgv_rng, (uint64_t)(s))

/* ----------------------------------------------------------------
 * Memory stubs
 * ---------------------------------------------------------------- */
#define palloc(sz)				   mkt_alloc(sz)
#define palloc0(sz)				   mkt_alloc0(sz)
#define palloc_extended(sz, flags) mkt_alloc0(sz)
#define pfree(p)				   mkt_free(p)

/* Memory context stubs (Elkan uses a temp context; we just use
 * direct alloc/free since we clean up explicitly) */
#define MemoryContext				   void *
#define CurrentMemoryContext		   NULL
#define AllocSetContextCreate(p, n, s) NULL
#define ALLOCSET_DEFAULT_SIZES		   0
#define MemoryContextSwitchTo(ctx)	   NULL
#define MemoryContextReset(ctx)		   ((void)0)
#define MemoryContextDelete(ctx)	   ((void)0)
#define MemoryContextGetParent(ctx)	   NULL
#define MCXT_ALLOC_ZERO				   0x01
#define MCXT_ALLOC_HUGE				   0x02

/* ----------------------------------------------------------------
 * PostgreSQL error/interrupt stubs
 * ---------------------------------------------------------------- */
#define CHECK_FOR_INTERRUPTS() ((void)0)

#define elog(level, ...)                                  \
	do                                                    \
	{                                                     \
		fprintf(stderr, "pgvector kmeans: " __VA_ARGS__); \
		fprintf(stderr, "\n");                            \
		if ((level) >= 20) /* ERROR */                    \
			abort();                                      \
	} while (0)

/* ERROR level in PostgreSQL */
#define ERROR	20
#define WARNING 19
#define INFO	17

/* ereport: skip entirely (only used for maintenance_work_mem check) */
#define ereport(level, ...)			   ((void)0)
#define errcode(c)					   0
#define errmsg(...)					   0
#define ERRCODE_PROGRAM_LIMIT_EXCEEDED 0

/* maintenance_work_mem: set high so the check always passes */
#define maintenance_work_mem (INT_MAX / 1024)

/* IvfflatNormValue: normalize a vector to unit length in-place */
static inline Datum
IvfflatNormValue(const IvfflatTypeInfo *typeInfo, Oid collation, Datum value)
{
	(void)typeInfo;
	(void)collation;
	Vector *vec	 = (Vector *)DatumGetPointer(value);
	float	norm = mkt_l2_norm(vec->x, (Dimension)vec->dim);
	if (norm > 1e-30f)
	{
		float inv = 1.0f / norm;
		for (int i = 0; i < vec->dim; i++)
			vec->x[i] *= inv;
	}
	return value;
}

/* Max dimensions (from pgvector) */
#define IVFFLAT_MAX_DIM 16000

/* Thread-local max iterations (read by ElkanKmeans loop) */
extern _Thread_local uint32_t pgv_max_iterations;

/* ----------------------------------------------------------------
 * Public entry point
 * ---------------------------------------------------------------- */
KMeansResult *pgvector_kmeans(
		const float			*vectors,
		uint32_t			 nvecs,
		Dimension			 dim,
		uint32_t			 nlist,
		DistanceMetric		 metric,
		const KMeansOptions *options);

#endif /* MKT_KMEANS_PGVECTOR_H */
