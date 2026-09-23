/*
 * test_parallel_scan.c - Work-stealing vector scan (standalone) tests
 *
 * Verifies that concurrent participants cover every vector exactly once, that
 * the synthesized TID round-trips to the vector index, and that the callback
 * receives the correct vector pointer.
 */

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>

#include "standalone/parallel_scan.h"
#include "vs_test.h"

TEST_GROUP(ParallelScan);

#define PS_NVECS   10000
#define PS_DIM	   4
#define PS_THREADS 4

typedef struct
{
	const float		  *base;
	Dimension		   dim;
	_Atomic(uint32_t) *counts; /* visits per index */
	_Atomic(uint32_t) *tid_errors;
	_Atomic(uint32_t) *ptr_errors;
} PSCovArg;

static void
ps_cov_cb(void *state, ItemPointerData tid, const float *vec)
{
	PSCovArg *a		= (PSCovArg *)state;
	uint32_t  index = ItemPointerGetBlockNumber(&tid);

	/* TID must decode back to a valid index, and offset must be the marker. */
	if (index >= PS_NVECS || ItemPointerGetOffsetNumber(&tid) != 1)
	{
		atomic_fetch_add(a->tid_errors, 1);
		return;
	}

	/* Callback must receive the vector at that index. */
	if (vec != a->base + (size_t)index * a->dim)
		atomic_fetch_add(a->ptr_errors, 1);

	atomic_fetch_add(&a->counts[index], 1);
}

typedef struct
{
	PrismParallelScan *ps;
	PSCovArg		  *arg;
} PSThreadArg;

static void *
ps_thread(void *p)
{
	PSThreadArg *t = (PSThreadArg *)p;
	prism_parallel_scan_run(t->ps, ps_cov_cb, t->arg);
	return NULL;
}

TEST(worksteal_covers_every_vector_once)
{
	float *vectors = malloc((size_t)PS_NVECS * PS_DIM * sizeof(float));
	for (uint32_t i = 0; i < PS_NVECS; i++)
		for (int d = 0; d < PS_DIM; d++)
			vectors[(size_t)i * PS_DIM + d] = (float)i;

	_Atomic(uint32_t) *counts = calloc(PS_NVECS, sizeof(_Atomic(uint32_t)));
	_Atomic(uint32_t)  tid_errors = 0;
	_Atomic(uint32_t)  ptr_errors = 0;

	PrismParallelScan ps;
	prism_parallel_scan_init(&ps, vectors, PS_NVECS, PS_DIM);

	PSCovArg arg = {
			.base		= vectors,
			.dim		= PS_DIM,
			.counts		= counts,
			.tid_errors = &tid_errors,
			.ptr_errors = &ptr_errors,
	};
	PSThreadArg ta = {.ps = &ps, .arg = &arg};

	pthread_t th[PS_THREADS];
	for (int i = 0; i < PS_THREADS; i++)
		pthread_create(&th[i], NULL, ps_thread, &ta);
	for (int i = 0; i < PS_THREADS; i++)
		pthread_join(th[i], NULL);

	ASSERT_EQ(
			0,
			(int)atomic_load(&tid_errors),
			"every TID decodes to its index");
	ASSERT_EQ(
			0,
			(int)atomic_load(&ptr_errors),
			"callback gets the right vector");

	uint32_t missing = 0;
	uint32_t dup	 = 0;
	for (uint32_t i = 0; i < PS_NVECS; i++)
	{
		uint32_t c = atomic_load(&counts[i]);
		if (c == 0)
			missing++;
		else if (c > 1)
			dup++;
	}
	ASSERT_EQ(0, (int)missing, "no vector is skipped");
	ASSERT_EQ(0, (int)dup, "no vector is visited twice");

	free(vectors);
	free(counts);
}

/* A single participant must also cover everything (degenerate work-stealing).
 */
TEST(single_thread_covers_all)
{
	uint32_t nvecs	 = 777; /* not a multiple of the chunk size */
	float	*vectors = calloc((size_t)nvecs * PS_DIM, sizeof(float));
	uint32_t visited = 0;

	PrismParallelScan ps;
	prism_parallel_scan_init(&ps, vectors, nvecs, PS_DIM);

	/* Reuse the coverage callback machinery via a local counter. */
	_Atomic(uint32_t) *counts	  = calloc(nvecs, sizeof(_Atomic(uint32_t)));
	_Atomic(uint32_t)  tid_errors = 0;
	_Atomic(uint32_t)  ptr_errors = 0;
	PSCovArg		   arg		  = {
							 .base		 = vectors,
							 .dim		 = PS_DIM,
							 .counts	 = counts,
							 .tid_errors = &tid_errors,
							 .ptr_errors = &ptr_errors,
	 };

	/* counts is sized nvecs, but ps_cov_cb bounds-checks against PS_NVECS;
	 * 777 < PS_NVECS so the index check passes. */
	prism_parallel_scan_run(&ps, ps_cov_cb, &arg);

	for (uint32_t i = 0; i < nvecs; i++)
		if (atomic_load(&counts[i]) == 1)
			visited++;

	ASSERT_EQ((int)nvecs, (int)visited, "single participant covers all");

	free(vectors);
	free(counts);
}
