/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * posting_fixtures.h - Shared fixtures for the posting-list unit tests
 *
 * Building a posting list and scoring a query against it takes the same
 * handful of steps in every test file, so they live here. The array-backed
 * storage these build on is in page_storage.h.
 */

#ifndef VS_TEST_POSTING_FIXTURES_H
#define VS_TEST_POSTING_FIXTURES_H

#include "core/memory.h"
#include "index/posting_build.h"
#include "index/posting_page.h"
#include "index/posting_scan.h"
#include "index/storage.h"
#include "page_storage.h"
#include "quant/rabitq.h"
#include "standalone/pg_compat.h"

/* Create a TID from a vector_id (standalone encoding). */
static inline ItemPointerData
vid_to_tid(uint32_t vid)
{
	ItemPointerData tid;
	prism_posting_set_vector_id(&tid, vid);
	return tid;
}

/* Deterministic vectors, spread over a range wide enough that scores differ.
 */
static inline float *
make_test_vectors(uint32_t nvecs, Dimension dim)
{
	float *vecs = vs_alloc((size_t)nvecs * dim * sizeof(float));
	for (uint32_t i = 0; i < nvecs; i++)
		for (Dimension d = 0; d < dim; d++)
			vecs[(size_t)i * dim + d] = (float)((i * 13 + d * 7) % 100 - 50) /
										10.0f;
	return vecs;
}

/*
 * A query state against `centroid`, ready to score. The rotated query and its
 * bit codes are allocated here and left to the test's memory context, so
 * callers need no scratch buffers of their own.
 */
static inline void
setup_query_state(
		RaBitQQueryState *qstate,
		RaBitQParams	 *params,
		const float		 *centroid,
		Dimension		  dim)
{
	float *query	= vs_alloc(dim * sizeof(float));
	float *pt_query = vs_alloc(dim * sizeof(float));
	for (Dimension d = 0; d < dim; d++)
		query[d] = (float)(d % 10) / 5.0f;
	qstate->transformed = vs_alloc_aligned(dim * sizeof(float), 64);
	qstate->query_bits	= vs_alloc_aligned(VS_RABITQ_BYTES(dim), 64);
	vs_rabitq_init_query_constants(qstate, dim);
	vs_rabitq_rotate(params, query, pt_query);
	vs_rabitq_init_query_state(
			qstate, pt_query, centroid, dim, VS_DISTANCE_MODE_ASYMMETRIC);
}

/* Build a single-cluster posting list of nbuilt vectors (AoS or fastscan). */
static inline BlockNumber
build_cluster(
		TestPageStorage *st,
		RaBitQParams	*params,
		Dimension		 dim,
		const float		*centroid,
		const float		*vecs,
		uint32_t		 nbuilt,
		bool			 fastscan)
{
	PrismPostingBuilder builder;
	if (fastscan)
		prism_posting_builder_init_fastscan(
				&builder, &st->base, params, dim, 0, centroid, centroid);
	else
		prism_posting_builder_init(
				&builder, &st->base, params, dim, 0, centroid, centroid);
	for (uint32_t i = 0; i < nbuilt; i++)
		prism_posting_builder_add(
				&builder, vid_to_tid(i), vecs + (size_t)i * dim);
	BlockNumber head = prism_posting_builder_finish(&builder);
	prism_posting_builder_cleanup(&builder);
	return head;
}

#endif /* VS_TEST_POSTING_FIXTURES_H */
