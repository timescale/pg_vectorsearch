/*
 * test_centroid_search.c - Unit tests for centroid beam search
 *
 * Tests cover:
 * - Batch SoA distance matches per-entry distance
 * - Single-level beam search (root only)
 * - Two-level beam search (root + leaf)
 * - Beam search finds correct nearest clusters
 */

#include <math.h>
#include <string.h>

#include "core/memory.h"
#include "core/pg_compat.h"
#include "index/centroid_search.h"
#include "mkt_test.h"
#include "quant/rabitq.h"

TEST_GROUP(CentroidSearch);
TEST_MEMCTX_FIXTURE();

/* ----------------------------------------------------------------
 * Standalone page accessor (pages stored in a flat array)
 * ---------------------------------------------------------------- */

static Page
standalone_page_read(void *ctx, BlockNumber blkno)
{
	char *pages = (char *)ctx;
	return pages + (size_t)blkno * BLCKSZ;
}

static void
standalone_page_release(void *ctx, BlockNumber blkno)
{
	(void)ctx;
	(void)blkno;
}

/* ----------------------------------------------------------------
 * Helpers
 * ---------------------------------------------------------------- */

static float *
make_test_vector(Dimension dim, int seed)
{
	float *data = mkt_alloc(dim * sizeof(float));
	for (Dimension i = 0; i < dim; i++)
		data[i] = (float)((i * 17 + seed) % 100 - 50) / 10.0f;
	return data;
}

static float
brute_force_l2(const float *a, const float *b, Dimension dim)
{
	float sum = 0.0f;
	for (Dimension i = 0; i < dim; i++)
	{
		float diff = a[i] - b[i];
		sum += diff * diff;
	}
	return sum;
}

/* ----------------------------------------------------------------
 * Batch SoA distance tests
 * ---------------------------------------------------------------- */

TEST(batch_soa_matches_single)
{
	Dimension dim	= 128;
	const int count = 8;

	RaBitQParams *params = mkt_rabitq_create(dim, 42);
	ASSERT_NOT_NULL(params, "params created");

	/* Zero centroid (global mean) */
	float *centroid = mkt_alloc(dim * sizeof(float));
	for (Dimension i = 0; i < dim; i++)
		centroid[i] = 0.0f;
	VectorRef cent_ref = {.data = centroid, .dim = dim};

	/* Encode vectors into SoA arrays */
	uint32_t packed_bytes = MKT_RABITQ_BYTES(dim);
	float	*f_add		  = mkt_alloc(count * sizeof(float));
	float	*f_rescale	  = mkt_alloc(count * sizeof(float));
	uint8_t *bits		  = mkt_alloc((size_t)count * packed_bytes);

	float	   **vecs	   = mkt_alloc(count * sizeof(void *));
	RaBitQData **encodings = mkt_alloc(count * sizeof(void *));

	for (int i = 0; i < count; i++)
	{
		vecs[i]			  = make_test_vector(dim, i * 7);
		VectorRef vec_ref = {.data = vecs[i], .dim = dim};
		encodings[i]	  = mkt_rabitq_encode(params, vec_ref, cent_ref);
		ASSERT_NOT_NULL(encodings[i], "encoding succeeded");

		f_add[i]	 = encodings[i]->f_add;
		f_rescale[i] = encodings[i]->f_rescale;
		memcpy(bits + (size_t)i * packed_bytes,
			   encodings[i]->bits,
			   packed_bytes);
	}

	/* Create query state */
	float			 *query		= make_test_vector(dim, 100);
	VectorRef		  query_ref = {.data = query, .dim = dim};
	RaBitQQueryState *qstate =
			mkt_rabitq_prepare_query(params, query_ref, cent_ref);
	ASSERT_NOT_NULL(qstate, "query state created");

	/* Compute batch distances */
	Distance *batch_dists = mkt_alloc(count * sizeof(Distance));
	mkt_rabitq_distance_batch_soa(
			qstate, f_add, f_rescale, bits, count, dim, batch_dists);

	/* Compare with single-entry distances */
	for (int i = 0; i < count; i++)
	{
		Distance single_dist = mkt_rabitq_distance(qstate, encodings[i], dim);

		char msg[128];
		snprintf(
				msg,
				sizeof(msg),
				"vec %d: batch=%.6f single=%.6f",
				i,
				batch_dists[i],
				single_dist);
		ASSERT_FLOAT_EQ(single_dist, batch_dists[i], 1e-6f, msg);
	}

	/* Cleanup */
	mkt_free(batch_dists);
	mkt_rabitq_free_query(qstate);
	for (int i = 0; i < count; i++)
		mkt_free(encodings[i]);
	mkt_free(encodings);
	mkt_free(vecs);
	mkt_free(bits);
	mkt_free(f_rescale);
	mkt_free(f_add);
	mkt_rabitq_destroy(params);
}

TEST(batch_soa_null_inputs)
{
	Dimension	  dim	   = 64;
	RaBitQParams *params   = mkt_rabitq_create(dim, 42);
	float		 *centroid = mkt_alloc(dim * sizeof(float));
	for (Dimension i = 0; i < dim; i++)
		centroid[i] = 0.0f;
	VectorRef cent_ref = {.data = centroid, .dim = dim};

	float			 *query		= make_test_vector(dim, 1);
	VectorRef		  query_ref = {.data = query, .dim = dim};
	RaBitQQueryState *qstate =
			mkt_rabitq_prepare_query(params, query_ref, cent_ref);

	float	 f_add[1]	  = {1.0f};
	float	 f_rescale[1] = {1.0f};
	uint8_t	 bits[16]	  = {0};
	Distance dists[1];

	/* Should not crash with null inputs */
	mkt_rabitq_distance_batch_soa(NULL, f_add, f_rescale, bits, 1, dim, dists);
	mkt_rabitq_distance_batch_soa(
			qstate, NULL, f_rescale, bits, 1, dim, dists);
	mkt_rabitq_distance_batch_soa(qstate, f_add, NULL, bits, 1, dim, dists);
	mkt_rabitq_distance_batch_soa(
			qstate, f_add, f_rescale, NULL, 1, dim, dists);
	mkt_rabitq_distance_batch_soa(
			qstate, f_add, f_rescale, bits, 0, dim, dists);
	mkt_rabitq_distance_batch_soa(
			qstate, f_add, f_rescale, bits, 1, dim, NULL);

	ASSERT_TRUE(1, "null inputs should not crash");

	mkt_rabitq_free_query(qstate);
	mkt_rabitq_destroy(params);
}

/* ----------------------------------------------------------------
 * Single-level beam search (root-only tree)
 * ---------------------------------------------------------------- */

TEST(beam_search_single_level)
{
	Dimension dim		 = 128;
	const int ncentroids = 16;

	RaBitQParams *params = mkt_rabitq_create(dim, 42);
	ASSERT_NOT_NULL(params, "params created");

	/* Zero centroid (global mean) */
	float *centroid = mkt_alloc(dim * sizeof(float));
	for (Dimension i = 0; i < dim; i++)
		centroid[i] = 0.0f;
	VectorRef cent_ref = {.data = centroid, .dim = dim};

	/* Allocate pages (just need 1 for root) */
	char *pages = mkt_alloc((size_t)2 * BLCKSZ);
	memset(pages, 0, (size_t)2 * BLCKSZ);
	Page root_page = pages + 0 * BLCKSZ;

	mkt_centroid_page_init(root_page, 0, 0);

	/* Generate centroid vectors and add to page */
	float **medoid_vecs = mkt_alloc(ncentroids * sizeof(void *));
	for (int i = 0; i < ncentroids; i++)
	{
		medoid_vecs[i]		= make_test_vector(dim, i * 31);
		VectorRef	vec_ref = {.data = medoid_vecs[i], .dim = dim};
		RaBitQData *enc		= mkt_rabitq_encode(params, vec_ref, cent_ref);
		ASSERT_NOT_NULL(enc, "encoding succeeded");

		ItemPointerData tid;
		ItemPointerSet(&tid, 0, (OffsetNumber)i);

		/* Leaf centroids: child_blkno would be posting list head */
		bool added = mkt_centroid_page_add(
				root_page,
				dim,
				100 + i, /* fake posting list blkno */
				0,
				MKT_CENTROID_FLAG_LEAF,
				&tid,
				enc);
		ASSERT_TRUE(added, "entry added");
		mkt_free(enc);
	}

	/* Create query */
	float			 *query		= make_test_vector(dim, 999);
	VectorRef		  query_ref = {.data = query, .dim = dim};
	RaBitQQueryState *qstate =
			mkt_rabitq_prepare_query(params, query_ref, cent_ref);
	ASSERT_NOT_NULL(qstate, "query state created");

	/* Set up accessor */
	MktPageAccessor accessor = {
			.read	 = standalone_page_read,
			.release = standalone_page_release,
			.ctx	 = pages,
	};

	/* Search */
	MktCentroidSearchState search_state = {
			.qstate		= qstate,
			.accessor	= &accessor,
			.beam_width = 4,
			.nprobe		= 4,
			.dim		= dim,
	};

	MktCentroidResult results[4];
	uint32_t nresults = mkt_centroid_beam_search(&search_state, 0, 1, results);

	ASSERT_EQ(4, nresults, "should return nprobe results");

	/* Results should be sorted by distance (ascending) */
	for (uint32_t i = 1; i < nresults; i++)
	{
		char msg[128];
		snprintf(
				msg,
				sizeof(msg),
				"result %u (%.4f) >= result %u (%.4f)",
				i - 1,
				results[i - 1].distance,
				i,
				results[i].distance);
		ASSERT_TRUE(results[i - 1].distance <= results[i].distance, msg);
	}

	/* Verify posting_head values are valid */
	for (uint32_t i = 0; i < nresults; i++)
	{
		ASSERT_TRUE(
				results[i].posting_head >= 100 &&
						results[i].posting_head < 100 + ncentroids,
				"posting_head should be in range");
	}

	/* Brute force: find the actual nearest centroid */
	int	  nearest_idx  = 0;
	float nearest_dist = brute_force_l2(query, medoid_vecs[0], dim);
	for (int i = 1; i < ncentroids; i++)
	{
		float d = brute_force_l2(query, medoid_vecs[i], dim);
		if (d < nearest_dist)
		{
			nearest_dist = d;
			nearest_idx	 = i;
		}
	}

	/*
	 * The top-1 result from beam search should match the brute-force
	 * nearest with high probability. RaBitQ estimates can differ from
	 * true distances, so we check the true nearest is in top-4.
	 */
	bool found_nearest = false;
	for (uint32_t i = 0; i < nresults; i++)
	{
		if (results[i].posting_head == (BlockNumber)(100 + nearest_idx))
		{
			found_nearest = true;
			break;
		}
	}

	char msg[128];
	snprintf(
			msg,
			sizeof(msg),
			"brute-force nearest (idx %d, dist %.4f) "
			"should be in top-%u results",
			nearest_idx,
			nearest_dist,
			nresults);
	ASSERT_TRUE(found_nearest, msg);

	/* Cleanup */
	mkt_rabitq_free_query(qstate);
	for (int i = 0; i < ncentroids; i++)
		mkt_free(medoid_vecs[i]);
	mkt_free(medoid_vecs);
	mkt_free(pages);
	mkt_rabitq_destroy(params);
}

/* ----------------------------------------------------------------
 * Two-level beam search
 * ---------------------------------------------------------------- */

TEST(beam_search_two_levels)
{
	Dimension dim		   = 64;
	const int level0_count = 4; /* 4 root centroids */
	const int level1_count = 8; /* 8 children per root centroid */

	RaBitQParams *params = mkt_rabitq_create(dim, 42);
	ASSERT_NOT_NULL(params, "params created");

	float *centroid = mkt_alloc(dim * sizeof(float));
	for (Dimension i = 0; i < dim; i++)
		centroid[i] = 0.0f;
	VectorRef cent_ref = {.data = centroid, .dim = dim};

	/*
	 * Page layout:
	 *   Page 0: root (level 0) - 4 centroids
	 *   Pages 1-4: leaf (level 1) - 8 centroids each
	 */
	const int total_pages = 1 + level0_count;
	char	 *pages		  = mkt_alloc((size_t)total_pages * BLCKSZ);
	memset(pages, 0, (size_t)total_pages * BLCKSZ);

	/* Initialize root page */
	Page root_page = pages + 0 * BLCKSZ;
	mkt_centroid_page_init(root_page, 0, 0);

	/* Initialize leaf pages and populate */
	float **all_leaf_vecs = mkt_alloc(
			(size_t)level0_count * level1_count * sizeof(void *));

	for (int r = 0; r < level0_count; r++)
	{
		/* Leaf page for this root centroid */
		BlockNumber leaf_blkno = (BlockNumber)(1 + r);
		Page		leaf_page  = pages + (size_t)leaf_blkno * BLCKSZ;
		mkt_centroid_page_init(leaf_page, 1, r * level1_count);

		for (int c = 0; c < level1_count; c++)
		{
			int vec_idx = r * level1_count + c;
			/* Spread leaf vectors to make them distinguishable */
			all_leaf_vecs[vec_idx] = make_test_vector(dim, r * 1000 + c * 37);

			VectorRef	vec_ref = {.data = all_leaf_vecs[vec_idx], .dim = dim};
			RaBitQData *enc		= mkt_rabitq_encode(params, vec_ref, cent_ref);
			ASSERT_NOT_NULL(enc, "leaf encoding succeeded");

			ItemPointerData tid;
			ItemPointerSet(&tid, 0, (OffsetNumber)vec_idx);

			/* Leaf children point to posting lists (fake blknos) */
			bool added = mkt_centroid_page_add(
					leaf_page,
					dim,
					200 + vec_idx,
					0,
					MKT_CENTROID_FLAG_LEAF,
					&tid,
					enc);
			ASSERT_TRUE(added, "leaf entry added");
			mkt_free(enc);
		}

		/* Add root centroid pointing to this leaf page.
		 * Use the first leaf vector as the root medoid. */
		float	   *root_vec = all_leaf_vecs[r * level1_count];
		VectorRef	root_ref = {.data = root_vec, .dim = dim};
		RaBitQData *root_enc = mkt_rabitq_encode(params, root_ref, cent_ref);
		ASSERT_NOT_NULL(root_enc, "root encoding succeeded");

		ItemPointerData root_tid;
		ItemPointerSet(&root_tid, 0, (OffsetNumber)(r * level1_count));

		bool added = mkt_centroid_page_add(
				root_page,
				dim,
				leaf_blkno,
				level1_count,
				0,
				&root_tid,
				root_enc);
		ASSERT_TRUE(added, "root entry added");
		mkt_free(root_enc);
	}

	/* Create query */
	float			 *query		= make_test_vector(dim, 5555);
	VectorRef		  query_ref = {.data = query, .dim = dim};
	RaBitQQueryState *qstate =
			mkt_rabitq_prepare_query(params, query_ref, cent_ref);
	ASSERT_NOT_NULL(qstate, "query state created");

	/* Set up accessor */
	MktPageAccessor accessor = {
			.read	 = standalone_page_read,
			.release = standalone_page_release,
			.ctx	 = pages,
	};

	MktCentroidSearchState search_state = {
			.qstate		= qstate,
			.accessor	= &accessor,
			.beam_width = 2,
			.nprobe		= 4,
			.dim		= dim,
	};

	MktCentroidResult results[4];
	uint32_t nresults = mkt_centroid_beam_search(&search_state, 0, 2, results);

	ASSERT_TRUE(nresults > 0, "should return at least one result");
	ASSERT_TRUE(nresults <= 4, "should return at most nprobe results");

	/* Results should be sorted by distance */
	for (uint32_t i = 1; i < nresults; i++)
	{
		char msg[128];
		snprintf(
				msg,
				sizeof(msg),
				"result %u dist (%.4f) >= result %u dist (%.4f)",
				i - 1,
				results[i - 1].distance,
				i,
				results[i].distance);
		ASSERT_TRUE(results[i - 1].distance <= results[i].distance, msg);
	}

	/* Brute force: find actual nearest leaf */
	int	  total_leaves = level0_count * level1_count;
	int	  nearest_idx  = 0;
	float nearest_dist = brute_force_l2(query, all_leaf_vecs[0], dim);
	for (int i = 1; i < total_leaves; i++)
	{
		float d = brute_force_l2(query, all_leaf_vecs[i], dim);
		if (d < nearest_dist)
		{
			nearest_dist = d;
			nearest_idx	 = i;
		}
	}

	/* Check the brute-force nearest is in the results */
	bool found = false;
	for (uint32_t i = 0; i < nresults; i++)
	{
		if (results[i].posting_head == (BlockNumber)(200 + nearest_idx))
		{
			found = true;
			break;
		}
	}

	char msg[128];
	snprintf(
			msg,
			sizeof(msg),
			"brute-force nearest leaf (idx %d, dist %.4f) "
			"should be in results",
			nearest_idx,
			nearest_dist);
	ASSERT_TRUE(found, msg);

	/* Cleanup */
	mkt_rabitq_free_query(qstate);
	for (int i = 0; i < total_leaves; i++)
		mkt_free(all_leaf_vecs[i]);
	mkt_free(all_leaf_vecs);
	mkt_free(pages);
	mkt_rabitq_destroy(params);
}

/* ----------------------------------------------------------------
 * Edge cases
 * ---------------------------------------------------------------- */

TEST(beam_search_null_state)
{
	MktCentroidResult results[1];
	uint32_t		  n = mkt_centroid_beam_search(NULL, 0, 1, results);
	ASSERT_EQ(0, n, "null state should return 0");
}

TEST(beam_search_null_results)
{
	MktPageAccessor accessor = {
			.read	 = standalone_page_read,
			.release = standalone_page_release,
			.ctx	 = NULL,
	};
	RaBitQQueryState	   dummy_qstate = {0};
	MktCentroidSearchState state		= {
				   .qstate	   = &dummy_qstate,
				   .accessor   = &accessor,
				   .beam_width = 4,
				   .nprobe	   = 4,
				   .dim		   = 64,
	   };
	uint32_t n = mkt_centroid_beam_search(&state, 0, 1, NULL);
	ASSERT_EQ(0, n, "null results should return 0");
}

TEST(beam_search_zero_levels)
{
	MktPageAccessor accessor = {
			.read	 = standalone_page_read,
			.release = standalone_page_release,
			.ctx	 = NULL,
	};
	RaBitQQueryState	   dummy_qstate = {0};
	MktCentroidSearchState state		= {
				   .qstate	   = &dummy_qstate,
				   .accessor   = &accessor,
				   .beam_width = 4,
				   .nprobe	   = 4,
				   .dim		   = 64,
	   };
	MktCentroidResult results[1];
	uint32_t		  n = mkt_centroid_beam_search(&state, 0, 0, results);
	ASSERT_EQ(0, n, "zero levels should return 0");
}

TEST(beam_search_invalid_blkno)
{
	MktPageAccessor accessor = {
			.read	 = standalone_page_read,
			.release = standalone_page_release,
			.ctx	 = NULL,
	};
	RaBitQQueryState	   dummy_qstate = {0};
	MktCentroidSearchState state		= {
				   .qstate	   = &dummy_qstate,
				   .accessor   = &accessor,
				   .beam_width = 4,
				   .nprobe	   = 4,
				   .dim		   = 64,
	   };
	MktCentroidResult results[1];
	uint32_t		  n =
			mkt_centroid_beam_search(&state, InvalidBlockNumber, 1, results);
	ASSERT_EQ(0, n, "invalid blkno should return 0");
}
