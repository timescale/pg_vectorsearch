/*
 * test_centroid_search.c - Unit tests for centroid beam search
 *
 * Tests cover:
 * - Two-level beam search (root + leaf)
 * - Beam search finds correct nearest clusters
 * - Edge cases (null inputs, zero levels, invalid block number)
 */

#include <string.h>

#include "core/memory.h"
#include "core/pg_compat.h"
#include "index/centroid_search.h"
#include "mkt_test.h"
#include "quant/rabitq.h"

TEST_GROUP(CentroidSearch);
TEST_MEMCTX_FIXTURE();

/* ----------------------------------------------------------------
 * Test storage callbacks (pages stored in a flat array)
 * ---------------------------------------------------------------- */

static Page
test_read_page(void *ctx, BlockNumber blkno)
{
	char *pages = (char *)ctx;
	return pages + (size_t)blkno * BLCKSZ;
}

static void
test_release_page(void *ctx, BlockNumber blkno)
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
	mkt_centroid_page_init(root_page, 0);

	/* Initialize leaf pages and populate */
	float **all_leaf_vecs = mkt_alloc(
			(size_t)level0_count * level1_count * sizeof(void *));

	for (int r = 0; r < level0_count; r++)
	{
		/* Leaf page for this root centroid */
		BlockNumber leaf_blkno = (BlockNumber)(1 + r);
		Page		leaf_page  = pages + (size_t)leaf_blkno * BLCKSZ;
		mkt_centroid_page_init(leaf_page, 1);

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

	/* Set up storage */
	MktStorage storage = {
			.read_page	  = test_read_page,
			.release_page = test_release_page,
			.ctx		  = pages,
	};

	MktCentroidSearchState search_state = {
			.qstate		= qstate,
			.storage	= &storage,
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
	MktStorage storage = {
			.read_page	  = test_read_page,
			.release_page = test_release_page,
			.ctx		  = NULL,
	};
	RaBitQQueryState	   dummy_qstate = {0};
	MktCentroidSearchState state		= {
				   .qstate	   = &dummy_qstate,
				   .storage	   = &storage,
				   .beam_width = 4,
				   .nprobe	   = 4,
				   .dim		   = 64,
	   };
	uint32_t n = mkt_centroid_beam_search(&state, 0, 1, NULL);
	ASSERT_EQ(0, n, "null results should return 0");
}

TEST(beam_search_zero_levels)
{
	MktStorage storage = {
			.read_page	  = test_read_page,
			.release_page = test_release_page,
			.ctx		  = NULL,
	};
	RaBitQQueryState	   dummy_qstate = {0};
	MktCentroidSearchState state		= {
				   .qstate	   = &dummy_qstate,
				   .storage	   = &storage,
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
	MktStorage storage = {
			.read_page	  = test_read_page,
			.release_page = test_release_page,
			.ctx		  = NULL,
	};
	RaBitQQueryState	   dummy_qstate = {0};
	MktCentroidSearchState state		= {
				   .qstate	   = &dummy_qstate,
				   .storage	   = &storage,
				   .beam_width = 4,
				   .nprobe	   = 4,
				   .dim		   = 64,
	   };
	MktCentroidResult results[1];
	uint32_t		  n =
			mkt_centroid_beam_search(&state, InvalidBlockNumber, 1, results);
	ASSERT_EQ(0, n, "invalid blkno should return 0");
}
