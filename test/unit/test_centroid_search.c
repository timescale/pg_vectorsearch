/*
 * test_centroid_search.c - Unit tests for centroid beam search
 *
 * Tests cover:
 * - Two-level beam search (root + leaf) for RaBitQ, float, half
 * - Beam search finds correct nearest clusters
 * - Edge cases (null inputs, zero levels, invalid block number)
 */

#include <float.h>
#include <string.h>

#include "algo/vecops.h"
#include "core/memory.h"
#include "core/pg_compat.h"
#include "index/centroid_search.h"
#include "mkt_halfvec.h"
#include "mkt_test.h"
#include "quant/rabitq.h"

TEST_GROUP(CentroidSearch);
TEST_MEMCTX_FIXTURE();

/* ----------------------------------------------------------------
 * Test storage: MktStorage base + flat page array
 * ---------------------------------------------------------------- */

typedef struct TestStorage
{
	MktStorage	  base; /* must be first */
	char		 *pages;
	const float **vecs; /* medoid vectors indexed by TID offset */
	Dimension	  dim;
} TestStorage;

static Page
test_read_page(MktStorage *self, BlockNumber blkno)
{
	TestStorage *ts = (TestStorage *)self;
	return ts->pages + (size_t)blkno * BLCKSZ;
}

static void
test_release_page(MktStorage *self, BlockNumber blkno)
{
	(void)self;
	(void)blkno;
}

/*
 * test_rerank - Rerank candidates using stored medoid vectors.
 *
 * Looks up each candidate's vector via TID offset, computes exact
 * L2 distance, then uses lower-bound ordering with threshold
 * pruning to return the best-keep results.
 */
static uint32_t
test_rerank(
		MktStorage			  *self,
		Datum				   query,
		Dimension			   dim,
		const ItemPointerData *tids,
		const Distance		  *distances,
		const Distance		  *errors,
		uint32_t			   count,
		uint32_t			   keep,
		uint32_t			  *out_indices,
		Distance			  *out_distances)
{
	TestStorage *ts		 = (TestStorage *)self;
	const float *query_f = (const float *)DatumGetPointer(query);

	if (ts->vecs == NULL || query_f == NULL || count == 0)
		return 0;

	/*
	 * Build sortable array of (lower_bound, index) pairs.
	 * Exact candidates (error==0) keep their distance as-is.
	 */
	typedef struct
	{
		Distance lower_bound;
		uint32_t idx;
	} LBEntry;

	LBEntry *lb = mkt_alloc(count * sizeof(LBEntry));
	for (uint32_t i = 0; i < count; i++)
	{
		lb[i].lower_bound = distances[i] - errors[i];
		lb[i].idx		  = i;
	}

	/* Sort by lower_bound ascending */
	for (uint32_t i = 1; i < count; i++)
	{
		LBEntry	 tmp = lb[i];
		uint32_t j	 = i;
		while (j > 0 && lb[j - 1].lower_bound > tmp.lower_bound)
		{
			lb[j] = lb[j - 1];
			j--;
		}
		lb[j] = tmp;
	}

	/* Iterate in lower_bound order with threshold pruning */
	Distance  threshold = FLT_MAX;
	uint32_t  nresults	= 0;
	Distance *exact		= mkt_alloc(count * sizeof(Distance));

	for (uint32_t i = 0; i < count; i++)
	{
		uint32_t idx = lb[i].idx;

		/* Prune: if lower_bound >= threshold and have k results */
		if (nresults >= keep && lb[i].lower_bound >= threshold)
			break;

		Distance d;
		if (errors[idx] == 0.0f)
		{
			d = distances[idx]; /* already exact */
		}
		else
		{
			OffsetNumber off = ItemPointerGetOffsetNumber(&tids[idx]);
			const float *vec = ts->vecs[off];
			if (vec == NULL)
				continue;
			d = mkt_l2_distance_squared(query_f, vec, dim);
		}
		exact[idx] = d;

		/* Insert into result set maintaining sorted order */
		uint32_t pos = nresults;
		while (pos > 0 && exact[out_indices[pos - 1]] > d)
		{
			if (pos < keep)
			{
				out_indices[pos]   = out_indices[pos - 1];
				out_distances[pos] = out_distances[pos - 1];
			}
			pos--;
		}
		if (pos < keep)
		{
			out_indices[pos]   = idx;
			out_distances[pos] = d;
			if (nresults < keep)
				nresults++;

			/* Update threshold to worst result */
			threshold = out_distances[nresults - 1];
		}
	}

	mkt_free(exact);
	mkt_free(lb);
	return nresults;
}

static const MktStorageOps test_storage_ops = {
		.read_page	  = test_read_page,
		.release_page = test_release_page,
		.write_page	  = NULL,
		.new_page	  = NULL,
		.commit_page  = NULL,
		.rerank		  = test_rerank,
};

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
			/* Spread leaf vectors to make them distinguishable.
			 * Use r*997 (coprime with 100) so root medoids differ
			 * given make_test_vector uses % 100 internally. */
			all_leaf_vecs[vec_idx] = make_test_vector(dim, r * 997 + c * 37);

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

	/* Set up storage with medoid vectors for reranking */
	TestStorage storage = {
			.base.ops = &test_storage_ops,
			.pages	  = pages,
			.vecs	  = (const float **)all_leaf_vecs,
			.dim	  = dim,
	};

	MktCentroidSearchState search_state = {
			.qstate		 = qstate,
			.query		 = query,
			.query_datum = PointerGetDatum(query),
			.storage	 = &storage.base,
			.beam_width	 = 2,
			.nprobe		 = 4,
			.dim		 = dim,
	};

	MktCentroidResult results[4];
	uint32_t		  nresults =
			mkt_centroid_beam_search(&search_state, 0, 2, results, NULL, NULL);

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
	uint32_t n = mkt_centroid_beam_search(NULL, 0, 1, results, NULL, NULL);
	ASSERT_EQ(0, n, "null state should return 0");
}

TEST(beam_search_null_results)
{
	TestStorage storage = {
			.base.ops = &test_storage_ops,
			.pages	  = NULL,
	};
	RaBitQQueryState	   dummy_qstate = {0};
	MktCentroidSearchState state		= {
				   .qstate	   = &dummy_qstate,
				   .storage	   = &storage.base,
				   .beam_width = 4,
				   .nprobe	   = 4,
				   .dim		   = 64,
	   };
	uint32_t n = mkt_centroid_beam_search(&state, 0, 1, NULL, NULL, NULL);
	ASSERT_EQ(0, n, "null results should return 0");
}

TEST(beam_search_zero_levels)
{
	TestStorage storage = {
			.base.ops = &test_storage_ops,
			.pages	  = NULL,
	};
	RaBitQQueryState	   dummy_qstate = {0};
	MktCentroidSearchState state		= {
				   .qstate	   = &dummy_qstate,
				   .storage	   = &storage.base,
				   .beam_width = 4,
				   .nprobe	   = 4,
				   .dim		   = 64,
	   };
	MktCentroidResult results[1];
	uint32_t n = mkt_centroid_beam_search(&state, 0, 0, results, NULL, NULL);
	ASSERT_EQ(0, n, "zero levels should return 0");
}

TEST(beam_search_invalid_blkno)
{
	TestStorage storage = {
			.base.ops = &test_storage_ops,
			.pages	  = NULL,
	};
	RaBitQQueryState	   dummy_qstate = {0};
	MktCentroidSearchState state		= {
				   .qstate	   = &dummy_qstate,
				   .storage	   = &storage.base,
				   .beam_width = 4,
				   .nprobe	   = 4,
				   .dim		   = 64,
	   };
	MktCentroidResult results[1];
	uint32_t		  n = mkt_centroid_beam_search(
			 &state, InvalidBlockNumber, 1, results, NULL, NULL);
	ASSERT_EQ(0, n, "invalid blkno should return 0");
}

/* ----------------------------------------------------------------
 * Float format beam search
 * ---------------------------------------------------------------- */

TEST(beam_search_float_two_levels)
{
	Dimension dim		   = 64;
	const int level0_count = 4;
	const int level1_count = 8;

	/*
	 * Page layout:
	 *   Page 0: root (level 0) - 4 centroids (float)
	 *   Pages 1-4: leaf (level 1) - 8 centroids each (float)
	 */
	const int total_pages = 1 + level0_count;
	char	 *pages		  = mkt_alloc((size_t)total_pages * BLCKSZ);
	memset(pages, 0, (size_t)total_pages * BLCKSZ);

	Page root_page = pages;
	mkt_centroid_page_init_fmt(root_page, 0, MKT_CENTROID_FMT_FLOAT);

	float **all_leaf_vecs = mkt_alloc(
			(size_t)level0_count * level1_count * sizeof(void *));

	for (int r = 0; r < level0_count; r++)
	{
		BlockNumber leaf_blkno = (BlockNumber)(1 + r);
		Page		leaf_page  = pages + (size_t)leaf_blkno * BLCKSZ;
		mkt_centroid_page_init_fmt(leaf_page, 1, MKT_CENTROID_FMT_FLOAT);

		for (int c = 0; c < level1_count; c++)
		{
			int	   vec_idx		   = r * level1_count + c;
			float *vec			   = make_test_vector(dim, r * 1000 + c * 37);
			all_leaf_vecs[vec_idx] = vec;

			bool added = mkt_centroid_page_add_entry(
					leaf_page,
					dim,
					200 + vec_idx,
					0,
					MKT_CENTROID_FLAG_LEAF,
					NULL,
					vec);
			ASSERT_TRUE(added, "leaf float entry added");
		}

		/* Root centroid = first leaf vector */
		float *root_vec = all_leaf_vecs[r * level1_count];

		bool added = mkt_centroid_page_add_entry(
				root_page, dim, leaf_blkno, level1_count, 0, NULL, root_vec);
		ASSERT_TRUE(added, "root float entry added");
	}

	/* Create query */
	float *query = make_test_vector(dim, 5555);

	TestStorage storage = {
			.base.ops = &test_storage_ops,
			.pages	  = pages,
			.vecs	  = (const float **)all_leaf_vecs,
			.dim	  = dim,
	};

	MktCentroidSearchState search_state = {
			.qstate		 = NULL, /* not needed for float pages */
			.query		 = query,
			.query_datum = PointerGetDatum(query),
			.storage	 = &storage.base,
			.beam_width	 = 2,
			.nprobe		 = 4,
			.dim		 = dim,
	};

	MktCentroidResult results[4];
	uint32_t		  nresults =
			mkt_centroid_beam_search(&search_state, 0, 2, results, NULL, NULL);

	ASSERT_TRUE(nresults > 0, "float: should return results");
	ASSERT_TRUE(nresults <= 4, "float: at most nprobe results");

	/* Results should be sorted by distance */
	for (uint32_t i = 1; i < nresults; i++)
	{
		ASSERT_TRUE(
				results[i - 1].distance <= results[i].distance,
				"float: results sorted by distance");
	}

	/* Float pages have zero error */
	for (uint32_t i = 0; i < nresults; i++)
	{
		ASSERT_FLOAT_EQ(
				0.0f, results[i].error, 1e-6f, "float: error should be 0");
	}

	/* Brute force nearest */
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

	bool found = false;
	for (uint32_t i = 0; i < nresults; i++)
	{
		if (results[i].posting_head == (BlockNumber)(200 + nearest_idx))
		{
			found = true;
			break;
		}
	}
	ASSERT_TRUE(found, "float: brute-force nearest in results");

	/* Cleanup */
	for (int i = 0; i < total_leaves; i++)
		mkt_free(all_leaf_vecs[i]);
	mkt_free(all_leaf_vecs);
	mkt_free(pages);
}

/* ----------------------------------------------------------------
 * Half format beam search
 * ---------------------------------------------------------------- */

TEST(beam_search_half_two_levels)
{
	Dimension dim		   = 64;
	const int level0_count = 4;
	const int level1_count = 8;

	const int total_pages = 1 + level0_count;
	char	 *pages		  = mkt_alloc((size_t)total_pages * BLCKSZ);
	memset(pages, 0, (size_t)total_pages * BLCKSZ);

	Page root_page = pages;
	mkt_centroid_page_init_fmt(root_page, 0, MKT_CENTROID_FMT_HALF);

	/* Keep float versions for brute-force ground truth */
	float **all_leaf_f32 = mkt_alloc(
			(size_t)level0_count * level1_count * sizeof(void *));
	half **all_leaf_half = mkt_alloc(
			(size_t)level0_count * level1_count * sizeof(void *));

	for (int r = 0; r < level0_count; r++)
	{
		BlockNumber leaf_blkno = (BlockNumber)(1 + r);
		Page		leaf_page  = pages + (size_t)leaf_blkno * BLCKSZ;
		mkt_centroid_page_init_fmt(leaf_page, 1, MKT_CENTROID_FMT_HALF);

		for (int c = 0; c < level1_count; c++)
		{
			int	   vec_idx = r * level1_count + c;
			float *fvec	   = make_test_vector(dim, r * 1000 + c * 37);
			half  *hvec	   = mkt_alloc(dim * sizeof(half));
			for (Dimension d = 0; d < dim; d++)
				hvec[d] = mkt_float_to_half(fvec[d]);

			all_leaf_f32[vec_idx]  = fvec;
			all_leaf_half[vec_idx] = hvec;

			bool added = mkt_centroid_page_add_entry(
					leaf_page,
					dim,
					200 + vec_idx,
					0,
					MKT_CENTROID_FLAG_LEAF,
					NULL,
					hvec);
			ASSERT_TRUE(added, "leaf half entry added");
		}

		/* Root centroid = first leaf vector (half) */
		half *root_hvec = all_leaf_half[r * level1_count];

		bool added = mkt_centroid_page_add_entry(
				root_page, dim, leaf_blkno, level1_count, 0, NULL, root_hvec);
		ASSERT_TRUE(added, "root half entry added");
	}

	float *query = make_test_vector(dim, 5555);

	/* For half pages, reranking uses float vecs (same ground truth) */
	TestStorage storage = {
			.base.ops = &test_storage_ops,
			.pages	  = pages,
			.vecs	  = (const float **)all_leaf_f32,
			.dim	  = dim,
	};

	MktCentroidSearchState search_state = {
			.qstate		 = NULL,
			.query		 = query,
			.query_datum = PointerGetDatum(query),
			.storage	 = &storage.base,
			.beam_width	 = 2,
			.nprobe		 = 4,
			.dim		 = dim,
	};

	MktCentroidResult results[4];
	uint32_t		  nresults =
			mkt_centroid_beam_search(&search_state, 0, 2, results, NULL, NULL);

	ASSERT_TRUE(nresults > 0, "half: should return results");
	ASSERT_TRUE(nresults <= 4, "half: at most nprobe results");

	/* Results sorted */
	for (uint32_t i = 1; i < nresults; i++)
	{
		ASSERT_TRUE(
				results[i - 1].distance <= results[i].distance,
				"half: results sorted by distance");
	}

	/* Half pages have zero error */
	for (uint32_t i = 0; i < nresults; i++)
	{
		ASSERT_FLOAT_EQ(
				0.0f, results[i].error, 1e-6f, "half: error should be 0");
	}

	/* Brute force nearest (using float vectors for ground truth) */
	int	  total_leaves = level0_count * level1_count;
	int	  nearest_idx  = 0;
	float nearest_dist = brute_force_l2(query, all_leaf_f32[0], dim);
	for (int i = 1; i < total_leaves; i++)
	{
		float d = brute_force_l2(query, all_leaf_f32[i], dim);
		if (d < nearest_dist)
		{
			nearest_dist = d;
			nearest_idx	 = i;
		}
	}

	bool found = false;
	for (uint32_t i = 0; i < nresults; i++)
	{
		if (results[i].posting_head == (BlockNumber)(200 + nearest_idx))
		{
			found = true;
			break;
		}
	}
	ASSERT_TRUE(found, "half: brute-force nearest in results");

	/* Cleanup */
	for (int i = 0; i < total_leaves; i++)
	{
		mkt_free(all_leaf_f32[i]);
		mkt_free(all_leaf_half[i]);
	}
	mkt_free(all_leaf_f32);
	mkt_free(all_leaf_half);
	mkt_free(pages);
}
