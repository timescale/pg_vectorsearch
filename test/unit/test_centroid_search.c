/*
 * test_centroid_search.c - Unit tests for centroid beam search
 *
 * Tests cover:
 * - Two-level beam search (root + leaf) for RaBitQ, float, half
 * - Beam search finds correct nearest clusters
 * - Edge cases (null inputs, zero levels, invalid block number)
 */

#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "algo/vecops.h"
#include "core/memory.h"
#include "index/centroid_page.h"
#include "index/centroid_search.h"
#include "quant/rabitq.h"
#include "standalone/pg_compat.h"
#include "types/vec16.h"
#include "vs_test.h"

TEST_GROUP(CentroidSearch);
TEST_MEMCTX_FIXTURE();

/* ----------------------------------------------------------------
 * Test storage: VsStorage base + flat page array
 * ---------------------------------------------------------------- */

typedef struct TestStorage
{
	VsStorage	  base; /* must be first */
	char		 *pages;
	const float **vecs; /* medoid vectors indexed by TID offset */
	Dimension	  dim;
} TestStorage;

static Page
test_read_page(VsStorage *self, BlockNumber blkno)
{
	TestStorage *ts = (TestStorage *)self;
	return ts->pages + (size_t)blkno * BLCKSZ;
}

static void
test_release_page(VsStorage *self, BlockNumber blkno)
{
	(void)self;
	(void)blkno;
}

static const VsStorageOps test_storage_ops = {
		.read_page	  = test_read_page,
		.release_page = test_release_page,
		.write_page	  = NULL,
		.new_page	  = NULL,
		.commit_page  = NULL,
		.rerank		  = NULL,
};

/* ----------------------------------------------------------------
 * Helpers
 * ---------------------------------------------------------------- */

static float *
make_test_vector(Dimension dim, int seed)
{
	float *data = vs_alloc(dim * sizeof(float));
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

	RaBitQParams *params = vs_rabitq_create(dim, 42);
	ASSERT_NOT_NULL(params, "params created");

	float *centroid = vs_alloc(dim * sizeof(float));
	for (Dimension i = 0; i < dim; i++)
		centroid[i] = 0.0f;
	Vec32Ref cent_ref = {.data = centroid, .dim = dim};

	/*
	 * Page layout:
	 *   Page 0: root (level 0) - 4 centroids
	 *   Pages 1-4: leaf (level 1) - 8 centroids each
	 */
	const int total_pages = 1 + level0_count;
	char	 *pages		  = vs_alloc((size_t)total_pages * BLCKSZ);
	memset(pages, 0, (size_t)total_pages * BLCKSZ);

	/* Initialize root page */
	Page root_page = pages + 0 * BLCKSZ;
	prism_centroid_page_init(root_page, 0);

	/* Initialize leaf pages and populate */
	float **all_leaf_vecs = vs_alloc(
			(size_t)level0_count * level1_count * sizeof(void *));

	for (int r = 0; r < level0_count; r++)
	{
		/* Leaf page for this root centroid */
		BlockNumber leaf_blkno = (BlockNumber)(1 + r);
		Page		leaf_page  = pages + (size_t)leaf_blkno * BLCKSZ;
		prism_centroid_page_init(leaf_page, 1);

		for (int c = 0; c < level1_count; c++)
		{
			int vec_idx = r * level1_count + c;
			/* Spread leaf vectors to make them distinguishable.
			 * Use r*997 (coprime with 100) so root medoids differ
			 * given make_test_vector uses % 100 internally. */
			all_leaf_vecs[vec_idx] = make_test_vector(dim, r * 997 + c * 37);

			Vec32Ref	vec_ref = {.data = all_leaf_vecs[vec_idx], .dim = dim};
			RaBitQData *enc		= vs_rabitq_encode(params, vec_ref, cent_ref);
			ASSERT_NOT_NULL(enc, "leaf encoding succeeded");

			/* Leaf children point to posting lists (fake blknos) */
			bool added = prism_centroid_page_add(
					leaf_page,
					dim,
					200 + vec_idx,
					0,
					PRISM_CENTROID_FLAG_LEAF,
					enc);
			ASSERT_TRUE(added, "leaf entry added");
			vs_free(enc);
		}

		/* Add root centroid pointing to this leaf page.
		 * Use the first leaf vector as the root medoid. */
		float	   *root_vec = all_leaf_vecs[r * level1_count];
		Vec32Ref	root_ref = {.data = root_vec, .dim = dim};
		RaBitQData *root_enc = vs_rabitq_encode(params, root_ref, cent_ref);
		ASSERT_NOT_NULL(root_enc, "root encoding succeeded");

		bool added = prism_centroid_page_add(
				root_page, dim, leaf_blkno, level1_count, 0, root_enc);
		ASSERT_TRUE(added, "root entry added");
		vs_free(root_enc);
	}

	/* Create query */
	float			 *query		= make_test_vector(dim, 5555);
	Vec32Ref		  query_ref = {.data = query, .dim = dim};
	RaBitQQueryState *qstate =
			vs_rabitq_prepare_query(params, query_ref, cent_ref);
	ASSERT_NOT_NULL(qstate, "query state created");

	/* Set up storage with medoid vectors for reranking */
	TestStorage storage = {
			.base.ops = &test_storage_ops,
			.pages	  = pages,
			.vecs	  = (const float **)all_leaf_vecs,
			.dim	  = dim,
	};

	PrismCentroidSearchState search_state = {
			.qstate		= qstate,
			.query		= query,
			.storage	= &storage.base,
			.beam_width = 4, /* explore all roots so descent cannot prune the
							  * nearest's branch on RaBitQ estimate noise */
			.nprobe = 8,
			.dim	= dim,
	};

	PrismCentroidResult results[8];
	uint32_t			nresults = prism_centroid_beam_search(
			   &search_state, 0, 2, results, NULL, NULL);

	ASSERT_TRUE(nresults > 0, "should return at least one result");
	ASSERT_TRUE(nresults <= 8, "should return at most nprobe results");

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
	vs_rabitq_free_query(qstate);
	for (int i = 0; i < total_leaves; i++)
		vs_free(all_leaf_vecs[i]);
	vs_free(all_leaf_vecs);
	vs_free(pages);
	vs_rabitq_destroy(params);
}

/* Regression (F3/F9): a page's entry_count is read straight off disk and
 * drives the per-entry loops in score_page, which fill the scratch arrays
 * sized to the format's real per-page capacity. A count past that capacity
 * (corruption, a truncated write) must be rejected before it overruns the
 * scratch. Build a valid root page, corrupt its entry_count beyond the
 * RABITQ capacity, and confirm the scan aborts (vs_error) rather than
 * writing out of bounds. Runs in a forked child so the abort doesn't take
 * the test process down; the parent asserts the child died by SIGABRT. */
TEST(beam_search_rejects_corrupt_entry_count)
{
	Dimension	  dim	 = 64;
	RaBitQParams *params = vs_rabitq_create(dim, 42);
	ASSERT_NOT_NULL(params, "params created");

	float *centroid = vs_alloc(dim * sizeof(float));
	for (Dimension i = 0; i < dim; i++)
		centroid[i] = 0.0f;
	Vec32Ref cent_ref = {.data = centroid, .dim = dim};

	char *pages = vs_alloc(2 * (size_t)BLCKSZ);
	memset(pages, 0, 2 * (size_t)BLCKSZ);
	Page root_page = pages;
	prism_centroid_page_init(root_page, 0);

	float **vecs = vs_alloc(4 * sizeof(void *));
	for (int c = 0; c < 4; c++)
	{
		vecs[c]			= make_test_vector(dim, c * 37 + 1);
		Vec32Ref	v	= {.data = vecs[c], .dim = dim};
		RaBitQData *enc = vs_rabitq_encode(params, v, cent_ref);
		ASSERT_TRUE(
				prism_centroid_page_add(root_page, dim, 1, 0, 0, enc),
				"root entry added");
		vs_free(enc);
	}

	/* Corrupt: claim one more entry than the format can hold. */
	PrismCentroidPageOpaque *op = PRISM_CENTROID_OPAQUE(root_page);
	op->entry_count				= (uint16_t)(prism_centroid_max_entries_fmt(
										 dim, PRISM_CENTROID_FMT_RABITQ) +
								 1);

	float			 *query	 = make_test_vector(dim, 5555);
	Vec32Ref		  qref	 = {.data = query, .dim = dim};
	RaBitQQueryState *qstate = vs_rabitq_prepare_query(params, qref, cent_ref);
	TestStorage		  storage = {
				  .base.ops = &test_storage_ops,
				  .pages	= pages,
				  .vecs		= (const float **)vecs,
				  .dim		= dim,
	  };
	PrismCentroidSearchState st = {
			.qstate		= qstate,
			.query		= query,
			.storage	= &storage.base,
			.beam_width = 2,
			.nprobe		= 4,
			.dim		= dim,
	};

	fflush(NULL);
	pid_t pid = fork();
	ASSERT_TRUE(pid >= 0, "fork succeeded");
	if (pid == 0)
	{
		/* Child: the corrupt count must trip vs_error -> abort() before
		 * any out-of-bounds scratch write. Silence stderr so the expected
		 * diagnostic doesn't clutter the test log. */
		if (freopen("/dev/null", "w", stderr) == NULL)
			_exit(2);
		PrismCentroidResult results[4];
		prism_centroid_beam_search(&st, 0, 2, results, NULL, NULL);
		_exit(0); /* reached only if the check failed to fire */
	}

	int status = 0;
	waitpid(pid, &status, 0);
	ASSERT_TRUE(
			WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT,
			"corrupt entry_count aborts instead of overflowing the scratch");

	vs_rabitq_free_query(qstate);
	for (int c = 0; c < 4; c++)
		vs_free(vecs[c]);
	vs_free(vecs);
	vs_free(pages);
	vs_free(centroid);
	vs_free(query);
	vs_rabitq_destroy(params);
}

/* ----------------------------------------------------------------
 * Edge cases
 * ---------------------------------------------------------------- */

TEST(beam_search_null_state)
{
	PrismCentroidResult results[1];
	uint32_t n = prism_centroid_beam_search(NULL, 0, 1, results, NULL, NULL);
	ASSERT_EQ(0, n, "null state should return 0");
}

TEST(beam_search_null_results)
{
	TestStorage storage = {
			.base.ops = &test_storage_ops,
			.pages	  = NULL,
	};
	RaBitQQueryState		 dummy_qstate = {0};
	PrismCentroidSearchState state		  = {
				   .qstate	   = &dummy_qstate,
				   .storage	   = &storage.base,
				   .beam_width = 4,
				   .nprobe	   = 4,
				   .dim		   = 64,
	   };
	uint32_t n = prism_centroid_beam_search(&state, 0, 1, NULL, NULL, NULL);
	ASSERT_EQ(0, n, "null results should return 0");
}

TEST(beam_search_zero_levels)
{
	TestStorage storage = {
			.base.ops = &test_storage_ops,
			.pages	  = NULL,
	};
	RaBitQQueryState		 dummy_qstate = {0};
	PrismCentroidSearchState state		  = {
				   .qstate	   = &dummy_qstate,
				   .storage	   = &storage.base,
				   .beam_width = 4,
				   .nprobe	   = 4,
				   .dim		   = 64,
	   };
	PrismCentroidResult results[1];
	uint32_t n = prism_centroid_beam_search(&state, 0, 0, results, NULL, NULL);
	ASSERT_EQ(0, n, "zero levels should return 0");
}

TEST(beam_search_invalid_blkno)
{
	TestStorage storage = {
			.base.ops = &test_storage_ops,
			.pages	  = NULL,
	};
	RaBitQQueryState		 dummy_qstate = {0};
	PrismCentroidSearchState state		  = {
				   .qstate	   = &dummy_qstate,
				   .storage	   = &storage.base,
				   .beam_width = 4,
				   .nprobe	   = 4,
				   .dim		   = 64,
	   };
	PrismCentroidResult results[1];
	uint32_t			n = prism_centroid_beam_search(
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
	char	 *pages		  = vs_alloc((size_t)total_pages * BLCKSZ);
	memset(pages, 0, (size_t)total_pages * BLCKSZ);

	Page root_page = pages;
	prism_centroid_page_init_fmt(root_page, 0, PRISM_CENTROID_FMT_FLOAT);

	float **all_leaf_vecs = vs_alloc(
			(size_t)level0_count * level1_count * sizeof(void *));

	for (int r = 0; r < level0_count; r++)
	{
		BlockNumber leaf_blkno = (BlockNumber)(1 + r);
		Page		leaf_page  = pages + (size_t)leaf_blkno * BLCKSZ;
		prism_centroid_page_init_fmt(leaf_page, 1, PRISM_CENTROID_FMT_FLOAT);

		for (int c = 0; c < level1_count; c++)
		{
			int	   vec_idx		   = r * level1_count + c;
			float *vec			   = make_test_vector(dim, r * 1000 + c * 37);
			all_leaf_vecs[vec_idx] = vec;

			bool added = prism_centroid_page_add_entry(
					leaf_page,
					dim,
					200 + vec_idx,
					0,
					PRISM_CENTROID_FLAG_LEAF,
					vec);
			ASSERT_TRUE(added, "leaf float entry added");
		}

		/* Root centroid = first leaf vector */
		float *root_vec = all_leaf_vecs[r * level1_count];

		bool added = prism_centroid_page_add_entry(
				root_page, dim, leaf_blkno, level1_count, 0, root_vec);
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

	PrismCentroidSearchState search_state = {
			.qstate		= NULL, /* not needed for float pages */
			.query		= query,
			.storage	= &storage.base,
			.beam_width = 2,
			.nprobe		= 4,
			.dim		= dim,
	};

	PrismCentroidResult results[4];
	uint32_t			nresults = prism_centroid_beam_search(
			   &search_state, 0, 2, results, NULL, NULL);

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
		vs_free(all_leaf_vecs[i]);
	vs_free(all_leaf_vecs);
	vs_free(pages);
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
	char	 *pages		  = vs_alloc((size_t)total_pages * BLCKSZ);
	memset(pages, 0, (size_t)total_pages * BLCKSZ);

	Page root_page = pages;
	prism_centroid_page_init_fmt(root_page, 0, PRISM_CENTROID_FMT_HALF);

	/* Keep float versions for brute-force ground truth */
	float **all_leaf_f32 = vs_alloc(
			(size_t)level0_count * level1_count * sizeof(void *));
	half **all_leaf_half = vs_alloc(
			(size_t)level0_count * level1_count * sizeof(void *));

	for (int r = 0; r < level0_count; r++)
	{
		BlockNumber leaf_blkno = (BlockNumber)(1 + r);
		Page		leaf_page  = pages + (size_t)leaf_blkno * BLCKSZ;
		prism_centroid_page_init_fmt(leaf_page, 1, PRISM_CENTROID_FMT_HALF);

		for (int c = 0; c < level1_count; c++)
		{
			int	   vec_idx = r * level1_count + c;
			float *fvec	   = make_test_vector(dim, r * 1000 + c * 37);
			half  *hvec	   = vs_alloc(dim * sizeof(half));
			for (Dimension d = 0; d < dim; d++)
				hvec[d] = vs_float_to_half(fvec[d]);

			all_leaf_f32[vec_idx]  = fvec;
			all_leaf_half[vec_idx] = hvec;

			bool added = prism_centroid_page_add_entry(
					leaf_page,
					dim,
					200 + vec_idx,
					0,
					PRISM_CENTROID_FLAG_LEAF,
					hvec);
			ASSERT_TRUE(added, "leaf half entry added");
		}

		/* Root centroid = first leaf vector (half) */
		half *root_hvec = all_leaf_half[r * level1_count];

		bool added = prism_centroid_page_add_entry(
				root_page, dim, leaf_blkno, level1_count, 0, root_hvec);
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

	PrismCentroidSearchState search_state = {
			.qstate		= NULL,
			.query		= query,
			.storage	= &storage.base,
			.beam_width = 2,
			.nprobe		= 4,
			.dim		= dim,
	};

	PrismCentroidResult results[4];
	uint32_t			nresults = prism_centroid_beam_search(
			   &search_state, 0, 2, results, NULL, NULL);

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
		vs_free(all_leaf_f32[i]);
		vs_free(all_leaf_half[i]);
	}
	vs_free(all_leaf_f32);
	vs_free(all_leaf_half);
	vs_free(pages);
}
