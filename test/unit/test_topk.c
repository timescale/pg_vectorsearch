/*
 * test_topk.c - Unit tests for bounded top-K collection
 *
 * Tests cover:
 * - Basic create/init/destroy
 * - Insert and threshold behavior
 * - Extract sorted order
 * - Brute-force validation against linear scan
 * - Error-aware threshold and candidate buffering
 * - Edge cases (k=1, all equal distances, empty)
 */

#include <math.h>
#include <string.h>

#include "algo/topk.h"
#include "core/memory.h"
#include "mkt_test.h"

TEST_GROUP(TopK);
TEST_MEMCTX_FIXTURE();

/* ----------------------------------------------------------------
 * Basic API tests
 * ---------------------------------------------------------------- */

TEST(topk_create_destroy)
{
	MktTopK *topk = mkt_topk_create(10);
	ASSERT_NOT_NULL(topk, "create should succeed");
	ASSERT_EQ(0, topk->cand_count, "cand_count should be 0");
	ASSERT_EQ(10, topk->k, "k should be 10");
	mkt_topk_destroy(topk);
}

TEST(topk_init_cleanup)
{
	MktTopK topk;
	mkt_topk_init(&topk, 5);

	ASSERT_EQ(0, topk.cand_count, "cand_count should be 0");
	ASSERT_EQ(5, topk.k, "k should be 5");
	mkt_topk_cleanup(&topk);
}

TEST(topk_threshold_empty)
{
	MktTopK topk;
	mkt_topk_init(&topk, 3);

	ASSERT_TRUE(
			isinf(mkt_topk_threshold(&topk)),
			"threshold should be INFINITY when not full");
	mkt_topk_cleanup(&topk);
}

/* ----------------------------------------------------------------
 * Insert and threshold (zero error)
 * ---------------------------------------------------------------- */

TEST(topk_insert_under_capacity)
{
	MktTopK topk;
	mkt_topk_init(&topk, 3);

	mkt_topk_insert(&topk, 5.0f, 0.0f, 1);
	ASSERT_TRUE(
			isinf(mkt_topk_threshold(&topk)),
			"threshold still INFINITY (< k upper bounds)");

	mkt_topk_insert(&topk, 3.0f, 0.0f, 2);
	mkt_topk_insert(&topk, 7.0f, 0.0f, 3);

	Distance thresh = mkt_topk_threshold(&topk);
	ASSERT_FLOAT_EQ(
			7.0f, thresh, 1e-6f, "threshold should be max upper_bound (7.0)");
	mkt_topk_cleanup(&topk);
}

TEST(topk_insert_tightens_threshold)
{
	MktTopK topk;
	mkt_topk_init(&topk, 3);

	mkt_topk_insert(&topk, 5.0f, 0.0f, 1);
	mkt_topk_insert(&topk, 3.0f, 0.0f, 2);
	mkt_topk_insert(&topk, 7.0f, 0.0f, 3);

	/* Insert better upper bound → tightens threshold */
	mkt_topk_insert(&topk, 4.0f, 0.0f, 4);

	Distance thresh = mkt_topk_threshold(&topk);
	ASSERT_FLOAT_EQ(5.0f, thresh, 1e-6f, "threshold should tighten to 5.0");
	mkt_topk_cleanup(&topk);
}

TEST(topk_insert_prunes_bad_candidates)
{
	MktTopK topk;
	mkt_topk_init(&topk, 3);

	mkt_topk_insert(&topk, 1.0f, 0.0f, 1);
	mkt_topk_insert(&topk, 2.0f, 0.0f, 2);
	mkt_topk_insert(&topk, 3.0f, 0.0f, 3);

	/* lower_bound=10 >= threshold=3 → pruned */
	mkt_topk_insert(&topk, 10.0f, 0.0f, 99);

	MktTopKEntry results[32];
	uint32_t	 count;
	mkt_topk_extract_sorted(&topk, results, &count);

	/* Should only have the first 3 entries */
	ASSERT_EQ(3, count, "pruned entry should not appear");
	mkt_topk_cleanup(&topk);
}

/* ----------------------------------------------------------------
 * Extract sorted
 * ---------------------------------------------------------------- */

TEST(topk_extract_sorted_ascending)
{
	MktTopK topk;
	mkt_topk_init(&topk, 5);

	mkt_topk_insert(&topk, 9.0f, 0.0f, 9);
	mkt_topk_insert(&topk, 1.0f, 0.0f, 1);
	mkt_topk_insert(&topk, 5.0f, 0.0f, 5);
	mkt_topk_insert(&topk, 3.0f, 0.0f, 3);
	mkt_topk_insert(&topk, 7.0f, 0.0f, 7);

	MktTopKEntry results[32];
	uint32_t	 count;
	mkt_topk_extract_sorted(&topk, results, &count);

	ASSERT_EQ(5, count, "should extract 5 entries");

	for (uint32_t i = 1; i < count; i++)
	{
		char msg[128];
		snprintf(
				msg,
				sizeof(msg),
				"result[%u] (%.1f) should be <= result[%u] (%.1f)",
				i - 1,
				results[i - 1].distance,
				i,
				results[i].distance);
		ASSERT_TRUE(results[i - 1].distance <= results[i].distance, msg);
	}

	/* Verify IDs match expected order */
	ASSERT_EQ(1, results[0].id, "first should be id=1 (dist 1.0)");
	ASSERT_EQ(3, results[1].id, "second should be id=3 (dist 3.0)");
	ASSERT_EQ(5, results[2].id, "third should be id=5 (dist 5.0)");
	ASSERT_EQ(7, results[3].id, "fourth should be id=7 (dist 7.0)");
	ASSERT_EQ(9, results[4].id, "fifth should be id=9 (dist 9.0)");
	mkt_topk_cleanup(&topk);
}

TEST(topk_extract_resets)
{
	MktTopK topk;
	mkt_topk_init(&topk, 3);

	mkt_topk_insert(&topk, 1.0f, 0.0f, 1);
	mkt_topk_insert(&topk, 2.0f, 0.0f, 2);

	MktTopKEntry results[32];
	uint32_t	 count;
	mkt_topk_extract_sorted(&topk, results, &count);
	ASSERT_EQ(2, count, "should extract 2");

	mkt_topk_reset(&topk);
	ASSERT_EQ(0, topk.cand_count, "cand_count should be 0 after reset");
	mkt_topk_cleanup(&topk);
}

/* ----------------------------------------------------------------
 * Brute-force validation
 * ---------------------------------------------------------------- */

TEST(topk_brute_force_validation)
{
	const uint32_t k = 10;
	const uint32_t n = 100;

	MktTopK topk;
	mkt_topk_init(&topk, k);

	/* Insert n candidates with pseudo-random distances, zero error */
	float *all_dists = mkt_alloc(n * sizeof(float));
	for (uint32_t i = 0; i < n; i++)
	{
		all_dists[i] = (float)((i * 97 + 13) % 1000) / 10.0f;
		mkt_topk_insert(&topk, all_dists[i], 0.0f, i);
	}

	MktTopKEntry *results = mkt_alloc(topk.cand_count * sizeof(*results));
	uint32_t	  count;
	mkt_topk_extract_sorted(&topk, results, &count);

	/* With zero error, threshold pruning eliminates candidates
	 * whose distance >= Kth-smallest, so count should be <= k+1
	 * (boundary candidates may survive). Check the first k. */
	ASSERT_TRUE(count >= k, "should have at least k results");

	/* Brute force: sort all distances and take first k */
	for (uint32_t i = 0; i < n; i++)
	{
		for (uint32_t j = i + 1; j < n; j++)
		{
			if (all_dists[j] < all_dists[i])
			{
				float tmp	 = all_dists[i];
				all_dists[i] = all_dists[j];
				all_dists[j] = tmp;
			}
		}
	}

	/* Verify first k results match brute force */
	for (uint32_t i = 0; i < k; i++)
	{
		char msg[128];
		snprintf(
				msg,
				sizeof(msg),
				"result[%u] dist %.1f should match brute-force %.1f",
				i,
				results[i].distance,
				all_dists[i]);
		ASSERT_FLOAT_EQ(all_dists[i], results[i].distance, 1e-6f, msg);
	}

	mkt_free(results);
	mkt_free(all_dists);
	mkt_topk_cleanup(&topk);
}

/* ----------------------------------------------------------------
 * Error-aware behavior
 * ---------------------------------------------------------------- */

TEST(topk_error_affects_threshold)
{
	MktTopK topk;
	mkt_topk_init(&topk, 2);

	/* dist=3, error=1 → upper_bound=4 */
	mkt_topk_insert(&topk, 3.0f, 1.0f, 1);
	/* dist=2, error=0.5 → upper_bound=2.5 */
	mkt_topk_insert(&topk, 2.0f, 0.5f, 2);

	/* Threshold = max of {4.0, 2.5} = 4.0 (Kth-smallest ub) */
	ASSERT_FLOAT_EQ(
			4.0f,
			mkt_topk_threshold(&topk),
			1e-6f,
			"threshold should be worst upper_bound");
	mkt_topk_cleanup(&topk);
}

TEST(topk_keeps_high_error_candidates)
{
	MktTopK topk;
	mkt_topk_init(&topk, 2);

	/* dist=2, error=0 → ub=2 */
	mkt_topk_insert(&topk, 2.0f, 0.0f, 1);
	/* dist=3, error=0 → ub=3 */
	mkt_topk_insert(&topk, 3.0f, 0.0f, 2);
	/* Threshold = 3.0 */

	/* dist=10, error=9 → lb=1, ub=19
	 * lower_bound=1 < threshold=3 → NOT pruned.
	 * This candidate might be the closest! */
	mkt_topk_insert(&topk, 10.0f, 9.0f, 3);

	MktTopKEntry results[32];
	uint32_t	 count;
	mkt_topk_extract_sorted(&topk, results, &count);

	/* All three should be kept (high-error entry not evicted) */
	ASSERT_EQ(3, count, "high-error candidate should be kept");

	bool has_3 = false;
	for (uint32_t i = 0; i < count; i++)
	{
		if (results[i].id == 3)
			has_3 = true;
	}
	ASSERT_TRUE(has_3, "high-error entry (id=3) must be in results");
	mkt_topk_cleanup(&topk);
}

TEST(topk_buffers_more_than_k)
{
	MktTopK topk;
	mkt_topk_init(&topk, 2);

	/* All with overlapping error bounds — none can be pruned */
	mkt_topk_insert(&topk, 5.0f, 3.0f, 1); /* lb=2, ub=8 */
	mkt_topk_insert(&topk, 6.0f, 3.0f, 2); /* lb=3, ub=9 */
	mkt_topk_insert(&topk, 7.0f, 3.0f, 3); /* lb=4, ub=10 */

	/* threshold = 9.0 (Kth-smallest ub from {8,9})
	 * Entry 3: lb=4 < 9 → kept */
	MktTopKEntry results[32];
	uint32_t	 count;
	mkt_topk_extract_sorted(&topk, results, &count);

	ASSERT_TRUE(
			count > 2, "should buffer more than K with overlapping bounds");
	ASSERT_EQ(3, count, "all 3 entries should be in buffer");
	mkt_topk_cleanup(&topk);
}

TEST(topk_error_preserved_in_extract)
{
	MktTopK topk;
	mkt_topk_init(&topk, 5);

	mkt_topk_insert(&topk, 2.0f, 0.5f, 1);
	mkt_topk_insert(&topk, 4.0f, 1.0f, 2);
	mkt_topk_insert(&topk, 6.0f, 0.1f, 3);

	MktTopKEntry results[32];
	uint32_t	 count;
	mkt_topk_extract_sorted(&topk, results, &count);
	ASSERT_EQ(3, count, "should extract 3");

	/* Results sorted by distance ascending, error preserved */
	ASSERT_FLOAT_EQ(2.0f, results[0].distance, 1e-6f, "first dist=2.0");
	ASSERT_FLOAT_EQ(0.5f, results[0].error, 1e-6f, "first error=0.5");
	ASSERT_FLOAT_EQ(4.0f, results[1].distance, 1e-6f, "second dist=4.0");
	ASSERT_FLOAT_EQ(1.0f, results[1].error, 1e-6f, "second error=1.0");
	ASSERT_FLOAT_EQ(6.0f, results[2].distance, 1e-6f, "third dist=6.0");
	ASSERT_FLOAT_EQ(0.1f, results[2].error, 1e-6f, "third error=0.1");
	mkt_topk_cleanup(&topk);
}

TEST(topk_filters_stale_candidates)
{
	MktTopK topk;
	mkt_topk_init(&topk, 2);

	/* First two: threshold starts at INFINITY, so accepted */
	mkt_topk_insert(&topk, 1.0f, 0.0f, 1); /* lb=1, ub=1 */
	mkt_topk_insert(&topk, 2.0f, 0.0f, 2); /* lb=2, ub=2 */
	/* threshold = 2.0 */

	/* This tightens the threshold heap */
	mkt_topk_insert(&topk, 0.5f, 0.0f, 3); /* lb=0.5, ub=0.5 */
	/* threshold ub_heap now {1.0, 0.5} → root=1.0 */

	/* Entry id=2 (lb=2) >= final threshold=1.0 → stale, filtered */
	MktTopKEntry results[32];
	uint32_t	 count;
	mkt_topk_extract_sorted(&topk, results, &count);

	ASSERT_EQ(2, count, "stale entry should be filtered");

	bool has_2 = false;
	for (uint32_t i = 0; i < count; i++)
	{
		if (results[i].id == 2)
			has_2 = true;
	}
	ASSERT_FALSE(has_2, "id=2 (lb=2 >= threshold=1) should be filtered");
	mkt_topk_cleanup(&topk);
}

/* ----------------------------------------------------------------
 * Edge cases
 * ---------------------------------------------------------------- */

TEST(topk_k_equals_one)
{
	MktTopK topk;
	mkt_topk_init(&topk, 1);

	mkt_topk_insert(&topk, 10.0f, 0.0f, 10);
	ASSERT_FLOAT_EQ(
			10.0f,
			mkt_topk_threshold(&topk),
			1e-6f,
			"threshold with one entry");

	mkt_topk_insert(&topk, 5.0f, 0.0f, 5);
	ASSERT_FLOAT_EQ(
			5.0f,
			mkt_topk_threshold(&topk),
			1e-6f,
			"threshold after better insert");

	mkt_topk_insert(&topk, 20.0f, 0.0f, 20);

	MktTopKEntry results[32];
	uint32_t	 count;
	mkt_topk_extract_sorted(&topk, results, &count);

	/* id=10 (lb=10 >= threshold=5) should be filtered.
	 * id=20 (lb=20 >= threshold=5) should be pruned.
	 * Only id=5 remains. */
	ASSERT_EQ(1, count, "should extract 1");
	ASSERT_EQ(5, results[0].id, "should be the best");
	mkt_topk_cleanup(&topk);
}

TEST(topk_all_equal_distances)
{
	MktTopK topk;
	mkt_topk_init(&topk, 5);

	for (uint32_t i = 0; i < 10; i++)
		mkt_topk_insert(&topk, 42.0f, 0.0f, i);

	MktTopKEntry results[32];
	uint32_t	 count;
	mkt_topk_extract_sorted(&topk, results, &count);

	/* With zero error: ub=42 for all. threshold=42.
	 * lb=42 >= threshold=42 → later entries pruned.
	 * Only the first 5 (before threshold fills) survive. */
	ASSERT_EQ(5, count, "should extract 5");

	for (uint32_t i = 0; i < count; i++)
	{
		ASSERT_FLOAT_EQ(
				42.0f,
				results[i].distance,
				1e-6f,
				"all distances should be equal");
	}
	mkt_topk_cleanup(&topk);
}

TEST(topk_extract_empty)
{
	MktTopK topk;
	mkt_topk_init(&topk, 3);

	MktTopKEntry results[1];
	uint32_t	 count;
	mkt_topk_extract_sorted(&topk, results, &count);
	ASSERT_EQ(0, count, "empty extraction should return 0");
	mkt_topk_cleanup(&topk);
}

TEST(topk_reset)
{
	MktTopK topk;
	mkt_topk_init(&topk, 3);

	mkt_topk_insert(&topk, 1.0f, 0.0f, 1);
	mkt_topk_insert(&topk, 2.0f, 0.0f, 2);
	ASSERT_TRUE(topk.cand_count > 0, "has candidates before reset");

	mkt_topk_reset(&topk);
	ASSERT_EQ(0, topk.cand_count, "cand_count after reset");
	ASSERT_TRUE(
			isinf(mkt_topk_threshold(&topk)),
			"threshold after reset should be INFINITY");

	/* Should work again after reset */
	mkt_topk_insert(&topk, 5.0f, 0.0f, 5);
	ASSERT_EQ(1, topk.cand_count, "cand_count after re-insert");
	mkt_topk_cleanup(&topk);
}

/* ----------------------------------------------------------------
 * Capped extraction (quickselect + dedup) vs brute force
 * ---------------------------------------------------------------- */

/* Adversarial distance distributions for the quickselect partition:
 * all-equal (pivot-equal band), sorted, reverse-sorted, two-valued,
 * and scattered, each with duplicate ids (SOAR-style replicas). The
 * capped extract must return exactly the cap best unique ids, sorted;
 * bounds bugs in the partition surface under the sanitizer jobs. */
TEST(topk_extract_capped_adversarial)
{
	const uint32_t n	  = 500;
	const uint32_t k	  = 10;
	const uint32_t caps[] = {1, 10, 33, 160, 500, 1000};

	for (int dist_kind = 0; dist_kind < 5; dist_kind++)
	{
		for (size_t c = 0; c < sizeof(caps) / sizeof(caps[0]); c++)
		{
			MktTopK topk;
			mkt_topk_init(&topk, k);
			/* Every id twice (replica), error large enough that no
			 * candidate is threshold-pruned: the buffer holds all. */
			for (uint32_t i = 0; i < n; i++)
			{
				float d;
				switch (dist_kind)
				{
				case 0:
					d = 1.0f;
					break;
				case 1:
					d = (float)i;
					break;
				case 2:
					d = (float)(n - i);
					break;
				case 3:
					d = (float)(i % 2);
					break;
				default:
					d = (float)((i * 7919u) % 257u);
					break;
				}
				mkt_topk_insert(&topk, d, 1000.0f, (uint64_t)(i % 250) + 1);
			}

			MktTopKEntry *res = mkt_alloc(
					topk.cand_count * sizeof(MktTopKEntry));
			uint32_t out;
			mkt_topk_extract_sorted_capped(&topk, res, &out, caps[c]);

			uint32_t expect = caps[c] < 250 ? caps[c] : 250;
			ASSERT_EQ(expect, out, "capped extract returns cap uniques");
			for (uint32_t i = 1; i < out; i++)
				ASSERT_TRUE(
						res[i - 1].distance <= res[i].distance,
						"capped extract sorted ascending");
			for (uint32_t i = 0; i < out; i++)
				for (uint32_t j = i + 1; j < out; j++)
					ASSERT_TRUE(res[i].id != res[j].id, "no duplicate ids");

			mkt_free(res);
			mkt_topk_cleanup(&topk);
		}
	}
}
