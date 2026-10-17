/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
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
#include "vs_test.h"

TEST_GROUP(TopK);
TEST_MEMCTX_FIXTURE();

/* ----------------------------------------------------------------
 * Basic API tests
 * ---------------------------------------------------------------- */

TEST(topk_create_destroy)
{
	VsTopK *topk = vs_topk_create(10);
	ASSERT_NOT_NULL(topk, "create should succeed");
	ASSERT_EQ(0, topk->cand_count, "cand_count should be 0");
	ASSERT_EQ(10, topk->k, "k should be 10");
	vs_topk_destroy(topk);
}

TEST(topk_init_cleanup)
{
	VsTopK topk;
	vs_topk_init(&topk, 5);

	ASSERT_EQ(0, topk.cand_count, "cand_count should be 0");
	ASSERT_EQ(5, topk.k, "k should be 5");
	vs_topk_cleanup(&topk);
}

TEST(topk_init_defines_src_stamp)
{
	/* Every insert stamps cur_src onto the entry, and only the query scan
	 * ever sets it, so init has to leave it defined. The poison makes a
	 * missing initialization show up as the poison rather than as a
	 * chance zero. */
	VsTopK topk;
	memset(&topk, 0xAB, sizeof(topk));
	vs_topk_init(&topk, 3);
	vs_topk_insert(&topk, 1.0f, 0.0f, 7);

	ASSERT_EQ(1, topk.cand_count, "insert buffered the candidate");
	ASSERT_EQ(0, topk.candidates[0].src, "init defines the src stamp");
	vs_topk_cleanup(&topk);
}

TEST(topk_create_reports_heap_capacity)
{
	/* k_capacity is what a reset compares a new k against; read wrong, the
	 * reset skips its reallocation and leaves ub_heap sized for the
	 * original k while k says otherwise -- inserts then run past it. */
	VsTopK *topk = vs_topk_create(2);
	ASSERT_EQ(2, topk->k_capacity, "create reports its heap capacity");
	ASSERT_EQ(0, topk->cur_src, "create defines the src stamp");

	vs_topk_reset_to_k(topk, 64);
	for (uint32_t i = 0; i < 64; i++)
		vs_topk_insert(topk, (float)(64 - i), 0.0f, i + 1);
	ASSERT_EQ(64, topk->ub_count, "the threshold heap grew with k");

	vs_topk_destroy(topk);
}

TEST(topk_threshold_empty)
{
	VsTopK topk;
	vs_topk_init(&topk, 3);

	ASSERT_TRUE(
			isinf(vs_topk_threshold(&topk)),
			"threshold should be INFINITY when not full");
	vs_topk_cleanup(&topk);
}

/* ----------------------------------------------------------------
 * Insert and threshold (zero error)
 * ---------------------------------------------------------------- */

TEST(topk_insert_under_capacity)
{
	VsTopK topk;
	vs_topk_init(&topk, 3);

	vs_topk_insert(&topk, 5.0f, 0.0f, 1);
	ASSERT_TRUE(
			isinf(vs_topk_threshold(&topk)),
			"threshold still INFINITY (< k upper bounds)");

	vs_topk_insert(&topk, 3.0f, 0.0f, 2);
	vs_topk_insert(&topk, 7.0f, 0.0f, 3);

	Distance thresh = vs_topk_threshold(&topk);
	ASSERT_FLOAT_EQ(
			7.0f, thresh, 1e-6f, "threshold should be max upper_bound (7.0)");
	vs_topk_cleanup(&topk);
}

TEST(topk_insert_tightens_threshold)
{
	VsTopK topk;
	vs_topk_init(&topk, 3);

	vs_topk_insert(&topk, 5.0f, 0.0f, 1);
	vs_topk_insert(&topk, 3.0f, 0.0f, 2);
	vs_topk_insert(&topk, 7.0f, 0.0f, 3);

	/* Insert better upper bound → tightens threshold */
	vs_topk_insert(&topk, 4.0f, 0.0f, 4);

	Distance thresh = vs_topk_threshold(&topk);
	ASSERT_FLOAT_EQ(5.0f, thresh, 1e-6f, "threshold should tighten to 5.0");
	vs_topk_cleanup(&topk);
}

TEST(topk_insert_prunes_bad_candidates)
{
	VsTopK topk;
	vs_topk_init(&topk, 3);

	vs_topk_insert(&topk, 1.0f, 0.0f, 1);
	vs_topk_insert(&topk, 2.0f, 0.0f, 2);
	vs_topk_insert(&topk, 3.0f, 0.0f, 3);

	/* lower_bound=10 >= threshold=3 → pruned */
	vs_topk_insert(&topk, 10.0f, 0.0f, 99);

	VsTopKEntry results[32];
	uint32_t	count;
	vs_topk_extract_sorted(&topk, results, &count);

	/* Should only have the first 3 entries */
	ASSERT_EQ(3, count, "pruned entry should not appear");
	vs_topk_cleanup(&topk);
}

/* ----------------------------------------------------------------
 * Extract sorted
 * ---------------------------------------------------------------- */

TEST(topk_extract_sorted_ascending)
{
	VsTopK topk;
	vs_topk_init(&topk, 5);

	vs_topk_insert(&topk, 9.0f, 0.0f, 9);
	vs_topk_insert(&topk, 1.0f, 0.0f, 1);
	vs_topk_insert(&topk, 5.0f, 0.0f, 5);
	vs_topk_insert(&topk, 3.0f, 0.0f, 3);
	vs_topk_insert(&topk, 7.0f, 0.0f, 7);

	VsTopKEntry results[32];
	uint32_t	count;
	vs_topk_extract_sorted(&topk, results, &count);

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
	vs_topk_cleanup(&topk);
}

TEST(topk_extract_resets)
{
	VsTopK topk;
	vs_topk_init(&topk, 3);

	vs_topk_insert(&topk, 1.0f, 0.0f, 1);
	vs_topk_insert(&topk, 2.0f, 0.0f, 2);

	VsTopKEntry results[32];
	uint32_t	count;
	vs_topk_extract_sorted(&topk, results, &count);
	ASSERT_EQ(2, count, "should extract 2");

	vs_topk_reset(&topk);
	ASSERT_EQ(0, topk.cand_count, "cand_count should be 0 after reset");
	vs_topk_cleanup(&topk);
}

/* ----------------------------------------------------------------
 * Brute-force validation
 * ---------------------------------------------------------------- */

TEST(topk_brute_force_validation)
{
	const uint32_t k = 10;
	const uint32_t n = 100;

	VsTopK topk;
	vs_topk_init(&topk, k);

	/* Insert n candidates with pseudo-random distances, zero error */
	float *all_dists = vs_alloc(n * sizeof(float));
	for (uint32_t i = 0; i < n; i++)
	{
		all_dists[i] = (float)((i * 97 + 13) % 1000) / 10.0f;
		vs_topk_insert(&topk, all_dists[i], 0.0f, i);
	}

	VsTopKEntry *results = vs_alloc(topk.cand_count * sizeof(*results));
	uint32_t	 count;
	vs_topk_extract_sorted(&topk, results, &count);

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

	vs_free(results);
	vs_free(all_dists);
	vs_topk_cleanup(&topk);
}

/* ----------------------------------------------------------------
 * Error-aware behavior
 * ---------------------------------------------------------------- */

TEST(topk_error_affects_threshold)
{
	VsTopK topk;
	vs_topk_init(&topk, 2);

	/* dist=3, error=1 → upper_bound=4 */
	vs_topk_insert(&topk, 3.0f, 1.0f, 1);
	/* dist=2, error=0.5 → upper_bound=2.5 */
	vs_topk_insert(&topk, 2.0f, 0.5f, 2);

	/* Threshold = max of {4.0, 2.5} = 4.0 (Kth-smallest ub) */
	ASSERT_FLOAT_EQ(
			4.0f,
			vs_topk_threshold(&topk),
			1e-6f,
			"threshold should be worst upper_bound");
	vs_topk_cleanup(&topk);
}

TEST(topk_keeps_high_error_candidates)
{
	VsTopK topk;
	vs_topk_init(&topk, 2);

	/* dist=2, error=0 → ub=2 */
	vs_topk_insert(&topk, 2.0f, 0.0f, 1);
	/* dist=3, error=0 → ub=3 */
	vs_topk_insert(&topk, 3.0f, 0.0f, 2);
	/* Threshold = 3.0 */

	/* dist=10, error=9 → lb=1, ub=19
	 * lower_bound=1 < threshold=3 → NOT pruned.
	 * This candidate might be the closest! */
	vs_topk_insert(&topk, 10.0f, 9.0f, 3);

	VsTopKEntry results[32];
	uint32_t	count;
	vs_topk_extract_sorted(&topk, results, &count);

	/* All three should be kept (high-error entry not evicted) */
	ASSERT_EQ(3, count, "high-error candidate should be kept");

	bool has_3 = false;
	for (uint32_t i = 0; i < count; i++)
	{
		if (results[i].id == 3)
			has_3 = true;
	}
	ASSERT_TRUE(has_3, "high-error entry (id=3) must be in results");
	vs_topk_cleanup(&topk);
}

TEST(topk_buffers_more_than_k)
{
	VsTopK topk;
	vs_topk_init(&topk, 2);

	/* All with overlapping error bounds — none can be pruned */
	vs_topk_insert(&topk, 5.0f, 3.0f, 1); /* lb=2, ub=8 */
	vs_topk_insert(&topk, 6.0f, 3.0f, 2); /* lb=3, ub=9 */
	vs_topk_insert(&topk, 7.0f, 3.0f, 3); /* lb=4, ub=10 */

	/* threshold = 9.0 (Kth-smallest ub from {8,9})
	 * Entry 3: lb=4 < 9 → kept */
	VsTopKEntry results[32];
	uint32_t	count;
	vs_topk_extract_sorted(&topk, results, &count);

	ASSERT_TRUE(
			count > 2, "should buffer more than K with overlapping bounds");
	ASSERT_EQ(3, count, "all 3 entries should be in buffer");
	vs_topk_cleanup(&topk);
}

TEST(topk_error_preserved_in_extract)
{
	VsTopK topk;
	vs_topk_init(&topk, 5);

	vs_topk_insert(&topk, 2.0f, 0.5f, 1);
	vs_topk_insert(&topk, 4.0f, 1.0f, 2);
	vs_topk_insert(&topk, 6.0f, 0.1f, 3);

	VsTopKEntry results[32];
	uint32_t	count;
	vs_topk_extract_sorted(&topk, results, &count);
	ASSERT_EQ(3, count, "should extract 3");

	/* Results sorted by distance ascending, error preserved */
	ASSERT_FLOAT_EQ(2.0f, results[0].distance, 1e-6f, "first dist=2.0");
	ASSERT_FLOAT_EQ(0.5f, results[0].error, 1e-6f, "first error=0.5");
	ASSERT_FLOAT_EQ(4.0f, results[1].distance, 1e-6f, "second dist=4.0");
	ASSERT_FLOAT_EQ(1.0f, results[1].error, 1e-6f, "second error=1.0");
	ASSERT_FLOAT_EQ(6.0f, results[2].distance, 1e-6f, "third dist=6.0");
	ASSERT_FLOAT_EQ(0.1f, results[2].error, 1e-6f, "third error=0.1");
	vs_topk_cleanup(&topk);
}

TEST(topk_filters_stale_candidates)
{
	VsTopK topk;
	vs_topk_init(&topk, 2);

	/* First two: threshold starts at INFINITY, so accepted */
	vs_topk_insert(&topk, 1.0f, 0.0f, 1); /* lb=1, ub=1 */
	vs_topk_insert(&topk, 2.0f, 0.0f, 2); /* lb=2, ub=2 */
	/* threshold = 2.0 */

	/* This tightens the threshold heap */
	vs_topk_insert(&topk, 0.5f, 0.0f, 3); /* lb=0.5, ub=0.5 */
	/* threshold ub_heap now {1.0, 0.5} → root=1.0 */

	/* Entry id=2 (lb=2) >= final threshold=1.0 → stale, filtered */
	VsTopKEntry results[32];
	uint32_t	count;
	vs_topk_extract_sorted(&topk, results, &count);

	ASSERT_EQ(2, count, "stale entry should be filtered");

	bool has_2 = false;
	for (uint32_t i = 0; i < count; i++)
	{
		if (results[i].id == 2)
			has_2 = true;
	}
	ASSERT_FALSE(has_2, "id=2 (lb=2 >= threshold=1) should be filtered");
	vs_topk_cleanup(&topk);
}

/* ----------------------------------------------------------------
 * Edge cases
 * ---------------------------------------------------------------- */

TEST(topk_k_equals_one)
{
	VsTopK topk;
	vs_topk_init(&topk, 1);

	vs_topk_insert(&topk, 10.0f, 0.0f, 10);
	ASSERT_FLOAT_EQ(
			10.0f,
			vs_topk_threshold(&topk),
			1e-6f,
			"threshold with one entry");

	vs_topk_insert(&topk, 5.0f, 0.0f, 5);
	ASSERT_FLOAT_EQ(
			5.0f,
			vs_topk_threshold(&topk),
			1e-6f,
			"threshold after better insert");

	vs_topk_insert(&topk, 20.0f, 0.0f, 20);

	VsTopKEntry results[32];
	uint32_t	count;
	vs_topk_extract_sorted(&topk, results, &count);

	/* id=10 (lb=10 >= threshold=5) should be filtered.
	 * id=20 (lb=20 >= threshold=5) should be pruned.
	 * Only id=5 remains. */
	ASSERT_EQ(1, count, "should extract 1");
	ASSERT_EQ(5, results[0].id, "should be the best");
	vs_topk_cleanup(&topk);
}

TEST(topk_all_equal_distances)
{
	VsTopK topk;
	vs_topk_init(&topk, 5);

	for (uint32_t i = 0; i < 10; i++)
		vs_topk_insert(&topk, 42.0f, 0.0f, i);

	VsTopKEntry results[32];
	uint32_t	count;
	vs_topk_extract_sorted(&topk, results, &count);

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
	vs_topk_cleanup(&topk);
}

TEST(topk_extract_empty)
{
	VsTopK topk;
	vs_topk_init(&topk, 3);

	VsTopKEntry results[1];
	uint32_t	count;
	vs_topk_extract_sorted(&topk, results, &count);
	ASSERT_EQ(0, count, "empty extraction should return 0");
	vs_topk_cleanup(&topk);
}

TEST(topk_reset)
{
	VsTopK topk;
	vs_topk_init(&topk, 3);

	vs_topk_insert(&topk, 1.0f, 0.0f, 1);
	vs_topk_insert(&topk, 2.0f, 0.0f, 2);
	ASSERT_TRUE(topk.cand_count > 0, "has candidates before reset");

	vs_topk_reset(&topk);
	ASSERT_EQ(0, topk.cand_count, "cand_count after reset");
	ASSERT_TRUE(
			isinf(vs_topk_threshold(&topk)),
			"threshold after reset should be INFINITY");

	/* Should work again after reset */
	vs_topk_insert(&topk, 5.0f, 0.0f, 5);
	ASSERT_EQ(1, topk.cand_count, "cand_count after re-insert");
	vs_topk_cleanup(&topk);
}

/* ----------------------------------------------------------------
 * Dedup map vs a linear-scan reference heap
 * ---------------------------------------------------------------- */

/* Threshold heap maintained with a linear id scan per insert. The
 * mapped heap must perform the same moves, so both arrays match
 * entry for entry. */
typedef struct RefHeap
{
	Distance ub[256];
	uint64_t id[256];
	uint32_t count;
	uint32_t k;
} RefHeap;

static void
ref_swap(RefHeap *h, uint32_t a, uint32_t b)
{
	Distance d = h->ub[a];
	h->ub[a]   = h->ub[b];
	h->ub[b]   = d;
	uint64_t t = h->id[a];
	h->id[a]   = h->id[b];
	h->id[b]   = t;
}

static void
ref_sift_down(RefHeap *h, uint32_t i)
{
	for (;;)
	{
		uint32_t l = 2 * i + 1, r = 2 * i + 2, m = i;
		if (l < h->count && h->ub[l] > h->ub[m])
			m = l;
		if (r < h->count && h->ub[r] > h->ub[m])
			m = r;
		if (m == i)
			return;
		ref_swap(h, i, m);
		i = m;
	}
}

static void
ref_insert(RefHeap *h, Distance lb, Distance ub, uint64_t id, bool unique)
{
	if (h->count == h->k && !(lb < h->ub[0]))
		return;
	for (uint32_t i = 0; !unique && i < h->count; i++)
	{
		if (h->id[i] == id)
		{
			if (ub < h->ub[i])
			{
				h->ub[i] = ub;
				ref_sift_down(h, i);
			}
			return;
		}
	}
	if (h->count < h->k)
	{
		uint32_t i = h->count++;
		h->ub[i]   = ub;
		h->id[i]   = id;
		while (i > 0 && h->ub[i] > h->ub[(i - 1) / 2])
		{
			ref_swap(h, i, (i - 1) / 2);
			i = (i - 1) / 2;
		}
	}
	else if (ub < h->ub[0])
	{
		h->ub[0] = ub;
		h->id[0] = id;
		ref_sift_down(h, 0);
	}
}

/* Heavy id reuse forces map hits, evictions and backward-shift
 * deletes. Resets cover shrinking k, growing it within the allocated
 * capacity and growing it past. Each round opens with unique inserts,
 * so the map is built over a non-empty heap after a growing reset and
 * kept current by unique inserts otherwise. */
TEST(topk_dedup_map_matches_linear_scan)
{
	const uint32_t ks[]			  = {1, 7, 64, 200, 33, 150, 256};
	uint64_t	   rng			  = 0x2545F4914F6CDD1DULL;
	uint64_t	   next_unique_id = 1u << 20;
	VsTopK		  *topk			  = vs_topk_create(ks[0]);

	for (size_t round = 0; round < sizeof(ks) / sizeof(ks[0]); round++)
	{
		RefHeap ref = {.count = 0, .k = ks[round]};
		vs_topk_reset_to_k(topk, ks[round]);

		for (uint32_t step = 0; step < 20000; step++)
		{
			rng ^= rng << 13;
			rng ^= rng >> 7;
			rng ^= rng << 17;

			Distance d	  = (float)(rng % 10000) / 100.0f;
			Distance err  = (float)((rng >> 16) % 300) / 100.0f;
			bool	 uniq = step < 8 || (rng >> 40) % 50 == 0;
			uint64_t id	  = uniq ? next_unique_id++
								 : (rng >> 24) % (3 * ks[round] + 1);

			if (uniq)
				vs_topk_insert_unique(topk, d, err, id);
			else
				vs_topk_insert(topk, d, err, id);
			ref_insert(&ref, d - err, d + err, id, uniq);

			ASSERT_EQ(ref.count, topk->ub_count, "heap size matches");
			for (uint32_t i = 0; i < ref.count; i++)
			{
				ASSERT_TRUE(ref.id[i] == topk->ub_ids[i], "heap ids match");
				ASSERT_TRUE(ref.ub[i] == topk->ub_heap[i], "heap ubs match");
			}

			if (topk->map_pos == NULL)
				continue;
			uint32_t used = 0;
			for (uint32_t s = 0; s <= topk->map_mask; s++)
				used += topk->map_pos[s] != UINT32_MAX;
			ASSERT_EQ(topk->ub_count, used, "one map slot per heap entry");
			for (uint32_t i = 0; i < topk->ub_count; i++)
			{
				uint32_t s = topk->ub_slot[i];
				ASSERT_EQ(
						i, topk->map_pos[s], "slot points back at its entry");
				ASSERT_TRUE(
						topk->map_ids[s] == topk->ub_ids[i],
						"slot holds its id");
			}
		}
	}
	vs_topk_destroy(topk);
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
			VsTopK topk;
			vs_topk_init(&topk, k);
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
				vs_topk_insert(&topk, d, 1000.0f, (uint64_t)(i % 250) + 1);
			}

			VsTopKEntry *res = vs_alloc(topk.cand_count * sizeof(VsTopKEntry));
			uint32_t	 out;
			vs_topk_extract_sorted_capped(&topk, res, &out, caps[c]);

			uint32_t expect = caps[c] < 250 ? caps[c] : 250;
			ASSERT_EQ(expect, out, "capped extract returns cap uniques");
			for (uint32_t i = 1; i < out; i++)
				ASSERT_TRUE(
						res[i - 1].distance <= res[i].distance,
						"capped extract sorted ascending");
			for (uint32_t i = 0; i < out; i++)
				for (uint32_t j = i + 1; j < out; j++)
					ASSERT_TRUE(res[i].id != res[j].id, "no duplicate ids");

			vs_free(res);
			vs_topk_cleanup(&topk);
		}
	}
}
