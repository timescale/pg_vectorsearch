/*
 * test_index_base.c - Unit tests for the shared index descriptor's helpers
 *
 * Tests cover:
 * - Posting page accounting: agreement with the block range on a freshly
 *   built index, and independence from it once a split has appended a
 *   centroid page past the posting region
 */

#include "index/index_base.h"
#include "vs_test.h"

TEST_GROUP(IndexBase);

/*
 * A build reserves [PRISM_FIRST_CENTROID_BLKNO, first_posting) for the
 * centroid pages, so ncentroid_pages is exactly first_posting - 1 and the
 * maintained count and the block range have to agree. This is the property
 * that makes the count form safe to adopt: it changes nothing until a split
 * appends.
 */
TEST(posting_pages_match_the_block_range_on_a_fresh_build)
{
	const double num_pages	   = 1058.0;
	const double first_posting = 2.0;

	ASSERT_FLOAT_EQ(
			num_pages - first_posting,
			prism_index_posting_pages(num_pages, first_posting - 1.0),
			0.0,
			"on a fresh build the count form equals the block range");
}

/*
 * A split with no room on a level-0 centroid page extends the relation and
 * chains the new centroid page past the posting region. ncentroid_pages
 * counts it, so a block range counts it a second time -- as a posting page.
 * The figures are from a real index: dim 1024, nlist 1, rebalanced down to a
 * target of 6, which leaves ncentroid_pages at 10 against a first_posting
 * that is still 2, so nine centroid pages sit above the posting region.
 */
TEST(appended_centroid_pages_are_not_counted_as_posting_pages)
{
	ASSERT_FLOAT_EQ(
			1047.0,
			prism_index_posting_pages(1058.0, 10.0),
			0.0,
			"the nine appended centroid pages are not posting pages");
}

/*
 * The same statement without the arithmetic: no page may be priced as both a
 * centroid page and a posting page, so the two counts together cannot exceed
 * the relation's non-metapage size.
 */
TEST(no_page_is_counted_in_both_terms)
{
	const double num_pages = 1058.0;
	const double ncentroid = 10.0;

	ASSERT_TRUE(
			ncentroid + prism_index_posting_pages(num_pages, ncentroid) <=
					num_pages - (double)PRISM_FIRST_CENTROID_BLKNO,
			"centroid and posting pages together fit in the relation");
}

/* An index whose posting region is empty still has to divide by something. */
TEST(posting_pages_are_floored_at_one)
{
	ASSERT_FLOAT_EQ(
			1.0,
			prism_index_posting_pages(2.0, 1.0),
			0.0,
			"an empty posting region reports one page, not zero");
}
