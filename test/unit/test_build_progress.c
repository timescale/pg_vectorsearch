/*
 * test_build_progress.c - the canonical index-build phase name table
 *
 * Guards that every build phase has a human-readable name (a phase added
 * without one would show blank in pg_stat_progress_create_index), that
 * out-of-range values are rejected, and that the strings the build_progress
 * isolation test depends on stay frozen.
 */

#include <stdbool.h>
#include <string.h>

#include "index/build_progress.h"
#include "mkt_test.h"

TEST_GROUP(BuildProgress);

/* Every value in [INITIALIZE, MAX] maps to a non-empty name. */
TEST(every_phase_has_a_name)
{
	for (int p = MKT_BUILD_PHASE_INITIALIZE; p <= MKT_BUILD_PHASE_MAX; p++)
	{
		const char *name = mkt_build_phase_name(p);
		ASSERT_NOT_NULL(name, "every phase value has a name");
		ASSERT_TRUE(name[0] != '\0', "phase name is non-empty");
	}
}

/* Out-of-range values return NULL rather than crashing / reading garbage. */
TEST(out_of_range_is_null)
{
	ASSERT_NULL(mkt_build_phase_name(0), "value 0 has no name");
	ASSERT_NULL(mkt_build_phase_name(-1), "negative value has no name");
	ASSERT_NULL(
			mkt_build_phase_name(MKT_BUILD_PHASE_MAX + 1),
			"value above max has no name");
}

/* The strings the build_progress isolation test asserts must not drift. */
TEST(frozen_phase_strings)
{
	ASSERT_TRUE(
			strcmp(mkt_build_phase_name(MKT_BUILD_PHASE_SCAN),
				   "scanning table") == 0,
			"SCAN string is frozen");
	ASSERT_TRUE(
			strcmp(mkt_build_phase_name(MKT_BUILD_PHASE_SCAN_PARALLEL),
				   "scanning table (parallel)") == 0,
			"SCAN_PARALLEL string is frozen");
}
