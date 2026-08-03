/*
 * idset.h - Minimal open-addressing set of uint64 ids
 *
 * Membership tracking for per-query scratch use: linear probing over a
 * power-of-two slot array, no deletion, no resizing. Callers size the
 * set once for the expected population; the load factor stays at or
 * below one half, keeping probe chains short.
 *
 * Id 0 doubles as the empty-slot marker, so a literal id 0 is tracked
 * in a dedicated flag instead of a slot. Posting-encoded TIDs are
 * never 0 (block 0 is the metapage), so the flag is idle on the scan
 * paths and exists for generic callers.
 *
 * Used by both standalone and PG builds.
 */
#ifndef MKT_CORE_IDSET_H
#define MKT_CORE_IDSET_H

#include <stdbool.h>
#include <stdint.h>

#include "core/log.h"
#include "core/memory.h"

/*
 * 2^64 divided by the golden ratio, rounded to odd (the splitmix64
 * increment). Multiplying by it scatters consecutive or structured
 * ids across the high bits, which the mask then folds onto the slot
 * range (Fibonacci hashing).
 */
#define MKT_HASH_GOLDEN_GAMMA 0x9E3779B97F4A7C15ULL

typedef struct MktIdSet
{
	uint64_t *slots;
	uint32_t  mask; /* nslots - 1; nslots is a power of two */
	bool	  has_zero;
} MktIdSet;

/*
 * Size for up to `expected` distinct ids at a load factor of at most
 * one half. The set neither resizes nor tracks its population:
 * inserting more than nslots distinct ids would probe forever, so the
 * caller's bound must hold. Populations needing more than the 2^31
 * slot ceiling (2^30 ids -- a 24+ GB candidate buffer upstream) are
 * rejected outright rather than proceeding undersized.
 */
static inline void
mkt_idset_init(MktIdSet *set, uint32_t expected)
{
	uint64_t want	= (uint64_t)expected * 2;
	uint32_t nslots = 2;

	if (expected > (1u << 30))
		mkt_error("id set population %u exceeds the slot ceiling", expected);

	while ((uint64_t)nslots < want)
		nslots <<= 1;
	set->slots	  = mkt_alloc0(nslots * sizeof(uint64_t));
	set->mask	  = nslots - 1;
	set->has_zero = false;
}

/*
 * Insert `id` if absent. Returns true when the id was newly added,
 * false when it was already a member.
 */
static inline bool
mkt_idset_test_add(MktIdSet *set, uint64_t id)
{
	if (id == 0)
	{
		if (set->has_zero)
			return false;
		set->has_zero = true;
		return true;
	}

	uint32_t slot = (uint32_t)(id * MKT_HASH_GOLDEN_GAMMA) & set->mask;
	while (set->slots[slot] != 0 && set->slots[slot] != id)
		slot = (slot + 1) & set->mask;
	if (set->slots[slot] == id)
		return false;
	set->slots[slot] = id;
	return true;
}

static inline void
mkt_idset_cleanup(MktIdSet *set)
{
	if (set->slots != NULL)
	{
		mkt_free(set->slots);
		set->slots = NULL;
	}
}

#endif /* MKT_CORE_IDSET_H */
