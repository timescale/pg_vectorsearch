/*
 * centroid_build.h - Generic centroid page writer
 *
 * Writes centroid entries to linked pages via MktStorage. Supports
 * all centroid formats (RaBitQ, float32, float16). Standalone-
 * compatible — all dependencies are portable.
 */

#ifndef MKT_CENTROID_BUILD_H
#define MKT_CENTROID_BUILD_H

#include "index/centroid_page.h"
#include "index/storage.h"

/*
 * Write centroid entries to linked pages.
 *
 * Creates one or more centroid pages via storage->new_page, filling
 * each to capacity before allocating the next. Pages are linked
 * via next_blkno in the opaque area.
 *
 * Parameters:
 *   level        — tree level for page init (0 = root)
 *   flags        — per-entry flags (e.g. MKT_CENTROID_FLAG_LEAF)
 *   child_count  — uniform child count for all entries
 *   data         — array of nlist pointers to entry payloads:
 *                    RABITQ → const RaBitQData *
 *                    FLOAT  → const float * (dim elements)
 *                    HALF   → const half * (dim elements)
 *   medoid_tids  — per-entry TIDs (NULL → no TID written)
 *   child_blknos — per-entry child block numbers
 *                   (NULL → InvalidBlockNumber for all entries)
 *
 * Returns the BlockNumber of the first centroid page.
 */
BlockNumber mkt_centroid_write_pages(
		MktStorage			  *storage,
		Dimension			   dim,
		uint32_t			   nlist,
		MktCentroidFormat	   fmt,
		uint8_t				   level,
		uint16_t			   flags,
		uint16_t			   child_count,
		const void			 **data,
		const ItemPointerData *medoid_tids,
		const BlockNumber	  *child_blknos);

#endif /* MKT_CENTROID_BUILD_H */
