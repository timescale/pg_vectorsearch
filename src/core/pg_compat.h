/*
 * pg_compat.h - Standalone definitions for PostgreSQL page types
 *
 * Provides binary-compatible definitions of BlockNumber, ItemPointerData,
 * Page, and page constants for standalone builds. In PG builds, the real
 * PostgreSQL headers are used instead.
 *
 * This enables centroid page code to compile and test without PostgreSQL
 * while producing pages with identical binary layout.
 */

#ifndef MKT_PG_COMPAT_H
#define MKT_PG_COMPAT_H

/* Only for standalone builds — PG builds use real headers */
#ifdef MKT_STANDALONE

#include <stdint.h>

typedef uint32_t BlockNumber;
typedef uint16_t OffsetNumber;
typedef char	*Page;

#define InvalidBlockNumber	((BlockNumber)0xFFFFFFFF)
#define InvalidOffsetNumber ((OffsetNumber)0)
#define BLCKSZ				8192

/*
 * Simplified page header — same size as PostgreSQL's PageHeaderData
 * (24 bytes). We only use it as reserved space; the SoA data starts
 * at offset 24 in both modes.
 */
#define SizeOfPageHeaderData 24

/*
 * ItemPointerData — 6-byte heap TID, binary-compatible with PG.
 * BlockIdData is two uint16 (high/low halves of BlockNumber).
 */
typedef struct BlockIdData
{
	uint16_t bi_hi;
	uint16_t bi_lo;
} BlockIdData;

typedef struct ItemPointerData
{
	BlockIdData	 ip_blkid;
	OffsetNumber ip_posid;
} __attribute__((packed, aligned(2))) ItemPointerData;

typedef ItemPointerData *ItemPointer;

static inline BlockNumber
ItemPointerGetBlockNumber(const ItemPointerData *tip)
{
	return ((BlockNumber)tip->ip_blkid.bi_hi << 16) |
		   (BlockNumber)tip->ip_blkid.bi_lo;
}

static inline OffsetNumber
ItemPointerGetOffsetNumber(const ItemPointerData *tip)
{
	return tip->ip_posid;
}

static inline void
ItemPointerSet(ItemPointerData *tip, BlockNumber blk, OffsetNumber off)
{
	tip->ip_blkid.bi_hi = (uint16_t)(blk >> 16);
	tip->ip_blkid.bi_lo = (uint16_t)blk;
	tip->ip_posid		= off;
}

static inline void
ItemPointerSetInvalid(ItemPointerData *tip)
{
	tip->ip_blkid.bi_hi = 0xFFFF;
	tip->ip_blkid.bi_lo = 0xFFFF;
	tip->ip_posid		= InvalidOffsetNumber;
}

#endif /* MKT_STANDALONE */
#endif /* MKT_PG_COMPAT_H */
