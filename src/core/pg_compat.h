/*
 * pg_compat.h - Standalone definitions for PostgreSQL page types
 *
 * Provides binary-compatible definitions of BlockNumber, ItemPointerData,
 * PageHeaderData, and page functions for standalone builds. In PG builds,
 * the real PostgreSQL headers are used instead.
 *
 * This enables centroid page code to compile and test without PostgreSQL
 * while producing pages with identical binary layout.
 */

#ifndef MKT_PG_COMPAT_H
#define MKT_PG_COMPAT_H

/* Only for standalone builds — PG builds use real headers */
#ifdef MKT_STANDALONE

#include <stdint.h>
#include <string.h>

typedef uintptr_t Datum;
typedef char	 *Pointer;

#define PointerGetDatum(p) ((Datum)(p))
#define DatumGetPointer(d) ((Pointer)(d))

typedef uint32_t BlockNumber;
typedef uint16_t OffsetNumber;
typedef uint16_t LocationIndex;
typedef char	*Page;

#define InvalidBlockNumber	((BlockNumber)0xFFFFFFFF)
#define InvalidOffsetNumber ((OffsetNumber)0)
#define BLCKSZ				8192

/* Alignment macros matching PostgreSQL (8-byte on 64-bit) */
#define MAXALIGN(len) (((uintptr_t)(len) + 7) & ~(uintptr_t)7)

/* Page layout version — matches PostgreSQL 8.3+ */
#define PG_PAGE_LAYOUT_VERSION 4

/*
 * PageHeaderData — binary-compatible with PostgreSQL's PageHeaderData
 * (24 bytes). We use pd_lower/pd_upper/pd_special to track page regions.
 */
typedef struct PageHeaderData
{
	uint64_t	  pd_lsn;	   /* 8B - WAL LSN (unused standalone) */
	uint16_t	  pd_checksum; /* 2B */
	uint16_t	  pd_flags;	   /* 2B */
	LocationIndex pd_lower;	   /* 2B - end of forward-growing region */
	LocationIndex pd_upper;	   /* 2B - start of backward-growing region */
	LocationIndex pd_special;  /* 2B - start of special/opaque area */
	uint16_t	  pd_pagesize_version; /* 2B - page size | version */
	uint32_t	  pd_prune_xid;		   /* 4B - (unused standalone) */
} PageHeaderData;

typedef PageHeaderData *PageHeader;

#define SizeOfPageHeaderData (sizeof(PageHeaderData))

/*
 * PageGetContents — returns pointer to page content area.
 * For pages without line pointers, content starts after the header.
 */
static inline char *
PageGetContents(Page page)
{
	return (char *)page + MAXALIGN(SizeOfPageHeaderData);
}

/*
 * PageGetSpecialPointer — returns pointer to special (opaque) area.
 * Matches PostgreSQL's PageGetSpecialPointer macro.
 */
#define PageGetSpecialPointer(page) ((page) + ((PageHeader)(page))->pd_special)

/*
 * PageIsNew — true if page has not been initialized by PageInit.
 */
static inline bool
PageIsNew(const Page page)
{
	return ((const PageHeaderData *)page)->pd_upper == 0;
}

/*
 * PageGetPageSize — extract page size from pd_pagesize_version.
 */
static inline uint16_t
PageGetPageSize(const Page page)
{
	return ((const PageHeaderData *)page)->pd_pagesize_version &
		   (uint16_t)0xFF00;
}

/*
 * PageInit — initialize a page with standard PG header layout.
 *
 * Matches PostgreSQL's PageInit: zeroes the page, sets pd_lower to
 * the end of the header, pd_upper and pd_special to pageSize minus
 * MAXALIGN'd special size, and packs size+version.
 */
static inline void
PageInit(Page page, uint16_t pageSize, uint16_t specialSize)
{
	PageHeader header = (PageHeader)page;

	specialSize = (uint16_t)MAXALIGN(specialSize);

	memset(page, 0, pageSize);

	header->pd_lower			= SizeOfPageHeaderData;
	header->pd_upper			= pageSize - specialSize;
	header->pd_special			= pageSize - specialSize;
	header->pd_pagesize_version = (pageSize & 0xFF00) | PG_PAGE_LAYOUT_VERSION;
}

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

/*
 * Types/macros the shared parallel build (parallel_build.h) expects from
 * PostgreSQL's postgres.h / utils/rel.h. Relation is opaque here — the
 * standalone build never dereferences it (it passes its vectors directly).
 */
typedef size_t				 Size;
typedef struct RelationData *Relation;

#define BUFFERALIGN(len) MAXALIGN(len)
#define UINT64CONST(x)	 (x##ULL)

#endif /* MKT_STANDALONE */
#endif /* MKT_PG_COMPAT_H */
