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
 * PostgreSQL's postgres.h / utils/rel.h.
 */
typedef size_t Size;

#define BUFFERALIGN(len) MAXALIGN(len)
#define UINT64CONST(x)	 (x##ULL)

/* PG checks for query cancellation in long loops; nothing to do standalone. */
#define CHECK_FOR_INTERRUPTS() ((void)0)

/* The wait-event class is PG instrumentation; the barrier/latch shims ignore
 * the argument, so any value will do. */
#define WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN 0

/*
 * Standalone Relation: the parallel build's "heap" carries the in-memory
 * vector array it scans; its "index" carries a live page count so the few
 * RelationGetNumberOfBlocks() call sites resolve. The shared driver never
 * dereferences a Relation directly — the standalone back-end opens these out
 * of its shared state — so only RelationGetNumberOfBlocks needs the fields.
 */
typedef struct RelationData
{
	const float	   *vectors;	/* heap: vectors[nvecs * dim] */
	uint32_t		nvecs;		/* heap: vector count */
	uint32_t		dim;		/* heap: dimension */
	const uint32_t *page_count; /* index: live page count (NULL for a heap) */
} RelationData;

typedef struct RelationData *Relation;

static inline BlockNumber
RelationGetNumberOfBlocks(Relation rel)
{
	if (rel->page_count != NULL)
		return (BlockNumber)*rel->page_count;

	/*
	 * Heap: report a block count from which the build's row estimate
	 * (blocks * BLCKSZ / bytes-per-tuple) recovers the true vector count.
	 */
	uint64_t bytes_per_tuple = (uint64_t)rel->dim * sizeof(float) + 32;
	return (BlockNumber)(((uint64_t)rel->nvecs * bytes_per_tuple + BLCKSZ -
						  1) /
						 BLCKSZ);
}

/*
 * The shared build asks for an IndexInfo only to read ii_ParallelWorkers (the
 * leader) and to hand back to mkt_build_scan (ignored in standalone). A small
 * struct covers both; BuildIndexInfo is unused by the standalone scan.
 */
typedef struct IndexInfo
{
	int ii_ParallelWorkers;
} IndexInfo;

static inline IndexInfo *
BuildIndexInfo(Relation rel)
{
	(void)rel;
	return NULL;
}

/* PG per-worker instrumentation — no-ops in standalone (single process). */
typedef struct WalUsage
{
	int unused;
} WalUsage;

typedef struct BufferUsage
{
	int unused;
} BufferUsage;

#define InstrAccumParallelQuery(buf, wal) ((void)(buf), (void)(wal))

#endif /* MKT_STANDALONE */
#endif /* MKT_PG_COMPAT_H */
