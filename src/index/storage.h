/*
 * storage.h - Unified I/O abstraction for index page access
 *
 * MktStorage provides a single vtable for all page and vector I/O,
 * replacing the separate MktPageAccessor, MktVectorAccessor, and
 * MktWALWriter interfaces. WAL logging and durability are internal
 * to each implementation — callers just see read/release/write/commit.
 *
 * Standalone: ctx = array of malloc'd 8KB buffers
 * PG mode:    ctx = Relation, wraps buffer cache + GenericXLog
 * Cloud:      ctx = remote block store (future)
 *
 * Caller pattern:
 *
 *   // Search (read-only hot path)
 *   Page page = mkt_storage_read_page(s, blkno);
 *   // ... scan page entries ...
 *   mkt_storage_release_page(s, blkno);
 *
 *   // Build (write path)
 *   Page page = mkt_storage_new_page(s, &blkno);
 *   mkt_centroid_page_init(page, level);
 *   mkt_centroid_page_add(page, dim, ...);
 *   mkt_storage_commit_page(s, blkno);
 */

#ifndef MKT_STORAGE_H
#define MKT_STORAGE_H

#include "mkt_types.h"

#ifdef MKT_STANDALONE
#include "core/pg_compat.h"
#else
#include <postgres.h>

#include <storage/bufpage.h>
#include <storage/itemptr.h>
#endif

typedef struct MktStorage
{
	/* Read path */
	Page (*read_page)(void *ctx, BlockNumber blkno);
	void (*release_page)(void *ctx, BlockNumber blkno);

	/* Write path (durability/WAL is internal to implementation) */
	Page (*write_page)(void *ctx, BlockNumber blkno);
	Page (*new_page)(void *ctx, BlockNumber *blkno_out);
	void (*commit_page)(void *ctx, BlockNumber blkno);

	/* Vector fetch */
	VectorRef (*fetch_vec)(void *ctx, ItemPointerData tid);

	void *ctx;
} MktStorage;

/* ----------------------------------------------------------------
 * Inline wrappers — hide s->fn(s->ctx, ...) behind clean calls
 * ---------------------------------------------------------------- */

static inline Page
mkt_storage_read_page(const MktStorage *s, BlockNumber blkno)
{
	return s->read_page(s->ctx, blkno);
}

static inline void
mkt_storage_release_page(const MktStorage *s, BlockNumber blkno)
{
	s->release_page(s->ctx, blkno);
}

static inline Page
mkt_storage_write_page(const MktStorage *s, BlockNumber blkno)
{
	return s->write_page(s->ctx, blkno);
}

static inline Page
mkt_storage_new_page(const MktStorage *s, BlockNumber *blkno_out)
{
	return s->new_page(s->ctx, blkno_out);
}

static inline void
mkt_storage_commit_page(const MktStorage *s, BlockNumber blkno)
{
	s->commit_page(s->ctx, blkno);
}

static inline VectorRef
mkt_storage_fetch_vec(const MktStorage *s, ItemPointerData tid)
{
	return s->fetch_vec(s->ctx, tid);
}

#endif /* MKT_STORAGE_H */
