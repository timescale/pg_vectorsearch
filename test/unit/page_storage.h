/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * page_storage.h - Array-backed VsStorage for unit tests
 *
 * What lets the paged code be exercised without PostgreSQL: pages are plain
 * memory, and commit/release are no-ops because nothing here is logged or
 * pinned. Every test that reads or writes a page needs it, so it lives here
 * instead of being copied per file.
 */

#ifndef VS_TEST_PAGE_STORAGE_H
#define VS_TEST_PAGE_STORAGE_H

#include "core/memory.h"
#include "index/storage.h"
#include "standalone/pg_compat.h"

typedef struct TestPageStorage
{
	VsStorage base;
	char	 *pages;
	uint32_t  next_blkno;
	uint32_t  page_cap;
} TestPageStorage;

static inline Page
test_read_page(VsStorage *self, BlockNumber blkno)
{
	TestPageStorage *s = (TestPageStorage *)self;
	return s->pages + (size_t)blkno * BLCKSZ;
}

static inline void
test_release_page(VsStorage *self, BlockNumber blkno)
{
	(void)self;
	(void)blkno;
}

static inline Page
test_write_page(VsStorage *self, BlockNumber blkno)
{
	TestPageStorage *s = (TestPageStorage *)self;
	return s->pages + (size_t)blkno * BLCKSZ;
}

static inline Page
test_new_page(VsStorage *self, BlockNumber *blkno_out)
{
	TestPageStorage *s = (TestPageStorage *)self;
	*blkno_out		   = s->next_blkno++;
	return s->pages + (size_t)*blkno_out * BLCKSZ;
}

static inline void
test_commit_page(VsStorage *self, BlockNumber blkno)
{
	(void)self;
	(void)blkno;
}

static const VsStorageOps test_storage_ops = {
		.read_page	  = test_read_page,
		.release_page = test_release_page,
		.write_page	  = test_write_page,
		.new_page	  = test_new_page,
		.commit_page  = test_commit_page,
};

/* In-place init, for tests that hand out the first block at something other
 * than 0 (a metapage, say). */
static inline void
test_storage_init(TestPageStorage *s, uint32_t page_cap, uint32_t next_blkno)
{
	*s = (TestPageStorage){
			.base		= {.ops = &test_storage_ops},
			.pages		= vs_alloc0((size_t)page_cap * BLCKSZ),
			.next_blkno = next_blkno,
			.page_cap	= page_cap,
	};
}

static inline TestPageStorage
make_test_storage(uint32_t num_pages)
{
	TestPageStorage s;
	test_storage_init(&s, num_pages, 0);
	return s;
}

#endif /* VS_TEST_PAGE_STORAGE_H */
