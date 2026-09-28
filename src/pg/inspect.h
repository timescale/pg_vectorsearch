/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * inspect.h - shared centroid-tree walk
 *
 * The read-only inspection functions (inspect.c) and the mutating
 * maintenance functions (maintenance.c) both need to enumerate the leaf
 * entries of the centroid tree. The walk itself is read-only; it lives with
 * the inspection code and is declared here so the maintenance code can reuse
 * it without duplicating the traversal.
 */

#ifndef INSPECT_H
#define INSPECT_H

#include <postgres.h>

#include <utils/rel.h>

#include "core/types.h"

/* One leaf centroid entry with its location in the centroid tree. */
typedef struct LeafEntry
{
	BlockNumber posting_head;
	BlockNumber centroid_page;
	uint16_t	entry_idx;
} LeafEntry;

/*
 * BFS the centroid tree and collect all leaf entries into a palloc'd array
 * (*out); returns the count. The caller pfrees the array. Read-only; defined
 * in inspect.c.
 */
int collect_leaf_entries(
		Relation	index,
		BlockNumber first_centroid,
		uint8_t		nlevels,
		Dimension	dim,
		LeafEntry **out);

#endif /* INSPECT_H */
