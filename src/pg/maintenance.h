/*
 * maintenance.h - index maintenance entry points (incremental split)
 */

#ifndef MAINTENANCE_H
#define MAINTENANCE_H

#include <postgres.h>

#include <utils/rel.h>

/*
 * Scan an already-open index and split every posting-list head flagged for
 * split (or, when mkt.max_postinglist_size > 0, over that size). Returns the
 * number of lists split. The caller holds the index lock. Shared by the SQL
 * mkt.compact() entry point and VACUUM cleanup.
 */
int32 mktann_compact_index(Relation index);

#endif /* MAINTENANCE_H */
