/*
 * test_helpers.c - test-only SQL entry points
 *
 * Built as its own always-built module, never part of the extension:
 * regression tests that need extra introspection CREATE their SQL
 * functions against this library explicitly and drop them afterward,
 * so the extension itself ships no test surface.
 *
 * The extension's own library must already be loaded in the backend
 * before this module is (any call of an extension C function does
 * that): these functions resolve symbols the extension library
 * exports, and PostgreSQL loads modules with RTLD_NOW.
 */

#include <postgres.h>

#include <access/relation.h>
#include <commands/defrem.h>
#include <fmgr.h>
#include <funcapi.h>

#include "index/centroid_page.h"
#include "pg/amcache.h"
#include "pg/bufstorage.h"

PG_MODULE_MAGIC;

/*
 * One row per cached RaBitQ rotation matrix in this backend's params
 * cache: (dim, refcount, usage). See prism_rabitq_cache_stats().
 */
PG_FUNCTION_INFO_V1(vs_test_rabitq_params_cache);

Datum
vs_test_rabitq_params_cache(PG_FUNCTION_ARGS)
{
	ReturnSetInfo		*rsinfo = (ReturnSetInfo *)fcinfo->resultinfo;
	PrismRabitqCacheStat stats[64];
	int					 n;

	InitMaterializedSRF(fcinfo, 0);

	n = prism_rabitq_cache_stats(stats, lengthof(stats));
	for (int i = 0; i < n; i++)
	{
		Datum values[3];
		bool  nulls[3] = {false, false, false};

		values[0] = Int32GetDatum(stats[i].dim);
		values[1] = Int32GetDatum(stats[i].refcount);
		values[2] = Float8GetDatum(stats[i].usage);
		tuplestore_putvalues(
				rsinfo->setResult, rsinfo->setDesc, values, nulls);
	}

	PG_RETURN_VOID();
}

/*
 * Reset the backend's params cache to its initial state; returns the
 * number of entries dropped. Errors if any entry is checked out. Lets
 * a test section start from an empty cache without reconnecting.
 */
PG_FUNCTION_INFO_V1(vs_test_rabitq_cache_clear);

Datum
vs_test_rabitq_cache_clear(PG_FUNCTION_ARGS)
{
	PG_RETURN_INT32(prism_rabitq_cache_clear());
}

/*
 * Append an empty, level-0 centroid page to the end of the given index
 * relation, mirroring the on-disk shape a posting-list split leaves when it
 * has to grow the centroid tree but finds no room on the level-0 page: it
 * extends the relation for the new centroid page, landing it past every
 * existing posting page (see posting_split.c's `!appended` branch). Lets a
 * regression test exercise that shape -- and what reads the index afterward
 * -- directly, without forcing enough real splits to overflow a page for
 * real. Returns the new page's block number.
 */
PG_FUNCTION_INFO_V1(vs_test_append_centroid_page);

Datum
vs_test_append_centroid_page(PG_FUNCTION_ARGS)
{
	Oid		 indexoid = PG_GETARG_OID(0);
	Relation index	  = relation_open(indexoid, ShareUpdateExclusiveLock);

	/* This extends whatever relation it's handed -- passing a heap table by
	 * mistake would silently append a garbage page to it instead of erroring
	 * out, since neither relation_open nor vs_pg_storage_init care what kind
	 * of relation they're given. */
	if (index->rd_rel->relam != get_am_oid("prism", false))
	{
		relation_close(index, ShareUpdateExclusiveLock);
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("\"%s\" is not a prism index",
						RelationGetRelationName(index))));
	}

	VsPgStorage storage;
	vs_pg_storage_init(&storage, index, NULL, DISTANCE_L2);

	BlockNumber blkno;
	Page		page = vs_storage_new_page(&storage.base, &blkno);
	prism_centroid_page_init_fmt(page, 0, PRISM_CENTROID_FMT_RABITQ);
	vs_storage_commit_page(&storage.base, blkno);

	relation_close(index, ShareUpdateExclusiveLock);

	PG_RETURN_INT32((int32)blkno);
}
