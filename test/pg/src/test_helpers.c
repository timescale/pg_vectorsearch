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

#include <fmgr.h>
#include <funcapi.h>

#include "pg/mktann_cache.h"

PG_MODULE_MAGIC;

/*
 * One row per cached RaBitQ rotation matrix in this backend's params
 * cache: (dim, refcount, usage). See mktann_rabitq_cache_stats().
 */
PG_FUNCTION_INFO_V1(mkt_test_rabitq_params_cache);

Datum
mkt_test_rabitq_params_cache(PG_FUNCTION_ARGS)
{
	ReturnSetInfo	  *rsinfo = (ReturnSetInfo *)fcinfo->resultinfo;
	MktRabitqCacheStat stats[64];
	int				   n;

	InitMaterializedSRF(fcinfo, 0);

	n = mktann_rabitq_cache_stats(stats, lengthof(stats));
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
