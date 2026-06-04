/*
 * parallel_backend.c - PostgreSQL implementations of the build's back-end
 * seams.
 *
 * The build driver is shared between the PG extension and the standalone
 * engine; the parts that genuinely differ by back-end are expressed as
 * same-named seam functions, with the PG versions here and the thread-based
 * versions under src/standalone. This file currently holds the heap-scan seam;
 * the DSM/launch/teardown seams join it as the driver moves to shared code.
 */

#include <postgres.h>

#include <access/tableam.h>
#include <catalog/index.h>
#include <utils/rel.h>

#include "mkt_pg.h"
#include "mkt_vector.h"
#include "parallel_build.h"

/*
 * Bridges PostgreSQL's heap-tuple callback to the back-end-neutral scan
 * callback: skip nulls and hand the vector's data pointer (into the varlena,
 * no copy) plus its tid to the shared logic.
 */
typedef struct MktPgScanAdapter
{
	MktBuildScanCb cb;
	void		  *state;
} MktPgScanAdapter;

static void
mkt_pg_scan_adapter(
		Relation	index,
		ItemPointer tid,
		Datum	   *values,
		bool	   *isnull,
		bool		tuple_is_alive,
		void	   *adapter_state)
{
	MktPgScanAdapter *a = (MktPgScanAdapter *)adapter_state;

	(void)index;
	(void)tuple_is_alive;

	if (isnull[0])
		return;

	a->cb(a->state, *tid, MKT_VECTOR_DATA(DatumGetMktVector(values[0])));
}

/*
 * Scan every vector via the shared parallel table scan, invoking cb per live
 * tuple. The standalone back-end provides a same-named function that iterates
 * its in-memory vector array (work-stealing) instead. allow_sync/anyvisible
 * are PostgreSQL table_index_build_scan flags, ignored in standalone.
 */
void
mkt_build_scan(
		Relation		  heap,
		Relation		  index,
		struct IndexInfo *indexInfo,
		MktBuildShared	 *shared,
		bool			  allow_sync,
		bool			  anyvisible,
		MktBuildScanCb	  cb,
		void			 *state)
{
	MktPgScanAdapter actx = {.cb = cb, .state = state};
	TableScanDesc	 scan = table_beginscan_parallel(
			   heap, ParallelTableScanFromMktShared(shared));

	table_index_build_scan(
			heap,
			index,
			indexInfo,
			allow_sync,
			anyvisible,
			mkt_pg_scan_adapter,
			&actx,
			scan);
}
