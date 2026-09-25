/*
 * iam_handler.c - prism index access method handler
 *
 * Registers the prism index access method with PostgreSQL. Build and
 * scan callbacks delegate to build.c and scan.c; trivial
 * stubs for unimplemented callbacks remain here.
 */

#include <postgres.h>

#include <access/amapi.h>
#include <access/reloptions.h>
#include <access/relscan.h>
#include <commands/vacuum.h>
#include <fmgr.h>
#include <storage/bufmgr.h>
#include <storage/lmgr.h>
#include <utils/float.h>
#include <utils/injection_point.h>
#include <utils/memutils.h>
#include <utils/selfuncs.h>

#include "algo/vecops.h"
#include "amcache.h"
#include "build.h"
#include "cost.h"
#include "index/index_base.h"
#include "index/posting_insert.h"
#include "index/query_scan.h"
#include "pg/bufstorage.h"
#include "quant/rabitq.h"
#include "scan.h"
#include "support_pg.h"
#include "typeinfo.h"
#include "types/vec32.h"

PG_FUNCTION_INFO_V1(prism_handler);

/* ----------------------------------------------------------------
 * Trivial stubs (no separate file needed)
 * ---------------------------------------------------------------- */

/*
 * ambuildempty populates the init fork of an unlogged index so that crash
 * recovery has a valid image to copy over the main fork. prism has no
 * notion of a valid "empty" index image -- even a build over zero rows
 * produces a real single-cluster tree -- so a no-op here would leave the
 * init fork at zero blocks and let recovery wipe the whole index, metadata
 * page included. Rather than seed the init fork with an index shape nothing
 * else ever exercises, refuse unlogged tables outright at CREATE INDEX time.
 */
static void
prism_buildempty(Relation index)
{
	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("prism indexes do not support unlogged tables"),
			 errhint("Use a logged table, or run ALTER TABLE ... SET LOGGED "
					 "before creating the index.")));
}

/*
 * Beam width used when routing an inserted vector to its nearest leaf. Wide
 * enough that multi-level tree descent lands the true nearest leaf; we still
 * insert into a single list (results[0]). Phase 0 does no SOAR replication on
 * insert — that's restored in bulk at rebuild / compaction.
 */
#define PRISM_INSERT_ROUTE_BEAM 8

/*
 * How many times an insert re-routes when the head it locked turns out to have
 * been split away. A split retires the head it replaces, so an insert that was
 * waiting on the head's lock has to route again -- bounded, so churn cannot
 * spin forever, and reaching the bound fails the insert rather than dropping
 * the tuple.
 */
#define PRISM_INSERT_ROUTE_ATTEMPTS 8

static bool
prism_insert(
		Relation		  index,
		Datum			 *values,
		bool			 *isnull,
		ItemPointer		  heap_tid,
		Relation		  heap,
		IndexUniqueCheck  check_unique,
		bool			  index_unchanged,
		struct IndexInfo *index_info)
{
	(void)heap;
	(void)check_unique;
	(void)index_info;

	/* index_unchanged is deliberately ignored. PostgreSQL sets it when an
	 * UPDATE left the indexed column untouched but still had to place the
	 * new tuple version elsewhere (a non-HOT update); it is a hint for
	 * access methods that can deduplicate against the old version, not a
	 * signal that the old entry still covers the new TID. A HOT update,
	 * where the old entry does still apply, never reaches aminsert at all.
	 * Skipping the insert here left the new version unreachable through
	 * the index. NULL vectors get no entry. */
	if (isnull[0])
		return false;

	/* Per-insert scratch context: beam-search + encode allocations are freed
	 * in one shot and don't accumulate in the inserting transaction. */
	MemoryContext insert_ctx = AllocSetContextCreate(
			CurrentMemoryContext, "prism insert", ALLOCSET_DEFAULT_SIZES);
	MemoryContext old_ctx = MemoryContextSwitchTo(insert_ctx);

	/* Immutable index parameters from the per-backend cache (metapage read at
	 * most once per backend, not once per insert). The checkout is
	 * registered with the current resource owner; capture it for the
	 * release below (and for the error paths in between, which release
	 * through the owner instead). */
	ResourceOwner  params_owner = CurrentResourceOwner;
	PrismIndexBase base;
	prism_index_base_init(index, &base);
	Dimension dim = base.dim;

	VsPgStorage storage;
	vs_pg_storage_init(&storage, index, NULL, base.metric);
	base.centroid_storage = &storage.base;
	base.posting_storage  = &storage.base;
	base.page_base		  = NULL;

	/* Inserted vector, converted to float32 when the column type is not. */
	Vec32Access input = vec32_access(
			prism_cache_type_info(index), dim, CurrentMemoryContext);
	Vec32Ref vref = vec32_read(&input, values[0]);

	if (vref.dim != dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("inserted vector dimension %u does not match index "
						"dimension %u",
						vref.dim,
						dim)));

	/* No defined distance under cosine: encoded as unreachable below.
	 * Squared norm -- zero iff the norm is zero, without the sqrt. */
	bool degenerate = base.metric == DISTANCE_COSINE &&
					  vs_l2_norm_squared(vref.data, dim) == 0.0f;

	/*
	 * Route to a leaf the same way a query does (prism_query_route normalizes
	 * for cosine, rotates into qs.pt_query, and runs the beam search;
	 * qs.pt_query is then exactly the rotated residual the encode needs), lock
	 * its head, and append. If the head was split away while we waited for the
	 * lock it is now tombstoned; re-route to the new head and retry. Bounded
	 * so a pathological churn can't spin forever -- and if the bound is
	 * reached the insert fails rather than returning as though it had indexed
	 * the tuple.
	 *
	 * The query state is initialized once and reused across attempts -- each
	 * prism_query_route re-runs the search from scratch on it -- so a retry
	 * does not re-allocate its beam buffers.
	 */
	RaBitQScratch	enc;
	bool			enc_init = false;
	PrismQueryState qs;
	prism_query_state_init(&qs, &base, 1, PRISM_INSERT_ROUTE_BEAM);
	bool inserted = false;
	bool routed	  = true;

	for (int attempt = 0; attempt < PRISM_INSERT_ROUTE_ATTEMPTS; attempt++)
	{
		uint32_t n = prism_query_route(
				&qs,
				vref.data,
				PRISM_INSERT_ROUTE_BEAM,
				VS_DISTANCE_MODE_ASYMMETRIC,
				NULL);

		BlockNumber head = (n > 0) ? qs.beam_results[0].posting_head
								   : InvalidBlockNumber;
		if (head == InvalidBlockNumber)
		{
			routed = false;
			break;
		}

		/*
		 * Serialize concurrent inserts into this cluster with a heavyweight
		 * page lock on the head — distinct from the buffer content locks the
		 * insert primitive takes per page, so it doesn't fight the
		 * single-buffer storage model. Released here, not held to xact end.
		 */
		LockPage(index, head, ExclusiveLock);

		if (!enc_init)
		{
			vs_rabitq_scratch_init(&enc, dim);
			enc_init = true;
		}
		/*
		 * Test hook: fires while this insert holds the per-cluster page lock,
		 * so an isolation test can pause here and observe a second insert into
		 * the same cluster block on the lock (proving the serialization and
		 * that it does not deadlock). No-op unless PG was built with injection
		 * points and a test attached an action.
		 */
		INJECTION_POINT("prism-insert-locked", NULL);
		/*
		 * If the head was split away while we waited for the lock it is now
		 * retired (TOMBSTONED immediate or DELETED XID-gated); insert_one
		 * reports that from the head read it does anyway, and we re-route.
		 */
		bool head_retired = false;

		/*
		 * Test hook: stands in for finding the head retired, so a test can
		 * drive the retry budget to its end. It replaces the insert rather
		 * than following it -- an insert that had already written the entry
		 * would write it again on every retry. Compiles to a constant false
		 * unless PostgreSQL was built with injection points, and is only
		 * true while a test holds the point attached.
		 */
		if (IS_INJECTION_POINT_ATTACHED("prism-insert-force-reroute"))
		{
			INJECTION_POINT("prism-insert-force-reroute", NULL);
			head_retired = true;
		}
		else
			prism_posting_insert_one(
					&storage.base,
					base.params,
					dim,
					head,
					*heap_tid,
					qs.pt_query,
					&enc,
					degenerate,
					&head_retired);
		UnlockPage(index, head, ExclusiveLock);
		if (head_retired)
			continue; /* head was split; re-route */
		inserted = true;
		break;
	}
	prism_query_state_cleanup(&qs);

	MemoryContextSwitchTo(old_ctx);
	MemoryContextDelete(insert_ctx);

	/* Check the RaBitQParams checkout back in — see prism_index_base_init
	 * above. */
	prism_release_params(dim, base.rabitq_seed, params_owner);

	/*
	 * Every path out of the loop above must have indexed the tuple. Returning
	 * normally without having done so would leave a committed row that no scan
	 * of this index can ever find, with nothing to say it happened -- so fail
	 * the insert and let the transaction that owns the row decide.
	 */
	if (!routed)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("no leaf partition found for an insert into index "
						"\"%s\"",
						RelationGetRelationName(index))));
	if (!inserted)
		ereport(ERROR,
				(errcode(ERRCODE_T_R_SERIALIZATION_FAILURE),
				 errmsg("insert into index \"%s\" gave way to concurrent "
						"maintenance %d times",
						RelationGetRelationName(index),
						PRISM_INSERT_ROUTE_ATTEMPTS),
				 errhint("Retry the transaction.")));

	/* bool result is only meaningful for unique indexes. */
	return false;
}

/*
 * Adapt PostgreSQL's IndexBulkDeleteCallback (takes ItemPointer) to the shared
 * tombstone predicate (takes ItemPointerData by value).
 */
typedef struct PrismBulkDeleteCtx
{
	IndexBulkDeleteCallback cb;
	void				   *cb_state;
} PrismBulkDeleteCtx;

static bool
tid_is_dead(ItemPointerData tid, void *state)
{
	PrismBulkDeleteCtx *c = (PrismBulkDeleteCtx *)state;
	return c->cb(&tid, c->cb_state);
}

/*
 * VACUUM's dead-tuple removal. Block-scans the index and tombstones each
 * posting chain from its FIRST (head) page via prism_posting_tombstone_chain —
 * the head walk covers the chain's overflow pages, so only heads are acted on.
 * Tombstoned entries are skipped by later scans; physical reclaim happens at a
 * later compaction/rebuild. This is the cleanup path for both explicit DELETEs
 * and the dead old-version of every vector-column UPDATE.
 */
static IndexBulkDeleteResult *
prism_bulkdelete(
		IndexVacuumInfo		   *info,
		IndexBulkDeleteResult  *stats,
		IndexBulkDeleteCallback callback,
		void				   *cb_state)
{
	if (stats == NULL)
		stats = palloc0(sizeof(IndexBulkDeleteResult));

	Relation	index	= info->index;
	BlockNumber nblocks = RelationGetNumberOfBlocks(index);
	if (nblocks <= 1)
		return stats; /* block 0 is the metadata page */

	/* dim + metric + first_posting from the per-backend cache (metapage read
	 * at most once per backend). prism_cache_meta skips the rotation-matrix
	 * work the scan / insert cache path does — VACUUM never needs it. */
	Dimension	   dim;
	DistanceMetric metric;
	BlockNumber	   first_posting;
	prism_cache_meta(index, &dim, &metric, &first_posting);

	VsPgStorage storage;
	vs_pg_storage_init(&storage, index, NULL, metric);

	PrismBulkDeleteCtx ctx = {.cb = callback, .cb_state = cb_state};

	/*
	 * The index is laid out as: block 0 metadata, then the contiguous
	 * centroid region [1, first_posting), then the posting pages.
	 * first_posting is fixed at build time and the relation only grows from
	 * there, so nothing below it is ever a posting page or a split-appended
	 * centroid page -- start there, as before, to skip that region without
	 * scanning it.
	 *
	 * A split that needs to grow the centroid tree but finds no room on the
	 * level-0 page appends a new centroid page by extending the relation
	 * (see posting_split.c), which lands past every existing posting page --
	 * i.e. still >= first_posting, just no longer separable from the
	 * posting region by block number alone. Classify each page in the
	 * scanned range by its own page_id instead of relying on position: a
	 * posting page is acted on, a centroid page and an empty
	 * (not-yet-initialized) page are both expected and skipped, and
	 * anything else is the only real anomaly.
	 *
	 * Within the posting region we act only on chain heads; overflow pages
	 * are reached via the chain from their head. A page that is none of the
	 * above is the only anomaly worth surfacing (corruption or a format
	 * bug); count those and emit a single WARNING after the walk rather
	 * than one per page, so a badly corrupt index can't flood the log
	 * (bulkdelete also runs once per dead-tuple batch, i.e. potentially
	 * many times per VACUUM).
	 */
	uint32_t	unrecognized = 0;
	BlockNumber first_bad	 = InvalidBlockNumber;
	BlockNumber start		 = Max(first_posting, 1);

	for (BlockNumber blk = start; blk < nblocks; blk++)
	{
		vacuum_delay_point(false);

		Buffer buf = ReadBuffer(index, blk);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		Page page		= BufferGetPage(buf);
		bool is_head	= false;
		bool recognized = PageIsNew(page); /* an empty page is expected */

		if (!recognized && prism_page_is_posting(page))
		{
			PrismPostingPageOpaque *op = prism_posting_opaque(page);
			recognized				   = true;
			/* Skip a retired (DELETED) chain: it is superseded by a split,
			 * its live_count slot now holds delete_xid, and cleanup
			 * reclaims it once safe. */
			is_head = (op->flags & PRISM_POSTING_PAGE_FIRST) != 0 &&
					  !(op->flags & PRISM_POSTING_PAGE_DELETED);
		}
		/* Centroid pages carry no heap TIDs — there is nothing for
		 * bulkdelete to do with one, only to not mistake it for corruption. */
		if (!recognized && prism_page_is_centroid(page))
			recognized = true;
		UnlockReleaseBuffer(buf);

		if (!recognized)
		{
			if (first_bad == InvalidBlockNumber)
				first_bad = blk;
			unrecognized++;
		}

		if (!is_head)
			continue;

		/*
		 * Mutating a cluster's chain requires the head's page lock -- the same
		 * lock inserts take, and the one a split holds while it rewrites the
		 * cluster. The check above ran under a buffer lock that has since been
		 * released, so without this a split could retire the chain in the gap:
		 * the head's live_count slot then holds delete_xid, and the tombstone
		 * pass would decrement that instead. A shrunken delete_xid reads as
		 * older than it is, which brings the chain's physical reclaim forward
		 * past the scans it was being kept alive for.
		 */
		LockPage(index, blk, ExclusiveLock);

		/* Re-read under the lock: a live head at this point stays live. */
		Page hp = vs_storage_read_page(&storage.base, blk);
		const PrismPostingPageOpaque *hop = prism_posting_opaque(hp);
		bool still_head = (hop->flags & PRISM_POSTING_PAGE_FIRST) != 0 &&
						  !(hop->flags & PRISM_POSTING_PAGE_DELETED) &&
						  !(hop->flags & PRISM_POSTING_PAGE_TOMBSTONED);
		vs_storage_release_page(&storage.base, blk);

		if (still_head)
		{
			stats->tuples_removed += prism_posting_tombstone_chain(
					&storage.base, dim, blk, tid_is_dead, &ctx);

			/* Live tuples remaining: the head's maintained live_count, which
			 * the tombstone pass just decremented (O(1), no rescan). */
			Page lp = vs_storage_read_page(&storage.base, blk);
			stats->num_index_tuples += prism_posting_head_live_count(lp);
			vs_storage_release_page(&storage.base, blk);
		}

		UnlockPage(index, blk, ExclusiveLock);
	}

	/* One summary line per call, not one per bad page (see loop comment). */
	if (unrecognized > 0)
		ereport(WARNING,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("skipped %u unrecognized page(s) in index \"%s\" "
						"during bulkdelete (first at block %u); index may be "
						"corrupt",
						unrecognized,
						RelationGetRelationName(index),
						first_bad)));

	return stats;
}

/*
 * The tuple count reported here becomes the index's pg_class.reltuples,
 * which the planner reads as indexinfo->tuples. Two things keep it honest.
 *
 * With no bulk delete (a VACUUM that found nothing to remove, or ANALYZE),
 * return NULL so the existing statistics stand. Returning a zeroed result
 * instead would set reltuples to 0 on every such VACUUM -- which is what
 * happened -- leaving the planner believing an index over millions of rows
 * is empty until the next ANALYZE.
 *
 * After a bulk delete, the count bulkdelete accumulated is the number of
 * live posting entries, and that is not the number of indexed heap tuples:
 * SOAR and boundary replication store a vector in more than one list, so
 * the entry count overstates the row count by the replication factor.
 * Report the heap's own live count instead, as nbtree does, and carry
 * estimated_count through so an estimate does not overwrite an exact
 * figure (vacuumlazy skips the update when it is set).
 */
static IndexBulkDeleteResult *
prism_vacuumcleanup(IndexVacuumInfo *info, IndexBulkDeleteResult *stats)
{
	if (info->analyze_only)
		return stats;
	if (stats == NULL)
		return NULL;

	stats->num_pages		= RelationGetNumberOfBlocks(info->index);
	stats->num_index_tuples = info->num_heap_tuples;
	stats->estimated_count	= info->estimated_count;
	return stats;
}

static void
prism_costestimate(
		PlannerInfo *root,
		IndexPath	*path,
		double		 loop_count,
		Cost		*startup_cost,
		Cost		*total_cost,
		Selectivity *selectivity,
		double		*correlation,
		double		*index_pages)
{
	/* Never use the index without ORDER BY <op> */
	if (path->indexorderbys == NIL)
	{
		*startup_cost			  = get_float8_infinity();
		*total_cost				  = get_float8_infinity();
		*selectivity			  = 0;
		*correlation			  = 0;
		*index_pages			  = 0;
		path->path.disabled_nodes = 2;
		return;
	}

	prism_cost_estimate(
			root,
			path,
			loop_count,
			startup_cost,
			total_cost,
			selectivity,
			correlation,
			index_pages);
}

static bytea *
prism_options(Datum reloptions, bool validate)
{
	static const relopt_parse_elt tab[] = {
			{"distance_mode",
			 RELOPT_TYPE_ENUM,
			 offsetof(PrismOptions, distance_mode)},
			{"fan_out", RELOPT_TYPE_INT, offsetof(PrismOptions, fan_out)},
			{"nlist", RELOPT_TYPE_INT, offsetof(PrismOptions, nlist)},
			{"kmeans_nredo",
			 RELOPT_TYPE_INT,
			 offsetof(PrismOptions, kmeans_nredo)},
			{"soar_lambda",
			 RELOPT_TYPE_REAL,
			 offsetof(PrismOptions, soar_lambda)},
			{"boundary_epsilon",
			 RELOPT_TYPE_REAL,
			 offsetof(PrismOptions, boundary_epsilon)},
			{"centroid_compression",
			 RELOPT_TYPE_ENUM,
			 offsetof(PrismOptions, centroid_compression)},
			{"fastscan", RELOPT_TYPE_ENUM, offsetof(PrismOptions, fastscan)},
			{"centroid_fastscan",
			 RELOPT_TYPE_ENUM,
			 offsetof(PrismOptions, centroid_fastscan)},
	};
	return (bytea *)build_reloptions(
			reloptions,
			validate,
			prism_relopt_kind,
			sizeof(PrismOptions),
			tab,
			lengthof(tab));
}

static bool
prism_validate(Oid opclassoid)
{
	return true;
}

/* ----------------------------------------------------------------
 * Handler
 * ---------------------------------------------------------------- */

Datum
prism_handler(PG_FUNCTION_ARGS)
{
	IndexAmRoutine *amroutine = makeNode(IndexAmRoutine);

	/* Properties */
	amroutine->amstrategies			   = 0;
	amroutine->amsupport			   = 3;
	amroutine->amoptsprocnum		   = 0;
	amroutine->amcanorder			   = false;
	amroutine->amcanorderbyop		   = true;
	amroutine->amcanhash			   = false;
	amroutine->amconsistentequality	   = false;
	amroutine->amconsistentordering	   = false;
	amroutine->amcanbackward		   = false;
	amroutine->amcanunique			   = false;
	amroutine->amcanmulticol		   = false;
	amroutine->amoptionalkey		   = true;
	amroutine->amsearcharray		   = false;
	amroutine->amsearchnulls		   = false;
	amroutine->amstorage			   = false;
	amroutine->amclusterable		   = false;
	amroutine->ampredlocks			   = false;
	amroutine->amcanparallel		   = false;
	amroutine->amcanbuildparallel	   = true;
	amroutine->amcaninclude			   = false;
	amroutine->amusemaintenanceworkmem = false;
	amroutine->amsummarizing		   = false;
	amroutine->amparallelvacuumoptions = VACUUM_OPTION_PARALLEL_BULKDEL;
	amroutine->amkeytype			   = InvalidOid;

	/* Build callbacks */
	amroutine->ambuild			= prism_build;
	amroutine->ambuildempty		= prism_buildempty;
	amroutine->ambuildphasename = prism_buildphasename;

	/* Insert / maintenance */
	amroutine->aminsert		   = prism_insert;
	amroutine->aminsertcleanup = NULL;
	amroutine->ambulkdelete	   = prism_bulkdelete;
	amroutine->amvacuumcleanup = prism_vacuumcleanup;

	/* Cost estimation / validation */
	amroutine->amcanreturn	   = NULL;
	amroutine->amcostestimate  = prism_costestimate;
	amroutine->amgettreeheight = NULL;
	amroutine->amoptions	   = prism_options;
	amroutine->amproperty	   = NULL;
	amroutine->amvalidate	   = prism_validate;
	amroutine->amadjustmembers = NULL;

	/* Scan callbacks */
	amroutine->ambeginscan = prism_beginscan;
	amroutine->amrescan	   = prism_rescan;
	amroutine->amgettuple  = prism_gettuple;
	amroutine->amgetbitmap = NULL;
	amroutine->amendscan   = prism_endscan;
	amroutine->ammarkpos   = NULL;
	amroutine->amrestrpos  = NULL;

	/* Parallel scan (not supported) */
	amroutine->amestimateparallelscan = NULL;
	amroutine->aminitparallelscan	  = NULL;
	amroutine->amparallelrescan		  = NULL;

	/* Planning */
	amroutine->amtranslatestrategy = NULL;
	amroutine->amtranslatecmptype  = NULL;

	PG_RETURN_POINTER(amroutine);
}
