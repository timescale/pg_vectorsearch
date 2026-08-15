/*
 * mktann_external_centroids.c - Load precomputed centroids from a table
 *
 * See mktann_external_centroids.h. All validation happens here and
 * raises a specific ereport(ERROR); mkt_hkmeans_build_from_external()
 * itself is the portable (PG + standalone) algorithm and only reports
 * pass/fail.
 */

#include <postgres.h>

#include <executor/spi.h>
#include <lib/stringinfo.h>
#include <string.h>
#include <utils/builtins.h>
#include <utils/fmgrprotos.h>
#include <utils/lsyscache.h>

#include "algo/hkmeans.h"
#include "mkt_pg.h"
#include "mktann_external_centroids.h"

/*
 * Resolve the centroids_table option to a safely quoted, canonical
 * identifier. regclassin raises its own descriptive ereport if the
 * name doesn't resolve; re-quoting the resolved schema/relation names,
 * rather than interpolating the option string directly, is what keeps
 * this safe against a maliciously or carelessly quoted option value.
 *
 * Index builds (like other maintenance commands) run with search_path
 * restricted to pg_catalog, pg_temp -- a hardening measure against
 * search-path hijacking during privileged operations -- so an
 * unqualified table name never resolves here even when it did in the
 * session that issued CREATE INDEX. centroids_table must therefore
 * always be schema-qualified (documented on the reloption itself).
 */
static char *
resolve_table_ident(const char *table_opt)
{
	Oid	 relid = DatumGetObjectId(
			 DirectFunctionCall1(regclassin, CStringGetDatum(table_opt)));
	char *nspname = get_namespace_name(get_rel_namespace(relid));
	char *relname = get_rel_name(relid);

	if (relname == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_TABLE),
				 errmsg("centroids_table \"%s\" does not exist", table_opt)));

	return quote_qualified_identifier(nspname, relname);
}

HKMeansResult *
mkt_external_centroids_build(
		const MktannOptions *opts, Dimension dim, uint32_t fan_out)
{
	const char *table_opt = GET_STRING_RELOPTION(opts, centroids_table);

	if (table_opt == NULL || table_opt[0] == '\0')
		return NULL;

	char *qualified = resolve_table_ident(table_opt);

	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "mktann: SPI_connect failed reading centroids_table");

	StringInfoData query;
	initStringInfo(&query);
	appendStringInfo(
			&query, "SELECT id, parent, vector FROM %s ORDER BY id", qualified);

	int rc = SPI_execute(query.data, true /* read-only */, 0);
	if (rc != SPI_OK_SELECT)
	{
		SPI_finish();
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("mktann: reading centroids_table \"%s\" failed "
						"(SPI status %d) -- expected columns id, parent, "
						"vector",
						table_opt,
						rc)));
	}

	uint64 n = SPI_processed;
	if (n == 0)
	{
		SPI_finish();
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("centroids_table \"%s\" is empty", table_opt)));
	}
	if (n > PG_UINT32_MAX)
	{
		SPI_finish();
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("centroids_table \"%s\" has too many rows (%llu)",
						table_opt,
						(unsigned long long)n)));
	}

	TupleDesc tupdesc	  = SPI_tuptable->tupdesc;
	int		  id_attn	  = SPI_fnumber(tupdesc, "id");
	int		  parent_attn = SPI_fnumber(tupdesc, "parent");
	int		  vec_attn	  = SPI_fnumber(tupdesc, "vector");
	if (id_attn <= 0 || parent_attn <= 0 || vec_attn <= 0)
	{
		SPI_finish();
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_COLUMN),
				 errmsg("centroids_table \"%s\" must have columns id, "
						"parent, vector",
						table_opt)));
	}

	/* Own output arrays in the caller's context so they survive the
	 * SPI_finish() below (which pops SPI's memory context away). */
	MemoryContext oldcontext = CurrentMemoryContext;
	int32_t		 *ids	   = MemoryContextAlloc(
			oldcontext, sizeof(int32_t) * (size_t)n);
	bool	*has_parent = MemoryContextAlloc(
			oldcontext, sizeof(bool) * (size_t)n);
	int32_t *parents = MemoryContextAlloc(
			oldcontext, sizeof(int32_t) * (size_t)n);
	float	*vectors = MemoryContextAlloc(
			oldcontext, sizeof(float) * (size_t)n * dim);

	for (uint64 i = 0; i < n; i++)
	{
		HeapTuple tuple = SPI_tuptable->vals[i];
		bool	  isnull;

		Datum id_d = SPI_getbinval(tuple, tupdesc, id_attn, &isnull);
		if (isnull)
		{
			SPI_finish();
			ereport(ERROR,
					(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
					 errmsg("centroids_table \"%s\": id is NULL at row "
							"%llu",
							table_opt,
							(unsigned long long)(i + 1))));
		}
		ids[i] = DatumGetInt32(id_d);

		Datum parent_d = SPI_getbinval(tuple, tupdesc, parent_attn, &isnull);
		has_parent[i]  = !isnull;
		parents[i]	   = isnull ? 0 : DatumGetInt32(parent_d);

		Datum vec_d = SPI_getbinval(tuple, tupdesc, vec_attn, &isnull);
		if (isnull)
		{
			SPI_finish();
			ereport(ERROR,
					(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
					 errmsg("centroids_table \"%s\": vector is NULL at "
							"row %llu",
							table_opt,
							(unsigned long long)(i + 1))));
		}

		MktVector *vec = DatumGetMktVector(vec_d);
		if (vec->dim != (int16_t)dim)
		{
			SPI_finish();
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("centroids_table \"%s\": vector at row %llu "
							"has %d dimensions, expected %u",
							table_opt,
							(unsigned long long)(i + 1),
							vec->dim,
							dim)));
		}
		memcpy(vectors + (size_t)i * dim,
			   vec->x,
			   sizeof(float) * (size_t)dim);
	}

	SPI_finish();

	HKMeansResult *tree = mkt_hkmeans_build_from_external(
			ids, has_parent, parents, vectors, (uint32_t)n, fan_out, dim);

	pfree(ids);
	pfree(has_parent);
	pfree(parents);
	pfree(vectors);

	if (tree == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("centroids_table \"%s\" does not describe a "
						"valid tree",
						table_opt),
				 errdetail("Check for: a duplicate id; a parent value "
						   "that does not match another row's id; a "
						   "cycle; a node whose children are a mix of "
						   "leaves and non-leaves; or a node with more "
						   "children than fan_out (%u).",
						   fan_out)));

	return tree;
}
