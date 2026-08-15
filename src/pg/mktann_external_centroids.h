/*
 * mktann_external_centroids.h - Load precomputed centroids from a table
 *
 * Backs the centroids_table index option: instead of clustering the
 * heap, build the routing tree directly from a user-populated table of
 * (id, parent, vector) rows -- the same shape as VectorChord's
 * external-build format, so centroids computed once (e.g. with a
 * full-dataset, many-iteration k-means run outside Postgres) can drive
 * both.
 */

#ifndef MKTANN_EXTERNAL_CENTROIDS_H
#define MKTANN_EXTERNAL_CENTROIDS_H

#include <access/reloptions.h> /* GET_STRING_RELOPTION */

#include "algo/hkmeans.h"
#include "mkt_pg.h"
#include "mkt_types.h"

/*
 * True when opts->centroids_table names a (non-empty) table, i.e. the
 * caller should skip sampling/k-means and take the external-centroids
 * path instead of clustering. Cheap and side-effect-free -- callers
 * that need the tree itself still call mkt_external_centroids_build.
 */
static inline bool
mkt_external_centroids_requested(const MktannOptions *opts)
{
	const char *table_opt =
			opts != NULL ? GET_STRING_RELOPTION(opts, centroids_table) : NULL;
	return table_opt != NULL && table_opt[0] != '\0';
}

/*
 * Build a routing tree from opts->centroids_table.
 *
 * Only called once mkt_external_centroids_requested(opts) is true; this
 * always either returns a valid tree or raises ereport(ERROR) with a
 * specific reason -- it never returns NULL.
 */
HKMeansResult *mkt_external_centroids_build(
		const MktannOptions *opts, Dimension dim, uint32_t fan_out);

#endif /* MKTANN_EXTERNAL_CENTROIDS_H */
