/*
 * mkt_params.h - Index build and search parameter limits
 *
 * Central place for all tunable parameter defaults, minimums, and
 * maximums. These are used by GUC definitions, reloptions, and CLI
 * benchmarks. Keeping them in one file makes it easy to see the full
 * set of knobs and their valid ranges.
 */

#ifndef MKT_PARAMS_H
#define MKT_PARAMS_H

/* ----------------------------------------------------------------
 * Partitioning — how the vector space is divided into clusters
 * ---------------------------------------------------------------- */

/*
 * nlist: number of leaf clusters (posting lists).
 * Each cluster gets one posting list in the index. More clusters
 * means smaller lists and faster scans, but coarser routing.
 * When 0 (auto), the build chooses sqrt(ntuples).
 */
#define MKT_DEFAULT_NLIST 0 /* 0 = auto: sqrt(ntuples) */
#define MKT_MIN_NLIST	  0
#define MKT_MAX_NLIST	  1000000

/*
 * fan_out: children per internal node in the centroid tree.
 * Controls the branching factor of the hierarchical k-means tree
 * used for routing queries to leaf clusters. Higher values give
 * shallower trees (fewer levels) but more work per level.
 * The build auto-tunes this from nlist when left at default.
 */
#define MKT_DEFAULT_FAN_OUT 32
#define MKT_MIN_FAN_OUT		2
#define MKT_MAX_FAN_OUT		65535

/* ----------------------------------------------------------------
 * Clustering — k-means parameters for centroid training
 * ---------------------------------------------------------------- */

/*
 * K-means iteration limits. 20 iterations is typically enough for
 * convergence on ANN workloads; tolerance triggers early exit when
 * centroid movement falls below this fraction of total cost.
 */
#define MKT_KMEANS_DEFAULT_MAX_ITER	 20
#define MKT_KMEANS_DEFAULT_TOLERANCE 1e-4f

/*
 * Random seed for k-means++ initialization. Fixed for
 * reproducible index builds.
 */
#define MKT_KMEANS_DEFAULT_SEED 42

/*
 * Number of k-means restarts. Each restart uses a different
 * k-means++ initialization and keeps the best result. More
 * restarts improve centroid quality at the cost of build time.
 */
#define MKT_KMEANS_DEFAULT_NREDO 1

/*
 * Training sample size: max(MIN_SAMPLES, nlist * PER_CLUSTER).
 * We need enough samples per cluster for stable centroids. 256
 * per cluster matches FAISS IVF defaults. The 10k floor ensures
 * reasonable quality even with very few clusters.
 */
#define MKT_KMEANS_MIN_SAMPLES		   10000
#define MKT_KMEANS_SAMPLES_PER_CLUSTER 256

/* ----------------------------------------------------------------
 * Quantization — RaBitQ encoding parameters
 * ---------------------------------------------------------------- */

/*
 * Random seed for the orthogonal rotation matrix used by RaBitQ.
 * The matrix is deterministic given (dim, seed) and stored in
 * index metadata so scans can reconstruct it. Fixed for
 * reproducible builds.
 */
#define MKT_RABITQ_DEFAULT_SEED 42

/* ----------------------------------------------------------------
 * Search — query-time parameters
 * ---------------------------------------------------------------- */

/*
 * nprobe: number of posting lists to scan per query. Higher values
 * improve recall at the cost of latency. Exposed as the mkt.nprobe
 * GUC so users can tune the recall/speed trade-off at runtime.
 */
#define MKT_DEFAULT_NPROBE 10
#define MKT_MIN_NPROBE	   1
#define MKT_MAX_NPROBE	   10000

/*
 * top-K fallback: used when the planner cannot extract a LIMIT
 * clause (e.g., subqueries, CTEs). Determines how many nearest
 * neighbors to collect before returning results.
 */
#define MKT_DEFAULT_TOPK 100

#endif /* MKT_PARAMS_H */
